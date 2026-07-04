#!/usr/bin/env python3
"""
Replacement data source for the dead `Bisher/as_vspoof_2019_la` Hugging Face
dataset (custom loading script; requires `trust_remote_code`, which the
installed `datasets` library no longer supports at all -- script-based HF
datasets cannot be loaded anymore regardless of the flag, and the repo itself
also returns DatasetNotFoundError).

This script rebuilds an equivalent **local** Hugging Face `DatasetDict` at
`data/asvspoof_2019_la/` with the exact schema the original scripts expect:

    ds["train"]        -- official ASVspoof2019 LA "train" partition
                           (bonafide + spoof attacks A01-A06)
    ds["validation"]    -- official ASVspoof2019 LA "dev" partition
                           (bonafide + spoof attacks A01-A06)
    ds["test"]          -- official ASVspoof2019 LA "eval" partition
                           (bonafide + spoof attacks A07-A19)

Each row has columns: `speaker_id` (str), `system_id` (str; "-" for
bonafide, "A01".."A19" for spoof), `audio` (HF Audio feature, 16 kHz).
This matches exactly what experiments/results/e_mini_goat_fusion/
prepare_mini_goat_data.py, experiments/scripts/i1_geometry_causal_decomp.py,
experiments/scripts/i4_asvspoof_position.py, experiments/scripts/
j3_axis_adaptive_head.py, and experiments/results/e_mini_goat_fusion/
score_and_fuse_mini_goat.py read via
`ds["system_id"]`, `ds.select(sel).cast_column("audio", ...)`,
`ex["audio"]["array"]`, `ex.get("speaker_id", ...)`.

Source: the official ASVspoof2019 LA archive (identical protocol format:
`speaker_id filename - system_id key`) mirrored, unmodified, as a plain
`LA.zip` file (no custom loading script, no trust_remote_code needed) at the
public Hugging Face dataset repo `RohitGENAICODER/ASVspoofLADataset`
(https://huggingface.co/datasets/RohitGENAICODER/ASVspoofLADataset). Verified:
train/dev/eval flac counts (25381/24987/71934 incl. dir entries) and cm
protocol files match the official ASVspoof2019 LA release exactly
(25380/24844/71237 utterances; A01-A06 in train/dev, A07-A19 in eval).

Rather than downloading the full ~8 GB archive, this script reads the zip's
central directory via a single small request (HTTP Range on the resolve URL,
which the HF CDN supports) and then fetches ONLY the individual flac member
byte ranges it needs (one ranged GET per file, no `datasets` /
`huggingface_hub` whole-file download) -- keeping this a genuinely "tiny"
subset consistent with the repo's mini/tiny philosophy, while still being
large enough to satisfy every downstream sampler:

    split       needed (bonafide + spoof)   sampled here (with margin)
    train       >= 250 + 250 (mini_goat)    400 + 400
    validation  >=  75 +  75 (mini_goat)    150 + 150
    test        >= 800 + 800 (I1/I4/J3/E)   950 + 950 (spread over A07-A19)

Usage:
    python experiments/scripts/build_asvspoof_2019_la_cache.py [--force]
"""
from __future__ import annotations

import argparse
import io
import struct
import sys
import time
import zlib
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "asvspoof_2019_la"

REPO_ID = "RohitGENAICODER/ASVspoofLADataset"
ZIP_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/LA.zip"

SEED = 20260704
N_WORKERS = 8

# (hf_split_name, official_dir_name, protocol_suffix, n_bonafide, n_spoof, valid_attacks)
SPLIT_SPECS = [
    ("train", "train", "train", 400, 400, {f"A{i:02d}" for i in range(1, 7)}),
    ("validation", "dev", "dev", 150, 150, {f"A{i:02d}" for i in range(1, 7)}),
    ("test", "eval", "eval", 950, 950, {f"A{i:02d}" for i in range(7, 20)}),
]


class HTTPRangeFile:
    """Minimal seekable file-like object backed by HTTP Range requests, so
    zipfile can read the central directory of a huge remote zip without
    downloading it."""

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        r = self.session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if (n is None or n < 0) else min(self.pos + n, self.size) - 1
        if self.pos > end:
            return b""
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
                r.raise_for_status()
                data = r.content
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        self.pos += len(data)
        return data


def fetch_member_bytes(session: requests.Session, url: str, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress a single zip member with exactly one ranged GET,
    using the (offset, compress_size) already known from the central
    directory -- no separate request to read the local file header first."""
    start = info.header_offset
    # Generous fixed buffer covering the 30-byte local header + filename +
    # any extra fields (Unix/NTFS timestamps etc.), sized safely.
    buf = 128 + len(info.filename.encode("utf-8"))
    end = start + buf + info.compress_size
    for attempt in range(12):
        r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
        if r.status_code in (200, 206):
            break
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
        time.sleep(wait)
    else:
        r.raise_for_status()
    data = r.content
    assert data[:4] == b"PK\x03\x04", f"bad local header for {info.filename}: {data[:4]!r}"
    fname_len = struct.unpack("<H", data[26:28])[0]
    extra_len = struct.unpack("<H", data[28:30])[0]
    data_start = 30 + fname_len + extra_len
    comp = data[data_start:data_start + info.compress_size]
    if len(comp) < info.compress_size:
        # buffer was too small (unusually large extra field) -- refetch precisely
        end2 = start + data_start + info.compress_size
        r2 = session.get(url, headers={"Range": f"bytes={start}-{end2}"}, timeout=60)
        data2 = r2.content
        comp = data2[data_start:data_start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(comp, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        raw = comp
    else:
        raise ValueError(f"unsupported compress_type {info.compress_type} for {info.filename}")
    assert len(raw) == info.file_size, (info.filename, len(raw), info.file_size)
    return raw


def parse_protocol(text: str):
    """Parse an ASVspoof2019 LA cm protocol file:
    'speaker_id filename - system_id key' -> list of dict rows."""
    rows = []
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        speaker_id, filename, _, system_id, key = parts[:5]
        rows.append({
            "speaker_id": speaker_id,
            "filename": filename,
            "system_id": system_id,  # "-" for bonafide already
            "label": key,  # "bonafide" / "spoof"
        })
    return rows


def sample_split(rows, n_bonafide, n_spoof, valid_attacks, rng):
    bona = [r for r in rows if r["system_id"] == "-"]
    spoof = [r for r in rows if r["system_id"] in valid_attacks]
    assert len(bona) >= n_bonafide, f"only {len(bona)} bonafide rows available, need {n_bonafide}"
    assert len(spoof) >= n_spoof, f"only {len(spoof)} spoof rows available, need {n_spoof}"

    # Spread the spoof sample evenly across attack types instead of pure
    # uniform random, so every attack id A0x is represented (I4/E5 compute
    # per-attack stats over the eval subset).
    by_attack = defaultdict(list)
    for r in spoof:
        by_attack[r["system_id"]].append(r)
    attacks = sorted(by_attack)
    per_attack = n_spoof // len(attacks)
    remainder = n_spoof - per_attack * len(attacks)
    sel_spoof = []
    for i, a in enumerate(attacks):
        pool = by_attack[a]
        k = per_attack + (1 if i < remainder else 0)
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        sel_spoof.extend(pool[j] for j in idx)

    idx_b = rng.choice(len(bona), size=n_bonafide, replace=False)
    sel_bona = [bona[j] for j in idx_b]

    selected = sel_bona + sel_spoof
    rng.shuffle(selected)
    return selected


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if cache already exists")
    args = parser.parse_args()

    if (CACHE_DIR / "dataset_dict.json").exists() and not args.force:
        print(f"[build_asvspoof_2019_la_cache] cache already present at {CACHE_DIR}, skipping "
              f"(pass --force to rebuild).")
        return

    from datasets import Dataset, DatasetDict, Features, Value, Sequence

    session = requests.Session()
    print(f"[build_asvspoof_2019_la_cache] opening remote zip central directory: {ZIP_URL}")
    rf = HTTPRangeFile(ZIP_URL, session)
    zf = zipfile.ZipFile(rf)
    print(f"  zip size={rf.size / 1e9:.2f} GB, {len(zf.namelist())} entries (central directory only, "
          f"no bulk download)")

    rng_master = np.random.default_rng(SEED)
    split_datasets = {}
    total_files = 0
    total_bytes = 0

    for hf_split, official_dir, proto_suffix, n_bona, n_spoof, valid_attacks in SPLIT_SPECS:
        proto_path = f"LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{proto_suffix}.trl.txt"
        print(f"\n[build_asvspoof_2019_la_cache] {hf_split}: reading protocol {proto_path}")
        proto_text = zf.read(proto_path).decode()
        rows = parse_protocol(proto_text)
        print(f"  official partition has {len(rows)} utterances "
              f"({sum(1 for r in rows if r['system_id']=='-')} bonafide, "
              f"{sum(1 for r in rows if r['system_id']!='-')} spoof)")

        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        selected = sample_split(rows, n_bona, n_spoof, valid_attacks, rng)
        print(f"  sampled {len(selected)} utterances for local cache "
              f"({n_bona} bonafide + {n_spoof} spoof across {len(valid_attacks)} attack types)")

        member_names = [
            f"LA/ASVspoof2019_LA_{official_dir}/flac/{r['filename']}.flac" for r in selected
        ]
        infos = [zf.getinfo(n) for n in member_names]

        print(f"  downloading {len(infos)} flac files "
              f"({sum(i.file_size for i in infos) / 1e6:.1f} MB uncompressed) with {N_WORKERS} workers ...")
        t0 = time.time()
        audio_bytes = [None] * len(infos)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(fetch_member_bytes, session, ZIP_URL, info): j
                    for j, info in enumerate(infos)}
            done = 0
            for fut in as_completed(futs):
                j = futs[fut]
                audio_bytes[j] = fut.result()
                done += 1
                if done % 200 == 0:
                    print(f"    ... {done}/{len(infos)}")
        elapsed = time.time() - t0
        nbytes = sum(len(b) for b in audio_bytes)
        total_files += len(infos)
        total_bytes += nbytes
        print(f"  done in {elapsed:.1f}s, {nbytes / 1e6:.1f} MB flac bytes")

        # Decode flac -> float32 PCM array *now*, at build time, with soundfile.
        # NOTE: we deliberately do NOT use datasets' `Audio` feature here. In
        # the installed `datasets==5.0.0`, Audio.encode_example/decode_example
        # unconditionally import torchcodec, and torchcodec fails to load in
        # this sandbox (missing libtorchcodec/CUDA runtime libs), independent
        # of which dataset we source from. Storing plain
        # {"array": [...], "sampling_rate": int} avoids torchcodec entirely
        # while still round-tripping through save_to_disk/load_from_disk, and
        # keeps `ex["audio"]["array"]` working exactly as the original
        # (pre-torchcodec) scripts expect.
        arrays = []
        for b in audio_bytes:
            wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16_000:
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav), sr, 16_000
                ).numpy()
            arrays.append(wav.astype(np.float32))

        data = {
            "speaker_id": [r["speaker_id"] for r in selected],
            "system_id": [r["system_id"] for r in selected],
            "audio": [
                {"array": arr, "sampling_rate": 16_000}
                for arr in arrays
            ],
        }
        features = Features({
            "speaker_id": Value("string"),
            "system_id": Value("string"),
            "audio": {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")},
        })
        split_datasets[hf_split] = Dataset.from_dict(data, features=features)

    ddict = DatasetDict(split_datasets)
    print(f"\n[build_asvspoof_2019_la_cache] saving DatasetDict to {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ddict.save_to_disk(str(CACHE_DIR))

    meta = {
        "source_repo": REPO_ID,
        "source_file": "LA.zip",
        "replaces": "Bisher/as_vspoof_2019_la (dead: DatasetNotFoundError; also required "
                    "trust_remote_code, unsupported by installed datasets>=5)",
        "seed": SEED,
        "splits": {
            hf_split: {"n_bonafide": n_bona, "n_spoof": n_spoof, "official_partition": official_dir}
            for hf_split, official_dir, _, n_bona, n_spoof, _ in SPLIT_SPECS
        },
        "total_files_downloaded": total_files,
        "total_flac_bytes": total_bytes,
    }
    (CACHE_DIR / "cache_build_info.json").write_text(__import__("json").dumps(meta, indent=2))

    print(f"\n[build_asvspoof_2019_la_cache] done. {total_files} files, "
          f"{total_bytes / 1e6:.1f} MB, splits={list(split_datasets)}")
    for name, ds in split_datasets.items():
        print(f"  {name}: {len(ds)} rows")


if __name__ == "__main__":
    build()
