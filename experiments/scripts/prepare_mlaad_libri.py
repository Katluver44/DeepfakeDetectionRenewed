#!/usr/bin/env python3
"""
Build a domain-matched MLAAD(spoof) + LibriSpeech-train-clean-100(bonafide)
processed dataset for training a calibrated deepfake detector.

Rationale (see continue_laugh.md / laughsmi_plan.md D1 laughter-evasion work):
existing MLAAD-tiny-trained checkpoints (models/mlaad_robust_goat*.ckpt) are
saturated on our eval set because their "bonafide" class (MLAAD's `original/`
audio, likely M-AILABS/LibriVox-derived, possibly missing/mismatched in the
current HF snapshot) does not match the real-speech distribution we evaluate
against (LibriSpeech test-clean, per laughsmi/scripts/build_eval_sets.py). The
model learns "clean read English narration = bonafide" from a *different*
recording/mastering pipeline than LibriSpeech, so at eval time it false-alarms
on LibriSpeech real speech.

Fix: train directly against the real distribution we evaluate on.
  SPOOF    = MLAAD English fakes, HF mueller91/MLAAD fake/en/<system>/*.wav
             (already cached locally; stratified across systems).
  BONAFIDE = LibriSpeech train-clean-100 (HF openslr/librispeech_asr,
             all/train.clean.100/*.parquet), decoded via soundfile from the
             embedded audio bytes (NOT the `datasets` Audio feature, which
             needs torchcodec and crashes in this environment).

Leakage safety: bonafide comes from train-clean-100 (different LibriSpeech
speakers AND a different split than test-clean). The eval set
(laughsmi/data/eval_mlaad) uses LibriSpeech test-clean bona-fide clips fetched
via kresnik/librispeech_asr_test — a disjoint split/dataset mirror. No overlap.

Output layout mirrors experiments/data/mlaad_tiny_processed/ so it's a drop-in
for train_mlaad_regular.py:
    <output-dir>/audio/*.pt          (48000,) float32 16kHz mono tensors
    <output-dir>/splits/train.json
    <output-dir>/splits/val.json
    <output-dir>/metadata.json

Usage:
    laughsmi/venv/bin/python experiments/scripts/prepare_mlaad_libri.py \
        --n-spoof 1200 --n-bona 1200 --val-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 16000
TARGET_SAMPLES = 3 * TARGET_SR  # 48000

MLAAD_SNAPSHOT = PROJECT_ROOT.parent / ".cache" / "huggingface"  # unused, resolved via hf cache below


def resample_to_16k_mono(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != TARGET_SR:
        t = torch.from_numpy(wav).unsqueeze(0)
        resampler = T.Resample(sr, TARGET_SR)
        t = resampler(t)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def center_crop_or_tile(wav: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    T_ = len(wav)
    if T_ == 0:
        return np.zeros(target_len, dtype=np.float32)
    if T_ < target_len:
        reps = -(-target_len // T_)
        wav = np.tile(wav, reps)[:target_len]
        return wav.astype(np.float32)
    start = (T_ - target_len) // 2
    return wav[start:start + target_len].astype(np.float32)


def save_tensor(wav: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(wav.astype(np.float32)), path)


# ---------------------------------------------------------------------------
# Spoof: MLAAD English fakes (local HF cache)
# ---------------------------------------------------------------------------
def collect_mlaad_spoof_files(n_spoof: int, seed: int) -> list:
    """Stratified sample of MLAAD English fake .wav *repo-relative paths*
    (mueller91/MLAAD is a HF dataset repo; most files are not in the local
    cache, so we list via the Hub API and download on demand)."""
    from huggingface_hub import HfApi

    rng = random.Random(seed)
    api = HfApi()
    info = api.dataset_info("mueller91/MLAAD")
    bysys = defaultdict(list)
    for s in info.siblings:
        f = s.rfilename
        if f.startswith("fake/en/") and f.endswith(".wav"):
            bysys[f.split("/")[2]].append(f)

    systems = sorted(bysys)
    log.info(f"[mlaad] found {len(systems)} English TTS systems on the Hub, "
              f"{sum(len(v) for v in bysys.values())} total wavs available")

    per_sys = max(1, n_spoof // max(1, len(systems)))
    plan = []
    for sysn in systems:
        files = list(bysys[sysn])
        rng.shuffle(files)
        for f in files[:per_sys]:
            plan.append((sysn, f))
    rng.shuffle(plan)
    plan = plan[:n_spoof]
    log.info(f"[mlaad] selected {len(plan)} spoof files across {len(systems)} systems "
              f"(~{per_sys}/system)")
    return plan


def process_mlaad_spoof(plan: list, audio_dir: Path) -> list:
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _dl(item):
        sysn, rel = item
        try:
            local = hf_hub_download(
                "mueller91/MLAAD", rel, repo_type="dataset", etag_timeout=10,
            )
            return (sysn, rel, local, None)
        except Exception as e:
            return (sysn, rel, None, e)

    log.info(f"[mlaad] downloading {len(plan)} spoof files (12 parallel workers) ...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_dl, item): item for item in plan}
        n_done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            n_done += 1
            if n_done % 100 == 0:
                log.info(f"[mlaad] downloaded {n_done}/{len(plan)} ({time.time()-t0:.0f}s elapsed)")
    log.info(f"[mlaad] download done in {time.time()-t0:.0f}s")

    records = []
    n_fail = 0
    for i, (sysn, rel, local, err) in enumerate(results):
        if err is not None:
            n_fail += 1
            log.warning(f"[mlaad] skip {rel}: {type(err).__name__} {err}")
            continue
        try:
            wav, sr = sf.read(local, dtype="float32", always_2d=False)
            wav = resample_to_16k_mono(wav, sr)
            wav = center_crop_or_tile(wav)
            sample_id = f"spoof_{i:05d}_{sysn.replace(' ', '_').replace('/', '_')}"
            out_path = audio_dir / f"{sample_id}.pt"
            save_tensor(wav, out_path)
            records.append({
                "sample_id": sample_id,
                "audio_path": f"audio/{sample_id}.pt",
                "label": "spoof",
                "attack_system": sysn,
                "language": "en",
                "source": "mlaad",
            })
        except Exception as e:
            n_fail += 1
            log.warning(f"[mlaad] skip {rel}: {type(e).__name__} {e}")
    log.info(f"[mlaad] processed {len(records)} spoof files ({n_fail} failed)")
    return records


# ---------------------------------------------------------------------------
# Bonafide: LibriSpeech train-clean-100 (parquet, soundfile decode)
# ---------------------------------------------------------------------------
def process_librispeech_bona(n_bona: int, audio_dir: Path, seed: int, shards: list) -> list:
    """Pull a SPEAKER-DIVERSE sample of LibriSpeech train-clean-100 bonafide.

    train-clean-100 parquet shards are laid out speaker-contiguously (each
    shard of ~2000 rows spans only ~18-19 unique speakers), so reading whole
    shards sequentially (as an earlier version of this function did) yields
    almost no speaker diversity and lets the model memorize a handful of
    recording chains instead of learning "real speech" in general -- that
    caused a large train/eval EER gap in an earlier run of this pipeline
    (val-eer ~0.11 in-distribution, ~0.27 on held-out LibriSpeech speakers).
    Fix: draw only a small, capped number of utterances PER SPEAKER, spread
    across many shards, so many distinct speakers are represented.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from collections import defaultdict

    rng = random.Random(seed + 1)
    records = []
    n_ok = 0
    per_speaker_cap = 8  # cap utterances/speaker so diversity dominates over shard size
    speaker_counts: dict = defaultdict(int)

    for shard in shards:
        if n_ok >= n_bona:
            break
        rel = f"all/train.clean.100/{shard}"
        log.info(f"[libri] downloading shard {rel} ...")
        local = hf_hub_download("openslr/librispeech_asr", rel, repo_type="dataset")
        pf = pq.ParquetFile(local)
        table = pf.read()
        n_rows = table.num_rows
        idx = list(range(n_rows))
        rng.shuffle(idx)

        audio_col = table.column("audio")
        spk_col = table.column("speaker_id")
        id_col = table.column("id")

        for i in idx:
            if n_ok >= n_bona:
                break
            spk = spk_col[i].as_py()
            if speaker_counts[spk] >= per_speaker_cap:
                continue
            try:
                cell = audio_col[i].as_py()
                audio_bytes = cell["bytes"]
                utt_id = id_col[i].as_py()
                wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
                wav = resample_to_16k_mono(wav, sr)
                wav = center_crop_or_tile(wav)
                sample_id = f"bona_{n_ok:05d}_{utt_id}"
                out_path = audio_dir / f"{sample_id}.pt"
                save_tensor(wav, out_path)
                records.append({
                    "sample_id": sample_id,
                    "audio_path": f"audio/{sample_id}.pt",
                    "label": "bonafide",
                    "attack_system": "none",
                    "language": "en",
                    "source": "librispeech_train_clean_100",
                    "speaker_id": spk,
                })
                speaker_counts[spk] += 1
                n_ok += 1
            except Exception as e:
                log.warning(f"[libri] skip row {i} in {shard}: {type(e).__name__} {e}")

    log.info(f"[libri] processed {n_ok} bonafide files from {len(speaker_counts)} speakers "
             f"(cap={per_speaker_cap}/speaker) across shards {shards}")
    return records


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def make_splits(spoof_records: list, bona_records: list, val_frac: float, seed: int):
    """Speaker-disjoint split for bonafide (a LibriSpeech speaker's utterances
    go entirely to train OR val, never both) so val EER isn't inflated by the
    model having seen the same speaker/recording-chain during training.
    Spoof (MLAAD) has no natural speaker id to key on; split by attack_system
    instead so val also holds out nothing structurally different there
    (system stratification isn't required for leakage-safety but keeps both
    splits balanced across systems)."""
    rng = random.Random(seed + 2)
    spoof = list(spoof_records)
    bona = list(bona_records)
    rng.shuffle(spoof)

    # --- bonafide: split whole speakers into train/val ---
    speakers = sorted({r.get("speaker_id") for r in bona})
    rng.shuffle(speakers)
    n_val_speakers = max(1, int(len(speakers) * val_frac))
    val_speakers = set(speakers[:n_val_speakers])
    bona_val = [r for r in bona if r.get("speaker_id") in val_speakers]
    bona_train = [r for r in bona if r.get("speaker_id") not in val_speakers]

    n_val_spoof = max(1, int(len(spoof) * val_frac))
    spoof_val = spoof[:n_val_spoof]
    spoof_train = spoof[n_val_spoof:]

    val = spoof_val + bona_val
    train = spoof_train + bona_train
    rng.shuffle(val)
    rng.shuffle(train)
    log.info(f"[splits] bonafide speakers: {len(speakers)} total, "
             f"{len(val_speakers)} held out for val (speaker-disjoint)")
    return train, val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/data/mlaad_libri_processed")
    p.add_argument("--n-spoof", type=int, default=1200)
    p.add_argument("--n-bona", type=int, default=750)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--libri-shards", type=int, default=14,
                    help="Number of train.clean.100 parquet shards to pull from "
                         "(each ~2000 rows / ~18-19 speakers; there are 14 total shards "
                         "covering all 251 train-clean-100 speakers). Use many shards with "
                         "a low per-speaker cap for speaker diversity, not few shards fully "
                         "exhausted (that gives near-zero speaker diversity).")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = args.output_dir
    audio_dir = out_dir / "audio"
    splits_dir = out_dir / "splits"

    if out_dir.exists() and not args.force and any(audio_dir.glob("*.pt")):
        log.info(f"{out_dir} already has processed audio; use --force to rebuild")
        return

    audio_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    log.info("=== Collecting MLAAD spoof file plan ===")
    spoof_plan = collect_mlaad_spoof_files(args.n_spoof, args.seed)
    log.info("=== Processing MLAAD spoof audio ===")
    spoof_records = process_mlaad_spoof(spoof_plan, audio_dir)

    log.info("=== Processing LibriSpeech train-clean-100 bonafide audio ===")
    shard_names = [f"{i:04d}.parquet" for i in range(args.libri_shards)]
    bona_records = process_librispeech_bona(args.n_bona, audio_dir, args.seed, shard_names)

    n_spoof_ok = len(spoof_records)
    n_bona_ok = len(bona_records)
    log.info(f"=== Balancing: {n_spoof_ok} spoof, {n_bona_ok} bonafide ===")
    n_bal = min(n_spoof_ok, n_bona_ok)
    if n_bal < n_spoof_ok:
        spoof_records = spoof_records[:n_bal]
    if n_bal < n_bona_ok:
        bona_records = bona_records[:n_bal]

    train_records, val_records = make_splits(spoof_records, bona_records, args.val_frac, args.seed)

    (splits_dir / "train.json").write_text(json.dumps(train_records, indent=2))
    (splits_dir / "val.json").write_text(json.dumps(val_records, indent=2))

    metadata = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "prepare_mlaad_libri.py",
        "n_spoof_total": len(spoof_records),
        "n_bona_total": len(bona_records),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "spoof_source": "mueller91/MLAAD fake/en/<system>/*.wav (local HF cache)",
        "bona_source": f"openslr/librispeech_asr all/train.clean.100 shards {shard_names}",
        "leakage_note": (
            "Bonafide drawn from LibriSpeech TRAIN-clean-100 (different speakers "
            "and different split than the eval set's LibriSpeech TEST-clean bona "
            "clips in laughsmi/data/eval_mlaad, sourced from kresnik/librispeech_asr_test). "
            "No overlap between train and eval bonafide audio."
        ),
        "target_sr": TARGET_SR,
        "target_samples": TARGET_SAMPLES,
        "val_frac": args.val_frac,
        "seed": args.seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total_time = time.time() - t0
    log.info(f"Done in {total_time:.0f}s. train={len(train_records)} val={len(val_records)}")
    log.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
