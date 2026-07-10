"""D1.1 — Build two tiny, balanced deepfake eval sets as (wav dir + meta.csv)
so they can be fed to `score_itw_detector.py` unchanged.

meta.csv schema (what the scorer expects): file,speaker,label
  label in {bona-fide, spoof}; file is a basename relative to the eval dir.

Set A — eval_asv19/  (in-domain for models/asv19-wavlm-gat-full.ckpt)
  Exported from the local HuggingFace arrow cache data/asvspoof_2019_la (test
  split). system_id == '-' is bona-fide, 'Axx' is spoof. Balanced subsample.

Set B — eval_mlaad/  (in-domain for models/mlaad_robust_goat.ckpt)
  spoof: stratified English sample downloaded per-file from HF mueller91/MLAAD
         (fake/en/<system>/*.wav), ~N/#systems per system.
  bona-fide (real anchor): LibriSpeech test-clean sample from HF
         kresnik/librispeech_asr_test. MLAAD's fakes are synthesized from
         read-English audiobook text (M-AILABS/LibriVox lineage), so clean
         read English is the domain-appropriate real companion (official
         M-AILABS has no usable English HF mirror; LibriSpeech test-clean is
         the closest directly-downloadable real read-English corpus).

All wavs written mono 16 kHz float32. Idempotent-ish: skips writing a set whose
meta.csv already has the target count unless --force.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

LAUGHSMI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAUGHSMI_ROOT.parent
TARGET_SR = 16000


def write_wav(path: Path, wav: np.ndarray, sr: int = TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)


def resample_if_needed(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def write_meta(meta_path: Path, rows: list):
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Set A: ASVspoof2019-LA test export
# ---------------------------------------------------------------------------
def build_asv19(out_dir: Path, n_per_class: int, seed: int):
    from datasets import load_from_disk
    rng = random.Random(seed)
    ds = load_from_disk(str(REPO_ROOT / "data" / "asvspoof_2019_la"))["test"]

    bona_idx = [i for i, s in enumerate(ds["system_id"]) if s == "-"]
    spoof_idx = [i for i, s in enumerate(ds["system_id"]) if s != "-"]
    rng.shuffle(bona_idx)
    rng.shuffle(spoof_idx)
    bona_idx = bona_idx[:n_per_class]
    spoof_idx = spoof_idx[:n_per_class]

    rows = []
    wav_dir = out_dir / "wavs"
    for tag, idxs, label in [("bona", bona_idx, "bona-fide"), ("spoof", spoof_idx, "spoof")]:
        for i in idxs:
            r = ds[i]
            wav = resample_if_needed(np.asarray(r["audio"]["array"], dtype=np.float32),
                                     r["audio"]["sampling_rate"])
            fname = f"{tag}_{i:05d}.wav"
            write_wav(wav_dir / fname, wav)
            rows.append({"file": f"wavs/{fname}", "speaker": r["speaker_id"], "label": label})
    write_meta(out_dir / "meta.csv", rows)
    print(f"[asv19] wrote {len(rows)} files -> {out_dir}")


# ---------------------------------------------------------------------------
# Set B: MLAAD spoof (download) + LibriSpeech real anchor
# ---------------------------------------------------------------------------
def build_mlaad(out_dir: Path, n_spoof: int, n_bona: int, seed: int, timeout_s: int):
    import time
    from huggingface_hub import HfApi, hf_hub_download
    rng = random.Random(seed)
    wav_dir = out_dir / "wavs"
    rows = []
    t_start = time.time()

    # --- spoof: stratified English MLAAD ---
    api = HfApi()
    info = api.dataset_info("mueller91/MLAAD")
    from collections import defaultdict
    bysys = defaultdict(list)
    for s in info.siblings:
        f = s.rfilename
        if f.startswith("fake/en/") and f.endswith(".wav"):
            bysys[f.split("/")[2]].append(f)
    systems = sorted(bysys)
    per_sys = max(1, n_spoof // len(systems))
    plan = []
    for sysn in systems:
        files = bysys[sysn]
        rng.shuffle(files)
        for f in files[:per_sys]:
            plan.append((sysn, f))
    rng.shuffle(plan)
    plan = plan[:n_spoof]

    n_ok = 0
    for sysn, rel in plan:
        if time.time() - t_start > timeout_s:
            print(f"[mlaad] TIMEOUT after {n_ok} spoof files; proceeding with what we have")
            break
        try:
            local = hf_hub_download("mueller91/MLAAD", rel, repo_type="dataset")
            wav, sr = sf.read(local, dtype="float32", always_2d=False)
            wav = resample_if_needed(wav, sr)
            fname = f"spoof_{n_ok:05d}_{sysn.replace(' ', '_').replace('/', '_')}.wav"
            write_wav(wav_dir / fname, wav)
            rows.append({"file": f"wavs/{fname}", "speaker": sysn, "label": "spoof"})
            n_ok += 1
        except Exception as e:
            print(f"[mlaad] skip {rel}: {type(e).__name__} {e}")
    print(f"[mlaad] got {n_ok} spoof files in {time.time()-t_start:.0f}s")

    # --- bona-fide: LibriSpeech test-clean anchor ---
    try:
        from datasets import load_dataset
        ls = load_dataset("kresnik/librispeech_asr_test", "clean", split="test",
                          streaming=True, trust_remote_code=True)
        n_b = 0
        for r in ls:
            if n_b >= n_bona:
                break
            a = r["audio"]
            wav = resample_if_needed(np.asarray(a["array"], dtype=np.float32), a["sampling_rate"])
            spk = str(r.get("speaker_id", r.get("id", n_b)))
            fname = f"bona_{n_b:05d}.wav"
            write_wav(wav_dir / fname, wav)
            rows.append({"file": f"wavs/{fname}", "speaker": spk, "label": "bona-fide"})
            n_b += 1
        print(f"[mlaad] got {n_b} bona-fide LibriSpeech anchor files")
    except Exception as e:
        print(f"[mlaad] LibriSpeech anchor FAILED: {type(e).__name__} {e}")

    write_meta(out_dir / "meta.csv", rows)
    nb = sum(1 for r in rows if r["label"] == "bona-fide")
    ns = sum(1 for r in rows if r["label"] == "spoof")
    print(f"[mlaad] wrote {len(rows)} files ({nb} bona / {ns} spoof) -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["asv19", "mlaad", "both"], default="both")
    p.add_argument("--n-per-class", type=int, default=150, help="asv19 per-class count")
    p.add_argument("--n-mlaad-spoof", type=int, default=156)  # 4 per 39 systems
    p.add_argument("--n-mlaad-bona", type=int, default=150)
    p.add_argument("--mlaad-timeout", type=int, default=1200, help="seconds budget for MLAAD dl")
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--out-root", default=str(LAUGHSMI_ROOT / "data"))
    args = p.parse_args()

    out_root = Path(args.out_root)
    if args.which in ("asv19", "both"):
        build_asv19(out_root / "eval_asv19", args.n_per_class, args.seed)
    if args.which in ("mlaad", "both"):
        build_mlaad(out_root / "eval_mlaad", args.n_mlaad_spoof, args.n_mlaad_bona,
                    args.seed, args.mlaad_timeout)


if __name__ == "__main__":
    main()
