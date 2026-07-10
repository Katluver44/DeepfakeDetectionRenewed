"""D1.4 — Augment an eval set by splicing synthetic (Bark) laughter into a
random 70% of its FAKE files, producing a parallel eval dir whose meta.csv
uses the SAME file basenames as the source set (so base/aug scores pair 1:1).

Augmentation = insertion (concatenation) of a laughter clip into the utterance
timeline at a varied position:
    position in {start, mid, end}  (recorded per file)
Bona-fide files are copied through unchanged (their file rows are identical).
Only spoof files are eligible; 70% of them are randomly augmented, the rest
copied unchanged (so the aug set is a superset-consistent mirror of base).

Insert source clips are read from one or more laughter dirs (Bark). Each is
peak-normalized and level-matched to the host RMS to avoid a level artifact
dominating the splice.

Usage:
    python scripts/augment_laughter.py \
        --src data/eval_asv19 --out data/eval_asv19_aug \
        --laugh-dirs data/laugh_bank_bark data/bark_probe/laughter_only \
        --frac 0.7 --seed 20260710

Writes:
    <out>/wavs/*.wav
    <out>/meta.csv        (file,speaker,label — same basenames as src)
    <out>/manifest.csv    (file,label,augmented,insert_file,position,insert_dur_s,host_dur_s)
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_16k_mono(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def level_match(insert: np.ndarray, host: np.ndarray) -> np.ndarray:
    """Peak-normalize insert, then scale to the host's RMS so the laughter sits
    at a comparable loudness to the surrounding speech (not louder/quieter)."""
    peak = np.max(np.abs(insert)) + 1e-9
    ins = insert / peak
    target = rms(host)
    cur = rms(ins)
    ins = ins * (target / (cur + 1e-9))
    return np.clip(ins, -1.0, 1.0).astype(np.float32)


def splice_in(host: np.ndarray, insert: np.ndarray, position: str) -> np.ndarray:
    if position == "start":
        return np.concatenate([insert, host])
    if position == "end":
        return np.concatenate([host, insert])
    mid = len(host) // 2
    return np.concatenate([host[:mid], insert, host[mid:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source eval dir (has meta.csv + wavs/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--laugh-dirs", nargs="+", required=True)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    # gather laughter inserts
    inserts = []
    for d in args.laugh_dirs:
        for p in sorted(Path(d).glob("*.wav")):
            inserts.append(p)
    if not inserts:
        raise SystemExit(f"no laughter inserts found in {args.laugh_dirs}")
    print(f"[augment] {len(inserts)} laughter inserts available")

    rows = list(csv.DictReader(open(src / "meta.csv")))
    spoof_rows = [r for r in rows if r["label"] == "spoof"]
    rng.shuffle(spoof_rows)
    n_aug = int(round(args.frac * len(spoof_rows)))
    aug_ids = {r["file"] for r in spoof_rows[:n_aug]}
    print(f"[augment] {len(spoof_rows)} spoof files, augmenting {n_aug} ({args.frac:.0%})")

    positions = ["start", "mid", "end"]
    meta_out, manifest = [], []
    for r in rows:
        rel = r["file"]  # e.g. wavs/xxx.wav
        host = load_16k_mono(src / rel)
        base = Path(rel).name
        augmented = rel in aug_ids
        insert_file, position, ins_dur = "", "", 0.0
        if augmented:
            ins_path = rng.choice(inserts)
            insert = load_16k_mono(ins_path)
            insert = level_match(insert, host)
            position = rng.choice(positions)
            wav = splice_in(host, insert, position)
            insert_file, ins_dur = str(ins_path), round(len(insert) / TARGET_SR, 3)
        else:
            wav = host
        sf.write(str(out / "wavs" / base), np.clip(wav, -1, 1).astype(np.float32), TARGET_SR)
        meta_out.append({"file": f"wavs/{base}", "speaker": r["speaker"], "label": r["label"]})
        manifest.append({"file": f"wavs/{base}", "label": r["label"],
                         "augmented": int(augmented), "insert_file": insert_file,
                         "position": position, "insert_dur_s": ins_dur,
                         "host_dur_s": round(len(host) / TARGET_SR, 3)})

    with open(out / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader(); w.writerows(meta_out)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "augmented", "insert_file",
                                          "position", "insert_dur_s", "host_dur_s"])
        w.writeheader(); w.writerows(manifest)
    print(f"[augment] wrote {len(meta_out)} files -> {out}  ({n_aug} augmented)")


if __name__ == "__main__":
    main()
