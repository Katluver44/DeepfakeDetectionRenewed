"""D1.6 interp control — is the aug score drop caused by the laughter content,
or merely by the splice (concatenation seam / displacing host audio out of the
3s center crop)?

Two controls per augmented fake, both re-scored with the matching checkpoint:
  (A) UNMASK: reconstruct the augmented waveform, then splice the laughter
      region back OUT -> should recover ~the base score if the drop is
      laughter-driven.
  (B) SILENCE: replace the laughter region with equal-duration silence (keeps
      the same timeline displacement / crop shift as the laughter, but removes
      laughter *content*) -> isolates content vs. geometry.

We compare, on the same augmented fakes:
   base score  vs  aug score  vs  unmask score  vs  silence score
If aug << base but unmask ~= base and silence ~= base, the drop is laughter
CONTENT (not the splice artifact).
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import soundfile as sf

LAUGHSMI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUGHSMI / "scripts"))
from score_itw_detector import load_detector, crop_or_tile_center, score_batch  # reuse

TARGET_SR = 16000


def load_wav(p):
    w, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if w.ndim > 1: w = w.mean(1)
    return w.astype(np.float32), sr


def reconstruct_regions(host_dur_s, insert_dur_s, position, total_len):
    """Return (laugh_start_idx, laugh_end_idx) within the augmented waveform."""
    ins = int(round(insert_dur_s * TARGET_SR))
    host = int(round(host_dur_s * TARGET_SR))
    if position == "start":
        return 0, ins
    if position == "end":
        return total_len - ins, total_len
    # mid: inserted after host//2
    mid = host // 2
    return mid, mid + ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda"
    lit = load_detector(args.ckpt, device)
    aug_dir = Path(args.aug_dir)
    manifest = [r for r in csv.DictReader(open(aug_dir / "manifest.csv")) if r["augmented"] == "1"]

    rows = []
    unmask_wavs, silence_wavs, ids = [], [], []
    for r in manifest:
        w, sr = load_wav(aug_dir / r["file"])
        s, e = reconstruct_regions(float(r["host_dur_s"]), float(r["insert_dur_s"]),
                                   r["position"], len(w))
        s = max(0, min(len(w), s)); e = max(0, min(len(w), e))
        unmask = np.concatenate([w[:s], w[e:]]) if e > s else w
        silence = w.copy(); silence[s:e] = 0.0
        unmask_wavs.append(crop_or_tile_center(unmask))
        silence_wavs.append(crop_or_tile_center(silence))
        ids.append(Path(r["file"]).name)

    def score_all(wavs):
        out = []
        for i in range(0, len(wavs), args.batch_size):
            out.extend(p for _, p in score_batch(lit, wavs[i:i+args.batch_size], device))
        return out

    unmask_scores = score_all(unmask_wavs)
    silence_scores = score_all(silence_wavs)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_id", "score_unmask", "score_silence"])
        w.writeheader()
        for i, fid in enumerate(ids):
            w.writerow({"file_id": fid, "score_unmask": unmask_scores[i], "score_silence": silence_scores[i]})
    print(f"wrote {len(ids)} rows -> {args.out}")


if __name__ == "__main__":
    main()
