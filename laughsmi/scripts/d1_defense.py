"""Test-time defense against the laughter-splice evasion (NO retraining).

Mechanism recap: the model mean-pools frames, so an inserted laughter/real
segment contributes non-spoof frames that pull the utterance score down. Defense:
score the utterance as the MOST-SPOOF sliding window instead of the whole-crop
mean — a genuinely-synthetic region still scores high even if a laughter window
dilutes the average.

For each file we take sliding 3s windows (hop 1s) over the FULL waveform, score
each window with the model (each is center-cropped/tiled to 3s as usual), and
aggregate with:
  - mean      (baseline: ~ the model's own whole-utterance behavior)
  - max       (worst-window / most-spoof)
  - p90       (90th-percentile window, robust version of max)

We evaluate each aggregator's EER on clean data and its evasion rate on the
laughter-augmented fakes, at the aggregator's OWN clean-EER threshold. A good
defense keeps clean EER low AND cuts the aug evasion rate.

Usage:
  python scripts/d1_defense.py --ckpt ../models/asv19-wavlm-gat-full.ckpt \
      --clean-dir data/eval_asv19 --aug-dir data/eval_asv19_aug --tag asv19
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve

L = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(L / "scripts"))
from score_itw_detector import load_detector, crop_or_tile_center, score_batch, load_wav_mono_16k

SR = 16000
WIN = 3 * SR


def window_scores(lit, wav, device, hop_s=1.0, bs=16):
    hop = int(hop_s * SR)
    T = len(wav)
    if T <= WIN:
        return np.array([p for _, p in score_batch(lit, [crop_or_tile_center(wav)], device)])
    starts = list(range(0, T - WIN + 1, hop))
    if starts[-1] != T - WIN:
        starts.append(T - WIN)
    wins = [crop_or_tile_center(wav[s:s + WIN]) for s in starts]
    out = []
    for i in range(0, len(wins), bs):
        out.extend(p for _, p in score_batch(lit, wins[i:i + bs], device))
    return np.array(out)


def aggregate(ws):
    return {"mean": float(ws.mean()), "max": float(ws.max()), "p90": float(np.percentile(ws, 90))}


def score_dir(lit, audio_dir, meta_rows, device):
    rows = []
    for r in meta_rows:
        wav = load_wav_mono_16k(str(audio_dir / r["file"]))
        agg = aggregate(window_scores(lit, wav, device))
        rows.append({"file": Path(r["file"]).name, "label": r["label"], **agg})
    return rows


def eer_thr(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--aug-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    device = "cuda"
    lit = load_detector(args.ckpt, device)

    clean_meta = list(csv.DictReader(open(Path(args.clean_dir) / "meta.csv")))
    clean = score_dir(lit, Path(args.clean_dir), clean_meta, device)

    # augmented fakes only (paired by basename via manifest)
    man = {Path(r["file"]).name: r for r in csv.DictReader(open(Path(args.aug_dir) / "manifest.csv")) if r["augmented"] == "1"}
    aug_meta = [r for r in csv.DictReader(open(Path(args.aug_dir) / "meta.csv")) if Path(r["file"]).name in man]
    aug = score_dir(lit, Path(args.aug_dir), aug_meta, device)

    # index clean augmented-fake baseline for pairing
    clean_by_id = {r["file"]: r for r in clean}

    print(f"\n{'agg':<6}{'cleanEER':>10}{'thr':>8}{'baseCatch':>11}{'augCatch':>10}{'evasion':>9}")
    results = []
    for agg in ["mean", "max", "p90"]:
        sc = np.array([r[agg] for r in clean]); yl = np.array([0 if r["label"] == "bona-fide" else 1 for r in clean])
        thr, eer = eer_thr(sc, yl)
        # on augmented fakes: baseline (clean version of same files) vs aug
        aug_ids = [r["file"] for r in aug]
        base_fake = np.array([clean_by_id[i][agg] for i in aug_ids])
        aug_fake = np.array([r[agg] for r in aug])
        base_catch = float((base_fake >= thr).mean())
        aug_catch = float((aug_fake >= thr).mean())
        evasion = float((aug_fake < thr).mean())
        print(f"{agg:<6}{eer*100:>9.2f}%{thr:>8.3f}{base_catch:>11.2f}{aug_catch:>10.2f}{evasion:>9.2f}")
        results.append({"agg": agg, "clean_eer_pct": round(eer*100, 2), "thr": round(thr, 4),
                        "base_catch": round(base_catch, 3), "aug_catch": round(aug_catch, 3),
                        "evasion_rate": round(evasion, 3)})

    with open(L / "tables" / f"table_defense_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f"\nwrote tables/table_defense_{args.tag}.csv")
    print("Read: 'mean' = undefended baseline; 'max'/'p90' = worst-window defense.")


if __name__ == "__main__":
    main()
