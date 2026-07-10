"""Investigate WHY inserting laughter/silence into fakes moves the ASV19
detector's spoof score DOWN but the calibrated-MLAAD detector's score UP
(both Phoneme_GAT_lit, same architecture). See SIGNFLIP_INVESTIGATION.md.

Parts (all results printed and dumped to detector_out/signflip_*.csv):
  A  logit-space paired deltas from the EXISTING score CSVs (H-A / H-D:
     saturation & operating-point artifacts), plus base-score-matched bins.
  C  pure-stimulus scoring: run PURE laughter (Bark + real), PURE silence,
     PURE bona speech (ASV19-LA & LibriSpeech) and PURE spoof clips through
     BOTH models as whole utterances (H-C: spoof-class geometry).
  X  cross-model paired deltas: score the OTHER model on each eval set's
     base/aug pairs (does the sign follow the MODEL or the DATASET?).
  B  frame-level sliding-window logits over augmented fakes for BOTH models:
     do windows covering the inserted region get low (bona) or high (spoof)
     logits in each model? (H-B)

Read-only w.r.t. existing data/models/scripts; reuses
score_itw_detector.load_detector / score_batch / crop_or_tile_center.

Usage: venv/bin/python scripts/investigate_signflip.py [--parts ACXB]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

LAUGHSMI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUGHSMI / "scripts"))

from score_itw_detector import (  # noqa: E402
    TARGET_LEN, TARGET_SR, crop_or_tile_center, load_detector,
    load_wav_mono_16k, score_batch,
)

DET = LAUGHSMI / "detector_out"
DATA = LAUGHSMI / "data"
MODELS = LAUGHSMI.parent / "models"

CKPTS = {
    "asv19": str(MODELS / "asv19-wavlm-gat-full.ckpt"),
    "mlaad_cal": str(MODELS / "mlaad_libri_calibrated.ckpt"),
}

RNG = np.random.default_rng(20260710)


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def aug_file_ids(manifest_path) -> list[str]:
    return [r["file"] for r in read_csv(manifest_path) if r["augmented"] == "1"]


def paired_deltas(base_csv, aug_csv, ids):
    base = {r["file_id"]: r for r in read_csv(base_csv)}
    aug = {r["file_id"]: r for r in read_csv(aug_csv)}
    dl, dp, bl, bp = [], [], [], []
    for fid in ids:
        if fid in base and fid in aug:
            dl.append(float(aug[fid]["logit"]) - float(base[fid]["logit"]))
            dp.append(float(aug[fid]["score"]) - float(base[fid]["score"]))
            bl.append(float(base[fid]["logit"]))
            bp.append(float(base[fid]["score"]))
    return np.array(dl), np.array(dp), np.array(bl), np.array(bp)


def summarize(name, dl, dp, bl, bp):
    from scipy import stats
    w = stats.wilcoxon(dl) if np.any(dl != 0) else None
    row = dict(
        cond=name, n=len(dl),
        base_prob_mean=float(bp.mean()), base_logit_mean=float(bl.mean()),
        dlogit_mean=float(dl.mean()), dlogit_median=float(np.median(dl)),
        dprob_mean=float(dp.mean()),
        frac_dlogit_neg=float((dl < 0).mean()),
        wilcoxon_p=float(w.pvalue) if w else float("nan"),
    )
    print(f"  {name:26s} n={row['n']:4d} baseP={row['base_prob_mean']:+.3f} "
          f"baseL={row['base_logit_mean']:+.2f} dL={row['dlogit_mean']:+.3f} "
          f"(med {row['dlogit_median']:+.3f}) dP={row['dprob_mean']:+.3f} "
          f"frac(dL<0)={row['frac_dlogit_neg']:.2f} p={row['wilcoxon_p']:.2g}")
    return row


# ============================================================================
# Part A: logit-space deltas + base-matched bins (H-A, H-D)
# ============================================================================

def part_a():
    print("\n=== PART A: paired deltas in LOGIT space (existing CSVs) ===")
    ids_a_bark = aug_file_ids(DATA / "eval_asv19_aug/manifest.csv")
    ids_a_real = aug_file_ids(DATA / "eval_asv19_augreal/manifest.csv")
    ids_m_bark = aug_file_ids(DATA / "eval_mlaad_aug/manifest.csv")
    ids_m_real = aug_file_ids(DATA / "eval_mlaad_augreal/manifest.csv")

    combos = [
        ("asv19 +bark", DET / "base_asv19.csv", DET / "aug_asv19.csv", ids_a_bark),
        ("asv19 +real", DET / "base_asv19.csv", DET / "augreal_asv19.csv", ids_a_real),
        ("mlaad_cal +bark", DET / "base_mlaad_cal.csv", DET / "aug_mlaad_cal.csv", ids_m_bark),
        ("mlaad_cal +real", DET / "base_mlaad_cal.csv", DET / "augreal_mlaad_cal.csv", ids_m_real),
    ]
    rows, cache = [], {}
    for name, b, a, ids in combos:
        dl, dp, bl, bp = paired_deltas(b, a, ids)
        rows.append(summarize(name, dl, dp, bl, bp))
        cache[name] = (dl, dp, bl, bp)
    write_csv(DET / "signflip_A_logit_deltas.csv", rows, list(rows[0].keys()))

    # base-score-matched bins: does mlaad's + sign persist where asv19's base
    # probs live (and vice versa)? Compare Bark condition of both models.
    print("\n  --- base-prob-matched bins (Bark condition) ---")
    bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]
    brow = []
    for lo, hi in bins:
        line = {"bin": f"[{lo},{hi})"}
        for name in ("asv19 +bark", "mlaad_cal +bark"):
            dl, dp, bl, bp = cache[name]
            m = (bp >= lo) & (bp < hi)
            line[f"{name} n"] = int(m.sum())
            line[f"{name} dlogit"] = float(dl[m].mean()) if m.any() else float("nan")
            line[f"{name} dprob"] = float(dp[m].mean()) if m.any() else float("nan")
        print("   ", line)
        brow.append(line)
    write_csv(DET / "signflip_A_matched_bins.csv", brow, list(brow[0].keys()))
    return cache


# ============================================================================
# Part C: pure stimuli through both models (H-C)
# ============================================================================

def _sample_wavs(dirpath, pattern, n, prefix_filter=None):
    files = sorted(Path(dirpath).glob(pattern))
    if prefix_filter:
        files = [f for f in files if f.name.startswith(prefix_filter)]
    if len(files) > n:
        idx = RNG.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def build_stimuli():
    """(stimulus_type, file_or_None, wav) list; wav already 3s-cropped/tiled."""
    stim = []

    def add_files(stype, files):
        for f in files:
            stim.append((stype, str(f), crop_or_tile_center(load_wav_mono_16k(str(f)))))

    add_files("laugh_bark", _sample_wavs(DATA / "laugh_bank_bark", "*.wav", 60))
    add_files("laugh_bark_probe", _sample_wavs(DATA / "bark_probe/laughter_only", "*.wav", 40))
    add_files("laugh_real_vocalsound", _sample_wavs(DATA / "laugh_inserts_real", "*.wav", 60))
    add_files("bona_asv19", _sample_wavs(DATA / "eval_asv19/wavs", "bona_*.wav", 50))
    add_files("spoof_asv19", _sample_wavs(DATA / "eval_asv19/wavs", "spoof_*.wav", 50))
    add_files("bona_libri", _sample_wavs(DATA / "eval_mlaad/wavs", "bona_*.wav", 50))
    add_files("spoof_mlaad", _sample_wavs(DATA / "eval_mlaad/wavs", "spoof_*.wav", 50))

    stim.append(("silence_zeros", "", np.zeros(TARGET_LEN, dtype=np.float32)))
    for i in range(5):  # near-silence: -60 dBFS noise (real recordings' "silence")
        stim.append(("silence_noise60db", f"noise{i}",
                     (RNG.standard_normal(TARGET_LEN) * 1e-3).astype(np.float32)))
    return stim


def score_stimuli(models, stim, device):
    rows = []
    for mname, lit in models.items():
        for i in range(0, len(stim), 8):
            chunk = stim[i:i + 8]
            res = score_batch(lit, [w for _, _, w in chunk], device)
            for (stype, f, _), (logit, prob) in zip(chunk, res):
                rows.append(dict(model=mname, stimulus=stype, file=f,
                                 logit=logit, prob=prob))
    return rows


def part_c(models, device):
    print("\n=== PART C: pure stimuli through both models ===")
    stim = build_stimuli()
    rows = score_stimuli(models, stim, device)
    write_csv(DET / "signflip_C_pure_stimuli.csv", rows,
              ["model", "stimulus", "file", "logit", "prob"])
    # summary table
    print(f"  {'stimulus':24s} {'asv19 logit(prob)':>22s} {'mlaad_cal logit(prob)':>24s}")
    stypes = sorted({r["stimulus"] for r in rows})
    for st in stypes:
        parts = []
        for m in ("asv19", "mlaad_cal"):
            sel = [r for r in rows if r["model"] == m and r["stimulus"] == st]
            L = np.array([r["logit"] for r in sel])
            P = np.array([r["prob"] for r in sel])
            parts.append(f"{L.mean():+7.2f} ({P.mean():.3f}) n={len(L)}")
        print(f"  {st:24s} {parts[0]:>22s} {parts[1]:>26s}")


# ============================================================================
# Part X: cross-model paired deltas (model vs dataset)
# ============================================================================

def score_set(lit, audio_dir, file_ids, device="cuda"):
    out = {}
    for i in range(0, len(file_ids), 8):
        chunk = file_ids[i:i + 8]
        wavs = [crop_or_tile_center(load_wav_mono_16k(str(Path(audio_dir) / f)))
                for f in chunk]
        res = score_batch(lit, wavs, device)
        for f, (logit, prob) in zip(chunk, res):
            out[f] = (logit, prob)
    return out


def part_x(models, device):
    print("\n=== PART X: cross-model paired deltas (sign follows model or data?) ===")
    jobs = [  # (model, base_dir, aug_dir, tag)
        ("mlaad_cal", DATA / "eval_asv19", DATA / "eval_asv19_aug", "mlaad_cal on asv19+bark"),
        ("asv19", DATA / "eval_mlaad", DATA / "eval_mlaad_aug", "asv19 on mlaad+bark"),
    ]
    rows, det_rows = [], []
    for mname, bdir, adir, tag in jobs:
        ids = aug_file_ids(adir / "manifest.csv")
        lit = models[mname]
        sb = score_set(lit, bdir, ids, device=device)
        sa = score_set(lit, adir, ids, device=device)
        dl = np.array([sa[f][0] - sb[f][0] for f in ids])
        dp = np.array([sa[f][1] - sb[f][1] for f in ids])
        bl = np.array([sb[f][0] for f in ids])
        bp = np.array([sb[f][1] for f in ids])
        rows.append(summarize(tag, dl, dp, bl, bp))
        for f in ids:
            det_rows.append(dict(tag=tag, file_id=f, base_logit=sb[f][0],
                                 aug_logit=sa[f][0], base_prob=sb[f][1],
                                 aug_prob=sa[f][1]))
    write_csv(DET / "signflip_X_crossmodel.csv", det_rows, list(det_rows[0].keys()))
    write_csv(DET / "signflip_X_summary.csv", rows, list(rows[0].keys()))


# ============================================================================
# Part B: frame-level window logits over the insert region (H-B)
# ============================================================================

def insert_span(row):
    ins, host = float(row["insert_dur_s"]), float(row["host_dur_s"])
    pos = row["position"]
    if pos == "start":
        return 0.0, ins
    if pos == "end":
        return host, host + ins
    return host / 2.0, host / 2.0 + ins  # mid: host[:mid]+insert+host[mid:]


def window_logits(lit, wav, device, window_s=1.0, hop_s=0.25):
    win = int(round(window_s * TARGET_SR))
    hop = int(round(hop_s * TARGET_SR))
    T = len(wav)
    starts = list(range(0, max(1, T - win + 1), hop))
    if starts[-1] + win < T:
        starts.append(max(0, T - win))
    centers, wavs = [], []
    for s in starts:
        e = min(T, s + win)
        wavs.append(crop_or_tile_center(wav[s:e], TARGET_LEN))
        centers.append((s + e) / 2.0 / TARGET_SR)
    logits = []
    for i in range(0, len(wavs), 16):
        logits.extend(l for l, _ in score_batch(lit, wavs[i:i + 16], device))
    return np.array(centers), np.array(logits)


def part_b(models, device, n_files=15):
    print("\n=== PART B: sliding-window logits inside vs outside insert region ===")
    rows = []
    for aset, adir in (("asv19+bark", DATA / "eval_asv19_aug"),
                       ("mlaad+bark", DATA / "eval_mlaad_aug")):
        man = [r for r in read_csv(adir / "manifest.csv")
               if r["augmented"] == "1" and float(r["insert_dur_s"]) >= 1.5]
        idx = RNG.choice(len(man), size=min(n_files, len(man)), replace=False)
        picks = [man[i] for i in sorted(idx)]
        for r in picks:
            wav = load_wav_mono_16k(str(adir / r["file"]))
            s0, s1 = insert_span(r)
            for mname, lit in models.items():
                cen, lg = window_logits(lit, wav, device)
                inside = (cen >= s0) & (cen <= s1)
                if inside.sum() == 0 or (~inside).sum() == 0:
                    continue
                rows.append(dict(
                    aug_set=aset, model=mname, file=r["file"], position=r["position"],
                    insert_s0=round(s0, 2), insert_s1=round(s1, 2),
                    n_in=int(inside.sum()), n_out=int((~inside).sum()),
                    logit_in=float(lg[inside].mean()),
                    logit_out=float(lg[~inside].mean()),
                    logit_in_minus_out=float(lg[inside].mean() - lg[~inside].mean()),
                ))
    write_csv(DET / "signflip_B_frame_logits.csv", rows, list(rows[0].keys()))
    print(f"  {'aug_set':12s} {'model':10s} {'mean logit IN':>14s} {'mean logit OUT':>15s} {'IN-OUT':>8s} n")
    for aset in ("asv19+bark", "mlaad+bark"):
        for m in ("asv19", "mlaad_cal"):
            sel = [r for r in rows if r["aug_set"] == aset and r["model"] == m]
            if not sel:
                continue
            li = np.mean([r["logit_in"] for r in sel])
            lo = np.mean([r["logit_out"] for r in sel])
            d = np.mean([r["logit_in_minus_out"] for r in sel])
            frac_neg = np.mean([r["logit_in_minus_out"] < 0 for r in sel])
            print(f"  {aset:12s} {m:10s} {li:+14.2f} {lo:+15.2f} {d:+8.2f} "
                  f"n={len(sel)} frac(IN<OUT)={frac_neg:.2f}")


# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ACXB", help="subset of A,C,X,B to run")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    if "A" in args.parts:
        part_a()

    need_gpu = any(p in args.parts for p in "CXB")
    models = {}
    if need_gpu:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        for name, ckpt in CKPTS.items():
            print(f"loading {name} on {device}: {ckpt}")
            models[name] = load_detector(ckpt, device)
    else:
        device = args.device or "cpu"

    if "C" in args.parts:
        part_c(models, device)
    if "X" in args.parts:
        part_x(models, device)
    if "B" in args.parts:
        part_b(models, device)


if __name__ == "__main__":
    main()
