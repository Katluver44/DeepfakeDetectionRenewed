#!/usr/bin/env python3
"""
p5_system_hardness_reweight.py
==============================
P5 ablation: system-level C/T-stratified loss reweighting.

Standard per-utterance BCE treats every system equally, under-weighting the
hard (compact-but-bursty) systems that dominate residual EER. This fine-tunes
robust_goat with the per-utterance classification loss scaled by a per-system
hardness weight derived from C (=−rog@L12) and T (=vel_entropy@L9).

Weight construction (from outputs/sig_layer_features.csv):
    hardness(s) = zscore(C_s) + zscore(T_s)          # higher = harder
    w_s = exp(beta · (hardness_s − max hardness))     # softmax-shaped, ≤ 1
    w_s = w_s / mean(w_s)                              # mean-normalized to ~1
    w_s = clip(w_s, max = max_weight)                 # cap at 3× mean (stability)
Bonafide samples and systems absent from the table keep weight 1.0.

Pre-registered criteria (on the hard-system buckets, Q4 of C and of T):
    Improved:  ΔEER(C-Q4) ≤ −0.02 AND ΔEER(overall) within ±0.02
    Neutral:   ΔEER(C-Q4) in (−0.02, +0.02)
    Degraded:  ΔEER(C-Q4) > +0.02  OR ΔEER(overall) > +0.03

Outputs: experiments/results/mlaad/p5_hardness_reweight/{summary.md,
         per_system_eer_seed*.csv, per_system_eer_mean.csv, system_weights.csv}

Usage:
    python p5_system_hardness_reweight.py --seeds 42 123 1024 --epochs 5 --beta 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _ablation_common as C
from _ablation_common import (
    PROJECT_ROOT, EXP_DIR, base_cfg, load_system_ct, load_lit_state_from_base,
    run_seeds, consistency_note,
)
from phoneme_GAT.modules_ablations import Phoneme_GAT_Reweight_lit

OUT_DIR = EXP_DIR / "results" / "mlaad" / "p5_hardness_reweight"


def build_system_weights(beta: float, max_weight: float) -> dict[str, float]:
    ct = load_system_ct()
    systems = list(ct)
    Cv = np.array([ct[s]["C"] for s in systems])   # −rog: higher = harder
    Tv = np.array([ct[s]["T"] for s in systems])   # vel-entropy: higher = harder
    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    hardness = z(Cv) + z(Tv)
    w = np.exp(beta * (hardness - hardness.max()))
    w = w / w.mean()
    w = np.clip(w, None, max_weight)
    return {s: float(w[i]) for i, s in enumerate(systems)}


def verdict(report: dict) -> str:
    q4 = report["C"][3]
    c_q4 = q4["delta_eer"]
    overall = report["overall"].get("delta_mean_eer", float("nan"))
    if not np.isnan(c_q4) and c_q4 <= -0.02 and abs(overall) <= 0.02:
        tag = "IMPROVED"
    elif not np.isnan(c_q4) and c_q4 > 0.02:
        tag = "DEGRADED"
    elif not np.isnan(overall) and overall > 0.03:
        tag = "DEGRADED (overall regression)"
    else:
        tag = "NEUTRAL"
    auc_d = q4.get("delta_auc", float("nan"))
    bal_d = q4.get("delta_bal_acc", float("nan"))
    return (f"**{tag}** — ΔEER(C-Q4 hard)={c_q4:+.4f}, ΔEER(overall)={overall:+.4f}\n"
            f"  hard-bucket corroboration: ΔAUC={auc_d:+.4f}, Δbal_acc={bal_d:+.4f}\n"
            f"  {consistency_note(report, 'C')}\n  {consistency_note(report, 'T')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 1024])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="softmax temperature on hardness; higher = sharper weighting")
    ap.add_argument("--max-weight", type=float, default=3.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = build_system_weights(args.beta, args.max_weight)

    # Persist the weight table for auditing.
    import pandas as pd
    ct = load_system_ct()
    pd.DataFrame([{"system": s, "weight": weights[s],
                   "C": ct[s]["C"], "T": ct[s]["T"]} for s in weights]
                 ).sort_values("weight", ascending=False
                 ).to_csv(OUT_DIR / "system_weights.csv", index=False)
    print(f"weights: min={min(weights.values()):.3f} max={max(weights.values()):.3f} "
          f"(n={len(weights)})  → {OUT_DIR/'system_weights.csv'}")

    def factory(seed):
        model = Phoneme_GAT_Reweight_lit(cfg=base_cfg())
        load_lit_state_from_base(model)
        model.configure_ablation(weights, max_weight=args.max_weight)
        return model

    run_seeds(factory, "p5_hardness_reweight", OUT_DIR,
              args.seeds, args.epochs, args.lr, criteria_fn=verdict)


if __name__ == "__main__":
    main()
