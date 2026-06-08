#!/usr/bin/env python3
"""
p6_diversity_regularization.py
==============================
P6 ablation: representation-diversity regularization on bonafide frames.

C measures compactness of the L12 frame cloud; hard (synthetic) systems sit on
a compact manifold. The direct fix is to train the encoder to produce MORE
spread-out representations for bonafide speech, widening the C-axis gap that the
classifier can exploit:

    total_loss += lambda_div · (−rog(bonafide encoder frames))

Implemented in Phoneme_GAT_Diversity_lit.calcuate_loss (gated on self.training).
Risk is that the regularizer fights BCE when compactness happens to be
discriminative, so lambda_div starts small (0.01) and is swept.

Pre-registered criteria (Q4 = hardest bucket on each axis):
    Improved:  ΔEER(C-Q4) ≤ −0.02 AND ΔEER(overall) ≤ +0.02
    Neutral:   |ΔEER(C-Q4)| < 0.02
    Degraded:  ΔEER(overall) > +0.03  (regularizer hurt discrimination)

Outputs: experiments/results/mlaad/p6_diversity_lambda{λ}/...

Usage:
    python p6_diversity_regularization.py --seeds 42 123 1024 --lambda-div 0.01
    python p6_diversity_regularization.py --lambda-div 0.05 --seeds 42   # sweep
"""
from __future__ import annotations

import argparse

import numpy as np

from _ablation_common import (
    EXP_DIR, base_cfg, load_lit_state_from_base, run_seeds, consistency_note,
)
from phoneme_GAT.modules_ablations import Phoneme_GAT_Diversity_lit


def verdict(report: dict) -> str:
    q4 = report["C"][3]
    c_q4 = q4["delta_eer"]
    overall = report["overall"].get("delta_mean_eer", float("nan"))
    if not np.isnan(c_q4) and c_q4 <= -0.02 and (np.isnan(overall) or overall <= 0.02):
        tag = "IMPROVED"
    elif not np.isnan(overall) and overall > 0.03:
        tag = "DEGRADED (overall regression)"
    elif not np.isnan(c_q4) and abs(c_q4) < 0.02:
        tag = "NEUTRAL"
    else:
        tag = "MIXED"
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
    ap.add_argument("--lambda-div", type=float, default=0.01)
    args = ap.parse_args()

    out_dir = EXP_DIR / "results" / "mlaad" / f"p6_diversity_lambda{args.lambda_div:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory(seed):
        model = Phoneme_GAT_Diversity_lit(cfg=base_cfg())
        load_lit_state_from_base(model)
        model.configure_ablation(lambda_div=args.lambda_div)
        return model

    run_seeds(factory, f"p6_diversity_lambda{args.lambda_div:g}", out_dir,
              args.seeds, args.epochs, args.lr, criteria_fn=verdict)


if __name__ == "__main__":
    main()
