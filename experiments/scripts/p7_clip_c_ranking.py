#!/usr/bin/env python3
"""
p7_clip_c_ranking.py
====================
P7 ablation: CLIP-loss reformulation as a compactness-ordered ranking loss.

Augments the standard objective with a pairwise margin-ranking term over the
spoof samples in each batch: within a batch, for every ordered pair of spoof
utterances (i, j) with compactness_i > compactness_j (compactness = −rog@L12,
more compact = harder), the model is penalized when logit_i is not at least
`margin` above logit_j. This encodes C's mechanistic relationship with detection
difficulty directly into the training signal.

    total_loss += lambda_rank · mean_over_pairs[ relu(margin − (logit_i − logit_j)) ]

Per-sample compactness comes from phoneme_feat (L12), already in batch_res — no
extra forward pass. Implemented in Phoneme_GAT_CRank_lit (gated on self.training).

HIGH RISK / speculative per the roadmap: per-sample compactness within a batch is
a noisy proxy for system-level C, and ranking losses are unstable in small
batches. lambda_rank is kept small.

Pre-registered criteria (Q4 = hardest bucket):
    Improved:  ΔEER(C-Q4) ≤ −0.02 AND ΔEER(overall) ≤ +0.02
    Neutral:   |ΔEER(C-Q4)| < 0.02 AND |ΔEER(overall)| < 0.02
    Degraded:  ΔEER(overall) > +0.03  (ranking destabilized training)

Outputs: experiments/results/mlaad/p7_crank_lambda{λ}/...

Usage:
    python p7_clip_c_ranking.py --seeds 42 123 1024 --lambda-rank 0.1 --margin 0.0
"""
from __future__ import annotations

import argparse

import numpy as np

from _ablation_common import (
    EXP_DIR, base_cfg, load_lit_state_from_base, run_seeds, consistency_note,
)
from phoneme_GAT.modules_ablations import Phoneme_GAT_CRank_lit


def verdict(report: dict) -> str:
    q4 = report["C"][3]
    c_q4 = q4["delta_eer"]
    overall = report["overall"].get("delta_mean_eer", float("nan"))
    if not np.isnan(c_q4) and c_q4 <= -0.02 and (np.isnan(overall) or overall <= 0.02):
        tag = "IMPROVED"
    elif not np.isnan(overall) and overall > 0.03:
        tag = "DEGRADED (unstable / overall regression)"
    elif (not np.isnan(c_q4) and abs(c_q4) < 0.02
          and (np.isnan(overall) or abs(overall) < 0.02)):
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
    ap.add_argument("--lambda-rank", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = EXP_DIR / "results" / "mlaad" / f"p7_crank_lambda{args.lambda_rank:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory(seed):
        model = Phoneme_GAT_CRank_lit(cfg=base_cfg())
        load_lit_state_from_base(model)
        model.configure_ablation(lambda_rank=args.lambda_rank, margin=args.margin)
        return model

    run_seeds(factory, f"p7_crank_lambda{args.lambda_rank:g}", out_dir,
              args.seeds, args.epochs, args.lr, criteria_fn=verdict)


if __name__ == "__main__":
    main()
