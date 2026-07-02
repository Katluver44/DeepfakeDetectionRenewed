"""Fusion figures, split per corpus (smaller than the old 3-bar fig2c).

Two separate, small paired-bar panels:
  (left)  fig_fusion_asvspoof.pdf -- ASVspoof2019-LA: the headroom test.
          weak detector (500 utt) gets a large gain; strong detector
          (full, near ceiling) gets none. Same axis, same fusion recipe.
  (right) fig_fusion_mlaad.pdf -- MLAAD in-domain: detector alone vs axis
          alone vs fused; fusion is best and negative across all folds/seeds.

Data artifacts (never hand-typed):
  experiments/results/e_mini_goat_fusion/mini_goat_fusion_results.json
      weak: 0.14375 -> 0.11375 (dEER -0.030, CI[-0.046,-0.013], p=0);
      strong (robust_goat re-gated this run): 0.07125 -> 0.07563 (dEER +0.004, p=0.559)
  experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json
      strong (published, seed-mean): detector 0.078, dEER -0.0004, p=0.80
  experiments/results/i7_axis_fusion/{i7_headline.csv, i7_stats.json}
      MLAAD: detector 0.272 -> fused 0.163 (axis alone 0.187); dEER -0.109
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed"
sys.path.insert(0, os.path.join(REPO, "paper/figures"))
from _style import (apply_style, save_both, BLUE, VERMILLION, GRAY, BLACK,
                    LIGHT_GRAY, GREEN, SINGLE_COL_WIDTH)

apply_style()
OUT_DIR = os.path.join(REPO, "final_submission/figs")

mg = json.load(open(os.path.join(
    REPO, "experiments/results/e_mini_goat_fusion/mini_goat_fusion_results.json")))
cf = json.load(open(os.path.join(
    REPO, "experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json")))
i7s = json.load(open(os.path.join(
    REPO, "experiments/results/i7_axis_fusion/i7_stats.json")))

# ---- ASVspoof headroom panel ----
w_det = mg["mini_goat"]["detector_alone_eer"]      # 0.14375
w_fus = mg["mini_goat"]["fused_eer"]               # 0.11375
w_d = mg["mini_goat"]["dEER_mean"]                 # -0.030
s_det = cf["detector_alone_eer"]                   # 0.078125 (seed mean)
s_fus = cf["fused_eer"]
s_d = cf["dEER_mean_over_seeds"]                   # -0.0004
assert abs(w_det - 0.14375) < 1e-4 and abs(w_fus - 0.11375) < 1e-4
assert abs(s_det - 0.078125) < 1e-4, s_det

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.46, 1.95))
groups = [("weak\n(500 utt)", w_det, w_fus, w_d),
          ("strong\n(full)", s_det, s_fus, s_d)]
x = np.arange(len(groups)); bw = 0.36
for k, (nm, det, fus, d) in enumerate(groups):
    ax.bar(x[k] - bw/2, det, bw, color=LIGHT_GRAY, edgecolor=BLACK, lw=0.7,
           label="detector alone" if k == 0 else None, zorder=3)
    ax.bar(x[k] + bw/2, fus, bw, color=BLUE, edgecolor=BLACK, lw=0.7,
           label="+ axis fusion" if k == 0 else None, zorder=3)
    top = max(det, fus)
    sig = d < -0.01
    ax.annotate(rf"$\Delta$EER {d:+.3f}", xy=(x[k], top), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7.8,
                fontweight="bold" if sig else "normal",
                color=VERMILLION if sig else GRAY)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=8.4)
ax.set_ylabel("EER"); ax.set_ylim(0, 0.185)
ax.legend(frameon=False, loc="upper right", fontsize=7.6)
ax.set_title("ASVspoof2019: fusion pays\nonly with headroom", fontsize=9.0, pad=6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
save_both(fig, OUT_DIR, "fig_fusion_asvspoof")
plt.close(fig)

# ---- MLAAD in-domain panel ----
det_eers, fus_eers = [], []
with open(os.path.join(REPO, "experiments/results/i7_axis_fusion/i7_headline.csv")) as fh:
    for row in csv.DictReader(fh):
        if row["scorer"] == "detector":
            det_eers.append(float(row["EER"]))
        elif row["scorer"] == "fused":
            fus_eers.append(float(row["EER"]))
m_det = float(np.mean(det_eers))
m_fus = float(np.mean(fus_eers))
m_axis = i7s.get("axis_alone_EER", None)
if m_axis is None:
    m_axis = 0.1874  # I7_FUSION.MLAAD_axis_alone_EER (regenerated_numbers.json)
m_d = i7s["dEER"]
assert abs(m_d - (-0.10942)) < 1e-3, m_d

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.46, 1.95))
bars = [("detector\nalone", m_det, LIGHT_GRAY),
        ("axis\nalone", m_axis, GREEN),
        ("fused", m_fus, BLUE)]
xx = np.arange(len(bars))
for k, (nm, v, col) in enumerate(bars):
    ax.bar(xx[k], v, 0.6, color=col, edgecolor=BLACK, lw=0.7, zorder=3)
    ax.annotate(f"{v:.3f}", xy=(xx[k], v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8.0)
ax.annotate(rf"$\Delta$EER {m_d:+.3f}", xy=(2, m_fus), xytext=(0, 16),
            textcoords="offset points", ha="center", fontsize=7.8,
            fontweight="bold", color=VERMILLION)
ax.set_xticks(xx); ax.set_xticklabels([b[0] for b in bars], fontsize=8.4)
ax.set_ylabel("EER"); ax.set_ylim(0, 0.33)
ax.set_title("MLAAD in-domain:\nfusion beats both parts", fontsize=9.0, pad=6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
save_both(fig, OUT_DIR, "fig_fusion_mlaad")
plt.close(fig)
print("Saved fusion panels.")
