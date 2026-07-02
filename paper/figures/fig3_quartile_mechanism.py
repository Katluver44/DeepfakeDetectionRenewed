"""Fig 3 (new): the axis-fusion gain lands on exactly the hardest systems.

The training-free axis fusion is not a uniform boost: on MLAAD (i7) the AUC
improvement grows monotonically with detector-difficulty quartile, from
+0.036 on the easiest quartile to +0.220 on the hardest. This is the
law-as-mechanism plot: the axis carries information precisely where the
detector fails, which is *why* fusion helps (and why it is not tuning).

Data artifact (never hand-typed):
  experiments/results/i7_axis_fusion/i7_quartiles.csv
      Q1_easy 0.9495->0.9851 (+0.036); Q2 0.8705->0.9522 (+0.082);
      Q3 0.7687->0.9050 (+0.136); Q4_hard 0.6167->0.8371 (+0.220)
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save_both, BLUE, VERMILLION, GRAY, BLACK,
                    LIGHT_GRAY, SINGLE_COL_WIDTH)

apply_style()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))

rows = list(csv.DictReader(open(os.path.join(
    REPO, "experiments/results/i7_axis_fusion/i7_quartiles.csv"))))
labels = ["Q1\neasiest", "Q2", "Q3", "Q4\nhardest"]
det = [float(r["AUC_detector"]) for r in rows]
fus = [float(r["AUC_fused"]) for r in rows]
dlt = [float(r["delta"]) for r in rows]

assert abs(dlt[0] - 0.0356) < 1e-3 and abs(dlt[-1] - 0.2203) < 1e-3, dlt
print("[fig3] dAUC by quartile:", [f"{d:+.3f}" for d in dlt])

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.66, 3.2))
x = np.arange(len(rows))

# detector AUC as base, fusion gain stacked on top in vermillion
ax.bar(x, det, 0.62, color=LIGHT_GRAY, edgecolor=BLACK, lw=0.7,
       label="detector AUC", zorder=3)
ax.bar(x, dlt, 0.62, bottom=det, color=VERMILLION, edgecolor=BLACK, lw=0.7,
       label="gain from axis fusion", zorder=3)
for k in range(len(rows)):
    ax.annotate(f"+{dlt[k]:.3f}", xy=(x[k], det[k] + dlt[k]),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8.0, fontweight="bold", color=VERMILLION)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.6)
ax.set_xlabel("detector-difficulty quartile (MLAAD systems)")
ax.set_ylabel("AUC (bona vs system)")
ax.set_ylim(0.55, 1.02)
ax.axhline(1.0, color=GRAY, lw=0.6, ls=":", zorder=1)
ax.legend(frameon=False, loc="lower left", fontsize=8.4)
ax.set_title("Fusion gain concentrates on the hardest systems",
             fontsize=9.6, pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
paths = save_both(fig, OUT_DIR, "fig3_quartile_mechanism")
print("Saved:", paths)
