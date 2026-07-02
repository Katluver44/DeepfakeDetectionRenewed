"""Fig 1 (concept, rebuilt): visualization of the natural-synthetic axis.

Rebuilt to fix the label overlaps in the earlier fig1_concept.png. Shows, in a
2-D sketch of frozen WavLM-L12 space:
  - the bona-fide cloud (blue) and its centroid,
  - the synthetic centroid, and the axis w = centroid(spoof)-centroid(bona),
  - an EASY system (green) whose projections are TIGHT along w (low sd_along),
  - a HARD system (orange) whose projections SPREAD across the boundary
    (high sd_along) -- some utterances land in the bona region and evade.

Illustrative geometry (not fit to a specific artifact); the empirical law is
Fig. 2a. Deterministic (fixed seed).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/paper/figures")
from _style import (apply_style, save_both, BLUE, VERMILLION, GREEN, GRAY,
                    BLACK, LIGHT_GRAY, SINGLE_COL_WIDTH)

apply_style()
OUT_DIR = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/final_submission/figs"
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.66, 3.0))

# axis direction w (unit), pointing lower-left(bona) -> upper-right(synthetic)
theta = np.deg2rad(22)
w = np.array([np.cos(theta), np.sin(theta)])

bona_c = np.array([0.0, 0.0])
synth_c = bona_c + 5.6 * w

# bona cloud
bona = bona_c + rng.normal(0, [0.85, 0.85], size=(70, 2))
ax.scatter(bona[:, 0], bona[:, 1], s=11, color=BLUE, alpha=0.55, lw=0, zorder=2)

# easy system: tight along w, offset off-axis (clearly synthetic)
easy_c = synth_c + np.array([-0.4, 1.9])
along = rng.normal(0, 0.35, 40)
orth = rng.normal(0, 0.32, 40)
easy = easy_c + np.outer(along, w) + np.outer(orth, [-w[1], w[0]])
ax.scatter(easy[:, 0], easy[:, 1], s=11, color=GREEN, alpha=0.7, lw=0, zorder=2)

# hard system: large spread ALONG w -> straddles the boundary toward bona
hard_c = synth_c + np.array([-0.9, -2.1])
along = rng.normal(0, 1.9, 46)
orth = rng.normal(0, 0.30, 46)
hard = hard_c + np.outer(along, w) + np.outer(orth, [-w[1], w[0]])
# keep the hard cloud clear of the bona-centroid label region
hard = hard[hard[:, 0] > 1.4]
ax.scatter(hard[:, 0], hard[:, 1], s=11, color=VERMILLION, alpha=0.7, lw=0, zorder=2)

# the axis w
arr = FancyArrowPatch(bona_c, synth_c, arrowstyle="-|>", mutation_scale=15,
                      lw=2.0, color=BLACK, zorder=4)
ax.add_patch(arr)
ax.text(*(bona_c + 2.9 * w + np.array([-0.1, 0.55])), r"axis $w$", fontsize=11,
        fontweight="bold", ha="center", rotation=22, rotation_mode="anchor")

# centroids
ax.scatter(*bona_c, s=70, color=BLUE, edgecolor=BLACK, lw=1.1, zorder=5)
ax.scatter(*synth_c, s=70, color=GRAY, edgecolor=BLACK, lw=1.1, zorder=5)
ax.text(bona_c[0] - 1.3, bona_c[1] - 0.2, "bona-fide\ncentroid", fontsize=8,
        ha="right", va="center", color=BLUE)
ax.text(synth_c[0] + 0.9, synth_c[1] + 0.05, "synthetic\ncentroid", fontsize=8,
        ha="left", va="center", color=GRAY)

# labels for the two systems, placed clear of points
ax.annotate("easy system\ntight along $w$", xy=(easy_c[0] + 0.6, easy_c[1] + 0.7),
            xytext=(easy_c[0] + 2.4, easy_c[1] + 1.0), fontsize=8.2, color=GREEN,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))
ax.annotate("hard system\nspreads across boundary",
            xy=(hard_c[0] + 1.9, hard_c[1] - 0.3),
            xytext=(hard_c[0] + 1.0, hard_c[1] - 2.4), fontsize=8.2,
            color=VERMILLION, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=VERMILLION, lw=0.8))

# a couple of "evading" hard utterances that fell toward the bona region
evaders = hard[np.argsort(hard @ w)[:3]]
ax.scatter(evaders[:, 0], evaders[:, 1], s=30, facecolor="none",
           edgecolor=VERMILLION, lw=1.3, zorder=6)
ax.annotate("evade", xy=(evaders[0, 0], evaders[0, 1]),
            xytext=(evaders[0, 0] - 0.3, evaders[0, 1] + 1.4), fontsize=7.6,
            color=VERMILLION, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=0.7))

ax.set_xlim(-4.6, 8.4)
ax.set_ylim(-4.6, 5.2)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Spread along a single axis separates easy from hard systems",
             fontsize=9.4, pad=6)

# small legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
              markersize=6, label="bona-fide"),
       Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
              markersize=6, label="easy synthetic system"),
       Line2D([0], [0], marker="o", color="none", markerfacecolor=VERMILLION,
              markersize=6, label="hard synthetic system")]
ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=7.6,
          handletextpad=0.3, borderpad=0.2)

fig.tight_layout()
paths = save_both(fig, OUT_DIR, "fig1_concept")
print("Saved:", paths)
