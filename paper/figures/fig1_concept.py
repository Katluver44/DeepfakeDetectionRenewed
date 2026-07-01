"""Fig 1: schematic concept figure for the natural-synthetic axis lever.

LEFT: 2-D sketch of frozen WavLM-L12 embedding space with a bonafide cluster
and two synthetic-system clusters, showing the natural-synthetic axis w and
"easy" (tight-along-w) vs "hard" (spread-along-w) synthetic systems.

RIGHT: training-free fusion pipeline as a box/arrow diagram.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, hide_top_right, save_both, BLUE, VERMILLION,
                     GREEN, PINK, ORANGE, GRAY, LIGHT_GRAY, BLACK,
                     SINGLE_COL_WIDTH)

apply_style()
rng = np.random.default_rng(7)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

fig = plt.figure(figsize=(SINGLE_COL_WIDTH, 3.1))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)

# ---------------------------------------------------------------------------
# LEFT PANEL: embedding-space sketch
# ---------------------------------------------------------------------------
axL = fig.add_subplot(gs[0, 0])

bona_center = np.array([-1.6, 0.0])
easy_center = np.array([1.6, 1.0])
hard_center = np.array([1.4, -1.1])

n = 55
bona_pts = bona_center + rng.normal(scale=[0.42, 0.42], size=(n, 2))

# "easy" system: tight cluster elongated perpendicular to axis (tight ALONG axis)
theta = np.deg2rad(18)
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
easy_local = rng.normal(scale=[0.14, 0.5], size=(n, 2))  # narrow along axis (x), spread perpendicular (y)
easy_pts = easy_center + easy_local @ rot.T

# "hard" system: spread along the axis direction (wide along w)
hard_local = rng.normal(scale=[0.75, 0.42], size=(n, 2))
hard_pts = hard_center + hard_local @ rot.T

axL.scatter(bona_pts[:, 0], bona_pts[:, 1], s=10, color=BLUE, alpha=0.65,
            edgecolors="none", label="Bonafide")
axL.scatter(easy_pts[:, 0], easy_pts[:, 1], s=10, color=GREEN, alpha=0.7,
            edgecolors="none", label="Synthetic system A (easy)")
axL.scatter(hard_pts[:, 0], hard_pts[:, 1], s=10, color=VERMILLION, alpha=0.7,
            edgecolors="none", label="Synthetic system B (hard)")

for c, col in [(bona_center, BLUE), (easy_center, GREEN), (hard_center, VERMILLION)]:
    axL.scatter(*c, s=70, color=col, edgecolors="black", linewidths=1.0, zorder=5)

# natural-synthetic axis w: arrow from bonafide centroid toward spoof centroid
spoof_overall_center = (easy_center + hard_center) / 2
w_start = bona_center
w_end = bona_center + 1.28 * (spoof_overall_center - bona_center)
arrow = FancyArrowPatch(w_start, w_end, arrowstyle="-|>", mutation_scale=16,
                         linewidth=2.0, color=BLACK, zorder=6)
axL.add_patch(arrow)
mid = w_start + 0.62 * (w_end - w_start)
axL.annotate(r"axis $w$", xy=mid, xytext=(mid[0] - 0.05, mid[1] + 0.55),
             fontsize=10, fontweight="bold", ha="center")

# annotate tight/spread along w
axL.annotate("tight along $w$\n(easy)", xy=easy_center, xytext=(easy_center[0] + 0.15, easy_center[1] + 1.05),
             fontsize=8.3, ha="center", color=GREEN,
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))
axL.annotate("spread along $w$\n(hard)", xy=hard_center, xytext=(hard_center[0] + 0.55, hard_center[1] - 1.55),
             fontsize=8.3, ha="center", color=VERMILLION,
             arrowprops=dict(arrowstyle="-", color=VERMILLION, lw=0.8))

axL.set_xlim(-3.2, 3.6)
axL.set_ylim(-2.8, 2.6)
axL.set_xticks([])
axL.set_yticks([])
for s in axL.spines.values():
    s.set_visible(False)
axL.set_title("Frozen WavLM-L12 embedding space", fontsize=10.5)
axL.legend(loc="lower left", frameon=False, fontsize=7.6, handletextpad=0.3,
           borderaxespad=0.1, markerscale=1.3)

# ---------------------------------------------------------------------------
# RIGHT PANEL: training-free fusion pipeline
# ---------------------------------------------------------------------------
axR = fig.add_subplot(gs[0, 1])
axR.set_xlim(0, 10)
axR.set_ylim(0, 10)
axR.axis("off")


def box(ax, xy, w, h, text, fc="white", ec=BLACK, fontsize=8.6, lw=1.2, style="round,pad=0.05"):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                        edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=BLACK, lw=1.3, style="-|>", connectionstyle=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                         linewidth=lw, color=color, zorder=2,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


b_audio = box(axR, (0.3, 8.1), 2.6, 1.15, "Audio", fc="#F2F2F2")
b_wavlm = box(axR, (0.3, 5.9), 2.6, 1.35, "Frozen\nWavLM-L12", fc="#EAF3FB", ec=BLUE)

arrow(axR, (1.6, 8.1), (1.6, 7.25))

b_logit = box(axR, (3.7, 7.05), 3.1, 1.15, "Detector logit\n$z(\\mathrm{detector})$", fc="#FBEFE8", ec=ORANGE)
b_proj = box(axR, (3.7, 4.75), 3.1, 1.15, "Projection on\ninternal axis $w$", fc="#EAF6F1", ec=GREEN)

arrow(axR, (2.9, 6.85), (3.7, 7.6))
arrow(axR, (2.9, 6.35), (3.7, 5.3))

b_fuse = box(axR, (7.3, 5.85), 2.45, 1.45, "$s_{\\mathrm{fused}} = z(\\mathrm{detector})$\n$+\\ \\lambda\\, z(\\mathrm{axis\\ proj})$",
             fc="#F6F0F5", ec=PINK, fontsize=8.0)

arrow(axR, (6.8, 7.6), (7.3, 6.75))
arrow(axR, (6.8, 5.3), (7.3, 6.15))

axR.text(5.0, 3.55, "no retraining, no labels", ha="center", va="center",
         fontsize=9.2, fontstyle="italic", color=GRAY,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFAFA", edgecolor=LIGHT_GRAY))
arrow(axR, (8.5, 5.85), (6.9, 3.9), connectionstyle="arc3,rad=-0.25", color=GRAY, lw=1.0, style="-")

axR.set_title("Training-free fusion pipeline", fontsize=10.5)

fig.suptitle("The natural-synthetic axis: a training-free lever for deepfake-speech detection",
             fontsize=10.8, y=1.03)

paths = save_both(fig, OUT_DIR, "fig1_concept")
print("Saved:", paths)
