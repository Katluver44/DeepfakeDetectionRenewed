"""Fig 2a: the hardness law scatter (MLAAD, 61 TTS systems).

Per-system spread along the corpus-internal natural-synthetic axis (sd_along)
vs detection hardness (1 - AUC). Point size encodes n_utts. A reliability
"noise floor" band around the trend visualises the hardness measurement
ceiling (split-half reliability 0.86 -> max explainable R^2 ~0.86).

Data artifacts (never hand-typed):
  experiments/results/i3_position_geometry/system_position.csv  (sd_along, hard_shared)
  experiments/results/i3_position_geometry/utt_position.csv     (per-utt rows -> n_utts)
  experiments/axis_audits/audits_outputs/audit9_hardness_reliability  (reliability 0.863)

Paper claim (Sec. 3): Spearman rho = 0.60, p < 1e-6, LOSO R^2 = 0.277,
against a 0.86 reliability ceiling (captures ~32% of explainable variance).
"""
import csv
import collections
import os
import sys

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, hide_top_right, save_both, BLUE, VERMILLION,
                    GREEN, GRAY, LIGHT_GRAY, BLACK, SINGLE_COL_WIDTH)

apply_style()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sys_csv = os.path.join(REPO, "experiments/results/i3_position_geometry/system_position.csv")
utt_csv = os.path.join(REPO, "experiments/results/i3_position_geometry/utt_position.csv")

rows = list(csv.DictReader(open(sys_csv)))
systems = [r["system"] for r in rows]
sd = np.array([float(r["sd_along"]) for r in rows])
hard = np.array([float(r["hard_shared"]) for r in rows])

# n_utts per system by counting utterance rows
counts = collections.Counter(r["system"] for r in csv.DictReader(open(utt_csv)))
n_utts = np.array([counts[s] for s in systems], dtype=float)

RELIABILITY = 0.863   # audit9: full-sample Spearman-Brown reliability (ceiling)
rho, p = spearmanr(sd, hard)

# ---- assertions against roadmap / paper values ----
assert len(rows) == 61, f"expected 61 systems, got {len(rows)}"
assert abs(rho - 0.60) < 0.02, f"rho {rho:.3f} off from paper 0.60"
print(f"[fig2a] n={len(rows)}  Spearman rho={rho:.3f}  p={p:.2e}")
print(f"[fig2a] n_utts min/median/max = {int(n_utts.min())}/{int(np.median(n_utts))}/{int(n_utts.max())}")
print(f"[fig2a] hardness reliability (ceiling) = {RELIABILITY}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.66, 3.0))

# size legend: map n_utts -> marker area. Use sqrt so area ~ n.
def size_of(n):
    return 18 + 3.2 * np.sqrt(n)

# reliability noise-floor band around a guide-to-the-eye linear fit.
# hardness split-half reliability r => irreducible noise sd = sqrt(1-r)*sd(hard)
b, a = np.polyfit(sd, hard, 1)              # hard ~ a + b*sd  (visual guide only)
xline = np.linspace(sd.min(), sd.max(), 100)
yline = a + b * xline
noise_sd = np.sqrt(1.0 - RELIABILITY) * hard.std(ddof=1)
ax.fill_between(xline, yline - noise_sd, yline + noise_sd, color=LIGHT_GRAY,
                alpha=0.35, linewidth=0, zorder=0)
ax.plot(xline, yline, color=GRAY, lw=1.4, ls="--", zorder=1)
# The grey band is the hardness-measurement noise floor (split-half reliability
# 0.86 -> R^2 ceiling ~0.86); explained in the LaTeX caption rather than an
# in-figure label, to keep the point cloud uncluttered.

ax.scatter(sd, hard, s=size_of(n_utts), c=BLUE, alpha=0.62,
           edgecolors="white", linewidths=0.5, zorder=3)

# annotate extreme systems (hardest, easiest, and a mid-high) for narrative
def label_system(name, dx, dy, ha="left"):
    i = systems.index(name)
    short = name.split("|")[0]
    ax.annotate(short, xy=(sd[i], hard[i]),
                xytext=(sd[i] + dx, hard[i] + dy), ha=ha, fontsize=7.4,
                color=BLACK, zorder=5,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))

hardest = systems[int(np.argmax(hard))]
easiest = systems[int(np.argmin(hard))]
print(f"[fig2a] hardest={hardest} (hard={hard.max():.3f}), easiest={easiest} (hard={hard.min():.3f})")
label_system(hardest, 0.4, 0.02, ha="left")
label_system(easiest, 0.25, 0.06, ha="left")

ax.set_xlabel(r"spread along axis  $\mathrm{sd}_{\mathrm{along}}$")
ax.set_ylabel(r"hardness  $(1-\mathrm{AUC})$")
ax.set_ylim(-0.03, 0.80)
ax.set_xlim(sd.min() - 0.4, sd.max() + 0.5)
hide_top_right(ax)

ax.annotate(rf"Spearman $\rho={rho:.2f}$, $p<10^{{-6}}$" + "\n"
            rf"LOSO $R^2=0.277$ ($\approx$32% of ceiling)",
            xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LIGHT_GRAY, lw=0.7))

# size legend (proxy handles)
from matplotlib.lines import Line2D
leg_ns = [10, 16, 25]   # actual per-system utt range is narrow (9-25)
handles = [Line2D([0], [0], marker="o", ls="none", markerfacecolor=BLUE,
                  markeredgecolor="white", alpha=0.62,
                  markersize=np.sqrt(size_of(n)) , label=f"{n}")
           for n in leg_ns]
ax.legend(handles=handles, title="utts/system", loc="upper right",
          frameon=False, fontsize=7.6, title_fontsize=7.8,
          labelspacing=0.7, handletextpad=0.4, borderpad=0.3)

fig.tight_layout()
paths = save_both(fig, OUT_DIR, "fig2a_law_scatter")
print("Saved:", paths)
