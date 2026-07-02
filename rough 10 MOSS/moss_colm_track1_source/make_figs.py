#!/usr/bin/env python3
"""MOSS@COLM figures: (1) redrawn WavLM-GAT architecture (original diagram, cited),
(2) 3-panel results: hardness law, axis rotation (MLAAD vs ASVspoof-2019 only),
headroom lever (weak vs full WavLM-GAT). No ITW / ASVspoof2021 content."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella"],
    "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Pagella",
    "mathtext.it": "TeX Gyre Pagella:italic", "mathtext.bf": "TeX Gyre Pagella:bold",
    "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#444",
    "xtick.color": "#333", "ytick.color": "#333", "axes.labelcolor": "#222",
    "text.color": "#222", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE = "#3b6fb0"; NAVY = "#14305a"; GREY = "#9aa0a6"; RED = "#c0392b"
FROZEN = "#d7dde3"; TRAIN = "#cfe0f2"; TRAIN_EDGE = "#3b6fb0"; FROZEN_EDGE = "#8a929b"
OUT = "figs/"

# ============================================================
# FIG 1 — WavLM-GAT architecture (original redrawn diagram)
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")

def box(y, label, kind, h=1.35, w=8.2, x=0.9):
    fc = FROZEN if kind == "frozen" else (TRAIN if kind == "train" else "#f4ede2")
    ec = FROZEN_EDGE if kind == "frozen" else (TRAIN_EDGE if kind == "train" else "#b58a4b")
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                       fc=fc, ec=ec, lw=1.1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)

def arrow(y0, y1, x=5.0):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color="#555"))

rows = [
    (17.9, "Raw audio  (16 kHz, 3 s)", "io"),
    (15.9, "WavLM encoder  [frozen]", "frozen"),
    (13.9, "CTC phoneme IDs  +  frame features", "frozen"),
    (11.9, "Adaptive phoneme pooling", "train"),
    (9.9,  "GAT  (3 layers, 6 heads, skip)", "train"),
    (7.9,  "BiLSTM  (2 layers)", "train"),
    (5.9,  "mean-pool over phonemes  +  L2", "train"),
    (3.9,  "Linear head", "train"),
    (1.9,  "spoof logit", "io"),
]
for (y, lab, kind) in rows:
    box(y, lab, kind)
for i in range(len(rows) - 1):
    arrow(rows[i][0], rows[i + 1][0] + 1.35)

# legend
ax.add_patch(Rectangle((0.9, 0.15), 0.6, 0.6, fc=FROZEN, ec=FROZEN_EDGE, lw=1))
ax.text(1.7, 0.45, "frozen", va="center", fontsize=7.5)
ax.add_patch(Rectangle((4.5, 0.15), 0.6, 0.6, fc=TRAIN, ec=TRAIN_EDGE, lw=1))
ax.text(5.3, 0.45, "trainable", va="center", fontsize=7.5)
fig.savefig(OUT + "fig1_architecture.pdf")
plt.close(fig)

# ============================================================
# FIG 2 — three results panels
# ============================================================
pts = json.load(open("fig1a_points.json"))
X = np.array(pts["x"]); Y = np.array(pts["y"])

fig = plt.figure(figsize=(11.0, 3.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.80, 1.05], wspace=0.40)

# (a) hardness law
axa = fig.add_subplot(gs[0, 0])
axa.scatter(X, Y, s=30, color=BLUE, alpha=0.78, edgecolors="white", linewidths=0.5, zorder=3)
m, b = np.polyfit(X, Y, 1); xs = np.array([X.min() - 0.1, X.max() + 0.1])
axa.plot(xs, m * xs + b, color=NAVY, lw=2.0, zorder=4)
axa.set_xlabel(r"spread along axis  $\mathrm{sd}_{\mathrm{along}}$")
axa.set_ylabel(r"hardness ($1-\mathrm{AUC}$)")
axa.set_xlim(0.6, 7.6); axa.set_ylim(-0.02, 0.63)
axa.set_xticks([2, 4, 6]); axa.set_yticks([0.0, 0.2, 0.4, 0.6])
axa.text(0.04, 0.96, r"$\rho = 0.60,\ p < 10^{-6}$", transform=axa.transAxes,
         va="top", ha="left", fontsize=9.5)
axa.set_title("(a) the hardness law (MLAAD, 61 systems)", fontsize=9.3, pad=5)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)

# (b) axis rotation: MLAAD vs ASVspoof-2019 only
axb = fig.add_subplot(gs[0, 1])
axb.axhspan(-0.03, 0.03, color=GREY, alpha=0.30, zorder=0)
axb.bar([0], [-0.08], width=0.5, color="#6f7174", edgecolor="#3d3f42", lw=0.7, zorder=3)
axb.axhline(0.93, ls="--", lw=1.4, color="#2c7a7b", zorder=2)
axb.axhline(0.0, lw=0.8, color="#444", zorder=2)
axb.set_ylim(-0.35, 1.08); axb.set_xlim(-0.7, 0.7)
axb.set_xticks([0]); axb.set_xticklabels(["MLAAD\n$\\leftrightarrow$ ASVspoof-2019"], fontsize=8)
axb.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axb.set_ylabel("cosine between axes")
axb.text(0.66, 0.945, "within-corpus\n$\\approx 0.93$", ha="right", va="bottom",
         fontsize=8, color="#2c7a7b")
axb.text(-0.66, 0.075, "chance", ha="left", va="bottom", fontsize=8, color="#5f6368")
axb.text(0.0, -0.14, "$-0.08$", ha="center", va="top", fontsize=8.2, color="#3d3f42")
axb.set_title("(b) the axis rotates across datasets", fontsize=9.3, pad=5)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

# (c) headroom lever
axc = fig.add_subplot(gs[0, 2])
cats = ["WavLM-GAT\n(500 utt)", "WavLM-GAT\n(full)", "WavLM-GAT\n(MLAAD)"]
base = [0.1438, 0.0781, 0.2721]; fused = [0.1138, 0.0777, 0.1627]
deltas = ["\u22120.030", "+0.000", "\u22120.109"]
sig = [True, False, True]
xx = np.arange(3); w = 0.38
axc.bar(xx - w / 2, base, w, color=GREY, edgecolor="#333", lw=0.5, label="detector alone", zorder=3)
axc.bar(xx + w / 2, fused, w, color=BLUE, edgecolor="#333", lw=0.5, label="+ axis fusion", zorder=3)
for i, d in enumerate(deltas):
    col = "#1f8a44" if sig[i] else "#888"
    axc.text(xx[i] + w / 2, fused[i] + 0.012, d, ha="center", va="bottom", fontsize=8, color=col)
axc.set_xticks(xx); axc.set_xticklabels(cats, fontsize=8)
axc.set_ylabel("EER"); axc.set_ylim(0, 0.32)
axc.legend(frameon=False, loc="upper right", fontsize=8)
axc.set_title("(c) the free lever pays off only with headroom", fontsize=9.3, pad=5)
for s in ("top", "right"):
    axc.spines[s].set_visible(False)

fig.savefig(OUT + "fig2_results.pdf")
plt.close(fig)
print("figures written")
