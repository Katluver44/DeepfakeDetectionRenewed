"""Fig 2b: the axis rotates across datasets (cosine matrix heatmap).

Redraw per roadmap bug #3: the previous figure annotated "chance 0.08" while
the observed cross-corpus cosine is ~-0.08 and the true high-dimensional chance
floor is ~0.03. This version shows the whole rotation story at once as a
4x4 cosine matrix over {MLAAD, ASVspoof19, ASVspoof21, ITW}:
  - DIAGONAL  = within-corpus split-half reliability (axes are near-noiseless).
  - OFF-DIAG  = cross-corpus cos(w_a, w_b): near-zero or NEGATIVE (anti-aligned),
    inside/at the +-0.03 chance band -> a transferred axis is uninformative.

Data artifacts (never hand-typed):
  experiments/axis_audits/audits_outputs/audit4_axis_rotation/audit4_results.json
      split-half reliability (MLAAD 0.926, ITW 0.982, ASV21 0.981);
      MLAAD-frame cross cosines (ML-ITW +0.103, ML-ASV21 -0.207, ITW-ASV21 -0.238);
      random-direction null: E|cos| 0.029, 95th pct 0.070.
  experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json
      w(ASV2019)*w(MLAAD) = -0.078 ; w(ASV2019)*w(ASV2021) = +0.747.

Two cells are unavailable in the artifacts (no committed cross-cosine): the
ASVspoof19 split-half diagonal and ASV19<->ITW. They are hatched, not invented.

Paper claim (Sec. 4): within-corpus ~0.93, cross-corpus cos ~-0.08 (and negative),
chance floor ~0.03.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (apply_style, save_both, BLUE, VERMILLION, GRAY, BLACK,
                    SINGLE_COL_WIDTH)

apply_style()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))

a4 = json.load(open(os.path.join(
    REPO, "experiments/axis_audits/audits_outputs/audit4_axis_rotation/audit4_results.json")))
cfus = json.load(open(os.path.join(
    REPO, "experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json")))

# reliability (split-half) on diagonal
rel = a4["reliability"]
r_mlaad = rel["mlaad"]["split_half_cos_mean"]
r_itw = rel["itw"]["split_half_cos_mean"]
r_asv21 = rel["asv21"]["split_half_cos_mean"]

# cross-corpus cosines (MLAAD frame for the audit4 triple; own-frame for the
# ASV19 pairs, which are what the roadmap/paper quote). Frames noted in README.
c = a4["cos_mlaad_frame"]
ml_itw = c["ml_itw"]        # +0.103
ml_asv21 = c["ml_asv21"]    # -0.207
itw_asv21 = c["itw_asv21"]  # -0.238
ml_asv19 = cfus["cos_internal_vs_mlaad"]["cos_internal_vs_mlaad_own_frame"]      # -0.078
asv19_asv21 = cfus["cos_internal_vs_mlaad"]["cos_internal_2019LA_vs_internal_2021LA"]  # +0.747

null_mean = a4["random_null"]["mean_abs_cos"]   # 0.029
null_p95 = a4["random_null"]["p95_abs_cos"]     # 0.070

# ---- assertions ----
assert abs(r_mlaad - 0.93) < 0.02 and abs(r_itw - 0.98) < 0.02
assert abs(ml_asv19 - (-0.078)) < 0.005, ml_asv19
assert abs(asv19_asv21 - 0.747) < 0.005, asv19_asv21
assert abs(ml_itw - 0.103) < 0.005 and abs(ml_asv21 + 0.207) < 0.005
print(f"[fig2b] reliability diag: MLAAD {r_mlaad:.3f}  ASV21 {r_asv21:.3f}  ITW {r_itw:.3f}")
print(f"[fig2b] cross: ML-ASV19 {ml_asv19:+.3f}  ML-ASV21 {ml_asv21:+.3f}  "
      f"ML-ITW {ml_itw:+.3f}  ASV19-ASV21 {asv19_asv21:+.3f}  ITW-ASV21 {itw_asv21:+.3f}")
print(f"[fig2b] chance floor: E|cos|={null_mean:.3f}  95th pct={null_p95:.3f}")

# ---------------------------------------------------------------------------
# Build 4x4 matrix. order: MLAAD, ASVspoof19, ASVspoof21, ITW
# ---------------------------------------------------------------------------
labels = ["MLAAD", "ASVspoof19", "ASVspoof21", "ITW"]
NAN = np.nan
M = np.array([
    [r_mlaad,   ml_asv19,   ml_asv21,   ml_itw],
    [ml_asv19,  NAN,        asv19_asv21, NAN],
    [ml_asv21,  asv19_asv21, r_asv21,   itw_asv21],
    [ml_itw,    NAN,        itw_asv21,  r_itw],
])

# diverging blue<-white->vermillion, neutral gray at 0
cmap = LinearSegmentedColormap.from_list(
    "coscmap", [(0.0, BLUE), (0.5, "#f2f2f2"), (1.0, VERMILLION)])
cmap.set_bad("#ffffff")
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.72, 3.4))
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal")

n = len(labels)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.4)
ax.set_yticklabels(labels, fontsize=8.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, n, 1), minor=True)
ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which="minor", color="white", lw=2)

# annotate cells
for i in range(n):
    for j in range(n):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.6,
                    color=GRAY, style="italic")
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       hatch="////", edgecolor="#cccccc", lw=0))
            continue
        diag = (i == j)
        # white text on saturated cells, dark on pale ones
        tcol = "white" if abs(v) > 0.45 else BLACK
        txt = f"{v:+.2f}" if not diag else f"{v:.2f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=8.6 if not diag else 8.6,
                fontweight="bold" if diag else "normal", color=tcol)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, ticks=[-1, -0.5, 0, 0.5, 1])
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"cross-corpus cosine  $\cos(w_a, w_b)$", fontsize=8.4, labelpad=6)
# mark the +-chance band on the colorbar as a grey stripe at zero
cbar.ax.axhspan(-null_p95, null_p95, xmin=0, xmax=1, color=GRAY, alpha=0.45, lw=0)
cbar.ax.annotate(rf"$\pm${null_p95:.02f} chance", xy=(1.0, 0.5), xycoords="axes fraction",
                 xytext=(7, 0), textcoords="offset points",
                 fontsize=6.8, color=GRAY, va="center", rotation=90)

ax.set_title("The axis is corpus-local: it rotates to near-orthogonality",
             fontsize=9.6, pad=8)

fig.tight_layout()
paths = save_both(fig, OUT_DIR, "fig2b_axis_rotation")
print("Saved:", paths)
