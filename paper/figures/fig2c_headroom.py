"""Fig 2c: training-free axis fusion helps only under detector headroom.

Redraw per roadmap bug #4. The previous version put three bars labelled
"WavLM-GAT (500 utt)/(full)/(MLAAD)" on one axis that mixed detector strength
and corpus, with a y-axis reading EER but annotations reading dEER. This
version is explicit: three regimes, each a PAIRED bar (detector alone vs
+ axis fusion), y-axis = EER, dEER annotated with its 95% CI.

Regimes (left->right by remaining headroom):
  weak   : mini_goat (WavLM-GAT trained on 500 utts), ASVspoof2019-LA
  mid    : MLAAD in-domain WavLM-GAT (i7), far from ceiling
  strong : robust_goat (WavLM-GAT, full ASVspoof2019-LA), near ceiling

Data artifacts (never hand-typed):
  experiments/results/e_mini_goat_fusion/mini_goat_fusion_results.json
      weak: detector 0.14375 -> fused 0.11375, dEER -0.030 CI[-0.046,-0.013] p=0
  experiments/results/i7_axis_fusion/{i7_headline.csv,i7_stats.json}
      mid : detector mean(EER over 3 seeds) -> fused mean; dEER -0.109 CI[-0.129,-0.091]
  experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json
      strong: detector 0.078125 -> fused 0.07771, dEER -0.0004 CI[-0.0075,+0.0062] p=0.80

Note: the paper's strong-detector EER 0.078 is the mean over seeds and is
CORRECT (per-seed s1 is 0.071; the mean is what should be reported).
"""
import csv
import json
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

mg = json.load(open(os.path.join(
    REPO, "experiments/results/e_mini_goat_fusion/mini_goat_fusion_results.json")))
cf = json.load(open(os.path.join(
    REPO, "experiments/results/c_asvspoof_fusion/asvspoof_fusion_results.json")))
i7s = json.load(open(os.path.join(
    REPO, "experiments/results/i7_axis_fusion/i7_stats.json")))

# --- weak (mini_goat) ---
w_det = mg["mini_goat"]["detector_alone_eer"]
w_fus = mg["mini_goat"]["fused_eer"]
w_d = mg["mini_goat"]["dEER_mean"]
w_ci = mg["mini_goat"]["dEER_ci95"]

# --- strong (robust_goat, c_asvspoof) ---
s_det = cf["detector_alone_eer"]
s_fus = cf["fused_eer"]
s_d = cf["dEER_mean_over_seeds"]
s_ci = cf["dEER_bootstrap_ci95"]

# --- mid (MLAAD i7): mean EER over the 3 detector seeds vs fused seeds ---
det_eers, fus_eers = [], []
with open(os.path.join(REPO, "experiments/results/i7_axis_fusion/i7_headline.csv")) as fh:
    for row in csv.DictReader(fh):
        if row["scorer"] == "detector":
            det_eers.append(float(row["EER"]))
        elif row["scorer"] == "fused":
            fus_eers.append(float(row["EER"]))
m_det = float(np.mean(det_eers))
m_fus = float(np.mean(fus_eers))
m_d = i7s["dEER"]
m_ci = i7s["dEER_ci"]

# assertions against roadmap/paper values
assert abs(w_det - 0.14375) < 1e-4 and abs(w_fus - 0.11375) < 1e-4
assert abs(s_det - 0.078125) < 1e-4, s_det
assert abs(m_d - (-0.10942)) < 1e-3, m_d
print(f"[fig2c] weak   det {w_det:.3f} fus {w_fus:.3f}  dEER {w_d:+.3f} CI[{w_ci[0]:+.3f},{w_ci[1]:+.3f}]")
print(f"[fig2c] mid    det {m_det:.3f} fus {m_fus:.3f}  dEER {m_d:+.3f} CI[{m_ci[0]:+.3f},{m_ci[1]:+.3f}]")
print(f"[fig2c] strong det {s_det:.3f} fus {s_fus:.3f}  dEER {s_d:+.4f} CI[{s_ci[0]:+.4f},{s_ci[1]:+.4f}]")

regimes = [
    ("weak detector\n(500 utt, ASVspoof19)",  w_det, w_fus, w_d, w_ci),
    ("mid detector\n(MLAAD, in-domain)",       m_det, m_fus, m_d, m_ci),
    ("strong detector\n(full, near ceiling)",  s_det, s_fus, s_d, s_ci),
]

fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 0.78, 3.3))
x = np.arange(len(regimes))
bw = 0.36
for k, (name, det, fus, d, ci) in enumerate(regimes):
    ax.bar(x[k] - bw / 2, det, bw, color=LIGHT_GRAY, edgecolor=BLACK, lw=0.7,
           label="detector alone" if k == 0 else None, zorder=3)
    ax.bar(x[k] + bw / 2, fus, bw, color=BLUE, edgecolor=BLACK, lw=0.7,
           label="+ axis fusion" if k == 0 else None, zorder=3)
    # dEER annotation with CI, above the taller bar
    top = max(det, fus)
    sig = ci[1] < 0  # CI excludes 0 (improvement)
    txt = rf"$\Delta$EER {d:+.3f}"
    ax.annotate(txt, xy=(x[k], top), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.2, fontweight="bold" if sig else "normal",
                color=VERMILLION if sig else GRAY)
    ax.annotate(f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", xy=(x[k], top),
                xytext=(0, 1.5), textcoords="offset points", ha="center",
                fontsize=6.6, color=GRAY)

ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in regimes], fontsize=8.2)
ax.set_ylabel("equal-error rate (EER)")
ax.set_ylim(0, 0.34)
ax.legend(frameon=False, loc="upper right", fontsize=8.6)
ax.set_title("Training-free axis fusion pays off only with headroom",
             fontsize=9.6, pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
paths = save_both(fig, OUT_DIR, "fig2c_headroom")
print("Saved:", paths)
