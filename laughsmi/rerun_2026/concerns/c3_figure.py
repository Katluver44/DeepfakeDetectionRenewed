"""C3 figure: is the laughter-insertion locality vulnerability + worst-window
defense architecture- and benchmark-independent?

Panel A: mean paired insertion score-shift direction (base->aug) per detector,
         expressed as rank-biserial effect size (scale-free, sign = direction).
         Negative = laughter pulls fakes toward bona-fide (the evasion direction).
Panel B: evasion rate at the clean-EER threshold, single-center-crop (the attack)
         vs worst-window max scoring (the defense), per detector. Scale-free.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = Path("/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi/rerun_2026/concerns")

# Okabe-Ito colorblind-safe
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"

# ---- gather numbers (from the CSVs / analyses just computed) ----
# detector: (label, in_domain_ok, clean_eer, rank_biserial, delta_mean_note,
#            evasion_single, evasion_worstwindow_max, defense_clean_eer)
detectors = [
    # ASV19 WavLM-GAT (reference, reproduces paper Fig 3)
    ("ASVspoof19\nWavLM-GAT\n(ref, in-domain)", True, 6.0, -0.682, 0.257, 0.04, 5.00),
    # AASIST second architecture, same ASV19 benchmark
    ("ASVspoof19\nAASIST\n(2nd arch, in-domain)", True, 1.0, -0.488, 0.057, 0.00, 2.00),
    # MLAAD WavLM-GAT second benchmark (DEGENERATE: clean EER 47%, out-of-domain)
    ("MLAAD-tiny\nWavLM-GAT\n(2nd bench, EER~47%)", False, 47.19, 0.646, 0.244, 0.09, 48.75),
]

labels = [d[0] for d in detectors]
ok = [d[1] for d in detectors]
eer = [d[2] for d in detectors]
rb = [d[3] for d in detectors]
ev_single = [d[4] for d in detectors]
ev_ww = [d[5] for d in detectors]

x = np.arange(len(detectors))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

# ---- Panel A: insertion direction (rank-biserial) ----
barcolors = [BLUE if o else GREY for o in ok]
bars = axA.bar(x, rb, color=barcolors, width=0.6, edgecolor="white", linewidth=1.2)
axA.axhline(0, color="#444444", lw=1)
axA.set_ylabel("Insertion effect (rank-biserial, base→aug)")
axA.set_title("A. Direction of laughter-insertion shift", fontsize=12, loc="left")
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=8.5)
axA.set_ylim(-0.9, 0.9)
for xi, v in zip(x, rb):
    axA.text(xi, v + (0.05 if v >= 0 else -0.08), f"{v:+.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
axA.text(0.02, -0.86, "↓ negative = laughter pulls fakes toward bona-fide (evasion direction)",
         fontsize=8, color=BLUE, transform=axA.get_yaxis_transform() if False else axA.transData)
axA.annotate("wrong sign: synthetic Bark laughter\nreads as spoof to a MLAAD-trained\ndetector (no dilution to repair)",
             xy=(2.0, 0.64), xytext=(0.75, 0.72), fontsize=7.5, color="#555555", ha="left",
             arrowprops=dict(arrowstyle="->", color=GREY))
for sp in ["top", "right"]:
    axA.spines[sp].set_visible(False)

# ---- Panel B: evasion single-crop vs worst-window ----
w = 0.36
b1 = axB.bar(x - w/2, ev_single, width=w, color=ORANGE, edgecolor="white",
             linewidth=1.2, label="single center-crop (attack)")
b2 = axB.bar(x + w/2, ev_ww, width=w, color=BLUE, edgecolor="white",
             linewidth=1.2, label="worst-window max (defense)")
axB.set_ylabel("Evasion rate at clean-EER threshold")
axB.set_title("B. Worst-window defense vs the insertion attack", fontsize=12, loc="left")
axB.set_xticks(x)
axB.set_xticklabels(labels, fontsize=8.5)
axB.set_ylim(0, 0.35)
for xi, v in zip(x - w/2, ev_single):
    axB.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in zip(x + w/2, ev_ww):
    axB.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
axB.legend(frameon=False, fontsize=9, loc="upper left")
# shade the degenerate detector
axB.axvspan(1.5, 2.5, color=GREY, alpha=0.08)
axB.text(2.0, 0.32, "degenerate\n(clean EER 47%,\nout-of-domain)", ha="center",
         fontsize=7.5, color=GREY)
for sp in ["top", "right"]:
    axB.spines[sp].set_visible(False)

fig.suptitle("C3 — Laughter-insertion locality: architecture & benchmark independence",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = C / "figures" / "c3_architecture_independence.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
