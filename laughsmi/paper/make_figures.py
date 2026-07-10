"""Publication figures for the LaugHSMI 2026 short paper.

Reads only existing result CSVs in laughsmi/{tables,detector_out}; writes
PDF+PNG into laughsmi/paper/figs/.
"""
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

L = Path(__file__).resolve().parents[1]
T = L / "tables"
D = L / "detector_out"
OUT = L / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference, light mode) --------------------------------
BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
SEQ = {250: "#86b6ef", 450: "#2a78d6", 650: "#104281"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLW = 3.33   # ACM \columnwidth in inches
TEXTW = 7.0   # ACM \textwidth


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", name)


# =============================================================================
# Figure 1 — acoustic taxonomy heatmap (Cliff's delta, synthetic vs real)
# =============================================================================
lit = pd.read_csv(T / "laughter_literature_features.csv")
GENS = ["bark_laughter_token", "bark_laughs_inline", "parler_tts", "xtts", "audioldm2"]
GEN_LAB = ["Bark\n(token)", "Bark\n(inline)", "Parler-\nTTS", "XTTS-v2", "Audio-\nLDM2"]
FEATS = [
    ("voiced_fraction", "Voiced fraction"),
    ("voiced_burst_mean_s", "Voiced-burst dur."),
    ("unvoiced_run_mean_s", "Unvoiced-run dur."),
    ("voicing_transition_rate_hz", "Voicing trans. rate"),
    ("harmonicity_acf_db", "Harmonicity (ACF)"),
    ("f0_mean_hz", "F0 mean"),
    ("f0_std_hz", "F0 variability"),
    ("spectral_cog_hz", "Spectral CoG"),
    ("bout_duration_s", "Bout duration"),
    ("inter_onset_cv", "Inter-onset CV"),
]

M = np.zeros((len(FEATS), len(GENS)))
P = np.ones_like(M)
for i, (f, _) in enumerate(FEATS):
    for j, g in enumerate(GENS):
        r = lit[(lit.generator == g) & (lit.feature == f)].iloc[0]
        M[i, j] = -r.cliffs_delta_real_gt_synth  # + => synthetic higher than real
        P[i, j] = r.p_value

cmap = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#5598e7", "#f0efec", "#ef9a99", "#b2312f"])
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig, ax = plt.subplots(figsize=(COLW, 2.9))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
# white cell separators
for k in range(1, len(GENS)):
    ax.axvline(k - 0.5, color="white", lw=1.2)
for k in range(1, len(FEATS)):
    ax.axhline(k - 0.5, color="white", lw=1.2)
ax.set_xticks(range(len(GENS)), GEN_LAB)
ax.set_yticks(range(len(FEATS)), [n for _, n in FEATS])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(FEATS)):
    for j in range(len(GENS)):
        v, p = M[i, j], P[i, j]
        dark = abs(v) > 0.55
        txt = f"{v:+.2f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=6.2, fontweight="bold" if p < 0.05 else "normal",
                color=("white" if dark else (INK if p < 0.05 else MUTED)))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=6.5, length=0, labelcolor=INK2)
cb.outline.set_visible(False)
cb.set_label("Cliff's $\\delta$ (synthetic $-$ real)", fontsize=7)
save(fig, "fig_taxonomy")

# =============================================================================
# Figure 2 — separability: (a) controlled AUC vs content-free floor,
#            (b) held-out-generator transfer by WavLM layer
# =============================================================================
GEN_SHORT = {"bark_laughter_token": "Bark (token)", "bark_laughs_inline": "Bark (inline)",
             "parler_tts": "Parler-TTS", "xtts": "XTTS-v2", "audioldm2": "AudioLDM2"}
d3 = pd.read_csv(T / "table_d3_fixed.csv").dropna(subset=["comparison"])
d3 = d3[d3.comparison.str.startswith("real vs ")]
d3["gen"] = d3.comparison.str.replace("real vs ", "")
d3 = d3.set_index("gen").loc[GENS]

held = pd.read_csv(T / "laughter_dynamics_wavlm_held_generator.csv")

fig, (a, b) = plt.subplots(
    2, 1, figsize=(COLW, 3.4), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.62})

y = np.arange(len(GENS))[::-1]
floor = d3.content_free.astype(float).values
wav = d3.wavlm_l12_pca20.astype(float).values
for yi, f0, w0 in zip(y, floor, wav):
    a.plot([f0, w0], [yi, yi], color=GRID, lw=1.6, zorder=1)
a.scatter(floor, y, s=26, color=MUTED, zorder=2)
a.scatter(wav, y, s=30, color=BLUE, zorder=3)
# direct labels above the first row's dots instead of a legend box
a.text(floor[0], y[0] + 0.34, "content-free floor", fontsize=6.4, color=INK2,
       ha="center", va="bottom")
a.text(wav[0], y[0] + 0.34, "WavLM L12+PCA", fontsize=6.4, color=BLUE,
       ha="right", va="bottom")
a.annotate("tie: confound-\nexplainable", xy=(wav[-1], y[-1] + 0.12),
           xytext=(0.76, y[-1] + 0.28), fontsize=6.2, color=INK2, va="center",
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
a.set_yticks(y, [GEN_SHORT[g] for g in GENS])
a.set_xlim(0.48, 1.02)
a.set_ylim(-0.55, 4.95)
a.axvline(0.5, color=BASE, lw=0.8, ls=(0, (3, 2)))
a.text(0.508, -0.42, "chance", fontsize=6, color=MUTED, va="bottom")
a.set_xlabel("AUC, real vs synthetic laughter (controlled clips)")
a.grid(axis="x", color=GRID, lw=0.5)
a.set_axisbelow(True)
a.spines[["top", "right", "left"]].set_visible(False)
a.tick_params(axis="y", length=0)
a.set_title("(a) Within-generator probe vs content-free baseline",
            loc="left", fontsize=7.5, color=INK, pad=4)

LAYERS = [0, 3, 12]
LCOL = {0: SEQ[250], 3: SEQ[450], 12: SEQ[650]}
off = {0: 0.22, 3: 0.0, 12: -0.22}
for lay in LAYERS:
    sub = held[held.wavlm_layer == lay].set_index("held_out_generator").loc[GENS]
    b.errorbar(sub.auc_mean, y + off[lay], xerr=sub.auc_std, fmt="o", ms=4.2,
               color=LCOL[lay], ecolor=LCOL[lay], elinewidth=1.0, capsize=1.6,
               capthick=1.0, label=f"layer {lay}", zorder=3)
b.set_yticks(y, [GEN_SHORT[g] for g in GENS])
b.axvline(0.5, color=BASE, lw=0.8, ls=(0, (3, 2)))
b.set_xlim(0.28, 1.02)
b.set_xlabel("AUC on held-out generator (mean $\\pm$ s.d., 50 resamples)")
b.grid(axis="x", color=GRID, lw=0.5)
b.set_axisbelow(True)
b.spines[["top", "right", "left"]].set_visible(False)
b.tick_params(axis="y", length=0)
b.legend(loc="upper left", frameon=False, borderpad=0.1, handletextpad=0.3,
         bbox_to_anchor=(0.0, 1.02), labelspacing=0.25)
b.set_title("(b) Transfer to a held-out generator, by WavLM layer",
            loc="left", fontsize=7.5, color=INK, pad=4)
save(fig, "fig_separability")

# =============================================================================
# Figure 3 — insertion effect on detector scores + test-time mitigation
# =============================================================================
def scores(fname, key="score"):
    with open(D / fname) as fh:
        return {Path(r["file_id"]).name: float(r[key]) for r in csv.DictReader(fh)}

base = scores("base_asv19.csv")
aug = scores("aug_asv19.csv")
augreal = scores("augreal_asv19.csv")
unm = scores("control_asv19.csv", "score_unmask")
sil = scores("control_asv19.csv", "score_silence")
man = pd.read_csv(L / "data" / "eval_asv19_aug" / "manifest.csv")
ids = [Path(f).name for f in man[man.augmented == 1].file]

conds = [
    ("Clean fake", [base[i] for i in ids], MUTED),
    ("+ synth.\nlaughter", [aug[i] for i in ids], RED),
    ("+ real\nlaughter", [augreal[i] for i in ids], YELLOW),
    ("silence\nregion", [sil[i] for i in ids], VIOLET),
    ("unmask\n(control)", [unm[i] for i in ids], AQUA),
]
THR = 0.3181

fig, (a, b) = plt.subplots(
    1, 2, figsize=(TEXTW, 2.15), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.28})

rng = np.random.default_rng(7)
for x, (lab, vals, col) in enumerate(conds):
    vals = np.asarray(vals)
    jit = rng.uniform(-0.16, 0.16, len(vals))
    a.scatter(x + jit, vals, s=5, color=col, alpha=0.45, lw=0, zorder=2)
    a.hlines(np.mean(vals), x - 0.26, x + 0.26, color=col, lw=1.8, zorder=3)
    a.text(x + 0.30, np.mean(vals), f"{np.mean(vals):.2f}", fontsize=6.5,
           va="center", color=INK)
a.axhline(THR, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
a.text(4.42, THR - 0.035, "EER threshold", fontsize=6.2, color=INK2,
       ha="right", va="top")
a.set_xticks(range(len(conds)), [c[0] for c in conds])
a.set_ylabel("Spoof score")
a.set_ylim(-0.04, 1.06)
a.set_xlim(-0.5, 4.55)
a.grid(axis="y", color=GRID, lw=0.5)
a.set_axisbelow(True)
a.spines[["top", "right"]].set_visible(False)
a.tick_params(axis="x", length=0)
a.set_title("(a) Detector scores on the same 105 fakes, per condition",
            loc="left", fontsize=7.5, color=INK, pad=4)

defense = [("Single\ncentre crop", 0.20, MUTED, "5.3"),
           ("Sliding\n+ mean", 0.029, SEQ[250], "3.7"),
           ("Sliding + max\n(worst window)", 0.010, SEQ[650], "4.3")]
xs = np.arange(len(defense))
ticklabs = []
for x, (lab, v, col, eer) in zip(xs, defense):
    b.bar(x, v * 100, width=0.58, color=col, zorder=2)
    b.text(x, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=7,
           color=INK, fontweight="bold")
    ticklabs.append(f"{lab}\nEER {eer}%")
b.set_xticks(xs, ticklabs)
for t in b.get_xticklabels():
    t.set_linespacing(1.35)
    t.set_fontsize(6.6)
b.set_ylabel("Evaded fakes (%)")
b.set_ylim(0, 23)
b.grid(axis="y", color=GRID, lw=0.5)
b.set_axisbelow(True)
b.spines[["top", "right"]].set_visible(False)
b.tick_params(axis="x", length=0)
b.set_title("(b) Laughter-augmented fakes that evade, by scoring",
            loc="left", fontsize=7.5, color=INK, pad=4)
save(fig, "fig_insertion")
print("done")
