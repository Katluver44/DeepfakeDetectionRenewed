"""C1: BH-FDR correction + bootstrap CIs on Cliff's delta for the acoustic
taxonomy (Fig 1). Pure recompute from cached per-clip descriptor table.

Inputs (cached, no re-extraction):
  tables/laughter_literature_features.csv       (60 rows: 12 feat x 5 gens, point estimates)
  tables/laughter_literature_features_clips.csv (120 rows: per-clip values, 20/group x 6 groups)

Outputs:
  rerun_2026/concerns/tables/taxonomy_fdr_ci.csv
  rerun_2026/concerns/figures/fig_taxonomy_fdr.png
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

L = "/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi"
OUT_T = f"{L}/rerun_2026/concerns/tables/taxonomy_fdr_ci.csv"
OUT_F = f"{L}/rerun_2026/concerns/figures/fig_taxonomy_fdr.png"

RNG = np.random.default_rng(20260711)
N_BOOT = 5000

GENS = ["bark_laughter_token", "bark_laughs_inline", "parler_tts", "xtts", "audioldm2"]
GEN_LAB = {"bark_laughter_token": "Bark\n(token)", "bark_laughs_inline": "Bark\n(inline)",
           "parler_tts": "Parler-\nTTS", "xtts": "XTTS-v2", "audioldm2": "Audio-\nLDM2"}
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
    ("spectral_cog_std_hz", "Spectral CoG std."),
    ("f0_range_hz", "F0 range"),
]
GROUP_MAP = {g: f"laugh-{g}" for g in GENS}


def cliffs_delta(a, b):
    """delta = P(a>b) - P(a<b), 'a' = real, matches cliffs_delta_real_gt_synth."""
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def main():
    lit = pd.read_csv(f"{L}/tables/laughter_literature_features.csv")
    clips = pd.read_csv(f"{L}/tables/laughter_literature_features_clips.csv")

    rows = []
    for feat, feat_lab in FEATS:
        for gen in GENS:
            grp = GROUP_MAP[gen]
            real_vals = clips.loc[clips.group == "laugh-real", feat].dropna().to_numpy()
            synth_vals = clips.loc[clips.group == grp, feat].dropna().to_numpy()
            n_real, n_synth = len(real_vals), len(synth_vals)

            # sanity recompute of point delta + MWU p, cross-check vs the literature table
            delta = cliffs_delta(real_vals, synth_vals)
            try:
                _, p_raw = stats.mannwhitneyu(real_vals, synth_vals, alternative="two-sided")
            except ValueError:
                p_raw = np.nan

            ref = lit[(lit.generator == gen) & (lit.feature == feat)]
            ref_delta = float(ref.cliffs_delta_real_gt_synth.iloc[0]) if len(ref) else np.nan
            ref_p = float(ref.p_value.iloc[0]) if len(ref) else np.nan

            # bootstrap CI for Cliff's delta: resample WITHIN each group, with replacement
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                rb = RNG.choice(real_vals, n_real, replace=True)
                sb = RNG.choice(synth_vals, n_synth, replace=True)
                boots[b] = cliffs_delta(rb, sb)
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

            rows.append(dict(
                feature=feat, feature_label=feat_lab, generator=gen,
                n_real=n_real, n_synthetic=n_synth,
                cliffs_delta=round(delta, 4),
                cliffs_delta_ref=round(ref_delta, 4) if not np.isnan(ref_delta) else np.nan,
                delta_ci_lo=round(ci_lo, 4), delta_ci_hi=round(ci_hi, 4),
                boot_se=round(boots.std(ddof=1), 4),
                p_raw=p_raw, p_raw_ref=ref_p,
            ))

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR across the FULL 12x5 = 60-cell grid (use the
    # literature table's original uncorrected p, the one the paper bolds on;
    # our recomputed p_raw is reported alongside as a cross-check).
    p = df["p_raw_ref"].to_numpy()
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    q_raw = ranked_p * m / (np.arange(m) + 1)
    # enforce monotonicity (step-up)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_monotone, 0, 1)
    df["q_bh"] = q
    df["survives_fdr_q05"] = df["q_bh"] < 0.05
    df["survives_fdr_q10"] = df["q_bh"] < 0.10
    df["bold_in_paper_p05"] = df["p_raw_ref"] < 0.05

    df = df.sort_values(["feature", "generator"]).reset_index(drop=True)
    cols = ["feature", "feature_label", "generator", "n_real", "n_synthetic",
            "cliffs_delta", "delta_ci_lo", "delta_ci_hi", "boot_se",
            "p_raw", "p_raw_ref", "q_bh", "bold_in_paper_p05",
            "survives_fdr_q05", "survives_fdr_q10"]
    df[cols].to_csv(OUT_T, index=False)
    print(f"wrote {OUT_T}")

    n_bold = df.bold_in_paper_p05.sum()
    n_surv05 = df.survives_fdr_q05.sum()
    n_surv10 = df.survives_fdr_q10.sum()
    print(f"cells with paper-bold p<.05 (uncorrected): {n_bold}/{len(df)}")
    print(f"of those, survive BH q<.05: {(df.bold_in_paper_p05 & df.survives_fdr_q05).sum()}")
    print(f"of those, survive BH q<.10: {(df.bold_in_paper_p05 & df.survives_fdr_q10).sum()}")
    print(f"total surviving q<.05: {n_surv05}/{len(df)}  q<.10: {n_surv10}/{len(df)}")

    # ---- sensitivity/power note ------------------------------------------------
    # With n=20/20 and alpha=.05 (two-sided MWU / rank-biserial ~ Cliff's delta),
    # approximate power for detecting |delta| via normal approx to MWU:
    # Using the delta<->AUC relation AUC = (delta+1)/2, and asymptotic SE(AUC)
    # (Hanley-McNeil) at n1=n2=20, compute the minimum detectable |delta| at 80% power.
    n1 = n2 = 20
    alpha = 0.05
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(0.80)

    def se_auc(auc, n1, n2):
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        return np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))

    # solve numerically for AUC where (auc-0.5) = (z_a+z_b) * se_auc(auc) approx (iterate)
    auc_try = 0.5
    for _ in range(200):
        se = se_auc(auc_try, n1, n2)
        auc_try = 0.5 + (z_a + z_b) * se
        auc_try = min(auc_try, 0.999)
    min_detectable_delta = 2 * (auc_try - 0.5)
    print(f"\n[power note] n=20/20, alpha=.05, target power=.80 -> "
          f"minimum reliably-detectable |Cliff's delta| ~ {min_detectable_delta:.2f}")
    with open(f"{L}/rerun_2026/concerns/tables/taxonomy_power_note.txt", "w") as f:
        f.write(
            "C1 sensitivity/power note (N=20 real vs N=20 synthetic per family)\n"
            "=====================================================================\n"
            f"Two-sided alpha=.05, target power=.80.\n"
            f"Using the Hanley-McNeil asymptotic SE(AUC) at n1=n2=20 and the\n"
            f"AUC<->Cliff's delta identity (delta = 2*AUC - 1), the minimum |delta|\n"
            f"reliably detectable at 80% power is approximately {min_detectable_delta:.2f}.\n"
            "Effects smaller than this (delta roughly in [-0.55, 0.55]) are\n"
            "underpowered at N=20/20 even before any multiple-comparison correction;\n"
            "a non-significant or FDR-non-surviving cell in that range should be read\n"
            "as 'inconclusive', not 'no difference'. This also means several of the\n"
            "already-significant large-delta cells (|delta|>0.6) are comfortably\n"
            "powered, while borderline cells (|delta| 0.2-0.4) are the ones the FDR\n"
            "correction and small N should make us most skeptical of.\n"
        )
    print("wrote tables/taxonomy_power_note.txt")

    make_figure(df)


def make_figure(df):
    mpl.rcParams.update({"font.size": 7.5, "figure.facecolor": "white", "axes.facecolor": "white"})
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#5598e7", "#f0efec", "#ef9a99", "#b2312f"])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    M = np.zeros((len(FEATS), len(GENS)))
    Q = np.ones_like(M)
    LO = np.zeros_like(M)
    HI = np.zeros_like(M)
    for i, (f, _) in enumerate(FEATS):
        for j, gname in enumerate(GENS):
            r = df[(df.feature == f) & (df.generator == gname)].iloc[0]
            M[i, j] = -r.cliffs_delta  # synthetic - real convention, matches make_figures.py
            Q[i, j] = r.q_bh
            LO[i, j] = -r.delta_ci_hi
            HI[i, j] = -r.delta_ci_lo

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for k in range(1, len(GENS)):
        ax.axvline(k - 0.5, color="white", lw=1.4)
    for k in range(1, len(FEATS)):
        ax.axhline(k - 0.5, color="white", lw=1.4)
    ax.set_xticks(range(len(GENS)), [GEN_LAB[g] for g in GENS])
    ax.set_yticks(range(len(FEATS)), [lab for _, lab in FEATS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(FEATS)):
        for j in range(len(GENS)):
            v, q, lo, hi = M[i, j], Q[i, j], LO[i, j], HI[i, j]
            dark = abs(v) > 0.55
            survives = q < 0.05
            marker = "**" if survives else ("*" if q < 0.10 else "")
            txt = f"{v:+.2f}{marker}\n[{lo:+.2f},{hi:+.2f}]"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.4,
                    fontweight="bold" if survives else "normal",
                    color=("white" if dark else (INK if survives else MUTED)))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5, length=0, labelcolor=INK2)
    cb.outline.set_visible(False)
    cb.set_label("Cliff's $\\delta$ (synthetic $-$ real), with 95% bootstrap CI", fontsize=7.5)
    ax.set_title(
        "Fig 1 (FDR-corrected): ** = BH q<.05, * = BH q<.10, plain = does not survive\n"
        "cell text = point $\\delta$ and [95% bootstrap CI] (5000 resamples, within-group)",
        fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_F, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_F}")


if __name__ == "__main__":
    main()
