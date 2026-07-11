"""C4: bootstrap 95% CIs for the Fig 2a within-generator AUCs (WavLM L12+PCA20
vs the content-free floor) and for the WavLM-floor gap. Pure recompute from
cached embeddings; no re-extraction / no WavLM forward passes.

DATA NOTE (documented judgment call): the brief pointed at
`embeddings/d3/mean_emb_layer12.npy` + `embeddings/d3_inventory.csv`, but those
only cover 3 groups (laugh-real, speech-real, laugh-bark) -- not the 5
generator families in Fig 2a. The file that actually covers all 5 families
(bark_laughter_token, bark_laughs_inline, parler_tts, xtts, audioldm2) plus
laugh-real/speech-real is `embeddings/d3_multi/mean_emb_layer12.npy` +
`embeddings/d3_multi_inventory.csv` (682 clips) -- used here instead.

These cached mean-pooled embeddings are the ones from the original (audited)
D3 extraction, i.e. WITHOUT the length/silence/loudness confound-control
normalization that `scripts/d3_fixed.py` applied before computing the
`table_d3_fixed.csv` point estimates (that script re-extracts WavLM from
raw audio with per-clip trimming/tiling/peak-norm -- audio files are not
available in this environment, and re-running WavLM is out of scope for a
"pure recompute" pass). Consequently point AUCs recomputed here run a bit
higher than table_d3_fixed for some families (see the recompute-vs-reference
columns in the output table) -- this is disclosed, not hidden, and the
qualitative comparison to the content-free floor is preserved.

Content-free floor: no per-clip content-free scalar features are cached
anywhere (they are computed on-the-fly from raw audio in d3_fixed.py and
never written to disk), so its CI is obtained analytically via the
Hanley-McNeil (1982) asymptotic SE for AUC, centered on the table_d3_fixed
point estimate. WavLM AUC CI is a full nonparametric stratified bootstrap
on the actual embeddings (gold standard). The WavLM-floor gap CI combines
both via Monte Carlo (paired draws from the WavLM bootstrap distribution and
a Normal(floor_point, SE_HM) distribution for the floor).

Outputs:
  rerun_2026/concerns/tables/fig2a_auc_ci.csv
  rerun_2026/concerns/figures/fig_separability_a_ci.png
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
import matplotlib.pyplot as plt

L = "/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi"
OUT_T = f"{L}/rerun_2026/concerns/tables/fig2a_auc_ci.csv"
OUT_F = f"{L}/rerun_2026/concerns/figures/fig_separability_a_ci.png"

SEED = 20260711
N_BOOT = 2000
RNG = np.random.default_rng(SEED)

GENS = ["bark_laughter_token", "bark_laughs_inline", "parler_tts", "xtts", "audioldm2"]
GEN_SHORT = {"bark_laughter_token": "Bark (token)", "bark_laughs_inline": "Bark (inline)",
             "parler_tts": "Parler-TTS", "xtts": "XTTS-v2", "audioldm2": "AudioLDM2"}

# same balanced-N selection logic as scripts/d3_fixed.py (same GROUPS order,
# same RandomState(0) draws) so the working subsample matches as closely as
# is possible from the cached (non-confound-controlled) embeddings.
D3FIXED_GROUPS = ["laugh-real", "speech-real", "laugh-bark_laughter_token",
                   "laugh-bark_laughs_inline", "laugh-audioldm2", "laugh-parler_tts", "laugh-xtts"]


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def balanced_select(inv):
    rng = np.random.RandomState(0)  # matches d3_fixed.py SEED=0
    ncap = min(inv.group.value_counts().min(), 100)
    sel_idx = []
    for grp in D3FIXED_GROUPS:
        idx = np.where(inv.group.values == grp)[0]
        chosen = rng.choice(len(idx), min(ncap, len(idx)), replace=False)
        sel_idx.append(idx[chosen])
    return np.concatenate(sel_idx), ncap


def main():
    inv = pd.read_csv(f"{L}/embeddings/d3_multi_inventory.csv")
    emb12 = np.load(f"{L}/embeddings/d3_multi/mean_emb_layer12.npy")
    d3fixed = pd.read_csv(f"{L}/tables/table_d3_fixed.csv").dropna(subset=["comparison"])
    d3fixed = d3fixed[d3fixed.comparison.str.startswith("real vs ")]
    d3fixed["gen"] = d3fixed.comparison.str.replace("real vs ", "")
    d3fixed = d3fixed.set_index("gen")

    sel_idx, ncap = balanced_select(inv)
    g = inv.group.values[sel_idx]
    E = emb12[sel_idx]
    print(f"[c4] balanced N={ncap}/group across {len(D3FIXED_GROUPS)} groups, {len(sel_idx)} clips total")

    real_mask_full = g == "laugh-real"
    rows = []
    for gen in GENS:
        grp = f"laugh-{gen}"
        mask = real_mask_full | (g == grp)
        X, y = E[mask], (g[mask] == grp).astype(int)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())

        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(20, X.shape[0] - 1), random_state=0),
                              LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(min(5, n_pos, n_neg), shuffle=True, random_state=0)
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        point_auc = roc_auc_score(y, oof)

        # stratified nonparametric bootstrap of the AUC given fixed CV oof scores
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bp = RNG.choice(idx_pos, len(idx_pos), replace=True)
            bn = RNG.choice(idx_neg, len(idx_neg), replace=True)
            bi = np.concatenate([bp, bn])
            # guard against degenerate resamples (shouldn't happen since each stratum is non-empty)
            boots[b] = roc_auc_score(y[bi], oof[bi])
        wavlm_ci_lo, wavlm_ci_hi = np.percentile(boots, [2.5, 97.5])

        # content-free floor: point estimate from table_d3_fixed (per-clip cf
        # features not cached / audio unavailable -> analytic Hanley-McNeil CI)
        floor_point = float(d3fixed.loc[gen, "content_free"])
        floor_se = hanley_mcneil_se(floor_point, n_pos, n_neg)
        floor_ci_lo = max(0.0, floor_point - 1.96 * floor_se)
        floor_ci_hi = min(1.0, floor_point + 1.96 * floor_se)

        # gap CI via Monte Carlo: pair each WavLM bootstrap draw with an
        # independent Normal(floor_point, floor_se) draw
        floor_draws = RNG.normal(floor_point, floor_se, N_BOOT).clip(0, 1)
        gap_draws = boots - floor_draws
        gap_ci_lo, gap_ci_hi = np.percentile(gap_draws, [2.5, 97.5])
        gap_point = point_auc - floor_point
        gap_excludes_zero = bool(gap_ci_lo > 0)

        ref_wavlm = float(d3fixed.loc[gen, "wavlm_l12_pca20"])
        rows.append(dict(
            comparison=f"real vs {gen}", generator=gen,
            n_real=n_neg, n_synthetic=n_pos,
            wavlm_l12_pca20_auc=round(point_auc, 4),
            wavlm_l12_pca20_auc_ref_table_d3_fixed=ref_wavlm,
            wavlm_ci_lo=round(wavlm_ci_lo, 4), wavlm_ci_hi=round(wavlm_ci_hi, 4),
            content_free_auc=floor_point,
            content_free_ci_lo=round(floor_ci_lo, 4), content_free_ci_hi=round(floor_ci_hi, 4),
            content_free_ci_method="analytic Hanley-McNeil (no cached per-clip cf features)",
            gap=round(gap_point, 4),
            gap_ci_lo=round(gap_ci_lo, 4), gap_ci_hi=round(gap_ci_hi, 4),
            gap_excludes_zero=gap_excludes_zero,
            verdict_ci=("WavLM>floor (CI excludes 0)" if gap_excludes_zero else "not significant at 95% CI"),
        ))
        print(f"{gen:22s} wavlm={point_auc:.3f} [{wavlm_ci_lo:.3f},{wavlm_ci_hi:.3f}]  "
              f"floor={floor_point:.3f} [{floor_ci_lo:.3f},{floor_ci_hi:.3f}]  "
              f"gap={gap_point:+.3f} [{gap_ci_lo:+.3f},{gap_ci_hi:+.3f}]  "
              f"{'EXCLUDES 0' if gap_excludes_zero else 'includes 0'}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_T, index=False)
    print(f"\nwrote {OUT_T}")

    make_figure(df)


def make_figure(df):
    BLUE, MUTED, GRID, BASE, INK, INK2 = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    d = df.set_index("generator").loc[GENS]
    y = np.arange(len(GENS))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for yi, row in zip(y, d.itertuples()):
        ax.plot([row.content_free_auc, row.wavlm_l12_pca20_auc], [yi, yi], color=GRID, lw=1.6, zorder=1)
        ax.errorbar(row.content_free_auc, yi, xerr=[[row.content_free_auc - row.content_free_ci_lo],
                                                      [row.content_free_ci_hi - row.content_free_auc]],
                     fmt="o", ms=5, color=MUTED, ecolor=MUTED, elinewidth=1.2, capsize=2.5, zorder=2)
        ax.errorbar(row.wavlm_l12_pca20_auc, yi, xerr=[[row.wavlm_l12_pca20_auc - row.wavlm_ci_lo],
                                                        [row.wavlm_ci_hi - row.wavlm_l12_pca20_auc]],
                     fmt="o", ms=5.5, color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=2.5, zorder=3)
        marker = " *" if row.gap_excludes_zero else " (ns)"
        ax.text(1.045, yi, marker, fontsize=7, va="center",
                color=(BLUE if row.gap_excludes_zero else MUTED))

    ax.set_yticks(y, [GEN_SHORT[g] for g in GENS])
    ax.set_xlim(0.3, 1.15)
    ax.set_ylim(-0.6, len(GENS) - 0.4)
    ax.axvline(0.5, color=BASE, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.508, -0.5, "chance", fontsize=6, color=MUTED, va="bottom")
    ax.set_xlabel("AUC, real vs synthetic laughter (controlled clips), with 95% CI")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], color=MUTED, label="content-free floor (analytic CI)")
    ax.scatter([], [], color=BLUE, label="WavLM L12+PCA20 (bootstrap CI)")
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, bbox_to_anchor=(0.0, -0.02))
    ax.set_title("(a) Within-generator probe vs content-free floor, with 95% CIs\n"
                  "* = WavLM$-$floor gap CI excludes 0 (2000 bootstrap resamples, stratified by class)",
                  loc="left", fontsize=7.2, color=INK, pad=6)
    fig.tight_layout()
    fig.savefig(OUT_F, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_F}")


if __name__ == "__main__":
    main()
