"""C2: second real-laughter anchor (ESC-50 'laughing', Freesound.org) vs
cached synthetic-family WavLM embeddings, and a direct corpus-leakage probe.

Two comparisons are run and clearly labeled by pipeline, because raw audio
for VocalSound (real, D2/D3's original anchor) and for all five synthetic
families is GONE -- only their cached mean WavLM embeddings survive:

  (A) RAW pipeline (whole-clip, no silence-trim/peak-norm): the pipeline that
      produced the cached embeddings/d3_multi/mean_emb_layer{9,12}.npy for
      laugh-real (VocalSound) and all synthetic families. We extract real2
      (ESC-50 laughing) the SAME way (real2_raw_emb_layer12.npy) so this
      comparison is apples-to-apples on pipeline, letting us pair real2
      against the cached synth embeddings AND against cached VocalSound.
      This is also the representation that gave the original (audited,
      confound-prone) AUC~1.0 D3 finding, so it is not by itself evidence of
      a laughter-content signal -- but it IS the only representation in which
      a same-pipeline real2-vs-synth AND real2-vs-VocalSound comparison is
      possible, which is exactly what's needed for the leakage probe.

  (B) FIXED/confound-controlled pipeline (2s center active segment,
      peak-normalized -- scripts/d3_fixed.py's protocol): real2 was also
      extracted this way (real2_emb_layer12.npy). This is the protocol the
      paper's CURRENT defensible D3 claim relies on. It cannot be applied to
      the synthetic families here because their raw audio no longer exists
      to reprocess, so no real2-vs-synth number can be reported in this
      pipeline. It CAN be used for a real2-only content-free sanity check.

Outputs:
  rerun_2026/concerns/tables/second_anchor_separability.csv
  rerun_2026/concerns/tables/leakage_probe.csv
  rerun_2026/concerns/figures/second_anchor_auc.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

L = Path(__file__).resolve().parents[2]
C = L / "rerun_2026" / "concerns"
RNG_SEED = 0
N_BOOT = 2000

FAMILIES = {
    "bark_laughter_token": "laugh-bark_laughter_token",
    "bark_laughs_inline": "laugh-bark_laughs_inline",
    "parler_tts": "laugh-parler_tts",
    "xtts": "laugh-xtts",
    "audioldm2": "laugh-audioldm2",
}


def cv_auc(X, y, pca=None, seed=RNG_SEED):
    ok = ~np.isnan(X).any(1)
    X, y = X[ok], y[ok]
    nmin = min((y == 0).sum(), (y == 1).sum())
    if nmin < 2:
        return np.nan
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, X.shape[0] - 1, X.shape[1]), random_state=seed))
    steps.append(LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(min(5, nmin), shuffle=True, random_state=seed)
    s = cross_val_score(make_pipeline(*steps), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean())


def bootstrap_auc_ci(X, y, pca=20, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap CI on AUC via a single held-representation logistic-regression
    probe refit on resamples of the pooled (X, y): stratified resample with
    replacement, refit + evaluate in-sample-CV each time (same probe family
    as the point estimate) to get a distribution of AUCs."""
    rng = np.random.RandomState(seed)
    ok = ~np.isnan(X).any(1)
    X, y = X[ok], y[ok]
    n = len(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    aucs = []
    for b in range(n_boot):
        bi0 = rng.choice(idx0, len(idx0), replace=True)
        bi1 = rng.choice(idx1, len(idx1), replace=True)
        bidx = np.concatenate([bi0, bi1])
        Xb, yb = X[bidx], y[bidx]
        # fit on one half, score on the other half of the bootstrap sample
        # (simple stratified 50/50 split, fast, avoids CV-in-bootstrap cost)
        rng2 = np.random.RandomState(seed * 100000 + b)
        perm0 = rng2.permutation(len(bi0)); perm1 = rng2.permutation(len(bi1))
        h0 = len(bi0) // 2; h1 = len(bi1) // 2
        if h0 < 1 or h1 < 1:
            continue
        tr_idx = np.concatenate([np.arange(len(bi0))[perm0[:h0]], len(bi0) + np.arange(len(bi1))[perm1[:h1]]])
        te_idx = np.concatenate([np.arange(len(bi0))[perm0[h0:]], len(bi0) + np.arange(len(bi1))[perm1[h1:]]])
        Xtr, ytr = Xb[tr_idx], yb[tr_idx]
        Xte, yte = Xb[te_idx], yb[te_idx]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=seed))
        steps.append(LogisticRegression(max_iter=2000))
        clf = make_pipeline(*steps)
        try:
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            continue
    aucs = np.array(aucs)
    if len(aucs) < 50:
        return np.nan, np.nan, np.nan, len(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi), len(aucs)


def main():
    # ---- load cached synth + VocalSound embeddings (RAW pipeline) ----
    inv = pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv")
    emb12 = np.load(L / "embeddings" / "d3_multi" / "mean_emb_layer12.npy")
    g = inv.group.to_numpy()

    # ---- load real2 (ESC-50 laughing) embeddings, both pipelines ----
    real2_raw12 = np.load(C / "data" / "real2_raw_emb_layer12.npy")
    real2_fix12 = np.load(C / "data" / "real2_emb_layer12.npy")
    real2_cf = np.load(C / "data" / "real2_contentfree.npy")
    real2_inv = pd.read_csv(C / "data" / "real2_inventory.csv")
    n_real2 = len(real2_inv)
    print(f"real2 (ESC-50 laughing): n={n_real2}")

    rows = []
    # === Step 3+4: replication -- real2 (RAW pipeline) vs each cached synth family (RAW) ===
    for short, grp in FAMILIES.items():
        Xsyn = emb12[g == grp]
        X = np.vstack([real2_raw12, Xsyn])
        y = np.r_[np.zeros(len(real2_raw12)), np.ones(len(Xsyn))]
        auc_point = cv_auc(X, y, pca=20)
        auc_mean, lo, hi, nboot = bootstrap_auc_ci(X, y, pca=20)
        rows.append({
            "comparison": f"real2 vs {short}", "pipeline": "raw (matches cached d3_multi)",
            "n_real2": len(real2_raw12), "n_synth": len(Xsyn),
            "wavlm_l12_pca20_auc": round(auc_point, 3),
            "boot_mean": round(auc_mean, 3) if not np.isnan(auc_mean) else np.nan,
            "boot_ci_lo": round(lo, 3) if not np.isnan(lo) else np.nan,
            "boot_ci_hi": round(hi, 3) if not np.isnan(hi) else np.nan,
            "n_boot_valid": nboot,
        })

    # === Step 5: leakage probe -- real2 (RAW) vs cached VocalSound (RAW, laugh-real) ===
    Xvs = emb12[g == "laugh-real"]
    X = np.vstack([real2_raw12, Xvs])
    y = np.r_[np.zeros(len(real2_raw12)), np.ones(len(Xvs))]
    auc_point = cv_auc(X, y, pca=20)
    auc_mean, lo, hi, nboot = bootstrap_auc_ci(X, y, pca=20)
    leak_row_wavlm = {
        "comparison": "real2 (ESC-50) vs real-VocalSound", "pipeline": "raw (matches cached d3_multi)",
        "n_real2": len(real2_raw12), "n_synth": len(Xvs),
        "wavlm_l12_pca20_auc": round(auc_point, 3),
        "boot_mean": round(auc_mean, 3) if not np.isnan(auc_mean) else np.nan,
        "boot_ci_lo": round(lo, 3) if not np.isnan(lo) else np.nan,
        "boot_ci_hi": round(hi, 3) if not np.isnan(hi) else np.nan,
        "n_boot_valid": nboot,
    }
    rows.append(leak_row_wavlm)

    # === Semi-controlled variant: real2 in the FIXED (2s, peak-norm) pipeline
    # vs cached synth in their native RAW pipeline. Real audio for synth is
    # gone so their side can't be re-processed; this at least removes the
    # real2-side duration/silence/loudness confound and checks whether the
    # raw-pipeline AUC=1.0 saturation (all families indistinguishable) is a
    # real2-side duration artifact (ESC-50 clips are all 5.0s, much longer
    # than typical synth clips). ===
    rows_semi = []
    for short, grp in FAMILIES.items():
        Xsyn = emb12[g == grp]
        X = np.vstack([real2_fix12, Xsyn])
        y = np.r_[np.zeros(len(real2_fix12)), np.ones(len(Xsyn))]
        auc_point = cv_auc(X, y, pca=20)
        auc_mean, lo, hi, nboot = bootstrap_auc_ci(X, y, pca=20)
        rows_semi.append({
            "comparison": f"real2 vs {short}", "pipeline": "real2=fixed(2s,peaknorm) / synth=raw(cached)",
            "n_real2": len(real2_fix12), "n_synth": len(Xsyn),
            "wavlm_l12_pca20_auc": round(auc_point, 3),
            "boot_mean": round(auc_mean, 3) if not np.isnan(auc_mean) else np.nan,
            "boot_ci_lo": round(lo, 3) if not np.isnan(lo) else np.nan,
            "boot_ci_hi": round(hi, 3) if not np.isnan(hi) else np.nan,
            "n_boot_valid": nboot,
        })
    Xvs = emb12[g == "laugh-real"]
    X = np.vstack([real2_fix12, Xvs])
    y = np.r_[np.zeros(len(real2_fix12)), np.ones(len(Xvs))]
    auc_point = cv_auc(X, y, pca=20)
    auc_mean, lo, hi, nboot = bootstrap_auc_ci(X, y, pca=20)
    rows_semi.append({
        "comparison": "real2 vs real-VocalSound", "pipeline": "real2=fixed(2s,peaknorm) / VocalSound=raw(cached)",
        "n_real2": len(real2_fix12), "n_synth": len(Xvs),
        "wavlm_l12_pca20_auc": round(auc_point, 3),
        "boot_mean": round(auc_mean, 3) if not np.isnan(auc_mean) else np.nan,
        "boot_ci_lo": round(lo, 3) if not np.isnan(lo) else np.nan,
        "boot_ci_hi": round(hi, 3) if not np.isnan(hi) else np.nan,
        "n_boot_valid": nboot,
    })
    df_semi = pd.DataFrame(rows_semi)
    df_semi.to_csv(C / "tables" / "second_anchor_separability_semicontrolled.csv", index=False)
    print("\n=== semi-controlled (real2 fixed vs synth raw-cached) ===")
    print(df_semi.to_string(index=False))

    df = pd.DataFrame(rows)
    (C / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(C / "tables" / "second_anchor_separability.csv", index=False)
    print(df.to_string(index=False))

    # === Sanity controls in the SAME raw pipeline: label-shuffle + real2 self-split ===
    rng = np.random.RandomState(RNG_SEED)
    Xbt = np.vstack([real2_raw12, emb12[g == "laugh-bark_laughter_token"]])
    ybt = np.r_[np.zeros(len(real2_raw12)), np.ones((g == "laugh-bark_laughter_token").sum())]
    shuf = np.mean([cv_auc(Xbt, rng.permutation(ybt), pca=20, seed=t) for t in range(5)])

    # real2 self-split control (confound-controlled pipeline, content-free floor)
    half = n_real2 // 2
    perm = rng.permutation(n_real2)
    y_self = np.zeros(n_real2); y_self[perm[:half]] = 1
    real2_selfsplit_wavlm = np.mean([cv_auc(real2_fix12, rng.permutation(y_self), pca=20, seed=t) for t in range(5)])
    # (permuting again is another shuffle; use a fixed non-random split instead)
    y_self2 = np.zeros(n_real2); y_self2[perm[:half]] = 1
    real2_selfsplit_wavlm_fixed = cv_auc(real2_fix12, y_self2, pca=20, seed=0)
    real2_selfsplit_cf = cv_auc(real2_cf, y_self2, pca=None, seed=0)

    print(f"\ncontrols (raw pipeline): label-shuffle real2-vs-bark_token={shuf:.3f} (want ~0.5)")
    print(f"real2-only random-half-split (fixed pipeline) WavLM AUC={real2_selfsplit_wavlm_fixed:.3f}, "
          f"content-free AUC={real2_selfsplit_cf:.3f} (both want ~0.5)")

    # === Leakage probe table ===
    # Reference numbers from tables/table_d3_fixed.csv (VocalSound-anchor, confound-controlled)
    d3fixed = pd.read_csv(L / "tables" / "table_d3_fixed.csv")
    ref_xtts_fixed = d3fixed.loc[d3fixed.comparison == "real vs xtts", "wavlm_l12_pca20"].values
    ref_xtts_fixed = float(ref_xtts_fixed[0]) if len(ref_xtts_fixed) else np.nan

    leak_rows = [
        {"probe": "real-VocalSound vs real2-ESC50 (WavLM L12+PCA20, raw pipeline)",
         "auc": leak_row_wavlm["wavlm_l12_pca20_auc"],
         "boot_ci_lo": leak_row_wavlm["boot_ci_lo"], "boot_ci_hi": leak_row_wavlm["boot_ci_hi"],
         "interpretation": "two independent REAL corpora; high AUC here = corpus/channel signal, not laughter authenticity"},
        {"probe": "real-VocalSound vs XTTS (WavLM L12+PCA20, confound-controlled, from table_d3_fixed.csv)",
         "auc": ref_xtts_fixed, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
         "interpretation": "prior reported real-vs-XTTS separability (VocalSound anchor, XTTS cloned FROM VocalSound)"},
        {"probe": "real2-ESC50 vs XTTS (WavLM L12+PCA20, raw pipeline)",
         "auc": df.loc[df.comparison == "real2 vs xtts", "wavlm_l12_pca20_auc"].values[0],
         "boot_ci_lo": df.loc[df.comparison == "real2 vs xtts", "boot_ci_lo"].values[0],
         "boot_ci_hi": df.loc[df.comparison == "real2 vs xtts", "boot_ci_hi"].values[0],
         "interpretation": "independent-anchor XTTS separability; XTTS was NOT cloned from ESC-50"},
        {"probe": "label-shuffle control (real2 vs bark_token, raw pipeline)",
         "auc": round(float(shuf), 3), "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
         "interpretation": "sanity: should be ~0.5"},
        {"probe": "real2 internal random-half-split, WavLM (confound-controlled pipeline)",
         "auc": round(float(real2_selfsplit_wavlm_fixed), 3), "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
         "interpretation": "sanity: should be ~0.5 (no true label difference within one corpus)"},
        {"probe": "real2 internal random-half-split, content-free 8-scalar (confound-controlled pipeline)",
         "auc": round(float(real2_selfsplit_cf), 3), "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
         "interpretation": "sanity: should be ~0.5"},
    ]
    leak_df = pd.DataFrame(leak_rows)
    leak_df.to_csv(C / "tables" / "leakage_probe.csv", index=False)
    print("\n" + leak_df.to_string(index=False))
    print("\nwrote tables/second_anchor_separability.csv, tables/leakage_probe.csv")

    # === figure ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_rows = df[df.comparison != "real2 (ESC-50) vs real-VocalSound"].copy()
        xs = np.arange(len(plot_rows))
        ax.bar(xs, plot_rows.wavlm_l12_pca20_auc, color="#2563eb", alpha=0.85, label="real2 (ESC-50) vs synth family")
        yerr_lo = plot_rows.wavlm_l12_pca20_auc - plot_rows.boot_ci_lo
        yerr_hi = plot_rows.boot_ci_hi - plot_rows.wavlm_l12_pca20_auc
        ax.errorbar(xs, plot_rows.wavlm_l12_pca20_auc, yerr=[yerr_lo, yerr_hi], fmt="none",
                     ecolor="black", capsize=4)
        ax.axhline(leak_row_wavlm["wavlm_l12_pca20_auc"], color="#dc2626", linestyle="--",
                    label=f"real-VocalSound vs real2 leakage probe = {leak_row_wavlm['wavlm_l12_pca20_auc']:.2f}")
        ax.axhline(0.5, color="gray", linestyle=":", label="chance")
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("real2 vs ", "") for c in plot_rows.comparison], rotation=20)
        ax.set_ylabel("WavLM L12+PCA20 AUC (real2 anchor, raw pipeline)")
        ax.set_ylim(0.3, 1.05)
        ax.set_title("C2: second real-anchor (ESC-50) separability vs. leakage probe")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        (C / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C / "figures" / "second_anchor_auc.png", dpi=200, bbox_inches="tight")
        print("wrote figures/second_anchor_auc.png")
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
