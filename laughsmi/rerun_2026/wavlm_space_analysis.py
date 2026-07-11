"""Quantify how laughter insertion moves augmented spoof files in WavLM space.

Uses the primary embedding: detector's own frozen WavLM-base backbone, layer 12
(last hidden state), mean-pooled over time, 768-d, as produced by
extract_detector_wavlm.py (saved raw/unstandardized on disk).

Axis w = centroid(spoof_base) - centroid(bona_base), computed in embeddings
STANDARDIZED per-coordinate by the bona-fide (base) mean/SD -- the same
convention as experiments/axis_audits/audit_common.py::loso_axis_features.
(Raw-space cosine/projection geometry was checked and is dominated by
high-variance nuisance WavLM dimensions unrelated to the spoof/bona-fide
distinction -- standardization is what makes the axis decision-relevant.)

Outputs:
  rerun_2026/tables/wavlm_space_changes.csv  -- per-file stats for paired
      augmented spoof files (base vs aug): L2 shift, cosine(base,aug),
      proj_along_w_base, proj_along_w_aug, delta_along_w,
      cos_dist_to_bona_centroid_base/aug, delta_cos_dist_to_bona, delta_score.
  rerun_2026/tables/wavlm_space_summary.csv  -- aggregate stats + correlation
      of delta_along_w with delta_score (Pearson + Spearman).
  rerun_2026/figures/wavlm_umap.png  -- UMAP of bona / spoof-base / spoof-aug.
  rerun_2026/figures/wavlm_pca.png   -- PCA fallback/companion view.
  rerun_2026/figures/along_w_vs_delta_score.png -- scatter + correlation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RERUN = Path(__file__).resolve().parent
LAYER = 12


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def main():
    npz = np.load(RERUN / "embeddings" / "wavlm_base_embeddings.npz")
    meta = pd.read_csv(RERUN / "embeddings" / "wavlm_base_meta.csv")
    X = npz[f"layer{LAYER}"]
    assert len(meta) == X.shape[0], f"{len(meta)} vs {X.shape[0]}"

    base_mask = meta["set"] == "base"
    aug_mask = meta["set"] == "aug"

    bona_base_mask = base_mask & (meta["label"] == "bona-fide")
    spoof_base_mask = base_mask & (meta["label"] == "spoof")

    # Standardize per-coordinate by the bona-fide (base) mean/SD before computing
    # the axis -- matches experiments/axis_audits/audit_common.py::loso_axis_features,
    # the established convention in this repo for the sd_along axis. Raw
    # (unstandardized) cosine/centroid geometry in 768-d WavLM space is dominated
    # by high-variance nuisance dimensions and gives a much noisier signal (verified
    # empirically here: raw-space projection barely separates base/aug). All
    # geometry below (w, projections, cosine distances) is computed in this
    # standardized space; embeddings on disk remain raw/unstandardized per the brief.
    mu_bona_raw = X[bona_base_mask.values].mean(0)
    sd_bona_raw = X[bona_base_mask.values].std(0) + 1e-9
    Xz = (X - mu_bona_raw) / sd_bona_raw

    mu_bona = Xz[bona_base_mask.values].mean(0)
    mu_spoof = Xz[spoof_base_mask.values].mean(0)
    w = unit(mu_spoof - mu_bona)
    X = Xz  # downstream code operates on standardized embeddings

    # per-file base index by basename (for pairing with aug)
    base_df = meta[base_mask].reset_index()
    aug_df = meta[aug_mask].reset_index()
    base_idx_by_name = dict(zip(base_df["basename"], base_df["index"]))

    # scores (for correlation)
    scores_base = pd.read_csv(RERUN / "base_asv19.csv")
    scores_aug = pd.read_csv(RERUN / "aug_asv19.csv")
    scores_base["basename"] = scores_base["file_id"].apply(lambda x: Path(x).name)
    scores_aug["basename"] = scores_aug["file_id"].apply(lambda x: Path(x).name)
    score_base_map = dict(zip(scores_base["basename"], scores_base["score"]))
    score_aug_map = dict(zip(scores_aug["basename"], scores_aug["score"]))

    rows = []
    aug_spoof_df = aug_df[(aug_df["label"] == "spoof") & (aug_df["augmented"].astype(str) == "1")]
    for _, r in aug_spoof_df.iterrows():
        bname = r["basename"]
        if bname not in base_idx_by_name:
            continue
        bi = base_idx_by_name[bname]
        x_base = X[bi]
        x_aug = X[r["index"]]

        l2_shift = float(np.linalg.norm(x_aug - x_base))
        cos_base_aug = float(np.dot(unit(x_base), unit(x_aug)))

        d_base = x_base - mu_bona
        d_aug = x_aug - mu_bona
        proj_base = float(np.dot(d_base, w))
        proj_aug = float(np.dot(d_aug, w))
        delta_along_w = proj_aug - proj_base
        orth_base = float(np.linalg.norm(d_base - proj_base * w))
        orth_aug = float(np.linalg.norm(d_aug - proj_aug * w))
        delta_orth = orth_aug - orth_base

        cosd_base = 1.0 - float(np.dot(unit(x_base), unit(mu_bona)))
        cosd_aug = 1.0 - float(np.dot(unit(x_aug), unit(mu_bona)))
        delta_cosd = cosd_aug - cosd_base

        sb = score_base_map.get(bname, np.nan)
        sa = score_aug_map.get(bname, np.nan)
        delta_score = sa - sb if not (np.isnan(sb) or np.isnan(sa)) else np.nan

        rows.append({
            "file": bname, "position": r.get("position", ""), "system_id": r.get("system_id", ""),
            "l2_shift": l2_shift, "cos_base_aug": cos_base_aug,
            "proj_along_w_base": proj_base, "proj_along_w_aug": proj_aug, "delta_along_w": delta_along_w,
            "orth_to_w_base": orth_base, "orth_to_w_aug": orth_aug, "delta_orth_to_w": delta_orth,
            "cos_dist_to_bona_base": cosd_base, "cos_dist_to_bona_aug": cosd_aug, "delta_cos_dist_to_bona": delta_cosd,
            "score_base": sb, "score_aug": sa, "delta_score": delta_score,
        })

    df = pd.DataFrame(rows)
    (RERUN / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RERUN / "tables" / "wavlm_space_changes.csv", index=False)
    print(f"wrote wavlm_space_changes.csv ({len(df)} paired augmented-spoof files)")

    # correlation of movement along w with delta score
    valid = df.dropna(subset=["delta_along_w", "delta_score"])
    pear_r, pear_p = stats.pearsonr(valid["delta_along_w"], valid["delta_score"])
    spear_r, spear_p = stats.spearmanr(valid["delta_along_w"], valid["delta_score"])

    summary = {
        "n_paired_aug_spoof": len(df),
        "mean_l2_shift": float(df["l2_shift"].mean()),
        "mean_cos_base_aug": float(df["cos_base_aug"].mean()),
        "mean_proj_along_w_base": float(df["proj_along_w_base"].mean()),
        "mean_proj_along_w_aug": float(df["proj_along_w_aug"].mean()),
        "mean_delta_along_w": float(df["delta_along_w"].mean()),
        "frac_moved_toward_bona_along_w": float((df["delta_along_w"] < 0).mean()),
        "mean_orth_to_w_base": float(df["orth_to_w_base"].mean()),
        "mean_orth_to_w_aug": float(df["orth_to_w_aug"].mean()),
        "mean_delta_orth_to_w": float(df["delta_orth_to_w"].mean()),
        "ratio_abs_delta_along_w_to_delta_orth": float(df["delta_along_w"].abs().mean() / (df["delta_orth_to_w"].abs().mean() + 1e-9)),
        "mean_cos_dist_to_bona_base": float(df["cos_dist_to_bona_base"].mean()),
        "mean_cos_dist_to_bona_aug": float(df["cos_dist_to_bona_aug"].mean()),
        "mean_delta_cos_dist_to_bona": float(df["delta_cos_dist_to_bona"].mean()),
        "pearson_r_deltaW_deltaScore": float(pear_r),
        "pearson_p_deltaW_deltaScore": float(pear_p),
        "spearman_r_deltaW_deltaScore": float(spear_r),
        "spearman_p_deltaW_deltaScore": float(spear_p),
    }
    with open(RERUN / "tables" / "wavlm_space_summary.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w_.writeheader()
        w_.writerow(summary)
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RERUN / "figures").mkdir(parents=True, exist_ok=True)

    # scatter: delta_along_w vs delta_score
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(valid["delta_along_w"], valid["delta_score"], alpha=0.6, s=20)
    ax.set_xlabel("Δ projection along w (spoof-bona axis), aug - base")
    ax.set_ylabel("Δ detector score (aug - base)")
    ax.set_title(f"Movement along w vs Δscore (Pearson r={pear_r:.2f}, p={pear_p:.1e})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(RERUN / "figures" / "along_w_vs_delta_score.png", dpi=150)
    plt.close(fig)

    # PCA of bona / spoof-base / spoof-aug (+ context: all base+aug points)
    from sklearn.decomposition import PCA
    plot_mask = bona_base_mask.values | spoof_base_mask.values
    aug_spoof_idx = aug_spoof_df["index"].values
    X_plot = np.concatenate([X[plot_mask], X[aug_spoof_idx]], axis=0)
    labels_plot = (["bona-fide (base)"] * int(bona_base_mask.sum())
                   + ["spoof (base)"] * int(spoof_base_mask.sum())
                   + ["spoof (aug, +laughter)"] * len(aug_spoof_idx))
    pca = PCA(n_components=2, random_state=0)
    Xp = pca.fit_transform(X_plot)

    colors = {"bona-fide (base)": "#2E86AB", "spoof (base)": "#C1443C", "spoof (aug, +laughter)": "#E8A33D"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in colors:
        m = np.array(labels_plot) == lab
        ax.scatter(Xp[m, 0], Xp[m, 1], s=18, alpha=0.65, label=lab, color=colors[lab])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("WavLM-base L12 mean-pooled embeddings, standardized (PCA)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RERUN / "figures" / "wavlm_pca.png", dpi=150)
    plt.close(fig)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
        Xu = reducer.fit_transform(X_plot)
        fig, ax = plt.subplots(figsize=(7, 6))
        for lab in colors:
            m = np.array(labels_plot) == lab
            ax.scatter(Xu[m, 0], Xu[m, 1], s=18, alpha=0.65, label=lab, color=colors[lab])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title("WavLM-base L12 mean-pooled embeddings, standardized (UMAP)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RERUN / "figures" / "wavlm_umap.png", dpi=150)
        plt.close(fig)
        print("wrote wavlm_umap.png")
    except Exception as e:
        print(f"UMAP failed ({e}); PCA figure still written")

    print("wrote wavlm_pca.png, along_w_vs_delta_score.png")


if __name__ == "__main__":
    main()
