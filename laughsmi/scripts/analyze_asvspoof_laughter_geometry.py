"""Analyze ASVspoof bonafide speech vs VocalSound laughter in WavLM space."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist


def load_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path.with_suffix(".csv"))
    return pd.read_csv(path)


def rank_biserial_from_u(u: float, n_x: int, n_y: int) -> float:
    return float((2.0 * u / (n_x * n_y)) - 1.0)


def unpaired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    laugh = df[df.group == "laugh-vs"]
    speech = df[df.group == "speech-asv-bona"]
    for feature in ("C_layer12", "T_layer9", "mean_cos_dist_layer12", "n_frames"):
        x = laugh[feature].to_numpy(float)
        y = speech[feature].to_numpy(float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "comparison": "laugh-vs_minus_speech-asv-bona",
            "feature": feature,
            "n_laugh": len(x),
            "n_speech": len(y),
            "mean_laugh": float(np.mean(x)),
            "mean_speech": float(np.mean(y)),
            "median_laugh": float(np.median(x)),
            "median_speech": float(np.median(y)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
            "rank_biserial_laugh_gt_speech": rank_biserial_from_u(float(u), len(x), len(y)),
        })
    return pd.DataFrame(rows)


def probe_auc(df: pd.DataFrame, emb: np.ndarray, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    sub = df[df.group.isin(["laugh-vs", "speech-asv-bona"])].copy()
    idx = sub.row_index.to_numpy(int)
    X = emb[idx]
    valid = ~np.isnan(X).any(axis=1)
    sub = sub[valid]
    X = X[valid]
    y = (sub.group == "laugh-vs").astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train, test in cv.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(X[train], y[train])
        score = clf.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], score))
    return {
        "probe": "logreg_laugh_vs_asv_bonafide_speech",
        "n": int(len(y)),
        "n_laugh": int(y.sum()),
        "n_speech": int((1 - y).sum()),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "n_folds": 5,
    }


def centroid_distances(df: pd.DataFrame, emb: np.ndarray) -> dict:
    out = {"metric": "centroid_cosine_distance_layer12"}
    centroids = {}
    for group in ("laugh-vs", "speech-asv-bona"):
        idx = df.loc[df.group == group, "row_index"].to_numpy(int)
        X = emb[idx]
        X = X[~np.isnan(X).any(axis=1)]
        centroids[group] = X.mean(axis=0)
    out["cosine_dist_laugh_vs_to_speech_asv_bona"] = float(
        cosine_dist(centroids["laugh-vs"], centroids["speech-asv-bona"])
    )
    return out


def plot_umap(df: pd.DataFrame, emb9: np.ndarray, emb12: np.ndarray, out_base: Path, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    colors = {"laugh-vs": "#d95f02", "speech-asv-bona": "#1b9e77"}
    markers = {"laugh-vs": "^", "speech-asv-bona": "o"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, emb, title in ((axes[0], emb9, "WavLM Layer 9"), (axes[1], emb12, "WavLM Layer 12")):
        idx = df.row_index.to_numpy(int)
        X = emb[idx]
        valid = ~np.isnan(X).any(axis=1)
        sub = df[valid]
        X = X[valid]
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
        coords = reducer.fit_transform(X)
        for group in ["speech-asv-bona", "laugh-vs"]:
            m = (sub.group == group).to_numpy()
            ax.scatter(coords[m, 0], coords[m, 1], s=16, alpha=0.75,
                       c=colors[group], marker=markers[group], label=group, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("ASVspoof bonafide speech vs VocalSound laughter")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("laughsmi/embeddings/asv_vs_laughter_features.parquet"))
    parser.add_argument("--mean-emb9", type=Path, default=Path("laughsmi/embeddings/asv_vs_laughter_embeddings/mean_emb_layer9.npy"))
    parser.add_argument("--mean-emb12", type=Path, default=Path("laughsmi/embeddings/asv_vs_laughter_embeddings/mean_emb_layer12.npy"))
    parser.add_argument("--out-stats", type=Path, default=Path("laughsmi/tables/asv_vs_laughter_stats.csv"))
    parser.add_argument("--out-probe", type=Path, default=Path("laughsmi/tables/asv_vs_laughter_probe.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("laughsmi/figures/asv_vs_laughter_umap"))
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    df = load_features(args.features)
    emb9 = np.load(args.mean_emb9)
    emb12 = np.load(args.mean_emb12)

    stats_df = unpaired_tests(df)
    probe_df = pd.DataFrame([probe_auc(df, emb12, args.seed), centroid_distances(df, emb12)])
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.out_stats, index=False)
    args.out_probe.parent.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(args.out_probe, index=False)
    plot_umap(df, emb9, emb12, args.out_fig, args.seed)
    print(f"Wrote {args.out_stats}")
    print(stats_df.to_string(index=False))
    print(f"Wrote {args.out_probe}")
    print(probe_df.to_string(index=False))
    print(f"Wrote {args.out_fig.with_suffix('.png')} and {args.out_fig.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
