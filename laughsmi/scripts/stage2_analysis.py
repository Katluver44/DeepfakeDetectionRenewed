"""Stage 2 statistical + visual analyses: paired Wilcoxon, UMAP, H4 probe.

Implements laughsmi_plan.md §5.3:

(a) Paired Wilcoxon signed-rank test, laugh-bona vs. speech-bona, on C
    (compactness, layer 12) and T (velocity entropy, layer 9), linked via
    `pair_id`. Reports p-values and rank-biserial effect size r.
    -> tables/table2.csv, tables/table2.tex  (H2 verdict)

(b) UMAP (n_neighbors=30, min_dist=0.1, metric=cosine, seed 42) of the
    segment-mean embeddings, two panels (layer 9 | layer 12), colored by
    group. -> figures/figure1.pdf, figures/figure1.png (300 dpi)

(c) If laugh-spoof rows exist: cosine distance of the laugh-spoof centroid
    to the laugh-bona centroid vs. the speech centroids, plus a 5-fold
    logistic-regression probe (genuine laughter [laugh-bona + laugh-vs] vs.
    synthetic laughter [laugh-spoof]) reporting AUC.
    -> appended to tables/table2.csv, and tables/h4_probe.csv  (H4 verdict)

Inputs:
    --features   embeddings/features.parquet (or .csv fallback), as produced
                 by extract_wavlm_features.py. Must have columns: row_index,
                 group, pair_id, C_layer12, T_layer9.
    --inventory  embeddings/segment_inventory.csv (for group/pair_id if not
                 already joined into features; used mainly as a cross-check).
    --mean-emb9  embeddings/mean_emb_layer9.npy
    --mean-emb12 embeddings/mean_emb_layer12.npy

Usage:
    python stage2_analysis.py \
        --features embeddings/features.parquet \
        --mean-emb9 embeddings/mean_emb_layer9.npy \
        --mean-emb12 embeddings/mean_emb_layer12.npy \
        --out-table2 tables/table2.csv --out-table2-tex tables/table2.tex \
        --out-fig figures/figure1 \
        --out-h4 tables/h4_probe.csv \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback)
            raise
    return pd.read_csv(path)


def rank_biserial_from_wilcoxon(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size r for a Wilcoxon signed-rank test.

    r = (sum of positive ranks - sum of negative ranks) / (total rank sum)
    Equivalent to (W+ - W-) / (n(n+1)/2) for n non-zero differences.
    """
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return float("nan")
    abs_ranks = stats.rankdata(np.abs(diffs))
    pos_rank_sum = abs_ranks[diffs > 0].sum()
    neg_rank_sum = abs_ranks[diffs < 0].sum()
    total = pos_rank_sum + neg_rank_sum
    if total == 0:
        return float("nan")
    return float((pos_rank_sum - neg_rank_sum) / total)


def paired_wilcoxon_analysis(df: pd.DataFrame) -> list[dict]:
    """Paired Wilcoxon signed-rank, laugh-bona vs speech-bona, on C_layer12 and T_layer9."""
    rows = []
    laugh = df[df["group"] == "laugh-bona"].dropna(subset=["pair_id"])
    speech = df[df["group"] == "speech-bona"].dropna(subset=["pair_id"])

    laugh = laugh.set_index(laugh["pair_id"].astype(str))
    speech = speech.set_index(speech["pair_id"].astype(str))
    common_ids = laugh.index.intersection(speech.index)

    n_pairs = len(common_ids)
    for feature in ("C_layer12", "T_layer9"):
        if n_pairs < 1:
            rows.append({
                "test": f"paired_wilcoxon_{feature}",
                "n_pairs": 0,
                "statistic": float("nan"),
                "p_value": float("nan"),
                "rank_biserial_r": float("nan"),
                "mean_laugh_bona": float("nan"),
                "mean_speech_bona": float("nan"),
            })
            continue

        x = laugh.loc[common_ids, feature].to_numpy(dtype=float)
        y = speech.loc[common_ids, feature].to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        diffs = x - y

        if len(diffs) > 0 and np.any(diffs != 0):
            statistic, p_value = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        else:
            statistic, p_value = float("nan"), float("nan")

        r = rank_biserial_from_wilcoxon(diffs)

        rows.append({
            "test": f"paired_wilcoxon_{feature}",
            "n_pairs": int(len(diffs)),
            "statistic": float(statistic) if statistic == statistic else float("nan"),
            "p_value": float(p_value) if p_value == p_value else float("nan"),
            "rank_biserial_r": r,
            "mean_laugh_bona": float(np.mean(x)) if len(x) else float("nan"),
            "mean_speech_bona": float(np.mean(y)) if len(y) else float("nan"),
        })
    return rows


def write_table2_latex(rows: list[dict], out_tex: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stage 2: paired Wilcoxon signed-rank tests (laugh-bona vs. "
                 r"speech-bona) on compactness $C$ (layer 12) and velocity entropy $T$ "
                 r"(layer 9), plus H4 synthetic-laughter probe results if available.}")
    lines.append(r"\label{tab:table2}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Test & N pairs & Statistic & $p$ & Rank-biserial $r$ & Mean (laugh / speech) \\")
    lines.append(r"\midrule")
    for row in rows:
        if "test" not in row:
            continue
        p_val = row.get("p_value", float("nan"))
        p_str = "$<$0.001" if (p_val == p_val and p_val < 0.001) else (
            f"{p_val:.3f}" if p_val == p_val else "--"
        )
        stat = row.get("statistic", float("nan"))
        stat_str = f"{stat:.2f}" if stat == stat else "--"
        r_val = row.get("rank_biserial_r", float("nan"))
        r_str = f"{r_val:.3f}" if r_val == r_val else "--"
        mean_l = row.get("mean_laugh_bona", float("nan"))
        mean_s = row.get("mean_speech_bona", float("nan"))
        mean_str = (
            f"{mean_l:.3f} / {mean_s:.3f}" if (mean_l == mean_l and mean_s == mean_s) else "--"
        )
        test_name_tex = row["test"].replace("_", r"\_")
        lines.append(
            f"{test_name_tex} & {row.get('n_pairs', 0)} & "
            f"{stat_str} & {p_str} & {r_str} & {mean_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n")


GROUP_COLORS = {
    "laugh-bona": "#1b9e77",
    "speech-bona": "#7570b3",
    "laugh-vs": "#d95f02",
    "speech-spoof": "#666666",
    "laugh-spoof": "#e7298a",
}
GROUP_MARKERS = {
    "laugh-bona": "o",
    "speech-bona": "s",
    "laugh-vs": "^",
    "speech-spoof": "x",
    "laugh-spoof": "D",
}


def plot_umap_panels(
    df: pd.DataFrame,
    mean_emb9: np.ndarray,
    mean_emb12: np.ndarray,
    out_fig_base: Path,
    seed: int = 42,
) -> None:
    """UMAP of segment-mean embeddings, two panels (layer 9 | layer 12)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, emb, layer_name in ((axes[0], mean_emb9, "Layer 9"), (axes[1], mean_emb12, "Layer 12")):
        valid_mask = ~np.isnan(emb).any(axis=1)
        row_indices = df["row_index"].to_numpy()
        # Align df rows to embedding rows: embedding arrays are indexed by
        # inventory row_index, so build a lookup.
        max_idx = emb.shape[0]
        keep = (row_indices < max_idx) & valid_mask[np.clip(row_indices, 0, max_idx - 1)]
        sub_df = df[keep]
        sub_emb = emb[sub_df["row_index"].to_numpy()]

        if len(sub_df) < 5:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(layer_name)
            continue

        n_neighbors = min(30, max(2, len(sub_df) - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=seed
        )
        coords = reducer.fit_transform(sub_emb)

        for group in sub_df["group"].unique():
            gm = (sub_df["group"] == group).to_numpy()
            ax.scatter(
                coords[gm, 0], coords[gm, 1],
                s=14, alpha=0.75,
                c=GROUP_COLORS.get(group, "#999999"),
                marker=GROUP_MARKERS.get(group, "o"),
                label=group,
                edgecolors="none",
            )
        ax.set_title(f"WavLM {layer_name}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels) or 1),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Segment-mean WavLM embeddings by group (UMAP, cosine metric)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_fig_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_fig_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def h4_probe_analysis(df: pd.DataFrame, mean_emb12: np.ndarray, seed: int = 42) -> dict | None:
    """H4: cosine-distance-to-centroid + 5-fold logistic regression AUC probe,
    genuine laughter (laugh-bona + laugh-vs) vs. synthetic laughter (laugh-spoof).

    Returns None if there are no laugh-spoof rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.spatial.distance import cosine as cosine_dist

    laugh_spoof = df[df["group"] == "laugh-spoof"]
    if len(laugh_spoof) == 0:
        return None

    def centroid_for(group: str) -> np.ndarray | None:
        sub = df[df["group"] == group]
        if len(sub) == 0:
            return None
        idx = sub["row_index"].to_numpy()
        idx = idx[idx < mean_emb12.shape[0]]
        vecs = mean_emb12[idx]
        vecs = vecs[~np.isnan(vecs).any(axis=1)]
        if len(vecs) == 0:
            return None
        return vecs.mean(axis=0)

    centroids = {g: centroid_for(g) for g in ("laugh-bona", "laugh-vs", "speech-bona", "speech-spoof", "laugh-spoof")}

    dist_results = {}
    ref = centroids.get("laugh-spoof")
    if ref is not None:
        for g, c in centroids.items():
            if g == "laugh-spoof" or c is None:
                continue
            dist_results[f"cosine_dist_laugh-spoof_to_{g}"] = float(cosine_dist(ref, c))

    genuine_idx = df[df["group"].isin(["laugh-bona", "laugh-vs"])]["row_index"].to_numpy()
    synthetic_idx = df[df["group"] == "laugh-spoof"]["row_index"].to_numpy()
    genuine_idx = genuine_idx[genuine_idx < mean_emb12.shape[0]]
    synthetic_idx = synthetic_idx[synthetic_idx < mean_emb12.shape[0]]

    X_genuine = mean_emb12[genuine_idx]
    X_synth = mean_emb12[synthetic_idx]
    valid_g = ~np.isnan(X_genuine).any(axis=1)
    valid_s = ~np.isnan(X_synth).any(axis=1)
    X_genuine, X_synth = X_genuine[valid_g], X_synth[valid_s]

    result = {**dist_results, "n_genuine": int(len(X_genuine)), "n_synthetic": int(len(X_synth))}

    if len(X_genuine) < 4 or len(X_synth) < 4:
        result["auc_mean"] = float("nan")
        result["auc_std"] = float("nan")
        result["note"] = "insufficient samples for 5-fold CV (need >=4 per class)"
        return result

    X = np.vstack([X_genuine, X_synth])
    y = np.concatenate([np.zeros(len(X_genuine)), np.ones(len(X_synth))])

    n_splits = min(5, np.bincount(y.astype(int)).min())
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        if len(np.unique(y[test_idx])) < 2:
            continue
        probs = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    result["auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    result["auc_std"] = float(np.std(aucs)) if aucs else float("nan")
    result["n_folds"] = len(aucs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 analyses: paired Wilcoxon (H2), UMAP figure, H4 synthetic-laughter probe."
    )
    parser.add_argument("--features", type=Path, default=Path("embeddings/features.parquet"))
    parser.add_argument("--mean-emb9", type=Path, default=Path("embeddings/mean_emb_layer9.npy"))
    parser.add_argument("--mean-emb12", type=Path, default=Path("embeddings/mean_emb_layer12.npy"))
    parser.add_argument("--out-table2", type=Path, default=Path("tables/table2.csv"))
    parser.add_argument("--out-table2-tex", type=Path, default=Path("tables/table2.tex"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/figure1"),
                        help="Base path (no extension) for figure1.pdf/.png")
    parser.add_argument("--out-h4", type=Path, default=Path("tables/h4_probe.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.features.exists():
        print(f"ERROR: features file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    df = load_features(args.features)
    if "row_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_index"})

    # --- (a) paired Wilcoxon ---
    wilcoxon_rows = paired_wilcoxon_analysis(df)
    table2_df = pd.DataFrame(wilcoxon_rows)

    # --- (c) H4 probe (computed before writing table2 so we can append rows) ---
    h4_result = None
    if args.mean_emb12.exists():
        mean_emb12 = np.load(args.mean_emb12)
        h4_result = h4_probe_analysis(df, mean_emb12, seed=args.seed)

    if h4_result is not None:
        h4_summary_row = {
            "test": "h4_probe_genuine_vs_synthetic_laughter_auc",
            "n_pairs": h4_result.get("n_genuine", 0) + h4_result.get("n_synthetic", 0),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "rank_biserial_r": float("nan"),
            "mean_laugh_bona": h4_result.get("auc_mean", float("nan")),
            "mean_speech_bona": h4_result.get("auc_std", float("nan")),
        }
        table2_df = pd.concat([table2_df, pd.DataFrame([h4_summary_row])], ignore_index=True)

        args.out_h4.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([h4_result]).to_csv(args.out_h4, index=False)
        print(f"Wrote {args.out_h4}")
    else:
        print("No laugh-spoof rows found; skipping H4 probe (per plan §5.3, conditional).")

    args.out_table2.parent.mkdir(parents=True, exist_ok=True)
    table2_df.to_csv(args.out_table2, index=False)
    write_table2_latex(wilcoxon_rows, args.out_table2_tex)
    print(f"Wrote {args.out_table2}")
    print(f"Wrote {args.out_table2_tex}")
    print(table2_df.to_string(index=False))

    # --- (b) UMAP figure ---
    if args.mean_emb9.exists() and args.mean_emb12.exists():
        mean_emb9 = np.load(args.mean_emb9)
        mean_emb12 = np.load(args.mean_emb12)
        try:
            plot_umap_panels(df, mean_emb9, mean_emb12, args.out_fig, seed=args.seed)
            print(f"Wrote {args.out_fig.with_suffix('.pdf')} and .png")
        except Exception as exc:
            print(f"WARNING: UMAP figure generation failed: {exc}", file=sys.stderr)
    else:
        print("WARNING: mean embedding .npy files not found; skipping Figure 1.", file=sys.stderr)


if __name__ == "__main__":
    main()
