"""D3 — real vs. synthetic laughter in WavLM (SSL) space.

Uses embeddings/d3_features.parquet (C_layer12, T_layer9, group) and
embeddings/d3/mean_emb_layer{9,12}.npy from extract_wavlm_features.py.

Groups: laugh-real (VocalSound), laugh-bark (Bark synthetic), speech-real
(LibriSpeech anchor).

Outputs:
  tables/table_d3.csv|.tex  — Mann-Whitney U on C & T (real vs bark laughter),
     rank-biserial effect size, group means; 5-fold linear-probe AUC
     real-vs-synth laughter (mean-embedding features, layer 9 & 12).
  figures/figure_d3.png|.pdf — UMAP of segment-mean embeddings (L9 & L12),
     colored by group.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LAUGHSMI = Path(__file__).resolve().parents[1]
COLORS = {"laugh-real": "#2563eb", "laugh-bark": "#dc2626", "speech-real": "#9ca3af"}


def rank_biserial(a, b):
    """Effect size for Mann-Whitney: 2*U/(n1*n2) - 1."""
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (len(a) * len(b)) - 1


def main():
    df = pd.read_parquet(LAUGHSMI / "embeddings" / "d3_features.parquet")
    emb9 = np.load(LAUGHSMI / "embeddings" / "d3" / "mean_emb_layer9.npy")
    emb12 = np.load(LAUGHSMI / "embeddings" / "d3" / "mean_emb_layer12.npy")

    real = df[df.group == "laugh-real"]
    bark = df[df.group == "laugh-bark"]

    rows = []
    for feat, layer in [("C_layer12", "L12 compactness (C)"), ("T_layer9", "L9 traj. irregularity (T)")]:
        r = real[feat].dropna().to_numpy(); b = bark[feat].dropna().to_numpy()
        U, p = mannwhitneyu(r, b, alternative="two-sided")
        rows.append({"feature": layer, "real_mean": round(float(r.mean()), 4),
                     "bark_mean": round(float(b.mean()), 4),
                     "mannwhitney_p": float(p), "rank_biserial": round(rank_biserial(r, b), 3)})

    # linear probe real-vs-bark laughter on mean embeddings (no clip leakage:
    # each clip is one row, StratifiedKFold splits by clip)
    lr_mask = df.group.isin(["laugh-real", "laugh-bark"]).to_numpy()
    y = (df.group.to_numpy()[lr_mask] == "laugh-bark").astype(int)
    aucs = {}
    for name, emb in [("L9", emb9), ("L12", emb12)]:
        X = emb[lr_mask]
        ok = ~np.isnan(X).any(axis=1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        auc = cross_val_score(clf, X[ok], y[ok], cv=cv, scoring="roc_auc")
        aucs[name] = (round(float(auc.mean()), 3), round(float(auc.std()), 3))

    # write table
    out_csv = LAUGHSMI / "tables" / "table_d3.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "real_mean", "bark_mean", "mannwhitney_p", "rank_biserial"])
        for r in rows:
            w.writerow([r["feature"], r["real_mean"], r["bark_mean"], f"{r['mannwhitney_p']:.2e}", r["rank_biserial"]])
        w.writerow([])
        w.writerow(["probe_real_vs_bark_AUC (5-fold)", "mean", "std"])
        for k, (m, s) in aucs.items():
            w.writerow([k, m, s])
    print("=== D3 C/T stats (laugh-real vs laugh-bark) ===")
    for r in rows:
        print(f"  {r['feature']}: real={r['real_mean']} bark={r['bark_mean']} "
              f"p={r['mannwhitney_p']:.2e} rank-biserial={r['rank_biserial']}")
    print("=== D3 linear-probe real-vs-synth laughter AUC (5-fold) ===")
    for k, (m, s) in aucs.items():
        print(f"  {k}: AUC={m} +/- {s}")

    # UMAP figure
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, emb, name in [(axes[0], emb9, "Layer 9"), (axes[1], emb12, "Layer 12")]:
            ok = ~np.isnan(emb).any(axis=1)
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
            Z = reducer.fit_transform(emb[ok])
            g = df.group.to_numpy()[ok]
            for grp in ["speech-real", "laugh-real", "laugh-bark"]:
                m = g == grp
                ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, c=COLORS[grp], label=grp)
            ax.set_title(f"WavLM {name}"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        axes[0].legend(loc="best", fontsize=9)
        fig.suptitle("Real vs. synthetic laughter in WavLM space (segment-mean embeddings)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(LAUGHSMI / "figures" / f"figure_d3.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote figures/figure_d3.png|pdf")
    except Exception as e:
        print(f"UMAP figure skipped: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
