"""Multi-method D3 — real laughter vs. EACH synthetic laughter technique in
WavLM space. Uses embeddings/d3_multi_features.parquet + d3_multi/*.npy.

Per synthetic method: 5-fold linear-probe AUC vs. laugh-real, and centroid
cosine distance to real laughter (L9 & L12). Plus a UMAP colored by method.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = Path(__file__).resolve().parents[1]
COLORS = {
    "laugh-real": "#2563eb", "speech-real": "#9ca3af",
    "laugh-bark_laughter_token": "#dc2626", "laugh-bark_laughs_inline": "#f59e0b",
    "laugh-parler_tts": "#7c3aed", "laugh-audioldm2": "#059669", "laugh-xtts": "#db2777",
}


def cos(a, b):
    a = a/(np.linalg.norm(a)+1e-9); b = b/(np.linalg.norm(b)+1e-9)
    return 1-float(a@b)


def main():
    df = pd.read_parquet(L/"embeddings"/"d3_multi_features.parquet")
    emb9 = np.load(L/"embeddings"/"d3_multi"/"mean_emb_layer9.npy")
    emb12 = np.load(L/"embeddings"/"d3_multi"/"mean_emb_layer12.npy")
    g = df.group.to_numpy()
    real_mask = g == "laugh-real"
    methods = [m for m in pd.unique(g) if m.startswith("laugh-") and m != "laugh-real"]

    rows = []
    for m in methods:
        rec = {"method": m.replace("laugh-", "")}
        for name, emb in [("L9", emb9), ("L12", emb12)]:
            ok = ~np.isnan(emb).any(axis=1)
            rm = real_mask & ok; mm = (g == m) & ok
            X = np.vstack([emb[rm], emb[mm]]); y = np.r_[np.zeros(rm.sum()), np.ones(mm.sum())]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            auc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
            d = cos(emb[rm].mean(0), emb[mm].mean(0))
            rec[f"auc_{name}"] = round(float(auc), 3)
            rec[f"cosdist_real_{name}"] = round(d, 3)
        rows.append(rec)

    with open(L/"tables"/"table_d3_multi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("=== real laughter vs each synthetic method (WavLM) ===")
    print(f"{'method':<22}{'AUC L9':>8}{'AUC L12':>9}{'cosD L12':>10}")
    for r in rows:
        print(f"{r['method']:<22}{r['auc_L9']:>8}{r['auc_L12']:>9}{r['cosdist_real_L12']:>10}")

    # UMAP
    try:
        import umap, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, emb, nm in [(axes[0], emb9, "Layer 9"), (axes[1], emb12, "Layer 12")]:
            ok = ~np.isnan(emb).any(axis=1)
            Z = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb[ok])
            gg = g[ok]
            for grp in ["speech-real", "laugh-real"] + methods:
                mk = gg == grp
                if mk.sum():
                    ax.scatter(Z[mk, 0], Z[mk, 1], s=13, alpha=0.7,
                               c=COLORS.get(grp, "#333"), label=grp.replace("laugh-", ""))
            ax.set_title(f"WavLM {nm}"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle("Real vs. synthetic laughter across 4 synthesis techniques (WavLM)")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(L/"figures"/f"figure_d3_multi.{e}", dpi=200, bbox_inches="tight")
        print("wrote figures/figure_d3_multi.png|pdf")
    except Exception as e:
        print(f"UMAP skipped: {type(e).__name__} {e}")
    print("wrote tables/table_d3_multi.csv")


if __name__ == "__main__":
    main()
