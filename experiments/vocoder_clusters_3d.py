"""
vocoder_clusters_3d.py
======================
Same analysis as vocoder_clusters.py but projects to 3-D instead of 2-D.

For each of the 22 representations:
  1. PCA  → 3-D scatter (two viewing angles side-by-side)
  2. t-SNE → 3-D scatter (two viewing angles side-by-side)
  3. Silhouette score on PCA-50 features (unchanged from 2-D version)

Summary outputs:
  silhouette_ranking.png    — same bar chart (scores don't change)
  grid_tsne3d_top.png       — 3-D t-SNE grid for top-6 probes (two views each)
  grid_pca3d_top.png        — 3-D PCA grid for top-6 probes (two views each)
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  registers 3-D projection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "experiments" / "results" / "linear_probe"
OUT_DIR      = REPO_ROOT / "experiments" / "results" / "vocoder_clusters_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
SYSTEMS     = ["A01", "A02", "A03", "A04", "A05", "A06"]
TOP_GRID_N  = 6
PCA50_DIM   = 50

# Two complementary viewing angles: (elevation, azimuth)
VIEWS = [(20, 45), (20, 135)]

_TAB10         = plt.cm.tab10.colors
SYSTEM_COLORS  = {s: _TAB10[i] for i, s in enumerate(SYSTEMS)}

N_GAT_LAYERS = 3
N_HEADS      = 6
PROBE_NAMES  = (
    ["bilstm"]
    + [f"gat_l{l}" for l in range(N_GAT_LAYERS)]
    + [f"head_l{l}_h{h}" for l in range(N_GAT_LAYERS) for h in range(N_HEADS)]
)

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_spoof_features(name: str):
    path = FEATURES_DIR / f"features_{name}.npz"
    d    = np.load(path, allow_pickle=True)
    X    = d["X"].astype(np.float32)
    mask = d["y"] == 1
    return X[mask], d["system_ids"][mask]


# ---------------------------------------------------------------------------
# 3-D scatter: two views side-by-side
# ---------------------------------------------------------------------------
def scatter3d(proj: np.ndarray, sids: np.ndarray,
              title: str, axis_labels: tuple[str, str, str],
              out_path: Path, extra_info: str = "") -> None:
    """Save a figure with two 3-D views of the same projection."""
    fig = plt.figure(figsize=(14, 6))
    for col, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for sid in SYSTEMS:
            mask = sids == sid
            if not mask.any():
                continue
            ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                       c=[SYSTEM_COLORS[sid]], s=8, alpha=0.55,
                       edgecolors="none", label=sid)
        ax.set_xlabel(axis_labels[0], fontsize=8, labelpad=2)
        ax.set_ylabel(axis_labels[1], fontsize=8, labelpad=2)
        ax.set_zlabel(axis_labels[2], fontsize=8, labelpad=2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}° azim={azim}°", fontsize=9)
        ax.tick_params(labelsize=6)

    handles = [mpatches.Patch(color=SYSTEM_COLORS[s], label=s) for s in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", ncol=len(SYSTEMS),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    full_title = title if not extra_info else f"{title}   [{extra_info}]"
    fig.suptitle(full_title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Per-probe analysis
# ---------------------------------------------------------------------------
def analyse_probe(name: str, X: np.ndarray, sids: np.ndarray,
                  le: LabelEncoder, methods: list[str]) -> dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_int    = le.transform(sids)

    n_pca50  = min(PCA50_DIM, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca50    = PCA(n_components=n_pca50, random_state=SEED)
    X_pca50  = pca50.fit_transform(X_scaled)

    result: dict = {"name": name, "silhouette": float("nan")}

    # Silhouette on PCA-50
    n_sil   = min(2000, len(X_pca50))
    rng     = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(X_pca50), n_sil, replace=False) if n_sil < len(X_pca50) else np.arange(len(X_pca50))
    if len(np.unique(y_int[sil_idx])) > 1:
        result["silhouette"] = float(silhouette_score(
            X_pca50[sil_idx], y_int[sil_idx], random_state=SEED))

    # 3-D PCA
    if "pca" in methods:
        n_comp = min(3, X_scaled.shape[1])
        pca3   = PCA(n_components=n_comp, random_state=SEED)
        proj   = pca3.fit_transform(X_scaled)
        var    = pca3.explained_variance_ratio_
        labels = tuple(f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(var))
        scatter3d(
            proj, sids,
            title=f"PCA 3-D  —  {name}",
            axis_labels=labels,
            extra_info=f"sil={result['silhouette']:.3f}",
            out_path=OUT_DIR / f"pca3d_{name}.png",
        )
        result["pca_proj"] = proj

    # 3-D t-SNE
    if "tsne" in methods:
        print(f"    running t-SNE 3-D for {name} ({len(X_pca50)} pts)…")
        tsne = TSNE(n_components=3, perplexity=40, n_iter=1000,
                    random_state=SEED, init="pca", learning_rate=200.0)
        proj = tsne.fit_transform(X_pca50)
        scatter3d(
            proj, sids,
            title=f"t-SNE 3-D  —  {name}",
            axis_labels=("dim 1", "dim 2", "dim 3"),
            extra_info=f"sil={result['silhouette']:.3f}",
            out_path=OUT_DIR / f"tsne3d_{name}.png",
        )
        result["tsne_proj"] = proj

    # 3-D UMAP
    if "umap" in methods and HAS_UMAP:
        print(f"    running UMAP 3-D for {name}…")
        reducer = umap_lib.UMAP(n_components=3, random_state=SEED,
                                n_neighbors=15, min_dist=0.1)
        proj = reducer.fit_transform(X_pca50)
        scatter3d(
            proj, sids,
            title=f"UMAP 3-D  —  {name}",
            axis_labels=("dim 1", "dim 2", "dim 3"),
            extra_info=f"sil={result['silhouette']:.3f}",
            out_path=OUT_DIR / f"umap3d_{name}.png",
        )
        result["umap_proj"] = proj

    return result


# ---------------------------------------------------------------------------
# Summary: grid of 3-D plots for top-N probes
# ---------------------------------------------------------------------------
def plot_3d_grid(results: list[dict], sids_map: dict[str, np.ndarray],
                 proj_key: str, method_label: str,
                 top_n: int, out_path: Path) -> None:
    ranked = sorted(
        [r for r in results if proj_key in r],
        key=lambda r: r["silhouette"], reverse=True,
    )[:top_n]
    if not ranked:
        return

    # Each probe gets 2 columns (two viewing angles), each row = one probe
    nrows = len(ranked)
    ncols = 2
    fig   = plt.figure(figsize=(11, 4.5 * nrows))

    for row, r in enumerate(ranked):
        proj = r[proj_key]
        sids = sids_map[r["name"]]
        for col, (elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1,
                                 projection="3d")
            for sid in SYSTEMS:
                mask = sids == sid
                if mask.any():
                    ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                               c=[SYSTEM_COLORS[sid]], s=7, alpha=0.5,
                               edgecolors="none")
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=5)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            subplot_title = (f"{r['name']}  (sil={r['silhouette']:.3f})"
                             if col == 0 else f"azim={azim}°")
            ax.set_title(subplot_title, fontsize=9)

    handles = [mpatches.Patch(color=SYSTEM_COLORS[s], label=s) for s in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", ncol=len(SYSTEMS),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{method_label} 3-D — top {len(ranked)} probes by silhouette score",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Silhouette ranking bar chart
# ---------------------------------------------------------------------------
def plot_silhouette_ranking(results: list[dict], out_path: Path) -> None:
    ranked = sorted(results, key=lambda r: r["silhouette"], reverse=True)
    names  = [r["name"] for r in ranked]
    scores = [r["silhouette"] for r in ranked]
    N      = len(names)

    def _colour(n):
        if n == "bilstm":       return "#4C72B0"
        if n.startswith("gat"): return "#DD8452"
        layer = int(n.split("_h")[0][-1])
        return ["#55A868", "#C44E52", "#8172B2"][layer]

    colors = [_colour(n) for n in names]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(N), scores, color=colors, width=0.65, alpha=0.88,
                  edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(N))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Silhouette score (PCA-50 features)", fontsize=10)
    ax.set_title("Vocoder-cluster silhouette score per representation\n"
                 "(higher = better separated by attack system)", fontsize=12)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002 * np.sign(v + 1e-9),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4C72B0", label="bilstm"),
        Patch(facecolor="#DD8452", label="gat full-layer"),
        Patch(facecolor="#55A868", label="head layer 0"),
        Patch(facecolor="#C44E52", label="head layer 1"),
        Patch(facecolor="#8172B2", label="head layer 2"),
    ]
    ax.legend(handles=legend_els, fontsize=8, ncol=5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    methods = ["pca", "tsne"]
    if HAS_UMAP:
        methods.append("umap")
        print("UMAP available — running PCA + t-SNE + UMAP (all 3-D)")
    else:
        print("Running PCA + t-SNE in 3-D  (install umap-learn for UMAP)")

    le = LabelEncoder()
    le.fit(SYSTEMS)

    print(f"\nProcessing {len(PROBE_NAMES)} probes...\n")
    results  : list[dict]            = []
    sids_map : dict[str, np.ndarray] = {}

    for name in PROBE_NAMES:
        print(f"[{name}]")
        X, sids = load_spoof_features(name)
        sids_map[name] = sids
        print(f"  {len(X)} spoof samples, D={X.shape[1]}")
        r = analyse_probe(name, X, sids, le, methods)
        results.append(r)
        print(f"  silhouette = {r['silhouette']:.4f}")

    # Ranking table
    ranked = sorted(results, key=lambda r: r["silhouette"], reverse=True)
    print(f"\n{'Rank':<5}  {'Probe':<28}  {'Silhouette':>10}")
    print("─" * 47)
    for rank, r in enumerate(ranked, 1):
        print(f"  {rank:<4}  {r['name']:<28}  {r['silhouette']:>10.4f}")

    # CSV
    csv_path = OUT_DIR / "silhouette_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "probe", "silhouette"])
        for rank, r in enumerate(ranked, 1):
            w.writerow([rank, r["name"], f"{r['silhouette']:.4f}"])
    print(f"\nSaved: {csv_path.name}")

    # Summary plots
    plot_silhouette_ranking(results, OUT_DIR / "silhouette_ranking.png")
    plot_3d_grid(results, sids_map, "tsne_proj", "t-SNE",
                 top_n=TOP_GRID_N, out_path=OUT_DIR / "grid_tsne3d_top.png")
    plot_3d_grid(results, sids_map, "pca_proj", "PCA",
                 top_n=TOP_GRID_N, out_path=OUT_DIR / "grid_pca3d_top.png")

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
