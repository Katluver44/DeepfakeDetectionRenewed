"""
vocoder_clusters.py
===================
Dimensionality-reduction visualisation of internal representations coloured
by vocoder / attack system (A01–A06), to see how well the model separates
spoofing systems in its internal feature space.

Uses feature matrices already saved by linear_probe.py:
  experiments/results/linear_probe/features_<name>.npz
  keys: X (N, D), y (N,), system_ids (N,)

Mirrors the data setup in multiclass_probe.py:
  - spoof-only samples (y == 1), systems A01–A06
  - same random seed (42)

For each of the 22 representations:
  1. PCA  → scatter (all samples)
  2. t-SNE → scatter (PCA-50 init, all spoof samples)
  3. UMAP  → scatter (if umap-learn is installed)
  4. Silhouette score on PCA-50 features

Saves to experiments/results/vocoder_clusters/:
  pca_{name}.png
  tsne_{name}.png
  umap_{name}.png          (only if umap-learn is installed)
  silhouette_ranking.png   — bar chart ranked by silhouette score
  silhouette_summary.csv   — table
  grid_tsne_top.png        — t-SNE grid for the top-N representations
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "experiments" / "results" / "linear_probe"
OUT_DIR      = REPO_ROOT / "experiments" / "results" / "vocoder_clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
SYSTEMS      = ["A01", "A02", "A03", "A04", "A05", "A06"]
TOP_GRID_N   = 6      # how many probes to include in the summary t-SNE grid
PCA50_DIM    = 50     # intermediate PCA dim for t-SNE / UMAP / silhouette

# One distinct colour per system (matplotlib tab10)
_TAB10 = plt.cm.tab10.colors
SYSTEM_COLORS = {s: _TAB10[i] for i, s in enumerate(SYSTEMS)}

# Probe names in canonical order (matches multiclass_probe.py)
N_GAT_LAYERS = 3
N_HEADS      = 6
PROBE_NAMES  = (
    ["bilstm"]
    + [f"gat_l{l}" for l in range(N_GAT_LAYERS)]
    + [f"head_l{l}_h{h}" for l in range(N_GAT_LAYERS) for h in range(N_HEADS)]
)

# Try to import UMAP
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------------
# Data loading (mirrors multiclass_probe.py)
# ---------------------------------------------------------------------------
def load_spoof_features(name: str):
    """Return X (N_spoof, D), system_id_labels (N_spoof,) for A01–A06."""
    path = FEATURES_DIR / f"features_{name}.npz"
    d    = np.load(path, allow_pickle=True)
    X    = d["X"].astype(np.float32)
    y    = d["y"]
    sids = d["system_ids"]
    mask = y == 1
    return X[mask], sids[mask]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _legend_handles():
    return [mpatches.Patch(color=SYSTEM_COLORS[s], label=s) for s in SYSTEMS]


def scatter2d(proj: np.ndarray, sids: np.ndarray,
              title: str, xlabel: str, ylabel: str,
              out_path: Path, extra_info: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for sid in SYSTEMS:
        mask = sids == sid
        if not mask.any():
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[SYSTEM_COLORS[sid]], s=14, alpha=0.65,
                   edgecolors="none", label=sid)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    full_title = title if not extra_info else f"{title}\n{extra_info}"
    ax.set_title(full_title, fontsize=11)
    ax.legend(handles=_legend_handles(), fontsize=9, ncol=3,
              loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Per-probe analysis
# ---------------------------------------------------------------------------
def analyse_probe(name: str, X: np.ndarray, sids: np.ndarray,
                  le: LabelEncoder, methods: list[str]) -> dict:
    """
    Run dimensionality reductions on X (spoof-only, shape (N, D)).
    Returns dict with silhouette score and 2-D projections.
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_int   = le.transform(sids)   # integer labels for silhouette

    # PCA-50 (used as input to t-SNE / UMAP and for silhouette)
    n_pca50 = min(PCA50_DIM, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca50   = PCA(n_components=n_pca50, random_state=SEED)
    X_pca50 = pca50.fit_transform(X_scaled)

    result: dict = {"name": name, "silhouette": float("nan")}

    # Silhouette score (subsample to max 2000 for speed)
    n_sil = min(2000, len(X_pca50))
    rng   = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(X_pca50), n_sil, replace=False) if n_sil < len(X_pca50) else np.arange(len(X_pca50))
    if len(np.unique(y_int[sil_idx])) > 1:
        result["silhouette"] = float(silhouette_score(
            X_pca50[sil_idx], y_int[sil_idx], random_state=SEED))

    # PCA 2-D
    if "pca" in methods:
        pca2  = PCA(n_components=2, random_state=SEED)
        proj  = pca2.fit_transform(X_scaled)
        var   = pca2.explained_variance_ratio_
        scatter2d(
            proj, sids,
            title=f"PCA  —  {name}",
            xlabel=f"PC1 ({var[0]*100:.1f}% var)",
            ylabel=f"PC2 ({var[1]*100:.1f}% var)",
            extra_info=f"silhouette = {result['silhouette']:.3f}",
            out_path=OUT_DIR / f"pca_{name}.png",
        )
        result["pca_proj"] = proj

    # t-SNE
    if "tsne" in methods:
        print(f"    running t-SNE for {name} ({len(X_pca50)} pts)…")
        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                    random_state=SEED, init="pca", learning_rate=200.0)
        proj = tsne.fit_transform(X_pca50)
        scatter2d(
            proj, sids,
            title=f"t-SNE  —  {name}",
            xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
            extra_info=f"silhouette = {result['silhouette']:.3f}",
            out_path=OUT_DIR / f"tsne_{name}.png",
        )
        result["tsne_proj"] = proj

    # UMAP
    if "umap" in methods and HAS_UMAP:
        print(f"    running UMAP for {name}…")
        reducer = umap_lib.UMAP(n_components=2, random_state=SEED, n_neighbors=15,
                                min_dist=0.1, metric="euclidean")
        proj = reducer.fit_transform(X_pca50)
        scatter2d(
            proj, sids,
            title=f"UMAP  —  {name}",
            xlabel="UMAP dim 1", ylabel="UMAP dim 2",
            extra_info=f"silhouette = {result['silhouette']:.3f}",
            out_path=OUT_DIR / f"umap_{name}.png",
        )
        result["umap_proj"] = proj

    return result


# ---------------------------------------------------------------------------
# Summary plots
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
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
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


def plot_tsne_grid(results: list[dict], sids_map: dict[str, np.ndarray],
                   top_n: int, out_path: Path) -> None:
    """Grid of t-SNE scatter plots for the top-N probes by silhouette score."""
    ranked = sorted(
        [r for r in results if "tsne_proj" in r],
        key=lambda r: r["silhouette"], reverse=True
    )[:top_n]

    if not ranked:
        return

    ncols = min(3, len(ranked))
    nrows = -(-len(ranked) // ncols)   # ceiling division

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, r in zip(axes_flat, ranked):
        proj = r["tsne_proj"]
        sids = sids_map[r["name"]]
        for sid in SYSTEMS:
            mask = sids == sid
            if mask.any():
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=[SYSTEM_COLORS[sid]], s=10, alpha=0.6,
                           edgecolors="none")
        ax.set_title(f"{r['name']}\n(sil={r['silhouette']:.3f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused axes
    for ax in axes_flat[len(ranked):]:
        ax.set_visible(False)

    # Shared legend at the bottom
    handles = _legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=len(SYSTEMS),
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"t-SNE — top {len(ranked)} probes by vocoder silhouette score",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
        print("UMAP available — will run PCA + t-SNE + UMAP")
    else:
        print("umap-learn not installed — running PCA + t-SNE only")
        print("  (install with: pip install umap-learn)")

    le = LabelEncoder()
    le.fit(SYSTEMS)

    print(f"\nProcessing {len(PROBE_NAMES)} probes...\n")
    results  : list[dict]          = []
    sids_map : dict[str, np.ndarray] = {}

    for name in PROBE_NAMES:
        print(f"[{name}]")
        X, sids = load_spoof_features(name)
        sids_map[name] = sids
        print(f"  {len(X)} spoof samples, D={X.shape[1]}")
        r = analyse_probe(name, X, sids, le, methods)
        results.append(r)
        print(f"  silhouette = {r['silhouette']:.4f}")

    # --- Ranking summary ---------------------------------------------------
    ranked = sorted(results, key=lambda r: r["silhouette"], reverse=True)
    print(f"\n{'Rank':<5}  {'Probe':<28}  {'Silhouette':>10}")
    print("─" * 47)
    for rank, r in enumerate(ranked, 1):
        print(f"  {rank:<4}  {r['name']:<28}  {r['silhouette']:>10.4f}")

    # --- CSV ---------------------------------------------------------------
    csv_path = OUT_DIR / "silhouette_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "probe", "silhouette"])
        for rank, r in enumerate(ranked, 1):
            w.writerow([rank, r["name"], f"{r['silhouette']:.4f}"])
    print(f"\nSaved: {csv_path.name}")

    # --- Summary plots -----------------------------------------------------
    plot_silhouette_ranking(results, OUT_DIR / "silhouette_ranking.png")
    plot_tsne_grid(results, sids_map, top_n=TOP_GRID_N,
                   out_path=OUT_DIR / "grid_tsne_top.png")

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
