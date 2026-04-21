"""
unseen_analysis.py
==================
Four analyses of gat_l0 embeddings for unseen vocoder systems.

Loads experiments/results/unseen_systems/embeddings.npz (from unseen_systems.py).

Produces (saved to experiments/results/unseen_analysis/):
  1. dendrogram.png        — hierarchical clustering of 19 system centroids
  2. centroid_heatmap.png  — 19×19 cosine-similarity heatmap of centroids
  3. umap.png              — UMAP scatter, coloured by system
  4. knn_confusion.png     — K-NN probe (trained on A01–A06) → predicted class
                             probabilities for each unseen system; + accuracy table
  4b.knn_summary.csv       — per-system KNN top-1 / top-2 predicted trained system
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
EMB_PATH   = REPO_ROOT / "experiments" / "results" / "unseen_systems" / "embeddings.npz"
OUT_DIR    = REPO_ROOT / "experiments" / "results" / "unseen_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

BONAFIDE_ID    = "-"
TRAIN_SYSTEMS  = ["A01","A02","A03","A04","A05","A06"]
UNSEEN_SYSTEMS = ["A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]
ALL_SYSTEMS    = [BONAFIDE_ID] + TRAIN_SYSTEMS + UNSEEN_SYSTEMS   # 20 total

# Colour helpers
_CMAP_TRAIN   = plt.cm.Reds
_CMAP_UNSEEN  = plt.cm.Blues
_BONAFIDE_COL = "#2CA02C"

def _sys_colour(sid):
    if sid == BONAFIDE_ID:
        return _BONAFIDE_COL
    if sid in TRAIN_SYSTEMS:
        t = TRAIN_SYSTEMS.index(sid) / max(len(TRAIN_SYSTEMS) - 1, 1)
        return _CMAP_TRAIN(0.35 + 0.55 * t)
    t = UNSEEN_SYSTEMS.index(sid) / max(len(UNSEEN_SYSTEMS) - 1, 1)
    return _CMAP_UNSEEN(0.35 + 0.55 * t)


# ── Load & scale ─────────────────────────────────────────────────────────────
print("Loading embeddings ...")
d           = np.load(EMB_PATH, allow_pickle=True)
X_raw       = d["X"].astype(np.float32)
sys_ids     = d["system_ids"]          # (N,) string array

scaler      = StandardScaler()
X           = scaler.fit_transform(X_raw)

# Per-system centroids (L2-normalised for cosine similarity)
systems_present = [s for s in ALL_SYSTEMS if (sys_ids == s).any()]
centroids   = {}
for sid in systems_present:
    m = X[sys_ids == sid].mean(axis=0)
    centroids[sid] = m / (np.linalg.norm(m) + 1e-12)

cent_matrix = np.stack([centroids[s] for s in systems_present])   # (20, D)
print(f"  {len(X)} embeddings, {len(systems_present)} systems")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dendrogram of system centroids
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/4] Dendrogram ...")

# Use cosine distance between L2-normed centroids = 1 – dot product
cos_dist  = squareform(pdist(cent_matrix, metric="cosine"))
linkage_m = linkage(cos_dist, method="ward", optimal_ordering=True)

label_colours = [_sys_colour(s) for s in systems_present]

fig, ax = plt.subplots(figsize=(14, 6))
ddata = dendrogram(
    linkage_m,
    labels=systems_present,
    leaf_rotation=45,
    leaf_font_size=10,
    color_threshold=0,        # all black lines; we colour tick labels instead
    above_threshold_color="k",
    ax=ax,
)
# Colour the x-axis tick labels by group
for lbl in ax.get_xmajorticklabels():
    sid = lbl.get_text()
    lbl.set_color(_sys_colour(sid))
    lbl.set_fontweight("bold" if sid in TRAIN_SYSTEMS else "normal")

ax.set_ylabel("Ward distance (cosine)", fontsize=10)
ax.set_title("Hierarchical clustering of system centroids (gat_l0)\n"
             "Red labels = trained  |  Blue labels = unseen  |  Green = bonafide",
             fontsize=11)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
fig.tight_layout()
fig.savefig(OUT_DIR / "dendrogram.png", dpi=150)
plt.close(fig)
print("  Saved: dendrogram.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pairwise centroid cosine-similarity heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("[2/4] Centroid similarity heatmap ...")

cos_sim   = 1 - cos_dist   # (20, 20)
N         = len(systems_present)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cos_sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
ax.set_xticks(range(N)); ax.set_xticklabels(systems_present, rotation=90, fontsize=8)
ax.set_yticks(range(N)); ax.set_yticklabels(systems_present, fontsize=8)

# Colour tick labels
for lbl in ax.get_xmajorticklabels() + ax.get_ymajorticklabels():
    sid = lbl.get_text()
    lbl.set_color(_sys_colour(sid))

# Cell annotations (only for cells ≥ 0.3 to avoid clutter)
for i in range(N):
    for j in range(N):
        v = cos_sim[i, j]
        if abs(v) >= 0.15 or i == j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if 0.2 < v < 0.8 else "white")

# Draw separator lines between groups (bonafide | trained | unseen)
n_bon   = sum(1 for s in systems_present if s == BONAFIDE_ID)
n_train = sum(1 for s in systems_present if s in TRAIN_SYSTEMS)
for cut in [n_bon - 0.5, n_bon + n_train - 0.5]:
    ax.axhline(cut, color="white", linewidth=2)
    ax.axvline(cut, color="white", linewidth=2)

ax.set_title("Pairwise cosine similarity of system centroids (gat_l0)\n"
             "Red labels = trained  |  Blue labels = unseen  |  Green = bonafide",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "centroid_heatmap.png", dpi=150)
plt.close(fig)
print("  Saved: centroid_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. UMAP
# ═══════════════════════════════════════════════════════════════════════════
print("[3/4] UMAP ...")
import umap as _umap   # import here so earlier stages don't fail if missing

UMAP_PER_SYS = 300

rng = np.random.default_rng(SEED)
umap_idx = []
for sid in systems_present:
    idx = np.where(sys_ids == sid)[0]
    chosen = rng.choice(idx, min(UMAP_PER_SYS, len(idx)), replace=False)
    umap_idx.extend(chosen.tolist())
umap_idx = np.array(umap_idx)

X_umap_in  = X[umap_idx]
sids_umap  = sys_ids[umap_idx]

reducer   = _umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                        metric="cosine", random_state=SEED)
umap_proj = reducer.fit_transform(X_umap_in)
print(f"  UMAP done ({len(umap_idx)} points)")

fig, ax = plt.subplots(figsize=(12, 9))
for sid in systems_present:
    mask   = sids_umap == sid
    if not mask.any(): continue
    col    = _sys_colour(sid)
    marker = "^" if sid == BONAFIDE_ID else ("s" if sid in TRAIN_SYSTEMS else "o")
    label  = "Bonafide" if sid == BONAFIDE_ID else sid
    ax.scatter(umap_proj[mask, 0], umap_proj[mask, 1],
               c=[col], s=18, alpha=0.65, marker=marker,
               edgecolors="none", label=label)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of gat_l0 embeddings — all systems\n"
             "(▲ bonafide  ■ trained A01–A06  ● unseen A07–A19)", fontsize=12)
ax.legend(fontsize=7, ncol=3, markerscale=1.5, loc="best", framealpha=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / "umap.png", dpi=150)
plt.close(fig)
print("  Saved: umap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. K-NN probe: trained on A01–A06, predict on A07–A19
# ═══════════════════════════════════════════════════════════════════════════
print("[4/4] K-NN probe ...")

# Training set: all A01-A06 samples
train_mask = np.array([s in TRAIN_SYSTEMS for s in sys_ids])
X_train    = X[train_mask]
y_train    = sys_ids[train_mask]

le = LabelEncoder(); le.fit(TRAIN_SYSTEMS)
y_train_enc = le.transform(y_train)

knn = KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance",
                            algorithm="brute", n_jobs=-1)
knn.fit(X_train, y_train_enc)
print(f"  KNN fitted on {len(X_train)} training samples ({len(TRAIN_SYSTEMS)} classes)")

# Predict on every system (including trained ones as a sanity check)
results = []
for sid in systems_present:
    mask   = sys_ids == sid
    X_sid  = X[mask]
    proba  = knn.predict_proba(X_sid)          # (N, 6) — soft votes
    mean_p = proba.mean(axis=0)                # (6,) average class probability
    top1   = le.classes_[mean_p.argmax()]
    sorted_idx = mean_p.argsort()[::-1]
    top2   = le.classes_[sorted_idx[1]]
    results.append({
        "system_id": sid,
        "top1": top1, "top1_prob": float(mean_p.max()),
        "top2": top2, "top2_prob": float(mean_p[sorted_idx[1]]),
        "mean_proba": mean_p,
    })

# ── Stacked bar: mean predicted class probability per query system ──────────
query_systems = [r["system_id"] for r in results]
proba_matrix  = np.stack([r["mean_proba"] for r in results])   # (20, 6)

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(query_systems))
bottoms = np.zeros(len(query_systems))
train_colours = [_sys_colour(s) for s in TRAIN_SYSTEMS]

for ci, (cls, col) in enumerate(zip(TRAIN_SYSTEMS, train_colours)):
    vals = proba_matrix[:, ci]
    bars = ax.bar(x, vals, bottom=bottoms, color=col, label=cls, alpha=0.85)
    bottoms += vals

# Mark which systems are trained vs unseen
for i, sid in enumerate(query_systems):
    tag = "" if sid == BONAFIDE_ID else ("★" if sid in TRAIN_SYSTEMS else "")
    if tag:
        ax.text(i, -0.07, tag, ha="center", va="top", fontsize=10,
                color="darkred" if sid in TRAIN_SYSTEMS else "navy",
                transform=ax.get_xaxis_transform())

ax.set_xticks(x)
ax.set_xticklabels(query_systems, rotation=45, ha="right", fontsize=9)
for lbl in ax.get_xmajorticklabels():
    sid = lbl.get_text()
    lbl.set_color(_sys_colour(sid))
ax.set_ylabel("Mean predicted class probability")
ax.set_ylim(0, 1.05)
ax.set_title("K-NN probe (k=15): predicted A01–A06 class distribution\n"
             "for every system  (★ = trained system, used as sanity check)",
             fontsize=11)
ax.legend(title="Predicted as", fontsize=8, loc="upper right", ncol=2)
ax.axhline(1/6, color="gray", linestyle="--", linewidth=0.8,
           alpha=0.6, label="Chance (1/6)")
# Draw separator between bonafide | trained | unseen
n_bon   = sum(1 for s in query_systems if s == BONAFIDE_ID)
n_train = sum(1 for s in query_systems if s in TRAIN_SYSTEMS)
for cut in [n_bon - 0.5, n_bon + n_train - 0.5]:
    ax.axvline(cut, color="black", linewidth=1.2, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "knn_confusion.png", dpi=150)
plt.close(fig)
print("  Saved: knn_confusion.png")

# ── Console summary ───────────────────────────────────────────────────────
print(f"\n{'System':<8}  {'Top-1':>6}  {'P(top1)':>8}  {'Top-2':>6}  {'P(top2)':>8}  {'Category'}")
print("─" * 64)
for r in results:
    sid  = r["system_id"]
    cat  = "bonafide" if sid == BONAFIDE_ID else ("trained" if sid in TRAIN_SYSTEMS else "UNSEEN")
    print(f"  {sid:<6}  {r['top1']:>6}  {r['top1_prob']:>8.3f}  {r['top2']:>6}  {r['top2_prob']:>8.3f}  {cat}")

# ── CSV ───────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "knn_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system_id","category","top1","top1_prob",
                                       "top2","top2_prob"] + TRAIN_SYSTEMS)
    w.writeheader()
    for r in results:
        sid  = r["system_id"]
        cat  = "bonafide" if sid == BONAFIDE_ID else ("trained" if sid in TRAIN_SYSTEMS else "unseen")
        row  = {"system_id": sid, "category": cat,
                "top1": r["top1"], "top1_prob": f"{r['top1_prob']:.4f}",
                "top2": r["top2"], "top2_prob": f"{r['top2_prob']:.4f}"}
        for ci, cls in enumerate(TRAIN_SYSTEMS):
            row[cls] = f"{r['mean_proba'][ci]:.4f}"
        w.writerow(row)
print(f"  Saved: knn_summary.csv")
print(f"\nAll artefacts saved to {OUT_DIR}/")
