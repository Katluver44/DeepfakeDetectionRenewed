"""One-off: re-run t-SNE + metrics bar from saved embeddings.npz / metrics.csv."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

OUT_DIR       = Path(__file__).resolve().parent / "results" / "unseen_systems"
SEED          = 42
TSNE_PER_SYS  = 200
BONAFIDE_ID   = "-"
TRAIN_SYSTEMS  = ["A01","A02","A03","A04","A05","A06"]
UNSEEN_SYSTEMS = ["A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

_CMAP_TRAIN   = plt.cm.Reds
_CMAP_UNSEEN  = plt.cm.Blues
_BONAFIDE_COL = "#2CA02C"


def _system_colour(sid):
    if sid == BONAFIDE_ID:
        return _BONAFIDE_COL
    elif sid in TRAIN_SYSTEMS:
        idx = TRAIN_SYSTEMS.index(sid) / max(len(TRAIN_SYSTEMS) - 1, 1)
        return _CMAP_TRAIN(0.35 + 0.55 * idx)
    else:
        idx = UNSEEN_SYSTEMS.index(sid) / max(len(UNSEEN_SYSTEMS) - 1, 1)
        return _CMAP_UNSEEN(0.35 + 0.55 * idx)


def scatter_plot(proj, sids, title, xlabel, ylabel, out_path):
    all_systems = [BONAFIDE_ID] + sorted(TRAIN_SYSTEMS) + sorted(UNSEEN_SYSTEMS)
    fig, ax = plt.subplots(figsize=(12, 9))
    for sid in all_systems:
        mask = sids == sid
        if not mask.any(): continue
        col    = _system_colour(sid)
        label  = sid if sid != BONAFIDE_ID else "Bonafide"
        marker = "^" if sid == BONAFIDE_ID else ("s" if sid in TRAIN_SYSTEMS else "o")
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[col], s=18, alpha=0.65, label=label, marker=marker,
                   edgecolors="none")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=3, markerscale=1.5, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_metrics(df_rows, out_path):
    systems  = [r["system_id"] for r in df_rows]
    aucs     = [float(r["auc"]) for r in df_rows]
    one_eer  = [1 - float(r["eer"]) for r in df_rows]
    x = np.arange(len(systems))

    train_set  = set(TRAIN_SYSTEMS)
    colours_auc = []
    for s in systems:
        if s == BONAFIDE_ID: colours_auc.append("#2CA02C")
        elif s in train_set: colours_auc.append("#D62728")
        else:                colours_auc.append("#1F77B4")

    fig, ax = plt.subplots(figsize=(16, 5))
    w = 0.38
    ax.bar(x - w/2, aucs,    width=w, color=colours_auc, alpha=0.85, label="AUC")
    ax.bar(x + w/2, one_eer, width=w, color=colours_auc, alpha=0.45,
           hatch="//", label="1 − EER")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUC  /  1−EER")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-system AUC and 1−EER\n"
                 "Red = trained (val split)  |  Blue = unseen (test split)  |  "
                 "Solid = AUC  |  Hatched = 1−EER", fontsize=10)
    for i, (a, e) in enumerate(zip(aucs, one_eer)):
        ax.text(i - w/2, a + 0.005, f"{a:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i + w/2, e + 0.005, f"{e:.2f}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Load embeddings ──────────────────────────────────────────────────────────
print("Loading embeddings.npz ...")
d = np.load(OUT_DIR / "embeddings.npz", allow_pickle=True)
X           = d["X"].astype(np.float32)
all_sys_ids = d["system_ids"]
print(f"  {len(X)} embeddings, shape {X.shape}")

# ── Scale ────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── t-SNE ────────────────────────────────────────────────────────────────────
print("Running t-SNE ...")
all_systems_list = [BONAFIDE_ID] + TRAIN_SYSTEMS + UNSEEN_SYSTEMS
rng     = np.random.default_rng(SEED)
tsne_idx = []
for sid in all_systems_list:
    idx = np.where(all_sys_ids == sid)[0]
    if len(idx) == 0: continue
    chosen = rng.choice(idx, min(TSNE_PER_SYS, len(idx)), replace=False)
    tsne_idx.extend(chosen.tolist())
tsne_idx = np.array(tsne_idx)

X_tsne_in = X_scaled[tsne_idx]
sids_tsne  = all_sys_ids[tsne_idx]

tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
            random_state=SEED, init="pca", learning_rate="auto")
tsne_proj = tsne.fit_transform(X_tsne_in)
print(f"  t-SNE done ({len(tsne_idx)} points)")

scatter_plot(
    tsne_proj, sids_tsne,
    title="t-SNE of GAT layer-0 embeddings — all systems\n"
          "(▲=bonafide  ■=trained A01–A06  ●=unseen A07–A19)",
    xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
    out_path=OUT_DIR / "tsne_unseen.png")

# ── Metrics bar chart ────────────────────────────────────────────────────────
print("Loading metrics.csv ...")
metrics_rows = []
with open(OUT_DIR / "metrics.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["system_id"] != BONAFIDE_ID:
            metrics_rows.append(row)

plot_metrics(metrics_rows, OUT_DIR / "metrics_bar.png")
print("Done.")
