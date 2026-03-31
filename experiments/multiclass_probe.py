"""
multiclass_probe.py
===================
6-way linear probe on spoof-only samples (A01–A06).

Uses the feature matrices already extracted by linear_probe.py.
For each of the 22 representations (bilstm, gat_l0/1/2, head_l{0-2}_h{0-5}):
  1. Filter to spoof rows only (system_id in A01–A06).
  2. Train a 6-class logistic regression (one-vs-rest, L2, balanced weights).
  3. Evaluate: per-class precision/recall/F1, macro averages, confusion matrix.

Plots saved to experiments/results/multiclass_probe/:
  confusion_<name>.png        — normalised confusion matrix per probe
  probe_similarity.png        — pairwise inter-probe agreement (prediction agreement)
  multiclass_ranking.png      — macro-F1 bar chart
  multiclass_summary.csv      — one row per probe
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "experiments" / "results" / "linear_probe"
OUT_DIR      = REPO_ROOT / "experiments" / "results" / "multiclass_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
TEST_FRAC    = 0.20
SYSTEMS      = ["A01", "A02", "A03", "A04", "A05", "A06"]

# Probe names in display order (matches linear_probe.py)
N_GAT_LAYERS = 3
N_HEADS      = 6
PROBE_NAMES  = (
    ["bilstm"]
    + [f"gat_l{l}" for l in range(N_GAT_LAYERS)]
    + [f"head_l{l}_h{h}" for l in range(N_GAT_LAYERS) for h in range(N_HEADS)]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_spoof_features(name: str):
    """Return X (N_spoof, D), y_sys (N_spoof,) of system IDs."""
    path = FEATURES_DIR / f"features_{name}.npz"
    d    = np.load(path, allow_pickle=True)
    X    = d["X"].astype(np.float32)
    y    = d["y"]                      # binary label (1 = spoof)
    sids = d["system_ids"]
    mask = y == 1
    return X[mask], sids[mask]


def run_probe(name: str, X: np.ndarray, y: np.ndarray,
              train_mask: np.ndarray) -> dict:
    """
    Train 6-class probe, return metrics dict + test predictions.
    y : integer class labels 0–5.
    """
    scaler    = StandardScaler()
    X_tr      = scaler.fit_transform(X[train_mask])
    X_te      = scaler.transform(X[~train_mask])
    y_tr, y_te = y[train_mask], y[~train_mask]

    clf = LogisticRegression(
        C=1.0, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs",
        max_iter=2000, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)   # (N_te, 6)

    acc      = float(accuracy_score(y_te, preds))
    macro_f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    report   = classification_report(
        y_te, preds, target_names=SYSTEMS,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_te, preds, normalize="true")  # row = true, col = pred

    # Save probe weights
    np.savez_compressed(
        OUT_DIR / f"probe_mc_{name}.npz",
        coef=clf.coef_,            # (6, D)
        intercept=clf.intercept_,  # (6,)
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=np.array(SYSTEMS),
    )

    return {
        "name":     name,
        "acc":      acc,
        "macro_f1": macro_f1,
        "report":   report,
        "cm":       cm,
        "preds":    preds,
        "y_te":     y_te,
        "coef":     clf.coef_,    # kept for similarity matrix
    }


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(SYSTEMS)
    ax.set_xticks(range(n)); ax.set_xticklabels(SYSTEMS, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(SYSTEMS, fontsize=9)
    ax.set_xlabel("Predicted system"); ax.set_ylabel("True system")
    ax.set_title(title, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i,j] > 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ranking(results: list[dict], out_path: Path) -> None:
    results_sorted = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    names  = [r["name"] for r in results_sorted]
    f1s    = [r["macro_f1"] for r in results_sorted]
    accs   = [r["acc"] for r in results_sorted]
    N      = len(names)

    # Colour by probe type
    def colour(n):
        if n == "bilstm":       return "#4C72B0"
        if n.startswith("gat"): return "#DD8452"
        layer = int(n.split("_h")[0][-1])
        return ["#55A868", "#C44E52", "#8172B2"][layer]

    colours = [colour(n) for n in names]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(N)
    bars = ax.bar(x, f1s, color=colours, width=0.6, alpha=0.88)
    ax.scatter(x, accs, color="black", s=18, zorder=5, label="Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1  (bars)  /  Accuracy  (dots)")
    ax.set_title("6-way attack-system probe — macro F1 ranking", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)

    # Legend patches for probe types
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4C72B0", label="bilstm"),
        Patch(facecolor="#DD8452", label="gat full-layer"),
        Patch(facecolor="#55A868", label="head layer 0"),
        Patch(facecolor="#C44E52", label="head layer 1"),
        Patch(facecolor="#8172B2", label="head layer 2"),
    ]
    ax.legend(handles=legend_els + [
        plt.scatter([], [], c="black", s=18, label="Accuracy")],
        fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_similarity(results: list[dict], out_path: Path) -> None:
    """
    Pairwise prediction-agreement matrix across probes.
    Agreement(i, j) = fraction of test samples where probe i and probe j
    predict the same class.  Uses the shared test indices.
    """
    names = [r["name"] for r in results]
    N     = len(names)
    preds_all = np.stack([r["preds"] for r in results], axis=0)  # (N_probes, N_te)

    agree = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            agree[i, j] = float((preds_all[i] == preds_all[j]).mean())

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(agree, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Prediction agreement")
    ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Inter-probe prediction agreement (6-way spoof classification)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED)

    # ── Load spoof features for all probes and verify they're consistent ──────
    print("Loading spoof features ...")
    all_X   = {}
    all_sid = {}
    for name in PROBE_NAMES:
        X, sids = load_spoof_features(name)
        all_X[name]   = X
        all_sid[name] = sids

    # All probes share the same sample ordering — use bilstm as reference
    ref_sids = all_sid["bilstm"]
    n_spoof  = len(ref_sids)
    print(f"Spoof samples per probe: {n_spoof}  "
          f"({', '.join(f'{s}={int((ref_sids==s).sum())}' for s in SYSTEMS)})")

    # Encode system IDs → integers
    le = LabelEncoder()
    le.fit(SYSTEMS)
    y = le.transform(ref_sids)   # 0–5

    # ── Stratified train/test split (shared across all probes) ───────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    train_idx, test_idx = next(sss.split(np.zeros(n_spoof), y))
    train_mask = np.zeros(n_spoof, dtype=bool)
    train_mask[train_idx] = True

    # ── Train probes ─────────────────────────────────────────────────────────
    results = []
    print(f"\n{'Probe':<28}  {'Acc':>6}  {'MacroF1':>8}")
    print("─" * 46)
    for name in PROBE_NAMES:
        m = run_probe(name, all_X[name], y, train_mask)
        results.append(m)
        print(f"  {name:<26}  {m['acc']:.4f}  {m['macro_f1']:.4f}")

        # Per-probe confusion matrix
        plot_confusion(
            m["cm"],
            title=f"Confusion — {name}",
            out_path=OUT_DIR / f"confusion_{name}.png",
        )

    # ── Ranking table (stdout) ────────────────────────────────────────────────
    ranked = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    print(f"\n {'Rank':<5}  {'Probe':<28}  {'Acc':>6}  {'MacroF1':>8}")
    print("─" * 54)
    for rank, m in enumerate(ranked, 1):
        print(f"  {rank:<4}  {m['name']:<28}  {m['acc']:.4f}  {m['macro_f1']:.4f}")

    # Per-class breakdown for the best probe
    best = ranked[0]
    print(f"\nBest probe: {best['name']}")
    print(classification_report(
        best["y_te"], best["preds"], target_names=SYSTEMS, zero_division=0))

    # ── CSV summary ──────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "multiclass_summary.csv"
    fieldnames = ["probe", "acc", "macro_f1"] + [
        f"{s}_{m}" for s in SYSTEMS for m in ("precision", "recall", "f1-score")
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in results:
            row = {"probe": m["name"], "acc": f"{m['acc']:.4f}",
                   "macro_f1": f"{m['macro_f1']:.4f}"}
            for s in SYSTEMS:
                for metric in ("precision", "recall", "f1-score"):
                    row[f"{s}_{metric}"] = f"{m['report'][s][metric]:.4f}"
            w.writerow(row)
    print(f"\nSaved: {csv_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_ranking(results, OUT_DIR / "multiclass_ranking.png")
    plot_similarity(results, OUT_DIR / "probe_similarity.png")

    print(f"\nAll artefacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
