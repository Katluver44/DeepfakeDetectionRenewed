#!/usr/bin/env python3
"""
e9_analysis.py — Mechanism analysis on E9 checkpoints.

Tests H1 (Gini split), H2 (TTS vs VC classifier), H3 (quasi-adjacency F1).
Also reports H4 (dominant feature identity).

Reads all_layers_artifacts.pt from each seed, runs the three analyses,
and produces a comparison table against original seed-1.

Usage:
  venv/bin/python3 experiments/e9_analysis.py --seeds 1 2
  venv/bin/python3 experiments/e9_analysis.py --seeds 1 2 --compare_original

Outputs:
  experiments/results/e9_mixed_training/analysis_seed{N}.json
  experiments/results/e9_mixed_training/gini_comparison.png
  experiments/results/e9_mixed_training/analysis_report.md
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",            type=int, nargs="+", default=[1, 2])
parser.add_argument("--compare_original", type=int, default=1,
                    help="Also load original ASVspoof seed-1 artifacts for comparison")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR      = REPO_ROOT / "experiments" / "results" / "e9_mixed_training"
ORIG_ARTS    = REPO_ROOT / "experiments" / "results" / "gat_attn_graphs" / "all_layers_artifacts.pt"
SWEEP_PERCS  = list(range(50, 100)) + [99.5, 99.9]


# ── A_agg construction (same as gat_quasi_adj.py) ────────────────────────────

def sparse_to_dense(attn_mean: np.ndarray, edge_index: np.ndarray, N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.float64)
    np.add.at(A, (edge_index[1], edge_index[0]), attn_mean)
    no_in = np.where(A.sum(axis=1) < 1e-10)[0]
    A[no_in, no_in] = 1.0
    return A


def build_agg_graph(rec: dict) -> np.ndarray:
    N  = rec["n_nodes"]
    ei = rec["edge_index"].numpy()
    layers = [sparse_to_dense(rec[f"attn_l{li}"].float().numpy().mean(axis=1), ei, N)
              for li in range(3)]
    A_agg = layers[2] @ layers[1] @ layers[0]
    rs = A_agg.sum(axis=1, keepdims=True)
    return A_agg / np.where(rs > 1e-10, rs, 1.0)


def gini_indegree(A_agg: np.ndarray) -> float:
    N = A_agg.shape[0]
    mask = ~np.eye(N, dtype=bool)
    indeg = A_agg[mask].reshape(N, N - 1).sum(axis=1)
    if indeg.sum() < 1e-12:
        return 0.0
    indeg = np.sort(indeg)
    n = len(indeg)
    return float((2 * np.arange(1, n + 1) @ indeg - (n + 1) * indeg.sum()) /
                 (n * indeg.sum()))


def offdiag_frobenius(A_agg: np.ndarray) -> float:
    N = A_agg.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return float(np.sqrt((A_agg[mask] ** 2).sum()))


def entropy_row(A_agg: np.ndarray) -> float:
    N = A_agg.shape[0]
    mask = ~np.eye(N, dtype=bool)
    rows = A_agg[mask].reshape(N, N - 1)
    rows = rows / (rows.sum(axis=1, keepdims=True) + 1e-12)
    H = -(rows * np.log(rows + 1e-12)).sum(axis=1)
    return float(H.mean())


def compute_features(A_agg: np.ndarray) -> np.ndarray:
    """10-feature vector matching E7 (gat_e7.py)."""
    N = A_agg.shape[0]
    mask = ~np.eye(N, dtype=bool)
    off  = A_agg[mask]
    indeg = A_agg[mask].reshape(N, N - 1).sum(axis=1)
    outdeg= A_agg[mask].reshape(N, N - 1).sum(axis=0) if N > 1 else np.zeros(1)

    feats = np.array([
        gini_indegree(A_agg),              # 0: gini_indegree
        offdiag_frobenius(A_agg),           # 1: offdiag_frobenius
        entropy_row(A_agg),                 # 2: entropy_row
        float(off.max()),                   # 3: max_offdiag
        float(indeg.mean()),                # 4: mean_indeg
        float(indeg.std() + 1e-12),         # 5: std_indeg
        float(outdeg.mean()),               # 6: mean_outdeg
        float(outdeg.std() + 1e-12),        # 7: std_outdeg
        float((off > 0.01).mean()),         # 8: density_threshold
        float(N),                           # 9: n_nodes
    ], dtype=np.float32)
    return feats


# ── Quasi-adjacency F1 (same as gat_quasi_adj.py) ─────────────────────────────

def build_input_matrix(rec: dict) -> np.ndarray:
    N  = rec["n_nodes"]
    ei = rec["edge_index"].numpy()
    A  = np.zeros((N, N), dtype=np.float64)
    A[ei[1], ei[0]] = 1.0
    return A


def per_sample_f1(A_agg: np.ndarray, A_input: np.ndarray) -> dict:
    N    = A_agg.shape[0]
    mask = ~np.eye(N, dtype=bool)
    flat_agg   = A_agg[mask]
    flat_input = A_input[mask].astype(bool)
    nz   = flat_agg[flat_agg > 0]
    if len(nz) == 0:
        return {"best_f1": 0.0, "best_tau": 0.0, "best_p": 0.0, "best_r": 0.0,
                "density_agg": 0.0, "density_input": float(flat_input.mean())}
    taus = np.unique(np.percentile(nz, SWEEP_PERCS))
    best = {"f1": 0.0, "tau": taus[0], "p": 0.0, "r": 0.0}
    for tau in taus:
        pred = flat_agg >= tau
        tp = (pred & flat_input).sum()
        fp = (pred & ~flat_input).sum()
        fn = (~pred & flat_input).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "tau": tau, "p": p, "r": r}
    # Random permutation baseline
    rng      = np.random.default_rng(42)
    perm_f1s = []
    for _ in range(50):
        perm = flat_agg.copy(); rng.shuffle(perm)
        perm_taus = np.unique(np.percentile(perm[perm > 0] if (perm > 0).any() else perm, SWEEP_PERCS))
        best_pf = 0.0
        for tau in perm_taus:
            pred = perm >= tau
            tp = (pred & flat_input).sum()
            fp = (pred & ~flat_input).sum()
            fn = (~pred & flat_input).sum()
            p_ = tp / (tp + fp + 1e-10)
            r_ = tp / (tp + fn + 1e-10)
            f1_ = 2 * p_ * r_ / (p_ + r_ + 1e-10)
            if f1_ > best_pf: best_pf = f1_
        perm_f1s.append(best_pf)
    return {
        "best_f1":       best["f1"],
        "best_tau":      float(best["tau"]),
        "best_p":        best["p"],
        "best_r":        best["r"],
        "density_agg":   float((flat_agg > 0).mean()),
        "density_input": float(flat_input.mean()),
        "random_f1":     float(np.mean(perm_f1s)),
        "delta_f1":      best["f1"] - float(np.mean(perm_f1s)),
    }


# ── H1: Gini direction split ──────────────────────────────────────────────────

def analyse_gini_split(records: list[dict], source_tag: str) -> dict:
    """
    Compute per-sample Gini of A_agg in-degree.
    Classify each sample as TTS / VC / bonafide by system_id prefix.
    Return mean Gini per class + direction test result.
    """
    by_class: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        if rec.get("is_degenerate", False):
            continue
        A_agg = build_agg_graph(rec)
        g     = gini_indegree(A_agg)
        sid   = rec["system_id"]

        if sid == "librispeech":
            cls = "bonafide"
        elif sid.startswith("VCC2020_"):
            cls = "VC"
        else:
            cls = "TTS"   # WaveFake systems: {vocoder}_{corpus}
        by_class[cls].append(g)

    stats = {}
    for cls, vals in sorted(by_class.items()):
        a = np.array(vals)
        stats[cls] = {
            "n":      len(a),
            "mean":   float(a.mean()),
            "std":    float(a.std()),
            "median": float(np.median(a)),
        }

    bona_mean = stats.get("bonafide", {}).get("mean", np.nan)
    tts_mean  = stats.get("TTS", {}).get("mean", np.nan)
    vc_mean   = stats.get("VC",  {}).get("mean", np.nan)

    h1_tts_direction = (tts_mean < bona_mean) if not np.isnan(tts_mean) else None
    h1_vc_direction  = (vc_mean  > bona_mean) if not np.isnan(vc_mean)  else None
    h1_pass = bool(h1_tts_direction and h1_vc_direction) if (
        h1_tts_direction is not None and h1_vc_direction is not None) else None

    print(f"\n[H1] Gini split ({source_tag}):")
    for cls, s in stats.items():
        print(f"  {cls:12s}: mean={s['mean']:.4f}±{s['std']:.4f}  n={s['n']}")
    print(f"  TTS < bonafide: {h1_tts_direction}  |  VC > bonafide: {h1_vc_direction}")
    print(f"  H1 {'PASS' if h1_pass else 'FAIL' if h1_pass is not None else 'INCONCLUSIVE'}")

    return {
        "class_stats": stats,
        "h1_tts_direction": h1_tts_direction,
        "h1_vc_direction":  h1_vc_direction,
        "h1_pass": h1_pass,
    }


# ── H2: TTS vs VC classifier ──────────────────────────────────────────────────

def analyse_tts_vc_classifier(records: list[dict], source_tag: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

    feats, labels = [], []
    for rec in records:
        if rec.get("is_degenerate", False):
            continue
        sid = rec["system_id"]
        if sid == "librispeech":
            continue  # exclude bonafide from TTS vs VC
        A_agg = build_agg_graph(rec)
        feats.append(compute_features(A_agg))
        labels.append(0 if sid.startswith("VCC2020_") else 1)  # 0=VC, 1=TTS

    X = np.array(feats)
    y = np.array(labels)

    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        print(f"\n[H2] Cannot run classifier ({source_tag}): insufficient data")
        return {"h2_balanced_acc": None, "h2_pass": None}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ba_scores, auc_scores = [], []

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xva = scaler.transform(X[val_idx])
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(Xtr, y[train_idx])
        preds = clf.predict(Xva)
        probs = clf.predict_proba(Xva)[:, 1]
        ba_scores.append(balanced_accuracy_score(y[val_idx], preds))
        if len(np.unique(y[val_idx])) > 1:
            auc_scores.append(roc_auc_score(y[val_idx], probs))

    mean_ba  = float(np.mean(ba_scores))
    std_ba   = float(np.std(ba_scores))
    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")
    h2_pass  = mean_ba >= 0.70

    # Feature importances from a full-data fit
    scaler = StandardScaler().fit(X)
    clf    = LogisticRegression(max_iter=1000, random_state=42, C=1.0).fit(
             scaler.transform(X), y)
    feat_names = ["gini_indeg", "offdiag_frob", "entropy_row", "max_offdiag",
                  "mean_indeg", "std_indeg", "mean_outdeg", "std_outdeg",
                  "density_001", "n_nodes"]
    coefs = list(zip(feat_names, np.abs(clf.coef_[0]).tolist()))
    coefs.sort(key=lambda x: -x[1])
    top_features = [name for name, _ in coefs[:3]]

    print(f"\n[H2] TTS vs VC classifier ({source_tag}):")
    print(f"  5-fold balanced accuracy: {mean_ba:.3f} ± {std_ba:.3f}")
    print(f"  AUC: {mean_auc:.3f}")
    print(f"  H2 {'PASS' if h2_pass else 'FAIL'}  (threshold ≥ 0.70)")
    print(f"  Top features: {top_features}")

    return {
        "n_samples": X.shape[0],
        "n_tts":     int((y == 1).sum()),
        "n_vc":      int((y == 0).sum()),
        "cv_balanced_acc_mean": mean_ba,
        "cv_balanced_acc_std":  std_ba,
        "cv_auc_mean":          mean_auc,
        "top_features":         top_features,
        "h2_pass":              h2_pass,
    }


# ── H3: Quasi-adjacency F1 ────────────────────────────────────────────────────

def analyse_quasi_adj(records: list[dict], source_tag: str) -> dict:
    by_sys: dict[str, list[float]] = defaultdict(list)
    random_by_sys: dict[str, list[float]] = defaultdict(list)
    delta_by_sys: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        if rec.get("is_degenerate", False):
            continue
        A_agg   = build_agg_graph(rec)
        A_input = build_input_matrix(rec)
        r       = per_sample_f1(A_agg, A_input)
        sid     = rec["system_id"]

        by_sys[sid].append(r["best_f1"])
        random_by_sys[sid].append(r["random_f1"])
        delta_by_sys[sid].append(r["delta_f1"])

    # Evaluate H3 at the CLASS level (mean delta per class), not per-sample
    class_mean_deltas = [float(np.mean(vals)) for vals in delta_by_sys.values() if vals]
    all_deltas = [d for vals in delta_by_sys.values() for d in vals]  # kept for other stats
    h3_pass = max(class_mean_deltas) < 0.10 if class_mean_deltas else None
    above_random = [sid for sid, deltas in delta_by_sys.items() if np.mean(deltas) > 0]

    print(f"\n[H3] Quasi-adjacency F1 ({source_tag}):")
    for sid in sorted(by_sys.keys()):
        f1s  = np.array(by_sys[sid])
        rnds = np.array(random_by_sys[sid])
        dl   = np.array(delta_by_sys[sid])
        print(f"  {sid:35s}: F1={f1s.mean():.3f}±{f1s.std():.3f}  "
              f"rand={rnds.mean():.3f}  Δ={dl.mean():+.3f}")
    print(f"  Classes above random floor: {above_random}")
    max_cls = max(class_mean_deltas) if class_mean_deltas else float("nan")
    print(f"  H3 {'PASS' if h3_pass else 'FAIL' if h3_pass is not None else 'INCONCLUSIVE'}"
          f"  (max class-mean Δ={max_cls:.3f} {'< 0.10' if h3_pass else '≥ 0.10'})")

    sys_stats = {}
    for sid in sorted(by_sys.keys()):
        sys_stats[sid] = {
            "f1_mean":  float(np.mean(by_sys[sid])),
            "f1_std":   float(np.std(by_sys[sid])),
            "rand_mean":float(np.mean(random_by_sys[sid])),
            "delta_mean":float(np.mean(delta_by_sys[sid])),
        }
    return {
        "per_system": sys_stats,
        "mean_class_delta": float(np.mean(class_mean_deltas)) if class_mean_deltas else None,
        "max_class_delta":  float(max(class_mean_deltas))      if class_mean_deltas else None,
        "above_random":     above_random,
        "h3_pass":          h3_pass,
    }


# ── Per-seed analysis ─────────────────────────────────────────────────────────

def analyse_seed(seed: int) -> dict | None:
    arts_path = OUT_DIR / f"seed{seed}" / "all_layers_artifacts.pt"
    eer_path  = OUT_DIR / f"seed{seed}" / "eer_eval.json"

    if not arts_path.exists():
        print(f"[SKIP] seed={seed}: artifacts not found at {arts_path}")
        return None

    print(f"\n{'='*70}")
    print(f"Seed {seed} analysis")
    print(f"{'='*70}")

    artifact = torch.load(str(arts_path), weights_only=False)
    n = len(artifact["sample_ids"])
    print(f"Loaded {n} samples")

    # Unpack into list-of-dicts (same structure as gat_attn_graphs records)
    records = []
    for i in range(n):
        rec = {
            "sample_id":        artifact["sample_ids"][i],
            "label":            artifact["labels"][i],
            "system_id":        artifact["system_ids"][i],
            "n_nodes":          artifact["n_nodes"][i],
            "n_edges":          artifact["n_edges"][i],
            "is_degenerate":    artifact["is_degenerate"][i],
            "node_phoneme_ids": artifact["node_phoneme_ids"][i],
            "edge_index":       artifact["edge_index"][i],
            "edge_type_adj":    artifact["edge_type_adj"][i],
            "attn_l0":          artifact["attn_l0"][i],
            "attn_l1":          artifact["attn_l1"][i],
            "attn_l2":          artifact["attn_l2"][i],
        }
        records.append(rec)

    eer = None
    if eer_path.exists():
        eer = json.load(open(eer_path))["eer"]
        print(f"EER (new domain): {eer*100:.2f}%")

    tag = f"E9 seed{seed}"
    h1  = analyse_gini_split(records, tag)
    h2  = analyse_tts_vc_classifier(records, tag)
    h3  = analyse_quasi_adj(records, tag)

    result = {
        "seed":  seed,
        "n":     n,
        "eer":   eer,
        "h1":    h1,
        "h2":    h2,
        "h3":    h3,
    }

    out = OUT_DIR / f"analysis_seed{seed}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    return result


# ── Cross-seed summary ────────────────────────────────────────────────────────

def cross_seed_summary(results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("CROSS-SEED SUMMARY")
    print(f"{'='*70}")

    for tag, passed_fn in [
        ("H1 (Gini TTS<bona AND VC>bona)",  lambda r: r["h1"]["h1_pass"]),
        ("H2 (bal-acc ≥ 0.70)",             lambda r: r["h2"]["h2_pass"]),
        ("H3 (max Δ < 0.10 vs random)",     lambda r: r["h3"]["h3_pass"]),
    ]:
        outcomes = [passed_fn(r) for r in results if r is not None]
        all_pass  = all(outcomes) if outcomes else None
        any_fail  = any(o is False for o in outcomes)
        status    = "ALL PASS" if all_pass else ("PARTIAL FAIL" if any_fail else "?")
        seeds     = [str(r["seed"]) for r in results if r is not None]
        values    = [f"seed{r['seed']}={'PASS' if passed_fn(r) else 'FAIL'}"
                     for r in results if r is not None]
        print(f"  {tag}: {status}  ({', '.join(values)})")

    # H4: feature overlap check
    all_tops = [set(r["h2"]["top_features"] or [])
                for r in results if r is not None and r["h2"].get("top_features")]
    canonical = {"gini_indeg", "offdiag_frob"}
    for i, r in enumerate(results):
        if r is None: continue
        tops = set(r["h2"].get("top_features", []))
        overlap = tops & canonical
        print(f"  H4 seed{r['seed']}: top features={list(tops)}  "
              f"overlap with canonical={list(overlap)}")

    # Gini values across seeds
    print("\n  Gini means per class:")
    for r in results:
        if r is None: continue
        for cls, s in r["h1"].get("class_stats", {}).items():
            print(f"    seed{r['seed']} {cls:12s}: {s['mean']:.4f}±{s['std']:.4f}")


def make_gini_comparison_plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load original seed-1 if available
    orig_gini: dict[str, float] = {}
    if args.compare_original and ORIG_ARTS.exists():
        print("\nLoading original ASVspoof seed-1 artifacts for Gini comparison...")
        art = torch.load(str(ORIG_ARTS), weights_only=False)
        n = len(art["sample_ids"])
        orig_recs = [{"n_nodes": art["n_nodes"][i], "n_edges": art["n_edges"][i],
                      "is_degenerate": art["is_degenerate"][i],
                      "system_id": art["system_ids"][i],
                      "edge_index": art["edge_index"][i],
                      "attn_l0": art["attn_l0"][i],
                      "attn_l1": art["attn_l1"][i],
                      "attn_l2": art["attn_l2"][i]}
                     for i in range(n)]
        for rec in orig_recs:
            if rec.get("is_degenerate"): continue
            try:
                A = build_agg_graph(rec)
                g = gini_indegree(A)
                sid = rec["system_id"]
                orig_gini.setdefault(sid, []).append(g)
            except Exception:
                pass
        orig_gini = {k: np.mean(v) for k, v in orig_gini.items()}

    fig, axes = plt.subplots(1, len(results) + (1 if orig_gini else 0),
                              figsize=(5 * (len(results) + 1), 4),
                              sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    palette = {"TTS": "#2ca02c", "VC": "#ff7f0e", "bonafide": "#1f77b4"}
    ax_idx = 0

    # Original reference
    if orig_gini:
        ax = axes[ax_idx]; ax_idx += 1
        for sid, g in sorted(orig_gini.items()):
            cls = ("bonafide" if sid == "-" else "TTS" if sid.startswith("A0") else "VC")
            color = palette.get(cls, "gray")
            ax.barh(sid, g, color=color, alpha=0.8)
        ax.set_title("Original ASVspoof\nseed-1 (ref)", fontsize=9)
        ax.set_xlabel("Mean Gini in-degree")

    for r in results:
        if r is None: continue
        ax = axes[ax_idx]; ax_idx += 1
        for cls, s in sorted(r["h1"]["class_stats"].items()):
            color = palette.get(cls, "gray")
            ax.barh(cls, s["mean"], xerr=s["std"], color=color,
                    alpha=0.8, capsize=3)
        ax.set_title(f"E9 seed{r['seed']}\nEER={r['eer']*100:.1f}%" if r.get("eer") else
                     f"E9 seed{r['seed']}", fontsize=9)
        ax.set_xlabel("Mean Gini in-degree")

    fig.suptitle("Gini in-degree: TTS vs VC vs bonafide\n"
                 "(green=TTS, orange=VC, blue=bonafide)", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "gini_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nGini plot saved: {out}")


def write_report(results: list[dict]) -> None:
    lines = [
        "# E9 Mechanism Analysis Report",
        "",
        "**Pre-registered hypotheses** (see preregistration.md)",
        "",
        "## Results",
        "",
        "| Seed | EER (new-domain) | H1 (Gini split) | H2 (bal-acc) | H3 (max Δ quasi-adj) |",
        "|------|-----------------|-----------------|-------------|---------------------|",
    ]
    for r in results:
        if r is None: continue
        eer_str = f"{r['eer']*100:.2f}%" if r.get("eer") is not None else "?"
        h1_str  = "PASS" if r["h1"].get("h1_pass") else "FAIL"
        h2_str  = f"{r['h2'].get('cv_balanced_acc_mean', float('nan')):.3f} " \
                  f"({'PASS' if r['h2'].get('h2_pass') else 'FAIL'})"
        h3_str  = f"Δ={r['h3'].get('max_class_delta', float('nan')):.3f} " \
                  f"({'PASS' if r['h3'].get('h3_pass') else 'FAIL'})"
        lines.append(f"| {r['seed']} | {eer_str} | {h1_str} | {h2_str} | {h3_str} |")

    lines += ["", "## Interpretation", "",
              "*(fill in after reviewing results against decision logic in preregistration.md)*",
              ""]

    out = OUT_DIR / "analysis_report.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {out}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    results = []
    for seed in args.seeds:
        r = analyse_seed(seed)
        results.append(r)

    valid = [r for r in results if r is not None]
    if len(valid) > 0:
        cross_seed_summary(valid)
        make_gini_comparison_plot(valid)
        write_report(valid)
    else:
        print("\nNo valid results to summarize. Run e9_attn_cache.py first.")


if __name__ == "__main__":
    main()
