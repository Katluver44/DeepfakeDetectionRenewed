#!/usr/bin/env python3
"""
mlaad_higher_order_mechanism_analysis.py
=========================================
Tests whether higher-order temporal and graph-level signals predict
per-system EER difficulty in MLAAD.

Motivated by the null result of WavLM distance and phoneme KL divergence
in the extreme-system analysis (ρ < -0.22, p > 0.08; group perm-p ≥ 0.147).

Metrics (all computed per utterance, aggregated per system):
  1. Phoneme segment embedding variance  — variance across phoneme node features
  2. Frame-to-frame variance             — adjacent-frame L2 distances in hidden_states
  3. Attention entropy                   — per-node, per-head entropy at GAT layer 0
  4. Graph sparsity                      — n_nodes, edge density, avg degree
  5. Graph concentration                 — Gini of attn weights, top-k mass fraction

Analysis pipeline:
  6. Spearman + Pearson correlation with mean EER (all 63 systems)
  7. Hard vs easy comparison: Cohen's d + permutation test
  8. Outlier detection (griffin_lim, OuteTTS highlighted)
  9. Multivariate: LinearRegression, Lasso, RandomForest with LOO-CV

Outputs → experiments/results/mlaad/higher_order_mechanism_analysis/
  phoneme_variance.csv
  frame_variance.csv
  attention_entropy.csv
  graph_sparsity.csv
  graph_concentration.csv
  best_predictors_of_difficulty.csv
  hard_vs_easy_comparison.csv
  outlier_report.md
  multivariate_explanation.json
  analysis_summary.md
"""
from __future__ import annotations

import csv
import json
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
E6_CSV        = EXP_DIR / "results" / "mlaad" / "e6_ranking_lock" / "per_seed_per_attack_eer.csv"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "higher_order_mechanism_analysis"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_PER_SYSTEM = 50
BATCH_SIZE     = 16
HARD_THRESHOLD = 7
EASY_THRESHOLD = 7
PERM_N         = 5000
RNG_SEED       = 42

SPOTLIGHT = {"griffin_lim", "OuteTTS", "orpheus-tts-0.1-finetune"}  # outlier candidates


def _best_ckpt(stem: str) -> Path:
    cands = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return cands[0] if cands else CKPT_DIR / f"{stem}.ckpt"


CKPT_PATH = _best_ckpt("mlaad_robust_goat")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def gini_coef(a: np.ndarray) -> float:
    """Gini coefficient: 0=uniform, 1=fully concentrated."""
    a = np.abs(a)
    if a.sum() < 1e-12 or len(a) < 2:
        return 0.0
    a = np.sort(a)
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = PERM_N,
                     rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    obs = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        count += abs(combined[:n_a].mean() - combined[n_a:].mean()) >= obs
    return float(count / n_perm)


def phoneme_segment_embeddings(hs: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Group consecutive frames with same phoneme ID and mean-pool to node embeddings.
    hs: (T, 768), pids: (T,) → (n_segments, 768)
    """
    T = hs.shape[0]
    if T == 0:
        return hs[:0]
    segs, start = [], 0
    for t in range(1, T):
        if pids[t] != pids[t - 1]:
            segs.append(hs[start:t].mean(axis=0))
            start = t
    segs.append(hs[start:].mean(axis=0))
    return np.stack(segs)


def compute_attn_stats(attn_e_nh: np.ndarray, tgt_nodes: np.ndarray,
                       n_nodes: int) -> dict:
    """
    attn_e_nh : (E, NH)  — attention weights (sum to 1 per target node per head)
    tgt_nodes : (E,)     — target node index for each edge
    n_nodes   : int

    Returns per-utterance aggregated stats across all valid (degree≥2) nodes.
    """
    NH = attn_e_nh.shape[1]
    entropies, ginis, top1_masses, top3_masses = [], [], [], []

    for node_i in range(n_nodes):
        mask = (tgt_nodes == node_i)
        a = attn_e_nh[mask]          # (deg_i, NH)
        if a.shape[0] < 2:
            continue
        deg = a.shape[0]

        # Entropy per head (already normalized, sum to 1 per head)
        H = -np.sum(a * np.log(a + 1e-12), axis=0)   # (NH,)
        entropies.extend(H.tolist())

        # Gini per head
        for h in range(NH):
            ginis.append(gini_coef(a[:, h]))

        # Top-1 and top-3 mass per head
        sorted_a = np.sort(a, axis=0)[::-1]           # (deg, NH) descending
        top1_masses.extend(sorted_a[0].tolist())
        k3 = min(3, deg)
        top3_masses.extend(sorted_a[:k3].sum(axis=0).tolist())

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    def _safe_median(lst):
        return float(np.median(lst)) if lst else float("nan")

    return {
        "mean_entropy":    _safe_mean(entropies),
        "median_entropy":  _safe_median(entropies),
        "max_entropy":     float(np.max(entropies)) if entropies else float("nan"),
        "var_entropy":     float(np.var(entropies)) if entropies else float("nan"),
        "mean_gini":       _safe_mean(ginis),
        "mean_top1_mass":  _safe_mean(top1_masses),
        "mean_top3_mass":  _safe_mean(top3_masses),
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_extreme_groups() -> tuple:
    rows = list(csv.DictReader(E6_CSV.open()))
    cond_seed_maps: dict = {}
    for r in rows:
        k = (r["condition"], r["seed"])
        cond_seed_maps.setdefault(k, {})[r["attack_system"]] = float(r["eer"])

    sys_eer_accum: dict = defaultdict(list)
    for r in rows:
        sys_eer_accum[r["attack_system"]].append(float(r["eer"]))
    mean_eer = {s: float(np.mean(v)) for s, v in sys_eer_accum.items()}

    hard_ctr: dict = defaultdict(int)
    easy_ctr: dict = defaultdict(int)
    for eer_map in cond_seed_maps.values():
        ranked = sorted(eer_map.items(), key=lambda x: -x[1])
        n_q = max(1, int(len(ranked) * 0.25))
        for s, _ in ranked[:n_q]:
            hard_ctr[s] += 1
        for s, _ in ranked[-n_q:]:
            easy_ctr[s] += 1

    hard = sorted([s for s, c in hard_ctr.items() if c >= HARD_THRESHOLD],
                  key=lambda s: -mean_eer[s])
    easy = sorted([s for s, c in easy_ctr.items() if c >= EASY_THRESHOLD],
                  key=lambda s: mean_eer[s])
    return hard, easy, mean_eer


class EvalDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(PROCESSED_DIR / rec["audio_path"]).unsqueeze(0)
        return {
            "audio":     wav,
            "label":     0 if rec["label"] == "bonafide" else 1,
            "system_id": rec.get("attack_system", "bonafide"),
        }


def _collate(batch):
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def sample_records(rng: np.random.Generator) -> list[dict]:
    all_recs = json.loads(INDIST_JSON.read_text())
    by_sys: dict = defaultdict(list)
    bona = []
    for r in all_recs:
        if r["label"] == "bonafide":
            bona.append(r)
        else:
            by_sys[r["attack_system"]].append(r)

    sampled = []
    for recs in by_sys.values():
        idxs = rng.choice(len(recs), size=min(MAX_PER_SYSTEM, len(recs)), replace=False)
        sampled.extend([recs[i] for i in idxs])

    bona_idxs = rng.choice(len(bona), size=min(MAX_PER_SYSTEM, len(bona)), replace=False)
    sampled.extend([bona[i] for i in bona_idxs])
    return sampled


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    import gat_l0_attention as gla
    gla.patch_phoneme_loader()
    lit = gla.load_model(CKPT_PATH, device)
    lit.eval()
    return lit


# ─── Pass 1: phoneme segment + frame variance ─────────────────────────────────

def run_pass1(lit, records: list[dict], device: torch.device) -> dict[str, dict]:
    """
    Returns {system_id: {'phoneme_var': [...], 'frame_mean_dist': [...], 'frame_var_dist': [...]}}
    """
    from gat_l0_attention import run_frozen_frontend

    gat_model = lit.model
    ds = EvalDataset(records)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, collate_fn=_collate, pin_memory=True)

    per_sys: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    total = len(records)

    print(f"  Pass 1 (phoneme/frame variance): {total} samples")
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            audio  = batch["audio"].to(device)
            labels = batch["label"]
            sys_ids = batch["system_id"]
            B = audio.shape[0]

            hs, pids = run_frozen_frontend(audio, gat_model, device)
            # hs: (B, T, 768), pids: (B, T)
            hs_np   = hs.cpu().numpy()
            pids_np = pids.cpu().numpy()

            for i in range(B):
                tag = sys_ids[i] if labels[i] == 1 else "bonafide"
                h = hs_np[i]      # (T, 768)
                p = pids_np[i]    # (T,)

                # Phoneme segment embeddings
                segs = phoneme_segment_embeddings(h, p)   # (n_segs, 768)
                if len(segs) > 1:
                    seg_var = float(segs.var(axis=0).mean())   # mean variance across dims
                else:
                    seg_var = 0.0

                # Frame-to-frame distances
                diffs = np.linalg.norm(h[1:] - h[:-1], axis=-1)  # (T-1,)
                mean_dist = float(diffs.mean())
                var_dist  = float(diffs.var())

                per_sys[tag]["phoneme_var"].append(seg_var)
                per_sys[tag]["frame_mean_dist"].append(mean_dist)
                per_sys[tag]["frame_var_dist"].append(var_dist)
                per_sys[tag]["n_segs"].append(float(len(segs)))

            if (bi + 1) % 10 == 0 or (bi + 1) == len(dl):
                print(f"    {min((bi + 1) * BATCH_SIZE, total)}/{total}")

    return per_sys


# ─── Pass 2: attention records via gla.extract_attention ──────────────────────

def run_pass2(lit, records: list[dict], device: torch.device) -> dict[str, dict]:
    """
    Returns {system_id: {entropy: [...], n_nodes: [...], edge_density: [...], gini: [...], ...}}
    """
    import gat_l0_attention as gla

    ds = EvalDataset(records)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, collate_fn=_collate, pin_memory=True)

    print(f"\n  Pass 2 (attention/graph metrics): {len(records)} samples")
    attn_records, n_deg = gla.extract_attention(lit, dl, device)
    valid = [r for r in attn_records if not r["is_degenerate"]]
    print(f"  {len(valid)} valid records (dropped {n_deg} degenerate)")

    per_sys: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for rec in valid:
        tag = rec["system_id"] if rec["label"] == 1 else "bonafide"
        attn   = rec["attn_l0"].numpy()          # (E, NH)
        ei     = rec["edge_index"].numpy()        # (2, E)
        n_nodes = rec["n_nodes"]
        n_edges = rec["n_edges"]

        tgt_nodes = ei[1]  # attention sums to 1 per target node

        # ── Attention entropy + concentration ─────────────────────────────────
        stats = compute_attn_stats(attn, tgt_nodes, n_nodes)
        for k, v in stats.items():
            per_sys[tag][k].append(v)

        # ── Graph sparsity ────────────────────────────────────────────────────
        per_sys[tag]["n_nodes"].append(float(n_nodes))
        per_sys[tag]["n_edges"].append(float(n_edges))
        if n_nodes > 1:
            # Max possible edges for directed graph excluding self-loops
            max_e = float(n_nodes * (n_nodes - 1))
            edge_density = n_edges / max_e if max_e > 0 else 0.0
            avg_degree   = n_edges / n_nodes
        else:
            edge_density = 0.0
            avg_degree   = 0.0
        per_sys[tag]["edge_density"].append(edge_density)
        per_sys[tag]["avg_degree"].append(avg_degree)

    return per_sys


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_sys: dict[str, dict], metrics: list[str]) -> dict[str, dict]:
    """Aggregate lists → mean per system per metric."""
    out = {}
    for sys_id, metric_lists in per_sys.items():
        row = {"system_id": sys_id}
        for m in metrics:
            vals = [v for v in metric_lists.get(m, []) if not (isinstance(v, float) and np.isnan(v))]
            row[f"{m}_mean"]   = float(np.mean(vals)) if vals else float("nan")
            row[f"{m}_median"] = float(np.median(vals)) if vals else float("nan")
            row["n_utts"]      = len(metric_lists.get(m, []))
        out[sys_id] = row
    return out


# ─── Correlation analysis ─────────────────────────────────────────────────────

def correlate_with_eer(feat_table: dict[str, dict], mean_eer: dict[str, float],
                       feature_cols: list[str]) -> list[dict]:
    systems = [s for s in feat_table if s != "bonafide" and s in mean_eer]
    eer_arr = np.array([mean_eer[s] for s in systems])

    results = []
    for col in feature_cols:
        vals = np.array([feat_table[s].get(col, float("nan")) for s in systems])
        valid = ~np.isnan(vals)
        if valid.sum() < 5:
            continue
        rho, p_s = spearmanr(vals[valid], eer_arr[valid])
        r,   p_p = pearsonr(vals[valid], eer_arr[valid])
        results.append({
            "feature":      col,
            "spearman_rho": float(rho),
            "spearman_p":   float(p_s),
            "pearson_r":    float(r),
            "pearson_p":    float(p_p),
            "n":            int(valid.sum()),
        })

    results.sort(key=lambda x: abs(x["spearman_rho"]), reverse=True)
    return results


# ─── Hard vs easy comparison ──────────────────────────────────────────────────

def hard_vs_easy(feat_table: dict[str, dict], hard: list[str], easy: list[str],
                 feature_cols: list[str]) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for col in feature_cols:
        h = np.array([feat_table[s][col] for s in hard
                      if s in feat_table and not np.isnan(feat_table[s].get(col, float("nan")))])
        e = np.array([feat_table[s][col] for s in easy
                      if s in feat_table and not np.isnan(feat_table[s].get(col, float("nan")))])
        if len(h) < 2 or len(e) < 2:
            continue
        d     = cohen_d(h, e)
        perm_p = permutation_test(h, e, rng=rng)
        rows.append({
            "feature":    col,
            "hard_mean":  float(h.mean()),
            "easy_mean":  float(e.mean()),
            "delta":      float(h.mean() - e.mean()),
            "cohen_d":    d,
            "perm_p":     perm_p,
        })
    rows.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)
    return rows


# ─── Multivariate ─────────────────────────────────────────────────────────────

def _loo_r2(model_cls, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """Manual LOO-CV R² — avoids sklearn's undefined-single-sample R² issue."""
    from sklearn.model_selection import LeaveOneOut
    n = len(y)
    y_pred = np.empty(n)
    for train_idx, test_idx in LeaveOneOut().split(X):
        m = model_cls(**kwargs)
        m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def multivariate_analysis(feat_table: dict[str, dict], mean_eer: dict[str, float],
                           feature_cols: list[str]) -> dict:
    from sklearn.linear_model import LinearRegression, LassoCV, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    systems = [s for s in feat_table if s != "bonafide" and s in mean_eer]
    y = np.array([mean_eer[s] for s in systems])

    # Build feature matrix, dropping NaN cols
    feat_arrays = {}
    for col in feature_cols:
        vals = np.array([feat_table[s].get(col, float("nan")) for s in systems])
        if not np.any(np.isnan(vals)):
            feat_arrays[col] = vals

    if len(feat_arrays) < 2:
        return {"error": "too few features without NaN"}

    col_names = list(feat_arrays.keys())
    X_raw = np.column_stack([feat_arrays[c] for c in col_names])  # (n, p)
    n, p  = X_raw.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Linear regression (manual LOO)
    lr_r2 = _loo_r2(LinearRegression, X, y)

    # Lasso: fit with CV alpha, then manual LOO
    lasso_cv = LassoCV(cv=min(5, n), max_iter=2000, random_state=RNG_SEED)
    lasso_cv.fit(X, y)
    best_alpha = float(lasso_cv.alpha_)
    lasso_r2  = _loo_r2(Lasso, X, y, alpha=best_alpha, max_iter=2000)
    lasso_coef = {col_names[i]: float(lasso_cv.coef_[i]) for i in range(p)}
    selected   = [c for c, v in lasso_coef.items() if abs(v) > 1e-6]

    # Random Forest (manual LOO)
    rf_r2 = _loo_r2(RandomForestRegressor, X, y,
                    n_estimators=200, random_state=RNG_SEED)
    rf = RandomForestRegressor(n_estimators=200, random_state=RNG_SEED)
    rf.fit(X, y)
    rf_importances = {col_names[i]: float(rf.feature_importances_[i]) for i in range(p)}
    top_rf = sorted(rf_importances.items(), key=lambda x: -x[1])[:5]

    return {
        "n_systems":   n,
        "n_features":  p,
        "features":    col_names,
        "linear_regression": {"loo_r2": lr_r2},
        "lasso": {
            "loo_r2":            lasso_r2,
            "best_alpha":        best_alpha,
            "coef":              lasso_coef,
            "selected_features": selected,
        },
        "random_forest": {
            "loo_r2":             rf_r2,
            "feature_importance": rf_importances,
            "top5":               [{"feature": f, "importance": v} for f, v in top_rf],
        },
    }


# ─── Output writers ───────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_outlier_report(path: Path,
                         feat_table: dict[str, dict],
                         corr_rows: list[dict],
                         hard: list[str],
                         easy: list[str],
                         mean_eer: dict[str, float]):
    top_feats = [r["feature"] for r in corr_rows[:5]]
    lines = [
        "# Outlier Report — Higher-Order Mechanism Analysis",
        "",
        "## Spotlight systems",
        "",
    ]
    for sys_id in SPOTLIGHT:
        if sys_id not in feat_table:
            continue
        group = "hard" if sys_id in hard else ("easy" if sys_id in easy else "mid")
        eer   = mean_eer.get(sys_id, float("nan"))
        lines.append(f"### {sys_id}  (group={group}, mean_EER={eer:.3f})")
        for feat in top_feats:
            val = feat_table[sys_id].get(feat, float("nan"))
            lines.append(f"  {feat}: {val:.4f}" if not np.isnan(val) else f"  {feat}: N/A")
        lines.append("")

    lines += [
        "## Correlation context",
        "",
        "Top 5 features by |ρ| with EER:",
        "",
    ]
    for r in corr_rows[:5]:
        lines.append(
            f"  {r['feature']}: ρ={r['spearman_rho']:+.3f} p={r['spearman_p']:.3f} "
            f"| r={r['pearson_r']:+.3f} p={r['pearson_p']:.3f}"
        )

    path.write_text("\n".join(lines))


def write_summary(path: Path,
                  corr_rows: list[dict],
                  hve_rows: list[dict],
                  mv: dict,
                  hard: list[str],
                  easy: list[str]):
    sig_corr = [r for r in corr_rows if r["spearman_p"] < 0.05]
    sig_hve  = [r for r in hve_rows  if r["perm_p"] < 0.05]

    lines = [
        "# Higher-Order Mechanism Analysis — Summary",
        "",
        f"Hard systems ({len(hard)}): {', '.join(hard)}",
        f"Easy systems ({len(easy)}): {', '.join(easy)}",
        "",
        "## Correlation with EER (n=63)",
        "",
    ]
    for r in corr_rows[:10]:
        sig = "*" if r["spearman_p"] < 0.05 else ""
        lines.append(
            f"  {r['feature']}: ρ={r['spearman_rho']:+.3f} p={r['spearman_p']:.3f} "
            f"| r={r['pearson_r']:+.3f} p={r['pearson_p']:.3f} {sig}"
        )

    lines += ["", f"Significant predictors (p<0.05): {len(sig_corr)}"]
    if sig_corr:
        for r in sig_corr:
            lines.append(f"  {r['feature']}: ρ={r['spearman_rho']:+.3f}")

    lines += ["", "## Hard vs Easy comparison", ""]
    for r in hve_rows[:10]:
        sig = "**" if r["perm_p"] < 0.01 else ("*" if r["perm_p"] < 0.05 else "")
        lines.append(
            f"  {r['feature']}: Δ={r['delta']:+.4f}  d={r['cohen_d']:+.3f}  "
            f"perm-p={r['perm_p']:.3f} {sig}"
        )

    lines += [
        "",
        "## Multivariate (LOO-CV R²)",
        "",
        f"  Linear regression: {mv.get('linear_regression', {}).get('loo_r2', 'N/A'):.3f}",
        f"  Lasso:             {mv.get('lasso', {}).get('loo_r2', 'N/A'):.3f}  "
        f"(α={mv.get('lasso', {}).get('best_alpha', 'N/A'):.4f})",
        f"  Random Forest:     {mv.get('random_forest', {}).get('loo_r2', 'N/A'):.3f}",
        "",
        f"Lasso selected features: {mv.get('lasso', {}).get('selected_features', [])}",
        "",
        "Top-5 RF importances:",
    ]
    for item in mv.get("random_forest", {}).get("top5", []):
        lines.append(f"  {item['feature']}: {item['importance']:.4f}")

    path.write_text("\n".join(lines))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(RNG_SEED)

    # ── Extreme groups ────────────────────────────────────────────────────────
    print("\nDefining extreme groups ...")
    hard, easy, mean_eer = load_extreme_groups()
    print(f"  Hard ({len(hard)}): {hard}")
    print(f"  Easy ({len(easy)}): {easy}")

    # ── Sample records ────────────────────────────────────────────────────────
    records = sample_records(rng)
    spoof_recs = [r for r in records if r["label"] != "bonafide"]
    print(f"\nSampled {len(records)} records ({len(records)-len(spoof_recs)} bonafide, "
          f"{len(spoof_recs)} spoof)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {CKPT_PATH.name} ...")
    lit = load_model(device)

    # ── Pass 1: phoneme segment + frame variance ──────────────────────────────
    print("\n--- Pass 1: phoneme & frame metrics ---")
    pass1_raw = run_pass1(lit, records, device)

    p1_metrics = ["phoneme_var", "frame_mean_dist", "frame_var_dist", "n_segs"]
    p1_agg = aggregate(pass1_raw, p1_metrics)

    # ── Pass 2: attention + graph metrics ─────────────────────────────────────
    print("\n--- Pass 2: attention & graph metrics ---")
    pass2_raw = run_pass2(lit, records, device)

    p2_metrics = [
        "mean_entropy", "median_entropy", "max_entropy", "var_entropy",
        "mean_gini", "mean_top1_mass", "mean_top3_mass",
        "n_nodes", "n_edges", "edge_density", "avg_degree",
    ]
    p2_agg = aggregate(pass2_raw, p2_metrics)

    # ── Merge into unified feature table ─────────────────────────────────────
    all_systems = set(p1_agg) | set(p2_agg)
    feat_table: dict[str, dict] = {}
    for s in all_systems:
        row = {"system_id": s}
        if s in p1_agg:
            for m in p1_metrics:
                row[f"{m}_mean"]   = p1_agg[s].get(f"{m}_mean",   float("nan"))
                row[f"{m}_median"] = p1_agg[s].get(f"{m}_median", float("nan"))
        if s in p2_agg:
            for m in p2_metrics:
                row[f"{m}_mean"]   = p2_agg[s].get(f"{m}_mean",   float("nan"))
                row[f"{m}_median"] = p2_agg[s].get(f"{m}_median", float("nan"))
        feat_table[s] = row

    # ── Feature columns for analysis ─────────────────────────────────────────
    feature_cols = [
        "phoneme_var_mean", "phoneme_var_median",
        "frame_mean_dist_mean", "frame_var_dist_mean",
        "n_segs_mean",
        "mean_entropy_mean", "median_entropy_mean", "max_entropy_mean", "var_entropy_mean",
        "mean_gini_mean", "mean_top1_mass_mean", "mean_top3_mass_mean",
        "n_nodes_mean", "n_edges_mean", "edge_density_mean", "avg_degree_mean",
    ]

    # ── Write per-metric CSVs ─────────────────────────────────────────────────
    print("\nWriting CSVs ...")

    # phoneme_variance.csv
    write_csv(OUT_DIR / "phoneme_variance.csv", [
        {"system_id": s, **{k: feat_table[s].get(k, float("nan"))
                            for k in ["phoneme_var_mean", "phoneme_var_median", "n_segs_mean"]}}
        for s in sorted(feat_table) if s != "bonafide"
    ])

    # frame_variance.csv
    write_csv(OUT_DIR / "frame_variance.csv", [
        {"system_id": s, **{k: feat_table[s].get(k, float("nan"))
                            for k in ["frame_mean_dist_mean", "frame_mean_dist_median",
                                      "frame_var_dist_mean"]}}
        for s in sorted(feat_table) if s != "bonafide"
    ])

    # attention_entropy.csv
    write_csv(OUT_DIR / "attention_entropy.csv", [
        {"system_id": s, **{k: feat_table[s].get(k, float("nan"))
                            for k in ["mean_entropy_mean", "median_entropy_mean",
                                      "max_entropy_mean", "var_entropy_mean"]}}
        for s in sorted(feat_table) if s != "bonafide"
    ])

    # graph_sparsity.csv
    write_csv(OUT_DIR / "graph_sparsity.csv", [
        {"system_id": s, **{k: feat_table[s].get(k, float("nan"))
                            for k in ["n_nodes_mean", "n_edges_mean",
                                      "edge_density_mean", "avg_degree_mean"]}}
        for s in sorted(feat_table) if s != "bonafide"
    ])

    # graph_concentration.csv
    write_csv(OUT_DIR / "graph_concentration.csv", [
        {"system_id": s, **{k: feat_table[s].get(k, float("nan"))
                            for k in ["mean_gini_mean", "mean_top1_mass_mean", "mean_top3_mass_mean"]}}
        for s in sorted(feat_table) if s != "bonafide"
    ])

    # ── Correlation analysis ──────────────────────────────────────────────────
    print("\nCorrelation with EER ...")
    corr_rows = correlate_with_eer(feat_table, mean_eer, feature_cols)
    write_csv(OUT_DIR / "best_predictors_of_difficulty.csv", corr_rows)

    print("\n  Top correlates:")
    for r in corr_rows[:8]:
        sig = "*" if r["spearman_p"] < 0.05 else ""
        print(f"    {r['feature']:35s}  ρ={r['spearman_rho']:+.3f}  p={r['spearman_p']:.3f}  {sig}")

    # ── Hard vs easy ──────────────────────────────────────────────────────────
    print("\nHard vs easy comparison ...")
    hve_rows = hard_vs_easy(feat_table, hard, easy, feature_cols)
    write_csv(OUT_DIR / "hard_vs_easy_comparison.csv", hve_rows)

    print("\n  Top group differences:")
    for r in hve_rows[:8]:
        sig = "*" if r["perm_p"] < 0.05 else ""
        print(f"    {r['feature']:35s}  Δ={r['delta']:+.4f}  d={r['cohen_d']:+.3f}  "
              f"perm-p={r['perm_p']:.3f}  {sig}")

    # ── Multivariate ──────────────────────────────────────────────────────────
    print("\nMultivariate analysis (LOO-CV) ...")
    mv = multivariate_analysis(feat_table, mean_eer, feature_cols)
    (OUT_DIR / "multivariate_explanation.json").write_text(
        json.dumps(mv, indent=2))
    if "error" not in mv:
        print(f"  LR LOO-R²: {mv['linear_regression']['loo_r2']:.3f}")
        print(f"  Lasso LOO-R²: {mv['lasso']['loo_r2']:.3f}  "
              f"selected: {mv['lasso']['selected_features']}")
        print(f"  RF LOO-R²: {mv['random_forest']['loo_r2']:.3f}")
        print(f"  RF top feature: {mv['random_forest']['top5'][0]}")

    # ── Outlier report ────────────────────────────────────────────────────────
    write_outlier_report(OUT_DIR / "outlier_report.md",
                         feat_table, corr_rows, hard, easy, mean_eer)

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(OUT_DIR / "analysis_summary.md",
                  corr_rows, hve_rows, mv, hard, easy)

    print(f"\nAll outputs → {OUT_DIR}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HIGHER-ORDER MECHANISM ANALYSIS — SUMMARY")
    print("=" * 70)
    sig_corr = [r for r in corr_rows if r["spearman_p"] < 0.05]
    print(f"\nSignificant correlates (p<0.05): {len(sig_corr)} of {len(corr_rows)} features")
    for r in corr_rows[:10]:
        sig = "  *" if r["spearman_p"] < 0.05 else ""
        print(f"  {r['feature']:35s}  ρ={r['spearman_rho']:+.3f}  p={r['spearman_p']:.3f}{sig}")
    print("\nHard vs easy (top by |d|):")
    for r in hve_rows[:5]:
        sig = "*" if r["perm_p"] < 0.05 else ""
        print(f"  {r['feature']:35s}  Δ={r['delta']:+.4f}  d={r['cohen_d']:+.3f}  "
              f"perm-p={r['perm_p']:.3f}  {sig}")
    if "error" not in mv:
        print(f"\nMultivariate LOO-R²:  LR={mv['linear_regression']['loo_r2']:.3f}  "
              f"Lasso={mv['lasso']['loo_r2']:.3f}  RF={mv['random_forest']['loo_r2']:.3f}")


if __name__ == "__main__":
    main()
