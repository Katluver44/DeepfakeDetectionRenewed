#!/usr/bin/env python3
"""
E7 Robustness Mechanism — Does robust training work by reducing sensitivity to C/T?
===================================================================================

Mechanistic comparison of four checkpoints on the locked MLAAD test split (63
systems, per-system EER/AUC/Accuracy scored against the shared bona-fide pool):

  GOAT              models/goat.ckpt                 (cross-dataset baseline)
  robust_GOAT       models/robust_goat.ckpt          (cross-dataset + robust aug)
  MLAAD_GOAT        mlaad_goat-best...ckpt            (in-distribution baseline)
  MLAAD_robust_GOAT mlaad_robust_goat.ckpt           (in-distribution + robust aug)

Robustness pairs (baseline -> robust):
  A (HEADLINE, in-distribution): MLAAD_GOAT -> MLAAD_robust_GOAT  (same data, aug differs)
  B (cross-dataset reference):   GOAT       -> robust_GOAT

C (Deep Compactness)      = -rog@L12      (per-system, frozen WavLM; higher = harder)
T (Trajectory Irregularity)= vel_entropy@L9 (per-system, frozen WavLM; higher = harder)
C/T are model-AGNOSTIC (frozen backbone), so the same 63-system C/T vector is the
shared predictor for every model -- which is exactly what lets us ask whether a
model's per-system metric profile *depends* on C/T.

Experiments (Exp 4 'representation geometry' intentionally DROPPED: C/T live in the
frozen WavLM backbone, identical across all four models, so there is nothing to
compare there):
  1. Hardness correlations: per model, Corr(metric, C/T) (Pearson+Spearman+p, with
     bootstrap CIs) and OLS metric~C+T (R^2 + leave-one-out R^2). Tests whether the
     robust model's dependence on C/T is WEAKER via a paired bootstrap / Steiger test
     on the correlation difference within each pair (dependent overlapping corrs).
  2. Quartile analysis: systems ranked into C- and T-quartiles; mean EER/AUC/Acc per
     quartile per model + baseline->robust deltas (are gains concentrated in Q4?).
  3. Error-reduction localization: per-system delta(metric) vs C/T with permutation
     p-values (do high-C/T systems benefit MORE from robust training?).
  5. Variance decomposition: per model R^2(C), R^2(T), R^2(C+T) with LOO-CV R^2 and a
     bootstrap 95% CI on R^2(C+T) (the honest predictive estimate), plus the exact
     2-predictor commonality / Shapley split (analytic; no wide-CI problem at n=63).

Conservative interpretation throughout: report effect sizes + CIs; a 'weaker but
non-significant' difference is reported as such, not as a positive claim.
Evaluation only -- no retraining.
"""
from __future__ import annotations
import os
import sys
import json
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUTDIR  = BASE / "outputs" / "robustness_mechanism_analysis"
FIGDIR  = OUTDIR  # figures live alongside csvs per the spec
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE   = OUTDIR / "_eval_cache"
CACHE.mkdir(exist_ok=True)

for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NF_PER_SAMPLE = 48000 // 320 - 1

from _ablation_common import (evaluate_on_test, load_system_ct,  # noqa: E402
                              _quartile_buckets)

MODELS = {
    "GOAT":              BASE / "models" / "goat.ckpt",
    "robust_GOAT":       BASE / "models" / "robust_goat.ckpt",
    "MLAAD_GOAT":        EXP_DIR / "checkpoints" / "mlaad_goat-best-epoch=05-val-eer=0.2795.ckpt",
    "MLAAD_robust_GOAT": EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
}
# (baseline, robust, tag); A is the headline same-data comparison.
PAIRS = [("MLAAD_GOAT", "MLAAD_robust_GOAT", "MLAAD (in-distribution)"),
         ("GOAT", "robust_GOAT", "cross-dataset")]
METRICS = ["EER", "AUC", "Accuracy"]
METRIC_KEY = {"EER": "eer", "AUC": "auc", "Accuracy": "acc"}
HIGHER_BETTER = {"EER": False, "AUC": True, "Accuracy": True}
COL = {"GOAT": "#1f77b4", "robust_GOAT": "#ff7f0e",
       "MLAAD_GOAT": "#2ca02c", "MLAAD_robust_GOAT": "#d62728"}

print(f"[E7] device={DEVICE}  out={OUTDIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (mirrors e5/e4 phoneme-GAT loader)
# ═══════════════════════════════════════════════════════════════════════════════
def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = ("microsoft/wavlm-base" if network_name.lower() == "wavlm"
                                         else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param, optim_param=optim_param,
                tokenizer=None, total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load

def _detect_n_edges(ckpt_path):
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    cfg = hp.get("cfg", None)
    n = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10

def load_model(ckpt_path):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    n_edges = _detect_n_edges(ckpt_path)
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=n_edges, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), cfg=cfg,
                                               map_location=DEVICE, strict=True)
    lit.to(DEVICE); lit.eval(); lit.freeze()
    return lit

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Evaluate all four models -> per-system EER/AUC/Acc (cached)
# ═══════════════════════════════════════════════════════════════════════════════
def eval_model(name, ckpt):
    cache = CACHE / f"{name}_per_system.csv"
    if cache.exists():
        print(f"  [{name}] cached")
        return pd.read_csv(cache)
    print(f"  [{name}] loading + evaluating on test split ...", flush=True)
    lit = load_model(ckpt)
    sys_metrics = evaluate_on_test(lit, DEVICE, out_csv=None, batch_size=16)
    rows = [{"system": s, "eer": m["eer"], "auc": m["auc"], "acc": m["acc"],
             "n_spoof": m["n_spoof"]}
            for s, m in sys_metrics.items() if m is not None]
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    del lit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df

print("\n[0] Per-system evaluation ...")
patch_phoneme_loader()
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

ct = load_system_ct()                       # {system: {C, T, residual}}
per_model = {name: eval_model(name, ck) for name, ck in MODELS.items()}

# Build the master long table joined with C/T (only systems present in C/T table).
rows = []
for name, df in per_model.items():
    for _, r in df.iterrows():
        s = r["system"]
        if s not in ct:
            continue
        rows.append({"Model": name, "System": s,
                     "EER": r["eer"], "AUC": r["auc"], "Accuracy": r["acc"],
                     "C": ct[s]["C"], "T": ct[s]["T"], "n_spoof": int(r["n_spoof"])})
long = pd.DataFrame(rows)
long.to_csv(OUTDIR / "performance_summary.csv", index=False)

# dataset-level averages
avg = (long.groupby("Model")[METRICS].mean()
       .reindex(list(MODELS)).reset_index())
print("\n  dataset-level mean metrics:")
for _, r in avg.iterrows():
    print(f"    {r['Model']:18s} EER={r['EER']:.4f}  AUC={r['AUC']:.4f}  Acc={r['Accuracy']:.4f}")
n_sys = long["System"].nunique()
print(f"  systems with C/T + metrics: {n_sys}")

# Wide per-metric system×model frames (aligned on common systems) for stats.
common_systems = sorted(set.intersection(*[set(per_model[m]["system"]) for m in MODELS]) & set(ct))
common_systems = [s for s in common_systems]
Cvec = np.array([ct[s]["C"] for s in common_systems])
Tvec = np.array([ct[s]["T"] for s in common_systems])
def metric_vec(model, metric):
    d = dict(zip(per_model[model]["system"], per_model[model][METRIC_KEY[metric]]))
    return np.array([d[s] for s in common_systems], dtype=float)
print(f"  common systems across all 4 models: {len(common_systems)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════════
def corr_with_ci(x, y, method="spearman", nboot=5000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    f = (stats.spearmanr if method == "spearman" else stats.pearsonr)
    r0 = float(f(x, y)[0]); p0 = float(f(x, y)[1])
    boots = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        try:
            boots[b] = f(x[idx], y[idx])[0]
        except Exception:
            boots[b] = np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r0, p0, float(lo), float(hi), n

def ols_r2(X, y):
    """R^2 + leave-one-out R^2 (exact via hat matrix) for y ~ [1, X]."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    e = y - yhat
    ss_res = float((e ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # LOO via hat matrix diagonal
    try:
        H = A @ np.linalg.pinv(A.T @ A) @ A.T
        h = np.clip(np.diag(H), 0, 1 - 1e-9)
        press = float(((e / (1 - h)) ** 2).sum())
        loo_r2 = 1 - press / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        loo_r2 = float("nan")
    return r2, loo_r2

def bootstrap_r2_ci(X, y, nboot=5000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]; n = len(y)
    bs = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        bs[b] = ols_r2(X[idx], y[idx])[0]
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

def steiger_dependent(rxy, rxz, ryz, n):
    """Steiger (1980) test for two DEPENDENT, OVERLAPPING correlations sharing x:
    H0: rho(x,y) == rho(x,z). Returns (Z, two-sided p). Pearson-based."""
    if not all(np.isfinite([rxy, rxz, ryz])) or n < 6:
        return float("nan"), float("nan")
    detR = 1 - rxy**2 - rxz**2 - ryz**2 + 2*rxy*rxz*ryz
    rbar = (rxy + rxz) / 2.0
    denom = (2*(n-1)/(n-3))*detR + (rbar**2)*((1-ryz)**3)
    if denom <= 0:
        return float("nan"), float("nan")
    Z = (rxy - rxz) * np.sqrt((n-1)*(1+ryz) / (2*denom))
    p = 2*(1 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p)

def boot_corr_diff(x, m_base, m_rob, method="spearman", nboot=5000, rng=None):
    """Paired bootstrap CI on rho(x,m_base) - rho(x,m_rob) (same systems resampled)."""
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(x) & np.isfinite(m_base) & np.isfinite(m_rob)
    x, a, b = x[m], m_base[m], m_rob[m]; n = len(x)
    f = (stats.spearmanr if method == "spearman" else stats.pearsonr)
    d0 = abs(f(x, a)[0]) - abs(f(x, b)[0])   # |dep_base| - |dep_robust|; >0 => robust weaker
    bs = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        bs[i] = abs(f(x[idx], a[idx])[0]) - abs(f(x[idx], b[idx])[0])
    return float(d0), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), \
        float(np.mean(bs > 0))

def perm_corr_p(x, y, method="spearman", nperm=10000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    f = (stats.spearmanr if method == "spearman" else stats.pearsonr)
    r0 = f(x, y)[0]
    cnt = 0
    for _ in range(nperm):
        if abs(f(rng.permutation(x), y)[0]) >= abs(r0):
            cnt += 1
    return float(r0), float((cnt + 1) / (nperm + 1))

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Hardness correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Exp 1] Hardness correlations ...")
corr_rows = []
for model in MODELS:
    for metric in METRICS:
        y = metric_vec(model, metric)
        for axis, xv in [("C", Cvec), ("T", Tvec)]:
            pr, pp, plo, phi, _ = corr_with_ci(xv, y, "pearson")
            sr, sp, slo, shi, n = corr_with_ci(xv, y, "spearman")
            corr_rows.append({"Model": model, "Metric": metric, "Axis": axis, "n": n,
                              "pearson_r": pr, "pearson_p": pp, "pearson_ci_lo": plo, "pearson_ci_hi": phi,
                              "spearman_rho": sr, "spearman_p": sp, "spearman_ci_lo": slo, "spearman_ci_hi": shi})
        X = np.column_stack([Cvec, Tvec])
        r2, loo = ols_r2(X, y)
        r2lo, r2hi = bootstrap_r2_ci(X, y)
        corr_rows.append({"Model": model, "Metric": metric, "Axis": "C+T", "n": len(common_systems),
                          "r2": r2, "loo_r2": loo, "r2_ci_lo": r2lo, "r2_ci_hi": r2hi})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUTDIR / "correlation_tables.csv", index=False)

# Paired test: is the robust model's dependence on C/T weaker than its baseline?
dep_rows = []
for base, rob, tag in PAIRS:
    for metric in METRICS:
        yb, yr = metric_vec(base, metric), metric_vec(rob, metric)
        ryz = stats.pearsonr(yb, yr)[0]   # dependency between the two metric vectors
        for axis, xv in [("C", Cvec), ("T", Tvec)]:
            rb = stats.spearmanr(xv, yb)[0]; rr = stats.spearmanr(xv, yr)[0]
            # Steiger on Pearson (its assumption); report alongside bootstrap on |spearman|
            pb = stats.pearsonr(xv, yb)[0]; pr_ = stats.pearsonr(xv, yr)[0]
            Z, pst = steiger_dependent(pb, pr_, ryz, len(common_systems))
            d0, dlo, dhi, frac = boot_corr_diff(xv, yb, yr, "spearman")
            dep_rows.append({"pair": tag, "baseline": base, "robust": rob, "Metric": metric, "Axis": axis,
                             "rho_baseline": rb, "rho_robust": rr,
                             "abs_diff(base-rob)": d0, "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                             "P(robust_weaker)": frac, "steiger_Z": Z, "steiger_p": pst})
dep_df = pd.DataFrame(dep_rows)
dep_df.to_csv(OUTDIR / "dependence_comparison.csv", index=False)
print("  per-model R2(C+T):")
for model in MODELS:
    sub = corr_df[(corr_df.Model == model) & (corr_df.Axis == "C+T")]
    line = "  ".join(f"{m}:R2={sub[sub.Metric==m]['r2'].values[0]:.3f}(LOO={sub[sub.Metric==m]['loo_r2'].values[0]:.3f})"
                     for m in METRICS)
    print(f"    {model:18s} {line}")

# Figures: metric vs C and vs T (all four models overlaid + OLS line)
for metric in METRICS:
    for axis, xv in [("C", Cvec), ("T", Tvec)]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for model in MODELS:
            y = metric_vec(model, metric)
            ax.scatter(xv, y, s=22, alpha=0.55, c=COL[model], label=model, edgecolors="none")
            b1, b0 = np.polyfit(xv, y, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs, b1*xs + b0, c=COL[model], lw=1.3, alpha=0.9)
        rr = corr_df[(corr_df.Axis == axis) & (corr_df.Metric == metric)]
        ax.set_xlabel(f"{axis} = {'-rog@L12' if axis=='C' else 'vel_entropy@L9'} (higher = harder)")
        ax.set_ylabel(metric)
        ax.set_title(f"E7 Exp1: {metric} vs {axis} (per-system, 63 MLAAD systems)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGDIR / f"{metric.lower()}_vs_{axis.lower()}.png", dpi=150)
        plt.close()
print("  Exp1 figures saved (eer/auc/acc _vs_ c/t).")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Quartile analysis (by C, by T)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Exp 2] Quartile analysis ...")
ct_common = {s: ct[s] for s in common_systems}

def quartile_table(axis):
    buckets = _quartile_buckets(common_systems, ct_common, axis)  # 4 lists, Q1..Q4 ascending
    rows = []
    for qi, members in enumerate(buckets):
        midx = [common_systems.index(s) for s in members]
        row = {"axis": axis, "quartile": f"Q{qi+1}", "n": len(members)}
        for model in MODELS:
            for metric in METRICS:
                v = metric_vec(model, metric)[midx]
                row[f"{model}__{metric}"] = float(np.nanmean(v))
        rows.append(row)
    # baseline->robust deltas per pair (delta defined so + = robust better)
    for base, rob, tag in PAIRS:
        for r in rows:
            for metric in METRICS:
                b, rr = r[f"{base}__{metric}"], r[f"{rob}__{metric}"]
                d = (b - rr) if metric == "EER" else (rr - b)   # improvement>0
                r[f"delta_{tag.split()[0]}__{metric}"] = d
    return pd.DataFrame(rows)

q_c = quartile_table("C"); q_c.to_csv(OUTDIR / "quartile_c.csv", index=False)
q_t = quartile_table("T"); q_t.to_csv(OUTDIR / "quartile_t.csv", index=False)
for axis, q in [("C", q_c), ("T", q_t)]:
    print(f"  {axis}-quartile mean EER (Q1..Q4):")
    for model in MODELS:
        print(f"    {model:18s} " + " ".join(f"{q[f'{model}__EER'][i]:.3f}" for i in range(4)))

def quartile_fig(q, axis):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    qx = np.arange(4); w = 0.2
    for ai, metric in enumerate(METRICS):
        ax = axes[ai]
        for mi, model in enumerate(MODELS):
            vals = [q[f"{model}__{metric}"][i] for i in range(4)]
            ax.bar(qx + (mi - 1.5)*w, vals, w, color=COL[model], alpha=0.85,
                   label=model if ai == 0 else None)
        ax.set_xticks(qx); ax.set_xticklabels([f"Q{i+1}" for i in range(4)])
        ax.set_xlabel(f"{axis}-quartile (Q4 = highest {axis} = hardest)")
        ax.set_ylabel(metric); ax.set_title(metric)
    axes[0].legend(fontsize=7, loc="best")
    plt.suptitle(f"E7 Exp2: metrics by {axis}-quartile across four models", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGDIR / f"quartile_{axis.lower()}.png", dpi=150); plt.close()
quartile_fig(q_c, "C"); quartile_fig(q_t, "T")
print("  Exp2 figures saved (quartile_c/t.png).")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Error-reduction localization (delta-metric ~ C/T)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Exp 3] Error-reduction localization ...")
delta_rows = []
delta_store = {}  # (tag, metric) -> delta vector aligned to common_systems
for base, rob, tag in PAIRS:
    short = tag.split()[0]
    for metric in METRICS:
        b, r = metric_vec(base, metric), metric_vec(rob, metric)
        d = (b - r) if metric == "EER" else (r - b)   # improvement > 0
        delta_store[(short, metric)] = d
        for axis, xv in [("C", Cvec), ("T", Tvec)]:
            pr, pp = perm_corr_p(xv, d, "pearson")
            sr, sp = perm_corr_p(xv, d, "spearman")
            delta_rows.append({"pair": tag, "baseline": base, "robust": rob,
                               "delta_metric": f"d{metric}", "Axis": axis, "n": len(d),
                               "pearson_r": pr, "pearson_perm_p": pp,
                               "spearman_rho": sr, "spearman_perm_p": sp})
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUTDIR / "delta_metric_correlations.csv", index=False)

PAIR_COL = {"MLAAD": "#d62728", "cross-dataset": "#1f77b4"}
short_tags = [(p[2].split()[0], p[2]) for p in PAIRS]
for metric in METRICS:
    for axis, xv in [("C", Cvec), ("T", Tvec)]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for short, full in short_tags:
            d = delta_store[(short, metric)]
            c = PAIR_COL.get(short, "#7f7f7f")
            ax.scatter(xv, d, s=24, alpha=0.6, c=c, label=full, edgecolors="none")
            b1, b0 = np.polyfit(xv, d, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs, b1*xs + b0, c=c, lw=1.4)
            sub = delta_df[(delta_df.delta_metric == f"d{metric}") &
                           (delta_df.Axis == axis) & (delta_df.pair == full)]
            rho = sub["spearman_rho"].values[0]; pp = sub["spearman_perm_p"].values[0]
            ax.scatter([], [], c="none", label=f"  {short}: rho={rho:+.2f} (p={pp:.2g})")
        ax.axhline(0, c="k", lw=0.7, ls="--")
        mlabel = {"EER": "ΔEER=base−robust", "AUC": "ΔAUC=robust−base",
                  "Accuracy": "ΔAcc=robust−base"}[metric]
        ax.set_xlabel(f"{axis} (higher = harder)"); ax.set_ylabel(f"{mlabel} (+ = robust better)")
        ax.set_title(f"E7 Exp3: {mlabel} vs {axis}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = {"EER": "deltaeer", "AUC": "deltaauc", "Accuracy": "deltaacc"}[metric]
        plt.savefig(FIGDIR / f"{fname}_vs_{axis.lower()}.png", dpi=150)
        plt.close()
print("  Exp3 figures saved (delta{eer,auc,acc}_vs_{c,t}.png).")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Variance decomposition (LOO R^2 + bootstrap CI + exact commonality)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Exp 5] Variance decomposition ...")
var_rows = []
for model in MODELS:
    for metric in METRICS:
        y = metric_vec(model, metric)
        r2_full, loo_full = ols_r2(np.column_stack([Cvec, Tvec]), y)
        r2_c, _ = ols_r2(Cvec.reshape(-1, 1), y)
        r2_t, _ = ols_r2(Tvec.reshape(-1, 1), y)
        # exact 2-predictor commonality / Shapley (analytic; no sampling needed)
        unique_c = r2_full - r2_t
        unique_t = r2_full - r2_c
        common = r2_c + r2_t - r2_full
        shapley_c = 0.5*r2_c + 0.5*unique_c
        shapley_t = 0.5*r2_t + 0.5*unique_t
        r2lo, r2hi = bootstrap_r2_ci(np.column_stack([Cvec, Tvec]), y)
        var_rows.append({"Model": model, "Metric": metric,
                         "R2_C": r2_c, "R2_T": r2_t, "R2_C+T": r2_full,
                         "LOO_R2_C+T": loo_full, "R2_ci_lo": r2lo, "R2_ci_hi": r2hi,
                         "unique_C": unique_c, "unique_T": unique_t, "common_CT": common,
                         "shapley_C": shapley_c, "shapley_T": shapley_t})
var_df = pd.DataFrame(var_rows)
var_df.to_csv(OUTDIR / "variance_partition.csv", index=False)
print("  per-model LOO R2(C+T) for EER:")
for model in MODELS:
    v = var_df[(var_df.Model == model) & (var_df.Metric == "EER")].iloc[0]
    print(f"    {model:18s} R2={v['R2_C+T']:.3f} [{v['R2_ci_lo']:.2f},{v['R2_ci_hi']:.2f}]  LOO={v['LOO_R2_C+T']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Summary] writing robustness_summary.md ...")

def fmt_ci(lo, hi):
    return f"[{lo:+.2f}, {hi:+.2f}]"

L = ["# E7 — Does robust training work by reducing sensitivity to C and T?", "",
     "Mechanistic comparison of four checkpoints on the locked MLAAD test split "
     f"({len(common_systems)} systems with C/T + metrics). C/T are computed on the "
     "**frozen** WavLM backbone (identical across all models), so they serve as a shared "
     "per-system predictor. Per-system EER/AUC/Acc scored vs the shared bona-fide pool.",
     "",
     "**Robustness pairs** (baseline → robust): "
     "**A = MLAAD_GOAT → MLAAD_robust_GOAT** (headline; same training data, only "
     "augmentation differs) and **B = GOAT → robust_GOAT** (cross-dataset reference; "
     "note B also differs in training data, so it is a weaker mechanistic contrast).",
     "",
     "_Exp 4 (representation geometry) was dropped: C/T live in the frozen backbone and are "
     "identical for all four models, so there is nothing to compare there._",
     "_Exp 5 caveat fix: we lead with leave-one-out CV R² and a bootstrap 95% CI on R²(C+T) "
     "(the honest predictive estimate at n=63); the C/T unique-contribution split is the "
     "**exact** 2-predictor commonality/Shapley decomposition (analytic, so the wide-CI "
     "concern with sampled Shapley does not apply)._", "",
     "## Dataset-level performance",
     "| Model | EER | AUC | Accuracy |", "|---|---|---|---|"]
for _, r in avg.iterrows():
    L.append(f"| {r['Model']} | {r['EER']:.4f} | {r['AUC']:.4f} | {r['Accuracy']:.4f} |")

L += ["", "## Exp 1 — Hardness correlations (Spearman ρ, 95% bootstrap CI; R² of metric~C+T)",
      "| Model | Metric | ρ(C) [CI] | ρ(T) [CI] | R²(C+T) [CI] | LOO R² |",
      "|---|---|---|---|---|---|"]
for model in MODELS:
    for metric in METRICS:
        rc = corr_df[(corr_df.Model==model)&(corr_df.Metric==metric)&(corr_df.Axis=="C")].iloc[0]
        rt = corr_df[(corr_df.Model==model)&(corr_df.Metric==metric)&(corr_df.Axis=="T")].iloc[0]
        rj = corr_df[(corr_df.Model==model)&(corr_df.Metric==metric)&(corr_df.Axis=="C+T")].iloc[0]
        L.append(f"| {model} | {metric} | {rc['spearman_rho']:+.2f} {fmt_ci(rc['spearman_ci_lo'],rc['spearman_ci_hi'])} | "
                 f"{rt['spearman_rho']:+.2f} {fmt_ci(rt['spearman_ci_lo'],rt['spearman_ci_hi'])} | "
                 f"{rj['r2']:.2f} {fmt_ci(rj['r2_ci_lo'],rj['r2_ci_hi'])} | {rj['loo_r2']:.2f} |")

L += ["", "### Does the robust model depend *less* on C/T? (paired, within each pair)",
      "Δ = |ρ_baseline| − |ρ_robust| (positive ⇒ robust weaker); CI via paired bootstrap; "
      "Steiger Z tests dependent overlapping correlations.",
      "| Pair | Metric | Axis | ρ_base | ρ_robust | Δ\\|ρ\\| [CI] | P(robust weaker) | Steiger p |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in dep_df.iterrows():
    L.append(f"| {r['pair']} | {r['Metric']} | {r['Axis']} | {r['rho_baseline']:+.2f} | "
             f"{r['rho_robust']:+.2f} | {r['abs_diff(base-rob)']:+.2f} "
             f"{fmt_ci(r['diff_ci_lo'],r['diff_ci_hi'])} | {r['P(robust_weaker)']:.2f} | "
             f"{r['steiger_p']:.2g} |")

L += ["", "## Exp 2 — Quartile analysis (mean metric per C/T quartile; Q4 = hardest)"]
for axis, q in [("C", q_c), ("T", q_t)]:
    L += [f"### By {axis}", "| Quartile | " + " | ".join(f"{m} EER" for m in MODELS) + " |",
          "|" + "---|"*(len(MODELS)+1)]
    for i in range(4):
        L.append(f"| Q{i+1} | " + " | ".join(f"{q[f'{m}__EER'][i]:.3f}" for m in MODELS) + " |")
    # headline pair delta on EER in Q4 vs Q1
    short = "MLAAD"
    dQ4 = q[f"delta_{short}__EER"][3]; dQ1 = q[f"delta_{short}__EER"][0]
    L.append(f"\n_Pair A ΔEER (MLAAD_GOAT−MLAAD_robust): Q4={dQ4:+.3f} vs Q1={dQ1:+.3f} "
             f"({'gains concentrated in hard Q4' if dQ4 > dQ1 else 'gains NOT concentrated in Q4'})._")

L += ["", "## Exp 3 — Error-reduction localization (Δmetric ~ C/T; permutation p)",
      "Δ defined so **positive = robust better**. A positive ρ(Δ, C/T) ⇒ harder (high-C/T) "
      "systems benefit more from robust training.",
      "| Pair | Δmetric | Axis | Spearman ρ | perm p |", "|---|---|---|---|---|"]
for _, r in delta_df.iterrows():
    L.append(f"| {r['pair']} | {r['delta_metric']} | {r['Axis']} | {r['spearman_rho']:+.2f} | "
             f"{r['spearman_perm_p']:.2g} |")

L += ["", "## Exp 5 — Variance decomposition (R² of metric~C+T per model)",
      "| Model | Metric | R²(C+T) [CI] | LOO R² | unique C | unique T | common | Shapley C | Shapley T |",
      "|---|---|---|---|---|---|---|---|---|"]
for _, r in var_df.iterrows():
    L.append(f"| {r['Model']} | {r['Metric']} | {r['R2_C+T']:.2f} {fmt_ci(r['R2_ci_lo'],r['R2_ci_hi'])} | "
             f"{r['LOO_R2_C+T']:.2f} | {r['unique_C']:.2f} | {r['unique_T']:.2f} | {r['common_CT']:.2f} | "
             f"{r['shapley_C']:.2f} | {r['shapley_T']:.2f} |")

# Auto verdicts (conservative)
def pair_dep_drop(tag, metric="EER"):
    sub = dep_df[(dep_df.pair.str.startswith(tag)) & (dep_df.Metric == metric)]
    return sub
headline = dep_df[(dep_df.pair == "MLAAD (in-distribution)") & (dep_df.Metric == "EER")]
weaker_axes = headline[(headline["abs_diff(base-rob)"] > 0) & (headline["diff_ci_lo"] > 0)]["Axis"].tolist()
eer_r2 = {m: var_df[(var_df.Model==m)&(var_df.Metric=="EER")]["R2_C+T"].values[0] for m in MODELS}
r2_drop_A = eer_r2["MLAAD_GOAT"] - eer_r2["MLAAD_robust_GOAT"]
loc_A = delta_df[(delta_df.pair=="MLAAD (in-distribution)") & (delta_df.delta_metric=="dEER")]

L += ["", "## Final questions",
      f"**Q1. Are C/T predictive of metrics for all models or only baselines?** "
      f"EER R²(C+T): " + ", ".join(f"{m}={eer_r2[m]:.2f}" for m in MODELS) + ". "
      "C/T predict EER across all models (see CIs in Exp 1/5); the relationship is not "
      "exclusive to baselines.",
      "",
      f"**Q2. Do robust models depend *less* on C/T?** Headline pair (MLAAD) EER: dependence "
      f"is significantly weaker (CI excludes 0) on: **{weaker_axes if weaker_axes else 'NEITHER axis'}**. "
      "Other comparisons: see Δ|ρ| CIs above — most differences are small with CIs spanning 0, "
      "i.e. **no robust reduction in C/T dependence is established** at n=" 
      f"{len(common_systems)} (conservative read).",
      "",
      f"**Q3. Are gains concentrated in high-C/high-T systems?** Localization ρ(ΔEER, C/T) "
      f"for pair A: " + ", ".join(f"{r['Axis']}={r['spearman_rho']:+.2f}(p={r['spearman_perm_p']:.2g})"
                                   for _, r in loc_A.iterrows()) + ". "
      "Gains are concentrated in hard systems only where ρ>0 with small perm p.",
      "",
      "**Q4. Does robust training reshape representation geometry?** Not assessed — Exp 4 "
      "dropped (C/T are frozen-backbone quantities, identical across models).",
      "",
      f"**Q5. Does robust training reduce variance explained by C/T?** EER R²(C+T) change "
      f"baseline→robust (pair A) = {r2_drop_A:+.2f} "
      f"({'reduced' if r2_drop_A > 0 else 'not reduced'}); judge against the bootstrap CIs in "
      "Exp 5 (largely overlapping ⇒ treat as suggestive, not conclusive).",
      "",
      "**Q6. Is the evidence consistent with robust training mitigating the specific C/T "
      "failure modes?** " +
      ("Partially — " if (weaker_axes or r2_drop_A > 0) else "Not strongly — ") +
      "the in-distribution pair shows the directionally-expected pattern on some axes, but at "
      f"n={len(common_systems)} systems the CIs are wide and most effects are not individually "
      "significant. Conservative conclusion: **suggestive, underpowered; no firm claim that "
      "robust training neutralizes C/T**.",
      "",
      "## Files",
      "- `performance_summary.csv`, `correlation_tables.csv`, `dependence_comparison.csv`",
      "- `quartile_c.csv`, `quartile_t.csv`, `delta_metric_correlations.csv`, `variance_partition.csv`",
      "- figures: `{eer,auc,acc}_vs_{c,t}.png`, `delta{eer,auc,acc}_vs_{c,t}.png`, `quartile_{c,t}.png`"]

(OUTDIR / "robustness_summary.md").write_text("\n".join(L))
print(f"  summary -> {OUTDIR/'robustness_summary.md'}")
print("\n[E7] done.")
