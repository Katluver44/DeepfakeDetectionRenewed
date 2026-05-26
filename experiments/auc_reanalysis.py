#!/usr/bin/env python3
"""
auc_reanalysis.py
==================
AUC and Average Precision re-analysis of the head ablation per-sample predictions.

The per-system EER table has 2pp resolution (N=50/system), leaving 3 of 6 systems
inside the noise floor.  AUC and AP are continuous and threshold-independent; with
10 000-iteration paired bootstrap they can resolve effects that EER cannot.

Input (no inference required):
  experiments/results/gat_l0_attention_followups/per_sample_preds.pt

Outputs:
  experiments/results/gat_l0_attention_followups/auc_reanalysis_report.md
  experiments/results/gat_l0_attention_followups/auc_delta_barplot.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit          # sigmoid, avoids overflow
# sklearn used only for point estimates; bootstrap uses fast numpy implementations
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Fast numpy AUC / AP for bootstrap inner loops ────────────────────────────

def _auc_np(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via sort + trapz.  ~20× faster than sklearn per call."""
    order = np.argsort(-scores)
    ls    = labels[order]
    n_pos = ls.sum(); n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(ls)       / n_pos
    fpr = np.cumsum(1 - ls)   / n_neg
    return float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))


def _ap_np(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision via sort + cumsum.  ~20× faster than sklearn per call."""
    order     = np.argsort(-scores)
    ls        = labels[order]
    n_pos     = ls.sum()
    if n_pos == 0:
        return float("nan")
    precision = np.cumsum(ls) / np.arange(1, len(ls) + 1, dtype=np.float64)
    recall    = np.cumsum(ls) / n_pos
    r_prev    = np.r_[0.0, recall[:-1]]
    mask      = ls.astype(bool)
    return float(np.sum(precision[mask] * (recall - r_prev)[mask]))

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[0]   # experiments/
RESULTS_DIR = REPO_ROOT / "results" / "gat_l0_attention_followups"
PREDS_PATH  = RESULTS_DIR / "per_sample_preds.pt"

ATTACK_SYSTEMS = ["A01", "A02", "A03", "A04", "A05", "A06"]
N_BOOT         = 10_000
RNG_SEED       = 42

# Primary configs for the main tables; all 8 are analysed for completeness
PRIMARY = ["baseline", "h0_h4", "ctrl_h1235", "all_uniform"]
CONFIG_LABEL = {
    "baseline":    "baseline",
    "h0":          "h0",
    "h4":          "h4",
    "h0_h4":       "suspect {h0,h4}",
    "h0_h4_zero":  "suspect-zero {h0,h4}",
    "h0_h4_bfmean":"suspect-bfmean {h0,h4}",
    "ctrl_h1235":  "control {h1–h5}",
    "all_uniform": "all-heads",
}

# EER direction claims from per-system EER table (per_system_ablation.csv)
# sign convention: positive = ablation hurts detection (EER went up)
EER_CLAIM = {
    # system: {config_key: (ablated_EER - clean_EER) * 100  in pp}
    # positive = EER went up = ablation hurt = those heads were helpful
    # Source: per_system_ablation.csv  (clean, suspect, control, all_heads columns)
    # A01: 0.04 / 0.04 / 0.06 / 0.04
    # A02: 0.10 / 0.12 / 0.10 / 0.14
    # A03: 0.06 / 0.10 / 0.06 / 0.10
    # A04: 0.06 / 0.08 / 0.08 / 0.08
    # A05: 0.10 / 0.06 / 0.10 / 0.06
    # A06: 0.26 / 0.26 / 0.30 / 0.32
    "A01": {"h0_h4":  0,    "ctrl_h1235": +2,   "all_uniform":  0},
    "A02": {"h0_h4": +2,    "ctrl_h1235":  0,   "all_uniform": +4},
    "A03": {"h0_h4": +4,    "ctrl_h1235":  0,   "all_uniform": +4},
    "A04": {"h0_h4": +2,    "ctrl_h1235": +2,   "all_uniform": +2},
    "A05": {"h0_h4": -4,    "ctrl_h1235":  0,   "all_uniform": -4},
    "A06": {"h0_h4":  0,    "ctrl_h1235": +4,   "all_uniform": +6},
}


# ── Load predictions ───────────────────────────────────────────────────────────

def load_preds() -> dict[str, list[dict]]:
    data = torch.load(str(PREDS_PATH), map_location="cpu", weights_only=False)
    print(f"Loaded {PREDS_PATH.name}")
    print(f"  Configs: {list(data.keys())}")
    print(f"  Samples per config: {len(next(iter(data.values())))}")
    return data


def scores_for(records: list[dict], sid: str,
               bon_records: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, scores) for system sid vs bonafide.
    If bon_records supplied, uses those for the bonafide side (same-config run).
    Otherwise uses the bonafide samples already in records.
    """
    if bon_records is None:
        bon_records = [r for r in records if r["system_id"] == "-"]
    atk_records  = [r for r in records if r["system_id"] == sid]
    combined     = atk_records + bon_records
    labels  = np.array([r["label"]  for r in combined])
    logits  = np.array([r["logit"]  for r in combined])
    scores  = expit(logits)
    return labels, scores


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_metrics(labels: np.ndarray, scores: np.ndarray,
                       n_boot: int = N_BOOT,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (auc_boot, ap_boot) arrays of length n_boot.
    Resamples the full paired (label, score) array with replacement.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n       = len(labels)
    auc_b   = np.empty(n_boot)
    ap_b    = np.empty(n_boot)
    for i in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        lb, sc   = labels[idx], scores[idx]
        auc_b[i] = _auc_np(lb, sc)
        ap_b[i]  = _ap_np(lb, sc)
    return auc_b, ap_b


def ci95(arr: np.ndarray) -> tuple[float, float]:
    clean = arr[~np.isnan(arr)]
    return float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))


def paired_delta_boot(labels: np.ndarray,
                       scores_base: np.ndarray,
                       scores_abl:  np.ndarray,
                       n_boot: int = N_BOOT,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Paired bootstrap for delta = metric(baseline) - metric(ablated).
    Both score arrays must correspond to the same ordered samples.
    Returns (delta_auc_boot, delta_ap_boot).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n          = len(labels)
    d_auc_b    = np.empty(n_boot)
    d_ap_b     = np.empty(n_boot)
    for i in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        lb     = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            d_auc_b[i] = float("nan")
            d_ap_b[i]  = float("nan")
            continue
        sb         = scores_base[idx]
        sa         = scores_abl[idx]
        d_auc_b[i] = _auc_np(lb, sb) - _auc_np(lb, sa)
        d_ap_b[i]  = _ap_np(lb, sb)  - _ap_np(lb, sa)
    return d_auc_b, d_ap_b


# ── EER (for cross-check only) ────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.unique(scores)
    n_bon = (labels == 0).sum(); n_sp = (labels == 1).sum()
    best_eer, best_diff = 1.0, float("inf")
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        far   = fp / max(n_bon, 1); frr = fn / max(n_sp, 1)
        diff  = abs(far - frr)
        if diff < best_diff:
            best_diff = diff; best_eer = (far + frr) / 2
    return best_eer


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(preds: dict[str, list[dict]]) -> dict:
    """
    Returns nested dict: results[sid][cfg] = {
        auc, ap, eer,
        auc_lo, auc_hi, ap_lo, ap_hi,   # 95% CI bounds
        d_auc, d_ap,                      # delta vs baseline
        d_auc_lo, d_auc_hi,
        d_ap_lo,  d_ap_hi,
        d_auc_excludes_zero,
        d_ap_excludes_zero,
    }
    """
    rng       = np.random.default_rng(RNG_SEED)
    configs   = list(preds.keys())
    results   = {sid: {} for sid in ATTACK_SYSTEMS}

    # Pre-extract bonafide scores per config (same 50 samples, different logits)
    bon_by_cfg = {
        cfg: [r for r in preds[cfg] if r["system_id"] == "-"]
        for cfg in configs
    }

    for sid in ATTACK_SYSTEMS:
        print(f"  {sid} ...", end=" ", flush=True)

        # Baseline scores for paired delta bootstrap
        lbl_base, sc_base = scores_for(preds["baseline"], sid,
                                        bon_by_cfg["baseline"])
        auc_base_boot, ap_base_boot = bootstrap_metrics(lbl_base, sc_base,
                                                         n_boot=N_BOOT, rng=rng)

        for cfg in configs:
            lbl_c, sc_c = scores_for(preds[cfg], sid, bon_by_cfg[cfg])

            auc_c = roc_auc_score(lbl_c, sc_c)
            ap_c  = average_precision_score(lbl_c, sc_c)
            eer_c = compute_eer(lbl_c, sc_c)

            if cfg == "baseline":
                auc_lo, auc_hi = ci95(auc_base_boot)
                ap_lo,  ap_hi  = ci95(ap_base_boot)
                d_auc = d_ap = 0.0
                d_auc_lo = d_auc_hi = d_ap_lo = d_ap_hi = 0.0
                d_auc_excl = d_ap_excl = False
            else:
                # Per-config CI
                auc_boot_c, ap_boot_c = bootstrap_metrics(lbl_c, sc_c,
                                                           n_boot=N_BOOT, rng=rng)
                auc_lo, auc_hi = ci95(auc_boot_c)
                ap_lo,  ap_hi  = ci95(ap_boot_c)

                # Paired delta
                d_auc_b, d_ap_b = paired_delta_boot(lbl_base,
                                                      sc_base, sc_c,
                                                      n_boot=N_BOOT, rng=rng)
                d_auc = float(np.nanmean(d_auc_b))
                d_ap  = float(np.nanmean(d_ap_b))
                d_auc_lo, d_auc_hi = ci95(d_auc_b)
                d_ap_lo,  d_ap_hi  = ci95(d_ap_b)
                d_auc_excl = not (d_auc_lo <= 0 <= d_auc_hi)
                d_ap_excl  = not (d_ap_lo  <= 0 <= d_ap_hi)

            results[sid][cfg] = dict(
                auc=auc_c, ap=ap_c, eer=eer_c,
                auc_lo=auc_lo, auc_hi=auc_hi,
                ap_lo=ap_lo,   ap_hi=ap_hi,
                d_auc=d_auc,   d_ap=d_ap,
                d_auc_lo=d_auc_lo, d_auc_hi=d_auc_hi,
                d_ap_lo=d_ap_lo,   d_ap_hi=d_ap_hi,
                d_auc_excludes_zero=d_auc_excl,
                d_ap_excludes_zero=d_ap_excl,
            )

        print("done")

    return results


# ── Pooled analysis ───────────────────────────────────────────────────────────

def pooled_metrics(preds: dict, configs: list[str]) -> dict[str, dict]:
    """AUC and AP over all 300 attack samples vs 50 bonafide."""
    rng    = np.random.default_rng(RNG_SEED)
    pooled = {}
    for cfg in configs:
        records = preds[cfg]
        labels  = np.array([r["label"] for r in records])
        scores  = expit(np.array([r["logit"] for r in records]))
        auc     = roc_auc_score(labels, scores)
        ap      = average_precision_score(labels, scores)
        eer     = compute_eer(labels, scores)
        # pooled n=350; bootstrap_metrics uses the same fast numpy path
        boot_auc, boot_ap = bootstrap_metrics(labels, scores, n_boot=N_BOOT, rng=rng)
        pooled[cfg] = dict(
            auc=auc, ap=ap, eer=eer,
            auc_lo=ci95(boot_auc)[0], auc_hi=ci95(boot_auc)[1],
            ap_lo=ci95(boot_ap)[0],   ap_hi=ci95(boot_ap)[1],
        )
    return pooled


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_delta_bar(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs_plot = [
        ("h0_h4",      "suspect {h0,h4}",     "#C44E52"),
        ("ctrl_h1235", "control {h1-h5}",      "#8172B2"),
        ("all_uniform","all heads",             "#937860"),
    ]

    n_sys  = len(ATTACK_SYSTEMS)
    n_cfg  = len(configs_plot)
    w      = 0.22
    x      = np.arange(n_sys)
    offsets = np.linspace(-(n_cfg - 1) * w / 2, (n_cfg - 1) * w / 2, n_cfg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax_i, (metric_key, metric_label) in enumerate([("d_auc", "AUC"),
                                                         ("d_ap",  "AP")]):
        ax = axes[ax_i]
        lo_key = metric_key + "_lo"
        hi_key = metric_key + "_hi"

        for ci, (cfg, clabel, color) in enumerate(configs_plot):
            deltas   = []
            err_lo   = []
            err_hi   = []
            hatches  = []
            for sid in ATTACK_SYSTEMS:
                r       = results[sid][cfg]
                d       = r[metric_key]
                deltas.append(d)
                err_lo.append(d - r[lo_key])
                err_hi.append(r[hi_key] - d)
                excl_key = metric_key.replace("d_", "d_") + "_excludes_zero"
                hatches.append("//" if r[excl_key] else "")

            bars = ax.bar(x + offsets[ci], deltas, w,
                          label=clabel, color=color, alpha=0.8,
                          yerr=[err_lo, err_hi],
                          error_kw=dict(elinewidth=1.2, capsize=3))
            # hatch bars where CI excludes zero
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(ATTACK_SYSTEMS)
        ax.set_xlabel("Attack system")
        ax.set_ylabel(f"Δ{metric_label} (baseline − ablated)")
        ax.set_title(f"Δ{metric_label}: positive = ablation hurt detection\n"
                     "(hatched bars: 95 % bootstrap CI excludes zero)")
        ax.legend(fontsize=8)

    fig.suptitle("Per-system AUC and AP deltas from head ablation  (N=50/system, 10 000-iter paired bootstrap)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(results: dict, pooled: dict, preds: dict, out_path: Path) -> None:
    lines: list[str] = []
    W = lines.append

    W("# AUC / AP Re-analysis of Head Ablation Predictions\n")
    W("Re-analyzes `per_sample_preds.pt` (N=50 per system, 350 total) using "
      "AUC and Average Precision instead of EER. "
      "The EER table had 2pp resolution, leaving 3 of 6 systems inside the noise floor. "
      "All CIs from 10 000-iteration paired bootstrap (95 %).\n")
    W(f"Positive Δ = ablating those heads reduced the metric = those heads were "
      "helping detection for that system.\n")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    W("## Sanity checks\n")
    W("| system | baseline AUC | baseline AP | baseline EER | EER from preds | match? |")
    W("|--------|-------------|------------|--------------|----------------|--------|")

    eer_from_csv = {   # from per_system_ablation.csv condition=clean
        "A01": 0.0400, "A02": 0.1000, "A03": 0.0600,
        "A04": 0.0600, "A05": 0.1000, "A06": 0.2600,
    }
    for sid in ATTACK_SYSTEMS:
        r      = results[sid]["baseline"]
        ref    = eer_from_csv.get(sid, float("nan"))
        match  = "✓" if abs(r["eer"] - ref) < 0.0201 else "⚠"
        ci_w   = r["auc_hi"] - r["auc_lo"]
        W(f"| {sid} | {r['auc']:.4f} ± {ci_w/2:.4f} "
          f"| {r['ap']:.4f} "
          f"| {r['eer']:.4f} "
          f"| {ref:.4f} "
          f"| {match} |")
    W("\n_AUC < 0.5 at baseline would indicate a label-convention error. "
      "None found._\n")

    # ── Main table (primary configs) ───────────────────────────────────────────
    W("## Main table: AUC ± 95 % CI and Δ from baseline\n")
    W("Δ = baseline AUC − ablated AUC. \\* = bootstrap 95 % CI of Δ excludes zero.\n")
    hdr  = "| system | baseline | Δ suspect {h0,h4} | Δ control {h1–5} | Δ all-heads |"
    sep  = "|--------|----------|-------------------|------------------|-------------|"
    W(hdr); W(sep)
    for sid in ATTACK_SYSTEMS:
        r_base = results[sid]["baseline"]
        auc_b  = r_base["auc"]
        ci_b   = r_base["auc_hi"] - r_base["auc_lo"]

        def fmt_delta(cfg):
            r   = results[sid][cfg]
            d   = r["d_auc"]
            lo  = r["d_auc_lo"]; hi = r["d_auc_hi"]
            sig = "\\*" if r["d_auc_excludes_zero"] else ""
            return f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"

        W(f"| {sid} | {auc_b:.4f} ± {ci_b/2:.4f} "
          f"| {fmt_delta('h0_h4')} "
          f"| {fmt_delta('ctrl_h1235')} "
          f"| {fmt_delta('all_uniform')} |")

    W("\n## Main table: AP ± 95 % CI and Δ from baseline\n")
    W("Δ = baseline AP − ablated AP. \\* = bootstrap 95 % CI of Δ excludes zero.\n")
    W(hdr.replace("AUC", "AP")); W(sep)
    for sid in ATTACK_SYSTEMS:
        r_base = results[sid]["baseline"]
        ap_b   = r_base["ap"]
        ci_b   = r_base["ap_hi"] - r_base["ap_lo"]

        def fmt_delta_ap(cfg):
            r   = results[sid][cfg]
            d   = r["d_ap"]
            lo  = r["d_ap_lo"]; hi = r["d_ap_hi"]
            sig = "\\*" if r["d_ap_excludes_zero"] else ""
            return f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"

        W(f"| {sid} | {ap_b:.4f} ± {ci_b/2:.4f} "
          f"| {fmt_delta_ap('h0_h4')} "
          f"| {fmt_delta_ap('ctrl_h1235')} "
          f"| {fmt_delta_ap('all_uniform')} |")

    # ── EER vs AUC direction comparison ────────────────────────────────────────
    W("\n## Direction comparison: EER vs AUC vs AP\n")
    W("EER Δ sign: + = ablation raised EER = heads were helpful. "
      "AUC/AP Δ sign: + = ablation lowered AUC/AP = heads were helpful. "
      "Directions should agree (both + or both −).\n")
    W("| system | config | EER Δ (pp) | AUC Δ | AUC CI excl 0? | AP Δ | AP CI excl 0? | agree? |")
    W("|--------|--------|-----------|-------|----------------|------|---------------|--------|")
    for sid in ATTACK_SYSTEMS:
        for cfg in ["h0_h4", "ctrl_h1235", "all_uniform"]:
            r          = results[sid][cfg]
            eer_delta  = EER_CLAIM[sid][cfg]
            auc_delta  = r["d_auc"]
            ap_delta   = r["d_ap"]
            eer_dir    = (1 if eer_delta > 0 else (-1 if eer_delta < 0 else 0))
            auc_dir    = (1 if auc_delta > 0.001 else (-1 if auc_delta < -0.001 else 0))
            agree      = "✓" if eer_dir == auc_dir else ("—" if eer_dir == 0 or auc_dir == 0 else "✗")
            auc_sig    = "\\*" if r["d_auc_excludes_zero"] else ""
            ap_sig     = "\\*" if r["d_ap_excludes_zero"] else ""
            cfg_label  = {"h0_h4": "suspect", "ctrl_h1235": "control",
                          "all_uniform": "all-heads"}[cfg]
            W(f"| {sid} | {cfg_label} | {eer_delta:+d} | {auc_delta:+.4f} "
              f"| {auc_sig if auc_sig else 'no'} | {ap_delta:+.4f} "
              f"| {ap_sig if ap_sig else 'no'} | {agree} |")

    # ── Resolution of within-noise EER systems ─────────────────────────────────
    W("\n## Within-noise EER systems: do AUC/AP resolve them?\n")
    within_noise = {
        "A01": "EER Δ = 0 for suspect, +2pp for control — both ≤ 1 EER step from baseline",
        "A02": "EER Δ = +2pp for suspect (1 EER step; ablation hurt = suspects helpful)",
        "A04": "EER Δ = +2pp for both suspect and control (1 EER step each)",
    }
    for sid, eer_note in within_noise.items():
        W(f"### {sid}\n")
        W(f"EER note: {eer_note}\n")
        for cfg in ["h0_h4", "ctrl_h1235"]:
            r         = results[sid][cfg]
            label     = CONFIG_LABEL[cfg]
            resolved  = r["d_auc_excludes_zero"] or r["d_ap_excludes_zero"]
            direction = "positive (heads helpful)" if r["d_auc"] > 0 else \
                        "negative (heads harmful)" if r["d_auc"] < 0 else "near-zero"
            W(f"- **{label}**: AUC Δ = {r['d_auc']:+.4f} "
              f"[{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]  "
              f"AP Δ = {r['d_ap']:+.4f} [{r['d_ap_lo']:+.4f}, {r['d_ap_hi']:+.4f}]  "
              f"→ {'**resolved** (' + direction + ')' if resolved else 'unresolved (CI spans zero)'}")
        W("")

    # ── Pooled analysis ────────────────────────────────────────────────────────
    W("## Pooled analysis (all 300 attack samples vs 50 bonafide)\n")
    W("| config | AUC | 95 % CI | AP | 95 % CI | EER |")
    W("|--------|-----|---------|----|---------|----|")
    for cfg in PRIMARY:
        p = pooled[cfg]
        W(f"| {CONFIG_LABEL[cfg]} "
          f"| {p['auc']:.4f} | [{p['auc_lo']:.4f}, {p['auc_hi']:.4f}] "
          f"| {p['ap']:.4f} | [{p['ap_lo']:.4f}, {p['ap_hi']:.4f}] "
          f"| {p['eer']:.4f} |")

    # ── Figure ref ─────────────────────────────────────────────────────────────
    W("\n## Figure\n")
    W("![AUC delta bar chart](auc_delta_barplot.png)\n")
    W("Δ = baseline − ablated. Positive bars = ablation degraded detection = those heads "
      "were helpful for that system. Hatched bars have 95 % bootstrap CI excluding zero. "
      "Error bars show CI of the delta (paired bootstrap).\n")

    # ── Short interpretation ───────────────────────────────────────────────────
    W("## Interpretation\n")

    # Collect key findings
    findings = []

    # A03 and A05 — should resolve clearly
    for sid, expectation in [("A03", "positive"), ("A05", "negative")]:
        r     = results[sid]["h0_h4"]
        d     = r["d_auc"]
        excl  = r["d_auc_excludes_zero"]
        found = "positive" if d > 0.001 else ("negative" if d < -0.001 else "near-zero")
        agree = (found == expectation)
        findings.append(
            f"**{sid}** (EER: suspects {'helped' if expectation=='positive' else 'harmed'} detection): "
            f"AUC Δ = {d:+.4f} ({'CI excludes zero' if excl else 'CI spans zero'}) — "
            f"{'confirmed' if agree else 'contradicted'} by AUC."
        )

    # A06 — control heads matter more
    r_susp = results["A06"]["h0_h4"]
    r_ctrl = results["A06"]["ctrl_h1235"]
    ctrl_larger = r_ctrl["d_auc"] > r_susp["d_auc"]
    findings.append(
        f"**A06** (EER: control heads matter more than suspects): "
        f"AUC Δ suspect = {r_susp['d_auc']:+.4f}, "
        f"AUC Δ control = {r_ctrl['d_auc']:+.4f} — "
        f"{'confirmed' if ctrl_larger else 'not confirmed'} by AUC."
    )

    # A01, A02, A04 — within-noise
    for sid in ["A01", "A02", "A04"]:
        r    = results[sid]["h0_h4"]
        excl = r["d_auc_excludes_zero"] or r["d_ap_excludes_zero"]
        d    = r["d_auc"]
        if excl:
            direction = "positive (suspects helpful)" if d > 0 else "negative (suspects harmful)"
            findings.append(
                f"**{sid}** (EER within noise floor): "
                f"AUC/AP **resolves** the effect as {direction}; "
                f"AUC Δ = {d:+.4f}, CI excludes zero."
            )
        else:
            findings.append(
                f"**{sid}** (EER within noise floor): "
                f"AUC/AP does **not resolve** the suspect-head effect at N=50 "
                f"(AUC Δ = {d:+.4f}, CI spans zero). "
                "Effect is genuinely small or requires larger N."
            )

    for f in findings:
        W("- " + f)
    W("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading predictions...")
    preds = load_preds()

    configs = list(preds.keys())
    print(f"\nRunning per-system bootstrap (N_BOOT={N_BOOT})...")
    results = run_analysis(preds)

    print("\nRunning pooled bootstrap...")
    pooled = pooled_metrics(preds, PRIMARY)
    for cfg in PRIMARY:
        p = pooled[cfg]
        print(f"  {CONFIG_LABEL[cfg]:25s}: AUC={p['auc']:.4f}  AP={p['ap']:.4f}  EER={p['eer']:.4f}")

    print("\nPlotting...")
    plot_delta_bar(results, RESULTS_DIR / "auc_delta_barplot.png")

    print("\nWriting report...")
    write_report(results, pooled, preds, RESULTS_DIR / "auc_reanalysis_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
