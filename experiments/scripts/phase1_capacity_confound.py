#!/usr/bin/env python3
"""
phase1_capacity_confound.py  —  Phase 1 capacity-confound check
===============================================================
Tests whether the cross-language EER improvement from ablating {top-1, top-2}
on mlaad_robust_goat is specific to those heads or explainable by capacity
reduction alone.

Null distribution: all C(6,2)-1 = 14 non-{top-1, top-2} 2-head pairs,
evaluated on BOTH in-distribution and cross-language splits.

Primary metric:
    xl_reduction = baseline_cross_lang_eer - ablated_cross_lang_eer
    (positive = cross-language EER went down = better generalisation)

Decision gate:
    PASS    — empirical_percentile >= 90  (target in top 10%)
    FAIL    — empirical_percentile <  80  OR target within ±1 SD of null mean
    MARGINAL — 80 <= percentile < 90

Writes:
    experiments/results/mlaad/phase1_confound/
        phase1_results.csv      machine-readable (null pairs + single-head runs)
        phase1_results.json     full structured output
        phase1_report.txt       concise human-readable summary + verdict

Usage:
    python experiments/scripts/phase1_capacity_confound.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

# ── torch.load compat ─────────────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
EXP_DIR      = PROJECT_ROOT / "experiments"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import head_ablation as ha
from mlaad_e4_ablation import MAALDAblationDataset, _best_or_last

PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
CKPT_DIR      = EXP_DIR / "checkpoints"
BASELINE_DIR  = EXP_DIR / "results" / "mlaad" / "baseline_eval"
HEAD_DISC_DIR = EXP_DIR / "results" / "mlaad" / "head_discovery"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "phase1_confound"

SEED          = 42
ABLATION_MODE = "zero"
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 8))


# ── JSON encoder ──────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):         return int(obj)
        if isinstance(obj, np.floating):        return float(obj)
        if isinstance(obj, np.ndarray):         return obj.tolist()
        if isinstance(obj, (frozenset, set)):   return sorted(obj)
        return super().default(obj)

def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ── Eval helpers ──────────────────────────────────────────────────────────────

def make_loader(records: list[dict]) -> torch.utils.data.DataLoader:
    ds = MAALDAblationDataset(records, PROCESSED_DIR)
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=ha.collate,
    )

def run_condition(lit, loader, device, heads: frozenset) -> list[dict]:
    return ha.run_eval(lit, loader, device, heads, ABLATION_MODE, bonafide_means={})

def pooled_eer(records: list[dict]) -> float:
    labels = np.array([r["label"] for r in records])
    scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in records], dtype=np.float64)))
    return float(ha.compute_eer(labels, scores))

def eval_pair(lit, loader_in, loader_xl, device, heads: frozenset,
              base_in: float, base_xl: float) -> dict:
    """Run one ablation condition on both splits; return summary dict."""
    t0      = time.time()
    recs_in = run_condition(lit, loader_in, device, heads)
    recs_xl = run_condition(lit, loader_xl, device, heads)
    elapsed = time.time() - t0

    p_in = pooled_eer(recs_in)
    p_xl = pooled_eer(recs_xl)
    return {
        "in_dist_eer":      round(p_in, 6),
        "cross_lang_eer":   round(p_xl, 6),
        "in_dist_delta":    round(p_in - base_in, 6),
        "cross_lang_delta": round(p_xl - base_xl, 6),
        "xl_reduction":     round(base_xl - p_xl, 6),   # positive = improved
        "elapsed_s":        round(elapsed, 1),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def dist_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "n":      len(a),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a, ddof=1) if len(a) > 1 else 0.0),
        "median": float(np.median(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "p10":    float(np.percentile(a, 10)),
        "p90":    float(np.percentile(a, 90)),
    }

def empirical_pct(target: float, null: list[float]) -> float:
    """Fraction of null values strictly below target, × 100."""
    return 100.0 * sum(v < target for v in null) / len(null)

def zscore(target: float, null: list[float]) -> float:
    a   = np.array(null)
    std = float(np.std(a, ddof=1))
    return (target - float(np.mean(a))) / std if std > 0.0 else float("inf")

def gate(target_xl_red: float, null_xl_reds: list[float]) -> tuple[str, str]:
    """Apply decision gate; return (verdict, explanation)."""
    pct   = empirical_pct(target_xl_red, null_xl_reds)
    z     = zscore(target_xl_red, null_xl_reds)
    n     = len(null_xl_reds)
    mu    = float(np.mean(null_xl_reds))
    sigma = float(np.std(null_xl_reds, ddof=1))
    n_above = sum(v >= target_xl_red for v in null_xl_reds)

    if pct >= 90.0:
        return "PASS", (
            f"empirical_percentile={pct:.1f}% (≥90%), z={z:+.2f}. "
            f"{n_above}/{n} null pairs match or beat the target. "
            "Cross-language gain is specific to {top-1, top-2}; "
            "capacity confound rejected."
        )

    fail_reasons = []
    if pct < 80.0:
        fail_reasons.append(f"percentile={pct:.1f}% < 80%")
    if abs(target_xl_red - mu) <= sigma:
        fail_reasons.append(f"target within ±1 SD of null mean (z={z:+.2f})")
    if fail_reasons:
        return "FAIL", (
            f"Target ({target_xl_red:.4f}) indistinguishable from random pairs: "
            + "; ".join(fail_reasons) + ". Capacity confound cannot be ruled out."
        )

    return "MARGINAL", (
        f"empirical_percentile={pct:.1f}% (80–90%), z={z:+.2f}. "
        "Borderline — interpret with caution."
    )


# ── CSV writer ────────────────────────────────────────────────────────────────

FIELDS = ["type", "pair_id", "ablated_heads",
          "in_dist_eer", "cross_lang_eer",
          "in_dist_delta", "cross_lang_delta", "xl_reduction"]

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Text report ───────────────────────────────────────────────────────────────

def write_report(res: dict, path: Path) -> None:
    tgt = res["target_pair"]
    st  = res["statistical_test"]
    ns  = res["null_distribution"]["xl_reduction_stats"]
    sg  = res["single_head_ablations"]
    asym = res["asymmetry_analysis"]
    dg  = res["decision_gate"]

    lines = [
        "=" * 70,
        "PHASE 1 CAPACITY-CONFOUND CHECK — mlaad_robust_goat",
        "=" * 70,
        "",
        "EXPERIMENT",
        f"  Checkpoint : mlaad_robust_goat",
        f"  Target pair: {{h{tgt['ablated_heads'][0]}, h{tgt['ablated_heads'][1]}}}  (top-1, top-2)",
        f"  Metric     : xl_reduction = baseline_cross_lang_eer − ablated_cross_lang_eer",
        f"               (positive = cross-language EER improved)",
        f"  Null pool  : {res['null_distribution']['n_pairs']} non-target 2-head pairs (exhaustive)",
        "",
        "BASELINE",
        f"  In-distribution EER : {res['baseline_in_dist_eer']:.4f}",
        f"  Cross-language EER  : {res['baseline_xl_eer']:.4f}",
        f"  Pipeline parity     : {'OK' if res['pipeline_parity_ok'] else 'WARN (>0.01 diff)'}",
        "",
        "TARGET PAIR RESULT",
        f"  Ablated heads  : {tgt['ablated_heads']}",
        f"  In-dist EER    : {tgt['in_dist_eer']:.4f}  (Δ={tgt['in_dist_delta']:+.4f})",
        f"  Cross-lang EER : {tgt['cross_lang_eer']:.4f}  (Δ={tgt['cross_lang_delta']:+.4f})",
        f"  xl_reduction   : {tgt['xl_reduction']:+.4f}",
        "",
        "NULL DISTRIBUTION (14 non-target pairs, mlaad_robust_goat, both splits)",
        f"  mean  = {ns['mean']:+.4f}",
        f"  std   = {ns['std']:.4f}",
        f"  min   = {ns['min']:+.4f}   max = {ns['max']:+.4f}",
        f"  p10   = {ns['p10']:+.4f}   p90 = {ns['p90']:+.4f}",
        "",
        "STATISTICAL TEST",
        f"  Target xl_reduction   : {st['target_xl_reduction']:+.4f}",
        f"  z-score               : {st['z_score']:+.4f}",
        f"  Empirical percentile  : {st['empirical_percentile']:.1f}%",
        f"  Pairs ≥ target        : {st['count_at_or_above']}/{st['n_null']}",
        "",
        "SINGLE-HEAD ABLATIONS",
    ]

    for r in sg:
        lines.append(
            f"  {r['condition']:12s}  ablate={r['ablated_heads']}  "
            f"in={r['in_dist_eer']:.4f}(Δ{r['in_dist_delta']:+.4f})  "
            f"xl={r['cross_lang_eer']:.4f}(Δ{r['cross_lang_delta']:+.4f})  "
            f"xl_red={r['xl_reduction']:+.4f}"
        )

    lines += [
        "",
        "ASYMMETRY (pair vs singles)",
        f"  top-1 only  xl_reduction = {asym['top1_xl_reduction']:+.4f}",
        f"  top-2 only  xl_reduction = {asym['top2_xl_reduction']:+.4f}",
        f"  pair        xl_reduction = {asym['pair_xl_reduction']:+.4f}",
        f"  sum-of-singles           = {asym['sum_of_singles']:+.4f}",
        f"  synergy (pair − sum)     = {asym['synergy']:+.4f}",
        f"  dominant head            : {asym['dominant_head']}",
        "",
        "DECISION GATE",
        f"  Rules : PASS if percentile ≥ 90%",
        f"          FAIL if percentile < 80% OR |z| ≤ 1",
        f"  VERDICT: {dg['verdict']}",
        f"  {dg['explanation']}",
        "",
        "SPECIFICITY vs CAPACITY INTERPRETATION",
    ]

    v = dg["verdict"]
    if v == "PASS":
        lines += [
            "  The cross-language gain is HEAD-SPECIFIC: ablating {top-1, top-2}",
            "  reduces cross-language EER far more than ablating any other pair of",
            "  equal cardinality. A pure capacity argument (removing any 2 heads",
            "  helps generalisation by reducing overfitting) is inconsistent with",
            "  these results. The improvement is attributable to the particular",
            "  routing pattern of the target heads, not to capacity reduction alone.",
        ]
    elif v == "FAIL":
        lines += [
            "  The cross-language gain CANNOT be attributed to head specificity:",
            "  random pairs of equal cardinality produce comparable or larger",
            "  improvements. This is consistent with a capacity confound — removing",
            "  any 2 attention heads may regularise the model and improve",
            "  cross-language generalisation regardless of which heads are removed.",
            "  The selective ablation claim is not supported.",
        ]
    else:
        lines += [
            "  The result is BORDERLINE. The target pair outperforms the null mean",
            "  but the margin is insufficient to confidently rule out a capacity",
            "  confound. Interpret the cross-language improvement with caution.",
        ]

    lines += ["", "=" * 70]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Target heads ──────────────────────────────────────────────────────────
    target_json   = json.loads((HEAD_DISC_DIR / "target_heads.json").read_text())
    target_heads  = target_json["target_heads"]   # [top-1, top-2] = [h2, h4]
    CRITICAL      = frozenset(target_heads)
    h_top1, h_top2 = target_heads[0], target_heads[1]
    print(f"Target heads (top-1=h{h_top1}, top-2=h{h_top2})  critical={sorted(CRITICAL)}")

    # ── Test records ──────────────────────────────────────────────────────────
    in_records = json.loads((BASELINE_DIR / "test_in_distribution.json").read_text())
    xl_records = json.loads((BASELINE_DIR / "test_cross_language.json").read_text())
    print(f"Records: in_dist={len(in_records)}, cross_lang={len(xl_records)}")

    # ── Checkpoint + loaders ──────────────────────────────────────────────────
    ckpt = _best_or_last("mlaad_robust_goat")
    if not ckpt.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt}")
    print(f"Checkpoint: {ckpt.name}")

    loader_in = make_loader(in_records)
    loader_xl = make_loader(xl_records)

    ha.patch_phoneme_loader()
    lit = ha.load_model(ckpt, device)
    NH  = lit.model.GAT.gat_net[0].num_of_heads
    print(f"Loaded. NH={NH}")
    ha.sanity_layer_untouched(lit)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline pass...")
    base_recs_in = run_condition(lit, loader_in, device, frozenset())
    base_recs_xl = run_condition(lit, loader_xl, device, frozenset())
    base_in = pooled_eer(base_recs_in)
    base_xl = pooled_eer(base_recs_xl)
    print(f"  in_dist={base_in:.6f}  cross_lang={base_xl:.6f}")

    # Parity vs stored mlaad_robust_e4 results
    stored_base_in = 0.360237   # from mlaad_robust_e4/in_distribution/pooled_summary.json
    stored_base_xl = 0.677390   # from mlaad_robust_e4/cross_language/pooled_summary.json
    parity_ok = (abs(base_in - stored_base_in) <= 0.01 and
                 abs(base_xl - stored_base_xl) <= 0.01)
    print(f"  Parity vs stored: in_diff={abs(base_in-stored_base_in):.4f}  "
          f"xl_diff={abs(base_xl-stored_base_xl):.4f}  "
          f"{'PASS' if parity_ok else 'WARN'}")

    # ── Single-head ablations (task item 2) ───────────────────────────────────
    single_conditions = [
        ("top1_only", frozenset({h_top1})),
        ("top2_only", frozenset({h_top2})),
        ("top1_top2", CRITICAL),
    ]

    print("\nSingle-head ablations...")
    single_rows: list[dict] = []
    target_result: dict = {}

    for cond_name, heads in single_conditions:
        r = eval_pair(lit, loader_in, loader_xl, device, heads, base_in, base_xl)
        row = {"condition": cond_name, "ablated_heads": sorted(heads), **r}
        single_rows.append(row)
        if cond_name == "top1_top2":
            target_result = r
        print(f"  {cond_name:12s}  ablate={sorted(heads)}  "
              f"in={r['in_dist_eer']:.4f}(Δ{r['in_dist_delta']:+.4f})  "
              f"xl={r['cross_lang_eer']:.4f}(Δ{r['cross_lang_delta']:+.4f})  "
              f"xl_red={r['xl_reduction']:+.4f}")

    target_xl_red = target_result["xl_reduction"]

    # ── Null distribution (task item 1) ───────────────────────────────────────
    all_pairs: list[frozenset] = sorted(
        (frozenset(c) for c in combinations(range(NH), 2) if frozenset(c) != CRITICAL),
        key=sorted,
    )
    expected_n = (NH * (NH - 1) // 2) - 1
    assert len(all_pairs) == expected_n, f"Expected {expected_n} null pairs, got {len(all_pairs)}"
    assert all(p != CRITICAL for p in all_pairs), "CRITICAL pair found in null set"
    print(f"\nNull distribution: {len(all_pairs)} pairs on mlaad_robust_goat (both splits)")
    print(f"  Pairs: {[sorted(p) for p in all_pairs]}")

    null_rows: list[dict] = []
    null_xl_reds: list[float] = []

    for i, pair in enumerate(all_pairs):
        pair_id = "h" + "_h".join(str(h) for h in sorted(pair))
        r = eval_pair(lit, loader_in, loader_xl, device, pair, base_in, base_xl)
        null_xl_reds.append(r["xl_reduction"])
        row = {"pair_id": pair_id, "ablated_heads": sorted(pair), **r}
        null_rows.append(row)
        print(f"  [{i+1:2d}/{len(all_pairs)}] {pair_id:10s}  "
              f"in={r['in_dist_eer']:.4f}(Δ{r['in_dist_delta']:+.4f})  "
              f"xl={r['cross_lang_eer']:.4f}(Δ{r['cross_lang_delta']:+.4f})  "
              f"xl_red={r['xl_reduction']:+.4f}  ({r['elapsed_s']:.1f}s)")

    # ── Statistics (task item 1, decision gate item 3) ────────────────────────
    null_stats  = dist_stats(null_xl_reds)
    pct         = empirical_pct(target_xl_red, null_xl_reds)
    z           = zscore(target_xl_red, null_xl_reds)
    n_above     = sum(v >= target_xl_red for v in null_xl_reds)
    verdict, verdict_expl = gate(target_xl_red, null_xl_reds)

    print(f"\n{'='*65}")
    print(f"PHASE 1 RESULTS")
    print(f"  Target xl_reduction = {target_xl_red:+.4f}")
    print(f"  Null: mean={null_stats['mean']:+.4f}  std={null_stats['std']:.4f}  "
          f"range=[{null_stats['min']:+.4f}, {null_stats['max']:+.4f}]")
    print(f"  z-score              = {z:+.4f}")
    print(f"  Empirical percentile = {pct:.1f}%")
    print(f"  Pairs ≥ target       = {n_above}/{len(all_pairs)}")
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_expl}")

    # ── Asymmetry analysis ────────────────────────────────────────────────────
    _by_cond = {r["condition"]: r for r in single_rows}
    top1_red = _by_cond["top1_only"]["xl_reduction"]
    top2_red = _by_cond["top2_only"]["xl_reduction"]
    pair_red = _by_cond["top1_top2"]["xl_reduction"]
    asym = {
        "top1_xl_reduction": round(float(top1_red), 6),
        "top2_xl_reduction": round(float(top2_red), 6),
        "pair_xl_reduction": round(float(pair_red), 6),
        "sum_of_singles":    round(float(top1_red + top2_red), 6),
        "synergy":           round(float(pair_red - (top1_red + top2_red)), 6),
        "dominant_head":     "top1" if top1_red > top2_red else "top2",
    }

    # ── Build output structures ───────────────────────────────────────────────
    results_json = {
        "checkpoint":          "mlaad_robust_goat",
        "target_heads":        sorted(CRITICAL),
        "baseline_in_dist_eer": round(base_in, 6),
        "baseline_xl_eer":      round(base_xl, 6),
        "pipeline_parity_ok":   parity_ok,
        "target_pair": {
            "pair_id":          f"h{h_top1}_h{h_top2}",
            "ablated_heads":    sorted(CRITICAL),
            **{k: round(float(v), 6) for k, v in target_result.items()
               if k not in ("elapsed_s",)},
            "elapsed_s":        target_result.get("elapsed_s"),
        },
        "null_distribution": {
            "n_pairs":             len(all_pairs),
            "xl_reduction_stats":  null_stats,
            "pairs":               [
                {"pair_id": r["pair_id"],
                 "ablated_heads": r["ablated_heads"],
                 "in_dist_eer":   r["in_dist_eer"],
                 "cross_lang_eer": r["cross_lang_eer"],
                 "in_dist_delta":  r["in_dist_delta"],
                 "cross_lang_delta": r["cross_lang_delta"],
                 "xl_reduction":   r["xl_reduction"]}
                for r in null_rows
            ],
        },
        "statistical_test": {
            "metric":               "xl_reduction = baseline_xl_eer - ablated_xl_eer",
            "target_xl_reduction":  round(target_xl_red, 6),
            "null_mean":            round(null_stats["mean"], 6),
            "null_std":             round(null_stats["std"], 6),
            "z_score":              round(z, 4),
            "empirical_percentile": round(pct, 1),
            "count_at_or_above":    n_above,
            "n_null":               len(all_pairs),
        },
        "decision_gate": {
            "verdict":     verdict,
            "explanation": verdict_expl,
            "gate_rules": {
                "PASS":     "empirical_percentile >= 90",
                "FAIL":     "empirical_percentile < 80 OR target within ±1 SD of null mean",
                "MARGINAL": "80 <= empirical_percentile < 90",
            },
        },
        "single_head_ablations": [
            {"condition": r["condition"],
             "ablated_heads": r["ablated_heads"],
             "in_dist_eer":   r["in_dist_eer"],
             "cross_lang_eer": r["cross_lang_eer"],
             "in_dist_delta":  r["in_dist_delta"],
             "cross_lang_delta": r["cross_lang_delta"],
             "xl_reduction":   r["xl_reduction"]}
            for r in single_rows
        ],
        "asymmetry_analysis": asym,
    }

    # ── CSV rows ──────────────────────────────────────────────────────────────
    csv_rows: list[dict] = [
        {"type": "baseline", "pair_id": "baseline", "ablated_heads": "[]",
         "in_dist_eer": round(base_in, 6), "cross_lang_eer": round(base_xl, 6),
         "in_dist_delta": 0.0, "cross_lang_delta": 0.0, "xl_reduction": 0.0},
    ]
    for r in null_rows:
        csv_rows.append({
            "type": "null_pair", "pair_id": r["pair_id"],
            "ablated_heads": str(r["ablated_heads"]),
            "in_dist_eer":    r["in_dist_eer"],
            "cross_lang_eer": r["cross_lang_eer"],
            "in_dist_delta":  r["in_dist_delta"],
            "cross_lang_delta": r["cross_lang_delta"],
            "xl_reduction":   r["xl_reduction"],
        })
    for r in single_rows:
        csv_rows.append({
            "type": "single_or_pair", "pair_id": r["condition"],
            "ablated_heads": str(r["ablated_heads"]),
            "in_dist_eer":    r["in_dist_eer"],
            "cross_lang_eer": r["cross_lang_eer"],
            "in_dist_delta":  r["in_dist_delta"],
            "cross_lang_delta": r["cross_lang_delta"],
            "xl_reduction":   r["xl_reduction"],
        })

    # ── Write ─────────────────────────────────────────────────────────────────
    write_csv(csv_rows, OUT_DIR / "phase1_results.csv")
    _dump(results_json, OUT_DIR / "phase1_results.json")
    write_report(results_json, OUT_DIR / "phase1_report.txt")

    print(f"\nOutputs:")
    print(f"  {OUT_DIR}/phase1_results.csv")
    print(f"  {OUT_DIR}/phase1_results.json")
    print(f"  {OUT_DIR}/phase1_report.txt")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
