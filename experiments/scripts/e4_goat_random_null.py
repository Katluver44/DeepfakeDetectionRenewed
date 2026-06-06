#!/usr/bin/env python3
"""
e4_goat_random_null.py
=======================
Null distribution for E4-style layer-0 head-pair ablation on goat.ckpt.

With NH=6 there are only C(6,2)-1 = 14 non-{h0,h4} head pairs — essentially
exhaustive for a 10-pair target — so all 14 are evaluated rather than
sub-sampling 10.

Each pair is evaluated in two directions (matching e4_goat_baseline):
  ablate_pair  — zero the sampled pair
  ablate_compl — zero the complement (keep only the sampled pair)

Imports ablation/eval utilities from head_ablation.py and gat_e3_e4.py.
{h0,h4} reference values are loaded from the e4_goat_baseline CSV.

Outputs → experiments/results/gat_l0_attention_followups/e4_goat_random_null/
  per_pair_results.json
  null_distribution_summary.json
  comparison_to_h0h4.json
  run_config.json
"""
from __future__ import annotations

import hashlib
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
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT    = EXP_DIR.parent
OUT_DIR      = EXP_DIR / "results" / "gat_l0_attention_followups" / "e4_goat_random_null"
BASELINE_CSV = EXP_DIR / "results" / "gat_l0_attention_followups" / "e4_goat_baseline" / \
               "e4_goat_baseline_per_system_eer.csv"
CKPT         = REPO_ROOT / "models" / "goat.ckpt"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import existing utilities ─────────────────────────────────────────────────
import head_ablation as ha
import gat_e3_e4 as e4lib

# ── Constants (must match e4_goat_baseline exactly) ───────────────────────────
HF_DATASET     = "Bisher/ASVspoof_2019_LA"
CACHE_DIR      = REPO_ROOT / "data" / "asvspoof_2019_la"
CRITICAL_HEADS = frozenset({0, 4})
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42
N_BOOTSTRAP    = 1_000
ABLATION_MODE  = "zero"
ABLATION_LAYER = 0
BASELINE_EER_TOLERANCE = 0.01   # ±1pp for pipeline-parity sanity check


# ── JSON serialisation ────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, (frozenset, set)): return sorted(obj)
        return super().default(obj)


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ── Helpers ───────────────────────────────────────────────────────────────────

def pooled_eer(records: list[dict]) -> float:
    labels = np.array([r["label"] for r in records])
    scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in records])))
    return ha.compute_eer(labels, scores)


def run_condition(lit, loader, device, heads: frozenset) -> list[dict]:
    return ha.run_eval(lit, loader, device, heads, ABLATION_MODE, bonafide_means={})


def eval_set_hash(dataset: ha.BalancedDataset) -> str:
    key = json.dumps(sorted(zip(dataset.indices, dataset.sys_ids)))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ckpt_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_baseline_values(csv_path: Path) -> dict:
    """Read pooled and per-attack EER/ΔEER from the e4_goat_baseline CSV."""
    import csv as _csv
    rows = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            rows[row["system"]] = row
    pooled = rows["pooled"]
    attack_rows = {s: rows[s] for s in rows if s != "pooled"}
    return {
        "baseline_pooled_eer":       float(pooled["eer_baseline"]),
        "ablate_h0h4_pooled_eer":    float(pooled["eer_ablate_h0_h4"]),
        "ablate_h0h4_pooled_delta":  float(pooled["delta_ablate_h0_h4"]),
        "keep_h0h4_pooled_eer":      float(pooled["eer_keep_h0_h4"]),
        "keep_h0h4_pooled_delta":    float(pooled["delta_keep_h0_h4"]),
        "per_attack_ablate": {s: float(v["eer_ablate_h0_h4"]) for s, v in attack_rows.items()},
        "per_attack_keep":   {s: float(v["eer_keep_h0_h4"])   for s, v in attack_rows.items()},
    }


def distribution_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "n":       len(a),
        "mean":    float(np.mean(a)),
        "median":  float(np.median(a)),
        "std":     float(np.std(a, ddof=1) if len(a) > 1 else 0.0),
        "min":     float(np.min(a)),
        "max":     float(np.max(a)),
        "percentiles": {str(p): float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)},
    }


def compare_to_null(observed: float, null_values: list[float], label: str) -> dict:
    """One-sided empirical p-values and rank for an observed statistic vs a null distribution."""
    n = len(null_values)
    a = np.array(null_values)
    rank_from_bottom = int((a < observed).sum())
    pct = rank_from_bottom / n * 100.0
    p_above = float((a >= observed).sum()) / n   # fraction at least as extreme upward
    p_below = float((a <= observed).sum()) / n   # fraction at least as extreme downward
    return {
        "label":             label,
        "observed":          float(observed),
        "n_null":            n,
        "null_values":       [float(v) for v in null_values],
        "rank_from_bottom":  rank_from_bottom,
        "empirical_percentile": round(pct, 1),
        "p_one_sided_above": round(p_above, 4),   # H1: observed > null
        "p_one_sided_below": round(p_below, 4),   # H1: observed < null
        "interpretation": (
            f"{rank_from_bottom}/{n} null pairs have ΔEER < {observed:.4f}; "
            f"{int((a > observed).sum())}/{n} have ΔEER > {observed:.4f}"
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}"); sys.exit(1)
    if not BASELINE_CSV.exists():
        print(f"ERROR: e4_goat_baseline CSV not found: {BASELINE_CSV}")
        print("Run experiments/scripts/e4_goat_baseline.py first."); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ha.patch_phoneme_loader()
    print(f"Loading model: {CKPT.name}")
    lit = ha.load_model(CKPT, device)
    print("  model loaded")

    # ── NH from model (not hardcoded) ─────────────────────────────────────────
    NH = lit.model.GAT.gat_net[0].num_of_heads
    print(f"  GAT layer 0: {NH} heads")

    # ── Enumerate all non-{h0,h4} pairs ──────────────────────────────────────
    all_pairs: list[frozenset] = sorted(
        (frozenset(c) for c in combinations(range(NH), len(CRITICAL_HEADS))
         if frozenset(c) != CRITICAL_HEADS),
        key=sorted,
    )
    n_possible = sum(1 for _ in combinations(range(NH), len(CRITICAL_HEADS))) - 1
    assert n_possible == len(all_pairs)
    note = (f"All {len(all_pairs)} non-{{h0,h4}} pairs enumerated "
            f"(C({NH},{len(CRITICAL_HEADS)})-1 = {n_possible}; "
            f"≈ exhaustive for a 10-pair target)")
    print(f"\n  {note}")
    print(f"  Pairs: {[sorted(p) for p in all_pairs]}")

    # ── Sanity (a): no pair is {h0,h4} ───────────────────────────────────────
    assert all(p != CRITICAL_HEADS for p in all_pairs), \
        "SANITY FAIL (a): {h0,h4} found in null pairs"
    print("  Sanity (a): no pair is {h0,h4} — PASS")

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    print(f"\nLoading dataset (N_PER_CLASS={N_PER_CLASS}, split=validation)...")
    dataset = ha.BalancedDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token, N_PER_CLASS, SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=ha.collate)

    ds_hash     = eval_set_hash(dataset)
    attack_sids = sorted(s for s in set(dataset.sys_ids) if s != "-")
    print(f"  eval set hash:  {ds_hash}")

    # ── Load reference {h0,h4} values ─────────────────────────────────────────
    ref = load_baseline_values(BASELINE_CSV)
    print(f"\n  Reference from e4_goat_baseline:")
    print(f"    baseline pooled EER     = {ref['baseline_pooled_eer']:.4f}")
    print(f"    ablate_h0_h4 pooled EER = {ref['ablate_h0h4_pooled_eer']:.4f}  "
          f"(Δ={ref['ablate_h0h4_pooled_delta']:+.4f})")
    print(f"    keep_h0_h4   pooled EER = {ref['keep_h0h4_pooled_eer']:.4f}  "
          f"(Δ={ref['keep_h0h4_pooled_delta']:+.4f})")

    # ── Determinism check ─────────────────────────────────────────────────────
    # ha.sanity_baseline_identity uses a 1e-5 threshold that can fire due to
    # GPU float32 non-determinism (inter-run kernel variation, not SpecAugment).
    # We use 1e-4, which is still >>100× tighter than the EER resolution (0.02).
    _batch   = next(iter(loader))
    _audio   = _batch["audio"].to(device)
    _num_f   = torch.full((_audio.shape[0],), ha.NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        _hs, _pids = ha.run_frozen_frontend(_audio, lit.model, device)
        _lg1 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
        _lg2 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
    _diff = (_lg1 - _lg2).abs().max().item()
    assert _diff < 1e-4, f"Baseline not deterministic: max logit diff={_diff:.2e}"
    print(f"  Determinism check PASSED: max logit diff = {_diff:.2e}")
    ha.sanity_layer_untouched(lit)

    # ── Baseline pass (sanity b: must match e4_goat_baseline within tolerance) ─
    print(f"\n  Running baseline pass...")
    t0 = time.time()
    base_recs   = run_condition(lit, loader, device, frozenset())
    base_pooled = pooled_eer(base_recs)
    print(f"    pooled EER={base_pooled:.4f}  ({time.time()-t0:.1f}s)")

    ref_base = ref["baseline_pooled_eer"]
    if abs(base_pooled - ref_base) <= BASELINE_EER_TOLERANCE:
        print(f"  Sanity (b): baseline EER {base_pooled:.4f} matches e4_goat_baseline "
              f"{ref_base:.4f} within ±{BASELINE_EER_TOLERANCE} — PASS")
        sanity_b_pass = True
    else:
        print(f"  Sanity (b): FAIL — this-run baseline {base_pooled:.4f} ≠ "
              f"e4_goat_baseline {ref_base:.4f} (diff={abs(base_pooled-ref_base):.4f})")
        sanity_b_pass = False

    base_per_attack = e4lib.per_system_eer(base_recs, attack_sids)

    # ── Run all pairs (two directions each) ───────────────────────────────────
    pair_records: dict[str, dict[str, list[dict]]] = {}   # key → {ablate,compl} → records

    for pair in all_pairs:
        key  = "_".join(str(h) for h in sorted(pair))
        compl = frozenset(range(NH)) - pair

        print(f"\n  Pair {sorted(pair)}:")
        t0 = time.time()

        recs_ablate = run_condition(lit, loader, device, pair)
        t1 = time.time()
        recs_compl  = run_condition(lit, loader, device, compl)
        t2 = time.time()

        pair_records[key] = {"ablate": recs_ablate, "compl": recs_compl, "pair": sorted(pair)}
        print(f"    ablate EER={pooled_eer(recs_ablate):.4f} ({t1-t0:.1f}s)  "
              f"compl EER={pooled_eer(recs_compl):.4f} ({t2-t1:.1f}s)")

    # ── Sanity (c): not all Δ EERs identical ─────────────────────────────────
    delta_ablates = [pooled_eer(v["ablate"]) - base_pooled for v in pair_records.values()]
    if len(set(f"{d:.6f}" for d in delta_ablates)) == 1:
        print("\n  Sanity (c): FAIL — all ablate-direction ΔEER values are identical; "
              "ablation may not be firing")
        sanity_c_pass = False
    else:
        print(f"\n  Sanity (c): ΔEER values vary across pairs "
              f"(range [{min(delta_ablates):.4f}, {max(delta_ablates):.4f}]) — PASS")
        sanity_c_pass = True

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    print("\n  Computing bootstrap CIs...")
    rng = np.random.default_rng(SEED)   # shared, advancing through all pairs+directions
    ci_ablate: dict[str, dict] = {}
    ci_compl:  dict[str, dict] = {}

    for pair in all_pairs:
        key = "_".join(str(h) for h in sorted(pair))
        ci_ablate[key] = e4lib.bootstrap_eer_ci(
            pair_records[key]["ablate"], attack_sids, N_BOOTSTRAP, rng=rng)
        ci_compl[key]  = e4lib.bootstrap_eer_ci(
            pair_records[key]["compl"],  attack_sids, N_BOOTSTRAP, rng=rng)

    # ── Build per_pair_results ────────────────────────────────────────────────
    per_pair: list[dict] = []
    for pair in all_pairs:
        key   = "_".join(str(h) for h in sorted(pair))
        compl = sorted(frozenset(range(NH)) - pair)
        recs_a = pair_records[key]["ablate"]
        recs_c = pair_records[key]["compl"]
        p_a    = pooled_eer(recs_a)
        p_c    = pooled_eer(recs_c)
        psa    = e4lib.per_system_eer(recs_a, attack_sids)
        psc    = e4lib.per_system_eer(recs_c, attack_sids)

        per_pair.append({
            "pair":         sorted(pair),
            "complement":   compl,
            "ablate_pair": {
                "ablated_heads":  sorted(pair),
                "pooled_eer":     float(p_a),
                "delta_eer":      float(p_a - base_pooled),
                "per_attack_eer": {s: float(v) for s, v in psa.items()},
                "per_attack_ci":  {s: [float(ci_ablate[key][s][0]),
                                       float(ci_ablate[key][s][1])]
                                   for s in attack_sids},
            },
            "ablate_compl": {
                "ablated_heads":  compl,
                "pooled_eer":     float(p_c),
                "delta_eer":      float(p_c - base_pooled),
                "per_attack_eer": {s: float(v) for s, v in psc.items()},
                "per_attack_ci":  {s: [float(ci_compl[key][s][0]),
                                       float(ci_compl[key][s][1])]
                                   for s in attack_sids},
            },
        })

    # ── Null distribution summary ─────────────────────────────────────────────
    delta_ablate_vals = [r["ablate_pair"]["delta_eer"] for r in per_pair]
    delta_compl_vals  = [r["ablate_compl"]["delta_eer"] for r in per_pair]

    null_summary = {
        "n_pairs":     len(all_pairs),
        "exhaustive_note": note,
        "baseline_pooled_eer": float(base_pooled),
        "ablate_pair": distribution_stats(delta_ablate_vals),
        "ablate_compl": distribution_stats(delta_compl_vals),
    }

    # ── Comparison to {h0,h4} ─────────────────────────────────────────────────
    comparison = {
        "source_file": str(BASELINE_CSV),
        "reference_baseline_eer":  ref["baseline_pooled_eer"],
        "this_run_baseline_eer":   float(base_pooled),
        "pipeline_parity_ok":      sanity_b_pass,
        "ablate_direction": compare_to_null(
            ref["ablate_h0h4_pooled_delta"], delta_ablate_vals, "ablate {h0,h4}"),
        "keep_direction": compare_to_null(
            ref["keep_h0h4_pooled_delta"], delta_compl_vals, "keep-only {h0,h4}"),
    }

    # ── Checkpoint hash ───────────────────────────────────────────────────────
    print("\n  Computing checkpoint hash (may take ~2s)...")
    ckpt_sha = ckpt_hash(CKPT)

    # ── run_config ────────────────────────────────────────────────────────────
    run_cfg = {
        "checkpoint":         str(CKPT),
        "checkpoint_name":    CKPT.name,
        "checkpoint_sha256":  ckpt_sha,
        "eval_set": {
            "hf_dataset":    HF_DATASET,
            "split":         "validation",
            "n_per_class":   N_PER_CLASS,
            "attack_systems": attack_sids,
            "dataset_hash":  ds_hash,
        },
        "ablation_spec": {
            "layer":            ABLATION_LAYER,
            "mode":             ABLATION_MODE,
            "critical_heads":   sorted(CRITICAL_HEADS),
            "n_possible_pairs": n_possible,
            "exhaustive":       True,
            "sampled_pairs":    [sorted(p) for p in all_pairs],
        },
        "seed":                SEED,
        "n_bootstrap":         N_BOOTSTRAP,
        "nh":                  NH,
        "sanity": {
            "a_no_critical_pair_sampled": True,
            "b_baseline_parity":          sanity_b_pass,
            "c_delta_eer_not_all_identical": sanity_c_pass,
        },
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    _dump(per_pair,    OUT_DIR / "per_pair_results.json")
    _dump(null_summary, OUT_DIR / "null_distribution_summary.json")
    _dump(comparison,   OUT_DIR / "comparison_to_h0h4.json")
    _dump(run_cfg,      OUT_DIR / "run_config.json")
    print(f"\n  Saved 4 JSON files to: {OUT_DIR}/")

    # ── Final print ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Results: {OUT_DIR}/")
    print()

    median_delta = null_summary["ablate_pair"]["median"]
    h0h4_delta   = ref["ablate_h0h4_pooled_delta"]
    comp_a        = comparison["ablate_direction"]
    n_worse       = int((np.array(delta_ablate_vals) > h0h4_delta).sum())
    n_equal       = int((np.array(delta_ablate_vals) == h0h4_delta).sum())

    print(f"Null distribution (ablate-pair direction), {len(all_pairs)} pairs:")
    for r in per_pair:
        p   = r["pair"]
        d   = r["ablate_pair"]["delta_eer"]
        tag = "  ← {h0,h4} baseline" if frozenset(p) == CRITICAL_HEADS else ""
        print(f"  pair {p}: pooled EER={r['ablate_pair']['pooled_eer']:.4f}  "
              f"Δ={d:+.4f}{tag}")
    print()
    print(f"Null distribution summary (ablate direction):")
    print(f"  median ΔEER = {median_delta:+.4f}")
    print(f"  mean   ΔEER = {null_summary['ablate_pair']['mean']:+.4f}")
    print(f"  range        [{null_summary['ablate_pair']['min']:+.4f}, "
          f"{null_summary['ablate_pair']['max']:+.4f}]")
    print()
    print(f"{{h0,h4}} ablate ΔEER = {h0h4_delta:+.4f}  "
          f"(from e4_goat_baseline, baseline={ref['baseline_pooled_eer']:.4f})")
    print(f"  {n_worse}/{len(all_pairs)} null pairs hurt more (ΔEER > {h0h4_delta:+.4f})")
    print(f"  {n_equal}/{len(all_pairs)} null pairs tied (ΔEER = {h0h4_delta:+.4f})")
    print(f"  empirical percentile = {comp_a['empirical_percentile']:.1f}%")
    print(f"  p (H1: h0,h4 ablation more damaging than null) = "
          f"{comp_a['p_one_sided_above']:.4f}")
    print(f"  p (H1: h0,h4 ablation less damaging than null) = "
          f"{comp_a['p_one_sided_below']:.4f}")
    print()
    comp_k = comparison["keep_direction"]
    print(f"{{h0,h4}} keep-only ΔEER = {ref['keep_h0h4_pooled_delta']:+.4f}")
    print(f"  empirical percentile = {comp_k['empirical_percentile']:.1f}%")
    print(f"  p (H1: keep-only more damaging than null) = "
          f"{comp_k['p_one_sided_above']:.4f}")
    print()
    all_pass = sanity_b_pass and sanity_c_pass
    print(f"Sanity: {'PASS' if all_pass else 'FAIL — review warnings above'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
