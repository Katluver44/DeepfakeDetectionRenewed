#!/usr/bin/env python3
"""
mlaad_e4_ablation.py
====================
E4-style head ablation on MLAAD checkpoints using the target heads discovered
in Phase 2 (mlaad_head_discovery.py).

Conditions (all GAT layer 0, "zero" mode matching E4):
  baseline           — no ablation
  ablate_top1        — zero h2 only (top-ranked head from Phase 2)
  ablate_top2        — zero h4 only (second-ranked head from Phase 2)
  ablate_h2_h4       — zero {h2, h4} (both target heads)
  random_single_ctrl — random single head chosen ∉ {h2, h4}, seed=42
  random_pair_ctrl   — random pair ≠ {h2, h4}, seed=42

Both checkpoints (mlaad_robust_goat, mlaad_goat) are evaluated using
robust_goat's target heads on both test sets.

Exhaustive null distribution run on mlaad_goat only: all C(6,2)-1 = 14
non-{h2,h4} pairs (ablate_pair and ablate_compl directions, in-dist only).

Usage:
    python experiments/scripts/mlaad_e4_ablation.py

Reads:
    experiments/results/mlaad/head_discovery/target_heads.json
    experiments/results/mlaad/baseline_eval/test_in_distribution.json
    experiments/results/mlaad/baseline_eval/test_cross_language.json
    experiments/checkpoints/mlaad_{goat,robust_goat}-best-*.ckpt

Writes:
    experiments/results/mlaad/e4_ablation/
        mlaad_robust_e4/{in_distribution,cross_language}/
            per_attack_per_condition.csv
            pooled_summary.json
        mlaad_goat_e4/{in_distribution,cross_language}/
            per_attack_per_condition.csv
            pooled_summary.json
        mlaad_goat_null_distribution/
            per_pair_results.json
            null_distribution_summary.json
        comparison_to_null/
            empirical_percentile.json
        cross_language_delta.json
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
from torch.utils.data import Dataset, DataLoader

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
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
CKPT_DIR      = EXP_DIR / "checkpoints"
HEAD_DISC_DIR = EXP_DIR / "results" / "mlaad" / "head_discovery"
BASELINE_DIR  = EXP_DIR / "results" / "mlaad" / "baseline_eval"
OUT_BASE      = EXP_DIR / "results" / "mlaad" / "e4_ablation"

for _p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import ablation utilities ─────────────────────────────────────────────────
import head_ablation as ha
import gat_e3_e4 as e4lib

# ── Constants ─────────────────────────────────────────────────────────────────
SEED           = 42
N_BOOTSTRAP    = 200
ABLATION_MODE  = "zero"
ABLATION_LAYER = 0
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
TARGET_SR      = 16_000


# ── JSON helper ───────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, np.bool_):         return bool(obj)
        if isinstance(obj, np.ndarray):       return obj.tolist()
        if isinstance(obj, (frozenset, set)): return sorted(obj)
        return super().default(obj)


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _best_or_last(stem: str) -> Path:
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else CKPT_DIR / f"{stem}.ckpt"


# ── MLAAD Dataset ─────────────────────────────────────────────────────────────

class MAALDAblationDataset(Dataset):
    """
    Wraps MLAAD test records for use with ha.run_eval.
    Returns batches compatible with ha.collate:
        audio: (1, 48000), label: long, system_id: "-" | attack_system
    """
    def __init__(self, records: list[dict], processed_dir: Path) -> None:
        self.records      = records
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(self.processed_dir / rec["audio_path"])  # (48000,)
        wav = wav.unsqueeze(0)                                      # (1, 48000)
        y   = 0 if rec["label"] == "bonafide" else 1
        sid = "-" if rec["label"] == "bonafide" else rec.get("attack_system", "unknown")
        return {
            "audio":     wav,
            "label":     torch.tensor(y, dtype=torch.long),
            "system_id": sid,
        }


def make_loader(records: list[dict], processed_dir: Path) -> DataLoader:
    ds = MAALDAblationDataset(records, processed_dir)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, collate_fn=ha.collate)


# ── Ablation helpers ──────────────────────────────────────────────────────────

def run_condition(lit, loader, device, heads: frozenset) -> list[dict]:
    return ha.run_eval(lit, loader, device, heads, ABLATION_MODE, bonafide_means={})


def _fast_bootstrap_eer_ci(
    records: list[dict],
    attack_sids: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """
    Vectorised bootstrap EER CIs using pre-extracted numpy arrays.
    ~10x faster than e4lib.bootstrap_eer_ci for large attack_sids lists.
    """
    bon_labels  = np.array([r["label"]  for r in records if r["system_id"] == "-"], dtype=np.int8)
    bon_logits  = np.array([r["logit"]  for r in records if r["system_id"] == "-"], dtype=np.float32)
    bon_scores  = 1.0 / (1.0 + np.exp(-bon_logits.astype(np.float64)))
    n_bon       = len(bon_labels)

    result: dict[str, tuple[float, float]] = {}
    for sid in attack_sids:
        atk_logits = np.array([r["logit"]  for r in records if r["system_id"] == sid], dtype=np.float32)
        atk_scores = 1.0 / (1.0 + np.exp(-atk_logits.astype(np.float64)))
        n_atk      = len(atk_logits)
        combined_scores = np.concatenate([bon_scores, atk_scores])
        combined_labels = np.concatenate([np.zeros(n_bon, dtype=np.int8),
                                          np.ones(n_atk,  dtype=np.int8)])
        n = len(combined_labels)
        boot_eers = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_eers.append(ha.compute_eer(combined_labels[idx], combined_scores[idx]))
        if boot_eers:
            result[sid] = (float(np.percentile(boot_eers, 2.5)),
                           float(np.percentile(boot_eers, 97.5)))
        else:
            result[sid] = (float("nan"), float("nan"))
    return result


def pooled_eer(records: list[dict]) -> float:
    labels = np.array([r["label"]  for r in records])
    scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in records])))
    return float(ha.compute_eer(labels, scores))


def pick_random_single(nh: int, exclude: frozenset, seed: int) -> frozenset:
    """Pick one head index not in exclude, seeded."""
    pool = [h for h in range(nh) if h not in exclude]
    rng  = random.Random(seed)
    rng.shuffle(pool)
    return frozenset({pool[0]})


def pick_random_pair(nh: int, critical: frozenset, seed: int) -> frozenset:
    """Pick a random pair of size 2 ≠ critical, seeded."""
    all_pairs = [frozenset(c) for c in combinations(range(nh), 2)
                 if frozenset(c) != critical]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[0]


# ── CSV writer ────────────────────────────────────────────────────────────────

COND_NAMES = [
    "baseline",
    "ablate_top1",
    "ablate_top2",
    "ablate_h2_h4",
    "random_single_ctrl",
    "random_pair_ctrl",
]


def write_per_attack_csv(
    attack_sids: list[str],
    pooled_eers: dict[str, float],
    eer_by_cond: dict[str, dict[str, float]],
    ci_by_cond:  dict[str, dict[str, tuple]],
    path: Path,
) -> None:
    fieldnames = ["system"]
    for c in COND_NAMES:
        fieldnames += [f"eer_{c}", f"delta_{c}",
                       f"ci_{c}_lo", f"ci_{c}_hi"]

    rows = []
    base = pooled_eers["baseline"]
    for sid in sorted(attack_sids):
        row = {"system": sid}
        for c in COND_NAMES:
            e  = eer_by_cond[c][sid]
            ci = ci_by_cond[c].get(sid, (float("nan"), float("nan")))
            row[f"eer_{c}"]   = round(e, 6)
            row[f"delta_{c}"] = round(e - eer_by_cond["baseline"][sid], 6)
            row[f"ci_{c}_lo"] = round(ci[0], 6)
            row[f"ci_{c}_hi"] = round(ci[1], 6)
        rows.append(row)

    # Pooled row
    _nan = float("nan")
    pooled_row = {"system": "pooled"}
    for c in COND_NAMES:
        pooled_row[f"eer_{c}"]   = round(pooled_eers[c], 6)
        pooled_row[f"delta_{c}"] = round(pooled_eers[c] - base, 6)
        pooled_row[f"ci_{c}_lo"] = _nan
        pooled_row[f"ci_{c}_hi"] = _nan
    rows.append(pooled_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Null distribution helpers (from e4_goat_random_null.py) ──────────────────

def distribution_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "n":      len(a),
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a, ddof=1) if len(a) > 1 else 0.0),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "percentiles": {str(p): float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)},
    }


def compare_to_null(observed: float, null_values: list[float], label: str) -> dict:
    n    = len(null_values)
    a    = np.array(null_values)
    rank = int((a < observed).sum())
    pct  = rank / n * 100.0
    return {
        "label":                label,
        "observed":             float(observed),
        "n_null":               n,
        "null_values":          [float(v) for v in null_values],
        "rank_from_bottom":     rank,
        "empirical_percentile": round(pct, 1),
        "p_one_sided_above":    round(float((a >= observed).sum()) / n, 4),
        "p_one_sided_below":    round(float((a <= observed).sum()) / n, 4),
        "interpretation": (
            f"{rank}/{n} null pairs have ΔEER < {observed:.4f}; "
            f"{int((a > observed).sum())}/{n} have ΔEER > {observed:.4f}"
        ),
    }


# ── Core ablation runner ──────────────────────────────────────────────────────

def run_ablation_for_checkpoint(
    ckpt_name: str,
    ckpt_path: Path,
    conditions: list[tuple[str, frozenset]],
    loader_in:  DataLoader,
    loader_xl:  DataLoader,
    attack_sids_in:  list[str],
    attack_sids_xl:  list[str],
    out_dir:    Path,
    device:     torch.device,
) -> dict:
    """
    Load checkpoint, run all conditions on both loaders, write CSVs + pooled JSON.
    Returns {in_distribution: {pooled_eers, ...}, cross_language: {...}}.
    """
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_name}  ({ckpt_path.name})")

    ha.patch_phoneme_loader()
    lit = ha.load_model(ckpt_path, device)
    print(f"  Loaded.  GAT layer 0 heads: {lit.model.GAT.gat_net[0].num_of_heads}")

    # Determinism sanity
    _batch = next(iter(loader_in))
    _audio = _batch["audio"].to(device)
    _num_f = torch.full((_audio.shape[0],), ha.NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        _hs, _pids = ha.run_frozen_frontend(_audio, lit.model, device)
        _lg1 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
        _lg2 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
    _diff = (_lg1 - _lg2).abs().max().item()
    status = "PASS" if _diff < 1e-4 else "FAIL"
    print(f"  Determinism: {status}  (max logit diff={_diff:.2e})")
    ha.sanity_layer_untouched(lit)

    results = {}
    for split_name, loader, attack_sids in [
        ("in_distribution", loader_in, attack_sids_in),
        ("cross_language",  loader_xl, attack_sids_xl),
    ]:
        print(f"\n  Split: {split_name}")
        all_records:  dict[str, list[dict]] = {}
        pooled_eers_s: dict[str, float]    = {}

        for cond_name, heads in conditions:
            t0   = time.time()
            recs = run_condition(lit, loader, device, heads)
            all_records[cond_name]   = recs
            pooled_eers_s[cond_name] = pooled_eer(recs)
            print(f"    {cond_name:22s}  ablate={sorted(heads)}  "
                  f"pooled EER={pooled_eers_s[cond_name]:.4f}  ({time.time()-t0:.1f}s)")

        # Per-system EERs + bootstrap CIs
        rng = np.random.default_rng(SEED)
        eer_by_cond: dict[str, dict[str, float]] = {}
        ci_by_cond:  dict[str, dict[str, tuple]] = {}
        print(f"    Computing bootstrap CIs ({N_BOOTSTRAP} resamples, fast)...")
        for cond_name, _ in conditions:
            eer_by_cond[cond_name] = e4lib.per_system_eer(
                all_records[cond_name], attack_sids)
            ci_by_cond[cond_name]  = _fast_bootstrap_eer_ci(
                all_records[cond_name], attack_sids, N_BOOTSTRAP, rng)

        # Sanity: pipeline parity — baseline EER should not be degenerate
        base_eer = pooled_eers_s["baseline"]
        parity_ok = base_eer < 0.50
        print(f"    Sanity baseline EER={base_eer:.4f}: "
              f"{'PASS (<0.50)' if parity_ok else 'FAIL (>=0.50)'}")

        # Sanity: ablate ≠ random ctrl (coarse check)
        diff_a_vs_ctrl = abs(pooled_eers_s["ablate_h2_h4"]
                             - pooled_eers_s["random_pair_ctrl"])
        print(f"    |ablate_h2_h4 - random_pair_ctrl| = {diff_a_vs_ctrl:.4f}")

        # Write CSV
        split_out = out_dir / split_name
        write_per_attack_csv(
            attack_sids, pooled_eers_s, eer_by_cond, ci_by_cond,
            split_out / "per_attack_per_condition.csv",
        )

        # Write pooled summary
        base = pooled_eers_s["baseline"]
        pooled_summary = {
            "checkpoint":    ckpt_name,
            "split":         split_name,
            "n_samples":     len(loader.dataset),
            "n_attack_sids": len(attack_sids),
            "pooled_eers":   {c: round(pooled_eers_s[c], 6) for c in COND_NAMES},
            "pooled_deltas": {c: round(pooled_eers_s[c] - base, 6)
                              for c in COND_NAMES},
            "sanity": {
                "baseline_eer_ok":      parity_ok,
                "ablate_vs_ctrl_diff":  round(diff_a_vs_ctrl, 4),
            },
            "pipeline_note": (
                "run_eval uses run_frozen_frontend (no _mask_hidden_states) for "
                "determinism. Baseline EER will be higher than the reported value from "
                "eval_mlaad_baseline.py (~0.07-0.10 absolute difference). "
                "Ablation ΔEER is still valid since all conditions use the same pipeline."
            ),
        }
        _dump(pooled_summary, split_out / "pooled_summary.json")
        print(f"    Wrote: {split_out}/per_attack_per_condition.csv + pooled_summary.json")

        results[split_name] = {
            "pooled_eers": pooled_eers_s,
            "all_records": all_records,
            "attack_sids": attack_sids,
        }

    return results


# ── Exhaustive null distribution ──────────────────────────────────────────────

def run_null_distribution(
    ckpt_name:   str,
    ckpt_path:   Path,
    critical:    frozenset,
    loader_in:   DataLoader,
    attack_sids: list[str],
    baseline_pooled_eer: float,
    out_dir:     Path,
    device:      torch.device,
) -> tuple[list[dict], dict]:
    """
    Enumerate all C(NH,2)-1 pairs ≠ critical.
    For each pair run ablate_pair and ablate_compl (in-dist only).
    Returns (per_pair, null_summary).
    """
    print(f"\n{'='*70}")
    print(f"Exhaustive null distribution: {ckpt_name}")

    ha.patch_phoneme_loader()
    lit = ha.load_model(ckpt_path, device)
    NH  = lit.model.GAT.gat_net[0].num_of_heads
    print(f"  NH={NH}, critical={sorted(critical)}")

    all_pairs: list[frozenset] = sorted(
        (frozenset(c) for c in combinations(range(NH), len(critical))
         if frozenset(c) != critical),
        key=sorted,
    )
    n_possible = sum(1 for _ in combinations(range(NH), len(critical))) - 1
    assert n_possible == len(all_pairs)
    note = (f"All {len(all_pairs)} non-{set(sorted(critical))} pairs "
            f"(C({NH},{len(critical)})-1 = {n_possible}; exhaustive)")
    print(f"  {note}")
    print(f"  Pairs: {[sorted(p) for p in all_pairs]}")

    # Sanity (a): no critical pair in null
    assert all(p != critical for p in all_pairs), "SANITY FAIL: critical pair in null set"
    print("  Sanity (a): no critical pair in null — PASS")

    # Baseline pass (sanity b: parity with main run)
    print(f"\n  Baseline pass...")
    base_recs    = run_condition(lit, loader_in, device, frozenset())
    this_base    = pooled_eer(base_recs)
    parity_ok    = abs(this_base - baseline_pooled_eer) <= 0.01
    print(f"  baseline EER={this_base:.4f}  ref={baseline_pooled_eer:.4f}  "
          f"diff={abs(this_base - baseline_pooled_eer):.4f}  "
          f"parity={'PASS' if parity_ok else 'FAIL'}")

    # Run all pairs
    pair_records: dict[str, dict] = {}
    for pair in all_pairs:
        key   = "_".join(str(h) for h in sorted(pair))
        compl = frozenset(range(NH)) - pair
        print(f"\n  Pair {sorted(pair)}:")
        t0 = time.time()
        recs_ablate = run_condition(lit, loader_in, device, pair)
        t1 = time.time()
        recs_compl  = run_condition(lit, loader_in, device, compl)
        t2 = time.time()
        pa = pooled_eer(recs_ablate)
        pc = pooled_eer(recs_compl)
        print(f"    ablate EER={pa:.4f} Δ={pa-this_base:+.4f} ({t1-t0:.1f}s)  "
              f"compl EER={pc:.4f} Δ={pc-this_base:+.4f} ({t2-t1:.1f}s)")
        pair_records[key] = {
            "pair": sorted(pair), "compl": sorted(compl),
            "ablate": recs_ablate, "compl_recs": recs_compl,
        }

    # Sanity (c): Δ EERs not all identical
    delta_ablates = [pooled_eer(v["ablate"]) - this_base for v in pair_records.values()]
    vary_ok = len(set(f"{d:.6f}" for d in delta_ablates)) > 1
    print(f"\n  Sanity (c): ΔEER values vary: {'PASS' if vary_ok else 'FAIL'}")

    # Bootstrap CIs
    print(f"  Computing bootstrap CIs...")
    rng = np.random.default_rng(SEED)
    ci_ablate: dict[str, dict] = {}
    ci_compl_:  dict[str, dict] = {}
    for pair in all_pairs:
        key = "_".join(str(h) for h in sorted(pair))
        ci_ablate[key] = _fast_bootstrap_eer_ci(
            pair_records[key]["ablate"],    attack_sids, N_BOOTSTRAP, rng)
        ci_compl_[key] = _fast_bootstrap_eer_ci(
            pair_records[key]["compl_recs"], attack_sids, N_BOOTSTRAP, rng)

    # Build per_pair list
    per_pair: list[dict] = []
    for pair in all_pairs:
        key    = "_".join(str(h) for h in sorted(pair))
        compl_ = sorted(frozenset(range(NH)) - pair)
        recs_a = pair_records[key]["ablate"]
        recs_c = pair_records[key]["compl_recs"]
        p_a    = pooled_eer(recs_a)
        p_c    = pooled_eer(recs_c)
        psa    = e4lib.per_system_eer(recs_a, attack_sids)
        psc    = e4lib.per_system_eer(recs_c, attack_sids)
        per_pair.append({
            "pair":       sorted(pair),
            "complement": compl_,
            "ablate_pair": {
                "ablated_heads":  sorted(pair),
                "pooled_eer":     float(p_a),
                "delta_eer":      float(p_a - this_base),
                "per_attack_eer": {s: float(v) for s, v in psa.items()},
                "per_attack_ci":  {s: [float(ci_ablate[key][s][0]),
                                       float(ci_ablate[key][s][1])]
                                   for s in attack_sids},
            },
            "ablate_compl": {
                "ablated_heads":  compl_,
                "pooled_eer":     float(p_c),
                "delta_eer":      float(p_c - this_base),
                "per_attack_eer": {s: float(v) for s, v in psc.items()},
                "per_attack_ci":  {s: [float(ci_compl_[key][s][0]),
                                       float(ci_compl_[key][s][1])]
                                   for s in attack_sids},
            },
        })

    delta_ablate_vals = [r["ablate_pair"]["delta_eer"] for r in per_pair]
    delta_compl_vals  = [r["ablate_compl"]["delta_eer"] for r in per_pair]

    null_summary = {
        "checkpoint":            ckpt_name,
        "n_pairs":               len(all_pairs),
        "exhaustive_note":       note,
        "baseline_pooled_eer":   float(this_base),
        "ref_baseline_pooled_eer": float(baseline_pooled_eer),
        "pipeline_parity_ok":    parity_ok,
        "ablate_pair":           distribution_stats(delta_ablate_vals),
        "ablate_compl":          distribution_stats(delta_compl_vals),
        "sanity": {
            "a_no_critical_in_null": True,
            "b_pipeline_parity":     parity_ok,
            "c_delta_eer_vary":      vary_ok,
        },
    }

    _dump(per_pair,     out_dir / "per_pair_results.json")
    _dump(null_summary, out_dir / "null_distribution_summary.json")
    print(f"\n  Wrote null distribution to: {out_dir}/")

    return per_pair, null_summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load target heads ─────────────────────────────────────────────────────
    target_json = json.loads((HEAD_DISC_DIR / "target_heads.json").read_text())
    target_heads = target_json["target_heads"]   # e.g. [2, 4]
    h_top1, h_top2 = target_heads[0], target_heads[1]
    CRITICAL = frozenset(target_heads)
    print(f"\nTarget heads (from Phase 2): {target_heads}")
    print(f"  decision: {target_json['decision']}")
    print(f"  kl_range: {target_json['kl_range']}")

    # ── Load test records ─────────────────────────────────────────────────────
    in_dist_records = json.loads((BASELINE_DIR / "test_in_distribution.json").read_text())
    xl_records      = json.loads((BASELINE_DIR / "test_cross_language.json").read_text())
    print(f"\nTest records — in_distribution: {len(in_dist_records)}, "
          f"cross_language: {len(xl_records)}")

    attack_sids_in = sorted(
        {r["attack_system"] for r in in_dist_records if r["label"] == "spoof"})
    attack_sids_xl = sorted(
        {r["attack_system"] for r in xl_records if r["label"] == "spoof"})
    print(f"  attack systems in-dist: {len(attack_sids_in)}, "
          f"cross-lang: {len(attack_sids_xl)}")

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_robust = _best_or_last("mlaad_robust_goat")
    ckpt_goat   = _best_or_last("mlaad_goat")
    for p in (ckpt_robust, ckpt_goat):
        if not p.exists():
            print(f"ERROR: checkpoint not found: {p}")
            sys.exit(1)
    print(f"\nCheckpoints:")
    print(f"  mlaad_robust_goat: {ckpt_robust.name}")
    print(f"  mlaad_goat:        {ckpt_goat.name}")

    # ── Build loaders ─────────────────────────────────────────────────────────
    loader_in = make_loader(in_dist_records, PROCESSED_DIR)
    loader_xl = make_loader(xl_records,      PROCESSED_DIR)

    # ── Define conditions (using a temporary NH=6 placeholder; assert later) ──
    # NH will be read from the model; we use a placeholder NH=6 for pair picking
    NH_PLACEHOLDER = 6
    random_single = pick_random_single(NH_PLACEHOLDER, CRITICAL, SEED)
    random_pair   = pick_random_pair(NH_PLACEHOLDER, CRITICAL, SEED)

    conditions: list[tuple[str, frozenset]] = [
        ("baseline",           frozenset()),
        ("ablate_top1",        frozenset({h_top1})),
        ("ablate_top2",        frozenset({h_top2})),
        ("ablate_h2_h4",       CRITICAL),
        ("random_single_ctrl", random_single),
        ("random_pair_ctrl",   random_pair),
    ]

    print(f"\nConditions:")
    for name, heads in conditions:
        print(f"  {name:22s}  ablate={sorted(heads)}")

    # ── Run mlaad_robust_goat ─────────────────────────────────────────────────
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    robust_results = run_ablation_for_checkpoint(
        ckpt_name   = "mlaad_robust_goat",
        ckpt_path   = ckpt_robust,
        conditions  = conditions,
        loader_in   = loader_in,
        loader_xl   = loader_xl,
        attack_sids_in  = attack_sids_in,
        attack_sids_xl  = attack_sids_xl,
        out_dir     = OUT_BASE / "mlaad_robust_e4",
        device      = device,
    )

    # ── Run mlaad_goat ────────────────────────────────────────────────────────
    goat_results = run_ablation_for_checkpoint(
        ckpt_name   = "mlaad_goat",
        ckpt_path   = ckpt_goat,
        conditions  = conditions,
        loader_in   = loader_in,
        loader_xl   = loader_xl,
        attack_sids_in  = attack_sids_in,
        attack_sids_xl  = attack_sids_xl,
        out_dir     = OUT_BASE / "mlaad_goat_e4",
        device      = device,
    )

    # ── Exhaustive null on mlaad_goat (in-dist only) ──────────────────────────
    goat_base_pooled = goat_results["in_distribution"]["pooled_eers"]["baseline"]
    per_pair, null_summary = run_null_distribution(
        ckpt_name           = "mlaad_goat",
        ckpt_path           = ckpt_goat,
        critical            = CRITICAL,
        loader_in           = loader_in,
        attack_sids         = attack_sids_in,
        baseline_pooled_eer = goat_base_pooled,
        out_dir             = OUT_BASE / "mlaad_goat_null_distribution",
        device              = device,
    )

    # ── Comparison to null ────────────────────────────────────────────────────
    goat_ablate_delta = (goat_results["in_distribution"]["pooled_eers"]["ablate_h2_h4"]
                         - goat_results["in_distribution"]["pooled_eers"]["baseline"])
    robust_ablate_delta = (robust_results["in_distribution"]["pooled_eers"]["ablate_h2_h4"]
                           - robust_results["in_distribution"]["pooled_eers"]["baseline"])

    delta_ablate_vals = [r["ablate_pair"]["delta_eer"] for r in per_pair]
    delta_compl_vals  = [r["ablate_compl"]["delta_eer"] for r in per_pair]

    empirical = {
        "critical_heads":    sorted(CRITICAL),
        "checkpoint_for_null": "mlaad_goat",
        "mlaad_goat": {
            "ablate_h2_h4_delta_eer":  round(goat_ablate_delta, 6),
            "ablate_direction":        compare_to_null(
                goat_ablate_delta, delta_ablate_vals, f"ablate {{{sorted(CRITICAL)}}}"),
        },
        "mlaad_robust_goat": {
            "ablate_h2_h4_delta_eer":  round(robust_ablate_delta, 6),
        },
    }
    _dump(empirical, OUT_BASE / "comparison_to_null" / "empirical_percentile.json")

    # ── Cross-language delta analysis ─────────────────────────────────────────
    xl_delta: dict = {"critical_heads": sorted(CRITICAL), "checkpoints": {}}
    for ckpt_name, res in [("mlaad_robust_goat", robust_results),
                            ("mlaad_goat",         goat_results)]:
        in_e = res["in_distribution"]["pooled_eers"]
        xl_e = res["cross_language"]["pooled_eers"]
        xl_delta["checkpoints"][ckpt_name] = {
            c: {
                "in_dist_eer":         round(in_e[c], 4),
                "cross_lang_eer":      round(xl_e[c], 4),
                "cross_lang_gap":      round(xl_e[c] - in_e[c], 4),
                "gap_vs_baseline":     round((xl_e[c] - in_e[c])
                                             - (xl_e["baseline"] - in_e["baseline"]), 4),
            }
            for c in COND_NAMES
        }
    _dump(xl_delta, OUT_BASE / "cross_language_delta.json")
    print(f"\nWrote: {OUT_BASE}/comparison_to_null/empirical_percentile.json")
    print(f"Wrote: {OUT_BASE}/cross_language_delta.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MLAAD E4 ABLATION SUMMARY")
    print(f"  Target heads: {sorted(CRITICAL)}  (from mlaad_robust_goat Phase 2)")
    print(f"  Mode: {ABLATION_MODE}  Layer: {ABLATION_LAYER}")
    print()
    for ckpt_name, res in [("mlaad_robust_goat", robust_results),
                            ("mlaad_goat",         goat_results)]:
        print(f"  {ckpt_name}:")
        for split in ("in_distribution", "cross_language"):
            pe = res[split]["pooled_eers"]
            b  = pe["baseline"]
            print(f"    {split:20s} "
                  f"baseline={b:.4f}  "
                  f"ablate_h2_h4={pe['ablate_h2_h4']:.4f} (Δ={pe['ablate_h2_h4']-b:+.4f})  "
                  f"random_pair={pe['random_pair_ctrl']:.4f} (Δ={pe['random_pair_ctrl']-b:+.4f})")

    ns = null_summary
    print(f"\n  Null distribution (mlaad_goat, in-dist, ablate direction):")
    print(f"    N pairs: {ns['n_pairs']}  "
          f"median ΔEER={ns['ablate_pair']['median']:+.4f}  "
          f"range=[{ns['ablate_pair']['min']:+.4f}, {ns['ablate_pair']['max']:+.4f}]")
    print(f"    {{h2,h4}} ΔEER = {goat_ablate_delta:+.4f}  "
          f"(empirical percentile = "
          f"{empirical['mlaad_goat']['ablate_direction']['empirical_percentile']:.1f}%)")
    print()
    all_sanity = (ns["sanity"]["a_no_critical_in_null"]
                  and ns["sanity"]["b_pipeline_parity"]
                  and ns["sanity"]["c_delta_eer_vary"])
    print(f"  Sanity: {'ALL PASS' if all_sanity else 'ISSUES — review above'}")
    print(f"  Output: {OUT_BASE}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
