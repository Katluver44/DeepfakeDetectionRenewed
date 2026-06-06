#!/usr/bin/env python3
"""
e4_goat_baseline.py
====================
E4-style per-system EER under GAT layer-0 attention ablation, evaluated on
goat.ckpt (non-robust baseline model).

Ablation conditions (layer 0 only, zero mode — matching original E4):
  baseline      — no ablation
  ablate_h0_h4  — zero heads {h0, h4} at layer 0
  keep_h0_h4    — zero complement heads (all except h0, h4) at layer 0
  random_ctrl   — zero a randomly chosen pair of equal size (seeded, ≠ {h0,h4})

Imports ablation utilities from head_ablation.py and eval pipeline from
gat_e3_e4.py. No logic duplication.

Outputs → experiments/results/gat_l0_attention_followups/e4_goat_baseline/
"""
from __future__ import annotations

import csv
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

# ── torch.load compat (must precede all torch.load calls) ────────────────────
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
EXP_DIR   = Path(__file__).resolve().parent.parent   # experiments/
REPO_ROOT = EXP_DIR.parent                            # repo root
OUT_DIR   = EXP_DIR / "results" / "gat_l0_attention_followups" / "e4_goat_baseline"
CKPT      = REPO_ROOT / "models" / "goat.ckpt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import existing utilities (no logic duplication) ─────────────────────────
import head_ablation as ha      # install_ablation, load_model, run_eval, …
import gat_e3_e4 as e4lib       # bootstrap_eer_ci, per_system_eer

# ── Constants ─────────────────────────────────────────────────────────────────
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
CRITICAL_HEADS = frozenset({0, 4})
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42
N_BOOTSTRAP    = 1_000
ABLATION_MODE  = "zero"   # matches original E4 (gat_e3_e4.py)
ABLATION_LAYER = 0        # layer-0 only


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_random_ctrl_heads(nh: int, critical: frozenset, seed: int) -> frozenset:
    """Return a randomly chosen pair (size == len(critical), ≠ critical), seeded."""
    k         = len(critical)
    all_pairs = [frozenset(c) for c in combinations(range(nh), k)
                 if frozenset(c) != critical]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[0]


def eval_set_hash(dataset: ha.BalancedDataset) -> str:
    key = json.dumps(sorted(zip(dataset.indices, dataset.sys_ids)))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def pooled_eer(records: list[dict]) -> float:
    labels = np.array([r["label"] for r in records])
    scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in records])))
    return ha.compute_eer(labels, scores)


def run_condition(lit, loader, device, heads: frozenset) -> list[dict]:
    """Zero-mode ablation on layer 0 only (baseline = empty heads = no-op)."""
    return ha.run_eval(lit, loader, device, heads, ABLATION_MODE, bonafide_means={})


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ha.patch_phoneme_loader()
    print(f"Loading model: {CKPT.name}")
    lit = ha.load_model(CKPT, device)
    print("  model loaded")

    # ── Read NH from model (not hardcoded) ────────────────────────────────────
    gat_net = lit.model.GAT.gat_net
    NH = gat_net[0].num_of_heads
    print(f"  GAT layer 0: {NH} heads")

    # ── Define conditions ─────────────────────────────────────────────────────
    complement_heads  = frozenset(range(NH)) - CRITICAL_HEADS
    random_ctrl_heads = pick_random_ctrl_heads(NH, CRITICAL_HEADS, SEED)

    print(f"  critical heads:    {sorted(CRITICAL_HEADS)}")
    print(f"  complement heads:  {sorted(complement_heads)}")
    print(f"  random ctrl heads: {sorted(random_ctrl_heads)}")

    conditions: list[tuple[str, frozenset]] = [
        ("baseline",     frozenset()),
        ("ablate_h0_h4", CRITICAL_HEADS),
        ("keep_h0_h4",   complement_heads),   # ablate complement → only h0,h4 active
        ("random_ctrl",  random_ctrl_heads),
    ]

    # ── Build dataloader (same split/preprocessing as original E4) ───────────
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
    print(f"  attack systems: {attack_sids}")

    # ── Sanity pre-checks ─────────────────────────────────────────────────────
    print("\n--- Sanity: baseline determinism ---")
    ha.sanity_baseline_identity(lit, loader, device)
    ha.sanity_layer_untouched(lit)

    # ── Run all conditions ────────────────────────────────────────────────────
    all_records: dict[str, list[dict]] = {}
    pooled_eers: dict[str, float]      = {}

    for cond_name, heads in conditions:
        print(f"\n  Condition: {cond_name}  ablate={sorted(heads)}")
        t0   = time.time()
        recs = run_condition(lit, loader, device, heads)
        all_records[cond_name] = recs
        p_eer = pooled_eer(recs)
        pooled_eers[cond_name] = p_eer
        print(f"    Done ({time.time()-t0:.1f}s)  pooled EER={p_eer:.4f}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n--- Sanity checks ---")
    sanity_results: dict[str, str] = {}

    # (a) baseline EER not degenerate
    base_eer = pooled_eers["baseline"]
    if base_eer > 0.5:
        msg = f"WARN — pooled EER={base_eer:.4f} > 0.5; model may be broken or inverted"
        print(f"  (a) {msg}")
    else:
        msg = f"OK — pooled EER={base_eer:.4f}; verify against goat.ckpt reported eval EER"
        print(f"  (a) {msg}")
    sanity_results["baseline_eer"] = msg

    # (c) ablate_h0_h4 ≠ keep_h0_h4 (numerically identical would indicate a bug)
    diff_c = abs(pooled_eers["ablate_h0_h4"] - pooled_eers["keep_h0_h4"])
    if diff_c < 1e-6:
        msg = f"FAIL — ablate_h0_h4 and keep_h0_h4 have identical pooled EER={pooled_eers['ablate_h0_h4']:.4f}; bug suspected"
        print(f"  (c) {msg}")
    else:
        msg = (f"OK — ablate_h0_h4={pooled_eers['ablate_h0_h4']:.4f} ≠ "
               f"keep_h0_h4={pooled_eers['keep_h0_h4']:.4f} (diff={diff_c:.4f})")
        print(f"  (c) {msg}")
    sanity_results["ablate_ne_keep"] = msg

    # ── Per-system EER + bootstrap CIs ───────────────────────────────────────
    print("\n--- Per-system EER + bootstrap CIs ---")
    eer_by_cond: dict[str, dict[str, float]] = {}
    ci_by_cond:  dict[str, dict[str, tuple]] = {}
    rng = np.random.default_rng(SEED)
    for cond_name, _ in conditions:
        eer_by_cond[cond_name] = e4lib.per_system_eer(all_records[cond_name], attack_sids)
        ci_by_cond[cond_name]  = e4lib.bootstrap_eer_ci(
            all_records[cond_name], attack_sids, n_bootstrap=N_BOOTSTRAP, rng=rng)

    # ── Sanity (b): random_ctrl CI overlaps baseline CI ───────────────────────
    # Average per-system CI bounds (coarse check — per-system CIs are in the CSV)
    base_lo = float(np.mean([ci_by_cond["baseline"][s][0] for s in attack_sids]))
    base_hi = float(np.mean([ci_by_cond["baseline"][s][1] for s in attack_sids]))
    ctrl_lo = float(np.mean([ci_by_cond["random_ctrl"][s][0] for s in attack_sids]))
    ctrl_hi = float(np.mean([ci_by_cond["random_ctrl"][s][1] for s in attack_sids]))
    ci_overlap = base_lo <= ctrl_hi and ctrl_lo <= base_hi
    if ci_overlap:
        msg = f"OK — random_ctrl avg CI [{ctrl_lo:.3f},{ctrl_hi:.3f}] overlaps baseline [{base_lo:.3f},{base_hi:.3f}]"
        print(f"  (b) {msg}")
    else:
        msg = f"WARN — random_ctrl avg CI [{ctrl_lo:.3f},{ctrl_hi:.3f}] does NOT overlap baseline [{base_lo:.3f},{base_hi:.3f}]"
        print(f"  (b) {msg}")
    sanity_results["random_ctrl_ci_overlap"] = msg

    sanity_pass = all(not s.startswith("FAIL") and not s.startswith("WARN")
                      for s in sanity_results.values())

    # ── Print per-system table ────────────────────────────────────────────────
    print(f"\n{'system':8s}  {'baseline':>10s}  {'ablate_h0_h4':>14s}  {'keep_h0_h4':>12s}  {'random_ctrl':>12s}")
    print("-" * 66)
    table_rows: list[dict] = []
    for sid in sorted(attack_sids):
        b   = eer_by_cond["baseline"][sid]
        aa  = eer_by_cond["ablate_h0_h4"][sid]
        ko  = eer_by_cond["keep_h0_h4"][sid]
        rc  = eer_by_cond["random_ctrl"][sid]
        ci_b  = ci_by_cond["baseline"][sid]
        ci_aa = ci_by_cond["ablate_h0_h4"][sid]
        ci_ko = ci_by_cond["keep_h0_h4"][sid]
        ci_rc = ci_by_cond["random_ctrl"][sid]
        print(f"  {sid:6s}  "
              f"{b:.4f}[{ci_b[0]:.3f},{ci_b[1]:.3f}]  "
              f"{aa:.4f}[{ci_aa[0]:.3f},{ci_aa[1]:.3f}]  "
              f"{ko:.4f}[{ci_ko[0]:.3f},{ci_ko[1]:.3f}]  "
              f"{rc:.4f}[{ci_rc[0]:.3f},{ci_rc[1]:.3f}]")
        table_rows.append({
            "system":               sid,
            "eer_baseline":         b,
            "eer_ablate_h0_h4":     aa,
            "eer_keep_h0_h4":       ko,
            "eer_random_ctrl":      rc,
            "delta_ablate_h0_h4":   aa - b,
            "delta_keep_h0_h4":     ko - b,
            "delta_random_ctrl":    rc - b,
            "ci_baseline_lo":       ci_b[0],   "ci_baseline_hi":       ci_b[1],
            "ci_ablate_h0_h4_lo":   ci_aa[0],  "ci_ablate_h0_h4_hi":   ci_aa[1],
            "ci_keep_h0_h4_lo":     ci_ko[0],  "ci_keep_h0_h4_hi":     ci_ko[1],
            "ci_random_ctrl_lo":    ci_rc[0],  "ci_random_ctrl_hi":    ci_rc[1],
        })

    # Pooled row (CIs not computed — use nan as sentinel)
    _nan = float("nan")
    table_rows.append({
        "system":               "pooled",
        "eer_baseline":         pooled_eers["baseline"],
        "eer_ablate_h0_h4":     pooled_eers["ablate_h0_h4"],
        "eer_keep_h0_h4":       pooled_eers["keep_h0_h4"],
        "eer_random_ctrl":      pooled_eers["random_ctrl"],
        "delta_ablate_h0_h4":   pooled_eers["ablate_h0_h4"] - pooled_eers["baseline"],
        "delta_keep_h0_h4":     pooled_eers["keep_h0_h4"]   - pooled_eers["baseline"],
        "delta_random_ctrl":    pooled_eers["random_ctrl"]  - pooled_eers["baseline"],
        "ci_baseline_lo":       _nan, "ci_baseline_hi":       _nan,
        "ci_ablate_h0_h4_lo":   _nan, "ci_ablate_h0_h4_hi":   _nan,
        "ci_keep_h0_h4_lo":     _nan, "ci_keep_h0_h4_hi":     _nan,
        "ci_random_ctrl_lo":    _nan, "ci_random_ctrl_hi":    _nan,
    })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "e4_goat_baseline_per_system_eer.csv"
    write_csv(table_rows, csv_path)

    # ── Save run_config.json ──────────────────────────────────────────────────
    run_config = {
        "checkpoint":      str(CKPT),
        "checkpoint_name": CKPT.name,
        "ablation_spec": {
            "layer":             ABLATION_LAYER,
            "mode":              ABLATION_MODE,
            "critical_heads":    sorted(CRITICAL_HEADS),
            "complement_heads":  sorted(complement_heads),
            "random_ctrl_heads": sorted(random_ctrl_heads),
        },
        "eval_set": {
            "hf_dataset":    HF_DATASET,
            "split":         "validation",
            "n_per_class":   N_PER_CLASS,
            "attack_systems": attack_sids,
            "dataset_hash":  ds_hash,
        },
        "seed":                          SEED,
        "n_bootstrap":                   N_BOOTSTRAP,
        "nh":                            NH,
        "sanity_pass":                   sanity_pass,
        "ci_overlap_random_ctrl_baseline": ci_overlap,
        "sanity_details":                sanity_results,
    }
    cfg_path = OUT_DIR / "run_config.json"
    cfg_path.write_text(json.dumps(run_config, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────────
    b  = pooled_eers["baseline"]
    aa = pooled_eers["ablate_h0_h4"]
    ko = pooled_eers["keep_h0_h4"]
    rc = pooled_eers["random_ctrl"]

    print(f"\n{'='*65}")
    print(f"Results: {OUT_DIR}/")
    print(f"  {csv_path.name}")
    print(f"  {cfg_path.name}")
    print(f"\nPooled EER — baseline: {b:.4f} | ablate_h0_h4: {aa:.4f} (Δ{aa-b:+.4f}) | "
          f"keep_h0_h4: {ko:.4f} (Δ{ko-b:+.4f}) | random_ctrl: {rc:.4f} (Δ{rc-b:+.4f})")
    print(f"Sanity: {'PASS' if sanity_pass else 'FAIL — review warnings above'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
