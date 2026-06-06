#!/usr/bin/env python3
"""
e4_goat_topheads_ablation.py
=============================
E4-style per-system EER ablation using goat.ckpt's own top-2 heads as identified
by goat_head_discovery.py (highest mean KL divergence heads).

Reads the top-2 pair from goat_head_discovery/head_ranking.json — never hardcoded.

Conditions (layer 0 only, zero mode):
  baseline         — no ablation
  ablate_topheads  — zero the top-2 pair
  keep_topheads    — zero the complement (keep only top-2 active)
  random_ctrl      — zero a randomly chosen pair of equal size (seeded, ≠ top-2)

Output schema matches e4_goat_baseline exactly (same column names, same metric
names, same JSON structure) so the two CSVs are directly comparable by `system`.

Outputs → experiments/results/gat_l0_attention_followups/e4_goat_topheads_ablation/
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
EXP_DIR   = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent
OUT_DIR   = EXP_DIR / "results" / "gat_l0_attention_followups" / "e4_goat_topheads_ablation"
DISCOVERY = EXP_DIR / "results" / "gat_l0_attention_followups" / "goat_head_discovery" / \
            "head_ranking.json"
BASELINE_CSV = EXP_DIR / "results" / "gat_l0_attention_followups" / "e4_goat_baseline" / \
               "e4_goat_baseline_per_system_eer.csv"
CKPT      = REPO_ROOT / "models" / "goat.ckpt"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import existing utilities (no logic duplication) ─────────────────────────
import head_ablation as ha
import gat_e3_e4 as e4lib

# ── Constants (must match e4_goat_baseline exactly) ───────────────────────────
HF_DATASET   = "Bisher/ASVspoof_2019_LA"
CACHE_DIR    = REPO_ROOT / "data" / "asvspoof_2019_la"
N_PER_CLASS  = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8))
SEED         = 42
N_BOOTSTRAP  = 1_000
ABLATION_MODE  = "zero"
ABLATION_LAYER = 0
BASELINE_EER_TOLERANCE = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_random_ctrl(nh: int, top2: frozenset, seed: int) -> frozenset:
    k         = len(top2)
    all_pairs = [frozenset(c) for c in combinations(range(nh), k)
                 if frozenset(c) != top2]
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
    return ha.run_eval(lit, loader, device, heads, ABLATION_MODE, bonafide_means={})


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_baseline_pooled_eer(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    import csv as _csv
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            if row["system"] == "pooled":
                return float(row["eer_baseline"])
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    # ── Read top-2 from discovery output (never hardcoded) ───────────────────
    if not DISCOVERY.exists():
        print(f"ERROR: head_ranking.json not found: {DISCOVERY}")
        print("Run experiments/scripts/goat_head_discovery.py first.")
        sys.exit(1)
    disc = json.loads(DISCOVERY.read_text())
    top2_heads = frozenset(disc["top2_heads"])
    top2_same_as_robust = disc.get("top2_same_as_robust_goat", False)
    head_ranking_doc = disc   # keep full record for run_config

    print(f"Top-2 heads from discovery: {sorted(top2_heads)}")
    if top2_same_as_robust:
        print("  [FLAG] Top-2 = {h0,h4} — same as robust_goat.ckpt. "
              "The random-null run is the primary comparison for this case.")

    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ha.patch_phoneme_loader()
    print(f"Loading model: {CKPT.name}")
    lit = ha.load_model(CKPT, device)
    NH  = lit.model.GAT.gat_net[0].num_of_heads
    print(f"  model loaded  (NH={NH})")

    # ── Define conditions ─────────────────────────────────────────────────────
    complement_heads  = frozenset(range(NH)) - top2_heads
    random_ctrl_heads = pick_random_ctrl(NH, top2_heads, SEED)

    print(f"  top-2 heads:       {sorted(top2_heads)}")
    print(f"  complement heads:  {sorted(complement_heads)}")
    print(f"  random ctrl heads: {sorted(random_ctrl_heads)}")

    conditions: list[tuple[str, frozenset]] = [
        ("baseline",        frozenset()),
        ("ablate_topheads", top2_heads),
        ("keep_topheads",   complement_heads),
        ("random_ctrl",     random_ctrl_heads),
    ]

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
    print(f"  eval set hash: {ds_hash}")

    # ── Determinism check (GPU-appropriate threshold) ─────────────────────────
    _batch = next(iter(loader))
    _audio = _batch["audio"].to(device)
    _num_f = torch.full((_audio.shape[0],), ha.NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        _hs, _pids = ha.run_frozen_frontend(_audio, lit.model, device)
        _lg1 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
        _lg2 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
    _diff = (_lg1 - _lg2).abs().max().item()
    assert _diff < 1e-4, f"Model not deterministic: max logit diff={_diff:.2e}"
    print(f"  Determinism check PASSED: max logit diff = {_diff:.2e}")
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

    # (b) Baseline must match e4_goat_baseline within tolerance
    ref_base = load_baseline_pooled_eer(BASELINE_CSV)
    base_eer = pooled_eers["baseline"]
    if ref_base is not None:
        diff_b = abs(base_eer - ref_base)
        if diff_b <= BASELINE_EER_TOLERANCE:
            msg = f"OK — baseline EER {base_eer:.4f} matches e4_goat_baseline {ref_base:.4f} (diff={diff_b:.4f})"
            print(f"  (b) {msg}")
        else:
            msg = f"WARN — baseline EER {base_eer:.4f} vs e4_goat_baseline {ref_base:.4f} (diff={diff_b:.4f} > tol={BASELINE_EER_TOLERANCE})"
            print(f"  (b) {msg}")
    else:
        msg = f"SKIP — e4_goat_baseline CSV not found; baseline EER={base_eer:.4f}"
        print(f"  (b) {msg}")
    sanity_results["baseline_parity"] = msg

    # ablate ≠ keep (bug check)
    diff_c = abs(pooled_eers["ablate_topheads"] - pooled_eers["keep_topheads"])
    if diff_c < 1e-6:
        msg = f"FAIL — ablate_topheads and keep_topheads have identical EER={pooled_eers['ablate_topheads']:.4f}"
        print(f"  (c) {msg}")
    else:
        msg = (f"OK — ablate_topheads={pooled_eers['ablate_topheads']:.4f} ≠ "
               f"keep_topheads={pooled_eers['keep_topheads']:.4f} (diff={diff_c:.4f})")
        print(f"  (c) {msg}")
    sanity_results["ablate_ne_keep"] = msg

    sanity_pass = all(not s.startswith(("FAIL", "WARN")) for s in sanity_results.values())

    # ── Per-system EER + bootstrap CIs ───────────────────────────────────────
    print("\n--- Per-system EER + bootstrap CIs ---")
    eer_by_cond: dict[str, dict[str, float]] = {}
    ci_by_cond:  dict[str, dict[str, tuple]] = {}
    rng = np.random.default_rng(SEED)
    for cond_name, _ in conditions:
        eer_by_cond[cond_name] = e4lib.per_system_eer(all_records[cond_name], attack_sids)
        ci_by_cond[cond_name]  = e4lib.bootstrap_eer_ci(
            all_records[cond_name], attack_sids, n_bootstrap=N_BOOTSTRAP, rng=rng)

    # CI overlap check for random_ctrl
    base_lo = float(np.mean([ci_by_cond["baseline"][s][0] for s in attack_sids]))
    base_hi = float(np.mean([ci_by_cond["baseline"][s][1] for s in attack_sids]))
    ctrl_lo = float(np.mean([ci_by_cond["random_ctrl"][s][0] for s in attack_sids]))
    ctrl_hi = float(np.mean([ci_by_cond["random_ctrl"][s][1] for s in attack_sids]))
    ci_overlap = base_lo <= ctrl_hi and ctrl_lo <= base_hi

    # ── Build output table (same schema as e4_goat_baseline) ─────────────────
    print(f"\n{'system':8s}  {'baseline':>10s}  {'ablate_top2':>13s}  {'keep_top2':>11s}  {'random_ctrl':>12s}")
    print("-" * 66)
    table_rows: list[dict] = []
    for sid in sorted(attack_sids):
        b   = eer_by_cond["baseline"][sid]
        at  = eer_by_cond["ablate_topheads"][sid]
        kt  = eer_by_cond["keep_topheads"][sid]
        rc  = eer_by_cond["random_ctrl"][sid]
        ci_b  = ci_by_cond["baseline"][sid]
        ci_at = ci_by_cond["ablate_topheads"][sid]
        ci_kt = ci_by_cond["keep_topheads"][sid]
        ci_rc = ci_by_cond["random_ctrl"][sid]
        print(f"  {sid:6s}  "
              f"{b:.4f}[{ci_b[0]:.3f},{ci_b[1]:.3f}]  "
              f"{at:.4f}[{ci_at[0]:.3f},{ci_at[1]:.3f}]  "
              f"{kt:.4f}[{ci_kt[0]:.3f},{ci_kt[1]:.3f}]  "
              f"{rc:.4f}[{ci_rc[0]:.3f},{ci_rc[1]:.3f}]")
        # Column names match e4_goat_baseline schema with "topheads" replacing "h0_h4"
        table_rows.append({
            "system":                   sid,
            "eer_baseline":             b,
            "eer_ablate_topheads":      at,
            "eer_keep_topheads":        kt,
            "eer_random_ctrl":          rc,
            "delta_ablate_topheads":    at - b,
            "delta_keep_topheads":      kt - b,
            "delta_random_ctrl":        rc - b,
            "ci_baseline_lo":           ci_b[0],  "ci_baseline_hi":           ci_b[1],
            "ci_ablate_topheads_lo":    ci_at[0], "ci_ablate_topheads_hi":    ci_at[1],
            "ci_keep_topheads_lo":      ci_kt[0], "ci_keep_topheads_hi":      ci_kt[1],
            "ci_random_ctrl_lo":        ci_rc[0], "ci_random_ctrl_hi":        ci_rc[1],
        })

    _nan = float("nan")
    table_rows.append({
        "system":                "pooled",
        "eer_baseline":          pooled_eers["baseline"],
        "eer_ablate_topheads":   pooled_eers["ablate_topheads"],
        "eer_keep_topheads":     pooled_eers["keep_topheads"],
        "eer_random_ctrl":       pooled_eers["random_ctrl"],
        "delta_ablate_topheads": pooled_eers["ablate_topheads"] - pooled_eers["baseline"],
        "delta_keep_topheads":   pooled_eers["keep_topheads"]   - pooled_eers["baseline"],
        "delta_random_ctrl":     pooled_eers["random_ctrl"]     - pooled_eers["baseline"],
        "ci_baseline_lo": _nan,        "ci_baseline_hi": _nan,
        "ci_ablate_topheads_lo": _nan, "ci_ablate_topheads_hi": _nan,
        "ci_keep_topheads_lo":  _nan,  "ci_keep_topheads_hi":  _nan,
        "ci_random_ctrl_lo":    _nan,  "ci_random_ctrl_hi":    _nan,
    })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "e4_goat_topheads_per_system_eer.csv"
    write_csv(table_rows, csv_path)

    # ── Save run_config.json ──────────────────────────────────────────────────
    run_config = {
        "checkpoint":      str(CKPT),
        "checkpoint_name": CKPT.name,
        "ablation_spec": {
            "layer":              ABLATION_LAYER,
            "mode":               ABLATION_MODE,
            "top2_heads":         sorted(top2_heads),
            "complement_heads":   sorted(complement_heads),
            "random_ctrl_heads":  sorted(random_ctrl_heads),
            "top2_same_as_robust_goat": top2_same_as_robust,
            "discovery_source":   str(DISCOVERY),
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
    run_cfg_path = OUT_DIR / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_config, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────────
    b  = pooled_eers["baseline"]
    at = pooled_eers["ablate_topheads"]
    kt = pooled_eers["keep_topheads"]
    rc = pooled_eers["random_ctrl"]

    print(f"\n{'='*70}")
    print(f"Results: {OUT_DIR}/")
    print(f"  {csv_path.name}")
    print(f"  {run_cfg_path.name}")
    print(f"\nTop-2 heads on goat.ckpt: {sorted(top2_heads)}"
          + ("  [same as robust_goat — see flag]" if top2_same_as_robust else ""))
    print(f"\nPooled EER:")
    print(f"  baseline:        {b:.4f}")
    print(f"  ablate top-2:    {at:.4f}  (Δ{at-b:+.4f})")
    print(f"  keep-only top-2: {kt:.4f}  (Δ{kt-b:+.4f})")
    print(f"  random ctrl:     {rc:.4f}  (Δ{rc-b:+.4f})")
    print(f"\nSanity: {'PASS' if sanity_pass else 'FAIL/WARN — review above'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
