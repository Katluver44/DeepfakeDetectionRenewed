#!/usr/bin/env python3
"""
mlaad_head_discovery.py
=======================
GAT layer-0 head specialization analysis on MLAAD checkpoints.

Imports the class-conditional KL divergence pipeline from gat_l0_attention.py
and per_head_mean_kl from goat_head_discovery.py — no logic duplication.

Runs on mlaad_robust_goat.ckpt and mlaad_goat.ckpt using
test_in_distribution.json and test_cross_language.json (written by
eval_mlaad_baseline.py).

Usage:
    python experiments/scripts/mlaad_head_discovery.py

Outputs: experiments/results/mlaad/head_discovery/
    mlaad_robust_in_distribution/head_ranking.json
    mlaad_robust_cross_language/head_ranking.json
    mlaad_goat_in_distribution/head_ranking.json
    mlaad_goat_cross_language/head_ranking.json
    comparison_summary.json
    target_heads.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR      = PROJECT_ROOT / "experiments"
SCRIPTS_DIR  = EXP_DIR / "scripts"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── torch.load compat (must run before importing gla / ghd) ──────────────────
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

# ── Import from existing scripts (no logic duplication) ──────────────────────
import gat_l0_attention as gla          # extract_attention, aggregate_9class,
                                        # compute_kl_per_system, build_vocab,
                                        # patch_phoneme_loader, load_model, collate
import goat_head_discovery as ghd       # per_head_mean_kl, ROBUST_GOAT_HEAD_RANKING

# ── Constants ─────────────────────────────────────────────────────────────────
CKPT_DIR         = PROJECT_ROOT / "experiments/checkpoints"
PROCESSED_DIR    = PROJECT_ROOT / "experiments/data/mlaad_tiny_processed"
BASELINE_EVAL_DIR = PROJECT_ROOT / "experiments/results/mlaad/baseline_eval"
OUT_DIR          = PROJECT_ROOT / "experiments/results/mlaad/head_discovery"

BATCH_SIZE = 8
SEED       = 42
TOP_K      = 2

# Reference ranges from ASVspoof runs (for comparison commentary)
ASVSPOOF_ROBUST_GOAT_KL_RANGE = 3.1
ASVSPOOF_GOAT_KL_RANGE        = 1.15

# Thresholds for the proceed/caveat/stop decision
THRESHOLD_PROCEED    = 2.0
THRESHOLD_CAVEAT     = 1.5


def _best_or_last(stem: str) -> Path:
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else CKPT_DIR / f"{stem}.ckpt"


CHECKPOINTS: dict[str, tuple[str, Path]] = {
    # display_name → (output_prefix, path)
    "mlaad_robust_goat": ("mlaad_robust", _best_or_last("mlaad_robust_goat")),
    "mlaad_goat":        ("mlaad_goat",   _best_or_last("mlaad_goat")),
}


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def _dump(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ── MLAAD dataset for head discovery ─────────────────────────────────────────

class MAALDHeadDataset(torch.utils.data.Dataset):
    """
    Loads MLAAD test split records and returns the format expected by
    gla.extract_attention: {audio (1,T), label, system_id}.

    system_id convention (matches compute_kl_per_system bonafide_key="-"):
      bonafide → "-"
      spoof    → attack_system value from the record
    """

    def __init__(self, records: list[dict], processed_dir: Path):
        self.records      = records
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(self.processed_dir / rec["audio_path"])   # (48000,) float32
        wav = wav.unsqueeze(0)                                       # (1, 48000)
        y   = 0 if rec["label"] == "bonafide" else 1
        sys_id = "-" if rec["label"] == "bonafide" else rec.get("attack_system", "unknown")
        return {
            "audio":     wav,
            "label":     torch.tensor(y, dtype=torch.long),
            "system_id": sys_id,
        }


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_head_discovery(
    lit,
    records: list[dict],
    processed_dir: Path,
    device: torch.device,
) -> dict:
    """
    Run the full KL head-ranking pipeline on one (model, split) pair.
    Returns a result dict with head_kl, ranking, top2, kl_range, etc.
    """
    _, id_to_cls = gla.build_vocab()
    NH = lit.model.GAT.gat_net[0].num_of_heads

    ds     = MAALDHeadDataset(records, processed_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=gla.collate,
    )

    t0 = time.time()
    attn_records, n_deg = gla.extract_attention(lit, loader, device)
    elapsed = time.time() - t0

    systems     = sorted(set(r["system_id"] for r in attn_records))
    attack_sids = [s for s in systems if s != "-"]

    accum, _    = gla.aggregate_9class(attn_records, id_to_cls, systems)
    kl_results  = gla.compute_kl_per_system(accum, systems)

    # Per-head mean KL (imported from goat_head_discovery — no duplication)
    head_kl = ghd.per_head_mean_kl(kl_results, attack_sids, NH)
    ranking  = sorted(head_kl, key=lambda h: -head_kl[h])
    top2     = ranking[:TOP_K]

    kl_vals  = list(head_kl.values())
    kl_range = max(kl_vals) / max(min(kl_vals), 1e-10)

    # Sanity (b): scores not all zero
    all_zero    = all(v < 1e-10 for v in kl_vals)
    all_uniform = (max(kl_vals) - min(kl_vals)) < 1e-6

    return {
        "NH":            NH,
        "head_kl":       head_kl,          # {head_idx: mean_kl}
        "ranking":       ranking,           # head indices, descending mean_kl
        "top2":          top2,
        "kl_range":      kl_range,
        "kl_results":    kl_results,        # {system: {head_kl, mean_kl}}
        "attack_sids":   attack_sids,
        "n_total":       len(attn_records),
        "n_degenerate":  n_deg,
        "elapsed_s":     elapsed,
        "sanity_b_all_zero":    all_zero,
        "sanity_b_all_uniform": all_uniform,
    }


def build_head_ranking_doc(
    ckpt_name: str,
    split_name: str,
    ckpt_path: Path,
    res: dict,
) -> dict:
    kl_results  = res["kl_results"]
    head_kl     = res["head_kl"]
    ranking     = res["ranking"]
    top2        = res["top2"]
    attack_sids = res["attack_sids"]

    per_system_kl = {
        sid: {
            "mean_kl": float(kl_results[sid]["mean_kl"]),
            "head_kl": [float(v) for v in kl_results[sid]["head_kl"]],
        }
        for sid in attack_sids
    }

    return {
        "checkpoint":  ckpt_name,
        "split":       split_name,
        "ckpt_path":   str(ckpt_path),
        "nh":          res["NH"],
        "criterion":   "mean KL(attack_h || bonafide_h) across all attack systems, "
                       "Laplace-smoothed 9-class (C×C) attention matrix",
        "kl_range_max_min": float(res["kl_range"]),
        "head_ranking": [
            {
                "rank":       rank + 1,
                "head":       int(h),
                "mean_kl":    float(head_kl[h]),
                "per_system": {s: float(kl_results[s]["head_kl"][h]) for s in attack_sids},
            }
            for rank, h in enumerate(ranking)
        ],
        "top2_heads":       [int(h) for h in top2],
        "n_attack_systems": len(attack_sids),
        "attack_systems":   attack_sids,
        "n_total":          res["n_total"],
        "n_degenerate":     res["n_degenerate"],
        "per_system_kl":    per_system_kl,
        "sanity_b_pass":    not res["sanity_b_all_zero"] and not res["sanity_b_all_uniform"],
    }


# ── Decision logic ────────────────────────────────────────────────────────────

def make_decision(robust_in_dist_range: float, top2_stable: bool) -> tuple[str, str]:
    """
    Returns (decision_code, explanation).
    decision_code: "proceed" | "proceed_with_caveat" | "stop"
    """
    if robust_in_dist_range >= THRESHOLD_PROCEED and top2_stable:
        return (
            "proceed",
            f"KL range {robust_in_dist_range:.2f}× > {THRESHOLD_PROCEED}× and top-2 stable "
            "across test sets. Specialization is robust; proceed to Phase 3 ablation."
        )
    elif robust_in_dist_range >= THRESHOLD_CAVEAT:
        return (
            "proceed_with_caveat",
            f"KL range {robust_in_dist_range:.2f}× is in the moderate zone "
            f"({THRESHOLD_CAVEAT}–{THRESHOLD_PROCEED}×). "
            "Specialization exists but is weaker than ASVspoof (3.1×). "
            "Proceed to Phase 3 but flag weaker specialization in the writeup. "
            + ("" if top2_stable else "Note: top-2 heads differ across test sets — "
               "use in-distribution ranking for ablation targets.")
        )
    else:
        return (
            "stop",
            f"KL range {robust_in_dist_range:.2f}× < {THRESHOLD_CAVEAT}×. "
            "Head specialization did not emerge on MLAAD. "
            "The 'robust training creates specialization' claim does not generalise here. "
            "Phase 3 ablation would likely return a null result. "
            "Recommended: report this as a negative finding. "
            "Optionally run Phase 3 to document the null result explicitly."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--baseline-eval-dir", type=Path, default=BASELINE_EVAL_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Verify inputs ─────────────────────────────────────────────────────────
    in_dist_path  = args.baseline_eval_dir / "test_in_distribution.json"
    cross_lang_path = args.baseline_eval_dir / "test_cross_language.json"
    for p in (in_dist_path, cross_lang_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run eval_mlaad_baseline.py first.")
            sys.exit(1)

    for ck_name, (_, ck_path) in CHECKPOINTS.items():
        if not ck_path.exists():
            print(f"ERROR: checkpoint not found: {ck_path}")
            sys.exit(1)

    in_dist_records   = json.loads(in_dist_path.read_text())
    cross_lang_records = json.loads(cross_lang_path.read_text())

    print(f"In-distribution split:  {len(in_dist_records)} records")
    print(f"Cross-language split:   {len(cross_lang_records)} records")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run discovery for each (checkpoint, split) ───────────────────────────
    # Load each model once; call extract_attention for both splits on the same lit.
    all_results: dict[tuple[str, str], dict] = {}

    SPLITS = [
        ("in_distribution", in_dist_records),
        ("cross_language",  cross_lang_records),
    ]

    gla.patch_phoneme_loader()

    for ckpt_name, (out_prefix, ckpt_path) in CHECKPOINTS.items():
        print(f"\n{'='*65}")
        print(f"Loading {ckpt_name} from {ckpt_path.name}")
        lit = gla.load_model(ckpt_path, device)

        for split_name, records in SPLITS:
            print(f"\n--- {ckpt_name} / {split_name} ({len(records)} records) ---")

            res = run_head_discovery(lit, records, args.processed_dir, device)
            all_results[(ckpt_name, split_name)] = res

            # Sanity (b) report
            if res["sanity_b_all_zero"]:
                print(f"  [SANITY (b) FAIL] All per-head KL scores are zero — "
                      "checkpoint load or hook may be broken.")
            elif res["sanity_b_all_uniform"]:
                print(f"  [SANITY (b) FAIL] All per-head KL scores identical — "
                      "no head differentiation.")
            else:
                print(f"  [SANITY (b) PASS] Per-head KL scores span "
                      f"{min(res['head_kl'].values()):.6f} – {max(res['head_kl'].values()):.6f}")

            kl_range = res["kl_range"]
            if kl_range < THRESHOLD_CAVEAT:
                print(f"  [SANITY (c) FLAG] KL range {kl_range:.3f}× < {THRESHOLD_CAVEAT}× — "
                      "specialization did not emerge on MLAAD for this checkpoint/split.")

            print(f"  KL range (max/min): {kl_range:.3f}×")
            print(f"  Top-{TOP_K} heads: {res['top2']}")
            print(f"  Head ranking (descending mean KL):")
            for rank, h in enumerate(res["ranking"]):
                tag = " ← top-2" if h in res["top2"] else ""
                print(f"    rank {rank+1}: h{h}  mean_KL={res['head_kl'][h]:.6f}{tag}")

            # Write per-run output dir
            run_dir = args.out_dir / f"{out_prefix}_{split_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            doc = build_head_ranking_doc(ckpt_name, split_name, ckpt_path, res)
            _dump(doc, run_dir / "head_ranking.json")
            print(f"  Wrote: {run_dir}/head_ranking.json")

    # ── Sanity (a): rankings must differ between checkpoints ─────────────────
    print(f"\n{'='*65}")
    print("SANITY CHECKS")
    robust_in_rank = all_results[("mlaad_robust_goat", "in_distribution")]["ranking"]
    goat_in_rank   = all_results[("mlaad_goat",        "in_distribution")]["ranking"]
    rankings_identical = (robust_in_rank == goat_in_rank)
    if rankings_identical:
        print("  [SANITY (a) FAIL] In-distribution head rankings for mlaad_robust_goat "
              "and mlaad_goat are IDENTICAL — possible checkpoint loading bug.")
    else:
        print(f"  [SANITY (a) PASS] Rankings differ: "
              f"robust_goat={robust_in_rank}  goat={goat_in_rank}")

    # ── Comparison summary ────────────────────────────────────────────────────
    summary: dict = {
        "asvspoof_reference": {
            "robust_goat_kl_range": ASVSPOOF_ROBUST_GOAT_KL_RANGE,
            "goat_kl_range":        ASVSPOOF_GOAT_KL_RANGE,
            "note": "KL max/min ratio across per-head mean KL scores, from original ASVspoof run",
        },
        "sanity_a_pass": not rankings_identical,
    }

    for ckpt_name, (out_prefix, ckpt_path) in CHECKPOINTS.items():
        ckpt_summary: dict = {}
        for split_name, _ in SPLITS:
            res = all_results[(ckpt_name, split_name)]
            ckpt_summary[split_name] = {
                "kl_range":  float(res["kl_range"]),
                "top2_heads": [int(h) for h in res["top2"]],
                "head_kl":   {str(h): float(v) for h, v in res["head_kl"].items()},
                "ranking":   [int(h) for h in res["ranking"]],
                "n_attack_systems": len(res["attack_sids"]),
                "sanity_b_pass": not res["sanity_b_all_zero"] and not res["sanity_b_all_uniform"],
            }

        # Stability: same top-2 on both test sets?
        top2_in  = frozenset(all_results[(ckpt_name, "in_distribution")]["top2"])
        top2_cl  = frozenset(all_results[(ckpt_name, "cross_language")]["top2"])
        stable   = (top2_in == top2_cl)
        ckpt_summary["top2_stable_across_splits"] = stable
        ckpt_summary["top2_overlap"] = int(len(top2_in & top2_cl))
        summary[ckpt_name] = ckpt_summary

    _dump(summary, args.out_dir / "comparison_summary.json")
    print(f"\n  Wrote: {args.out_dir}/comparison_summary.json")

    # ── Target heads selection ────────────────────────────────────────────────
    # Use mlaad_robust_goat in-distribution ranking as primary.
    robust_res    = all_results[("mlaad_robust_goat", "in_distribution")]
    robust_range  = robust_res["kl_range"]
    target_top2   = [int(h) for h in robust_res["top2"]]
    robust_stable = summary["mlaad_robust_goat"]["top2_stable_across_splits"]

    decision_code, decision_explanation = make_decision(robust_range, robust_stable)

    # Also compare with ASVspoof top-2 (h0, h4)
    same_as_asvspoof = frozenset(target_top2) == frozenset({0, 4})

    target_heads = {
        "selected_checkpoint":  "mlaad_robust_goat",
        "ranking_source":       "in_distribution",
        "target_heads":         target_top2,
        "rationale": (
            f"Top-{TOP_K} heads by descending mean KL(attack_h || bonafide_h) on "
            "mlaad_robust_goat in-distribution test set. "
            f"KL range = {robust_range:.3f}×. "
            f"Stable across splits: {robust_stable}."
        ),
        "kl_range":              float(robust_range),
        "top2_stable":           robust_stable,
        "same_heads_as_asvspoof": same_as_asvspoof,
        "asvspoof_note": (
            "Head identity is initialization-dependent; "
            "exact index match with ASVspoof {h0,h4} is not expected. "
            f"Match: {same_as_asvspoof}."
        ),
        "decision":              decision_code,
        "decision_explanation":  decision_explanation,
        "cross_language_top2":   [int(h) for h in all_results[
            ("mlaad_robust_goat", "cross_language")]["top2"]],
    }
    _dump(target_heads, args.out_dir / "target_heads.json")
    print(f"  Wrote: {args.out_dir}/target_heads.json")

    # ── Print decision ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MLAAD HEAD DISCOVERY — DECISION SUMMARY")
    print("=" * 65)
    print(f"\n  KL ranges (max/min ratio across per-head mean KL):")
    for ckpt_name, (_, _) in CHECKPOINTS.items():
        for split_name, _ in SPLITS:
            r = all_results[(ckpt_name, split_name)]["kl_range"]
            print(f"    {ckpt_name:25s} / {split_name:20s}: {r:.3f}×")

    print(f"\n  ASVspoof reference: robust_goat={ASVSPOOF_ROBUST_GOAT_KL_RANGE}×  "
          f"goat={ASVSPOOF_GOAT_KL_RANGE}×")

    print(f"\n  Top-{TOP_K} heads selected (from mlaad_robust_goat / in_distribution):")
    print(f"    {target_top2}")
    print(f"  Stable across test sets: {robust_stable}")
    print(f"  Same as ASVspoof {{h0,h4}}: {same_as_asvspoof}")

    print(f"\n  DECISION: {decision_code.upper()}")
    print(f"  {decision_explanation}")
    print("=" * 65)
    print(f"\nAll outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
