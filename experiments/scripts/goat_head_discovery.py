#!/usr/bin/env python3
"""
goat_head_discovery.py
======================
Mirror the original class-conditional attention-divergence analysis from
gat_l0_attention.py, but evaluated on goat.ckpt instead of robust_goat.ckpt.

Metric: for each GAT layer-0 head h, compute
    mean_KL(h) = mean over attack systems of KL(attack_h || bonafide_h)
where KL is computed on Laplace-smoothed 9-class (C×C) attention matrices.
This is the exact criterion used to identify h0,h4 on robust_goat.ckpt.

Top-2 heads are selected by descending mean_KL(h).

Outputs → experiments/results/gat_l0_attention_followups/goat_head_discovery/
  head_ranking.json   — per-head scores, full ranking, selected top-2 pair
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
OUT_DIR   = EXP_DIR / "results" / "gat_l0_attention_followups" / "goat_head_discovery"
CKPT      = REPO_ROOT / "models" / "goat.ckpt"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import original divergence-analysis utilities (no logic duplication) ──────
import gat_l0_attention as gla   # extract_attention, aggregate_9class,
                                  # compute_kl_per_system, build_vocab,
                                  # patch_phoneme_loader, load_model,
                                  # BalancedDataset, collate

# ── Constants (must match e4_goat_baseline / gat_l0_attention exactly) ────────
HF_DATASET   = "Bisher/ASVspoof_2019_LA"
CACHE_DIR    = REPO_ROOT / "data" / "asvspoof_2019_la"
N_PER_CLASS  = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8))
SEED         = 42

# Reference ranking from robust_goat.ckpt (h4 > h0 > h1 > h3 > h2 > h5)
# Used for sanity check (a): goat ranking must differ.
ROBUST_GOAT_HEAD_RANKING = [4, 0, 1, 3, 2, 5]   # descending mean-KL order

TOP_K = 2   # number of top heads to select


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


def ckpt_hash_prefix(path: Path, nbytes: int = 65536) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(nbytes)).hexdigest()[:16]


def eval_set_hash(dataset: gla.BalancedDataset) -> str:
    key = json.dumps(sorted(zip(dataset.indices, dataset.sys_ids)))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def per_head_mean_kl(kl_results: dict, attack_sids: list[str], nh: int) -> dict[int, float]:
    """For each head h: mean of KL(attack_system_h || bonafide_h) across all attack systems."""
    return {
        h: float(np.mean([kl_results[s]["head_kl"][h] for s in attack_sids]))
        for h in range(nh)
    }


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    gla.patch_phoneme_loader()
    print(f"Loading model: {CKPT.name}")
    lit = gla.load_model(CKPT, device)
    NH = lit.model.GAT.gat_net[0].num_of_heads
    print(f"  model loaded  (NH={NH})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    print(f"\nLoading dataset (N_PER_CLASS={N_PER_CLASS}, split=validation)...")
    dataset = gla.BalancedDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token, N_PER_CLASS, SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=gla.collate)

    ds_hash     = eval_set_hash(dataset)
    attack_sids = sorted(s for s in set(dataset.sys_ids) if s != "-")
    print(f"  eval set hash: {ds_hash}  attack systems: {attack_sids}")

    # ── Extract attention (same hook+pipeline as gat_l0_attention.py) ─────────
    print("\n--- Extracting GAT layer-0 attention ---")
    t0 = time.time()
    records, n_deg = gla.extract_attention(lit, loader, device)
    print(f"  {len(records)} samples ({n_deg} degenerate) in {time.time()-t0:.1f}s")

    systems = sorted(set(r["system_id"] for r in records))

    # ── 9-class aggregation ───────────────────────────────────────────────────
    print("\n--- 9-class aggregation ---")
    _, id_to_cls = gla.build_vocab()
    accum, _counts = gla.aggregate_9class(records, id_to_cls, systems)

    # ── KL divergence per head (exact same function as original) ─────────────
    print("\n--- KL divergence ---")
    kl_results = gla.compute_kl_per_system(accum, systems)

    for sid in sorted(kl_results, key=lambda s: -kl_results[s]["mean_kl"]):
        hkl = "  ".join(f"h{h}={v:.4f}" for h, v in enumerate(kl_results[sid]["head_kl"]))
        print(f"  {sid}: mean_KL={kl_results[sid]['mean_kl']:.4f}  [{hkl}]")

    # ── Per-head mean KL (the criterion used to select h0,h4 on robust_goat) ──
    head_kl = per_head_mean_kl(kl_results, attack_sids, NH)
    ranking  = sorted(head_kl, key=lambda h: -head_kl[h])   # descending
    top2     = frozenset(ranking[:TOP_K])

    print(f"\n--- Head ranking by mean KL(attack || bonafide) ---")
    for rank, h in enumerate(ranking):
        tag = " ← TOP-2" if h in top2 else ""
        print(f"  rank {rank+1}: h{h}  mean_KL={head_kl[h]:.6f}{tag}")

    print(f"\n  Top-{TOP_K} selected: {sorted(top2)}")

    # ── Sanity (a): ranking must differ from robust_goat's ───────────────────
    goat_ranking   = ranking
    robust_ranking = ROBUST_GOAT_HEAD_RANKING
    ranking_identical = goat_ranking == robust_ranking
    goat_top2_is_h0h4 = top2 == frozenset({0, 4})

    if ranking_identical:
        print("\n  [SANITY (a) WARN] Full head ranking is IDENTICAL to robust_goat.ckpt.")
        print("  This may indicate the wrong checkpoint was loaded or cached results are stale.")
    else:
        print(f"\n  Sanity (a): PASS — goat ranking {goat_ranking} ≠ robust_goat {robust_ranking}")

    if goat_top2_is_h0h4:
        print("\n  [SANITY (c) FLAG] goat.ckpt top-2 = {h0, h4} — same as robust_goat.ckpt.")
        print("  This INVALIDATES the working hypothesis that robust training specifically")
        print("  concentrates discriminative signal into h0,h4. The random-null run becomes")
        print("  the primary comparison. Proceed with ablation but interpret accordingly.")
    else:
        print(f"  Top-2 = {sorted(top2)} ≠ {{h0,h4}} — goat.ckpt uses different critical heads")

    # ── Build output ──────────────────────────────────────────────────────────
    per_system_kl = {
        sid: {
            "mean_kl": float(kl_results[sid]["mean_kl"]),
            "head_kl": [float(v) for v in kl_results[sid]["head_kl"]],
        }
        for sid in attack_sids
    }

    head_ranking_doc = {
        "checkpoint":       CKPT.name,
        "nh":               NH,
        "criterion":        "mean KL(attack_h || bonafide_h) across all attack systems, "
                            "Laplace-smoothed 9-class attention matrix",
        "head_ranking": [
            {
                "rank":        rank + 1,
                "head":        int(h),
                "mean_kl":     float(head_kl[h]),
                "per_system":  {s: float(kl_results[s]["head_kl"][h]) for s in attack_sids},
            }
            for rank, h in enumerate(ranking)
        ],
        "top2_heads":             sorted(top2),
        "top2_same_as_robust_goat": goat_top2_is_h0h4,
        "full_ranking_same_as_robust_goat": ranking_identical,
        "robust_goat_top2":       [0, 4],
        "robust_goat_ranking":    ROBUST_GOAT_HEAD_RANKING,
        "per_system_kl":          per_system_kl,
        "sanity_a_pass":          not ranking_identical,
        "sanity_c_flag":          goat_top2_is_h0h4,
    }

    run_cfg = {
        "checkpoint":         str(CKPT),
        "checkpoint_name":    CKPT.name,
        "checkpoint_hash_prefix": ckpt_hash_prefix(CKPT),
        "eval_set": {
            "hf_dataset":    HF_DATASET,
            "split":         "validation",
            "n_per_class":   N_PER_CLASS,
            "attack_systems": attack_sids,
            "dataset_hash":  ds_hash,
        },
        "seed":       SEED,
        "nh":         NH,
        "top_k":      TOP_K,
        "top2_heads": sorted(top2),
        "sanity_a_pass":  not ranking_identical,
        "sanity_c_flag":  goat_top2_is_h0h4,
    }

    _dump(head_ranking_doc, OUT_DIR / "head_ranking.json")
    _dump(run_cfg,           OUT_DIR / "run_config.json")

    # ── Final print ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Results: {OUT_DIR}/")
    print(f"  head_ranking.json")
    print(f"  run_config.json")
    print(f"\ngoat.ckpt head ranking (descending mean KL):")
    for rank, h in enumerate(ranking):
        tag = " ← top-2" if h in top2 else ""
        print(f"  rank {rank+1}: h{h}  mean_KL={head_kl[h]:.6f}{tag}")
    print(f"\nTop-2 selected: {sorted(top2)}")
    if goat_top2_is_h0h4:
        print("  WARNING: top-2 is {h0,h4} — same as robust_goat; see flag above")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
