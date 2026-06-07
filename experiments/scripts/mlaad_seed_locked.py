#!/usr/bin/env python3
"""
mlaad_seed_locked.py
====================
Seed-locked reproducibility study for MLAAD head-specialisation findings.

Trains mlaad_goat and mlaad_robust_goat with multiple random seeds, then
runs the full head-discovery + ablation + entropy pipeline on each checkpoint.
Aggregates across seeds and applies pre-registered stability criteria.

Usage:
    python experiments/scripts/mlaad_seed_locked.py [--n-seeds 3|5]
                                                    [--skip-training]
                                                    [--reuse-seed42]

Pre-registered criteria (stated before any data is seen):
  Strong   : same head pair in top-2 across >=4/5 robust seeds
  Moderate : same pair in >=3/5 robust seeds, OR one head consistently
             appears (top-1 or top-2) in >=4/5
  Weak     : head identity varies but KL widening + ablation DEER + entropy
             collapse all go in the right direction and within +-50% CV
  Failed   : none of the above

Outputs:
    experiments/results/mlaad/seed_locked_reproducibility/
        per_seed_results.json
        head_identity_table.csv
        mechanism_magnitudes.csv
        paired_robust_vs_goat.csv
        seed_stability_summary.json
        run_config.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
BASELINE_DIR  = EXP_DIR / "results" / "mlaad" / "baseline_eval"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "seed_locked_reproducibility"
LOG_BASE      = EXP_DIR / "results" / "mlaad" / "training_logs"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# torch.load compat (avoid weights_only warning)
# ---------------------------------------------------------------------------
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

try:
    from argparse import Namespace as _NS
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([_NS, _PS, _PTR])
except Exception:
    from argparse import Namespace as _NS
    torch.serialization.add_safe_globals([_NS])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_SEEDS  = [42, 123, 456, 789, 1024]
CONDITIONS = ["goat", "robust_goat"]
BATCH_SIZE = 32
TOP_K      = 2

TRAIN_SCRIPTS = {
    "goat":        SCRIPTS_DIR / "train_mlaad_regular.py",
    "robust_goat": SCRIPTS_DIR / "train_mlaad_adversarial.py",
}


def _resolve_python() -> str:
    """
    Find a Python interpreter that has the project dependencies (librosa, torch, etc.).
    Prefers the project venv, then falls back to sys.executable.
    """
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        import subprocess as _sp
        check = _sp.run([str(venv_python), "-c", "import librosa"],
                        capture_output=True)
        if check.returncode == 0:
            return str(venv_python)
    return sys.executable


PYTHON_BIN = _resolve_python()

# ---------------------------------------------------------------------------
# Pre-registered criteria text (printed before any data is seen)
# ---------------------------------------------------------------------------
CRITERIA_TEXT = """
+--------------------------------------------------------------------------+
|       PRE-REGISTERED CRITERIA -- SEED-LOCKED REPRODUCIBILITY            |
|       (stated before training or analysis begins)                       |
+--------------------------------------------------------------------------+
|  Metric: across N_SEEDS seeds, for mlaad_robust_goat condition.         |
|                                                                          |
|  STRONG   : the SAME head pair appears in top-2 across >=4/5 robust     |
|             seeds -> head identity is a property of the architecture +  |
|             training regime + dataset, not initialization.              |
|                                                                          |
|  MODERATE : the same head pair appears in >=3/5, OR one single head     |
|             consistently appears (top-1 or top-2) in >=4/5 -> moderate |
|             head-identity stability claim.                              |
|                                                                          |
|  WEAK     : head identity varies, but KL range widening, ablation DEER  |
|             > 0, and entropy collapse are all directionally consistent  |
|             and within +-50% CV across seeds -> mechanism-level claim   |
|             is robust, head-identity claim is not.                      |
|                                                                          |
|  FAILED   : none of the above -> mechanism is not seed-stable; the      |
|             central claim is masked by the single-seed run.             |
+--------------------------------------------------------------------------+
""".strip()


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):    return int(o)
        if isinstance(o, (np.floating,)):   return float(o)
        if isinstance(o, np.ndarray):       return o.tolist()
        if isinstance(o, (set, frozenset)): return sorted(o)
        return super().default(o)


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_stem(cond: str, seed: int) -> str:
    return f"mlaad_{cond}_seed{seed}"


def find_best_ckpt(stem: str) -> Optional[Path]:
    """Best-epoch checkpoint first, then stem.ckpt, then None."""
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    if candidates:
        return candidates[0]
    fallback = CKPT_DIR / f"{stem}.ckpt"
    return fallback if fallback.exists() else None


def find_original_ckpt(cond: str) -> Optional[Path]:
    """Original seed=42 checkpoint under canonical name."""
    stem = f"mlaad_{cond}"
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    if candidates:
        return candidates[0]
    fallback = CKPT_DIR / f"{stem}.ckpt"
    return fallback if fallback.exists() else None


def md5_ckpt(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def train_checkpoint(cond: str, seed: int, n_epochs: int = 7) -> bool:
    """Train one checkpoint. Returns True on success. Skips if already done."""
    stem = ckpt_stem(cond, seed)
    out  = find_best_ckpt(stem)
    if out is not None:
        print(f"  [{stem}] Already trained: {out.name}", flush=True)
        return True

    ckpt_path = CKPT_DIR / f"{stem}.ckpt"
    log_dir   = LOG_BASE / stem
    script    = TRAIN_SCRIPTS[cond]

    cmd = [
        PYTHON_BIN, str(script),
        "--seed",            str(seed),
        "--checkpoint-path", str(ckpt_path),
        "--log-dir",         str(log_dir),
        "--epochs",          str(n_epochs),
    ]

    print(f"\n  [{stem}] Training ({n_epochs} epochs) ...", flush=True)
    t0     = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [{stem}] TRAINING FAILED (exit {result.returncode})", flush=True)
        return False

    best = find_best_ckpt(stem)
    if best is None:
        print(f"  [{stem}] Training OK but no checkpoint found!", flush=True)
        return False

    print(f"  [{stem}] Done in {elapsed/60:.1f} min -> {best.name}", flush=True)
    return True


def train_seed_parallel(seed: int, n_epochs: int = 7) -> bool:
    """
    Train both conditions for one seed simultaneously.
    Conditions that already have checkpoints are skipped.
    Falls back to sequential if only one condition needs training.
    """
    needed = [c for c in CONDITIONS if find_best_ckpt(ckpt_stem(c, seed)) is None]

    # Nothing to do
    if not needed:
        for cond in CONDITIONS:
            stem = ckpt_stem(cond, seed)
            print(f"  [{stem}] Already trained: {find_best_ckpt(stem).name}", flush=True)
        return True

    # Only one condition needs training — run sequentially
    if len(needed) == 1:
        return train_checkpoint(needed[0], seed, n_epochs)

    # Both conditions need training — launch in parallel
    print(f"\n  [seed={seed}] Launching both conditions in parallel ...", flush=True)
    procs: dict[str, subprocess.Popen] = {}
    t0 = time.time()
    for cond in needed:
        stem      = ckpt_stem(cond, seed)
        ckpt_path = CKPT_DIR / f"{stem}.ckpt"
        log_dir   = LOG_BASE / stem
        cmd = [
            PYTHON_BIN, str(TRAIN_SCRIPTS[cond]),
            "--seed",            str(seed),
            "--checkpoint-path", str(ckpt_path),
            "--log-dir",         str(log_dir),
            "--epochs",          str(n_epochs),
        ]
        print(f"  [{stem}] Spawning ...", flush=True)
        procs[cond] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)

    all_ok = True
    for cond, proc in procs.items():
        proc.wait()
        stem    = ckpt_stem(cond, seed)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  [{stem}] TRAINING FAILED (exit {proc.returncode})", flush=True)
            all_ok = False
        else:
            best = find_best_ckpt(stem)
            if best is None:
                print(f"  [{stem}] OK but no checkpoint found!", flush=True)
                all_ok = False
            else:
                print(f"  [{stem}] Done -> {best.name}  (wall {elapsed/60:.1f} min)",
                      flush=True)
    return all_ok


# ---------------------------------------------------------------------------
# Per-head entropy
# ---------------------------------------------------------------------------

def per_head_attention_entropy(attn_records: list, nh: int) -> dict:
    """
    Mean Shannon entropy of the per-node incoming-attention distribution,
    computed independently for each head. Lower -> more specialised.

    Records use edge_index (2, E) where [1] is dst, and n_nodes for node count.
    Degenerate samples are skipped.
    """
    head_ents: dict = defaultdict(list)
    for rec in attn_records:
        if rec.get("is_degenerate"):
            continue
        attn = rec["attn_l0"]
        if hasattr(attn, "numpy"):
            attn = attn.numpy()
        attn = attn.astype(np.float64)              # (E, NH)
        ei   = rec["edge_index"]                    # tensor (2, E) or ndarray
        if hasattr(ei, "numpy"):
            ei = ei.numpy()
        dst  = np.asarray(ei[1])                    # (E,)
        n    = int(rec["n_nodes"])
        for t in range(n):
            mask = (dst == t)
            if not mask.any():
                continue
            a_t = attn[mask]                        # (k, NH)
            eps = 1e-10
            ent = -(a_t * np.log(a_t + eps)).sum(0) # (NH,)
            for h in range(nh):
                head_ents[h].append(float(ent[h]))
    return {h: float(np.mean(head_ents[h])) if head_ents[h] else float("nan")
            for h in range(nh)}


# ---------------------------------------------------------------------------
# Analysis for one checkpoint
# ---------------------------------------------------------------------------

def analyse_checkpoint(
    ckpt_path: Path,
    in_dist_records: list,
    device: torch.device,
) -> dict:
    """Head-discovery + entropy + baseline/ablation for one checkpoint."""
    import gat_l0_attention as gla
    import goat_head_discovery as ghd
    import head_ablation as ha

    print(f"    Loading {ckpt_path.name}", flush=True)
    gla.patch_phoneme_loader()
    lit = gla.load_model(ckpt_path, device)
    NH  = lit.model.GAT.gat_net[0].num_of_heads

    _, id_to_cls = gla.build_vocab()

    # Build data loaders
    from mlaad_head_discovery import MAALDHeadDataset
    hd_ds = MAALDHeadDataset(in_dist_records, PROCESSED_DIR)
    hd_loader = torch.utils.data.DataLoader(
        hd_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=gla.collate,
    )

    # -- Head discovery -------------------------------------------------------
    print("      Head discovery ...", flush=True)
    t0 = time.time()
    attn_records, n_deg = gla.extract_attention(lit, hd_loader, device)
    systems     = sorted(set(r["system_id"] for r in attn_records))
    attack_sids = [s for s in systems if s != "-"]
    accum, _    = gla.aggregate_9class(attn_records, id_to_cls, systems)
    kl_results  = gla.compute_kl_per_system(accum, systems)
    head_kl     = ghd.per_head_mean_kl(kl_results, attack_sids, NH)
    ranking     = sorted(head_kl, key=lambda h: -head_kl[h])
    top2        = ranking[:TOP_K]
    kl_vals     = list(head_kl.values())
    kl_range    = max(kl_vals) / max(min(kl_vals), 1e-10)
    elapsed_hd  = time.time() - t0
    print(f"      top-2={top2}  kl_range={kl_range:.3f}x  ({elapsed_hd:.0f}s)",
          flush=True)

    # -- Per-head entropy -----------------------------------------------------
    print("      Per-head entropy ...", flush=True)
    head_entropy = per_head_attention_entropy(attn_records, NH)
    top2_mean_entropy = float(np.mean([head_entropy[h] for h in top2]))

    # -- Ablation: baseline + ablate own top-2 --------------------------------
    from mlaad_e4_ablation import MAALDAblationDataset, pooled_eer, run_condition

    abl_ds = MAALDAblationDataset(in_dist_records, PROCESSED_DIR)
    abl_loader = torch.utils.data.DataLoader(
        abl_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=ha.collate,
    )

    print("      Baseline eval ...", flush=True)
    t0        = time.time()
    base_recs = run_condition(lit, abl_loader, device, frozenset())
    base_eer  = pooled_eer(base_recs)
    print(f"      baseline EER={base_eer:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"      Ablate {{h{top2[0]},h{top2[1]}}} ...", flush=True)
    t0       = time.time()
    abl_recs = run_condition(lit, abl_loader, device, frozenset(top2))
    abl_eer  = pooled_eer(abl_recs)
    delta    = float(base_eer - abl_eer)   # positive = ablation helps
    print(f"      ablated EER={abl_eer:.4f}  DEER={delta:+.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    ha.sanity_layer_untouched(lit)

    return {
        "checkpoint":        ckpt_path.name,
        "nh":                NH,
        "n_attack_systems":  len(attack_sids),
        "n_degenerate":      n_deg,
        "head_kl":           {str(h): round(float(head_kl[h]), 6)    for h in range(NH)},
        "head_ranking":      [int(h) for h in ranking],
        "top2":              [int(h) for h in top2],
        "top1":              int(ranking[0]),
        "kl_range":          round(float(kl_range), 4),
        "head_entropy":      {str(h): round(float(head_entropy[h]), 6) for h in range(NH)},
        "top2_mean_entropy": round(top2_mean_entropy, 6),
        "baseline_eer":      round(float(base_eer), 6),
        "ablated_eer":       round(float(abl_eer), 6),
        "delta_eer":         round(float(delta), 6),
    }


# ---------------------------------------------------------------------------
# Frozen-frontend sanity check
# ---------------------------------------------------------------------------

def check_frozen_frontend(ckpt_paths: list, device: torch.device,
                           in_dist_records: list) -> bool:
    """
    Sanity (c): verify WavLM produces identical pre-GAT features regardless
    of GAT training seed. Compares first-batch hidden states across checkpoints.
    """
    import gat_l0_attention as gla
    import head_ablation as ha
    from mlaad_e4_ablation import MAALDAblationDataset

    sample_records = in_dist_records[:BATCH_SIZE * 2]
    ds = MAALDAblationDataset(sample_records, PROCESSED_DIR)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=ha.collate,
    )
    batch = next(iter(loader))
    audio = batch["audio"].to(device)

    ref_hs  = None
    all_ok  = True
    for ckpt_path in ckpt_paths:
        gla.patch_phoneme_loader()
        lit = gla.load_model(ckpt_path, device)
        with torch.no_grad():
            hs, _ = ha.run_frozen_frontend(audio, lit.model, device)
        hs_np = hs.cpu().numpy()
        if ref_hs is None:
            ref_hs = hs_np
        else:
            max_diff = float(np.abs(hs_np - ref_hs).max())
            if max_diff > 1e-4:
                print(f"  [SANITY (c) FAIL] {ckpt_path.name}: "
                      f"max hidden-state diff = {max_diff:.2e} (expected <1e-4)",
                      flush=True)
                all_ok = False
    if all_ok:
        print(f"  [SANITY (c) PASS] Pre-GAT WavLM features identical across "
              f"all {len(ckpt_paths)} checkpoints (tol 1e-4).", flush=True)
    return all_ok


# ---------------------------------------------------------------------------
# Aggregate & pre-registered criteria
# ---------------------------------------------------------------------------

def describe(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    cv  = std / (float(np.mean(np.abs(a))) + 1e-10)
    return {
        "mean": round(float(np.mean(a)), 4),
        "std":  round(std, 4),
        "min":  round(float(np.min(a)), 4),
        "max":  round(float(np.max(a)), 4),
        "cv":   round(cv, 4),
    }


def head_pair(top2: list) -> tuple:
    return tuple(sorted(top2))


def evaluate_criteria(robust_results: list, n_seeds: int) -> dict:
    """Apply pre-registered stability criteria to robust_goat results."""
    pairs      = [head_pair(r["top2"]) for r in robust_results]
    pair_counts = Counter(pairs)
    most_common_pair, pair_freq = pair_counts.most_common(1)[0]

    all_heads_in_top2 = [h for p in pairs for h in p]
    head_counts        = Counter(all_heads_in_top2)
    most_common_head, head_freq = head_counts.most_common(1)[0]

    kl_ranges  = [r["kl_range"]          for r in robust_results]
    delta_eers = [r["delta_eer"]          for r in robust_results]
    entropies  = [r["top2_mean_entropy"]  for r in robust_results]

    # Strong
    strong_thresh = max(4, math.ceil(4 * n_seeds / 5))
    if pair_freq >= strong_thresh:
        criterion   = "strong"
        explanation = (
            f"Pair {sorted(most_common_pair)} appeared in {pair_freq}/{n_seeds} "
            f"robust seeds (>={strong_thresh}). Head identity is a property of "
            "architecture+regime+data, not initialization."
        )
    else:
        # Moderate
        mod_pair_thresh = math.ceil(3 * n_seeds / 5)
        mod_head_thresh = max(4, math.ceil(4 * n_seeds / 5))
        if pair_freq >= mod_pair_thresh or head_freq >= mod_head_thresh:
            criterion   = "moderate"
            explanation = (
                f"Pair {sorted(most_common_pair)} in {pair_freq}/{n_seeds} seeds; "
                f"head h{most_common_head} appears in {head_freq}/{n_seeds} seeds. "
                "Moderate head-identity stability."
            )
        else:
            # Weak: effect consistent in direction, CV < 0.5
            n_pos_eer  = sum(1 for d in delta_eers if d > 0)
            all_kl_ok  = all(k > 1.0 for k in kl_ranges)
            kl_cv      = describe(kl_ranges)["cv"]
            eer_cv     = describe([abs(d) for d in delta_eers])["cv"] if delta_eers else 1.0
            direction_ok = n_pos_eer >= math.ceil(2 * n_seeds / 3) and all_kl_ok
            magnitude_ok = kl_cv < 0.5 and eer_cv < 0.5

            if direction_ok and magnitude_ok:
                criterion   = "weak"
                explanation = (
                    f"Head pair varies (best pair in {pair_freq}/{n_seeds} seeds) but "
                    f"KL range always > 1x and DEER > 0 in {n_pos_eer}/{n_seeds} seeds; "
                    f"KL CV={kl_cv:.2f}, DEER CV={eer_cv:.2f} (both <0.5). "
                    "Mechanism-level claim robust; head-identity claim not."
                )
            else:
                criterion   = "failed"
                explanation = (
                    f"Head pair varies; best pair in only {pair_freq}/{n_seeds} seeds. "
                    f"DEER>0 in {n_pos_eer}/{n_seeds} seeds; "
                    f"KL CV={kl_cv:.2f}, DEER CV={eer_cv:.2f}. "
                    "Mechanism is not seed-stable."
                )

    return {
        "criterion":         criterion,
        "explanation":       explanation,
        "n_seeds":           n_seeds,
        "most_common_pair":  list(most_common_pair),
        "pair_frequency":    pair_freq,
        "most_common_head":  int(most_common_head),
        "head_frequency":    int(head_freq),
        "pair_counts":       {str(list(p)): c for p, c in pair_counts.items()},
        "head_counts":       {str(h): c for h, c in head_counts.items()},
        "kl_range_stats":    describe(kl_ranges),
        "delta_eer_stats":   describe(delta_eers),
        "entropy_stats":     describe(entropies),
    }


def robust_beats_goat(robust_r: dict, goat_r: dict) -> dict:
    return {
        "kl_range_wider":   robust_r["kl_range"]         > goat_r["kl_range"],
        "delta_eer_larger": robust_r["delta_eer"]         > goat_r["delta_eer"],
        "entropy_lower":    robust_r["top2_mean_entropy"] < goat_r["top2_mean_entropy"],
        "robust_kl_range":  robust_r["kl_range"],
        "goat_kl_range":    goat_r["kl_range"],
        "robust_delta_eer": robust_r["delta_eer"],
        "goat_delta_eer":   goat_r["delta_eer"],
        "robust_entropy":   robust_r["top2_mean_entropy"],
        "goat_entropy":     goat_r["top2_mean_entropy"],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_head_identity_table(per_seed: dict, seeds: list, out_path: Path) -> None:
    rows = []
    for seed in seeds:
        for cond in CONDITIONS:
            r = per_seed[str(seed)].get(f"mlaad_{cond}", {})
            if not r:
                continue
            row     = {"seed": seed, "condition": cond}
            ranking = r.get("head_ranking", [])
            for rank_idx in range(6):
                h = ranking[rank_idx] if rank_idx < len(ranking) else ""
                row[f"rank{rank_idx + 1}"] = h
            row["top2"] = str(r.get("top2", []))
            rows.append(row)
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_path}", flush=True)


def write_mechanism_magnitudes(per_seed: dict, seeds: list, out_path: Path) -> None:
    rows = []
    for seed in seeds:
        for cond in CONDITIONS:
            r = per_seed[str(seed)].get(f"mlaad_{cond}", {})
            if not r:
                continue
            rows.append({
                "seed":              seed,
                "condition":         cond,
                "kl_range":          r.get("kl_range", ""),
                "delta_eer":         r.get("delta_eer", ""),
                "baseline_eer":      r.get("baseline_eer", ""),
                "ablated_eer":       r.get("ablated_eer", ""),
                "top2_mean_entropy": r.get("top2_mean_entropy", ""),
                "top2":              str(r.get("top2", [])),
                "best_val_eer":      r.get("best_val_eer", ""),
                "training_time_s":   r.get("training_time_s", ""),
            })
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_path}", flush=True)


def write_paired_comparisons(per_seed: dict, seeds: list, out_path: Path) -> None:
    rows = []
    for seed in seeds:
        rob = per_seed[str(seed)].get("mlaad_robust_goat", {})
        goa = per_seed[str(seed)].get("mlaad_goat", {})
        if not (rob and goa):
            continue
        rows.append({"seed": seed, **robust_beats_goat(rob, goa)})
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Final print
# ---------------------------------------------------------------------------

def print_final_report(per_seed: dict, seeds: list, stability: dict,
                        paired_wins: dict) -> None:
    print("\n" + "=" * 76, flush=True)
    print("SEED-LOCKED REPRODUCIBILITY -- FINAL REPORT", flush=True)
    print("=" * 76, flush=True)

    for cond in CONDITIONS:
        key = f"mlaad_{cond}"
        print(f"\n{'-'*40}", flush=True)
        print(f"Head identity table: {cond}", flush=True)
        print(f"{'Seed':>6}  {'Rank1':>5}  {'Rank2':>5}  {'Rank3':>5}  "
              f"{'Top-2':>10}  {'KL range':>10}  {'DEER':>8}", flush=True)
        for seed in seeds:
            r  = per_seed[str(seed)].get(key, {})
            if not r:
                continue
            rk = r.get("head_ranking", [])
            print(f"  {seed:>4}  "
                  f"h{rk[0] if len(rk)>0 else '?':>4}  "
                  f"h{rk[1] if len(rk)>1 else '?':>4}  "
                  f"h{rk[2] if len(rk)>2 else '?':>4}  "
                  f"  {str(r.get('top2',[])):>10}  "
                  f"  {r.get('kl_range', float('nan')):>8.3f}x"
                  f"  {r.get('delta_eer', float('nan')):>+8.4f}",
                  flush=True)

    print(f"\n{'-'*40}", flush=True)
    print("Distribution across seeds (robust_goat):", flush=True)
    for metric, label in [("kl_range_stats",  "KL range (x)"),
                           ("delta_eer_stats", "Ablation DEER"),
                           ("entropy_stats",   "Top-2 entropy")]:
        s = stability.get(metric, {})
        print(f"  {label:20s}: mean={s.get('mean','?'):.4f}  "
              f"std={s.get('std','?'):.4f}  "
              f"[{s.get('min','?'):.4f}, {s.get('max','?'):.4f}]  "
              f"CV={s.get('cv','?'):.3f}", flush=True)

    print(f"\n{'-'*40}", flush=True)
    n_kl = paired_wins["n_kl_wider"]
    n_ee = paired_wins["n_eer_larger"]
    n_en = paired_wins["n_entropy_lower"]
    n    = paired_wins["n_seeds"]
    print("Robust vs Goat -- per-seed mechanism signature:", flush=True)
    print(f"  KL range wider:       {n_kl}/{n} seeds", flush=True)
    print(f"  Ablation DEER larger: {n_ee}/{n} seeds", flush=True)
    print(f"  Top-2 entropy lower:  {n_en}/{n} seeds", flush=True)

    print(f"\n{'='*76}", flush=True)
    print(f"CRITERION MET: {stability['criterion'].upper()}", flush=True)
    print(f"  {stability['explanation']}", flush=True)
    print(f"  Most common pair: {stability['most_common_pair']}  "
          f"({stability['pair_frequency']}/{stability['n_seeds']} seeds)", flush=True)
    print("=" * 76, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-locked reproducibility study")
    p.add_argument("--n-seeds", type=int, choices=[3, 5], default=5,
                   help="Number of seeds (3=tight compute, 5=full study)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training; analyse existing checkpoints only")
    p.add_argument("--reuse-seed42", action="store_true",
                   help="Reuse original mlaad_{goat,robust_goat}.ckpt as seed42 "
                        "instead of retraining (saves ~44 min)")
    p.add_argument("--epochs", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = ALL_SEEDS[: args.n_seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Pre-registered criteria (before any training or data)
    # ------------------------------------------------------------------
    print(CRITERIA_TEXT, flush=True)
    print(f"\nSeeds:      {seeds}", flush=True)
    print(f"Conditions: {CONDITIONS}", flush=True)

    # ------------------------------------------------------------------
    # 1. Training
    # ------------------------------------------------------------------
    if args.skip_training:
        print("\n[SKIP] --skip-training set; skipping all training.", flush=True)
    else:
        print(f"\n{'-'*60}", flush=True)
        print("PHASE 1: Training", flush=True)
        print(f"{'-'*60}", flush=True)

        if args.reuse_seed42 and 42 in seeds:
            for cond in CONDITIONS:
                orig = find_original_ckpt(cond)
                if orig:
                    dest_name = ckpt_stem(cond, 42) + "-best-reused.ckpt"
                    dest = CKPT_DIR / dest_name
                    if not dest.exists():
                        dest.symlink_to(orig)
                    print(f"  [seed42/{cond}] Reused {orig.name} -> {dest.name}",
                          flush=True)

        for seed in seeds:
            ok = train_seed_parallel(seed, n_epochs=args.epochs)
            if not ok:
                print(f"\nFATAL: training failed for seed={seed}. Aborting.",
                      flush=True)
                sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Verify all checkpoints exist
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}", flush=True)
    print("PHASE 2: Verifying checkpoints", flush=True)
    missing:  list = []
    ckpt_map: dict = {}   # "{cond}_{seed}" -> Path
    for seed in seeds:
        for cond in CONDITIONS:
            stem = ckpt_stem(cond, seed)
            p    = find_best_ckpt(stem)
            if p is None:
                missing.append(stem)
            else:
                ckpt_map[f"{cond}_{seed}"] = p
                print(f"  {stem}: {p.name}", flush=True)
    if missing:
        print(f"\nERROR: Missing checkpoints: {missing}", flush=True)
        sys.exit(1)

    # Sanity (a): all checkpoint hashes distinct
    print("\n[SANITY (a)] Checking checkpoint distinctness ...", flush=True)
    hashes: dict = defaultdict(list)
    for key, path in ckpt_map.items():
        h = md5_ckpt(path)
        hashes[h].append(key)
        print(f"  {key}: {h[:16]}...", flush=True)
    dup_groups = {h: ks for h, ks in hashes.items() if len(ks) > 1}
    if dup_groups:
        print(f"  [SANITY (a) WARN] Duplicate checkpoints: {dup_groups}", flush=True)
    else:
        print(f"  [SANITY (a) PASS] All {len(ckpt_map)} checkpoints have "
              "distinct weights.", flush=True)

    # ------------------------------------------------------------------
    # 3. Analysis
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}", flush=True)
    print("PHASE 3: Head discovery + entropy + ablation", flush=True)
    print(f"{'-'*60}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    in_dist_path = BASELINE_DIR / "test_in_distribution.json"
    if not in_dist_path.exists():
        print(f"ERROR: {in_dist_path} not found.", flush=True)
        sys.exit(1)
    in_dist_records = json.loads(in_dist_path.read_text())
    print(f"In-distribution test records: {len(in_dist_records)}", flush=True)

    # Load cached results if available (allows resuming after interruption)
    per_seed_path = OUT_DIR / "per_seed_results.json"
    per_seed: dict = (json.loads(per_seed_path.read_text())
                      if per_seed_path.exists() else {})

    for seed in seeds:
        seed_str = str(seed)
        if seed_str not in per_seed:
            per_seed[seed_str] = {}
        for cond in CONDITIONS:
            model_key = f"mlaad_{cond}"
            if model_key in per_seed[seed_str]:
                print(f"  [seed={seed}/{cond}] Cached -- skipping.", flush=True)
                continue

            ckpt_path = ckpt_map[f"{cond}_{seed}"]
            print(f"\n  -- Seed {seed} / {cond} --", flush=True)
            t0_total = time.time()
            metrics  = analyse_checkpoint(ckpt_path, in_dist_records, device)

            # Try to read training time from log dir
            log_dir  = LOG_BASE / ckpt_stem(cond, seed)
            fm_path  = log_dir / "final_metrics.json"
            best_val_eer  = float("nan")
            training_time = float("nan")
            if fm_path.exists():
                fm = json.loads(fm_path.read_text())
                best_val_eer  = fm.get("best_val_eer",
                                       fm.get("best_val_EER", float("nan")))
                training_time = fm.get("total_training_time_seconds", float("nan"))

            metrics["best_val_eer"]    = round(float(best_val_eer), 6)
            metrics["training_time_s"] = round(float(training_time), 1)
            metrics["ckpt_md5"]        = md5_ckpt(ckpt_path)[:16]
            metrics["elapsed_total_s"] = round(time.time() - t0_total, 1)

            per_seed[seed_str][model_key] = metrics
            # Persist after each checkpoint so we can resume if interrupted
            _dump(per_seed, per_seed_path)

    print(f"\nWrote: {per_seed_path}", flush=True)

    # Sanity (b): baseline EER < 0.30
    print("\n[SANITY (b)] Checking baseline EER ...", flush=True)
    flagged_b: list = []
    for seed in seeds:
        for cond in CONDITIONS:
            r   = per_seed[str(seed)].get(f"mlaad_{cond}", {})
            eer = r.get("baseline_eer", 1.0)
            if eer >= 0.30:
                flagged_b.append(f"seed={seed}/{cond}: EER={eer:.4f}")
    if flagged_b:
        for f in flagged_b:
            print(f"  [SANITY (b) FLAG] {f} >= 0.30 -- possible training failure!",
                  flush=True)
    else:
        print("  [SANITY (b) PASS] All checkpoints baseline EER < 0.30.", flush=True)

    # Sanity (c): frozen frontend byte-identical
    print("\n[SANITY (c)] Checking frozen WavLM feature identity ...", flush=True)
    for cond in CONDITIONS:
        cond_ckpts = [ckpt_map[f"{cond}_{s}"] for s in seeds]
        check_frozen_frontend(cond_ckpts, device, in_dist_records)

    # Sanity (d): training time consistency
    print("\n[SANITY (d)] Training time consistency ...", flush=True)
    for cond in CONDITIONS:
        times = [per_seed[str(s)].get(f"mlaad_{cond}", {}).get("training_time_s",
                                                                 float("nan"))
                 for s in seeds]
        valid = [t for t in times if not math.isnan(t)]
        if valid:
            cv   = (float(np.std(valid)) / (float(np.mean(valid)) + 1e-10)
                    if len(valid) > 1 else 0.0)
            flag = " [FLAG: high variance]" if cv > 0.2 else ""
            print(f"  {cond}: mean={np.mean(valid)/60:.1f}min  "
                  f"std={np.std(valid)/60:.1f}min  CV={cv:.2f}{flag}", flush=True)
        else:
            print(f"  {cond}: no training-time data available "
                  "(final_metrics.json not found)", flush=True)

    # ------------------------------------------------------------------
    # 4. Aggregate
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}", flush=True)
    print("PHASE 4: Aggregate analysis", flush=True)
    print(f"{'-'*60}", flush=True)

    robust_results = [per_seed[str(s)].get("mlaad_robust_goat", {}) for s in seeds]
    goat_results   = [per_seed[str(s)].get("mlaad_goat", {})        for s in seeds]
    robust_results = [r for r in robust_results if r]
    goat_results   = [r for r in goat_results   if r]

    stability = evaluate_criteria(robust_results, len(seeds))

    paired_data = [
        robust_beats_goat(
            per_seed[str(s)].get("mlaad_robust_goat", {}),
            per_seed[str(s)].get("mlaad_goat", {}),
        )
        for s in seeds
        if (per_seed[str(s)].get("mlaad_robust_goat")
            and per_seed[str(s)].get("mlaad_goat"))
    ]
    paired_wins = {
        "n_seeds":         len(paired_data),
        "n_kl_wider":      sum(1 for d in paired_data if d["kl_range_wider"]),
        "n_eer_larger":    sum(1 for d in paired_data if d["delta_eer_larger"]),
        "n_entropy_lower": sum(1 for d in paired_data if d["entropy_lower"]),
    }

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}", flush=True)
    print("PHASE 5: Writing outputs", flush=True)
    print(f"{'-'*60}", flush=True)

    write_head_identity_table(per_seed, seeds, OUT_DIR / "head_identity_table.csv")
    write_mechanism_magnitudes(per_seed, seeds, OUT_DIR / "mechanism_magnitudes.csv")
    write_paired_comparisons(per_seed, seeds, OUT_DIR / "paired_robust_vs_goat.csv")

    seed_stability = {
        "criterion_met":        stability["criterion"],
        "explanation":          stability["explanation"],
        "n_seeds":              len(seeds),
        "seeds":                seeds,
        "robust_goat_analysis": stability,
        "paired_wins":          paired_wins,
        "sanity_b_flagged":     flagged_b,
        "sanity_a_dups":        list(dup_groups.values()),
    }
    _dump(seed_stability, OUT_DIR / "seed_stability_summary.json")
    print(f"Wrote: {OUT_DIR}/seed_stability_summary.json", flush=True)

    run_config = {
        "seeds":      seeds,
        "n_seeds":    len(seeds),
        "conditions": CONDITIONS,
        "epochs":     args.epochs,
        "skip_training": args.skip_training,
        "reuse_seed42":  args.reuse_seed42,
        "top_k":      TOP_K,
        "batch_size": BATCH_SIZE,
        "checkpoints": {k: str(v) for k, v in ckpt_map.items()},
        "total_training_min": round(
            sum(
                per_seed[str(s)].get(f"mlaad_{c}", {}).get("training_time_s", 0) / 60
                for s in seeds for c in CONDITIONS
            ), 1,
        ),
        "pre_registered_criteria": {
            "strong":   "same pair in >=4/5 robust seeds",
            "moderate": "same pair in >=3/5, or one head in >=4/5",
            "weak":     "effect consistent in direction and within +-50% CV",
            "failed":   "none of the above",
        },
    }
    _dump(run_config, OUT_DIR / "run_config.json")
    print(f"Wrote: {OUT_DIR}/run_config.json", flush=True)

    # ------------------------------------------------------------------
    # 6. Final report
    # ------------------------------------------------------------------
    print_final_report(per_seed, seeds, stability, paired_wins)
    print(f"\nAll outputs in: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
