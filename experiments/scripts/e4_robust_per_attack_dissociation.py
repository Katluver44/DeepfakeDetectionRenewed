#!/usr/bin/env python3
"""
e4_robust_per_attack_dissociation.py
=====================================
Per-attack TTS/VC dissociation under single- and pair-head ablation on
robust_goat.ckpt.

Re-runs E4 on robust_goat.ckpt with finer-grained ablation conditions and
per-attack EER reporting, testing whether h0 and h4 specialise in different
attack types (TTS vs VC) or are functionally redundant.

Ablation conditions (GAT layer 0, zero mode):
  baseline       — no ablation
  ablate_h0      — zero {h0} only
  ablate_h4      — zero {h4} only
  ablate_h0_h4   — zero {h0, h4}  ← consistency check vs published robust E4
  random_single  — zero one head sampled from {h1,h2,h3,h5}, fixed seed
  random_pair    — zero one pair sampled from {h1,h2,h3,h5}, fixed seed

Head count, layer count, and attack-system metadata are read from the model
and dataset — not hardcoded.

TTS/VC classification is derived from gat_e7.ROUTING_SYSTEMS / SKIP_SYSTEMS
(the repo-canonical source, itself derived from ASVspoof 2019 official docs).
The script fails loudly if any attack system remains unclassified.

Outputs → experiments/results/gat_l0_attention_followups/e4_robust_per_attack_dissociation/
  per_attack_per_condition.csv     — long-form: attack × condition × EER × CI
  dissociation_per_attack.json     — per-attack dissociation scores with CIs
  dissociation_tts_vs_vc.json      — aggregate TTS/VC dissociation, perm p-value
  run_config.json                  — checkpoint hash, eval set, seeds, mapping
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
REPO_ROOT = EXP_DIR.parent
OUT_DIR   = EXP_DIR / "results" / "gat_l0_attention_followups" / \
            "e4_robust_per_attack_dissociation"
PRIOR_E4_CSV = EXP_DIR / "results" / "gat_l0_attention_followups" / \
               "ablation_summary.csv"   # head_ablation.py output on robust_goat
CKPT      = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import existing utilities (no logic duplication) ─────────────────────────
import head_ablation as ha      # load_model, run_eval, compute_eer, BalancedDataset
import gat_e3_e4   as e4lib     # per_system_eer, bootstrap_eer_ci

# ── Constants (no hardcoded model/dataset values — all read at runtime) ───────
HF_DATASET   = "Bisher/ASVspoof_2019_LA"
CACHE_DIR    = REPO_ROOT / "data" / "asvspoof_2019_la"
CRITICAL_HEADS = frozenset({0, 4})   # from prior gat_l0_attention analysis
N_PER_CLASS  = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8))
SEED         = 42
N_BOOTSTRAP  = 1_000
N_PERM       = 1_000
ABLATION_MODE    = "zero"
ABLATION_LAYER   = 0
PRIOR_EER_TOL    = 0.01   # ±1pp tolerance for pipeline-parity sanity check


# ── JSON encoder ─────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, (frozenset, set)): return sorted(obj)
        return super().default(obj)


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))
    print(f"  Saved: {path}")


# ── TTS/VC mapping ────────────────────────────────────────────────────────────

def load_tts_vc_mapping(attack_systems: list[str]) -> dict[str, str]:
    """
    Derives TTS/VC classification from gat_e7.ROUTING_SYSTEMS / SKIP_SYSTEMS
    (the repo-canonical source, itself derived from ASVspoof 2019 official
    system descriptions).

    For any attack system absent from both sets (e.g. A02, excluded from gat_e7
    as "ambiguous mechanism"), falls back to a small lookup table derived from
    the ASVspoof 2019 paper (Table 2). Raises ValueError for truly unknown IDs.

    Returns {system_id: "TTS" | "VC"}.
    """
    import gat_e7 as e7
    routing = e7.ROUTING_SYSTEMS   # {"A01", "A03", "A04"}
    skip    = e7.SKIP_SYSTEMS      # {"A05", "A06"}

    # Fallback for systems gat_e7 deliberately excluded (e.g. A02).
    # Source: ASVspoof 2019 LA official system descriptions, Table 2.
    asvspoof_fallback: dict[str, str] = {
        "A02": "TTS",   # WaveNet-based TTS (excluded from E7 cluster as ambiguous)
    }

    mapping: dict[str, str] = {}
    for sid in sorted(attack_systems):
        if sid in routing:
            mapping[sid] = "TTS"
        elif sid in skip:
            mapping[sid] = "VC"
        elif sid in asvspoof_fallback:
            mapping[sid] = asvspoof_fallback[sid]
            print(f"  [WARN] {sid}: absent from gat_e7 canonical sets; "
                  f"classified as {asvspoof_fallback[sid]} "
                  f"per ASVspoof 2019 official description (WaveNet TTS). "
                  f"Note: gat_e7 excludes A02 as ambiguous in cluster analysis.")
        else:
            raise ValueError(
                f"FATAL: attack system {sid!r} has no TTS/VC classification in "
                f"gat_e7.ROUTING_SYSTEMS, gat_e7.SKIP_SYSTEMS, or the ASVspoof "
                f"2019 fallback table. Add it explicitly and re-run."
            )
    return mapping


# ── Control-head selection ────────────────────────────────────────────────────

def pick_random_single(control_pool: list[int], seed: int) -> frozenset:
    """One head sampled from control_pool (heads ≠ CRITICAL_HEADS), fixed seed."""
    rng = random.Random(seed)
    return frozenset({rng.choice(control_pool)})


def pick_random_pair(control_pool: list[int], n: int, seed: int) -> frozenset:
    """One size-n pair sampled from C(control_pool, n), fixed seed."""
    all_pairs = [frozenset(c) for c in combinations(control_pool, n)]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[0]


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


def load_prior_robust_e4_eers() -> dict[str, float] | None:
    """
    Loads per-condition pooled EER from the existing ablation_summary.csv
    produced by head_ablation.py (robust_goat.ckpt).
    Returns {config_name: eer} or None if file not found.
    """
    if not PRIOR_E4_CSV.exists():
        return None
    import csv as _csv
    out: dict[str, float] = {}
    with open(PRIOR_E4_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            out[row["config"]] = float(row["eer"])
    return out


# ── Bootstrap CI for dissociation score ──────────────────────────────────────

def bootstrap_dissociation_ci(
    records_baseline: list[dict],
    records_h0: list[dict],
    records_h4: list[dict],
    attack_sys: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for dissociation_score[attack] = ΔEER_h0only − ΔEER_h4only.
    Resamples bonafide + attack records jointly across all three conditions so
    the bootstrap preserves within-sample correlation across ablation conditions.
    """
    bon_b  = [r for r in records_baseline if r["system_id"] == "-"]
    bon_0  = [r for r in records_h0       if r["system_id"] == "-"]
    bon_4  = [r for r in records_h4       if r["system_id"] == "-"]
    atk_b  = [r for r in records_baseline if r["system_id"] == attack_sys]
    atk_0  = [r for r in records_h0       if r["system_id"] == attack_sys]
    atk_4  = [r for r in records_h4       if r["system_id"] == attack_sys]

    comb_b = bon_b + atk_b
    comb_0 = bon_0 + atk_0
    comb_4 = bon_4 + atk_4
    n = len(comb_b)

    def _eer(recs: list[dict]) -> float:
        labels = np.array([r["label"] for r in recs])
        logits = np.array([r["logit"] for r in recs])
        scores = 1.0 / (1.0 + np.exp(-logits))
        if labels.sum() == 0 or labels.sum() == n:
            return float("nan")
        return ha.compute_eer(labels, scores)

    boot_diss: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        eb  = _eer([comb_b[i] for i in idx])
        e0  = _eer([comb_0[i] for i in idx])
        e4  = _eer([comb_4[i] for i in idx])
        if any(np.isnan(x) for x in (eb, e0, e4)):
            continue
        boot_diss.append((e0 - eb) - (e4 - eb))

    if boot_diss:
        return (float(np.percentile(boot_diss, 2.5)),
                float(np.percentile(boot_diss, 97.5)))
    return (float("nan"), float("nan"))


def bootstrap_aggregate_dissociation_ci(
    records_baseline: list[dict],
    records_h0: list[dict],
    records_h4: list[dict],
    systems: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for mean dissociation over a group of attack systems.
    Resamples across systems within the group (case-resampling bootstrap).
    """
    if not systems:
        return (float("nan"), float("nan"))

    # Get per-system dissociation from each bootstrap resample of per-attack data
    # We resample systems with replacement within the group.
    diss_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    for s in systems:
        bon_b = [r for r in records_baseline if r["system_id"] == "-"]
        bon_0 = [r for r in records_h0       if r["system_id"] == "-"]
        bon_4 = [r for r in records_h4       if r["system_id"] == "-"]
        atk_b = [r for r in records_baseline if r["system_id"] == s]
        atk_0 = [r for r in records_h0       if r["system_id"] == s]
        atk_4 = [r for r in records_h4       if r["system_id"] == s]
        comb_b = bon_b + atk_b
        comb_0 = bon_0 + atk_0
        comb_4 = bon_4 + atk_4
        nc = len(comb_b)

        def _eer(recs: list[dict]) -> float:
            lbls = np.array([r["label"] for r in recs])
            logi = np.array([r["logit"] for r in recs])
            scrs = 1.0 / (1.0 + np.exp(-logi))
            if lbls.sum() == 0 or lbls.sum() == len(lbls):
                return float("nan")
            return ha.compute_eer(lbls, scrs)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, nc, size=nc)
            eb = _eer([comb_b[i] for i in idx])
            e0 = _eer([comb_0[i] for i in idx])
            e4 = _eer([comb_4[i] for i in idx])
            if any(np.isnan(x) for x in (eb, e0, e4)):
                diss_by_sys[s].append(float("nan"))
            else:
                diss_by_sys[s].append((e0 - eb) - (e4 - eb))

    # Aggregate: mean dissociation over the group for each bootstrap replicate
    n_sys = len(systems)
    agg_boot: list[float] = []
    for bi in range(n_bootstrap):
        vals = [diss_by_sys[s][bi] for s in systems]
        if any(np.isnan(v) for v in vals):
            continue
        agg_boot.append(float(np.mean(vals)))

    if agg_boot:
        return (float(np.percentile(agg_boot, 2.5)),
                float(np.percentile(agg_boot, 97.5)))
    return (float("nan"), float("nan"))


# ── Permutation test ──────────────────────────────────────────────────────────

def permutation_test_tts_vc_dissociation(
    records_baseline: list[dict],
    records_h0: list[dict],
    records_h4: list[dict],
    tts_systems: list[str],
    vc_systems: list[str],
    attack_systems: list[str],
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> tuple[float, float, list[float]]:
    """
    Permutation test for (TTS aggregate dissociation) − (VC aggregate dissociation).

    Null hypothesis: h0 and h4 are interchangeable (do the same job for each
    sample). Under the null, randomly swapping the h0/h4 delta-logit assignments
    per sample should produce the same aggregate TTS-vs-VC dissociation.

    Implementation: for each permutation, each sample independently has its
    Δlogit_h0 and Δlogit_h4 values swapped with probability 0.5. The per-attack
    EERs and dissociation scores are recomputed from the permuted logits.

    Returns (D_observed, two_sided_p_value, null_D_list).
    """
    rng = np.random.default_rng(seed)

    n = len(records_baseline)
    assert len(records_h0) == n and len(records_h4) == n, (
        f"Record counts differ: baseline={n}, h0={len(records_h0)}, h4={len(records_h4)}")

    base_logits = np.array([r["logit"] for r in records_baseline])
    h0_logits   = np.array([r["logit"] for r in records_h0])
    h4_logits   = np.array([r["logit"] for r in records_h4])
    labels      = np.array([r["label"] for r in records_baseline])
    sys_ids     = [r["system_id"] for r in records_baseline]

    delta_h0 = h0_logits - base_logits
    delta_h4 = h4_logits - base_logits

    def _compute_D(d0: np.ndarray, d4: np.ndarray) -> float:
        recs_b  = [{"label": int(labels[i]), "system_id": sys_ids[i],
                    "logit": float(base_logits[i])} for i in range(n)]
        recs_h0 = [{"label": int(labels[i]), "system_id": sys_ids[i],
                    "logit": float(base_logits[i] + d0[i])} for i in range(n)]
        recs_h4 = [{"label": int(labels[i]), "system_id": sys_ids[i],
                    "logit": float(base_logits[i] + d4[i])} for i in range(n)]

        eer_b  = e4lib.per_system_eer(recs_b,  attack_systems)
        eer_h0 = e4lib.per_system_eer(recs_h0, attack_systems)
        eer_h4 = e4lib.per_system_eer(recs_h4, attack_systems)

        diss = {s: (eer_h0[s] - eer_b[s]) - (eer_h4[s] - eer_b[s])
                for s in attack_systems}

        tts_d = (float(np.mean([diss[s] for s in tts_systems if s in diss]))
                 if tts_systems else 0.0)
        vc_d  = (float(np.mean([diss[s] for s in vc_systems  if s in diss]))
                 if vc_systems  else 0.0)
        return tts_d - vc_d

    D_obs = _compute_D(delta_h0, delta_h4)

    D_null: list[float] = []
    for _ in range(n_perm):
        swap    = rng.integers(0, 2, size=n).astype(bool)
        d0_perm = np.where(swap, delta_h4, delta_h0)
        d4_perm = np.where(swap, delta_h0, delta_h4)
        D_null.append(_compute_D(d0_perm, d4_perm))

    p = float(np.mean(np.abs(D_null) >= abs(D_obs)))
    return D_obs, p, D_null


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT}")

    # ── Load model ────────────────────────────────────────────────────────────
    ha.patch_phoneme_loader()
    lit = ha.load_model(CKPT, device)

    # ── Read layer count and head count from model (never hardcoded) ──────────
    gat_net = lit.model.GAT.gat_net
    N_LAYERS = len(gat_net)
    NH       = gat_net[ABLATION_LAYER].num_of_heads
    print(f"  Model: {N_LAYERS} GAT layers, {NH} heads at layer {ABLATION_LAYER}")

    # ── Define conditions ─────────────────────────────────────────────────────
    control_pool = sorted(frozenset(range(NH)) - CRITICAL_HEADS)
    random_single_head = pick_random_single(control_pool, seed=SEED)
    random_pair_head   = pick_random_pair(control_pool, n=len(CRITICAL_HEADS), seed=SEED)

    print(f"  critical heads:     {sorted(CRITICAL_HEADS)}")
    print(f"  control pool:       {control_pool}")
    print(f"  random_single head: {sorted(random_single_head)}")
    print(f"  random_pair  heads: {sorted(random_pair_head)}")

    conditions: list[tuple[str, frozenset]] = [
        ("baseline",      frozenset()),
        ("ablate_h0",     frozenset({0})),
        ("ablate_h4",     frozenset({4})),
        ("ablate_h0_h4",  CRITICAL_HEADS),
        ("random_single", random_single_head),
        ("random_pair",   random_pair_head),
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
    print(f"  eval set hash:  {ds_hash}")
    print(f"  attack systems: {attack_sids}")

    # ── TTS/VC mapping (read from gat_e7 utility, not hardcoded) ─────────────
    print("\n--- TTS/VC attack mapping (from gat_e7.ROUTING_SYSTEMS/SKIP_SYSTEMS) ---")
    tts_vc_map = load_tts_vc_mapping(attack_sids)
    tts_systems = sorted(s for s, t in tts_vc_map.items() if t == "TTS")
    vc_systems  = sorted(s for s, t in tts_vc_map.items() if t == "VC")
    for sid in attack_sids:
        print(f"  {sid}: {tts_vc_map[sid]}")
    # Sanity: all attack systems must be mapped (load_tts_vc_mapping raises on failure)
    assert set(tts_vc_map.keys()) == set(attack_sids), (
        f"FATAL: TTS/VC mapping incomplete — mapped: {sorted(tts_vc_map)}, "
        f"expected: {attack_sids}")
    print(f"  TTS systems: {tts_systems}")
    print(f"  VC  systems: {vc_systems}")

    # ── Determinism check ─────────────────────────────────────────────────────
    print("\n--- Determinism check ---")
    _batch = next(iter(loader))
    _audio = _batch["audio"].to(device)
    _num_f = torch.full((_audio.shape[0],), ha.NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        _hs, _pids = ha.run_frozen_frontend(_audio, lit.model, device)
        _lg1 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
        _lg2 = lit.model.encoder_and_GAT(_hs, _num_f, _pids)[5].cpu()
    _diff = (_lg1 - _lg2).abs().max().item()
    assert _diff < 1e-4, f"Model not deterministic: max logit diff={_diff:.2e}"
    print(f"  Determinism PASSED: max logit diff = {_diff:.2e}")
    ha.sanity_layer_untouched(lit)

    # ── Run all conditions ────────────────────────────────────────────────────
    print("\n--- Running ablation conditions ---")
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

    # ── Per-attack EER + bootstrap CIs ───────────────────────────────────────
    print("\n--- Per-attack EER + bootstrap CIs ---")
    eer_by_cond: dict[str, dict[str, float]] = {}
    ci_by_cond:  dict[str, dict[str, tuple]] = {}
    rng = np.random.default_rng(SEED)

    for cond_name, _ in conditions:
        eer_by_cond[cond_name] = e4lib.per_system_eer(all_records[cond_name], attack_sids)
        ci_by_cond[cond_name]  = e4lib.bootstrap_eer_ci(
            all_records[cond_name], attack_sids, n_bootstrap=N_BOOTSTRAP, rng=rng)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n--- Sanity checks ---")
    sanity_results: dict[str, str] = {}

    # (a) "ablate_h0_h4" pooled EER matches previously published robust E4 result
    prior = load_prior_robust_e4_eers()
    if prior is not None and "h0_h4_zero" in prior:
        ref_h0h4 = prior["h0_h4_zero"]
        diff_a = abs(pooled_eers["ablate_h0_h4"] - ref_h0h4)
        if diff_a <= PRIOR_EER_TOL:
            msg = (f"OK — ablate_h0_h4 EER {pooled_eers['ablate_h0_h4']:.4f} matches "
                   f"prior h0_h4_zero {ref_h0h4:.4f} (diff={diff_a:.4f}) — pipeline parity confirmed")
        else:
            msg = (f"WARN — ablate_h0_h4 EER {pooled_eers['ablate_h0_h4']:.4f} vs "
                   f"prior h0_h4_zero {ref_h0h4:.4f} (diff={diff_a:.4f} > tol={PRIOR_EER_TOL})")
        print(f"  (a) {msg}")
        sanity_results["ablate_h0h4_parity"] = msg
    else:
        msg = (f"SKIP — ablation_summary.csv not found or lacks h0_h4_zero; "
               f"ablate_h0_h4 EER={pooled_eers['ablate_h0_h4']:.4f}")
        print(f"  (a) {msg}")
        sanity_results["ablate_h0h4_parity"] = msg

    # (b) Baseline pooled EER matches prior run
    if prior is not None and "baseline" in prior:
        ref_base = prior["baseline"]
        diff_b = abs(pooled_eers["baseline"] - ref_base)
        if diff_b <= PRIOR_EER_TOL:
            msg = (f"OK — baseline EER {pooled_eers['baseline']:.4f} matches "
                   f"prior baseline {ref_base:.4f} (diff={diff_b:.4f}) — eval pipeline parity")
        else:
            msg = (f"WARN — baseline EER {pooled_eers['baseline']:.4f} vs "
                   f"prior baseline {ref_base:.4f} (diff={diff_b:.4f} > tol={PRIOR_EER_TOL})")
        print(f"  (b) {msg}")
        sanity_results["baseline_parity"] = msg
    else:
        msg = f"SKIP — prior baseline EER unavailable; this-run baseline={pooled_eers['baseline']:.4f}"
        print(f"  (b) {msg}")
        sanity_results["baseline_parity"] = msg

    # (c) Random controls produce ΔEERs not wildly larger than {h0,h4} effect
    delta_h0h4    = abs(pooled_eers["ablate_h0_h4"] - pooled_eers["baseline"])
    delta_single  = abs(pooled_eers["random_single"] - pooled_eers["baseline"])
    delta_pair    = abs(pooled_eers["random_pair"]   - pooled_eers["baseline"])
    CTRL_RATIO_WARN = 2.0
    if delta_h0h4 > 0:
        r_single = delta_single / delta_h0h4
        r_pair   = delta_pair   / delta_h0h4
        if r_single > CTRL_RATIO_WARN or r_pair > CTRL_RATIO_WARN:
            msg = (f"WARN — random controls show unusually large ΔEERs relative to "
                   f"ablate_h0_h4: |Δ_single|={delta_single:.4f} (ratio={r_single:.2f}), "
                   f"|Δ_pair|={delta_pair:.4f} (ratio={r_pair:.2f}) vs "
                   f"|Δ_h0h4|={delta_h0h4:.4f}; check for bugs or model anomaly")
        else:
            msg = (f"OK — random control ΔEERs in line with null: "
                   f"|Δ_single|={delta_single:.4f} ({r_single:.2f}×), "
                   f"|Δ_pair|={delta_pair:.4f} ({r_pair:.2f}×) vs "
                   f"|Δ_h0h4|={delta_h0h4:.4f}")
    else:
        msg = (f"NOTE — ablate_h0_h4 ΔEER≈0; cannot compute ratio. "
               f"|Δ_single|={delta_single:.4f}, |Δ_pair|={delta_pair:.4f}")
    print(f"  (c) {msg}")
    sanity_results["random_ctrl_ratio"] = msg

    sanity_pass = all(not s.startswith(("FAIL", "WARN")) for s in sanity_results.values())

    # ── Per-attack EER table ──────────────────────────────────────────────────
    print(f"\n{'system':8s}  {'baseline':>9s}  {'ablate_h0':>11s}  "
          f"{'ablate_h4':>11s}  {'ablate_h0_h4':>14s}  "
          f"{'rand_single':>13s}  {'rand_pair':>11s}")
    print("-" * 90)
    for sid in sorted(attack_sids):
        b  = eer_by_cond["baseline"][sid]
        h0 = eer_by_cond["ablate_h0"][sid]
        h4 = eer_by_cond["ablate_h4"][sid]
        hh = eer_by_cond["ablate_h0_h4"][sid]
        rs = eer_by_cond["random_single"][sid]
        rp = eer_by_cond["random_pair"][sid]
        tvc = tts_vc_map.get(sid, "?")
        print(f"  {sid:6s}[{tvc}]  "
              f"{b:.4f}  "
              f"{h0:.4f}(Δ{h0-b:+.3f})  "
              f"{h4:.4f}(Δ{h4-b:+.3f})  "
              f"{hh:.4f}(Δ{hh-b:+.3f})  "
              f"{rs:.4f}(Δ{rs-b:+.3f})  "
              f"{rp:.4f}(Δ{rp-b:+.3f})")

    # ── Dissociation analysis ─────────────────────────────────────────────────
    print("\n--- Dissociation analysis ---")

    diss_per_atk: dict[str, dict] = {}
    rng_diss = np.random.default_rng(SEED)   # fresh rng for dissociation CIs

    for sid in sorted(attack_sids):
        b  = eer_by_cond["baseline"][sid]
        h0 = eer_by_cond["ablate_h0"][sid]
        h4 = eer_by_cond["ablate_h4"][sid]
        d0 = h0 - b
        d4 = h4 - b
        disc = d0 - d4   # positive → h0 matters more; negative → h4 matters more

        ci = bootstrap_dissociation_ci(
            all_records["baseline"], all_records["ablate_h0"], all_records["ablate_h4"],
            sid, N_BOOTSTRAP, rng_diss)

        diss_per_atk[sid] = {
            "delta_eer_h0_only": float(d0),
            "delta_eer_h4_only": float(d4),
            "dissociation_score": float(disc),
            "ci_low":  float(ci[0]),
            "ci_high": float(ci[1]),
            "tts_vc":  tts_vc_map[sid],
        }
        direction = "h0>h4" if disc > 0 else "h4>h0" if disc < 0 else "equal"
        print(f"  {sid:6s}[{tts_vc_map[sid]}]  "
              f"Δh0={d0:+.4f}  Δh4={d4:+.4f}  "
              f"dissociation={disc:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  ({direction})")

    # TTS and VC aggregate dissociation
    tts_diss_vals = [diss_per_atk[s]["dissociation_score"] for s in tts_systems]
    vc_diss_vals  = [diss_per_atk[s]["dissociation_score"] for s in vc_systems]
    tts_agg_diss  = float(np.mean(tts_diss_vals)) if tts_diss_vals else float("nan")
    vc_agg_diss   = float(np.mean(vc_diss_vals))  if vc_diss_vals  else float("nan")
    diff_agg      = float(tts_agg_diss - vc_agg_diss) if not any(np.isnan([tts_agg_diss, vc_agg_diss])) else float("nan")

    print(f"\n  TTS aggregate dissociation: {tts_agg_diss:+.4f}  (systems: {tts_systems})")
    print(f"  VC  aggregate dissociation: {vc_agg_diss:+.4f}  (systems: {vc_systems})")
    print(f"  Difference (TTS − VC):      {diff_agg:+.4f}")

    # Bootstrap CIs on aggregate dissociation
    print("\n  Computing bootstrap CIs for aggregate dissociation...")
    rng_agg = np.random.default_rng(SEED)
    tts_agg_ci = bootstrap_aggregate_dissociation_ci(
        all_records["baseline"], all_records["ablate_h0"],
        all_records["ablate_h4"], tts_systems, N_BOOTSTRAP, rng_agg)
    vc_agg_ci  = bootstrap_aggregate_dissociation_ci(
        all_records["baseline"], all_records["ablate_h0"],
        all_records["ablate_h4"], vc_systems,  N_BOOTSTRAP, rng_agg)

    print(f"  TTS CI: [{tts_agg_ci[0]:+.4f}, {tts_agg_ci[1]:+.4f}]")
    print(f"  VC  CI: [{vc_agg_ci[0]:+.4f},  {vc_agg_ci[1]:+.4f}]")

    # ── Permutation test ──────────────────────────────────────────────────────
    print(f"\n--- Permutation test (N={N_PERM} permutations) ---")
    print("  Shuffling h0/h4 delta-logit assignments per sample...")
    t_perm = time.time()
    D_obs, p_val, D_null = permutation_test_tts_vc_dissociation(
        all_records["baseline"], all_records["ablate_h0"], all_records["ablate_h4"],
        tts_systems, vc_systems, attack_sids, n_perm=N_PERM, seed=SEED)
    print(f"  Done ({time.time()-t_perm:.1f}s)")
    print(f"  Observed D (TTS_diss − VC_diss) = {D_obs:+.4f}")
    print(f"  Permutation p-value (two-sided)  = {p_val:.4f}")

    D_null_arr = np.array(D_null)
    print(f"  Null distribution: "
          f"mean={float(D_null_arr.mean()):+.4f}  "
          f"std={float(D_null_arr.std()):+.4f}  "
          f"range=[{float(D_null_arr.min()):+.4f}, {float(D_null_arr.max()):+.4f}]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n--- Saving outputs ---")

    # 1. per_attack_per_condition.csv (long-form)
    csv_rows: list[dict] = []
    for sid in sorted(attack_sids):
        for cond_name, _ in conditions:
            eer  = eer_by_cond[cond_name][sid]
            ci   = ci_by_cond[cond_name][sid]
            csv_rows.append({
                "attack":    sid,
                "tts_vc":    tts_vc_map[sid],
                "condition": cond_name,
                "eer":       round(eer, 6),
                "ci_low":    round(ci[0], 6),
                "ci_high":   round(ci[1], 6),
            })
    # Pooled rows (no CIs)
    _nan = float("nan")
    for cond_name, _ in conditions:
        csv_rows.append({
            "attack":    "pooled",
            "tts_vc":    "n/a",
            "condition": cond_name,
            "eer":       round(pooled_eers[cond_name], 6),
            "ci_low":    _nan,
            "ci_high":   _nan,
        })
    csv_path = OUT_DIR / "per_attack_per_condition.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "tts_vc", "condition", "eer", "ci_low", "ci_high"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"  Saved: {csv_path}")

    # 2. dissociation_per_attack.json
    _dump(diss_per_atk, OUT_DIR / "dissociation_per_attack.json")

    # 3. dissociation_tts_vs_vc.json
    tts_vc_out = {
        "tts_aggregate": {
            "systems":           tts_systems,
            "dissociation_score": tts_agg_diss,
            "ci_low":            float(tts_agg_ci[0]),
            "ci_high":           float(tts_agg_ci[1]),
        },
        "vc_aggregate": {
            "systems":           vc_systems,
            "dissociation_score": vc_agg_diss,
            "ci_low":            float(vc_agg_ci[0]),
            "ci_high":           float(vc_agg_ci[1]),
        },
        "difference": {
            "score":               diff_agg,
            "interpretation":     ("TTS h0-dominant" if diff_agg > 0 else
                                   "VC h0-dominant"  if diff_agg < 0 else "equal"),
            "permutation_p_value": p_val,
            "n_permutations":      N_PERM,
            "D_observed":          D_obs,
            "null_mean":           float(D_null_arr.mean()),
            "null_std":            float(D_null_arr.std()),
            "null_p5":             float(np.percentile(D_null_arr, 5)),
            "null_p95":            float(np.percentile(D_null_arr, 95)),
        },
    }
    _dump(tts_vc_out, OUT_DIR / "dissociation_tts_vs_vc.json")

    # 4. run_config.json
    print("\n  Computing checkpoint hash...")
    ckpt_sha = ckpt_hash(CKPT)
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
            "layer":              ABLATION_LAYER,
            "n_layers_total":     N_LAYERS,
            "mode":               ABLATION_MODE,
            "critical_heads":     sorted(CRITICAL_HEADS),
            "control_pool":       control_pool,
            "random_single_head": sorted(random_single_head),
            "random_pair_heads":  sorted(random_pair_head),
            "nh":                 NH,
        },
        "tts_vc_mapping": tts_vc_map,
        "tts_vc_source":  "gat_e7.ROUTING_SYSTEMS / SKIP_SYSTEMS (ASVspoof 2019 official)",
        "tts_systems":    tts_systems,
        "vc_systems":     vc_systems,
        "seed":           SEED,
        "n_bootstrap":    N_BOOTSTRAP,
        "n_permutations": N_PERM,
        "sanity_pass":    sanity_pass,
        "sanity_details": sanity_results,
    }
    _dump(run_cfg, OUT_DIR / "run_config.json")

    # ── Final summary print ───────────────────────────────────────────────────
    b  = pooled_eers["baseline"]
    h0 = pooled_eers["ablate_h0"]
    h4 = pooled_eers["ablate_h4"]
    hh = pooled_eers["ablate_h0_h4"]

    print(f"\n{'='*72}")
    print(f"Results: {OUT_DIR}/")
    print()

    print("TTS/VC attack mapping used:")
    for sid in attack_sids:
        print(f"  {sid}: {tts_vc_map[sid]}")
    print()

    print("Pooled EER (4-row summary):")
    print(f"  {'condition':20s}  {'EER':>8s}  {'ΔEER':>8s}")
    print(f"  {'-'*40}")
    for cname, val in [("baseline", b), ("ablate_h0", h0),
                       ("ablate_h4", h4), ("ablate_h0_h4", hh)]:
        print(f"  {cname:20s}  {val:.4f}   {val-b:+.4f}")
    print()

    print("TTS-aggregate ΔEER (h0 vs h4 ablation):")
    print(f"  Δh0: {float(np.mean([eer_by_cond['ablate_h0'][s] - eer_by_cond['baseline'][s] for s in tts_systems])):+.4f}  "
          f"Δh4: {float(np.mean([eer_by_cond['ablate_h4'][s] - eer_by_cond['baseline'][s] for s in tts_systems])):+.4f}")
    print("VC-aggregate ΔEER (h0 vs h4 ablation):")
    print(f"  Δh0: {float(np.mean([eer_by_cond['ablate_h0'][s] - eer_by_cond['baseline'][s] for s in vc_systems])):+.4f}  "
          f"Δh4: {float(np.mean([eer_by_cond['ablate_h4'][s] - eer_by_cond['baseline'][s] for s in vc_systems])):+.4f}")
    print()

    print(f"TTS aggregate dissociation:    {tts_agg_diss:+.4f}  "
          f"CI [{tts_agg_ci[0]:+.4f}, {tts_agg_ci[1]:+.4f}]")
    print(f"VC  aggregate dissociation:    {vc_agg_diss:+.4f}  "
          f"CI [{vc_agg_ci[0]:+.4f},  {vc_agg_ci[1]:+.4f}]")
    print(f"Difference (TTS − VC):         {diff_agg:+.4f}")
    print(f"Permutation p-value (two-sid): {p_val:.4f}  (N={N_PERM})")
    print()

    print(f"Sanity: {'PASS' if sanity_pass else 'FAIL/WARN — review output above'}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
