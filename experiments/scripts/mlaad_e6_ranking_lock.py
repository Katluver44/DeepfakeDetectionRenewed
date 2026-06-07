#!/usr/bin/env python3
"""
mlaad_e6_ranking_lock.py
========================
E6 analog for MLAAD: cross-seed agreement on per-attack-system EER ranking.

Evaluates 4 robust_goat and 5 goat seed checkpoints on the in-distribution
test split, computes per-attack EER, and tests cross-seed agreement.

Pre-registered criteria (stated before running):
  Strong   : mean pairwise Spearman rho >= 0.75 system-level,
             OR >= 0.85 family-level, AND <= 2 category flips.
  Moderate : mean rho in [0.55, 0.75] system-level OR [0.65, 0.85] family-level,
             AND <= 4 category flips.
  Weak     : mean rho < 0.55 system-level. Ranking unstable across seeds.

Comparison finding: Does robust_goat have more stable rankings than goat?
  Robust rho_bar - goat rho_bar with bootstrap CI.
  If robust > goat by >=0.05 with non-overlapping CI -> "augmentation stabilizes ranking".

ASVspoof E6 reference: rho = 0.9429 (both cross-seed pairs), seed1 vs seed2 = 1.000,
  zero category flips. 6 systems, permutation-test p-values (not EER).

Usage:
    venv/bin/python experiments/scripts/mlaad_e6_ranking_lock.py

Outputs:
    experiments/results/mlaad/e6_ranking_lock/
        per_seed_per_attack_eer.csv
        pairwise_spearman.json
        family_level_agreement.json
        category_flips.json
        robust_vs_goat_comparison.json
        e6_analog_verdict.json
        run_config.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from argparse import Namespace
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "e6_ranking_lock"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# torch.load compat
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

# ─── Config ──────────────────────────────────────────────────────────────────
# robust_goat: exclude seed 456 (best at epoch 0 — diverged)
ROBUST_SEEDS = [42, 123, 789, 1024]
# goat: seed 456 trained normally (epoch 5, val-eer=0.3030) — include all 5
GOAT_SEEDS   = [42, 123, 456, 789, 1024]

MIN_SAMPLES = 30   # pre-registered threshold; NOTE: max in test split is ~25
                   # so this will drop all systems. We fall back to MIN_SAMPLES=5
                   # and document this prominently.
BATCH_SIZE  = 32
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
TOP_QUARTILE = 0.25   # top 25% = hardest systems

# ASVspoof E6 reference numbers (from experiments/results/gat_attn_graphs/e6/)
ASVSPOOF_E6 = {
    "rho_seed3_vs_seed1": 0.9429,
    "rho_seed3_vs_seed2": 0.9429,
    "rho_seed1_vs_seed2": 1.0000,
    "n_category_flips":   0,
    "n_systems":          6,
    "method":             "permutation_test_p_values",
    "note":               "6 systems A01-A06; ρ on family-permutation p-values, not EER",
}

# ─── TTS family map ──────────────────────────────────────────────────────────
# Derived from system names. Groups version variants together.
# Phase 0 metadata.json has no family field; this mapping is based on naming.
def build_family_map(systems: list[str]) -> dict[str, str]:
    """Map each system name to its architectural family."""
    rules = [
        # prefix/substring -> family label (checked in order; first match wins)
        ("Index-TTS",               "IndexTTS"),
        ("Kitten-TTS",              "KittenTTS"),
        ("Llasa",                   "Llasa"),
        ("VoxCPM",                  "VoxCPM"),
        ("Nari Dia",                "NariDia"),
        ("minimax_speech",          "MiniMax"),
        ("MiniCPM",                 "MiniCPM"),
        ("Microsoft VibeVoice",     "MicrosoftVibeVoice"),
        ("microsoft_speecht5",      "MicrosoftVibeVoice"),   # same vendor
        ("Edge-TTS",                "MicrosoftEdge"),
        ("parler_tts",              "ParlerTTS"),
        ("suno_bark",               "SunoBark"),
        ("e2-tts",                  "E2F5"),
        ("f5-tts",                  "E2F5"),
        ("facebook_mms",            "MetaMMS"),
        ("Spark-TTS",               "SparkTTS"),
        ("FireRedTTS",              "FireRedTTS"),
        ("OuteTTS",                 "OuteTTS"),
        ("WhisperSpeech",           "WhisperSpeech"),
        ("Openaudio",               "Openaudio"),
        ("kokoro",                  "Kokoro"),
        ("vixTTS",                  "Kokoro"),               # Kokoro variant
        ("Higgs-Audio",             "HiggsAudio"),
        ("Kyutai-TTS",              "Kyutai"),
        ("Voxtream",                "Voxtream"),
        ("Resemble",                "Resemble"),
        ("Ringg",                   "Ringg"),
        ("sesame_csm",              "SesameCsm"),
    ]
    mapping: dict[str, str] = {}
    for sys in systems:
        matched = False
        for prefix, family in rules:
            if prefix.lower() in sys.lower():
                mapping[sys] = family
                matched = True
                break
        if not matched:
            # Singleton: use system name as own family
            mapping[sys] = sys
    return mapping


# ─── Checkpoint helpers ───────────────────────────────────────────────────────
def find_best_ckpt(stem: str) -> Path | None:
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else None


def ckpt_for(condition: str, seed: int) -> Path:
    stem = f"mlaad_{condition}_seed{seed}"
    best = find_best_ckpt(stem)
    if best is None:
        raise FileNotFoundError(f"No checkpoint found for {stem}")
    return best


# ─── Dataset & inference ─────────────────────────────────────────────────────
from loader import TARGET_SR


class EvalDataset(Dataset):
    def __init__(self, records: list[dict], processed_dir: Path):
        self.records = records
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(self.processed_dir / rec["audio_path"]).unsqueeze(0)
        y   = 0 if rec["label"] == "bonafide" else 1
        return {
            "audio":         wav,
            "label":         torch.tensor(y, dtype=torch.long),
            "sample_rate":   TARGET_SR,
            "attack_system": rec.get("attack_system", "unknown"),
        }


def collate_fn(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit.eval()
    lit.to(device)
    return lit


def run_inference(
    model,
    records: list[dict],
    processed_dir: Path,
    device: torch.device,
) -> tuple[list[int], list[float], list[str]]:
    """Return (labels, logits, attack_systems) for all records."""
    ds = EvalDataset(records, processed_dir)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, collate_fn=collate_fn, pin_memory=True)
    all_labels: list[int] = []
    all_logits: list[float] = []
    all_systems: list[str] = []

    with torch.no_grad():
        for batch in dl:
            audio = batch["audio"].to(device)
            B     = audio.shape[0]
            num_frames = torch.full((B,), 48000 // 320 - 1, device=device)
            out   = model.model(
                audio, num_frames,
                profiler=None, use_aug=False, stage="val",
                ground_truth_labels=batch["label"].to(device),
            )
            logits = out["logit"].cpu().float()
            all_labels.extend(batch["label"].tolist())
            all_logits.extend(logits.tolist())
            all_systems.extend(batch["attack_system"])

    return all_labels, all_logits, all_systems


# ─── EER computation ──────────────────────────────────────────────────────────
from callbacks import compute_eer as _compute_eer


def eer_safe(labels: list[int], scores: list[float]) -> float | None:
    y, s = np.array(labels), np.array(scores)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(_compute_eer(y, s, positive_label=1))
    except Exception:
        return None


def per_system_eer(
    all_labels: list[int],
    all_logits: list[float],
    all_systems: list[str],
    min_samples: int,
) -> tuple[dict[str, float | None], dict[str, int], list[str]]:
    """
    Per-attack-system EER using system spoof + ALL bonafide.
    Returns:
        system_eer  : {system: eer_or_None}
        system_counts: {system: n_spoof_samples}
        dropped     : systems below min_samples threshold
    """
    bf_labels: list[int] = []
    bf_logits: list[float] = []
    sp_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for y, s, sys in zip(all_labels, all_logits, all_systems):
        if sys == "bonafide" or y == 0:
            bf_labels.append(y)
            bf_logits.append(s)
        else:
            sp_groups[sys][0].append(y)
            sp_groups[sys][1].append(s)

    results: dict[str, float | None] = {}
    counts:  dict[str, int]          = {}
    dropped: list[str]               = []

    for sys, (sp_lab, sp_log) in sorted(sp_groups.items()):
        counts[sys] = len(sp_lab)
        if len(sp_lab) < min_samples:
            dropped.append(sys)
            results[sys] = None
            continue
        combined_labels = sp_lab + bf_labels
        combined_logits = sp_log + bf_logits
        results[sys] = eer_safe(combined_labels, combined_logits)

    return results, counts, dropped


# ─── Spearman helpers ─────────────────────────────────────────────────────────
def pairwise_spearman(
    seed_eers: dict[int, dict[str, float]],
    valid_systems: list[str],
) -> dict[str, float]:
    """Compute all C(n,2) pairwise Spearman rho on EER ranking."""
    seeds = sorted(seed_eers.keys())
    pairs: dict[str, float] = {}
    for s1, s2 in combinations(seeds, 2):
        eer1 = [seed_eers[s1].get(sys, float("nan")) for sys in valid_systems]
        eer2 = [seed_eers[s2].get(sys, float("nan")) for sys in valid_systems]
        # Drop systems where either seed has NaN
        mask = [not (np.isnan(e1) or np.isnan(e2)) for e1, e2 in zip(eer1, eer2)]
        e1_clean = [e for e, m in zip(eer1, mask) if m]
        e2_clean = [e for e, m in zip(eer2, mask) if m]
        if len(e1_clean) < 3:
            pairs[f"{s1}_vs_{s2}"] = float("nan")
        else:
            rho, pval = spearmanr(e1_clean, e2_clean)
            pairs[f"{s1}_vs_{s2}"] = round(float(rho), 4)
    return pairs


def family_spearman(
    seed_eers: dict[int, dict[str, float]],
    family_map: dict[str, str],
    valid_systems: list[str],
) -> dict[str, float]:
    """Compute pairwise Spearman on family-level mean EER."""
    seeds = sorted(seed_eers.keys())
    # Per seed: mean EER per family
    families = sorted(set(family_map[s] for s in valid_systems))

    def family_mean_eers(seed: int) -> dict[str, float]:
        fam_eers: dict[str, list] = defaultdict(list)
        for sys in valid_systems:
            e = seed_eers[seed].get(sys, float("nan"))
            if not np.isnan(e):
                fam_eers[family_map[sys]].append(e)
        return {f: float(np.mean(v)) for f, v in fam_eers.items() if v}

    pairs: dict[str, float] = {}
    for s1, s2 in combinations(seeds, 2):
        fe1 = family_mean_eers(s1)
        fe2 = family_mean_eers(s2)
        common = sorted(set(fe1) & set(fe2))
        if len(common) < 3:
            pairs[f"{s1}_vs_{s2}"] = float("nan")
        else:
            rho, _ = spearmanr([fe1[f] for f in common], [fe2[f] for f in common])
            pairs[f"{s1}_vs_{s2}"] = round(float(rho), 4)
    return pairs


def category_flips(
    seed_eers: dict[int, dict[str, float]],
    valid_systems: list[str],
    top_frac: float = TOP_QUARTILE,
) -> dict:
    """
    For each system: is it in the top quartile (hardest)?
    Count how many systems are NOT consistent across all seeds.
    """
    seeds = sorted(seed_eers.keys())
    n_top = max(1, int(len(valid_systems) * top_frac))

    per_seed_top: dict[int, set] = {}
    for seed in seeds:
        eers = {sys: seed_eers[seed].get(sys, float("nan")) for sys in valid_systems}
        # Drop NaN
        valid = {sys: e for sys, e in eers.items() if not np.isnan(e)}
        # Highest EER = hardest
        ranked = sorted(valid, key=lambda s: -valid[s])
        per_seed_top[seed] = set(ranked[:n_top])

    # A system flips if it's not in the top-quartile in at least one seed
    # but IS in at least one other seed
    flip_systems = []
    for sys in valid_systems:
        in_top = [sys in per_seed_top[s] for s in seeds]
        if any(in_top) and not all(in_top):
            flip_systems.append(sys)

    per_seed_top_lists = {s: sorted(per_seed_top[s]) for s in seeds}
    return {
        "n_flips":          len(flip_systems),
        "flip_systems":     flip_systems,
        "n_top":            n_top,
        "n_total_systems":  len(valid_systems),
        "per_seed_top":     per_seed_top_lists,
    }


def bootstrap_delta_rho(
    robust_pairs: dict[str, float],
    goat_pairs:   dict[str, float],
    n_bootstrap:  int = N_BOOTSTRAP,
    rng_seed:     int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap CI for robust_mean_rho - goat_mean_rho."""
    rng = np.random.default_rng(rng_seed)
    r_vals = np.array([v for v in robust_pairs.values() if not np.isnan(v)])
    g_vals = np.array([v for v in goat_pairs.values()   if not np.isnan(v)])
    obs_delta = float(r_vals.mean() - g_vals.mean())

    boot_deltas = []
    for _ in range(n_bootstrap):
        r_boot = rng.choice(r_vals, size=len(r_vals), replace=True).mean()
        g_boot = rng.choice(g_vals, size=len(g_vals), replace=True).mean()
        boot_deltas.append(float(r_boot - g_boot))

    boot = np.array(boot_deltas)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "robust_mean_rho":   round(float(r_vals.mean()), 4),
        "goat_mean_rho":     round(float(g_vals.mean()), 4),
        "observed_delta":    round(obs_delta, 4),
        "ci_95_lo":          round(ci_lo, 4),
        "ci_95_hi":          round(ci_hi, 4),
        "n_bootstrap":       n_bootstrap,
        "significant":       bool(ci_lo > 0 or ci_hi < 0),
        "robust_more_stable": bool(obs_delta > 0.05 and ci_lo > 0),
        "interpretation": (
            "augmented training stabilizes ranking (robust > goat by >= 0.05, CI excludes 0)"
            if (obs_delta > 0.05 and ci_lo > 0) else
            "stability is similar — not specific to augmentation"
            if abs(obs_delta) < 0.05 else
            "goat more stable than robust — unexpected"
        ),
    }


# ─── Verdict ─────────────────────────────────────────────────────────────────
def apply_criteria(mean_rho_sys: float, mean_rho_fam: float, n_flips: int) -> dict:
    if (mean_rho_sys >= 0.75 or mean_rho_fam >= 0.85) and n_flips <= 2:
        criterion = "strong"
        explanation = (
            f"mean system rho={mean_rho_sys:.3f} >= 0.75 OR "
            f"family rho={mean_rho_fam:.3f} >= 0.85, AND {n_flips} flips <= 2. "
            "Per-attack story is robust across seeds on MLAAD."
        )
    elif (0.55 <= mean_rho_sys < 0.75 or 0.65 <= mean_rho_fam < 0.85) and n_flips <= 4:
        criterion = "moderate"
        explanation = (
            f"mean system rho={mean_rho_sys:.3f} in [0.55, 0.75) OR "
            f"family rho={mean_rho_fam:.3f} in [0.65, 0.85), AND {n_flips} flips <= 4. "
            "Per-attack story is mostly robust with some noise."
        )
    else:
        criterion = "weak"
        # Explain which criterion actually failed
        if mean_rho_sys >= 0.75 and n_flips > 2:
            reason = (
                f"rho={mean_rho_sys:.3f} meets STRONG rho threshold (>=0.75) "
                f"but {n_flips} category flips >> 2-flip limit. "
                "High rank correlation but ranking category boundaries are unstable."
            )
        elif mean_rho_sys < 0.55:
            reason = f"mean system rho={mean_rho_sys:.3f} < 0.55."
        else:
            reason = (
                f"rho={mean_rho_sys:.3f} in [0.55,0.75) / "
                f"family={mean_rho_fam:.3f} in [0.65,0.85) "
                f"but {n_flips} category flips > 4."
            )
        explanation = (
            reason + " Per-attack ranking partially reproducible on MLAAD "
            "(strong correlation but top-quartile membership unstable); "
            "ASVspoof E6 stability (ρ=0.943, 0 flips on 6 systems) "
            "does not fully transfer to 63-system setting."
        )
    return {"criterion": criterion, "explanation": explanation}


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # ── Load test records ─────────────────────────────────────────────────────
    records: list[dict] = json.loads(INDIST_JSON.read_text())
    print(f"In-distribution records: {len(records)}")

    # Sanity (b): same eval split used across all seeds (deterministic)
    import hashlib
    split_hash = hashlib.md5(INDIST_JSON.read_bytes()).hexdigest()
    print(f"Split hash (sanity b): {split_hash}")

    # ── Sample counts per system ──────────────────────────────────────────────
    from collections import Counter
    spoof_counts = Counter(r["attack_system"] for r in records if r["label"] == "spoof")
    bonafide_count = sum(1 for r in records if r["label"] == "bonafide")
    max_spoof = max(spoof_counts.values()) if spoof_counts else 0

    print(f"\nBonafide samples: {bonafide_count}")
    print(f"Spoof systems:    {len(spoof_counts)}")
    print(f"Max spoof per system: {max_spoof}")

    # Pre-registered threshold check
    dropped_preregistered = [s for s, n in spoof_counts.items() if n < MIN_SAMPLES]
    effective_min_samples = MIN_SAMPLES
    if len(dropped_preregistered) == len(spoof_counts):
        effective_min_samples = 5
        print(f"\n[NOTE] Pre-registered MIN_SAMPLES={MIN_SAMPLES} would drop ALL "
              f"{len(spoof_counts)} systems (max={max_spoof}). "
              f"Falling back to effective MIN_SAMPLES={effective_min_samples}.")
        print("       EER estimates are noisy with small N; reported for completeness.")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    condition_seeds = {
        "robust_goat": ROBUST_SEEDS,
        "goat":        GOAT_SEEDS,
    }
    all_ckpts: dict[str, dict[int, Path]] = {}
    for cond, seeds in condition_seeds.items():
        all_ckpts[cond] = {}
        for seed in seeds:
            p = ckpt_for(cond, seed)
            all_ckpts[cond][seed] = p
            print(f"  [{cond}/seed={seed}] {p.name}")

    # Sanity (c): verify different checkpoints (not cache)
    print("\n[Sanity c] Checkpoint identity check:")
    for cond, seed_map in all_ckpts.items():
        paths = list(seed_map.values())
        names = [p.name for p in paths]
        assert len(set(names)) == len(names), f"Duplicate checkpoint for {cond}!"
        print(f"  {cond}: {len(names)} distinct checkpoints OK")

    # ── Per-seed inference ────────────────────────────────────────────────────
    # seed_eers[condition][seed][system] = eer
    seed_eers: dict[str, dict[int, dict[str, float]]] = {
        "robust_goat": {}, "goat": {}
    }
    system_counts_all: dict[str, int] = {}
    dropped_systems: list[str] = []
    all_seed_logit_cache: dict[str, dict[int, tuple]] = {
        "robust_goat": {}, "goat": {}
    }

    print("\n" + "=" * 60)
    print("Per-seed inference")
    print("=" * 60)

    for cond, seeds in condition_seeds.items():
        for seed in seeds:
            ckpt_path = all_ckpts[cond][seed]
            print(f"\n  [{cond}/seed={seed}] Loading {ckpt_path.name} ...", flush=True)
            t0 = time.time()
            model = load_model(ckpt_path, device)
            labels, logits, systems = run_inference(model, records, PROCESSED_DIR, device)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

            elapsed = time.time() - t0
            sys_eer, counts, dropped = per_system_eer(
                labels, logits, systems, effective_min_samples
            )
            system_counts_all.update(counts)
            dropped_systems.extend(d for d in dropped if d not in dropped_systems)

            # Store only valid (not None) EERs
            seed_eers[cond][seed] = {
                sys: e for sys, e in sys_eer.items() if e is not None
            }
            print(f"    Done in {elapsed:.1f}s — {len(seed_eers[cond][seed])} valid systems",
                  flush=True)

    # ── Valid systems ─────────────────────────────────────────────────────────
    # A system is valid if it has a non-None EER in ALL seeds of at least one condition
    def valid_for(cond: str) -> list[str]:
        seeds = condition_seeds[cond]
        all_sys = set(seed_eers[cond][seeds[0]].keys())
        for s in seeds[1:]:
            all_sys &= set(seed_eers[cond][s].keys())
        return sorted(all_sys)

    valid_robust = valid_for("robust_goat")
    valid_goat   = valid_for("goat")
    valid_union  = sorted(set(valid_robust) | set(valid_goat))

    print(f"\nValid systems (robust_goat): {len(valid_robust)}")
    print(f"Valid systems (goat):        {len(valid_goat)}")
    print(f"Dropped systems (sample count <{effective_min_samples}):")
    for sys in sorted(dropped_systems):
        print(f"  {system_counts_all.get(sys, '?'):4}  {sys}")

    # ── Family map ────────────────────────────────────────────────────────────
    family_map = build_family_map(valid_union)
    # Sanity (d): report family groupings
    from collections import defaultdict as _dd
    fam_to_sys: dict[str, list] = _dd(list)
    for sys, fam in family_map.items():
        fam_to_sys[fam].append(sys)
    multi = {f: sorted(ss) for f, ss in sorted(fam_to_sys.items()) if len(ss) > 1}
    print(f"\nFamily groupings (multi-system families only, {len(multi)} total):")
    for fam, ss in multi.items():
        print(f"  {fam}: {ss}")

    # ── Per-seed per-attack EER CSV ───────────────────────────────────────────
    csv_rows = []
    for cond in ("robust_goat", "goat"):
        for seed, eer_map in seed_eers[cond].items():
            for sys in sorted(eer_map.keys()):
                csv_rows.append({
                    "condition":     cond,
                    "seed":          seed,
                    "attack_system": sys,
                    "n_spoof":       system_counts_all.get(sys, ""),
                    "eer":           round(eer_map[sys], 6),
                })

    csv_path = OUT_DIR / "per_seed_per_attack_eer.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "attack_system",
                                               "n_spoof", "eer"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {csv_path}")

    # ── Pairwise Spearman ─────────────────────────────────────────────────────
    robust_sys_pairs = pairwise_spearman(seed_eers["robust_goat"], valid_robust)
    goat_sys_pairs   = pairwise_spearman(seed_eers["goat"],        valid_goat)
    robust_fam_pairs = family_spearman(seed_eers["robust_goat"], family_map, valid_robust)
    goat_fam_pairs   = family_spearman(seed_eers["goat"],         family_map, valid_goat)

    def summarize_pairs(pairs: dict[str, float]) -> dict:
        vals = [v for v in pairs.values() if not np.isnan(v)]
        return {
            "pairs":  pairs,
            "mean":   round(float(np.mean(vals)), 4)  if vals else None,
            "min":    round(float(np.min(vals)), 4)   if vals else None,
            "max":    round(float(np.max(vals)), 4)   if vals else None,
            "n_pairs": len(vals),
        }

    spearman_out = {
        "robust_goat": {
            "system_level": summarize_pairs(robust_sys_pairs),
            "family_level": summarize_pairs(robust_fam_pairs),
            "n_valid_systems": len(valid_robust),
        },
        "goat": {
            "system_level": summarize_pairs(goat_sys_pairs),
            "family_level": summarize_pairs(goat_fam_pairs),
            "n_valid_systems": len(valid_goat),
        },
    }
    (OUT_DIR / "pairwise_spearman.json").write_text(json.dumps(spearman_out, indent=2))
    print(f"Wrote {OUT_DIR / 'pairwise_spearman.json'}")

    # ── Family-level agreement detail ─────────────────────────────────────────
    def per_seed_family_eers(cond: str) -> dict[int, dict[str, float]]:
        seeds = condition_seeds[cond]
        valid = valid_robust if cond == "robust_goat" else valid_goat
        result: dict[int, dict[str, float]] = {}
        for seed in seeds:
            fam_eers: dict[str, list] = _dd(list)
            for sys in valid:
                e = seed_eers[cond][seed].get(sys, float("nan"))
                if not np.isnan(e):
                    fam_eers[family_map[sys]].append(e)
            result[seed] = {f: round(float(np.mean(v)), 4)
                            for f, v in fam_eers.items() if v}
        return result

    fam_level_out = {
        "family_map_multi_system": multi,
        "robust_goat": {
            "per_seed_family_eers": {str(s): v for s, v in
                                     per_seed_family_eers("robust_goat").items()},
            "pairwise_spearman": summarize_pairs(robust_fam_pairs),
        },
        "goat": {
            "per_seed_family_eers": {str(s): v for s, v in
                                     per_seed_family_eers("goat").items()},
            "pairwise_spearman": summarize_pairs(goat_fam_pairs),
        },
    }
    (OUT_DIR / "family_level_agreement.json").write_text(
        json.dumps(fam_level_out, indent=2))
    print(f"Wrote {OUT_DIR / 'family_level_agreement.json'}")

    # ── Category flips ────────────────────────────────────────────────────────
    robust_flips = category_flips(seed_eers["robust_goat"], valid_robust)
    goat_flips   = category_flips(seed_eers["goat"],        valid_goat)
    flips_out = {
        "top_quartile_frac": TOP_QUARTILE,
        "robust_goat":       robust_flips,
        "goat":              goat_flips,
    }
    (OUT_DIR / "category_flips.json").write_text(json.dumps(flips_out, indent=2))
    print(f"Wrote {OUT_DIR / 'category_flips.json'}")

    # ── Robust vs goat stability comparison ──────────────────────────────────
    comparison = bootstrap_delta_rho(robust_sys_pairs, goat_sys_pairs)
    (OUT_DIR / "robust_vs_goat_comparison.json").write_text(
        json.dumps(comparison, indent=2))
    print(f"Wrote {OUT_DIR / 'robust_vs_goat_comparison.json'}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    robust_mean_sys = spearman_out["robust_goat"]["system_level"]["mean"] or 0.0
    robust_mean_fam = spearman_out["robust_goat"]["family_level"]["mean"] or 0.0
    robust_n_flips  = robust_flips["n_flips"]
    verdict = apply_criteria(robust_mean_sys, robust_mean_fam, robust_n_flips)

    verdict_out = {
        "pre_registered_criteria": {
            "strong":   "mean system rho >= 0.75 OR family rho >= 0.85, AND flips <= 2",
            "moderate": "mean system rho in [0.55,0.75) OR family rho in [0.65,0.85), AND flips <= 4",
            "weak":     "mean system rho < 0.55 OR flips > 4",
        },
        "robust_goat": {
            "mean_system_rho":   robust_mean_sys,
            "mean_family_rho":   robust_mean_fam,
            "n_category_flips":  robust_n_flips,
            **verdict,
        },
        "asvspoof_e6_reference": ASVSPOOF_E6,
        "note_min_samples": (
            f"Pre-registered MIN_SAMPLES={MIN_SAMPLES}; max in test split is {max_spoof}. "
            f"Effective threshold used: {effective_min_samples}. "
            "EERs with small N should be interpreted cautiously."
        ),
    }
    (OUT_DIR / "e6_analog_verdict.json").write_text(json.dumps(verdict_out, indent=2))
    print(f"Wrote {OUT_DIR / 'e6_analog_verdict.json'}")

    # ── run_config ────────────────────────────────────────────────────────────
    run_config = {
        "robust_seeds":         ROBUST_SEEDS,
        "goat_seeds":           GOAT_SEEDS,
        "min_samples_preregistered": MIN_SAMPLES,
        "min_samples_effective": effective_min_samples,
        "batch_size":           BATCH_SIZE,
        "n_bootstrap":          N_BOOTSTRAP,
        "bootstrap_seed":       BOOTSTRAP_SEED,
        "top_quartile_frac":    TOP_QUARTILE,
        "test_split_hash":      split_hash,
        "n_test_records":       len(records),
        "n_bonafide":           bonafide_count,
        "n_spoof_systems":      len(spoof_counts),
        "n_valid_robust":       len(valid_robust),
        "n_valid_goat":         len(valid_goat),
        "checkpoints": {
            cond: {str(s): p.name for s, p in seed_map.items()}
            for cond, seed_map in all_ckpts.items()
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"Wrote {OUT_DIR / 'run_config.json'}")

    # ══════════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MLAAD E6 RANKING LOCK — SUMMARY")
    print("=" * 70)

    for cond in ("robust_goat", "goat"):
        valid  = valid_robust if cond == "robust_goat" else valid_goat
        seeds  = condition_seeds[cond]
        flips  = robust_flips if cond == "robust_goat" else goat_flips
        s_info = spearman_out[cond]
        print(f"\n── {cond.upper()} ({len(seeds)} seeds, {len(valid)} systems) ──")

        # Per-seed EER ranking: top-5 hardest and top-5 easiest
        for seed in seeds:
            eer_map = seed_eers[cond][seed]
            ranked  = sorted(eer_map.items(), key=lambda x: -x[1])
            top5    = ranked[:5]
            bot5    = ranked[-5:]
            print(f"\n  seed={seed} — top-5 hardest (highest EER):")
            for sys, e in top5:
                print(f"    {e:.4f}  {sys}")
            print(f"  seed={seed} — top-5 easiest (lowest EER):")
            for sys, e in reversed(bot5):
                print(f"    {e:.4f}  {sys}")

        # Pairwise Spearman matrix
        print(f"\n  Pairwise Spearman rho (system-level):")
        for pair, rho in s_info["system_level"]["pairs"].items():
            print(f"    {pair:30s}  rho={rho:+.4f}")
        print(f"  → mean={s_info['system_level']['mean']:+.4f}  "
              f"[{s_info['system_level']['min']:+.4f}, "
              f"{s_info['system_level']['max']:+.4f}]")

        print(f"\n  Family-level Spearman rho:")
        for pair, rho in s_info["family_level"]["pairs"].items():
            print(f"    {pair:30s}  rho={rho:+.4f}")
        fam_m = s_info['family_level']['mean']
        print(f"  → mean={fam_m:+.4f}  "
              f"[{s_info['family_level']['min']:+.4f}, "
              f"{s_info['family_level']['max']:+.4f}]")

        print(f"\n  Category flips (top-{int(TOP_QUARTILE*100)}%): {flips['n_flips']} / {len(valid)}")
        if flips["flip_systems"]:
            print(f"  Flipped: {flips['flip_systems'][:10]}")

    # ASVspoof E6 comparison
    print("\n" + "─" * 70)
    print("ASVspoof E6 reference (6 systems, permutation-test p-values):")
    print(f"  rho seed3 vs seed1 : {ASVSPOOF_E6['rho_seed3_vs_seed1']:+.4f}")
    print(f"  rho seed3 vs seed2 : {ASVSPOOF_E6['rho_seed3_vs_seed2']:+.4f}")
    print(f"  rho seed1 vs seed2 : {ASVSPOOF_E6['rho_seed1_vs_seed2']:+.4f}")
    print(f"  Category flips     : {ASVSPOOF_E6['n_category_flips']}")
    print(f"MLAAD robust_goat mean rho: {robust_mean_sys:+.4f}  "
          f"(system), {robust_mean_fam:+.4f}  (family)")

    # Verdict
    print("\n" + "─" * 70)
    print(f"PRE-REGISTERED VERDICT: {verdict['criterion'].upper()}")
    print(f"  {verdict['explanation']}")

    # Robust vs goat comparison
    print("\n" + "─" * 70)
    print("Robust vs Goat stability comparison:")
    print(f"  robust mean rho : {comparison['robust_mean_rho']:+.4f}")
    print(f"  goat   mean rho : {comparison['goat_mean_rho']:+.4f}")
    print(f"  delta           : {comparison['observed_delta']:+.4f}  "
          f"95% CI [{comparison['ci_95_lo']:+.4f}, {comparison['ci_95_hi']:+.4f}]")
    print(f"  interpretation  : {comparison['interpretation']}")

    print(f"\nResults: {OUT_DIR}")
    print(f"Elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
