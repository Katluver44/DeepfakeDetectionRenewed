#!/usr/bin/env python3
"""
asvspoof_e6_ranking_lock.py
===========================
E6 ranking-stability analysis for ASVspoof 2019-LA robust_goat and goat models.

Mirrors the approach of mlaad_e6_ranking_lock.py but uses:
  - ASVspoof 2019-LA validation split (6 attack systems, 3716 samples each)
  - models/robust_goat.ckpt (seed1), robust_goat_seed7.ckpt (seed2),
    robust_goat_seed3.ckpt (seed3)
  - models/goat.ckpt (1 seed only — no pairwise possible)

Computes per-attack EER via logit-based inference (not hub-mass / permutation),
then applies the same pre-registered criteria as the MLAAD analog for comparison.

Pre-registered criteria (from mlaad_e6_ranking_lock.py, stated before running):
  Strong   : mean pairwise rho >= 0.75 system-level, AND <= 2 category flips.
  Moderate : mean rho in [0.55, 0.75), AND <= 4 category flips.
  Weak     : mean rho < 0.55 OR flips > 4.

Note: With only 6 systems, top-quartile = top-2 for category flip analysis.

Outputs:
    experiments/results/asvspoof/e6_ranking_lock/
        per_seed_per_attack_eer.csv
        pairwise_spearman.json
        category_flips.json
        comparison_with_mlaad.json
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

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
EXP_DIR    = Path(__file__).resolve().parent
OUT_DIR    = EXP_DIR / "results" / "asvspoof" / "e6_ranking_lock"
MODELS_DIR = REPO_ROOT / "models"
CACHE_DIR  = REPO_ROOT / "data" / "asvspoof_2019_la"
VOCAB_DIR  = REPO_ROOT / "vocab_phoneme"

for _p in (str(REPO_ROOT), str(EXP_DIR)):
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
HF_DATASET  = "Bisher/ASVspoof_2019_LA"
HF_SPLIT    = "validation"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
BATCH_SIZE  = 8
TARGET_SR   = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # 149
N_PER_CLASS = 50   # per attack system, matching original gat_e6.py balanced eval

ROBUST_CHECKPOINTS = {
    "seed1": MODELS_DIR / "robust_goat.ckpt",
    "seed2": MODELS_DIR / "robust_goat_seed7.ckpt",
    "seed3": MODELS_DIR / "robust_goat_seed3.ckpt",
}
GOAT_CHECKPOINTS = {
    "seed1": MODELS_DIR / "goat.ckpt",
}

N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 42
TOP_QUARTILE   = 0.25
MIN_SAMPLES    = 30   # enforced — all ASVspoof systems have 3716 samples

# MLAAD E6 results for cross-comparison (from mlaad_e6_ranking_lock.py output)
MLAAD_E6 = {
    "robust_goat": {"mean_system_rho": 0.7936, "mean_family_rho": 0.8061,
                    "n_category_flips": 21, "verdict": "weak", "n_seeds": 4},
    "goat":        {"mean_system_rho": 0.8000, "mean_family_rho": 0.8070,
                    "n_category_flips": 22, "verdict": "weak", "n_seeds": 5},
}

# ─── Audio helpers (from gat_e6.py) ──────────────────────────────────────────
import io as _io
import soundfile as _sf
import torchaudio.transforms as _T


def _decode(entry: dict) -> torch.Tensor:
    raw  = entry.get("bytes"); path = entry.get("path")
    arr, sr = (_sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else
               _sf.read(path, dtype="float32", always_2d=False))
    w = torch.tensor(arr)
    if w.ndim == 1: w = w.unsqueeze(0)
    elif w.ndim == 2: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR: w = _T.Resample(sr, TARGET_SR)(w)
    return w


def _crop(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    if n < TARGET_SAMPLES:
        w = w.repeat(1, -(-TARGET_SAMPLES // n))
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s: s + TARGET_SAMPLES]


def _lbl(raw) -> int:
    if isinstance(raw, str):
        return 0 if raw.strip().lower() in ("0", "bonafide", "real", "genuine") else 1
    return int(raw)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class ASVspoofDataset(torch.utils.data.Dataset):
    """Full validation split — no balancing, deterministic order."""
    def __init__(self):
        from datasets import load_dataset, Audio as HFAudio
        token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
        ds = load_dataset(HF_DATASET, split=HF_SPLIT,
                          cache_dir=str(CACHE_DIR), token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        self.label_key = next(
            (k for k in ds[0] if k != "audio" and
             ("label" in k.lower() or k.lower() == "key")), "label")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        ex = self.ds[idx]
        return {
            "audio":     _crop(_decode(ex["audio"])),
            "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
            "system_id": ex["system_id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "audio":     torch.stack([b["audio"] for b in batch]),
        "label":     torch.stack([b["label"] for b in batch]),
        "system_id": [b["system_id"] for b in batch],
    }


# ─── Model loading ────────────────────────────────────────────────────────────
def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name    = network_name
        network_param.pretrained_name = (
            "microsoft/wavlm-base" if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None,
                total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)

    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load


def _detect_n_edges(ckpt_path: Path) -> int:
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    hp   = ckpt.get("hyper_parameters", {})
    cfg  = hp.get("cfg", None)
    n    = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10


def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    n_edges = _detect_n_edges(ckpt_path)
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=n_edges, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit, n_edges


# ─── Inference ───────────────────────────────────────────────────────────────
def run_frozen_frontend(audio: torch.Tensor, gat_model, device: torch.device):
    x = audio
    if x.ndim == 3 and x.size(1) == 1: x = x[:, 0, :]
    with torch.no_grad():
        feat1 = gat_model.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hs, _ = gat_model.transformer_in_phoneme_model.feature_projection(feat1)
        pf    = gat_model.transformer_in_phoneme_model.encoder(hs)[0]
        pl    = gat_model.phoneme_model.model.model.lm_head(pf)
        pids  = torch.argmax(pl, dim=-1)
    return hs, pids


def run_inference(
    lit,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Return list of {label, system_id, logit} for all samples."""
    gm = lit.model
    records: list[dict] = []
    n_total = len(loader.dataset)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            audio   = batch["audio"].to(device)
            labels  = batch["label"].tolist()
            sys_ids = batch["system_id"]
            B       = len(labels)
            num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)
            hs, pids = run_frozen_frontend(audio, gm, device)
            result   = gm.encoder_and_GAT(hs, num_f, pids)
            logits   = result[5].cpu()
            for i in range(B):
                records.append({
                    "label":     labels[i],
                    "system_id": sys_ids[i],
                    "logit":     float(logits[i].item()),
                })
            if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                print(f"  {min((bi+1)*BATCH_SIZE, n_total)}/{n_total}", flush=True)

    return records


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
    records: list[dict],
    min_samples: int = MIN_SAMPLES,
) -> tuple[dict[str, float | None], dict[str, int], list[str]]:
    """
    Per-attack-system EER: system spoof + ALL bonafide.
    Returns (eer_map, count_map, dropped_list).
    """
    bf_labels: list[int] = []
    bf_logits: list[float] = []
    sp_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for rec in records:
        if rec["system_id"] == "-":   # bonafide
            bf_labels.append(rec["label"])
            bf_logits.append(rec["logit"])
        else:
            sp_groups[rec["system_id"]][0].append(rec["label"])
            sp_groups[rec["system_id"]][1].append(rec["logit"])

    results: dict[str, float | None] = {}
    counts:  dict[str, int]          = {}
    dropped: list[str]               = []

    for sys, (sp_lab, sp_log) in sorted(sp_groups.items()):
        counts[sys] = len(sp_lab)
        if len(sp_lab) < min_samples:
            dropped.append(sys)
            results[sys] = None
            continue
        results[sys] = eer_safe(sp_lab + bf_labels, sp_log + bf_logits)

    return results, counts, dropped


# ─── Spearman / flip helpers (same logic as mlaad_e6_ranking_lock.py) ────────
def pairwise_spearman(
    seed_eers: dict[str, dict[str, float]],
    valid_systems: list[str],
) -> dict[str, float]:
    seeds = sorted(seed_eers.keys())
    pairs: dict[str, float] = {}
    for s1, s2 in combinations(seeds, 2):
        e1 = [seed_eers[s1].get(sys, float("nan")) for sys in valid_systems]
        e2 = [seed_eers[s2].get(sys, float("nan")) for sys in valid_systems]
        mask = [not (np.isnan(a) or np.isnan(b)) for a, b in zip(e1, e2)]
        c1 = [a for a, m in zip(e1, mask) if m]
        c2 = [b for b, m in zip(e2, mask) if m]
        if len(c1) < 3:
            pairs[f"{s1}_vs_{s2}"] = float("nan")
        else:
            rho, _ = spearmanr(c1, c2)
            pairs[f"{s1}_vs_{s2}"] = round(float(rho), 4)
    return pairs


def category_flips(
    seed_eers: dict[str, dict[str, float]],
    valid_systems: list[str],
    top_frac: float = TOP_QUARTILE,
) -> dict:
    seeds = sorted(seed_eers.keys())
    n_top = max(1, int(len(valid_systems) * top_frac))

    per_seed_top: dict[str, set] = {}
    for seed in seeds:
        eers  = {s: seed_eers[seed].get(s, float("nan")) for s in valid_systems}
        valid = {s: e for s, e in eers.items() if not np.isnan(e)}
        ranked = sorted(valid, key=lambda s: -valid[s])
        per_seed_top[seed] = set(ranked[:n_top])

    flip_systems = []
    for sys in valid_systems:
        in_top = [sys in per_seed_top[s] for s in seeds]
        if any(in_top) and not all(in_top):
            flip_systems.append(sys)

    return {
        "n_flips":         len(flip_systems),
        "flip_systems":    flip_systems,
        "n_top":           n_top,
        "n_total_systems": len(valid_systems),
        "per_seed_top":    {s: sorted(per_seed_top[s]) for s in seeds},
    }


def summarize_pairs(pairs: dict[str, float]) -> dict:
    vals = [v for v in pairs.values() if not np.isnan(v)]
    return {
        "pairs":   pairs,
        "mean":    round(float(np.mean(vals)), 4) if vals else None,
        "min":     round(float(np.min(vals)),  4) if vals else None,
        "max":     round(float(np.max(vals)),  4) if vals else None,
        "n_pairs": len(vals),
    }


def apply_criteria(mean_rho: float, n_flips: int) -> dict:
    if mean_rho >= 0.75 and n_flips <= 2:
        criterion = "strong"
        explanation = (
            f"mean rho={mean_rho:.3f} >= 0.75 AND {n_flips} flips <= 2. "
            "Per-attack ranking stable across seeds."
        )
    elif 0.55 <= mean_rho < 0.75 and n_flips <= 4:
        criterion = "moderate"
        explanation = (
            f"mean rho={mean_rho:.3f} in [0.55,0.75), AND {n_flips} flips <= 4. "
            "Mostly stable with some noise."
        )
    else:
        if mean_rho >= 0.75 and n_flips > 2:
            reason = (
                f"rho={mean_rho:.3f} meets STRONG threshold but "
                f"{n_flips} category flips > 2-flip limit"
            )
        elif mean_rho < 0.55:
            reason = f"mean rho={mean_rho:.3f} < 0.55"
        else:
            reason = (
                f"rho={mean_rho:.3f} in [0.55,0.75) but {n_flips} flips > 4"
            )
        criterion = "weak"
        explanation = reason + ". Per-attack ranking partially or fully unstable."
    return {"criterion": criterion, "explanation": explanation}


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    patch_phoneme_loader()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading ASVspoof 2019-LA validation split ...", flush=True)
    ds = ASVspoofDataset()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)
    print(f"  {len(ds)} samples loaded")

    # Count systems
    from collections import Counter
    sys_counts_all = Counter(ds.ds[i]["system_id"] for i in range(len(ds.ds)))
    print(f"  Systems: {dict(sorted(sys_counts_all.items()))}")
    bonafide_count = sys_counts_all.get("-", 0)

    # ── Per-seed inference ────────────────────────────────────────────────────
    condition_ckpts = {
        "robust_goat": ROBUST_CHECKPOINTS,
        "goat":        GOAT_CHECKPOINTS,
    }
    seed_eers:    dict[str, dict[str, dict[str, float]]] = {}
    seed_records: dict[str, dict[str, list]]             = {}
    all_counts:   dict[str, int]                         = {}
    dropped_all:  list[str]                              = []

    print("\n" + "=" * 60)
    print("Per-seed inference")
    print("=" * 60)

    for cond, ckpt_map in condition_ckpts.items():
        seed_eers[cond] = {}
        seed_records[cond] = {}

        for seed_name, ckpt_path in ckpt_map.items():
            if not ckpt_path.exists():
                print(f"  [MISSING] {cond}/{seed_name}: {ckpt_path}")
                continue
            print(f"\n  [{cond}/{seed_name}] {ckpt_path.name}", flush=True)
            t0 = time.time()
            model, n_edges = load_model(ckpt_path, device)
            print(f"    n_edges={n_edges}", flush=True)
            records = run_inference(model, loader, device)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()

            sys_eer, counts, dropped = per_system_eer(records, MIN_SAMPLES)
            all_counts.update(counts)
            dropped_all.extend(d for d in dropped if d not in dropped_all)
            seed_eers[cond][seed_name]    = {
                s: e for s, e in sys_eer.items() if e is not None
            }
            seed_records[cond][seed_name] = records
            print(f"    Done in {time.time()-t0:.1f}s — "
                  f"{len(seed_eers[cond][seed_name])} valid systems", flush=True)

    # ── Valid systems ─────────────────────────────────────────────────────────
    def valid_for(cond: str) -> list[str]:
        if not seed_eers[cond]: return []
        sets = [set(seed_eers[cond][s].keys()) for s in seed_eers[cond]]
        common = sets[0]
        for s in sets[1:]: common &= s
        return sorted(common)

    valid_robust = valid_for("robust_goat")
    valid_goat   = valid_for("goat")
    print(f"\nValid systems (robust_goat): {valid_robust}")
    print(f"Valid systems (goat):        {valid_goat}")
    if dropped_all:
        print(f"Dropped (<{MIN_SAMPLES} samples): {dropped_all}")

    # ── Sanity (c): distinct checkpoints ─────────────────────────────────────
    print("\n[Sanity c] Checkpoint identity:")
    for cond, ckpt_map in condition_ckpts.items():
        names = [p.name for p in ckpt_map.values() if p.exists()]
        ok = len(set(names)) == len(names)
        print(f"  {cond}: {names} — {'OK' if ok else 'DUPLICATE!'}")

    # ── Per-seed per-attack EER CSV ───────────────────────────────────────────
    csv_rows = []
    for cond in ("robust_goat", "goat"):
        for seed_name, eer_map in seed_eers[cond].items():
            for sys in sorted(eer_map.keys()):
                csv_rows.append({
                    "condition":  cond,
                    "seed":       seed_name,
                    "system_id":  sys,
                    "n_spoof":    all_counts.get(sys, ""),
                    "eer":        round(eer_map[sys], 6),
                })
    csv_path = OUT_DIR / "per_seed_per_attack_eer.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "system_id",
                                               "n_spoof", "eer"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {csv_path}")

    # ── Pairwise Spearman (robust_goat only — goat has 1 seed) ───────────────
    robust_pairs = pairwise_spearman(seed_eers["robust_goat"], valid_robust) \
                   if len(seed_eers["robust_goat"]) >= 2 else {}
    spearman_out = {
        "robust_goat": {
            "system_level": summarize_pairs(robust_pairs),
            "n_valid_systems": len(valid_robust),
            "n_seeds": len(seed_eers["robust_goat"]),
        },
        "goat": {
            "note": "Only 1 seed available — pairwise Spearman not computable.",
            "n_valid_systems": len(valid_goat),
            "n_seeds": 1,
            "per_attack_eer": {
                "seed1": seed_eers["goat"].get("seed1", {})
            },
        },
    }
    (OUT_DIR / "pairwise_spearman.json").write_text(json.dumps(spearman_out, indent=2))
    print(f"Wrote {OUT_DIR / 'pairwise_spearman.json'}")

    # ── Category flips ────────────────────────────────────────────────────────
    robust_flips = category_flips(seed_eers["robust_goat"], valid_robust) \
                   if len(seed_eers["robust_goat"]) >= 2 else {"n_flips": "N/A"}
    flips_out = {
        "top_quartile_frac": TOP_QUARTILE,
        "n_systems":         len(valid_robust),
        "n_top":             max(1, int(len(valid_robust) * TOP_QUARTILE)),
        "robust_goat":       robust_flips,
        "note": (f"With {len(valid_robust)} systems, top quartile = "
                 f"{max(1, int(len(valid_robust)*TOP_QUARTILE))} system(s)."),
    }
    (OUT_DIR / "category_flips.json").write_text(json.dumps(flips_out, indent=2))
    print(f"Wrote {OUT_DIR / 'category_flips.json'}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    robust_mean_rho = spearman_out["robust_goat"]["system_level"].get("mean") or 0.0
    robust_n_flips  = robust_flips.get("n_flips", 0)
    verdict = apply_criteria(robust_mean_rho, robust_n_flips) \
              if isinstance(robust_n_flips, int) else \
              {"criterion": "N/A", "explanation": "Insufficient seeds for analysis."}

    verdict_out = {
        "pre_registered_criteria": {
            "strong":   "mean rho >= 0.75 AND flips <= 2",
            "moderate": "mean rho in [0.55,0.75) AND flips <= 4",
            "weak":     "mean rho < 0.55 OR flips > 4",
        },
        "robust_goat": {
            "mean_system_rho":  robust_mean_rho,
            "n_category_flips": robust_n_flips,
            "n_seeds":          len(seed_eers["robust_goat"]),
            "n_systems":        len(valid_robust),
            **verdict,
        },
        "goat": {
            "note": "Only 1 seed — pairwise stability not computable. "
                    "Ranking shown in per_seed_per_attack_eer.csv for reference.",
            "per_attack_eer": seed_eers["goat"].get("seed1", {}),
        },
    }
    (OUT_DIR / "e6_analog_verdict.json").write_text(json.dumps(verdict_out, indent=2))
    print(f"Wrote {OUT_DIR / 'e6_analog_verdict.json'}")

    # ── Cross-dataset comparison ──────────────────────────────────────────────
    comparison_out = {
        "asvspoof_robust_goat": {
            "mean_system_rho":  robust_mean_rho,
            "n_category_flips": robust_n_flips,
            "n_seeds":          len(seed_eers["robust_goat"]),
            "n_systems":        len(valid_robust),
            "verdict":          verdict.get("criterion", "N/A"),
        },
        "mlaad_robust_goat": MLAAD_E6["robust_goat"],
        "mlaad_goat":        MLAAD_E6["goat"],
        "asvspoof_e6_original": {
            "rho_seed3_vs_seed1": 0.9429,
            "rho_seed3_vs_seed2": 0.9429,
            "rho_seed1_vs_seed2": 1.0000,
            "n_category_flips":   0,
            "n_systems":          6,
            "method":  "permutation_test_p_values (hub-mass, not EER)",
            "note": (
                "Original E6 used hub-mass attention patterns + permutation tests, "
                "not logit-based EER. EER-based rho will differ due to method change."
            ),
        },
        "interpretation": (
            "ASVspoof has only 6 systems — Spearman rho is highly sensitive to "
            "a single rank swap; compare cautiously with MLAAD's 63-system result. "
            "MLAAD and ASVspoof both use the same pre-registered criteria applied "
            "to EER-based rankings."
        ),
    }
    (OUT_DIR / "comparison_with_mlaad.json").write_text(
        json.dumps(comparison_out, indent=2))
    print(f"Wrote {OUT_DIR / 'comparison_with_mlaad.json'}")

    # ── run_config ────────────────────────────────────────────────────────────
    run_config = {
        "dataset": HF_DATASET,
        "split":   HF_SPLIT,
        "n_total_records":  len(ds),
        "n_bonafide":       bonafide_count,
        "n_attack_systems": len([s for s in sys_counts_all if s != "-"]),
        "min_samples":      MIN_SAMPLES,
        "batch_size":       BATCH_SIZE,
        "top_quartile_frac": TOP_QUARTILE,
        "checkpoints": {
            cond: {k: p.name for k, p in ckpt_map.items() if p.exists()}
            for cond, ckpt_map in condition_ckpts.items()
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"Wrote {OUT_DIR / 'run_config.json'}")

    # ══════════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ASVSPOOF E6 RANKING LOCK — SUMMARY")
    print("=" * 70)

    # Per-seed EER rankings
    for cond in ("robust_goat", "goat"):
        valid = valid_robust if cond == "robust_goat" else valid_goat
        print(f"\n── {cond.upper()} ({len(seed_eers[cond])} seed(s), {len(valid)} systems) ──")
        for seed_name in sorted(seed_eers[cond].keys()):
            eer_map = seed_eers[cond][seed_name]
            ranked  = sorted(eer_map.items(), key=lambda x: -x[1])
            print(f"\n  {seed_name} — ranking (hardest to easiest):")
            for sys, e in ranked:
                print(f"    {e:.4f}  {sys}")

    # Pairwise Spearman
    if robust_pairs:
        print(f"\n── ROBUST_GOAT pairwise Spearman rho (system-level) ──")
        for pair, rho in robust_pairs.items():
            print(f"  {pair:30s}  rho={rho:+.4f}")
        sl = spearman_out["robust_goat"]["system_level"]
        print(f"  → mean={sl['mean']:+.4f}  [{sl['min']:+.4f}, {sl['max']:+.4f}]")

    # Category flips
    if isinstance(robust_flips.get("n_flips"), int):
        print(f"\nCategory flips (top-{int(TOP_QUARTILE*100)}% = "
              f"top-{robust_flips['n_top']} system(s)): "
              f"{robust_flips['n_flips']} / {len(valid_robust)}")
        for seed_name, top_set in robust_flips.get("per_seed_top", {}).items():
            print(f"  {seed_name} top: {sorted(top_set)}")

    # Comparison
    print("\n" + "─" * 70)
    print("Cross-dataset comparison (system-level mean Spearman rho):")
    print(f"  ASVspoof robust_goat (EER-based, 6 sys, 3 seeds): "
          f"{robust_mean_rho:+.4f}")
    print(f"  MLAAD    robust_goat (EER-based, 63 sys, 4 seeds): "
          f"{MLAAD_E6['robust_goat']['mean_system_rho']:+.4f}")
    print(f"  MLAAD    goat        (EER-based, 63 sys, 5 seeds): "
          f"{MLAAD_E6['goat']['mean_system_rho']:+.4f}")
    print(f"  ASVspoof E6 original (hub-mass p-values, 3 seeds): "
          f"mean ≈ 0.962 [0.943, 1.000]")

    # Verdict
    print("\n" + "─" * 70)
    print(f"PRE-REGISTERED VERDICT (robust_goat): "
          f"{verdict.get('criterion', 'N/A').upper()}")
    print(f"  {verdict.get('explanation', '')}")

    print(f"\nResults: {OUT_DIR}")
    print(f"Elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
