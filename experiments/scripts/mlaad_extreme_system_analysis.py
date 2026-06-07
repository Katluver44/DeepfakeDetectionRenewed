#!/usr/bin/env python3
"""
mlaad_extreme_system_analysis.py
=================================
Investigates WHY consistently hard and easy MLAAD systems are hard/easy.

Two analyses:
  1. WavLM embedding distance to bonafide (computed fresh)
  2. Phoneme-level attention KL divergence from bonafide (reused from head discovery)

Uses the original mlaad_robust_goat checkpoint and the in-distribution test split.
Covers all 63 systems for full correlation analysis; labels extremes from E6 ranking.

Outputs → experiments/results/mlaad/extreme_system_analysis/
  extreme_systems.json
  wavlm_distances.csv
  phoneme_kl.csv
  difficulty_correlations.json
  analysis_report.md
  wavlm_vs_eer.png
  phoneme_kl_vs_eer.png
  hard_vs_easy_boxplots.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, permutation_test
from torch.utils.data import DataLoader, Dataset

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
E6_CSV        = EXP_DIR / "results" / "mlaad" / "e6_ranking_lock" / "per_seed_per_attack_eer.csv"
HEAD_DISC_JSON = (EXP_DIR / "results" / "mlaad" / "head_discovery" /
                  "mlaad_robust_in_distribution" / "head_ranking.json")
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "extreme_system_analysis"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
MAX_PER_SYSTEM   = 50    # utterances to sample per system for WavLM embeddings
BATCH_SIZE       = 16
HARD_THRESHOLD   = 7     # appear in top-quartile in >= this many of 9 seed-condition slots
EASY_THRESHOLD   = 7
PERM_N           = 5000
RNG_SEED         = 42

def _best_ckpt(stem: str) -> Path:
    cands = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return cands[0] if cands else CKPT_DIR / f"{stem}.ckpt"

CKPT_PATH = _best_ckpt("mlaad_robust_goat")


# ─── Step 0: Define extreme groups ───────────────────────────────────────────
def load_extreme_groups() -> tuple[list[str], list[str], dict[str, float]]:
    """
    Returns (hard_systems, easy_systems, mean_eer_per_system).
    Hard = in top-quartile (highest EER) in >= HARD_THRESHOLD of 9 seed-condition slots.
    Easy = in bottom-quartile (lowest EER) in >= EASY_THRESHOLD slots.
    """
    rows = list(csv.DictReader(E6_CSV.open()))

    # Build per (condition, seed) EER maps
    cond_seed_maps: dict[tuple, dict[str, float]] = {}
    for r in rows:
        k = (r["condition"], r["seed"])
        cond_seed_maps.setdefault(k, {})[r["attack_system"]] = float(r["eer"])

    # Mean EER across all slots
    sys_eer_accum: dict[str, list] = defaultdict(list)
    for r in rows:
        sys_eer_accum[r["attack_system"]].append(float(r["eer"]))
    mean_eer = {s: float(np.mean(v)) for s, v in sys_eer_accum.items()}

    # Top-quartile counts (hardest)
    hard_counter: dict[str, int] = defaultdict(int)
    easy_counter: dict[str, int] = defaultdict(int)

    for (cond, seed), eer_map in cond_seed_maps.items():
        ranked = sorted(eer_map.items(), key=lambda x: -x[1])   # high EER first
        n_q    = max(1, int(len(ranked) * 0.25))
        for sys, _ in ranked[:n_q]:
            hard_counter[sys] += 1
        for sys, _ in ranked[-n_q:]:
            easy_counter[sys] += 1

    hard = sorted([s for s, c in hard_counter.items() if c >= HARD_THRESHOLD],
                  key=lambda s: -mean_eer[s])
    easy = sorted([s for s, c in easy_counter.items() if c >= EASY_THRESHOLD],
                  key=lambda s:  mean_eer[s])

    return hard, easy, mean_eer, hard_counter, easy_counter


# ─── Dataset ─────────────────────────────────────────────────────────────────
from loader import TARGET_SR


class EvalDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(PROCESSED_DIR / rec["audio_path"]).unsqueeze(0)
        y   = 0 if rec["label"] == "bonafide" else 1
        return {"audio": wav, "label": y,
                "attack_system": rec.get("attack_system", "bonafide")}


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


# ─── Model + frozen frontend ──────────────────────────────────────────────────
import gat_l0_attention as gla


def load_model(device: torch.device):
    gla.patch_phoneme_loader()
    lit = gla.load_model(CKPT_PATH, device)
    lit.eval()
    return lit


def extract_wavlm_embeddings(
    model,
    records: list[dict],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    For each system (including 'bonafide'), extract mean-pooled WavLM
    hidden_states per utterance. Returns {system: (N, 768)} array.
    """
    from gat_l0_attention import run_frozen_frontend

    gat_model = model.model
    ds = EvalDataset(records)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, collate_fn=collate_fn, pin_memory=True)

    sys_embeds: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for batch in dl:
            audio = batch["audio"].to(device)
            hs, _ = run_frozen_frontend(audio, gat_model, device)
            # hs: (B, T, 768) — mean-pool over time
            emb = hs.mean(dim=1).cpu().numpy()  # (B, 768)
            for i, sys_id in enumerate(batch["attack_system"]):
                tag = sys_id if batch["label"][i] == 1 else "bonafide"
                sys_embeds[tag].append(emb[i])

    return {k: np.stack(v) for k, v in sys_embeds.items()}


# ─── Step 1: WavLM distances ─────────────────────────────────────────────────
def compute_wavlm_distances(
    sys_embeds: dict[str, np.ndarray],
    all_systems: list[str],
) -> dict[str, dict]:
    """
    For each system, compare its embedding distribution to bonafide.
    Returns per-system dict with cosine_dist, l2_dist, per_utt_cosine stats.
    """
    bf_embs  = sys_embeds["bonafide"]            # (N_bf, 768)
    bf_mean  = bf_embs.mean(axis=0)              # (768,)
    bf_mean_norm = bf_mean / (np.linalg.norm(bf_mean) + 1e-10)

    results = {}
    for sys in all_systems:
        if sys not in sys_embeds:
            results[sys] = None
            continue
        embs = sys_embeds[sys]                   # (N, 768)
        sys_mean = embs.mean(axis=0)             # (768,)
        sys_mean_norm = sys_mean / (np.linalg.norm(sys_mean) + 1e-10)

        # Centroid-to-centroid
        cos_cent  = float(1.0 - np.dot(sys_mean_norm, bf_mean_norm))
        l2_cent   = float(np.linalg.norm(sys_mean - bf_mean))

        # Per-utterance cosine distance to bonafide centroid
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        per_utt_cos = 1.0 - embs_norm @ bf_mean_norm   # (N,)

        results[sys] = {
            "n_utterances":    len(embs),
            "cosine_centroid": cos_cent,
            "l2_centroid":     l2_cent,
            "per_utt_cosine_mean": float(per_utt_cos.mean()),
            "per_utt_cosine_std":  float(per_utt_cos.std()),
            "per_utt_cosine_median": float(np.median(per_utt_cos)),
        }

    return results


# ─── Step 2: Phoneme KL from head discovery ───────────────────────────────────
def load_phoneme_kl() -> dict[str, float]:
    """
    Load per-system mean KL divergence from bonafide attention distribution.
    Already computed in existing head discovery results.
    Higher KL = attention patterns more different from bonafide.
    """
    data = json.loads(HEAD_DISC_JSON.read_text())
    psk  = data["per_system_kl"]
    return {sys: float(v["mean_kl"]) for sys, v in psk.items()}


# ─── Correlation helpers ──────────────────────────────────────────────────────
def correlate_with_eer(
    x_values: list[float],
    y_eer:    list[float],
    name:     str,
) -> dict:
    x  = np.array(x_values)
    y  = np.array(y_eer)
    rho,  p_spearman = spearmanr(x, y)
    r,    p_pearson  = pearsonr(x, y)
    return {
        "predictor":   name,
        "n":           len(x),
        "spearman_rho": round(float(rho), 4),
        "spearman_p":   round(float(p_spearman), 4),
        "pearson_r":    round(float(r), 4),
        "pearson_p":    round(float(p_pearson), 4),
    }


def group_permutation_test(hard_vals: list, easy_vals: list) -> dict:
    """Permutation test: is mean(hard) != mean(easy)?"""
    h, e = np.array(hard_vals), np.array(easy_vals)
    obs  = float(h.mean() - e.mean())
    combined = np.concatenate([h, e])
    rng = np.random.default_rng(RNG_SEED)
    count = 0
    for _ in range(PERM_N):
        perm = rng.permutation(combined)
        diff = perm[:len(h)].mean() - perm[len(h):].mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = count / PERM_N
    d_cohen = obs / (combined.std() + 1e-10)
    return {
        "hard_mean":  round(float(h.mean()), 5),
        "easy_mean":  round(float(e.mean()), 5),
        "difference": round(obs, 5),
        "cohen_d":    round(float(d_cohen), 4),
        "perm_p":     round(p, 4),
        "n_perm":     PERM_N,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────
HARD_COLOR = "#d62728"
EASY_COLOR = "#1f77b4"
GREY_COLOR = "#aaaaaa"


def _label_color(sys: str, hard: set, easy: set) -> tuple:
    if sys in hard: return HARD_COLOR, "hard"
    if sys in easy: return EASY_COLOR, "easy"
    return GREY_COLOR, "mid"


def plot_scatter(
    x_vals:  list,
    y_vals:  list,
    labels:  list[str],
    hard_set: set,
    easy_set: set,
    xlabel: str,
    ylabel: str,
    title:  str,
    path:   Path,
    rho:    float,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for x, y, lbl in zip(x_vals, y_vals, labels):
        color, group = _label_color(lbl, hard_set, easy_set)
        ax.scatter(x, y, color=color, s=55, alpha=0.8, zorder=3,
                   edgecolors="white", linewidths=0.4)
        if group != "mid":
            ax.annotate(lbl, (x, y), fontsize=6, alpha=0.85,
                        xytext=(3, 3), textcoords="offset points")

    # Trend line
    z = np.polyfit(x_vals, y_vals, 1)
    xr = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(xr, np.polyval(z, xr), "k--", lw=1, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}  (ρ={rho:+.3f})", fontsize=12)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=HARD_COLOR, label="hard (≥7/9 slots)"),
                  Patch(facecolor=EASY_COLOR, label="easy (≥7/9 slots)"),
                  Patch(facecolor=GREY_COLOR, label="mid")]
    ax.legend(handles=legend_els, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplots(
    hard_wavlm: list, easy_wavlm: list,
    hard_kl:    list, easy_kl:    list,
    path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, hard_v, easy_v, title, ylabel in [
        (axes[0], hard_wavlm, easy_wavlm,
         "WavLM cosine distance\nto bonafide centroid",
         "Cosine distance"),
        (axes[1], hard_kl, easy_kl,
         "Phoneme attention KL\nfrom bonafide distribution",
         "Mean KL divergence"),
    ]:
        bp = ax.boxplot([hard_v, easy_v], patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(HARD_COLOR + "aa")
        bp["boxes"][1].set_facecolor(EASY_COLOR + "aa")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Hard\n(≥7/9)", "Easy\n(≥7/9)"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─── Report ───────────────────────────────────────────────────────────────────
def write_report(
    hard: list[str], easy: list[str],
    mean_eer: dict,
    wavlm_res: dict,
    kl_map: dict,
    wavlm_corr: dict,
    kl_corr: dict,
    wavlm_group: dict,
    kl_group: dict,
    out_path: Path,
):
    lines = [
        "# MLAAD Extreme System Analysis",
        "",
        "## Extreme Groups",
        "",
        f"**Hard systems** (≥{HARD_THRESHOLD}/9 seed-condition top-quartile slots):",
        "",
        "| System | Mean EER | WavLM dist | Phoneme KL |",
        "|--------|----------|-----------|-----------|",
    ]
    for s in hard:
        eer  = mean_eer.get(s, float("nan"))
        wd   = wavlm_res[s]["cosine_centroid"] if wavlm_res.get(s) else float("nan")
        kl   = kl_map.get(s, float("nan"))
        lines.append(f"| {s} | {eer:.3f} | {wd:.4f} | {kl:.4f} |")

    lines += ["", f"**Easy systems** (≥{EASY_THRESHOLD}/9 slots):", "",
              "| System | Mean EER | WavLM dist | Phoneme KL |",
              "|--------|----------|-----------|-----------|"]
    for s in easy:
        eer  = mean_eer.get(s, float("nan"))
        wd   = wavlm_res[s]["cosine_centroid"] if wavlm_res.get(s) else float("nan")
        kl   = kl_map.get(s, float("nan"))
        lines.append(f"| {s} | {eer:.3f} | {wd:.4f} | {kl:.4f} |")

    lines += [
        "",
        "## Correlation with EER (all 63 systems)",
        "",
        "| Predictor | Spearman ρ | p | Pearson r | p |",
        "|-----------|-----------|---|----------|---|",
        (f"| WavLM cosine dist | {wavlm_corr['spearman_rho']:+.3f} | "
         f"{wavlm_corr['spearman_p']:.3f} | {wavlm_corr['pearson_r']:+.3f} | "
         f"{wavlm_corr['pearson_p']:.3f} |"),
        (f"| Phoneme KL div    | {kl_corr['spearman_rho']:+.3f} | "
         f"{kl_corr['spearman_p']:.3f} | {kl_corr['pearson_r']:+.3f} | "
         f"{kl_corr['pearson_p']:.3f} |"),
        "",
        "## Hard vs Easy Group Comparison",
        "",
        "### WavLM cosine distance to bonafide",
        (f"Hard mean: {wavlm_group['hard_mean']:.4f} | "
         f"Easy mean: {wavlm_group['easy_mean']:.4f} | "
         f"Δ={wavlm_group['difference']:+.4f} | "
         f"Cohen d={wavlm_group['cohen_d']:+.3f} | "
         f"perm p={wavlm_group['perm_p']:.3f}"),
        "",
        "### Phoneme attention KL from bonafide",
        (f"Hard mean: {kl_group['hard_mean']:.4f} | "
         f"Easy mean: {kl_group['easy_mean']:.4f} | "
         f"Δ={kl_group['difference']:+.4f} | "
         f"Cohen d={kl_group['cohen_d']:+.3f} | "
         f"perm p={kl_group['perm_p']:.3f}"),
        "",
        "## Interpretation",
        "",
    ]

    # Auto-generate interpretation
    wavlm_sig = wavlm_group["perm_p"] < 0.05
    kl_sig    = kl_group["perm_p"] < 0.05
    wavlm_dir = wavlm_group["difference"] < 0   # hard < easy means harder = closer

    interp = []
    if wavlm_corr["spearman_rho"] < -0.2 and wavlm_corr["spearman_p"] < 0.05:
        interp.append(
            "WavLM distance to bonafide **negatively correlates** with EER "
            f"(ρ={wavlm_corr['spearman_rho']:+.3f}): systems closer to bonafide in "
            "representation space are harder to detect. "
            "**Consistent with the proximity hypothesis.**"
        )
    elif wavlm_corr["spearman_rho"] > 0.2 and wavlm_corr["spearman_p"] < 0.05:
        interp.append(
            "WavLM distance to bonafide **positively correlates** with EER "
            f"(ρ={wavlm_corr['spearman_rho']:+.3f}): more distant systems are harder. "
            "**Inconsistent with the proximity hypothesis.** "
            "Hard systems may differ from bonafide in ways the detector cannot leverage."
        )
    else:
        interp.append(
            f"WavLM distance does NOT significantly predict EER "
            f"(ρ={wavlm_corr['spearman_rho']:+.3f}, p={wavlm_corr['spearman_p']:.3f}). "
            "Embedding proximity to bonafide does not explain detection difficulty."
        )

    if kl_corr["spearman_rho"] < -0.2 and kl_corr["spearman_p"] < 0.05:
        interp.append(
            "Phoneme KL divergence **negatively correlates** with EER "
            f"(ρ={kl_corr['spearman_rho']:+.3f}): systems with attention patterns "
            "closer to bonafide are harder. **Consistent with phonemic proximity hypothesis.**"
        )
    elif kl_corr["spearman_rho"] > 0.2 and kl_corr["spearman_p"] < 0.05:
        interp.append(
            "Phoneme KL divergence **positively correlates** with EER "
            f"(ρ={kl_corr['spearman_rho']:+.3f}): systems with MORE different "
            "attention patterns are harder. **Unexpected — phonemic divergence makes "
            "the system harder, not easier, to detect.**"
        )
    else:
        interp.append(
            f"Phoneme KL divergence does NOT significantly predict EER "
            f"(ρ={kl_corr['spearman_rho']:+.3f}, p={kl_corr['spearman_p']:.3f}). "
            "Attention-level differences from bonafide do not explain difficulty."
        )

    if wavlm_sig:
        direction = "closer to" if wavlm_dir else "further from"
        interp.append(
            f"Group comparison confirms hard systems are **{direction} bonafide** in "
            f"WavLM space (perm p={wavlm_group['perm_p']:.3f}, d={wavlm_group['cohen_d']:+.3f})."
        )
    else:
        interp.append(
            f"Hard vs easy WavLM distance difference is **not significant** "
            f"(perm p={wavlm_group['perm_p']:.3f})."
        )

    if kl_sig:
        kl_dir = "lower" if kl_group["difference"] < 0 else "higher"
        interp.append(
            f"Group comparison: hard systems have **{kl_dir} phoneme KL** than easy "
            f"(perm p={kl_group['perm_p']:.3f}, d={kl_group['cohen_d']:+.3f})."
        )
    else:
        interp.append(
            f"Hard vs easy phoneme KL difference is **not significant** "
            f"(perm p={kl_group['perm_p']:.3f})."
        )

    # Notable violations
    violations = []
    for s in hard:
        if kl_map.get(s, 0) < 0.05:
            violations.append(f"{s} (hard, KL={kl_map[s]:.4f} — very low)")
    for s in easy:
        if kl_map.get(s, 0) > 0.12:
            violations.append(f"{s} (easy, KL={kl_map[s]:.4f} — very high)")
    if violations:
        interp.append(
            "**Notable violations of the KL hypothesis:** " + ", ".join(violations)
        )

    lines += ["- " + i for i in interp]
    out_path.write_text("\n".join(lines) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ── Step 0: extreme groups ────────────────────────────────────────────────
    print("Defining extreme groups ...", flush=True)
    hard, easy, mean_eer, hard_ctr, easy_ctr = load_extreme_groups()
    all_systems = sorted(mean_eer.keys())
    hard_set, easy_set = set(hard), set(easy)

    print(f"  Hard ({len(hard)}): {hard}")
    print(f"  Easy ({len(easy)}): {easy}")

    extreme_json = {
        "hard_threshold_slots": HARD_THRESHOLD,
        "easy_threshold_slots": EASY_THRESHOLD,
        "total_seed_condition_slots": 9,
        "hard_systems": [
            {"system": s, "slots": hard_ctr[s], "mean_eer": round(mean_eer[s], 4)}
            for s in hard
        ],
        "easy_systems": [
            {"system": s, "slots": easy_ctr[s], "mean_eer": round(mean_eer[s], 4)}
            for s in easy
        ],
    }
    (OUT_DIR / "extreme_systems.json").write_text(json.dumps(extreme_json, indent=2))
    print(f"  Wrote extreme_systems.json")

    # ── Load test records ──────────────────────────────────────────────────────
    records: list[dict] = json.loads(INDIST_JSON.read_text())
    # Stratified sample: up to MAX_PER_SYSTEM per system, all bonafide
    rng = np.random.default_rng(RNG_SEED)
    by_sys: dict[str, list] = defaultdict(list)
    for r in records:
        tag = r["attack_system"] if r["label"] == "spoof" else "bonafide"
        by_sys[tag].append(r)

    sampled: list[dict] = []
    for tag, recs in by_sys.items():
        idx = rng.choice(len(recs), min(len(recs), MAX_PER_SYSTEM), replace=False)
        sampled.extend(recs[i] for i in idx)

    print(f"\nSampled {len(sampled)} records "
          f"({sum(1 for r in sampled if r['label']=='bonafide')} bonafide, "
          f"{sum(1 for r in sampled if r['label']=='spoof')} spoof)", flush=True)

    # ── Step 1: WavLM embeddings ──────────────────────────────────────────────
    print(f"\nLoading {CKPT_PATH.name} ...", flush=True)
    model = load_model(device)

    print("Extracting WavLM embeddings ...", flush=True)
    sys_embeds = extract_wavlm_embeddings(model, sampled, device)
    print(f"  Done — {len(sys_embeds)} systems including bonafide", flush=True)

    wavlm_res = compute_wavlm_distances(sys_embeds, all_systems)

    # Write CSV
    wavlm_rows = []
    for sys in sorted(all_systems):
        r = wavlm_res.get(sys)
        if r is None:
            continue
        wavlm_rows.append({
            "attack_system":      sys,
            "group":              "hard" if sys in hard_set else
                                  "easy" if sys in easy_set else "mid",
            "mean_eer":           round(mean_eer.get(sys, float("nan")), 4),
            "n_utterances":       r["n_utterances"],
            "cosine_centroid":    round(r["cosine_centroid"], 6),
            "l2_centroid":        round(r["l2_centroid"], 6),
            "per_utt_cosine_mean": round(r["per_utt_cosine_mean"], 6),
            "per_utt_cosine_std":  round(r["per_utt_cosine_std"], 6),
        })
    with (OUT_DIR / "wavlm_distances.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(wavlm_rows[0].keys()))
        writer.writeheader(); writer.writerows(wavlm_rows)
    print(f"  Wrote wavlm_distances.csv")

    # ── Step 2: Phoneme KL (reuse head discovery) ─────────────────────────────
    print("\nLoading phoneme KL from head discovery ...", flush=True)
    kl_map = load_phoneme_kl()
    print(f"  {len(kl_map)} systems")

    kl_rows = []
    for sys in sorted(all_systems):
        kl = kl_map.get(sys)
        if kl is None:
            continue
        kl_rows.append({
            "attack_system": sys,
            "group":         "hard" if sys in hard_set else
                             "easy" if sys in easy_set else "mid",
            "mean_eer":      round(mean_eer.get(sys, float("nan")), 4),
            "mean_kl":       round(kl, 6),
        })
    with (OUT_DIR / "phoneme_kl.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(kl_rows[0].keys()))
        writer.writeheader(); writer.writerows(kl_rows)
    print(f"  Wrote phoneme_kl.csv")

    # ── Step 3: Correlations ─────────────────────────────────────────────────
    print("\nComputing correlations ...", flush=True)
    common = [s for s in all_systems
              if wavlm_res.get(s) and kl_map.get(s) is not None]

    eer_v  = [mean_eer[s]                         for s in common]
    wd_v   = [wavlm_res[s]["cosine_centroid"]      for s in common]
    kl_v   = [kl_map[s]                            for s in common]

    wavlm_corr = correlate_with_eer(wd_v, eer_v, "wavlm_cosine")
    kl_corr    = correlate_with_eer(kl_v, eer_v, "phoneme_kl")
    print(f"  WavLM corr: ρ={wavlm_corr['spearman_rho']:+.3f} "
          f"(p={wavlm_corr['spearman_p']:.3f})")
    print(f"  KL corr:    ρ={kl_corr['spearman_rho']:+.3f} "
          f"(p={kl_corr['spearman_p']:.3f})")

    # Group comparisons
    hard_wd = [wavlm_res[s]["cosine_centroid"] for s in hard if wavlm_res.get(s)]
    easy_wd = [wavlm_res[s]["cosine_centroid"] for s in easy if wavlm_res.get(s)]
    hard_kl = [kl_map[s] for s in hard if s in kl_map]
    easy_kl = [kl_map[s] for s in easy if s in kl_map]

    wavlm_group = group_permutation_test(hard_wd, easy_wd)
    kl_group    = group_permutation_test(hard_kl, easy_kl)
    print(f"  Hard WavLM mean: {wavlm_group['hard_mean']:.4f}  "
          f"Easy: {wavlm_group['easy_mean']:.4f}  "
          f"Δ={wavlm_group['difference']:+.4f}  perm-p={wavlm_group['perm_p']:.3f}")
    print(f"  Hard KL mean:    {kl_group['hard_mean']:.4f}  "
          f"Easy: {kl_group['easy_mean']:.4f}  "
          f"Δ={kl_group['difference']:+.4f}  perm-p={kl_group['perm_p']:.3f}")

    corr_out = {
        "n_systems":     len(common),
        "wavlm_vs_eer":  wavlm_corr,
        "kl_vs_eer":     kl_corr,
        "group_test_wavlm": wavlm_group,
        "group_test_kl":    kl_group,
    }
    (OUT_DIR / "difficulty_correlations.json").write_text(
        json.dumps(corr_out, indent=2))
    print(f"  Wrote difficulty_correlations.json")

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots ...", flush=True)

    plot_scatter(wd_v, eer_v, common, hard_set, easy_set,
                 "WavLM cosine distance to bonafide centroid",
                 "Mean EER (across 9 seed-condition slots)",
                 "WavLM embedding proximity vs attack difficulty",
                 OUT_DIR / "wavlm_vs_eer.png",
                 wavlm_corr["spearman_rho"])

    plot_scatter(kl_v, eer_v, common, hard_set, easy_set,
                 "Phoneme attention KL divergence from bonafide",
                 "Mean EER (across 9 seed-condition slots)",
                 "Phoneme KL divergence vs attack difficulty",
                 OUT_DIR / "phoneme_kl_vs_eer.png",
                 kl_corr["spearman_rho"])

    plot_boxplots(hard_wd, easy_wd, hard_kl, easy_kl,
                  OUT_DIR / "hard_vs_easy_boxplots.png")
    print(f"  Wrote 3 figures")

    # ── Step 5: Report ────────────────────────────────────────────────────────
    write_report(hard, easy, mean_eer, wavlm_res, kl_map,
                 wavlm_corr, kl_corr, wavlm_group, kl_group,
                 OUT_DIR / "analysis_report.md")
    print(f"  Wrote analysis_report.md")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXTREME SYSTEM ANALYSIS — SUMMARY")
    print("=" * 65)

    print(f"\nHard systems ({len(hard)})  |  EER  |  WavLM dist  |  Phoneme KL")
    for s in hard:
        eer = mean_eer.get(s, float("nan"))
        wd  = wavlm_res[s]["cosine_centroid"] if wavlm_res.get(s) else float("nan")
        kl  = kl_map.get(s, float("nan"))
        print(f"  {s:<35s}  {eer:.3f}  {wd:.4f}  {kl:.4f}")

    print(f"\nEasy systems ({len(easy)})  |  EER  |  WavLM dist  |  Phoneme KL")
    for s in easy:
        eer = mean_eer.get(s, float("nan"))
        wd  = wavlm_res[s]["cosine_centroid"] if wavlm_res.get(s) else float("nan")
        kl  = kl_map.get(s, float("nan"))
        print(f"  {s:<35s}  {eer:.3f}  {wd:.4f}  {kl:.4f}")

    print(f"\nCorrelations with EER (n={len(common)}):")
    print(f"  WavLM cosine dist:  ρ={wavlm_corr['spearman_rho']:+.3f}  "
          f"p={wavlm_corr['spearman_p']:.3f}")
    print(f"  Phoneme KL div:     ρ={kl_corr['spearman_rho']:+.3f}  "
          f"p={kl_corr['spearman_p']:.3f}")

    print(f"\nGroup comparison (hard vs easy):")
    print(f"  WavLM:  Δ={wavlm_group['difference']:+.4f}  "
          f"d={wavlm_group['cohen_d']:+.3f}  perm-p={wavlm_group['perm_p']:.3f}")
    print(f"  KL:     Δ={kl_group['difference']:+.4f}  "
          f"d={kl_group['cohen_d']:+.3f}  perm-p={kl_group['perm_p']:.3f}")

    print(f"\nResults: {OUT_DIR}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
