#!/usr/bin/env python3
"""
mlaad_temporal_intervention.py
================================
Causal intervention: does temporal smoothness CAUSALLY affect EER,
or is the ρ=-0.48 correlation merely observational?

Strategy:
  1. Cache WavLM hidden_states for all test samples once (slow frontend runs once).
  2. Apply 12 controlled perturbations to the cached embeddings:
       (A) Temporal smoothing — moving avg k∈{3,5,10} + exp smooth α∈{0.3,0.5,0.7}
       (B) Temporal jitter   — local perm k∈{3,5,10} + Gauss noise σ∈{0.01,0.05,0.1}
       (C) Baseline          — no modification
  3. Run encoder→GAT→BiLSTM→classifier on each perturbed version (WavLM skipped).
  4. Measure actual phoneme_var and frame_mean_dist after each perturbation.
  5. Compute EER: overall, hard-group, easy-group.
  6. Plot variance vs EER curve; write intervention table and interpretation.

Outputs → experiments/results/mlaad/temporal_intervention/
  temporal_intervention.csv
  variance_vs_eer.png
  hard_vs_easy_sensitivity.png
  intervention_table.md
  causal_interpretation.md
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
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
E6_CSV        = EXP_DIR / "results" / "mlaad" / "e6_ranking_lock" / "per_seed_per_attack_eer.csv"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "temporal_intervention"

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

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
HARD_THRESHOLD = 7
EASY_THRESHOLD = 7
NF_PER_SAMPLE  = 149    # TARGET_SAMPLES // 320 - 1; WavLM frames per 3-second clip
RNG_SEED       = 42

def _best_ckpt(stem: str) -> Path:
    cands = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return cands[0] if cands else CKPT_DIR / f"{stem}.ckpt"

CKPT_PATH = _best_ckpt("mlaad_robust_goat")

# ─── Conditions ───────────────────────────────────────────────────────────────
CONDITIONS = [
    # (name, group, kind, param)
    ("smooth_ma_k3",    "smooth", "moving_avg",  {"k": 3}),
    ("smooth_ma_k5",    "smooth", "moving_avg",  {"k": 5}),
    ("smooth_ma_k10",   "smooth", "moving_avg",  {"k": 10}),
    ("smooth_exp_a03",  "smooth", "exp_smooth",  {"alpha": 0.3}),
    ("smooth_exp_a05",  "smooth", "exp_smooth",  {"alpha": 0.5}),
    ("smooth_exp_a07",  "smooth", "exp_smooth",  {"alpha": 0.7}),
    ("baseline",        "baseline", "none",       {}),
    ("jitter_perm_k3",  "jitter", "local_perm",  {"k": 3}),
    ("jitter_perm_k5",  "jitter", "local_perm",  {"k": 5}),
    ("jitter_perm_k10", "jitter", "local_perm",  {"k": 10}),
    ("jitter_gauss_s001", "jitter", "gauss_noise", {"sigma": 0.01}),
    ("jitter_gauss_s005", "jitter", "gauss_noise", {"sigma": 0.05}),
    ("jitter_gauss_s01",  "jitter", "gauss_noise", {"sigma": 0.1}),
]

# ─── Perturbation functions ────────────────────────────────────────────────────
# All operate on (N, T, D) numpy float32 arrays; return same shape.

def perturb_moving_avg(hs: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average with window k; edges reflect."""
    N, T, D = hs.shape
    # Use F.avg_pool1d via torch for speed
    x = torch.from_numpy(hs).permute(0, 2, 1)     # (N, D, T)
    pad = k // 2
    x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=pad)
    if x.shape[-1] > T:
        x = x[:, :, :T]
    elif x.shape[-1] < T:
        # Pad tail with last value
        tail = x[:, :, -1:].expand(-1, -1, T - x.shape[-1])
        x = torch.cat([x, tail], dim=-1)
    return x.permute(0, 2, 1).numpy()              # (N, T, D)


def perturb_exp_smooth(hs: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential smoothing: h'_t = α·h_t + (1−α)·h'_{t−1}."""
    N, T, D = hs.shape
    result = hs.copy()
    for t in range(1, T):
        result[:, t] = alpha * hs[:, t] + (1 - alpha) * result[:, t - 1]
    return result


def perturb_local_perm(hs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly permute frames within non-overlapping windows of size k."""
    result = hs.copy()
    N, T, D = hs.shape
    for start in range(0, T, k):
        end = min(start + k, T)
        seg_len = end - start
        if seg_len < 2:
            continue
        perm = rng.permutation(seg_len)
        result[:, start:end, :] = result[:, start:end, :][:, perm, :]
    return result


def perturb_gauss_noise(hs: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise: h'_t = h_t + ε, ε ~ N(0, σ²)."""
    noise = rng.standard_normal(hs.shape).astype(np.float32) * sigma
    return hs + noise


def apply_perturbation(hs: np.ndarray, kind: str, params: dict,
                       rng: np.random.Generator) -> np.ndarray:
    if kind == "none":
        return hs
    elif kind == "moving_avg":
        return perturb_moving_avg(hs, params["k"])
    elif kind == "exp_smooth":
        return perturb_exp_smooth(hs, params["alpha"])
    elif kind == "local_perm":
        return perturb_local_perm(hs, params["k"], rng)
    elif kind == "gauss_noise":
        return perturb_gauss_noise(hs, params["sigma"], rng)
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")


# ─── Variance metrics ─────────────────────────────────────────────────────────

def compute_variance_metrics(hs: np.ndarray, pids: np.ndarray) -> tuple[float, float]:
    """
    hs: (N, T, D), pids: (N, T)
    Returns (mean_phoneme_var, mean_frame_mean_dist) across all N samples.
    """
    phon_vars, frame_dists = [], []
    N, T, D = hs.shape
    for i in range(N):
        h = hs[i]      # (T, D)
        p = pids[i]    # (T,)

        # Phoneme segment variance
        segs, start = [], 0
        for t in range(1, T):
            if p[t] != p[t - 1]:
                segs.append(h[start:t].mean(axis=0))
                start = t
        segs.append(h[start:].mean(axis=0))
        if len(segs) > 1:
            phon_vars.append(float(np.stack(segs).var(axis=0).mean()))

        # Frame-to-frame distances
        diffs = np.linalg.norm(h[1:] - h[:-1], axis=-1)
        frame_dists.append(float(diffs.mean()))

    return (float(np.mean(phon_vars)) if phon_vars else float("nan"),
            float(np.mean(frame_dists)) if frame_dists else float("nan"))


# ─── EER helper ───────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """EER from labels (0=bonafide, 1=spoof) and raw logit scores."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_extreme_groups() -> tuple[list[str], list[str]]:
    rows = list(csv.DictReader(E6_CSV.open()))
    cond_seed_maps: dict = {}
    for r in rows:
        k = (r["condition"], r["seed"])
        cond_seed_maps.setdefault(k, {})[r["attack_system"]] = float(r["eer"])

    hard_ctr: dict = defaultdict(int)
    easy_ctr: dict = defaultdict(int)
    for eer_map in cond_seed_maps.values():
        ranked = sorted(eer_map.items(), key=lambda x: -x[1])
        n_q = max(1, int(len(ranked) * 0.25))
        for s, _ in ranked[:n_q]:
            hard_ctr[s] += 1
        for s, _ in ranked[-n_q:]:
            easy_ctr[s] += 1

    hard = [s for s, c in hard_ctr.items() if c >= HARD_THRESHOLD]
    easy = [s for s, c in easy_ctr.items() if c >= EASY_THRESHOLD]
    return hard, easy


class EvalDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(PROCESSED_DIR / rec["audio_path"]).unsqueeze(0)
        return {
            "audio":     wav,
            "label":     0 if rec["label"] == "bonafide" else 1,
            "system_id": rec.get("attack_system", "bonafide"),
        }


def _collate(batch):
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    import gat_l0_attention as gla
    gla.patch_phoneme_loader()
    lit = gla.load_model(CKPT_PATH, device)
    lit.eval()
    return lit


# ─── Cache frontend ───────────────────────────────────────────────────────────

def cache_frontend(lit, records: list[dict], device: torch.device) -> dict:
    """
    Run frozen WavLM frontend once for all records.
    Returns {
      'hs':        (N, T, 768) float32 numpy,
      'pids':      (N, T)      int64  numpy,
      'labels':    (N,)        int32  numpy,
      'system_ids': [N strings],
    }
    """
    from gat_l0_attention import run_frozen_frontend

    gat_model = lit.model
    ds = EvalDataset(records)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, collate_fn=_collate, pin_memory=True)

    all_hs, all_pids, all_labels, all_sids = [], [], [], []
    N = len(records)

    print(f"  Caching WavLM frontend for {N} samples ...")
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            audio = batch["audio"].to(device)
            hs, pids = run_frozen_frontend(audio, gat_model, device)
            all_hs.append(hs.cpu().numpy().astype(np.float32))
            all_pids.append(pids.cpu().numpy())
            all_labels.extend(batch["label"].tolist())
            all_sids.extend(batch["system_id"])
            if (bi + 1) % 20 == 0 or (bi + 1) == len(dl):
                print(f"    {min((bi + 1) * BATCH_SIZE, N)}/{N}")

    return {
        "hs":         np.concatenate(all_hs,  axis=0),  # (N, T, 768)
        "pids":       np.concatenate(all_pids, axis=0),  # (N, T)
        "labels":     np.array(all_labels, dtype=np.int32),
        "system_ids": all_sids,
    }


# ─── Inference with perturbed hidden states ───────────────────────────────────

def infer_perturbed(lit, hs_np: np.ndarray, pids_np: np.ndarray,
                    device: torch.device) -> np.ndarray:
    """
    Run encoder_and_GAT on (N, T, 768) perturbed hidden_states.
    Returns (N,) float32 logits.
    """
    gat_model = lit.model
    N = hs_np.shape[0]
    all_logits = []

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        hs_b   = torch.from_numpy(hs_np[start:end]).to(device)   # (B, T, 768)
        pids_b = torch.from_numpy(pids_np[start:end]).to(device)  # (B, T)
        B      = hs_b.shape[0]
        num_f  = torch.full((B,), NF_PER_SAMPLE, device=device)

        with torch.no_grad():
            result = gat_model.encoder_and_GAT(hs_b, num_f, pids_b)
        logit = result[5]  # (B,) — classification logit
        all_logits.append(logit.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


# ─── Output helpers ───────────────────────────────────────────────────────────

def write_table_md(path: Path, rows: list[dict]):
    header = "| Condition | Group | Overall EER | Hard EER | Easy EER | phoneme_var | frame_dist |"
    sep    = "|-----------|-------|:-----------:|:--------:|:--------:|:-----------:|:----------:|"
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['group']} | {r['overall_eer']:.4f} | "
            f"{r['hard_eer']:.4f} | {r['easy_eer']:.4f} | "
            f"{r['phoneme_var']:.5f} | {r['frame_dist']:.4f} |"
        )
    path.write_text("\n".join(lines))


def write_interpretation(path: Path, rows: list[dict]):
    baseline = next(r for r in rows if r["group"] == "baseline")
    smooth_rows = [r for r in rows if r["group"] == "smooth"]
    jitter_rows = [r for r in rows if r["group"] == "jitter"]

    smooth_eer_delta = np.mean([r["overall_eer"] for r in smooth_rows]) - baseline["overall_eer"]
    jitter_eer_delta = np.mean([r["overall_eer"] for r in jitter_rows]) - baseline["overall_eer"]

    smooth_var_delta = np.mean([r["phoneme_var"] for r in smooth_rows]) - baseline["phoneme_var"]
    jitter_var_delta = np.mean([r["phoneme_var"] for r in jitter_rows]) - baseline["phoneme_var"]

    smooth_direction = "increases" if smooth_eer_delta > 0 else "decreases"
    jitter_direction = "decreases" if jitter_eer_delta < 0 else "increases"

    smooth_consistent = smooth_eer_delta > 0
    jitter_consistent = jitter_eer_delta < 0
    hypothesis_supported = smooth_consistent and jitter_consistent

    hardest_smooth = max(smooth_rows, key=lambda r: r["overall_eer"])
    strongest_jitter = min(jitter_rows, key=lambda r: r["overall_eer"])

    verdict = "SUPPORTED" if hypothesis_supported else "NOT SUPPORTED"
    lines = [
        "# Causal Interpretation — Temporal Smoothness Intervention",
        "",
        f"## Verdict: Hypothesis {verdict}",
        "",
        "## Evidence",
        "",
        f"Baseline EER: {baseline['overall_eer']:.4f} (phoneme_var={baseline['phoneme_var']:.5f})",
        "",
        f"Temporal smoothing (mean Δ EER = {smooth_eer_delta:+.4f}, "
        f"mean Δ phoneme_var = {smooth_var_delta:+.5f}):",
        f"  Smoothing {smooth_direction} EER. "
        f"Strongest smoothing condition: {hardest_smooth['name']} "
        f"(EER={hardest_smooth['overall_eer']:.4f}).",
        "",
        f"Temporal jitter (mean Δ EER = {jitter_eer_delta:+.4f}, "
        f"mean Δ phoneme_var = {jitter_var_delta:+.5f}):",
        f"  Jitter {jitter_direction} EER. "
        f"Strongest jitter condition: {strongest_jitter['name']} "
        f"(EER={strongest_jitter['overall_eer']:.4f}).",
        "",
        "## Causal vs Correlational Assessment",
        "",
    ]
    if hypothesis_supported:
        lines += [
            "The intervention establishes a causal link between temporal embedding variance "
            "and detection difficulty. Artificially reducing variance (smoothing) degrades "
            "the detector's ability to separate bonafide from spoof audio, while adding "
            "temporal inconsistency (jitter) improves separability — consistent with "
            "the detector exploiting temporal artifact signals in the WavLM feature space. "
            "The observational correlation (ρ=-0.48) is therefore not merely a confound of "
            "dataset composition; temporal smoothness is a mechanistically relevant "
            "property that the GAT-based detector depends on.",
        ]
    else:
        lines += [
            "The intervention does not cleanly support a causal interpretation. "
            "While the observational correlation (ρ=-0.48) between phoneme variance and EER "
            "is significant, directly manipulating temporal variance does not produce a "
            "consistent directional effect on EER across smoothing and jitter conditions. "
            "This suggests the correlation is mediated by confounders such as system-level "
            "differences in spectral structure, TTS architecture, or training data — rather "
            "than temporal variance per se being the causal mechanism.",
        ]
    path.write_text("\n".join(lines))


# ─── Plots ────────────────────────────────────────────────────────────────────

GROUP_COLORS = {"smooth": "#2166ac", "baseline": "#000000", "jitter": "#d6604d"}
GROUP_MARKERS = {"smooth": "o", "baseline": "D", "jitter": "s"}


def plot_variance_vs_eer(rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (x_key, x_label) in zip(axes, [
        ("phoneme_var",  "Mean phoneme embedding variance"),
        ("frame_dist",   "Mean frame-to-frame distance"),
    ]):
        for group in ["smooth", "baseline", "jitter"]:
            grp = [r for r in rows if r["group"] == group]
            xs  = [r[x_key] for r in grp]
            ys  = [r["overall_eer"] for r in grp]
            ax.scatter(xs, ys, c=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                       s=80, label=group, zorder=3, alpha=0.9)
            for r in grp:
                ax.annotate(r["name"], (r[x_key], r["overall_eer"]),
                            fontsize=6, ha="left", va="bottom", alpha=0.7)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("EER", fontsize=11)
        ax.set_title(f"{x_label.split()[1].capitalize()} vs EER", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Variance vs EER: Causal Intervention Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_group_sensitivity(rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    group_labels = ["smooth", "baseline", "jitter"]
    eer_keys = [("overall_eer", "All systems"), ("hard_eer", "Hard systems"),
                ("easy_eer", "Easy systems")]

    x_positions = range(len(CONDITIONS))
    cond_names  = [r["name"] for r in rows]

    for ax, (eer_key, title) in zip(axes, eer_keys):
        eers = [r[eer_key] for r in rows]
        colors = [GROUP_COLORS[r["group"]] for r in rows]
        bars = ax.bar(x_positions, eers, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("EER" if eer_key == "overall_eer" else "")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(rows[6]["overall_eer"], color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="baseline")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.suptitle("EER Sensitivity to Temporal Perturbations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(RNG_SEED)

    # ── Extreme groups ────────────────────────────────────────────────────────
    print("\nLoading extreme groups ...")
    hard, easy = load_extreme_groups()
    hard_set = set(hard)
    easy_set  = set(easy)
    print(f"  Hard: {len(hard)}  Easy: {len(easy)}")

    # ── Load all test records ─────────────────────────────────────────────────
    all_records = json.loads(INDIST_JSON.read_text())
    print(f"\nTotal test records: {len(all_records)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {CKPT_PATH.name} ...")
    lit = load_model(device)
    gat_model = lit.model

    # ── Cache frontend (runs WavLM once) ──────────────────────────────────────
    print("\nCaching WavLM frontend ...")
    cache = cache_frontend(lit, all_records, device)
    hs_all   = cache["hs"]        # (N, T, 768)
    pids_all = cache["pids"]      # (N, T)
    labels   = cache["labels"]    # (N,)
    sys_ids  = cache["system_ids"]

    N = len(labels)
    hard_mask = np.array([s in hard_set for s in sys_ids])
    easy_mask = np.array([s in easy_set for s in sys_ids])
    spoof_mask = (labels == 1)

    print(f"  {N} samples cached  (bonafide={int((labels==0).sum())}  "
          f"spoof={int(spoof_mask.sum())})")
    print(f"  Hard-system samples: {int(hard_mask.sum())}  "
          f"Easy-system samples: {int(easy_mask.sum())}")

    # ── Run each condition ────────────────────────────────────────────────────
    results = []
    print(f"\nRunning {len(CONDITIONS)} conditions ...")
    for cond_name, group, kind, params in CONDITIONS:
        print(f"\n  [{group}] {cond_name}")

        # 1. Perturb cached hidden states (CPU, numpy)
        hs_p = apply_perturbation(hs_all, kind, params, rng)

        # 2. Measure actual variance after perturbation (on spoof samples for comparability)
        phoneme_var, frame_dist = compute_variance_metrics(
            hs_p[spoof_mask], pids_all[spoof_mask])
        print(f"    phoneme_var={phoneme_var:.5f}  frame_dist={frame_dist:.4f}")

        # 3. Run encoder_and_GAT on perturbed hs
        logits = infer_perturbed(lit, hs_p, pids_all, device)

        # 4. Compute EERs
        overall_eer = compute_eer(labels, logits)
        hard_eer    = compute_eer(
            labels[hard_mask | (labels == 0)],
            logits[hard_mask | (labels == 0)])
        easy_eer    = compute_eer(
            labels[easy_mask | (labels == 0)],
            logits[easy_mask | (labels == 0)])
        print(f"    EER: overall={overall_eer:.4f}  hard={hard_eer:.4f}  easy={easy_eer:.4f}")

        results.append({
            "name":        cond_name,
            "group":       group,
            "kind":        kind,
            "params":      str(params),
            "overall_eer": overall_eer,
            "hard_eer":    hard_eer,
            "easy_eer":    easy_eer,
            "phoneme_var": phoneme_var,
            "frame_dist":  frame_dist,
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting outputs ...")

    write_csv = lambda path, rows: (
        None if not rows else (
            setattr(write_csv, "_", None),
            (lambda: [
                f.close() or None
                for f in [open(path, "w", newline="")] * 0
            ])()
        )
    )

    # temporal_intervention.csv
    csv_path = OUT_DIR / "temporal_intervention.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["name", "group", "kind", "params", "overall_eer",
                      "hard_eer", "easy_eer", "phoneme_var", "frame_dist"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"  Wrote temporal_intervention.csv")

    # intervention_table.md
    write_table_md(OUT_DIR / "intervention_table.md", results)
    print(f"  Wrote intervention_table.md")

    # causal_interpretation.md
    write_interpretation(OUT_DIR / "causal_interpretation.md", results)
    print(f"  Wrote causal_interpretation.md")

    # variance_vs_eer.png
    plot_variance_vs_eer(results, OUT_DIR / "variance_vs_eer.png")

    # hard_vs_easy_sensitivity.png
    plot_group_sensitivity(results, OUT_DIR / "hard_vs_easy_sensitivity.png")

    print(f"\nAll outputs → {OUT_DIR}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Console summary ───────────────────────────────────────────────────────
    baseline = next(r for r in results if r["group"] == "baseline")
    smooth_rows = [r for r in results if r["group"] == "smooth"]
    jitter_rows = [r for r in results if r["group"] == "jitter"]

    print("\n" + "=" * 70)
    print("TEMPORAL INTERVENTION — SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<22}  {'Group':<10}  {'EER(all)':<10}  "
          f"{'EER(hard)':<10}  {'EER(easy)':<10}  {'phon_var':<12}  {'frame_dist'}")
    print("-" * 88)
    for r in results:
        marker = "<-- baseline" if r["group"] == "baseline" else ""
        print(f"{r['name']:<22}  {r['group']:<10}  {r['overall_eer']:.4f}      "
              f"{r['hard_eer']:.4f}      {r['easy_eer']:.4f}      "
              f"{r['phoneme_var']:.5f}       {r['frame_dist']:.4f}  {marker}")

    smooth_eer_delta = np.mean([r["overall_eer"] for r in smooth_rows]) - baseline["overall_eer"]
    jitter_eer_delta = np.mean([r["overall_eer"] for r in jitter_rows]) - baseline["overall_eer"]
    print(f"\nMean Δ EER (smooth vs baseline):  {smooth_eer_delta:+.4f}  "
          f"({'↑ worse detection' if smooth_eer_delta>0 else '↓ better detection'})")
    print(f"Mean Δ EER (jitter vs baseline):  {jitter_eer_delta:+.4f}  "
          f"({'↓ better detection' if jitter_eer_delta<0 else '↑ worse detection'})")
    verdict = (smooth_eer_delta > 0 and jitter_eer_delta < 0)
    print(f"\nHypothesis (temporal smoothness is causal): "
          f"{'SUPPORTED' if verdict else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
