#!/usr/bin/env python3
"""
mlaad_waveform_intervention.py
================================
Waveform-level causal test: do temporal smoothness manipulations applied to
raw audio — before WavLM re-encoding — produce the same EER trends as the
embedding-level intervention (mlaad_temporal_intervention.py)?

If yes → the causal mechanism is end-to-end (true acoustic property).
If no  → the previous finding was a representation-level artifact (WavLM
         smoothing≠audio smoothing).

Waveform conditions (all applied BEFORE WavLM runs):
  (A) Smoothing: moving average {5ms, 10ms, 20ms} + Gaussian LPF {σ=1ms, 2.5ms, 5ms}
  (B) Jitter:    segment shuffle {5ms, 10ms, 20ms} + additive noise {σ=0.001, 0.005, 0.01}
  (C) Baseline:  original waveform unchanged

Strategy:
  1. Cache raw waveforms in RAM (avoid re-reading disk 13×).
  2. For each condition: transform waveforms (CPU numpy) → GPU → full WavLM+GAT pipeline.
  3. Capture hidden_states from WavLM to measure post-encoding phoneme_var and frame_dist.
  4. Compute EER + AUC overall, hard group, easy group.
  5. Overlay vs embedding-level results (loaded from temporal_intervention.csv).

Outputs → experiments/results/mlaad/waveform_intervention/
  waveform_intervention.csv
  waveform_vs_eer.png          (waveform-level curve + embedding-level overlay)
  hard_vs_easy_waveform.png
  waveform_intervention_table.md
  end_to_end_conclusion.md
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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
E6_CSV        = EXP_DIR / "results" / "mlaad" / "e6_ranking_lock" / "per_seed_per_attack_eer.csv"
EMBED_CSV     = EXP_DIR / "results" / "mlaad" / "temporal_intervention" / "temporal_intervention.csv"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "waveform_intervention"

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
SR             = 16000      # WavLM sample rate
BATCH_SIZE     = 32
HARD_THRESHOLD = 7
EASY_THRESHOLD = 7
NF_PER_SAMPLE  = 149        # WavLM frames per 3-second clip
RNG_SEED       = 42


def _best_ckpt(stem: str) -> Path:
    cands = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return cands[0] if cands else CKPT_DIR / f"{stem}.ckpt"


CKPT_PATH = _best_ckpt("mlaad_robust_goat")

# ─── Waveform conditions ──────────────────────────────────────────────────────
# Each: (name, group, kind, params)
CONDITIONS = [
    ("wav_ma_5ms",      "smooth",   "moving_avg",    {"window_ms": 5}),
    ("wav_ma_10ms",     "smooth",   "moving_avg",    {"window_ms": 10}),
    ("wav_ma_20ms",     "smooth",   "moving_avg",    {"window_ms": 20}),
    ("wav_glpf_1ms",    "smooth",   "gaussian_lpf",  {"sigma_ms": 1.0}),
    ("wav_glpf_2p5ms",  "smooth",   "gaussian_lpf",  {"sigma_ms": 2.5}),
    ("wav_glpf_5ms",    "smooth",   "gaussian_lpf",  {"sigma_ms": 5.0}),
    ("wav_baseline",    "baseline", "none",          {}),
    ("wav_shuf_5ms",    "jitter",   "seg_shuffle",   {"window_ms": 5}),
    ("wav_shuf_10ms",   "jitter",   "seg_shuffle",   {"window_ms": 10}),
    ("wav_shuf_20ms",   "jitter",   "seg_shuffle",   {"window_ms": 20}),
    ("wav_noise_001",   "jitter",   "gauss_noise",   {"sigma": 0.001}),
    ("wav_noise_005",   "jitter",   "gauss_noise",   {"sigma": 0.005}),
    ("wav_noise_01",    "jitter",   "gauss_noise",   {"sigma": 0.01}),
]

# ─── Waveform perturbation functions ─────────────────────────────────────────
# All operate on (N, L) numpy float32 arrays.

def wav_moving_avg(wavs: np.ndarray, window_ms: float) -> np.ndarray:
    """Centered moving average over waveform samples."""
    k = max(1, int(window_ms * SR / 1000))
    if k == 1:
        return wavs
    # Use cumsum trick for O(N·L) instead of O(N·L·k)
    out = np.zeros_like(wavs)
    cs = np.cumsum(np.pad(wavs, ((0, 0), (k // 2, k // 2)), mode="edge"), axis=1)
    out = (cs[:, k:] - cs[:, :-k]) / k
    # Trim to original length
    return out[:, :wavs.shape[1]].astype(np.float32)


def wav_gaussian_lpf(wavs: np.ndarray, sigma_ms: float) -> np.ndarray:
    """Gaussian low-pass filter applied per sample."""
    sigma = sigma_ms * SR / 1000
    result = np.empty_like(wavs)
    for i in range(len(wavs)):
        result[i] = gaussian_filter1d(wavs[i].astype(np.float64), sigma=sigma)
    return result.astype(np.float32)


def wav_seg_shuffle(wavs: np.ndarray, window_ms: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Shuffle audio samples within non-overlapping windows."""
    k = max(2, int(window_ms * SR / 1000))
    result = wavs.copy()
    L = wavs.shape[1]
    for start in range(0, L, k):
        end = min(start + k, L)
        seg_len = end - start
        if seg_len < 2:
            continue
        perm = rng.permutation(seg_len)
        result[:, start:end] = result[:, start:end][:, perm]
    return result


def wav_gauss_noise(wavs: np.ndarray, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Additive i.i.d. Gaussian noise at the waveform level."""
    noise = rng.standard_normal(wavs.shape).astype(np.float32) * sigma
    return wavs + noise


def apply_waveform_transform(wavs: np.ndarray, kind: str, params: dict,
                              rng: np.random.Generator) -> np.ndarray:
    """wavs: (N, L) float32."""
    if kind == "none":
        return wavs
    elif kind == "moving_avg":
        return wav_moving_avg(wavs, params["window_ms"])
    elif kind == "gaussian_lpf":
        return wav_gaussian_lpf(wavs, params["sigma_ms"])
    elif kind == "seg_shuffle":
        return wav_seg_shuffle(wavs, params["window_ms"], rng)
    elif kind == "gauss_noise":
        return wav_gauss_noise(wavs, params["sigma"], rng)
    else:
        raise ValueError(kind)


# ─── Variance metrics (same as higher-order analysis) ─────────────────────────

def phoneme_segment_var(hs: np.ndarray, pids: np.ndarray) -> float:
    """hs (T, 768), pids (T,) → mean variance across phoneme segment embeddings."""
    segs, start = [], 0
    T = len(pids)
    for t in range(1, T):
        if pids[t] != pids[t - 1]:
            segs.append(hs[start:t].mean(axis=0))
            start = t
    segs.append(hs[start:].mean(axis=0))
    if len(segs) < 2:
        return 0.0
    return float(np.stack(segs).var(axis=0).mean())


def frame_mean_dist(hs: np.ndarray) -> float:
    """hs (T, 768) → mean adjacent-frame L2 distance."""
    diffs = np.linalg.norm(hs[1:] - hs[:-1], axis=-1)
    return float(diffs.mean())


# ─── EER / AUC ────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


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

    return ([s for s, c in hard_ctr.items() if c >= HARD_THRESHOLD],
            [s for s, c in easy_ctr.items() if c >= EASY_THRESHOLD])


# ─── Cache raw waveforms ──────────────────────────────────────────────────────

def cache_waveforms(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load all waveforms into RAM.
    Returns (wavs, labels, system_ids):
      wavs:       (N, L) float32
      labels:     (N,)   int32
      system_ids: list[str]
    """
    N = len(records)
    L = 48000   # 3 seconds × 16kHz
    wavs       = np.empty((N, L), dtype=np.float32)
    labels     = np.empty(N, dtype=np.int32)
    system_ids = []

    print(f"  Loading {N} waveforms into RAM ...")
    for i, rec in enumerate(records):
        t = torch.load(PROCESSED_DIR / rec["audio_path"])   # (L,)
        if t.shape[0] != L:
            # Pad or trim
            if t.shape[0] < L:
                t = torch.nn.functional.pad(t, (0, L - t.shape[0]))
            else:
                t = t[:L]
        wavs[i]   = t.numpy()
        labels[i] = 0 if rec["label"] == "bonafide" else 1
        system_ids.append(rec.get("attack_system", "bonafide"))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{N}")

    print(f"  Waveform cache: {wavs.nbytes / 1e6:.0f} MB")
    return wavs, labels, system_ids


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    import gat_l0_attention as gla
    gla.patch_phoneme_loader()
    lit = gla.load_model(CKPT_PATH, device)
    lit.eval()
    return lit


# ─── Run one condition ────────────────────────────────────────────────────────

def run_condition(gat_model, wavs_np: np.ndarray, device: torch.device,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run WavLM + GAT + classifier on pre-transformed waveforms.
    Returns (logits, phon_vars, frame_dists) each (N,).
    wavs_np: (N, L) float32 — already transformed.
    """
    from gat_l0_attention import run_frozen_frontend

    N = wavs_np.shape[0]
    all_logits     = np.empty(N, dtype=np.float32)
    all_phon_vars  = np.empty(N, dtype=np.float32)
    all_frame_dists = np.empty(N, dtype=np.float32)

    num_f_const = torch.full((BATCH_SIZE,), NF_PER_SAMPLE, device=device)

    for start in range(0, N, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, N)
        B     = end - start
        batch = torch.from_numpy(wavs_np[start:end]).to(device)   # (B, L)

        with torch.no_grad():
            hs, pids = run_frozen_frontend(
                batch.unsqueeze(1), gat_model, device)            # (B, T, 768), (B, T)
            num_f = num_f_const[:B]
            result = gat_model.encoder_and_GAT(hs, num_f, pids)
            logits = result[5]                                     # (B,)

        hs_np   = hs.cpu().numpy()
        pids_np = pids.cpu().numpy()
        all_logits[start:end] = logits.cpu().numpy()

        for i in range(B):
            all_phon_vars[start + i]   = phoneme_segment_var(hs_np[i], pids_np[i])
            all_frame_dists[start + i] = frame_mean_dist(hs_np[i])

    return all_logits, all_phon_vars, all_frame_dists


# ─── Outputs ──────────────────────────────────────────────────────────────────

def write_table_md(path: Path, rows: list[dict]):
    header = ("| Condition | Group | EER | AUC | Δ EER | phoneme_var | frame_dist |")
    sep    = ("|-----------|-------|:---:|:---:|:-----:|:-----------:|:----------:|")
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['group']} | {r['overall_eer']:.4f} | "
            f"{r['overall_auc']:.4f} | {r['delta_eer']:+.4f} | "
            f"{r['phoneme_var']:.5f} | {r['frame_dist']:.4f} |"
        )
    path.write_text("\n".join(lines))


def write_conclusion(path: Path, wav_rows: list[dict], emb_rows: list[dict]):
    wav_base = next(r for r in wav_rows if r["group"] == "baseline")
    wav_smooth = [r for r in wav_rows if r["group"] == "smooth"]
    wav_jitter = [r for r in wav_rows if r["group"] == "jitter"]

    emb_smooth = [r for r in emb_rows if r["group"] == "smooth"]

    wav_smooth_delta = np.mean([r["delta_eer"] for r in wav_smooth])
    wav_jitter_delta = np.mean([r["delta_eer"] for r in wav_jitter])

    emb_smooth_delta = np.mean([r["delta_eer"] for r in emb_smooth])

    # Correlation between waveform-level phoneme_var and EER
    wav_phon_var = np.array([r["phoneme_var"] for r in wav_rows])
    wav_eer      = np.array([r["overall_eer"]  for r in wav_rows])
    rho, p_rho   = spearmanr(wav_phon_var, wav_eer)

    # Compare directional consistency
    smooth_consistent = wav_smooth_delta > 0
    # Jitter (noise only, not shuffle) comparison
    wav_noise_rows = [r for r in wav_jitter if "noise" in r["name"]]
    noise_delta    = np.mean([r["delta_eer"] for r in wav_noise_rows])

    effect_ratio = wav_smooth_delta / emb_smooth_delta if abs(emb_smooth_delta) > 0.001 else float("nan")

    lines = [
        "# End-to-End Causal Assessment",
        "",
        "## Results Summary",
        "",
        f"Waveform-level baseline EER: {wav_base['overall_eer']:.4f}",
        "",
        f"Waveform smoothing (mean Δ EER = {wav_smooth_delta:+.4f}):",
        f"  Expected from hypothesis: positive (smoothing → worse detection)",
        f"  Observed: {'positive — CONSISTENT' if smooth_consistent else 'negative — INCONSISTENT'}",
        "",
        f"Waveform noise jitter (mean Δ EER = {noise_delta:+.4f}):",
        f"  Expected from hypothesis: zero or negative",
        "",
        f"Embedding-level smoothing (reference Δ EER = {emb_smooth_delta:+.4f})",
        f"Waveform/embedding effect ratio: {effect_ratio:.2f}"
        if not np.isnan(effect_ratio) else "",
        "",
        f"Phoneme_var vs EER (waveform level): ρ={rho:+.3f}  p={p_rho:.3f}",
        "",
        "## Conclusion",
        "",
    ]

    if smooth_consistent and abs(wav_smooth_delta) > 0.01:
        if abs(effect_ratio) > 0.3 if not np.isnan(effect_ratio) else False:
            lines.append(
                "Waveform-level smoothing produces a directionally consistent and "
                f"substantive EER increase (mean Δ={wav_smooth_delta:+.4f} vs baseline), "
                "confirming that the causal mechanism operates end-to-end through the "
                "WavLM encoder. The acoustic temporal structure of the audio — not merely "
                "the embedding-space representation — is the key determinant of detection "
                "difficulty. Systems that produce temporally smooth audio (low phoneme "
                "segment variance) evade detection because WavLM faithfully encodes "
                "that smoothness into its representations. This is a strong result: "
                "the correlation (ρ=−0.48 observed across MLAAD systems) reflects a "
                "genuine end-to-end causal mechanism."
            )
        else:
            lines.append(
                f"Waveform-level smoothing produces a directionally consistent EER increase "
                f"(mean Δ={wav_smooth_delta:+.4f}), but the magnitude is substantially "
                f"attenuated relative to embedding-level smoothing (ratio={effect_ratio:.2f}). "
                f"This suggests partial mediation: WavLM partially normalizes mild waveform "
                f"smoothing through its powerful encoder, so the effect is weaker but not "
                f"absent. The causal mechanism survives end-to-end but is partially buffered "
                f"by WavLM's encoding invariances."
            )
    else:
        lines.append(
            f"Waveform-level smoothing does not consistently degrade detection "
            f"(mean Δ={wav_smooth_delta:+.4f}). The previous embedding-level finding "
            f"(Δ={emb_smooth_delta:+.4f}) does not replicate at the waveform level, "
            f"implying that the detector exploits properties of the WavLM representation "
            f"space that are not directly tied to the acoustic temporal structure of the raw "
            f"audio. Waveform smoothing at 5–20ms scales is substantially attenuated by "
            f"WavLM's 25ms convolutional front-end before it reaches the phoneme graph, "
            f"making it an ineffective intervention at these scales."
        )

    path.write_text("\n".join(lines))


# ─── Plots ────────────────────────────────────────────────────────────────────

GROUP_COLORS  = {"smooth": "#2166ac", "baseline": "#000000", "jitter": "#d6604d"}
GROUP_MARKERS = {"smooth": "o", "baseline": "D", "jitter": "s"}


def plot_comparison(wav_rows: list[dict], emb_rows: list[dict], out_path: Path):
    """
    Two-panel plot:
    Left: waveform-level phoneme_var vs EER
    Right: overlay of waveform-level (solid) and embedding-level (dashed) smooth conditions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: waveform-level scatter ──────────────────────────────────────
    ax = axes[0]
    for group in ["smooth", "baseline", "jitter"]:
        grp = [r for r in wav_rows if r["group"] == group]
        xs  = [r["phoneme_var"] for r in grp]
        ys  = [r["overall_eer"] for r in grp]
        ax.scatter(xs, ys, c=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                   s=80, label=f"wav·{group}", zorder=3, alpha=0.9)
        for r in grp:
            ax.annotate(r["name"], (r["phoneme_var"], r["overall_eer"]),
                        fontsize=6.5, ha="left", va="bottom", alpha=0.7)
    ax.set_xlabel("Phoneme embedding variance (post-WavLM)", fontsize=11)
    ax.set_ylabel("EER", fontsize=11)
    ax.set_title("Waveform-level intervention\n(WavLM re-encoded)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel B: overlay comparison (smooth conditions only) ─────────────────
    ax = axes[1]
    # Waveform smooth
    wav_s = sorted([r for r in wav_rows if r["group"] == "smooth"],
                   key=lambda r: r["phoneme_var"])
    xs_w  = [r["phoneme_var"] for r in wav_s]
    ys_w  = [r["overall_eer"] for r in wav_s]
    ax.plot(xs_w, ys_w, "-o", color="#2166ac", label="waveform-level smooth",
            linewidth=2, markersize=7)

    # Embedding smooth (from previous experiment)
    emb_s = sorted([r for r in emb_rows if r["group"] == "smooth"],
                   key=lambda r: r["phoneme_var"])
    xs_e  = [r["phoneme_var"] for r in emb_s]
    ys_e  = [r["overall_eer"] for r in emb_s]
    ax.plot(xs_e, ys_e, "--o", color="#b2182b", label="embedding-level smooth",
            linewidth=2, markersize=7)

    # Baselines
    wav_b = next(r for r in wav_rows if r["group"] == "baseline")
    emb_b = next(r for r in emb_rows if r["group"] == "baseline")
    ax.scatter([wav_b["phoneme_var"]], [wav_b["overall_eer"]], marker="D",
               color="#000000", s=100, zorder=5, label="wav-baseline")
    ax.scatter([emb_b["phoneme_var"]], [emb_b["overall_eer"]], marker="D",
               color="#888888", s=100, zorder=5, label="emb-baseline")

    ax.set_xlabel("Phoneme embedding variance (post-WavLM)", fontsize=11)
    ax.set_ylabel("EER", fontsize=11)
    ax.set_title("Waveform vs Embedding smoothing\n(causal comparison)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Waveform-Level Causal Intervention — Does Effect Survive Re-Encoding?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_hard_easy(wav_rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cond_names = [r["name"] for r in wav_rows]
    x_pos      = range(len(wav_rows))
    colors     = [GROUP_COLORS[r["group"]] for r in wav_rows]

    for ax, (eer_key, title) in zip(axes, [
        ("overall_eer", "All systems"),
        ("hard_eer",    "Hard systems"),
        ("easy_eer",    "Easy systems"),
    ]):
        eers = [r[eer_key] for r in wav_rows]
        ax.bar(x_pos, eers, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("EER" if eer_key == "overall_eer" else "")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        base_eer = next(r[eer_key] for r in wav_rows if r["group"] == "baseline")
        ax.axhline(base_eer, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    from matplotlib.patches import Patch
    axes[-1].legend(
        handles=[Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()],
        loc="upper right", fontsize=9)
    plt.suptitle("Waveform-Level EER by System Group", fontsize=12, fontweight="bold")
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
    hard_set, easy_set = set(hard), set(easy)

    # ── Load test records ─────────────────────────────────────────────────────
    records = json.loads(INDIST_JSON.read_text())
    print(f"Total test records: {len(records)}")

    # ── Cache raw waveforms ───────────────────────────────────────────────────
    print("\nCaching waveforms ...")
    wavs, labels, sys_ids = cache_waveforms(records)
    N = len(labels)

    hard_mask = np.array([s in hard_set for s in sys_ids])
    easy_mask = np.array([s in easy_set for s in sys_ids])
    spoof_mask = (labels == 1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {CKPT_PATH.name} ...")
    lit = load_model(device)
    gat_model = lit.model

    # ── Run each waveform condition ───────────────────────────────────────────
    results = []
    baseline_eer = None

    print(f"\nRunning {len(CONDITIONS)} waveform conditions ...")
    for cond_name, group, kind, params in CONDITIONS:
        print(f"\n  [{group}] {cond_name}")

        # Apply waveform transform (CPU numpy)
        wavs_t = apply_waveform_transform(wavs, kind, params, rng)

        # Run full WavLM + GAT pipeline
        logits, phon_vars, frame_dists = run_condition(gat_model, wavs_t, device)

        # Variance metrics (spoof samples only, for comparability)
        mean_phon_var   = float(phon_vars[spoof_mask].mean())
        mean_frame_dist = float(frame_dists[spoof_mask].mean())
        print(f"    phoneme_var={mean_phon_var:.5f}  frame_dist={mean_frame_dist:.4f}")

        # EERs
        overall_eer = compute_eer(labels, logits)
        hard_eer    = compute_eer(labels[hard_mask | ~spoof_mask],
                                   logits[hard_mask | ~spoof_mask])
        easy_eer    = compute_eer(labels[easy_mask | ~spoof_mask],
                                   logits[easy_mask | ~spoof_mask])
        overall_auc = compute_auc(labels, logits)

        if baseline_eer is None and group == "baseline":
            baseline_eer = overall_eer

        delta_eer = overall_eer - (baseline_eer or overall_eer)
        print(f"    EER: overall={overall_eer:.4f}  hard={hard_eer:.4f}  "
              f"easy={easy_eer:.4f}  AUC={overall_auc:.4f}  Δ={delta_eer:+.4f}")

        results.append({
            "name":        cond_name,
            "group":       group,
            "kind":        kind,
            "params":      str(params),
            "overall_eer": overall_eer,
            "hard_eer":    hard_eer,
            "easy_eer":    easy_eer,
            "overall_auc": overall_auc,
            "delta_eer":   delta_eer,
            "phoneme_var": mean_phon_var,
            "frame_dist":  mean_frame_dist,
        })

    # Backfill delta_eer (baseline runs after some smooth conditions)
    bline = next(r for r in results if r["group"] == "baseline")
    for r in results:
        r["delta_eer"] = r["overall_eer"] - bline["overall_eer"]

    # ── Load embedding-level results for comparison ───────────────────────────
    emb_rows = []
    if EMBED_CSV.exists():
        emb_rows = list(csv.DictReader(EMBED_CSV.open()))
        for r in emb_rows:
            for k in ("overall_eer", "hard_eer", "easy_eer", "phoneme_var", "frame_dist"):
                if k in r:
                    r[k] = float(r[k])
        emb_base = next((x for x in emb_rows if x["group"] == "baseline"), None)
        for r in emb_rows:
            r["delta_eer"] = r["overall_eer"] - (float(emb_base["overall_eer"]) if emb_base else 0.0)
        print(f"\n  Loaded {len(emb_rows)} embedding-level rows for overlay")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting outputs ...")

    # CSV
    csv_path = OUT_DIR / "waveform_intervention.csv"
    with csv_path.open("w", newline="") as f:
        fields = ["name", "group", "kind", "params", "overall_eer", "hard_eer",
                  "easy_eer", "overall_auc", "delta_eer", "phoneme_var", "frame_dist"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print("  Wrote waveform_intervention.csv")

    # Markdown table
    write_table_md(OUT_DIR / "waveform_intervention_table.md", results)
    print("  Wrote waveform_intervention_table.md")

    # Conclusion
    write_conclusion(OUT_DIR / "end_to_end_conclusion.md", results, emb_rows)
    print("  Wrote end_to_end_conclusion.md")

    # Plots
    plot_comparison(results, emb_rows, OUT_DIR / "waveform_vs_eer.png")
    plot_hard_easy(results, OUT_DIR / "hard_vs_easy_waveform.png")

    print(f"\nAll outputs → {OUT_DIR}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WAVEFORM INTERVENTION — SUMMARY")
    print("=" * 72)
    print(f"\n{'Condition':<22}  {'Group':<10}  {'EER':<8}  {'AUC':<8}  "
          f"{'Δ EER':<8}  {'phon_var':<12}  frame_dist")
    print("-" * 90)
    for r in results:
        mk = "  ← BASELINE" if r["group"] == "baseline" else ""
        print(f"{r['name']:<22}  {r['group']:<10}  {r['overall_eer']:.4f}    "
              f"{r['overall_auc']:.4f}    {r['delta_eer']:+.4f}    "
              f"{r['phoneme_var']:.5f}       {r['frame_dist']:.4f}{mk}")

    smooth_delta = np.mean([r["delta_eer"] for r in results if r["group"] == "smooth"])
    jitter_delta = np.mean([r["delta_eer"] for r in results if r["group"] == "jitter"])
    emb_smooth   = np.mean([r["overall_eer"] for r in emb_rows if r["group"] == "smooth"]) if emb_rows else float("nan")
    emb_base_eer = next((float(r["overall_eer"]) for r in emb_rows if r["group"] == "baseline"), float("nan"))

    print(f"\nMean Δ EER  smooth: {smooth_delta:+.4f}   jitter: {jitter_delta:+.4f}")
    if emb_rows:
        print(f"Embedding-level smooth mean EER: {emb_smooth:.4f}  "
              f"(Δ from emb-baseline: {emb_smooth - emb_base_eer:+.4f})")
    verdict = "END-TO-END CAUSAL" if smooth_delta > 0.01 else "REPRESENTATION-LEVEL ARTIFACT"
    print(f"\nMechanism: {verdict}")


if __name__ == "__main__":
    main()
