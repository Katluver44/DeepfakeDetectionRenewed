#!/usr/bin/env python3
"""
train_mlaad_smoothing_aug.py
==============================
Wraps train_mlaad_adversarial.py to add temporal-smoothing augmentation
applied to WavLM hidden_states before the trainable encoder+GAT layers.

Motivation: The causal intervention (mlaad_temporal_intervention.py +
mlaad_waveform_intervention.py) established that temporal embedding smoothness
is a genuine end-to-end causal mechanism of evasion. This script tests whether
training the detector under smoothed-embedding conditions improves its
robustness to smooth TTS systems.

Hook design:
  - Inserted at the INPUT to encoder_and_GAT (after frozen WavLM + SpecAugment)
  - Active only during training (model.training=True)
  - Applies per-sample: randomly one of {moving-average, causal EMA}
  - Applied symmetrically across bonafide and fake (identical p_smooth)

Seeds: 42, 123, 1024 (3 best from seed-locked run)
Saves:  experiments/checkpoints/mlaad_robust_smoothaug_seed{N}.ckpt

Pre-registered criteria (printed at start):
  Strong:   ΔEER(hard) ≤ −0.05  AND |ΔEER(overall)| ≤ 0.02  AND ΔEER(crosslang) ≤ 0
  Partial:  ΔEER(hard) ≤ −0.02  AND ΔEER(overall) ≤ +0.02
  Tradeoff: ΔEER(hard) ≤ −0.05  BUT ΔEER(overall) > +0.02
  Failed:   ΔEER(hard) > −0.02  OR  ΔEER(overall) > +0.05

Outputs → experiments/results/mlaad/smoothing_augmentation/
  per_seed_metrics.json
  paired_deltas.csv
  extreme_systems_breakdown.csv
  cross_language_comparison.json
  criterion_verdict.json
  aug_validation.json
  run_config.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR      = PROJECT_ROOT / "experiments"
CKPT_DIR     = EXP_DIR / "checkpoints"
PROC_DIR     = EXP_DIR / "data" / "mlaad_tiny_processed"
RESULTS_DIR  = EXP_DIR / "results" / "mlaad"
OUT_DIR      = RESULTS_DIR / "smoothing_augmentation"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(EXP_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# modules.py uses a hardcoded relative path "pretrained/..." — must run from project root
os.chdir(PROJECT_ROOT)

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import torch.serialization
from argparse import Namespace
try:
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination
    from pandas import Series
    torch.serialization.add_safe_globals([Namespace, Phonemer_Tokenizer_Recombination, Series])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ─── Import baseline training infrastructure ──────────────────────────────────
from train_mlaad_adversarial import (
    HP, DEVIATIONS, MAALDSplitDataset, AUG_FNS,
    set_seed,
)
from phoneme_GAT.modules import Phoneme_GAT_lit
import gat_l0_attention as gla

# ─── Constants ────────────────────────────────────────────────────────────────
TRAIN_SEEDS   = [42, 123, 1024]
P_SMOOTH      = 0.5
MA_KERNELS    = [3, 5, 7, 9, 11]
EMA_ALPHA_LO  = 0.2
EMA_ALPHA_HI  = 0.7
N_BOOT        = 2000
BOOT_SEED     = 42
NF_PER_SAMPLE = 149

TEST_IN_DIST_JSON  = RESULTS_DIR / "baseline_eval" / "test_in_distribution.json"
TEST_CROSSLANG_JSON = RESULTS_DIR / "baseline_eval" / "test_cross_language.json"

HARD_SYSTEMS = [
    "FireRedTTS-2.0", "Index-TTS-1.5", "Spark-TTS-0.5B",
    "VoxCPM-1.5", "Higgs-Audio-V2", "ZipVoice", "OuteTTS", "griffin_lim",
]
EASY_SYSTEMS = [
    "orpheus-tts-0.1-finetune", "Kitten-TTS-Nano-0.1", "Veena", "Supertonic",
    "kokoro", "Kitten-TTS-Nano-0.2", "Ringg Squirrel TTS v1.0",
]

# Pre-registered criteria (thresholds)
STRONG_HARD_EER_DELTA   = -0.05
STRONG_OVERALL_WINDOW   =  0.02
PARTIAL_HARD_EER_DELTA  = -0.02
PARTIAL_OVERALL_MAX     =  0.02
FAILED_OVERALL_MAX      =  0.05


def _best_ckpt_for(stem: str) -> Path | None:
    cands = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    if cands:
        return cands[0]
    p = CKPT_DIR / f"{stem}.ckpt"
    return p if p.exists() else None


# ─── Temporal smoothing functions (GPU-compatible) ────────────────────────────

def _smooth_ma(hs: torch.Tensor, k: int) -> torch.Tensor:
    """hs: (T, D). Centered moving average, reflect-padded."""
    T, D = hs.shape
    pad  = k // 2
    x    = hs.T.unsqueeze(0)                            # (1, D, T)
    x    = F_nn.pad(x, (pad, pad), mode="reflect")
    x    = F_nn.avg_pool1d(x, kernel_size=k, stride=1)  # (1, D, T)
    if x.shape[-1] > T:
        x = x[..., :T]
    return x.squeeze(0).T                               # (T, D)


def _smooth_ema(hs: torch.Tensor, alpha: float) -> torch.Tensor:
    """hs: (T, D). Causal exponential moving average."""
    T, D = hs.shape
    result = hs.clone()
    one_minus = 1.0 - alpha
    for t in range(1, T):
        result[t] = alpha * hs[t] + one_minus * result[t - 1]
    return result


# ─── Augmentation tracker (sanity checks) ─────────────────────────────────────

class AugTracker:
    """Tracks smoothing application statistics for sanity checks."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_presented       = 0
        self.n_applied         = 0
        self.n_presented_bona  = 0
        self.n_applied_bona    = 0
        self.n_presented_fake  = 0
        self.n_applied_fake    = 0
        # Norm stats (first batch only)
        self.pre_norm_mean  = None
        self.post_norm_mean = None
        self.norms_captured = False

    def log_batch(self, labels, applied_mask, pre_norms, post_norms):
        B = len(labels)
        self.n_presented += B
        self.n_applied   += int(applied_mask.sum())
        for i in range(B):
            is_bona = int(labels[i]) == 0
            applied = bool(applied_mask[i])
            if is_bona:
                self.n_presented_bona += 1
                if applied:
                    self.n_applied_bona += 1
            else:
                self.n_presented_fake += 1
                if applied:
                    self.n_applied_fake += 1

        if not self.norms_captured and pre_norms is not None:
            self.pre_norm_mean  = float(pre_norms.mean())
            self.post_norm_mean = float(post_norms.mean())
            self.norms_captured = True

    def summary(self, p_smooth: float) -> dict:
        def rate(n, d):
            return float(n / d) if d > 0 else float("nan")

        bona_rate = rate(self.n_applied_bona, self.n_presented_bona)
        fake_rate = rate(self.n_applied_fake, self.n_presented_fake)
        overall   = rate(self.n_applied, self.n_presented)
        tol = 0.05
        symm_ok = (abs(bona_rate - p_smooth) < tol and abs(fake_rate - p_smooth) < tol)
        norm_ok = (self.pre_norm_mean is not None and
                   self.post_norm_mean is not None and
                   abs(self.pre_norm_mean - self.post_norm_mean) > 1e-4)
        return {
            "p_smooth_target":         p_smooth,
            "overall_application_rate": overall,
            "bonafide_application_rate": bona_rate,
            "fake_application_rate":    fake_rate,
            "symmetry_within_5pct":     symm_ok,
            "pre_norm_mean":            self.pre_norm_mean,
            "post_norm_mean":           self.post_norm_mean,
            "norms_differ":             norm_ok,
            "sanity_a_pass":            norm_ok,
            "sanity_b_pass":            symm_ok,
        }


# ─── Smoothing hook ───────────────────────────────────────────────────────────

def install_smoothing_hook(gat_model, p_smooth: float,
                           tracker: AugTracker, seed: int) -> callable:
    """
    Monkey-patches gat_model.encoder_and_GAT to apply temporal smoothing
    to hidden_states (only during training, only when use_encoder=True).

    Returns a restore callable that removes the hook.
    """
    rng = random.Random(seed)
    original = gat_model.encoder_and_GAT.__func__

    def _smoothed(self_inner, hidden_states, num_frames, phoneme_ids,
                  profiler=None, use_encoder=True, ground_truth_labels=None):

        if self_inner.training and use_encoder:
            B = hidden_states.shape[0]
            hs = hidden_states.clone()  # out-of-place; don't modify caller's tensor

            applied_mask = torch.zeros(B, dtype=torch.bool)
            pre_norms    = torch.zeros(B)
            post_norms   = torch.zeros(B)

            for i in range(B):
                pre_norms[i] = hs[i].norm()
                if rng.random() < p_smooth:
                    applied_mask[i] = True
                    if rng.random() < 0.5:
                        k = rng.choice(MA_KERNELS)
                        hs[i] = _smooth_ma(hs[i], k)
                    else:
                        alpha = rng.uniform(EMA_ALPHA_LO, EMA_ALPHA_HI)
                        hs[i] = _smooth_ema(hs[i], alpha)
                    post_norms[i] = hs[i].norm()
                else:
                    post_norms[i] = pre_norms[i]

            labels_list = (ground_truth_labels.cpu().tolist()
                           if ground_truth_labels is not None
                           else [None] * B)
            tracker.log_batch(labels_list, applied_mask, pre_norms, post_norms)
            hidden_states = hs

        return original(self_inner, hidden_states, num_frames, phoneme_ids,
                        profiler=profiler, use_encoder=use_encoder,
                        ground_truth_labels=ground_truth_labels)

    gat_model.encoder_and_GAT = types.MethodType(_smoothed, gat_model)

    def restore():
        gat_model.encoder_and_GAT = types.MethodType(original, gat_model)

    return restore


# ─── Sanity check (d): WavLM feature_extractor is frozen ──────────────────────

def check_wavlm_frozen(gat_model) -> bool:
    """Confirm CNN feature_extractor parameters have requires_grad=False."""
    for p in gat_model.transformer_in_phoneme_model.feature_extractor.parameters():
        if p.requires_grad:
            return False
    for p in gat_model.transformer_in_phoneme_model.feature_projection.parameters():
        if p.requires_grad:
            return False
    return True


# ─── Training one seed ────────────────────────────────────────────────────────

def train_one_seed(seed: int, device: torch.device,
                   fast_dev_run: bool = False) -> dict:
    """Train smoothaug model for one seed. Returns final_metrics dict."""
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger as PLCSVLogger
    from callbacks import EER_Callback
    from callbacks_rational import (
        BinaryACC_Callback, BinaryAUC_Callback,
        TPR_Callback, TNR_Callback, FPR_Callback, FNR_Callback,
    )

    set_seed(seed)

    ckpt_stem   = f"mlaad_robust_smoothaug_seed{seed}"
    ckpt_path   = CKPT_DIR / f"{ckpt_stem}.ckpt"
    log_dir     = RESULTS_DIR / "training_logs" / ckpt_stem
    log_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    splits_dir = PROC_DIR / "splits"

    train_ds = MAALDSplitDataset(
        splits_dir / "train.json", PROC_DIR,
        mode="train", balance=True, seed=seed,
        aug_prob=HP["aug_prob"], aug_fns=AUG_FNS, aug_weights=HP["aug_weights"],
    )
    val_ds = MAALDSplitDataset(
        splits_dir / "val.json", PROC_DIR,
        mode="eval", balance=True, seed=seed,
    )

    train_dl = DataLoader(
        train_ds, batch_size=HP["batch_size"], shuffle=True,
        num_workers=HP["num_workers"], pin_memory=True,
        drop_last=HP["drop_last"], persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=HP["num_workers"], pin_memory=True,
        drop_last=False, persistent_workers=True,
    )

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone=HP["backbone"], use_raw=False,
        use_GAT=HP["use_GAT"], n_edges=HP["n_edges"],
        use_aug=HP["use_aug"], use_pool=HP["use_pool"], use_clip=HP["use_clip"],
    ))
    lit = Phoneme_GAT_lit(cfg=cfg)

    # ── Sanity (d): confirm WavLM CNN is frozen BEFORE installing hook ─────────
    frozen_ok = check_wavlm_frozen(lit.model)
    print(f"  [sanity-d] WavLM feature_extractor frozen: {frozen_ok}")

    # ── Install smoothing hook ────────────────────────────────────────────────
    tracker = AugTracker()
    restore_hook = install_smoothing_hook(lit.model, P_SMOOTH, tracker, seed)
    print(f"  [hook] Temporal smoothing installed (p_smooth={P_SMOOTH})")

    metric_cbs = [
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
        TPR_Callback(batch_key="label", output_key="logit"),
        TNR_Callback(batch_key="label", output_key="logit"),
        FPR_Callback(batch_key="label", output_key="logit"),
        FNR_Callback(batch_key="label", output_key="logit"),
    ]

    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename=f"{ckpt_stem}-best-{{epoch:02d}}-{{val-eer:.4f}}",
        monitor="val-eer", mode="min", save_last=True, verbose=True,
    )

    trainer = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=HP["max_epochs"],
        logger=PLCSVLogger(save_dir=str(log_dir), name="", version="",
                           flush_logs_every_n_steps=10),
        callbacks=[*metric_cbs, ckpt_cb],
        log_every_n_steps=10,
        deterministic=False,
        fast_dev_run=fast_dev_run,
    )

    print(f"\n  Training seed={seed} for {HP['max_epochs']} epochs ...")
    t0 = time.time()
    trainer.fit(lit, train_dl, val_dl)
    elapsed = time.time() - t0

    # Save last checkpoint
    trainer.save_checkpoint(str(ckpt_path))

    # Extract best-checkpoint info
    best_path = ckpt_cb.best_model_path or ""
    best_eer  = float("nan")
    best_epoch = -1
    if best_path:
        m = re.search(r"val-eer=([0-9]+\.[0-9]+)", best_path)
        if m:
            best_eer = float(m.group(1))
        m2 = re.search(r"epoch=(\d+)", best_path)
        if m2:
            best_epoch = int(m2.group(1))

    # Aug sanity summary
    aug_summary = tracker.summary(P_SMOOTH)

    final = {
        "seed":           seed,
        "model":          ckpt_stem,
        "best_val_eer":   best_eer,
        "best_epoch":     best_epoch,
        "best_ckpt":      Path(best_path).name if best_path else "",
        "elapsed_s":      elapsed,
        "wavlm_frozen":   frozen_ok,
        "aug_validation": aug_summary,
    }

    restore_hook()  # remove hook (eval will load clean checkpoint)
    print(f"  Seed {seed} done: val-eer={best_eer:.4f}  elapsed={elapsed/60:.1f}min")
    return final


# ─── Inference helpers ────────────────────────────────────────────────────────

class _EvalDS(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        w = torch.load(PROC_DIR / r["audio_path"])  # (48000,)
        return {
            "audio":         w.unsqueeze(0),         # (1, 48000)
            "label":         torch.tensor(0 if r["label"] == "bonafide" else 1),
            "system_id":     r.get("attack_system", "bonafide"),
            "sample_rate":   torch.tensor(16000),
        }


def _collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def _load_for_eval(ckpt_path: Path, device: torch.device):
    lit = gla.load_model(ckpt_path, device)
    lit.eval()
    return lit


def _infer(lit, records, device, batch_size=32):
    """Run inference; returns (labels np, scores np, system_ids list)."""
    from gat_l0_attention import run_frozen_frontend
    ds = _EvalDS(records)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=2, collate_fn=_collate, pin_memory=True)
    gat_model = lit.model
    all_labels, all_scores, all_sids = [], [], []
    num_f = torch.full((batch_size,), NF_PER_SAMPLE, device=device)

    with torch.no_grad():
        for batch in dl:
            audio = batch["audio"].to(device)
            B = audio.shape[0]
            hs, pids = run_frozen_frontend(audio, gat_model, device)
            result = gat_model.encoder_and_GAT(hs, num_f[:B], pids)
            logits = result[5].cpu().numpy()
            all_labels.extend(batch["label"].tolist())
            all_scores.extend(logits.tolist())
            all_sids.extend(batch["system_id"])

    return np.array(all_labels), np.array(all_scores), all_sids


# ─── EER + bootstrap CI ──────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_eer_ci(labels: np.ndarray, scores: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Returns (lo_95, hi_95) bootstrap CI on EER."""
    rng = np.random.default_rng(seed)
    n   = len(labels)
    boot_eers = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e   = compute_eer(labels[idx], scores[idx])
        if not np.isnan(e):
            boot_eers.append(e)
    if not boot_eers:
        return float("nan"), float("nan")
    lo = float(np.percentile(boot_eers, 2.5))
    hi = float(np.percentile(boot_eers, 97.5))
    return lo, hi


def _eer_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    eer = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    lo, hi = bootstrap_eer_ci(labels, scores)
    return {"eer": eer, "auc": auc, "eer_ci_lo": lo, "eer_ci_hi": hi,
            "n_samples": int(len(labels)),
            "n_spoof": int((labels == 1).sum()),
            "n_bonafide": int((labels == 0).sum())}


# ─── Evaluation for one checkpoint ──────────────────────────────────────────

def evaluate_ckpt(ckpt_path: Path, device: torch.device,
                  test_records, crosslang_records,
                  hard_set: set, easy_set: set) -> dict:
    """Full evaluation suite for one checkpoint."""
    lit = _load_for_eval(ckpt_path, device)

    # ── In-distribution ───────────────────────────────────────────────────────
    labels, scores, sids = _infer(lit, test_records, device)
    spoof_mask = (labels == 1)
    bona_mask  = (labels == 0)

    overall = _eer_metrics(labels, scores)

    # Hard / easy pooled (spoof vs ALL bonafide)
    hard_mask  = np.array([s in hard_set for s in sids]) & spoof_mask
    easy_mask  = np.array([s in easy_set for s in sids]) & spoof_mask
    hard_labels  = np.concatenate([labels[hard_mask],  labels[bona_mask]])
    hard_scores  = np.concatenate([scores[hard_mask],  scores[bona_mask]])
    easy_labels  = np.concatenate([labels[easy_mask],  labels[bona_mask]])
    easy_scores  = np.concatenate([scores[easy_mask],  scores[bona_mask]])

    hard_metrics = _eer_metrics(hard_labels, hard_scores)
    easy_metrics = _eer_metrics(easy_labels, easy_scores)

    # Per-system EER
    per_sys = {}
    unique_sys = sorted(set(s for s in sids if s not in ("bonafide", None)))
    for sys_id in unique_sys:
        sys_mask = np.array([s == sys_id for s in sids]) & spoof_mask
        if sys_mask.sum() < 3:
            continue
        s_lab = np.concatenate([labels[sys_mask], labels[bona_mask]])
        s_scr = np.concatenate([scores[sys_mask], scores[bona_mask]])
        per_sys[sys_id] = compute_eer(s_lab, s_scr)

    # ── Cross-language ────────────────────────────────────────────────────────
    cl_lab, cl_scr, _ = _infer(lit, crosslang_records, device)
    crosslang = _eer_metrics(cl_lab, cl_scr)

    return {
        "overall":   overall,
        "hard":      hard_metrics,
        "easy":      easy_metrics,
        "per_sys":   per_sys,
        "crosslang": crosslang,
    }


# ─── Criterion evaluation ──────────────────────────────────────────────────────

def evaluate_criterion(mean_hard_delta: float, mean_overall_delta: float,
                        mean_crosslang_delta: float) -> str:
    if mean_hard_delta <= STRONG_HARD_EER_DELTA:
        if abs(mean_overall_delta) <= STRONG_OVERALL_WINDOW and mean_crosslang_delta <= 0:
            return "strong"
        elif mean_overall_delta > STRONG_OVERALL_WINDOW:
            return "tradeoff"
    if mean_hard_delta <= PARTIAL_HARD_EER_DELTA and mean_overall_delta <= PARTIAL_OVERALL_MAX:
        return "partial"
    if mean_overall_delta > FAILED_OVERALL_MAX:
        return "failed"
    return "failed"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Run 1 batch per epoch (debug)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training, go straight to evaluation")
    p.add_argument("--eval-only-seed", type=int, default=None,
                   help="Evaluate only this seed (skip training)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # ── Print pre-registered criteria ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRE-REGISTERED EVALUATION CRITERIA")
    print("=" * 70)
    print(f"  Strong:   ΔEER(hard) ≤ {STRONG_HARD_EER_DELTA:+.2f}  "
          f"AND |ΔEER(overall)| ≤ {STRONG_OVERALL_WINDOW:.2f}  AND ΔEER(crosslang) ≤ 0")
    print(f"  Partial:  ΔEER(hard) ≤ {PARTIAL_HARD_EER_DELTA:+.2f}  "
          f"AND ΔEER(overall) ≤ {PARTIAL_OVERALL_MAX:+.2f}")
    print(f"  Tradeoff: ΔEER(hard) ≤ {STRONG_HARD_EER_DELTA:+.2f}  BUT ΔEER(overall) > {STRONG_OVERALL_WINDOW:+.2f}")
    print(f"  Failed:   ΔEER(hard) > {PARTIAL_HARD_EER_DELTA:+.2f}  OR ΔEER(overall) > {FAILED_OVERALL_MAX:+.2f}")
    print("  (all deltas: smoothaug − baseline, mean across 3 seeds)")
    print("=" * 70 + "\n")

    # ── Load test data ────────────────────────────────────────────────────────
    test_records      = json.loads(TEST_IN_DIST_JSON.read_text())
    crosslang_records = json.loads(TEST_CROSSLANG_JSON.read_text())
    hard_set = set(HARD_SYSTEMS)
    easy_set = set(EASY_SYSTEMS)
    print(f"Test in-dist: {len(test_records)} records")
    print(f"Test crosslang: {len(crosslang_records)} records")
    print(f"Hard systems: {len(hard_set)}  Easy systems: {len(easy_set)}\n")

    # ── Determine seeds to process ────────────────────────────────────────────
    seeds = [args.eval_only_seed] if args.eval_only_seed else TRAIN_SEEDS

    # ── Training ──────────────────────────────────────────────────────────────
    training_results = {}
    aug_validation   = {}

    if not args.skip_training:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"TRAINING seed={seed}")
            print(f"{'='*60}")

            # Check if best checkpoint already exists
            existing = _best_ckpt_for(f"mlaad_robust_smoothaug_seed{seed}")
            if existing and existing.exists() and "-best-" in existing.name:
                print(f"  [SKIP] Best checkpoint already exists: {existing.name}")
                training_results[seed] = {"seed": seed, "skipped": True, "best_ckpt": existing.name}
                continue

            tr = train_one_seed(seed, device, fast_dev_run=args.fast_dev_run)
            training_results[seed] = tr
            aug_validation[seed]   = tr.get("aug_validation", {})
    else:
        print("[skip-training mode] Loading existing smoothaug checkpoints ...\n")
        for seed in seeds:
            existing = _best_ckpt_for(f"mlaad_robust_smoothaug_seed{seed}")
            if existing and existing.exists():
                print(f"  seed={seed}: {existing.name}")
            else:
                print(f"  seed={seed}: NOT FOUND — training required")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    per_seed_metrics = {}

    for seed in seeds:
        print(f"\n  Evaluating seed={seed} ...")

        # Baseline
        baseline_ckpt = _best_ckpt_for(f"mlaad_robust_goat_seed{seed}")
        if baseline_ckpt is None or not baseline_ckpt.exists():
            print(f"    [WARN] baseline not found for seed={seed}, skipping")
            continue

        # Smoothaug
        smoothaug_ckpt = _best_ckpt_for(f"mlaad_robust_smoothaug_seed{seed}")
        if smoothaug_ckpt is None or not smoothaug_ckpt.exists():
            print(f"    [WARN] smoothaug not found for seed={seed}, skipping")
            continue

        print(f"    Baseline:  {baseline_ckpt.name}")
        print(f"    Smoothaug: {smoothaug_ckpt.name}")

        base_metrics  = evaluate_ckpt(baseline_ckpt,  device, test_records,
                                       crosslang_records, hard_set, easy_set)
        saug_metrics  = evaluate_ckpt(smoothaug_ckpt, device, test_records,
                                       crosslang_records, hard_set, easy_set)

        def _delta(a, b): return float(b - a) if (not np.isnan(a) and not np.isnan(b)) else float("nan")

        per_seed_metrics[seed] = {
            "seed":              seed,
            "baseline_ckpt":     baseline_ckpt.name,
            "smoothaug_ckpt":    smoothaug_ckpt.name,
            "baseline":          base_metrics,
            "smoothaug":         saug_metrics,
            "delta_overall_eer": _delta(base_metrics["overall"]["eer"],
                                         saug_metrics["overall"]["eer"]),
            "delta_hard_eer":    _delta(base_metrics["hard"]["eer"],
                                         saug_metrics["hard"]["eer"]),
            "delta_easy_eer":    _delta(base_metrics["easy"]["eer"],
                                         saug_metrics["easy"]["eer"]),
            "delta_crosslang_eer": _delta(base_metrics["crosslang"]["eer"],
                                           saug_metrics["crosslang"]["eer"]),
        }

        print(f"    Overall:    base={base_metrics['overall']['eer']:.4f}  "
              f"aug={saug_metrics['overall']['eer']:.4f}  "
              f"Δ={per_seed_metrics[seed]['delta_overall_eer']:+.4f}")
        print(f"    Hard:       base={base_metrics['hard']['eer']:.4f}  "
              f"aug={saug_metrics['hard']['eer']:.4f}  "
              f"Δ={per_seed_metrics[seed]['delta_hard_eer']:+.4f}")
        print(f"    Easy:       base={base_metrics['easy']['eer']:.4f}  "
              f"aug={saug_metrics['easy']['eer']:.4f}  "
              f"Δ={per_seed_metrics[seed]['delta_easy_eer']:+.4f}")
        print(f"    CrossLang:  base={base_metrics['crosslang']['eer']:.4f}  "
              f"aug={saug_metrics['crosslang']['eer']:.4f}  "
              f"Δ={per_seed_metrics[seed]['delta_crosslang_eer']:+.4f}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    if per_seed_metrics:
        all_d_overall   = [v["delta_overall_eer"]   for v in per_seed_metrics.values()]
        all_d_hard      = [v["delta_hard_eer"]       for v in per_seed_metrics.values()]
        all_d_easy      = [v["delta_easy_eer"]       for v in per_seed_metrics.values()]
        all_d_crosslang = [v["delta_crosslang_eer"]  for v in per_seed_metrics.values()]

        def _nanmean(lst):
            valid = [x for x in lst if not np.isnan(x)]
            return float(np.mean(valid)) if valid else float("nan")

        mean_d_overall   = _nanmean(all_d_overall)
        mean_d_hard      = _nanmean(all_d_hard)
        mean_d_easy      = _nanmean(all_d_easy)
        mean_d_crosslang = _nanmean(all_d_crosslang)

        verdict = evaluate_criterion(mean_d_hard, mean_d_overall, mean_d_crosslang)

        # Per-system breakdown across seeds
        all_systems = sorted(hard_set | easy_set)
        sys_rows = []
        for sys_id in all_systems:
            group = "hard" if sys_id in hard_set else "easy"
            for seed, data in per_seed_metrics.items():
                base_eer = data["baseline"]["per_sys"].get(sys_id, float("nan"))
                saug_eer = data["smoothaug"]["per_sys"].get(sys_id, float("nan"))
                delta    = saug_eer - base_eer if not (np.isnan(base_eer) or np.isnan(saug_eer)) else float("nan")
                sys_rows.append({
                    "system": sys_id, "group": group, "seed": seed,
                    "baseline_eer": base_eer, "smoothaug_eer": saug_eer,
                    "delta_eer": delta,
                })

        # ── Write outputs ─────────────────────────────────────────────────────
        print(f"\n\nWriting outputs → {OUT_DIR}")

        # per_seed_metrics.json
        (OUT_DIR / "per_seed_metrics.json").write_text(
            json.dumps({str(k): v for k, v in per_seed_metrics.items()}, indent=2,
                       default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))

        # paired_deltas.csv
        delta_rows = []
        for seed, data in per_seed_metrics.items():
            delta_rows.append({
                "seed": seed,
                "delta_overall": data["delta_overall_eer"],
                "delta_hard":    data["delta_hard_eer"],
                "delta_easy":    data["delta_easy_eer"],
                "delta_crosslang": data["delta_crosslang_eer"],
                "baseline_overall": data["baseline"]["overall"]["eer"],
                "smoothaug_overall": data["smoothaug"]["overall"]["eer"],
                "baseline_hard": data["baseline"]["hard"]["eer"],
                "smoothaug_hard": data["smoothaug"]["hard"]["eer"],
                "baseline_crosslang": data["baseline"]["crosslang"]["eer"],
                "smoothaug_crosslang": data["smoothaug"]["crosslang"]["eer"],
            })
        with (OUT_DIR / "paired_deltas.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            w.writeheader(); w.writerows(delta_rows)

        # extreme_systems_breakdown.csv
        with (OUT_DIR / "extreme_systems_breakdown.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "system", "group", "seed", "baseline_eer", "smoothaug_eer", "delta_eer"])
            w.writeheader(); w.writerows(sys_rows)

        # cross_language_comparison.json
        cl_data = {}
        for seed, data in per_seed_metrics.items():
            cl_data[str(seed)] = {
                "baseline_eer":  data["baseline"]["crosslang"]["eer"],
                "smoothaug_eer": data["smoothaug"]["crosslang"]["eer"],
                "delta_eer":     data["delta_crosslang_eer"],
            }
        (OUT_DIR / "cross_language_comparison.json").write_text(json.dumps(cl_data, indent=2))

        # criterion_verdict.json
        verdict_data = {
            "verdict":              verdict,
            "criteria": {
                "strong":   f"ΔEER(hard)≤{STRONG_HARD_EER_DELTA:+.2f} AND |ΔEER(overall)|≤{STRONG_OVERALL_WINDOW:.2f} AND ΔEER(crosslang)≤0",
                "partial":  f"ΔEER(hard)≤{PARTIAL_HARD_EER_DELTA:+.2f} AND ΔEER(overall)≤{PARTIAL_OVERALL_MAX:+.2f}",
                "tradeoff": f"ΔEER(hard)≤{STRONG_HARD_EER_DELTA:+.2f} BUT ΔEER(overall)>{STRONG_OVERALL_WINDOW:+.2f}",
                "failed":   f"ΔEER(hard)>{PARTIAL_HARD_EER_DELTA:+.2f} OR ΔEER(overall)>{FAILED_OVERALL_MAX:+.2f}",
            },
            "mean_delta_hard_eer":      mean_d_hard,
            "mean_delta_overall_eer":   mean_d_overall,
            "mean_delta_easy_eer":      mean_d_easy,
            "mean_delta_crosslang_eer": mean_d_crosslang,
            "per_seed_delta_hard":      {str(k): v["delta_hard_eer"] for k, v in per_seed_metrics.items()},
            "per_seed_delta_overall":   {str(k): v["delta_overall_eer"] for k, v in per_seed_metrics.items()},
        }
        (OUT_DIR / "criterion_verdict.json").write_text(json.dumps(verdict_data, indent=2))

        # aug_validation.json
        (OUT_DIR / "aug_validation.json").write_text(
            json.dumps({str(k): v for k, v in aug_validation.items()}, indent=2))

        # run_config.json
        run_config = {
            "script":       "train_mlaad_smoothing_aug.py",
            "seeds":        TRAIN_SEEDS,
            "p_smooth":     P_SMOOTH,
            "ma_kernels":   MA_KERNELS,
            "ema_alpha_lo": EMA_ALPHA_LO,
            "ema_alpha_hi": EMA_ALPHA_HI,
            "n_boot":       N_BOOT,
            "hp":           HP,
            "hard_systems": HARD_SYSTEMS,
            "easy_systems": EASY_SYSTEMS,
            "elapsed_total_s": time.time() - t_start,
        }
        (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))

        # ── Console summary ───────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("SMOOTHING AUGMENTATION — FINAL SUMMARY")
        print("=" * 70)
        print(f"\nMean Δ EER (smoothaug − baseline), n={len(per_seed_metrics)} seeds:")
        print(f"  Hard systems:  {mean_d_hard:+.4f}  (positive = aug HURTS hard, negative = HELPS)")
        print(f"  Easy systems:  {mean_d_easy:+.4f}")
        print(f"  Overall:       {mean_d_overall:+.4f}")
        print(f"  Cross-lang:    {mean_d_crosslang:+.4f}")

        print(f"\nPer-seed hard-system ΔEER:")
        for seed, data in per_seed_metrics.items():
            print(f"  Seed {seed}: Δhard={data['delta_hard_eer']:+.4f}  "
                  f"Δoverall={data['delta_overall_eer']:+.4f}  "
                  f"Δcrosslang={data['delta_crosslang_eer']:+.4f}")

        print(f"\nPer-system EER changes (extreme systems):")
        print(f"  {'System':<35}  {'Group':<8}  {'ΔEER (mean across seeds)':>25}")
        for sys_id in all_systems:
            rows_for = [r for r in sys_rows if r["system"] == sys_id]
            deltas = [r["delta_eer"] for r in rows_for if not np.isnan(r["delta_eer"])]
            mean_d = float(np.mean(deltas)) if deltas else float("nan")
            group  = "hard" if sys_id in hard_set else "easy"
            print(f"  {sys_id:<35}  {group:<8}  {mean_d:+.4f}")

        print(f"\n{'='*70}")
        print(f"VERDICT: {verdict.upper()}")
        print(f"{'='*70}")

        print(f"\nResults → {OUT_DIR}")
        print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
