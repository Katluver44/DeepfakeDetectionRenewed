#!/usr/bin/env python3
"""
p4_smoothing_aug_finetune.py
=============================
P4 ablation: Fine-tune robust_goat with temporal-smoothing augmentation
restricted to kernels [3,5,7] at p_smooth=0.3.

Rationale: E3 shows smoothing (MA window≥3) directly reduces C by averaging
frames toward a local mean, simulating compact-embedding behavior of hard systems.
Training on smoothed bonafide speech forces the model to distinguish compact-but-real
from compact-and-synthetic.

Key differences from the existing train_mlaad_smoothing_aug.py:
  - Kernels: [3, 5, 7] only (NOT 9, 11 — E3 shows window≥11 REVERSES the correlation)
  - p_smooth: 0.3 (vs 0.5 in the original)
  - Starting weights: robust_goat.ckpt (not training from scratch)
  - Epochs: 5 (fine-tune only)
  - Tracks val-C distribution each epoch to confirm broader training distribution

Pre-registered criteria:
  Strong:   ΔEER(hard-C) ≤ −0.02 AND |ΔEER(overall)| ≤ 0.03
  Partial:  ΔEER(hard-C) ≤ −0.01 AND ΔEER(overall) ≤ +0.02
  Failed:   ΔEER(hard-C) > 0  OR ΔEER(overall) > +0.05

Outputs:
    experiments/checkpoints/mlaad_ct_smooth_seed{seed}-best-*.ckpt
    experiments/results/mlaad/p4_smoothing_aug/
        per_system_eer_seed{seed}.csv
        c_distribution_epochs_seed{seed}.json
        summary.md
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
EXP_DIR      = PROJECT_ROOT / "experiments"
CKPT_DIR     = EXP_DIR / "checkpoints"
PROC_DIR     = EXP_DIR / "data" / "mlaad_tiny_processed"
OUT_DIR      = EXP_DIR / "results" / "mlaad" / "p4_smoothing_aug"
BASE_CKPT    = PROJECT_ROOT / "experiments" / "checkpoints" / "mlaad_robust_goat.ckpt"
TEST_JSON    = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(PROJECT_ROOT)

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from argparse import Namespace
try:
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination
    from pandas import Series
    torch.serialization.add_safe_globals([Namespace, Phonemer_Tokenizer_Recombination, Series])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

from train_mlaad_adversarial import (
    HP, MAALDSplitDataset, AUG_FNS, set_seed,
)
from phoneme_GAT.modules import Phoneme_GAT_lit
from loader import TARGET_SR
from callbacks import EER_Callback
from callbacks_rational import BinaryACC_Callback, BinaryAUC_Callback

# ─── Config ──────────────────────────────────────────────────────────────────
SEEDS         = [42, 123, 1024]
P_SMOOTH      = 0.3         # lower than original 0.5 (conservative)
MA_KERNELS    = [3, 5, 7]   # window≥11 reverses the C-axis correlation (E3 finding)
EMA_ALPHA_LO  = 0.3
EMA_ALPHA_HI  = 0.7
MAX_EPOCHS    = 5
BATCH_SIZE    = HP["batch_size"]   # 20


# ─── Smoothing functions ──────────────────────────────────────────────────────

def _smooth_ma(hs: torch.Tensor, k: int) -> torch.Tensor:
    """Centered moving average. hs: (T, D)."""
    T, D = hs.shape
    pad  = k // 2
    x    = hs.T.unsqueeze(0)
    x    = F_nn.pad(x, (pad, pad), mode="reflect")
    x    = F_nn.avg_pool1d(x, kernel_size=k, stride=1)
    if x.shape[-1] > T:
        x = x[..., :T]
    return x.squeeze(0).T


def _smooth_ema(hs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal exponential moving average. hs: (T, D)."""
    T, D = hs.shape
    result = hs.clone()
    one_minus = 1.0 - alpha
    for t in range(1, T):
        result[t] = alpha * hs[t] + one_minus * result[t - 1]
    return result


# ─── Smoothing hook ───────────────────────────────────────────────────────────

def install_smoothing_hook(gat_model, p_smooth: float, seed: int):
    """
    Monkey-patches gat_model.encoder_and_GAT to apply temporal smoothing
    before the trainable encoder (active during training only).

    Restricted to MA_KERNELS=[3,5,7] (E3 finding: window≥11 reverses C signal).
    Returns a restore callable.
    """
    rng = random.Random(seed)
    original = gat_model.encoder_and_GAT.__func__

    def _smoothed(self_inner, hidden_states, num_frames, phoneme_ids,
                  profiler=None, use_encoder=True, ground_truth_labels=None):
        if self_inner.training and use_encoder:
            B, T, D = hidden_states.shape
            aug_list = []
            for i in range(B):
                if rng.random() < p_smooth:
                    hs_i = hidden_states[i]
                    choice = rng.randint(0, len(MA_KERNELS))  # 0..len: 0=EMA, else MA
                    if choice == len(MA_KERNELS):
                        alpha = rng.uniform(EMA_ALPHA_LO, EMA_ALPHA_HI)
                        aug_list.append(_smooth_ema(hs_i, alpha))
                    else:
                        k = MA_KERNELS[choice]
                        aug_list.append(_smooth_ma(hs_i, k))
                else:
                    aug_list.append(hidden_states[i])
            hidden_states = torch.stack(aug_list, dim=0)

        return original(self_inner, hidden_states, num_frames, phoneme_ids,
                        profiler=profiler, use_encoder=use_encoder,
                        ground_truth_labels=ground_truth_labels)

    gat_model.encoder_and_GAT = types.MethodType(_smoothed, gat_model)

    def restore():
        gat_model.encoder_and_GAT = types.MethodType(original, gat_model)

    return restore


# ─── C-distribution tracker (per epoch on validation) ────────────────────────

def extract_val_c_stats(model, device) -> dict:
    """
    Run one forward pass through the validation set, extract rog@L12 per utterance.
    Returns dict with mean/std/q25/q75 of the C distribution.
    """
    from phoneme_GAT.modules_ct import _rog_batch

    splits_dir = PROC_DIR / "splits"
    val_ds = MAALDSplitDataset(
        splits_dir / "val.json", PROC_DIR, mode="eval", balance=True, seed=42,
    )
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    wavlm = model.model.transformer_in_phoneme_model
    rogs = []
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            audio = batch["audio"].to(device)  # (B, 1, 48000)
            audio_1d = audio[:, 0, :]
            wout = wavlm(input_values=audio_1d, output_hidden_states=True)
            frames_L12 = wout.hidden_states[12]  # (B, T, 768)
            rog_vals = _rog_batch(frames_L12).cpu().numpy()
            rogs.extend(rog_vals.tolist())

    rogs = np.array(rogs)
    return {
        "mean": float(rogs.mean()),
        "std":  float(rogs.std()),
        "q25":  float(np.percentile(rogs, 25)),
        "q75":  float(np.percentile(rogs, 75)),
        "min":  float(rogs.min()),
        "max":  float(rogs.max()),
    }


# ─── EER computation ─────────────────────────────────────────────────────────

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from collections import defaultdict
import pandas as pd


def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return float(eer)
    except Exception:
        return None


@torch.no_grad()
def evaluate_on_test(model, device, out_csv=None):
    class TestDS(torch.utils.data.Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            r = self.records[i]
            wav = torch.load(PROC_DIR / r["audio_path"]).unsqueeze(0)
            return {"audio": wav, "label": 0 if r["label"]=="bonafide" else 1,
                    "system": r["attack_system"]}

    records = json.loads(TEST_JSON.read_text())
    dl = DataLoader(TestDS(records), batch_size=16, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda b: {
                        "audio": torch.stack([x["audio"] for x in b]),
                        "label": [x["label"] for x in b],
                        "system": [x["system"] for x in b],
                    }, pin_memory=True)

    NF = 48000 // 320 - 1
    model.eval()
    all_labels, all_logits, all_systems = [], [], []
    for batch in dl:
        audio = batch["audio"].to(device)
        nf = torch.full((audio.shape[0],), NF, device=device)
        out = model.model(audio, nf, use_aug=False, stage="val")
        all_logits.extend(out["logit"].cpu().tolist())
        all_labels.extend(batch["label"])
        all_systems.extend(batch["system"])

    bf_labels, bf_logits = [], []
    sp_groups = defaultdict(lambda: ([], []))
    for y, s, sys in zip(all_labels, all_logits, all_systems):
        if y == 0:
            bf_labels.append(y); bf_logits.append(s)
        else:
            sp_groups[sys][0].append(y); sp_groups[sys][1].append(s)

    results = {}
    for sys, (sp_lab, sp_log) in sorted(sp_groups.items()):
        if len(sp_lab) < 5:
            results[sys] = None; continue
        results[sys] = compute_eer(
            np.array(sp_lab + bf_labels),
            np.array(sp_log + bf_logits)
        )

    if out_csv is not None:
        rows = [{"system": s, "eer": v} for s, v in results.items() if v is not None]
        pd.DataFrame(rows).sort_values("eer", ascending=False).to_csv(out_csv, index=False)
    return results


# ─── Per-seed training ────────────────────────────────────────────────────────

def train_seed(seed: int, device: torch.device) -> dict:
    set_seed(seed)
    ckpt_stem = f"mlaad_ct_smooth_seed{seed}"
    ckpt_path = CKPT_DIR / f"{ckpt_stem}.ckpt"
    log_dir   = OUT_DIR / f"logs_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P4 Smoothing Augmentation — seed {seed}")
    print(f"  p_smooth={P_SMOOTH}, kernels={MA_KERNELS}")
    print(f"  Base checkpoint: {BASE_CKPT}")
    print(f"{'='*60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    splits_dir = PROC_DIR / "splits"
    train_ds = MAALDSplitDataset(
        splits_dir / "train.json", PROC_DIR, mode="train",
        balance=True, seed=seed, aug_prob=HP["aug_prob"],
        aug_fns=AUG_FNS, aug_weights=HP["aug_weights"],
    )
    val_ds = MAALDSplitDataset(
        splits_dir / "val.json", PROC_DIR, mode="eval", balance=True, seed=seed,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=HP["num_workers"], pin_memory=True,
                          drop_last=HP["drop_last"], persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=HP["num_workers"], pin_memory=True,
                          drop_last=False, persistent_workers=True)

    # ── Model: fine-tune from robust_goat checkpoint ───────────────────────────
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone=HP["backbone"], use_raw=False,
        use_GAT=HP["use_GAT"], n_edges=HP["n_edges"],
        use_aug=HP["use_aug"], use_pool=HP["use_pool"], use_clip=HP["use_clip"],
    ))
    model = Phoneme_GAT_lit.load_from_checkpoint(str(BASE_CKPT), map_location=device)
    model.train()

    # Install smoothing hook (before trainable encoder)
    restore_fn = install_smoothing_hook(model.model, P_SMOOTH, seed=seed)
    print(f"  Smoothing hook installed: kernels={MA_KERNELS}, p_smooth={P_SMOOTH}")

    # ── Track C distribution per epoch ───────────────────────────────────────
    c_epoch_stats = {}
    # Baseline (before any training)
    model.eval()
    c_epoch_stats["epoch_0_baseline"] = extract_val_c_stats(model, device)
    print(f"  Baseline C stats (validation): {c_epoch_stats['epoch_0_baseline']}")
    model.train()

    # ── Trainer with per-epoch C tracking ─────────────────────────────────────
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning.loggers import CSVLogger as PLCSVLogger

    class CDistributionTracker(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            ep = trainer.current_epoch
            stats = extract_val_c_stats(pl_module, device)
            c_epoch_stats[f"epoch_{ep+1}"] = stats
            print(f"  [C tracker] Epoch {ep+1}: "
                  f"mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    callbacks = [
        EER_Callback(batch_key="label", output_key="logit"),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        CDistributionTracker(),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename=ckpt_stem + "-best-{epoch:02d}-{val-eer:.4f}",
            monitor="val-eer", mode="min", save_last=True, verbose=True,
        ),
    ]
    logger = PLCSVLogger(save_dir=str(log_dir), name="", version="")

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=MAX_EPOCHS,
        logger=logger, callbacks=callbacks,
        log_every_n_steps=10, deterministic=False,
    )

    start = time.time()
    trainer.fit(model, train_dl, val_dl)
    print(f"  Training time: {time.time()-start:.1f}s")

    # Remove hook after training
    restore_fn()
    trainer.save_checkpoint(str(ckpt_path))

    # Save C epoch stats
    (log_dir / f"c_distribution_epochs_seed{seed}.json").write_text(
        json.dumps(c_epoch_stats, indent=2)
    )

    # ── Evaluate on test split ─────────────────────────────────────────────────
    model.eval().to(device)
    eer_dict = evaluate_on_test(
        model, device,
        out_csv=OUT_DIR / f"per_system_eer_seed{seed}.csv"
    )
    valid = [v for v in eer_dict.values() if v is not None]
    print(f"  Test mean EER: {np.mean(valid):.4f} ({len(valid)} systems)")
    return eer_dict


# ─── Summary ─────────────────────────────────────────────────────────────────

def write_summary(all_seed_eers: dict, baseline_csv=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = sorted(set(s for d in all_seed_eers.values() for s in d))
    mean_eer = {}
    for sys in systems:
        vals = [d[sys] for d in all_seed_eers.values() if d.get(sys) is not None]
        mean_eer[sys] = float(np.mean(vals)) if vals else None

    ct_eers = [v for v in mean_eer.values() if v is not None]
    hard_thresh = np.percentile(ct_eers, 75)
    hard_eers   = [v for v in ct_eers if v >= hard_thresh]

    baseline = {}
    if baseline_csv and Path(baseline_csv).exists():
        bdf = pd.read_csv(baseline_csv)
        baseline = dict(zip(bdf["system"], bdf.get("eer", bdf.get("eer_before", []))))

    lines = [
        "# P4: Smoothing Augmentation Fine-tuning",
        "",
        f"**p_smooth={P_SMOOTH}  |  kernels={MA_KERNELS}  |  seeds={sorted(all_seed_eers.keys())}  |  epochs={MAX_EPOCHS}**",
        "",
        "## Motivation",
        "E3 shows temporal smoothing (MA window≥3) directly reduces C (rog) by averaging frames.",
        "Training on smoothed bonafide speech forces the model to distinguish compact-but-real from",
        "compact-and-synthetic. Kernels ≥11 EXCLUDED because E3 shows they reverse the C correlation.",
        "",
        "## Results (mean across seeds)",
        "",
        f"| Metric | Smooth-aug model |",
        f"|--------|-----------------|",
        f"| Mean EER (all systems) | {np.mean(ct_eers):.4f} |",
        f"| Median EER | {np.median(ct_eers):.4f} |",
        f"| Hard-C-quartile EER (≥75th pct) | {np.mean(hard_eers):.4f} |",
        "",
        "## Per-system delta vs baseline",
        "",
        "| System | Smooth EER | Baseline EER | ΔEER |",
        "|--------|-----------|-------------|------|",
    ]
    for s, v in sorted(mean_eer.items(), key=lambda x: -(x[1] or 0)):
        if v is None:
            continue
        bl = baseline.get(s)
        delta = f"{v - bl:+.4f}" if bl else "N/A"
        lines.append(f"| {s[:40]:40s} | {v:.4f} | {bl:.4f if bl else 'N/A':>9} | {delta} |")

    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"Summary written to {OUT_DIR / 'summary.md'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--p-smooth", type=float, default=P_SMOOTH)
    p.add_argument("--baseline-csv", type=Path, default=None)
    args = p.parse_args()

    global MAX_EPOCHS, P_SMOOTH
    MAX_EPOCHS = args.epochs
    P_SMOOTH   = args.p_smooth

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}")

    all_seed_eers = {}
    for seed in args.seeds:
        all_seed_eers[seed] = train_seed(seed, device)

    write_summary(all_seed_eers, baseline_csv=args.baseline_csv)
    all_eers = [v for d in all_seed_eers.values() for v in d.values() if v is not None]
    print(f"\nP4 done. Mean test EER: {np.mean(all_eers):.4f}")


if __name__ == "__main__":
    main()
