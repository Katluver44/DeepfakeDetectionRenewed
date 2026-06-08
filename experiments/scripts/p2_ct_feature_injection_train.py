#!/usr/bin/env python3
"""
p2_ct_feature_injection_train.py
=================================
P2 ablation: Fine-tune robust_goat with C and T features injected at the
classification head. Uses Phoneme_GAT_CT from phoneme_GAT/modules_ct.py.

Changes from baseline (train_mlaad_adversarial.py):
  - Model: Phoneme_GAT_CT with cls_head input 768+2=770 (C and T appended)
  - Starting weights: loaded from models/robust_goat.ckpt (base checkpoint)
  - Epochs: 5 (fine-tune only; fewer than training from scratch)
  - LR: 5e-5 (encoder) / 5e-5 (head) — lower than baseline 1e-4 to preserve features

Pre-registered criteria:
  Improved:   mean EER on hard-C quartile decreases by ≥0.02 absolute vs baseline
  Neutral:    hard-C ΔEER in (−0.02, +0.02)
  Degraded:   hard-C ΔEER > +0.02  (C/T injection hurts)

Outputs:
    experiments/checkpoints/mlaad_ct_feat_seed{seed}-best-*.ckpt
    experiments/results/mlaad/ct_feature_injection/
        per_system_eer_seed{seed}.csv
        training_curves_seed{seed}.csv
        summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F_audio
from fractions import Fraction
from torch.utils.data import DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
EXP_DIR      = PROJECT_ROOT / "experiments"
CKPT_DIR     = EXP_DIR / "checkpoints"
PROC_DIR     = EXP_DIR / "data" / "mlaad_tiny_processed"
OUT_DIR      = EXP_DIR / "results" / "mlaad" / "ct_feature_injection"
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
    HP, DEVIATIONS, MAALDSplitDataset, AUG_FNS, set_seed,
)
from phoneme_GAT.modules_ct import Phoneme_GAT_CT_lit
from loader import TARGET_SR
from callbacks import EER_Callback
from callbacks_rational import (
    BinaryACC_Callback, BinaryAUC_Callback,
    TPR_Callback, TNR_Callback, FPR_Callback, FNR_Callback,
)

# ─── Config ──────────────────────────────────────────────────────────────────
SEEDS      = [42, 123, 1024]
N_CT       = 2          # C (rog@L12) and T (vel_entropy@L9)
MAX_EPOCHS = 5          # fine-tuning (vs 7 for training from scratch)
LR_ENCODER = 5e-5       # lower than baseline 5e-5 (encoder already tuned)
LR_HEAD    = 5e-5       # new dimensions start at zero; small LR is safe
BATCH_SIZE = HP["batch_size"]  # 20

# ─── Evaluation ───────────────────────────────────────────────────────────────
# Multi-metric eval + C/T-quartile stratification (EER cross-checked vs AUC/acc).
from _ablation_common import (
    evaluate_on_test as eval_test_multimetric,
    load_baseline_metrics, load_system_ct, stratify_by_ct, stratified_markdown,
    consistency_note, METRIC_KEYS,
)


# Per-system EER/AUC/accuracy on the locked test split is provided by
# _ablation_common.evaluate_on_test (imported above as eval_test_multimetric);
# the previous EER-only inline copy was removed to avoid divergence.


# ─── Per-seed training ────────────────────────────────────────────────────────

def train_seed(seed: int, device: torch.device) -> dict:
    """Fine-tune CT model for one seed. Returns per-system EER dict."""
    set_seed(seed)
    ckpt_stem  = (f"mlaad_ct_feat_seed{seed}" if N_CT == 2
                  else f"mlaad_ct{N_CT}_feat_seed{seed}")
    ckpt_path  = CKPT_DIR / f"{ckpt_stem}.ckpt"
    log_dir    = OUT_DIR / f"logs_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P2 CT Feature Injection — seed {seed}")
    print(f"  Base checkpoint: {BASE_CKPT}")
    print(f"  Output:          {ckpt_path}")
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

    # ── Model: load base weights into CT variant ───────────────────────────────
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone=HP["backbone"], use_raw=False,
        use_GAT=HP["use_GAT"], n_edges=HP["n_edges"],
        use_aug=HP["use_aug"], use_pool=HP["use_pool"], use_clip=HP["use_clip"],
    ))
    model = Phoneme_GAT_CT_lit.load_from_base_checkpoint(
        str(BASE_CKPT), cfg=cfg, n_ct=N_CT
    )

    # Override LR for CT model (lower for stable fine-tuning)
    model.lr = LR_HEAD

    # ── Trainer ───────────────────────────────────────────────────────────────
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger as PLCSVLogger

    callbacks = [
        EER_Callback(batch_key="label", output_key="logit"),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename=ckpt_stem + "-best-{epoch:02d}-{val-eer:.4f}",
            monitor="val-eer", mode="min", save_last=False, verbose=True,
        ),
    ]
    logger = PLCSVLogger(save_dir=str(log_dir), name="", version="",
                         flush_logs_every_n_steps=10)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=MAX_EPOCHS,
        precision="bf16-mixed",        # A100 fast path; save_last off for concurrency
        logger=logger, callbacks=callbacks,
        log_every_n_steps=10, deterministic=False,
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    start = time.time()
    trainer.fit(model, train_dl, val_dl)
    print(f"  Training time: {time.time()-start:.1f}s")

    trainer.save_checkpoint(str(ckpt_path))

    # ── Evaluate on test split (EER + AUC + accuracy + balanced accuracy) ──────
    model.eval().to(device)
    metrics = eval_test_multimetric(
        model, device,
        out_csv=OUT_DIR / f"per_system_metrics_seed{seed}.csv"
    )
    valid = [m for m in metrics.values() if m is not None]
    print(f"  Test mean EER: {np.mean([m['eer'] for m in valid]):.4f}  "
          f"AUC: {np.mean([m['auc'] for m in valid]):.4f}  "
          f"bal_acc: {np.mean([m['bal_acc'] for m in valid]):.4f} ({len(valid)} systems)")
    return metrics


# ─── Comparison with baseline ─────────────────────────────────────────────────

def write_summary(all_seed_metrics: dict[int, dict], baseline_csv: Path | None = None):
    """Write summary.md comparing the CT model vs baseline across seeds, with
    EER cross-checked against AUC and accuracy and stratified by C/T quartile."""
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mean of each metric per system across seeds.
    systems = sorted(set(s for d in all_seed_metrics.values() for s in d))
    mean_metrics = {}
    for sysn in systems:
        ms = [d[sysn] for d in all_seed_metrics.values() if d.get(sysn) is not None]
        if not ms:
            mean_metrics[sysn] = None; continue
        mean_metrics[sysn] = {k: float(np.mean([m[k] for m in ms if m.get(k) is not None]))
                              for k in METRIC_KEYS if any(m.get(k) is not None for m in ms)}

    baseline = load_baseline_metrics(baseline_csv) if baseline_csv else load_baseline_metrics()
    ct = load_system_ct()
    report = stratify_by_ct(mean_metrics, baseline, ct)

    lines = [
        "# P2: CT Feature Injection — Training Summary",
        "",
        f"**n_ct={N_CT}  |  seeds={sorted(all_seed_metrics.keys())}  |  epochs={MAX_EPOCHS}**",
        "",
        stratified_markdown(report, "Per-system metrics vs mlaad_robust_goat baseline"),
        "",
        "## Config",
        f"- Base checkpoint: {BASE_CKPT}",
        f"- n_ct: {N_CT} (C=−rog@L12, T=vel_entropy@L9)",
        f"- cls_head input: 768+{N_CT}={768+N_CT}",
        f"- Fine-tune epochs: {MAX_EPOCHS}  |  LR encoder/head: {LR_ENCODER}/{LR_HEAD}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))

    # Machine-readable per-system table with all metrics.
    rows = []
    for s in systems:
        if mean_metrics[s] is None:
            continue
        row = {"system": s, "C": ct.get(s, {}).get("C"), "T": ct.get(s, {}).get("T")}
        for k in METRIC_KEYS:
            row[f"ct_{k}"] = mean_metrics[s].get(k)
            row[f"baseline_{k}"] = baseline.get(s, {}).get(k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_system_metrics_mean.csv", index=False)
    print(f"Summary written to {OUT_DIR / 'summary.md'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n-ct", type=int, default=N_CT,
                   help="Number of CT features: 1=C only, 2=C+T, 3=C+T+T_window11")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--baseline-csv", type=Path, default=None,
                   help="Per-system EER CSV from baseline for comparison")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}  |  n_ct={args.n_ct}  |  epochs={args.epochs}")

    # Update globals for dynamic args. n_ct=2 keeps the legacy P2 dir; n_ct=3 is
    # the P3 (T_window11) variant and writes to its own dir so summaries don't
    # collide.
    global N_CT, MAX_EPOCHS, OUT_DIR
    N_CT        = args.n_ct
    MAX_EPOCHS  = args.epochs
    if N_CT != 2:
        label = {1: "p2b_c_only", 3: "p3_ct_window11"}.get(N_CT, f"ct_n{N_CT}")
        OUT_DIR = EXP_DIR / "results" / "mlaad" / label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_eers = {}
    for seed in args.seeds:
        all_seed_eers[seed] = train_seed(seed, device)

    write_summary(all_seed_eers, baseline_csv=args.baseline_csv)

    # Overall stats (values are per-system metric dicts now)
    all_eers = [m["eer"] for d in all_seed_eers.values() for m in d.values()
                if m is not None and m.get("eer") is not None]
    print(f"\n{'='*60}")
    print(f"P2 done. Mean test EER: {np.mean(all_eers):.4f}")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
