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

# ─── Evaluation imports ───────────────────────────────────────────────────────
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from collections import defaultdict


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return float(eer)
    except Exception:
        return None


def per_system_eer_from_dict(all_labels, all_logits, all_systems, min_n=5):
    bf_labels, bf_logits = [], []
    sp_groups = defaultdict(lambda: ([], []))
    for y, s, sys in zip(all_labels, all_logits, all_systems):
        if y == 0:
            bf_labels.append(y); bf_logits.append(s)
        else:
            sp_groups[sys][0].append(y); sp_groups[sys][1].append(s)
    results = {}
    for sys, (sp_lab, sp_log) in sorted(sp_groups.items()):
        if len(sp_lab) < min_n:
            results[sys] = None; continue
        comb_labels = np.array(sp_lab + bf_labels)
        comb_scores = np.array(sp_log + bf_logits)
        results[sys] = compute_eer(comb_labels, comb_scores)
    return results


# ─── Evaluation loop ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(model, device, out_csv: Path | None = None):
    """Run inference on the full test split, return per-system EER dict."""
    from torch.utils.data import Dataset
    import json

    class TestDS(torch.utils.data.Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            r = self.records[i]
            wav = torch.load(PROC_DIR / r["audio_path"]).unsqueeze(0)
            return {"audio": wav, "label": 0 if r["label"]=="bonafide" else 1,
                    "system": r["attack_system"]}

    def collate(batch):
        return {
            "audio":  torch.stack([b["audio"] for b in batch]),
            "label":  [b["label"] for b in batch],
            "system": [b["system"] for b in batch],
        }

    records = json.loads(TEST_JSON.read_text())
    dl = DataLoader(TestDS(records), batch_size=16, shuffle=False,
                    num_workers=4, collate_fn=collate, pin_memory=True)

    model.eval()
    all_labels, all_logits, all_systems = [], [], []
    NF = 48000 // 320 - 1

    for batch in dl:
        audio = batch["audio"].to(device)
        num_frames = torch.full((audio.shape[0],), NF, device=device)
        out = model.model(audio, num_frames, use_aug=False, stage="val")
        all_logits.extend(out["logit"].cpu().tolist())
        all_labels.extend(batch["label"])
        all_systems.extend(batch["system"])

    sys_eer = per_system_eer_from_dict(all_labels, all_logits, all_systems)

    if out_csv is not None:
        import pandas as pd
        rows = [{"system": s, "eer": v} for s, v in sys_eer.items() if v is not None]
        pd.DataFrame(rows).sort_values("eer", ascending=False).to_csv(out_csv, index=False)

    return sys_eer


# ─── Per-seed training ────────────────────────────────────────────────────────

def train_seed(seed: int, device: torch.device) -> dict:
    """Fine-tune CT model for one seed. Returns per-system EER dict."""
    set_seed(seed)
    ckpt_stem  = f"mlaad_ct_feat_seed{seed}"
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
            monitor="val-eer", mode="min", save_last=True, verbose=True,
        ),
    ]
    logger = PLCSVLogger(save_dir=str(log_dir), name="", version="",
                         flush_logs_every_n_steps=10)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=MAX_EPOCHS,
        logger=logger, callbacks=callbacks,
        log_every_n_steps=10, deterministic=False,
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    start = time.time()
    trainer.fit(model, train_dl, val_dl)
    print(f"  Training time: {time.time()-start:.1f}s")

    trainer.save_checkpoint(str(ckpt_path))

    # ── Evaluate on test split ─────────────────────────────────────────────────
    model.eval().to(device)
    eer_dict = evaluate_on_test(
        model, device,
        out_csv=OUT_DIR / f"per_system_eer_seed{seed}.csv"
    )
    valid = [v for v in eer_dict.values() if v is not None]
    print(f"  Test mean EER: {np.mean(valid):.4f} ({len(valid)} systems)")
    return eer_dict


# ─── Comparison with baseline ─────────────────────────────────────────────────

def write_summary(all_seed_eers: dict[int, dict], baseline_csv: Path | None = None):
    """Write summary.md comparing CT model vs baseline across seeds."""
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Average EER across seeds per system
    systems = sorted(set(s for d in all_seed_eers.values() for s in d))
    mean_eer_ct = {}
    for sys in systems:
        vals = [d[sys] for d in all_seed_eers.values() if d.get(sys) is not None]
        mean_eer_ct[sys] = np.mean(vals) if vals else None

    lines = [
        "# P2: CT Feature Injection — Training Summary",
        "",
        f"**n_ct={N_CT}  |  seeds={sorted(all_seed_eers.keys())}  |  epochs={MAX_EPOCHS}**",
        "",
        "## Per-system EER (mean across seeds)",
        "",
        "| System | CT model EER | Baseline EER | ΔEER |",
        "|--------|-------------|-------------|------|",
    ]

    baseline = {}
    if baseline_csv and Path(baseline_csv).exists():
        bdf = pd.read_csv(baseline_csv)
        baseline = dict(zip(bdf["system"], bdf.get("eer", bdf.get("eer_before", []))))

    valid_ct = [(s, v) for s, v in sorted(mean_eer_ct.items()) if v is not None]
    ct_eers  = [v for _, v in valid_ct]

    for s, ct_e in sorted(valid_ct, key=lambda x: -x[1]):
        bl = baseline.get(s, None)
        delta = f"{ct_e - bl:+.4f}" if bl is not None else "N/A"
        bl_str = f"{bl:.4f}" if bl is not None else "N/A"
        lines.append(f"| {s[:40]:40s} | {ct_e:.4f} | {bl_str} | {delta} |")

    hard_thresh = np.percentile(ct_eers, 75)
    hard_eers   = [v for _, v in valid_ct if v >= hard_thresh]

    lines += [
        "",
        "## Summary Statistics",
        "",
        f"| Metric | CT model |",
        f"|--------|---------|",
        f"| Mean EER (all systems) | {np.mean(ct_eers):.4f} |",
        f"| Median EER | {np.median(ct_eers):.4f} |",
        f"| Hard-quartile EER (≥75th pct) | {np.mean(hard_eers):.4f} |",
        "",
        "## Config",
        f"- Base checkpoint: {BASE_CKPT}",
        f"- n_ct: {N_CT} (C=rog@L12, T=vel_entropy@L9)",
        f"- cls_head input: 768+{N_CT}={768+N_CT}",
        f"- Fine-tune epochs: {MAX_EPOCHS}",
        f"- LR encoder/head: {LR_ENCODER}/{LR_HEAD}",
    ]

    (OUT_DIR / "summary.md").write_text("\n".join(lines))
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}  |  n_ct={args.n_ct}  |  epochs={args.epochs}")

    # Update global for dynamic args
    global N_CT, MAX_EPOCHS
    N_CT        = args.n_ct
    MAX_EPOCHS  = args.epochs

    all_seed_eers = {}
    for seed in args.seeds:
        all_seed_eers[seed] = train_seed(seed, device)

    write_summary(all_seed_eers, baseline_csv=args.baseline_csv)

    # Overall stats
    all_eers = [v for d in all_seed_eers.values() for v in d.values() if v is not None]
    print(f"\n{'='*60}")
    print(f"P2 done. Mean test EER: {np.mean(all_eers):.4f}")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
