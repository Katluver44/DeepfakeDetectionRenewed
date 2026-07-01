#!/usr/bin/env python3
"""
Workstream E — train a deliberately WEAK ASVspoof detector ("mini_goat").

Copied/adapted from experiments/scripts/train_mlaad_regular.py (the WORKING
WavLM-GAT training recipe): identical model (Phoneme_GAT_lit, same cfg),
identical Lightning Trainer / callback stack. The only changes are:
  (i)   dataset: ASVspoof-2019-LA `train` split (attacks A01-A06), ~500
        preprocessed .pt files built by prepare_mini_goat_data.py, instead of
        MLAAD-tiny splits.
  (ii)  deliberately small train set (~500 files, 250 bona + 250 spoof) and
        few epochs -- mini_goat is SUPPOSED to be weak (this is the point of
        Workstream E's headroom hypothesis test), not a from-scratch SOTA
        detector.
  (iii) checkpoint path -> models/mini_goat.ckpt

Usage:
    venv/bin/python experiments/results/e_mini_goat_fusion/train_mini_goat.py

Produces:
    models/mini_goat.ckpt
    experiments/checkpoints/mini_goat*.ckpt (best + last, via ModelCheckpoint)
    experiments/results/e_mini_goat_fusion/training_logs/
        training_curves.csv, final_metrics.json, hyperparameters.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch torch.load for legacy checkpoints (matches train_mlaad_regular.py)
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

import torch.serialization
from argparse import Namespace
from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination
from pandas import Series
torch.serialization.add_safe_globals([Namespace, Phonemer_Tokenizer_Recombination, Series])

from phoneme_GAT.modules import Phoneme_GAT_lit
from loader import TARGET_SR, TARGET_SAMPLES
from callbacks import EER_Callback
from callbacks_rational import (
    BinaryACC_Callback, BinaryAUC_Callback,
    TPR_Callback, TNR_Callback, FPR_Callback, FNR_Callback,
)

HERE = Path(__file__).resolve().parent

# ─── Hyperparameters ─────────────────────────────────────────────────────────
# Same model/optim recipe as train_mlaad_regular.py / robust_goat. Deliberately
# small data + few epochs is the ONLY lever pulled to make mini_goat weak.
HP = {
    "backbone": "wavlm",
    "use_GAT": True,
    "n_edges": 10,
    "use_aug": True,
    "use_pool": True,
    "use_clip": True,
    "batch_size": 10,       # small batch: only 500 train files
    "max_epochs": 4,        # deliberately undertrained
    "reverb_prob": 0.0,     # no augmentation -- keep it weak / simple
    "num_workers": 4,
    "drop_last": True,
    "seed": 42,
}

DEVIATIONS = [
    "Deliberately weak detector: ~500 train files (250 bonafide + 250 spoof) "
    "from ASVspoof-2019-LA `train` split (attacks A01-A06), vs robust_goat's "
    "limit=15000. Only 4 epochs. This is the intended manipulation for "
    "Workstream E's headroom hypothesis test, not a bug.",
    "No reverb augmentation (reverb_prob=0) to keep the weak-detector regime simple.",
    "Center-cropped 48000-sample tensors (same as prepare_mlaad_tiny.py); no "
    "additional random-crop freedom since source tensors are already exactly 3s.",
    "Val set (150 utts) drawn from ASVspoof `validation` split (also A01-A06, "
    "attack-disjoint from the A07-A19 eval set mini_goat is scored on).",
]


class ASVspoofMiniDataset(Dataset):
    """Reads preprocessed .pt tensors from prepare_mini_goat_data.py's splits."""

    def __init__(self, split_json: Path, processed_dir: Path, mode: str = "train"):
        self.processed_dir = processed_dir / "audio"
        self.mode = mode
        self.records = json.loads(split_json.read_text())
        n_bf = sum(1 for r in self.records if r["label"] == "bonafide")
        n_sp = sum(1 for r in self.records if r["label"] == "spoof")
        print(f"  ASVspoofMiniDataset({split_json.name}, {mode}): "
              f"{len(self.records)} samples ({n_bf} bonafide + {n_sp} spoof)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tensor = torch.load(self.processed_dir / rec["audio_path"])  # (48000,) float32
        wav = tensor.unsqueeze(0)  # (1, 48000)
        y = 0 if rec["label"] == "bonafide" else 1
        return {
            "audio": wav,
            "label": torch.tensor(y, dtype=torch.long),
            "sample_rate": TARGET_SR,
            "attack_system": rec.get("attack_system", "unknown"),
        }


class CSVTrainingLogger:
    COLUMNS = ["epoch", "train_loss", "val_loss", "val_eer", "val_acc", "val_auc"]

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "training_curves.csv"
        with open(self.csv_path, "w") as f:
            f.write(",".join(self.COLUMNS) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/data/mini_goat_processed")
    p.add_argument("--checkpoint-path", type=Path,
                    default=PROJECT_ROOT / "models/mini_goat.ckpt")
    p.add_argument("--ckpt-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/checkpoints")
    p.add_argument("--log-dir", type=Path,
                    default=HERE / "training_logs")
    p.add_argument("--seed", type=int, default=HP["seed"])
    p.add_argument("--epochs", type=int, default=HP["max_epochs"])
    p.add_argument("--batch-size", type=int, default=HP["batch_size"])
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    hp_snapshot = dict(HP)
    hp_snapshot.update({
        "seed": args.seed, "max_epochs": args.epochs, "batch_size": args.batch_size,
        "checkpoint_path": str(args.checkpoint_path), "processed_dir": str(args.processed_dir),
        "script": "train_mini_goat.py", "model_variant": "mini_goat",
        "deviations_from_robust_goat": DEVIATIONS,
    })
    (args.log_dir / "hyperparameters.json").write_text(json.dumps(hp_snapshot, indent=2))
    print("Hyperparameters written.")

    print("Loading datasets ...")
    splits_dir = args.processed_dir / "splits"
    train_ds = ASVspoofMiniDataset(splits_dir / "train.json", args.processed_dir, mode="train")
    val_ds = ASVspoofMiniDataset(splits_dir / "val.json", args.processed_dir, mode="eval")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=HP["num_workers"], pin_memory=True,
                           drop_last=HP["drop_last"], persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=HP["num_workers"], pin_memory=True,
                         drop_last=False, persistent_workers=True)

    print(f"Train: {len(train_ds)} samples ({len(train_dl)} batches/epoch)\n"
          f"Val:   {len(val_ds)} samples ({len(val_dl)} batches)\n")

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone=HP["backbone"], use_raw=False, use_GAT=HP["use_GAT"],
        n_edges=HP["n_edges"], use_aug=HP["use_aug"], use_pool=HP["use_pool"],
        use_clip=HP["use_clip"],
    ))
    model = Phoneme_GAT_lit(cfg=cfg)

    from pytorch_lightning.callbacks import ModelCheckpoint
    csv_logger = CSVTrainingLogger(args.log_dir)

    metric_callbacks = [
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
        TPR_Callback(batch_key="label", output_key="logit"),
        TNR_Callback(batch_key="label", output_key="logit"),
        FPR_Callback(batch_key="label", output_key="logit"),
        FNR_Callback(batch_key="label", output_key="logit"),
    ]

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="mini_goat-best-{epoch:02d}-{val-eer:.4f}",
        monitor="val-eer", mode="min", save_last=True, verbose=True,
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger as PLCSVLogger

    pl_csv_logger = PLCSVLogger(save_dir=str(args.log_dir), name="", version="",
                                 flush_logs_every_n_steps=10)

    trainer = Trainer(
        accelerator="gpu", devices=1, max_epochs=args.epochs,
        logger=pl_csv_logger, callbacks=[*metric_callbacks, ckpt_callback],
        log_every_n_steps=5, deterministic=False, fast_dev_run=args.fast_dev_run,
    )

    print(f"\nStarting training: {args.epochs} epochs, batch={args.batch_size}")
    start_time = time.time()
    trainer.fit(model, train_dl, val_dl)
    total_time = time.time() - start_time

    trainer.save_checkpoint(str(args.checkpoint_path))
    print(f"\nSaved checkpoint: {args.checkpoint_path}")

    pl_metrics_path = args.log_dir / "metrics.csv"
    if pl_metrics_path.exists():
        import csv
        with open(pl_metrics_path) as f:
            pl_rows = list(csv.DictReader(f))
        from collections import defaultdict
        epoch_data = defaultdict(dict)
        for row in pl_rows:
            ep = row.get("epoch", "")
            if ep == "" or ep is None:
                continue
            ep = int(float(ep))
            for k, v in row.items():
                if v and v != "" and k != "step":
                    try:
                        epoch_data[ep][k] = float(v)
                    except (ValueError, TypeError):
                        pass
        with open(args.log_dir / "training_curves.csv", "w") as f:
            cols = ["epoch", "train_loss", "val_loss", "val_eer", "val_acc", "val_auc",
                    "val_tpr", "val_tnr", "val_fpr", "val_fnr"]
            f.write(",".join(cols) + "\n")
            for ep in sorted(epoch_data):
                d = epoch_data[ep]
                vals = [ep, d.get("train-loss", ""), d.get("val-loss", ""),
                        d.get("val-eer", ""), d.get("val-acc", ""), d.get("val-auc", ""),
                        d.get("val-tpr", ""), d.get("val-tnr", ""), d.get("val-fpr", ""),
                        d.get("val-fnr", "")]
                f.write(",".join(str(v) for v in vals) + "\n")

    best_val_eer = float("nan")
    best_epoch = -1
    best_checkpoint_name = ""
    try:
        best_model_path = ckpt_callback.best_model_path or ""
        if not best_model_path:
            candidates = sorted(args.ckpt_dir.glob("mini_goat-best-*.ckpt"))
            if candidates:
                best_model_path = str(candidates[0])
        if best_model_path:
            best_checkpoint_name = Path(best_model_path).name
            m = re.search(r"val-eer=([0-9]+(?:\.[0-9]+)?)", best_model_path)
            if m:
                best_val_eer = float(m.group(1))
            m2 = re.search(r"epoch=(\d+)", best_model_path)
            if m2:
                best_epoch = int(m2.group(1))
    except Exception:
        pass

    last_metrics = {}
    try:
        last_metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
    except Exception:
        pass

    final_metrics = {
        "model_variant": "mini_goat",
        "total_training_time_seconds": total_time,
        "total_training_time_hms": time.strftime("%H:%M:%S", time.gmtime(total_time)),
        "actual_epochs": trainer.current_epoch,
        "best_epoch": best_epoch,
        "best_val_eer": best_val_eer,
        "best_checkpoint": best_checkpoint_name,
        "last_epoch_metrics": last_metrics,
        "checkpoint_path": str(args.checkpoint_path),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    (args.log_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"Training complete: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Best val EER:      {best_val_eer:.4f} (epoch {best_epoch})")
    print(f"Checkpoint:        {args.checkpoint_path}")
    print(f"Logs:              {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
