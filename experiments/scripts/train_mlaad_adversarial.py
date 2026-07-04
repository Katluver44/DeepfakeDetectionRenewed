#!/usr/bin/env python3
"""
Train the adversarial (mlaad_robust_goat) model on MLAAD-tiny.

Converted from robust.ipynb. Training logic preserved exactly; changes:
  (i)  dataset path from --processed-dir argument
  (ii) checkpoint path from --checkpoint-path argument
  (iii) deterministic seed throughout

Usage:
    python experiments/scripts/train_mlaad_adversarial.py

Produces:
    experiments/checkpoints/mlaad_robust_goat.ckpt
    experiments/results/mlaad/training_logs/mlaad_robust_goat/
        training_curves.csv
        final_metrics.json
        hyperparameters.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F_audio
from torch.utils.data import Dataset, DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from loader import _make_balanced_indices, TARGET_SR, TARGET_SAMPLES
from callbacks import EER_Callback
from callbacks_rational import (
    BinaryACC_Callback, BinaryAUC_Callback,
    TPR_Callback, TNR_Callback, FPR_Callback, FNR_Callback,
)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
# Match ASVspoof robust.ipynb exactly; deviations documented below.
HP = {
    "backbone": "wavlm",
    "use_GAT": True,
    "n_edges": 10,
    "use_aug": True,
    "use_pool": True,
    "use_clip": True,
    "batch_size": 20,
    "max_epochs": 7,
    # Augmentation — proportional to goat.pth struggle scores (from robust.ipynb)
    "aug_prob": 0.35,             # 35% of samples get augmented
    "aug_weights": [0.50, 0.30, 0.20],   # noise, pitch_up, reverb
    "aug_names": ["noise", "pitch_up", "reverb"],
    "noise_snr_db_range": (8.0, 40.0),
    "pitch_semitone_range": (1.0, 5.0),
    "reverb_t60_values": [0.20, 0.35, 0.50, 0.70, 0.90],
    "num_workers": 4,
    "drop_last": True,
    "seed": 42,
}

DEVIATIONS = [
    "No random crop: preprocessed .pt files are center-cropped to 48000 samples "
    "(ASVspoof used random crop during training).",
    "No train-sample limit: ASVspoof used limit=30000; MLAAD-tiny has fewer samples, "
    "so all available balanced pairs are used.",
    "No wandb logging: replaced by CSV + JSON for reproducibility.",
    "Dataset source: MLAAD-tiny splits (train.json / val.json) instead of "
    "Bisher/ASVspoof_2019_LA HuggingFace dataset.",
]

# ─── Augmentation functions (verbatim from robust.ipynb) ─────────────────────

def _match_len(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[-1] < n:
        x = torch.nn.functional.pad(x, (0, n - x.shape[-1]))
    return x[..., :n]


def aug_noise(wav: torch.Tensor) -> torch.Tensor:
    lo, hi = HP["noise_snr_db_range"]
    snr_db = random.uniform(lo, hi)
    sig_power = wav.pow(2).mean()
    noise = torch.randn_like(wav)
    noise_power = noise.pow(2).mean()
    scale = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return wav + scale * noise


def aug_pitch_up(wav: torch.Tensor) -> torch.Tensor:
    lo, hi = HP["pitch_semitone_range"]
    semitones = random.uniform(lo, hi)
    factor = 2 ** (semitones / 12)
    frac = Fraction(factor).limit_denominator(20)
    return _match_len(F_audio.resample(wav, frac.numerator, frac.denominator), wav.shape[-1])


def aug_reverb(wav: torch.Tensor) -> torch.Tensor:
    t60 = random.choice(HP["reverb_t60_values"])
    room_scale = random.uniform(0.25, 0.80)
    T = wav.shape[-1]
    rir_len = min(int(t60 * TARGET_SR), 1600)
    t = torch.linspace(0, t60, rir_len)
    decay = torch.exp(-6.9 * t / t60)
    rir = torch.randn(rir_len) * decay
    for d_ms in [15, 30, 50]:
        d = int(d_ms * 1e-3 * TARGET_SR * room_scale)
        if d < rir_len:
            rir[d] += 0.4 * room_scale * decay[d]
    rir = rir / (rir.abs().max() + 1e-8)
    n_fft = 2 ** math.ceil(math.log2(T + rir_len - 1))
    out = torch.fft.irfft(
        torch.fft.rfft(wav, n=n_fft) * torch.fft.rfft(rir, n=n_fft), n=n_fft
    )
    return out[..., :T]


AUG_FNS = [aug_noise, aug_pitch_up, aug_reverb]

# ─── Dataset ─────────────────────────────────────────────────────────────────

class MAALDSplitDataset(Dataset):
    """
    Reads preprocessed .pt tensors from a MLAAD-tiny split JSON file.
    Each .pt file is shape (48000,) float32 (3 s at 16 kHz, center-cropped).
    Labels: 0=bonafide, 1=spoof.
    """

    def __init__(
        self,
        split_json: Path,
        processed_dir: Path,
        mode: str = "train",
        balance: bool = True,
        limit: int | None = None,
        seed: int = 42,
        aug_prob: float = 0.0,
        aug_fns: list | None = None,
        aug_weights: list[float] | None = None,
    ):
        self.processed_dir = processed_dir
        self.mode = mode
        self.aug_prob = aug_prob
        self.aug_fns = aug_fns or []
        self.aug_weights = aug_weights or []

        records = json.loads(split_json.read_text())
        labels = [0 if r["label"] == "bonafide" else 1 for r in records]

        n_bonafide = sum(1 for y in labels if y == 0)
        n_spoof = sum(1 for y in labels if y == 1)

        if n_bonafide == 0 or n_spoof == 0:
            raise RuntimeError(
                f"Split {split_json} has {n_bonafide} bonafide and {n_spoof} spoof. "
                "Both classes required. Re-run prepare_mlaad_tiny.py after bonafide download."
            )

        if balance:
            indices = _make_balanced_indices(labels, seed=seed, limit=limit)
            self.records = [records[i] for i in indices]
        else:
            self.records = records

        n_bf = sum(1 for r in self.records if r["label"] == "bonafide")
        n_sp = sum(1 for r in self.records if r["label"] == "spoof")
        print(
            f"  MAALDSplitDataset({split_json.name}, {mode}): "
            f"{len(self.records)} samples ({n_bf} bonafide + {n_sp} spoof)"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        tensor = torch.load(self.processed_dir / rec["audio_path"])  # (48000,)
        wav = tensor.unsqueeze(0)  # (1, 48000)

        if self.mode == "train" and self.aug_fns and random.random() < self.aug_prob:
            aug_fn = random.choices(self.aug_fns, weights=self.aug_weights, k=1)[0]
            wav = aug_fn(wav)

        y = 0 if rec["label"] == "bonafide" else 1
        return {
            "audio": wav,
            "label": torch.tensor(y, dtype=torch.long),
            "sample_rate": TARGET_SR,
            "attack_system": rec.get("attack_system", "unknown"),
            "language": rec.get("language", "unknown"),
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir", type=Path,
        default=PROJECT_ROOT / "experiments/data/mlaad_tiny_processed",
    )
    p.add_argument(
        "--checkpoint-path", type=Path,
        default=PROJECT_ROOT / "experiments/checkpoints/mlaad_robust_goat.ckpt",
    )
    p.add_argument(
        "--log-dir", type=Path,
        default=PROJECT_ROOT / "experiments/results/mlaad/training_logs/mlaad_robust_goat",
    )
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

    # ── Write hyperparameters ────────────────────────────────────────────────
    hp_snapshot = dict(HP)
    hp_snapshot.update({
        "seed": args.seed,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_path": str(args.checkpoint_path),
        "processed_dir": str(args.processed_dir),
        "script": "train_mlaad_adversarial.py",
        "model_variant": "mlaad_robust_goat",
        "aug_rationale": (
            "Proportional to goat.pth struggle scores from attack sweep "
            "(noise=50%, pitch_up=30%, reverb=20%) — from robust.ipynb"
        ),
        "deviations_from_asvspoof": DEVIATIONS,
    })
    (args.log_dir / "hyperparameters.json").write_text(json.dumps(hp_snapshot, indent=2))
    print("Hyperparameters written.")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets …")
    splits_dir = args.processed_dir / "splits"

    train_ds = MAALDSplitDataset(
        splits_dir / "train.json",
        args.processed_dir,
        mode="train",
        balance=True,
        seed=args.seed,
        aug_prob=HP["aug_prob"],
        aug_fns=AUG_FNS,
        aug_weights=HP["aug_weights"],
    )
    val_ds = MAALDSplitDataset(
        splits_dir / "val.json",
        args.processed_dir,
        mode="eval",
        balance=True,
        seed=args.seed,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=HP["num_workers"],
        pin_memory=True,
        drop_last=HP["drop_last"],
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=HP["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    print(
        f"Train: {len(train_ds)} samples ({len(train_dl)} batches/epoch)\n"
        f"Val:   {len(val_ds)} samples ({len(val_dl)} batches)\n"
        f"Aug breakdown: noise={HP['aug_weights'][0]*100:.0f}%  "
        f"pitch_up={HP['aug_weights'][1]*100:.0f}%  "
        f"reverb={HP['aug_weights'][2]*100:.0f}%  (of {HP['aug_prob']*100:.0f}% augmented)\n"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = Namespace(
        PhonemeGAT=Namespace(
            backbone=HP["backbone"],
            use_raw=False,
            use_GAT=HP["use_GAT"],
            n_edges=HP["n_edges"],
            use_aug=HP["use_aug"],
            use_pool=HP["use_pool"],
            use_clip=HP["use_clip"],
        )
    )
    model = Phoneme_GAT_lit(cfg=cfg)

    # ── Callbacks + Trainer ──────────────────────────────────────────────────
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger as PLCSVLogger

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
        dirpath=args.checkpoint_path.parent,
        filename=args.checkpoint_path.stem + "-best-{epoch:02d}-{val-eer:.4f}",
        monitor="val-eer",
        mode="min",
        save_last=True,
        verbose=True,
    )

    pl_csv_logger = PLCSVLogger(
        save_dir=str(args.log_dir),
        name="",
        version="",
        flush_logs_every_n_steps=10,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=pl_csv_logger,
        callbacks=[*metric_callbacks, ckpt_callback],
        log_every_n_steps=10,
        deterministic=False,
        fast_dev_run=args.fast_dev_run,
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"Starting training: {args.epochs} epochs, batch={args.batch_size}")
    start_time = time.time()
    trainer.fit(model, train_dl, val_dl)
    total_time = time.time() - start_time

    # ── Save final checkpoint ─────────────────────────────────────────────────
    trainer.save_checkpoint(str(args.checkpoint_path))
    print(f"\nSaved checkpoint: {args.checkpoint_path}")

    # ── Write training_curves.csv ─────────────────────────────────────────────
    pl_metrics_path = args.log_dir / "metrics.csv"
    if pl_metrics_path.exists():
        import csv
        from collections import defaultdict
        with open(pl_metrics_path) as f:
            pl_rows = list(csv.DictReader(f))
        epoch_data: dict[int, dict] = defaultdict(dict)
        for row in pl_rows:
            ep = row.get("epoch", "")
            if not ep:
                continue
            ep = int(float(ep))
            for k, v in row.items():
                if v and v != "" and k != "step":
                    try:
                        epoch_data[ep][k] = float(v)
                    except (ValueError, TypeError):
                        pass
        with open(args.log_dir / "training_curves.csv", "w") as f:
            cols = [
                "epoch", "train_loss", "val_loss",
                "val_eer", "val_acc", "val_auc",
                "val_tpr", "val_tnr", "val_fpr", "val_fnr",
            ]
            f.write(",".join(cols) + "\n")
            for ep in sorted(epoch_data):
                d = epoch_data[ep]
                vals = [
                    ep,
                    d.get("train-loss", ""),
                    d.get("val-loss", ""),
                    d.get("val-eer", ""),
                    d.get("val-acc", ""),
                    d.get("val-auc", ""),
                    d.get("val-tpr", ""),
                    d.get("val-tnr", ""),
                    d.get("val-fpr", ""),
                    d.get("val-fnr", ""),
                ]
                f.write(",".join(str(v) for v in vals) + "\n")

    # ── Final metrics ─────────────────────────────────────────────────────────
    import re as _re
    best_val_eer = float("nan")
    best_epoch = -1
    best_checkpoint_name = ""
    try:
        best_model_path = ckpt_callback.best_model_path or ""
        if not best_model_path:
            candidates = sorted(args.checkpoint_path.parent.glob(
                args.checkpoint_path.stem + "-best-*.ckpt"
            ))
            if candidates:
                best_model_path = str(candidates[0])
        if best_model_path:
            best_checkpoint_name = Path(best_model_path).name
            m = _re.search(r"val-eer=([0-9]+(?:\.[0-9]+)?)", best_model_path)
            if m:
                best_val_eer = float(m.group(1))
            m2 = _re.search(r"epoch=(\d+)", best_model_path)
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
        "model_variant": "mlaad_robust_goat",
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
