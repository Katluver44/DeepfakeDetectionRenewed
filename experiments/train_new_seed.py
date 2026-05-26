#!/usr/bin/env python3
"""
train_new_seed.py
=================
Retrains robust_goat with a different random seed for E5 replication study.
Replicates the training procedure from robust.ipynb exactly, only changing seed.

Usage:
  venv/bin/python3 experiments/train_new_seed.py --seed 7
  venv/bin/python3 experiments/train_new_seed.py --seed 7 --epochs 7

Output: models/robust_goat_seed{SEED}.ckpt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F_nn

# ── Reproducibility ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",   type=int, default=7)
parser.add_argument("--epochs", type=int, default=7)
parser.add_argument("--wandb",  type=int, default=0,
                    help="1 to enable W&B logging (default: off for replication runs)")
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import pytorch_lightning as pl
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_CKPT    = REPO_ROOT / "models" / f"robust_goat_seed{SEED}.ckpt"
CACHE_DIR   = REPO_ROOT / "data" / "asvspoof_2019_la"
SECRET_PATH = REPO_ROOT / "secret.txt"

DATASET_NAME = "Bisher/ASVspoof_2019_LA"
BATCH_SIZE   = 20
AUG_PROB     = 0.35
AUG_WEIGHTS  = [0.50, 0.30, 0.20]   # noise / pitch_up / reverb

# ── torch.load compat ─────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

from argparse import Namespace
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ── Dataset + loader helpers ──────────────────────────────────────────────────
from loader import (
    _make_balanced_indices, _label_to_int, _crop_policy,
    TARGET_SR, TARGET_SAMPLES, get_eval_dataloader,
)
import torchaudio.functional as F_audio
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio as HFAudio, concatenate_datasets


# ── Augmentation (identical to robust.ipynb) ─────────────────────────────────

def _match_len(x, n):
    if x.shape[-1] < n:
        x = F_nn.pad(x, (0, n - x.shape[-1]))
    return x[..., :n]

def aug_noise(wav):
    snr_db = random.uniform(8, 40)
    sig_power   = wav.pow(2).mean()
    noise       = torch.randn_like(wav)
    noise_power = noise.pow(2).mean()
    scale = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return wav + scale * noise

def aug_pitch_up(wav):
    semitones = random.uniform(1, 5)
    factor    = 2 ** (semitones / 12)
    frac      = Fraction(factor).limit_denominator(20)
    return _match_len(F_audio.resample(wav, frac.numerator, frac.denominator),
                      wav.shape[-1])

def aug_reverb(wav):
    t60        = random.choice([0.20, 0.35, 0.50, 0.70, 0.90])
    room_scale = random.uniform(0.25, 0.80)
    T          = wav.shape[-1]
    rir_len    = min(int(t60 * TARGET_SR), 1600)
    t          = torch.linspace(0, t60, rir_len)
    decay      = torch.exp(-6.9 * t / t60)
    rir        = torch.randn(rir_len) * decay
    for d_ms in [15, 30, 50]:
        d = int(d_ms * 1e-3 * TARGET_SR * room_scale)
        if d < rir_len:
            rir[d] += 0.4 * room_scale * decay[d]
    rir   = rir / (rir.abs().max() + 1e-8)
    n_fft = 2 ** math.ceil(math.log2(T + rir_len - 1))
    out   = torch.fft.irfft(
        torch.fft.rfft(wav, n=n_fft) * torch.fft.rfft(rir, n=n_fft), n=n_fft)
    return out[..., :T]

AUG_FNS   = [aug_noise, aug_pitch_up, aug_reverb]
AUG_NAMES = ["noise",   "pitch_up",   "reverb"]


class RobustDataset(Dataset):
    def __init__(self, hf_ds, indices, label_key):
        self.ds        = hf_ds
        self.indices   = indices
        self.label_key = label_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex  = self.ds[self.indices[idx]]
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        wav = _crop_policy(wav, "train")
        if random.random() < AUG_PROB:
            fn  = random.choices(AUG_FNS, weights=AUG_WEIGHTS, k=1)[0]
            wav = fn(wav)
        y = _label_to_int(ex[self.label_key])
        return {"audio": wav,
                "label": torch.tensor(y).long(),
                "sample_rate": TARGET_SR}


# ── Model ─────────────────────────────────────────────────────────────────────

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name   = network_name
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Replication training  seed={SEED}  epochs={args.epochs} ===")
    print(f"Output: {OUT_CKPT}")

    hf_token = SECRET_PATH.read_text().strip() if SECRET_PATH.exists() else None

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading HF train + test splits...")
    hf_train = load_dataset(DATASET_NAME, split="train",
                             cache_dir=str(CACHE_DIR), token=hf_token)
    hf_test  = load_dataset(DATASET_NAME, split="test",
                             cache_dir=str(CACHE_DIR), token=hf_token)
    hf_train = hf_train.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
    hf_test  = hf_test.cast_column( "audio", HFAudio(sampling_rate=TARGET_SR))
    hf_pool  = concatenate_datasets([hf_train, hf_test])
    print(f"Pool size: {len(hf_pool)}")

    label_key = next(
        (k for k in hf_pool[0].keys()
         if "label" in k.lower() or k.lower() == "key"), "label")
    print(f"Label key: '{label_key}'")

    labels  = [_label_to_int(hf_pool[i][label_key]) for i in range(len(hf_pool))]
    indices = _make_balanced_indices(labels, seed=SEED, limit=30000)
    counts  = Counter(_label_to_int(hf_pool[i][label_key]) for i in indices)
    print(f"Balanced pool (seed={SEED}): {len(indices)} "
          f"({counts[0]} bonafide + {counts[1]} spoof)")

    train_ds = RobustDataset(hf_pool, indices, label_key)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    print(f"Train: {len(train_ds)} samples | {len(train_dl)} batches/epoch")

    val_dl = get_eval_dataloader(
        source="hf", split="validation", hf_name=DATASET_NAME,
        batch_size=BATCH_SIZE, hf_cache_dir=str(CACHE_DIR),
        limit=3000, hf_token=hf_token)
    print(f"Val: {len(val_dl.dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    from callbacks_rational import (
        BinaryACC_Callback, BinaryAUC_Callback, EER_Callback,
        TPR_Callback, TNR_Callback, FPR_Callback, FNR_Callback,
    )
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))

    model = Phoneme_GAT_lit(cfg=cfg)
    print(f"Model: Phoneme_GAT_lit  "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")

    # ── Logger ────────────────────────────────────────────────────────────────
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        if wandb.run is not None:
            wandb.finish()
        logger = WandbLogger(
            project="DeepfakeDetectionRenewed",
            name=f"robust-goat-seed{SEED}",
            tags=["e5-replication", f"seed{SEED}"],
            log_model=False,
        )
        logger.experiment.config.update({
            "seed": SEED, "max_epochs": args.epochs, "batch_size": BATCH_SIZE,
            "aug_prob": AUG_PROB, "n_edges": 10,
        }, allow_val_change=True)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(REPO_ROOT / "logs"), name=f"e5_seed{SEED}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[
            BinaryACC_Callback(batch_key="label", output_key="logit"),
            BinaryAUC_Callback(batch_key="label", output_key="logit"),
            EER_Callback(batch_key="label", output_key="logit"),
            TPR_Callback(batch_key="label", output_key="logit"),
            TNR_Callback(batch_key="label", output_key="logit"),
            FPR_Callback(batch_key="label", output_key="logit"),
            FNR_Callback(batch_key="label", output_key="logit"),
        ],
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dl, val_dl)

    trainer.save_checkpoint(str(OUT_CKPT))
    print(f"\n✅ Saved: {OUT_CKPT}")


if __name__ == "__main__":
    main()
