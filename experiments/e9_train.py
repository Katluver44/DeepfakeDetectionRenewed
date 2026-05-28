#!/usr/bin/env python3
"""
e9_train.py — E9: Train on mixed WaveFake (TTS) + VCC2020 (VC) + LibriSpeech (bonafide).

Tests whether the routing-vs-skip-route pathway split emerges in models trained
on a completely different attack distribution (neural-vocoder TTS + voice conversion).

Architecture: identical to existing seed-1 model (WavLM + BiLSTM + 3-layer GAT).
Training data:
  - Spoof TTS: WaveFake (Jöhren & Kolossa 2021) — vocoder resynthesis of LJSpeech/VCTK
  - Spoof VC:  VCC2020 Task 1 (Yi et al. 2020) — intra-lingual voice conversion
  - Bonafide:  LibriSpeech train-clean-100 (Panayotov et al. 2015)
  Composition: 50/50 bonafide/spoof; 50/50 TTS/VC within spoof; ~9,920 total.

Usage:
  venv/bin/python3 experiments/e9_train.py --seed 1
  venv/bin/python3 experiments/e9_train.py --seed 2

Output:
  models/e9_seed{SEED}.ckpt
  experiments/results/e9_mixed_training/seed{SEED}_train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F_nn

parser = argparse.ArgumentParser()
parser.add_argument("--seed",   type=int, default=1)
parser.add_argument("--epochs", type=int, default=7)
parser.add_argument("--wandb",  type=int, default=0)
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_CKPT    = REPO_ROOT / "models" / f"e9_seed{SEED}.ckpt"
LOG_DIR     = REPO_ROOT / "experiments" / "results" / "e9_mixed_training"
CACHE_DIR   = REPO_ROOT / "data"
SECRET_PATH = REPO_ROOT / "secret.txt"

BATCH_SIZE  = 20
AUG_PROB    = 0.35
AUG_WEIGHTS = [0.50, 0.30, 0.20]   # noise / pitch_up / reverb

# ── torch.load compat ────────────────────────────────────────────────────────
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

from loader import _label_to_int, _crop_policy, _ensure_sr, TARGET_SR, TARGET_SAMPLES
import torchaudio.functional as F_audio
from fractions import Fraction
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio as HFAudio


# ── Augmentation (identical to train_new_seed.py) ────────────────────────────

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

def _augment(wav: torch.Tensor) -> torch.Tensor:
    if random.random() < AUG_PROB:
        fn = random.choices(AUG_FNS, weights=AUG_WEIGHTS, k=1)[0]
        wav = fn(wav)
    return wav


# ── Dataset: generic record list ─────────────────────────────────────────────

class MixedDataset(Dataset):
    """
    Unified dataset for mixed training.
    Records: list of dicts with keys {path, label (int), system_id (str)}.
    Accepts either:
      - file path on disk  (path != "__hf__")
      - HF dataset item    (path == "__hf__", row in hf_ds at index)
    """
    def __init__(self, records: list[dict], mode: str = "train",
                 hf_ds=None):
        self.records = records
        self.mode    = mode
        self.hf_ds   = hf_ds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if rec["path"] == "__hf__":
            ex  = self.hf_ds[rec["hf_idx"]]
            arr = ex["audio"]["array"]
            wav = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            assert ex["audio"]["sampling_rate"] == TARGET_SR
        else:
            import torchaudio
            wav, sr = torchaudio.load(rec["path"])
            wav = _ensure_sr(wav, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

        wav = _crop_policy(wav, self.mode)
        if self.mode == "train":
            wav = _augment(wav)

        return {
            "audio":     wav,
            "label":     torch.tensor(rec["label"]).long(),
            "system_id": rec["system_id"],
            "sample_rate": TARGET_SR,
        }


# ── Build records lists ───────────────────────────────────────────────────────

def build_wavefake_records(wavefake_root: Path, n_target: int, seed: int) -> list[dict]:
    """
    Enumerate all WAV files under wavefake_root/generated_audio/**/*.wav.
    Actual layout: generated_audio/{system_name}/**/*.wav (flat, one dir per system).
    system_name examples: ljspeech_hifiGAN, ljspeech_melgan, etc.
    Skips jsut_* directories (Japanese, not English).
    All files labelled spoof (1). system_id = directory name.
    Returns n_target randomly-sampled records (without replacement if possible).
    """
    audio_root = wavefake_root / "generated_audio"
    records = []
    for sys_dir in sorted(audio_root.iterdir()):
        if not sys_dir.is_dir():
            continue
        sys_name = sys_dir.name
        if sys_name.startswith("jsut"):   # Japanese — exclude
            continue
        for fpath in sys_dir.rglob("*.wav"):
            records.append({
                "path":      str(fpath),
                "label":     1,
                "system_id": sys_name,
            })

    if len(records) == 0:
        raise RuntimeError(f"No WAV files found under {audio_root}. "
                           "Is WaveFake extracted?")

    rng = random.Random(seed)
    rng.shuffle(records)

    if n_target <= len(records):
        return records[:n_target]

    print(f"[WARN] WaveFake only has {len(records)} English files; need {n_target}. "
          "Repeating samples.")
    repeats = []
    while len(repeats) < n_target:
        repeats.extend(records)
    return repeats[:n_target]


def build_vcc2020_records(vcc2020_root: Path) -> list[dict]:
    """
    All WAV files in vcc2020_root/audio/*/task1/*.wav.
    system_id = parent system directory name (T01, T02, ...).
    """
    import json
    manifest = vcc2020_root / "manifest_task1.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        records = []
        for system_id, paths in m.items():
            for p in paths:
                if os.path.exists(p):
                    records.append({
                        "path":      p,
                        "label":     1,
                        "system_id": f"VCC2020_{system_id}",
                    })
        return records

    # Fallback: scan directory
    records = []
    for sysdir in sorted((vcc2020_root / "audio").iterdir()):
        if not sysdir.is_dir():
            continue
        task1 = sysdir / "task1"
        if not task1.exists():
            continue
        for fpath in sorted(task1.glob("*.wav")):
            records.append({
                "path":      str(fpath),
                "label":     1,
                "system_id": f"VCC2020_{sysdir.name}",
            })
    return records


def build_librispeech_records(n_target: int, seed: int, token=None,
                              cache_dir: str = None) -> tuple[list[dict], object]:
    """
    Stream n_target samples from LibriSpeech train-clean-100 on HF.
    Returns (records_list, hf_dataset_object).
    Records use path="__hf__" + hf_idx to avoid materialising audio into RAM.
    """
    print(f"Loading LibriSpeech train.100 from HF (n={n_target})...")
    ds = load_dataset(
        "openslr/librispeech_asr", "clean",
        split="train.100",
        cache_dir=cache_dir or str(CACHE_DIR / "librispeech"),
        token=token,
    )
    ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

    # Random subset
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_target]

    records = [
        {
            "path":      "__hf__",
            "hf_idx":    i,
            "label":     0,
            "system_id": "librispeech",
        }
        for i in indices
    ]
    print(f"  → {len(records)} LibriSpeech bonafide records")
    return records, ds


# ── Model patch (identical to train_new_seed.py) ──────────────────────────────

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name    = network_name
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== E9 Mixed Training  seed={SEED}  epochs={args.epochs} ===")
    print(f"Output checkpoint: {OUT_CKPT}")

    token = SECRET_PATH.read_text().strip() if SECRET_PATH.exists() else None

    # ── Build spoof records ───────────────────────────────────────────────────
    wavefake_root = REPO_ROOT / "data" / "wavefake"
    vcc2020_root  = REPO_ROOT / "data" / "vcc2020"

    # VCC2020: use all available
    vcc_records = build_vcc2020_records(vcc2020_root)
    n_vc = len(vcc_records)
    print(f"VCC2020 Task1: {n_vc} records across "
          f"{len(set(r['system_id'] for r in vcc_records))} systems")

    # WaveFake: sample to match VCC2020 count (50/50 TTS/VC within spoof)
    wf_records = build_wavefake_records(wavefake_root, n_target=n_vc, seed=SEED)
    n_tts = len(wf_records)
    print(f"WaveFake TTS:  {n_tts} records across "
          f"{len(set(r['system_id'] for r in wf_records))} systems")

    # LibriSpeech: match total spoof count
    n_bona = n_vc + n_tts
    ls_records, librispeech_ds = build_librispeech_records(
        n_target=n_bona, seed=SEED, token=token,
        cache_dir=str(CACHE_DIR / "librispeech"))

    # Combine and split 80/20
    all_records = vcc_records + wf_records + ls_records
    rng = random.Random(SEED)
    rng.shuffle(all_records)

    n_val   = max(1, int(0.2 * len(all_records)))
    val_rec = all_records[:n_val]
    trn_rec = all_records[n_val:]

    label_counts = Counter(r["label"] for r in trn_rec)
    sys_counts   = Counter(r["system_id"] for r in trn_rec)
    print(f"\nTrain: {len(trn_rec)} samples  (bonafide={label_counts[0]}, spoof={label_counts[1]})")
    print(f"Val:   {len(val_rec)} samples")
    print(f"Systems in train: {dict(sys_counts)}\n")

    # Log manifest for reproducibility
    manifest_path = LOG_DIR / f"seed{SEED}_train_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "path", "label", "system_id", "hf_idx"],
                           extrasaction="ignore")
        w.writeheader()
        for rec in trn_rec:
            w.writerow({**rec, "split": "train", "hf_idx": rec.get("hf_idx", "")})
        for rec in val_rec:
            w.writerow({**rec, "split": "val", "hf_idx": rec.get("hf_idx", "")})
    print(f"Manifest saved: {manifest_path}")

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_ds = MixedDataset(trn_rec, mode="train",  hf_ds=librispeech_ds)
    val_ds   = MixedDataset(val_rec, mode="eval",   hf_ds=librispeech_ds)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True,
                          collate_fn=_collate)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, drop_last=False,
                          collate_fn=_collate)

    print(f"Train batches/epoch: {len(train_dl)}  Val batches: {len(val_dl)}")

    # ── Model (identical config to seed-1) ───────────────────────────────────
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Phoneme_GAT_lit  trainable={n_params/1e6:.1f}M params")

    # ── Logger ────────────────────────────────────────────────────────────────
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        if wandb.run is not None:
            wandb.finish()
        logger = WandbLogger(
            project="DeepfakeDetectionRenewed",
            name=f"e9-mixed-seed{SEED}",
            tags=["e9", f"seed{SEED}", "wavefake", "vcc2020"],
            log_model=False,
        )
        logger.experiment.config.update({
            "seed": SEED, "max_epochs": args.epochs, "batch_size": BATCH_SIZE,
            "n_tts": n_tts, "n_vc": n_vc, "n_bona": n_bona,
        }, allow_val_change=True)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(LOG_DIR), name=f"seed{SEED}")

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


def _collate(batch):
    """Drop system_id from batch tensors; keep as list for logging."""
    audio  = torch.stack([b["audio"]  for b in batch])
    labels = torch.stack([b["label"]  for b in batch])
    return {
        "audio":       audio,
        "label":       labels,
        "sample_rate": TARGET_SR,
        "system_ids":  [b["system_id"] for b in batch],
    }


if __name__ == "__main__":
    main()
