#!/usr/bin/env python3
"""
Workstream E — prepare a deliberately tiny (~500-file) ASVspoof-2019-LA
training set + small val set for mini_goat.

Mirrors experiments/scripts/prepare_mlaad_tiny.py's OUTPUT SHAPE (preprocessed
.pt tensors + train.json/val.json split files consumable by a
MAALDSplitDataset-style loader), but sources audio directly from the already
-cached HF dataset `Bisher/as_vspoof_2019_la` (arrow cache under
data/asvspoof_2019_la/, HF_DATASETS_OFFLINE=1, no network).

Attack/split mapping (confirmed by direct inspection of the HF dataset):
  train      : system_id in {-, A01..A06}   (25380 utts) -- TRAIN mini_goat here
  validation : system_id in {-, A01..A06}   (24844 utts) -- small val set for monitoring
  test       : system_id in {-, A07..A19}   (71237 utts) -- the 1600-utt EVAL
               selection scored by i1_geometry_causal_decomp.py / robust_goat.

Because mini_goat trains on `train` (A01-A06) and is scored on `test`
(A07-A19), it is both UTTERANCE- and ATTACK-disjoint from the eval set --
identical unseen-attack protocol robust_goat was trained under.

Outputs -> experiments/data/mini_goat_processed/
    audio/*.pt              (48000,) float32, 16 kHz, center-cropped
    splits/train.json       ~500 records (250 bonafide + 250 spoof), seed=42
    splits/val.json         ~150 records (balanced), from `validation` split
    prep_meta.json
"""
from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "experiments" / "data" / "mini_goat_processed"
AUDIO_DIR = OUT_DIR / "audio"
SPLITS_DIR = OUT_DIR / "splits"
DATA_CACHE = PROJECT_ROOT / "data" / "asvspoof_2019_la"

TARGET_SR = 16_000
TARGET_SAMPLES = 3 * TARGET_SR  # 48000
SEED = 42

N_TRAIN_PER_CLASS = 250   # -> 500 total
N_VAL_PER_CLASS = 75      # -> 150 total


def _crop_center(wav: np.ndarray) -> torch.Tensor:
    """Center-crop / tile-pad to TARGET_SAMPLES, matches loader.py eval policy."""
    x = torch.tensor(wav, dtype=torch.float32)
    if x.ndim > 1:
        x = x.mean(0)
    T = x.shape[0]
    if T < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // T)
        x = x.repeat(reps)
        T = x.shape[0]
    start = (T - TARGET_SAMPLES) // 2
    return x[start:start + TARGET_SAMPLES].clone()


def build_split(ds, split_name: str, n_per_class: int, seed: int, prefix: str):
    sysids = ds["system_id"]
    bona_idx = [i for i, s in enumerate(sysids) if s == "-"]
    spoof_idx = [i for i, s in enumerate(sysids) if s != "-"]
    rng = np.random.default_rng(seed)
    sel_bona = rng.choice(bona_idx, n_per_class, replace=False).tolist()
    sel_spoof = rng.choice(spoof_idx, n_per_class, replace=False).tolist()
    sel = sorted(sel_bona + sel_spoof)
    print(f"  [{split_name}] selected {len(sel)} utts "
          f"({len(sel_bona)} bonafide + {len(sel_spoof)} spoof) from HF split "
          f"with {len(bona_idx)} bonafide / {len(spoof_idx)} spoof available")

    from datasets import Audio as HFAudio
    sub = ds.select(sel).cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

    records = []
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for j in range(len(sub)):
        ex = sub[j]
        orig_idx = sel[j]
        label = "bonafide" if sysids[orig_idx] == "-" else "spoof"
        attack = sysids[orig_idx]
        wav = _crop_center(ex["audio"]["array"])
        fname = f"{prefix}_{orig_idx:06d}.pt"
        torch.save(wav, AUDIO_DIR / fname)
        records.append({
            "audio_path": fname,
            "label": label,
            "attack_system": attack,
            "language": "en",
            "orig_hf_index": orig_idx,
            "hf_split": split_name,
            "speaker_id": ex.get("speaker_id", "unknown"),
        })
    return records


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[prepare_mini_goat_data] loading HF dataset Bisher/as_vspoof_2019_la (offline) ...")
    from datasets import load_dataset
    ds = load_dataset("Bisher/as_vspoof_2019_la", cache_dir=str(DATA_CACHE),
                       trust_remote_code=True)

    train_ds = ds["train"]
    val_ds = ds["validation"]

    # sanity: train/val must be A01-A06 only (disjoint from eval A07-A19)
    train_attacks = set(a for a in set(train_ds["system_id"]) if a != "-")
    val_attacks = set(a for a in set(val_ds["system_id"]) if a != "-")
    eval_attacks = {f"A{i:02d}" for i in range(7, 20)}
    assert train_attacks.isdisjoint(eval_attacks), \
        f"train attacks {train_attacks} overlap eval attacks!"
    assert val_attacks.isdisjoint(eval_attacks), \
        f"val attacks {val_attacks} overlap eval attacks!"
    print(f"  train attacks: {sorted(train_attacks)} (disjoint from eval {sorted(eval_attacks)}: OK)")
    print(f"  val attacks:   {sorted(val_attacks)} (disjoint from eval: OK)")

    print("\n[prepare_mini_goat_data] building ~500-file balanced TRAIN split from HF `train` ...")
    train_records = build_split(train_ds, "train", N_TRAIN_PER_CLASS, SEED, prefix="tr")

    print("\n[prepare_mini_goat_data] building small balanced VAL split from HF `validation` ...")
    val_records = build_split(val_ds, "validation", N_VAL_PER_CLASS, SEED + 1, prefix="va")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "train.json").write_text(json.dumps(train_records, indent=2))
    (SPLITS_DIR / "val.json").write_text(json.dumps(val_records, indent=2))

    meta = {
        "seed": SEED,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "train_source_hf_split": "train (attacks A01-A06 + bonafide)",
        "val_source_hf_split": "validation (attacks A01-A06 + bonafide)",
        "eval_source_hf_split": "test (attacks A07-A19 + bonafide) -- scored separately, NOT prepared here",
        "target_sr": TARGET_SR,
        "target_samples": TARGET_SAMPLES,
        "crop_policy": "center crop for both train and val preprocessing "
                        "(same deviation as prepare_mlaad_tiny.py; train-time "
                        "random-crop augmentation, if any, is applied at load "
                        "time by the training script from these 3s tensors -- "
                        "since tensors are already exactly 3s there is no "
                        "additional crop freedom, matching the MLAAD-tiny deviation)",
        "note": "mini_goat trains ONLY on HF `train` split (A01-A06), which is "
                "both utterance- and attack-disjoint from the 1600-utt A07-A19 "
                "eval selection used by i1_geometry_causal_decomp.py / robust_goat.",
    }
    (OUT_DIR / "prep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[prepare_mini_goat_data] done. train={len(train_records)} val={len(val_records)}")
    print(f"  -> {OUT_DIR}")


if __name__ == "__main__":
    main()
