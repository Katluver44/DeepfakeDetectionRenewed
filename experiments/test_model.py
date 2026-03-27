"""
test_model.py — smoke-test robust_goat.ckpt on a holdout split.

Evaluates on the ASVspoof 2019 LA validation set (never seen during training).
Uses soundfile to decode audio directly, bypassing the torchcodec dependency
that datasets>=4.4.0 introduced.
"""
from __future__ import annotations

import argparse
import io
import os
import random
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# Patch torch.load to default weights_only=False (same as the training notebook).
# The checkpoint contains Namespace and other non-tensor objects.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

# Register safe globals so weights_only=True paths also work if called elsewhere.
try:
    from pandas import Series as _PandasSeries
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PandasSeries, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT       = REPO_ROOT / "models" / "robust_goat.ckpt"
DEFAULT_HF_DATASET = "Bisher/ASVspoof_2019_LA"
DEFAULT_CACHE_DIR  = REPO_ROOT / "data" / "asvspoof_2019_la"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR  # 48 000 frames = 3 s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test robust_goat.ckpt on the validation holdout.")
    p.add_argument("--checkpoint",    type=Path,  default=DEFAULT_CKPT)
    p.add_argument("--dataset-name",             default=DEFAULT_HF_DATASET)
    p.add_argument("--cache-dir",     type=Path,  default=DEFAULT_CACHE_DIR)
    p.add_argument("--limit",         type=int,   default=100,
                   help="Total examples to evaluate (balanced 50/50 by default).")
    p.add_argument("--batch-size",    type=int,   default=10)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--device",                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--hf-token-path", type=Path,  default=REPO_ROOT / "secret.txt")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hf_token(path: Path) -> str | None:
    if path.exists():
        tok = path.read_text(encoding="utf-8").strip()
        return tok or None
    return None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_audio_entry(entry: dict) -> torch.Tensor:
    """
    Decode a datasets Audio entry (decode=False) → (1, T) float32 tensor at TARGET_SR.

    The entry has keys 'path' and 'bytes'.  We prefer bytes (already in memory)
    and fall back to path for file-backed caches.
    """
    raw_bytes = entry.get("bytes")
    path      = entry.get("path")

    if raw_bytes is not None:
        arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    elif path is not None:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio entry has neither 'bytes' nor 'path'.")

    wave = torch.from_numpy(arr)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)          # (1, T)
    elif wave.ndim == 2:
        wave = wave.mean(0, keepdim=True) # stereo → mono

    if sr != TARGET_SR:
        wave = T.Resample(sr, TARGET_SR)(wave)

    return wave


def _crop_center(wave: torch.Tensor) -> torch.Tensor:
    """Center-crop (or tile-pad) to exactly TARGET_SAMPLES frames."""
    T = wave.shape[-1]
    if T < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // T)   # ceiling division
        wave = wave.repeat(1, reps)
    start = (wave.shape[-1] - TARGET_SAMPLES) // 2
    return wave[:, start : start + TARGET_SAMPLES]


def _label_to_int(raw) -> int:
    """'bonafide'→0, 'spoof'→1 (or any integer-like)."""
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "bonafide", "real", "genuine"):
            return 0
        return 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset: loads HF with decode=False, decodes audio with soundfile
# ---------------------------------------------------------------------------

class SoundfileHFDataset(torch.utils.data.Dataset):
    """
    Wraps an HF dataset split.  Audio is decoded with soundfile so torchcodec
    is not required.  Labels are balanced 50/50 up to `limit` total examples.
    """

    def __init__(self, hf_name: str, split: str, cache_dir: str,
                 limit: int | None, token: str | None, seed: int = 42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir,
                          token=token)
        # decode=False → gives {"path": ..., "bytes": ...} without torchcodec
        self.ds = ds.cast_column("audio", HFAudio(decode=False))

        # Auto-detect label column (access via no-decode version)
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )

        # Build label list without touching audio
        labels = [_label_to_int(self.ds[i][self.label_key]) for i in range(len(self.ds))]

        # Balanced indices
        idx0 = [i for i, y in enumerate(labels) if y == 0]
        idx1 = [i for i, y in enumerate(labels) if y == 1]
        rng = random.Random(seed)
        rng.shuffle(idx0); rng.shuffle(idx1)

        k = min(len(idx0), len(idx1))
        if limit is not None:
            k = min(k, limit // 2)

        self.indices = idx0[:k] + idx1[:k]
        rng.shuffle(self.indices)

        counts = Counter(_label_to_int(self.ds[i][self.label_key]) for i in self.indices)
        print(f"Holdout split '{split}': {len(self.indices)} examples  "
              f"({counts[0]} bonafide/normal  +  {counts[1]} spoof/deepfake)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex   = self.ds[self.indices[idx]]
        wave = _decode_audio_entry(ex["audio"])
        wave = _crop_center(wave)
        y    = _label_to_int(ex[self.label_key])
        return {
            "audio":       wave,
            "label":       torch.tensor(y, dtype=torch.long),
            "sample_rate": TARGET_SR,
        }


# ---------------------------------------------------------------------------
# Phoneme-model loader patch
# ---------------------------------------------------------------------------

def patch_phoneme_loader() -> None:
    """
    Monkey-patch load_phoneme_model so it can construct the backbone from
    HuggingFace weights without needing a local phoneme pretrain checkpoint
    (the full model weights come from robust_goat.ckpt anyway).
    """
    import phoneme_GAT.modules as modules_mod
    import phoneme_GAT.phoneme_model as pm_mod
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        if network_name.lower() == "wavlm":
            network_param.pretrained_name = "microsoft/wavlm-base"
        else:
            network_param.pretrained_name = "facebook/wav2vec2-base-960h"
        network_param.vocab_size = total_num_phonemes

        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path),
                network_param=network_param,
                optim_param=optim_param,
                tokenizer=None,
                total_num_phonemes=total_num_phonemes,
                weights_only=False,
            ).cpu()

        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)

    pm_mod.load_phoneme_model    = _load
    modules_mod.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Pass --checkpoint /path/to/robust_goat.ckpt"
        )

    # Silence the deprecated torchaudio backend setter if needed
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    hf_token = load_hf_token(args.hf_token_path)

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = SoundfileHFDataset(
        hf_name   = args.dataset_name,
        split     = "validation",
        cache_dir = str(args.cache_dir),
        limit     = args.limit,
        token     = hf_token,
        seed      = args.seed,
    )
    label_counts   = Counter(
        _label_to_int(dataset.ds[dataset.indices[i]][dataset.label_key])
        for i in range(len(dataset))
    )
    normal_count   = label_counts[0]
    deepfake_count = label_counts[1]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        drop_last   = False,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    patch_phoneme_loader()

    from phoneme_GAT.modules import Phoneme_GAT_lit

    cfg = Namespace(
        PhonemeGAT=Namespace(
            backbone  = "wavlm",
            use_raw   = True,   # no local phoneme ckpt needed; weights come from robust_goat.ckpt
            use_GAT   = True,
            n_edges   = 10,
            use_aug   = True,
            use_pool  = True,
            use_clip  = True,
        )
    )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = Phoneme_GAT_lit.load_from_checkpoint(
        str(args.checkpoint),
        cfg          = cfg,
        map_location = "cpu",
        strict       = True,
    )
    model.eval()
    model.freeze()
    device = torch.device(args.device)
    model.to(device)
    print(f"Model loaded on {device}.\n")

    # ── Eval loop ────────────────────────────────────────────────────────────
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(device)
            labels = batch["label"].to(device)
            B      = labels.shape[0]
            num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)

            out    = model.model(audio, num_frames, profiler=None, use_aug=False, stage="eval")
            all_logits.append(out["logit"].detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.sigmoid(logits)
    preds  = (probs >= 0.5).long()

    acc = (preds == labels).float().mean().item()
    tp  = int(((preds == 1) & (labels == 1)).sum())
    tn  = int(((preds == 0) & (labels == 0)).sum())
    fp  = int(((preds == 1) & (labels == 0)).sum())
    fn  = int(((preds == 0) & (labels == 1)).sum())

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Checkpoint : {args.checkpoint.name}")
    print(f"Dataset    : {args.dataset_name}  split=validation")
    print(f"Device     : {device}")
    print("-" * 55)
    print(f"Total examples evaluated : {len(labels)}")
    print(f"  Normal (bonafide)      : {normal_count}")
    print(f"  Deepfake (spoof)       : {deepfake_count}")
    print("-" * 55)
    print(f"Accuracy   : {acc:.4f}  ({int(acc*len(labels))}/{len(labels)} correct)")
    print(f"Confusion  : TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("=" * 55)


if __name__ == "__main__":
    main()
