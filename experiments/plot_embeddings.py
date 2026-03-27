"""
plot_embeddings.py — PCA visualisation of embeddings at three stages:

  1. Frozen phoneme encoder output  (mean-pooled over time)
  2. Trainable encoder output       (mean-pooled, pre-GAT)
  3. Post-GAT + LSTM + pool         (final classification embedding)

Red   = deepfake / spoof  (label 1)
Light blue = normal / bonafide (label 0)

Plots are saved as PNGs in the same directory as this script.
"""
from __future__ import annotations

import io
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

# ── torch.load compat (checkpoint stores Namespace + non-tensor objects) ──
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

try:
    from pandas import Series as _PandasSeries
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PandasSeries, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT    = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET      = "Bisher/ASVspoof_2019_LA"
CACHE_DIR       = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH   = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
N_SAMPLES      = 100   # total holdout examples (50 normal + 50 deepfake)
BATCH_SIZE     = 10
SEED           = 42


# ---------------------------------------------------------------------------
# Audio helpers  (identical to test_model.py)
# ---------------------------------------------------------------------------

def _decode_audio_entry(entry: dict) -> torch.Tensor:
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
        wave = wave.unsqueeze(0)
    elif wave.ndim == 2:
        wave = wave.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wave = T.Resample(sr, TARGET_SR)(wave)
    return wave


def _crop_center(wave: torch.Tensor) -> torch.Tensor:
    n = wave.shape[-1]
    if n < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // n)
        wave = wave.repeat(1, reps)
    start = (wave.shape[-1] - TARGET_SAMPLES) // 2
    return wave[:, start : start + TARGET_SAMPLES]


def _label_to_int(raw) -> int:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "bonafide", "real", "genuine"):
            return 0
        return 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SoundfileHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, limit, token, seed=42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))

        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )

        labels = [_label_to_int(self.ds[i][self.label_key]) for i in range(len(self.ds))]

        idx0 = [i for i, y in enumerate(labels) if y == 0]
        idx1 = [i for i, y in enumerate(labels) if y == 1]
        rng  = random.Random(seed)
        rng.shuffle(idx0); rng.shuffle(idx1)
        k = min(len(idx0), len(idx1), limit // 2)

        self.indices = idx0[:k] + idx1[:k]
        rng.shuffle(self.indices)

        counts = Counter(_label_to_int(self.ds[i][self.label_key]) for i in self.indices)
        print(f"Holdout '{split}': {len(self.indices)} examples  "
              f"({counts[0]} normal  +  {counts[1]} deepfake)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex   = self.ds[self.indices[idx]]
        wave = _decode_audio_entry(ex["audio"])
        wave = _crop_center(wave)
        y    = _label_to_int(ex[self.label_key])
        return {"audio": wave, "label": torch.tensor(y, dtype=torch.long), "sample_rate": TARGET_SR}


# ---------------------------------------------------------------------------
# Phoneme-loader patch
# ---------------------------------------------------------------------------

def patch_phoneme_loader() -> None:
    import phoneme_GAT.modules as modules_mod
    import phoneme_GAT.phoneme_model as pm_mod
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = (
            "microsoft/wavlm-base" if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h"
        )
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None,
                total_num_phonemes=total_num_phonemes, weights_only=False,
            ).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)

    pm_mod.load_phoneme_model     = _load
    modules_mod.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# PCA plot helper
# ---------------------------------------------------------------------------

def pca_plot(embeddings: np.ndarray, labels: np.ndarray,
             title: str, out_path: Path) -> None:
    """
    Project embeddings to 2-D with PCA and save a scatter plot.
    Red = deepfake (label 1), light blue = normal (label 0).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca  = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(embeddings)        # (N, 2)
    var  = pca.explained_variance_ratio_

    mask0 = labels == 0
    mask1 = labels == 1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj[mask0, 0], proj[mask0, 1],
               c="#6EB5FF", edgecolors="none", s=60, alpha=0.85, label="Normal (bonafide)")
    ax.scatter(proj[mask1, 0], proj[mask1, 1],
               c="#E03030", edgecolors="none", s=60, alpha=0.85, label="Deepfake (spoof)")

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)")
    ax.set_title(title, fontsize=13)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}  (PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SoundfileHFDataset(
        hf_name=HF_DATASET, split="validation",
        cache_dir=str(CACHE_DIR), limit=N_SAMPLES,
        token=hf_token, seed=SEED,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=True, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True,
    ))

    print(f"\nLoading checkpoint: {DEFAULT_CKPT}")
    model = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location="cpu", strict=True,
    )
    model.eval()
    model.freeze()
    device = torch.device("cpu")
    model.to(device)
    print("Model ready.\n")

    # ── Collect embeddings at three stages ───────────────────────────────────
    frozen_enc_embs   = []   # mean-pooled frozen WavLM encoder output
    trainable_enc_embs = []  # mean-pooled trainable encoder output (pre-GAT)
    gat_embs          = []   # post-GAT + LSTM + mean pool (final embedding)
    all_labels        = []

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(device)
            labels = batch["label"]
            B      = labels.shape[0]
            num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)

            out = model.model(audio, num_frames, profiler=None, use_aug=False, stage="eval")

            # phoneme_feat: (B, T, 768)  — frozen encoder (CTC head reads from here)
            frozen_enc_embs.append(out["phoneme_feat"].mean(dim=1).cpu())

            # encoder_feat: (B, T, 768)  — trainable encoder, before GAT
            trainable_enc_embs.append(out["encoder_feat"].mean(dim=1).cpu())

            # hidden_states: (B, 768)    — after GAT + LSTM + pool + L2-norm
            gat_embs.append(out["hidden_states"].cpu())

            all_labels.append(labels)

    frozen_enc_embs    = torch.cat(frozen_enc_embs).numpy()
    trainable_enc_embs = torch.cat(trainable_enc_embs).numpy()
    gat_embs           = torch.cat(gat_embs).numpy()
    labels_np          = torch.cat(all_labels).numpy()

    n_normal   = int((labels_np == 0).sum())
    n_deepfake = int((labels_np == 1).sum())
    print(f"Embeddings collected: {len(labels_np)} total  ({n_normal} normal, {n_deepfake} deepfake)\n")

    # ── PCA plots ─────────────────────────────────────────────────────────────
    pca_plot(
        frozen_enc_embs, labels_np,
        title="Frozen phoneme encoder  (mean-pooled over time)",
        out_path=EXPERIMENTS_DIR / "pca_frozen_encoder.png",
    )
    pca_plot(
        trainable_enc_embs, labels_np,
        title="Trainable encoder output  (pre-GAT, mean-pooled)",
        out_path=EXPERIMENTS_DIR / "pca_pre_gat.png",
    )
    pca_plot(
        gat_embs, labels_np,
        title="Post-GAT + LSTM + pool  (classification embedding)",
        out_path=EXPERIMENTS_DIR / "pca_post_gat.png",
    )

    print("\nDone. All plots saved to experiments/")


if __name__ == "__main__":
    main()
