#!/usr/bin/env python3
"""
e9_precheck.py — Pre-training checks (b) and (c) for E9.

(b) Phoneme count distribution: compare new training data vs ASVspoof 2019 LA.
(c) Pipeline sanity check: one batch through Phoneme_GAT forward pass.

Usage (run after WaveFake is extracted):
  venv/bin/python3 experiments/e9_precheck.py

Outputs to experiments/results/e9_mixed_training/precheck_*.
"""
from __future__ import annotations
import os, random, sys, json, time, warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "experiments" / "results" / "e9_mixed_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECRET_PATH = REPO_ROOT / "secret.txt"
TOKEN = SECRET_PATH.read_text().strip() if SECRET_PATH.exists() else None

N_SAMPLE = 200   # files per source for distribution check
SEED     = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("medium")

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

import torchaudio
from loader import _crop_policy, _ensure_sr, TARGET_SR, TARGET_SAMPLES


# ── Phoneme model loader (same patch as e9_train.py) ────────────────────────

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


def get_phoneme_count(wav_1d: torch.Tensor, phoneme_model, transformer) -> int:
    """
    Returns the number of unique phonemes after CTC collapse (reduced_num_frames).
    wav_1d: (L,) tensor at 16kHz.
    """
    with torch.no_grad():
        x = wav_1d.unsqueeze(0).cuda()   # (1, L)
        feat1 = transformer.feature_extractor(x).transpose(1, 2)
        hidden, _ = transformer.feature_projection(feat1)
        phoneme_feat = transformer.encoder(hidden)[0]
        logits = phoneme_model.model.model.lm_head(phoneme_feat)  # (1, T, V)
        ids    = torch.argmax(logits, dim=-1)[0]  # (T,)
        # collapse consecutive identical phonemes (CTC blank is id=0)
        uniq = ids.unique_consecutive()
        uniq = uniq[uniq != 0]  # remove blanks
    return uniq.numel()


# ── (b) Phoneme distribution check ──────────────────────────────────────────

def sample_files_from_source(source_name: str, n: int) -> list[str]:
    """Return n audio file paths for the given source."""
    if source_name == "asvspoof":
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train",
                          cache_dir=str(REPO_ROOT / "data" / "asvspoof_2019_la"),
                          token=TOKEN)
        ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
        idx = random.sample(range(len(ds)), n)
        return [("__hf__", i, ds) for i in idx]

    if source_name == "vcc2020":
        import glob
        files = glob.glob(str(REPO_ROOT / "data/vcc2020/audio/*/task1/*.wav"))
        return random.sample(files, min(n, len(files)))

    if source_name == "wavefake":
        import glob
        root = REPO_ROOT / "data/wavefake/generated_audio"
        files = glob.glob(str(root / "**/*.wav"), recursive=True)
        if not files:
            print("  [WARN] WaveFake not yet extracted — skipping")
            return []
        return random.sample(files, min(n, len(files)))

    if source_name == "librispeech":
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100",
                          cache_dir=str(REPO_ROOT / "data" / "librispeech"),
                          token=TOKEN)
        ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
        idx = random.sample(range(len(ds)), min(n, len(ds)))
        return [("__hf__", i, ds) for i in idx]

    raise ValueError(f"Unknown source: {source_name}")


def load_wave(item) -> torch.Tensor:
    """Load waveform and return (L,) tensor at 16kHz, 3s crop."""
    if isinstance(item, tuple) and item[0] == "__hf__":
        _, idx, ds = item
        ex  = ds[idx]
        arr = ex["audio"]["array"]
        wav = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, L)
    else:
        wav, sr = torchaudio.load(item)
        wav = _ensure_sr(wav, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
    wav = _crop_policy(wav, "eval")   # center 3s
    return wav[0]  # (L,)


def check_phoneme_distributions():
    print("\n" + "="*70)
    print("(b) Phoneme count distribution check")
    print("="*70)

    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True, n_edges=10,
        use_aug=False, use_pool=True, use_clip=False))
    model_wrapper = Phoneme_GAT(backbone="wavlm", use_raw=False, use_GAT=False, n_edges=10)
    model_wrapper = model_wrapper.cuda().eval()
    phoneme_model   = model_wrapper.phoneme_model
    transformer_blk = model_wrapper.transformer_in_phoneme_model

    sources = {
        "asvspoof":   "ASVspoof 2019 LA (reference)",
        "vcc2020":    "VCC2020 Task 1",
        "wavefake":   "WaveFake",
        "librispeech":"LibriSpeech train-100",
    }

    stats = {}
    for src_key, src_label in sources.items():
        items = sample_files_from_source(src_key, N_SAMPLE)
        if not items:
            stats[src_key] = {"counts": [], "label": src_label}
            continue

        counts = []
        for it in items:
            try:
                wav = load_wave(it).cuda()
                c   = get_phoneme_count(wav, phoneme_model, transformer_blk)
                counts.append(c)
            except Exception as e:
                pass

        arr = np.array(counts)
        print(f"\n{src_label}:")
        print(f"  n={len(arr)}  mean={arr.mean():.1f}  std={arr.std():.1f}  "
              f"min={arr.min()}  p25={np.percentile(arr,25):.0f}  "
              f"median={np.median(arr):.0f}  p75={np.percentile(arr,75):.0f}  "
              f"max={arr.max()}")
        stats[src_key] = {"counts": arr.tolist(), "label": src_label,
                          "mean": float(arr.mean()), "std": float(arr.std()),
                          "median": float(np.median(arr)),
                          "p5":  float(np.percentile(arr, 5)),
                          "p95": float(np.percentile(arr, 95))}

    # Save stats JSON
    out_path = OUT_DIR / "precheck_phoneme_dist.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"asvspoof": "#1f77b4", "vcc2020": "#ff7f0e",
              "wavefake": "#2ca02c", "librispeech": "#d62728"}
    for src_key, d in stats.items():
        if d["counts"]:
            ax.hist(d["counts"], bins=30, alpha=0.5, label=d["label"],
                    color=colors.get(src_key, "gray"), density=True)
    ax.set_xlabel("Phoneme count per sample (after CTC collapse, 3s crop)")
    ax.set_ylabel("Density")
    ax.set_title("E9 Pre-check: Phoneme count distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = OUT_DIR / "precheck_phoneme_dist.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")

    # Compatibility flag
    asv_med = stats.get("asvspoof", {}).get("median", 0)
    issues = []
    for src_key in ["vcc2020", "wavefake", "librispeech"]:
        d = stats.get(src_key, {})
        if not d.get("counts"):
            continue
        ratio = d["median"] / asv_med if asv_med > 0 else 0
        if ratio < 0.5 or ratio > 2.0:
            issues.append(f"{d['label']}: median={d['median']:.0f} "
                          f"vs ASVspoof={asv_med:.0f} (ratio={ratio:.2f}) — WARN")
    if issues:
        print("\n[COMPATIBILITY WARNINGS]")
        for w in issues:
            print(" ", w)
    else:
        print("\n[OK] All sources within 2× ASVspoof median phoneme count.")

    return stats


# ── (c) Pipeline sanity check ────────────────────────────────────────────────

def pipeline_sanity_check():
    print("\n" + "="*70)
    print("(c) Pipeline sanity check — one batch through Phoneme_GAT")
    print("="*70)

    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=False, use_pool=True, use_clip=True))
    model = Phoneme_GAT_lit(cfg=cfg).cuda().eval()

    import glob
    # Use VCC2020 (24kHz → should resample) and MLAAD local (16kHz) for sanity
    vcc_files  = glob.glob(str(REPO_ROOT / "data/vcc2020/audio/T01/task1/*.wav"))[:2]
    mlaad_files= glob.glob(str(REPO_ROOT / "data/mlaad_en/_hf_cache/fake/en/**/*.wav"),
                           recursive=True)[:2]
    all_files  = vcc_files + mlaad_files
    if not all_files:
        print("No test files found — check data paths")
        return

    wavs = []
    srs  = []
    for f in all_files:
        w, sr = torchaudio.load(f)
        w = _ensure_sr(w, sr)
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        w = _crop_policy(w, "eval")
        wavs.append(w)
        srs.append(sr)
        print(f"  loaded: {Path(f).name}  orig_sr={sr}  shape_after={w.shape}")

    batch_audio = torch.stack(wavs).cuda()   # (B, 1, 48000)
    B = len(wavs)
    num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1).cuda()

    t0 = time.time()
    with torch.no_grad():
        out = model.model(batch_audio, num_frames, use_aug=False, stage="eval")
    elapsed = time.time() - t0

    logits = out["logit"]
    print(f"\nBatch size: {B}")
    print(f"Logit shape: {logits.shape}   values: {logits.cpu().tolist()}")
    print(f"Forward pass time: {elapsed*1000:.0f}ms")

    assert logits.shape == (B,), f"Expected ({B},) got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    print("\n[OK] Pipeline sanity check passed.")

    out_path = OUT_DIR / "precheck_pipeline.json"
    with open(out_path, "w") as f:
        import json
        json.dump({
            "files": [Path(f).name for f in all_files],
            "orig_sample_rates": srs,
            "logits": logits.cpu().tolist(),
            "elapsed_ms": elapsed * 1000,
        }, f, indent=2)
    print(f"Saved: {out_path}")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline_sanity_check()
    check_phoneme_distributions()
    print("\n=== Pre-training checks complete. See", OUT_DIR, "===")
