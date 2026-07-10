#!/usr/bin/env python
"""D1.3 / D2 — Generate a bank of Bark synthetic *laughter-only* clips across
many speaker presets, for (a) splice-in inserts in D1 augmentation and (b) the
D3 real-vs-synthetic laughter SSL analysis + the D2 dataset slice.

Run in venv_bark with SUNO_USE_SMALL_MODELS=1:
    source venv_bark/bin/activate && export SUNO_USE_SMALL_MODELS=1
    python scripts/generate_laugh_bank.py --n-per-preset 6

Output:
    data/laugh_bank_bark/{preset}_{idx:03d}.wav   (16kHz mono PCM16)
    data/laugh_bank_bark/manifest.csv             (file,source,method,speaker_preset,prompt,dur_s,sr)
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "laugh_bank_bark"

# Broader speaker variety than the original probe (10 presets, mixed gender/lang-accent).
PRESETS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_3", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_9",
    "v2/de_speaker_3", "v2/fr_speaker_1", "v2/es_speaker_8",
]

LAUGHTER_VARIANTS = [
    "[laughter]",
    "[laughs]",
    "[laughs] [laughs]",
    "[laughter] [laughter]",
    "Haha [laughs]",
    "[laughs] hahaha!",
]


def resample_to_16k_mono(audio, orig_sr, target_sr=16000):
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-preset", type=int, default=6)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from bark import SAMPLE_RATE, generate_audio, preload_models

    os.environ.setdefault("SUNO_USE_SMALL_MODELS", "1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Preloading Bark (small)...")
    preload_models()

    rows = []
    device_mode = "gpu" if torch.cuda.is_available() else "cpu"
    total = len(PRESETS) * args.n_per_preset
    ci = 0
    t0all = time.time()
    for preset in PRESETS:
        tag = preset.split("/")[-1]
        for idx in range(args.n_per_preset):
            ci += 1
            prompt = LAUGHTER_VARIANTS[idx % len(LAUGHTER_VARIANTS)]
            out_path = OUT_DIR / f"{tag}_{idx:03d}.wav"
            if out_path.exists():
                continue
            t0 = time.time()
            try:
                audio = generate_audio(prompt, history_prompt=preset)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                torch.cuda.is_available = lambda: False  # noqa: E731
                device_mode = "cpu"
                audio = generate_audio(prompt, history_prompt=preset)
            except Exception:
                print(f"FAILED {out_path.name} prompt={prompt!r}")
                traceback.print_exc()
                continue
            a16 = resample_to_16k_mono(audio, SAMPLE_RATE, 16000)
            a16 = np.clip(a16, -1.0, 1.0)
            sf.write(str(out_path), a16, 16000, subtype="PCM_16")
            dur = len(a16) / 16000.0
            rows.append({"file": str(out_path.relative_to(REPO_ROOT)), "source": "bark",
                         "method": "bark_laughter_token", "speaker_preset": preset,
                         "prompt": prompt, "dur_s": round(dur, 3), "sr": 16000})
            print(f"[{ci}/{total}] {out_path.name} dur={dur:.2f}s gen={time.time()-t0:.2f}s {device_mode}")

    mpath = OUT_DIR / "manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "method", "speaker_preset", "prompt", "dur_s", "sr"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} clips in {time.time()-t0all:.0f}s -> {mpath}")


if __name__ == "__main__":
    main()
