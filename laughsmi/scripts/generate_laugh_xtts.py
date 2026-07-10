#!/usr/bin/env python
"""
Generate synthetic laughter clips using Coqui XTTS-v2 (reference/voice-conditioned TTS).

Method: pick REAL laughter clips from VocalSound as speaker-reference wavs, feed
XTTS-v2 laughter-like text prompts ("Hahaha! Ha ha ha, haha!", etc.), and save the
synthesized audio as candidate synthetic-laughter clips. Output resampled to 16kHz mono.

Usage:
    source venv_xtts/bin/activate
    python scripts/generate_laugh_xtts.py
"""
import csv
import os
import random
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import librosa

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(REPO_ROOT, "data", "vocalsound", "audio_16k_raw", "subset1")
OUT_DIR = os.path.join(REPO_ROOT, "data", "laugh_oss", "xtts")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")

SR_OUT = 16000

PROMPTS = [
    "Hahaha! Ha ha ha, haha!",
    "Ahahaha, hahaha!",
    "Hehe, haha, ho ho ho!",
    "Hahaha, that is so funny, hahaha!",
    "Ha ha ha ha ha!",
    "Hehehe, hahaha, ahaha!",
    "Ho ho ho, hahaha!",
    "Haha, hahaha, hehehe!",
]

random.seed(1234)


def pick_reference_wavs(n):
    files = sorted(f for f in os.listdir(REF_DIR) if f.endswith(".wav"))
    random.shuffle(files)
    return files[:n]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from TTS.api import TTS

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    print(f"Loading XTTS-v2 on device={device} ...")
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_clips_target = 30
    n_refs = 10
    ref_files = pick_reference_wavs(n_refs)
    print("Reference clips:", ref_files)

    rows = []
    idx = 0
    attempts = 0
    max_attempts = n_clips_target * 3

    while len(rows) < n_clips_target and attempts < max_attempts:
        attempts += 1
        ref_file = ref_files[idx % len(ref_files)]
        prompt = PROMPTS[idx % len(PROMPTS)]
        idx += 1

        ref_path = os.path.join(REF_DIR, ref_file)
        out_name = f"xtts_laugh_{idx:03d}.wav"
        out_path = os.path.join(OUT_DIR, out_name)

        try:
            wav = tts.tts(
                text=prompt,
                speaker_wav=ref_path,
                language="en",
            )
            wav = np.asarray(wav, dtype=np.float32)

            # XTTS outputs at 24kHz internally; resample to 16kHz mono for output.
            native_sr = getattr(tts.synthesizer, "output_sample_rate", 24000)
            if native_sr != SR_OUT:
                wav16 = librosa.resample(wav, orig_sr=native_sr, target_sr=SR_OUT)
            else:
                wav16 = wav

            sf.write(out_path, wav16, SR_OUT, subtype="PCM_16")
            dur_s = len(wav16) / SR_OUT

            rows.append({
                "file": os.path.join("data", "laugh_oss", "xtts", out_name),
                "source": "xtts",
                "method": "xtts_v2_refcloned_laughter",
                "speaker_or_voice": ref_file,
                "prompt": prompt,
                "dur_s": f"{dur_s:.3f}",
                "sr": SR_OUT,
            })
            print(f"[{len(rows)}/{n_clips_target}] wrote {out_name} "
                  f"(ref={ref_file}, dur={dur_s:.2f}s)")

        except Exception as e:
            print(f"FAILED on ref={ref_file}, prompt={prompt!r}: {e}")
            traceback.print_exc()
            # If CUDA OOM, bail out to let caller retry on CPU.
            if "CUDA out of memory" in str(e) or "cuda" in str(e).lower() and "memory" in str(e).lower():
                print("Detected possible CUDA OOM; stopping generation loop.")
                break

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "source", "method", "speaker_or_voice", "prompt", "dur_s", "sr"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone. Wrote {len(rows)} clips to {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
