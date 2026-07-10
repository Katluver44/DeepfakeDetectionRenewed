#!/usr/bin/env python
"""
Generate synthetic laughter clips using AudioLDM2 (cvssp/audioldm2), a
text-to-audio diffusion model. This is a DISTINCT synthesis method from
Bark, Parler-TTS, and Coqui-XTTS: it's a latent-diffusion audio generator
conditioned on GPT-2/CLAP text embeddings, not an autoregressive TTS model.

Output: 16kHz mono wav clips + manifest.csv under data/laugh_oss/audioldm2/
"""
import os
import csv
import time
import argparse

import numpy as np
import torch
import soundfile as sf
import librosa

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "laugh_oss", "audioldm2")
OUT_DIR = os.path.abspath(OUT_DIR)
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")
SR = 16000

PROMPTS = [
    "a person laughing",
    "hearty human laughter",
    "giggling and laughing",
    "a woman laughing out loud",
    "a man laughing heartily",
    "a group of people laughing together",
    "a child giggling and laughing",
    "loud boisterous laughter",
    "a person chuckling softly",
    "someone bursting into laughter",
]

NEG_PROMPT = "low quality, noise, silence, music"


def build_pipe():
    from diffusers import AudioLDM2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2", torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
    except torch.cuda.OutOfMemoryError:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
        pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float32)
        pipe = pipe.to(device)
    return pipe, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--max_clips", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pipe, device = build_pipe()
    print(f"[audioldm2] running on {device}")

    rows = []
    clip_idx = 0
    t0 = time.time()
    for p_idx, prompt in enumerate(PROMPTS):
        for rep in range(args.n_per_prompt):
            if clip_idx >= args.max_clips:
                break
            seed = 1000 * p_idx + rep
            generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
            try:
                audio = pipe(
                    prompt=prompt,
                    negative_prompt=NEG_PROMPT,
                    num_inference_steps=args.steps,
                    audio_length_in_s=args.dur,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                ).audios[0]
            except torch.cuda.OutOfMemoryError:
                print("CUDA OOM -> falling back to CPU")
                torch.cuda.empty_cache()
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                pipe = pipe.to("cpu")
                device = "cpu"
                generator = torch.Generator(device="cpu").manual_seed(seed)
                audio = pipe(
                    prompt=prompt,
                    negative_prompt=NEG_PROMPT,
                    num_inference_steps=args.steps,
                    audio_length_in_s=args.dur,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                ).audios[0]

            audio = np.asarray(audio, dtype=np.float32)
            # AudioLDM2 vocoder outputs at 16kHz already, but resample defensively
            native_sr = 16000
            if native_sr != SR:
                audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SR)
            peak = np.max(np.abs(audio)) + 1e-9
            audio = audio / peak * 0.9

            fname = f"audioldm2_{clip_idx:03d}_p{p_idx}_r{rep}.wav"
            fpath = os.path.join(OUT_DIR, fname)
            sf.write(fpath, audio, SR, subtype="PCM_16")

            dur_s = len(audio) / SR
            rows.append({
                "file": fname,
                "source": "cvssp/audioldm2",
                "method": "audioldm2_diffusion_true_audio",
                "speaker_or_voice": f"seed{seed}",
                "prompt": prompt,
                "dur_s": round(dur_s, 3),
                "sr": SR,
            })
            clip_idx += 1
            print(f"[{clip_idx}/{args.max_clips}] {fname}  prompt='{prompt}'  t={time.time()-t0:.1f}s")
        if clip_idx >= args.max_clips:
            break

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "method", "speaker_or_voice", "prompt", "dur_s", "sr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} clips to {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
