"""
Generate synthetic laughter clips using Parler-TTS (parler-tts/parler-tts-mini-v1),
as a second synthesis method besides Bark, for the laughter research dataset.

Outputs 16kHz mono wav clips into data/laugh_oss/parler_tts/ with a manifest.csv.

Usage:
    source venv/bin/activate
    python scripts/generate_laugh_oss.py
"""
import csv
import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from parler_tts import ParlerTTSForConditionalGeneration

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = "/home/ubuntu/DeepfakeDetectionRenewed/laughsmi/data/laugh_oss/parler_tts"
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")
MODEL_ID = "parler-tts/parler-tts-mini-v1"
TARGET_SR = 16000

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...", flush=True)
t0 = time.time()
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

MODEL_SR = model.config.sampling_rate

# Prompts: laughter-style transcripts + descriptions that push for laughing delivery
descriptions = [
    "A woman laughing warmly and heartily, with a clear, close-sounding, expressive voice, bursting into laughter.",
    "A man bursting into loud hearty laughter, clear audio, very expressive and energetic delivery.",
    "A woman giggling and laughing uncontrollably, close recording, animated and joyful tone.",
    "A man chuckling and laughing, warm expressive voice, high quality close-up recording.",
]

texts = [
    "Hahaha! Ha ha ha ha ha! Hahahaha!",
    "Bwahaha! Haha, ha ha ha ha! Hahaha, stop, hahaha!",
    "Hehehe, haha! Ha ha ha ha ha ha!",
    "Hahaha ha ha! Oh man, hahaha, ha ha!",
]

rows = []
clip_idx = 0
N_PER_COMBO = 3  # 4 descriptions x 4 texts won't all be used; we sample combos

combos = []
for i in range(len(descriptions)):
    for j in range(len(texts)):
        combos.append((descriptions[i], texts[j]))

# We want ~24-32 clips total, generate a few seeds per combo
import itertools
random_seeds = [0, 1]

t_start = time.time()
for (desc, text) in combos:
    for seed in random_seeds:
        torch.manual_seed(seed)
        input_ids = desc_tokenizer(desc, return_tensors="pt").input_ids.to(DEVICE)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

        gen_t0 = time.time()
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )
        gen_dt = time.time() - gen_t0

        audio = generation.cpu().numpy().squeeze()
        dur_s_native = len(audio) / MODEL_SR

        # resample to 16k
        if MODEL_SR != TARGET_SR:
            import librosa
            audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=MODEL_SR, target_sr=TARGET_SR)
        else:
            audio_16k = audio.astype(np.float32)

        dur_s = len(audio_16k) / TARGET_SR

        fname = f"parler_{clip_idx:03d}.wav"
        fpath = os.path.join(OUT_DIR, fname)
        sf.write(fpath, audio_16k, TARGET_SR)

        rows.append({
            "file": fname,
            "source": "parler-tts/parler-tts-mini-v1",
            "method": "parler_tts",
            "speaker_or_voice": f"desc_seed{seed}",
            "prompt": f"DESC: {desc} | TEXT: {text}",
            "dur_s": f"{dur_s:.3f}",
            "sr": TARGET_SR,
        })
        print(f"[{clip_idx}] gen_time={gen_dt:.2f}s dur={dur_s:.2f}s -> {fname}", flush=True)
        clip_idx += 1

        if clip_idx >= 32:
            break
    if clip_idx >= 32:
        break

total_dt = time.time() - t_start
print(f"Generated {clip_idx} clips in {total_dt:.1f}s ({total_dt/max(clip_idx,1):.2f}s/clip)", flush=True)

with open(MANIFEST, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "source", "method", "speaker_or_voice", "prompt", "dur_s", "sr"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Manifest written to {MANIFEST}")
