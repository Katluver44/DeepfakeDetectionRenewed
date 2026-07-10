#!/usr/bin/env python
"""Generate a synthetic-laughter probe set with Bark (suno-ai/bark).

Implements laughsmi_plan.md §3.4 (Decision D2 / H4 probe).

Must be run inside the dedicated `venv_bark` environment (NOT the main
`venv`), with SUNO_USE_SMALL_MODELS=1 set to keep VRAM usage modest since
the GPU is shared with another job:

    source venv_bark/bin/activate
    export SUNO_USE_SMALL_MODELS=1
    python scripts/generate_bark_probe.py

Generates 3 conditions x 20 prompts x 2 speaker presets = 120 clips:
  (a) speech_only   -- 20 short natural conversational sentences
  (b) speech_laugh  -- same 20 sentences with a "[laughs]" token inserted
                       mid-utterance
  (c) laughter_only -- laughter-token-only prompts (variants of
                       "[laughter]" / "[laughs] [laughs]")

Output layout:
  data/bark_probe/{condition}/{speaker}_{idx:03d}.wav   (16kHz mono PCM16)
  data/bark_probe/manifest.csv                          (file,condition,speaker_preset,prompt)

If a CUDA OOM is hit, the script automatically falls back to CPU generation
for the remainder of the run (slower, but acceptable for ~100 short clips).
"""

import csv
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "bark_probe"

SPEAKER_PRESETS = ["v2/en_speaker_6", "v2/en_speaker_9"]

# ---------------------------------------------------------------------------
# 20 short, varied, conversational-register sentences.
# ---------------------------------------------------------------------------
SENTENCES = [
    "I can't believe you actually did that.",
    "Honestly, I think we should just order pizza tonight.",
    "Wait, did you hear what she just said?",
    "I'm pretty sure we took a wrong turn back there.",
    "You know, I never thought I'd say this, but I miss the office.",
    "So I opened the fridge and there was absolutely nothing in it.",
    "He told me the meeting got moved to Friday.",
    "I spent the whole afternoon trying to fix that leaky faucet.",
    "Can you believe it's already almost summer?",
    "She walked in wearing the most ridiculous hat I've ever seen.",
    "I think my phone battery dies faster every single day.",
    "We should really plan that road trip we keep talking about.",
    "I forgot my umbrella again, of course it started raining.",
    "The new coffee place downtown is surprisingly good.",
    "I swear the dog understood every word I said.",
    "They asked me to give a speech and I had zero time to prepare.",
    "I just realized I've been wearing mismatched socks all day.",
    "It took three tries to parallel park in that tiny spot.",
    "My neighbor's cat keeps showing up on my porch at midnight.",
    "I finally finished that book you recommended last month.",
]

assert len(SENTENCES) == 20

# ---------------------------------------------------------------------------
# Condition (b): same sentences with "[laughs]" inserted mid-utterance.
# Insert after roughly the midpoint clause / punctuation for naturalness.
# ---------------------------------------------------------------------------
def insert_laugh_midutterance(sentence: str) -> str:
    """Insert a [laughs] token near the middle of the sentence.

    Splits on whitespace and inserts the tag after the middle word (or after
    a comma if one exists near the middle), so it reads as a laugh occurring
    mid-speech rather than at an edge.
    """
    # Prefer splitting at a comma if present.
    if "," in sentence:
        idx = sentence.index(",")
        before = sentence[: idx + 1]
        after = sentence[idx + 1 :].strip()
        return f"{before} [laughs] {after}"
    words = sentence.rstrip(".?!").split(" ")
    mid = len(words) // 2
    before = " ".join(words[:mid])
    after = " ".join(words[mid:])
    trailing_punct = sentence[-1] if sentence[-1] in ".?!" else "."
    return f"{before} [laughs] {after}{trailing_punct}"


SPEECH_LAUGH_PROMPTS = [insert_laugh_midutterance(s) for s in SENTENCES]

# ---------------------------------------------------------------------------
# Condition (c): laughter-only prompts. 20 variants cycling through a small
# set of laughter-token patterns bark recognizes.
# ---------------------------------------------------------------------------
LAUGHTER_TOKEN_VARIANTS = [
    "[laughter]",
    "[laughs]",
    "[laughs] [laughs]",
    "[laughter] [laughter]",
    "Haha [laughs]",
    "[laughs] Hahaha!",
    "[laughter] Oh my gosh [laughs]",
    "Hahahaha [laughter]",
]


def build_laughter_only_prompts(n=20):
    prompts = []
    for i in range(n):
        prompts.append(LAUGHTER_TOKEN_VARIANTS[i % len(LAUGHTER_TOKEN_VARIANTS)])
    return prompts


LAUGHTER_ONLY_PROMPTS = build_laughter_only_prompts(20)

CONDITIONS = {
    "speech_only": SENTENCES,
    "speech_laugh": SPEECH_LAUGH_PROMPTS,
    "laughter_only": LAUGHTER_ONLY_PROMPTS,
}


def resample_to_16k_mono(audio, orig_sr, target_sr=16000):
    """Resample a 1-D float32 array from orig_sr to target_sr using scipy."""
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def write_wav_16k_mono_pcm16(path, audio_f32, sr=16000):
    import numpy as np
    import soundfile as sf

    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    sf.write(str(path), audio_f32, sr, subtype="PCM_16")


def main():
    if "bark" not in sys.modules:
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
        except ImportError as e:
            print(
                "ERROR: could not import bark. This script must be run "
                "inside venv_bark (pip install git+https://github.com/suno-ai/bark.git).",
                file=sys.stderr,
            )
            raise

    import torch

    if os.environ.get("SUNO_USE_SMALL_MODELS") != "1":
        print(
            "WARNING: SUNO_USE_SMALL_MODELS is not set to 1. Setting it now "
            "for this process (small models keep VRAM usage modest).",
        )
        os.environ["SUNO_USE_SMALL_MODELS"] = "1"

    for cond in CONDITIONS:
        (OUT_DIR / cond).mkdir(parents=True, exist_ok=True)

    print("Preloading Bark models (small)...")
    t0 = time.time()
    preload_models()
    print(f"Preload took {time.time() - t0:.1f}s")

    device_mode = "gpu" if torch.cuda.is_available() else "cpu"

    manifest_rows = []
    durations = []
    per_clip_times = []
    total_clips = sum(len(v) for v in CONDITIONS.values()) * len(SPEAKER_PRESETS)
    clip_i = 0
    t_start_all = time.time()

    for cond, prompts in CONDITIONS.items():
        for speaker in SPEAKER_PRESETS:
            for idx, prompt in enumerate(prompts):
                clip_i += 1
                speaker_tag = speaker.split("/")[-1]
                out_name = f"{speaker_tag}_{idx:03d}.wav"
                out_path = OUT_DIR / cond / out_name

                t0 = time.time()
                try:
                    audio = generate_audio(prompt, history_prompt=speaker)
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[{clip_i}/{total_clips}] CUDA OOM on {cond}/{out_name}; "
                        "falling back to CPU for remainder of run.",
                    )
                    torch.cuda.empty_cache()
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    device_mode = "cpu"
                    # bark reads device lazily via generation module's global;
                    # simplest robust fallback: force torch to report no cuda
                    # by monkeypatching is_available.
                    torch.cuda.is_available = lambda: False  # noqa: E731
                    audio = generate_audio(prompt, history_prompt=speaker)
                except Exception:
                    print(f"FAILED on {cond}/{out_name} prompt={prompt!r}")
                    traceback.print_exc()
                    continue
                gen_time = time.time() - t0
                per_clip_times.append(gen_time)

                from bark import SAMPLE_RATE

                audio_16k = resample_to_16k_mono(audio, SAMPLE_RATE, 16000)
                write_wav_16k_mono_pcm16(out_path, audio_16k, 16000)
                dur_s = len(audio_16k) / 16000.0
                durations.append(dur_s)

                manifest_rows.append(
                    {
                        "file": str(out_path.relative_to(REPO_ROOT)),
                        "condition": cond,
                        "speaker_preset": speaker,
                        "prompt": prompt,
                    }
                )
                print(
                    f"[{clip_i}/{total_clips}] {cond}/{out_name} "
                    f"dur={dur_s:.2f}s gen_time={gen_time:.2f}s device={device_mode}"
                )

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "condition", "speaker_preset", "prompt"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_time = time.time() - t_start_all
    print("\n=== Summary ===")
    print(f"Total clips generated: {len(manifest_rows)} / {total_clips}")
    print(f"Total wall-clock (generation loop): {total_time:.1f}s")
    if per_clip_times:
        print(
            f"Per-clip gen time: mean={sum(per_clip_times) / len(per_clip_times):.2f}s "
            f"min={min(per_clip_times):.2f}s max={max(per_clip_times):.2f}s"
        )
    if durations:
        print(
            f"Clip durations: mean={sum(durations) / len(durations):.2f}s "
            f"min={min(durations):.2f}s max={max(durations):.2f}s"
        )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
