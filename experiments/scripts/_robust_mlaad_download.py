#!/usr/bin/env python3
"""Robust, resumable snapshot download of the correct MLAAD-tiny revision.

The stock prepare_mlaad_tiny.py download wrapper gives up after 3 quick tries
and silently proceeds with whatever partial files landed. Under HF 429
(Xet-token) throttling that yielded a German-only partial set. This script
retries snapshot_download (which resumes) with exponential backoff until the
full en+de fake + original(bonafide) tree is present.
"""
import os, time, sys
from pathlib import Path

REPO_ID = "mueller91/MLAAD-tiny"
REVISION = "4130a5b9955d86617e5e83c34a8057aa7efb6857"
CACHE_DIR = "/mnt/sagemaker-nvme/mlaad/mlaad_tiny_raw/snapshot"
TARGET_WAVS = 15290

# Dodge the Xet-token endpoint that was returning 429.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
tok = None
sp = Path("secret.txt")
if sp.exists():
    t = sp.read_text().strip()
    if t.startswith("hf_"):
        tok = t
tok = tok or os.environ.get("HF_TOKEN")

from huggingface_hub import snapshot_download

def wav_count():
    return sum(1 for _ in Path(CACHE_DIR).rglob("*.wav")) if Path(CACHE_DIR).exists() else 0

for attempt in range(1, 41):
    n = wav_count()
    print(f"[attempt {attempt}] current wavs={n}/{TARGET_WAVS}", flush=True)
    if n >= TARGET_WAVS:
        print("COMPLETE", flush=True)
        break
    try:
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
            cache_dir=CACHE_DIR, token=tok,
            ignore_patterns=["*.parquet"],
            max_workers=4,
        )
        print("snapshot_download returned OK", flush=True)
    except Exception as e:
        wait = min(60, 5 * attempt)
        print(f"  attempt {attempt} err: {repr(e)[:160]} -> sleep {wait}s", flush=True)
        time.sleep(wait)

n = wav_count()
print(f"FINAL wavs={n}/{TARGET_WAVS} -> {'OK' if n>=TARGET_WAVS else 'INCOMPLETE'}", flush=True)
sys.exit(0 if n >= TARGET_WAVS else 1)
