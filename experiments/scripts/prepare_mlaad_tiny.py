#!/usr/bin/env python3
"""
Phase 0: Prepare MLAAD-tiny dataset with preprocessed audio and clean splits.

Downloads mueller91/MLAAD-tiny from HuggingFace (resumes partial snapshot),
preprocesses audio to match the WavLM/ASVspoof pipeline (16 kHz mono, 3-s
centre-crop, float32), and creates stratified speaker-disjoint 70/15/15 splits.

Usage:
    python experiments/scripts/prepare_mlaad_tiny.py [--output-dir DIR] [--force]

Outputs:
    experiments/data/mlaad_tiny_processed/
        audio/                      preprocessed .pt tensors (48000,) float32
        splits/
            train.json
            val.json
            test.json
            test_LOCKED.md          SHA-256 of sorted test sample_ids
        dataset_inventory.json
        dataset_inventory.md
        metadata.json
        splits_config.json
        run_config.json
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

REPO_ID = "mueller91/MLAAD-tiny"
# NOTE: revision 9143e5ea (2026-05-27) is a German-only, spoof-only regression of
# this repo and does NOT match the reference manifests (English fake+bonafide).
# 4130a5b9 (2026-02-10) is the en+de, fake+original(bonafide) snapshot whose 64
# English systems + bonafide exactly match experiments/results/mlaad/baseline_eval.
SNAP_REVISION = "4130a5b9955d86617e5e83c34a8057aa7efb6857"
REPO_TYPE = "dataset"

# Relative to project root (CWD when this script is invoked)
RAW_CACHE_DIR = Path("experiments/data/mlaad_tiny_raw/snapshot")
PROCESSED_DIR = Path("experiments/data/mlaad_tiny_processed")

TARGET_SR = 16_000
TARGET_SAMPLES = 3 * TARGET_SR   # 48 000 samples  (3 seconds)
SPLIT_SEED = 42
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 30  # seconds between retries

SCRIPT_VERSION = "1.0.0"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_name(s: str) -> str:
    """Replace characters that are unsafe in filenames / JSON keys."""
    return re.sub(r"[^\w\-.]", "_", s)


def parse_speaker_from_original_file(original_file: str) -> str:
    """
    Extract the M-AILABS speaker (reader) ID from an original_file path.

    Path format: {locale}/by_book/{gender_or_mix}/{speaker_or_book}/{book}/wavs/...
    We use parts[3] as the speaker ID in both 'mix' and normal cases.
    """
    parts = original_file.split("/")
    if len(parts) >= 4:
        return parts[3]
    return "unknown"


def parse_speaker_from_filename(stem: str) -> str:
    """
    Derive a book-level speaker proxy from the M-AILABS filename stem.

    Handles naming variants:
      book_ch_fNNNNNN              → book
      book_ch_subch_fNNNNNN        → book
      book_ch_extra_64kb_fNNNNNN   → book  (e.g. piratesofersatz, rinkitink)
      book_NNN_fNNNNNN             → book  (3-digit chapter)

    Algorithm: strip the utterance suffix (_fNNNNN), then find the first
    _\d+ segment — everything before it is the book/speaker proxy.
    """
    # Strip utterance ID (_f followed by digits) at end
    base = re.sub(r"_f\d+$", "", stem, flags=re.IGNORECASE)
    # Find first _digits segment; everything before it = speaker proxy
    m = re.match(r"^(.+?)_\d+", base)
    if m:
        return m.group(1).lower()
    return base.lower()


# ─── Step 1: Download (resume) ────────────────────────────────────────────────

def download_snapshot() -> Path:
    """
    Download or resume the MLAAD-tiny snapshot into RAW_CACHE_DIR.
    Returns the path to the local snapshot directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed — cannot download. "
                  "Run: pip install huggingface_hub")
        sys.exit(1)

    snap_path = RAW_CACHE_DIR / f"datasets--{REPO_ID.replace('/', '--')}" / \
                "snapshots" / SNAP_REVISION

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            log.info("Download attempt %d/%d …", attempt, MAX_DOWNLOAD_RETRIES)
            result = snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=SNAP_REVISION,
                cache_dir=str(RAW_CACHE_DIR),
                ignore_patterns=["*.parquet"],  # no parquet files expected
            )
            log.info("Snapshot ready at: %s", result)
            return Path(result)
        except Exception as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_DOWNLOAD_RETRIES:
                log.info("Retrying in %d s …", DOWNLOAD_RETRY_DELAY)
                time.sleep(DOWNLOAD_RETRY_DELAY)

    log.warning(
        "All %d download attempts failed. "
        "Proceeding with locally available files in %s",
        MAX_DOWNLOAD_RETRIES, snap_path,
    )
    return snap_path


# ─── Step 2: Load speaker metadata from meta.csv files ────────────────────────

def load_speaker_map(snapshot_dir: Path) -> dict[tuple[str, str, str], str]:
    """
    Read every meta.csv in the snapshot and return a lookup:
        (lang, system, wav_stem) -> speaker_id
    """
    speaker_map: dict[tuple[str, str, str], str] = {}
    for lang in ("en", "de"):
        lang_dir = snapshot_dir / "fake" / lang
        if not lang_dir.is_dir():
            continue
        for sys_dir in lang_dir.iterdir():
            if not sys_dir.is_dir():
                continue
            meta_csv = sys_dir / "meta.csv"
            if not meta_csv.exists():
                continue
            system = sys_dir.name
            with open(meta_csv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="|")
                for row in reader:
                    of = row.get("original_file", "")
                    spk = parse_speaker_from_original_file(of)
                    # path column: ./fake/en/SYSTEM/filename.wav
                    path_col = row.get("path", "")
                    stem = Path(path_col).stem
                    speaker_map[(lang, system, stem)] = spk
    return speaker_map


# ─── Step 3: Discover all WAV files ───────────────────────────────────────────

def discover_samples(
    snapshot_dir: Path,
    speaker_map: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """
    Walk the snapshot and return one record per WAV file.

    Record schema:
        sample_id     str   unique filesystem-safe identifier
        wav_path      str   absolute path to the source WAV
        label         str   "spoof" | "bonafide"
        attack_system str   TTS system name, or "bonafide"
        language      str   "en" | "de" | "unknown"
        speaker_id    str   M-AILABS reader / book proxy
    """
    records: list[dict[str, Any]] = []

    # ── Spoof: fake/{lang}/{system}/*.wav
    fake_dir = snapshot_dir / "fake"
    if fake_dir.is_dir():
        for lang_dir in sorted(fake_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            for sys_dir in sorted(lang_dir.iterdir()):
                if not sys_dir.is_dir():
                    continue
                system = sys_dir.name
                for wav in sorted(sys_dir.glob("*.wav")):
                    stem = wav.stem
                    # Use filename-based speaker ID for consistency with bonafide
                    # (bonafide uses same M-AILABS filenames as spoof source texts)
                    spk = parse_speaker_from_filename(stem)
                    sid = f"fake__{lang}__{safe_name(system)}__{safe_name(stem)}"
                    records.append({
                        "sample_id": sid,
                        "wav_path": str(wav),
                        "label": "spoof",
                        "attack_system": system,
                        "language": lang,
                        "speaker_id": spk,
                    })

    # ── Bonafide: original/{en,de}/*.wav  (may be partially downloaded)
    for bonafide_root_name in ("original", "real", "bonafide"):
        bonafide_dir = snapshot_dir / bonafide_root_name
        if not bonafide_dir.is_dir():
            continue

        # Check for language subdirectories (original/en/, original/de/)
        lang_subdirs = sorted(
            d for d in bonafide_dir.iterdir()
            if d.is_dir() and d.name in ("en", "de")
        )
        if lang_subdirs:
            for lang_dir in lang_subdirs:
                lang = lang_dir.name
                for wav in sorted(lang_dir.glob("*.wav")):
                    stem = wav.stem
                    spk = parse_speaker_from_filename(stem)
                    sid = f"real__{lang}__{safe_name(stem)}"
                    records.append({
                        "sample_id": sid,
                        "wav_path": str(wav),
                        "label": "bonafide",
                        "attack_system": "bonafide",
                        "language": lang,
                        "speaker_id": spk,
                    })
        else:
            # Flat structure — assume English
            for wav in sorted(bonafide_dir.rglob("*.wav")):
                stem = wav.stem
                spk = parse_speaker_from_filename(stem)
                sid = f"real__en__{safe_name(stem)}"
                records.append({
                    "sample_id": sid,
                    "wav_path": str(wav),
                    "label": "bonafide",
                    "attack_system": "bonafide",
                    "language": "en",
                    "speaker_id": spk,
                })

        n_bf = sum(1 for r in records if r["label"] == "bonafide")
        log.info("Found bonafide directory: %s (%d WAVs)", bonafide_dir, n_bf)
        break
    else:
        log.warning(
            "No bonafide directory found in snapshot. "
            "Splits will be spoof-only. Re-run after completing the download."
        )

    log.info("Discovered %d samples total.", len(records))
    return records


# ─── Step 4: Audio preprocessing ──────────────────────────────────────────────

def _decode(wav_path: str) -> torch.Tensor:
    """
    Load a WAV file → float32 mono tensor of shape (1, T).
    Resamples to TARGET_SR if needed (matches head_ablation.py/_decode).
    """
    arr, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    w = torch.from_numpy(arr)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    elif w.ndim == 2:
        w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = T.Resample(sr, TARGET_SR)(w)
    return w                        # (1, T)


def _crop(w: torch.Tensor) -> torch.Tensor:
    """
    Centre-crop or tile-pad to exactly TARGET_SAMPLES.
    Returns (1, TARGET_SAMPLES).
    """
    n = w.shape[-1]
    if n < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // n)   # ceiling division
        w = w.repeat(1, reps)
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s: s + TARGET_SAMPLES]


def preprocess_all(
    records: list[dict[str, Any]],
    out_audio_dir: Path,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Preprocess every WAV and write .pt tensors to out_audio_dir.
    Returns (updated_records, errors).
    """
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    ok_records: list[dict[str, Any]] = []
    n = len(records)
    log_every = max(1, n // 20)

    for i, rec in enumerate(records):
        pt_name = rec["sample_id"] + ".pt"
        pt_path = out_audio_dir / pt_name
        rel_path = f"audio/{pt_name}"

        if pt_path.exists() and not force:
            rec = dict(rec, audio_path=rel_path)
            ok_records.append(rec)
            if i % log_every == 0:
                log.info("[%d/%d] cached: %s", i + 1, n, rec["sample_id"])
            continue

        try:
            tensor = _crop(_decode(rec["wav_path"])).squeeze(0)  # (48000,)
            torch.save(tensor, pt_path)
            rec = dict(rec, audio_path=rel_path)
            ok_records.append(rec)
        except Exception as exc:
            log.warning("Skipping %s — decode error: %s", rec["wav_path"], exc)
            errors.append(f"{rec['sample_id']}: {exc}")
            continue

        if (i + 1) % log_every == 0 or i == n - 1:
            log.info("[%d/%d] processed", i + 1, n)

    log.info(
        "Preprocessing complete: %d ok, %d errors.", len(ok_records), len(errors)
    )
    return ok_records, errors


# ─── Step 5: Stratified speaker-disjoint splits ───────────────────────────────

def _optimal_speaker_split(
    speaker_sizes: dict[str, int],
    targets: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """
    Exhaustively search all speaker→split assignments to minimise MSE from
    targets = (train, val, test).  Works well for ≤ ~10 speakers; for larger
    speaker sets a greedy algorithm is used instead.

    Returns {speaker_id: split_name}.
    """
    speakers = sorted(speaker_sizes.keys())
    n_spk = len(speakers)
    split_names = ("train", "val", "test")
    total = sum(speaker_sizes.values())

    if n_spk > 12:
        # Greedy fallback for large sets
        return _greedy_speaker_split(speaker_sizes, targets, seed)

    best_assignment: tuple[int, ...] | None = None
    best_mse = float("inf")

    for assignment in product(range(3), repeat=n_spk):
        sizes = [0, 0, 0]
        for spk_idx, split_idx in zip(range(n_spk), assignment):
            sizes[split_idx] += speaker_sizes[speakers[spk_idx]]
        # Require at least one speaker in each split (if we have ≥3 speakers)
        if n_spk >= 3 and any(s == 0 for s in sizes):
            continue
        ratios = [s / total for s in sizes]
        mse = sum((r - t) ** 2 for r, t in zip(ratios, targets))
        if mse < best_mse:
            best_mse = mse
            best_assignment = assignment

    if best_assignment is None:
        # Can't satisfy all-splits constraint — relax it
        best_mse = float("inf")
        for assignment in product(range(3), repeat=n_spk):
            sizes = [0, 0, 0]
            for spk_idx, split_idx in zip(range(n_spk), assignment):
                sizes[split_idx] += speaker_sizes[speakers[spk_idx]]
            ratios = [s / total for s in sizes]
            mse = sum((r - t) ** 2 for r, t in zip(ratios, targets))
            if mse < best_mse:
                best_mse = mse
                best_assignment = assignment

    result: dict[str, str] = {}
    for spk, split_idx in zip(speakers, best_assignment):   # type: ignore[arg-type]
        result[spk] = split_names[split_idx]

    actual = [0, 0, 0]
    for spk, si in zip(speakers, best_assignment):           # type: ignore[arg-type]
        actual[si] += speaker_sizes[spk]
    actual_ratios = [a / total for a in actual]
    log.info(
        "Speaker split: train=%.1f%% val=%.1f%% test=%.1f%% "
        "(target 70/15/15, MSE=%.4f)",
        *[r * 100 for r in actual_ratios],
        best_mse,
    )
    return result


def _greedy_speaker_split(
    speaker_sizes: dict[str, int],
    targets: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """
    Greedy speaker assignment: sort by size desc, assign each speaker to the
    split with the largest remaining deficit below its target count.
    """
    total = sum(speaker_sizes.values())
    target_counts = {
        "train": targets[0] * total,
        "val":   targets[1] * total,
        "test":  targets[2] * total,
    }
    # Sort by size descending (largest speakers first → better packing)
    speakers = sorted(speaker_sizes.keys(), key=lambda s: -speaker_sizes[s])
    running = {"train": 0.0, "val": 0.0, "test": 0.0}
    assigned: dict[str, str] = {}
    for spk in speakers:
        # Assign to the split furthest BELOW its target count
        best_split = max(
            ("train", "val", "test"),
            key=lambda s: target_counts[s] - running[s],
        )
        assigned[spk] = best_split
        running[best_split] += speaker_sizes[spk]

    actual = [running["train"], running["val"], running["test"]]
    actual_ratios = [a / total for a in actual]
    log.info(
        "Greedy speaker split: train=%.1f%% val=%.1f%% test=%.1f%%",
        *[r * 100 for r in actual_ratios],
    )
    return assigned


def make_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Create stratified speaker-disjoint train/val/test splits.

    Strategy:
    1. Per language, collect all (spoof+bonafide) speakers and their total counts.
    2. Find an optimal speaker→split assignment per language (minimise MSE from 70/15/15).
    3. Apply the SAME assignment to bonafide and spoof — ensuring that the same
       source text (book) lands in the same split regardless of label.
    4. Speakers not seen in the assignment default to 'train'.
    """
    # Step 1: collect per-language speaker counts across ALL labels
    lang_speaker_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        lang_speaker_counts[rec["language"]][rec["speaker_id"]] += 1

    # Step 2: compute one speaker→split assignment per language
    lang_speaker_assignment: dict[str, dict[str, str]] = {}
    for lang, spk_sizes in sorted(lang_speaker_counts.items()):
        log.info(
            "Language %s: %d total samples, %d speakers",
            lang, sum(spk_sizes.values()), len(spk_sizes),
        )
        assignment = _optimal_speaker_split(spk_sizes)
        lang_speaker_assignment[lang] = assignment
        for spk, split in sorted(assignment.items()):
            log.info("  %s -> %s (%d samples)", spk, split, spk_sizes[spk])

    # Step 3: assign each record using its (language, speaker_id)
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [], "val": [], "test": []
    }
    for rec in records:
        lang = rec["language"]
        spk = rec["speaker_id"]
        split = lang_speaker_assignment.get(lang, {}).get(spk, "train")
        split_records[split].append(rec)

    for split, recs in split_records.items():
        log.info("Split %s: %d records", split, len(recs))

    return split_records


# ─── Step 6: Lock test set ────────────────────────────────────────────────────

def compute_test_hash(test_records: list[dict[str, Any]]) -> str:
    """SHA-256 of lexicographically-sorted sample_ids, newline-separated."""
    ids_sorted = sorted(r["sample_id"] for r in test_records)
    content = "\n".join(ids_sorted).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# ─── Step 7: Write all outputs ────────────────────────────────────────────────

def write_inventory(
    records: list[dict[str, Any]],
    processed_dir: Path,
) -> None:
    """Write dataset_inventory.json and .md."""
    by_label: dict[str, int] = defaultdict(int)
    by_lang: dict[str, int] = defaultdict(int)
    by_system: dict[str, int] = defaultdict(int)
    by_speaker: dict[str, int] = defaultdict(int)
    for r in records:
        by_label[r["label"]] += 1
        by_lang[r["language"]] += 1
        by_system[r["attack_system"]] += 1
        by_speaker[r["speaker_id"]] += 1

    inv = {
        "total_samples": len(records),
        "by_label": dict(sorted(by_label.items())),
        "by_language": dict(sorted(by_lang.items())),
        "by_attack_system": dict(sorted(by_system.items())),
        "by_speaker": dict(sorted(by_speaker.items())),
        "n_spoof_systems": sum(1 for k in by_system if k != "bonafide"),
        "n_speakers": len(by_speaker),
    }

    json_path = processed_dir / "dataset_inventory.json"
    json_path.write_text(json.dumps(inv, indent=2))
    log.info("Wrote %s", json_path)

    md_lines = [
        "# MLAAD-tiny Dataset Inventory",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"\n**Total samples:** {inv['total_samples']}",
        "\n## By Label",
    ]
    for k, v in inv["by_label"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines += ["\n## By Language"]
    for k, v in inv["by_language"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines += ["\n## By Attack System"]
    for sys_name, cnt in sorted(inv["by_attack_system"].items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{sys_name}`: {cnt}")
    md_lines += ["\n## By Speaker (M-AILABS reader / book proxy)"]
    for spk, cnt in sorted(inv["by_speaker"].items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{spk}`: {cnt}")

    md_path = processed_dir / "dataset_inventory.md"
    md_path.write_text("\n".join(md_lines))
    log.info("Wrote %s", md_path)


def write_splits(
    split_records: dict[str, list[dict[str, Any]]],
    splits_dir: Path,
) -> str:
    """Write train/val/test JSON files and test_LOCKED.md. Returns test hash."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, recs in split_records.items():
        path = splits_dir / f"{split_name}.json"
        path.write_text(json.dumps(recs, indent=2))
        log.info("Wrote %s (%d records)", path, len(recs))

    test_hash = compute_test_hash(split_records["test"])
    lock_path = splits_dir / "test_LOCKED.md"
    lock_path.write_text(
        f"# Test Set Lock\n\n"
        f"**SHA-256 of sorted test sample_ids:**  \n`{test_hash}`\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Do not modify the test split without updating this hash.\n"
        f"Verification:\n"
        f"```python\n"
        f"import json, hashlib\n"
        f"recs = json.load(open('splits/test.json'))\n"
        f"ids = '\\n'.join(sorted(r['sample_id'] for r in recs)).encode()\n"
        f"assert hashlib.sha256(ids).hexdigest() == '{test_hash}'\n"
        f"```\n"
    )
    log.info("Wrote %s (hash: %s…)", lock_path, test_hash[:16])
    return test_hash


def write_metadata(
    records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    test_hash: str,
    processed_dir: Path,
    snapshot_dir: Path,
) -> None:
    """Write metadata.json with provenance and split statistics."""
    split_stats: dict[str, Any] = {}
    for split_name, recs in split_records.items():
        n = len(recs)
        label_counts: dict[str, int] = defaultdict(int)
        lang_counts: dict[str, int] = defaultdict(int)
        speaker_set: set[str] = set()
        system_set: set[str] = set()
        for r in recs:
            label_counts[r["label"]] += 1
            lang_counts[r["language"]] += 1
            speaker_set.add(r["speaker_id"])
            system_set.add(r["attack_system"])
        split_stats[split_name] = {
            "n": n,
            "fraction": n / max(len(records), 1),
            "by_label": dict(sorted(label_counts.items())),
            "by_language": dict(sorted(lang_counts.items())),
            "n_speakers": len(speaker_set),
            "n_systems": len(system_set),
        }

    meta = {
        "dataset": "MLAAD-tiny",
        "source_repo": f"huggingface.co/datasets/{REPO_ID}",
        "revision": SNAP_REVISION,
        "snapshot_dir": str(snapshot_dir),
        "processed_dir": str(processed_dir),
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audio_preprocessing": {
            "target_sr": TARGET_SR,
            "target_samples": TARGET_SAMPLES,
            "duration_seconds": TARGET_SAMPLES / TARGET_SR,
            "format": "torch.Tensor float32, shape (48000,)",
            "normalization": "none (soundfile default: float32 in [-1, 1])",
            "channels": "mono (mean of channels if stereo)",
            "padding": "tile-pad then centre-crop",
        },
        "split_seed": SPLIT_SEED,
        "split_strategy": "speaker-disjoint (M-AILABS reader), optimal MSE from 70/15/15",
        "test_sha256": test_hash,
        "total_samples": len(records),
        "split_stats": split_stats,
    }
    path = processed_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s", path)


def write_splits_config(
    split_records: dict[str, list[dict[str, Any]]],
    processed_dir: Path,
) -> None:
    """Write splits_config.json with per-split speaker assignments."""
    speaker_assignments: dict[str, list[str]] = {
        split: sorted({r["speaker_id"] for r in recs})
        for split, recs in split_records.items()
    }
    cfg = {
        "split_seed": SPLIT_SEED,
        "split_targets": SPLIT_TARGETS,
        "speaker_assignments": speaker_assignments,
        "description": (
            "Each speaker's samples appear in exactly one split "
            "(speaker-disjoint). Speaker = M-AILABS reader from "
            "original_file field of meta.csv."
        ),
    }
    path = processed_dir / "splits_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    log.info("Wrote %s", path)


def write_run_config(processed_dir: Path) -> None:
    """Write run_config.json for downstream experiment consumption."""
    cfg = {
        "processed_dir": str(processed_dir),
        "audio_dir": str(processed_dir / "audio"),
        "splits": {
            split: str(processed_dir / "splits" / f"{split}.json")
            for split in ("train", "val", "test")
        },
        "test_locked": str(processed_dir / "splits" / "test_LOCKED.md"),
        "target_sr": TARGET_SR,
        "target_samples": TARGET_SAMPLES,
        "labels": ["bonafide", "spoof"],
        "usage": (
            "Load splits/*.json to get record dicts. "
            "Load audio_dir/{sample_id}.pt with torch.load() "
            "to get a (48000,) float32 tensor. "
            "No further preprocessing needed."
        ),
    }
    path = processed_dir / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    log.info("Wrote %s", path)


# ─── Step 8: Sanity checks ────────────────────────────────────────────────────

def run_sanity_checks(
    records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    processed_dir: Path,
    test_hash: str,
) -> bool:
    """Run 8 sanity checks. Returns True if all pass."""
    checks: list[tuple[str, bool, str]] = []

    # (a) total WAV files discovered matches inventory
    inv_path = processed_dir / "dataset_inventory.json"
    inv = json.loads(inv_path.read_text())
    n_inv = inv["total_samples"]
    check_a = n_inv == len(records)
    checks.append(("(a) inventory count matches records", check_a,
                   f"inventory={n_inv}, records={len(records)}"))

    # (b) no duplicate sample_ids
    all_ids = [r["sample_id"] for r in records]
    check_b = len(set(all_ids)) == len(all_ids)
    dupes = len(all_ids) - len(set(all_ids))
    checks.append(("(b) no duplicate sample_ids", check_b,
                   f"{dupes} duplicates"))

    # (c) split sizes within ±10 pp of 70/15/15
    total = sum(len(r) for r in split_records.values())
    size_ok = True
    size_msg_parts = []
    for split, target in SPLIT_TARGETS.items():
        actual = len(split_records[split]) / max(total, 1)
        diff = abs(actual - target)
        ok = diff <= 0.10
        size_ok = size_ok and ok
        size_msg_parts.append(f"{split}={actual:.3f} (target={target:.2f})")
    checks.append(("(c) split sizes within ±10 pp of target",
                   size_ok, ", ".join(size_msg_parts)))

    # (d) bonafide:spoof ratio consistent across splits (skip if no bonafide)
    has_bonafide = any(r["label"] == "bonafide" for r in records)
    if has_bonafide:
        ratios = {}
        ratio_ok = True
        for split, recs in split_records.items():
            if not recs:
                continue
            bf = sum(1 for r in recs if r["label"] == "bonafide")
            ratio = bf / len(recs)
            ratios[split] = ratio
        if len(ratios) >= 2:
            vals = list(ratios.values())
            ratio_ok = (max(vals) - min(vals)) <= 0.10
        checks.append(("(d) bonafide:spoof ratio consistent across splits",
                       ratio_ok, str(ratios)))
    else:
        checks.append(("(d) bonafide:spoof ratio [SKIPPED — no bonafide]",
                       True, "bonafide not yet downloaded"))

    # (e) language balance in spoof consistent across splits
    lang_ok = True
    lang_msg_parts = []
    for split, recs in split_records.items():
        spoof_recs = [r for r in recs if r["label"] == "spoof"]
        if not spoof_recs:
            continue
        n_de = sum(1 for r in spoof_recs if r["language"] == "de")
        de_ratio = n_de / len(spoof_recs)
        lang_msg_parts.append(f"{split}: de_ratio={de_ratio:.3f}")
    if lang_msg_parts:
        # Check spread ≤ 20 pp
        de_ratios = []
        for split, recs in split_records.items():
            spoof_recs = [r for r in recs if r["label"] == "spoof"]
            if not spoof_recs:
                continue
            n_de = sum(1 for r in spoof_recs if r["language"] == "de")
            de_ratios.append(n_de / len(spoof_recs))
        if len(de_ratios) >= 2:
            lang_ok = (max(de_ratios) - min(de_ratios)) <= 0.20
    checks.append(("(e) language balance (de fraction) consistent across splits",
                   lang_ok, ", ".join(lang_msg_parts)))

    # (f) all attack systems represented in train
    all_systems = {r["attack_system"] for r in records}
    train_systems = {r["attack_system"] for r in split_records["train"]}
    missing_in_train = all_systems - train_systems
    check_f = len(missing_in_train) == 0
    checks.append(("(f) all attack systems represented in train",
                   check_f, f"missing in train: {sorted(missing_in_train)}"))

    # (g) speaker IDs disjoint between train and test
    train_speakers = {r["speaker_id"] for r in split_records["train"]}
    test_speakers = {r["speaker_id"] for r in split_records["test"]}
    leaked = train_speakers & test_speakers
    check_g = len(leaked) == 0
    checks.append(("(g) speaker IDs disjoint between train and test",
                   check_g, f"leaked speakers: {sorted(leaked)}"))

    # (h) test SHA-256 stored in test_LOCKED.md matches
    lock_path = processed_dir / "splits" / "test_LOCKED.md"
    lock_text = lock_path.read_text()
    check_h = test_hash in lock_text
    checks.append(("(h) test SHA-256 in test_LOCKED.md matches computed hash",
                   check_h, f"hash={test_hash[:16]}…"))

    # ── Print results
    print("\n" + "═" * 70)
    print("SANITY CHECKS")
    print("═" * 70)
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")
        all_pass = all_pass and passed
    print("═" * 70)
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("═" * 70 + "\n")
    return all_pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=PROCESSED_DIR,
        help="Output directory for processed dataset (default: %(default)s)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing preprocessed audio files",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip HuggingFace download and use existing snapshot",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Discover and inspect only; skip preprocessing and splits",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir: Path = args.output_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MLAAD-tiny Phase 0 — Dataset Preparation")
    log.info("=" * 60)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if args.skip_download:
        snapshot_dir = (
            RAW_CACHE_DIR
            / f"datasets--{REPO_ID.replace('/', '--')}"
            / "snapshots"
            / SNAP_REVISION
        )
        log.info("Skipping download; using existing snapshot: %s", snapshot_dir)
    else:
        snapshot_dir = download_snapshot()

    if not snapshot_dir.is_dir():
        log.error("Snapshot directory not found: %s", snapshot_dir)
        sys.exit(1)

    log.info("Snapshot directory: %s", snapshot_dir)

    # ── Step 2: Load speaker metadata ─────────────────────────────────────────
    log.info("Loading speaker metadata from meta.csv files …")
    speaker_map = load_speaker_map(snapshot_dir)
    log.info("Speaker map: %d entries", len(speaker_map))

    # ── Step 3: Discover samples ──────────────────────────────────────────────
    log.info("Discovering WAV files …")
    records = discover_samples(snapshot_dir, speaker_map)

    if not records:
        log.error("No WAV files found in snapshot. Aborting.")
        sys.exit(1)

    # ── Step 4: Write inventory (pre-preprocessing) ───────────────────────────
    write_inventory(records, processed_dir)

    if args.dry_run:
        log.info("--dry-run: stopping after inventory.")
        return

    # ── Step 5: Preprocess audio ──────────────────────────────────────────────
    log.info("Preprocessing audio …")
    audio_dir = processed_dir / "audio"
    ok_records, errors = preprocess_all(records, audio_dir, force=args.force)

    if errors:
        log.warning("%d files failed preprocessing:", len(errors))
        for e in errors[:10]:
            log.warning("  %s", e)
        if len(errors) > 10:
            log.warning("  … and %d more", len(errors) - 10)

    if not ok_records:
        log.error("No records survived preprocessing. Aborting.")
        sys.exit(1)

    # Re-write inventory with ok_records only
    write_inventory(ok_records, processed_dir)

    # ── Step 6: Create splits ─────────────────────────────────────────────────
    log.info("Creating stratified speaker-disjoint splits …")
    split_records = make_splits(ok_records)

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    splits_dir = processed_dir / "splits"
    test_hash = write_splits(split_records, splits_dir)
    write_metadata(ok_records, split_records, test_hash, processed_dir, snapshot_dir)
    write_splits_config(split_records, processed_dir)
    write_run_config(processed_dir)

    # ── Step 8: Sanity checks ─────────────────────────────────────────────────
    all_pass = run_sanity_checks(ok_records, split_records, processed_dir, test_hash)

    if not all_pass:
        log.warning("Some sanity checks failed — review output above.")
        sys.exit(2)

    log.info("Phase 0 complete. Processed dataset at: %s", processed_dir)
    log.info(
        "Run config: %s", processed_dir / "run_config.json"
    )


if __name__ == "__main__":
    main()
