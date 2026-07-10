"""Build the segment inventory for Stage 2 (WavLM geometry analysis).

Implements laughsmi_plan.md §5.1:

Groups produced (column `group`):
  - laugh-bona:   ITW bona-fide laughter segments from Stage 1, padded to
                  >= 0.5 s duration.
  - speech-bona:  for each laugh-bona segment, a same-file, same-duration
                  speech segment that is >= 1 s away from ANY laughter
                  segment in that file (controls speaker + channel; this is
                  the key confound-control pairing). Linked to its laugh-bona
                  segment via `pair_id`. Skipped if no room is found in the
                  file.
  - laugh-vs:     300 random whole VocalSound laughter clips (external
                  genuine-laughter anchor).
  - speech-spoof: 300 random ITW spoof files, each contributing one random
                  segment whose duration is drawn from the empirical
                  laugh-bona duration distribution (duration-matched).
  - laugh-spoof:  any spoof-side laughter detections from Stage 1 (n_laugh_segs
                  >= 1 among spoof-labeled files), plus optionally Bark probe
                  clips (whole-clip segments) if a bark directory is given.

Output: embeddings/segment_inventory.csv with columns:
    group, file_path, start_s, end_s, pair_id, speaker

`pair_id` is only populated for laugh-bona / speech-bona rows (shared integer
ID linking a laugh segment to its paired speech segment); all other rows have
an empty pair_id.

Usage:
    python build_segment_inventory.py \
        --itw-csv detector_out/itw_laughter.csv \
        --itw-audio-dir data/in_the_wild \
        --vocalsound-csv data/vocalsound/laughter_list.csv \
        --vocalsound-dir data/vocalsound/audio \
        --bark-dir data/bark_probe \
        --out embeddings/segment_inventory.csv \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_LAUGH_PAD_S = 0.5
SPEECH_LAUGH_GAP_S = 1.0
N_VOCALSOUND_CLIPS = 300
N_SPOOF_SPEECH_CLIPS = 300


def parse_seg_lists(row) -> list[tuple[float, float]]:
    """Parse ';'-joined seg_starts/seg_ends strings into a list of (start, end)."""
    starts_raw = row.get("seg_starts", "")
    ends_raw = row.get("seg_ends", "")
    if pd.isna(starts_raw) or pd.isna(ends_raw) or str(starts_raw).strip() == "":
        return []
    starts = [float(x) for x in str(starts_raw).split(";") if x.strip() != ""]
    ends = [float(x) for x in str(ends_raw).split(";") if x.strip() != ""]
    return list(zip(starts, ends))


def pad_segment(start: float, end: float, min_dur: float, file_dur: float) -> tuple[float, float]:
    """Pad a segment symmetrically to at least `min_dur` seconds, clamped to [0, file_dur]."""
    dur = end - start
    if dur >= min_dur:
        return start, end
    deficit = min_dur - dur
    new_start = max(0.0, start - deficit / 2)
    new_end = min(file_dur, new_start + min_dur)
    new_start = max(0.0, new_end - min_dur)
    return new_start, new_end


def find_paired_speech_segment(
    file_dur: float,
    target_dur: float,
    laugh_segments: list[tuple[float, float]],
    rng: np.random.Generator,
    gap_s: float = SPEECH_LAUGH_GAP_S,
    max_tries: int = 200,
) -> tuple[float, float] | None:
    """Find a same-duration speech window >= gap_s away from all laugh_segments.

    Tries random start offsets within the file; returns the first valid
    candidate, or None if no room is found after max_tries attempts (falls
    back to a deterministic scan before giving up).
    """
    if file_dur < target_dur:
        return None

    def is_far_enough(cand_start: float, cand_end: float) -> bool:
        for (ls, le) in laugh_segments:
            # Overlap or within gap_s of a laughter segment on either side.
            if cand_start < le + gap_s and cand_end > ls - gap_s:
                return False
        return True

    max_start = file_dur - target_dur
    if max_start < 0:
        return None

    for _ in range(max_tries):
        cand_start = float(rng.uniform(0.0, max_start))
        cand_end = cand_start + target_dur
        if is_far_enough(cand_start, cand_end):
            return cand_start, cand_end

    # Deterministic fallback scan on a grid.
    n_grid = 200
    for i in range(n_grid):
        cand_start = max_start * i / max(1, n_grid - 1)
        cand_end = cand_start + target_dur
        if is_far_enough(cand_start, cand_end):
            return cand_start, cand_end

    return None


def resolve_audio_path(audio_dir: Path, file_id: str) -> Path | None:
    """Resolve a file_id to an actual audio file path, trying common extensions."""
    candidates = [audio_dir / file_id]
    if not any(str(file_id).endswith(ext) for ext in (".wav", ".flac", ".mp3")):
        for ext in (".wav", ".flac", ".mp3"):
            candidates.append(audio_dir / f"{file_id}{ext}")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def build_laugh_bona_and_paired_speech(
    itw_df: pd.DataFrame, itw_audio_dir: Path, rng: np.random.Generator
) -> tuple[list[dict], list[dict]]:
    """Build laugh-bona rows and their paired speech-bona rows.

    Returns (laugh_bona_rows, speech_bona_rows).
    """
    laugh_rows: list[dict] = []
    speech_rows: list[dict] = []
    pair_counter = 0

    bona_df = itw_df[itw_df["label"] == "bona-fide"]
    for _, row in bona_df.iterrows():
        segs = parse_seg_lists(row)
        if not segs:
            continue
        file_id = row["file_id"]
        file_dur = float(row["dur_s"])
        speaker = row.get("speaker", "")
        audio_path = resolve_audio_path(itw_audio_dir, file_id)
        if audio_path is None:
            continue

        for (raw_start, raw_end) in segs:
            start, end = pad_segment(raw_start, raw_end, MIN_LAUGH_PAD_S, file_dur)
            target_dur = end - start
            if target_dur <= 0:
                continue

            paired = find_paired_speech_segment(file_dur, target_dur, segs, rng)
            if paired is None:
                continue  # skip if no room, per plan §5.1

            pair_id = pair_counter
            pair_counter += 1

            laugh_rows.append({
                "group": "laugh-bona",
                "file_path": str(audio_path),
                "start_s": round(start, 4),
                "end_s": round(end, 4),
                "pair_id": pair_id,
                "speaker": speaker,
            })
            speech_rows.append({
                "group": "speech-bona",
                "file_path": str(audio_path),
                "start_s": round(paired[0], 4),
                "end_s": round(paired[1], 4),
                "pair_id": pair_id,
                "speaker": speaker,
            })

    return laugh_rows, speech_rows


def build_laugh_vs(
    vocalsound_csv: Path, vocalsound_dir: Path, n: int, rng: np.random.Generator
) -> list[dict]:
    """Sample n random whole VocalSound laughter clips."""
    vs_df = pd.read_csv(vocalsound_csv)
    # Try to find a laughter-only subset if a label column exists; otherwise
    # assume the provided CSV already lists only laughter clips (per plan,
    # "laughter_list.csv").
    label_col = None
    for cand in ("label", "class", "category"):
        if cand in vs_df.columns:
            label_col = cand
            break
    if label_col is not None:
        vs_df = vs_df[vs_df[label_col].astype(str).str.contains("laugh", case=False, na=False)]

    file_col = None
    for cand in ("file", "filename", "file_id", "path", "clip"):
        if cand in vs_df.columns:
            file_col = cand
            break
    if file_col is None:
        file_col = vs_df.columns[0]

    n_sample = min(n, len(vs_df))
    if n_sample < n:
        print(f"WARNING: requested {n} VocalSound laughter clips but only {len(vs_df)} available; "
              f"using {n_sample}.", file=sys.stderr)
    sampled = vs_df.sample(n=n_sample, random_state=rng.integers(0, 2**31 - 1)) if n_sample else vs_df.iloc[0:0]

    rows = []
    for _, row in sampled.iterrows():
        fname = str(row[file_col])
        audio_path = resolve_audio_path(vocalsound_dir, fname)
        if audio_path is None:
            continue
        rows.append({
            "group": "laugh-vs",
            "file_path": str(audio_path),
            "start_s": 0.0,
            "end_s": -1.0,  # -1.0 sentinel = whole clip (resolved at feature-extraction time)
            "pair_id": "",
            "speaker": row.get("speaker", ""),
        })
    return rows


def build_speech_spoof(
    itw_df: pd.DataFrame,
    itw_audio_dir: Path,
    n: int,
    laugh_dur_distribution: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample n random ITW spoof files, each with one random segment whose
    duration is drawn from the empirical laugh-bona duration distribution."""
    spoof_df = itw_df[itw_df["label"] == "spoof"]
    n_sample = min(n, len(spoof_df))
    if n_sample < n:
        print(f"WARNING: requested {n} spoof files but only {len(spoof_df)} available; "
              f"using {n_sample}.", file=sys.stderr)
    if n_sample == 0:
        return []
    sampled = spoof_df.sample(n=n_sample, random_state=rng.integers(0, 2**31 - 1))

    rows = []
    if laugh_dur_distribution.size == 0:
        laugh_dur_distribution = np.array([1.0])  # fallback: 1s segments

    for _, row in sampled.iterrows():
        file_id = row["file_id"]
        file_dur = float(row["dur_s"])
        audio_path = resolve_audio_path(itw_audio_dir, file_id)
        if audio_path is None:
            continue
        target_dur = float(rng.choice(laugh_dur_distribution))
        target_dur = min(target_dur, file_dur) if file_dur > 0 else target_dur
        if target_dur <= 0 or file_dur <= 0:
            continue
        max_start = max(0.0, file_dur - target_dur)
        start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        end = start + target_dur
        rows.append({
            "group": "speech-spoof",
            "file_path": str(audio_path),
            "start_s": round(start, 4),
            "end_s": round(end, 4),
            "pair_id": "",
            "speaker": row.get("speaker", ""),
        })
    return rows


def build_laugh_spoof(itw_df: pd.DataFrame, itw_audio_dir: Path, bark_dir: Path | None) -> list[dict]:
    """Spoof-side laughter detections from Stage 1, plus optional Bark probe clips."""
    rows: list[dict] = []
    spoof_df = itw_df[itw_df["label"] == "spoof"]
    for _, row in spoof_df.iterrows():
        segs = parse_seg_lists(row)
        if not segs:
            continue
        file_id = row["file_id"]
        file_dur = float(row["dur_s"])
        audio_path = resolve_audio_path(itw_audio_dir, file_id)
        if audio_path is None:
            continue
        for (raw_start, raw_end) in segs:
            start, end = pad_segment(raw_start, raw_end, MIN_LAUGH_PAD_S, file_dur)
            if end <= start:
                continue
            rows.append({
                "group": "laugh-spoof",
                "file_path": str(audio_path),
                "start_s": round(start, 4),
                "end_s": round(end, 4),
                "pair_id": "",
                "speaker": row.get("speaker", ""),
            })

    if bark_dir is not None and bark_dir.exists():
        for audio_path in sorted(bark_dir.glob("*.wav")):
            rows.append({
                "group": "laugh-spoof",
                "file_path": str(audio_path),
                "start_s": 0.0,
                "end_s": -1.0,  # whole clip
                "pair_id": "",
                "speaker": "bark",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Stage-2 segment inventory (laugh-bona, speech-bona, "
                     "laugh-vs, speech-spoof, laugh-spoof)."
    )
    parser.add_argument("--itw-csv", type=Path, default=Path("detector_out/itw_laughter.csv"))
    parser.add_argument("--itw-audio-dir", type=Path, default=Path("data/in_the_wild"))
    parser.add_argument("--vocalsound-csv", type=Path, default=Path("data/vocalsound/laughter_list.csv"))
    parser.add_argument("--vocalsound-dir", type=Path, default=Path("data/vocalsound/audio"))
    parser.add_argument("--bark-dir", type=Path, default=None,
                        help="Optional directory of Bark synthetic-laughter probe clips (H4).")
    parser.add_argument("--out", type=Path, default=Path("embeddings/segment_inventory.csv"))
    parser.add_argument("--n-vocalsound", type=int, default=N_VOCALSOUND_CLIPS)
    parser.add_argument("--n-speech-spoof", type=int, default=N_SPOOF_SPEECH_CLIPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.itw_csv.exists():
        print(f"ERROR: itw csv not found: {args.itw_csv}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    itw_df = pd.read_csv(args.itw_csv, dtype={"file_id": str})

    laugh_bona_rows, speech_bona_rows = build_laugh_bona_and_paired_speech(
        itw_df, args.itw_audio_dir, rng
    )
    print(f"laugh-bona: {len(laugh_bona_rows)} segments paired with speech-bona "
          f"({len(speech_bona_rows)})")

    laugh_durs = np.array(
        [r["end_s"] - r["start_s"] for r in laugh_bona_rows], dtype=np.float64
    )

    laugh_vs_rows = []
    if args.vocalsound_csv.exists():
        laugh_vs_rows = build_laugh_vs(
            args.vocalsound_csv, args.vocalsound_dir, args.n_vocalsound, rng
        )
    else:
        print(f"WARNING: vocalsound csv not found ({args.vocalsound_csv}); "
              f"skipping laugh-vs group.", file=sys.stderr)
    print(f"laugh-vs: {len(laugh_vs_rows)} segments")

    speech_spoof_rows = build_speech_spoof(
        itw_df, args.itw_audio_dir, args.n_speech_spoof, laugh_durs, rng
    )
    print(f"speech-spoof: {len(speech_spoof_rows)} segments")

    laugh_spoof_rows = build_laugh_spoof(itw_df, args.itw_audio_dir, args.bark_dir)
    print(f"laugh-spoof: {len(laugh_spoof_rows)} segments")

    all_rows = laugh_bona_rows + speech_bona_rows + laugh_vs_rows + speech_spoof_rows + laugh_spoof_rows
    out_df = pd.DataFrame(
        all_rows,
        columns=["group", "file_path", "start_s", "end_s", "pair_id", "speaker"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows total)")
    print(out_df["group"].value_counts())


if __name__ == "__main__":
    main()
