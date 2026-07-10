"""Export detector-marked laughter clips for manual review."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def parse_list(x) -> list[float]:
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [float(v) for v in str(x).split(";") if v.strip()]


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=80)
    parser.add_argument("--context-s", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    df = pd.read_csv(args.detections)
    df["n_laugh_segs"] = pd.to_numeric(df["n_laugh_segs"], errors="coerce").fillna(0).astype(int)
    hits = df[df.n_laugh_segs > 0].copy()
    if hits.empty:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "manifest.csv").write_text("")
        print("No detected laughter segments.")
        return

    rng = np.random.default_rng(args.seed)
    if len(hits) > args.max_clips:
        # Mix high-confidence clips with random clips.
        top = hits.sort_values("max_prob", ascending=False).head(args.max_clips // 2)
        rest = hits.drop(index=top.index)
        rand_n = args.max_clips - len(top)
        rand = rest.sample(n=min(rand_n, len(rest)), random_state=int(rng.integers(0, 2**31 - 1)))
        hits = pd.concat([top, rand]).drop_duplicates("file_id")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in hits.iterrows():
        starts = parse_list(row.get("seg_starts", ""))
        ends = parse_list(row.get("seg_ends", ""))
        segs = [(s, e) for s, e in zip(starts, ends) if e > s]
        if not segs:
            continue
        # Export the longest segment per source file for fast QC.
        s, e = max(segs, key=lambda p: p[1] - p[0])
        src = args.audio_dir / str(row.file_id)
        if not src.exists():
            continue
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur_s = len(wav) / sr
        cs = max(0.0, s - args.context_s)
        ce = min(dur_s, e + args.context_s)
        a = int(round(cs * sr))
        b = int(round(ce * sr))
        clip = wav[a:b]
        peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
        rms = float(np.sqrt(np.mean(clip * clip))) if len(clip) else 0.0
        name = (
            f"{safe(str(row.file_id))}_p{float(row.max_prob):.3f}_"
            f"{s:.2f}-{e:.2f}_{safe(str(row.get('speaker', '')))}.wav"
        )
        sf.write(args.out_dir / name, clip, sr)
        rows.append({
            "clip_file": name,
            "file_id": row.file_id,
            "speaker": row.get("speaker", ""),
            "detected_start_s": s,
            "detected_end_s": e,
            "clip_start_s": cs,
            "clip_end_s": ce,
            "max_prob": row.max_prob,
            "n_laugh_segs_in_file": row.n_laugh_segs,
            "rms": rms,
            "peak": peak,
            "review_laughter": "",
            "review_english": "",
            "review_notes": "",
        })

    with (args.out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "README.md").write_text(
        "# VoxCeleb2 laughter review clips\n\n"
        "Fill `manifest.csv` columns `review_laughter`, `review_english`, and `review_notes`.\n"
        "Suggested labels: yes / no / ambiguous.\n"
    )
    print(f"Wrote {len(rows)} review clips to {args.out_dir}")


if __name__ == "__main__":
    main()
