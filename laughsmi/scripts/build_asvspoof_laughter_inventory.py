"""Build a clean inventory for ASVspoof bonafide speech vs VocalSound laughter.

This pivot avoids ITW detector labels entirely. It exports a sampled set of
ASVspoof2019 LA bonafide utterances from the local DatasetDict cache to WAV,
samples VocalSound laughter clips, and writes an inventory compatible with
extract_wavlm_features.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def resolve_audio_path(audio_dir: Path, file_id: str) -> Path | None:
    candidates = [audio_dir / file_id]
    if not any(str(file_id).endswith(ext) for ext in (".wav", ".flac", ".mp3")):
        candidates.extend(audio_dir / f"{file_id}{ext}" for ext in (".wav", ".flac", ".mp3"))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def export_asvspoof_bonafide(cache_dir: Path, wav_dir: Path, n: int, seed: int) -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(cache_dir))
    rows = []
    for split in ds.keys():
        split_ds = ds[split]
        for idx, ex in enumerate(split_ds):
            if ex["system_id"] != "-":
                continue
            rows.append((split, idx, ex))

    rng = np.random.default_rng(seed)
    n_sample = min(n, len(rows))
    choice = rng.choice(len(rows), size=n_sample, replace=False)
    wav_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for out_i, row_i in enumerate(choice):
        split, idx, ex = rows[int(row_i)]
        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        speaker = ex.get("speaker_id", "")
        wav_path = wav_dir / f"asv2019_{split}_{idx:05d}_{safe_name(speaker)}.wav"
        if not wav_path.exists():
            sf.write(wav_path, arr, sr)
        out_rows.append({
            "group": "speech-asv-bona",
            "file_path": str(wav_path),
            "start_s": 0.0,
            "end_s": -1.0,
            "pair_id": "",
            "speaker": speaker,
            "source": "ASVspoof2019_LA",
            "source_split": split,
        })
    return out_rows


def sample_vocalsound_laughter(csv_path: Path, audio_dir: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    file_col = None
    for cand in ("file", "filename", "file_id", "path", "clip"):
        if cand in df.columns:
            file_col = cand
            break
    if file_col is None:
        file_col = df.columns[0]

    label_col = None
    for cand in ("label", "class", "category"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is not None:
        df = df[df[label_col].astype(str).str.contains("laugh", case=False, na=False)]

    n_sample = min(n, len(df))
    sampled = df.sample(n=n_sample, random_state=seed) if n_sample else df.iloc[0:0]
    rows = []
    for _, row in sampled.iterrows():
        path = resolve_audio_path(audio_dir, str(row[file_col]))
        if path is None:
            continue
        rows.append({
            "group": "laugh-vs",
            "file_path": str(path),
            "start_s": 0.0,
            "end_s": -1.0,
            "pair_id": "",
            "speaker": row.get("speaker_id", row.get("speaker", "")),
            "source": "VocalSound",
            "source_split": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof-cache", type=Path, default=Path("data/asvspoof_2019_la"))
    parser.add_argument("--asv-wav-dir", type=Path, default=Path("laughsmi/data/asvspoof2019_bonafide_wavs"))
    parser.add_argument("--vocalsound-csv", type=Path, default=Path("laughsmi/data/vocalsound/laughter_list.csv"))
    parser.add_argument("--vocalsound-dir", type=Path, default=Path("laughsmi/data/vocalsound/audio_16k_raw"))
    parser.add_argument("--n-asv", type=int, default=300)
    parser.add_argument("--n-vocalsound", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--out", type=Path, default=Path("laughsmi/embeddings/asv_vs_laughter_inventory.csv"))
    args = parser.parse_args()

    asv_rows = export_asvspoof_bonafide(args.asvspoof_cache, args.asv_wav_dir, args.n_asv, args.seed)
    laugh_rows = sample_vocalsound_laughter(
        args.vocalsound_csv, args.vocalsound_dir, args.n_vocalsound, args.seed + 1
    )
    out_df = pd.DataFrame(asv_rows + laugh_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")
    print(out_df["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
