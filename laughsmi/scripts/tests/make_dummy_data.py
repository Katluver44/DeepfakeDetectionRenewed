"""Fabricate tiny synthetic data to end-to-end test table1_prevalence.py,
build_segment_inventory.py, and stage2_analysis.py without any real ITW/
VocalSound data or a torch/transformers install.

Produces (under --out-dir, default scripts/tests/_dummy):
  itw_laughter.csv         -- synthetic Stage-1 detector output
  audio/itw/*.wav          -- dummy ITW wav files (sine tone + noise bursts)
  vocalsound/laughter_list.csv, vocalsound/audio/*.wav -- dummy VocalSound clips
  bark/*.wav               -- a couple of dummy "Bark" synthetic laughter clips
  segment_inventory.csv    -- NOT produced here; build_segment_inventory.py
                              is run separately by run_dummy_pipeline.sh/py
  features.parquet (or csv), mean_emb_layer9.npy, mean_emb_layer12.npy
                            -- fabricated directly (skips extract_wavlm_features.py,
                               since that needs torch) so stage2_analysis.py can
                               be tested end-to-end.

Usage:
    python make_dummy_data.py --out-dir scripts/tests/_dummy --seed 42
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    """Write a mono float32 [-1,1] array to a 16-bit PCM wav file using stdlib wave."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm = (audio_clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_tone(dur_s: float, freq: float, rng: np.random.Generator, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, dur_s, int(dur_s * SR), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    tone += 0.02 * rng.normal(size=tone.shape)
    return tone.astype(np.float32)


def make_noise_burst(dur_s: float, rng: np.random.Generator, amp: float = 0.4) -> np.ndarray:
    n = int(dur_s * SR)
    return (amp * rng.normal(size=n)).astype(np.float32)


def build_itw_file(
    file_id: str,
    label: str,
    has_laughter: bool,
    rng: np.random.Generator,
    audio_dir: Path,
) -> dict:
    """Build one dummy ITW file: some speech tone, optionally with 1-2 'laughter'
    noise bursts inserted, and return its itw_laughter.csv row."""
    total_dur = float(rng.uniform(4.0, 8.0))
    audio = make_tone(total_dur, freq=float(rng.uniform(150, 300)), rng=rng)

    seg_starts, seg_ends = [], []
    if has_laughter:
        n_segs = int(rng.integers(1, 3))
        cursor = 0.5
        for _ in range(n_segs):
            seg_dur = float(rng.uniform(0.3, 1.2))
            if cursor + seg_dur + 1.0 > total_dur:
                break
            start = cursor
            end = start + seg_dur
            n_samples = int(seg_dur * SR)
            start_idx = int(start * SR)
            audio[start_idx:start_idx + n_samples] += make_noise_burst(seg_dur, rng, amp=0.5)[:n_samples]
            seg_starts.append(round(start, 3))
            seg_ends.append(round(end, 3))
            cursor = end + float(rng.uniform(1.2, 2.0))

    wav_path = audio_dir / f"{file_id}.wav"
    write_wav(wav_path, audio)

    n_laugh_segs = len(seg_starts)
    laugh_dur_s = sum(e - s for s, e in zip(seg_starts, seg_ends))
    laugh_ratio = laugh_dur_s / total_dur if total_dur > 0 else 0.0
    max_prob = float(rng.uniform(0.55, 0.95)) if n_laugh_segs > 0 else float(rng.uniform(0.05, 0.25))

    return {
        "file_id": f"{file_id}.wav",
        "label": label,
        "speaker": f"spk{int(rng.integers(0, 20)):03d}",
        "dur_s": round(total_dur, 3),
        "n_laugh_segs": n_laugh_segs,
        "laugh_dur_s": round(laugh_dur_s, 3),
        "laugh_ratio": round(laugh_ratio, 5),
        "max_prob": round(max_prob, 4),
        "seg_starts": ";".join(str(s) for s in seg_starts),
        "seg_ends": ";".join(str(e) for e in seg_ends),
    }


def build_vocalsound(rng: np.random.Generator, out_dir: Path, n: int = 20) -> None:
    audio_dir = out_dir / "vocalsound" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        dur = float(rng.uniform(0.5, 2.0))
        audio = make_noise_burst(dur, rng, amp=0.4)
        fname = f"vs_laugh_{i:03d}.wav"
        write_wav(audio_dir / fname, audio)
        rows.append({"file": fname, "label": "laughter", "speaker": f"vs_spk{i % 5}"})
    pd.DataFrame(rows).to_csv(out_dir / "vocalsound" / "laughter_list.csv", index=False)


def build_bark(rng: np.random.Generator, out_dir: Path, n: int = 4) -> None:
    bark_dir = out_dir / "bark"
    bark_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        dur = float(rng.uniform(0.8, 1.5))
        audio = make_noise_burst(dur, rng, amp=0.45)
        write_wav(bark_dir / f"bark_probe_{i:03d}.wav", audio)


def build_fake_features_and_embeddings(
    inventory_csv: Path, out_dir: Path, seed: int = 42, D: int = 32
) -> None:
    """Fabricate features.parquet/.csv + mean_emb_layer{9,12}.npy directly from
    an already-built segment_inventory.csv, bypassing WavLM (no torch needed).
    This lets stage2_analysis.py be tested end-to-end."""
    rng = np.random.default_rng(seed + 1)
    inv = pd.read_csv(inventory_csv)
    n = len(inv)

    # Give laugh-* groups a mean-shifted, higher-entropy/lower-compactness
    # profile than speech-* groups so the Wilcoxon test has real signal, and
    # laugh-spoof its own cluster so the H4 probe has signal too.
    group_mean_shift = {
        "laugh-bona": np.concatenate([[3.0], np.zeros(D - 1)]),
        "laugh-vs": np.concatenate([[3.2], np.zeros(D - 1)]),
        "speech-bona": np.zeros(D),
        "speech-spoof": np.zeros(D),
        "laugh-spoof": np.concatenate([[-3.0], np.zeros(D - 1)]),
    }

    mean_emb9 = np.full((n, D), np.nan, dtype=np.float32)
    mean_emb12 = np.full((n, D), np.nan, dtype=np.float32)
    records = []

    for row_index, row in inv.iterrows():
        group = row["group"]
        shift = group_mean_shift.get(group, np.zeros(D))
        emb9 = shift + rng.normal(scale=1.0, size=D)
        emb12 = shift + rng.normal(scale=1.0, size=D)
        mean_emb9[row_index] = emb9
        mean_emb12[row_index] = emb12

        is_laugh = group.startswith("laugh")
        C = float(rng.normal(loc=(-2.0 if is_laugh else -1.0), scale=0.3))
        T = float(np.clip(rng.normal(loc=(0.7 if is_laugh else 0.4), scale=0.1), 0, 1))
        mean_cos = float(np.clip(rng.normal(loc=(0.5 if is_laugh else 0.2), scale=0.1), 0, 2))

        records.append({
            "row_index": row_index,
            "group": group,
            "file_path": row["file_path"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
            "pair_id": row["pair_id"],
            "speaker": row.get("speaker", ""),
            "C_layer12": C,
            "T_layer9": T,
            "mean_cos_dist_layer12": mean_cos,
            "n_frames": int(rng.integers(20, 100)),
        })

    features_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"
    try:
        features_df.to_parquet(features_path, index=False)
    except Exception:
        features_path = out_dir / "features.csv"
        features_df.to_csv(features_path, index=False)

    np.save(out_dir / "mean_emb_layer9.npy", mean_emb9)
    np.save(out_dir / "mean_emb_layer12.npy", mean_emb12)
    print(f"Wrote fabricated {features_path}, mean_emb_layer9.npy, mean_emb_layer12.npy "
          f"({n} rows, D={D})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dummy data for end-to-end testing.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "_dummy")
    parser.add_argument("--n-bona", type=int, default=40)
    parser.add_argument("--n-spoof", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    itw_audio_dir = out_dir / "audio" / "itw"
    itw_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Bona-fide: high laughter prevalence (~60%), spoof: low (~15%) -- mimics H1.
    for i in range(args.n_bona):
        has_laughter = rng.uniform() < 0.6
        rows.append(build_itw_file(f"bona_{i:03d}", "bona-fide", has_laughter, rng, itw_audio_dir))
    for i in range(args.n_spoof):
        has_laughter = rng.uniform() < 0.15
        rows.append(build_itw_file(f"spoof_{i:03d}", "spoof", has_laughter, rng, itw_audio_dir))

    itw_df = pd.DataFrame(rows)
    itw_csv = out_dir / "itw_laughter.csv"
    itw_df.to_csv(itw_csv, index=False)
    print(f"Wrote {itw_csv} ({len(itw_df)} rows); "
          f"bona w/ laughter={sum((itw_df.label=='bona-fide') & (itw_df.n_laugh_segs>=1))}, "
          f"spoof w/ laughter={sum((itw_df.label=='spoof') & (itw_df.n_laugh_segs>=1))}")

    build_vocalsound(rng, out_dir, n=20)
    build_bark(rng, out_dir, n=4)
    print(f"Dummy data written under {out_dir}")


if __name__ == "__main__":
    main()
