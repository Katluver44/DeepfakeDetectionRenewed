"""Per-generator, literature-aligned acoustic analysis of verified laughter.

The spontaneous/volitional laughter literature motivates full-bout measures:
duration, F0 level/range/variability, unvoiced material, voiced-call duration,
inter-voicing intervals, harmonicity, temporal regularity, and spectral centre
of gravity.  This script retains each active bout instead of centre-cropping it
and compares the real anchor with each generator separately.  Its
autocorrelation harmonicity measure is a transparent proxy; final paper values
should be replicated with a Praat/Parselmouth implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000
HOP = 256
FRAME = 1_024
REAL = "laugh-real"


def active_bout(path: str) -> tuple[np.ndarray, float] | None:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    peak = float(np.max(np.abs(audio))) + 1e-9
    idx = np.flatnonzero(np.abs(audio) > 0.01 * peak)
    if len(idx) == 0:
        return None
    audio = audio[idx[0]:idx[-1] + 1]
    return (audio / (np.max(np.abs(audio)) + 1e-9) * 0.9).astype(np.float32), len(audio) / SR


def run_lengths(mask: np.ndarray, value: bool) -> np.ndarray:
    runs, start = [], None
    for i, current in enumerate(mask):
        if current == value and start is None:
            start = i
        if current != value and start is not None:
            runs.append(i - start)
            start = None
    if start is not None:
        runs.append(len(mask) - start)
    return np.asarray(runs, dtype=float) * HOP / SR


def autocorr_harmonicity(audio: np.ndarray) -> float:
    """Framewise periodicity in dB, analogous to but not identical to Praat HNR."""
    frames = librosa.util.frame(audio, frame_length=FRAME, hop_length=HOP).T
    values = []
    min_lag, max_lag = int(SR / 500), int(SR / 60)
    for frame in frames:
        frame = frame - frame.mean()
        energy = np.dot(frame, frame)
        if energy < 1e-7:
            continue
        acf = np.correlate(frame, frame, mode="full")[len(frame) - 1:] / energy
        periodicity = float(np.max(acf[min_lag:max_lag + 1]))
        if periodicity > 0:
            values.append(10 * np.log10(periodicity / max(1e-6, 1 - periodicity)))
    return float(np.median(values)) if values else np.nan


def features(audio: np.ndarray, duration_s: float) -> dict[str, float]:
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = np.asarray(voiced, dtype=bool)
    f0v = f0[~np.isnan(f0)]
    uv_runs, v_runs = run_lengths(voiced, False), run_lengths(voiced, True)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=FRAME, hop_length=HOP)[0]
    onset_times = librosa.onset.onset_detect(y=audio, sr=SR, hop_length=HOP, units="time")
    intervals = np.diff(onset_times)
    return {
        "bout_duration_s": duration_s,
        "voiced_fraction": float(voiced.mean()),
        "unvoiced_run_mean_s": float(uv_runs.mean()) if len(uv_runs) else np.nan,
        "voiced_burst_mean_s": float(v_runs.mean()) if len(v_runs) else np.nan,
        "voicing_transition_rate_hz": float(np.count_nonzero(np.diff(voiced.astype(int))) / duration_s),
        "f0_mean_hz": float(np.mean(f0v)) if len(f0v) else np.nan,
        "f0_range_hz": float(np.max(f0v) - np.min(f0v)) if len(f0v) else np.nan,
        "f0_std_hz": float(np.std(f0v)) if len(f0v) else np.nan,
        "harmonicity_acf_db": autocorr_harmonicity(audio),
        "spectral_cog_hz": float(np.mean(centroid)),
        "spectral_cog_std_hz": float(np.std(centroid)),
        "inter_onset_cv": float(np.std(intervals) / (np.mean(intervals) + 1e-9)) if len(intervals) >= 2 else np.nan,
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=ROOT / "tables" / "laughter_dynamics_clips.csv", type=Path)
    ap.add_argument("--out", default=ROOT / "tables" / "laughter_literature_features.csv", type=Path)
    args = ap.parse_args()
    clips = pd.read_csv(args.clips)
    rows = []
    for row in clips.itertuples(index=False):
        loaded = active_bout(row.file_path)
        if loaded is None:
            continue
        audio, duration = loaded
        rows.append({"group": row.group, "file_path": row.file_path, **features(audio, duration)})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out.with_name(args.out.stem + "_clips.csv"), index=False)

    real = frame[frame.group == REAL]
    result = []
    for group in sorted(g for g in frame.group.unique() if g != REAL):
        synth = frame[frame.group == group]
        for feature in frame.columns[2:]:
            a = real[feature].dropna().to_numpy()
            b = synth[feature].dropna().to_numpy()
            if min(len(a), len(b)) < 4:
                continue
            test = mannwhitneyu(a, b, alternative="two-sided")
            result.append({"generator": group.removeprefix("laugh-"), "feature": feature,
                           "real_mean": float(a.mean()), "synthetic_mean": float(b.mean()),
                           "cliffs_delta_real_gt_synth": cliffs_delta(a, b), "p_value": float(test.pvalue),
                           "n_real": len(a), "n_synthetic": len(b)})
    result = pd.DataFrame(result).sort_values(["generator", "p_value"])
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(result)} per-generator comparisons)")
    print(result.groupby("generator").head(4).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
