"""Characterize verified real and synthetic laughter with interpretable dynamics.

This is deliberately separate from the raw WavLM separability probes.  It uses
active-audio normalization, reports acoustic effects directly, and evaluates a
descriptor-only probe on a generator held out during training.  The latter is
the relevant test for a generator-agnostic claim; it is not a cross-corpus
real-laughter generalization test because the current real anchor is one
source.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000
SEED = 20260710
REAL = "laugh-real"


def active_audio(path: str, seconds: float = 2.0) -> np.ndarray | None:
    """Trim boundary silence and take a fixed active centre; reject short clips.

    Rejecting rather than tiling clips avoids creating artificial periodicity at
    the boundary.  All retained inputs therefore contain the same amount of
    active audio and have equal peak level.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    peak = float(np.max(np.abs(audio))) + 1e-9
    active = np.flatnonzero(np.abs(audio) > 0.01 * peak)
    wanted = int(seconds * SR)
    if len(active) == 0:
        return None
    audio = audio[active[0]:active[-1] + 1]
    if len(audio) < wanted:
        return None
    start = (len(audio) - wanted) // 2
    audio = audio[start:start + wanted]
    return (audio / (np.max(np.abs(audio)) + 1e-9) * 0.9).astype(np.float32)


def _cv(values: np.ndarray) -> float:
    return float(np.std(values) / (np.mean(values) + 1e-9))


def descriptors(audio: np.ndarray) -> dict[str, float]:
    """Measures tied to laugh call timing, voicing, and acoustic texture."""
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=SR, frame_length=1024)
    f0_voiced = f0[~np.isnan(f0)]
    voiced_frac = float(np.mean(voiced)) if voiced is not None else np.nan

    onset_frames = librosa.onset.onset_detect(y=audio, sr=SR, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=SR)
    intervals = np.diff(onset_times)
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    S_norm = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-9)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
    harmonic, percussive = librosa.effects.hpss(audio)
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    env = rms / (np.mean(rms) + 1e-9)
    env_entropy = -np.sum((env / env.sum()) * np.log(env / env.sum() + 1e-12)) / np.log(len(env))

    return {
        "onset_rate_hz": len(onset_times) / (len(audio) / SR),
        "inter_onset_cv": _cv(intervals) if len(intervals) >= 2 else np.nan,
        "voiced_fraction": voiced_frac,
        "f0_mean_hz": float(np.mean(f0_voiced)) if len(f0_voiced) else np.nan,
        "f0_cv": _cv(f0_voiced) if len(f0_voiced) >= 2 else np.nan,
        "envelope_cv": _cv(rms),
        "envelope_entropy": float(env_entropy),
        "harmonic_percussive_ratio": float(np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-9)),
        "spectral_flux": float(np.mean(flux)),
    }


def cliffs_delta(real: np.ndarray, synth: np.ndarray) -> float:
    # Positive means larger values for real laughter.
    return float((np.greater.outer(real, synth).sum() - np.less.outer(real, synth).sum()) / (len(real) * len(synth)))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    values = np.asarray(pvals, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank in range(len(values), 0, -1):
        idx = order[rank - 1]
        running = min(running, values[idx] * len(values) / rank)
        adjusted[idx] = running
    return adjusted.tolist()


def held_generator_auc(df: pd.DataFrame, feature_names: list[str], repeats: int) -> pd.DataFrame:
    """Train on all but one generator; test it against unseen real clips."""
    rows = []
    methods = sorted(g for g in df.group.unique() if g.startswith("laugh-") and g != REAL)
    for method_i, method in enumerate(methods):
        synth_test = df[df.group == method]
        synth_train = df[(df.group != REAL) & (df.group != method)]
        real = df[df.group == REAL]
        aucs = []
        # Reserve genuine clips for both training and testing.  Reusing the
        # same real examples on both sides would inflate held-generator AUC.
        n = min(len(synth_test), len(real) // 2)
        for repeat in range(repeats):
            rng = np.random.default_rng(SEED + method_i * 100 + repeat)
            real_test_idx = rng.choice(real.index.to_numpy(), size=n, replace=False)
            real_train = real.drop(index=real_test_idx)
            # Cap each synthetic training method to avoid its sample count dominating.
            train_parts = [real_train]
            for grp, part in synth_train.groupby("group"):
                take = min(len(part), len(real_train))
                train_parts.append(part.iloc[rng.choice(len(part), size=take, replace=False)])
            train = pd.concat(train_parts)
            synth_test_idx = rng.choice(synth_test.index.to_numpy(), size=n, replace=False)
            test = pd.concat([real.loc[real_test_idx], synth_test.loc[synth_test_idx]])
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, class_weight="balanced"))
            model.fit(train[feature_names], (train.group != REAL).astype(int))
            aucs.append(roc_auc_score((test.group != REAL).astype(int), model.predict_proba(test[feature_names])[:, 1]))
        rows.append({"held_out_generator": method.removeprefix("laugh-"), "n_per_class": n,
                     "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "repeats": repeats})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=ROOT / "embeddings" / "d3_multi_inventory.csv", type=Path)
    ap.add_argument("--per-group", type=int, default=20,
                    help="balanced clips per group; 20 is the strict active-audio limit across generators")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out-prefix", default=ROOT / "tables" / "laughter_dynamics", type=Path)
    args = ap.parse_args()

    inv = pd.read_csv(args.inventory)
    rng = np.random.default_rng(SEED)
    selected = []
    for group, part in inv[inv.group.str.startswith("laugh-")].groupby("group", sort=True):
        valid = []
        for row in part.itertuples(index=False):
            audio = active_audio(row.file_path)
            if audio is not None:
                valid.append((row, audio))
        if len(valid) < args.per_group:
            raise SystemExit(f"{group}: only {len(valid)} clips have >=2 s active audio; need {args.per_group}")
        for idx in rng.choice(len(valid), args.per_group, replace=False):
            row, audio = valid[idx]
            selected.append({"group": group, "file_path": row.file_path, "speaker": row.speaker, **descriptors(audio)})

    df = pd.DataFrame(selected)
    feature_names = [c for c in df.columns if c not in ("group", "file_path", "speaker")]
    if df[feature_names].isna().any().any():
        # Inter-onset CV / F0 can be undefined for one-call laughs. Median
        # imputation is fitted only on the selected study set, then documented.
        df[feature_names] = df[feature_names].fillna(df[feature_names].median())
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_clips.csv"), index=False)

    real = df[df.group == REAL]
    synth = df[df.group != REAL]
    rows, pvals = [], []
    for feature in feature_names:
        a, b = real[feature].to_numpy(), synth[feature].to_numpy()
        test = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": feature, "real_mean": float(a.mean()), "synth_mean": float(b.mean()),
                     "cliffs_delta_real_gt_synth": cliffs_delta(a, b), "mannwhitney_p": float(test.pvalue)})
        pvals.append(float(test.pvalue))
    summary = pd.DataFrame(rows)
    summary["bh_fdr_q"] = benjamini_hochberg(pvals)
    summary.sort_values("mannwhitney_p").to_csv(args.out_prefix.with_name(args.out_prefix.name + "_effects.csv"), index=False)
    held = held_generator_auc(df, feature_names, args.repeats)
    held.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_held_generator.csv"), index=False)

    print(f"selected {len(df)} verified laughter clips: {df.group.value_counts().to_dict()}")
    print("\nDescriptor effects, positive delta = higher in real:")
    print(summary.sort_values("mannwhitney_p").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nDescriptor-only leave-one-generator-out AUC:")
    print(held.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
