"""Evaluate fixed-length WavLM laughter features on an unseen generator.

Unlike per-generator cross-validation, this trains on real laughter plus all
but one synthetic generator and tests on the held generator against unseen real
clips.  It therefore answers whether the representation contains a shared
synthetic-laughter direction, within the current single-source real anchor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_laughter_dynamics import REAL, SEED, active_audio

ROOT = Path(__file__).resolve().parents[1]


def embeddings_for(df: pd.DataFrame, layers: list[int], batch_size: int) -> dict[int, np.ndarray]:
    import torch
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"extracting WavLM-Large layers {layers} on {device}")
    # The research environment already has the model cached.  Cache-only
    # loading avoids failed metadata requests in offline runs and makes the
    # exact model dependency explicit.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-large", local_files_only=True
    )
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", output_hidden_states=True, local_files_only=True
    ).to(device).eval()
    out = {layer: [] for layer in layers}
    audio = [active_audio(path) for path in df.file_path]
    if any(wav is None for wav in audio):
        raise RuntimeError("input table contains a clip that no longer has 2 s active audio")
    with torch.no_grad():
        for start in range(0, len(audio), batch_size):
            values = feature_extractor(audio[start:start + batch_size], sampling_rate=16_000,
                                       return_tensors="pt", padding=True)["input_values"].to(device)
            hidden = model(values).hidden_states
            for layer in layers:
                out[layer].append(hidden[layer].mean(dim=1).cpu().numpy())
            print(f"  {min(start + batch_size, len(audio))}/{len(audio)}", flush=True)
    return {layer: np.concatenate(chunks) for layer, chunks in out.items()}


def held_generator_auc(X: np.ndarray, groups: np.ndarray, repeats: int) -> pd.DataFrame:
    rows = []
    methods = sorted(g for g in np.unique(groups) if g != REAL)
    real_indices = np.flatnonzero(groups == REAL)
    for method_i, method in enumerate(methods):
        synth_test = np.flatnonzero(groups == method)
        synth_train = np.flatnonzero((groups != REAL) & (groups != method))
        n = min(len(synth_test), len(real_indices) // 2)
        aucs = []
        for repeat in range(repeats):
            rng = np.random.default_rng(SEED + method_i * 100 + repeat)
            real_test = rng.choice(real_indices, size=n, replace=False)
            real_train = np.setdiff1d(real_indices, real_test)
            # Fit preprocessing only on training data; limit components because
            # the genuine training set is deliberately small and independent.
            train_indices = np.concatenate([real_train, synth_train])
            synth_test_sample = rng.choice(synth_test, size=n, replace=False)
            test_indices = np.concatenate([real_test, synth_test_sample])
            pca_dims = min(10, len(train_indices) - 1, X.shape[1])
            model = make_pipeline(StandardScaler(), PCA(n_components=pca_dims, random_state=SEED),
                                  LogisticRegression(max_iter=2_000, class_weight="balanced"))
            model.fit(X[train_indices], (groups[train_indices] != REAL).astype(int))
            scores = model.predict_proba(X[test_indices])[:, 1]
            aucs.append(roc_auc_score((groups[test_indices] != REAL).astype(int), scores))
        rows.append({"held_out_generator": method.removeprefix("laugh-"), "n_per_class": n,
                     "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "repeats": repeats})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=ROOT / "tables" / "laughter_dynamics_clips.csv", type=Path)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 3, 12])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default=ROOT / "tables" / "laughter_dynamics_wavlm_held_generator.csv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.clips)
    emb = embeddings_for(df, args.layers, args.batch_size)
    groups = df.group.to_numpy()
    rows = []
    for layer, X in emb.items():
        np.save(args.out.with_name(args.out.stem + f"_layer{layer}.npy"), X)
        result = held_generator_auc(X, groups, args.repeats)
        result.insert(0, "wavlm_layer", layer)
        rows.append(result)
    output = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print("\nWavLM leave-one-generator-out AUC:")
    print(output.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
