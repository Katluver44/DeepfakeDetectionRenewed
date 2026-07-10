"""Extract WavLM-Large frame-embedding geometric features for the segment inventory.

Implements laughsmi_plan.md §5.2. For each row of embeddings/segment_inventory.csv,
loads the corresponding audio segment, resamples to 16 kHz mono, runs it through
microsoft/wavlm-large with output_hidden_states=True, and computes:

  - C: negative radius of gyration of frame embeddings at layer 12 (compactness).
  - T: velocity entropy (Shannon entropy of the histogram of frame-to-frame
       delta norms, 32 bins by default) at layer 9 (trajectory irregularity).
  - mean_cos_dist_l12: mean frame-to-frame cosine distance at layer 12 (sanity metric).
  - Segment-mean embeddings at layers 9 and 12, saved as .npy arrays keyed by
    row index (for the UMAP visualization in stage2_analysis.py).

All the actual geometry math lives in geom_features.py (no torch import there),
so it can be unit-tested independently of torch/transformers availability.

Outputs:
    embeddings/features.parquet (falls back to embeddings/features.csv if
        pyarrow/fastparquet aren't available) with columns:
        row_index, group, file_path, start_s, end_s, pair_id, speaker,
        C_layer12, T_layer9, mean_cos_dist_layer12, n_frames
    embeddings/mean_emb_layer9.npy  -- (N, D) array, row i = segment-mean
        embedding at layer 9 for inventory row i (NaN row if extraction failed).
    embeddings/mean_emb_layer12.npy -- same, layer 12.

Resumability: if the output feature table already exists, rows whose
row_index is already present (and whose mean-embedding row is non-NaN) are
skipped, so a killed/restarted job resumes rather than recomputing everything.

Usage:
    python extract_wavlm_features.py \
        --inventory embeddings/segment_inventory.csv \
        --out-features embeddings/features.parquet \
        --out-emb-dir embeddings \
        --layers 9 12 \
        --model microsoft/wavlm-large \
        --device cuda \
        --fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geom_features import compactness_C, mean_frame_cosine_distance, velocity_entropy

DEFAULT_MODEL = "microsoft/wavlm-large"
TARGET_SR = 16000
LAYER_C = 12  # compactness (radius of gyration) layer
LAYER_T = 9   # velocity entropy layer


def load_audio_segment(file_path: str, start_s: float, end_s: float, target_sr: int = TARGET_SR):
    """Load a (possibly whole-clip) audio segment, resampled to target_sr mono.

    end_s == -1.0 is a sentinel meaning "whole clip" (used for laugh-vs and
    Bark probe clips in the inventory, which are stored as whole clips).

    Lazy-imports soundfile/librosa/numpy resampling so this module can be
    imported without those installed (only needed at actual extraction time).
    """
    import soundfile as sf

    info = sf.info(file_path)
    sr = info.samplerate
    if end_s is None or end_s < 0:
        start_frame = 0
        end_frame = info.frames
    else:
        start_frame = max(0, int(round(start_s * sr)))
        end_frame = min(info.frames, int(round(end_s * sr)))
        if end_frame <= start_frame:
            return None

    audio, _ = sf.read(file_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except ImportError:
            # Fallback: simple linear-interpolation resampling if librosa is
            # unavailable (lower quality, but keeps the pipeline runnable).
            n_target = int(round(len(audio) * target_sr / sr))
            if n_target <= 0:
                return None
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_target, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio


class WavLMExtractor:
    """Thin wrapper around a WavLM model + feature extractor. All torch/
    transformers imports happen inside __init__ so importing this module
    doesn't require torch to be installed."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda", fp16: bool = False):
        import torch
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        if device == "cuda" and self.device == "cpu":
            print("WARNING: CUDA requested but not available; falling back to CPU.", file=sys.stderr)

        self.fp16 = fp16 and self.device == "cuda"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

    def embed(self, audio: np.ndarray, layers: tuple[int, ...] = (LAYER_T, LAYER_C)) -> dict[int, np.ndarray]:
        """Run one audio array (16kHz mono float32) through the model.

        Returns a dict {layer_idx: (T, D) numpy array of frame embeddings}
        for each requested layer.
        """
        torch = self.torch
        inputs = self.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        if self.fp16:
            input_values = input_values.half()

        with torch.no_grad():
            outputs = self.model(input_values)
        hidden_states = outputs.hidden_states  # tuple: (num_layers+1, B, T, D)

        result = {}
        for layer in layers:
            h = hidden_states[layer][0]  # (T, D)
            result[layer] = h.float().cpu().numpy()
        return result


def compute_row_features(embeddings_by_layer: dict[int, np.ndarray]) -> dict:
    """Compute C, T, mean cosine distance, and mean embeddings from a segment's
    per-layer frame embeddings dict {9: (T,D), 12: (T,D)}."""
    emb_t = embeddings_by_layer[LAYER_T]
    emb_c = embeddings_by_layer[LAYER_C]

    C = compactness_C(emb_c)
    T = velocity_entropy(emb_t, n_bins=32)
    mean_cos_dist = mean_frame_cosine_distance(emb_c)
    mean_emb_9 = emb_t.mean(axis=0) if emb_t.shape[0] > 0 else None
    mean_emb_12 = emb_c.mean(axis=0) if emb_c.shape[0] > 0 else None

    return {
        "C_layer12": C,
        "T_layer9": T,
        "mean_cos_dist_layer12": mean_cos_dist,
        "n_frames": int(emb_c.shape[0]),
        "mean_emb_9": mean_emb_9,
        "mean_emb_12": mean_emb_12,
    }


def load_existing_features(out_features: Path):
    if out_features.suffix == ".parquet" and out_features.exists():
        try:
            return pd.read_parquet(out_features)
        except Exception:
            return None
    csv_fallback = out_features.with_suffix(".csv")
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)
    return None


def save_features(df: pd.DataFrame, out_features: Path) -> Path:
    """Save features to parquet if possible, else fall back to CSV. Returns
    the path actually written."""
    if out_features.suffix == ".parquet":
        try:
            df.to_parquet(out_features, index=False)
            return out_features
        except Exception as exc:
            print(f"WARNING: parquet write failed ({exc}); falling back to CSV.", file=sys.stderr)
            csv_path = out_features.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    else:
        df.to_csv(out_features, index=False)
        return out_features


def checkpoint_mean_embeddings(
    mean_emb_9_arr: np.ndarray | None,
    mean_emb_12_arr: np.ndarray | None,
    new_mean9: dict[int, np.ndarray],
    new_mean12: dict[int, np.ndarray],
    n_rows: int,
    out_emb_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Persist mean embeddings alongside the resumable feature table.

    A process can be interrupted between table checkpoints.  Saving these
    arrays at the same cadence makes the existing-resume contract true rather
    than recomputing all completed rows after an interruption.
    """
    if not new_mean9:
        return mean_emb_9_arr, mean_emb_12_arr
    d = next(iter(new_mean9.values())).shape[0]
    if mean_emb_9_arr is None:
        mean_emb_9_arr = np.full((n_rows, d), np.nan, dtype=np.float32)
        mean_emb_12_arr = np.full((n_rows, d), np.nan, dtype=np.float32)
    elif mean_emb_9_arr.shape[0] < n_rows:
        pad = np.full((n_rows - mean_emb_9_arr.shape[0], d), np.nan, dtype=np.float32)
        mean_emb_9_arr = np.vstack([mean_emb_9_arr, pad])
        mean_emb_12_arr = np.vstack([mean_emb_12_arr, pad])
    for idx, vec in new_mean9.items():
        mean_emb_9_arr[idx] = vec
        mean_emb_12_arr[idx] = new_mean12[idx]
    np.save(out_emb_dir / "mean_emb_layer9.npy", mean_emb_9_arr)
    np.save(out_emb_dir / "mean_emb_layer12.npy", mean_emb_12_arr)
    return mean_emb_9_arr, mean_emb_12_arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract WavLM-Large geometric features (C, T, mean cosine "
                     "distance, mean embeddings) for the Stage-2 segment inventory."
    )
    parser.add_argument("--inventory", type=Path, default=Path("embeddings/segment_inventory.csv"))
    parser.add_argument("--out-features", type=Path, default=Path("embeddings/features.parquet"))
    parser.add_argument("--out-emb-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-log-every", type=int, default=50)
    args = parser.parse_args()

    if not args.inventory.exists():
        print(f"ERROR: inventory not found: {args.inventory}", file=sys.stderr)
        sys.exit(1)

    inventory = pd.read_csv(args.inventory)
    inventory = inventory.reset_index(drop=True)
    n_rows = len(inventory)

    args.out_emb_dir.mkdir(parents=True, exist_ok=True)
    emb9_path = args.out_emb_dir / "mean_emb_layer9.npy"
    emb12_path = args.out_emb_dir / "mean_emb_layer12.npy"

    existing = load_existing_features(args.out_features)
    if existing is not None and emb9_path.exists() and emb12_path.exists():
        mean_emb_9_arr = np.load(emb9_path)
        mean_emb_12_arr = np.load(emb12_path)
        # A table checkpoint can predate its embedding-array checkpoint (for
        # example after an interrupted process).  Only skip rows whose feature
        # record *and* both mean embeddings are present.
        done_rows = {
            int(i) for i in existing["row_index"].tolist()
            if 0 <= int(i) < len(mean_emb_9_arr)
            and np.isfinite(mean_emb_9_arr[int(i)]).all()
            and np.isfinite(mean_emb_12_arr[int(i)]).all()
        }
        records = existing.drop_duplicates(subset="row_index", keep="last").to_dict("records")
        print(f"Resuming: {len(done_rows)}/{n_rows} rows already computed.")
    else:
        done_rows = set()
        mean_emb_9_arr = None
        mean_emb_12_arr = None
        records = []

    extractor = WavLMExtractor(model_name=args.model, device=args.device, fp16=args.fp16)

    try:
        from tqdm import tqdm
        iterator = tqdm(inventory.itertuples(index=False), total=n_rows, desc="extracting")
    except ImportError:
        iterator = inventory.itertuples(index=False)

    new_mean9 = {}
    new_mean12 = {}
    D = None

    for row_index, row in enumerate(iterator):
        if row_index in done_rows:
            continue
        row_d = row._asdict() if hasattr(row, "_asdict") else dict(zip(inventory.columns, row))

        audio = load_audio_segment(row_d["file_path"], row_d["start_s"], row_d["end_s"])
        if audio is None or len(audio) < 400:  # ~25ms min, avoid degenerate WavLM input
            print(f"WARNING: skipping row {row_index} ({row_d['file_path']}): "
                  f"could not load audio segment.", file=sys.stderr)
            continue

        try:
            emb_by_layer = extractor.embed(audio, layers=(LAYER_T, LAYER_C))
            feats = compute_row_features(emb_by_layer)
        except Exception as exc:
            print(f"WARNING: extraction failed for row {row_index} "
                  f"({row_d['file_path']}): {exc}", file=sys.stderr)
            continue

        mean_emb_9 = feats.pop("mean_emb_9")
        mean_emb_12 = feats.pop("mean_emb_12")
        if mean_emb_9 is not None:
            D = mean_emb_9.shape[0]
            new_mean9[row_index] = mean_emb_9
            new_mean12[row_index] = mean_emb_12

        record = {
            "row_index": row_index,
            "group": row_d.get("group"),
            "file_path": row_d.get("file_path"),
            "start_s": row_d.get("start_s"),
            "end_s": row_d.get("end_s"),
            "pair_id": row_d.get("pair_id"),
            "speaker": row_d.get("speaker"),
            **feats,
        }
        records.append(record)

        if args.batch_log_every and row_index % args.batch_log_every == 0:
            df_partial = pd.DataFrame(records).drop_duplicates(subset="row_index", keep="last")
            save_features(df_partial, args.out_features)
            mean_emb_9_arr, mean_emb_12_arr = checkpoint_mean_embeddings(
                mean_emb_9_arr, mean_emb_12_arr, new_mean9, new_mean12,
                n_rows, args.out_emb_dir,
            )

    df_out = pd.DataFrame(records).drop_duplicates(subset="row_index", keep="last")
    written_path = save_features(df_out, args.out_features)
    print(f"Wrote {written_path} ({len(df_out)} rows)")

    if D is not None or new_mean9:
        mean_emb_9_arr, mean_emb_12_arr = checkpoint_mean_embeddings(
            mean_emb_9_arr, mean_emb_12_arr, new_mean9, new_mean12,
            n_rows, args.out_emb_dir,
        )
        print(f"Wrote {emb9_path}, {emb12_path} (shape {mean_emb_9_arr.shape})")
    else:
        print("WARNING: no embeddings computed; mean-embedding .npy files not written.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
