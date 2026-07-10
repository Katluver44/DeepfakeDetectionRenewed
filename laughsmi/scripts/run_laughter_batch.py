#!/usr/bin/env python
"""
Stage 1 batch runner: Gillick laughter detector over a directory of audio files.

Implements laughsmi_plan.md section 4:
  - Loads the pretrained resnet_with_augmentation checkpoint ONCE (GPU if available).
  - For each file listed in --meta (columns: file, speaker, label), runs
    inference directly via the laughter-detection repo's model/feature code
    (no shelling out to segment_laughter.py, for speed).
  - Writes one CSV row per file:
      file_id,label,speaker,dur_s,n_laugh_segs,laugh_dur_s,laugh_ratio,
      max_prob,seg_starts,seg_ends
    (seg_starts / seg_ends are ';'-joined floats; max_prob is the max
    frame-level laughter probability over the whole file, post lowpass filter,
    so other threshold operating points can be derived later without
    re-running inference.)
  - Resumable: if --out already exists, already-processed file_ids are
    skipped and new rows are appended.
  - Robust: per-file exceptions are caught, logged to <out>.failures.log,
    and processing continues.

Example:
    python scripts/run_laughter_batch.py \\
        --audio_dir data/in_the_wild --meta data/in_the_wild/meta.csv \\
        --threshold 0.5 --min_length 0.2 --out detector_out/itw_laughter.csv
"""

import argparse
import csv
import os
import sys
import time
import traceback
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- Wire up the jrgillick laughter-detection repo -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LAUGHTER_DETECTION_DIR = os.path.normpath(
    os.path.join(THIS_DIR, "..", "laughter-detection")
)
UTILS_DIR = os.path.join(LAUGHTER_DETECTION_DIR, "utils")

for p in (LAUGHTER_DETECTION_DIR, UTILS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import audio_utils  # noqa: E402
import configs  # noqa: E402
import data_loaders  # noqa: E402
import laugh_segmenter  # noqa: E402
import torch_utils  # noqa: E402

SAMPLE_RATE = 8000  # native rate expected by the Gillick feature/model pipeline

CSV_FIELDS = [
    "file_id",
    "label",
    "speaker",
    "dur_s",
    "n_laugh_segs",
    "laugh_dur_s",
    "laugh_ratio",
    "max_prob",
    "seg_starts",
    "seg_ends",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", required=True, type=str,
                         help="Directory containing audio files referenced in --meta")
    parser.add_argument("--meta", required=True, type=str,
                         help="CSV with columns: file, speaker, label "
                              "(label in {bona-fide, spoof})")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Frame probability threshold for a laughter segment")
    parser.add_argument("--min_length", type=float, default=0.2,
                         help="Minimum segment length in seconds")
    parser.add_argument("--out", required=True, type=str,
                         help="Output CSV path (resumable: appends & skips done rows)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of files to process (testing)")
    parser.add_argument("--model_path", type=str,
                         default=os.path.join(LAUGHTER_DETECTION_DIR,
                                               "checkpoints/in_use/resnet_with_augmentation"),
                         help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, default="resnet_with_augmentation",
                         help="Key into laughter-detection configs.CONFIG_MAP")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="DataLoader batch size for frame-level inference")
    parser.add_argument("--num_workers", type=int, default=2,
                         help="DataLoader num_workers")
    return parser.parse_args()


def load_model(config_key, model_path, device):
    config = configs.CONFIG_MAP[config_key]
    model = config["model"](
        dropout_rate=0.0,
        linear_layer_size=config["linear_layer_size"],
        filter_sizes=config["filter_sizes"],
    )
    model.set_device(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    torch_utils.load_checkpoint(os.path.join(model_path, "best.pth.tar"), model)
    model.eval()
    return model, config


def get_audio_duration(path):
    return audio_utils.get_audio_length(path)


def run_inference_on_file(model, config, device, audio_path, batch_size, num_workers):
    """Returns (probs_filtered, fps, dur_s) for one audio file."""
    feature_fn = config["feature_fn"]

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=SAMPLE_RATE
    )

    collate_fn = partial(
        audio_utils.pad_sequences_with_labels,
        expand_channel_dim=config["expand_channel_dim"],
    )

    try:
        ds_len = len(inference_dataset)
    except ValueError:
        # __len__ computed a negative value (file shorter than one window);
        # len() raises before returning it.
        ds_len = 0
    if ds_len <= 0:
        # File too short to produce even one frame window.
        dur_s = get_audio_duration(audio_path)
        return np.array([]), 0.0, dur_s

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    probs = []
    with torch.no_grad():
        for model_inputs, _ in inference_generator:
            if model_inputs is None:
                continue
            x = torch.from_numpy(model_inputs).float().to(device)
            preds = model(x).cpu().detach().numpy().squeeze()
            if preds.ndim == 0:
                preds = [float(preds)]
            else:
                preds = list(preds)
            probs += preds
    probs = np.array(probs)

    dur_s = get_audio_duration(audio_path)
    fps = len(probs) / float(dur_s) if dur_s > 0 else 0.0

    if len(probs) > 0:
        probs = laugh_segmenter.lowpass(probs)

    return probs, fps, dur_s


def process_one_file(model, config, device, audio_path, threshold, min_length,
                      batch_size, num_workers):
    probs, fps, dur_s = run_inference_on_file(
        model, config, device, audio_path, batch_size, num_workers
    )

    if len(probs) == 0:
        return {
            "dur_s": dur_s,
            "n_laugh_segs": 0,
            "laugh_dur_s": 0.0,
            "laugh_ratio": 0.0,
            "max_prob": 0.0,
            "seg_starts": "",
            "seg_ends": "",
        }

    instances = laugh_segmenter.get_laughter_instances(
        probs, threshold=threshold, min_length=min_length, fps=fps
    )

    seg_starts = [round(float(s), 4) for s, _ in instances]
    seg_ends = [round(float(e), 4) for _, e in instances]
    laugh_dur_s = float(sum(e - s for s, e in instances))
    laugh_ratio = laugh_dur_s / dur_s if dur_s > 0 else 0.0
    max_prob = float(np.max(probs)) if len(probs) > 0 else 0.0

    return {
        "dur_s": dur_s,
        "n_laugh_segs": len(instances),
        "laugh_dur_s": laugh_dur_s,
        "laugh_ratio": laugh_ratio,
        "max_prob": max_prob,
        "seg_starts": ";".join(str(x) for x in seg_starts),
        "seg_ends": ";".join(str(x) for x in seg_ends),
    }


def load_done_ids(out_path):
    if not os.path.exists(out_path):
        return set()
    try:
        df = pd.read_csv(out_path)
        if "file_id" in df.columns:
            return set(df["file_id"].astype(str).tolist())
    except Exception:
        pass
    return set()


def resolve_audio_path(audio_dir, file_id):
    """Try the file_id as-is, then with common extensions appended."""
    candidate = os.path.join(audio_dir, file_id)
    if os.path.exists(candidate):
        return candidate
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        c2 = candidate + ext
        if os.path.exists(c2):
            return c2
    return candidate  # fall through; will raise FileNotFoundError downstream


def main():
    args = parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    failures_log_path = args.out + ".failures.log"

    meta = pd.read_csv(args.meta)
    required_cols = {"file", "speaker", "label"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"--meta is missing required columns: {missing}")

    if args.limit is not None:
        meta = meta.iloc[: args.limit]

    done_ids = load_done_ids(args.out)
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(args.config, args.model_path, device)

    n_total = len(meta)
    n_skipped = 0
    n_processed = 0
    n_failed = 0

    with open(args.out, "a", newline="") as out_f, \
         open(failures_log_path, "a") as fail_f:

        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            out_f.flush()

        pbar = tqdm(meta.itertuples(index=False), total=n_total, desc="laughter-batch")
        for row in pbar:
            row_d = row._asdict()
            file_id = str(row_d["file"])
            speaker = row_d["speaker"]
            label = row_d["label"]

            if file_id in done_ids:
                n_skipped += 1
                continue

            try:
                audio_path = resolve_audio_path(args.audio_dir, file_id)
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                result = process_one_file(
                    model, config, device, audio_path,
                    threshold=args.threshold, min_length=args.min_length,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                )

                out_row = {
                    "file_id": file_id,
                    "label": label,
                    "speaker": speaker,
                    **result,
                }
                writer.writerow(out_row)
                out_f.flush()
                n_processed += 1

            except Exception as e:
                n_failed += 1
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                fail_f.write(f"[{ts}] {file_id}: {repr(e)}\n")
                fail_f.write(traceback.format_exc())
                fail_f.write("\n")
                fail_f.flush()
                continue

            pbar.set_postfix(done=n_processed, skipped=n_skipped, failed=n_failed)

    print(
        f"Finished. total={n_total} processed={n_processed} "
        f"skipped(resume)={n_skipped} failed={n_failed}"
    )
    if n_failed > 0:
        print(f"See failures log: {failures_log_path}")


if __name__ == "__main__":
    main()
