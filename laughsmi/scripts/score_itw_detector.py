"""Stage 3 (laughsmi_plan.md §6): score In-the-Wild audio with the trained
WavLM -> BiLSTM -> GAT deepfake detector (`Phoneme_GAT_lit`, defined in
`phoneme_GAT/modules.py`, checkpoints trained/saved as PyTorch Lightning
`.ckpt` files in `models/`).

Checkpoint loading
------------------
The checkpoint's `hyper_parameters` (saved via `save_hyperparameters()` in
`Phoneme_GAT_lit.__init__`) contain a `cfg` Namespace with the exact
architecture config (backbone, use_GAT, n_edges, use_aug, use_pool, use_clip)
used at train time; this script reads that back out of the checkpoint file
before constructing the model, so config mismatches can't silently corrupt
scores.

The `Phoneme_GAT` submodule's constructor unconditionally tries to build its
frozen "phoneme_model" backbone from a hardcoded, non-existent path
(`/lambda/nfs/.../vocab_phoneme` and `pretrained/best-epoch=42-...ckpt`, see
`phoneme_GAT/phoneme_model.py`). Since `load_from_checkpoint` overlays the
*full* state_dict (including that frozen submodule's weights) immediately
after construction, we monkeypatch `load_phoneme_model` to build a bare
`BaseModule` (WavLM CTC head, no tokenizer) instead of touching the broken
paths -- this is the same monkeypatch used elsewhere in this repo, e.g.
`experiments/scripts/i1_geometry_causal_decomp.py` and
`experiments/scripts/px_common.py`.

Score semantics (confirmed empirically on ITW bona-fide/spoof files, see
laughsmi_plan.md §6 report)
------------------------------------------------------------------------
- The model outputs a single scalar **logit** per utterance (`out["logit"]`),
  trained with `nn.BCEWithLogitsLoss()` against label 0=bona-fide, 1=spoof
  (`Phoneme_GAT_lit.calcuate_loss`). So:
      higher logit / prob  => model thinks SPOOF
      lower  logit / prob  => model thinks BONA-FIDE (real)
  `score` in this script's output CSV is `sigmoid(logit)` in [0, 1] (a spoof
  probability); the raw logit is also written for reference.
- This is a per-utterance (not per-frame) score: the whole waveform is fed
  through WavLM -> encoder -> adaptive phoneme pooling -> GAT -> BiLSTM ->
  mean-pool -> linear classifier head in one shot. There is no native
  per-frame/segment score. See `--frame-scores-dir` below for how we derive
  frame-level scores anyway (sliding-window re-scoring), used for Stage 3
  step 2 (per-frame score traces).

Modes
-----
1. Default: score every file in `--meta` (bona-fide/spoof CSV with columns
   file,speaker,label), write `file_id,label,speaker,score,logit` to `--out`.
   Resumable: if `--out` already exists, already-scored file_ids are skipped
   and new rows are appended.
2. `--mask-laughter --laughter-csv <path>`: only for files with
   `n_laugh_segs >= 1` in the laughter CSV, splice OUT the laughter segments
   (concatenate the remaining non-laughter audio) before scoring. Output CSV
   has the same laughter-CSV schema plus a `score_masked` (and
   `logit_masked`) column.
3. `--frame-scores-dir <dir>`: additionally save a per-file .npy of
   sliding-window scores (only for file_ids listed in `--files-subset`, since
   saving arrays for all ~31k files is wasteful). Window/hop are configurable
   (`--window-s` / `--hop-s`, default 1.0 s / 0.25 s).

Usage
-----
    python score_itw_detector.py \
        --audio_dir data/in_the_wild --meta data/in_the_wild/meta.csv \
        --ckpt ../models/asv19-wavlm-gat-full.ckpt \
        --out detector_out/itw_scores_asv19.csv --batch-size 8

    python score_itw_detector.py \
        --audio_dir data/in_the_wild --meta data/in_the_wild/meta.csv \
        --ckpt ../models/asv19-wavlm-gat-full.ckpt \
        --mask-laughter --laughter-csv detector_out/itw_laughter.csv \
        --out detector_out/itw_scores_asv19_masked.csv

    python score_itw_detector.py \
        --audio_dir data/in_the_wild --meta data/in_the_wild/meta.csv \
        --ckpt ../models/asv19-wavlm-gat-full.ckpt \
        --out detector_out/itw_scores_asv19.csv \
        --frame-scores-dir detector_out/frame_scores_asv19 \
        --files-subset detector_out/laughter_examples.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../DeepfakeDetectionRenewed
LAUGHSMI_ROOT = Path(__file__).resolve().parents[1]  # .../DeepfakeDetectionRenewed/laughsmi

TARGET_SR = 16000
TARGET_LEN = 3 * TARGET_SR  # 48000 samples = 3s, the fixed input length the model was trained on


# ============================================================================
# Model loading
# ============================================================================

def _patch_load_phoneme_model():
    """See module docstring: bypass the checkpoint's hardcoded/broken paths.

    `load_from_checkpoint` overlays the full state_dict right after
    construction, so the values used to build the placeholder BaseModule here
    (network_name/vocab_size) don't matter as long as shapes end up matching
    what's in the checkpoint (they do: total_num_phonemes=687 is hardcoded in
    `Phoneme_GAT.__init__` regardless of this patch).
    """
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        return BaseModule(
            network_param, optim_param, tokenizer=None,
            total_num_phonemes=total_num_phonemes,
        )

    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load


def load_detector(ckpt_path: str, device: str):
    """Load a Phoneme_GAT_lit checkpoint, reading its own saved cfg back out
    of hyper_parameters so we reconstruct the exact architecture it was
    trained with (rather than guessing n_edges etc.)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _patch_load_phoneme_model()

    from argparse import Namespace
    torch.serialization.add_safe_globals([Namespace])

    from phoneme_GAT.modules import Phoneme_GAT_lit

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = raw.get("hyper_parameters", {}).get("cfg", None)
    if saved_cfg is not None and hasattr(saved_cfg, "PhonemeGAT"):
        cfg = saved_cfg
    else:
        cfg = Namespace(PhonemeGAT=Namespace(
            backbone="wavlm", use_raw=False, use_GAT=True, n_edges=10,
            use_aug=True, use_pool=True, use_clip=True,
        ))
    del raw

    lit = Phoneme_GAT_lit.load_from_checkpoint(
        ckpt_path, cfg=cfg, map_location=device, strict=True
    )
    lit.to(device)
    lit.eval()
    return lit


# ============================================================================
# Audio IO / cropping
# ============================================================================

def load_wav_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def crop_or_tile_center(wav: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Paper eval policy: center crop to target_len; tile-pad if shorter."""
    T = len(wav)
    if T == 0:
        return np.zeros(target_len, dtype=np.float32)
    if T < target_len:
        reps = -(-target_len // T)
        wav = np.tile(wav, reps)[:target_len]
        return wav
    start = (T - target_len) // 2
    return wav[start:start + target_len]


def splice_out_segments(wav: np.ndarray, sr: int, seg_starts: list, seg_ends: list) -> np.ndarray:
    """Remove [start,end) spans (seconds) from wav, concatenating what's left."""
    if not seg_starts:
        return wav
    T = len(wav)
    spans = sorted(zip(seg_starts, seg_ends))
    keep = []
    cursor = 0
    for s, e in spans:
        s_idx = max(0, min(T, int(round(s * sr))))
        e_idx = max(0, min(T, int(round(e * sr))))
        if s_idx > cursor:
            keep.append(wav[cursor:s_idx])
        cursor = max(cursor, e_idx)
    if cursor < T:
        keep.append(wav[cursor:T])
    if not keep:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(keep)


# ============================================================================
# Scoring
# ============================================================================

@torch.no_grad()
def score_batch(lit, wavs: list, device: str) -> list:
    """wavs: list of 1-D float32 np arrays already cropped/padded to TARGET_LEN.
    Returns list of (logit, prob) tuples."""
    x = torch.from_numpy(np.stack(wavs, axis=0)).to(device)  # (B, T)
    B = x.shape[0]
    num_frames = torch.full((B,), TARGET_LEN // 320 - 1)
    out = lit.model(
        x, num_frames, profiler=None, use_aug=False, stage="val",
        ground_truth_labels=None,
    )
    logits = out["logit"].float().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return list(zip(logits.tolist(), probs.tolist()))


@torch.no_grad()
def sliding_window_scores(lit, wav: np.ndarray, device: str, window_s: float, hop_s: float) -> tuple:
    """Derive frame-level scores from an inherently utterance-level model via
    sliding-window re-scoring: score each window independently, tile/pad
    windows shorter than TARGET_LEN up to the model's fixed 3s input length.
    Returns (times_s, probs) where times_s are window CENTER times."""
    win = int(round(window_s * TARGET_SR))
    hop = int(round(hop_s * TARGET_SR))
    T = len(wav)
    if T <= 0:
        return np.array([]), np.array([])

    starts = list(range(0, max(1, T - win + 1), hop))
    if not starts or starts[-1] + win < T:
        last_start = max(0, T - win)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    windows = []
    centers = []
    for s in starts:
        e = min(T, s + win)
        seg = wav[s:e]
        seg = crop_or_tile_center(seg, TARGET_LEN)
        windows.append(seg)
        centers.append((s + e) / 2.0 / TARGET_SR)

    probs = []
    bs = 16
    for i in range(0, len(windows), bs):
        chunk = windows[i:i + bs]
        res = score_batch(lit, chunk, device)
        probs.extend(p for _, p in res)

    return np.array(centers), np.array(probs)


# ============================================================================
# CSV helpers
# ============================================================================

def read_meta(meta_path: str) -> list:
    with open(meta_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_laughter_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_segs(row: dict) -> tuple:
    starts_raw = (row.get("seg_starts") or "").strip()
    ends_raw = (row.get("seg_ends") or "").strip()
    if not starts_raw:
        return [], []
    starts = [float(x) for x in starts_raw.split(";") if x != ""]
    ends = [float(x) for x in ends_raw.split(";") if x != ""]
    return starts, ends


def already_done_ids(out_path: str, key_col: str = "file_id") -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        with open(out_path, newline="", encoding="utf-8") as f:
            return {r[key_col] for r in csv.DictReader(f)}
    except Exception:
        return set()


# ============================================================================
# Main modes
# ============================================================================

def run_default_mode(args, lit, device):
    rows = read_meta(args.meta)
    if args.limit:
        rows = rows[:args.limit]

    done = already_done_ids(args.out, key_col="file_id")
    todo = [r for r in rows if r["file"] not in done]
    print(f"[score_itw_detector] {len(rows)} total, {len(done)} already scored, {len(todo)} to do")

    fields = ["file_id", "label", "speaker", "score", "logit"]
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fail_log = args.out + ".failures.log"

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(todo), args.batch_size), desc="scoring")
    except ImportError:
        iterator = range(0, len(todo), args.batch_size)

    with open(args.out, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        if write_header:
            writer.writeheader()
            fout.flush()

        for i in iterator:
            batch_rows = todo[i:i + args.batch_size]
            wavs = []
            ok_rows = []
            for r in batch_rows:
                path = os.path.join(args.audio_dir, r["file"])
                try:
                    wav = load_wav_mono_16k(path)
                    wav = crop_or_tile_center(wav)
                    wavs.append(wav)
                    ok_rows.append(r)
                except Exception:
                    with open(fail_log, "a") as flog:
                        flog.write(f"{r['file']}\t{traceback.format_exc()}\n")

            if not wavs:
                continue

            try:
                scored = score_batch(lit, wavs, device)
            except Exception:
                with open(fail_log, "a") as flog:
                    for r in ok_rows:
                        flog.write(f"{r['file']}\t{traceback.format_exc()}\n")
                continue

            for r, (logit, prob) in zip(ok_rows, scored):
                writer.writerow({
                    "file_id": r["file"], "label": r["label"], "speaker": r.get("speaker", ""),
                    "score": prob, "logit": logit,
                })
            fout.flush()


def run_mask_mode(args, lit, device):
    laugh_rows = read_laughter_csv(args.laughter_csv)
    laugh_rows = [r for r in laugh_rows if int(float(r.get("n_laugh_segs", 0) or 0)) >= 1]
    if args.limit:
        laugh_rows = laugh_rows[:args.limit]

    done = already_done_ids(args.out, key_col="file_id")
    todo = [r for r in laugh_rows if r["file_id"] not in done]
    print(f"[score_itw_detector:mask] {len(laugh_rows)} files w/ laughter, "
          f"{len(done)} already scored, {len(todo)} to do")

    fields = ["file_id", "label", "speaker", "dur_s", "n_laugh_segs", "laugh_dur_s",
              "laugh_ratio", "max_prob", "seg_starts", "seg_ends", "score_masked", "logit_masked"]
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fail_log = args.out + ".failures.log"

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(todo), args.batch_size), desc="masking+scoring")
    except ImportError:
        iterator = range(0, len(todo), args.batch_size)

    with open(args.out, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        if write_header:
            writer.writeheader()
            fout.flush()

        for i in iterator:
            batch_rows = todo[i:i + args.batch_size]
            wavs, ok_rows = [], []
            for r in batch_rows:
                path = os.path.join(args.audio_dir, r["file_id"])
                try:
                    wav = load_wav_mono_16k(path)
                    starts, ends = parse_segs(r)
                    wav = splice_out_segments(wav, TARGET_SR, starts, ends)
                    wav = crop_or_tile_center(wav)
                    wavs.append(wav)
                    ok_rows.append(r)
                except Exception:
                    with open(fail_log, "a") as flog:
                        flog.write(f"{r['file_id']}\t{traceback.format_exc()}\n")

            if not wavs:
                continue

            try:
                scored = score_batch(lit, wavs, device)
            except Exception:
                with open(fail_log, "a") as flog:
                    for r in ok_rows:
                        flog.write(f"{r['file_id']}\t{traceback.format_exc()}\n")
                continue

            for r, (logit, prob) in zip(ok_rows, scored):
                out_row = {k: r.get(k, "") for k in fields if k not in ("score_masked", "logit_masked")}
                out_row["score_masked"] = prob
                out_row["logit_masked"] = logit
                writer.writerow(out_row)
            fout.flush()


def run_frame_scores_mode(args, lit, device):
    if not args.files_subset:
        raise ValueError("--frame-scores-dir requires --files-subset (a CSV with a file_id/file column) "
                          "to avoid saving per-frame arrays for all ~31k files.")
    with open(args.files_subset, newline="", encoding="utf-8") as f:
        subset_rows = list(csv.DictReader(f))
    key = "file_id" if "file_id" in (subset_rows[0].keys() if subset_rows else []) else "file"

    out_dir = Path(args.frame_scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        iterator = tqdm(subset_rows, desc="frame-scoring")
    except ImportError:
        iterator = subset_rows

    fail_log = str(out_dir / "failures.log")
    for r in iterator:
        file_id = r[key]
        npy_path = out_dir / (Path(file_id).stem + ".npy")
        if npy_path.exists():
            continue
        path = os.path.join(args.audio_dir, file_id)
        try:
            wav = load_wav_mono_16k(path)
            centers, probs = sliding_window_scores(
                lit, wav, device, window_s=args.window_s, hop_s=args.hop_s
            )
            np.save(npy_path, np.stack([centers, probs], axis=0).astype(np.float32))
        except Exception:
            with open(fail_log, "a") as flog:
                flog.write(f"{file_id}\t{traceback.format_exc()}\n")


# ============================================================================
# CLI
# ============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio_dir", required=True, help="Directory containing ITW wav files")
    p.add_argument("--meta", required=True, help="meta.csv with columns file,speaker,label")
    p.add_argument("--ckpt", required=True, help="Path to a Phoneme_GAT_lit .ckpt checkpoint")
    p.add_argument("--out", required=True, help="Output CSV path (resumable/appendable)")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke tests)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")

    p.add_argument("--mask-laughter", action="store_true",
                    help="Splice out laughter segments before scoring (needs --laughter-csv)")
    p.add_argument("--laughter-csv", default=None,
                    help="detector_out/itw_laughter.csv - required with --mask-laughter")

    p.add_argument("--frame-scores-dir", default=None,
                    help="If set, additionally save sliding-window frame-level scores as .npy "
                         "per file (only for --files-subset files)")
    p.add_argument("--files-subset", default=None,
                    help="CSV with a file_id (or file) column restricting --frame-scores-dir output")
    p.add_argument("--window-s", type=float, default=1.0, help="Sliding window length (s) for frame scores")
    p.add_argument("--hop-s", type=float, default=0.25, help="Sliding window hop (s) for frame scores")
    return p


def main():
    args = build_argparser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.mask_laughter and not args.laughter_csv:
        raise ValueError("--mask-laughter requires --laughter-csv")

    print(f"[score_itw_detector] loading {args.ckpt} on {device} ...")
    t0 = time.time()
    lit = load_detector(args.ckpt, device)
    print(f"[score_itw_detector] loaded in {time.time()-t0:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mask_laughter:
        run_mask_mode(args, lit, device)
    else:
        run_default_mode(args, lit, device)

    if args.frame_scores_dir:
        run_frame_scores_mode(args, lit, device)


if __name__ == "__main__":
    main()
