"""Extract mean-pooled WavLM-base embeddings from the DETECTOR'S OWN frozen
backbone (the exact WavLM the Phoneme_GAT model reads: microsoft/wavlm-base,
loaded inside the checkpoint's phoneme_model submodule), for every file in
one or more eval sets.

Applies the SAME 3s center-crop/tile the scorer uses (crop_or_tile_center ->
48000 samples) BEFORE running WavLM, so the embedding matches exactly what
the detector saw when scoring.

Saves layers 0, 9, 12 (last hidden state) mean-pooled over time, 768-d each,
RAW / unstandardized (no per-coordinate normalization) -- consumers should
standardize themselves (e.g. by the bona-fide mean/SD) as in
experiments/axis_audits/audit_common.py::loso_axis_features.

Usage:
    python rerun_2026/extract_detector_wavlm.py \
        --ckpt ../models/asv19-wavlm-gat-full.ckpt \
        --sets base=data/eval_asv19 aug=data/eval_asv19_aug \
        --attack-ids data/eval_asv19/attack_ids.csv \
        --manifest-aug data/eval_asv19_aug/manifest.csv \
        --out-npz rerun_2026/embeddings/wavlm_base_embeddings.npz \
        --out-meta rerun_2026/embeddings/wavlm_base_meta.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

LAUGHSMI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUGHSMI_ROOT / "scripts"))

from score_itw_detector import load_detector, load_wav_mono_16k, crop_or_tile_center  # noqa: E402

LAYERS = (0, 9, 12)


def read_meta(meta_path):
    with open(meta_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_attack_ids(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: r["system_id"] for r in csv.DictReader(f)}


def read_manifest(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {Path(r["file"]).name: r for r in csv.DictReader(f)}


@torch.no_grad()
def embed_batch(wavlm_model, feat_extractor, wavs, device, fp16=False):
    """wavs: list of (48000,) float32 arrays (already cropped/tiled).
    Returns dict {layer: (B, 768) mean-pooled np array}."""
    x = torch.from_numpy(np.stack(wavs, axis=0)).to(device)
    if fp16:
        x = x.half()
    out = wavlm_model(x, output_hidden_states=True)
    hs = out.hidden_states  # tuple length 13 (embeddings + 12 layers)
    result = {}
    for layer in LAYERS:
        h = hs[layer]  # (B, T, 768)
        result[layer] = h.float().mean(dim=1).cpu().numpy()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True,
                     help="tag=dir pairs, e.g. base=data/eval_asv19 aug=data/eval_asv19_aug")
    ap.add_argument("--attack-ids", default=None, help="csv with file,system_id (for base set)")
    ap.add_argument("--manifest-aug", default=None,
                     help="manifest.csv from augment_laughter.py (file,label,augmented,insert_file,position,...)")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] loading detector {args.ckpt} on {device} ...")
    t0 = time.time()
    lit = load_detector(args.ckpt, device)
    wavlm_model = lit.model.transformer_in_phoneme_model  # frozen WavLM-base backbone (transformers WavLMModel)
    wavlm_model.eval()
    print(f"[extract] loaded in {time.time()-t0:.1f}s; backbone={type(wavlm_model).__name__}")

    attack_ids = read_attack_ids(args.attack_ids)

    all_rows = []
    embs_by_layer = {layer: [] for layer in LAYERS}

    for spec in args.sets:
        tag, d = spec.split("=", 1)
        d = Path(d)
        rows = read_meta(d / "meta.csv")
        manifest = {}
        if tag == "aug" and args.manifest_aug:
            manifest = read_manifest(args.manifest_aug)
        print(f"[extract] set={tag} dir={d} n={len(rows)}")

        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            wavs = []
            ok_rows = []
            for r in batch:
                path = d / r["file"]
                try:
                    wav = load_wav_mono_16k(str(path))
                    wav = crop_or_tile_center(wav)
                    wavs.append(wav)
                    ok_rows.append(r)
                except Exception as e:
                    print(f"[extract] SKIP {path}: {e}")
            if not wavs:
                continue
            res = embed_batch(wavlm_model, None, wavs, device)
            for layer in LAYERS:
                embs_by_layer[layer].append(res[layer])
            for r in ok_rows:
                base = Path(r["file"]).name
                mrow = manifest.get(base, {})
                all_rows.append({
                    "file": r["file"],
                    "basename": base,
                    "set": tag,
                    "label": r["label"],
                    "speaker": r.get("speaker", ""),
                    "system_id": attack_ids.get(r["file"], ""),
                    "augmented": mrow.get("augmented", "0" if tag == "base" else ""),
                    "position": mrow.get("position", ""),
                    "insert_file": mrow.get("insert_file", ""),
                })
            if (i // args.batch_size) % 5 == 0:
                print(f"  ...{tag} {i+len(batch)}/{len(rows)}")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {f"layer{layer}": np.concatenate(embs_by_layer[layer], axis=0) for layer in LAYERS}
    np.savez(out_npz, **save_dict)

    out_meta = Path(args.out_meta)
    with open(out_meta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"[extract] wrote {out_npz} (layers {LAYERS}, shapes: "
          f"{[save_dict[f'layer{l}'].shape for l in LAYERS]})")
    print(f"[extract] wrote {out_meta} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
