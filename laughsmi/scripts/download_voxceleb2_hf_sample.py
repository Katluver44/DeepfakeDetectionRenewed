"""Download a small VoxCeleb2 WAV sample from a Hugging Face mirror.

The selected mirror (`humanify/voxceleb2_dev`) stores individual WAV files,
which is much easier to sample than full VoxCeleb2 archives. It does not expose
speaker language metadata, so the manifest marks language as unverified.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def parse_file(path: str) -> dict:
    stem = Path(path).stem
    # Example: id00015_MRDbSqDD6ZI_00232.wav
    parts = stem.split("_")
    speaker_id = parts[0] if parts else ""
    video_id = parts[1] if len(parts) > 1 else ""
    utt_id = parts[2] if len(parts) > 2 else ""
    return {"speaker_id": speaker_id, "video_id": video_id, "utt_id": utt_id}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="humanify/voxceleb2_dev")
    parser.add_argument("--out-dir", type=Path, default=Path("laughsmi/data/voxceleb2_sample"))
    parser.add_argument("--n-files", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--hf-token-file", type=Path, default=Path("secret.txt"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and args.hf_token_file.exists():
        token = args.hf_token_file.read_text().strip()

    api = HfApi(token=token)
    print(f"Listing {args.repo_id} files...")
    all_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    wavs = [f for f in all_files if f.lower().endswith(".wav")]
    if not wavs:
        raise RuntimeError(f"No WAV files found in {args.repo_id}")

    rng = random.Random(args.seed)
    selected = rng.sample(wavs, k=min(args.n_files, len(wavs)))

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, repo_path in enumerate(selected):
        meta = parse_file(repo_path)
        dst_name = f"{i:05d}_{safe_name(Path(repo_path).name)}"
        dst = audio_dir / dst_name
        if not dst.exists():
            src = hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=repo_path,
                token=token,
            )
            shutil.copy2(src, dst)
        rows.append({
            "file": dst.name,
            "speaker": meta["speaker_id"],
            "label": "bona-fide",
            "source_repo": args.repo_id,
            "source_path": repo_path,
            "speaker_id": meta["speaker_id"],
            "video_id": meta["video_id"],
            "utt_id": meta["utt_id"],
            "language": "unverified",
        })
        if (i + 1) % 50 == 0:
            print(f"Downloaded/copied {i + 1}/{len(selected)}")

    meta_df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "meta.csv"
    meta_df[["file", "speaker", "label"]].to_csv(meta_path, index=False)
    meta_df.to_csv(args.out_dir / "manifest_full.csv", index=False)
    print(f"Wrote {meta_path} ({len(meta_df)} rows)")
    print(f"Wrote {args.out_dir / 'manifest_full.csv'}")


if __name__ == "__main__":
    main()
