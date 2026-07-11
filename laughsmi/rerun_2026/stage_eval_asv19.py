"""Stage data/eval_asv19 from Bisher/ASVspoof_2019_LA parquet (test split).

150 bona-fide (key==0, system_id=='-') + 150 spoof (key==1, system_id=='Axx'),
seed 20260710. Writes meta.csv (file,speaker,label) + wavs/ (mono 16kHz) +
attack_ids.csv (file,system_id) sidecar for per-attack analysis downstream.
"""
import csv
import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SEED = 20260710
N_PER_CLASS = 150
PARQUET = "/home/sagemaker-user/.cache/huggingface/hub/datasets--Bisher--ASVspoof_2019_LA/snapshots/aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c/data/test-00000-of-00001.parquet"
OUT_DIR = Path("/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi/data/eval_asv19")
TARGET_SR = 16000


def resample_if_needed(wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def main():
    print("reading parquet...")
    tbl = pq.read_table(PARQUET, columns=["speaker_id", "audio_file_name", "audio", "system_id", "key"])
    df = tbl.to_pandas()
    print(f"rows: {len(df)}")

    rng = random.Random(SEED)
    bona_idx = df.index[df["key"] == 0].tolist()
    spoof_idx = df.index[df["key"] == 1].tolist()
    rng.shuffle(bona_idx)
    rng.shuffle(spoof_idx)
    bona_idx = bona_idx[:N_PER_CLASS]
    spoof_idx = spoof_idx[:N_PER_CLASS]

    wav_dir = OUT_DIR / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    attack_rows = []

    for tag, idxs, label in [("bona", bona_idx, "bona-fide"), ("spoof", spoof_idx, "spoof")]:
        for i in idxs:
            r = df.loc[i]
            audio_bytes = r["audio"]["bytes"]
            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            wav = resample_if_needed(wav, sr)
            fname = f"{tag}_{i:05d}.wav"
            sf.write(str(wav_dir / fname), wav, TARGET_SR)
            meta_rows.append({"file": f"wavs/{fname}", "speaker": r["speaker_id"], "label": label})
            attack_rows.append({"file": f"wavs/{fname}", "system_id": r["system_id"]})

    with open(OUT_DIR / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        w.writerows(meta_rows)

    with open(OUT_DIR / "attack_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "system_id"])
        w.writeheader()
        w.writerows(attack_rows)

    print(f"wrote {len(meta_rows)} files -> {OUT_DIR}")
    print(f"bona={sum(1 for r in meta_rows if r['label']=='bona-fide')} spoof={sum(1 for r in meta_rows if r['label']=='spoof')}")


if __name__ == "__main__":
    main()
