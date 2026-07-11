"""C3(a) — build a small MLAAD English-spoof eval set for the SECOND-BENCHMARK
replication (mlaad_wavlm-gat.ckpt), matching the eval_asv19 schema (meta.csv:
file,speaker,label).

DEVIATION FROM build_eval_sets.py::build_mlaad, documented:
  - `mueller91/MLAAD` (the source build_eval_sets.py expects) is now a GATED
    HF repo and our token, despite having `canReadGatedRepos` scope, is not on
    the dataset's authorized-user list (403 GatedRepoError on every file).
    Manual approval on the HF web UI wasn't obtainable in an agent session.
  - Fallback (documented): `mueller91/MLAAD-tiny` is the SAME MLAAD corpus
    (same lab / same generators), publicly accessible with our token, and is
    already referenced elsewhere in this repo as the MLAAD source
    (experiments/results/mlaad/baseline_eval/test_in_distribution.json points
    at `datasets--mueller91--MLAAD-tiny` snapshot paths) -- so this is not a
    novel substitution, it's the same corpus this codebase already uses.
  - `kresnik/librispeech_asr_test` (the real anchor build_eval_sets.py expects)
    is a now-unsupported script-dataset under datasets 5.0
    ("Dataset scripts are no longer supported"). Fallback: `openslr/librispeech_asr`
    parquet mirror (`clean/test/0000.parquet`), no dataset script, same corpus
    (LibriSpeech test-clean).
"""
from __future__ import annotations
import io, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

LAUGHSMI = Path("/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi")
TARGET_SR = 16000
SEED = 20260710
N_SPOOF = 120
N_BONA = 120
TIMEOUT_S = 900

OUT = LAUGHSMI / "data" / "eval_mlaad"


def resample_if_needed(wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def write_wav(path, wav, sr=TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)


def main():
    from huggingface_hub import HfApi, hf_hub_download
    rng = random.Random(SEED)
    wav_dir = OUT / "wavs"
    rows = []
    t0 = time.time()

    api = HfApi()
    info = api.dataset_info("mueller91/MLAAD-tiny")
    from collections import defaultdict
    bysys = defaultdict(list)
    for s in info.siblings:
        f = s.rfilename
        if f.startswith("fake/en/") and f.endswith(".wav"):
            bysys[f.split("/")[2]].append(f)
    systems = sorted(bysys)
    per_sys = max(1, N_SPOOF // len(systems))
    plan = []
    for sysn in systems:
        files = bysys[sysn]
        rng.shuffle(files)
        for f in files[:per_sys]:
            plan.append((sysn, f))
    rng.shuffle(plan)
    plan = plan[:N_SPOOF]
    print(f"[mlaad-tiny] {len(systems)} EN systems, sampling {len(plan)} spoof files")

    n_ok = 0
    for sysn, rel in plan:
        if time.time() - t0 > TIMEOUT_S:
            print(f"[mlaad-tiny] TIMEOUT after {n_ok} spoof files")
            break
        try:
            local = hf_hub_download("mueller91/MLAAD-tiny", rel, repo_type="dataset")
            wav, sr = sf.read(local, dtype="float32", always_2d=False)
            wav = resample_if_needed(wav, sr)
            fname = f"spoof_{n_ok:05d}_{sysn.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.wav"
            write_wav(wav_dir / fname, wav)
            rows.append({"file": f"wavs/{fname}", "speaker": sysn, "label": "spoof"})
            n_ok += 1
        except Exception as e:
            print(f"[mlaad-tiny] skip {rel}: {type(e).__name__} {e}")
    print(f"[mlaad-tiny] got {n_ok} spoof files in {time.time()-t0:.0f}s")

    # bona-fide anchor: LibriSpeech test-clean (openslr parquet mirror)
    try:
        local = hf_hub_download("openslr/librispeech_asr", "clean/test/0000.parquet", repo_type="dataset")
        df = pd.read_parquet(local)
        df = df.sample(n=min(N_BONA, len(df)), random_state=SEED).reset_index(drop=True)
        n_b = 0
        for i, r in df.iterrows():
            audio = r["audio"]
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            wav = resample_if_needed(wav, sr)
            fname = f"bona_{n_b:05d}.wav"
            write_wav(wav_dir / fname, wav)
            rows.append({"file": f"wavs/{fname}", "speaker": str(r["speaker_id"]), "label": "bona-fide"})
            n_b += 1
        print(f"[mlaad-tiny] got {n_b} bona-fide LibriSpeech(openslr) anchor files")
    except Exception as e:
        print(f"[mlaad-tiny] LibriSpeech anchor FAILED: {type(e).__name__} {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.csv"
    import csv
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        w.writerows(rows)
    nb = sum(1 for r in rows if r["label"] == "bona-fide")
    ns = sum(1 for r in rows if r["label"] == "spoof")
    print(f"[mlaad-tiny] wrote {len(rows)} files ({nb} bona / {ns} spoof) -> {OUT}")


if __name__ == "__main__":
    main()
