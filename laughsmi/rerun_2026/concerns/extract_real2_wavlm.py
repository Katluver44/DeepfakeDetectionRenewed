"""Extract WavLM-Large L9/L12 mean embeddings for the SECOND independent real
laughter corpus (ESC-50 'laughing' category, Freesound.org recordings), using
the exact same confound-controlled protocol as scripts/d3_fixed.py:
  1. energy-trim leading/trailing silence (>1% of peak)
  2. fixed 2.0s central active segment (tile-pad if shorter)
  3. peak-normalize to 0.9
  4. WavLM-Large hidden_states[9] and [12], mean-pooled over frames

Outputs:
  rerun_2026/concerns/data/real2_emb_layer9.npy
  rerun_2026/concerns/data/real2_emb_layer12.npy
  rerun_2026/concerns/data/real2_inventory.csv  (file, group='laugh-real2', dur_s)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

L = Path(__file__).resolve().parents[2]  # laughsmi/
SR = 16000
FIXED_DUR = 2.0
N_SAMP = int(FIXED_DUR * SR)

REAL2_DIR = L / "rerun_2026" / "concerns" / "data" / "real_laughter_2"
OUT_DIR = L / "rerun_2026" / "concerns" / "data"


def load_fixed(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import librosa
        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
    peak = np.max(np.abs(a)) + 1e-9
    above = np.abs(a) > 0.01 * peak
    if above.any():
        first = np.argmax(above); last = len(a) - 1 - np.argmax(above[::-1])
        a = a[first:last + 1]
    if len(a) == 0:
        a = np.zeros(N_SAMP, dtype=np.float32)
    if len(a) >= N_SAMP:
        s = (len(a) - N_SAMP) // 2
        a = a[s:s + N_SAMP]
    else:
        reps = -(-N_SAMP // len(a))
        a = np.tile(a, reps)[:N_SAMP]
    a = a / (np.max(np.abs(a)) + 1e-9) * 0.9
    return a.astype(np.float32)


def content_free_feats(a):
    rms = float(np.sqrt(np.mean(a ** 2)))
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1.0 / SR)
    centroid = float((freqs * spec).sum() / (spec.sum() + 1e-12))
    cum = np.cumsum(spec); rolloff = float(freqs[np.searchsorted(cum, 0.85 * cum[-1])]) if cum[-1] > 0 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a))) > 0))
    hf = float(spec[freqs > 6000].sum() / (spec.sum() + 1e-12))
    dc = float(np.mean(a))
    lf = float(spec[freqs < 500].sum() / (spec.sum() + 1e-12))
    return [rms, centroid, rolloff, zcr, hf, dc, lf, float(spec.max() / (spec.mean() + 1e-9))]


def load_raw(path):
    """Whole-clip, unnormalized load matching scripts/extract_wavlm_features.py's
    load_audio_segment (no silence trim, no peak-norm) -- the pipeline that
    produced the CACHED embeddings/d3_multi/*.npy for the synthetic families
    (raw audio for those families no longer exists, so this is the only way
    to get an apples-to-apples-pipeline comparison against them)."""
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import librosa
        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


def main():
    files = sorted(REAL2_DIR.glob("*.wav"))
    print(f"found {len(files)} real2 wav files")
    assert len(files) >= 20, "too few clips for a usable second anchor"

    import torch
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(dev).eval()

    emb9, emb12, cf, names, durs = [], [], [], [], []
    emb9_raw, emb12_raw = [], []
    with torch.no_grad():
        for fp in files:
            raw, sr = sf.read(fp, dtype="float32", always_2d=False)
            if raw.ndim > 1:
                raw = raw.mean(1)
            durs.append(len(raw) / sr)

            # confound-controlled (2s center active segment, peak-normalized)
            a = load_fixed(fp)
            cf.append(content_free_feats(a))
            iv = fe(a, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            hs = model(iv).hidden_states
            emb9.append(hs[9][0].mean(0).cpu().numpy())
            emb12.append(hs[12][0].mean(0).cpu().numpy())

            # raw whole-clip (matches the cached d3_multi synth embeddings' pipeline)
            araw = load_raw(fp)
            ivr = fe(araw, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            hsr = model(ivr).hidden_states
            emb9_raw.append(hsr[9][0].mean(0).cpu().numpy())
            emb12_raw.append(hsr[12][0].mean(0).cpu().numpy())

            names.append(fp.name)

    emb9 = np.array(emb9); emb12 = np.array(emb12); cf = np.array(cf)
    emb9_raw = np.array(emb9_raw); emb12_raw = np.array(emb12_raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "real2_emb_layer9.npy", emb9)
    np.save(OUT_DIR / "real2_emb_layer12.npy", emb12)
    np.save(OUT_DIR / "real2_contentfree.npy", cf)
    np.save(OUT_DIR / "real2_raw_emb_layer9.npy", emb9_raw)
    np.save(OUT_DIR / "real2_raw_emb_layer12.npy", emb12_raw)
    inv = pd.DataFrame({"file": names, "group": "laugh-real2", "dur_s_orig": durs})
    inv.to_csv(OUT_DIR / "real2_inventory.csv", index=False)
    print(f"wrote fixed {emb12.shape}, raw {emb12_raw.shape}, cf {cf.shape} embeddings + inventory")


if __name__ == "__main__":
    main()
