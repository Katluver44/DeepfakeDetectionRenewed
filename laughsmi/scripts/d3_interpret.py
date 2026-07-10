"""What is WavLM picking up on when it separates real from synthetic laughter?

Uses the SAME confound-controlled preprocessing as d3_fixed.py (energy-trim ->
fixed 2s active segment -> peak-normalize), so we interpret the *residual*
signal, not recording artifacts.

Three probes:
  (A) LAYER SWEEP: real-vs-synth linear-probe AUC at EVERY WavLM layer (0..24).
      Early layers = low-level acoustics; late layers = higher-level/phonetic.
      Tells us WHERE in the network the real/synth distinction lives.
  (B) ACOUSTIC CORRELATES: compute interpretable per-clip descriptors that
      capture laughter's physical character —
        harmonic-to-noise ratio (HNR, voicing/breathiness),
        spectral flatness (noisiness/aperiodicity),
        F0 mean & F0 std (pitch level & variability across the laugh),
        voiced-fraction, spectral-centroid & its temporal std (dynamics),
        onset density (burst rate — "ha-ha-ha" syllabicity).
      Then: (i) how well do THESE separate real vs synth (interpretable-probe
      AUC, a semantic floor above the pure-DSP content-free floor); (ii)
      correlate each descriptor with the WavLM real-vs-synth probe score to
      NAME which acoustic property aligns with WavLM's decision.
  (C) which single descriptor differs most (real vs synth), per generator.

Excludes AudioLDM2 from the "signal" pooling (audit: its separation is
confound-level). Writes tables/table_d3_interpret.csv + prints summary.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = Path(__file__).resolve().parents[1]
SR = 16000
N_SAMP = int(2.0 * SR)
GEN_SIGNAL = ["laugh-bark_laughter_token", "laugh-bark_laughs_inline", "laugh-parler_tts", "laugh-xtts"]
ALL_GEN = GEN_SIGNAL + ["laugh-audioldm2"]


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
        f = np.argmax(above); l = len(a) - 1 - np.argmax(above[::-1]); a = a[f:l + 1]
    if len(a) == 0:
        a = np.zeros(N_SAMP, np.float32)
    if len(a) >= N_SAMP:
        s = (len(a) - N_SAMP) // 2; a = a[s:s + N_SAMP]
    else:
        a = np.tile(a, -(-N_SAMP // len(a)))[:N_SAMP]
    return (a / (np.max(np.abs(a)) + 1e-9) * 0.9).astype(np.float32)


def acoustic_descriptors(a):
    import librosa
    # F0 via pyin (voiced pitch of the laugh)
    try:
        f0, vflag, _ = librosa.pyin(a, fmin=60, fmax=500, sr=SR, frame_length=1024)
        f0v = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0v)) if len(f0v) else 0.0
        f0_std = float(np.std(f0v)) if len(f0v) else 0.0
        voiced_frac = float(np.mean(vflag)) if vflag is not None else 0.0
    except Exception:
        f0_mean = f0_std = voiced_frac = 0.0
    S = np.abs(librosa.stft(a, n_fft=1024, hop_length=256))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    cent_mean = float(np.mean(cent)); cent_std = float(np.std(cent))
    # HNR proxy: harmonic/percussive energy ratio
    Hh, Pp = librosa.effects.hpss(a)
    hnr = float(np.sum(Hh ** 2) / (np.sum(Pp ** 2) + 1e-9))
    # onset density (burst rate ~ syllabicity of laughter)
    onsets = librosa.onset.onset_detect(y=a, sr=SR, units="time")
    onset_rate = float(len(onsets) / (len(a) / SR))
    # temporal RMS variability (laughs pulse; steady speech doesn't)
    rms = librosa.feature.rms(y=a)[0]
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-9))
    return {"f0_mean": f0_mean, "f0_std": f0_std, "voiced_frac": voiced_frac,
            "spec_flatness": flatness, "cent_mean": cent_mean, "cent_std": cent_std,
            "hnr": hnr, "onset_rate": onset_rate, "rms_cv": rms_cv}


def probe(X, y, seed=0):
    ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
    nmin = min((y == 0).sum(), (y == 1).sum())
    if nmin < 2:
        return np.nan
    cv = StratifiedKFold(min(5, nmin), shuffle=True, random_state=seed)
    return float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                 X, y, cv=cv, scoring="roc_auc").mean())


def main():
    inv = pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv")
    rng = np.random.RandomState(0)
    ncap = 30
    groups_use = ["laugh-real"] + ALL_GEN
    sel = pd.concat([inv[inv.group == g].iloc[rng.choice((inv.group == g).sum(), min(ncap, (inv.group == g).sum()), replace=False)]
                     for g in groups_use]).reset_index(drop=True)

    import torch
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(dev).eval()

    n_layers = model.config.num_hidden_layers + 1  # incl embedding layer
    layer_embs = [[] for _ in range(n_layers)]
    descs, groups = [], []
    print(f"[interpret] extracting {len(sel)} clips, all {n_layers} layers + acoustic descriptors...")
    with torch.no_grad():
        for _, r in sel.iterrows():
            a = load_fixed(r.file_path)
            descs.append(acoustic_descriptors(a))
            iv = fe(a, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            hs = model(iv).hidden_states
            for li in range(n_layers):
                layer_embs[li].append(hs[li][0].mean(0).cpu().numpy())
            groups.append(r.group)
    layer_embs = [np.array(e) for e in layer_embs]
    g = np.array(groups)
    desc_df = pd.DataFrame(descs)
    real = g == "laugh-real"
    synth_signal = np.isin(g, GEN_SIGNAL)

    # (A) layer sweep: real vs pooled-synth(signal generators)
    print("\n(A) LAYER SWEEP — real vs synthetic(4 signal generators) AUC per WavLM layer:")
    mask = real | synth_signal
    y = synth_signal[mask].astype(int)
    layer_aucs = []
    for li in range(n_layers):
        auc = probe(layer_embs[li][mask], y)
        layer_aucs.append(auc)
    for li, auc in enumerate(layer_aucs):
        bar = "#" * int((auc - 0.5) * 60) if auc > 0.5 else ""
        print(f"  layer {li:2d}: AUC={auc:.3f} {bar}")
    best_layer = int(np.nanargmax(layer_aucs))
    print(f"  -> peak at layer {best_layer} (AUC={layer_aucs[best_layer]:.3f})")

    # (B) interpretable acoustic-descriptor probe (semantic floor)
    desc_cols = list(desc_df.columns)
    Xd = desc_df.to_numpy()
    auc_desc = probe(Xd[mask], y)
    print(f"\n(B) interpretable acoustic-descriptor probe (9 named features): AUC={auc_desc:.3f}")
    print("    (compare to pure-DSP content-free floor ~0.71-0.89 and WavLM ~0.97)")

    # correlate each descriptor with WavLM decision (cross-val predicted prob at best layer)
    from sklearn.base import clone
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    Xb = layer_embs[best_layer][mask]
    okm = ~np.isnan(Xb).any(1)
    wavlm_prob = cross_val_predict(pipe, Xb[okm], y[okm],
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print("\n    correlation of each acoustic descriptor with WavLM's real-vs-synth decision score:")
    dfm = desc_df[mask].reset_index(drop=True).loc[okm].reset_index(drop=True)
    corrs = {}
    for c in desc_cols:
        v = dfm[c].to_numpy()
        if np.std(v) > 0:
            r_, p_ = pearsonr(v, wavlm_prob); corrs[c] = (r_, p_)
    for c, (r_, p_) in sorted(corrs.items(), key=lambda kv: -abs(kv[1][0])):
        print(f"      {c:<14} r={r_:+.3f} (p={p_:.1e})")

    # (C) which descriptor differs most real vs synth (per single-feature AUC)
    print("\n(C) single-descriptor real-vs-synth AUC (which acoustic property separates most):")
    single = {}
    for c in desc_cols:
        single[c] = probe(desc_df[c].to_numpy()[mask].reshape(-1, 1), y)
    for c, a in sorted(single.items(), key=lambda kv: -(kv[1] if not np.isnan(kv[1]) else 0)):
        rmean = desc_df[c][real].mean(); smean = desc_df[c][synth_signal].mean()
        print(f"      {c:<14} AUC={a:.3f}   real={rmean:.3f}  synth={smean:.3f}")

    # save
    with open(L / "tables" / "table_d3_interpret.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "real_vs_synth_auc"]); [w.writerow([li, round(a, 3)]) for li, a in enumerate(layer_aucs)]
        w.writerow([]); w.writerow(["acoustic_descriptor_probe_auc", round(auc_desc, 3)])
        w.writerow(["best_layer", best_layer, "auc", round(layer_aucs[best_layer], 3)])
        w.writerow([]); w.writerow(["descriptor", "corr_with_wavlm", "single_feat_auc", "real_mean", "synth_mean"])
        for c in desc_cols:
            r_ = corrs.get(c, (np.nan, np.nan))[0]
            w.writerow([c, round(r_, 3) if not np.isnan(r_) else "nan", round(single[c], 3),
                        round(float(desc_df[c][real].mean()), 3), round(float(desc_df[c][synth_signal].mean()), 3)])
    print("\nwrote tables/table_d3_interpret.csv")


if __name__ == "__main__":
    main()
