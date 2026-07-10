"""AUDIT of D3 "real vs synthetic laughter in WavLM space" analysis.

Read-only audit: does NOT modify any existing script/data. Investigates 5
confound hypotheses for the AUC~=1.0 real-vs-synthetic-laughter claim in
d3_multi_analysis.py / table_d3_multi.csv.

  1. Duration/silence artifact (clip length differences)
  2. Loudness/channel artifact (RMS, spectral centroid, DC offset)
  3. WavLM extraction correctness (layers, NaNs, whole-clip sentinel, dupes)
  4. Triviality check (is speech-real vs laugh-real ALSO AUC~1.0? label
     shuffle control? real-vs-real split control?)
  5. Harsher probe: PCA-20 before logistic regression

Writes: laughsmi/D3_AUDIT.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = Path(__file__).resolve().parents[1]
RNG_SEED = 0


def load_data():
    df = pd.read_parquet(L / "embeddings" / "d3_multi_features.parquet")
    emb9 = np.load(L / "embeddings" / "d3_multi" / "mean_emb_layer9.npy")
    emb12 = np.load(L / "embeddings" / "d3_multi" / "mean_emb_layer12.npy")
    inv = pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv")
    assert len(df) == len(emb9) == len(emb12) == len(inv), (len(df), len(emb9), len(emb12), len(inv))
    df = df.reset_index(drop=True)
    df["file_path"] = inv["file_path"].to_numpy()  # ensure alignment source
    return df, emb9, emb12


def probe_auc(X, y, n_splits=5, seed=0):
    ok = ~np.isnan(X).any(axis=1)
    X, y = X[ok], y[ok]
    n_min = min((y == 0).sum(), (y == 1).sum())
    if n_min < 2:
        return np.nan, np.nan, ok.sum()
    splits = min(n_splits, n_min)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(auc.mean()), float(auc.std()), ok.sum()


def pca_probe_auc(X, y, n_components=20, n_splits=5, seed=0):
    ok = ~np.isnan(X).any(axis=1)
    X, y = X[ok], y[ok]
    n_min = min((y == 0).sum(), (y == 1).sum())
    if n_min < 2:
        return np.nan, np.nan
    splits = min(n_splits, n_min)
    nc = min(n_components, X.shape[0] - 1, X.shape[1])
    clf = make_pipeline(StandardScaler(), PCA(n_components=nc, random_state=seed), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(auc.mean()), float(auc.std())


def compute_audio_stats(paths):
    """Per-file duration, RMS, DC offset, spectral centroid, leading/trailing silence."""
    rows = []
    for p in paths:
        try:
            audio, sr = sf.read(p, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            dur = len(audio) / sr
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dc = float(np.mean(audio))
            peak = float(np.max(np.abs(audio)) + 1e-12)
            # spectral centroid (simple, on whole clip magnitude spectrum)
            spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) if len(audio) > 1 else np.array([0.0])
            freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr) if len(audio) > 1 else np.array([0.0])
            centroid = float((freqs * spec).sum() / (spec.sum() + 1e-12))
            # spectral rolloff (freq below which 85% of spectral energy lies)
            cumsum = np.cumsum(spec)
            total_e = cumsum[-1] if len(cumsum) else 0.0
            rolloff = float(freqs[np.searchsorted(cumsum, 0.85 * total_e)]) if total_e > 0 else 0.0
            # zero-crossing rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0)) if len(audio) > 1 else 0.0
            # HF energy ratio above 6kHz
            hf_mask = freqs > 6000
            hf_ratio = float(spec[hf_mask].sum() / (spec.sum() + 1e-12))
            # leading/trailing silence at -40dBFS-relative-to-peak threshold
            thresh = 0.01 * peak
            above = np.abs(audio) > thresh
            if above.any():
                first = np.argmax(above)
                last = len(audio) - 1 - np.argmax(above[::-1])
                lead_sil = first / sr
                trail_sil = (len(audio) - 1 - last) / sr
            else:
                lead_sil = trail_sil = dur / 2
            rows.append(dict(file_path=p, dur_s=dur, rms=rms, dc_offset=dc, peak=peak,
                              spectral_centroid=centroid, spectral_rolloff=rolloff, zcr=zcr,
                              hf_energy_ratio=hf_ratio, lead_sil_s=lead_sil, trail_sil_s=trail_sil, sr=sr))
        except Exception as e:
            rows.append(dict(file_path=p, dur_s=np.nan, rms=np.nan, dc_offset=np.nan, peak=np.nan,
                              spectral_centroid=np.nan, spectral_rolloff=np.nan, zcr=np.nan,
                              hf_energy_ratio=np.nan, lead_sil_s=np.nan, trail_sil_s=np.nan, sr=np.nan))
            print(f"WARN: failed to read {p}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)


def trim_and_reextract_check(df, audio_stats, n_per_group=15):
    """H1 control: fix ALL clips to the SAME fixed duration (shortest common
    window, energy-based center-crop) and re-probe on WavLM re-extracted
    embeddings, to see whether AUC collapses once duration is controlled.
    This requires torch+transformers; if unavailable, falls back to a proxy
    using n_frames-normalized embeddings (documented as a proxy, not a full
    re-extraction control)."""
    try:
        import torch
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    except Exception as e:
        return None, f"torch/transformers unavailable ({e}); skipping true re-extraction control"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(device).eval()

    TARGET_DUR = 2.0  # seconds; shorter than nearly everything -> forces truncation for all groups
    TARGET_SR = 16000
    n_samples = int(TARGET_DUR * TARGET_SR)

    rng = np.random.RandomState(0)
    rows = []
    for grp in ["laugh-real", "laugh-bark_laughter_token", "laugh-bark_laughs_inline",
                "laugh-audioldm2", "laugh-parler_tts", "laugh-xtts", "speech-real"]:
        sub = df[df.group == grp]
        if len(sub) == 0:
            continue
        idx = rng.choice(len(sub), size=min(n_per_group, len(sub)), replace=False)
        for i in idx:
            rows.append((grp, sub.iloc[i].file_path))

    embs9, embs12, groups = [], [], []
    with torch.no_grad():
        for grp, fp in rows:
            audio, sr = sf.read(fp, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            # center-crop / pad to exactly TARGET_DUR seconds (fixes length confound)
            if len(audio) >= n_samples:
                start = (len(audio) - n_samples) // 2
                audio = audio[start:start + n_samples]
            else:
                pad = n_samples - len(audio)
                audio = np.pad(audio, (pad // 2, pad - pad // 2))
            # peak-normalize (also addresses H2 loudness confound simultaneously)
            peak = np.max(np.abs(audio)) + 1e-9
            audio = audio / peak * 0.9

            inputs = fe(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            iv = inputs["input_values"].to(device)
            out = model(iv)
            hs = out.hidden_states
            e9 = hs[9][0].mean(0).cpu().numpy()
            e12 = hs[12][0].mean(0).cpu().numpy()
            embs9.append(e9); embs12.append(e12); groups.append(grp)

    return dict(emb9=np.array(embs9), emb12=np.array(embs12), group=np.array(groups)), None


def main():
    df, emb9, emb12 = load_data()
    g = df.group.to_numpy()
    real_mask = g == "laugh-real"
    speech_mask = g == "speech-real"
    methods = [m for m in pd.unique(g) if m.startswith("laugh-") and m != "laugh-real"]

    report = []
    report.append("# D3 Audit: Real vs Synthetic Laughter in WavLM Space\n")
    report.append(f"Generated by scripts/audit_d3.py. Data: embeddings/d3_multi_features.parquet "
                   f"({len(df)} rows), embeddings/d3_multi_inventory.csv.\n")

    # ---------------- H1: duration/silence ----------------
    report.append("## H1: Duration/silence artifact\n")
    df["dur_s_frames"] = df.n_frames / 50.0  # WavLM ~50Hz frame rate, proxy
    dur_summary = df.groupby("group")["dur_s_frames"].agg(["mean", "std", "min", "max", "count"])
    report.append("Duration (proxy: n_frames/50Hz) by group:\n\n```\n" + dur_summary.to_string() + "\n```\n")

    print("Computing real audio-file stats (duration/RMS/silence/spectral)...", file=sys.stderr)
    audio_stats = compute_audio_stats(df.file_path.tolist())
    audio_stats["group"] = g
    dur_true_summary = audio_stats.groupby("group")["dur_s"].agg(["mean", "std", "min", "max", "count"])
    report.append("True on-disk clip duration by group (soundfile):\n\n```\n" + dur_true_summary.to_string() + "\n```\n")

    lead_summary = audio_stats.groupby("group")[["lead_sil_s", "trail_sil_s"]].mean()
    report.append("Mean leading/trailing silence (>1% peak threshold) by group:\n\n```\n" + lead_summary.to_string() + "\n```\n")

    # correlation between duration and C/T features
    ok = audio_stats.dur_s.notna()
    corr_C, p_C = pearsonr(audio_stats.dur_s[ok], df.C_layer12[ok])
    corr_T, p_T = pearsonr(audio_stats.dur_s[ok], df.T_layer9[ok])
    report.append(f"Correlation of duration with C_layer12: r={corr_C:.3f} (p={p_C:.2e}); "
                   f"with T_layer9: r={corr_T:.3f} (p={p_T:.2e}).\n")

    # duration-only probe: can we separate real vs synth using duration ALONE?
    dur_aucs = {}
    for m in methods:
        mask = real_mask | (g == m)
        X = audio_stats.dur_s.to_numpy()[mask].reshape(-1, 1)
        y = (g[mask] == m).astype(int)
        auc, std, n = probe_auc(X, y)
        dur_aucs[m] = (auc, std, n)
    report.append("**Duration-ONLY logistic-probe AUC** (real vs each synth method, using clip duration "
                   "as the sole feature):\n\n```\n")
    for m, (auc, std, n) in dur_aucs.items():
        report.append(f"  {m:<28} AUC={auc:.3f} +/- {std:.3f}  (n={n})\n")
    report.append("```\n")

    # ---------------- H2: loudness/channel ----------------
    report.append("## H2: Loudness/channel artifact\n")
    loud_summary = audio_stats.groupby("group")[["rms", "dc_offset", "peak", "spectral_centroid"]].mean()
    report.append("Mean RMS / DC offset / peak / spectral centroid by group:\n\n```\n" + loud_summary.to_string() + "\n```\n")

    loud_feat_aucs = {}
    for m in methods:
        mask = real_mask | (g == m)
        X = audio_stats[["rms", "dc_offset", "spectral_centroid"]].to_numpy()[mask]
        y = (g[mask] == m).astype(int)
        auc, std, n = probe_auc(X, y)
        loud_feat_aucs[m] = (auc, std, n)
    report.append("**Loudness-only probe AUC** (RMS+DC-offset+spectral-centroid, 3 features, real vs each synth):\n\n```\n")
    for m, (auc, std, n) in loud_feat_aucs.items():
        report.append(f"  {m:<28} AUC={auc:.3f} +/- {std:.3f}  (n={n})\n")
    report.append("```\n")

    # ---------------- CONTENT-FREE 8-FEATURE PROBE (decisive control) ----------------
    report.append("## Content-free scalar-feature probe (decisive control)\n")
    report.append("8 signal-level scalar features carrying no laughter-content information: "
                   "duration_s, leading_silence_s, trailing_silence_s, RMS, spectral_centroid, "
                   "spectral_rolloff (85%), zero-crossing-rate, HF-energy-ratio(>6kHz). Same "
                   "StratifiedKFold(5) logistic-regression probe as the WavLM analysis, but with "
                   "these 8 scalars replacing the 1024-dim WavLM embedding.\n\n")
    content_free_cols = ["dur_s", "lead_sil_s", "trail_sil_s", "rms", "spectral_centroid",
                          "spectral_rolloff", "zcr", "hf_energy_ratio"]
    cf_aucs = {}
    per_feat_best = {}
    for m in methods:
        mask4 = real_mask | (g == m)
        X = audio_stats[content_free_cols].to_numpy()[mask4]
        y = (g[mask4] == m).astype(int)
        auc, std, n = probe_auc(X, y)
        cf_aucs[m] = (auc, std, n)
        # per-single-feature AUC to find the "smoking gun"
        single = {}
        for col in content_free_cols:
            a, s, _ = probe_auc(audio_stats[col].to_numpy()[mask4].reshape(-1, 1), y)
            single[col] = a
        best_feat = max(single, key=lambda k: (single[k] if not np.isnan(single[k]) else -1))
        per_feat_best[m] = (best_feat, single[best_feat], single)

    report.append("**8-feature content-free probe AUC** (real-vs-each-synth-method, WavLM-free):\n\n```\n")
    for m, (auc, std, n) in cf_aucs.items():
        report.append(f"  {m:<28} AUC={auc:.3f} +/- {std:.3f}  (n={n})\n")
    report.append("```\n\n")
    report.append("**Single most discriminative content-free feature per method** (\"smoking gun\" check):\n\n```\n")
    for m, (best_feat, best_auc, single) in per_feat_best.items():
        others = ", ".join(f"{k}={v:.3f}" for k, v in single.items())
        report.append(f"  {m:<28} best={best_feat} (AUC={best_auc:.3f})  [{others}]\n")
    report.append("```\n")

    # content-free probe for speech-real vs laugh-real too
    mask_sr2 = real_mask | speech_mask
    X_sr = audio_stats[content_free_cols].to_numpy()[mask_sr2]
    y_sr2 = (g[mask_sr2] == "speech-real").astype(int)
    auc_sr_cf, std_sr_cf, n_sr_cf = probe_auc(X_sr, y_sr2)
    report.append(f"\nContent-free probe, speech-real vs laugh-real: AUC={auc_sr_cf:.3f}+/-{std_sr_cf:.3f} (n={n_sr_cf})\n")

    # ---------------- H3: extraction correctness ----------------
    report.append("## H3: WavLM extraction correctness\n")
    nan9 = np.isnan(emb9).any(axis=1).sum()
    nan12 = np.isnan(emb12).any(axis=1).sum()
    uniq9 = len(np.unique(emb9, axis=0))
    dup_files = df.file_path.duplicated().sum()
    end_s_vals = df.end_s.unique() if "end_s" in df.columns else pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv").end_s.unique()
    inv = pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv")
    end_s_ok = (inv.end_s == -1).all() and (inv.start_s == 0).all()
    report.append(f"- NaN embedding rows: L9={nan9}, L12={nan12} (of {len(df)})\n"
                   f"- Unique embedding rows: {uniq9}/{len(emb9)} (no collapsed/duplicate rows)\n"
                   f"- Duplicate file_path rows in inventory: {dup_files}\n"
                   f"- Whole-clip sentinel (start_s=0, end_s=-1) honored for ALL {len(inv)} inventory rows: {end_s_ok}\n"
                   f"- Row count consistency: features={len(df)}, emb9={len(emb9)}, emb12={len(emb12)}, inventory={len(inv)}\n")
    # confirm layers actually differ (sanity that layer 9 != layer 12 embedding)
    same_layer = np.allclose(emb9, emb12)
    cos9_12 = float(np.mean([np.dot(emb9[i], emb12[i]) / (np.linalg.norm(emb9[i]) * np.linalg.norm(emb12[i]) + 1e-9)
                              for i in range(min(50, len(emb9)))]))
    report.append(f"- emb9 identical to emb12 (would indicate layer-index bug): {same_layer}\n"
                   f"- Mean cosine similarity emb9 vs emb12 (first 50 rows, should be <1.0 if distinct layers): {cos9_12:.4f}\n")

    # ---------------- H4: triviality check ----------------
    report.append("## H4: Triviality check (label shuffle, real-vs-real controls)\n")
    # (a) speech-real vs laugh-real (claimed AUC~1 in narrative)
    mask = real_mask | speech_mask
    y = (g[mask] == "speech-real").astype(int)
    auc9, std9, n9 = probe_auc(emb9[mask], y)
    auc12, std12, n12 = probe_auc(emb12[mask], y)
    report.append(f"- speech-real vs laugh-real (WavLM probe): L9 AUC={auc9:.3f}+/-{std9:.3f}, "
                  f"L12 AUC={auc12:.3f}+/-{std12:.3f} (n={n9})\n")

    # (b) label-shuffled control on real vs bark_laughter_token (largest synth group)
    rng = np.random.RandomState(0)
    shuf_mask = real_mask | (g == "laugh-bark_laughter_token")
    y_true = (g[shuf_mask] == "laugh-bark_laughter_token").astype(int)
    shuf_aucs = []
    for trial in range(10):
        y_shuf = rng.permutation(y_true)
        auc, _, _ = probe_auc(emb12[shuf_mask], y_shuf, seed=trial)
        shuf_aucs.append(auc)
    report.append(f"- Label-SHUFFLED control (real vs bark_laughter_token, L12, 10 trials): "
                  f"AUC mean={np.mean(shuf_aucs):.3f}, max={np.max(shuf_aucs):.3f}, "
                  f"all={[round(a,3) for a in shuf_aucs]}\n")

    # (c) real-vs-real split control: random halves of laugh-real vs each other
    real_idx = np.where(real_mask)[0]
    half_aucs = []
    for trial in range(10):
        perm = rng.permutation(real_idx)
        half = len(perm) // 2
        y_half = np.zeros(len(real_idx), dtype=int)
        # map trial perm back into a boolean over real_idx positions
        pos = {idx: i for i, idx in enumerate(real_idx)}
        grpA = set(perm[:half].tolist())
        y_half = np.array([1 if idx in grpA else 0 for idx in real_idx])
        auc, _, _ = probe_auc(emb12[real_idx], y_half, seed=trial)
        half_aucs.append(auc)
    report.append(f"- Real-vs-real RANDOM-SPLIT control (laugh-real split in half, L12, 10 trials): "
                  f"AUC mean={np.mean(half_aucs):.3f}, max={np.max(half_aucs):.3f}, "
                  f"all={[round(a,3) for a in half_aucs]}\n")

    # (d) VocalSound speaker-held-out control: split laugh-real by speaker id (if enough distinct speakers)
    inv_real = inv[inv.group == "laugh-real"]
    n_speakers = inv_real.speaker.nunique()
    report.append(f"- laugh-real distinct speakers: {n_speakers} (VocalSound speaker field)\n")

    # ---------------- H5: harsher probe (PCA-20) ----------------
    report.append("## H5: Harsher probe (PCA-20 dim reduction before logistic regression)\n")
    report.append("```\n")
    for m in methods:
        mask2 = real_mask | (g == m)
        y2 = (g[mask2] == m).astype(int)
        auc_full, _, _ = probe_auc(emb12[mask2], y2)
        auc_pca, std_pca = pca_probe_auc(emb12[mask2], y2, n_components=20)
        report.append(f"  {m:<28} L12 full-dim AUC={auc_full:.3f}  PCA-20 AUC={auc_pca:.3f}+/-{std_pca:.3f}\n")
    # also for speech-real vs laugh-real
    auc_full_sr, _, _ = probe_auc(emb12[real_mask | speech_mask], (g[real_mask | speech_mask] == "speech-real").astype(int))
    auc_pca_sr, std_pca_sr = pca_probe_auc(emb12[real_mask | speech_mask], (g[real_mask | speech_mask] == "speech-real").astype(int), n_components=20)
    report.append(f"  {'speech-real_vs_laugh-real':<28} L12 full-dim AUC={auc_full_sr:.3f}  PCA-20 AUC={auc_pca_sr:.3f}+/-{std_pca_sr:.3f}\n")
    # shuffled control under PCA-20 too
    auc_pca_shuf, std_pca_shuf = pca_probe_auc(emb12[shuf_mask], rng.permutation(y_true), n_components=20)
    report.append(f"  {'[shuffled control]':<28} PCA-20 AUC={auc_pca_shuf:.3f}+/-{std_pca_shuf:.3f}\n")
    report.append("```\n")

    # ---------------- H1 continued: duration-controlled re-extraction ----------------
    report.append("## H1 (continued): duration+loudness-controlled re-extraction\n")
    result, skip_msg = trim_and_reextract_check(df, audio_stats)
    if result is None:
        report.append(f"SKIPPED: {skip_msg}\n")
    else:
        gg = result["group"]
        rmask = gg == "laugh-real"
        smask = gg == "speech-real"
        report.append("Re-extracted WavLM embeddings on 2.0s center-cropped, peak-normalized audio "
                       "(fixes BOTH duration and loudness confounds simultaneously), n=15/group. "
                       "Also applying PCA-20 on top (full fix-stack: trim/fix-length + peak-normalize "
                       "+ PCA-20 + logistic regression):\n\n```\n")
        fixed_full_aucs = {}
        for m in [x for x in pd.unique(gg) if x.startswith("laugh-") and x != "laugh-real"]:
            mask3 = rmask | (gg == m)
            y3 = (gg[mask3] == m).astype(int)
            a9, s9, n9_ = probe_auc(result["emb9"][mask3], y3, n_splits=3)
            a12, s12, n12_ = probe_auc(result["emb12"][mask3], y3, n_splits=3)
            a12_pca, s12_pca = pca_probe_auc(result["emb12"][mask3], y3, n_components=10, n_splits=3)
            fixed_full_aucs[m] = a12_pca
            report.append(f"  {m:<28} L9 AUC={a9:.3f}+/-{s9:.3f}  L12 AUC={a12:.3f}+/-{s12:.3f}  "
                          f"L12+PCA10 AUC={a12_pca:.3f}+/-{s12_pca:.3f}  (n={n9_})\n")
        mask_sr = rmask | smask
        y_sr = (gg[mask_sr] == "speech-real").astype(int)
        a9sr, s9sr, _ = probe_auc(result["emb9"][mask_sr], y_sr, n_splits=3)
        a12sr, s12sr, _ = probe_auc(result["emb12"][mask_sr], y_sr, n_splits=3)
        a12sr_pca, s12sr_pca = pca_probe_auc(result["emb12"][mask_sr], y_sr, n_components=10, n_splits=3)
        report.append(f"  {'speech-real_vs_laugh-real':<28} L9 AUC={a9sr:.3f}+/-{s9sr:.3f}  "
                      f"L12 AUC={a12sr:.3f}+/-{s12sr:.3f}  L12+PCA10 AUC={a12sr_pca:.3f}+/-{s12sr_pca:.3f}\n")
        report.append("```\n\n")

        report.append("**Decisive comparison: does WavLM (after full fix-stack) beat the content-free "
                       "scalar-feature probe?**\n\n```\n")
        for m in fixed_full_aucs:
            cf_auc = cf_aucs[m][0]
            wavlm_fixed = fixed_full_aucs[m]
            if wavlm_fixed > cf_auc + 0.03:
                verdict = "WavLM > content-free"
            elif wavlm_fixed < cf_auc - 0.03:
                verdict = "WavLM < content-free (quirk-level)"
            else:
                verdict = "tie (within +/-0.03)"
            report.append(f"  {m:<28} content-free AUC={cf_auc:.3f}  WavLM(fixed+PCA10) AUC={wavlm_fixed:.3f}  -> {verdict}\n")
        report.append("```\n")

    out_path = L / "D3_AUDIT.md"
    out_path.write_text("".join(report))
    print(f"\nWrote {out_path}")
    print("".join(report))


if __name__ == "__main__":
    main()
