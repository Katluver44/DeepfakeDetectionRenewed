"""D3 (confound-controlled rerun) — real vs synthetic laughter in WavLM space,
after removing the recording-pipeline confounds the audit (D3_AUDIT.md) exposed.

Per-clip normalization applied to EVERY clip before WavLM:
  1. energy-trim leading/trailing silence (>1% of peak),
  2. take a fixed FIXED_DUR-second segment of *active* audio (center of the
     trimmed signal; tile-pad if shorter) -> equalizes clip length AND silence,
  3. peak-normalize to 0.9 -> equalizes loudness.
Then extract WavLM-Large L9/L12 mean embeddings and run the same probes.

Balanced-N per group (min group size, capped) so no group-size artifact.

Reports, for real-vs-each-synthetic and the speech-vs-laughter sanity pair:
  - WavLM AUC (L9, L12, and L12+PCA-20)   -- the corrected finding
  - CONTENT-FREE 8-scalar AUC on the SAME fixed clips  -- the floor/baseline
  - label-shuffle and real-vs-real controls (must be ~0.5)
A claim of a genuine laughter-authenticity signal requires WavLM to BEAT the
content-free floor by a clear margin AFTER these fixes.

Outputs: tables/table_d3_fixed.csv, figures/figure_d3_fixed.png|pdf, and prints
a summary. Does not overwrite the old (audited) d3 artifacts.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = Path(__file__).resolve().parents[1]
SR = 16000
FIXED_DUR = 2.0
N_SAMP = int(FIXED_DUR * SR)
SEED = 0
GROUPS = ["laugh-real", "speech-real", "laugh-bark_laughter_token",
          "laugh-bark_laughs_inline", "laugh-audioldm2", "laugh-parler_tts", "laugh-xtts"]
COLORS = {"laugh-real": "#2563eb", "speech-real": "#9ca3af",
          "laugh-bark_laughter_token": "#dc2626", "laugh-bark_laughs_inline": "#f59e0b",
          "laugh-parler_tts": "#7c3aed", "laugh-audioldm2": "#059669", "laugh-xtts": "#db2777"}


def load_fixed(path):
    """Load, energy-trim silence, take fixed central active segment, peak-norm."""
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
    """8 scalars carrying no laughter-content info, on the FIXED clip (so
    duration/silence are already equalized -> this is the residual floor)."""
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


def probe(X, y, seed=0, pca=None):
    ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
    nmin = min((y == 0).sum(), (y == 1).sum())
    if nmin < 2:
        return np.nan, np.nan
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, X.shape[0] - 1, X.shape[1]), random_state=seed))
    steps.append(LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(min(5, nmin), shuffle=True, random_state=seed)
    s = cross_val_score(make_pipeline(*steps), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def main():
    inv = pd.read_csv(L / "embeddings" / "d3_multi_inventory.csv")
    # balanced N per group
    rng = np.random.RandomState(SEED)
    ncap = min(inv.group.value_counts().min(), 100)
    sel = []
    for grp in GROUPS:
        sub = inv[inv.group == grp]
        idx = rng.choice(len(sub), min(ncap, len(sub)), replace=False)
        sel.append(sub.iloc[idx])
    sel = pd.concat(sel).reset_index(drop=True)
    print(f"[d3_fixed] balanced N={ncap}/group, {len(sel)} clips total")

    import torch
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(dev).eval()

    emb9, emb12, cf, groups = [], [], [], []
    with torch.no_grad():
        for i, r in sel.iterrows():
            a = load_fixed(r.file_path)
            cf.append(content_free_feats(a))
            iv = fe(a, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            hs = model(iv).hidden_states
            emb9.append(hs[9][0].mean(0).cpu().numpy())
            emb12.append(hs[12][0].mean(0).cpu().numpy())
            groups.append(r.group)
    emb9, emb12, cf, g = np.array(emb9), np.array(emb12), np.array(cf), np.array(groups)

    real = g == "laugh-real"; speech = g == "speech-real"
    methods = [m for m in GROUPS if m not in ("laugh-real", "speech-real")]
    rows = []
    print(f"\n{'comparison':<30}{'WavLM L12':>11}{'L12+PCA20':>11}{'content-free':>14}{'verdict':>22}")
    for m in methods + ["speech-real"]:
        if m == "speech-real":
            mask = real | speech; y = (g[mask] == "speech-real").astype(int); label = "speech vs laugh (sanity)"
        else:
            mask = real | (g == m); y = (g[mask] == m).astype(int); label = f"real vs {m.replace('laugh-','')}"
        a12, s12 = probe(emb12[mask], y)
        a12p, _ = probe(emb12[mask], y, pca=20)
        acf, _ = probe(cf[mask], y)
        if a12p > acf + 0.05:
            v = "WavLM>floor (signal)"
        elif a12p < acf - 0.05:
            v = "WavLM<floor"
        else:
            v = "tie w/ floor (quirk)"
        rows.append({"comparison": label, "wavlm_l12": round(a12, 3), "wavlm_l12_pca20": round(a12p, 3),
                     "content_free": round(acf, 3), "verdict": v})
        print(f"{label:<30}{a12:>11.3f}{a12p:>11.3f}{acf:>14.3f}{v:>22}")

    # controls
    y_bt = (g[real | (g == 'laugh-bark_laughter_token')] == 'laugh-bark_laughter_token').astype(int)
    Xbt = emb12[real | (g == 'laugh-bark_laughter_token')]
    shuf = np.mean([probe(Xbt, rng.permutation(y_bt), seed=t)[0] for t in range(5)])
    ridx = np.where(real)[0]
    rr = np.mean([probe(emb12[ridx], np.array([1 if x in set(rng.permutation(ridx)[:len(ridx)//2]) else 0 for x in ridx]), seed=t)[0] for t in range(5)])
    print(f"\ncontrols: label-shuffle={shuf:.3f} (want ~0.5), real-vs-real={rr:.3f} (want ~0.5)")

    with open(L / "tables" / "table_d3_fixed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        w.writerow({}); w.writerow({"comparison": f"label_shuffle_control={shuf:.3f}", "verdict": f"real_vs_real={rr:.3f}"})
    print("wrote tables/table_d3_fixed.csv")

    # UMAP on fixed embeddings
    try:
        import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, emb, nm in [(axes[0], emb9, "L9"), (axes[1], emb12, "L12")]:
            Z = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
            for grp in GROUPS:
                mk = g == grp
                if mk.sum():
                    ax.scatter(Z[mk, 0], Z[mk, 1], s=14, alpha=0.7, c=COLORS[grp], label=grp.replace("laugh-", ""))
            ax.set_title(f"WavLM {nm} (length/silence/loudness-controlled)"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        axes[0].legend(fontsize=7)
        fig.suptitle("D3 (confound-controlled): real vs synthetic laughter")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(L / "figures" / f"figure_d3_fixed.{e}", dpi=200, bbox_inches="tight")
        print("wrote figures/figure_d3_fixed.png|pdf")
    except Exception as e:
        print(f"UMAP skipped: {e}")


if __name__ == "__main__":
    main()
