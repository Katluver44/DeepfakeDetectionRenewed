#!/usr/bin/env python3
"""
E3d patch: Gaussian noise injection + write complete E3 CSV and figure.
Re-extracts WavLM L9/L12 frames (fast, single encoder only).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio.transforms as TAT
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path(__file__).parents[2]
OUT  = BASE / 'outputs'
FIG  = OUT / 'figures'

MANIFEST    = BASE / 'data/mlaad_en/manifest.json'
RESIDUAL_CSV = BASE / 'experiments/results/mlaad/residual_hardness_2/full_feature_table.csv'

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16_000
TARGET_LEN = 48_000

# Load data
resid_df = pd.read_csv(RESIDUAL_CSV)
y_map    = dict(zip(resid_df['system'], resid_df['residual']))
with open(MANIFEST) as f:
    manifest = json.load(f)
systems = [s for s in manifest if s in y_map]
y_arr   = np.array([y_map[s] for s in systems])

print(f"N={len(systems)} systems, device={DEVICE}")

# Load WavLM
from transformers import WavLMForCTC
wavlm = WavLMForCTC.from_pretrained('microsoft/wavlm-base').wavlm.eval().to(DEVICE)

def load_wav(path):
    arr, sr = sf.read(path, dtype='float32', always_2d=False)
    wav = torch.tensor(arr)
    if wav.ndim > 1: wav = wav.mean(0)
    if sr != TARGET_SR:
        wav = TAT.Resample(sr, TARGET_SR)(wav.unsqueeze(0)).squeeze(0)
    if len(wav) < TARGET_LEN:
        wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN].unsqueeze(0)

def extract_layers(wav_tensor, layers=(9, 12)):
    with torch.no_grad():
        hs = wavlm(wav_tensor.to(DEVICE), output_hidden_states=True).hidden_states
    return {li: hs[li][0].cpu().float().numpy() for li in layers}

def rog(frames):
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c)**2, 1))))

def vel_entropy(frames):
    vels = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    n = max(5, min(20, len(vels)//4))
    h, _ = np.histogram(vels, bins=n)
    h = h.astype(float) + 1e-8; h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def moving_avg(frames, window):
    T, D = frames.shape
    out = np.zeros_like(frames)
    half = window // 2
    for t in range(T):
        lo = max(0, t - half); hi = min(T, t + half + 1)
        out[t] = frames[lo:hi].mean(0)
    return out

def pca_compress(frames, k):
    mu = frames.mean(0)
    fc = frames - mu
    U, s, Vt = np.linalg.svd(fc, full_matrices=False)
    return U[:, :k] * s[:k]

def shuffle_frames(frames, rng):
    return frames[rng.permutation(len(frames))]

def gaussian_noise(frames, sigma, rng):
    return frames + rng.normal(0, sigma, frames.shape)

# Extract WavLM L9/L12 frames for all systems
print("Extracting WavLM L9/L12 frames...", flush=True)
sys_frames_l9  = []
sys_frames_l12 = []
for si, sname in enumerate(systems):
    wavs = manifest[sname][:2]
    utts_l9 = []
    utts_l12 = []
    for wp in wavs:
        wav = load_wav(wp)
        layers = extract_layers(wav)
        utts_l9.append(layers[9])
        utts_l12.append(layers[12])
    sys_frames_l9.append(utts_l9)
    sys_frames_l12.append(utts_l12)
    if (si+1) % 20 == 0:
        print(f"  {si+1}/{len(systems)}", flush=True)

del wavlm; torch.cuda.empty_cache()

# Reference values
C_orig = np.array([np.mean([-rog(u) for u in sys_frames_l12[si]]) for si in range(len(systems))])
T_orig = np.array([np.mean([vel_entropy(u) for u in sys_frames_l9[si]]) for si in range(len(systems))])
r_C_orig, _ = stats.pearsonr(C_orig, y_arr)
r_T_orig, _ = stats.pearsonr(T_orig, y_arr)
print(f"\nOriginal: r(C)={r_C_orig:.3f}  r(T)={r_T_orig:.3f}")
print(f"T_orig mean={T_orig.mean():.4f}, C_orig mean={C_orig.mean():.4f}")

# ─── E3d: Gaussian noise ──────────────────────────────────────────────────────
# Compute embedding std from all L9 frames
all_l9 = np.vstack([u for si in range(len(systems)) for u in sys_frames_l9[si]])
X_std = float(np.std(all_l9))
print(f"Embedding std (L9): {X_std:.4f}")

print("\nE3d: Gaussian noise injection:")
e3d_rows = []
for sigma_scale in [0.1, 0.5, 1.0, 2.0]:
    sigma = sigma_scale * X_std
    rng2 = np.random.default_rng(42)
    T_noise = np.array([np.mean([vel_entropy(gaussian_noise(u, sigma, rng2))
                                   for u in sys_frames_l9[si]])
                         for si in range(len(systems))])
    rng3 = np.random.default_rng(42)
    C_noise = np.array([np.mean([-rog(gaussian_noise(u, sigma, rng3))
                                   for u in sys_frames_l12[si]])
                         for si in range(len(systems))])
    r_T, _ = stats.pearsonr(T_noise, y_arr)
    r_C, _ = stats.pearsonr(C_noise, y_arr)
    print(f"  σ={sigma_scale:.1f}σ_emb: T {T_noise.mean():.3f} (Δ={T_noise.mean()-T_orig.mean():+.3f}) r(T,h)={r_T:+.3f} | "
          f"C {C_noise.mean():.3f} (Δ={C_noise.mean()-C_orig.mean():+.3f}) r(C,h)={r_C:+.3f}")
    e3d_rows.extend([
        {'perturbation': 'gaussian_noise', 'param': f'sigma={sigma_scale}', 'metric': 'T',
         'mean_val': float(T_noise.mean()), 'r_with_hardness': r_T,
         'delta_mean': float(T_noise.mean() - T_orig.mean()), 'delta_r': r_T - r_T_orig},
        {'perturbation': 'gaussian_noise', 'param': f'sigma={sigma_scale}', 'metric': 'C',
         'mean_val': float(C_noise.mean()), 'r_with_hardness': r_C,
         'delta_mean': float(C_noise.mean() - C_orig.mean()), 'delta_r': r_C - r_C_orig},
    ])

# ─── Build complete E3 CSV from stdout results + E3d ─────────────────────────
# Reconstruct E3a/b/c from re-running on newly extracted frames
e3_rows = [
    {'perturbation': 'none', 'param': 'original', 'metric': 'C',
     'mean_val': float(C_orig.mean()), 'r_with_hardness': r_C_orig, 'delta_mean': 0.0, 'delta_r': 0.0},
    {'perturbation': 'none', 'param': 'original', 'metric': 'T',
     'mean_val': float(T_orig.mean()), 'r_with_hardness': r_T_orig, 'delta_mean': 0.0, 'delta_r': 0.0},
]

# E3a: smoothing
print("\nRe-running E3a smoothing...")
for win in [3, 5, 11, 21]:
    T_sm = np.array([np.mean([vel_entropy(moving_avg(u, win)) for u in sys_frames_l9[si]])
                      for si in range(len(systems))])
    C_sm = np.array([np.mean([-rog(moving_avg(u, win)) for u in sys_frames_l12[si]])
                      for si in range(len(systems))])
    r_T, _ = stats.pearsonr(T_sm, y_arr)
    r_C, _ = stats.pearsonr(C_sm, y_arr)
    e3_rows += [
        {'perturbation': 'smoothing', 'param': f'win={win}', 'metric': 'T',
         'mean_val': float(T_sm.mean()), 'r_with_hardness': r_T,
         'delta_mean': float(T_sm.mean()-T_orig.mean()), 'delta_r': r_T - r_T_orig},
        {'perturbation': 'smoothing', 'param': f'win={win}', 'metric': 'C',
         'mean_val': float(C_sm.mean()), 'r_with_hardness': r_C,
         'delta_mean': float(C_sm.mean()-C_orig.mean()), 'delta_r': r_C - r_C_orig},
    ]

# E3b: PCA
print("Re-running E3b PCA...")
for k in [32, 64, 128, 256, 512]:
    C_pca = np.array([np.mean([-rog(pca_compress(u, k)) for u in sys_frames_l12[si]])
                       for si in range(len(systems))])
    r_C, _ = stats.pearsonr(C_pca, y_arr)
    e3_rows.append({'perturbation': 'pca_compress', 'param': f'k={k}', 'metric': 'C',
                     'mean_val': float(C_pca.mean()), 'r_with_hardness': r_C,
                     'delta_mean': float(C_pca.mean()-C_orig.mean()), 'delta_r': r_C - r_C_orig})

# E3c: shuffle (use mean of 3 trials)
print("Re-running E3c shuffle...")
rng_s = np.random.default_rng(42)
T_shuf_all = []
C_shuf_all = []
for _ in range(3):
    T_shuf = np.array([np.mean([vel_entropy(shuffle_frames(u, rng_s)) for u in sys_frames_l9[si]])
                        for si in range(len(systems))])
    C_shuf = np.array([np.mean([-rog(shuffle_frames(u, rng_s)) for u in sys_frames_l12[si]])
                        for si in range(len(systems))])
    T_shuf_all.append(T_shuf); C_shuf_all.append(C_shuf)
T_shuf_mean = np.mean(T_shuf_all, 0)
C_shuf_mean = np.mean(C_shuf_all, 0)
r_T_shuf, _ = stats.pearsonr(T_shuf_mean, y_arr)
r_C_shuf, _ = stats.pearsonr(C_shuf_mean, y_arr)
e3_rows += [
    {'perturbation': 'shuffle', 'param': 'mean3trials', 'metric': 'T',
     'mean_val': float(T_shuf_mean.mean()), 'r_with_hardness': r_T_shuf,
     'delta_mean': float(T_shuf_mean.mean()-T_orig.mean()), 'delta_r': r_T_shuf - r_T_orig},
    {'perturbation': 'shuffle', 'param': 'mean3trials', 'metric': 'C',
     'mean_val': float(C_shuf_mean.mean()), 'r_with_hardness': r_C_shuf,
     'delta_mean': float(C_shuf_mean.mean()-C_orig.mean()), 'delta_r': r_C_shuf - r_C_orig},
]

# Add E3d
e3_rows.extend(e3d_rows)
pd.DataFrame(e3_rows).to_csv(OUT / 'e3_intervention_lite.csv', index=False)
print("\nSaved: e3_intervention_lite.csv")

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: smoothing
wins = [0, 3, 5, 11, 21]
T_sm_means = [float(T_orig.mean())] + [r['mean_val'] for r in e3_rows if r['perturbation']=='smoothing' and r['metric']=='T']
T_sm_rs    = [r_T_orig] + [r['r_with_hardness'] for r in e3_rows if r['perturbation']=='smoothing' and r['metric']=='T']
ax = axes[0][0]
ax.plot(wins, T_sm_means, 'o-', color='#1f77b4', label='T mean')
ax2 = ax.twinx()
ax2.plot(wins, T_sm_rs, 's--', color='#d62728', label='r(T,h)')
ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel('Smoothing window'); ax.set_ylabel('Mean T', color='#1f77b4')
ax2.set_ylabel('r(T, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')
ax.set_title('3a: Smoothing → T decreases\nHardness correlation weakens → zero → negative', fontsize=9)
ax.legend(loc='upper right', fontsize=8); ax2.legend(loc='lower right', fontsize=8)

# 3b: PCA
ks = [32, 64, 128, 256, 512, 768]
C_pca_means = [r['mean_val'] for r in e3_rows if r['perturbation']=='pca_compress' and r['metric']=='C'] + [float(C_orig.mean())]
C_pca_rs    = [r['r_with_hardness'] for r in e3_rows if r['perturbation']=='pca_compress' and r['metric']=='C'] + [r_C_orig]
ax = axes[0][1]
ax.plot(ks, C_pca_means, 'o-', color='#ff7f0e', label='C mean')
ax2 = ax.twinx()
ax2.plot(ks, C_pca_rs, 's--', color='#d62728', label='r(C,h)')
ax.set_xlabel('PCA dim (of 768)'); ax.set_ylabel('Mean C', color='#ff7f0e')
ax2.set_ylabel('r(C, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')
ax.set_title('3b: PCA compression → C robust\n(signal concentrated in top components)', fontsize=9)
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='lower right', fontsize=8)

# 3c: shuffle
shuf_r_T = [r['r_with_hardness'] for r in e3_rows if r['perturbation']=='shuffle' and r['metric']=='T'][0]
shuf_r_C = [r['r_with_hardness'] for r in e3_rows if r['perturbation']=='shuffle' and r['metric']=='C'][0]
ax = axes[1][0]
ax.bar(['Original T', 'Shuffled T (mean)'], [float(T_orig.mean()), float(T_shuf_mean.mean())],
       color=['#1f77b4', '#aec7e8'], edgecolor='black')
ax.set_ylabel('Mean T (vel_entropy)')
ax2 = ax.twinx()
ax2.plot(['Original T', 'Shuffled T (mean)'], [r_T_orig, shuf_r_T], 'D-', color='#d62728')
ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax2.set_ylabel('r(T, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')
ax.set_title(f'3c: Frame shuffle\nT changes, r(T,h): {r_T_orig:.3f} → {shuf_r_T:.3f}\nC unchanged r(C,h): {r_C_orig:.3f} → {shuf_r_C:.3f}', fontsize=9)

# 3d: noise
sigmas = [0.0, 0.1, 0.5, 1.0, 2.0]
T_noise_rs = [r_T_orig] + [r['r_with_hardness'] for r in e3d_rows if r['metric']=='T']
C_noise_rs = [r_C_orig] + [r['r_with_hardness'] for r in e3d_rows if r['metric']=='C']
ax = axes[1][1]
ax.plot(sigmas, T_noise_rs, 'o-', color='#1f77b4', label='r(T, hardness)', linewidth=1.8)
ax.plot(sigmas, C_noise_rs, 's-', color='#ff7f0e', label='r(C, hardness)', linewidth=1.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Noise amplitude (× embedding SD)')
ax.set_ylabel('Pearson r with residual hardness')
ax.set_title('3d: Gaussian noise → signal degrades\n(T more noise-sensitive than C)', fontsize=9)
ax.legend(fontsize=9)

plt.suptitle('E3: Intervention-lite — Perturbation Response of C and T', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'e3_intervention_lite.png', dpi=150)
plt.close()
print("Saved: e3_intervention_lite.png")

# Print summary table for easy reading
print("\n=== E3 SUMMARY ===")
print(f"{'Perturbation':<30} {'Param':<12} {'Metric':>6} {'Δmean':>8} {'r':>8} {'Δr':>8}")
print("-"*75)
for row in e3_rows:
    print(f"{row['perturbation']:<30} {row['param']:<12} {row['metric']:>6} "
          f"{row['delta_mean']:>8.3f} {row['r_with_hardness']:>8.3f} {row['delta_r']:>8.3f}")
