#!/usr/bin/env python3
"""
Experiments 1, 2, 3: Representation Invariance · Graph Independence · Intervention-lite

E1 — Re-extract C (rog) and T (vel_entropy) with three encoders (WavLM, HuBERT, wav2vec2)
     across all 13 layers and correlate with residual hardness.

E2 — Test whether T's rank ordering is stable across alternative velocity definitions
     (cosine, different strides, pseudo-phoneme segments, window sizes).

E3 — Apply directional perturbations (trajectory smoothing, PCA compression, frame shuffle)
     and check if C and T respond in the expected direction, and whether the
     hardness correlation breaks predictably.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio.transforms as TAT
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
OUT  = BASE / 'outputs'
FIG  = OUT  / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / '.frames_cache'
CACHE_DIR.mkdir(exist_ok=True)

MANIFEST    = BASE / 'data/mlaad_en/manifest.json'
RESIDUAL_CSV = BASE / 'experiments/results/mlaad/residual_hardness_2/full_feature_table.csv'
SIG_CSV     = OUT  / 'sig_layer_features.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16_000
print(f"Device: {DEVICE}")

# ─── Load outcomes ────────────────────────────────────────────────────────────
resid_df = pd.read_csv(RESIDUAL_CSV)
sig_df   = pd.read_csv(SIG_CSV)
y_map    = dict(zip(resid_df['system'], resid_df['residual']))
eer_map  = dict(zip(resid_df['system'], resid_df['observed_eer']))

# ─── Load manifest ────────────────────────────────────────────────────────────
with open(MANIFEST) as f:
    manifest = json.load(f)

# Filter to systems with known residual
systems = [s for s in manifest if s in y_map]
print(f"Systems with residual: {len(systems)} (manifest has {len(manifest)} total)")

# ─── Audio helpers ────────────────────────────────────────────────────────────
TARGET_LEN = 48_000   # 3 s

def load_wav(path: str) -> torch.Tensor:
    """Load wav, mono, 16 kHz, crop/pad to 3 s."""
    arr, sr = sf.read(path, dtype='float32', always_2d=False)
    wav = torch.tensor(arr)
    if wav.ndim > 1:
        wav = wav.mean(0)
    if sr != TARGET_SR:
        wav = TAT.Resample(sr, TARGET_SR)(wav.unsqueeze(0)).squeeze(0)
    if len(wav) < TARGET_LEN:
        wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN].unsqueeze(0)  # (1, T)

# ─── Trajectory metrics ───────────────────────────────────────────────────────

def rog(frames: np.ndarray) -> float:
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c)**2, 1))))

def vel_entropy(frames: np.ndarray, stride: int = 1, metric: str = 'l2', n_bins: int = None) -> float:
    if len(frames) <= stride:
        return 0.0
    f1 = frames[:-stride]
    f2 = frames[stride:]
    if metric == 'l2':
        vels = np.linalg.norm(f2 - f1, axis=1)
    elif metric == 'cosine':
        n1 = np.linalg.norm(f1, axis=1, keepdims=True).clip(1e-8)
        n2 = np.linalg.norm(f2, axis=1, keepdims=True).clip(1e-8)
        vels = 1.0 - np.sum((f1/n1) * (f2/n2), axis=1)
    else:
        raise ValueError(metric)
    n = n_bins or max(5, min(20, len(vels) // 4))
    hist, _ = np.histogram(vels, bins=n)
    h = hist.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def vel_entropy_window(frames: np.ndarray, window: int) -> float:
    """Velocity between center of consecutive windows."""
    k = window // 2
    T = len(frames)
    vels = []
    for t in range(k, T - k - 1):
        f1 = frames[t-k:t+k+1].mean(0)
        f2 = frames[t-k+1:t+k+2].mean(0)
        vels.append(np.linalg.norm(f2 - f1))
    if not vels:
        return 0.0
    vels = np.array(vels)
    n = max(5, min(20, len(vels) // 4))
    hist, _ = np.histogram(vels, bins=n)
    h = hist.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def vel_entropy_kmeans(frames: np.ndarray, k: int = 40) -> float:
    """Pseudo-phoneme velocity entropy: k-means cluster transitions."""
    if len(frames) < k:
        k = max(2, len(frames) // 2)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(frames)
    centroids = km.cluster_centers_
    seq = [centroids[labels[t]] for t in range(len(labels))]
    vels = [np.linalg.norm(seq[t+1] - seq[t]) for t in range(len(seq)-1)]
    if not vels:
        return 0.0
    vels = np.array(vels)
    n = max(5, min(20, len(vels) // 4))
    hist, _ = np.histogram(vels, bins=n)
    h = hist.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def loo_r2_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        idx = list(range(n)); idx.pop(i)
        sc = StandardScaler()
        xt = sc.fit_transform(x[idx].reshape(-1,1))
        xe = sc.transform(x[[i]].reshape(-1,1))
        preds[i] = Ridge(alpha=alpha).fit(xt, y[idx]).predict(xe)[0]
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res / ss_tot

# ─── Encoder loading ──────────────────────────────────────────────────────────
def load_encoders():
    from transformers import WavLMForCTC, HubertModel, Wav2Vec2Model
    encoders = {}

    print("Loading WavLM...", flush=True)
    wavlm = WavLMForCTC.from_pretrained('microsoft/wavlm-base').wavlm.eval().to(DEVICE)
    encoders['WavLM'] = wavlm

    print("Loading HuBERT...", flush=True)
    hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960').eval().to(DEVICE)
    encoders['HuBERT'] = hubert

    print("Loading wav2vec2...", flush=True)
    w2v2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').eval().to(DEVICE)
    encoders['wav2vec2'] = w2v2

    return encoders

# ─── Frame extraction ─────────────────────────────────────────────────────────
def extract_frames(encoder, wav_tensor: torch.Tensor) -> list:
    """Returns list of 13 np.ndarray (T', 768), one per layer."""
    x = wav_tensor.to(DEVICE)
    with torch.no_grad():
        out = encoder(x, output_hidden_states=True)
    return [hs[0].cpu().float().numpy() for hs in out.hidden_states]

# ─── E1: Representation Invariance ───────────────────────────────────────────
print("\n" + "="*70)
print("EXPERIMENT 1: REPRESENTATION INVARIANCE")
print("="*70)

encoders = load_encoders()
N_LAYERS = 13

# Results table: encoder × layer × metric → (pearson_r, spearman_rho, loo_r2)
e1_results = {}
# Cache: system → encoder → utterance → layer → frames
frames_cache = {}  # system → encoder → [utt_idx] → [layer] → np.ndarray

y_arr     = np.array([y_map[s]   for s in systems])
eer_arr   = np.array([eer_map[s] for s in systems])

for enc_name, encoder in encoders.items():
    print(f"\n--- Encoder: {enc_name} ---")
    frames_cache[enc_name] = {}

    # Per-system per-layer metrics
    sys_rog  = np.full((len(systems), N_LAYERS), np.nan)
    sys_vent = np.full((len(systems), N_LAYERS), np.nan)

    for si, sname in enumerate(systems):
        wavs = manifest[sname][:2]
        utt_rog  = np.zeros((len(wavs), N_LAYERS))
        utt_vent = np.zeros((len(wavs), N_LAYERS))
        frames_cache[enc_name][sname] = []

        for ui, wav_path in enumerate(wavs):
            wav = load_wav(wav_path)
            layers = extract_frames(encoder, wav)   # list of 13 arrays
            frames_cache[enc_name][sname].append(layers)
            for li, frs in enumerate(layers):
                utt_rog[ui, li]  = rog(frs)
                utt_vent[ui, li] = vel_entropy(frs)

        sys_rog[si]  = utt_rog.mean(0)
        sys_vent[si] = utt_vent.mean(0)

        if (si + 1) % 15 == 0:
            print(f"  {si+1}/{len(systems)} systems processed", flush=True)

    # Correlate per-layer values with residual hardness and EER
    enc_results = {}
    for li in range(N_LAYERS):
        # rog: negative correlation → use -rog as C proxy
        neg_rog = -sys_rog[:, li]
        vent    = sys_vent[:, li]

        # Filter NaN
        mask = np.isfinite(neg_rog) & np.isfinite(vent)

        r_rog,  p_rog  = stats.pearsonr(-sys_rog[mask, li], y_arr[mask])
        rho_rog, _     = stats.spearmanr(-sys_rog[mask, li], y_arr[mask])
        r_vent, p_vent = stats.pearsonr(vent[mask], y_arr[mask])
        rho_vent, _    = stats.spearmanr(vent[mask], y_arr[mask])

        loo_rog  = loo_r2_ridge(-sys_rog[mask, li], y_arr[mask])
        loo_vent = loo_r2_ridge(vent[mask], y_arr[mask])

        enc_results[li] = {
            'rog_r': r_rog, 'rog_p': p_rog, 'rog_rho': rho_rog, 'rog_loo': loo_rog,
            'vent_r': r_vent, 'vent_p': p_vent, 'vent_rho': rho_vent, 'vent_loo': loo_vent,
        }

    e1_results[enc_name] = enc_results

    # Print summary
    print(f"  Layer | rog_r  rog_p   rog_loo | vent_r  vent_p  vent_loo")
    print(f"  {'-'*60}")
    for li in range(N_LAYERS):
        r = enc_results[li]
        print(f"  L{li:02d}  | {r['rog_r']:+.3f}  {r['rog_p']:.3f}  {r['rog_loo']:+.3f}  | "
              f"{r['vent_r']:+.3f}  {r['vent_p']:.3f}  {r['vent_loo']:+.3f}")

# Free encoder memory (keep frames_cache)
del encoders; torch.cuda.empty_cache()

# ─── Save E1 CSV ──────────────────────────────────────────────────────────────
e1_rows = []
for enc_name, enc_res in e1_results.items():
    for li, metrics in enc_res.items():
        e1_rows.append({'encoder': enc_name, 'layer': li, **metrics})
e1_df = pd.DataFrame(e1_rows)
e1_df.to_csv(OUT / 'e1_representation_invariance.csv', index=False)

# ─── E1 Figures ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
encoder_colors = {'WavLM': '#1f77b4', 'HuBERT': '#ff7f0e', 'wav2vec2': '#2ca02c'}
layers = list(range(N_LAYERS))

for ax_idx, (metric_key, title, ylabel) in enumerate([
    ('rog_r', 'Pearson r: C (−rog) vs residual hardness', 'Pearson r'),
    ('vent_r', 'Pearson r: T (vel_entropy) vs residual hardness', 'Pearson r'),
    ('rog_loo', 'LOO R²: C (−rog) vs residual hardness', 'LOO R²'),
    ('vent_loo', 'LOO R²: T (vel_entropy) vs residual hardness', 'LOO R²'),
]):
    ax = axes[ax_idx // 2][ax_idx % 2]
    for enc_name, enc_res in e1_results.items():
        vals = [enc_res[li][metric_key] for li in layers]
        ax.plot(layers, vals, 'o-', label=enc_name,
                color=encoder_colors[enc_name], linewidth=1.8, markersize=5)
    ax.axhline(0, color='black', linewidth=0.8)
    if 'loo' not in metric_key:
        ax.axhline(-0.05, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('WavLM / HuBERT / wav2vec2 Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{i}' for i in layers], fontsize=7, rotation=45)
    ax.legend(fontsize=9)
plt.suptitle('E1: Representation Invariance — C and T across Encoders', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'e1_representation_invariance.png', dpi=150)
plt.close()
print("\nSaved: e1_representation_invariance.png")

# ─── E2: Graph Independence (T robustness) ────────────────────────────────────
print("\n" + "="*70)
print("EXPERIMENT 2: GRAPH INDEPENDENCE / T ROBUSTNESS")
print("="*70)

# Use WavLM frames (already cached) at L9 for T variants and L12 for C variants
TARGET_LAYER_T = 9   # vel_entropy peak layer for WavLM
TARGET_LAYER_C = 12  # rog peak layer for WavLM

enc_name = 'WavLM'

# Collect per-system per-utterance frames at L9 and L12
sys_frames_l9  = []
sys_frames_l12 = []
for sname in systems:
    utts_l9  = [frames_cache[enc_name][sname][ui][TARGET_LAYER_T]
                for ui in range(len(manifest[sname][:2]))]
    utts_l12 = [frames_cache[enc_name][sname][ui][TARGET_LAYER_C]
                for ui in range(len(manifest[sname][:2]))]
    sys_frames_l9.append(utts_l9)
    sys_frames_l12.append(utts_l12)

def compute_sys_T_variant(frames_list, fn):
    vals = []
    for utts in frames_list:
        utt_vals = [fn(u) for u in utts]
        vals.append(np.mean(utt_vals))
    return np.array(vals)

# T variants
T_variants = {
    'T_l2_s1 (original)': lambda f: vel_entropy(f, stride=1, metric='l2'),
    'T_cosine_s1':         lambda f: vel_entropy(f, stride=1, metric='cosine'),
    'T_l2_s2':             lambda f: vel_entropy(f, stride=2, metric='l2'),
    'T_l2_s4':             lambda f: vel_entropy(f, stride=4, metric='l2'),
    'T_window5':           lambda f: vel_entropy_window(f, window=5),
    'T_window11':          lambda f: vel_entropy_window(f, window=11),
    'T_kmeans40':          lambda f: vel_entropy_kmeans(f, k=40),
}

# C variants
C_variants = {
    'C_L12 (original)': lambda f: rog(f),
    'C_L11':            None,  # computed from L11 frames separately
    'C_L10':            None,
}

print("\nComputing T variants...")
T_sys = {}
for name, fn in T_variants.items():
    T_sys[name] = compute_sys_T_variant(sys_frames_l9, fn)
    r, p = stats.pearsonr(T_sys[name], y_arr)
    rho, _ = stats.spearmanr(T_sys[name], y_arr)
    loo = loo_r2_ridge(T_sys[name], y_arr)
    print(f"  {name:<28}: r={r:+.3f}  ρ={rho:+.3f}  LOO R²={loo:+.4f}  p={p:.3f}")

# Rank consistency
T_ref = T_sys['T_l2_s1 (original)']
ranks_ref = stats.rankdata(T_ref)
print("\nSpearman ρ between T_original and each variant:")
for name, t_vals in T_sys.items():
    if name == 'T_l2_s1 (original)': continue
    rho_vs_ref, _ = stats.spearmanr(T_ref, t_vals)
    print(f"  T_original vs {name:<28}: ρ={rho_vs_ref:.3f}")

# C at alternative layers
print("\nC (rog) at alternative layers (WavLM):")
C_sys = {}
for li in [9, 10, 11, 12]:
    c_vals = []
    for sname in systems:
        utts = [frames_cache[enc_name][sname][ui][li]
                for ui in range(len(manifest[sname][:2]))]
        c_vals.append(np.mean([-rog(u) for u in utts]))  # negate
    C_sys[f'C_L{li}'] = np.array(c_vals)
    r, p = stats.pearsonr(C_sys[f'C_L{li}'], y_arr)
    rho, _ = stats.spearmanr(C_sys[f'C_L{li}'], y_arr)
    loo = loo_r2_ridge(C_sys[f'C_L{li}'], y_arr)
    print(f"  C_L{li}: r={r:+.3f}  ρ={rho:+.3f}  LOO R²={loo:+.4f}  p={p:.3f}")

# Save E2 CSV
e2_rows = []
for name, t_vals in T_sys.items():
    r, p = stats.pearsonr(t_vals, y_arr)
    rho, _ = stats.spearmanr(t_vals, y_arr)
    loo = loo_r2_ridge(t_vals, y_arr)
    rho_vs_ref, _ = stats.spearmanr(T_ref, t_vals)
    e2_rows.append({'variant': name, 'type': 'T', 'pearson_r': r, 'spearman_rho': rho,
                    'loo_r2': loo, 'p': p, 'rho_vs_reference': rho_vs_ref})
for name, c_vals in C_sys.items():
    r, p = stats.pearsonr(c_vals, y_arr)
    rho, _ = stats.spearmanr(c_vals, y_arr)
    loo = loo_r2_ridge(c_vals, y_arr)
    e2_rows.append({'variant': name, 'type': 'C', 'pearson_r': r, 'spearman_rho': rho,
                    'loo_r2': loo, 'p': p, 'rho_vs_reference': 1.0})
pd.DataFrame(e2_rows).to_csv(OUT / 'e2_graph_independence.csv', index=False)

# E2 Figure: T variant rank correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
variant_names = list(T_sys.keys())
rhos_vs_ref = [stats.spearmanr(T_ref, T_sys[n])[0] for n in variant_names]
colors = ['#d62728' if n == 'T_l2_s1 (original)' else '#1f77b4' for n in variant_names]
bars = ax.barh(range(len(variant_names)), rhos_vs_ref, color=colors)
ax.set_yticks(range(len(variant_names))); ax.set_yticklabels(variant_names, fontsize=9)
ax.set_xlabel("Spearman ρ vs. T_original (rank consistency)")
ax.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='ρ=0.8 threshold')
ax.axvline(0.6, color='orange', linestyle='--', alpha=0.7, label='ρ=0.6 threshold')
ax.legend(fontsize=8); ax.set_title("T Variant Rank Consistency\n(ρ>0.8 = robust)", fontsize=10)

ax2 = axes[1]
vent_corrs = [stats.pearsonr(T_sys[n], y_arr)[0] for n in variant_names]
ax2.barh(range(len(variant_names)), vent_corrs, color=colors)
ax2.set_yticks(range(len(variant_names))); ax2.set_yticklabels(variant_names, fontsize=9)
ax2.set_xlabel("Pearson r with residual hardness")
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_title("T Variant Hardness Correlation\n(all variants should agree in sign)", fontsize=10)

plt.tight_layout()
plt.savefig(FIG / 'e2_graph_independence.png', dpi=150)
plt.close()
print("\nSaved: e2_graph_independence.png")

# ─── E3: Intervention-lite ────────────────────────────────────────────────────
print("\n" + "="*70)
print("EXPERIMENT 3: INTERVENTION-LITE")
print("="*70)

# For each perturbation: compute C and T from perturbed frames,
# then compute correlation with hardness. Report pre/post changes.

def moving_avg(frames, window):
    T = len(frames)
    out = np.zeros_like(frames)
    half = window // 2
    for t in range(T):
        lo = max(0, t - half); hi = min(T, t + half + 1)
        out[t] = frames[lo:hi].mean(0)
    return out

def pca_compress(frames, k):
    """Project to k-dim PCA subspace and stay in k-dim (return k-d array)."""
    mu = frames.mean(0)
    fc = frames - mu
    U, s, Vt = np.linalg.svd(fc, full_matrices=False)
    return U[:, :k] * s[:k]  # shape (T, k) — k-dim representations

def shuffle_frames(frames, rng=None):
    rng = rng or np.random.default_rng(42)
    idx = rng.permutation(len(frames))
    return frames[idx]

def gaussian_noise(frames, sigma, rng=None):
    rng = rng or np.random.default_rng(42)
    return frames + rng.normal(0, sigma, frames.shape)

# Original reference values
C_orig = np.array([np.mean([-rog(u) for u in sys_frames_l12[si]])
                   for si in range(len(systems))])
T_orig = np.array([np.mean([vel_entropy(u) for u in sys_frames_l9[si]])
                   for si in range(len(systems))])

r_C_orig, _ = stats.pearsonr(C_orig, y_arr)
r_T_orig, _ = stats.pearsonr(T_orig, y_arr)
print(f"\nOriginal: r(C, hardness)={r_C_orig:+.3f}  r(T, hardness)={r_T_orig:+.3f}")

e3_rows = [{'perturbation': 'none', 'param': 'original', 'metric': 'C',
             'mean_val': float(C_orig.mean()), 'r_with_hardness': r_C_orig,
             'delta_mean': 0.0, 'delta_r': 0.0},
           {'perturbation': 'none', 'param': 'original', 'metric': 'T',
             'mean_val': float(T_orig.mean()), 'r_with_hardness': r_T_orig,
             'delta_mean': 0.0, 'delta_r': 0.0}]

# 3a. Moving average smoothing (should decrease T, keep C roughly stable)
print("\n3a. Moving-average smoothing (should decrease T):")
for win in [3, 5, 11, 21]:
    T_smooth = np.array([np.mean([vel_entropy(moving_avg(u, win))
                                   for u in sys_frames_l9[si]])
                          for si in range(len(systems))])
    C_smooth = np.array([np.mean([-rog(moving_avg(u, win))
                                   for u in sys_frames_l12[si]])
                          for si in range(len(systems))])
    r_T, _ = stats.pearsonr(T_smooth, y_arr)
    r_C, _ = stats.pearsonr(C_smooth, y_arr)
    dT = T_smooth.mean() - T_orig.mean()
    dC = C_smooth.mean() - C_orig.mean()
    print(f"  win={win:2d}: T mean {T_smooth.mean():.3f} (Δ={dT:+.3f})  r(T,hardness)={r_T:+.3f} (Δ={r_T-r_T_orig:+.3f}) | "
          f"C mean {C_smooth.mean():.3f} (Δ={dC:+.3f})  r(C,hardness)={r_C:+.3f}")
    for metric, vals, r_val, ref_mean, ref_r in [
            ('T_smooth', T_smooth, r_T, float(T_orig.mean()), r_T_orig),
            ('C_smooth', C_smooth, r_C, float(C_orig.mean()), r_C_orig)]:
        e3_rows.append({'perturbation': 'smoothing', 'param': f'win={win}',
                         'metric': metric[:1],
                         'mean_val': float(vals.mean()),
                         'r_with_hardness': r_val,
                         'delta_mean': float(vals.mean()) - ref_mean,
                         'delta_r': r_val - ref_r})

# 3b. PCA compression (should decrease C in reduced space)
print("\n3b. PCA compression of L12 (should decrease C):")
for k in [32, 64, 128, 256, 512]:
    C_pca = np.array([np.mean([-rog(pca_compress(u, k))
                                for u in sys_frames_l12[si]])
                       for si in range(len(systems))])
    r_C, _ = stats.pearsonr(C_pca, y_arr)
    dC = C_pca.mean() - C_orig.mean()
    print(f"  k={k:3d}: C mean {C_pca.mean():.3f} (Δ={dC:+.3f})  r(C,hardness)={r_C:+.3f} (Δ={r_C-r_C_orig:+.3f})")
    e3_rows.append({'perturbation': 'pca_compress', 'param': f'k={k}',
                     'metric': 'C',
                     'mean_val': float(C_pca.mean()),
                     'r_with_hardness': r_C,
                     'delta_mean': float(C_pca.mean()) - float(C_orig.mean()),
                     'delta_r': r_C - r_C_orig})

# 3c. Frame shuffling (should increase T toward maximum; C should not change)
print("\n3c. Frame shuffling (should increase T, leave C unchanged):")
rng = np.random.default_rng(42)
for trial in range(3):
    T_shuf = np.array([np.mean([vel_entropy(shuffle_frames(u, rng))
                                  for u in sys_frames_l9[si]])
                        for si in range(len(systems))])
    C_shuf = np.array([np.mean([-rog(shuffle_frames(u, rng))
                                  for u in sys_frames_l12[si]])
                        for si in range(len(systems))])
    r_T, _ = stats.pearsonr(T_shuf, y_arr)
    r_C, _ = stats.pearsonr(C_shuf, y_arr)
    print(f"  trial {trial+1}: T mean {T_shuf.mean():.3f} (Δ={T_shuf.mean()-T_orig.mean():+.3f})  r(T,h)={r_T:+.3f} | "
          f"C mean {C_shuf.mean():.3f} (Δ={C_shuf.mean()-C_orig.mean():+.3f})  r(C,h)={r_C:+.3f}")
e3_rows.append({'perturbation': 'shuffle', 'param': 'random',
                 'metric': 'T',
                 'mean_val': float(T_shuf.mean()),
                 'r_with_hardness': r_T,
                 'delta_mean': float(T_shuf.mean() - T_orig.mean()),
                 'delta_r': r_T - r_T_orig})
e3_rows.append({'perturbation': 'shuffle', 'param': 'random',
                 'metric': 'C',
                 'mean_val': float(C_shuf.mean()),
                 'r_with_hardness': r_C,
                 'delta_mean': float(C_shuf.mean() - C_orig.mean()),
                 'delta_r': r_C - r_C_orig})

# 3d. Gaussian noise (should increase both T and C)
print("\n3d. Gaussian noise injection (should increase T and C):")
X_std = np.std(np.vstack([np.vstack(u for u in sys_frames_l9[si])
                            for si in range(len(systems))]))
print(f"  Embedding std ≈ {X_std:.4f}")
for sigma_scale in [0.1, 0.5, 1.0, 2.0]:
    sigma = sigma_scale * X_std
    rng2 = np.random.default_rng(42)
    T_noise = np.array([np.mean([vel_entropy(gaussian_noise(u, sigma, rng2))
                                   for u in sys_frames_l9[si]])
                         for si in range(len(systems))])
    C_noise = np.array([np.mean([-rog(gaussian_noise(u, sigma, rng2))
                                   for u in sys_frames_l12[si]])
                         for si in range(len(systems))])
    r_T, _ = stats.pearsonr(T_noise, y_arr)
    r_C, _ = stats.pearsonr(C_noise, y_arr)
    print(f"  σ={sigma_scale:.1f}σ_emb: T {T_noise.mean():.3f} (Δ={T_noise.mean()-T_orig.mean():+.3f}) r={r_T:+.3f} | "
          f"C {C_noise.mean():.3f} (Δ={C_noise.mean()-C_orig.mean():+.3f}) r={r_C:+.3f}")
    for metric, vals, ref_mean, ref_r in [
            ('T', T_noise, float(T_orig.mean()), r_T_orig),
            ('C', C_noise, float(C_orig.mean()), r_C_orig)]:
        e3_rows.append({'perturbation': 'gaussian_noise', 'param': f'sigma={sigma_scale}',
                         'metric': metric,
                         'mean_val': float(vals.mean()),
                         'r_with_hardness': r_T if metric=='T' else r_C,
                         'delta_mean': float(vals.mean()) - ref_mean,
                         'delta_r': (r_T if metric=='T' else r_C) - ref_r})

# Save E3 CSV
pd.DataFrame(e3_rows).to_csv(OUT / 'e3_intervention_lite.csv', index=False)

# E3 Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 3a: smoothing → T
ax = axes[0][0]
wins = [3, 5, 11, 21]
T_smoothed_means = []
T_smoothed_rs = []
for win in wins:
    rows_win = [r for r in e3_rows if r['perturbation']=='smoothing' and r['param']==f'win={win}' and r['metric']=='T']
    if rows_win:
        T_smoothed_means.append(rows_win[0]['mean_val'])
        T_smoothed_rs.append(rows_win[0]['r_with_hardness'])
ax.plot([0]+wins, [float(T_orig.mean())]+T_smoothed_means, 'o-', color='#1f77b4', label='T mean')
ax.set_xlabel('Smoothing window size'); ax.set_ylabel('Mean T (vel_entropy)')
ax.set_title('3a: Smoothing → T (expected: T decreases)', fontsize=10)
ax2 = ax.twinx()
ax2.plot([0]+wins, [r_T_orig]+T_smoothed_rs, 's--', color='#d62728', label='r(T, hardness)')
ax2.set_ylabel('r(T, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')

# Plot 3b: PCA → C
ax = axes[0][1]
ks = [32, 64, 128, 256, 512, 768]
C_pca_means = []; C_pca_rs = []
for k in ks:
    if k == 768:
        C_pca_means.append(float(C_orig.mean())); C_pca_rs.append(r_C_orig)
    else:
        rows_k = [r for r in e3_rows if r['perturbation']=='pca_compress' and r['param']==f'k={k}']
        if rows_k:
            C_pca_means.append(rows_k[0]['mean_val']); C_pca_rs.append(rows_k[0]['r_with_hardness'])
ax.plot(ks, C_pca_means, 'o-', color='#ff7f0e', label='C mean')
ax.set_xlabel('PCA components kept (of 768)'); ax.set_ylabel('Mean C (−rog)')
ax.set_title('3b: PCA compression → C (expected: C decreases)', fontsize=10)
ax2 = ax.twinx()
ax2.plot(ks, C_pca_rs, 's--', color='#d62728', label='r(C, hardness)')
ax2.set_ylabel('r(C, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')

# Plot 3c: shuffle → T
ax = axes[1][0]
shuf_rows_T = [r for r in e3_rows if r['perturbation']=='shuffle' and r['metric']=='T']
shuf_rows_C = [r for r in e3_rows if r['perturbation']=='shuffle' and r['metric']=='C']
ax.bar(['Original T', 'Shuffled T'], [float(T_orig.mean()), shuf_rows_T[0]['mean_val']],
       color=['#1f77b4', '#aec7e8'])
ax.set_ylabel('Mean T (vel_entropy)')
ax.set_title('3c: Frame shuffle → T\n(expected: T increases; r(T,h) should break)', fontsize=10)
ax2 = ax.twinx()
ax2.plot(['Original T', 'Shuffled T'],
         [r_T_orig, shuf_rows_T[0]['r_with_hardness']], 's-', color='#d62728')
ax2.set_ylabel('r(T, hardness)', color='#d62728')
ax2.tick_params(axis='y', colors='#d62728')

# Plot 3d: noise → T and C
ax = axes[1][1]
sigmas = [0.1, 0.5, 1.0, 2.0]
T_noise_rs = []; C_noise_rs = []
for s in sigmas:
    rows_T = [r for r in e3_rows if r['perturbation']=='gaussian_noise'
              and r['param']==f'sigma={s}' and r['metric']=='T']
    rows_C = [r for r in e3_rows if r['perturbation']=='gaussian_noise'
              and r['param']==f'sigma={s}' and r['metric']=='C']
    T_noise_rs.append(rows_T[0]['r_with_hardness'] if rows_T else np.nan)
    C_noise_rs.append(rows_C[0]['r_with_hardness'] if rows_C else np.nan)
ax.axhline(r_T_orig, color='#1f77b4', linestyle='--', alpha=0.6, label=f'r(T,h) original={r_T_orig:.3f}')
ax.axhline(r_C_orig, color='#ff7f0e', linestyle='--', alpha=0.6, label=f'r(C,h) original={r_C_orig:.3f}')
ax.plot(sigmas, T_noise_rs, 'o-', color='#1f77b4', label='r(T,h) noisy')
ax.plot(sigmas, C_noise_rs, 's-', color='#ff7f0e', label='r(C,h) noisy')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Noise amplitude (×embedding SD)'); ax.set_ylabel('r with residual hardness')
ax.set_title('3d: Gaussian noise → signal degradation\n(signal should degrade with noise)', fontsize=10)
ax.legend(fontsize=8)

plt.suptitle('E3: Intervention-lite — Perturbation Response', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'e3_intervention_lite.png', dpi=150)
plt.close()
print("\nSaved: e3_intervention_lite.png")

print("\n" + "="*70)
print("E1 / E2 / E3 COMPLETE — all CSVs and figures saved to outputs/")
print("="*70)
