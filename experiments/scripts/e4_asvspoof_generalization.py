#!/usr/bin/env python3
"""
Experiment 4: System-level Generalization to ASVspoof A01–A06

1. Sample 1000 utterances from ASVspoof 2019 LA train split, A01–A06 (≈167 per system).
2. Run robust_goat inference → per-system EER (6 systems).
3. Extract WavLM L9/L12 frame embeddings → per-system C (−rog@L12) and T (vel_entropy@L9).
4. Test: do C and T predict per-system EER ordering across A01–A06?
   (This tests generalization of the MLAAD-derived mechanism to ASVspoof data.)
"""

from __future__ import annotations
import sys
import json
import warnings
from argparse import Namespace
from collections import defaultdict
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
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parents[2]
EXP_DIR  = Path(__file__).resolve().parent
OUT      = BASE / 'outputs'
FIG      = OUT  / 'figures'
REPO     = BASE

for _p in [str(BASE), str(EXP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR    = 16_000
TARGET_LEN   = 3 * TARGET_SR   # 3 s
NF_PER_SAMPLE = TARGET_LEN // 320 - 1   # 149
BATCH_SIZE   = 8
N_PER_SYSTEM = 167   # ≈167 × 6 = 1002 total
CACHE_DIR    = BASE / 'data/asvspoof_2019_la'
MIN_SPOOF    = 30

ROBUST_CKPT = BASE / 'models/robust_goat.ckpt'

# ─── ASVspoof dataset ─────────────────────────────────────────────────────────
print("Loading ASVspoof dataset...", flush=True)
from datasets import load_dataset, Audio as HFAudio
ds_full = load_dataset('Bisher/as_vspoof_2019_la', cache_dir=str(CACHE_DIR), trust_remote_code=True)
ds = ds_full['train'].cast_column('audio', HFAudio(decode=False))

# Separate bonafide and per-system spoof
bonafide_idxs = []
sys_idxs: dict[str, list] = defaultdict(list)

for i, ex in enumerate(ds):
    sid = ex['system_id']
    if sid == '-':
        bonafide_idxs.append(i)
    elif sid in ('A01','A02','A03','A04','A05','A06'):
        sys_idxs[sid].append(i)

print(f"  Bonafide: {len(bonafide_idxs)}  |  Spoofed: "
      + "  ".join(f"{k}={len(v)}" for k,v in sorted(sys_idxs.items())))

rng = np.random.default_rng(42)
# Sample N_PER_SYSTEM utterances per attack system
sampled_sys = {sid: rng.choice(idxs, N_PER_SYSTEM, replace=False).tolist()
               for sid, idxs in sys_idxs.items()}
# Sample equal bonafide (N_PER_SYSTEM × 6)
n_bf = N_PER_SYSTEM * len(sampled_sys)
sampled_bf = rng.choice(bonafide_idxs, n_bf, replace=False).tolist()

print(f"  Sampled: {N_PER_SYSTEM} per system × {len(sampled_sys)} systems + {len(sampled_bf)} bonafide")
print(f"  Total:   {sum(len(v) for v in sampled_sys.values()) + len(sampled_bf)} utterances")

# ─── Audio decode ──────────────────────────────────────────────────────────────
import io as _io

def decode_entry(ex) -> torch.Tensor:
    audio_field = ex['audio']
    raw  = audio_field.get('bytes'); path = audio_field.get('path')
    try:
        if raw is not None:
            arr, sr = sf.read(_io.BytesIO(raw), dtype='float32', always_2d=False)
        else:
            arr, sr = sf.read(path, dtype='float32', always_2d=False)
    except Exception:
        return None
    wav = torch.tensor(arr)
    if wav.ndim > 1: wav = wav.mean(0)
    if sr != TARGET_SR:
        wav = TAT.Resample(sr, TARGET_SR)(wav.unsqueeze(0)).squeeze(0)
    if len(wav) < TARGET_LEN:
        wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN]  # (T,)

# ─── WavLM frame extraction ────────────────────────────────────────────────────
print("\nLoading WavLM...", flush=True)
from transformers import WavLMForCTC
wavlm_full = WavLMForCTC.from_pretrained('microsoft/wavlm-base')
wavlm = wavlm_full.wavlm.eval().to(DEVICE)

def extract_wavlm(wav_1d: torch.Tensor, layers=(9, 12)):
    x = wav_1d.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        hs = wavlm(x, output_hidden_states=True).hidden_states
    return {li: hs[li][0].cpu().float().numpy() for li in layers}

# Trajectory metrics
def rog(frames):
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c)**2, 1))))

def vel_entropy(frames):
    vels = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    n = max(5, min(20, len(vels)//4))
    h, _ = np.histogram(vels, bins=n)
    h = h.astype(float) + 1e-8; h /= h.sum()
    return float(-np.sum(h * np.log(h)))

# ─── Collect WavLM features per system ────────────────────────────────────────
print("\nExtracting WavLM C and T for ASVspoof utterances...", flush=True)

# sys_wavlm[sid] = {'rog_l12': [], 'vel_l9': []}
sys_wavlm = defaultdict(lambda: {'rog_l12': [], 'vel_l9': []})
bf_wavlm  = {'rog_l12': [], 'vel_l9': []}

# Process spoof systems
for sid in sorted(sampled_sys.keys()):
    idxs = sampled_sys[sid]
    for i, idx in enumerate(idxs):
        ex = ds[idx]
        wav = decode_entry(ex)
        if wav is None: continue
        layers = extract_wavlm(wav)
        sys_wavlm[sid]['rog_l12'].append(rog(layers[12]))
        sys_wavlm[sid]['vel_l9'].append(vel_entropy(layers[9]))
        if (i+1) % 50 == 0:
            print(f"  {sid}: {i+1}/{len(idxs)}", flush=True)
    print(f"  {sid} done: n={len(sys_wavlm[sid]['rog_l12'])}")

# Process bonafide
print("  Processing bonafide...")
for i, idx in enumerate(sampled_bf):
    ex = ds[idx]
    wav = decode_entry(ex)
    if wav is None: continue
    layers = extract_wavlm(wav)
    bf_wavlm['rog_l12'].append(rog(layers[12]))
    bf_wavlm['vel_l9'].append(vel_entropy(layers[9]))

print(f"  Bonafide done: n={len(bf_wavlm['rog_l12'])}")
del wavlm, wavlm_full; torch.cuda.empty_cache()

# ─── Load robust_goat and run inference ───────────────────────────────────────
print("\nLoading robust_goat for inference...", flush=True)

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name    = network_name
        network_param.pretrained_name = (
            "microsoft/wavlm-base" if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None,
                total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

patch_phoneme_loader()

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

def _detect_n_edges(ckpt_path):
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    hp   = ckpt.get("hyper_parameters", {})
    cfg  = hp.get("cfg", None)
    n    = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10

def load_model(ckpt_path):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    n_edges = _detect_n_edges(ckpt_path)
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=n_edges, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=DEVICE, strict=True)
    lit.to(DEVICE); lit.eval(); lit.freeze()
    return lit

lit = load_model(ROBUST_CKPT)
gm  = lit.model

def run_frontend(wav_1d: torch.Tensor):
    x = wav_1d.unsqueeze(0).to(DEVICE)
    num_f = torch.full((1,), NF_PER_SAMPLE, device=DEVICE)
    with torch.no_grad():
        feat1 = gm.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hs, _ = gm.transformer_in_phoneme_model.feature_projection(feat1)
        pf    = gm.transformer_in_phoneme_model.encoder(hs)[0]
        pl    = gm.phoneme_model.model.model.lm_head(pf)
        pids  = torch.argmax(pl, dim=-1)
        result = gm.encoder_and_GAT(hs, num_f, pids)
        logit  = float(result[5].cpu().item())
    return logit

# ─── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning robust_goat inference...", flush=True)

# sys_scores[sid] = {labels, logits}
sys_scores: dict[str, dict] = defaultdict(lambda: {'labels': [], 'logits': []})
bf_logits = []

for sid in sorted(sampled_sys.keys()):
    idxs = sampled_sys[sid]
    for i, idx in enumerate(idxs):
        ex = ds[idx]
        wav = decode_entry(ex)
        if wav is None: continue
        logit = run_frontend(wav)
        sys_scores[sid]['labels'].append(1)   # spoof = 1
        sys_scores[sid]['logits'].append(logit)
        if (i+1) % 50 == 0:
            print(f"  {sid}: {i+1}/{len(idxs)}", flush=True)
    print(f"  {sid} inference done: n={len(sys_scores[sid]['labels'])}")

print("  Processing bonafide inference...")
for i, idx in enumerate(sampled_bf):
    ex = ds[idx]
    wav = decode_entry(ex)
    if wav is None: continue
    logit = run_frontend(wav)
    bf_logits.append(logit)

print(f"  Bonafide inference done: n={len(bf_logits)}")

# ─── Compute per-system EER ────────────────────────────────────────────────────
from callbacks import compute_eer as _compute_eer

bf_labels = [0] * len(bf_logits)
sys_eer = {}
sys_n   = {}

for sid in sorted(sys_scores.keys()):
    sp_lab = sys_scores[sid]['labels']
    sp_log = sys_scores[sid]['logits']
    all_lab = sp_lab + bf_labels
    all_log = sp_log + bf_logits
    try:
        eer = float(_compute_eer(np.array(all_lab), np.array(all_log), positive_label=1))
    except Exception as e:
        print(f"  EER failed for {sid}: {e}")
        eer = np.nan
    sys_eer[sid] = eer
    sys_n[sid]   = len(sp_lab)

print("\nPer-system EER (robust_goat on ASVspoof A01–A06):")
for sid in sorted(sys_eer):
    print(f"  {sid}: EER={sys_eer[sid]:.4f}  (N spoof={sys_n[sid]})")

# ─── Aggregate per-system C and T ─────────────────────────────────────────────
systems_asv = sorted(sys_wavlm.keys())
asv_C = np.array([-np.mean(sys_wavlm[s]['rog_l12']) for s in systems_asv])  # negate: compact=harder
asv_T = np.array([np.mean(sys_wavlm[s]['vel_l9'])   for s in systems_asv])
asv_EER = np.array([sys_eer[s] for s in systems_asv])

print("\nPer-system C, T, and EER:")
print(f"  {'System':<8} {'C (−rog@L12)':>14} {'T (vel@L9)':>12} {'EER':>8}")
for i, sid in enumerate(systems_asv):
    print(f"  {sid:<8} {asv_C[i]:>14.4f} {asv_T[i]:>12.4f} {asv_EER[i]:>8.4f}")

# Correlations
if len(systems_asv) >= 4:
    r_C_eer, p_C_eer = stats.pearsonr(asv_C, asv_EER)
    r_T_eer, p_T_eer = stats.pearsonr(asv_T, asv_EER)
    rho_C_eer, _ = stats.spearmanr(asv_C, asv_EER)
    rho_T_eer, _ = stats.spearmanr(asv_T, asv_EER)
    print(f"\nCorrelations with EER (N={len(systems_asv)} systems):")
    print(f"  C: Pearson r={r_C_eer:+.3f} (p={p_C_eer:.3f})  Spearman ρ={rho_C_eer:+.3f}")
    print(f"  T: Pearson r={r_T_eer:+.3f} (p={p_T_eer:.3f})  Spearman ρ={rho_T_eer:+.3f}")
else:
    r_C_eer = r_T_eer = rho_C_eer = rho_T_eer = np.nan
    p_C_eer = p_T_eer = np.nan

# Compare MLAAD directions with ASVspoof directions
print("\nDirection check (MLAAD-trained: higher C → harder, higher T → harder):")
print(f"  C direction in ASVspoof: r={r_C_eer:+.3f} — {'CONSISTENT' if r_C_eer > 0 else 'REVERSED'}")
print(f"  T direction in ASVspoof: r={r_T_eer:+.3f} — {'CONSISTENT' if r_T_eer > 0 else 'REVERSED'}")

# ─── Rank comparison ──────────────────────────────────────────────────────────
rank_eer = stats.rankdata(-asv_EER)   # rank 1 = hardest (lowest EER)
rank_C   = stats.rankdata(-asv_C)     # rank 1 = most compact
rank_T   = stats.rankdata(-asv_T)     # rank 1 = most irregular

print("\nRank ordering (by decreasing EER difficulty = harder first):")
sorted_idx = np.argsort(-asv_EER)  # sort by descending EER (harder = lower EER, so ascending EER)
sorted_idx = np.argsort(asv_EER)   # ascending EER = hardest first
for i in sorted_idx:
    print(f"  {systems_asv[i]}: EER={asv_EER[i]:.4f}  C_rank={rank_C[i]:.0f}  T_rank={rank_T[i]:.0f}")

# ─── Save results ─────────────────────────────────────────────────────────────
asv_df = pd.DataFrame({
    'system': systems_asv,
    'C_neg_rog_L12': asv_C,
    'T_vel_entropy_L9': asv_T,
    'EER_robust_goat': asv_EER,
    'n_spoof': [sys_n.get(s, 0) for s in systems_asv],
    'rog_L12_mean': [-asv_C[i] for i in range(len(systems_asv))],  # original rog (positive)
    'vel_L9_mean': asv_T.tolist(),
    'n_wavlm_utts': [len(sys_wavlm[s]['rog_l12']) for s in systems_asv],
})
asv_df.to_csv(OUT / 'e4_asvspoof_per_system.csv', index=False)
print(f"\nSaved: e4_asvspoof_per_system.csv")

# ─── Figures ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(systems_asv)))
sys_colors = {sid: colors[i] for i, sid in enumerate(systems_asv)}

# Plot 1: C vs EER
ax = axes[0]
for i, sid in enumerate(systems_asv):
    ax.scatter(asv_C[i], asv_EER[i], color=sys_colors[sid], s=120, zorder=5)
    ax.annotate(sid, (asv_C[i], asv_EER[i]), textcoords='offset points',
                xytext=(5, 5), fontsize=10, fontweight='bold')
if not np.isnan(r_C_eer):
    m, b = np.polyfit(asv_C, asv_EER, 1)
    x_line = np.linspace(asv_C.min(), asv_C.max(), 50)
    ax.plot(x_line, m*x_line+b, '--', color='grey', alpha=0.7)
ax.set_xlabel('C = −rog@L12 (compact = harder in MLAAD)')
ax.set_ylabel('System EER (robust_goat on ASVspoof)')
ax.set_title(f'C vs EER\nr={r_C_eer:.3f}  ρ={rho_C_eer:.3f}', fontsize=11)

# Plot 2: T vs EER
ax = axes[1]
for i, sid in enumerate(systems_asv):
    ax.scatter(asv_T[i], asv_EER[i], color=sys_colors[sid], s=120, zorder=5)
    ax.annotate(sid, (asv_T[i], asv_EER[i]), textcoords='offset points',
                xytext=(5, 5), fontsize=10, fontweight='bold')
if not np.isnan(r_T_eer):
    m, b = np.polyfit(asv_T, asv_EER, 1)
    x_line = np.linspace(asv_T.min(), asv_T.max(), 50)
    ax.plot(x_line, m*x_line+b, '--', color='grey', alpha=0.7)
ax.set_xlabel('T = vel_entropy@L9 (irregular = harder in MLAAD)')
ax.set_ylabel('System EER (robust_goat on ASVspoof)')
ax.set_title(f'T vs EER\nr={r_T_eer:.3f}  ρ={rho_T_eer:.3f}', fontsize=11)

# Plot 3: C-T 2D manifold, colored by EER
ax = axes[2]
sc = ax.scatter(asv_C, asv_T, c=asv_EER, cmap='RdYlGn_r',
                s=200, zorder=5, edgecolors='black', linewidths=0.7)
plt.colorbar(sc, ax=ax, label='EER (robust_goat)', shrink=0.8)
for i, sid in enumerate(systems_asv):
    ax.annotate(sid, (asv_C[i], asv_T[i]), textcoords='offset points',
                xytext=(6, 4), fontsize=10, fontweight='bold')
ax.set_xlabel('C = −rog@L12'); ax.set_ylabel('T = vel_entropy@L9')
ax.set_title('C–T manifold (ASVspoof A01–A06)\ncolored by EER', fontsize=11)

plt.suptitle('E4: ASVspoof A01–A06 Generalization Test', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'e4_asvspoof_generalization.png', dpi=150)
plt.close()
print("Saved: e4_asvspoof_generalization.png")

# ─── Comparison with MLAAD ────────────────────────────────────────────────────
print("\n" + "="*70)
print("MLAAD vs ASVspoof comparison:")
print("  MLAAD: r(C, hardness)=+0.329  r(T, hardness)=+0.276  (N=63 systems)")
print(f"  ASVspoof: r(C, EER)={r_C_eer:.3f}  r(T, EER)={r_T_eer:.3f}  (N={len(systems_asv)} systems)")
print("  Note: EER ↑ means harder to detect in ASVspoof convention.")
print("="*70)
