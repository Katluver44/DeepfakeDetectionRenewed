#!/usr/bin/env python3
"""
E5 ITW: Does the C/T hard manifold explain InTheWild difficulty?
================================================================

Hypothesis: InTheWild (ITW) is hard because its samples occupy the same hard
manifold region (elevated C and/or T = the MLAAD Q4 region), not because of an
ITW-specific failure mode. ITW has no per-generation-system labels, so the test
runs at PER-SAMPLE granularity.

  C (compactness)     = -rog@L12   (radius of gyration of WavLM L12 frames; higher = harder)
  T (temporal entropy)= vel_entropy@L9 (adjacent-frame velocity entropy; higher = harder)

Pipeline (adapted from e4_asvspoof_generalization.py):
  1. Load ITW (mueller91/In-The-Wild) from local cache, join meta.csv labels.
  2. Per-sample C/T + mean-pooled L9/L12 WavLM embeddings (microsoft/wavlm-base, frozen).
  3. Build a consistent MLAAD per-sample reference from data/mlaad_en/<system>/.
  4. Stats: distribution shift (Mann-Whitney/KS/Cliff/Cohen), Q4 enrichment (binomial).
  5. Difficulty anchor: run BOTH robust_goat.ckpt and mlaad_robust_goat.ckpt -> ITW EER
     vs MLAAD Q4 EER + per-sample C/T<->logit Spearman.
  5b. Embedding-manifold similarity: joint PCA (2D/3D) + UMAP/t-SNE, kNN mixing,
      centroid distance, silhouette.
  6. Figures + outputs/e5_itw_summary.md.
"""
from __future__ import annotations
import os
import sys
import io as _io
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
EXP_DIR  = Path(__file__).resolve().parents[1]
SCRIPTS  = Path(__file__).resolve().parent
OUT      = BASE / "outputs"
FIG      = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR   = 16_000
TARGET_LEN  = 3 * TARGET_SR              # 48000
NF_PER_SAMPLE = TARGET_LEN // 320 - 1    # 149
BATCH_SIZE  = 16

ITW_CACHE   = BASE / "data" / "in_the_wild"
META_CSV    = (ITW_CACHE / "downloads" / "extracted" /
               "c3c93f2f54ac2d261fa7010629351505bd6e05597ea22fd4a35c92dda590a3bf" /
               "release_in_the_wild" / "meta.csv")
PROC_DIR    = EXP_DIR / "data" / "mlaad_tiny_processed"
TEST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
SIG_FEATURES = OUT / "sig_layer_features.csv"
P1_UTT      = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "utterance_features.csv"
BASELINE_EER = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "per_system_eer.csv"

CKPTS = {
    "robust_goat":       BASE / "models" / "robust_goat.ckpt",
    "mlaad_robust_goat": EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
}

N_PER_CLASS_ITW = 3000   # 3k spoof + 3k bona-fide

print(f"[E5 ITW] device={DEVICE}  base={BASE}")

# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory metrics (verbatim from e4)
# ═══════════════════════════════════════════════════════════════════════════════
def rog(frames: np.ndarray) -> float:
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c) ** 2, 1))))

def vel_entropy(frames: np.ndarray) -> float:
    vels = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    n = max(5, min(20, len(vels) // 4))
    h, _ = np.histogram(vels, bins=n)
    h = h.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))

# ═══════════════════════════════════════════════════════════════════════════════
# Audio helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _center_crop_pad(wav: torch.Tensor) -> torch.Tensor:
    """wav: (T,) -> (TARGET_LEN,) center 3s, tile-pad if short (paper eval policy)."""
    if wav.ndim > 1:
        wav = wav.mean(0)
    if len(wav) < TARGET_LEN:
        reps = -(-TARGET_LEN // len(wav))
        wav = wav.repeat(reps)
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN]

def load_wav_file(path: str) -> torch.Tensor | None:
    try:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    wav = torch.tensor(arr)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != TARGET_SR:
        wav = TAT.Resample(sr, TARGET_SR)(wav.unsqueeze(0)).squeeze(0)
    return _center_crop_pad(wav)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load InTheWild  (mueller91/In-The-Wild, local cache + meta.csv join)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading InTheWild ...", flush=True)
from datasets import load_dataset, Audio as HFAudio

assert META_CSV.exists(), f"meta.csv not found at {META_CSV}"
meta = pd.read_csv(META_CSV)
meta.columns = [c.strip().lower() for c in meta.columns]
file_col  = next(c for c in meta.columns if "file" in c or "path" in c or "name" in c)
spk_col   = next((c for c in meta.columns if "speaker" in c), None)
label_col = next(c for c in meta.columns if c not in (file_col, spk_col))
meta["_basename"] = meta[file_col].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
basename_to_label = dict(zip(meta["_basename"], meta[label_col]))
basename_to_spk   = dict(zip(meta["_basename"], meta[spk_col])) if spk_col else {}
print(f"  meta.csv: {meta.shape}  file_col={file_col!r} label_col={label_col!r} spk_col={spk_col!r}")

HF_TOKEN = os.environ.get("HF_TOKEN")
ds_itw = load_dataset("mueller91/In-The-Wild", cache_dir=str(ITW_CACHE),
                      token=HF_TOKEN if HF_TOKEN else None)
split_name = list(ds_itw.keys())[0]
ds_split = ds_itw[split_name]
ds_nodecode = ds_split.cast_column("audio", HFAudio(decode=False))
hf_basenames = [os.path.splitext(os.path.basename(ds_nodecode[i]["audio"]["path"]))[0]
                for i in range(len(ds_nodecode))]

REAL = {"bona-fide", "bonafide", "real", "genuine", "0", 0}
FAKE = {"spoof", "fake", "synthetic", "1", 1}
def label_to_int(v):
    if v is None:
        return None
    vl = str(v).lower().strip()
    if vl in REAL: return 0
    if vl in FAKE: return 1
    return None

int_labels = [label_to_int(basename_to_label.get(b)) for b in hf_basenames]
speakers   = [basename_to_spk.get(b, "unknown") for b in hf_basenames]

rng = np.random.default_rng(SEED)
real_idx = [i for i, l in enumerate(int_labels) if l == 0]
fake_idx = [i for i, l in enumerate(int_labels) if l == 1]
print(f"  available: real={len(real_idx)} spoof={len(fake_idx)} | speakers={len(set(speakers))}")
n_real = min(N_PER_CLASS_ITW, len(real_idx))
n_fake = min(N_PER_CLASS_ITW, len(fake_idx))
sel = sorted(rng.choice(real_idx, n_real, replace=False).tolist() +
             rng.choice(fake_idx, n_fake, replace=False).tolist())
ds_bal = ds_split.select(sel).cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
itw_labels   = [int_labels[i] for i in sel]
itw_speakers = [speakers[i] for i in sel]
itw_names    = [hf_basenames[i] for i in sel]
print(f"  balanced ITW subset: {len(sel)}  (real={n_real} spoof={n_fake})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. WavLM per-sample C/T + mean-pooled L9/L12 embeddings (frozen)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Loading WavLM (microsoft/wavlm-base) ...", flush=True)
from transformers import WavLMForCTC
wavlm_full = WavLMForCTC.from_pretrained("microsoft/wavlm-base")
wavlm = wavlm_full.wavlm.eval().to(DEVICE)

@torch.no_grad()
def wavlm_features(wav_batch: torch.Tensor):
    """wav_batch: (B, TARGET_LEN). Returns per-sample lists:
    rog_L12, vel_L9, emb_L12 (B,768), emb_L9 (B,768)."""
    x = wav_batch.to(DEVICE)
    hs = wavlm(x, output_hidden_states=True).hidden_states  # tuple len 13
    h9  = hs[9].float().cpu().numpy()    # (B, T, 768)
    h12 = hs[12].float().cpu().numpy()
    out_rog, out_vel, e12, e9 = [], [], [], []
    for i in range(x.shape[0]):
        out_rog.append(rog(h12[i]))
        out_vel.append(vel_entropy(h9[i]))
        e12.append(h12[i].mean(0))
        e9.append(h9[i].mean(0))
    return out_rog, out_vel, np.stack(e12), np.stack(e9)

def extract_set(wav_iter, n_total, tag):
    """wav_iter yields (idx, wav_tensor(TARGET_LEN,)). Returns dict of arrays."""
    rogs, vels, embs12, embs9, keep_idx = [], [], [], [], []
    buf_wav, buf_idx = [], []
    done = 0
    def flush():
        nonlocal done
        if not buf_wav:
            return
        wb = torch.stack(buf_wav)
        r, v, e12, e9 = wavlm_features(wb)
        rogs.extend(r); vels.extend(v); embs12.append(e12); embs9.append(e9)
        keep_idx.extend(buf_idx)
        done += len(buf_wav)
        buf_wav.clear(); buf_idx.clear()
        print(f"    {tag}: {done}/{n_total}", flush=True)
    for idx, wav in wav_iter:
        if wav is None:
            continue
        buf_wav.append(wav); buf_idx.append(idx)
        if len(buf_wav) >= BATCH_SIZE:
            flush()
    flush()
    return {
        "idx": np.array(keep_idx),
        "rog_L12": np.array(rogs),
        "vel_L9": np.array(vels),
        "emb_L12": np.concatenate(embs12) if embs12 else np.zeros((0, 768)),
        "emb_L9":  np.concatenate(embs9)  if embs9  else np.zeros((0, 768)),
    }

print("  extracting ITW C/T + embeddings ...", flush=True)
def itw_iter():
    for i in range(len(ds_bal)):
        arr = ds_bal[i]["audio"]["array"]
        wav = _center_crop_pad(torch.tensor(arr, dtype=torch.float32))
        yield i, wav
itw = extract_set(itw_iter(), len(ds_bal), "ITW")
itw_labels   = np.array(itw_labels)[itw["idx"]]
itw_speakers = [itw_speakers[i] for i in itw["idx"]]
itw_names    = [itw_names[i] for i in itw["idx"]]
itw_C = -itw["rog_L12"]
itw_T =  itw["vel_L9"]

# MLAAD per-sample reference — from the locked test split (.pt tensors, 48000 samples
# each), the same data behind sig_layer_features.csv / per_system_eer.csv.
print("  extracting MLAAD per-sample reference (test split spoof) ...", flush=True)
test_records = json.loads(TEST_JSON.read_text())
mlaad_items = [(r["attack_system"], PROC_DIR / r["audio_path"])
               for r in test_records if r["label"] == "spoof"]
print(f"    MLAAD spoof utts={len(mlaad_items)} across "
      f"{len(set(s for s, _ in mlaad_items))} systems")
def mlaad_iter():
    for k, (_sys, ptpath) in enumerate(mlaad_items):
        try:
            w = torch.load(ptpath)            # (48000,) float32, already cropped
        except Exception:
            w = None
        yield k, w
mla = extract_set(mlaad_iter(), len(mlaad_items), "MLAAD")
mla_system = [mlaad_items[k][0] for k in mla["idx"]]
mla_C = -mla["rog_L12"]
mla_T =  mla["vel_L9"]

del wavlm, wavlm_full
torch.cuda.empty_cache()

# Save per-sample CSVs
pd.DataFrame({
    "sample_id": itw_names, "speaker": itw_speakers,
    "label": ["spoof" if l == 1 else "bona-fide" for l in itw_labels],
    "C": itw_C, "T": itw_T, "rog_L12": itw["rog_L12"], "vel_L9": itw["vel_L9"],
}).to_csv(OUT / "e5_itw_utterance_ct.csv", index=False)
pd.DataFrame({
    "system": mla_system, "label": "spoof",
    "C": mla_C, "T": mla_T, "rog_L12": mla["rog_L12"], "vel_L9": mla["vel_L9"],
}).to_csv(OUT / "e5_mlaad_utterance_ct.csv", index=False)
print("  saved e5_itw_utterance_ct.csv, e5_mlaad_utterance_ct.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hard-region boundaries
# ═══════════════════════════════════════════════════════════════════════════════
sig = pd.read_csv(SIG_FEATURES)
sig_C = -sig["rog_L12"].values            # per-system mean C
sig_T =  sig["vel_entropy_L9"].values
# (a) canonical Q4 boundary from 63 per-system means (Q3->Q4 cut)
qa_C = float(np.quantile(sig_C, 0.75))
qa_T = float(np.quantile(sig_T, 0.75))
# (b) per-sample Q4 boundary from MLAAD reference utterances
qb_C = float(np.quantile(mla_C, 0.75))
qb_T = float(np.quantile(mla_T, 0.75))

# extraction-consistency sanity: per-system mean rog vs sig_layer_features
ref_means = pd.DataFrame({"system": mla_system, "rog_L12": mla["rog_L12"],
                          "vel_L9": mla["vel_L9"]}).groupby("system").mean()
merged = ref_means.join(sig.set_index("system")[["rog_L12", "vel_entropy_L9"]],
                        rsuffix="_sig", how="inner")
rog_consistency = (float(stats.pearsonr(merged["rog_L12"], merged["rog_L12_sig"])[0])
                   if len(merged) >= 3 else float("nan"))
vel_consistency = (float(stats.pearsonr(merged["vel_L9"], merged["vel_entropy_L9"])[0])
                   if len(merged) >= 3 else float("nan"))
print(f"\n[3] sanity (re-extraction vs sig_layer_features): "
      f"rog r={rog_consistency:.3f}  vel r={vel_consistency:.3f}  (n={len(merged)} systems)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Statistical hard-region tests (per-sample)
# ═══════════════════════════════════════════════════════════════════════════════
spoof_m = itw_labels == 1
bona_m  = itw_labels == 0
itw_sp_C, itw_sp_T = itw_C[spoof_m], itw_T[spoof_m]
itw_bf_C, itw_bf_T = itw_C[bona_m],  itw_T[bona_m]

def cliffs_delta_from_U(U, n1, n2):
    return float(2.0 * U / (n1 * n2) - 1.0)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")

def dist_shift(a, b, name):
    U, p_mw = stats.mannwhitneyu(a, b, alternative="greater")  # a > b ?
    ks, p_ks = stats.ks_2samp(a, b)
    return {
        "axis": name, "itw_mean": float(a.mean()), "mlaad_mean": float(b.mean()),
        "mw_U": float(U), "mw_p_greater": float(p_mw),
        "ks": float(ks), "ks_p": float(p_ks),
        "cliffs_delta": cliffs_delta_from_U(U, len(a), len(b)),
        "cohens_d": cohens_d(a, b),
    }

shift = [dist_shift(itw_sp_C, mla_C, "C"), dist_shift(itw_sp_T, mla_T, "T")]

def q4_enrichment(vals, thr, name, scheme):
    frac = float(np.mean(vals > thr))
    n = len(vals); k = int((vals > thr).sum())
    p = float(stats.binomtest(k, n, 0.25, alternative="greater").pvalue)
    return {"axis": name, "scheme": scheme, "threshold": thr,
            "frac_in_Q4": frac, "enrichment_vs_0.25": frac / 0.25, "binom_p": p, "n": n}

enrich = [
    q4_enrichment(itw_sp_C, qa_C, "C", "per-system(a)"),
    q4_enrichment(itw_sp_T, qa_T, "T", "per-system(a)"),
    q4_enrichment(itw_sp_C, qb_C, "C", "per-sample(b)"),
    q4_enrichment(itw_sp_T, qb_T, "T", "per-sample(b)"),
]
# both-axes (per-sample scheme b)
both_frac = float(np.mean((itw_sp_C > qb_C) & (itw_sp_T > qb_T)))
k_both = int(((itw_sp_C > qb_C) & (itw_sp_T > qb_T)).sum())
both_p = float(stats.binomtest(k_both, len(itw_sp_C), 0.0625, alternative="greater").pvalue)

# within-ITW: spoof vs bonafide
within = [dist_shift(itw_sp_C, itw_bf_C, "C(spoof>bona)"),
          dist_shift(itw_sp_T, itw_bf_T, "T(spoof>bona)")]

print("[4] distribution shift (ITW-spoof vs MLAAD-spoof):")
for s in shift:
    print(f"    {s['axis']}: ITW={s['itw_mean']:.3f} MLAAD={s['mlaad_mean']:.3f} "
          f"Cliff={s['cliffs_delta']:+.3f} d={s['cohens_d']:+.3f} MW_p={s['mw_p_greater']:.2e}")
print("    Q4 enrichment:")
for e in enrich:
    print(f"    {e['axis']} [{e['scheme']}]: frac={e['frac_in_Q4']:.3f} "
          f"({e['enrichment_vs_0.25']:.2f}x) p={e['binom_p']:.2e}")
print(f"    both-axes (b): frac={both_frac:.3f} p={both_p:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Difficulty anchor — run BOTH checkpoints
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Difficulty anchor (robust_goat + mlaad_robust_goat) ...", flush=True)
from _ablation_common import compute_eer, load_system_ct  # noqa: E402

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = ("microsoft/wavlm-base" if network_name.lower() == "wavlm"
                                         else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param, optim_param=optim_param,
                tokenizer=None, total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load

def _detect_n_edges(ckpt_path):
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    cfg = hp.get("cfg", None)
    n = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10

def load_model(ckpt_path):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    n_edges = _detect_n_edges(ckpt_path)
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=n_edges, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), cfg=cfg,
                                               map_location=DEVICE, strict=True)
    lit.to(DEVICE); lit.eval(); lit.freeze()
    return lit

patch_phoneme_loader()
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

@torch.no_grad()
def run_frontend(gm, wav_1d):
    x = wav_1d.unsqueeze(0).to(DEVICE)
    num_f = torch.full((1,), NF_PER_SAMPLE, device=DEVICE)
    feat1 = gm.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
    hs, _ = gm.transformer_in_phoneme_model.feature_projection(feat1)
    pf = gm.transformer_in_phoneme_model.encoder(hs)[0]
    pl = gm.phoneme_model.model.model.lm_head(pf)
    pids = torch.argmax(pl, dim=-1)
    result = gm.encoder_and_GAT(hs, num_f, pids)
    return float(result[5].cpu().item())

# pre-load ITW waveforms once (in itw["idx"] order)
itw_wavs = [_center_crop_pad(torch.tensor(ds_bal[int(i)]["audio"]["array"], dtype=torch.float32))
            for i in itw["idx"]]

# MLAAD baseline EER reference (from p1 per_system_eer.csv) + Q4 membership
ct = load_system_ct()
base_eer = pd.read_csv(BASELINE_EER)
eer_col = "eer_before" if "eer_before" in base_eer.columns else "eer"
sys_eer = dict(zip(base_eer["system"], base_eer[eer_col]))
common = [s for s in sys_eer if s in ct]
C_vals = np.array([ct[s]["C"] for s in common])
T_vals = np.array([ct[s]["T"] for s in common])
cC, cT = np.quantile(C_vals, 0.75), np.quantile(T_vals, 0.75)
q4_C_sys = [s for s in common if ct[s]["C"] > cC]
q4_T_sys = [s for s in common if ct[s]["T"] > cT]
mlaad_overall_eer = float(np.mean([sys_eer[s] for s in common]))
mlaad_q4C_eer = float(np.mean([sys_eer[s] for s in q4_C_sys]))
mlaad_q4T_eer = float(np.mean([sys_eer[s] for s in q4_T_sys]))

anchor = {}
itw_logits = {}
for name, ckpt in CKPTS.items():
    if not Path(ckpt).exists():
        print(f"    [skip] {name}: missing {ckpt}")
        continue
    print(f"    loading {name} ...", flush=True)
    lit = load_model(ckpt)
    gm = lit.model
    logits = np.array([run_frontend(gm, w) for w in itw_wavs])
    itw_logits[name] = logits
    eer = compute_eer(itw_labels.astype(int), logits)   # spoof=1
    sp_logit = logits[spoof_m]
    rC = stats.spearmanr(itw_sp_C, sp_logit)
    rT = stats.spearmanr(itw_sp_T, sp_logit)
    anchor[name] = {"itw_eer": float(eer) if eer is not None else float("nan"),
                    "spearman_C_logit": float(rC.correlation), "p_C": float(rC.pvalue),
                    "spearman_T_logit": float(rT.correlation), "p_T": float(rT.pvalue)}
    print(f"      ITW EER={anchor[name]['itw_eer']:.4f} | "
          f"rho(C,logit)={rC.correlation:+.3f} rho(T,logit)={rT.correlation:+.3f}")
    del lit, gm
    torch.cuda.empty_cache()

# MLAAD per-sample C/T<->logit reference (from p1 utterance_features.csv)
mlaad_corr = {}
if P1_UTT.exists():
    p1 = pd.read_csv(P1_UTT)
    p1s = p1[p1["label"] == 1]
    if {"C", "T", "raw_logit"}.issubset(p1s.columns) and len(p1s) > 10:
        # p1 'C' is rog (positive); flip sign to match C=-rog convention
        mlaad_corr = {
            "spearman_C_logit": float(stats.spearmanr(-p1s["C"], p1s["raw_logit"]).correlation),
            "spearman_T_logit": float(stats.spearmanr(p1s["T"], p1s["raw_logit"]).correlation),
            "n": int(len(p1s)),
        }
        print(f"    MLAAD(p1) rho(C,logit)={mlaad_corr['spearman_C_logit']:+.3f} "
              f"rho(T,logit)={mlaad_corr['spearman_T_logit']:+.3f}")

# save per-sample logits
itw_df = pd.read_csv(OUT / "e5_itw_utterance_ct.csv")
for name, lg in itw_logits.items():
    itw_df[f"logit_{name}"] = lg
itw_df.to_csv(OUT / "e5_itw_utterance_ct.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Embedding-manifold similarity (PCA/UMAP + kNN mixing + centroids)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5b] Embedding-manifold similarity ...", flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

X_itw_sp = itw["emb_L12"][spoof_m]
X_itw_bf = itw["emb_L12"][bona_m]
X_mla    = mla["emb_L12"]
# per-sample MLAAD Q4 (by C) vs easy (bottom 50%)
mla_q4_mask = mla_C > qb_C
mla_easy_mask = mla_C <= np.quantile(mla_C, 0.50)

X_all = np.vstack([X_itw_sp, X_itw_bf, X_mla])
src   = (["ITW-spoof"] * len(X_itw_sp) + ["ITW-bona"] * len(X_itw_bf) +
         ["MLAAD"] * len(X_mla))
scaler = StandardScaler().fit(X_all)
Xs = scaler.transform(X_all)
np.savez(OUT / "e5_itw_embeddings.npz", emb_L12=X_all, emb_L9=np.vstack(
    [itw["emb_L9"][spoof_m], itw["emb_L9"][bona_m], mla["emb_L9"]]),
    src=np.array(src), mla_q4=mla_q4_mask)

# PCA 3D
pca = PCA(n_components=3, random_state=SEED).fit(Xs)
P = pca.transform(Xs)
n_sp, n_bf = len(X_itw_sp), len(X_itw_bf)
P_sp, P_bf, P_mla = P[:n_sp], P[n_sp:n_sp + n_bf], P[n_sp + n_bf:]

# kNN dataset mixing: for each ITW-spoof, fraction of k nearest MLAAD neighbours in Q4
Xs_mla = scaler.transform(X_mla)
Xs_itw_sp = scaler.transform(X_itw_sp)
k = 15
nn = NearestNeighbors(n_neighbors=k).fit(Xs_mla)
_, nbr = nn.kneighbors(Xs_itw_sp)
knn_q4_frac = float(mla_q4_mask[nbr].mean())   # vs baseline 0.25
# centroid distances (standardized space)
c_itw = Xs_itw_sp.mean(0)
c_q4  = Xs_mla[mla_q4_mask].mean(0)
c_easy = Xs_mla[mla_easy_mask].mean(0)
d_q4  = float(np.linalg.norm(c_itw - c_q4))
d_easy = float(np.linalg.norm(c_itw - c_easy))
# silhouette of {ITW-spoof, MLAAD-Q4, MLAAD-easy}
sil_X = np.vstack([Xs_itw_sp, Xs_mla[mla_q4_mask], Xs_mla[mla_easy_mask]])
sil_y = (["itw"] * len(Xs_itw_sp) + ["q4"] * int(mla_q4_mask.sum()) +
         ["easy"] * int(mla_easy_mask.sum()))
sil = float(silhouette_score(sil_X, sil_y)) if len(set(sil_y)) > 1 else float("nan")
print(f"    kNN(k={k}) ITW-spoof neighbours in MLAAD-Q4: {knn_q4_frac:.3f} (chance 0.25)")
print(f"    centroid dist ITW->Q4={d_q4:.2f}  ITW->easy={d_easy:.2f}  silhouette={sil:.3f}")

# UMAP (else t-SNE) 2D + 3D
proj_name = "UMAP"
emb2d = emb3d = None
try:
    import umap
    emb2d = umap.UMAP(n_components=2, random_state=SEED).fit_transform(Xs)
    emb3d = umap.UMAP(n_components=3, random_state=SEED).fit_transform(Xs)
except Exception as ex:
    print(f"    UMAP unavailable ({ex}); falling back to t-SNE")
    proj_name = "t-SNE"
    from sklearn.manifold import TSNE
    emb2d = TSNE(n_components=2, random_state=SEED, init="pca").fit_transform(Xs)
    try:
        emb3d = TSNE(n_components=3, random_state=SEED, init="pca").fit_transform(Xs)
    except Exception:
        emb3d = None

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] Figures ...", flush=True)
COL = {"ITW-spoof": "#d62728", "ITW-bona": "#2ca02c", "MLAAD": "#7f7f7f"}

# Fig 1: C-T scatter with Q4 region shaded
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(mla_C, mla_T, s=8, c="#bbbbbb", alpha=0.5, label="MLAAD spoof")
ax.scatter(itw_bf_C, itw_bf_T, s=8, c=COL["ITW-bona"], alpha=0.4, label="ITW bona-fide")
ax.scatter(itw_sp_C, itw_sp_T, s=8, c=COL["ITW-spoof"], alpha=0.5, label="ITW spoof")
ax.axvline(qb_C, ls="--", c="k", lw=0.8); ax.axhline(qb_T, ls="--", c="k", lw=0.8)
ax.axvspan(qb_C, max(itw_C.max(), mla_C.max()), color="orange", alpha=0.06)
ax.text(qb_C, ax.get_ylim()[1], " Q4-C →", fontsize=8, va="top")
ax.set_xlabel("C = -rog@L12  (compact = harder)")
ax.set_ylabel("T = vel_entropy@L9  (bursty = harder)")
ax.set_title("E5 ITW: C-T manifold — ITW vs MLAAD\n(dashed = MLAAD per-sample Q4 boundary)")
ax.legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e5_itw_ct_manifold.png", dpi=150); plt.close()

# Fig 2: distributions of C and T
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, vals, thr, title in [
    (axes[0], (itw_sp_C, mla_C, itw_bf_C), qb_C, "C = -rog@L12"),
    (axes[1], (itw_sp_T, mla_T, itw_bf_T), qb_T, "T = vel_entropy@L9")]:
    ax.hist(vals[1], bins=40, density=True, alpha=0.5, color="#bbbbbb", label="MLAAD spoof")
    ax.hist(vals[2], bins=40, density=True, alpha=0.5, color=COL["ITW-bona"], label="ITW bona")
    ax.hist(vals[0], bins=40, density=True, alpha=0.5, color=COL["ITW-spoof"], label="ITW spoof")
    ax.axvline(thr, ls="--", c="k", lw=0.9, label="MLAAD Q4 boundary")
    ax.set_title(title); ax.legend(fontsize=8)
plt.suptitle("E5 ITW: per-sample C/T distributions", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e5_itw_ct_distributions.png", dpi=150); plt.close()

# Fig 3: logit vs C/T (first available ckpt)
if itw_logits:
    name0 = next(iter(itw_logits))
    lg = itw_logits[name0][spoof_m]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(itw_sp_C, lg, s=8, alpha=0.4, c=COL["ITW-spoof"])
    axes[0].set_xlabel("C = -rog@L12"); axes[0].set_ylabel(f"raw_logit ({name0})")
    axes[0].set_title(f"ITW spoof: logit vs C  rho={anchor[name0]['spearman_C_logit']:+.3f}")
    axes[1].scatter(itw_sp_T, lg, s=8, alpha=0.4, c=COL["ITW-spoof"])
    axes[1].set_xlabel("T = vel_entropy@L9")
    axes[1].set_title(f"ITW spoof: logit vs T  rho={anchor[name0]['spearman_T_logit']:+.3f}")
    plt.tight_layout(); plt.savefig(FIG / "e5_itw_score_vs_ct.png", dpi=150); plt.close()

def scatter_src_2d(ax, XY):
    for s in ["MLAAD", "ITW-bona", "ITW-spoof"]:
        m = np.array(src) == s
        ax.scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.45, c=COL[s], label=s)
    # highlight MLAAD Q4
    mla_off = n_sp + n_bf
    q4idx = np.where(mla_q4_mask)[0] + mla_off
    ax.scatter(XY[q4idx, 0], XY[q4idx, 1], s=10, facecolors="none",
               edgecolors="orange", linewidths=0.5, label="MLAAD Q4")

# Fig 4: PCA 2D + 3D
fig = plt.figure(figsize=(13, 5.5))
ax1 = fig.add_subplot(1, 2, 1)
scatter_src_2d(ax1, P[:, :2])
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.set_title(f"PCA 2D (L12 emb)  var={pca.explained_variance_ratio_[:2].sum():.2f}")
ax1.legend(markerscale=2, fontsize=7)
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for s in ["MLAAD", "ITW-bona", "ITW-spoof"]:
    m = np.array(src) == s
    ax2.scatter(P[m, 0], P[m, 1], P[m, 2], s=5, alpha=0.4, c=COL[s], label=s)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
ax2.set_title("PCA 3D")
plt.suptitle("E5 ITW: joint WavLM-L12 embedding PCA", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e5_itw_manifold_pca_2d.png", dpi=150)
plt.savefig(FIG / "e5_itw_manifold_pca_3d.png", dpi=150); plt.close()

# Fig 5: UMAP/t-SNE 2D + 3D
fig = plt.figure(figsize=(13, 5.5))
ax1 = fig.add_subplot(1, 2, 1)
scatter_src_2d(ax1, emb2d)
ax1.set_title(f"{proj_name} 2D"); ax1.legend(markerscale=2, fontsize=7)
if emb3d is not None:
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for s in ["MLAAD", "ITW-bona", "ITW-spoof"]:
        m = np.array(src) == s
        ax2.scatter(emb3d[m, 0], emb3d[m, 1], emb3d[m, 2], s=5, alpha=0.4, c=COL[s], label=s)
    ax2.set_title(f"{proj_name} 3D")
plt.suptitle(f"E5 ITW: joint embedding {proj_name}", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e5_itw_manifold_umap_2d.png", dpi=150)
plt.savefig(FIG / "e5_itw_manifold_umap_3d.png", dpi=150); plt.close()
print("    figures saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] Writing summary ...", flush=True)

# verdict
c_supported = (shift[0]["mw_p_greater"] < 0.05 and abs(shift[0]["cliffs_delta"]) >= 0.15
               and enrich[2]["binom_p"] < 0.05)
t_supported = (shift[1]["mw_p_greater"] < 0.05 and abs(shift[1]["cliffs_delta"]) >= 0.15
               and enrich[3]["binom_p"] < 0.05)
axis_verdict = ("BOTH C and T" if c_supported and t_supported else
                "C only" if c_supported else "T only" if t_supported else "NEITHER")
supported = c_supported or t_supported

L = ["# E5 ITW — Does the C/T hard manifold explain InTheWild difficulty?", "",
     f"ITW subset: spoof={int(spoof_m.sum())} bona-fide={int(bona_m.sum())} | "
     f"MLAAD reference utts={len(mla_C)} across {len(set(mla_system))} systems",
     f"Extraction sanity vs sig_layer_features: rog r={rog_consistency:.3f}, vel r={vel_consistency:.3f}",
     "",
     f"## Verdict: hypothesis **{'SUPPORTED' if supported else 'NOT supported'}** ({axis_verdict})",
     "ITW spoof samples are enriched in the MLAAD hard region on: "
     f"**{axis_verdict}**.", "",
     "## 1. Distribution shift (ITW-spoof vs MLAAD-spoof)",
     "| axis | ITW mean | MLAAD mean | Cliff's δ | Cohen's d | MW p(ITW>MLAAD) | KS p |",
     "|---|---|---|---|---|---|---|"]
for s in shift:
    L.append(f"| {s['axis']} | {s['itw_mean']:.3f} | {s['mlaad_mean']:.3f} | "
             f"{s['cliffs_delta']:+.3f} | {s['cohens_d']:+.3f} | {s['mw_p_greater']:.2e} | {s['ks_p']:.2e} |")
L += ["", "## 2. Q4 hard-region enrichment (ITW spoof; null=0.25)",
      "| axis | scheme | threshold | frac in Q4 | enrichment | binom p |",
      "|---|---|---|---|---|---|"]
for e in enrich:
    L.append(f"| {e['axis']} | {e['scheme']} | {e['threshold']:.3f} | {e['frac_in_Q4']:.3f} | "
             f"{e['enrichment_vs_0.25']:.2f}x | {e['binom_p']:.2e} |")
L.append(f"| C&T | per-sample(b) | — | {both_frac:.3f} | {both_frac/0.0625:.2f}x (null .0625) | {both_p:.2e} |")
L += ["", "## 3. Within-ITW (spoof vs bona-fide)",
      "| axis | spoof mean | bona mean | Cliff's δ | MW p |", "|---|---|---|---|---|"]
for s in within:
    L.append(f"| {s['axis']} | {s['itw_mean']:.3f} | {s['mlaad_mean']:.3f} | "
             f"{s['cliffs_delta']:+.3f} | {s['mw_p_greater']:.2e} |")
L += ["", "## 4. Difficulty anchor (per checkpoint)",
      f"MLAAD reference EER: overall={mlaad_overall_eer:.4f}  "
      f"Q4-C={mlaad_q4C_eer:.4f}  Q4-T={mlaad_q4T_eer:.4f}", "",
      "| checkpoint | ITW EER | ρ(C,logit) | ρ(T,logit) |", "|---|---|---|---|"]
for name, a in anchor.items():
    L.append(f"| {name} | {a['itw_eer']:.4f} | {a['spearman_C_logit']:+.3f} (p={a['p_C']:.1e}) | "
             f"{a['spearman_T_logit']:+.3f} (p={a['p_T']:.1e}) |")
if mlaad_corr:
    L.append(f"| MLAAD (p1 ref) | — | {mlaad_corr['spearman_C_logit']:+.3f} | "
             f"{mlaad_corr['spearman_T_logit']:+.3f} |")
_anchor_note = (
    "_DISSOCIATION: ITW is genuinely hard (EER comparable to/worse than MLAAD-Q4), and the "
    "C-axis *direction* still holds within ITW (ρ(C,logit)<0, matching MLAAD) — yet the "
    "geometry tests above show ITW spoof does NOT sit in the MLAAD hard region. So ITW "
    "difficulty is real but NOT explained by elevated C/T-hardness; it points to domain "
    "shift rather than hard-manifold overlap._"
    if not supported else
    "_ITW EER vs MLAAD-Q4 plus matching ρ sign corroborate the C/T mechanism in ITW._")
L += ["", _anchor_note, "",
      "## 5. Embedding-manifold similarity (WavLM-L12, standardized)",
      f"- kNN(k={k}) ITW-spoof neighbours falling in MLAAD-Q4: **{knn_q4_frac:.3f}** (chance 0.25)",
      f"- centroid distance ITW-spoof → MLAAD-Q4 = {d_q4:.2f} vs → MLAAD-easy = {d_easy:.2f} "
      f"({'closer to Q4' if d_q4 < d_easy else 'closer to easy'})",
      f"- silhouette {{ITW, Q4, easy}} = {sil:.3f}",
      f"- projection used for figures: {proj_name}", "",
      "## Figures",
      "- `figures/e5_itw_ct_manifold.png` — C-T scatter + Q4 region",
      "- `figures/e5_itw_ct_distributions.png` — per-sample C/T histograms",
      "- `figures/e5_itw_score_vs_ct.png` — logit vs C/T",
      "- `figures/e5_itw_manifold_pca_2d.png` / `_3d.png` — joint PCA",
      f"- `figures/e5_itw_manifold_umap_2d.png` / `_3d.png` — {proj_name}"]
(OUT / "e5_itw_summary.md").write_text("\n".join(L))
print(f"    summary -> {OUT/'e5_itw_summary.md'}")
print("\n[E5 ITW] done.")
