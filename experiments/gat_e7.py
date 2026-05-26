#!/usr/bin/env python3
"""
gat_e7.py
=========
E7: Per-sample routing-cluster classifier from structural A_agg features.

Goal: test whether the TTS-vs-VC structural split is recoverable from
individual samples using features derived from aggregated-attention matrices
that are invariant to phoneme vocabulary permutation.

A02 is EXCLUDED (ambiguous mechanism; would blur the cluster boundary).

Pre-registered hypotheses:
  H1 [primary]:   Structural-feature LR achieves bal_acc >= 0.65 AND
                  CI lower bound (mean - 2*std) > 0.55; AUC >= 0.70.
                  Falsification: bal_acc <= 0.55 or AUC <= 0.60.

  H2 [primary]:   Structural features outperform phoneme-distribution
                  baseline by >= 0.05 bal_acc (paired across folds).
                  Falsification: structural <= phoneme + 0.02.

  H3 [secondary]: Top-3 standardized LR coefs include >= 1 of:
                  {gini_in, gini_out, top1_mass, top5_mass}.
                  Falsification: none of the four in top 3.

  H4 [secondary]: Seeds 2 and 3 bal_acc within 0.10 of seed 1.
                  Falsification: either seed differs by > 0.10.

Usage:
  # Step 1: verify permutation invariance (extraction + invariance only)
  venv/bin/python3 experiments/gat_e7.py --verify-only

  # Step 2: full seed-1 primary analysis
  venv/bin/python3 experiments/gat_e7.py --seed1-only

  # Step 3: all seeds (H4 replication)
  venv/bin/python3 experiments/gat_e7.py

  # Force re-extract (ignore cache)
  venv/bin/python3 experiments/gat_e7.py --force-extract

Outputs -> experiments/results/gat_attn_graphs/e7/
"""
from __future__ import annotations

import argparse
import csv
import io as _io
import json
import os
import random
import sys
import time
import types
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── torch.load compat ─────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
EXP_DIR    = Path(__file__).resolve().parent
OUT_BASE   = EXP_DIR / "results" / "gat_attn_graphs" / "e7"
VOCAB_DIR  = REPO_ROOT / "vocab_phoneme"
CACHE_DIR  = OUT_BASE / "cache"
for d in [OUT_BASE, CACHE_DIR]: d.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Experiment constants ──────────────────────────────────────────────────────
ROUTING_SYSTEMS = {"A01", "A03", "A04"}   # class 0 (routing-dependent)
SKIP_SYSTEMS    = {"A05", "A06"}           # class 1 (skip-route / VC)
INCLUDE_SYSTEMS = ROUTING_SYSTEMS | SKIP_SYSTEMS

FEATURE_NAMES = [
    "gini_in",      # Gini of col-sums (in-degree) — concentration of hub reception
    "gini_out",     # Gini of row-sums (out-degree) — DEGENERATE for row-stochastic A_agg
    "entropy_in",   # Shannon entropy of normalized in-degree distribution
    "entropy_out",  # Shannon entropy of normalized out-degree — DEGENERATE for row-stochastic
    "spectral_gap", # |λ_1| - |λ_2| of A_agg — mixing speed proxy
    "top1_mass",    # max(d_in) / sum(d_in) — maximum hub concentration
    "top5_mass",    # sum(top-5 d_in) / sum(d_in) — top-hub mass
    "diag_mass",    # trace(A_agg) / sum(A_agg) — self-routing proxy
    "offdiag_frob", # ||A_agg - diag(A_agg)||_F — cross-routing magnitude
    "eff_rank",     # exp(H of norm. singular values) — matrix effective rank
]
CONCENTRATION_FEATURES = {"gini_in", "gini_out", "top1_mass", "top5_mass"}

SEED          = 42
N_PERM        = 5       # samples for permutation invariance test
PERM_TOL      = 1e-6
CV_SPLITS     = 5
NF_PER_SAMPLE = 3 * 16_000 // 320 - 1
N_PER_CLASS   = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 8))
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DATA    = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
LANG_ORDER    = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL       = ["|", "</s>", "<s>", "<unk>", "<pad>"]
TARGET_SR     = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
MIN_NODES     = 3


# ── Structural feature helpers ────────────────────────────────────────────────

def _gini(v: np.ndarray) -> float:
    v = v[v > 0]
    if len(v) == 0: return 0.0
    v = np.sort(v); n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)


def _entropy(v: np.ndarray) -> float:
    s = v.sum()
    if s < 1e-12: return 0.0
    p = v / s
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_structural_features(A_agg: np.ndarray) -> np.ndarray:
    """
    10 permutation-invariant features of a row-stochastic A_agg.
    Permutation-invariance argument for each feature:
      1-4: Gini/entropy depend only on the multiset of row/col sums, which
           is invariant under simultaneous row+col permutation.
      5:   Eigenvalues of A_agg are invariant under similarity transform;
           P @ A @ P^T (orthogonal P) is a similarity transform.
      6-7: Order statistics of col-sums; col-sums are a multiset invariant.
      8:   trace(P @ A @ P^T) = trace(A) and sum unchanged by permutation.
      9:   Frobenius norm is unitarily invariant; the off-diagonal block's
           F-norm equals ||P@(A - diag A)@P^T||_F = ||A - diag A||_F.
      10:  Singular values of P@A@P^T equal those of A (unitary invariance).

    NOTE: Features 2 and 4 (gini_out, entropy_out) are degenerate for
    row-stochastic A_agg because all row sums equal 1. They will be
    dropped by VarianceThreshold in the classifier pipeline.
    """
    N = A_agg.shape[0]
    d_in  = A_agg.sum(axis=0)   # col sums
    d_out = A_agg.sum(axis=1)   # row sums (= 1 for row-stochastic)

    f1 = _gini(d_in)
    f2 = _gini(d_out)                            # degenerate (always ≈ 0)
    f3 = _entropy(d_in)
    f4 = _entropy(d_out)                         # degenerate (always ≈ 0)

    if N > 1:
        eigv = np.sort(np.abs(np.linalg.eigvals(A_agg)))[::-1]
        f5 = float(eigv[0] - eigv[1])
    else:
        f5 = 0.0

    s = d_in.sum() + 1e-12
    f6 = float(d_in.max() / s)
    k  = min(5, N)
    f7 = float(np.partition(d_in, -k)[-k:].sum() / s)

    f8 = float(np.trace(A_agg) / (A_agg.sum() + 1e-12))

    A_od = A_agg - np.diag(np.diag(A_agg))
    f9   = float(np.sqrt((A_od ** 2).sum()))

    sv   = np.linalg.svd(A_agg, compute_uv=False)
    sv_n = sv / (sv.sum() + 1e-12)
    f10  = float(np.exp(-np.sum(sv_n * np.log(sv_n + 1e-12))))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float64)


# ── Vocabulary builder (verbatim from gat_e6.py) ─────────────────────────────

def build_vocab():
    _CAT_RULES = [
        ("tʃ","Affricates"),("dʒ","Affricates"),
        ("ʃ","Sibilants"),("ʒ","Sibilants"),("s","Sibilants"),("z","Sibilants"),
        ("ŋ","Nasals"),("n̩","Nasals"),("nʲ","Nasals"),("m̩","Nasals"),
        ("n","Nasals"),("m","Nasals"),
        ("ʔ","Stops"),("ɡʲ","Stops"),("ɡ","Stops"),("p","Stops"),("b","Stops"),
        ("t","Stops"),("d","Stops"),("k","Stops"),
        ("θ","Fricatives"),("ð","Fricatives"),("ɬ","Fricatives"),("ç","Fricatives"),
        ("x","Fricatives"),("f","Fricatives"),("v","Fricatives"),("h","Fricatives"),
        ("ɹ","Approximants"),("ɾ","Approximants"),("ʁ","Approximants"),
        ("əl","Approximants"),("l","Approximants"),("r","Approximants"),
        ("w","Approximants"),("j","Approximants"),
        ("aɪɚ","Diphthongs"),("aɪə","Diphthongs"),("oʊ","Diphthongs"),
        ("eɪ","Diphthongs"),("aɪ","Diphthongs"),("aʊ","Diphthongs"),
        ("ɔɪ","Diphthongs"),("iə","Diphthongs"),
        ("ɚ","Vowels"),("ɜː","Vowels"),("ɛɹ","Vowels"),("ɪɹ","Vowels"),
        ("ɔːɹ","Vowels"),("ɑːɹ","Vowels"),("ʊɹ","Vowels"),("oːɹ","Vowels"),
        ("iː","Vowels"),("uː","Vowels"),("ɪː","Vowels"),("ɛː","Vowels"),
        ("ɔː","Vowels"),("ɑː","Vowels"),("oː","Vowels"),
        ("ɪ","Vowels"),("ɛ","Vowels"),("æ","Vowels"),("ʌ","Vowels"),
        ("ɑ","Vowels"),("ɔ","Vowels"),("ʊ","Vowels"),("ə","Vowels"),
        ("ᵻ","Vowels"),("ɐ","Vowels"),("ɜ","Vowels"),
        ("a","Vowels"),("e","Vowels"),("i","Vowels"),("o","Vowels"),("u","Vowels"),
        ("ææ","Vowels"),
    ]
    def _sym(sym):
        if sym in SPECIAL or sym.isdigit(): return "Other"
        for sub, cls in _CAT_RULES:
            if sub in sym: return cls
        return "Other"
    total = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists(): continue
        for sym, _ in sorted(json.load(open(p)).items(), key=lambda x: x[1]):
            if sym not in SPECIAL: total.append(f"{lang}-{sym}")
    id_to_sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    return id_to_sym


def build_symbol_vocab(records, id_to_sym):
    seen = set()
    for rec in records:
        if rec["is_degenerate"]: continue
        for pid in rec["node_phoneme_ids"].tolist():
            seen.add(id_to_sym.get(int(pid), "?"))
    sym_list = sorted(seen)
    return sym_list, {s: i for i, s in enumerate(sym_list)}


# ── Audio & dataset (verbatim from gat_e6.py) ────────────────────────────────

def _decode(entry):
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else sf.read(path, dtype="float32", always_2d=False))
    w = torch.tensor(arr)
    if w.ndim == 1: w = w.unsqueeze(0)
    elif w.ndim == 2: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR: w = T.Resample(sr, TARGET_SR)(w)
    return w

def _crop(w):
    n = w.shape[-1]
    if n < TARGET_SAMPLES: w = w.repeat(1, -(-TARGET_SAMPLES // n))
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s: s + TARGET_SAMPLES]

def _lbl(raw):
    if isinstance(raw, str):
        return 0 if raw.strip().lower() in ("0","bonafide","real","genuine") else 1
    return int(raw)


class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, n_per_class, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=str(cache_dir), token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label")
        by_sys = defaultdict(list)
        for i in range(len(self.ds)):
            by_sys[self.ds[i].get("system_id","unknown")].append(i)
        rng = random.Random(seed)
        selected, sys_ids = [], []
        for sid, idxs in sorted(by_sys.items()):
            rng.shuffle(idxs); chosen = idxs[:n_per_class]
            selected.extend(chosen); sys_ids.extend([sid]*len(chosen))
        combined = list(zip(selected, sys_ids)); rng.shuffle(combined)
        self.indices, self.sys_ids = map(list, zip(*combined)) if combined else ([],[])

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        return {"audio":     _crop(_decode(ex["audio"])),
                "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
                "system_id": self.sys_ids[idx]}

def collate(batch):
    return {"audio":     torch.stack([b["audio"] for b in batch]),
            "label":     torch.stack([b["label"] for b in batch]),
            "system_id": [b["system_id"] for b in batch]}


# ── Model loading (verbatim) ──────────────────────────────────────────────────

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


def load_model(ckpt_path, device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit


# ── Extraction (verbatim from gat_e6.py) ─────────────────────────────────────

class PhonemeCapture:
    def __init__(self, gat_model):
        self.node_phoneme_ids = self.node_sample_idx = self.reduced_num_frames = None
        cap = self; orig = gat_model.encoder_and_GAT.__func__

        def _patched(self_i, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_i, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            rids = result[2].detach().cpu(); rnf = result[3].detach().cpu()
            flat_ids, flat_samp = [], []
            for i in range(len(rnf)):
                n = int(rnf[i].item())
                flat_ids.append(rids[i, :n])
                flat_samp.append(torch.full((n,), i, dtype=torch.long))
            cap.node_phoneme_ids   = torch.cat(flat_ids)
            cap.node_sample_idx    = torch.cat(flat_samp)
            cap.reduced_num_frames = rnf
            return result

        gat_model.encoder_and_GAT = types.MethodType(_patched, gat_model)


def run_frozen_frontend(audio, gm, device):
    x = audio
    if x.ndim == 3 and x.size(1) == 1: x = x[:, 0, :]
    with torch.no_grad():
        feat1  = gm.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hs, _  = gm.transformer_in_phoneme_model.feature_projection(feat1)
        pf     = gm.transformer_in_phoneme_model.encoder(hs)[0]
        pl     = gm.phoneme_model.model.model.lm_head(pf)
        pids   = torch.argmax(pl, dim=-1)
    return hs, pids


def extract_all_layers(lit, loader, device):
    gm = lit.model; n_layers = 3
    for i in range(n_layers): gm.GAT.gat_net[i].log_attention_weights = True
    buf = {f"attn_{i}": None for i in range(n_layers)}
    buf.update({f"eidx_{i}": None for i in range(n_layers)})
    hooks = []
    for li in range(n_layers):
        def _make_hook(l):
            def _hook(module, inp, out):
                buf[f"eidx_{l}"] = out[1].detach().cpu()
                if module.attention_weights is not None:
                    buf[f"attn_{l}"] = module.attention_weights.squeeze(-1).detach().cpu()
            return _hook
        hooks.append(gm.GAT.gat_net[li].register_forward_hook(_make_hook(li)))

    phoneme_cap = PhonemeCapture(gm)
    records = []; n_degenerate = 0; total = len(loader.dataset)

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)
                hs, pids = run_frozen_frontend(audio, gm, device)
                gm.encoder_and_GAT(hs, num_f, pids)

                ei_global = buf["eidx_0"]
                attn_all  = [buf[f"attn_{li}"] for li in range(n_layers)]
                if any(a is None for a in attn_all) or ei_global is None: continue

                node_pids     = phoneme_cap.node_phoneme_ids
                node_samp_idx = phoneme_cap.node_sample_idx
                rnf           = phoneme_cap.reduced_num_frames
                node_offsets  = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                base_id = len(records)
                for i in range(B):
                    n_nodes_i = int(rnf[i].item()); off = node_offsets[i]
                    edge_mask = (node_samp_idx[ei_global[0]] == i)
                    ei_local  = ei_global[:, edge_mask] - off
                    pids_i    = node_pids[node_samp_idx == i]
                    is_deg    = n_nodes_i < MIN_NODES
                    if is_deg: n_degenerate += 1
                    rec = {"sample_id": base_id + i, "label": labels[i],
                           "system_id": sys_ids[i],
                           "node_phoneme_ids": pids_i.clone(),
                           "edge_index": ei_local.clone(),
                           "n_nodes": n_nodes_i, "is_degenerate": is_deg}
                    for li in range(n_layers):
                        rec[f"attn_l{li}"] = attn_all[li][edge_mask].clone()
                    records.append(rec)

                for li in range(n_layers):
                    buf[f"attn_{li}"] = None; buf[f"eidx_{li}"] = None

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    print(f"  {min((bi+1)*BATCH_SIZE, total)}/{total}")
    finally:
        for h in hooks: h.remove()

    print(f"  Extracted {len(records)} samples ({n_degenerate} degenerate)")
    return records


def get_or_extract(ckpt_path: Path, loader, device, force=False):
    cache_file = CACHE_DIR / f"records_{ckpt_path.stem}.pt"
    if cache_file.exists() and not force:
        print(f"  Loading cached records: {cache_file.name}")
        return torch.load(str(cache_file))
    print(f"  Extracting attention from {ckpt_path.name}...")
    lit = load_model(ckpt_path, device)
    records = extract_all_layers(lit, loader, device)
    torch.save(records, str(cache_file))
    print(f"  Cached to {cache_file.name}")
    return records


# ── A_agg builder (verbatim) ──────────────────────────────────────────────────

def sparse_to_dense(attn_mean, edge_index, N):
    A = np.zeros((N, N), dtype=np.float64)
    np.add.at(A, (edge_index[1], edge_index[0]), attn_mean)
    row_sums = A.sum(axis=1); no_in = np.where(row_sums < 1e-10)[0]
    A[no_in, no_in] = 1.0
    return A


def build_agg_graph(rec):
    N = rec["n_nodes"]; ei = rec["edge_index"].numpy(); layers = []
    for li in range(3):
        attn_np = rec[f"attn_l{li}"].float().numpy()
        layers.append(sparse_to_dense(attn_np.mean(axis=1), ei, N))
    A_agg = layers[2] @ layers[1] @ layers[0]
    rs    = A_agg.sum(axis=1, keepdims=True)
    return A_agg / np.where(rs > 1e-10, rs, 1.0)


# ── Permutation invariance verification ───────────────────────────────────────

def verify_permutation_invariance(records: list, n_samples: int = N_PERM) -> bool:
    """
    For n_samples valid records, permute A_agg rows+cols with the same permutation
    and verify all 10 structural features are unchanged within PERM_TOL.
    """
    rng   = np.random.default_rng(0)
    valid = [r for r in records if not r["is_degenerate"]]
    if len(valid) < n_samples:
        print(f"  Warning: only {len(valid)} valid records, using all.")
        n_samples = len(valid)

    chosen_idx = rng.choice(len(valid), size=n_samples, replace=False)
    chosen     = [valid[i] for i in chosen_idx]

    feat_deltas = np.zeros((len(FEATURE_NAMES), n_samples))

    for si, rec in enumerate(chosen):
        A_agg  = build_agg_graph(rec)
        f_orig = compute_structural_features(A_agg)
        N      = A_agg.shape[0]
        perm   = rng.permutation(N)
        A_perm = A_agg[np.ix_(perm, perm)]
        f_perm = compute_structural_features(A_perm)
        feat_deltas[:, si] = np.abs(f_orig - f_perm)

    print()
    print("=" * 66)
    print("PERMUTATION INVARIANCE VERIFICATION")
    print(f"  (n={n_samples} random samples, same row+col permutation, tol={PERM_TOL:.0e})")
    print("=" * 66)
    print(f"  {'feature':16s}  {'max |Δ|':>12s}  {'status':>8s}  note")
    print(f"  {'-'*16}  {'-'*12}  {'-'*8}  ----")
    all_pass = True
    degenerate_idx = set()
    for fi, name in enumerate(FEATURE_NAMES):
        max_delta = feat_deltas[fi].max()
        status    = "PASS" if max_delta < PERM_TOL else "FAIL"
        if status == "FAIL": all_pass = False
        note = ""
        if name in ("gini_out", "entropy_out"):
            note = "← degenerate (row-stochastic A_agg)"
            degenerate_idx.add(fi)
        print(f"  {name:16s}  {max_delta:12.2e}  {status:>8s}  {note}")
    print("=" * 66)
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")
    if degenerate_idx:
        print(f"  Note: features {[FEATURE_NAMES[i] for i in sorted(degenerate_idx)]} are")
        print(f"        expected to be zero for row-stochastic A_agg and will be dropped")
        print(f"        by VarianceThreshold in the classifier pipeline.")
    print()
    return all_pass


# ── Feature matrix builders ───────────────────────────────────────────────────

def build_structural_matrix(records: list) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, sids) for the 250 samples in INCLUDE_SYSTEMS.
    Samples are sorted by (system_id, sample_id) for deterministic order.
    Degenerate records are dropped; warns if count differs from expected.
    """
    rows = []
    for rec in records:
        if rec["system_id"] not in INCLUDE_SYSTEMS: continue
        if rec["is_degenerate"]: continue
        A_agg = build_agg_graph(rec)
        feat  = compute_structural_features(A_agg)
        label = 0 if rec["system_id"] in ROUTING_SYSTEMS else 1
        rows.append((rec["system_id"], rec["sample_id"], feat, label))

    rows.sort(key=lambda r: (r[0], r[1]))

    X    = np.array([r[2] for r in rows], dtype=np.float64)
    y    = np.array([r[3] for r in rows], dtype=np.int64)
    sids = [r[0] for r in rows]

    counts = {s: sum(1 for r in rows if r[0] == s) for s in INCLUDE_SYSTEMS}
    print(f"  Structural matrix: N={len(rows)} samples")
    for s in sorted(INCLUDE_SYSTEMS):
        cls = "routing" if s in ROUTING_SYSTEMS else "skip"
        print(f"    {s} ({cls}): {counts.get(s, 0)}")
    return X, y, sids


def build_phoneme_matrix(records: list, id_to_sym: dict, sym_to_idx: dict,
                         V: int) -> tuple[np.ndarray, list[str]]:
    """
    Build (X_phon, sids) — normalized phoneme count vectors (V-dim simplex).
    Uses the SAME sample ordering as build_structural_matrix.
    """
    rows = []
    for rec in records:
        if rec["system_id"] not in INCLUDE_SYSTEMS: continue
        if rec["is_degenerate"]: continue
        pids = rec["node_phoneme_ids"].tolist()
        cnt  = np.zeros(V, dtype=np.float64)
        for pid in pids:
            sym = id_to_sym.get(int(pid), "?")
            idx = sym_to_idx.get(sym, 0)
            cnt[idx] += 1
        s = cnt.sum()
        if s > 0: cnt /= s
        rows.append((rec["system_id"], rec["sample_id"], cnt))

    rows.sort(key=lambda r: (r[0], r[1]))
    X_phon = np.array([r[2] for r in rows], dtype=np.float64)
    sids   = [r[0] for r in rows]
    return X_phon, sids


# ── Classifier evaluation ─────────────────────────────────────────────────────

def make_lr_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("thresh", VarianceThreshold(threshold=1e-10)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])

def make_gbt_pipeline():
    return Pipeline([
        ("thresh", VarianceThreshold(threshold=1e-10)),
        ("clf",   GradientBoostingClassifier(max_depth=3, n_estimators=100,
                                              random_state=SEED)),
    ])


def eval_pipeline(pipe, X: np.ndarray, y: np.ndarray,
                  splits: list) -> dict:
    """
    Evaluate pipeline on pre-defined CV splits.
    Returns per-fold and aggregate metrics.
    """
    bal_accs, f1s, aucs = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        bal_accs.append(balanced_accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, y_prob))
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    return {
        "bal_acc":  (float(np.mean(bal_accs)), float(np.std(bal_accs))),
        "f1":       (float(np.mean(f1s)),      float(np.std(f1s))),
        "auc":      (float(np.mean(aucs)),      float(np.std(aucs))),
        "folds":    {"bal_acc": bal_accs, "f1": f1s, "auc": aucs},
        "all_true": all_y_true,
        "all_pred": all_y_pred,
        "all_prob": all_y_prob,
    }


def get_lr_feature_importances(pipe, X: np.ndarray, y: np.ndarray) -> list[tuple]:
    """
    Fit pipeline on all data; extract standardized LR coefficients.
    Maps through VarianceThreshold support back to FEATURE_NAMES.
    Returns list of (feature_name, abs_coef, coef) sorted by abs descending.
    """
    pipe.fit(X, y)
    vt      = pipe.named_steps["thresh"]
    support = vt.get_support()
    retained = [FEATURE_NAMES[i] for i, keep in enumerate(support) if keep]
    coefs    = pipe.named_steps["clf"].coef_[0]
    result   = [(name, abs(c), c) for name, c in zip(retained, coefs)]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def get_gbt_feature_importances(pipe, X: np.ndarray, y: np.ndarray) -> list[tuple]:
    pipe.fit(X, y)
    vt      = pipe.named_steps["thresh"]
    support = vt.get_support()
    retained = [FEATURE_NAMES[i] for i, keep in enumerate(support) if keep]
    imps     = pipe.named_steps["clf"].feature_importances_
    result   = [(name, imp, imp) for name, imp in zip(retained, imps)]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ── Hypothesis evaluation ─────────────────────────────────────────────────────

def evaluate_hypotheses(seed1_lr: dict, seed1_phon: dict,
                        all_lr: dict[str, dict],
                        lr_importances: list[tuple]) -> dict:
    """
    Apply pre-registered decision rules H1–H4.
    all_lr: {seed_label: metrics_dict}, must include "seed1".
    """
    print("\n" + "=" * 70)
    print("PRE-REGISTERED HYPOTHESIS OUTCOMES  (E7)")
    print("=" * 70)
    results = {}

    # H1: structural LR performance
    ba_mean, ba_std = seed1_lr["bal_acc"]
    auc_mean, _     = seed1_lr["auc"]
    ci_lo = ba_mean - 2 * ba_std  # informal lower bound
    h1_pass = (ba_mean >= 0.65) and (ci_lo > 0.55) and (auc_mean >= 0.70)
    print(f"\nH1 [primary]: structural LR bal_acc >= 0.65 (CI lo > 0.55), AUC >= 0.70")
    print(f"  bal_acc = {ba_mean:.4f} ± {ba_std:.4f}  (CI lo ≈ {ci_lo:.4f})")
    print(f"  AUC     = {auc_mean:.4f}")
    print(f"  Thresholds: bal_acc >= 0.65 {'✓' if ba_mean >= 0.65 else '✗'}  "
          f"CI lo > 0.55 {'✓' if ci_lo > 0.55 else '✗'}  "
          f"AUC >= 0.70 {'✓' if auc_mean >= 0.70 else '✗'}")
    print(f"  Result: {'PASS ✓' if h1_pass else 'FAIL ✗'}")
    results["H1"] = {"pass": h1_pass, "ba_mean": ba_mean, "ba_std": ba_std,
                     "ci_lo": ci_lo, "auc_mean": auc_mean}

    # H2: structural vs phoneme baseline (paired, per-fold)
    fold_struct = seed1_lr["folds"]["bal_acc"]
    fold_phon   = seed1_phon["folds"]["bal_acc"]
    fold_deltas = [s - p for s, p in zip(fold_struct, fold_phon)]
    delta_mean  = float(np.mean(fold_deltas))
    delta_std   = float(np.std(fold_deltas))
    h2_pass = delta_mean >= 0.05
    print(f"\nH2 [primary]: structural outperforms phoneme baseline by >= 0.05 bal_acc")
    print(f"  Structural:  {seed1_lr['bal_acc'][0]:.4f} ± {seed1_lr['bal_acc'][1]:.4f}")
    print(f"  Phoneme bl:  {seed1_phon['bal_acc'][0]:.4f} ± {seed1_phon['bal_acc'][1]:.4f}")
    print(f"  Δ (paired):  {delta_mean:+.4f} ± {delta_std:.4f}")
    print(f"  Threshold: Δ >= 0.05  Falsification: Δ <= 0.02")
    print(f"  Result: {'PASS ✓' if h2_pass else 'FAIL ✗'}")
    results["H2"] = {"pass": h2_pass, "delta_mean": delta_mean, "delta_std": delta_std}

    # H3: top-3 LR importances include >= 1 concentration feature
    top3_names = [name for name, _, _ in lr_importances[:3]]
    top3_conc  = [n for n in top3_names if n in CONCENTRATION_FEATURES]
    h3_pass    = len(top3_conc) >= 1
    print(f"\nH3 [secondary]: top-3 LR coefs include >= 1 concentration feature")
    print(f"  Top-3 features: {top3_names}")
    print(f"  Concentration in top-3: {top3_conc if top3_conc else 'none'}")
    print(f"  Result: {'PASS ✓' if h3_pass else 'FAIL ✗'}")
    results["H3"] = {"pass": h3_pass, "top3": top3_names, "conc_in_top3": top3_conc}

    # H4: seeds 2 and 3 within 0.10 of seed 1
    seed1_ba = seed1_lr["bal_acc"][0]
    h4_diffs  = {}
    for slabel, metrics in all_lr.items():
        if slabel == "seed1": continue
        diff = abs(metrics["bal_acc"][0] - seed1_ba)
        h4_diffs[slabel] = diff
    h4_pass = all(d <= 0.10 for d in h4_diffs.values())
    print(f"\nH4 [secondary]: seeds 2 & 3 bal_acc within 0.10 of seed 1 ({seed1_ba:.4f})")
    for slabel, diff in h4_diffs.items():
        ba = all_lr[slabel]["bal_acc"][0]
        ok = "✓" if diff <= 0.10 else "✗"
        print(f"  {slabel}: {ba:.4f}  |Δ|={diff:.4f}  {ok}")
    if not h4_diffs:
        print(f"  (no replication seeds run yet)")
    print(f"  Result: {'PASS ✓' if h4_pass else ('FAIL ✗' if h4_diffs else 'N/A (single seed)')}")
    results["H4"] = {"pass": h4_pass, "diffs": h4_diffs}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    for h, res in results.items():
        status = "PASS ✓" if res["pass"] else "FAIL ✗"
        print(f"  {h}: {status}")

    if results["H1"]["pass"] and results["H2"]["pass"]:
        verdict = ("H1 + H2 PASS → per-sample cluster split is real and "
                   "attention-driven. Taxonomy moves from population-level "
                   "to sample-level claim. Add to paper.")
    elif results["H1"]["pass"]:
        verdict = ("H1 PASS, H2 FAIL → cluster split is real but recoverable "
                   "from phoneme distribution alone. Honest negative on "
                   "attention-mediation claim. Mention as caveat, do not headline.")
    else:
        verdict = ("H1 FAIL → cluster split is not a per-sample property. "
                   "Stays as population-level claim only. Do not add to paper.")

    print(f"\n  Decision: {verdict}")
    print("=" * 70)
    return results, verdict


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics_table(all_results: dict[str, dict],
                        phon_results: dict | None = None):
    """Print cross-seed metrics comparison table."""
    print(f"\n{'='*70}")
    print("CROSS-SEED METRICS (5-fold CV)")
    print(f"{'='*70}")
    header = f"  {'seed':8s}  {'bal_acc':>12s}  {'f1':>12s}  {'auc':>12s}"
    print(header); print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for slabel, res in all_results.items():
        ba_m, ba_s = res["bal_acc"]
        f1_m, f1_s = res["f1"]
        au_m, au_s = res["auc"]
        print(f"  {slabel:8s}  "
              f"{ba_m:.3f}±{ba_s:.3f}  "
              f"{f1_m:.3f}±{f1_s:.3f}  "
              f"{au_m:.3f}±{au_s:.3f}")
    if phon_results:
        ba_m, ba_s = phon_results["bal_acc"]
        f1_m, f1_s = phon_results["f1"]
        au_m, au_s = phon_results["auc"]
        print(f"  {'phoneme_bl':8s}  "
              f"{ba_m:.3f}±{ba_s:.3f}  "
              f"{f1_m:.3f}±{f1_s:.3f}  "
              f"{au_m:.3f}±{au_s:.3f}")
    print()


def print_importance_table(importances: list[tuple], title: str):
    print(f"\n{title}")
    print(f"  {'rank':4s}  {'feature':16s}  {'|coef|/imp':>12s}  {'conc?':>6s}")
    print(f"  {'----':4s}  {'-'*16}  {'-'*12}  {'-'*6}")
    for rank, (name, abs_val, _) in enumerate(importances, 1):
        conc = "✓" if name in CONCENTRATION_FEATURES else ""
        print(f"  {rank:4d}  {name:16s}  {abs_val:12.4f}  {conc:>6s}")


def print_confusion_matrix(y_true: list, y_pred: list):
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion matrix (from concatenated CV folds):")
    print(f"             predicted:  routing  skip-route")
    print(f"  actual routing:        {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  actual skip-route:     {cm[1,0]:6d}  {cm[1,1]:6d}")


def save_results(seed_label: str, lr_res: dict, gbt_res: dict,
                 phon_res: dict | None, lr_imp: list, gbt_imp: list,
                 hyp_results: dict | None, verdict: str | None,
                 all_lr: dict | None):
    """Write CSV + markdown for one seed (or the final multi-seed report)."""
    # Per-seed CSV
    rows = []
    for metric in ("bal_acc", "f1", "auc"):
        m, s = lr_res[metric]
        rows.append({"seed": seed_label, "model": "lr_struct",
                     "metric": metric, "mean": m, "std": s})
        m, s = gbt_res[metric]
        rows.append({"seed": seed_label, "model": "gbt_struct",
                     "metric": metric, "mean": m, "std": s})
        if phon_res:
            m, s = phon_res[metric]
            rows.append({"seed": seed_label, "model": "lr_phoneme",
                         "metric": metric, "mean": m, "std": s})

    csv_path = OUT_BASE / f"e7_metrics_{seed_label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Importance CSV
    imp_path = OUT_BASE / f"e7_importances_{seed_label}.csv"
    with open(imp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","feature","lr_abs_coef","gbt_importance"])
        w.writeheader()
        lr_d  = {n: a for n, a, _ in lr_imp}
        gbt_d = {n: a for n, a, _ in gbt_imp}
        all_feats = sorted(set(lr_d) | set(gbt_d), key=lambda n: lr_d.get(n, 0), reverse=True)
        for rank, name in enumerate(all_feats, 1):
            w.writerow({"rank": rank, "feature": name,
                        "lr_abs_coef": f"{lr_d.get(name, 0):.6f}",
                        "gbt_importance": f"{gbt_d.get(name, 0):.6f}"})

    print(f"  Saved: {csv_path.name}, {imp_path.name}")


# ── Per-seed analysis ─────────────────────────────────────────────────────────

def run_seed_analysis(records: list, seed_label: str, cv_splits: list,
                      id_to_sym: dict, sym_to_idx: dict, V: int,
                      phon_X: np.ndarray | None = None,
                      phon_y: np.ndarray | None = None) -> tuple[dict, dict, dict, list, list]:
    """
    Run full E7 analysis for one seed checkpoint.
    Returns (lr_res, gbt_res, phon_res, lr_importances, gbt_importances).
    phon_X/phon_y: if provided, use these instead of re-computing (seed 2/3 use seed-1 phonemes).
    """
    print(f"\n{'='*70}")
    print(f"SEED ANALYSIS: {seed_label}")
    print(f"{'='*70}")

    X_struct, y_struct, sids = build_structural_matrix(records)

    if phon_X is None:
        X_phon_raw, phon_sids = build_phoneme_matrix(records, id_to_sym, sym_to_idx, V)
        # Verify same order
        assert sids == phon_sids, "Phoneme and structural sample ordering mismatch!"
        phon_X = X_phon_raw
        phon_y = y_struct

    # LR on structural features
    print(f"\n  Logistic Regression (structural features):")
    lr_pipe = make_lr_pipeline()
    lr_res  = eval_pipeline(lr_pipe, X_struct, y_struct, cv_splits)
    ba_m, ba_s = lr_res["bal_acc"]
    au_m, _    = lr_res["auc"]
    print(f"    bal_acc={ba_m:.4f}±{ba_s:.4f}  AUC={au_m:.4f}")

    # GBT on structural features
    print(f"  Gradient Boosted Trees (structural features):")
    gbt_pipe = make_gbt_pipeline()
    gbt_res  = eval_pipeline(gbt_pipe, X_struct, y_struct, cv_splits)
    ba_m, ba_s = gbt_res["bal_acc"]
    au_m, _    = gbt_res["auc"]
    print(f"    bal_acc={ba_m:.4f}±{ba_s:.4f}  AUC={au_m:.4f}")

    # Phoneme baseline (LR on phoneme counts)
    print(f"  Phoneme distribution baseline:")
    phon_pipe = make_lr_pipeline()
    phon_res  = eval_pipeline(phon_pipe, phon_X, phon_y, cv_splits)
    ba_m, ba_s = phon_res["bal_acc"]
    au_m, _    = phon_res["auc"]
    print(f"    bal_acc={ba_m:.4f}±{ba_s:.4f}  AUC={au_m:.4f}")

    # Feature importances (fit on all data)
    lr_imp  = get_lr_feature_importances(make_lr_pipeline(), X_struct, y_struct)
    gbt_imp = get_gbt_feature_importances(make_gbt_pipeline(), X_struct, y_struct)

    print_importance_table(lr_imp,  f"\n  LR feature importances (standardized |coef|):")
    print_importance_table(gbt_imp, f"\n  GBT feature importances:")
    print_confusion_matrix(lr_res["all_true"], lr_res["all_pred"])

    save_results(seed_label, lr_res, gbt_res, phon_res, lr_imp, gbt_imp,
                 None, None, None)

    return lr_res, gbt_res, phon_res, lr_imp, gbt_imp, X_struct, y_struct, phon_X, phon_y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+",
        default=["models/robust_goat.ckpt",
                 "models/robust_goat_seed7.ckpt",
                 "models/robust_goat_seed3.ckpt"],
        help="Checkpoints for seed1, seed2, seed3 (in order)")
    parser.add_argument("--seed-labels", nargs="+",
        default=["seed1", "seed2", "seed3"])
    parser.add_argument("--verify-only", action="store_true",
        help="Extract seed-1 features and run permutation invariance check only")
    parser.add_argument("--seed1-only", action="store_true",
        help="Run full analysis for seed 1 only")
    parser.add_argument("--force-extract", action="store_true",
        help="Re-extract attention (ignore cache)")
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.seed_labels), \
        "Number of --ckpts must match --seed-labels"

    ckpt_paths = []
    for c in args.ckpts:
        p = Path(c)
        if not p.is_absolute(): p = REPO_ROOT / p
        if not p.exists():
            print(f"ERROR: checkpoint not found: {p}"); sys.exit(1)
        ckpt_paths.append(p)

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    patch_phoneme_loader()

    # ── Load dataset (same for all seeds) ────────────────────────────────────
    print(f"\n=== E7: Per-sample Routing-Cluster Classifier ===")
    print(f"  Seeds: {list(zip(args.seed_labels, [p.name for p in ckpt_paths]))}")
    print(f"  Device: {device}")
    print(f"  CV: {CV_SPLITS}-fold stratified, random_state={SEED}")
    print(f"  A02 excluded per pre-registration.")

    dataset = BalancedDataset(HF_DATASET, "validation", CACHE_DATA,
                              hf_token, N_PER_CLASS, SEED)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate)

    # ── Extract seed-1 records ────────────────────────────────────────────────
    print(f"\n--- Extracting: {args.seed_labels[0]} ({ckpt_paths[0].name}) ---")
    records_s1 = get_or_extract(ckpt_paths[0], loader, device, args.force_extract)
    valid_s1   = [r for r in records_s1 if not r["is_degenerate"]]

    id_to_sym = build_vocab()
    sym_list, sym_to_idx = build_symbol_vocab(records_s1, id_to_sym)
    V = len(sym_list)
    print(f"  Vocab V={V}")

    # ── Permutation invariance check ──────────────────────────────────────────
    inv_pass = verify_permutation_invariance(records_s1, N_PERM)
    if not inv_pass:
        print("ABORT: permutation invariance check failed. Fix features before proceeding.")
        sys.exit(3)

    if args.verify_only:
        print("  [--verify-only: stopping after invariance check]")
        return

    # ── Generate CV splits (fixed; same for all seeds + baselines) ───────────
    # Build from seed-1 structural matrix to get stable y ordering
    X_s1_struct, y_s1, sids_s1 = build_structural_matrix(records_s1)
    skf     = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    cv_splits = list(skf.split(X_s1_struct, y_s1))
    print(f"\n  Generated {CV_SPLITS} CV splits (stratified, seed={SEED}).")
    for fi, (tr, te) in enumerate(cv_splits):
        print(f"    Fold {fi+1}: train={len(tr)}, test={len(te)}, "
              f"pos_rate_test={y_s1[te].mean():.2f}")

    # ── Seed-1 full analysis ──────────────────────────────────────────────────
    s1_lr, s1_gbt, s1_phon, s1_lr_imp, s1_gbt_imp, \
        X_s1, y_s1, phon_X, phon_y = run_seed_analysis(
            records_s1, args.seed_labels[0], cv_splits,
            id_to_sym, sym_to_idx, V)

    if args.seed1_only:
        print("\n  [--seed1-only: stopping after seed 1]")
        # Still evaluate H1-H3 (H4 N/A)
        all_lr  = {args.seed_labels[0]: s1_lr}
        hyp_res, verdict = evaluate_hypotheses(
            s1_lr, s1_phon, all_lr, s1_lr_imp)
        print_metrics_table({args.seed_labels[0]: s1_lr}, s1_phon)
        return

    # ── Seeds 2 and 3 ─────────────────────────────────────────────────────────
    all_lr = {args.seed_labels[0]: s1_lr}

    for i in range(1, len(ckpt_paths)):
        slabel = args.seed_labels[i]
        print(f"\n--- Extracting: {slabel} ({ckpt_paths[i].name}) ---")
        records_si = get_or_extract(ckpt_paths[i], loader, device, args.force_extract)

        lr_si, gbt_si, phon_si, lr_imp_si, gbt_imp_si, \
            X_si, y_si, _, _ = run_seed_analysis(
                records_si, slabel, cv_splits,
                id_to_sym, sym_to_idx, V,
                phon_X=phon_X, phon_y=phon_y)

        all_lr[slabel] = lr_si

    # ── All-seed metrics table ────────────────────────────────────────────────
    print_metrics_table(all_lr, s1_phon)

    # ── Hypothesis evaluation (full H1–H4) ────────────────────────────────────
    hyp_res, verdict = evaluate_hypotheses(
        s1_lr, s1_phon, all_lr, s1_lr_imp)

    # ── Save final report ─────────────────────────────────────────────────────
    lines = [
        "# E7: Per-sample Routing-Cluster Classifier",
        "",
        "## Exclusion note",
        "A02 excluded (ambiguous mechanism; would blur the cluster boundary).",
        "",
        "## Data",
        "- Routing-dependent (class 0): A01, A03, A04 — 50 samples each = 150",
        "- Skip-route (class 1): A05, A06 — 50 samples each = 100",
        "- Total N = 250  (60/40 class split)",
        "",
        "## Cross-seed metrics (structural LR, 5-fold CV)",
        "",
        "| seed | bal_acc | F1 | AUC |",
        "|------|---------|----|-----|",
    ]
    for slabel, res in all_lr.items():
        ba_m, ba_s = res["bal_acc"]
        f1_m, f1_s = res["f1"]
        au_m, au_s = res["auc"]
        lines.append(f"| {slabel} | {ba_m:.3f}±{ba_s:.3f} | "
                     f"{f1_m:.3f}±{f1_s:.3f} | {au_m:.3f}±{au_s:.3f} |")

    ba_m, ba_s = s1_phon["bal_acc"]
    f1_m, f1_s = s1_phon["f1"]
    au_m, au_s = s1_phon["auc"]
    lines.append(f"| phoneme_bl | {ba_m:.3f}±{ba_s:.3f} | "
                 f"{f1_m:.3f}±{f1_s:.3f} | {au_m:.3f}±{au_s:.3f} |")

    lines += ["", "## Feature importances (seed 1, LR standardized |coef|)", ""]
    lines.append("| rank | feature | |coef| | conc? |")
    lines.append("|------|---------|-------|-------|")
    for rank, (name, abs_val, _) in enumerate(s1_lr_imp, 1):
        conc = "✓" if name in CONCENTRATION_FEATURES else ""
        lines.append(f"| {rank} | {name} | {abs_val:.4f} | {conc} |")

    lines += ["", "## Hypothesis outcomes", ""]
    for h, res in hyp_res.items():
        status = "PASS" if res["pass"] else "FAIL"
        lines.append(f"- **{h}**: {status}")

    lines += ["", f"**Decision: {verdict}**"]

    md_path = OUT_BASE / "e7_report.md"
    md_path.write_text("\n".join(lines))
    print(f"\n  Saved: {md_path}")
    print(f"\nAll E7 outputs in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
