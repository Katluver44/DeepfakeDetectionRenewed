#!/usr/bin/env python3
"""
gat_e8.py
=========
E8 — Combined TTS+VC Generalization Test for E7 Classifier

Goal: test whether the E7 routing-vs-skip-route classifier, trained on
ASVspoof 2019 LA (A01–A06), generalises to UNSEEN TTS and VC systems from
entirely different datasets.

Datasets
--------
MLAAD (English subset, TTS, ground-truth label: routing-dependent / class 0)
  Source: HuggingFace mueller91/MLAAD-tiny
  Citation: Müller et al. 2024, arXiv:2401.09512
  License: CC-BY-NC-4.0 (non-commercial research use)

VCC2020 Task 1 (English intra-lingual VC, ground-truth label: skip-route / class 1)
  Source: Zenodo record 4433173 (NOT 4345998 which is metadata only)
  Citation: Yi et al. 2020, DOI 10.21437/VCC_BC.2020-14
  License: ODbL (audio) / DbCL (individual contents)

Usage
-----
  # Stage 1-2: download, inspect, phoneme-dist, sanity check (stop before classification)
  venv/bin/python3 experiments/gat_e8.py

  # Stage 1-3: full classification (run AFTER committing pre-registration)
  venv/bin/python3 experiments/gat_e8.py --run-classification

  # Force re-download both datasets
  venv/bin/python3 experiments/gat_e8.py --force-download

  # Force re-extract features (ignore cache)
  venv/bin/python3 experiments/gat_e8.py --force-extract

Outputs → experiments/results/e8_combined_generalization/
"""
from __future__ import annotations

import argparse
import csv
import io as _io
import json
import os
import pickle
import random
import re
import sys
import time
import types
import urllib.request
import zipfile
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
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
REPO_ROOT    = Path(__file__).resolve().parents[1]
EXP_DIR      = Path(__file__).resolve().parent
E7_OUT       = EXP_DIR / "results" / "gat_attn_graphs" / "e7"
E8_OUT       = EXP_DIR / "results" / "e8_combined_generalization"
E8_CACHE     = E8_OUT / "cache"
DATA_VCC2020 = REPO_ROOT / "data" / "vcc2020"
DATA_MLAAD   = REPO_ROOT / "data" / "mlaad_en"
VOCAB_DIR    = REPO_ROOT / "vocab_phoneme"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

for d in [E8_OUT, E8_CACHE, DATA_VCC2020, DATA_MLAAD]:
    d.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Constants (verbatim from E7 where shared) ─────────────────────────────────
ROUTING_SYSTEMS = {"A01", "A03", "A04"}   # class 0: TTS / routing-dependent
SKIP_SYSTEMS    = {"A05", "A06"}           # class 1: VC / skip-route
INCLUDE_SYSTEMS = ROUTING_SYSTEMS | SKIP_SYSTEMS

FEATURE_NAMES = [
    "gini_in", "gini_out", "entropy_in", "entropy_out",
    "spectral_gap", "top1_mass", "top5_mass", "diag_mass",
    "offdiag_frob", "eff_rank",
]
CONCENTRATION_FEATURES = {"gini_in", "gini_out", "top1_mass", "top5_mass"}

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR       # 48000 (3 seconds)
NF_PER_SAMPLE  = 3 * TARGET_SR // 320 - 1   # 149 frames
MIN_NODES      = 3
BATCH_SIZE     = 4                   # smaller batch for E8 (mixed-SR audio)
SEED           = 42

LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]

# E8-specific
N_MLAAD       = 100
N_VCC2020     = 100
E8_SEED       = 42   # for sample selection and length matching
N_BOOT        = 1000
PHONEME_MATCH_THRESH = 0.20   # >20% median difference → flag + subsample

# ── E7 structural feature functions (verbatim) ────────────────────────────────

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
    """10 permutation-invariant features of a row-stochastic A_agg. Verbatim from E7."""
    N = A_agg.shape[0]
    d_in  = A_agg.sum(axis=0)
    d_out = A_agg.sum(axis=1)

    f1 = _gini(d_in)
    f2 = _gini(d_out)
    f3 = _entropy(d_in)
    f4 = _entropy(d_out)

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


# ── Vocabulary (verbatim from E7) ─────────────────────────────────────────────

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


# ── Audio utilities (verbatim from E7) ───────────────────────────────────────

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

def load_audio_file(path: str | Path) -> torch.Tensor:
    """Load wav file, resample to 16kHz, crop/pad to 3 s. Returns (1, 48000)."""
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    w = torch.tensor(arr)
    if w.ndim == 1: w = w.unsqueeze(0)
    elif w.ndim == 2: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = T.Resample(sr, TARGET_SR)(w)
    return _crop(w)


# ── Model loading (verbatim from E7) ─────────────────────────────────────────

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


# ── Extraction (verbatim from E7) ─────────────────────────────────────────────

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


# ── New E8: File-list dataset ─────────────────────────────────────────────────

class FileListDataset(torch.utils.data.Dataset):
    """Dataset backed by a list of (path, label, system_id, dataset_src) tuples."""
    def __init__(self, file_list: list[tuple]):
        self.files = file_list  # (path, label, system_id, dataset_src)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path, label, system_id, dataset_src = self.files[idx]
        w = load_audio_file(path)
        return {"audio":      w,
                "label":      torch.tensor(label, dtype=torch.long),
                "system_id":  system_id,
                "dataset_src": dataset_src,
                "path":       str(path)}


def collate_e8(batch):
    return {"audio":       torch.stack([b["audio"]  for b in batch]),
            "label":       torch.stack([b["label"]  for b in batch]),
            "system_id":   [b["system_id"]   for b in batch],
            "dataset_src": [b["dataset_src"] for b in batch],
            "path":        [b["path"]        for b in batch]}


# ── New E8: Extraction (wraps E7 extract_all_layers logic) ────────────────────

def extract_all_layers_e8(lit, loader, device):
    """Extract per-sample GAT attention records for E8 file-list dataset.
    Stores extra fields: dataset_src, path. Core logic verbatim from E7."""
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
                audio      = batch["audio"].to(device)
                labels     = batch["label"].tolist()
                sys_ids    = batch["system_id"]
                ds_srcs    = batch["dataset_src"]
                paths      = batch["path"]
                B          = len(labels)
                num_f      = torch.full((B,), NF_PER_SAMPLE, device=device)
                hs, pids   = run_frozen_frontend(audio, gm, device)
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
                    rec = {"sample_id":  base_id + i,
                           "label":      labels[i],
                           "system_id":  sys_ids[i],
                           "dataset_src": ds_srcs[i],
                           "path":       paths[i],
                           "node_phoneme_ids": pids_i.clone(),
                           "edge_index":  ei_local.clone(),
                           "n_nodes":    n_nodes_i,
                           "is_degenerate": is_deg}
                    for li in range(n_layers):
                        rec[f"attn_l{li}"] = attn_all[li][edge_mask].clone()
                    records.append(rec)

                for li in range(n_layers):
                    buf[f"attn_{li}"] = None; buf[f"eidx_{li}"] = None

                done = min((bi + 1) * BATCH_SIZE, total)
                if (bi + 1) % 5 == 0 or done == total:
                    print(f"  {done}/{total}")
    finally:
        for h in hooks: h.remove()

    print(f"  Extracted {len(records)} samples ({n_degenerate} degenerate)")
    return records


# ── New E8: LR pipeline (verbatim from E7) ───────────────────────────────────

def make_lr_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("thresh", VarianceThreshold(threshold=1e-10)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])


# ── New E8: Train E7 classifier on full seed-1 ASVspoof data ──────────────────

def train_e7_classifier(e7_cache_path: Path) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Load E7 seed-1 records, train LR on all 250 ASVspoof samples (no CV).
    Returns (trained_pipeline, X_train, y_train).
    """
    print(f"\n  Loading E7 seed-1 records from: {e7_cache_path.name}")
    records = torch.load(str(e7_cache_path))
    valid   = [r for r in records
               if r["system_id"] in INCLUDE_SYSTEMS and not r["is_degenerate"]]
    valid.sort(key=lambda r: (r["system_id"], r["sample_id"]))

    X = np.array([compute_structural_features(build_agg_graph(r)) for r in valid],
                 dtype=np.float64)
    y = np.array([0 if r["system_id"] in ROUTING_SYSTEMS else 1 for r in valid],
                 dtype=np.int64)

    counts = Counter(r["system_id"] for r in valid)
    print(f"  Training set: N={len(valid)} (A01={counts['A01']}, A03={counts['A03']}, "
          f"A04={counts['A04']}, A05={counts['A05']}, A06={counts['A06']})")

    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    print(f"  LR trained. Class 0 (TTS/routing): {(y==0).sum()}, "
          f"Class 1 (VC/skip): {(y==1).sum()}")
    return pipe, X, y


# ── New E8: VCC2020 download ──────────────────────────────────────────────────

def _progress_hook(count, block_size, total_size):
    pct = min(100.0, count * block_size * 100.0 / total_size) if total_size > 0 else 0
    mb  = count * block_size / 1e6
    tot = total_size / 1e6
    print(f"\r    {pct:5.1f}%  {mb:.0f}/{tot:.0f} MB", end="", flush=True)


def download_vcc2020(dest: Path, force: bool = False) -> dict:
    """Download VCC2020 listening test audio (643 MB) from Zenodo 4433173.
    Extracts task1 (intra-lingual English) wav files only.
    Returns dict: team_id → list of local wav paths.

    NOTE: Record 4433173 is the correct audio record. The DOI 10.5281/zenodo.4345998
    referenced in the experiment spec points to a metadata-only placeholder (24 KB);
    the actual audio is at zenodo.org/record/4433173.
    """
    manifest_path = dest / "manifest_task1.json"
    if manifest_path.exists() and not force:
        with open(manifest_path) as f:
            return json.load(f)

    zip_path = dest / "VCC2020-listeningtest-v1.0.1.zip"
    if not zip_path.exists() or force:
        url = ("https://zenodo.org/api/records/4433173/files/"
               "nii-yamagishilab/VCC2020-listeningtest-v1.0.1.zip/content")
        print(f"  Downloading VCC2020 listening test data (~642 MB) from Zenodo...")
        urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress_hook)
        print()

    print(f"  Extracting task1 wav files...")
    audio_dir = dest / "audio"
    audio_dir.mkdir(exist_ok=True)
    system_files: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as outer:
        inner_names = outer.namelist()
        team_zips   = sorted(n for n in inner_names
                             if re.search(r'/(T\d+)\.zip$', n))

        for tz_name in team_zips:
            m       = re.search(r'/(T\d+)\.zip$', tz_name)
            team_id = m.group(1)
            tz_bytes = outer.read(tz_name)

            with zipfile.ZipFile(_io.BytesIO(tz_bytes)) as inner:
                task1_wavs = [n for n in inner.namelist()
                              if '/task1/' in n and n.endswith('.wav')]

                for wav_name in task1_wavs:
                    rel     = wav_name.replace(f"{team_id}/task1/", "")
                    out_dir = audio_dir / team_id / "task1"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / rel
                    if not out_path.exists():
                        out_path.write_bytes(inner.read(wav_name))
                    system_files[team_id].append(str(out_path))

            n_files = len(system_files[team_id])
            print(f"    {team_id}: {n_files} task1 wavs")

    manifest = dict(system_files)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"  VCC2020 manifest saved: {manifest_path}")
    return manifest


# ── New E8: MLAAD download ─────────────────────────────────────────────────────

def download_mlaad_en(dest: Path, token: str, force: bool = False,
                      min_per_system: int = 2) -> dict:
    """Download English subset of MLAAD-tiny from HuggingFace.
    Lists all English files, selects 2–3 per system for diversity,
    downloads only selected files.
    Returns dict: system_id → list of local wav paths.

    Dataset: mueller91/MLAAD-tiny (CC-BY-NC-4.0, non-commercial research use)
    Note: MLAAD-tiny is not gated; the full MLAAD (mueller91/MLAAD) is gated.
    """
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and not force:
        with open(manifest_path) as f:
            return json.load(f)

    from huggingface_hub import HfApi, hf_hub_download

    print("  Listing MLAAD-tiny English files from HuggingFace...")
    api        = HfApi(token=token)
    all_files  = list(api.list_repo_files("mueller91/MLAAD-tiny", repo_type="dataset"))
    # wav files only; exclude non-audio metadata files
    en_files   = [f for f in all_files
                  if f.startswith("fake/en/") and f.endswith(".wav")]

    # Group by system
    by_system: dict[str, list[str]] = defaultdict(list)
    for hf_path in en_files:
        parts = hf_path.split("/")
        if len(parts) >= 3:
            by_system[parts[2]].append(hf_path)

    print(f"  Found {len(by_system)} English TTS systems, "
          f"{len(en_files)} total files.")

    # Select min_per_system per system for download
    rng = random.Random(E8_SEED)
    selected_hf: dict[str, list[str]] = {}
    for sys_id, paths in sorted(by_system.items()):
        shuffled = sorted(paths); rng.shuffle(shuffled)
        selected_hf[sys_id] = shuffled[:min_per_system]

    total_dl = sum(len(v) for v in selected_hf.values())
    print(f"  Downloading {total_dl} files ({min_per_system}/system × {len(by_system)} systems)...")

    system_local: dict[str, list[str]] = defaultdict(list)
    for i, (sys_id, hf_paths) in enumerate(sorted(selected_hf.items())):
        sys_dir = dest / sys_id
        sys_dir.mkdir(exist_ok=True)
        for hf_path in hf_paths:
            fname    = Path(hf_path).name
            out_path = sys_dir / fname
            if not out_path.exists():
                dl_path = hf_hub_download(
                    repo_id="mueller91/MLAAD-tiny",
                    filename=hf_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(dest / "_hf_cache"),
                    local_dir_use_symlinks=False,
                )
                import shutil
                shutil.copy2(dl_path, out_path)
            system_local[sys_id].append(str(out_path))
        if (i + 1) % 10 == 0 or (i + 1) == len(selected_hf):
            print(f"    {i+1}/{len(selected_hf)} systems done")

    manifest = dict(system_local)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"  MLAAD manifest saved: {manifest_path}")
    return manifest


# ── New E8: Sample selection + length matching ────────────────────────────────

def balanced_sample(system_files: dict[str, list[str]], n_total: int,
                    seed: int = E8_SEED) -> list[tuple[str, str]]:
    """Select n_total files balanced across systems. Returns list of (path, system_id)."""
    rng  = random.Random(seed)
    systems = sorted(system_files.keys())
    n_sys   = len(systems)
    base    = n_total // n_sys
    extra   = n_total % n_sys

    result = []
    for i, sys_id in enumerate(systems):
        paths    = list(system_files[sys_id])
        rng.shuffle(paths)
        n_pick   = base + (1 if i < extra else 0)
        picked   = paths[:n_pick]
        result.extend((p, sys_id) for p in picked)

    rng.shuffle(result)
    return result[:n_total]   # exact n_total


def check_phoneme_distributions(mlaad_recs: list, vcc2020_recs: list,
                                 e7_recs: list) -> tuple[bool, str]:
    """Compare n_nodes (phoneme count) distributions across the three sources.
    Returns (ok, message). Flags if MLAAD or VCC2020 median differs >20% from ASVspoof.
    """
    def _counts(recs):
        return [r["n_nodes"] for r in recs if not r["is_degenerate"]]

    asvspoof_counts = _counts(e7_recs)
    mlaad_counts    = _counts(mlaad_recs)
    vcc2020_counts  = _counts(vcc2020_recs)

    med_asv  = np.median(asvspoof_counts)
    med_mla  = np.median(mlaad_counts)   if mlaad_counts  else np.nan
    med_vcc  = np.median(vcc2020_counts) if vcc2020_counts else np.nan

    rel_mla  = abs(med_mla - med_asv) / (med_asv + 1e-12)
    rel_vcc  = abs(med_vcc - med_asv) / (med_asv + 1e-12)

    mla_ok   = rel_mla <= PHONEME_MATCH_THRESH
    vcc_ok   = rel_vcc <= PHONEME_MATCH_THRESH
    ok       = mla_ok and vcc_ok

    msg_parts = [
        f"ASVspoof: median={med_asv:.1f}  ({len(asvspoof_counts)} samples)",
        f"MLAAD:    median={med_mla:.1f}  rel_diff={rel_mla:.1%}  {'OK' if mla_ok else '⚠ EXCEEDS 20%'}",
        f"VCC2020:  median={med_vcc:.1f}  rel_diff={rel_vcc:.1%}  {'OK' if vcc_ok else '⚠ EXCEEDS 20%'}",
    ]
    return ok, "\n".join(msg_parts), (asvspoof_counts, mlaad_counts, vcc2020_counts)


def length_match_by_subsample(mlaad_sample: list[tuple], vcc2020_sample: list[tuple],
                               mlaad_recs: list, vcc2020_recs: list,
                               seed: int = E8_SEED) -> tuple[list, list, list, list]:
    """If MLAAD and VCC2020 median phoneme counts differ >20%, subsample the
    longer-tailed distribution to better match.
    Returns (mlaad_sample_out, vcc2020_sample_out, mlaad_recs_out, vcc2020_recs_out).
    """
    # Build n_nodes lookup by path
    def _nc_map(recs):
        return {r["path"]: r["n_nodes"] for r in recs if not r["is_degenerate"]}

    mla_nc = _nc_map(mlaad_recs)
    vcc_nc = _nc_map(vcc2020_recs)

    mla_valid = [(p, s) for p, s in mlaad_sample   if p in mla_nc]
    vcc_valid = [(p, s) for p, s in vcc2020_sample  if p in vcc_nc]

    med_mla = np.median([mla_nc[p] for p, _ in mla_valid])
    med_vcc = np.median([vcc_nc[p] for p, _ in vcc_valid])

    rel_diff = abs(med_mla - med_vcc) / (max(med_mla, med_vcc) + 1e-12)
    if rel_diff <= PHONEME_MATCH_THRESH:
        # No subsampling needed
        mla_recs_out = [r for r in mlaad_recs   if r["path"] in {p for p, _ in mla_valid}]
        vcc_recs_out = [r for r in vcc2020_recs  if r["path"] in {p for p, _ in vcc_valid}]
        return mla_valid, vcc_valid, mla_recs_out, vcc_recs_out

    # Subsample the longer-tail set to match the shorter-tail median
    print(f"  Length mismatch: MLAAD median={med_mla:.1f}, VCC2020 median={med_vcc:.1f}")
    print(f"  Subsampling to match distributions (seed={seed})...")
    rng = random.Random(seed)

    target_med = min(med_mla, med_vcc)
    n_target   = min(len(mla_valid), len(vcc_valid))

    def _subsample_to_target(sample, nc_map, target_n):
        # Keep samples with n_nodes ≤ 2× target_med (remove outliers on high end)
        filtered = [(p, s) for p, s in sample if nc_map[p] <= 2 * target_med]
        rng.shuffle(filtered)
        return filtered[:target_n]

    if med_mla > med_vcc:
        mla_valid = _subsample_to_target(mla_valid, mla_nc, n_target)
    else:
        vcc_valid = _subsample_to_target(vcc_valid, vcc_nc, n_target)

    mla_paths = {p for p, _ in mla_valid}
    vcc_paths = {p for p, _ in vcc_valid}
    mla_recs_out = [r for r in mlaad_recs  if r["path"] in mla_paths]
    vcc_recs_out = [r for r in vcc2020_recs if r["path"] in vcc_paths]

    print(f"  After subsampling: MLAAD={len(mla_valid)}, VCC2020={len(vcc_valid)}")
    return mla_valid, vcc_valid, mla_recs_out, vcc_recs_out


def histogram_match_subsample(mlaad_sample: list[tuple], vcc2020_sample: list[tuple],
                               mlaad_recs: list, vcc2020_recs: list,
                               bin_width: int = 5,
                               seed: int = E8_SEED) -> tuple[list, list, list, list]:
    """Bin-based histogram matching: for each n_nodes bin keep min(n_mla, n_vcc)
    samples from each dataset, giving identical marginal distributions.
    Returns (mlaad_sample_out, vcc2020_sample_out, mlaad_recs_out, vcc2020_recs_out).
    """
    def _nc_map(recs):
        return {r["path"]: r["n_nodes"] for r in recs if not r["is_degenerate"]}

    mla_nc = _nc_map(mlaad_recs)
    vcc_nc = _nc_map(vcc2020_recs)

    mla_valid = [(p, s) for p, s in mlaad_sample  if p in mla_nc]
    vcc_valid = [(p, s) for p, s in vcc2020_sample if p in vcc_nc]

    # Determine overlapping range
    all_nc = [mla_nc[p] for p, _ in mla_valid] + [vcc_nc[p] for p, _ in vcc_valid]
    lo = int(np.percentile(all_nc, 5))
    hi = int(np.percentile(all_nc, 95))
    bins = range(lo, hi + bin_width, bin_width)

    rng = random.Random(seed)

    def _bin_idx(nc): return (nc - lo) // bin_width

    # Group each dataset by bin
    mla_bins: dict[int, list] = {}
    for p, s in mla_valid:
        nc = mla_nc[p]
        if lo <= nc <= hi:
            b = _bin_idx(nc)
            mla_bins.setdefault(b, []).append((p, s))

    vcc_bins: dict[int, list] = {}
    for p, s in vcc_valid:
        nc = vcc_nc[p]
        if lo <= nc <= hi:
            b = _bin_idx(nc)
            vcc_bins.setdefault(b, []).append((p, s))

    all_bins = sorted(set(mla_bins) | set(vcc_bins))
    mla_out, vcc_out = [], []
    for b in all_bins:
        mla_b = mla_bins.get(b, [])
        vcc_b = vcc_bins.get(b, [])
        keep  = min(len(mla_b), len(vcc_b))
        if keep == 0:
            continue
        rng.shuffle(mla_b); rng.shuffle(vcc_b)
        mla_out.extend(mla_b[:keep])
        vcc_out.extend(vcc_b[:keep])

    mla_paths = {p for p, _ in mla_out}
    vcc_paths = {p for p, _ in vcc_out}
    mla_recs_out = [r for r in mlaad_recs  if r["path"] in mla_paths]
    vcc_recs_out = [r for r in vcc2020_recs if r["path"] in vcc_paths]

    med_mla_after = np.median([mla_nc[p] for p, _ in mla_out]) if mla_out else float("nan")
    med_vcc_after = np.median([vcc_nc[p] for p, _ in vcc_out]) if vcc_out else float("nan")
    print(f"  Histogram-matched: MLAAD={len(mla_out)} (median n_nodes={med_mla_after:.1f}), "
          f"VCC2020={len(vcc_out)} (median n_nodes={med_vcc_after:.1f})")
    print(f"  n_nodes range kept: [{lo}, {hi}]  bin_width={bin_width}")
    return mla_out, vcc_out, mla_recs_out, vcc_recs_out


# ── New E8: Feature matrix from records ───────────────────────────────────────

def build_e8_feature_matrix(records: list) -> tuple[np.ndarray, np.ndarray,
                                                     list[str], list[str], list[str]]:
    """Extract structural features from E8 records.
    Returns (X, y, system_ids, dataset_srcs, paths).
    """
    rows = []
    for rec in records:
        if rec["is_degenerate"]: continue
        A_agg = build_agg_graph(rec)
        feat  = compute_structural_features(A_agg)
        rows.append((rec["system_id"], rec["dataset_src"], rec["path"],
                     feat, rec["label"]))

    X    = np.array([r[3] for r in rows], dtype=np.float64)
    y    = np.array([r[4] for r in rows], dtype=np.int64)
    sids = [r[0] for r in rows]
    dsrc = [r[1] for r in rows]
    paths = [r[2] for r in rows]
    return X, y, sids, dsrc, paths


# ── New E8: Plotting ──────────────────────────────────────────────────────────

def plot_phoneme_distributions(asv_counts, mla_counts, vcc_counts,
                                out_path: Path):
    """Three-histogram overlay: ASVspoof eval vs MLAAD subsample vs VCC2020 subsample."""
    fig, ax = plt.subplots(figsize=(8, 4))
    kw = dict(alpha=0.55, bins=30, density=True)
    if asv_counts: ax.hist(asv_counts, **kw, color="steelblue",  label=f"ASVspoof eval (N={len(asv_counts)})")
    if mla_counts: ax.hist(mla_counts, **kw, color="darkorange", label=f"MLAAD EN (N={len(mla_counts)})")
    if vcc_counts: ax.hist(vcc_counts, **kw, color="seagreen",   label=f"VCC2020 task1 (N={len(vcc_counts)})")
    ax.set_xlabel("Phoneme-graph node count (n_nodes)")
    ax.set_ylabel("Density")
    ax.set_title("E8: Phoneme-count distributions — pipeline mismatch detector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_2d_scatter(e7_records: list, e8_records: list, out_path: Path):
    """offdiag_frob vs gini_in: 4 point types overlaid.
    Training TTS (A01/A03/A04), training VC (A05/A06),
    test TTS (MLAAD), test VC (VCC2020).
    """
    fi_gini    = FEATURE_NAMES.index("gini_in")
    fi_frob    = FEATURE_NAMES.index("offdiag_frob")

    def _feats(recs, filter_fn):
        out = []
        for r in recs:
            if filter_fn(r) and not r["is_degenerate"]:
                A = build_agg_graph(r)
                f = compute_structural_features(A)
                out.append(f)
        return np.array(out, dtype=np.float64) if out else np.zeros((0, 10))

    tts_train = _feats(e7_records, lambda r: r["system_id"] in ROUTING_SYSTEMS)
    vc_train  = _feats(e7_records, lambda r: r["system_id"] in SKIP_SYSTEMS)
    tts_test  = _feats(e8_records, lambda r: r["dataset_src"] == "MLAAD" and not r["is_degenerate"])
    vc_test   = _feats(e8_records, lambda r: r["dataset_src"] == "VCC2020" and not r["is_degenerate"])

    fig, ax = plt.subplots(figsize=(7, 5))
    kw_train = dict(alpha=0.45, s=20, zorder=2)
    kw_test  = dict(alpha=0.75, s=40, zorder=3, edgecolors="k", linewidths=0.4)

    if len(tts_train): ax.scatter(tts_train[:, fi_frob], tts_train[:, fi_gini],
                                   color="steelblue",  marker="o", label="Train TTS (A01/A03/A04)", **kw_train)
    if len(vc_train):  ax.scatter(vc_train[:, fi_frob],  vc_train[:, fi_gini],
                                   color="firebrick",  marker="o", label="Train VC (A05/A06)",      **kw_train)
    if len(tts_test):  ax.scatter(tts_test[:, fi_frob],  tts_test[:, fi_gini],
                                   color="darkorange", marker="^", label="Test TTS (MLAAD EN)",     **kw_test)
    if len(vc_test):   ax.scatter(vc_test[:, fi_frob],   vc_test[:, fi_gini],
                                   color="seagreen",   marker="s", label="Test VC (VCC2020 T1)",   **kw_test)

    ax.set_xlabel("offdiag_frob")
    ax.set_ylabel("gini_in")
    ax.set_title("E8: Structural feature space — training vs. unseen systems")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["TTS (pred)", "VC (pred)"])
    ax.set_yticklabels(["TTS (true)", "VC (true)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("E8: Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── New E8: Bootstrap CI ──────────────────────────────────────────────────────

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = E8_SEED):
    """Returns dict with 95% bootstrap CIs for bal_acc, recall_tts, recall_vc, auc."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    metrics = {"bal_acc": [], "recall_tts": [], "recall_vc": [], "auc": []}

    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        yt   = y_true[idx]; yp = y_pred[idx]; ypr = y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        metrics["bal_acc"].append(balanced_accuracy_score(yt, yp))
        prec, rec, _, _ = precision_recall_fscore_support(yt, yp, average=None,
                                                            zero_division=0)
        metrics["recall_tts"].append(rec[0])
        metrics["recall_vc"].append(rec[1])
        try: metrics["auc"].append(roc_auc_score(yt, ypr))
        except: pass

    ci = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


# ── New E8: Hypothesis evaluation ─────────────────────────────────────────────

def evaluate_hypotheses_e8(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray, sids: list, dsrc: list,
                            e7_lr: Pipeline, phon_counts: np.ndarray,
                            e7_records: list, out_dir: Path) -> dict:
    """Evaluate H1–H5 for E8. Returns results dict."""
    print("\n" + "=" * 70)
    print("PRE-REGISTERED HYPOTHESIS OUTCOMES  (E8)")
    print("=" * 70)
    results = {}

    bal_acc = balanced_accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average=None,
                                                         zero_division=0)
    try: auc = roc_auc_score(y, y_prob)
    except: auc = float("nan")
    ci = bootstrap_ci(y, y_pred, y_prob)
    recall_tts = rec[0]; recall_vc = rec[1]

    # ── H1: balanced accuracy ≥ 0.70 ─────────────────────────────────────────
    h1_pass = bal_acc >= 0.70
    h1_fals = bal_acc <= 0.55
    print(f"\nH1 [primary]: Balanced accuracy ≥ 0.70 on combined 200-sample set")
    print(f"  bal_acc = {bal_acc:.4f}  95% CI [{ci['bal_acc'][0]:.4f}, {ci['bal_acc'][1]:.4f}]")
    print(f"  AUC     = {auc:.4f}")
    print(f"  Threshold: ≥ 0.70 {'✓' if h1_pass else '✗'}  "
          f"Falsification (≤ 0.55): {'✗ (falsified)' if h1_fals else 'not falsified'}")
    print(f"  Result: {'PASS ✓' if h1_pass else 'FAIL ✗'}")
    results["H1"] = {"pass": h1_pass, "falsified": h1_fals,
                     "bal_acc": bal_acc, "auc": auc, "ci_bal_acc": ci["bal_acc"]}

    # ── H2: both recalls ≥ 0.65 ───────────────────────────────────────────────
    h2_pass = (recall_tts >= 0.65) and (recall_vc >= 0.65)
    h2_fals = (recall_tts <= 0.50) or (recall_vc <= 0.50)
    print(f"\nH2 [primary]: Both per-class recalls ≥ 0.65")
    print(f"  MLAAD (TTS) recall  = {recall_tts:.4f}  CI [{ci['recall_tts'][0]:.4f}, {ci['recall_tts'][1]:.4f}]  "
          f"{'≥ 0.65 ✓' if recall_tts >= 0.65 else '< 0.65 ✗'}")
    print(f"  VCC2020 (VC) recall = {recall_vc:.4f}  CI [{ci['recall_vc'][0]:.4f}, {ci['recall_vc'][1]:.4f}]  "
          f"{'≥ 0.65 ✓' if recall_vc >= 0.65 else '< 0.65 ✗'}")
    print(f"  Falsification (either ≤ 0.50): {'✗ (falsified)' if h2_fals else 'not falsified'}")
    print(f"  Result: {'PASS ✓' if h2_pass else 'FAIL ✗'}")
    results["H2"] = {"pass": h2_pass, "falsified": h2_fals,
                     "recall_tts": recall_tts, "recall_vc": recall_vc}

    # ── H3: Gini coefficient distribution comparison ──────────────────────────
    fi_gini = FEATURE_NAMES.index("gini_in")
    tts_train_gini = np.mean([compute_structural_features(build_agg_graph(r))[fi_gini]
                              for r in e7_records
                              if r["system_id"] in ROUTING_SYSTEMS and not r["is_degenerate"]])
    vc_train_gini  = np.mean([compute_structural_features(build_agg_graph(r))[fi_gini]
                              for r in e7_records
                              if r["system_id"] in SKIP_SYSTEMS and not r["is_degenerate"]])

    e8_is_tts = np.array([d == "MLAAD"  for d in dsrc])
    e8_is_vc  = np.array([d == "VCC2020" for d in dsrc])
    mla_gini  = np.mean(X[e8_is_tts, fi_gini]) if e8_is_tts.any() else np.nan
    vcc_gini  = np.mean(X[e8_is_vc,  fi_gini]) if e8_is_vc.any()  else np.nan

    h3_tts_ok = abs(mla_gini - tts_train_gini) < abs(mla_gini - vc_train_gini)
    h3_vc_ok  = abs(vcc_gini - vc_train_gini)  < abs(vcc_gini - tts_train_gini)
    h3_pass   = h3_tts_ok and h3_vc_ok
    print(f"\nH3 [secondary]: Gini coefficient distribution alignment")
    print(f"  Training TTS gini_in = {tts_train_gini:.4f}  Training VC = {vc_train_gini:.4f}")
    print(f"  MLAAD gini_in = {mla_gini:.4f}  "
          f"→ closer to {'TTS ✓' if h3_tts_ok else 'VC ✗'} training")
    print(f"  VCC2020 gini_in = {vcc_gini:.4f}  "
          f"→ closer to {'VC ✓' if h3_vc_ok else 'TTS ✗'} training")
    print(f"  Result: {'PASS ✓' if h3_pass else 'FAIL ✗'} (informational)")
    results["H3"] = {"pass": h3_pass, "mla_gini": mla_gini, "vcc_gini": vcc_gini,
                     "tts_train_gini": tts_train_gini, "vc_train_gini": vc_train_gini}

    # ── H4: per-system breakdown ───────────────────────────────────────────────
    sys_results: dict[str, dict] = {}
    for sys_id in sorted(set(sids)):
        mask    = np.array([s == sys_id for s in sids])
        n       = mask.sum()
        if n < 10: continue
        n_corr  = (y[mask] == y_pred[mask]).sum()
        pct     = n_corr / n
        ds      = dsrc[next(i for i, s in enumerate(sids) if s == sys_id)]
        sys_results[sys_id] = {"n": int(n), "n_correct": int(n_corr),
                                "pct_correct": float(pct), "dataset": ds}

    h4_pass = all(v["pct_correct"] >= 0.60 for v in sys_results.values())
    print(f"\nH4 [secondary]: Per-system breakdown (systems with ≥ 10 samples)")
    print(f"  {'system':30s}  {'dataset':8s}  {'N':>4s}  {'correct':>7s}  {'% corr':>7s}  {'≥60%':>5s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*5}")
    for sys_id, v in sys_results.items():
        ok = "✓" if v["pct_correct"] >= 0.60 else "✗"
        print(f"  {sys_id:30s}  {v['dataset']:8s}  {v['n']:4d}  {v['n_correct']:7d}  "
              f"{v['pct_correct']:7.3f}  {ok:>5s}")
    if not sys_results:
        print("  (no system with ≥ 10 samples; H4 informational only)")
    print(f"  Result: {'PASS ✓' if h4_pass else 'FAIL ✗'} (informational)")
    results["H4"] = {"pass": h4_pass, "per_system": sys_results}

    # ── H5: phoneme-count confound check ──────────────────────────────────────
    print(f"\nH5 [confound check]: Phoneme-count-only logistic regression baseline")
    X_phon = phon_counts.reshape(-1, 1).astype(np.float64)
    phon_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
    ])
    phon_pipe.fit(X_phon, y)
    phon_pred = phon_pipe.predict(X_phon)
    phon_ba   = balanced_accuracy_score(y, phon_pred)
    gap       = bal_acc - phon_ba
    h5_fals   = gap < 0.05   # structural within 0.05 of phoneme-count-only → confounded
    print(f"  Structural classifier bal_acc   = {bal_acc:.4f}")
    print(f"  Phoneme-count baseline bal_acc  = {phon_ba:.4f}")
    print(f"  Gap = {gap:+.4f}  Falsification threshold: gap < 0.05")
    if h5_fals:
        print(f"  ⚠ H5 FALSIFIED: result is length-confounded. Re-run with "
              f"length-matched subsampling.")
    else:
        print(f"  H5 not falsified: structural features capture more than length.")
    results["H5"] = {"falsified": h5_fals, "structural_ba": bal_acc,
                     "phon_ba": phon_ba, "gap": gap}

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'-'*70}")
    for h, res in results.items():
        if h == "H5":
            status = "FALSIFIED ✗" if res["falsified"] else "NOT FALSIFIED ✓"
        else:
            status = "PASS ✓" if res["pass"] else "FAIL ✗"
        print(f"  {h}: {status}")

    # Decision logic
    h1 = results["H1"]["pass"]
    h2 = results["H2"]["pass"]
    h5 = results["H5"]["falsified"]
    if h5:
        verdict = ("H5 FALSIFIED → re-run with length-matched subsampling. "
                   "If still positive, claim survives.")
    elif h1 and h2:
        verdict = ("H1 + H2 PASS + H5 not falsified → cross-system generalization "
                   "holds. Structural mechanism-class distinction survives both TTS "
                   "and VC shifts across datasets. Major addition to the paper.")
    elif h1 and not h2:
        dir_fail = "VCC2020 (VC)" if results["H2"]["recall_vc"] < 0.65 else "MLAAD (TTS)"
        verdict = (f"H1 PASS, H2 FAIL → asymmetric generalization. Classifier handles "
                   f"{dir_fail} less well than the other class. "
                   f"Report direction explicitly; investigate whether failing class "
                   f"shares architectural properties with its training analog.")
    else:
        verdict = ("H1 FAIL → taxonomy does not generalise cross-dataset. "
                   "Original E7 result stands as within-protocol; broader "
                   "mechanism-class claim does not survive. Report as honest negative.")

    print(f"\n  Decision: {verdict}")
    print(f"{'='*70}")
    results["verdict"] = verdict
    return results


# ── New E8: Sanity check ──────────────────────────────────────────────────────

def print_sanity_check(rec: dict, name: str, e7_X: np.ndarray):
    """Print A_agg row sums and 10 structural features for one sample.
    Compare features to E7 training distribution.
    """
    A_agg = build_agg_graph(rec)
    row_sums = A_agg.sum(axis=1)
    feat     = compute_structural_features(A_agg)

    print(f"\n  [{name}] system={rec['system_id']}  n_nodes={rec['n_nodes']}")
    print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}, "
          f"mean={row_sums.mean():.6f}  (expected: 1.000000 ± 1e-6)")
    rowsum_ok = np.allclose(row_sums, 1.0, atol=1e-4)
    print(f"  Row-stochastic check: {'PASS ✓' if rowsum_ok else 'FAIL ✗ (pipeline issue!)'}")

    e7_mean = e7_X.mean(axis=0)
    e7_std  = e7_X.std(axis=0)
    print(f"\n  {'feature':16s}  {'value':>10s}  {'E7 mean':>10s}  {'E7 std':>8s}  {'z-score':>8s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    flag = False
    for i, fname in enumerate(FEATURE_NAMES):
        v = feat[i]
        m = e7_mean[i]; s = e7_std[i]
        z = (v - m) / (s + 1e-12)
        warn = " ← OUTLIER" if abs(z) > 4 else ""
        if warn: flag = True
        print(f"  {fname:16s}  {v:10.6f}  {m:10.6f}  {s:8.6f}  {z:8.3f}{warn}")
    if flag:
        print(f"\n  ⚠ OUTLIER FEATURES DETECTED — pipeline mismatch likely. STOP.")
    else:
        print(f"\n  All features within ±4σ of E7 training distribution. OK ✓")
    return feat, rowsum_ok and not flag


# ── New E8: Reporting ─────────────────────────────────────────────────────────

def save_manifest_csv(file_list: list, records: list, out_path: Path):
    """Save per-sample manifest CSV."""
    # Build n_nodes lookup
    nc_map = {r["path"]: r["n_nodes"] for r in records}
    dur_map = {}
    for path, sys_id in file_list:
        try:
            arr, sr = sf.read(str(path), dtype="float32")
            dur_map[path] = len(arr) / sr
        except: dur_map[path] = -1.0

    rows = []
    for path, sys_id in file_list:
        nc = nc_map.get(path, -1)
        gt = "TTS" if sys_id.startswith("T") is False else "VC"
        # Infer ground truth from file path
        if "/mlaad_en/" in path or "fake/en/" in path:
            gt_label = "TTS"; gt_class = 0
        else:
            gt_label = "VC"; gt_class = 1
        rows.append({"path": path, "system_id": sys_id,
                     "gt_label": gt_label, "gt_class": gt_class,
                     "n_nodes": nc, "duration_s": f"{dur_map.get(path, -1):.2f}"})

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved manifest: {out_path.name}")


def save_results_csv(y_true, y_pred, y_prob, sids, dsrc, paths, out_path: Path):
    rows = []
    for i in range(len(y_true)):
        rows.append({"path": paths[i], "system_id": sids[i],
                     "dataset_src": dsrc[i],
                     "gt_label": "TTS" if y_true[i] == 0 else "VC",
                     "pred_label": "TTS" if y_pred[i] == 0 else "VC",
                     "pred_prob_vc": f"{y_prob[i]:.6f}",
                     "correct": int(y_true[i] == y_pred[i])})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved per-sample results: {out_path.name}")


def save_metrics_md(results: dict, bal_acc: float, prec, rec, f1,
                    auc: float, ci: dict, out_path: Path):
    lines = [
        "# E8: Combined TTS+VC Generalization Test",
        "",
        "## Datasets",
        "- **MLAAD** (English subset, 64 TTS systems): mueller91/MLAAD-tiny",
        "  - License: CC-BY-NC-4.0",
        "  - Citation: Müller et al. 2024, arXiv:2401.09512",
        "- **VCC2020 Task 1** (intra-lingual English VC, T01–T33): Zenodo 4433173",
        "  - License: ODbL (audio)",
        "  - Citation: Yi et al. 2020, DOI 10.21437/VCC_BC.2020-14",
        "",
        "## Exclusion notes",
        "- MLAAD: English subset only (other languages confound phoneme-recognizer analysis).",
        "  Language identified via directory path `fake/en/{system}/`.",
        "- VCC2020: Task 1 (intra-lingual English) only. Task 2 (cross-lingual: Finnish/",
        "  German/Mandarin targets) excluded to avoid language confound.",
        "",
        "## Headline metrics (200-sample combined set)",
        "",
        f"| metric | value | 95% CI |",
        "|--------|-------|--------|",
        f"| Balanced accuracy | {bal_acc:.4f} | [{ci['bal_acc'][0]:.4f}, {ci['bal_acc'][1]:.4f}] |",
        f"| TTS recall (MLAAD) | {rec[0]:.4f} | [{ci['recall_tts'][0]:.4f}, {ci['recall_tts'][1]:.4f}] |",
        f"| VC recall (VCC2020) | {rec[1]:.4f} | [{ci['recall_vc'][0]:.4f}, {ci['recall_vc'][1]:.4f}] |",
        f"| TTS precision | {prec[0]:.4f} | - |",
        f"| VC precision | {prec[1]:.4f} | - |",
        f"| F1 TTS | {f1[0]:.4f} | - |",
        f"| F1 VC | {f1[1]:.4f} | - |",
        f"| AUC | {auc:.4f} | [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}] |",
        "",
        "## Hypothesis outcomes",
        "",
    ]
    for h, res in results.items():
        if h == "verdict": continue
        if h == "H5":
            status = "FALSIFIED" if res["falsified"] else "NOT FALSIFIED"
        else:
            status = "PASS" if res["pass"] else "FAIL"
        lines.append(f"- **{h}**: {status}")

    lines += ["", f"**Decision: {results.get('verdict', '')}**"]
    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/robust_goat.ckpt",
                        help="Seed-1 checkpoint (default: models/robust_goat.ckpt)")
    parser.add_argument("--run-classification", action="store_true",
                        help="Run full classification (only after committing pre-registration)")
    parser.add_argument("--length-match", action="store_true",
                        help="Force histogram-based n_nodes matching between MLAAD and VCC2020 "
                             "(H5 follow-up; implies --run-classification)")
    parser.add_argument("--force-download",  action="store_true",
                        help="Force re-download both datasets")
    parser.add_argument("--force-extract",   action="store_true",
                        help="Force re-extract features (ignore cache)")
    args = parser.parse_args()
    if args.length_match:
        args.run_classification = True

    np.random.seed(E8_SEED); random.seed(E8_SEED); torch.manual_seed(E8_SEED)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute(): ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}"); sys.exit(1)

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    # Length-match run writes to a separate directory to preserve original results
    run_out = (EXP_DIR / "results" / "e8_combined_generalization_lm"
               if args.length_match else E8_OUT)
    run_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== E8: Combined TTS+VC Generalization Test ===")
    if args.length_match:
        print(f"  Mode: LENGTH-MATCHED (H5 follow-up)")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Output: {run_out}/")

    # ── Stage 1: Download datasets ────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"STAGE 1: DATA DOWNLOAD")
    print(f"{'='*66}")

    print(f"\n[MLAAD] Downloading English subset from HuggingFace (mueller91/MLAAD-tiny)...")
    mlaad_manifest = download_mlaad_en(DATA_MLAAD, hf_token or "",
                                       force=args.force_download)

    print(f"\n[VCC2020] Downloading listening test data from Zenodo record 4433173...")
    vcc2020_manifest = download_vcc2020(DATA_VCC2020, force=args.force_download)

    # ── (a) Directory layouts ─────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("(a) DIRECTORY LAYOUTS AFTER DOWNLOAD")
    print(f"{'='*66}")

    print(f"\nMLAAD English subset:  {DATA_MLAAD}/")
    for sys_id in sorted(mlaad_manifest.keys())[:10]:
        n = len(mlaad_manifest[sys_id])
        print(f"  {sys_id}/  ({n} files)")
    if len(mlaad_manifest) > 10:
        print(f"  ... ({len(mlaad_manifest) - 10} more systems)")

    print(f"\nVCC2020 task1 audio:  {DATA_VCC2020}/audio/")
    for team_id in sorted(vcc2020_manifest.keys()):
        n = len(vcc2020_manifest[team_id])
        print(f"  {team_id}/task1/  ({n} files)")

    # ── (b) Per-system file counts ────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("(b) PER-SYSTEM FILE COUNTS")
    print(f"{'='*66}")

    print(f"\nMLAAD (TTS, class 0 / routing-dependent):")
    print(f"  Total systems: {len(mlaad_manifest)}")
    print(f"  Total files:   {sum(len(v) for v in mlaad_manifest.values())}")
    print(f"  Files/system (downloaded sample):  "
          f"min={min(len(v) for v in mlaad_manifest.values())}, "
          f"max={max(len(v) for v in mlaad_manifest.values())}")

    print(f"\nVCC2020 task1 (VC, class 1 / skip-route):")
    print(f"  Total systems (teams): {len(vcc2020_manifest)}")
    print(f"  Total task1 files:     {sum(len(v) for v in vcc2020_manifest.values())}")
    cnt_vcc = {t: len(v) for t, v in vcc2020_manifest.items()}
    print(f"  Files/team:  min={min(cnt_vcc.values())}, max={max(cnt_vcc.values())}, "
          f"mean={sum(cnt_vcc.values())/len(cnt_vcc.values()):.1f}")

    # ── Sample selection ──────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("SAMPLE SELECTION")
    print(f"{'='*66}")
    print(f"  E8_SEED = {E8_SEED} (reproducible)")

    mlaad_sample   = balanced_sample(mlaad_manifest,   N_MLAAD,   seed=E8_SEED)
    vcc2020_sample = balanced_sample(vcc2020_manifest, N_VCC2020, seed=E8_SEED)
    print(f"  Selected {len(mlaad_sample)} MLAAD samples across "
          f"{len({s for _, s in mlaad_sample})} systems")
    print(f"  Selected {len(vcc2020_sample)} VCC2020 samples across "
          f"{len({s for _, s in vcc2020_sample})} teams")

    # ── Model loading ─────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("MODEL LOADING & FEATURE EXTRACTION")
    print(f"{'='*66}")
    patch_phoneme_loader()
    print(f"  Loading model: {ckpt_path.name}...")
    lit = load_model(ckpt_path, device)
    gm  = lit.model

    # ── Extract features for all 200 samples ─────────────────────────────────
    mlaad_cache   = E8_CACHE / "records_mlaad.pt"
    vcc2020_cache = E8_CACHE / "records_vcc2020.pt"

    def _make_loader(sample, label):
        file_list = [(path, label, sys_id,
                      "MLAAD" if label == 0 else "VCC2020")
                     for path, sys_id in sample]
        ds = FileListDataset(file_list)
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_e8)

    if mlaad_cache.exists() and not args.force_extract:
        print(f"\n  Loading cached MLAAD records: {mlaad_cache.name}")
        mlaad_records = torch.load(str(mlaad_cache))
    else:
        print(f"\n  Extracting MLAAD ({len(mlaad_sample)} samples)...")
        loader = _make_loader(mlaad_sample, label=0)
        mlaad_records = extract_all_layers_e8(lit, loader, device)
        torch.save(mlaad_records, str(mlaad_cache))
        print(f"  Cached → {mlaad_cache.name}")

    if vcc2020_cache.exists() and not args.force_extract:
        print(f"\n  Loading cached VCC2020 records: {vcc2020_cache.name}")
        vcc2020_records = torch.load(str(vcc2020_cache))
    else:
        print(f"\n  Extracting VCC2020 ({len(vcc2020_sample)} samples)...")
        loader = _make_loader(vcc2020_sample, label=1)
        vcc2020_records = extract_all_layers_e8(lit, loader, device)
        torch.save(vcc2020_records, str(vcc2020_cache))
        print(f"  Cached → {vcc2020_cache.name}")

    # ── Load E7 records for comparison ───────────────────────────────────────
    e7_cache = E7_OUT / "cache" / "records_robust_goat.pt"
    e7_records = torch.load(str(e7_cache))
    e7_valid   = [r for r in e7_records
                  if r["system_id"] in INCLUDE_SYSTEMS and not r["is_degenerate"]]

    # Build E7 feature matrix for z-score reference
    e7_X = np.array([compute_structural_features(build_agg_graph(r))
                     for r in e7_valid], dtype=np.float64)

    # ── (c) Phoneme distribution comparison ───────────────────────────────────
    print(f"\n{'='*66}")
    print("(c) PHONEME-COUNT DISTRIBUTION COMPARISON")
    print(f"{'='*66}")

    e7_asv_counts = [r["n_nodes"] for r in e7_valid]
    dist_ok, dist_msg, (asv_c, mla_c, vcc_c) = check_phoneme_distributions(
        mlaad_records, vcc2020_records, e7_valid)

    print(f"\n  {dist_msg}")

    # Cross-MLAAD/VCC2020 length balance check
    cross_rel = 0.0   # default: no imbalance
    mla_valid_nodes = [r["n_nodes"] for r in mlaad_records if not r["is_degenerate"]]
    vcc_valid_nodes = [r["n_nodes"] for r in vcc2020_records if not r["is_degenerate"]]
    if mla_valid_nodes and vcc_valid_nodes:
        med_mla = np.median(mla_valid_nodes)
        med_vcc = np.median(vcc_valid_nodes)
        cross_rel = abs(med_mla - med_vcc) / (max(med_mla, med_vcc) + 1e-12)
        print(f"\n  Cross-set balance: MLAAD median={med_mla:.1f}, "
              f"VCC2020 median={med_vcc:.1f}  rel_diff={cross_rel:.1%}")
        if cross_rel > PHONEME_MATCH_THRESH:
            print(f"  ⚠ Cross-set imbalance > 20%: will subsample to match.")
        else:
            print(f"  Cross-set balance OK ✓")

    # Phoneme dist plot
    phon_plot_path = run_out / "phoneme_distributions.png"
    plot_phoneme_distributions(asv_c, mla_c, vcc_c, phon_plot_path)

    if not dist_ok:
        print(f"\n  ⚠ WARNING: Phoneme distributions differ > 20% from ASVspoof eval.")
        print(f"  Applying length-matched subsampling before classification.")

    # ── (d) Sanity check on 1 MLAAD + 1 VCC2020 sample ──────────────────────
    print(f"\n{'='*66}")
    print("(d) SINGLE-SAMPLE SANITY CHECK")
    print(f"{'='*66}")

    mla_valid_recs = [r for r in mlaad_records  if not r["is_degenerate"]]
    vcc_valid_recs = [r for r in vcc2020_records if not r["is_degenerate"]]

    sanity_ok = True
    if mla_valid_recs:
        _, ok = print_sanity_check(mla_valid_recs[0], "MLAAD sample 0", e7_X)
        sanity_ok = sanity_ok and ok
    if vcc_valid_recs:
        _, ok = print_sanity_check(vcc_valid_recs[0], "VCC2020 sample 0", e7_X)
        sanity_ok = sanity_ok and ok

    if not sanity_ok:
        print(f"\n  ⚠ SANITY CHECK FAILED — stop and investigate pipeline mismatch "
              f"before running full classification.")

    print(f"\n{'='*66}")
    print("PRE-CHECK COMPLETE")
    print(f"{'='*66}")
    print(f"""
Pre-registration checklist:
  (a) Dir layouts              ✓ printed above
  (b) Per-system file counts   ✓ printed above
  (c) Phoneme-dist comparison  ✓ plot saved → {phon_plot_path.name}
  (d) Sanity check features    ✓ {'PASS' if sanity_ok else 'FAIL — investigate before continuing'}

Next steps:
  1. Review (a)–(d) above.
  2. Commit this experiment file (gat_e8.py) before running classification.
  3. Run with --run-classification to execute the full classification.
     Add --length-match to force histogram-based n_nodes matching (H5 follow-up).

Commands:
  venv/bin/python3 experiments/gat_e8.py --run-classification
  venv/bin/python3 experiments/gat_e8.py --length-match
""")

    if not args.run_classification:
        print("  [Stopping here — pass --run-classification to proceed]")
        return

    # ── Stage 3: Full classification ──────────────────────────────────────────
    print(f"\n{'='*66}")
    print("STAGE 3: FULL CLASSIFICATION")
    print(f"{'='*66}")

    # Length-match subsampling
    mlaad_sample_final   = mlaad_sample
    vcc2020_sample_final = vcc2020_sample
    mla_recs_final       = [r for r in mlaad_records  if not r["is_degenerate"]]
    vcc_recs_final       = [r for r in vcc2020_records if not r["is_degenerate"]]

    if args.length_match:
        print(f"\n  [H5 follow-up] Forcing histogram-based n_nodes matching...")
        mlaad_sample_final, vcc2020_sample_final, mla_recs_final, vcc_recs_final = \
            histogram_match_subsample(mlaad_sample, vcc2020_sample,
                                      mlaad_records, vcc2020_records)
    elif cross_rel > PHONEME_MATCH_THRESH:
        mlaad_sample_final, vcc2020_sample_final, mla_recs_final, vcc_recs_final = \
            length_match_by_subsample(mlaad_sample, vcc2020_sample,
                                      mlaad_records, vcc2020_records)

    # Combine E8 records
    all_e8_records = mla_recs_final + vcc_recs_final
    X, y, sids, dsrc, paths = build_e8_feature_matrix(all_e8_records)

    print(f"\n  E8 feature matrix: N={len(X)}  "
          f"(TTS={int((y==0).sum())}, VC={int((y==1).sum())})")

    # Train E7 LR on all ASVspoof seed-1 data
    e7_lr, X_e7, y_e7 = train_e7_classifier(e7_cache)

    # Apply classifier
    y_pred = e7_lr.predict(X)
    y_prob = e7_lr.predict_proba(X)[:, 1]
    bal_acc = balanced_accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average=None,
                                                         zero_division=0)
    try: auc = roc_auc_score(y, y_prob)
    except: auc = float("nan")
    ci = bootstrap_ci(y, y_pred, y_prob)

    print(f"\n  Balanced accuracy: {bal_acc:.4f}  "
          f"95% CI [{ci['bal_acc'][0]:.4f}, {ci['bal_acc'][1]:.4f}]")
    print(f"  TTS recall (MLAAD):  {rec[0]:.4f}  prec={prec[0]:.4f}  f1={f1[0]:.4f}")
    print(f"  VC  recall (VCC2020): {rec[1]:.4f}  prec={prec[1]:.4f}  f1={f1[1]:.4f}")
    print(f"  AUC: {auc:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"              TTS(pred)  VC(pred)")
    print(f"  TTS(true):  {cm[0,0]:9d}  {cm[0,1]:7d}")
    print(f"  VC (true):  {cm[1,0]:9d}  {cm[1,1]:7d}")

    # H5 phoneme counts
    phon_counts = np.array([r["n_nodes"] for r in all_e8_records
                             if not r["is_degenerate"]], dtype=np.float64)

    # Hypothesis evaluation
    hyp_results = evaluate_hypotheses_e8(
        X, y, y_pred, y_prob, sids, dsrc, e7_lr, phon_counts, e7_valid, run_out)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("SAVING OUTPUTS")
    print(f"{'='*66}")

    plot_2d_scatter(e7_valid, all_e8_records, run_out / "scatter_2d.png")
    plot_confusion_matrix(y, y_pred, run_out / "confusion_matrix.png")

    # Manifest CSV
    all_sample_final = (mlaad_sample_final or mlaad_sample) + \
                       (vcc2020_sample_final or vcc2020_sample)
    save_manifest_csv(all_sample_final, all_e8_records, run_out / "manifest.csv")
    save_results_csv(y, y_pred, y_prob, sids, dsrc, paths,
                     run_out / "per_sample_results.csv")
    save_metrics_md(hyp_results, bal_acc, prec, rec, f1, auc, ci,
                    run_out / "e8_report.md")

    print(f"\n  All E8 outputs in: {run_out}/")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
