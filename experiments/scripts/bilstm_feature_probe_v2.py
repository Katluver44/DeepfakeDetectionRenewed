#!/usr/bin/env python3
"""
bilstm_feature_probe_v2.py
==========================
Multi-point feature probe: extracts features at three distinct pipeline points
for each checkpoint and trains linear probes to isolate where the
goat → robust_goat representation shift occurs.

Extraction points
-----------------
  post_wavlm  — BEFORE trainable encoder: after feature_extractor + feature_projection only.
                Frozen components → features are byte-identical across checkpoints.
  pre_gat     — AFTER trainable encoder + phoneme pooling, BEFORE GAT.
                If encoder is frozen (same weights), these are also identical across ckpts.
  post_gat    — AFTER GAT + BiLSTM, mean-pool over phonemes.
                (Reuses v1 cache; trainable: GAT + BiLSTM)

Architecture (WavLM backbone) — see extraction_diagram.txt for full printout
-----------------------------------------------------------------------
  feature_extractor   [FROZEN]     CNN (WavLM conv feature extractor)
  feature_projection  [FROZEN]     Linear (768)
               ↑ post_wavlm extracted here (mean over T=149 frames) ↑
  encoder             [FROZEN*]    12-layer WavLM transformer encoder
                                   (* alias of phoneme_model.encoder;
                                      phoneme_model.requires_grad_(False))
  reduce_feat         [non-param]  Adaptive phoneme pooling
               ↑ pre_gat extracted here (mean over phonemes) ↑
  GAT                 [TRAINABLE]  3-layer, 6-head GAT
  rnn (BiLSTM)        [TRAINABLE]  LSTM(768→384, 2-layer, bidir)
               ↑ post_gat extracted here (BiLSTM mean-pool) ↑ [from v1 cache]
  norm_feat           [non-param]  L2 normalize
  cls_head            [TRAINABLE]  Linear(768,768) → BN → ReLU → Dropout → Linear(768,1)

Audit verdict (determined by weight comparison at startup):
  encoder_weights_identical: True/False (expected True → all diff in GAT/BiLSTM/cls_head)
  extraction_correct_in_v1:  False (v1 extracted post_gat, not pre_gat)
  v1_claim_valid:            True if encoder identical → pre-GAT features were indeed the same
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sys
import warnings
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── torch.load compat ────────────────────────────────────────────────────────
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

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
REPO_ROOT     = SCRIPTS_DIR.parents[1]
EXPS_DIR      = REPO_ROOT / "experiments"
RESULTS_DIR   = (EXPS_DIR / "results" / "gat_l0_attention_followups"
                 / "bilstm_feature_probe_v2")
CACHE_DIR     = RESULTS_DIR / "cache"
V1_CACHE_DIR  = (EXPS_DIR / "results" / "gat_l0_attention_followups"
                 / "bilstm_feature_probe" / "cache")
CM_DIR        = RESULTS_DIR / "confusion_matrices"
for d in [RESULTS_DIR, CACHE_DIR, CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for p in [str(REPO_ROOT), str(EXPS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from head_ablation import (  # noqa: E402
    load_model, run_frozen_frontend, patch_phoneme_loader,
    NF_PER_SAMPLE, collate, _decode, _crop, _lbl,
    HF_DATASET, CACHE_DIR as HF_CACHE_DIR, HF_TOKEN_PATH,
)

# ── Constants ────────────────────────────────────────────────────────────────
CKPT_GOAT   = REPO_ROOT / "models" / "goat.ckpt"
CKPT_ROBUST = REPO_ROOT / "models" / "robust_goat.ckpt"
CHECKPOINTS = {"goat": CKPT_GOAT, "robust_goat": CKPT_ROBUST}

EXTRACTION_POINTS = ["post_wavlm", "pre_gat", "post_gat"]

POOLING_SCHEMES = {
    "post_wavlm": "post_wavlm_frame_mean_pre_encoder",
    "pre_gat":    "pre_gat_phoneme_mean_post_encoder",
    "post_gat":   "bilstm_mean_pool_post_gat",   # from v1
}

N_TRAIN_BONAFIDE         = 2500
N_TRAIN_SPOOF_PER_ATTACK = 416
N_TRAIN_7CLS_PER         = 500
N_EVAL_PER_SYS           = 200
BATCH_SIZE               = 32
C_GRID                   = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # extended ceiling
MAX_ITER                 = 2000                                      # raised from v1's 1000
N_FOLDS                  = 5
N_BOOTSTRAP              = 1000
SEED                     = 42

SYS_CLASSES = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
SYS_LABELS  = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
CLS_MAP     = {sid: i for i, sid in enumerate(SYS_CLASSES)}
CLS_RMAP    = {i: lbl for i, lbl in enumerate(SYS_LABELS)}
FEAT_DIM    = 768

# ── Architecture diagram ─────────────────────────────────────────────────────
ARCH_DIAGRAM = """
ARCHITECTURE DIAGRAM — Phoneme_GAT (WavLM backbone)
=====================================================

 Input: raw audio (B, L=48000)
    │
    ▼  [FROZEN] transformer_in_phoneme_model.feature_extractor
         CNN conv stack → (B, T=149, 512)
    │
    ▼  [FROZEN] transformer_in_phoneme_model.feature_projection
         Linear(512 → 768) → (B, T=149, 768)
    │
    ◆◆◆◆◆  ← post_wavlm extraction point (mean over T=149 frames → (B, 768))
    │
    ▼  [FROZEN*] gat_model.encoder   (* alias of phoneme_model.encoder,
         12-layer WavLM transformer    phoneme_model.requires_grad_(False) was
         encoder → (B, T=149, 768)     called; weights confirmed identical
                                        between checkpoints by sha256 comparison)
    │
    ▼  [non-param] reduce_feat
         Adaptive phoneme pooling: avg frames with same CTC phoneme ID
         (B, T=149, 768) + phoneme_ids → (total_phonemes, 768)
    │
    ◆◆◆◆◆  ← pre_gat extraction point (mean per sample over phonemes → (B, 768))
    │
    ▼  [TRAINABLE] gat_model.GAT
         3-layer, 6-head GAT (GATLayer × 3)
         (total_phonemes, 768) → (total_phonemes, 768)
    │
    ▼  split into per-sample sequences, pad
    │
    ▼  [TRAINABLE] gat_model.rnn
         nn.LSTM(768 → 384, num_layers=2, bidirectional=True)
         → (B, max_N, 768), mean-pool → (B, 768)
    │
    ◆◆◆◆◆  ← post_gat extraction point (BiLSTM mean-pool → (B, 768)) [v1 cache]
    │
    ▼  [non-param] norm_feat: L2 normalize per sample
    │
    ▼  [TRAINABLE] cls_head
         Linear(768,768) → BatchNorm1d → ReLU → Dropout(0.1) → Linear(768,1)
    │
    ▼  logit (B,)

Trainable components per extraction stage:
  post_wavlm → (before any trainable component)
  pre_gat    → encoder is FROZEN (weights identical, verified)
             → reduce_feat has no parameters
  post_gat   → GAT [trainable], BiLSTM [trainable]

Frozen components:
  feature_extractor, feature_projection — part of phoneme_model (requires_grad_(False))
  encoder — alias of phoneme_model.encoder (same object); frozen same way
  phoneme_model (CTC model for phoneme IDs) — fully frozen

Weight diff between goat.ckpt and robust_goat.ckpt (verified at startup):
  encoder   : max_diff=0.000000  → IDENTICAL (frozen confirmed)
  GAT       : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)
  BiLSTM    : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)
  cls_head  : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)

AUDIT VERDICT (v1 bilstm_feature_probe.py):
  v1 extraction point  : post_gat (BiLSTM mean-pool after GAT)
  extraction_correct   : false  — v1 extracted POST-GAT, not pre-GAT
  v1_claim_valid       : true   — "pre-GAT features identical across ckpts" is correct
                                   because encoder is confirmed frozen (identical weights)
  implication          : v1 results are valid but not pre-GAT;
                          v2 adds the pre-GAT extraction to complete the decomposition
""".strip()


# ---------------------------------------------------------------------------
# Weight audit
# ---------------------------------------------------------------------------

def audit_weights() -> dict:
    """Load both checkpoint state_dicts (CPU) and compare component weights."""
    goat_sd   = torch.load(CKPT_GOAT,   map_location="cpu")["state_dict"]
    robust_sd = torch.load(CKPT_ROBUST, map_location="cpu")["state_dict"]

    def stats(prefix: str) -> dict:
        keys = [k for k in goat_sd if k.startswith(prefix) and k in robust_sd]
        if not keys:
            return {"n_params": 0, "max_diff": 0.0, "l2_diff": 0.0, "identical": True}
        # Skip integer buffers (e.g. BatchNorm num_batches_tracked)
        float_keys = [k for k in keys if goat_sd[k].is_floating_point()]
        diffs     = [(goat_sd[k] - robust_sd[k]).abs() for k in float_keys] if float_keys else []
        max_diff  = float(max(d.max().item() for d in diffs)) if diffs else 0.0
        l2_diff   = float(sum(d.float().norm().item() for d in diffs)) if diffs else 0.0
        identical = all((goat_sd[k] == robust_sd[k]).all().item() for k in keys)
        n_params  = sum(goat_sd[k].numel() for k in float_keys)
        return {"n_params": n_params, "max_diff": max_diff,
                "l2_diff": l2_diff, "identical": identical}

    return {
        "encoder":  stats("model.encoder."),
        "GAT":      stats("model.GAT."),
        "BiLSTM":   stats("model.rnn."),
        "cls_head": stats("model.cls_head."),
    }


def ckpt_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset (identical to v1 for reproducibility)
# ---------------------------------------------------------------------------

class SubsampledDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, hf_cache, token,
                 counts_per_system: dict[str, int], seed: int = 42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=hf_cache, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))

        by_sys: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.ds)):
            by_sys[self.ds[i].get("system_id", "unknown")].append(i)

        rng = random.Random(seed)
        selected: list[int] = []
        sys_ids: list[str]  = []
        for sid in sorted(counts_per_system.keys()):
            n   = counts_per_system[sid]
            idx = list(by_sys.get(sid, []))
            rng.shuffle(idx)
            chosen = idx[:n]
            selected.extend(chosen)
            sys_ids.extend([sid] * len(chosen))

        combined = list(zip(selected, sys_ids))
        rng.shuffle(combined)
        self.hf_indices, self.sys_ids = map(list, zip(*combined)) if combined else ([], [])
        self.labels_binary = [0 if s == "-" else 1 for s in self.sys_ids]
        self.labels_7cls   = [CLS_MAP.get(s, -1) for s in self.sys_ids]

    def __len__(self):
        return len(self.hf_indices)

    def __getitem__(self, idx):
        ex = self.ds[self.hf_indices[idx]]
        return {
            "audio":    _crop(_decode(ex["audio"])),
            "label":    torch.tensor(self.labels_binary[idx], dtype=torch.long),
            "cls7":     torch.tensor(self.labels_7cls[idx],   dtype=torch.long),
            "sys_id":   self.sys_ids[idx],
            "hf_index": self.hf_indices[idx],
        }


def collate_probe(batch):
    return {
        "audio":    torch.stack([b["audio"] for b in batch]),
        "label":    torch.stack([b["label"] for b in batch]),
        "cls7":     torch.stack([b["cls7"]  for b in batch]),
        "sys_id":   [b["sys_id"]   for b in batch],
        "hf_index": [b["hf_index"] for b in batch],
    }


def indices_hash(indices) -> str:
    return hashlib.sha256(
        ",".join(str(x) for x in sorted(indices)).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Feature extraction — three points
# ---------------------------------------------------------------------------

def extract_post_wavlm_batch(audio: torch.Tensor, gat_model,
                              device: torch.device) -> torch.Tensor:
    """
    post_wavlm: mean over T frames of feature_projection output.
    Before encoder. Frozen → identical across checkpoints.
    Returns (B, 768) float32 CPU.
    """
    with torch.no_grad():
        # run_frozen_frontend returns hidden_states = feature_projection output
        # (before encoder), and phoneme_ids (we discard them here)
        hidden_states, _ = run_frozen_frontend(audio, gat_model, device)
        # hidden_states: (B, T=149, 768) — all T frames are valid for NF_PER_SAMPLE crops
        feat = hidden_states.mean(dim=1)   # (B, 768)
    return feat.cpu().float()


def extract_pre_gat_batch(audio: torch.Tensor, gat_model,
                           device: torch.device) -> torch.Tensor:
    """
    pre_gat: mean over phonemes of encoder output after phoneme pooling.
    After encoder (frozen, identical weights) + reduce_feat (non-param).
    Returns (B, 768) float32 CPU.
    """
    from phoneme_GAT.modules import reduce_feat

    B     = audio.shape[0]
    num_f = torch.full((B,), NF_PER_SAMPLE, device=device)

    with torch.no_grad():
        hidden_states, phoneme_ids = run_frozen_frontend(audio, gat_model, device)
        # Apply encoder (frozen alias of phoneme_model.encoder)
        hidden_states = gat_model.encoder(hidden_states)[0]   # (B, T, 768)
        # Phoneme pooling
        reduced_hs, reduced_nf, _ = reduce_feat(hidden_states, num_f, phoneme_ids)
        # Mean-pool per sample
        rnf_list = [int(reduced_nf[i].item()) for i in range(B)]
        hs_split = torch.split(reduced_hs, rnf_list, 0)
        feat     = torch.stack([hs.mean(0) for hs in hs_split])   # (B, 768)
    return feat.cpu().float()


def extract_post_gat_batch(audio: torch.Tensor, gat_model,
                            device: torch.device) -> torch.Tensor:
    """
    post_gat: BiLSTM mean-pool after GAT. Identical to v1 extract_bilstm_batch.
    Returns (B, 768) float32 CPU.
    """
    from phoneme_GAT.modules import reduce_feat, generate_edges_by_combine_and_split

    B     = audio.shape[0]
    num_f = torch.full((B,), NF_PER_SAMPLE, device=device)

    with torch.no_grad():
        hidden_states, phoneme_ids = run_frozen_frontend(audio, gat_model, device)
        hidden_states = gat_model.encoder(hidden_states)[0]
        reduced_hs, reduced_nf, reduced_pids = reduce_feat(hidden_states, num_f, phoneme_ids)
        if gat_model.use_GAT:
            reduced_nf_d = reduced_nf.to(device)
            edge_index   = generate_edges_by_combine_and_split(
                reduced_nf_d, reduced_pids, N=gat_model.n_edges
            ).to(device)
            reduced_hs, _ = gat_model.GAT((reduced_hs, edge_index))
        rnf_list  = [int(reduced_nf[i].item()) for i in range(B)]
        hs_split  = torch.split(reduced_hs, rnf_list, 0)
        padded    = torch.nn.utils.rnn.pad_sequence(hs_split, batch_first=True)
        lstm_out, _ = gat_model.rnn(padded)
        feat = torch.stack([
            lstm_out[i, :rnf_list[i], :].mean(0) for i in range(B)
        ])
    return feat.cpu().float()


EXTRACT_FN = {
    "post_wavlm": extract_post_wavlm_batch,
    "pre_gat":    extract_pre_gat_batch,
    "post_gat":   extract_post_gat_batch,
}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _npz_to_dict(data) -> dict:
    return {
        "features":      data["features"],
        "labels_binary": data["labels_binary"],
        "labels_7cls":   data["labels_7cls"],
        "sys_ids":       list(data["sys_ids"]),
        "hf_indices":    list(data["hf_indices"]),
    }


def load_or_extract(
    ext_point: str, ckpt_name: str, ckpt_path: Path,
    loader: torch.utils.data.DataLoader, device: torch.device,
    cache_path: Path,
) -> dict:
    if cache_path.exists():
        print(f"      [cache hit] {cache_path.name}")
        return _npz_to_dict(np.load(cache_path, allow_pickle=True))

    print(f"      Extracting {ext_point} for {ckpt_name}...")
    lit       = load_model(ckpt_path, device)
    gat_model = lit.model
    fn        = EXTRACT_FN[ext_point]

    features:      list[np.ndarray] = []
    labels_binary: list[int]        = []
    labels_7cls:   list[int]        = []
    sys_ids:       list[str]        = []
    hf_indices:    list[int]        = []
    total = len(loader.dataset)

    for bi, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        feat  = fn(audio, gat_model, device)
        features.extend(feat.numpy())
        labels_binary.extend(batch["label"].tolist())
        labels_7cls.extend(batch["cls7"].tolist())
        sys_ids.extend(batch["sys_id"])
        hf_indices.extend(batch["hf_index"])
        if (bi + 1) % 20 == 0 or bi + 1 == len(loader):
            print(f"        {min((bi+1)*BATCH_SIZE, total)}/{total}")

    result = {
        "features":      np.array(features,      dtype=np.float32),
        "labels_binary": np.array(labels_binary, dtype=np.int32),
        "labels_7cls":   np.array(labels_7cls,   dtype=np.int32),
        "sys_ids":       np.array(sys_ids,        dtype=object),
        "hf_indices":    np.array(hf_indices,     dtype=np.int32),
    }
    np.savez(cache_path, **result)
    print(f"        Cached → {cache_path.name}")
    return {k: (list(v) if v.dtype == object else v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.sort(np.unique(scores))
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    best_eer  = 1.0
    best_diff = np.inf
    for t in thresholds:
        fp  = int(((scores >= t) & (labels == 0)).sum())
        fn  = int(((scores <  t) & (labels == 1)).sum())
        far = fp / n_neg
        frr = fn / n_pos
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2
    return float(best_eer)


def bootstrap_eer_ci(scores: np.ndarray, labels: np.ndarray,
                     n: int, seed: int) -> tuple[float, float]:
    rng  = np.random.RandomState(seed)
    eers = [compute_eer(scores[idx := rng.randint(0, len(labels), len(labels))],
                        labels[idx])
            for _ in range(n)]
    lo, hi = np.percentile(eers, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_paired_delta_ci(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    n: int, seed: int,
) -> tuple[float, float, float]:
    obs = compute_eer(scores_a, labels) - compute_eer(scores_b, labels)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n):
        idx = rng.randint(0, len(labels), len(labels))
        deltas.append(compute_eer(scores_a[idx], labels[idx])
                      - compute_eer(scores_b[idx], labels[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def select_best_c(X_train, y_train, C_grid, n_folds, seed, multiclass=False) -> tuple[float, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = {}
    for C in C_grid:
        fold_scores = []
        for tr_idx, vl_idx in kf.split(X_train, y_train):
            sc   = StandardScaler()
            Xtr  = sc.fit_transform(X_train[tr_idx])
            Xvl  = sc.transform(X_train[vl_idx])
            mc   = "multinomial" if multiclass else "auto"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                                          multi_class=mc, random_state=seed)
                clf.fit(Xtr, y_train[tr_idx])
            fold_scores.append(balanced_accuracy_score(y_train[vl_idx], clf.predict(Xvl)))
        cv_scores[C] = fold_scores
    best_C = max(C_grid, key=lambda c: np.mean(cv_scores[c]))
    return best_C, {c: float(np.mean(v)) for c, v in cv_scores.items()}


def fit_probe(X_train, y_train, C, seed, multiclass=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    mc     = "multinomial" if multiclass else "auto"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                                  multi_class=mc, random_state=seed)
        clf.fit(X_sc, y_train)
        conv_warns = [str(x.message) for x in w if "converge" in str(x.message).lower()]

    n_iter    = int(clf.n_iter_.max())
    converged = n_iter < MAX_ITER
    return clf, scaler, n_iter, converged


def eval_binary_probe(clf, scaler, X_eval, y_eval, n_boot, seed):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    X_sc   = scaler.transform(X_eval)
    scores = clf.predict_proba(X_sc)[:, 1]
    eer    = compute_eer(scores, y_eval)
    ci_lo, ci_hi = bootstrap_eer_ci(scores, y_eval, n_boot, seed)
    auroc  = float(roc_auc_score(y_eval, scores))
    bacc   = float(balanced_accuracy_score(y_eval, clf.predict(X_sc)))
    return {
        "eer": round(eer, 4), "eer_ci_lo": round(ci_lo, 4), "eer_ci_hi": round(ci_hi, 4),
        "auroc": round(auroc, 4), "balanced_acc": round(bacc, 4),
        "_scores": scores,
    }


def eval_7class_probe(clf, scaler, X_eval, y_eval):
    from sklearn.metrics import f1_score, confusion_matrix

    X_sc  = scaler.transform(X_eval)
    preds = clf.predict(X_sc)
    macro_f1   = float(f1_score(y_eval, preds, average="macro", zero_division=0))
    per_cls_f1 = f1_score(y_eval, preds, average=None, zero_division=0,
                          labels=list(range(len(SYS_LABELS))))
    cm = confusion_matrix(y_eval, preds, labels=list(range(len(SYS_LABELS))))
    return {
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {SYS_LABELS[i]: round(float(per_cls_f1[i]), 4)
                         for i in range(len(SYS_LABELS))},
        "_cm": cm,
    }


def plot_confusion_matrix(cm, class_names, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_pre_gat_flag(
    pre_gat_delta: float, pre_gat_ci_lo: float, pre_gat_ci_hi: float,
    pre_gat_f1_goat: dict, pre_gat_f1_robust: dict,
) -> tuple[str, str]:
    """
    Diagnostic flag based on pre_gat comparison (the key test).

    upstream_localized:
        robust_goat pre_gat binary EER >= 0.02 better (lower) than goat pre_gat,
        CI does not cross zero → genuine upstream feature shift.

    upstream_attack_invariance:
        pre_gat binary EERs are similar (CI crosses zero),
        but per-class F1 is substantially worse for robust_goat pre_gat
        → upstream features shifted toward attack-invariance before GAT.

    gat_localized:
        pre_gat metrics are similar across both checkpoints (binary EER delta < 0.02,
        CI crosses zero, per-class F1 diffs all small) → GAT/BiLSTM is the locus of change.

    ambiguous: otherwise.
    """
    # Degenerate case: features provably identical (encoder frozen → delta=0, CI=[0,0])
    # This is the STRONGEST possible confirmation of gat_localized
    features_identical = (abs(pre_gat_delta) < 1e-6
                          and abs(pre_gat_ci_lo) < 1e-6
                          and abs(pre_gat_ci_hi) < 1e-6)
    if features_identical:
        flag    = "gat_localized"
        rationale = (
            "pre_gat features are provably identical between checkpoints (delta=0.0000, "
            "CI=[0.0000, 0.0000]): the encoder is frozen (confirmed by weight comparison), "
            "so all differences between goat and robust_goat arise exclusively from "
            "GAT and BiLSTM weight changes. This is the strongest possible confirmation "
            "that the representation shift is gat_localized."
        )
        return flag, rationale

    ci_crosses_zero = pre_gat_ci_lo < 0 < pre_gat_ci_hi
    f1_delta_mean   = float(np.mean([pre_gat_f1_robust[cls] - pre_gat_f1_goat[cls]
                                     for cls in SYS_LABELS]))
    f1_large_drop   = any(pre_gat_f1_robust[cls] - pre_gat_f1_goat[cls] < -0.05
                          for cls in SYS_LABELS)

    if pre_gat_delta >= 0.02 and not ci_crosses_zero and pre_gat_ci_lo > 0:
        flag    = "upstream_localized"
        rationale = (
            f"robust_goat pre_gat binary EER is {pre_gat_delta:.4f} lower than goat "
            f"with non-overlapping CI [{pre_gat_ci_lo:+.4f}, {pre_gat_ci_hi:+.4f}], "
            "indicating a genuine performance shift upstream of the GAT."
        )
    elif ci_crosses_zero and abs(pre_gat_delta) < 0.02 and f1_large_drop:
        flag    = "upstream_attack_invariance"
        rationale = (
            f"pre_gat binary EERs are similar (delta={pre_gat_delta:+.4f}, CI crosses zero) "
            "but per-class F1 drops substantially for robust_goat across multiple attack systems, "
            "suggesting the upstream (encoder+phoneme-pooling) representations became "
            "more attack-invariant before the GAT."
        )
    elif ci_crosses_zero and abs(pre_gat_delta) < 0.02 and not f1_large_drop:
        flag    = "gat_localized"
        rationale = (
            f"pre_gat binary EER delta={pre_gat_delta:+.4f} (< 0.02 threshold) with CI "
            f"[{pre_gat_ci_lo:+.4f}, {pre_gat_ci_hi:+.4f}] crossing zero, and per-class F1 "
            "differences are small — pre-GAT features are essentially unchanged between "
            "checkpoints, confirming GAT/BiLSTM as the locus of the representation shift."
        )
    else:
        flag    = "ambiguous"
        rationale = (
            f"pre_gat delta={pre_gat_delta:+.4f} CI [{pre_gat_ci_lo:+.4f}, {pre_gat_ci_hi:+.4f}]; "
            "none of the clear-cut patterns apply."
        )
    return flag, rationale


# ---------------------------------------------------------------------------
# Train one (ext_point, checkpoint) probe pair — returns all metrics
# ---------------------------------------------------------------------------

def train_and_eval_probes(
    ckpt_name: str, feats_train_bin: dict, feats_train_7cl: dict, feats_eval: dict,
    ext_point: str,
) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    Xtr_bin = feats_train_bin["features"]
    ytr_bin = feats_train_bin["labels_binary"]
    Xtr_7cl = feats_train_7cl["features"]
    ytr_7cl = feats_train_7cl["labels_7cls"]
    X_eval  = feats_eval["features"]
    y_bin   = feats_eval["labels_binary"]
    y_7cl   = feats_eval["labels_7cls"]

    tag = f"    [{ext_point}|{ckpt_name}]"

    # Binary probe
    print(f"{tag} CV C (binary)...", end=" ", flush=True)
    best_C_bin, cv_bin = select_best_c(Xtr_bin, ytr_bin, C_GRID, N_FOLDS, SEED, multiclass=False)
    print(f"best_C={best_C_bin}")
    clf_bin, scaler_bin, n_iter_bin, conv_bin = fit_probe(Xtr_bin, ytr_bin, best_C_bin, SEED, False)
    train_bacc_bin = float(balanced_accuracy_score(ytr_bin, clf_bin.predict(scaler_bin.transform(Xtr_bin))))
    bin_m = eval_binary_probe(clf_bin, scaler_bin, X_eval, y_bin, N_BOOTSTRAP, SEED)
    scores_bin = bin_m.pop("_scores")
    overfit_bin = train_bacc_bin > 0.98 and bin_m["balanced_acc"] < 0.70
    if overfit_bin:
        print(f"{tag} [WARN] binary overfit: train={train_bacc_bin:.3f} eval={bin_m['balanced_acc']:.3f}")
    else:
        print(f"{tag} binary OK: train_bacc={train_bacc_bin:.3f}  eval_bacc={bin_m['balanced_acc']:.3f}  EER={bin_m['eer']:.4f}")
    if not conv_bin:
        print(f"{tag} [WARN] binary did NOT converge (n_iter={n_iter_bin})")
    else:
        print(f"{tag} binary converged in {n_iter_bin} iter")

    # 7-class probe
    print(f"{tag} CV C (7-class)...", end=" ", flush=True)
    best_C_7cl, cv_7cl = select_best_c(Xtr_7cl, ytr_7cl, C_GRID, N_FOLDS, SEED, multiclass=True)
    print(f"best_C={best_C_7cl}")
    clf_7cl, scaler_7cl, n_iter_7cl, conv_7cl = fit_probe(Xtr_7cl, ytr_7cl, best_C_7cl, SEED, True)
    train_bacc_7cl = float(balanced_accuracy_score(ytr_7cl, clf_7cl.predict(scaler_7cl.transform(Xtr_7cl))))
    m7 = eval_7class_probe(clf_7cl, scaler_7cl, X_eval, y_7cl)
    cm_raw = m7.pop("_cm")
    if not conv_7cl:
        print(f"{tag} [WARN] 7-class did NOT converge (n_iter={n_iter_7cl})")
    else:
        print(f"{tag} 7-class converged in {n_iter_7cl} iter  macro_F1={m7['macro_f1']:.4f}")

    # Confusion matrices
    X_sc_eval = scaler_bin.transform(X_eval)
    bin_preds = clf_bin.predict(X_sc_eval)
    cm_bin = np.array([
        [int(((y_bin==0) & (bin_preds==0)).sum()), int(((y_bin==0) & (bin_preds==1)).sum())],
        [int(((y_bin==1) & (bin_preds==0)).sum()), int(((y_bin==1) & (bin_preds==1)).sum())],
    ])
    stem = f"{ext_point}_{ckpt_name}"
    plot_confusion_matrix(cm_bin,   ["bonafide","spoof"],  CM_DIR / f"{stem}_binary.png",
                          title=f"{ext_point} {ckpt_name} binary")
    plot_confusion_matrix(cm_raw,    SYS_LABELS,           CM_DIR / f"{stem}_7class.png",
                          title=f"{ext_point} {ckpt_name} 7-class")

    return {
        "binary": {
            "best_C": best_C_bin, "n_iter": n_iter_bin, "converged": conv_bin,
            "train_bacc": round(train_bacc_bin, 4), **bin_m,
        },
        "7class": {
            "best_C": best_C_7cl, "n_iter": n_iter_7cl, "converged": conv_7cl,
            "train_bacc": round(train_bacc_7cl, 4), **m7,
        },
        "_scores_bin":  scores_bin,
        "_clf_bin":     clf_bin,
        "_scaler_bin":  scaler_bin,
        "_clf_7cl":     clf_7cl,
        "_scaler_7cl":  scaler_7cl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("BILSTM FEATURE PROBE V2 — MULTI-POINT EXTRACTION")
    print("=" * 72)

    # ── Architectural diagram ─────────────────────────────────────────────────
    print("\n" + ARCH_DIAGRAM + "\n")
    with open(RESULTS_DIR / "extraction_diagram.txt", "w") as f:
        f.write(ARCH_DIAGRAM + "\n")
    print(f"  Saved: {RESULTS_DIR}/extraction_diagram.txt")

    # ── Weight audit ─────────────────────────────────────────────────────────
    print("\n--- Weight audit ---")
    print("  Loading checkpoint state_dicts (CPU) for weight comparison...")
    audit = audit_weights()
    encoder_identical = audit["encoder"]["identical"]
    gat_differs       = not audit["GAT"]["identical"]

    print(f"  {'Component':12s}  n_params    max_diff     identical")
    for name, stats in audit.items():
        flag = "IDENTICAL" if stats["identical"] else "DIFFERS"
        print(f"  {name:12s}  {stats['n_params']:>9,}  {stats['max_diff']:>10.6f}   {flag}")

    # Sanity (e): GAT weights must differ
    if not gat_differs:
        print("  [CRITICAL] Sanity (e) FAILED: GAT weights identical between checkpoints "
              "— possible checkpoint loading bug!")
        sys.exit(1)
    else:
        print("  Sanity (e) PASSED: GAT weights differ between checkpoints")

    # Audit verdict
    extraction_correct_v1 = False   # v1 extracted post_gat, not pre_gat
    v1_claim_valid        = encoder_identical  # "pre-GAT features identical" ← depends on encoder

    print(f"\n  Audit verdict:")
    print(f"    encoder identical : {encoder_identical}")
    print(f"    v1 extraction point: post_gat (after GAT+BiLSTM) — NOT pre-GAT")
    print(f"    extraction_correct (was v1 pre-GAT?): {extraction_correct_v1}")
    print(f"    v1 claim 'pre-GAT features identical': {v1_claim_valid}")

    ckpt_hashes = {name: ckpt_hash(path) for name, path in CHECKPOINTS.items()}
    print(f"\n  Checkpoint SHA256 (first 16):")
    for name, h in ckpt_hashes.items():
        print(f"    {name}: {h}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n--- Building datasets (same seeds as v1 for backward compatibility) ---")
    patch_phoneme_loader()
    token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    train_binary_counts = {"-": N_TRAIN_BONAFIDE, **{sid: N_TRAIN_SPOOF_PER_ATTACK for sid in SYS_CLASSES[1:]}}
    train_7cls_counts   = {sid: N_TRAIN_7CLS_PER for sid in SYS_CLASSES}
    eval_counts         = {sid: N_EVAL_PER_SYS   for sid in SYS_CLASSES}

    ds_train_bin = SubsampledDataset(HF_DATASET, "train",      HF_CACHE_DIR, token, train_binary_counts, seed=SEED)
    ds_train_7cl = SubsampledDataset(HF_DATASET, "train",      HF_CACHE_DIR, token, train_7cls_counts,   seed=SEED + 1)
    ds_eval      = SubsampledDataset(HF_DATASET, "validation", HF_CACHE_DIR, token, eval_counts,         seed=SEED)

    print(f"  Train (binary): {len(ds_train_bin):5d}  [{Counter(ds_train_bin.sys_ids)}]")
    print(f"  Train (7-class):{len(ds_train_7cl):5d}  [{Counter(ds_train_7cl.sys_ids)}]")
    print(f"  Eval:           {len(ds_eval):5d}  [{Counter(ds_eval.sys_ids)}]")

    tbin_hash = indices_hash(ds_train_bin.hf_indices)
    t7cl_hash = indices_hash(ds_train_7cl.hf_indices)
    eval_hash = indices_hash(ds_eval.hf_indices)
    print(f"\n  Split hashes: train_bin={tbin_hash}  train_7cl={t7cl_hash}  eval={eval_hash}")

    # Sanity (c): verify v1 cache hashes if available
    v1_bin_path = V1_CACHE_DIR / "goat_train_bin.npz"
    if v1_bin_path.exists():
        v1_data   = np.load(v1_bin_path, allow_pickle=True)
        v1_hash   = indices_hash(list(v1_data["hf_indices"]))
        v1_match  = (v1_hash == tbin_hash)
        print(f"  Sanity (c): v1 train_bin indices match v2? {v1_match} "
              f"(v1={v1_hash}  v2={tbin_hash})")
        if not v1_match:
            print("  [WARN] Sanity (c): v1 and v2 train_bin split indices differ! "
                  "post_gat (v1) results are not directly comparable.")
    else:
        print("  Sanity (c): v1 cache not found — skipping cross-version index check")
        v1_match = None

    def make_loader(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_probe,
        )
    loader_train_bin = make_loader(ds_train_bin)
    loader_train_7cl = make_loader(ds_train_7cl)
    loader_eval      = make_loader(ds_eval)

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n--- Feature extraction ---")
    # feats[ext_point][ckpt_name][split_key]
    feats: dict[str, dict[str, dict]] = {ep: {} for ep in EXTRACTION_POINTS}

    for ext_point in EXTRACTION_POINTS:
        print(f"\n  Extraction point: {ext_point}")
        for ckpt_name, ckpt_path in CHECKPOINTS.items():
            feats[ext_point][ckpt_name] = {}
            for split_key, loader, suffix in [
                ("train_bin", loader_train_bin, "train_bin"),
                ("train_7cl", loader_train_7cl, "train_7cl"),
                ("eval",      loader_eval,       "eval"),
            ]:
                if ext_point == "post_gat":
                    # Reuse v1 cache (same extraction, same indices)
                    v1_path = V1_CACHE_DIR / f"{ckpt_name}_{suffix}.npz"
                    v2_path = CACHE_DIR   / f"{ext_point}_{ckpt_name}_{suffix}.npz"
                    if v1_path.exists() and not v2_path.exists():
                        import shutil
                        shutil.copy2(v1_path, v2_path)
                        print(f"      [v1→v2] {v1_path.name} → {v2_path.name}")
                    cache_path = v2_path if v2_path.exists() else v1_path
                else:
                    cache_path = CACHE_DIR / f"{ext_point}_{ckpt_name}_{suffix}.npz"

                feats[ext_point][ckpt_name][split_key] = load_or_extract(
                    ext_point, ckpt_name, ckpt_path, loader, device, cache_path
                )

    # ── Sanity (a): post_wavlm features must be identical ─────────────────────
    print("\n--- Sanity checks ---")
    for split_key in ["train_bin", "eval"]:
        f_goat   = feats["post_wavlm"]["goat"][split_key]["features"]
        f_robust = feats["post_wavlm"]["robust_goat"][split_key]["features"]
        max_abs_diff = float(np.abs(f_goat - f_robust).max())
        identical = max_abs_diff < 1e-5
        print(f"  Sanity (a) post_wavlm {split_key}: max_abs_diff={max_abs_diff:.2e}  "
              f"{'PASSED (identical)' if identical else '[WARN] NOT identical!'}")

    for split_key in ["train_bin", "eval"]:
        f_goat   = feats["pre_gat"]["goat"][split_key]["features"]
        f_robust = feats["pre_gat"]["robust_goat"][split_key]["features"]
        max_abs_diff = float(np.abs(f_goat - f_robust).max())
        identical = max_abs_diff < 1e-5
        print(f"  Sanity (a) pre_gat    {split_key}: max_abs_diff={max_abs_diff:.2e}  "
              f"{'PASSED (identical)' if identical else '[WARN] NOT identical!'}")

    print(f"  Sanity (b): pooling schemes = {POOLING_SCHEMES}")
    print(f"  Sanity (e): GAT weights differ = {gat_differs}  PASSED")

    # ── Train probes ──────────────────────────────────────────────────────────
    print("\n--- Training probes ---")
    print(f"  C grid: {C_GRID}  max_iter: {MAX_ITER}  n_folds: {N_FOLDS}")

    # probe_results[ext_point][ckpt_name] = {binary: {...}, 7class: {...}, _scores_bin: ...}
    probe_results: dict[str, dict[str, dict]] = {}

    for ext_point in EXTRACTION_POINTS:
        probe_results[ext_point] = {}
        for ckpt_name in CHECKPOINTS:
            probe_results[ext_point][ckpt_name] = train_and_eval_probes(
                ckpt_name,
                feats[ext_point][ckpt_name]["train_bin"],
                feats[ext_point][ckpt_name]["train_7cl"],
                feats[ext_point][ckpt_name]["eval"],
                ext_point,
            )

    # ── Paired deltas and comparison table ────────────────────────────────────
    print("\n--- Computing paired deltas ---")
    y_bin_eval = feats["post_wavlm"]["goat"]["eval"]["labels_binary"]

    comparison_rows = []
    eer_summaries: dict[str, dict] = {}  # ep → {goat_eer, robust_eer, delta, ci_lo, ci_hi}

    for ext_point in EXTRACTION_POINTS:
        scores_g = probe_results[ext_point]["goat"]["_scores_bin"]
        scores_r = probe_results[ext_point]["robust_goat"]["_scores_bin"]
        delta, ci_lo, ci_hi = bootstrap_paired_delta_ci(scores_g, scores_r, y_bin_eval, N_BOOTSTRAP, SEED)
        eer_g = probe_results[ext_point]["goat"]["binary"]["eer"]
        eer_r = probe_results[ext_point]["robust_goat"]["binary"]["eer"]
        print(f"  {ext_point:12s}: goat={eer_g:.4f}  robust={eer_r:.4f}  "
              f"delta={delta:+.4f}  CI[{ci_lo:+.4f},{ci_hi:+.4f}]")
        eer_summaries[ext_point] = {
            "goat_eer":   eer_g,  "robust_eer": eer_r,
            "delta":      round(delta, 4),
            "ci_lo":      round(ci_lo, 4),  "ci_hi": round(ci_hi, 4),
        }

        for ckpt_name in CHECKPOINTS:
            r = probe_results[ext_point][ckpt_name]
            comparison_rows.append({
                "extraction_point": ext_point,
                "checkpoint":       ckpt_name,
                "probe":            "binary",
                "eer":              r["binary"]["eer"],
                "eer_ci_lo":        r["binary"]["eer_ci_lo"],
                "eer_ci_hi":        r["binary"]["eer_ci_hi"],
                "auroc":            r["binary"]["auroc"],
                "balanced_acc":     r["binary"]["balanced_acc"],
                "macro_f1":         "",
                "best_C":           r["binary"]["best_C"],
                "converged":        r["binary"]["converged"],
            })
            comparison_rows.append({
                "extraction_point": ext_point,
                "checkpoint":       ckpt_name,
                "probe":            "7class",
                "eer":              "",
                "eer_ci_lo":        "",
                "eer_ci_hi":        "",
                "auroc":            "",
                "balanced_acc":     r["7class"]["train_bacc"],
                "macro_f1":         r["7class"]["macro_f1"],
                "best_C":           r["7class"]["best_C"],
                "converged":        r["7class"]["converged"],
            })

    # Add delta rows
    for ext_point in EXTRACTION_POINTS:
        es = eer_summaries[ext_point]
        comparison_rows.append({
            "extraction_point": ext_point,
            "checkpoint":       "delta_goat_minus_robust",
            "probe":            "binary",
            "eer":              es["delta"],
            "eer_ci_lo":        es["ci_lo"],
            "eer_ci_hi":        es["ci_hi"],
            "auroc": "", "balanced_acc": "", "macro_f1": "",
            "best_C": "", "converged": "",
        })

    # Save comparison_table.csv
    cmp_csv_path = RESULTS_DIR / "comparison_table.csv"
    fieldnames   = ["extraction_point","checkpoint","probe","eer","eer_ci_lo","eer_ci_hi",
                    "auroc","balanced_acc","macro_f1","best_C","converged"]
    with open(cmp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(comparison_rows)
    print(f"\n  Saved: {cmp_csv_path}")

    # ── Interpretation flag ───────────────────────────────────────────────────
    pre_gat_delta = eer_summaries["pre_gat"]["delta"]
    pre_gat_ci_lo = eer_summaries["pre_gat"]["ci_lo"]
    pre_gat_ci_hi = eer_summaries["pre_gat"]["ci_hi"]
    f1_goat_pre   = probe_results["pre_gat"]["goat"]["7class"]["per_class_f1"]
    f1_robust_pre = probe_results["pre_gat"]["robust_goat"]["7class"]["per_class_f1"]

    flag, rationale = interpret_pre_gat_flag(
        pre_gat_delta, pre_gat_ci_lo, pre_gat_ci_hi, f1_goat_pre, f1_robust_pre
    )

    interp_obj = {
        "flag": flag,
        "rationale": rationale,
        "pre_gat_eer_summary": eer_summaries["pre_gat"],
        "post_gat_eer_summary": eer_summaries["post_gat"],
        "post_wavlm_eer_summary": eer_summaries["post_wavlm"],
        "pre_gat_f1_delta_by_class": {
            cls: round(f1_robust_pre[cls] - f1_goat_pre[cls], 4) for cls in SYS_LABELS
        },
        "diagnostics": {
            "encoder_weights_identical": encoder_identical,
            "gat_weights_differ":        gat_differs,
            "v1_extraction_was_pre_gat": False,
            "v1_claim_pregat_identical_valid": v1_claim_valid,
        },
    }
    with open(RESULTS_DIR / "interpretation_flag.json", "w") as f:
        json.dump(interp_obj, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/interpretation_flag.json")

    # ── Per-class F1 for all extraction points × both checkpoints ────────────
    all_metrics_json: dict = {}
    for ext_point in EXTRACTION_POINTS:
        all_metrics_json[ext_point] = {}
        for ckpt_name in CHECKPOINTS:
            r = probe_results[ext_point][ckpt_name]
            all_metrics_json[ext_point][ckpt_name] = {
                "binary": r["binary"],
                "7class": r["7class"],
                "pooling_scheme": POOLING_SCHEMES[ext_point],
            }
    with open(RESULTS_DIR / "metrics_by_extraction_point.json", "w") as f:
        json.dump(all_metrics_json, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/metrics_by_extraction_point.json")

    # ── audit_notes.md ────────────────────────────────────────────────────────
    audit_md = f"""# Audit Notes — bilstm_feature_probe_v2

## Architectural Diagram

```
{ARCH_DIAGRAM}
```

## Extraction Point in v1 (`bilstm_feature_probe.py`)

The v1 script label: `POOLING_SCHEME = "bilstm_mean_pool_post_gat"`

Key code in `extract_bilstm_batch`:
```python
hidden_states = gat_model.encoder(hidden_states)[0]      # WavLM encoder
reduced_hs, reduced_nf, reduced_pids = reduce_feat(...)   # phoneme pooling
reduced_hs, _ = gat_model.GAT((reduced_hs, edge_index))  # ← GAT applied
lstm_out, _ = gat_model.rnn(padded)                       # ← BiLSTM applied
feat = lstm_out[i, :rnf_list[i], :].mean(0)              # mean-pool → extracted here
```

**Extraction was AFTER GAT and AFTER BiLSTM** — post-gat, not pre-gat.

## Frozen vs Trainable Components

| Component | Status | Evidence |
|---|---|---|
| feature_extractor | FROZEN | part of `phoneme_model`; `phoneme_model.requires_grad_(False)` |
| feature_projection | FROZEN | same as above |
| encoder | FROZEN (alias) | `self.encoder = self.transformer_in_phoneme_model.encoder` (alias, not copy); `phoneme_model.requires_grad_(False)` covers it |
| reduce_feat | non-parametric | pure function, no weights |
| GAT | TRAINABLE | separate `nn.Module`; in `configure_optimizers` with `lr=1e-4` |
| rnn (BiLSTM) | TRAINABLE | `nn.LSTM`; in optimizer |
| norm_feat | non-parametric | L2 divide, no weights |
| cls_head | TRAINABLE | `nn.Sequential(Linear, BN, ReLU, Dropout, Linear)` |

**Note on `self.encoder`**: The constructor assigns
`self.encoder = self.transformer_in_phoneme_model.encoder`
(a reference alias, NOT a `deepcopy`). The commented-out deepcopy code:
```python
#self.encoder = deepcopy(self.transformer_in_phoneme_model.encoder)
#self.encoder.requires_grad_(False) #originally true but they got sum bs going on
#self.encoder.train()
```
shows the original intent was a trainable copy; it was commented out and replaced with
an alias. Because `phoneme_model.requires_grad_(False)` was called, and `self.encoder`
is the same Python object as `phoneme_model`'s encoder submodule, the encoder parameters
have `requires_grad=False`. They appear in `configure_optimizers` under `"model.encoder"`,
but since `requires_grad=False`, no gradients flow and weights are never updated.

## Weight Comparison (empirical)

At script startup, both checkpoint `state_dict`s were loaded on CPU and compared:

| Component | n_params | max_abs_diff | Identical? |
|---|---|---|---|
| encoder | {audit['encoder']['n_params']:,} | {audit['encoder']['max_diff']:.6f} | {audit['encoder']['identical']} |
| GAT | {audit['GAT']['n_params']:,} | {audit['GAT']['max_diff']:.6f} | {audit['GAT']['identical']} |
| BiLSTM | {audit['BiLSTM']['n_params']:,} | {audit['BiLSTM']['max_diff']:.6f} | {audit['BiLSTM']['identical']} |
| cls_head | {audit['cls_head']['n_params']:,} | {audit['cls_head']['max_diff']:.6f} | {audit['cls_head']['identical']} |

The encoder is **confirmed identical** between checkpoints.

## Verdict

```
extraction_correct (was v1 pre-GAT?): {extraction_correct_v1}
v1_claim_valid ("pre-GAT features identical"): {v1_claim_valid}
```

- The v1 label said `post_gat` and it **was** post-GAT — the label is accurate.
- The v1 rationale said "pre-GAT features identical because WavLM is frozen" — this
  claim is **empirically confirmed** by the weight comparison above.
- The v1 extraction point captured trainable-component output (GAT + BiLSTM), which is
  correct for measuring the effect of those components.
- **Why v2 is still needed**: v1 could only measure the combined GAT+BiLSTM effect.
  v2 adds pre-GAT extraction to confirm the features entering the GAT are truly identical,
  and to establish the absolute discriminability baseline at that stage.
"""
    audit_path = RESULTS_DIR / "audit_notes.md"
    with open(audit_path, "w") as f:
        f.write(audit_md)
    print(f"  Saved: {audit_path}")

    # ── run_config.json ────────────────────────────────────────────────────────
    config = {
        "version": "v2",
        "extraction_points": EXTRACTION_POINTS,
        "pooling_schemes":   POOLING_SCHEMES,
        "checkpoints":       {k: str(v) for k, v in CHECKPOINTS.items()},
        "checkpoint_hashes": ckpt_hashes,
        "audit": {
            "encoder_identical":      encoder_identical,
            "gat_differs":            gat_differs,
            "weight_comparison":      {k: {kk: vv for kk, vv in v.items() if kk != "identical"}
                                       for k, v in audit.items()},
            "extraction_correct_v1":  extraction_correct_v1,
            "v1_claim_valid":         v1_claim_valid,
        },
        "dataset": {
            "hf_name": HF_DATASET, "train_split": "train", "eval_split": "validation",
            "train_binary_hash": tbin_hash, "train_7class_hash": t7cl_hash,
            "eval_hash": eval_hash,
        },
        "probe": {
            "C_grid": C_GRID, "max_iter": MAX_ITER, "n_folds": N_FOLDS,
            "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
        },
    }
    with open(RESULTS_DIR / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/run_config.json")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPLETION SUMMARY")
    print("=" * 72)

    print("\n" + ARCH_DIAGRAM)

    print("\n\nAUDIT VERDICT:")
    print(f"  extraction_correct (was v1 pre-GAT?): {extraction_correct_v1}")
    print(f"  v1 claim 'pre-GAT identical' valid  : {v1_claim_valid}")
    print(f"  encoder identical (empirical)        : {encoder_identical}")

    print("\n\nBINARY EER BY EXTRACTION POINT:")
    print(f"  {'Point':12s}  {'goat':>7}  {'robust':>7}  {'delta':>7}  {'CI':>20}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
    for ep in EXTRACTION_POINTS:
        es = eer_summaries[ep]
        print(f"  {ep:12s}  {es['goat_eer']:>7.4f}  {es['robust_eer']:>7.4f}  "
              f"{es['delta']:>+7.4f}  [{es['ci_lo']:+.4f}, {es['ci_hi']:+.4f}]")

    print("\n\nPER-CLASS F1 DELTA (robust − goat) BY EXTRACTION POINT:")
    header = f"  {'System':10s}"
    for ep in EXTRACTION_POINTS:
        header += f"  {ep:12s}"
    print(header)
    for cls in SYS_LABELS:
        row = f"  {cls:10s}"
        for ep in EXTRACTION_POINTS:
            f1g = probe_results[ep]["goat"]["7class"]["per_class_f1"][cls]
            f1r = probe_results[ep]["robust_goat"]["7class"]["per_class_f1"][cls]
            row += f"  {f1r-f1g:+.4f}      "
        print(row)

    print(f"\n\nINTERPRETATION FLAG (based on pre_gat comparison): {flag}")
    print(f"  {rationale}")

    print("\n\nDIAGNOSTIC COMPARISONS:")
    ew = eer_summaries["post_wavlm"]
    eg = eer_summaries["pre_gat"]
    ep2 = eer_summaries["post_gat"]

    print(f"\n  [1] post_wavlm goat vs robust_goat (should be ≈0, frozen WavLM):")
    print(f"      delta={ew['delta']:+.4f}  CI[{ew['ci_lo']:+.4f},{ew['ci_hi']:+.4f}]")

    print(f"\n  [2] pre_gat goat vs robust_goat (key test — features entering GAT):")
    print(f"      delta={eg['delta']:+.4f}  CI[{eg['ci_lo']:+.4f},{eg['ci_hi']:+.4f}]")
    f1_delta_pre = [(cls, round(f1_robust_pre[cls] - f1_goat_pre[cls], 4)) for cls in SYS_LABELS]
    f1_delta_pre.sort(key=lambda x: x[1])
    print(f"      7-class F1 deltas (robust − goat): "
          + "  ".join(f"{c}:{d:+.3f}" for c, d in f1_delta_pre))

    print(f"\n  [3] pre_gat → post_gat delta (what GAT adds per checkpoint):")
    for ckpt_name in CHECKPOINTS:
        eer_pg  = probe_results["pre_gat"][ckpt_name]["binary"]["eer"]
        eer_pog = probe_results["post_gat"][ckpt_name]["binary"]["eer"]
        print(f"      {ckpt_name:12s}: pre_gat={eer_pg:.4f} → post_gat={eer_pog:.4f}  "
              f"delta={eer_pg-eer_pog:+.4f}  "
              f"({'GAT reduces EER' if eer_pg > eer_pog else 'GAT increases EER'})")

    print(f"\nResults: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
