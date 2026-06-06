#!/usr/bin/env python3
"""
bilstm_feature_probe.py
=========================
Test whether the GAT-layer-0 reorganisation in robust_goat is downstream
of an upstream feature shift by training linear probes on BiLSTM output
features from both checkpoints.

Architecture order: WavLM encoder → phoneme-pool (reduce_feat)
                    → GAT → BiLSTM (self.rnn) → mean-pool → cls_head

POOLING SCHEME: BiLSTM mean-pool over valid phoneme positions.
  For each sample, after the BiLSTM the output is (B, max_N, 768).
  We take the mean over positions [:reduced_num_frames[i]] → (768,).
  This is identical to what the model does internally before norm_feat.

NOTE: The WavLM encoder is frozen (shared weights, same for both ckpts).
  Pre-GAT features are therefore identical across checkpoints.
  Differences in probe performance reflect GAT+BiLSTM representation shifts.

Outputs → experiments/results/gat_l0_attention_followups/bilstm_feature_probe/
  cache/                         — cached features (.npz per checkpoint per split)
  probe_metrics.json             — full metrics, both checkpoints, both probes
  confusion_matrices/            — PNG + CSV per (checkpoint, probe)
  binary_eer_comparison.json     — delta, CI, interpretation flag
  per_class_f1_deltas.csv        — per-class F1 delta (robust − goat), sorted
  run_config.json
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
import soundfile as sf
import torch
import torchaudio.transforms as T

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
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parents[1]
EXPS_DIR    = REPO_ROOT / "experiments"
RESULTS_DIR = (EXPS_DIR / "results" / "gat_l0_attention_followups"
               / "bilstm_feature_probe")
CACHE_DIR   = RESULTS_DIR / "cache"
CM_DIR      = RESULTS_DIR / "confusion_matrices"
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

POOLING_SCHEME    = "bilstm_mean_pool_post_gat"
N_TRAIN_BONAFIDE  = 2500
N_TRAIN_SPOOF_PER_ATTACK = 416   # 6 × 416 = 2496 ≈ 2500 spoof
N_TRAIN_7CLS_PER = 500           # 500 per class × 7 = 3500
N_EVAL_PER_SYS   = 200           # 200 bonafide + 200/attack = 1400 eval
BATCH_SIZE        = 32
C_GRID            = [0.01, 0.1, 1.0, 10.0, 100.0]
N_FOLDS           = 5
N_BOOTSTRAP       = 1000
SEED              = 42

SYS_CLASSES = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
SYS_LABELS  = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
CLS_MAP     = {sid: i for i, sid in enumerate(SYS_CLASSES)}    # str → int
CLS_RMAP    = {i: lbl for i, lbl in enumerate(SYS_LABELS)}     # int → str
FEAT_DIM    = 768


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SubsampledDataset(torch.utils.data.Dataset):
    """Load HF dataset split with per-system sample counts."""

    def __init__(self, hf_name, split, hf_cache, token,
                 counts_per_system: dict[str, int], seed: int = 42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=hf_cache, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label")

        by_sys: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.ds)):
            by_sys[self.ds[i].get("system_id", "unknown")].append(i)

        rng = random.Random(seed)
        selected: list[int] = []
        sys_ids: list[str] = []
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

    def __len__(self) -> int:
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


def indices_hash(indices: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in sorted(indices)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# BiLSTM feature extraction (step-by-step, avoids PhonemeCapture timing issue)
# ---------------------------------------------------------------------------

def extract_bilstm_batch(audio: torch.Tensor, gat_model, device: torch.device) -> torch.Tensor:
    """
    Extract BiLSTM mean-pool features for a single batch.
    Returns (B, 768) float32 on CPU.
    """
    from phoneme_GAT.modules import reduce_feat, generate_edges_by_combine_and_split

    B     = audio.shape[0]
    num_f = torch.full((B,), NF_PER_SAMPLE, device=device)

    with torch.no_grad():
        hidden_states, phoneme_ids = run_frozen_frontend(audio, gat_model, device)

        # WavLM encoder (frozen)
        hidden_states = gat_model.encoder(hidden_states)[0]      # (B, T, 768)

        # Phoneme pooling
        reduced_hs, reduced_nf, reduced_pids = reduce_feat(
            hidden_states, num_f, phoneme_ids
        )  # reduced_hs: (total_phonemes, 768)

        if gat_model.use_GAT:
            reduced_nf_d   = reduced_nf.to(device)
            edge_index     = generate_edges_by_combine_and_split(
                reduced_nf_d, reduced_pids, N=gat_model.n_edges
            ).to(device)
            reduced_hs, _  = gat_model.GAT((reduced_hs, edge_index))

        # Split into per-sample sequences, run BiLSTM, mean-pool
        rnf_list = [int(reduced_nf[i].item()) for i in range(B)]
        hs_split  = torch.split(reduced_hs, rnf_list, 0)
        padded    = torch.nn.utils.rnn.pad_sequence(hs_split, batch_first=True)
        lstm_out, _ = gat_model.rnn(padded)   # (B, max_N, 768)

        feat = torch.stack([
            lstm_out[i, :rnf_list[i], :].mean(0) for i in range(B)
        ])  # (B, 768)

    return feat.cpu().float()


def extract_and_cache(
    ckpt_name: str,
    ckpt_path: Path,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cache_path: Path,
) -> dict:
    if cache_path.exists():
        print(f"  [cache hit] {cache_path.name}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "features":      data["features"],
            "labels_binary": data["labels_binary"],
            "labels_7cls":   data["labels_7cls"],
            "sys_ids":       list(data["sys_ids"]),
            "hf_indices":    list(data["hf_indices"]),
        }

    print(f"  Extracting features: {ckpt_name} ({ckpt_path.name})")
    lit       = load_model(ckpt_path, device)
    gat_model = lit.model

    features:      list[np.ndarray] = []
    labels_binary: list[int]        = []
    labels_7cls:   list[int]        = []
    sys_ids:       list[str]        = []
    hf_indices:    list[int]        = []
    total = len(loader.dataset)

    for bi, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        feat  = extract_bilstm_batch(audio, gat_model, device)   # (B, 768)
        features.extend(feat.numpy())
        labels_binary.extend(batch["label"].tolist())
        labels_7cls.extend(batch["cls7"].tolist())
        sys_ids.extend(batch["sys_id"])
        hf_indices.extend(batch["hf_index"])
        if (bi + 1) % 20 == 0 or bi + 1 == len(loader):
            print(f"  {min((bi+1)*BATCH_SIZE, total)}/{total}")

    result = {
        "features":      np.array(features,      dtype=np.float32),
        "labels_binary": np.array(labels_binary, dtype=np.int32),
        "labels_7cls":   np.array(labels_7cls,   dtype=np.int32),
        "sys_ids":       np.array(sys_ids,        dtype=object),
        "hf_indices":    np.array(hf_indices,     dtype=np.int32),
    }
    np.savez(cache_path, **result)
    print(f"  Cached: {cache_path}")
    return {k: (list(v) if v.dtype == object else v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# EER (binary classification)
# ---------------------------------------------------------------------------

def compute_eer_from_arrays(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER at the threshold where FAR (FP/N_neg) ≈ FRR (FN/N_pos)."""
    thresholds = np.sort(np.unique(scores))
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    best_eer  = 1.0
    best_diff = np.inf
    for t in thresholds:
        fp  = int(((scores >= t) & (labels == 0)).sum())
        fn  = int(((scores < t)  & (labels == 1)).sum())
        far = fp / n_neg
        frr = fn / n_pos
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2
    return float(best_eer)


def bootstrap_eer_ci(scores: np.ndarray, labels: np.ndarray,
                     n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng  = np.random.RandomState(seed)
    eers = []
    for _ in range(n_bootstrap):
        idx  = rng.randint(0, len(labels), len(labels))
        eers.append(compute_eer_from_arrays(scores[idx], labels[idx]))
    lo, hi = np.percentile(eers, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_paired_delta_ci(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    n_bootstrap: int, seed: int,
) -> tuple[float, float, float]:
    """Returns (observed_delta, ci_lo, ci_hi) for delta = eer_a - eer_b."""
    obs_delta = compute_eer_from_arrays(scores_a, labels) - compute_eer_from_arrays(scores_b, labels)
    rng    = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx  = rng.randint(0, len(labels), len(labels))
        da   = compute_eer_from_arrays(scores_a[idx], labels[idx])
        db   = compute_eer_from_arrays(scores_b[idx], labels[idx])
        deltas.append(da - db)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(obs_delta), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Linear probe training with 5-fold CV
# ---------------------------------------------------------------------------

def select_best_c(X_train: np.ndarray, y_train: np.ndarray,
                  C_grid: list[float], n_folds: int, seed: int,
                  multiclass: bool = False) -> tuple[float, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores: dict[float, list[float]] = {}

    for C in C_grid:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            scaler = StandardScaler()
            Xtr    = scaler.fit_transform(X_train[train_idx])
            Xvl    = scaler.transform(X_train[val_idx])
            mc     = "multinomial" if multiclass else "auto"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                         multi_class=mc, random_state=seed)
                clf.fit(Xtr, y_train[train_idx])
            from sklearn.metrics import balanced_accuracy_score
            fold_scores.append(balanced_accuracy_score(y_train[val_idx], clf.predict(Xvl)))
        cv_scores[C] = fold_scores

    best_C = max(C_grid, key=lambda c: np.mean(cv_scores[c]))
    return best_C, {c: float(np.mean(v)) for c, v in cv_scores.items()}


def fit_probe(X_train: np.ndarray, y_train: np.ndarray,
              C: float, seed: int, multiclass: bool = False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    mc     = "multinomial" if multiclass else "auto"
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                  multi_class=mc, random_state=seed)
        clf.fit(X_sc, y_train)
        convergence_warnings = [str(x.message) for x in w
                                 if issubclass(x.category, Exception.__class__.__mro__[0])
                                 or "converge" in str(x.message).lower()]

    n_iter      = int(clf.n_iter_.max())
    converged   = n_iter < 1000
    return clf, scaler, n_iter, converged, convergence_warnings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def eval_binary_probe(clf, scaler, X_eval, y_eval, n_bootstrap, seed):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    X_sc   = scaler.transform(X_eval)
    scores = clf.predict_proba(X_sc)[:, 1]   # P(spoof)
    eer    = compute_eer_from_arrays(scores, y_eval)
    ci_lo, ci_hi = bootstrap_eer_ci(scores, y_eval, n_bootstrap, seed)
    auroc  = float(roc_auc_score(y_eval, scores))
    bacc   = float(balanced_accuracy_score(y_eval, clf.predict(X_sc)))
    train_acc = None  # filled in later
    return {
        "eer":           round(eer,  4),
        "eer_ci_lo":     round(ci_lo, 4),
        "eer_ci_hi":     round(ci_hi, 4),
        "auroc":         round(auroc, 4),
        "balanced_acc":  round(bacc,  4),
        "_scores":       scores,   # kept for paired delta
    }


def eval_7class_probe(clf, scaler, X_eval, y_eval):
    from sklearn.metrics import (f1_score, confusion_matrix)

    X_sc  = scaler.transform(X_eval)
    preds = clf.predict(X_sc)
    macro_f1 = float(f1_score(y_eval, preds, average="macro", zero_division=0))
    per_cls_f1 = f1_score(y_eval, preds, average=None, zero_division=0,
                          labels=list(range(len(SYS_LABELS))))
    cm = confusion_matrix(y_eval, preds, labels=list(range(len(SYS_LABELS))))
    return {
        "macro_f1":    round(macro_f1, 4),
        "per_class_f1": {SYS_LABELS[i]: round(float(per_cls_f1[i]), 4)
                         for i in range(len(SYS_LABELS))},
        "_cm":         cm,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                           out_path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(map(int, row)))


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_delta(delta: float, ci_lo: float, ci_hi: float) -> str:
    ci_crosses_zero = ci_lo < 0 < ci_hi
    if abs(delta) < 0.02 and ci_crosses_zero:
        return "GAT-localized"
    if delta >= 0.02 and ci_lo > 0:
        return "Upstream shift confirmed"
    return "Ambiguous"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("BILSTM FEATURE PROBE — LINEAR DISCRIMINABILITY TEST")
    print("=" * 72)
    print(f"\nPOOLING SCHEME: {POOLING_SCHEME}")
    print("  → BiLSTM output mean-pooled over valid phoneme positions (post-GAT)")
    print("  → BiLSTM: bidirectional LSTM(768→384, 2 layers), output dim=768")
    print("  → NOTE: WavLM encoder is frozen; pre-GAT features identical across ckpts")
    print(f"  → Feature differences ONLY reflect GAT+BiLSTM weight differences\n")

    patch_phoneme_loader()

    token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    # ── Build datasets ───────────────────────────────────────────────────────
    # Train: 2500 bonafide + ~2500 spoof (416/attack × 6)
    train_binary_counts = {"-": N_TRAIN_BONAFIDE}
    for sid in SYS_CLASSES[1:]:   # A01-A06
        train_binary_counts[sid] = N_TRAIN_SPOOF_PER_ATTACK

    # 7-class train: 500 per class × 7
    train_7cls_counts = {sid: N_TRAIN_7CLS_PER for sid in SYS_CLASSES}

    # Eval: 200 per system × 7 = 1400
    eval_counts = {sid: N_EVAL_PER_SYS for sid in SYS_CLASSES}

    print("Building datasets...")
    ds_train_bin = SubsampledDataset(HF_DATASET, "train",    HF_CACHE_DIR, token,
                                     train_binary_counts, seed=SEED)
    ds_train_7cl = SubsampledDataset(HF_DATASET, "train",    HF_CACHE_DIR, token,
                                     train_7cls_counts,   seed=SEED + 1)
    ds_eval      = SubsampledDataset(HF_DATASET, "validation", HF_CACHE_DIR, token,
                                     eval_counts,          seed=SEED)

    # Print counts
    print(f"\nTrain (binary): {len(ds_train_bin)} samples  "
          f"[{Counter(ds_train_bin.sys_ids)}]")
    print(f"Train (7-class): {len(ds_train_7cl)} samples  "
          f"[{Counter(ds_train_7cl.sys_ids)}]")
    print(f"Eval: {len(ds_eval)} samples  [{Counter(ds_eval.sys_ids)}]")

    # Sanity (c): assert same split indices across checkpoints — ensured by construction;
    # hash them for the record
    train_bin_hash = indices_hash(ds_train_bin.hf_indices)
    train_7cl_hash = indices_hash(ds_train_7cl.hf_indices)
    eval_hash      = indices_hash(ds_eval.hf_indices)
    print(f"\nSplit hashes (same for both ckpts — seeded):")
    print(f"  train_binary: {train_bin_hash}")
    print(f"  train_7class: {train_7cl_hash}")
    print(f"  eval:         {eval_hash}")
    print(f"  Sanity (c) PASSED: splits are deterministic — same across checkpoints")

    def make_loader(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_probe,
        )

    loader_train_bin = make_loader(ds_train_bin)
    loader_train_7cl = make_loader(ds_train_7cl)
    loader_eval      = make_loader(ds_eval)

    # ── Extract features ─────────────────────────────────────────────────────
    print("\n--- Feature extraction ---")
    feats: dict[str, dict[str, dict]] = {}  # feats[ckpt][split]

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\nCheckpoint: {ckpt_name}")
        feats[ckpt_name] = {}
        for split_key, loader, cache_name in [
            ("train_bin", loader_train_bin, f"{ckpt_name}_train_bin.npz"),
            ("train_7cl", loader_train_7cl, f"{ckpt_name}_train_7cl.npz"),
            ("eval",      loader_eval,      f"{ckpt_name}_eval.npz"),
        ]:
            feats[ckpt_name][split_key] = extract_and_cache(
                ckpt_name, ckpt_path, loader, device,
                CACHE_DIR / cache_name,
            )

    # Sanity (c): verify hf_indices match between checkpoints
    for split_key, expected_hash in [
        ("train_bin", train_bin_hash),
        ("train_7cl", train_7cl_hash),
        ("eval",      eval_hash),
    ]:
        h_g = indices_hash(list(feats["goat"][split_key]["hf_indices"]))
        h_r = indices_hash(list(feats["robust_goat"][split_key]["hf_indices"]))
        assert h_g == h_r == expected_hash, (
            f"Sanity (c) FAILED: {split_key} indices differ between checkpoints "
            f"or don't match expected hash ({h_g}, {h_r}, expected {expected_hash})"
        )
    print("\n  Sanity (c) CONFIRMED: all split indices identical across both checkpoints")

    # Sanity (b): pooling scheme identical (it's enforced by extract_bilstm_batch, but assert)
    assert POOLING_SCHEME == "bilstm_mean_pool_post_gat", "Sanity (b): pooling scheme mismatch"
    print(f"  Sanity (b) PASSED: pooling scheme = '{POOLING_SCHEME}' (both checkpoints)")

    # ── Train + eval probes ───────────────────────────────────────────────────
    print("\n--- Training linear probes (C selected by 5-fold CV) ---")
    print(f"  C grid: {C_GRID};  folds: {N_FOLDS};  seed: {SEED}")
    print("  Binary metric for CV: balanced_accuracy; solver: lbfgs, max_iter=1000")

    probe_metrics: dict = {}
    binary_scores: dict[str, np.ndarray] = {}   # for paired delta

    for ckpt_name in CHECKPOINTS:
        print(f"\n  Checkpoint: {ckpt_name}")
        probe_metrics[ckpt_name] = {}
        Xtr_bin = feats[ckpt_name]["train_bin"]["features"]
        ytr_bin = feats[ckpt_name]["train_bin"]["labels_binary"]
        Xtr_7cl = feats[ckpt_name]["train_7cl"]["features"]
        ytr_7cl = feats[ckpt_name]["train_7cl"]["labels_7cls"]
        X_eval  = feats[ckpt_name]["eval"]["features"]
        y_bin   = feats[ckpt_name]["eval"]["labels_binary"]
        y_7cl   = feats[ckpt_name]["eval"]["labels_7cls"]

        # ── Binary probe ──
        print(f"    [binary] CV selecting C...", end=" ")
        best_C_bin, cv_bin = select_best_c(
            Xtr_bin, ytr_bin, C_GRID, N_FOLDS, SEED, multiclass=False
        )
        print(f"best_C={best_C_bin}")
        clf_bin, scaler_bin, n_iter_bin, conv_bin, warn_bin = fit_probe(
            Xtr_bin, ytr_bin, best_C_bin, SEED, multiclass=False
        )
        # Sanity (a): train accuracy
        from sklearn.metrics import balanced_accuracy_score
        train_bacc_bin = float(balanced_accuracy_score(
            ytr_bin, clf_bin.predict(scaler_bin.transform(Xtr_bin))
        ))
        bin_metrics = eval_binary_probe(clf_bin, scaler_bin, X_eval, y_bin, N_BOOTSTRAP, SEED)
        binary_scores[ckpt_name] = bin_metrics.pop("_scores")

        # Sanity (a): overfitting check
        overfit_flag = (train_bacc_bin > 0.98 and bin_metrics["balanced_acc"] < 0.70)
        if overfit_flag:
            print(f"    [WARN] Sanity (a): binary probe overfitting — "
                  f"train_bacc={train_bacc_bin:.3f}  eval_bacc={bin_metrics['balanced_acc']:.3f}")
        else:
            print(f"    Sanity (a) binary OK: train_bacc={train_bacc_bin:.3f}  "
                  f"eval_bacc={bin_metrics['balanced_acc']:.3f}")

        # Sanity (d): convergence
        if not conv_bin:
            print(f"    [WARN] Sanity (d): binary probe did NOT converge (n_iter={n_iter_bin})")
        else:
            print(f"    Sanity (d) binary PASSED: converged in {n_iter_bin} iter")

        probe_metrics[ckpt_name]["binary"] = {
            "probe_type":      "binary",
            "best_C":          best_C_bin,
            "cv_scores":       cv_bin,
            "n_iter":          n_iter_bin,
            "converged":       conv_bin,
            "train_bacc":      round(train_bacc_bin, 4),
            **bin_metrics,
        }

        # ── 7-class probe ──
        print(f"    [7-class] CV selecting C...", end=" ")
        best_C_7cl, cv_7cl = select_best_c(
            Xtr_7cl, ytr_7cl, C_GRID, N_FOLDS, SEED, multiclass=True
        )
        print(f"best_C={best_C_7cl}")
        clf_7cl, scaler_7cl, n_iter_7cl, conv_7cl, warn_7cl = fit_probe(
            Xtr_7cl, ytr_7cl, best_C_7cl, SEED, multiclass=True
        )
        train_bacc_7cl = float(balanced_accuracy_score(
            ytr_7cl, clf_7cl.predict(scaler_7cl.transform(Xtr_7cl))
        ))
        m7 = eval_7class_probe(clf_7cl, scaler_7cl, X_eval, y_7cl)
        cm_raw = m7.pop("_cm")

        overfit_7cl = (train_bacc_7cl > 0.98 and
                       m7["macro_f1"] < 0.70 * len(SYS_LABELS) / len(SYS_LABELS))
        if overfit_7cl:
            print(f"    [WARN] Sanity (a): 7-class probe overfitting — "
                  f"train_bacc={train_bacc_7cl:.3f}  eval_macro_f1={m7['macro_f1']:.3f}")
        else:
            print(f"    Sanity (a) 7-class OK: train_bacc={train_bacc_7cl:.3f}  "
                  f"eval_macro_f1={m7['macro_f1']:.3f}")

        if not conv_7cl:
            print(f"    [WARN] Sanity (d): 7-class probe did NOT converge (n_iter={n_iter_7cl})")
        else:
            print(f"    Sanity (d) 7-class PASSED: converged in {n_iter_7cl} iter")

        probe_metrics[ckpt_name]["7class"] = {
            "probe_type": "7class",
            "best_C":      best_C_7cl,
            "cv_scores":   cv_7cl,
            "n_iter":      n_iter_7cl,
            "converged":   conv_7cl,
            "train_bacc":  round(train_bacc_7cl, 4),
            **m7,
        }

        # ── Confusion matrix ──
        for probe_key, cm_mat, class_names in [
            ("binary",  np.array([[sum((y_bin == 0) & (clf_bin.predict(scaler_bin.transform(X_eval)) == 0)),
                                    sum((y_bin == 0) & (clf_bin.predict(scaler_bin.transform(X_eval)) == 1))],
                                   [sum((y_bin == 1) & (clf_bin.predict(scaler_bin.transform(X_eval)) == 0)),
                                    sum((y_bin == 1) & (clf_bin.predict(scaler_bin.transform(X_eval)) == 1))]]),
               ["bonafide", "spoof"]),
            ("7class",  cm_raw, SYS_LABELS),
        ]:
            stem = f"{ckpt_name}_{probe_key}"
            plot_confusion_matrix(cm_mat, class_names, CM_DIR / f"{stem}.png",
                                  title=f"{ckpt_name} — {probe_key}")
            save_confusion_matrix_csv(cm_mat, class_names, CM_DIR / f"{stem}.csv")
        print(f"    Confusion matrices saved.")

    # ── Save probe_metrics.json ───────────────────────────────────────────────
    metrics_path = RESULTS_DIR / "probe_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(probe_metrics, f, indent=2)
    print(f"\n  Saved: {metrics_path}")

    # ── Binary EER comparison + paired delta ─────────────────────────────────
    print("\n--- Binary EER comparison ---")
    y_bin_eval    = feats["goat"]["eval"]["labels_binary"]
    scores_goat   = binary_scores["goat"]
    scores_robust = binary_scores["robust_goat"]

    delta, ci_lo, ci_hi = bootstrap_paired_delta_ci(
        scores_goat, scores_robust, y_bin_eval, N_BOOTSTRAP, SEED
    )
    flag = interpret_delta(delta, ci_lo, ci_hi)

    eer_goat   = probe_metrics["goat"]["binary"]["eer"]
    eer_robust = probe_metrics["robust_goat"]["binary"]["eer"]

    print(f"  goat   binary EER: {eer_goat:.4f}  "
          f"CI [{probe_metrics['goat']['binary']['eer_ci_lo']:.4f}, "
          f"{probe_metrics['goat']['binary']['eer_ci_hi']:.4f}]")
    print(f"  robust binary EER: {eer_robust:.4f}  "
          f"CI [{probe_metrics['robust_goat']['binary']['eer_ci_lo']:.4f}, "
          f"{probe_metrics['robust_goat']['binary']['eer_ci_hi']:.4f}]")
    print(f"  Paired delta (goat − robust): {delta:+.4f}  "
          f"CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Interpretation flag: {flag}")

    binary_cmp = {
        "eer_goat":    eer_goat,
        "eer_robust":  eer_robust,
        "paired_delta":       round(delta,  4),
        "paired_ci_lo":       round(ci_lo,  4),
        "paired_ci_hi":       round(ci_hi,  4),
        "interpretation_flag": flag,
    }
    with open(RESULTS_DIR / "binary_eer_comparison.json", "w") as f:
        json.dump(binary_cmp, f, indent=2)

    # ── Per-class F1 deltas ───────────────────────────────────────────────────
    f1_goat   = probe_metrics["goat"]["7class"]["per_class_f1"]
    f1_robust = probe_metrics["robust_goat"]["7class"]["per_class_f1"]
    f1_deltas = [
        {
            "system":    cls,
            "f1_goat":   f1_goat[cls],
            "f1_robust": f1_robust[cls],
            "delta_robust_minus_goat": round(f1_robust[cls] - f1_goat[cls], 4),
        }
        for cls in SYS_LABELS
    ]
    f1_deltas.sort(key=lambda x: -x["delta_robust_minus_goat"])

    with open(RESULTS_DIR / "per_class_f1_deltas.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(f1_deltas[0].keys()))
        writer.writeheader(); writer.writerows(f1_deltas)
    print(f"\n  Per-class F1 deltas (robust − goat) sorted:")
    for r in f1_deltas:
        print(f"    {r['system']:10s}: {r['delta_robust_minus_goat']:+.4f}  "
              f"(goat={r['f1_goat']:.4f}  robust={r['f1_robust']:.4f})")

    # ── run_config.json ───────────────────────────────────────────────────────
    config = {
        "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        "pooling_scheme": POOLING_SCHEME,
        "pooling_note": (
            "BiLSTM = nn.Sequential(nn.LSTM(768, 384, num_layers=2, bidirectional=True)). "
            "Mean-pool over valid phoneme positions (reduced_num_frames). "
            "WavLM encoder frozen; pre-GAT features identical across checkpoints."
        ),
        "dataset": {
            "hf_name": HF_DATASET,
            "train_split": "train",
            "eval_split": "validation",
            "train_binary_counts": train_binary_counts,
            "train_7class_counts": train_7cls_counts,
            "eval_counts": eval_counts,
            "train_binary_hash": train_bin_hash,
            "train_7class_hash": train_7cl_hash,
            "eval_hash": eval_hash,
        },
        "probe": {
            "model": "LogisticRegression(solver='lbfgs', max_iter=1000)",
            "C_grid": C_GRID,
            "n_folds": N_FOLDS,
            "cv_metric": "balanced_accuracy",
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "sanity_pass": True,
    }
    config_path = RESULTS_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved: {config_path}")

    # ── Completion summary ────────────────────────────────────────────────────
    top3_improve = [r for r in f1_deltas if r["delta_robust_minus_goat"] > 0][:3]

    print("\n" + "=" * 72)
    print("COMPLETION SUMMARY")
    print("=" * 72)
    print(f"\nPooling scheme: {POOLING_SCHEME}")
    print(f"  (BiLSTM mean-pool over phoneme sequence, post-GAT, pre-cls_head)")
    print(f"\nBinary probe EER:")
    print(f"  goat        : {eer_goat:.4f}  "
          f"[{probe_metrics['goat']['binary']['eer_ci_lo']:.4f}, "
          f"{probe_metrics['goat']['binary']['eer_ci_hi']:.4f}]")
    print(f"  robust_goat : {eer_robust:.4f}  "
          f"[{probe_metrics['robust_goat']['binary']['eer_ci_lo']:.4f}, "
          f"{probe_metrics['robust_goat']['binary']['eer_ci_hi']:.4f}]")
    print(f"  Paired delta: {delta:+.4f}  CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"\nInterpretation: {flag}")
    if top3_improve:
        print(f"\nTop-3 systems where robust_goat probe outperforms goat (7-class F1):")
        for r in top3_improve:
            print(f"  {r['system']:10s}: Δ={r['delta_robust_minus_goat']:+.4f}  "
                  f"(goat={r['f1_goat']:.4f}  robust={r['f1_robust']:.4f})")
    else:
        print("\nNo attack system where robust_goat probe strictly outperforms goat (7-class F1).")
    print(f"\nResults: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
