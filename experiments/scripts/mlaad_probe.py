#!/usr/bin/env python3
"""
mlaad_probe.py
==============
Three-point linear probe on MLAAD checkpoints.

Extracts features at three pipeline points for both checkpoints:
  post_wavlm  — after feature_extractor + feature_projection (before encoder)
                Frozen → features identical across checkpoints
  pre_gat     — after encoder + phoneme pooling (encoder is frozen/identical)
                Should also be identical → confirms gat_localized hypothesis
  post_gat    — after GAT + BiLSTM mean-pool (trainable; checkpoint-specific)

Trains a binary logistic regression probe at each extraction point × checkpoint.
Evaluates on both test sets (in-distribution and cross-language).

Usage:
    python experiments/scripts/mlaad_probe.py

Reads:
    experiments/data/mlaad_tiny_processed/splits/train.json
    experiments/results/mlaad/baseline_eval/test_{in_distribution,cross_language}.json
    experiments/checkpoints/mlaad_{goat,robust_goat}-best-*.ckpt

Writes:
    experiments/results/mlaad/probe_three_point/
        metrics_by_extraction_point.json
        in_distribution_comparison.csv
        cross_language_comparison.csv
        cross_language_generalization.json
        interpretation_flags.json
        feature_cache/   — cached .npz files (skipped on re-run)
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
CKPT_DIR      = EXP_DIR / "checkpoints"
BASELINE_DIR  = EXP_DIR / "results" / "mlaad" / "baseline_eval"
OUT_BASE      = EXP_DIR / "results" / "mlaad" / "probe_three_point"
CACHE_DIR     = OUT_BASE / "feature_cache"

for _p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from head_ablation import (
    load_model, run_frozen_frontend, patch_phoneme_loader, NF_PER_SAMPLE,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED              = 42
BATCH_SIZE        = 32
N_TRAIN_BONAFIDE  = 2500
N_TRAIN_SPOOF     = 2500
C_GRID            = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER          = 2000
N_FOLDS           = 5
N_BOOTSTRAP       = 1000
FEAT_DIM          = 768
EXTRACTION_POINTS = ["post_wavlm", "pre_gat", "post_gat"]
CHECKPOINTS       = ["mlaad_goat", "mlaad_robust_goat"]


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))

def _best_or_last(stem: str) -> Path:
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else CKPT_DIR / f"{stem}.ckpt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class MAALDProbeDataset(Dataset):
    def __init__(self, records: list[dict], processed_dir: Path) -> None:
        self.records       = records
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(self.processed_dir / rec["audio_path"])
        wav = wav.unsqueeze(0)
        y   = 0 if rec["label"] == "bonafide" else 1
        sid = "-" if rec["label"] == "bonafide" else rec.get("attack_system", "unknown")
        return {"audio": wav, "label": torch.tensor(y, dtype=torch.long), "system_id": sid}


def _collate(batch):
    return {
        "audio":     torch.stack([b["audio"]  for b in batch]),
        "label":     torch.stack([b["label"]  for b in batch]),
        "system_id": [b["system_id"] for b in batch],
    }

def make_loader(records):
    return DataLoader(MAALDProbeDataset(records, PROCESSED_DIR),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=_collate)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_post_wavlm(audio: torch.Tensor, gat_model, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        hs, _ = run_frozen_frontend(audio, gat_model, device)
        return hs.mean(dim=1).cpu().float()


def extract_pre_gat(audio: torch.Tensor, gat_model, device: torch.device) -> torch.Tensor:
    from phoneme_GAT.modules import reduce_feat
    B     = audio.shape[0]
    num_f = torch.full((B,), NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        hs, pids = run_frozen_frontend(audio, gat_model, device)
        hs       = gat_model.encoder(hs)[0]
        rhs, rnf, _ = reduce_feat(hs, num_f, pids)
        rnf_list = [int(rnf[i].item()) for i in range(B)]
        splits   = torch.split(rhs, rnf_list, 0)
        feat     = torch.stack([s.mean(0) for s in splits])
    return feat.cpu().float()


def extract_post_gat(audio: torch.Tensor, gat_model, device: torch.device) -> torch.Tensor:
    from phoneme_GAT.modules import reduce_feat, generate_edges_by_combine_and_split
    B     = audio.shape[0]
    num_f = torch.full((B,), NF_PER_SAMPLE, device=device)
    with torch.no_grad():
        hs, pids = run_frozen_frontend(audio, gat_model, device)
        hs = gat_model.encoder(hs)[0]
        rhs, rnf, rpids = reduce_feat(hs, num_f, pids)
        if gat_model.use_GAT:
            rnf_d  = rnf.to(device)
            ei     = generate_edges_by_combine_and_split(
                rnf_d, rpids, N=gat_model.n_edges).to(device)
            rhs, _ = gat_model.GAT((rhs, ei))
        rnf_list = [int(rnf[i].item()) for i in range(B)]
        splits   = torch.split(rhs, rnf_list, 0)
        padded   = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)
        out, _   = gat_model.rnn(padded)
        feat     = torch.stack([out[i, :rnf_list[i], :].mean(0) for i in range(B)])
    return feat.cpu().float()


EXTRACT_FN = {
    "post_wavlm": extract_post_wavlm,
    "pre_gat":    extract_pre_gat,
    "post_gat":   extract_post_gat,
}


def load_or_extract(
    ext_point: str,
    ckpt_name: str,
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (features (N, 768), labels (N,)) as float32 and int32 arrays.
    Caches to CACHE_DIR for re-runs.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ext_point}_{ckpt_name}_{split_name}.npz"

    if cache_path.exists():
        print(f"    [cache hit] {cache_path.name}")
        data = np.load(cache_path)
        return data["features"], data["labels"]

    print(f"    Extracting {ext_point} for {ckpt_name} on {split_name}...")
    patch_phoneme_loader()
    lit       = load_model(ckpt_path, device)
    gat_model = lit.model
    fn        = EXTRACT_FN[ext_point]

    feats:  list[np.ndarray] = []
    labels: list[int]        = []
    total = len(loader.dataset)

    for bi, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        feat  = fn(audio, gat_model, device)
        feats.extend(feat.numpy())
        labels.extend(batch["label"].tolist())
        if (bi + 1) % 20 == 0 or bi + 1 == len(loader):
            print(f"      {min((bi+1)*BATCH_SIZE, total)}/{total}")

    X = np.array(feats,  dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    np.savez(cache_path, features=X, labels=y)
    print(f"      Cached → {cache_path.name}")
    return X, y


# ── EER ───────────────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.sort(np.unique(scores))
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    best_eer, best_diff = 1.0, np.inf
    for t in thresholds:
        fp = int(((scores >= t) & (labels == 0)).sum())
        fn = int(((scores <  t) & (labels == 1)).sum())
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
    eers = []
    for _ in range(n):
        idx = rng.randint(0, len(labels), len(labels))
        eers.append(compute_eer(scores[idx], labels[idx]))
    lo, hi = np.percentile(eers, [2.5, 97.5])
    return float(lo), float(hi)


# ── Probe ─────────────────────────────────────────────────────────────────────

def select_best_c(X_train, y_train, C_grid, n_folds, seed) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = {C: [] for C in C_grid}
    for C in C_grid:
        for tr, vl in kf.split(X_train, y_train):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X_train[tr])
            Xvl = sc.transform(X_train[vl])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                                         random_state=seed)
                clf.fit(Xtr, y_train[tr])
            scores[C].append(balanced_accuracy_score(y_train[vl], clf.predict(Xvl)))
    return max(C_grid, key=lambda c: np.mean(scores[c]))


def fit_probe(X_train, y_train, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, random_state=seed)
        clf.fit(X_sc, y_train)
    return clf, scaler


def eval_probe(clf, scaler, X_eval, y_eval, n_boot, seed) -> dict:
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    X_sc   = scaler.transform(X_eval)
    scores = clf.predict_proba(X_sc)[:, 1]
    eer    = compute_eer(scores, y_eval)
    ci_lo, ci_hi = bootstrap_eer_ci(scores, y_eval, n_boot, seed)
    auroc  = float(roc_auc_score(y_eval, scores)) if len(np.unique(y_eval)) > 1 else float("nan")
    bacc   = float(balanced_accuracy_score(y_eval, clf.predict(X_sc)))
    return {
        "eer":          round(eer,  4),
        "eer_ci_lo":    round(ci_lo, 4),
        "eer_ci_hi":    round(ci_hi, 4),
        "auroc":        round(auroc, 4),
        "balanced_acc": round(bacc, 4),
    }


# ── Train subsampler ──────────────────────────────────────────────────────────

def subsample_train(records: list[dict], n_bon: int, n_sp: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    bon   = [r for r in records if r["label"] == "bonafide"]
    spoof = [r for r in records if r["label"] != "bonafide"]
    rng.shuffle(bon);  rng.shuffle(spoof)
    return bon[:n_bon] + spoof[:n_sp]


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret(
    pre_gat_eer_goat:   float,
    pre_gat_eer_robust: float,
    pre_gat_ci_goat:    tuple[float, float],
    pre_gat_ci_robust:  tuple[float, float],
) -> dict:
    delta = pre_gat_eer_robust - pre_gat_eer_goat
    ci_overlap = (pre_gat_ci_goat[0] <= pre_gat_ci_robust[1]
                  and pre_gat_ci_robust[0] <= pre_gat_ci_goat[1])
    if abs(delta) < 1e-4 and ci_overlap:
        flag = "gat_localized"
        rationale = (
            f"pre_gat EER identical (delta={delta:+.4f}): encoder is frozen, "
            "confirming all representation changes are post-encoder (GAT/BiLSTM)."
        )
    elif not ci_overlap and delta > 0.02:
        flag = "upstream_localized"
        rationale = (
            f"robust_goat pre_gat EER higher by {delta:.4f} with non-overlapping CI: "
            "robust_goat upstream representations are less discriminative."
        )
    elif ci_overlap and abs(delta) < 0.02:
        flag = "gat_localized"
        rationale = (
            f"pre_gat EER delta={delta:+.4f} with overlapping CI: pre-GAT features "
            "are not significantly different; differences localized to GAT/BiLSTM."
        )
    else:
        flag = "ambiguous"
        rationale = (
            f"pre_gat delta={delta:+.4f}; neither clearly gat_localized "
            "nor upstream_localized."
        )
    return {"flag": flag, "rationale": rationale,
            "pre_gat_eer_goat": round(pre_gat_eer_goat, 4),
            "pre_gat_eer_robust": round(pre_gat_eer_robust, 4),
            "delta": round(delta, 4),
            "ci_overlap": ci_overlap}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load records ──────────────────────────────────────────────────────────
    train_all  = json.loads((PROCESSED_DIR / "splits/train.json").read_text())
    in_dist    = json.loads((BASELINE_DIR  / "test_in_distribution.json").read_text())
    cross_lang = json.loads((BASELINE_DIR  / "test_cross_language.json").read_text())

    train_sub = subsample_train(train_all, N_TRAIN_BONAFIDE, N_TRAIN_SPOOF, SEED)
    print(f"\nData:")
    print(f"  train (subsampled): {len(train_sub)} "
          f"({sum(1 for r in train_sub if r['label']=='bonafide')} bon + "
          f"{sum(1 for r in train_sub if r['label']!='bonafide')} spoof)")
    print(f"  in_distribution:   {len(in_dist)}")
    print(f"  cross_language:    {len(cross_lang)}")

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_paths = {
        "mlaad_goat":        _best_or_last("mlaad_goat"),
        "mlaad_robust_goat": _best_or_last("mlaad_robust_goat"),
    }
    for name, path in ckpt_paths.items():
        if not path.exists():
            print(f"ERROR: checkpoint not found: {path}")
            sys.exit(1)
        print(f"  {name}: {path.name}")

    # ── Loaders ───────────────────────────────────────────────────────────────
    loader_train = make_loader(train_sub)
    loader_in    = make_loader(in_dist)
    loader_xl    = make_loader(cross_lang)

    # ── Extract features for all (ext_point, ckpt, split) combos ─────────────
    feats: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    # feats[ext_point][ckpt_name][split_name] = (X, y)

    for ext_point in EXTRACTION_POINTS:
        feats[ext_point] = {}
        for ckpt_name, ckpt_path in ckpt_paths.items():
            feats[ext_point][ckpt_name] = {}
            print(f"\n  [{ext_point}|{ckpt_name}]")
            for split_name, loader in [
                ("train",          loader_train),
                ("in_distribution", loader_in),
                ("cross_language",  loader_xl),
            ]:
                X, y = load_or_extract(ext_point, ckpt_name, ckpt_path,
                                       loader, device, split_name)
                feats[ext_point][ckpt_name][split_name] = (X, y)

    # ── Train probes + evaluate ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training probes and evaluating...")

    # metrics[ext_point][ckpt_name][split] = {eer, eer_ci_lo, eer_ci_hi, auroc, balanced_acc}
    metrics: dict = {}
    probes:  dict = {}  # (ext_point, ckpt_name) → (clf, scaler)

    for ext_point in EXTRACTION_POINTS:
        metrics[ext_point] = {}
        for ckpt_name in CHECKPOINTS:
            X_tr, y_tr = feats[ext_point][ckpt_name]["train"]

            print(f"\n  [{ext_point}|{ckpt_name}] CV C selection...", end=" ")
            best_C = select_best_c(X_tr, y_tr, C_GRID, N_FOLDS, SEED)
            print(f"best_C={best_C}")

            clf, scaler = fit_probe(X_tr, y_tr, best_C, SEED)
            probes[(ext_point, ckpt_name)] = (clf, scaler)

            metrics[ext_point][ckpt_name] = {}
            for split_name in ("in_distribution", "cross_language"):
                X_ev, y_ev = feats[ext_point][ckpt_name][split_name]
                m = eval_probe(clf, scaler, X_ev, y_ev, N_BOOTSTRAP, SEED)
                metrics[ext_point][ckpt_name][split_name] = m
                print(f"    {split_name:20s}  EER={m['eer']:.4f} "
                      f"[{m['eer_ci_lo']:.4f},{m['eer_ci_hi']:.4f}]  "
                      f"AUROC={m['auroc']:.4f}")

    # ── metrics_by_extraction_point.json ─────────────────────────────────────
    _dump(metrics, OUT_BASE / "metrics_by_extraction_point.json")
    print(f"\nWrote: {OUT_BASE}/metrics_by_extraction_point.json")

    # ── in_distribution_comparison.csv ───────────────────────────────────────
    in_dist_rows = []
    for ext_point in EXTRACTION_POINTS:
        for ckpt_name in CHECKPOINTS:
            m  = metrics[ext_point][ckpt_name]["in_distribution"]
            in_dist_rows.append({
                "extraction_point": ext_point,
                "checkpoint":       ckpt_name,
                **m,
            })

    _write_csv(in_dist_rows, OUT_BASE / "in_distribution_comparison.csv")
    print(f"Wrote: {OUT_BASE}/in_distribution_comparison.csv")

    # ── cross_language_comparison.csv ────────────────────────────────────────
    xl_rows = []
    for ext_point in EXTRACTION_POINTS:
        for ckpt_name in CHECKPOINTS:
            m = metrics[ext_point][ckpt_name]["cross_language"]
            xl_rows.append({
                "extraction_point": ext_point,
                "checkpoint":       ckpt_name,
                **m,
            })

    _write_csv(xl_rows, OUT_BASE / "cross_language_comparison.csv")
    print(f"Wrote: {OUT_BASE}/cross_language_comparison.csv")

    # ── cross_language_generalization.json ───────────────────────────────────
    xl_gen: dict = {}
    for ext_point in EXTRACTION_POINTS:
        xl_gen[ext_point] = {}
        for ckpt_name in CHECKPOINTS:
            m_in = metrics[ext_point][ckpt_name]["in_distribution"]
            m_xl = metrics[ext_point][ckpt_name]["cross_language"]
            xl_gen[ext_point][ckpt_name] = {
                "in_dist_eer":    m_in["eer"],
                "cross_lang_eer": m_xl["eer"],
                "gap":            round(m_xl["eer"] - m_in["eer"], 4),
            }

    _dump(xl_gen, OUT_BASE / "cross_language_generalization.json")
    print(f"Wrote: {OUT_BASE}/cross_language_generalization.json")

    # ── interpretation_flags.json ─────────────────────────────────────────────
    pre_gat_goat   = metrics["pre_gat"]["mlaad_goat"]["in_distribution"]
    pre_gat_robust = metrics["pre_gat"]["mlaad_robust_goat"]["in_distribution"]
    post_gat_goat  = metrics["post_gat"]["mlaad_goat"]["in_distribution"]
    post_gat_robust= metrics["post_gat"]["mlaad_robust_goat"]["in_distribution"]

    interpretation = interpret(
        pre_gat_eer_goat   = pre_gat_goat["eer"],
        pre_gat_eer_robust = pre_gat_robust["eer"],
        pre_gat_ci_goat    = (pre_gat_goat["eer_ci_lo"],   pre_gat_goat["eer_ci_hi"]),
        pre_gat_ci_robust  = (pre_gat_robust["eer_ci_lo"], pre_gat_robust["eer_ci_hi"]),
    )
    flags = {
        "interpretation":    interpretation,
        "post_gat_delta_eer": round(
            post_gat_robust["eer"] - post_gat_goat["eer"], 4),
        "post_gat_delta_auroc": round(
            post_gat_robust["auroc"] - post_gat_goat["auroc"], 4),
        "note": (
            "Positive delta_eer = robust_goat is harder to linearly separate "
            "(less discriminative post-GAT features). Negative delta = easier."
        ),
    }
    _dump(flags, OUT_BASE / "interpretation_flags.json")
    print(f"Wrote: {OUT_BASE}/interpretation_flags.json")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MLAAD PROBE SUMMARY (in-distribution)")
    print(f"{'ext_point':14s}  {'checkpoint':22s}  {'EER':>8s}  "
          f"{'AUROC':>8s}  {'xl_gap':>8s}")
    print("-" * 68)
    for ext_point in EXTRACTION_POINTS:
        for ckpt_name in CHECKPOINTS:
            m  = metrics[ext_point][ckpt_name]["in_distribution"]
            xl = xl_gen[ext_point][ckpt_name]
            print(f"  {ext_point:14s}  {ckpt_name:22s}  "
                  f"{m['eer']:8.4f}  {m['auroc']:8.4f}  {xl['gap']:+8.4f}")

    print(f"\nInterpretation: {interpretation['flag']}")
    print(f"  {interpretation['rationale']}")
    print(f"\nOutput: {OUT_BASE}/")
    print(f"{'='*70}")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
