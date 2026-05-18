#!/usr/bin/env python3
"""
head_ablation.py
================
Head ablation experiment at GAT layer 0.

Tests whether h0 and h4 (identified as high-KL heads in gat_l0_attention.py)
causally carry the attack-detection circuit.

Ablation: patches GATLayer.neighborhood_aware_softmax on layer 0 only.
The skip connection (skip_proj) is intentionally left intact — this tests
whether learned attention ROUTING (not the full head slot) is what carries
the discriminative signal. The floor check ({all 6} uniform) will therefore
not reach chance; that residual performance reflects the skip path.

Ablation modes:
  uniform      — replace ablated heads' attention with 1/in-degree per target node
  zero         — set ablated heads' attention to 0
  bonafide_mean — replace with global per-head mean attention over bonafide samples
                  (loaded from existing gat_l0_attention artifacts)

Configs (evaluated in order):
  baseline      {}            uniform   — no ablation, identity check
  h0            {0}           uniform
  h4            {4}           uniform
  h0_h4         {0,4}         uniform   — primary discriminative pair
  h0_h4_zero    {0,4}         zero      — secondary mode check
  h0_h4_bfmean  {0,4}         bonafide_mean — secondary mode check
  ctrl_h1235    {1,2,3,5}     uniform   — non-discriminative heads, control
  all_uniform   {0,1,2,3,4,5} uniform   — floor (skip path still active)

Outputs → experiments/results/gat_l0_attention_followups/
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import textwrap
import types
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
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
REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "gat_l0_attention_followups"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ATTN_ARTIFACTS = (Path(__file__).resolve().parent / "results"
                  / "gat_l0_attention" / "attention_artifacts.pt")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT          = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # 149
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42

# Ablation configurations: (name, heads_frozenset, mode)
CONFIGS: list[tuple[str, frozenset, str]] = [
    ("baseline",      frozenset(),          "uniform"),
    ("h0",            frozenset({0}),        "uniform"),
    ("h4",            frozenset({4}),        "uniform"),
    ("h0_h4",         frozenset({0, 4}),     "uniform"),
    ("h0_h4_zero",    frozenset({0, 4}),     "zero"),
    ("h0_h4_bfmean",  frozenset({0, 4}),     "bonafide_mean"),
    ("ctrl_h1235",    frozenset({1,2,3,5}),  "uniform"),
    ("all_uniform",   frozenset(range(6)),   "uniform"),
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode(entry: dict) -> torch.Tensor:
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else sf.read(path, dtype="float32", always_2d=False))
    w = torch.tensor(arr)
    if w.ndim == 1: w = w.unsqueeze(0)
    elif w.ndim == 2: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR: w = T.Resample(sr, TARGET_SR)(w)
    return w

def _crop(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    if n < TARGET_SAMPLES: w = w.repeat(1, -(-TARGET_SAMPLES // n))
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s : s + TARGET_SAMPLES]

def _lbl(raw) -> int:
    if isinstance(raw, str):
        s = raw.strip().lower()
        return 0 if s in ("0", "bonafide", "real", "genuine") else 1
    return int(raw)


# ---------------------------------------------------------------------------
# Balanced dataset
# ---------------------------------------------------------------------------

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, n_per_class, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label")
        by_system: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.ds)):
            sid = self.ds[i].get("system_id", "unknown")
            by_system[sid].append(i)
        rng = random.Random(seed)
        selected, sys_ids = [], []
        for sid, idxs in sorted(by_system.items()):
            rng.shuffle(idxs)
            chosen = idxs[:n_per_class]
            selected.extend(chosen)
            sys_ids.extend([sid] * len(chosen))
        combined = list(zip(selected, sys_ids))
        rng.shuffle(combined)
        self.indices, self.sys_ids = zip(*combined) if combined else ([], [])
        self.indices = list(self.indices); self.sys_ids = list(self.sys_ids)
        print(f"\nDataset: {len(self.indices)} samples across {len(by_system)} systems")
        for sid in sorted(by_system):
            cnt = sum(1 for s in self.sys_ids if s == sid)
            print(f"  {sid:8s}: {cnt}")

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


# ---------------------------------------------------------------------------
# Phoneme loader patch
# ---------------------------------------------------------------------------

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit


# ---------------------------------------------------------------------------
# Frozen frontend — SpecAugment completely off
# ---------------------------------------------------------------------------

def run_frozen_frontend(audio: torch.Tensor, gat_model, device: torch.device):
    """Run WavLM pipeline without _mask_hidden_states. Deterministic."""
    x = audio
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0, :]
    with torch.no_grad():
        feat1 = gat_model.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hidden_states, _ = gat_model.transformer_in_phoneme_model.feature_projection(feat1)
        phoneme_feat = gat_model.transformer_in_phoneme_model.encoder(hidden_states)[0]
        phoneme_logits = gat_model.phoneme_model.model.model.lm_head(phoneme_feat)
        phoneme_ids = torch.argmax(phoneme_logits, dim=-1)
    return hidden_states, phoneme_ids


# ---------------------------------------------------------------------------
# Bonafide mean attention loader
# ---------------------------------------------------------------------------

def load_bonafide_means(artifacts_path: Path) -> dict[int, float]:
    """
    Load per-head global mean attention from existing gat_l0_attention artifacts.
    Returns {head_idx: mean_attn_weight} where mean is over all bonafide edges.
    These are used as replacement values in bonafide_mean mode.
    """
    if not artifacts_path.exists():
        print(f"  [WARN] Artifacts not found at {artifacts_path}, using uniform 1/3 fallback")
        return {h: 1/3 for h in range(6)}

    data = torch.load(artifacts_path)
    labels  = data["labels"]
    attn_l0 = data["attn_l0"]   # list of (E_i, NH) tensors

    bon_tensors = [attn_l0[i] for i, lbl in enumerate(labels) if lbl == 0]
    if not bon_tensors:
        print("  [WARN] No bonafide samples in artifacts, using 1/3 fallback")
        return {h: 1/3 for h in range(6)}

    all_bon = torch.cat(bon_tensors, dim=0)   # (E_total_bonafide, NH)
    NH = all_bon.shape[1]
    means = {h: float(all_bon[:, h].mean().item()) for h in range(NH)}
    print(f"  Bonafide means per head: "
          + "  ".join(f"h{h}={means[h]:.4f}" for h in range(NH)))
    return means


# ---------------------------------------------------------------------------
# Ablation hook — patches neighborhood_aware_softmax on layer 0 instance only
# ---------------------------------------------------------------------------

def install_ablation(layer0, heads: frozenset, mode: str,
                     bonafide_means: dict[int, float] | None = None):
    """
    Monkey-patches layer0.neighborhood_aware_softmax.
    When heads is empty, the patch is a no-op (identity).
    Returns a remove() callable that restores the original method.

    Only attentions_per_edge is modified; skip_proj is unaffected.
    Shapes: attn = (E, NH, 1), trg_index = (E,), num_of_nodes = int.
    """
    orig_nas = layer0.neighborhood_aware_softmax  # already a bound method

    def _ablated_nas(scores_per_edge, trg_index, num_of_nodes):
        attn = orig_nas(scores_per_edge, trg_index, num_of_nodes)   # (E, NH, 1)
        if not heads:
            return attn
        attn = attn.clone()

        for h in sorted(heads):
            if mode == "uniform":
                # 1/in-degree per target node, broadcast back to edges
                degree = torch.zeros(num_of_nodes, dtype=attn.dtype, device=attn.device)
                ones   = torch.ones(len(trg_index), dtype=attn.dtype, device=attn.device)
                degree.scatter_add_(0, trg_index, ones)
                # nodes with no incoming edges get degree=0; clamp avoids div/0
                uniform_w = 1.0 / degree[trg_index].clamp(min=1.0)  # (E,)
                attn[:, h, 0] = uniform_w

            elif mode == "zero":
                attn[:, h, 0] = 0.0

            elif mode == "bonafide_mean":
                mean_val = float(bonafide_means[h]) if bonafide_means else 1/3
                attn[:, h, 0] = mean_val

            else:
                raise ValueError(f"Unknown ablation mode: {mode!r}")

        return attn

    # Assign as instance attribute — Python's descriptor protocol means
    # self.neighborhood_aware_softmax(...) inside forward() will find this
    # on the instance dict before the class, and call it WITHOUT auto-passing self.
    # orig_nas is already bound, so orig_nas(...) works correctly inside.
    layer0.neighborhood_aware_softmax = _ablated_nas

    def remove():
        # Restore by deleting the instance attribute → falls back to class method
        if "neighborhood_aware_softmax" in layer0.__dict__:
            del layer0.__dict__["neighborhood_aware_softmax"]

    return remove


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_baseline_identity(lit, loader, device) -> None:
    """
    Run two independent forward passes with no ablation on the same batch.
    Max logit difference must be < 1e-5 to confirm determinism.
    """
    gat_model = lit.model
    batch = next(iter(loader))
    audio   = batch["audio"].to(device)
    B       = audio.shape[0]
    num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

    with torch.no_grad():
        hs, pids = run_frozen_frontend(audio, gat_model, device)
        r1 = gat_model.encoder_and_GAT(hs, num_f, pids)
        logits1 = r1[5].cpu()
        r2 = gat_model.encoder_and_GAT(hs, num_f, pids)
        logits2 = r2[5].cpu()

    max_diff = (logits1 - logits2).abs().max().item()
    if max_diff >= 1e-5:
        raise AssertionError(
            f"SANITY FAILED — baseline not deterministic: max logit diff = {max_diff:.2e}\n"
            "SpecAugment or another stochastic component is active. Check eval mode."
        )
    print(f"  Baseline identity check PASSED: max logit diff = {max_diff:.2e}")


def sanity_layer_untouched(lit) -> None:
    """
    Confirm no instance-level override exists on layers 1 and 2
    after installing (and removing) the ablation on layer 0.
    """
    gat_net = lit.model.GAT.gat_net
    for idx in (1, 2):
        if "neighborhood_aware_softmax" in gat_net[idx].__dict__:
            raise AssertionError(
                f"SANITY FAILED — layer {idx} has an instance-level "
                "neighborhood_aware_softmax override. Ablation leaked to wrong layer."
            )
    print("  Layer-1/2 untouched check PASSED")


def sanity_floor_note(metrics_all: dict, config_name: str = "all_uniform") -> None:
    """
    For {all 6} uniform ablation: print residual attack accuracy and explain
    why it's above chance (skip_proj bypass). Does NOT fail — just reports.
    """
    if config_name not in metrics_all:
        return
    m = metrics_all[config_name]
    base = metrics_all["baseline"]
    print(f"\n  Floor check ({config_name}): "
          f"bonafide_acc={m['bonafide_acc']:.3f}  attack_acc={m['attack_acc']:.3f}")
    print(f"    Baseline was: bonafide_acc={base['bonafide_acc']:.3f}  "
          f"attack_acc={base['attack_acc']:.3f}")
    print("    NOTE: residual attack accuracy above chance is expected — "
          "skip_proj routes features around attention (see audit).")


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(lit, loader, device, heads: frozenset, mode: str,
             bonafide_means: dict[int, float]) -> list[dict]:
    """
    Run full dataset through model with the given ablation config.
    Returns list of {sample_id, label, system_id, logit (float), pred (int)}.
    """
    gat_model = lit.model
    layer0    = gat_model.GAT.gat_net[0]

    remove = install_ablation(layer0, heads, mode, bonafide_means)

    records: list[dict] = []
    sample_id = 0

    try:
        with torch.no_grad():
            for batch in loader:
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hs, pids = run_frozen_frontend(audio, gat_model, device)
                result   = gat_model.encoder_and_GAT(hs, num_f, pids)
                logits   = result[5].cpu()   # (B,)

                for i in range(B):
                    logit_i = float(logits[i].item())
                    records.append({
                        "sample_id": sample_id + i,
                        "label":     labels[i],
                        "system_id": sys_ids[i],
                        "logit":     logit_i,
                        "pred":      int(logit_i > 0),
                    })
                sample_id += B
    finally:
        remove()
        sanity_layer_untouched(lit)

    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """EER via linear threshold sweep. labels: 1=spoof, scores: sigmoid(logit)."""
    thresholds = np.unique(scores)
    n_bon   = (labels == 0).sum()
    n_spoof = (labels == 1).sum()
    best_eer, best_diff = 1.0, float("inf")
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        far   = fp / max(n_bon, 1)
        frr   = fn / max(n_spoof, 1)
        diff  = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2
    return best_eer


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via trapezoidal rule."""
    order  = np.argsort(-scores)
    ls     = labels[order]
    n_pos  = ls.sum()
    n_neg  = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(ls)     / n_pos
    fpr = np.cumsum(1 - ls) / n_neg
    # Prepend (0,0) for trapezoidal integration
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(abs(np.trapz(tpr, fpr)))


def compute_metrics(records: list[dict]) -> dict:
    labels  = np.array([r["label"]  for r in records])
    preds   = np.array([r["pred"]   for r in records])
    logits  = np.array([r["logit"]  for r in records])
    scores  = 1.0 / (1.0 + np.exp(-logits))   # sigmoid → attack probability

    bon_mask = labels == 0
    att_mask = labels == 1

    bon_acc  = float((preds[bon_mask] == labels[bon_mask]).mean()) if bon_mask.any() else float("nan")
    att_acc  = float((preds[att_mask] == labels[att_mask]).mean()) if att_mask.any() else float("nan")

    # Per-system
    sids = [r["system_id"] for r in records]
    per_sys: dict[str, float] = {}
    for sid in sorted(set(sids)):
        mask = np.array([s == sid for s in sids])
        per_sys[sid] = float((preds[mask] == labels[mask]).mean())

    eer = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)

    return {
        "bonafide_acc": bon_acc,
        "attack_acc":   att_acc,
        "per_system":   per_sys,
        "eer":          eer,
        "auc":          auc,
    }


def specificity_score(baseline: dict, ablated: dict) -> str:
    """
    (drop_in_attack_acc) / (drop_in_bonafide_acc).
    Returns "attack-only effect" if bonafide_acc does not drop (≤ 0.002 drop).
    """
    drop_bon = baseline["bonafide_acc"] - ablated["bonafide_acc"]
    drop_att = baseline["attack_acc"]   - ablated["attack_acc"]
    if drop_bon <= 0.002:
        return "attack-only effect"
    return f"{drop_att / drop_bon:.2f}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_main_bar(metrics_all: dict, out_path: Path) -> None:
    """
    Bar chart: accuracy drop from baseline for discriminative pair, control, and floor.
    Rows: bona-fide and attack accuracy drop. Side-by-side bars per config.
    """
    plt = _plt()
    baseline = metrics_all["baseline"]

    configs_to_plot = [
        ("h0",         "{h0}",         "#4C72B0"),
        ("h4",         "{h4}",         "#55A868"),
        ("h0_h4",      "{h0,h4}",      "#C44E52"),
        ("ctrl_h1235", "{h1,h2,h3,h5}","#8172B2"),
        ("all_uniform","all 6 heads",  "#937860"),
    ]

    n = len(configs_to_plot)
    x = np.arange(n)
    w = 0.35

    bon_drops = []
    att_drops = []
    labels_x  = []

    for cname, clabel, _ in configs_to_plot:
        m = metrics_all.get(cname, {"bonafide_acc": baseline["bonafide_acc"],
                                     "attack_acc":   baseline["attack_acc"]})
        bon_drops.append(baseline["bonafide_acc"] - m["bonafide_acc"])
        att_drops.append(baseline["attack_acc"]   - m["attack_acc"])
        labels_x.append(clabel)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_bon = ax.bar(x - w/2, bon_drops, w, label="Bona-fide acc drop",  color="#4393C3", alpha=0.85)
    bars_att = ax.bar(x + w/2, att_drops, w, label="Attack acc drop",     color="#D6604D", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_ylabel("Accuracy drop from baseline")
    ax.set_title("GAT Layer-0 Head Ablation: Accuracy Drop by Configuration")
    ax.legend(fontsize=10)

    # Annotate bars
    for bar in list(bars_bon) + list(bars_att):
        h = bar.get_height()
        if abs(h) > 0.002:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f"{h:+.3f}", ha="center", va="bottom", fontsize=8)

    # Add a vertical separator before ctrl group
    ctrl_idx = next(i for i, (c,_,__) in enumerate(configs_to_plot) if c == "ctrl_h1235")
    ax.axvline(ctrl_idx - 0.5, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_system(metrics_all: dict, out_path: Path) -> None:
    """Per-system attack accuracy for baseline, h0, h4, h0_h4, ctrl."""
    plt = _plt()
    configs_to_show = ["baseline", "h0", "h4", "h0_h4", "ctrl_h1235"]
    colors = ["#333333", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    attack_sids = sorted(s for s in metrics_all["baseline"]["per_system"] if s != "-")
    x = np.arange(len(attack_sids))
    n_configs = len(configs_to_show)
    w = 0.15
    offsets = np.linspace(-(n_configs-1)*w/2, (n_configs-1)*w/2, n_configs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, (cname, color) in enumerate(zip(configs_to_show, colors)):
        m = metrics_all.get(cname)
        if m is None:
            continue
        vals = [m["per_system"].get(sid, float("nan")) for sid in attack_sids]
        ax.bar(x + offsets[ci], vals, w, label=cname, color=color, alpha=0.8)

    ax.set_xticks(x); ax.set_xticklabels(attack_sids)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.set_ylabel("Attack accuracy")
    ax.set_title("Per-System Attack Accuracy by Head Ablation")
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CSV / report helpers
# ---------------------------------------------------------------------------

def write_summary_csv(metrics_all: dict, configs: list, out_path: Path) -> None:
    systems = sorted(metrics_all["baseline"]["per_system"].keys())
    header  = (["config", "heads", "mode", "bonafide_acc", "attack_acc"]
               + [f"acc_{s}" for s in systems] + ["eer", "auc", "specificity"])
    baseline = metrics_all["baseline"]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cname, heads, mode in configs:
            m = metrics_all[cname]
            spec = specificity_score(baseline, m)
            row = ([cname, str(sorted(heads)), mode,
                    f"{m['bonafide_acc']:.4f}", f"{m['attack_acc']:.4f}"]
                   + [f"{m['per_system'].get(s, float('nan')):.4f}" for s in systems]
                   + [f"{m['eer']:.4f}", f"{m['auc']:.4f}", spec])
            w.writerow(row)
    print(f"Saved: {out_path}")


def write_modes_csv(metrics_all: dict, out_path: Path) -> None:
    """Secondary table: {h0,h4} under all three ablation modes."""
    rows = [
        ("h0_h4",        "uniform"),
        ("h0_h4_zero",   "zero"),
        ("h0_h4_bfmean", "bonafide_mean"),
    ]
    header = ["config", "mode", "bonafide_acc", "attack_acc", "eer", "auc"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cname, mode in rows:
            m = metrics_all.get(cname)
            if m is None:
                continue
            w.writerow([cname, mode,
                        f"{m['bonafide_acc']:.4f}", f"{m['attack_acc']:.4f}",
                        f"{m['eer']:.4f}", f"{m['auc']:.4f}"])
    print(f"Saved: {out_path}")


def write_report(metrics_all: dict, configs: list,
                 sanity_msgs: list[str], out_path: Path) -> None:
    baseline = metrics_all["baseline"]
    systems  = sorted(baseline["per_system"].keys())

    # Main results table
    header_cols = (["Config", "Heads", "Mode", "BF acc", "Atk acc"]
                   + systems + ["EER", "AUC", "Specificity"])
    sep = "|" + "|".join("---" for _ in header_cols) + "|"
    hdr = "|" + "|".join(header_cols) + "|"

    table_rows = []
    for cname, heads, mode in configs:
        m    = metrics_all[cname]
        spec = specificity_score(baseline, m)
        bold = "**" if cname in ("h0_h4", "ctrl_h1235") else ""
        sys_vals = [f"{m['per_system'].get(s, float('nan')):.3f}" for s in systems]
        row = (f"|{bold}{cname}{bold}|{sorted(heads)}|{mode}"
               f"|{m['bonafide_acc']:.3f}|{m['attack_acc']:.3f}"
               + "|" + "|".join(sys_vals)
               + f"|{m['eer']:.3f}|{m['auc']:.3f}|{spec}|")
        table_rows.append(row)

    # Mode comparison table
    mode_rows = []
    for cname in ("h0_h4", "h0_h4_zero", "h0_h4_bfmean"):
        m = metrics_all.get(cname)
        if m is None:
            continue
        mode_rows.append(f"|{cname}|{m['bonafide_acc']:.3f}|{m['attack_acc']:.3f}"
                         f"|{m['eer']:.3f}|{m['auc']:.3f}|")

    # Causal claim assessment
    h0h4 = metrics_all["h0_h4"]
    ctrl  = metrics_all["ctrl_h1235"]
    bon_drop_disc = baseline["bonafide_acc"] - h0h4["bonafide_acc"]
    att_drop_disc = baseline["attack_acc"]   - h0h4["attack_acc"]
    att_drop_ctrl = baseline["attack_acc"]   - ctrl["attack_acc"]

    if att_drop_disc >= 0.05 and att_drop_ctrl < 0.02:
        claim = "**TRUE** — ablating h0 and h4 specifically degrades attack detection while the control ablation does not."
    elif att_drop_disc >= 0.05 and att_drop_ctrl >= 0.02:
        claim = "**PARTIAL** — ablating h0 and h4 degrades attack detection, but control heads also degrade performance, suggesting the effect is not head-specific."
    elif att_drop_disc < 0.02:
        claim = "**FALSE** — ablating h0 and h4 does not substantially degrade attack detection. The attention routing in those heads may not be causal."
    else:
        claim = f"**PARTIAL** — modest attack acc drop ({att_drop_disc:.3f}) for discriminative heads vs control ({att_drop_ctrl:.3f})."

    lines = [
        "# GAT Layer-0 Head Ablation Report",
        "",
        f"**Model**: `robust_goat.ckpt`",
        f"**Dataset**: ASVspoof 2019 LA validation, {N_PER_CLASS} samples per system",
        "**Ablation target**: `gat_net[0].neighborhood_aware_softmax` (attention routing only;",
        "skip_proj bypass is intentionally intact — see audit).",
        "",
        "## Sanity checks",
        "",
    ] + ["- " + s for s in sanity_msgs] + [
        "",
        "> **Floor check note**: `all_uniform` retains above-chance attack accuracy because",
        "> `skip_proj = Linear(768,768)` routes information to all head slots regardless of",
        "> attention routing. This is expected given the Option-A design choice.",
        "",
        "## Main results",
        "",
        hdr,
        sep,
    ] + table_rows + [
        "",
        "## Causal claim",
        "",
        "Causal claim: \"ablating h0 and h4 specifically degrades attack detection while",
        "preserving bona-fide classification.\"",
        "",
        f"Assessment: {claim}",
        "",
        "Key numbers:",
        f"- Attack acc drop {{h0,h4}} uniform: **{att_drop_disc:+.3f}**",
        f"- Bona-fide acc drop {{h0,h4}} uniform: **{bon_drop_disc:+.3f}**",
        f"- Attack acc drop control {{h1,h2,h3,h5}} uniform: **{att_drop_ctrl:+.3f}**",
        "",
        "## Mode-independence check for {h0, h4}",
        "",
        "| Config | BF acc | Atk acc | EER | AUC |",
        "|--------|--------|---------|-----|-----|",
    ] + mode_rows + [
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `ablation_summary.csv` | Full metrics for all 8 configs |",
        "| `ablation_modes_table.csv` | {h0,h4} comparison across 3 modes |",
        "| `ablation_bar.png` | Accuracy drop bar chart (main figure) |",
        "| `per_system_attack_acc.png` | Per-system breakdown |",
        "| `per_sample_preds.pt` | Raw per-sample logits/preds for all configs |",
        "",
    ]
    txt = "\n".join(lines)

    out_path.write_text(txt)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    patch_phoneme_loader()
    print(f"Loading model: {CKPT}")
    lit = load_model(CKPT, device)
    print(f"  model loaded")

    print(f"Loading dataset (N_PER_CLASS={N_PER_CLASS})...")
    dataset = BalancedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                              hf_token, N_PER_CLASS, SEED)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, collate_fn=collate,
                                          num_workers=0)

    print("\nLoading bonafide means for bonafide_mean ablation mode...")
    bonafide_means = load_bonafide_means(ATTN_ARTIFACTS)

    # ── Sanity: baseline determinism ─────────────────────────────────────────
    print("\n--- Sanity: baseline identity ---")
    sanity_baseline_identity(lit, loader, device)

    # ── Sanity: layer 1/2 untouched (pre-check with no ablation installed) ───
    sanity_layer_untouched(lit)

    # ── Run all configs ───────────────────────────────────────────────────────
    metrics_all:  dict[str, dict]      = {}
    all_preds:    dict[str, list[dict]] = {}
    sanity_msgs:  list[str]            = []

    for cname, heads, mode in CONFIGS:
        print(f"\n--- Config: {cname}  heads={sorted(heads)}  mode={mode} ---")
        records = run_eval(lit, loader, device, heads, mode, bonafide_means)
        metrics_all[cname] = compute_metrics(records)
        all_preds[cname]   = records

        m = metrics_all[cname]
        print(f"  bonafide_acc={m['bonafide_acc']:.4f}  "
              f"attack_acc={m['attack_acc']:.4f}  "
              f"eer={m['eer']:.4f}  auc={m['auc']:.4f}")
        for sid in sorted(m["per_system"]):
            print(f"    {sid:8s}: {m['per_system'][sid]:.4f}")

    # ── Post-run sanity messages ──────────────────────────────────────────────
    base_logits = np.array([r["logit"] for r in all_preds["baseline"]])
    sanity_msgs.append("Baseline identity check PASSED (max logit diff < 1e-5 across two runs)")
    sanity_msgs.append("Layer-1/2 untouched check PASSED (no instance override on gat_net[1/2])")

    # Floor check
    floor_m   = metrics_all["all_uniform"]
    floor_msg = (f"`all_uniform` attack_acc = {floor_m['attack_acc']:.3f}  "
                 f"bonafide_acc = {floor_m['bonafide_acc']:.3f}  "
                 "(above chance due to skip_proj bypass)")
    sanity_msgs.append(floor_msg)
    sanity_floor_note(metrics_all)

    # ── Save per-sample predictions ───────────────────────────────────────────
    preds_path = RESULTS_DIR / "per_sample_preds.pt"
    torch.save(all_preds, preds_path)
    print(f"\nSaved: {preds_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_main_bar(metrics_all, RESULTS_DIR / "ablation_bar.png")
    plot_per_system(metrics_all, RESULTS_DIR / "per_system_attack_acc.png")

    # ── Tables ────────────────────────────────────────────────────────────────
    write_summary_csv(metrics_all, CONFIGS, RESULTS_DIR / "ablation_summary.csv")
    write_modes_csv(metrics_all, RESULTS_DIR / "ablation_modes_table.csv")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(metrics_all, CONFIGS, sanity_msgs,
                 RESULTS_DIR / "head_ablation_report.md")

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
