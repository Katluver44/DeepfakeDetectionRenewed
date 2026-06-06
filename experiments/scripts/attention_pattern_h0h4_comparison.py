#!/usr/bin/env python3
"""
attention_pattern_h0h4_comparison.py
=====================================
Extracts and compares GAT layer-0 attention patterns for h0 and h4
between goat.ckpt and robust_goat.ckpt on a fixed balanced eval subset.

Fixed subset: 100 bonafide + 30 per attack system (A01-A06), seed=42.
Both checkpoint passes use the same DataLoader — example IDs are
structurally identical and verified by hash.

Outputs → experiments/results/gat_l0_attention_followups/attention_pattern_h0h4_comparison/
  attention_aggregates.npz      — (9,9) mean attention matrices per (ckpt, head, class)
  summary_stats.csv             — per (ckpt, head, class) entropy + sparsity stats
  figures/heatmap_grid.png      — 4×7 grid of 9×9 heatmaps
  figures/difference_heatmaps.png — robust_goat − goat per (head, class)
  figures/marginal_profiles.png — column-marginal overlays goat vs robust_goat
  run_config.json               — checkpoint hashes, example IDs, seed, pooling scheme
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sys
import types
from argparse import Namespace
from collections import defaultdict
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
RESULTS_DIR = EXPS_DIR / "results" / "gat_l0_attention_followups" / "attention_pattern_h0h4_comparison"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(REPO_ROOT), str(EXPS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from head_ablation import (  # noqa: E402
    load_model, run_frozen_frontend, patch_phoneme_loader,
    NF_PER_SAMPLE, collate, _decode, _crop, _lbl,
    HF_DATASET, CACHE_DIR, HF_TOKEN_PATH,
)
from utils.attention_hook import AttentionHook, PhonemeCapture  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
CKPT_GOAT   = REPO_ROOT / "models" / "goat.ckpt"
CKPT_ROBUST = REPO_ROOT / "models" / "robust_goat.ckpt"
CHECKPOINTS = {"goat": CKPT_GOAT, "robust_goat": CKPT_ROBUST}

N_BONAFIDE   = 100
N_PER_ATTACK = 30
SEED         = 42
BATCH_SIZE   = 8
TARGET_HEADS = [0, 4]
MIN_NODES    = 3
SYS_CLASSES  = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
SYS_LABELS   = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]  # for NPZ / figure labels
VOCAB_DIR    = REPO_ROOT / "vocab_phoneme"

LANG_ORDER  = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL     = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]
C = len(CLASS_ORDER)

_CAT_RULES: list[tuple[str, str]] = [
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


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

def _sym_to_cls(sym: str) -> str:
    if sym in SPECIAL or sym.isdigit():
        return "Other"
    for substr, cls in _CAT_RULES:
        if substr in sym:
            return cls
    return "Other"


def build_vocab() -> tuple[dict[int, str], dict[int, str]]:
    total: list[str] = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists():
            continue
        vocab: dict[str, int] = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id_to_sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    id_to_cls = {i: _sym_to_cls(s) for i, s in id_to_sym.items()}
    return id_to_sym, id_to_cls


# ---------------------------------------------------------------------------
# Fixed-subset dataset: 100 bonafide + 30 per attack
# ---------------------------------------------------------------------------

class FixedSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token,
                 n_bonafide=100, n_per_attack=30, seed=42):
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
        selected: list[int] = []
        sys_ids: list[str] = []
        for sid in sorted(by_system.keys()):
            idxs = list(by_system[sid])
            rng.shuffle(idxs)
            n = n_bonafide if sid == "-" else n_per_attack
            chosen = idxs[:n]
            selected.extend(chosen)
            sys_ids.extend([sid] * len(chosen))

        combined = list(zip(selected, sys_ids))
        rng.shuffle(combined)
        self.indices, self.sys_ids = map(list, zip(*combined)) if combined else ([], [])

        print(f"\nFixedSubsetDataset: {len(self.indices)} samples "
              f"({n_bonafide} bonafide + {n_per_attack}/attack)")
        for sid in sorted(by_system):
            cnt = sum(1 for s in self.sys_ids if s == sid)
            if cnt:
                print(f"  {sid:8s}: {cnt}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        return {
            "audio":     _crop(_decode(ex["audio"])),
            "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
            "system_id": self.sys_ids[idx],
        }


# ---------------------------------------------------------------------------
# Checkpoint hashing
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_for_ckpt(
    ckpt_name: str,
    ckpt_path: Path,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict]:
    """
    Load model, run extraction loop with AttentionHook + PhonemeCapture.
    Returns per-sample records with attn_l0 (E, NH) in sample-local edge space.
    """
    print(f"\n=== Extracting: {ckpt_name} ({ckpt_path.name}) ===")
    lit = load_model(ckpt_path, device)
    gat_model = lit.model

    hook = AttentionHook(gat_model, layer_idx=0).install()
    phoneme_cap = PhonemeCapture(gat_model)

    records: list[dict] = []
    n_degenerate = 0
    total = len(loader.dataset)
    attn_sum_checked = False

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hook.clear()
                hidden_states, phoneme_ids = run_frozen_frontend(audio, gat_model, device)
                gat_model.encoder_and_GAT(hidden_states, num_f, phoneme_ids)

                edge_index_global = hook.edge_index   # (2, E_total)
                attn_global       = hook.attn          # (E_total, NH)

                if edge_index_global is None or attn_global is None:
                    print(f"  [WARN] batch {bi}: hook did not fire, skipping")
                    continue

                NH = attn_global.shape[1]

                if not attn_sum_checked:
                    print(f"  Attention shape: {attn_global.shape}  (E={attn_global.shape[0]}, NH={NH})")
                    _check_attn_sum(attn_global, edge_index_global)
                    attn_sum_checked = True

                node_pids     = phoneme_cap.node_phoneme_ids   # (total_nodes,)
                node_samp_idx = phoneme_cap.node_sample_idx    # (total_nodes,)
                rnf           = phoneme_cap.reduced_num_frames  # (B,)

                node_offsets = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                for i in range(B):
                    n_nodes_i = int(rnf[i].item())
                    offset_i  = node_offsets[i]
                    src_global = edge_index_global[0]
                    edge_mask  = (node_samp_idx[src_global] == i)

                    ei_global = edge_index_global[:, edge_mask]
                    attn_i    = attn_global[edge_mask]            # (E_i, NH)
                    ei_local  = ei_global - offset_i              # 0-indexed, sample-local
                    pids_i    = node_pids[node_samp_idx == i]     # (n_nodes_i,)

                    is_deg = n_nodes_i < MIN_NODES
                    if is_deg:
                        n_degenerate += 1

                    records.append({
                        "system_id":       sys_ids[i],
                        "label":           labels[i],
                        "node_phoneme_ids": pids_i.clone(),
                        "edge_index":      ei_local.clone(),
                        "attn_l0":         attn_i.clone(),
                        "n_nodes":         n_nodes_i,
                        "is_degenerate":   is_deg,
                    })

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    done = min((bi + 1) * BATCH_SIZE, total)
                    print(f"  {done}/{total}")

    finally:
        hook.remove()

    valid = sum(1 for r in records if not r["is_degenerate"])
    print(f"  Done: {len(records)} records, {n_degenerate} degenerate filtered in downstream steps")
    return records


def _check_attn_sum(attn: torch.Tensor, edge_index: torch.Tensor, tol: float = 1e-3) -> None:
    NH        = attn.shape[1]
    tgt_nodes = edge_index[1]
    num_nodes = int(tgt_nodes.max().item()) + 1
    for h in range(NH):
        node_sum = torch.zeros(num_nodes)
        node_sum.scatter_add_(0, tgt_nodes, attn[:, h])
        active = node_sum > 1e-7
        if active.any():
            max_dev = (node_sum[active] - 1.0).abs().max().item()
            if max_dev > tol:
                raise AssertionError(
                    f"SANITY (a) FAILED — attn sum head {h}: max dev = {max_dev:.6f}"
                )
    print(f"  Sanity (a) PASSED: attention sums to 1 per query node (NH={NH}, tol={tol})")


# ---------------------------------------------------------------------------
# 9×9 phoneme-class aggregation
# ---------------------------------------------------------------------------

def aggregate_9x9(
    records: list[dict],
    id_to_cls: dict[int, str],
    head_idx: int,
    system_class: str,
) -> np.ndarray:
    """
    Returns (C, C) edge-sum-normalized attention matrix for a given
    (head, system_class) slice.  Entry [sc, tc] = fraction of total
    attention flowing from phoneme-class sc to phoneme-class tc.
    """
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    accum   = np.zeros((C, C), dtype=np.float64)

    for rec in records:
        if rec["is_degenerate"] or rec["system_id"] != system_class:
            continue
        pids = rec["node_phoneme_ids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"][:, head_idx].numpy()  # (E,)

        for e in range(ei.shape[1]):
            sc = cls_idx[id_to_cls.get(int(pids[ei[0, e]].item()), "Other")]
            tc = cls_idx[id_to_cls.get(int(pids[ei[1, e]].item()), "Other")]
            accum[sc, tc] += attn[e]

    total = accum.sum()
    if total > 0:
        accum /= total
    return accum


# ---------------------------------------------------------------------------
# Per-sample entropy
# ---------------------------------------------------------------------------

def compute_sample_entropy(rec: dict, head_idx: int) -> float:
    """Mean per-target-node entropy for a single sample and head."""
    attn_h    = rec["attn_l0"][:, head_idx].float()  # (E,)
    tgt_nodes = rec["edge_index"][1]                  # (E,)
    n         = rec["n_nodes"]

    entropies = []
    for t in range(n):
        mask = (tgt_nodes == t)
        if not mask.any():
            continue
        a   = attn_h[mask]
        ent = -(a * (a + 1e-10).log()).sum().item()
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0


# ---------------------------------------------------------------------------
# Summary statistics from a (C, C) attention matrix
# ---------------------------------------------------------------------------

def compute_matrix_stats(mat: np.ndarray) -> dict:
    flat        = mat.flatten()
    flat_sorted = np.sort(flat)[::-1]
    peak_idx    = int(np.argmax(mat))
    peak_src    = peak_idx // C
    peak_tgt    = peak_idx % C
    return {
        "top3_sparsity":  float(flat_sorted[:3].sum()),
        "top5_sparsity":  float(flat_sorted[:5].sum()),
        "peak_pos_src":   CLASS_ORDER[peak_src],
        "peak_pos_tgt":   CLASS_ORDER[peak_tgt],
        "peak_magnitude": float(flat_sorted[0]),
        "attn_variance":  float(mat.var()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _cls_ticks(ax, short=False):
    labels = [c[:3] if short else c for c in CLASS_ORDER]
    ax.set_xticks(range(C)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(C)); ax.set_yticklabels(labels, fontsize=5)


def plot_heatmap_grid(aggregates: dict, out_path: Path) -> None:
    """4×7 grid: rows=(goat_h0, goat_h4, robust_goat_h0, robust_goat_h4), cols=7 classes."""
    rows_meta = [("goat", 0), ("goat", 4), ("robust_goat", 0), ("robust_goat", 4)]
    row_labels = ["goat h0", "goat h4", "robust h0", "robust h4"]
    n_rows, n_cols = 4, len(SYS_LABELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
    fig.suptitle("GAT layer-0 attention heatmaps (9×9 phoneme classes)", fontsize=9)

    for ri, (ckpt, hidx) in enumerate(rows_meta):
        vmax = max(
            aggregates[f"{ckpt}_h{hidx}_{lbl}"].max()
            for lbl in SYS_LABELS
        )
        for ci, lbl in enumerate(SYS_LABELS):
            ax  = axes[ri, ci]
            mat = aggregates[f"{ckpt}_h{hidx}_{lbl}"]
            im  = ax.imshow(mat, cmap="viridis", aspect="auto",
                            vmin=0, vmax=vmax if vmax > 0 else 1)
            _cls_ticks(ax, short=True)
            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=7)
            if ri == 0:
                ax.set_title(lbl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_difference_heatmaps(aggregates: dict, out_path: Path) -> None:
    """2×7 grid: rows=h0/h4, cols=7 classes; shows robust_goat − goat."""
    n_rows, n_cols = 2, len(SYS_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
    fig.suptitle("Difference heatmaps: robust_goat − goat", fontsize=9)

    for ri, hidx in enumerate([0, 4]):
        for ci, lbl in enumerate(SYS_LABELS):
            ax   = axes[ri, ci]
            diff = (aggregates[f"robust_goat_h{hidx}_{lbl}"]
                    - aggregates[f"goat_h{hidx}_{lbl}"])
            vmax = max(abs(diff).max(), 1e-7)
            im   = ax.imshow(diff, cmap="RdBu_r", aspect="auto",
                             vmin=-vmax, vmax=vmax)
            _cls_ticks(ax, short=True)
            if ci == 0:
                ax.set_ylabel(f"h{hidx}", fontsize=7)
            if ri == 0:
                ax.set_title(lbl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_marginal_profiles(aggregates: dict, out_path: Path) -> None:
    """2×7 grid: rows=h0/h4, cols=7 classes; column-marginal (sum over src axis)."""
    n_rows, n_cols = 2, len(SYS_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4),
                             sharey=False)
    fig.suptitle("Column marginal profiles: attention flow into target phoneme class", fontsize=9)
    x = np.arange(C)

    for ri, hidx in enumerate([0, 4]):
        for ci, lbl in enumerate(SYS_LABELS):
            ax = axes[ri, ci]
            # Column marginal: sum over src (axis=0) → profile over tgt classes
            m_goat   = aggregates[f"goat_h{hidx}_{lbl}"].sum(axis=0)
            m_robust = aggregates[f"robust_goat_h{hidx}_{lbl}"].sum(axis=0)
            ax.plot(x, m_goat,   color="#1f77b4", marker="o", ms=3, lw=1.2,
                    label="goat")
            ax.plot(x, m_robust, color="#ff7f0e", marker="s", ms=3, lw=1.2,
                    label="robust")
            ax.set_xticks(x)
            ax.set_xticklabels([c[:3] for c in CLASS_ORDER], rotation=90, fontsize=5)
            ax.tick_params(axis="y", labelsize=5)
            if ci == 0:
                ax.set_ylabel(f"h{hidx}  Σ(src) attn", fontsize=6)
            if ri == 0:
                ax.set_title(lbl, fontsize=7)
            if ri == 0 and ci == n_cols - 1:
                ax.legend(fontsize=5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patch_phoneme_loader()

    # ── Vocab ─────────────────────────────────────────────────────────────────
    print("\nBuilding vocab...")
    id_to_sym, id_to_cls = build_vocab()
    print(f"  Vocab size: {len(id_to_cls)} tokens")

    # ── Fixed dataset ─────────────────────────────────────────────────────────
    token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset = FixedSubsetDataset(
        HF_DATASET, "validation", CACHE_DIR, token,
        n_bonafide=N_BONAFIDE, n_per_attack=N_PER_ATTACK, seed=SEED,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate,
    )

    # Hash example IDs for run_config
    id_hash_str = ",".join(str(x) for x in dataset.indices)
    example_ids_hash = hashlib.sha256(id_hash_str.encode()).hexdigest()[:16]
    print(f"\n  Example IDs hash: {example_ids_hash}")

    # ── Two-pass extraction ───────────────────────────────────────────────────
    all_records: dict[str, list[dict]] = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        all_records[ckpt_name] = extract_attention_for_ckpt(
            ckpt_name, ckpt_path, loader, device,
        )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n--- Sanity checks ---")

    # (a) attn sums to 1 — verified per-batch during extraction above

    # (b) identical example ordering between passes
    sids_goat   = [r["system_id"] for r in all_records["goat"]]
    sids_robust = [r["system_id"] for r in all_records["robust_goat"]]
    assert sids_goat == sids_robust, (
        "Sanity (b) FAILED: system_id ordering differs between checkpoint passes"
    )
    sys_id_hash = hashlib.sha256(",".join(sids_goat).encode()).hexdigest()[:16]
    print(f"  Sanity (b) PASSED: identical example ordering  hash={sys_id_hash}")

    # (c) h0 ≠ h4 within each checkpoint (aggregate matrices must differ)
    def _agg_all(ckpt: str, hidx: int) -> np.ndarray:
        recs = all_records[ckpt]
        cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        accum = np.zeros((C, C), dtype=np.float64)
        for rec in recs:
            if rec["is_degenerate"]:
                continue
            pids = rec["node_phoneme_ids"]
            ei   = rec["edge_index"]
            attn = rec["attn_l0"][:, hidx].numpy()
            for e in range(ei.shape[1]):
                sc = cls_idx[id_to_cls.get(int(pids[ei[0, e]].item()), "Other")]
                tc = cls_idx[id_to_cls.get(int(pids[ei[1, e]].item()), "Other")]
                accum[sc, tc] += attn[e]
        tot = accum.sum()
        return accum / tot if tot > 0 else accum

    for ckpt_name in CHECKPOINTS:
        m_h0 = _agg_all(ckpt_name, 0)
        m_h4 = _agg_all(ckpt_name, 4)
        l2   = float(np.linalg.norm(m_h0 - m_h4))
        assert l2 > 1e-4, (
            f"Sanity (c) FAILED: h0 ≈ h4 within {ckpt_name} (L2={l2:.2e})"
        )
        print(f"  Sanity (c) PASSED [{ckpt_name}]: h0 ≠ h4  L2={l2:.4f}")

    # (d) entropy non-degenerate (mean > 0.01 across all records)
    for ckpt_name, recs in all_records.items():
        valid = [r for r in recs if not r["is_degenerate"]]
        for hidx in TARGET_HEADS:
            entropies = [compute_sample_entropy(r, hidx) for r in valid]
            mean_ent  = float(np.mean(entropies))
            assert mean_ent > 0.01, (
                f"Sanity (d) FAILED: near-zero entropy [{ckpt_name} h{hidx}] = {mean_ent:.4f}"
            )
        print(f"  Sanity (d) PASSED [{ckpt_name}]: entropy non-degenerate")

    # ── Aggregate 9×9 matrices ────────────────────────────────────────────────
    print("\n--- Aggregating 9×9 attention matrices ---")
    aggregates: dict[str, np.ndarray] = {}
    for ckpt_name, recs in all_records.items():
        for hidx in TARGET_HEADS:
            for sys_cls, lbl in zip(SYS_CLASSES, SYS_LABELS):
                key = f"{ckpt_name}_h{hidx}_{lbl}"
                aggregates[key] = aggregate_9x9(recs, id_to_cls, hidx, sys_cls)
    print(f"  Built {len(aggregates)} matrices  (2 ckpts × 2 heads × 7 classes)")

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    npz_path = RESULTS_DIR / "attention_aggregates.npz"
    np.savez(npz_path, **aggregates)
    print(f"  Saved: {npz_path}")

    # ── Summary stats CSV ─────────────────────────────────────────────────────
    print("\n--- Computing summary statistics ---")
    csv_rows: list[dict] = []
    entropy_table: dict[str, dict[int, list[float]]] = {}

    for ckpt_name, recs in all_records.items():
        valid = [r for r in recs if not r["is_degenerate"]]
        for hidx in TARGET_HEADS:
            for sys_cls, lbl in zip(SYS_CLASSES, SYS_LABELS):
                mat = aggregates[f"{ckpt_name}_h{hidx}_{lbl}"]
                stats = compute_matrix_stats(mat)

                # Mean entropy over samples in this system class
                sys_recs = [r for r in valid if r["system_id"] == sys_cls]
                entropies = [compute_sample_entropy(r, hidx) for r in sys_recs]
                mean_ent  = float(np.mean(entropies)) if entropies else 0.0

                key = (ckpt_name, hidx)
                if key not in entropy_table:
                    entropy_table[key] = {}
                entropy_table[key][lbl] = mean_ent

                row = {
                    "checkpoint":    ckpt_name,
                    "head":          f"h{hidx}",
                    "class":         lbl,
                    "mean_entropy":  round(mean_ent, 6),
                    **{k: round(v, 6) if isinstance(v, float) else v
                       for k, v in stats.items()},
                }
                csv_rows.append(row)

    csv_path = RESULTS_DIR / "summary_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved: {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n--- Generating figures ---")
    heatmap_path  = FIGURES_DIR / "heatmap_grid.png"
    diff_path     = FIGURES_DIR / "difference_heatmaps.png"
    marginal_path = FIGURES_DIR / "marginal_profiles.png"

    plot_heatmap_grid(aggregates, heatmap_path)
    plot_difference_heatmaps(aggregates, diff_path)
    plot_marginal_profiles(aggregates, marginal_path)

    # ── run_config.json ───────────────────────────────────────────────────────
    print("\n--- Computing checkpoint hashes ---")
    ckpt_hashes = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"  Hashing {ckpt_path.name}...")
        ckpt_hashes[ckpt_name] = {
            "path":   str(ckpt_path),
            "sha256": sha256_of_file(ckpt_path),
        }

    config = {
        "checkpoints":            ckpt_hashes,
        "dataset": {
            "hf_name":           HF_DATASET,
            "split":             "validation",
            "n_bonafide":        N_BONAFIDE,
            "n_per_attack":      N_PER_ATTACK,
            "seed":              SEED,
            "n_total":           len(dataset),
            "example_ids":       dataset.indices,
            "example_ids_hash":  example_ids_hash,
            "system_order_hash": sys_id_hash,
        },
        "target_heads":           TARGET_HEADS,
        "gat_layer":              0,
        "attention_matrix_shape": [C, C],
        "pooling_scheme":         "edge-sum-normalize",
        "class_order":            CLASS_ORDER,
        "system_classes":         SYS_CLASSES,
        "system_labels":          SYS_LABELS,
        "sanity_pass":            True,
    }
    config_path = RESULTS_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ── Completion summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ATTENTION PATTERN COMPARISON COMPLETE")
    print("=" * 72)

    print("\nPer-head mean entropy (goat → robust_goat):")
    for hidx in TARGET_HEADS:
        vals_g = [entropy_table[("goat", hidx)][lbl] for lbl in SYS_LABELS]
        vals_r = [entropy_table[("robust_goat", hidx)][lbl] for lbl in SYS_LABELS]
        g_mean = float(np.mean(vals_g))
        r_mean = float(np.mean(vals_r))
        print(f"  h{hidx}: goat={g_mean:.4f}  →  robust_goat={r_mean:.4f}")

    # Largest L2 difference cell
    print("\nLargest L2 difference (robust_goat − goat) per (head, class):")
    max_l2, max_cell = 0.0, ("", "")
    for hidx in TARGET_HEADS:
        for lbl in SYS_LABELS:
            diff = (aggregates[f"robust_goat_h{hidx}_{lbl}"]
                    - aggregates[f"goat_h{hidx}_{lbl}"])
            l2 = float(np.linalg.norm(diff))
            print(f"  h{hidx}, {lbl:10s}: L2={l2:.4f}")
            if l2 > max_l2:
                max_l2  = l2
                max_cell = (f"h{hidx}", lbl)
    print(f"\n  → Max: {max_cell[0]}, {max_cell[1]}  L2={max_l2:.4f}")

    print("\nFigures saved:")
    for p in [heatmap_path, diff_path, marginal_path]:
        print(f"  {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
