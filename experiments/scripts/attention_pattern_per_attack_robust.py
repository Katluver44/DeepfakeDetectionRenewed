#!/usr/bin/env python3
"""
attention_pattern_per_attack_robust.py
========================================
Per-attack attention pattern analysis for robust_goat.ckpt, with A03/A05 focus.

Uses the same fixed eval subset as Prompt 1 (100 bonafide + 30/attack, seed=42).
If any attack system has < 20 examples in the subset, sampling is expanded
for that system specifically.

Outputs → experiments/results/gat_l0_attention_followups/attention_pattern_per_attack_robust/
  per_attack_attention_stats.csv     — entropy distribution + sparsity + variance per (head, system)
  per_attack_mean_patterns.npz       — (9,9) mean attention matrix per (head, system)
  per_attack_position_variance.npz   — (9,9) per-position variance per (head, system)
  attack_attention_similarity.csv    — 14×14 cosine similarity matrix
  attack_attention_similarity.png    — heatmap
  a03_vs_a05/
    side_by_side_h0.png              — mean attention: A03 vs A05 for h0
    side_by_side_h4.png              — same for h4
    difference_h0.png                — (h0 A03) − (h0 A05)
    difference_h4.png                — (h4 A03) − (h4 A05)
    a03_vs_a05_test.json             — per-head L2/KL distances + permutation p-values
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
import types
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
               / "attention_pattern_per_attack_robust")
A03A05_DIR  = RESULTS_DIR / "a03_vs_a05"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
A03A05_DIR.mkdir(parents=True, exist_ok=True)

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
CKPT_ROBUST  = REPO_ROOT / "models" / "robust_goat.ckpt"
N_BONAFIDE   = 100
N_PER_ATTACK = 30
MIN_PER_SYS  = 20
SEED         = 42
BATCH_SIZE   = 8
TARGET_HEADS = [0, 4]
MIN_NODES    = 3
N_PERM       = 1000
SYS_CLASSES  = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
SYS_LABELS   = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
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


def build_vocab() -> dict[int, str]:
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
    return {i: _sym_to_cls(s) for i, s in id_to_sym.items()}


# ---------------------------------------------------------------------------
# Fixed-subset dataset — identical to Prompt 1 (seed=42, N_BONAFIDE=100, N_PER_ATTACK=30)
# Expand any system below MIN_PER_SYS.
# ---------------------------------------------------------------------------

class FixedSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token,
                 n_bonafide=100, n_per_attack=30, min_per_sys=20, seed=42):
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

        # Initial selection — same algorithm + seed as Prompt 1
        rng = random.Random(seed)
        selected: list[int] = []
        sys_ids: list[str] = []
        selected_sets: dict[str, set[int]] = {}
        for sid in sorted(by_system.keys()):
            idxs = list(by_system[sid])
            rng.shuffle(idxs)
            n = n_bonafide if sid == "-" else n_per_attack
            chosen = idxs[:n]
            selected.extend(chosen)
            sys_ids.extend([sid] * len(chosen))
            selected_sets[sid] = set(chosen)

        combined = list(zip(selected, sys_ids))
        rng.shuffle(combined)
        self.indices, self.sys_ids = map(list, zip(*combined)) if combined else ([], [])

        # Expand any attack system below min_per_sys
        by_cnt = Counter(self.sys_ids)
        for sid in sorted(by_system.keys()):
            if sid == "-":
                continue
            cnt = by_cnt.get(sid, 0)
            if cnt < min_per_sys:
                already = selected_sets.get(sid, set())
                remaining = [x for x in by_system[sid] if x not in already]
                need = min_per_sys - cnt
                extra = remaining[:need]
                self.indices += extra
                self.sys_ids += [sid] * len(extra)
                print(f"  [EXPAND] {sid}: {cnt} → {cnt + len(extra)} (needed ≥ {min_per_sys})")

        self._by_system = by_system
        self.by_cnt = Counter(self.sys_ids)

        print(f"\nDataset: {len(self.indices)} samples  "
              f"({n_bonafide} bonafide + {n_per_attack}/attack, seed={seed})")
        for sid in sorted(by_system.keys()):
            cnt = self.by_cnt.get(sid, 0)
            if cnt:
                flag = f"  [<{min_per_sys}]" if cnt < min_per_sys else ""
                print(f"  {sid:8s}: {cnt}{flag}")

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
# Attention extraction (single checkpoint)
# ---------------------------------------------------------------------------

def extract_records(ckpt_path: Path, loader, device) -> list[dict]:
    print(f"\nExtracting: {ckpt_path.name}")
    lit = load_model(ckpt_path, device)
    gat_model = lit.model

    hook        = AttentionHook(gat_model, layer_idx=0).install()
    phoneme_cap = PhonemeCapture(gat_model)

    records: list[dict] = []
    n_deg   = 0
    total   = len(loader.dataset)

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

                ei_g   = hook.edge_index   # (2, E_total)
                attn_g = hook.attn          # (E_total, NH)
                if ei_g is None or attn_g is None:
                    print(f"  [WARN] batch {bi}: hook did not fire")
                    continue

                if bi == 0:
                    _check_attn_sum(attn_g, ei_g)

                rnf           = phoneme_cap.reduced_num_frames
                node_pids     = phoneme_cap.node_phoneme_ids
                node_samp_idx = phoneme_cap.node_sample_idx
                node_offsets  = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                for i in range(B):
                    offset_i  = node_offsets[i]
                    src_g     = ei_g[0]
                    mask      = (node_samp_idx[src_g] == i)
                    ei_loc    = ei_g[:, mask] - offset_i
                    attn_i    = attn_g[mask]
                    pids_i    = node_pids[node_samp_idx == i]
                    n_nodes_i = int(rnf[i].item())
                    is_deg    = n_nodes_i < MIN_NODES
                    if is_deg:
                        n_deg += 1
                    records.append({
                        "system_id":        sys_ids[i],
                        "label":            labels[i],
                        "node_phoneme_ids": pids_i.clone(),
                        "edge_index":       ei_loc.clone(),
                        "attn_l0":          attn_i.clone(),
                        "n_nodes":          n_nodes_i,
                        "is_degenerate":    is_deg,
                    })

                if (bi + 1) % 10 == 0 or bi + 1 == len(loader):
                    print(f"  {min((bi+1)*BATCH_SIZE, total)}/{total}")

    finally:
        hook.remove()

    print(f"  {len(records)} records, {n_deg} degenerate")
    return records


def _check_attn_sum(attn: torch.Tensor, edge_index: torch.Tensor, tol: float = 1e-3):
    tgt   = edge_index[1]
    n     = int(tgt.max().item()) + 1
    NH    = attn.shape[1]
    for h in range(NH):
        s = torch.zeros(n)
        s.scatter_add_(0, tgt, attn[:, h])
        active = s > 1e-7
        if active.any():
            dev = (s[active] - 1.0).abs().max().item()
            if dev > tol:
                raise AssertionError(
                    f"Sanity (a) FAILED — attn sum h{h}: max dev={dev:.2e}"
                )
    print(f"  Sanity (a) PASSED: attn sums to 1 per node (NH={NH})")


# ---------------------------------------------------------------------------
# Per-example (9×9) matrix extraction
# ---------------------------------------------------------------------------

def per_example_matrices(
    records: list[dict],
    id_to_cls: dict[int, str],
    head_idx: int,
    system_class: str,
) -> list[np.ndarray]:
    """Returns list of (C, C) edge-sum-normalized matrices, one per example."""
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    mats = []
    for rec in records:
        if rec["is_degenerate"] or rec["system_id"] != system_class:
            continue
        pids = rec["node_phoneme_ids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"][:, head_idx].numpy()
        mat  = np.zeros((C, C), dtype=np.float64)
        for e in range(ei.shape[1]):
            sc = cls_idx[id_to_cls.get(int(pids[ei[0, e]].item()), "Other")]
            tc = cls_idx[id_to_cls.get(int(pids[ei[1, e]].item()), "Other")]
            mat[sc, tc] += attn[e]
        tot = mat.sum()
        if tot > 0:
            mat /= tot
        mats.append(mat)
    return mats


# ---------------------------------------------------------------------------
# Per-example entropy
# ---------------------------------------------------------------------------

def sample_entropy(rec: dict, head_idx: int) -> float:
    attn_h    = rec["attn_l0"][:, head_idx].float()
    tgt_nodes = rec["edge_index"][1]
    n         = rec["n_nodes"]
    ents = []
    for t in range(n):
        mask = (tgt_nodes == t)
        if not mask.any():
            continue
        a = attn_h[mask]
        ents.append(-(a * (a + 1e-10).log()).sum().item())
    return float(np.mean(ents)) if ents else 0.0


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    denom  = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# KL divergence (Laplace-smoothed)
# ---------------------------------------------------------------------------

def kl_div(P: np.ndarray, Q: np.ndarray, eps: float = 1e-6) -> float:
    Ps = P.flatten() + eps;  Ps /= Ps.sum()
    Qs = Q.flatten() + eps;  Qs /= Qs.sum()
    return float(np.sum(Ps * np.log(Ps / Qs)))


# ---------------------------------------------------------------------------
# Permutation test: L2 distance between group-mean matrices
# ---------------------------------------------------------------------------

def permutation_test_l2(
    mats_a: list[np.ndarray],
    mats_b: list[np.ndarray],
    n_perm: int = 1000,
    seed:   int = 42,
) -> tuple[float, float, list[float]]:
    na, nb  = len(mats_a), len(mats_b)
    pool    = np.stack(mats_a + mats_b)            # (na+nb, C, C)
    mean_a  = pool[:na].mean(axis=0)
    mean_b  = pool[na:].mean(axis=0)
    obs     = float(np.linalg.norm(mean_a - mean_b))

    rng      = np.random.RandomState(seed)
    null     = []
    for _ in range(n_perm):
        perm = rng.permutation(na + nb)
        d    = float(np.linalg.norm(
            pool[perm[:na]].mean(axis=0) - pool[perm[na:]].mean(axis=0)
        ))
        null.append(d)

    p_val = float(np.mean([d >= obs for d in null]))
    return obs, p_val, null


# ---------------------------------------------------------------------------
# Matrix statistics helper
# ---------------------------------------------------------------------------

def matrix_stats(mat: np.ndarray) -> dict:
    flat        = mat.flatten()
    flat_sorted = np.sort(flat)[::-1]
    peak_idx    = int(np.argmax(mat))
    return {
        "top3_sparsity":  float(flat_sorted[:3].sum()),
        "top5_sparsity":  float(flat_sorted[:5].sum()),
        "peak_pos_src":   CLASS_ORDER[peak_idx // C],
        "peak_pos_tgt":   CLASS_ORDER[peak_idx % C],
        "peak_magnitude": float(flat_sorted[0]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _cls_ticks(ax, short: bool = True):
    labels = [c[:3] if short else c for c in CLASS_ORDER]
    ax.set_xticks(range(C)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(C)); ax.set_yticklabels(labels, fontsize=5)


def plot_similarity_heatmap(sim: np.ndarray, labels: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Cross-attack cosine similarity (h0 & h4 attention patterns)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_side_by_side(mat_a03: np.ndarray, mat_a05: np.ndarray,
                      hidx: int, out: Path):
    vmax = max(mat_a03.max(), mat_a05.max())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax, mat, lbl in zip(axes, [mat_a03, mat_a05], ["A03", "A05"]):
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        _cls_ticks(ax)
        ax.set_title(f"h{hidx} — {lbl}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Mean attention pattern: A03 vs A05  (h{hidx})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_difference(mat_a03: np.ndarray, mat_a05: np.ndarray,
                    hidx: int, out: Path):
    diff = mat_a03 - mat_a05
    vmax = max(abs(diff).max(), 1e-7)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    _cls_ticks(ax)
    ax.set_title(f"(h{hidx} A03) − (h{hidx} A05)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patch_phoneme_loader()

    print("\nBuilding vocab...")
    id_to_cls = build_vocab()
    print(f"  Vocab size: {len(id_to_cls)} tokens")

    # ── Dataset (identical to Prompt 1) ──────────────────────────────────────
    token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset = FixedSubsetDataset(
        HF_DATASET, "validation", CACHE_DIR, token,
        n_bonafide=N_BONAFIDE, n_per_attack=N_PER_ATTACK,
        min_per_sys=MIN_PER_SYS, seed=SEED,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate,
    )

    # Sanity (b): per-system counts ≥ MIN_PER_SYS
    print("\n--- Sanity (b): per-system example counts ---")
    below_threshold = []
    for sid, lbl in zip(SYS_CLASSES, SYS_LABELS):
        cnt = dataset.by_cnt.get(sid, 0)
        flag = "  [WARN: < threshold]" if cnt < MIN_PER_SYS else ""
        print(f"  {lbl:10s}: {cnt}{flag}")
        if cnt < MIN_PER_SYS:
            below_threshold.append(lbl)
    if below_threshold:
        print(f"  [WARN] Systems below {MIN_PER_SYS}: {below_threshold}")
    else:
        print(f"  Sanity (b) PASSED: all systems ≥ {MIN_PER_SYS} examples")

    id_hash = hashlib.sha256(",".join(str(x) for x in dataset.indices).encode()).hexdigest()[:16]

    # ── Extract records ───────────────────────────────────────────────────────
    records = extract_records(CKPT_ROBUST, loader, device)

    # ── Per-example matrices and stats ───────────────────────────────────────
    print("\n--- Computing per-example matrices ---")
    all_mats: dict[str, list[np.ndarray]] = {}   # key: f"h{hidx}_{lbl}"
    for hidx in TARGET_HEADS:
        for sys_cls, lbl in zip(SYS_CLASSES, SYS_LABELS):
            key = f"h{hidx}_{lbl}"
            all_mats[key] = per_example_matrices(records, id_to_cls, hidx, sys_cls)
            print(f"  {key}: {len(all_mats[key])} examples")

    # Mean patterns and per-position variance
    mean_pats: dict[str, np.ndarray] = {}
    pos_vars:  dict[str, np.ndarray] = {}
    for key, mats in all_mats.items():
        stack = np.stack(mats)           # (N, C, C)
        mean_pats[key] = stack.mean(axis=0)
        pos_vars[key]  = stack.var(axis=0)

    # Save NPZs
    np.savez(RESULTS_DIR / "per_attack_mean_patterns.npz", **mean_pats)
    np.savez(RESULTS_DIR / "per_attack_position_variance.npz", **pos_vars)
    print(f"  Saved NPZs: per_attack_mean_patterns.npz, per_attack_position_variance.npz")

    # ── Summary stats CSV ─────────────────────────────────────────────────────
    print("\n--- Computing summary statistics ---")
    csv_rows: list[dict] = []
    for hidx in TARGET_HEADS:
        for sys_cls, lbl in zip(SYS_CLASSES, SYS_LABELS):
            key  = f"h{hidx}_{lbl}"
            mats = all_mats[key]
            recs_sys = [r for r in records
                        if r["system_id"] == sys_cls and not r["is_degenerate"]]

            entropies = [sample_entropy(r, hidx) for r in recs_sys]
            ents_arr  = np.array(entropies)
            q25, med, q75 = np.percentile(ents_arr, [25, 50, 75]) if len(ents_arr) > 0 else (0,0,0)

            mean_pat = mean_pats[key]
            mstats   = matrix_stats(mean_pat)
            pvar_mat = pos_vars[key]

            csv_rows.append({
                "head":                 f"h{hidx}",
                "system":               lbl,
                "n_examples":           len(mats),
                "mean_entropy":         round(float(ents_arr.mean()) if len(ents_arr) else 0.0, 6),
                "std_entropy":          round(float(ents_arr.std())  if len(ents_arr) else 0.0, 6),
                "q25_entropy":          round(float(q25), 6),
                "median_entropy":       round(float(med), 6),
                "q75_entropy":          round(float(q75), 6),
                "top3_sparsity":        round(mstats["top3_sparsity"], 6),
                "top5_sparsity":        round(mstats["top5_sparsity"], 6),
                "peak_pos_src":         mstats["peak_pos_src"],
                "peak_pos_tgt":         mstats["peak_pos_tgt"],
                "peak_magnitude":       round(mstats["peak_magnitude"], 6),
                "mean_position_variance": round(float(pvar_mat.mean()), 8),
                "max_position_variance":  round(float(pvar_mat.max()),  8),
            })

    stats_path = RESULTS_DIR / "per_attack_attention_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"  Saved: {stats_path}")

    # ── Cross-attack similarity ───────────────────────────────────────────────
    print("\n--- Cross-attack cosine similarity ---")
    sim_labels = [f"h{h}_{lbl}" for h in TARGET_HEADS for lbl in SYS_LABELS]
    N_SIM = len(sim_labels)
    sim_matrix = np.zeros((N_SIM, N_SIM))
    for i, li in enumerate(sim_labels):
        for j, lj in enumerate(sim_labels):
            sim_matrix[i, j] = cosine_sim(mean_pats[li], mean_pats[lj])

    # Sanity (a): symmetric + diagonal ≈ 1
    sym_ok  = np.allclose(sim_matrix, sim_matrix.T, atol=1e-6)
    diag_ok = np.allclose(np.diag(sim_matrix), 1.0, atol=1e-4)
    assert sym_ok,  "Sanity (a) FAILED: similarity matrix not symmetric"
    assert diag_ok, f"Sanity (a) FAILED: diagonal not ≈ 1 (min={np.diag(sim_matrix).min():.6f})"
    print(f"  Sanity (a) PASSED: symmetric={sym_ok}, diagonal≈1={diag_ok}")

    # Save CSV
    sim_csv_path = RESULTS_DIR / "attack_attention_similarity.csv"
    with open(sim_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + sim_labels)
        for i, row_lbl in enumerate(sim_labels):
            writer.writerow([row_lbl] + [f"{sim_matrix[i, j]:.6f}" for j in range(N_SIM)])
    print(f"  Saved: {sim_csv_path}")

    # Save heatmap
    sim_png_path = RESULTS_DIR / "attack_attention_similarity.png"
    plot_similarity_heatmap(sim_matrix, sim_labels, sim_png_path)

    # ── A03 vs A05 focused analysis ───────────────────────────────────────────
    print("\n--- A03 vs A05 focused analysis ---")
    a03a05_results: dict[str, dict] = {}

    for hidx in TARGET_HEADS:
        key_a03 = f"h{hidx}_A03"
        key_a05 = f"h{hidx}_A05"
        mats_a03 = all_mats[key_a03]
        mats_a05 = all_mats[key_a05]
        mean_a03 = mean_pats[key_a03]
        mean_a05 = mean_pats[key_a05]

        # L2 and KL distances
        l2_dist    = float(np.linalg.norm(mean_a03 - mean_a05))
        kl_a03_a05 = kl_div(mean_a03, mean_a05)
        kl_a05_a03 = kl_div(mean_a05, mean_a03)

        # Permutation test
        print(f"  Permutation test h{hidx} (N={N_PERM})...")
        obs_l2, p_val, null_dist = permutation_test_l2(
            mats_a03, mats_a05, n_perm=N_PERM, seed=SEED
        )
        # Sanity (c): null has non-trivial spread
        null_std = float(np.std(null_dist))
        assert null_std > 1e-6, (
            f"Sanity (c) FAILED: permutation null collapsed (std={null_std:.2e}) for h{hidx}"
        )
        print(f"  Sanity (c) PASSED h{hidx}: null std={null_std:.4f}")

        a03a05_results[f"h{hidx}"] = {
            "n_a03":            len(mats_a03),
            "n_a05":            len(mats_a05),
            "observed_l2":      round(obs_l2, 6),
            "kl_a03_to_a05":    round(kl_a03_a05, 6),
            "kl_a05_to_a03":    round(kl_a05_a03, 6),
            "permutation_n":    N_PERM,
            "permutation_p_l2": round(p_val, 4),
            "null_mean":        round(float(np.mean(null_dist)), 6),
            "null_std":         round(null_std, 6),
            "null_min":         round(float(np.min(null_dist)), 6),
            "null_max":         round(float(np.max(null_dist)), 6),
        }

        # Figures
        plot_side_by_side(
            mean_a03, mean_a05, hidx,
            A03A05_DIR / f"side_by_side_h{hidx}.png"
        )
        plot_difference(
            mean_a03, mean_a05, hidx,
            A03A05_DIR / f"difference_h{hidx}.png"
        )

    test_json_path = A03A05_DIR / "a03_vs_a05_test.json"
    with open(test_json_path, "w") as f:
        json.dump(a03a05_results, f, indent=2)
    print(f"  Saved: {test_json_path}")

    # ── run_config.json ───────────────────────────────────────────────────────
    print("\n  Hashing checkpoint...")
    config = {
        "checkpoint":      str(CKPT_ROBUST),
        "checkpoint_sha256": sha256_of_file(CKPT_ROBUST),
        "dataset": {
            "hf_name":         HF_DATASET,
            "split":           "validation",
            "n_bonafide":      N_BONAFIDE,
            "n_per_attack":    N_PER_ATTACK,
            "min_per_sys":     MIN_PER_SYS,
            "seed":            SEED,
            "n_total":         len(dataset),
            "example_ids":     dataset.indices,
            "example_ids_hash": id_hash,
        },
        "target_heads":          TARGET_HEADS,
        "gat_layer":             0,
        "attention_matrix_shape": [C, C],
        "pooling_scheme":        "edge-sum-normalize per example then mean",
        "class_order":           CLASS_ORDER,
        "system_classes":        SYS_CLASSES,
        "n_permutations":        N_PERM,
        "sanity_pass":           True,
    }
    config_path = RESULTS_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ── Completion summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ATTENTION PATTERN PER-ATTACK ANALYSIS COMPLETE")
    print("=" * 72)

    # Most/least similar attack-system pairs (within-head, off-diagonal)
    for hidx in TARGET_HEADS:
        h_labels = [f"h{hidx}_{lbl}" for lbl in SYS_LABELS]
        h_idx    = [sim_labels.index(lb) for lb in h_labels]
        sub      = sim_matrix[np.ix_(h_idx, h_idx)]
        # Mask diagonal
        mask = np.ones_like(sub, dtype=bool)
        np.fill_diagonal(mask, False)
        flat_idx = np.argwhere(mask)
        sims     = [(sub[r, c], h_labels[r], h_labels[c]) for r, c in flat_idx]
        sims.sort()
        least = sims[0]
        most  = sims[-1]
        print(f"\nh{hidx} most  similar pair:  {most[1]:20s}  ↔  {most[2]:20s}  sim={most[0]:.4f}")
        print(f"h{hidx} least similar pair:  {least[1]:20s}  ↔  {least[2]:20s}  sim={least[0]:.4f}")

    # A03 ↔ A05 summary
    print("\nA03 ↔ A05 distance and permutation test:")
    for hidx in TARGET_HEADS:
        r = a03a05_results[f"h{hidx}"]
        print(f"  h{hidx}: L2={r['observed_l2']:.4f}  "
              f"KL(A03→A05)={r['kl_a03_to_a05']:.4f}  "
              f"KL(A05→A03)={r['kl_a05_to_a03']:.4f}  "
              f"perm_p={r['permutation_p_l2']:.4f}  "
              f"(null: mean={r['null_mean']:.4f}  std={r['null_std']:.4f})")

    # Qualitative: is A05's h0 pattern structurally different?
    h0_idx     = sim_labels.index("h0_A05")
    h0_others  = [sim_labels.index(f"h0_{lbl}") for lbl in SYS_LABELS if lbl != "A05"]
    a05_sims   = sim_matrix[h0_idx, h0_others]
    all_off_diag = sim_matrix[np.ix_(
        [sim_labels.index(f"h0_{l}") for l in SYS_LABELS],
        [sim_labels.index(f"h0_{l}") for l in SYS_LABELS]
    )]
    mask_diag = ~np.eye(len(SYS_LABELS), dtype=bool)
    median_cross = np.median(all_off_diag[mask_diag])
    a05_mean_sim = float(a05_sims.mean())
    a05_r = a03a05_results["h0"]
    qualitative = (
        f"h0/A05 mean similarity to other h0 patterns = {a05_mean_sim:.4f} "
        f"(cross-system median = {median_cross:.4f}); "
        f"A03↔A05 L2={a05_r['observed_l2']:.4f} p={a05_r['permutation_p_l2']:.4f} — "
        + ("A05 h0 pattern is a significant outlier vs other attacks."
           if a05_r["permutation_p_l2"] < 0.05 and a05_mean_sim < median_cross
           else "A05 h0 pattern is not a statistically significant outlier from other attacks.")
    )
    print(f"\nQualitative (A05 h0 structure): {qualitative}")

    print(f"\nResults: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
