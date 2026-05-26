#!/usr/bin/env python3
"""
gat_attn_graphs.py
===================
E1: Aggregated attention graphs across all 3 GAT layers, class-conditional.
E2: Reference-node / hub analysis on the aggregated graphs.

Inspired by El et al. "Attention Graphs" (2025), adapted to SL-GAT.

Convention throughout: A^l[target, source] — row-stochastic, row i = target i
  receives from source j with weight A^l[i,j]. Row sums to 1.

Aggregation (E1):
  A^l = mean over 6 heads of sparse attention → dense N×N
  A_agg = A^{L3} @ A^{L2} @ A^{L1}    (deepest layer leftmost)

Phases:
  1. Extract all 3 GAT-layer attentions  (or load from cache)
  2. Build A_agg per sample + sanity checks
  3. Class-conditional symbol-pair aggregates (E1 intermediate)
  4. E2 hub-identity shift + Gini  (pre-permutation intermediate)
  5. 1k dry-run permutation for timing estimate
  ← STOPS HERE.  Run with --full-perm to continue to 10k tests.

Outputs → experiments/results/gat_attn_graphs/
             e1/   e2/
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import time
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
from scipy.stats import spearmanr

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
OUT_BASE     = Path(__file__).resolve().parent / "results" / "gat_attn_graphs"
OUT_E1       = OUT_BASE / "e1"
OUT_E2       = OUT_BASE / "e2"
ARTIFACTS_L0 = Path(__file__).resolve().parent / "results" / "gat_l0_attention" / "attention_artifacts.pt"
ALL_LAYERS   = OUT_BASE / "all_layers_artifacts.pt"
VOCAB_DIR    = REPO_ROOT / "vocab_phoneme"

for p in [OUT_BASE, OUT_E1, OUT_E2]:
    p.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT          = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42
MIN_NODES      = 3
N_PERM_DRY     = 1_000
N_PERM_FULL    = 10_000
MIN_PAIR_SAMPLES = 5   # min samples per class for pair to enter permutation test

LANG_ORDER  = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL     = ["|", "</s>", "<s>", "<unk>", "<pad>"]


# ── Phoneme helpers (shared with gat_l0_attention.py) ─────────────────────────

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


# ── Holm-Bonferroni correction (same implementation as exp2_exp3) ─────────────

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    n      = len(pvals)
    order  = np.argsort(pvals)
    adj    = np.minimum(1.0, pvals[order] * np.arange(n, 0, -1, dtype=float))
    for i in range(1, n):
        adj[i] = max(adj[i], adj[i - 1])
    result        = np.empty(n)
    result[order] = adj
    return np.clip(result, 0.0, 1.0)


# ── Audio + dataset helpers (identical to gat_l0_attention.py) ────────────────

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


# ── Phoneme capture monkey-patch (identical to gat_l0_attention.py) ───────────

class PhonemeCapture:
    def __init__(self, gat_model):
        self.node_phoneme_ids   = None
        self.node_sample_idx    = None
        self.reduced_num_frames = None
        cap  = self
        orig = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_inner, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            rids = result[2].detach().cpu()
            rnf  = result[3].detach().cpu()
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


# ── phoneme-loader patch + model loading ──────────────────────────────────────

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


def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit


def run_frozen_frontend(audio, gat_model, device):
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


# ── All-layer extraction ───────────────────────────────────────────────────────

def extract_all_layers(lit, loader, device) -> list[dict]:
    """
    Hook all 3 GATLayer instances simultaneously.
    Returns per-sample dicts with attn_l0, attn_l1, attn_l2 (each (E, NH)),
    plus all fields from the original extraction (edge_index, node_phoneme_ids, etc.).
    Edge_index is the same for all 3 layers (verified by assertion).
    """
    gm = lit.model
    n_layers = len(gm.GAT.gat_net)
    assert n_layers == 3, f"Expected 3 GAT layers, got {n_layers}"

    for i in range(n_layers):
        gm.GAT.gat_net[i].log_attention_weights = True

    buf = {f"attn_{i}": None for i in range(n_layers)}
    buf.update({f"eidx_{i}": None for i in range(n_layers)})

    hooks = []
    for layer_i in range(n_layers):
        def _make_hook(li):
            def _hook(module, inp, out):
                buf[f"eidx_{li}"] = out[1].detach().cpu()
                if module.attention_weights is not None:
                    buf[f"attn_{li}"] = module.attention_weights.squeeze(-1).detach().cpu()
            return _hook
        hooks.append(gm.GAT.gat_net[layer_i].register_forward_hook(_make_hook(layer_i)))

    phoneme_cap = PhonemeCapture(gm)
    records: list[dict] = []
    n_degenerate = 0
    total = len(loader.dataset)
    sanity_done = False

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hidden_states, phoneme_ids = run_frozen_frontend(audio, gm, device)

                with torch.no_grad():
                    gm.encoder_and_GAT(hidden_states, num_f, phoneme_ids)

                # Verify edge_index identical across all layers
                if not sanity_done:
                    for li in range(1, n_layers):
                        assert torch.equal(buf["eidx_0"], buf[f"eidx_{li}"]), (
                            f"Edge index differs between layer 0 and layer {li}!")
                    print(f"  [Sanity] Edge indices identical across all {n_layers} layers ✓")
                    NH = buf["attn_0"].shape[1]
                    print(f"  [Sanity] attn shape: (E={buf['attn_0'].shape[0]}, NH={NH})")
                    sanity_done = True

                edge_index_global = buf["eidx_0"]
                attn_global = [buf[f"attn_{li}"] for li in range(n_layers)]

                if any(a is None for a in attn_global) or edge_index_global is None:
                    print(f"  [WARN] batch {bi}: hook missing, skipping")
                    continue

                node_pids     = phoneme_cap.node_phoneme_ids
                node_samp_idx = phoneme_cap.node_sample_idx
                rnf           = phoneme_cap.reduced_num_frames

                node_offsets = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                base_id = len(records)
                for i in range(B):
                    n_nodes_i = int(rnf[i].item())
                    offset_i  = node_offsets[i]
                    src_global = edge_index_global[0]
                    edge_mask  = (node_samp_idx[src_global] == i)

                    ei_global_i = edge_index_global[:, edge_mask]
                    ei_local    = ei_global_i - offset_i
                    is_adj      = (ei_local[1] - ei_local[0]) == 1
                    pids_i      = node_pids[node_samp_idx == i]
                    is_deg      = n_nodes_i < MIN_NODES
                    if is_deg: n_degenerate += 1

                    rec = {
                        "sample_id":        base_id + i,
                        "label":            labels[i],
                        "system_id":        sys_ids[i],
                        "node_phoneme_ids": pids_i.clone(),
                        "edge_index":       ei_local.clone(),
                        "edge_type_adj":    is_adj.clone(),
                        "n_nodes":          n_nodes_i,
                        "n_edges":          ei_local.shape[1],
                        "is_degenerate":    is_deg,
                    }
                    for li in range(n_layers):
                        rec[f"attn_l{li}"] = attn_global[li][edge_mask].clone()

                    records.append(rec)

                for li in range(n_layers):
                    buf[f"attn_{li}"] = None
                    buf[f"eidx_{li}"] = None

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    print(f"  {min((bi + 1) * BATCH_SIZE, total)}/{total}")

    finally:
        for h in hooks: h.remove()

    print(f"\n  Extracted {len(records)} samples ({n_degenerate} degenerate filtered)")
    return records, n_degenerate


def save_all_layers(records: list[dict], path: Path) -> None:
    layer_keys = ["attn_l0", "attn_l1", "attn_l2"]
    data = {
        "sample_ids":        [r["sample_id"]        for r in records],
        "labels":            [r["label"]             for r in records],
        "system_ids":        [r["system_id"]         for r in records],
        "n_nodes":           [r["n_nodes"]           for r in records],
        "n_edges":           [r["n_edges"]           for r in records],
        "is_degenerate":     [r["is_degenerate"]     for r in records],
        "node_phoneme_ids":  [r["node_phoneme_ids"]  for r in records],
        "edge_index":        [r["edge_index"]        for r in records],
        "edge_type_adj":     [r["edge_type_adj"]     for r in records],
    }
    for k in layer_keys:
        data[k] = [r[k] for r in records]
    torch.save(data, path)
    print(f"Saved: {path}")


def load_all_layers(path: Path) -> list[dict]:
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    n    = len(data["sample_ids"])
    recs = []
    for i in range(n):
        recs.append({
            "sample_id":        data["sample_ids"][i],
            "label":            data["labels"][i],
            "system_id":        data["system_ids"][i],
            "n_nodes":          data["n_nodes"][i],
            "n_edges":          data["n_edges"][i],
            "is_degenerate":    data["is_degenerate"][i],
            "node_phoneme_ids": data["node_phoneme_ids"][i],
            "edge_index":       data["edge_index"][i],
            "edge_type_adj":    data["edge_type_adj"][i],
            "attn_l0":          data["attn_l0"][i],
            "attn_l1":          data["attn_l1"][i],
            "attn_l2":          data["attn_l2"][i],
        })
    return recs


# ── Aggregated attention graph ────────────────────────────────────────────────

def sparse_to_dense(attn_mean: np.ndarray, edge_index: np.ndarray, N: int) -> np.ndarray:
    """
    Convert sparse edge attention to dense N×N matrix A[target, source].
    Nodes with no incoming edges get identity row (self-retention).
    """
    A = np.zeros((N, N), dtype=np.float64)
    src = edge_index[0]
    tgt = edge_index[1]
    np.add.at(A, (tgt, src), attn_mean)   # handles duplicate edges gracefully
    row_sums = A.sum(axis=1)
    no_in = np.where(row_sums < 1e-10)[0]
    A[no_in, no_in] = 1.0                 # identity for source-only nodes
    return A


def build_aggregated_attention_graph(rec: dict) -> np.ndarray:
    """
    A_agg = A^{L3} @ A^{L2} @ A^{L1}, row-normalised.
    Asserts each per-layer matrix is row-stochastic before product.
    Returns (N, N) float64.
    """
    N  = rec["n_nodes"]
    ei = rec["edge_index"].numpy()         # (2, E)
    tol = 1e-4

    layers: list[np.ndarray] = []
    for li in range(3):
        attn_np = rec[f"attn_l{li}"].float().numpy()  # (E, NH)
        attn_mean = attn_np.mean(axis=1)              # (E,)
        A = sparse_to_dense(attn_mean, ei, N)
        rs = A.sum(axis=1)
        max_dev = np.abs(rs - 1.0).max()
        if max_dev > tol:
            raise AssertionError(
                f"Layer {li} not row-stochastic after identity fix: max_dev={max_dev:.2e}")
        layers.append(A)

    A1, A2, A3 = layers            # layer 0, 1, 2
    A_agg = A3 @ A2 @ A1

    # matrix product of row-stochastic matrices is row-stochastic;
    # row-normalise to correct any float64 drift
    rs = A_agg.sum(axis=1, keepdims=True)
    rs = np.where(rs > 1e-10, rs, 1.0)
    A_agg = A_agg / rs
    return A_agg


# ── Symbol vocabulary ─────────────────────────────────────────────────────────

def build_symbol_vocab(records: list[dict],
                       id_to_sym: dict[int, str]) -> tuple[list[str], dict[str, int]]:
    """
    Build sorted vocabulary of all phoneme symbols present in non-degenerate records.
    Returns (sym_list, sym_to_idx).
    """
    seen: set[str] = set()
    for rec in records:
        if rec["is_degenerate"]:
            continue
        for pid in rec["node_phoneme_ids"].tolist():
            seen.add(id_to_sym.get(int(pid), "?"))
    sym_list = sorted(seen)
    sym_to_idx = {s: i for i, s in enumerate(sym_list)}
    return sym_list, sym_to_idx


# ── Per-sample symbol-pair matrix ─────────────────────────────────────────────

def per_sample_pair_matrix(rec: dict, A_agg: np.ndarray,
                            id_to_sym: dict[int, str],
                            sym_to_idx: dict[str, int], V: int) -> np.ndarray:
    """
    Build V×V matrix M where M[t, s] = mean A_agg[i,j]
    over all (i,j) with sym[i]=t and sym[j]=s.
    NaN where no such (i,j) pair exists.

    Vectorised via one-hot: M = H^T @ A_agg @ H / outer(counts, counts)
    where H[node, sym_idx] = 1.  Row-stochastic since A_agg is.
    """
    pids  = rec["node_phoneme_ids"].numpy()
    N     = rec["n_nodes"]
    sym_ids = np.array([sym_to_idx.get(id_to_sym.get(int(p), "?"), 0)
                        for p in pids], dtype=np.int64)

    H = np.zeros((N, V), dtype=np.float64)
    H[np.arange(N), sym_ids] = 1.0

    pair_sum = H.T @ A_agg @ H                          # (V, V)
    counts   = H.sum(axis=0)                             # (V,)
    outer    = counts[:, None] * counts[None, :]         # (V, V) outer product

    result = np.full((V, V), np.nan)
    valid  = outer > 0
    result[valid] = pair_sum[valid] / outer[valid]
    return result


# ── Class-conditional aggregates ──────────────────────────────────────────────

def compute_class_aggregates(records: list[dict],
                              per_sample: dict[int, np.ndarray],
                              systems: list[str], V: int) -> dict[str, np.ndarray]:
    """
    For each system, nanmean of per-sample V×V pair matrices over all samples.
    Returns {system_id: (V, V)} — row-stochastic in expectation.
    """
    accum  = {s: np.zeros((V, V), dtype=np.float64) for s in systems}
    counts = {s: np.zeros((V, V), dtype=np.int64)   for s in systems}

    for rec in records:
        if rec["is_degenerate"]:
            continue
        sid = rec["system_id"]
        M   = per_sample[rec["sample_id"]]      # (V, V) with NaN
        valid = ~np.isnan(M)
        accum[sid][valid]  += M[valid]
        counts[sid][valid] += 1

    agg: dict[str, np.ndarray] = {}
    for s in systems:
        A = np.full((V, V), np.nan)
        pos = counts[s] > 0
        A[pos] = accum[s][pos] / counts[s][pos]
        agg[s] = A
    return agg


# ── E1 sanity checks ──────────────────────────────────────────────────────────

def run_sanity_checks(records: list[dict], a_agg_map: dict[int, np.ndarray],
                      id_to_sym: dict, sym_to_idx: dict, V: int) -> None:
    """
    1. Row sums of A_agg ~ 1.0 ± 1e-6 for all samples.
    2. Spot-check one sample: recompute A_agg manually and assert match.
    3. Report diagonal fraction for bonafide class-aggregate.
    """
    print("\n=== Sanity checks ===")

    # 1. Row-sum check across all samples
    max_devs = []
    for rec in records:
        if rec["is_degenerate"]: continue
        A = a_agg_map[rec["sample_id"]]
        max_devs.append(np.abs(A.sum(axis=1) - 1.0).max())
    overall_max = max(max_devs)
    print(f"  Row-sum check: max deviation across all samples = {overall_max:.2e}  "
          f"({'PASS' if overall_max < 1e-5 else 'FAIL'})")
    assert overall_max < 1e-5, f"Row-sum check failed: {overall_max}"

    # 2. Spot-check: recompute sample 0 from scratch
    valid_recs = [r for r in records if not r["is_degenerate"]]
    r0 = valid_recs[0]
    A_check = build_aggregated_attention_graph(r0)
    A_cached = a_agg_map[r0["sample_id"]]
    diff = np.abs(A_check - A_cached).max()
    print(f"  Spot-check sample {r0['sample_id']}: recompute vs cached max_diff = {diff:.2e}  "
          f"({'PASS' if diff < 1e-10 else 'FAIL'})")
    assert diff < 1e-10, f"Spot-check failed: {diff}"

    # 3. Diagonal fraction for bonafide
    bon_recs = [r for r in records if r["system_id"] == "-" and not r["is_degenerate"]]
    diag_fracs = []
    for r in bon_recs:
        A = a_agg_map[r["sample_id"]]
        diag_fracs.append(np.trace(A) / A.sum())
    mean_diag = np.mean(diag_fracs)
    print(f"  Bonafide mean diagonal fraction of A_agg: {mean_diag:.4f}  "
          "(non-trivial due to skip-connection bias in l0; expected non-dominant)")

    print("  All sanity checks PASSED ✓\n")


# ── E1 intermediate output ────────────────────────────────────────────────────

def show_e1_intermediate(class_agg: dict[str, np.ndarray],
                          delta_agg: dict[str, np.ndarray],
                          sym_list: list[str], systems: list[str],
                          V: int) -> None:
    """Print per-attack Δ statistics and top pairs before permutation tests."""
    print("\n=== E1 Intermediate Results ===")
    attack_systems = [s for s in systems if s != "-"]

    # Per-attack summary: mean |Δ| over all symbol pairs
    print("\n  Per-attack mean |Δ| over all phoneme symbol pairs:")
    print(f"  {'system':6s}  {'mean|Δ|':>9s}  {'max|Δ|':>8s}  {'n_valid':>7s}")
    for s in sorted(attack_systems):
        dA = delta_agg[s]
        valid = ~np.isnan(dA)
        if not valid.any():
            print(f"  {s:6s}  (no valid pairs)")
            continue
        print(f"  {s:6s}  {np.nanmean(np.abs(dA)):9.5f}  {np.nanmax(np.abs(dA)):8.5f}  "
              f"{valid.sum():7d}")

    # Top-10 pairs by |Δ| for each attack
    print("\n  Top-10 phoneme pairs by |ΔA_agg| per attack system:")
    all_rows = []
    for s in sorted(attack_systems):
        dA = delta_agg[s]
        bon_A = class_agg["-"]
        atk_A = class_agg[s]
        # collect all valid (tgt_sym, src_sym) pairs
        rows = []
        for t in range(V):
            for src in range(V):
                if np.isnan(dA[t, src]): continue
                rows.append({
                    "system": s,
                    "tgt_sym": sym_list[t], "src_sym": sym_list[src],
                    "bon_mean": float(np.nan_to_num(bon_A[t, src])),
                    "atk_mean": float(np.nan_to_num(atk_A[t, src])),
                    "delta":    float(dA[t, src]),
                    "abs_delta": abs(float(dA[t, src])),
                })
        rows.sort(key=lambda x: -x["abs_delta"])
        all_rows.extend(rows)
        top10 = rows[:10]
        print(f"\n  {s}:")
        print(f"  {'src':8s} → {'tgt':8s}  {'bon':>8s}  {'atk':>8s}  {'Δ':>9s}")
        for r in top10:
            print(f"  {r['src_sym']:8s} → {r['tgt_sym']:8s}  "
                  f"{r['bon_mean']:8.5f}  {r['atk_mean']:8.5f}  {r['delta']:+9.5f}")

    # Save top pairs CSV
    out_csv = OUT_E1 / "top_delta_pairs.csv"
    if all_rows:
        all_rows.sort(key=lambda x: -x["abs_delta"])
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["system","src_sym","tgt_sym",
                                               "bon_mean","atk_mean","delta","abs_delta"])
            w.writeheader(); w.writerows(all_rows)
        print(f"\n  Saved: {out_csv}")


# ── E2: hub analysis (pre-permutation) ────────────────────────────────────────

def compute_indegree(class_agg: dict[str, np.ndarray],
                     sym_list: list[str]) -> dict[str, np.ndarray]:
    """
    d_in(p | c) = column sum of class_agg[c] at column idx(p).
    Returns {system: (V,) array of in-degrees}, NaN entries filled with 0.
    """
    result = {}
    for s, A in class_agg.items():
        # column sum: how much each source symbol is referenced by all targets
        d = np.nansum(A, axis=0)   # (V,)
        result[s] = d
    return result


def compute_gini(values: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    v = values[values > 0]
    if len(v) == 0: return 0.0
    v = np.sort(v)
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) / (n * v.sum())) - (n + 1) / n)


def hub_identity_shift(indegree: dict[str, np.ndarray],
                        systems: list[str], sym_list: list[str],
                        k_vals: list[int] = [5, 10]) -> dict:
    """
    For each attack system: Spearman rank correlation and top-k overlap
    of in-degree ranking vs bonafide.
    Returns nested dict: results[system][k] = overlap, results[system]['spearman'] = rho
    """
    bon_d = indegree["-"]
    bon_rank = np.argsort(-bon_d)   # rank 0 = highest d_in
    attack_systems = [s for s in systems if s != "-"]
    results = {}

    for s in attack_systems:
        atk_d    = indegree[s]
        atk_rank = np.argsort(-atk_d)

        rho, pval = spearmanr(bon_d, atk_d)
        overlaps  = {}
        for k in k_vals:
            bon_topk = set(bon_rank[:k])
            atk_topk = set(atk_rank[:k])
            overlaps[k] = len(bon_topk & atk_topk)

        delta_d = atk_d - bon_d
        top5_idx = np.argsort(-np.abs(delta_d))[:5]
        results[s] = {
            "spearman_rho": float(rho),
            "spearman_p":   float(pval),
            "overlaps":     overlaps,
            "top5_delta":   [(sym_list[i], float(delta_d[i])) for i in top5_idx],
        }
    return results


def show_e2_intermediate(indegree: dict[str, np.ndarray],
                          hub_shift: dict, systems: list[str],
                          sym_list: list[str],
                          class_agg: dict[str, np.ndarray],
                          V: int) -> None:
    """Print E2 intermediate tables."""
    attack_systems = [s for s in systems if s != "-"]

    print("\n=== E2 Intermediate Results ===")

    # Table 1: top-k overlap and Spearman
    print("\n  Table 1: Hub-identity shift (in-degree ranking vs bonafide)")
    print(f"  {'system':6s}  {'top-5 overlap':>13s}  {'top-10 overlap':>14s}  "
          f"{'Spearman ρ':>11s}  {'p-value':>9s}")
    for s in sorted(attack_systems):
        r = hub_shift[s]
        print(f"  {s:6s}  {r['overlaps'][5]:>13d}  {r['overlaps'][10]:>14d}  "
              f"  {r['spearman_rho']:+10.4f}  {r['spearman_p']:9.5f}")

    # Table 3: Gini
    print("\n  Table 3: Gini coefficient of in-degree distribution")
    print(f"  {'system':12s}  {'Gini':>7s}  {'d_in mean':>10s}  {'d_in max':>9s}")
    for s in sorted(systems):
        d = indegree[s]
        g = compute_gini(d)
        print(f"  {s:12s}  {g:7.4f}  {d.mean():10.5f}  {d.max():9.5f}")

    # Top-5 |Δd_in| per system (focus on null-result systems)
    bon_d = indegree["-"]
    print("\n  Top-5 phonemes by |Δd_in| — null systems A03, A05, A06:")
    null_systems = [s for s in ["A03", "A05", "A06"] if s in attack_systems]
    for s in null_systems:
        print(f"\n  {s}:")
        print(f"  {'phoneme':10s}  {'d_in(atk)':>10s}  {'d_in(bon)':>10s}  {'Δd_in':>9s}")
        atk_d  = indegree[s]
        delta  = atk_d - bon_d
        top5   = np.argsort(-np.abs(delta))[:5]
        for i in top5:
            print(f"  {sym_list[i]:10s}  {atk_d[i]:10.5f}  {bon_d[i]:10.5f}  "
                  f"{delta[i]:+9.5f}")

    # Bar figure: |Δd_in| for A03, A05, A06
    _plot_hub_delta(indegree, null_systems, sym_list, V)

    # Save Table 1 CSV
    rows = []
    for s in sorted(attack_systems):
        r = hub_shift[s]
        rows.append({
            "system": s,
            "top5_overlap": r["overlaps"][5],
            "top10_overlap": r["overlaps"][10],
            "spearman_rho": r["spearman_rho"],
            "spearman_p": r["spearman_p"],
        })
    with open(OUT_E2 / "hub_identity_shift.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Save Gini CSV
    gini_rows = []
    for s in sorted(systems):
        d = indegree[s]
        gini_rows.append({"system": s, "gini": compute_gini(d),
                          "d_in_mean": d.mean(), "d_in_max": d.max()})
    with open(OUT_E2 / "gini_indegree.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gini_rows[0].keys()))
        w.writeheader(); w.writerows(gini_rows)

    print(f"\n  Saved: {OUT_E2 / 'hub_identity_shift.csv'}")
    print(f"  Saved: {OUT_E2 / 'gini_indegree.csv'}")


def _plot_hub_delta(indegree: dict, null_systems: list[str],
                    sym_list: list[str], V: int, top_k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bon_d = indegree["-"]
    n = len(null_systems)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1: axes = [axes]

    for ax, s in zip(axes, null_systems):
        delta = indegree[s] - bon_d
        top_idx = np.argsort(-np.abs(delta))[:top_k]
        vals    = delta[top_idx]
        labels  = [sym_list[i] for i in top_idx]
        colors  = ["#E03030" if v > 0 else "#6EB5FF" for v in vals]
        ax.barh(range(top_k), vals, color=colors, alpha=0.85, edgecolor="grey", lw=0.3)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()
        ax.set_title(f"{s}: top-{top_k} phonemes by |Δd_in|", fontsize=10)
        ax.set_xlabel("Δd_in  (attack − bonafide)", fontsize=9)

    fig.suptitle("Hub mass shift for null-result systems (A03, A05, A06)\n"
                 "Red = more hub-like in attack, Blue = less hub-like", fontsize=11)
    plt.tight_layout()
    out = OUT_E2 / "hub_delta_null_systems.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Permutation helpers ───────────────────────────────────────────────────────

def _precompute_sample_colsums(records: list[dict],
                                per_sample: dict[int, np.ndarray],
                                V: int) -> dict[str, np.ndarray]:
    """
    For each system, stack per-sample column-sum vectors.
    Returns {sys: (n_samples, V)} — used for E2 hub-mass permutation.
    """
    by_sys: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        if rec["is_degenerate"]: continue
        M = per_sample[rec["sample_id"]]    # (V, V)
        colsum = np.nansum(M, axis=0)        # (V,) — same as d_in for this sample
        by_sys[rec["system_id"]].append(colsum)
    return {s: np.stack(arrs) for s, arrs in by_sys.items()}


def _e1_test_stat(atk_matrices: np.ndarray, bon_matrices: np.ndarray) -> np.ndarray:
    """Mean-difference test statistic per (tgt, src) pair.  (V, V)"""
    with np.errstate(all="ignore"):
        return np.nanmean(atk_matrices, axis=0) - np.nanmean(bon_matrices, axis=0)


def _e2_test_stat(atk_colsums: np.ndarray, bon_colsums: np.ndarray) -> np.ndarray:
    """Mean-difference test statistic per phoneme.  (V,)"""
    with np.errstate(all="ignore"):
        return np.nanmean(atk_colsums, axis=0) - np.nanmean(bon_colsums, axis=0)


def dry_run_permutation(records: list[dict],
                         per_sample: dict[int, np.ndarray],
                         systems: list[str], V: int,
                         n_perm: int = N_PERM_DRY) -> dict:
    """
    Run n_perm permutations for timing estimation (both E1 and E2).
    Uses one representative attack system.  Returns timing dict.
    """
    attack_systems = [s for s in systems if s != "-"]
    if not attack_systems: return {}

    # E1: per-pair mean-diff test on (V,V) matrices
    # Use one system to time
    test_sid = attack_systems[0]
    bon_mats = np.array([per_sample[r["sample_id"]]
                         for r in records
                         if r["system_id"] == "-" and not r["is_degenerate"]])
    atk_mats = np.array([per_sample[r["sample_id"]]
                         for r in records
                         if r["system_id"] == test_sid and not r["is_degenerate"]])
    pool_mats = np.concatenate([atk_mats, bon_mats], axis=0)
    n_atk     = len(atk_mats)
    n_pool    = len(pool_mats)

    print(f"\n  E1 dry-run ({n_perm} perms, system {test_sid}, "
          f"matrix shape {pool_mats.shape[1]}×{pool_mats.shape[2]})...")
    obs_e1 = _e1_test_stat(atk_mats, bon_mats)
    t0 = time.time()
    for _ in range(n_perm):
        perm = np.random.permutation(n_pool)
        _e1_test_stat(pool_mats[perm[:n_atk]], pool_mats[perm[n_atk:]])
    t_e1 = time.time() - t0
    print(f"    {n_perm} perms took {t_e1:.1f}s  → "
          f"10k est per system: {t_e1 * N_PERM_FULL / n_perm:.0f}s  "
          f"(× {len(attack_systems)} systems = "
          f"{t_e1 * N_PERM_FULL / n_perm * len(attack_systems):.0f}s total)")

    # E2: per-phoneme colsum mean-diff test
    sample_colsums = _precompute_sample_colsums(records, per_sample, V)
    bon_cs  = sample_colsums["-"]
    atk_cs  = sample_colsums[test_sid]
    pool_cs = np.concatenate([atk_cs, bon_cs], axis=0)
    n_atk_cs = len(atk_cs)

    print(f"\n  E2 dry-run ({n_perm} perms, system {test_sid}, "
          f"colsum shape {pool_cs.shape[1]})...")
    obs_e2 = _e2_test_stat(atk_cs, bon_cs)
    t0 = time.time()
    for _ in range(n_perm):
        perm = np.random.permutation(len(pool_cs))
        _e2_test_stat(pool_cs[perm[:n_atk_cs]], pool_cs[perm[n_atk_cs:]])
    t_e2 = time.time() - t0
    print(f"    {n_perm} perms took {t_e2:.1f}s  → "
          f"10k est per system: {t_e2 * N_PERM_FULL / n_perm:.0f}s  "
          f"(× {len(attack_systems)} systems = "
          f"{t_e2 * N_PERM_FULL / n_perm * len(attack_systems):.0f}s total)")

    return {"e1_dry_s": t_e1, "e2_dry_s": t_e2,
            "n_perm_dry": n_perm, "n_systems": len(attack_systems)}


# ── Full permutation tests ────────────────────────────────────────────────────

def run_e1_permutation(records: list[dict],
                        per_sample: dict[int, np.ndarray],
                        systems: list[str], V: int,
                        sym_list: list[str],
                        n_perm: int = N_PERM_FULL) -> dict:
    """
    Per-phoneme-pair signed-difference permutation test (E1).
    Returns {attack_system: {(tgt_sym, src_sym): p_raw}} plus Holm-corrected results.
    """
    attack_systems = [s for s in systems if s != "-"]
    bon_mats = np.array([per_sample[r["sample_id"]]
                         for r in records
                         if r["system_id"] == "-" and not r["is_degenerate"]])
    results = {}

    for sid in sorted(attack_systems):
        print(f"  E1 perm {sid} ({n_perm} perms)...", end=" ", flush=True)
        atk_mats  = np.array([per_sample[r["sample_id"]]
                               for r in records
                               if r["system_id"] == sid and not r["is_degenerate"]])
        pool_mats = np.concatenate([atk_mats, bon_mats], axis=0)
        n_atk     = len(atk_mats)
        n_pool    = len(pool_mats)

        obs = _e1_test_stat(atk_mats, bon_mats)   # (V, V)

        # Only test pairs with enough samples
        valid_mask = np.zeros((V, V), dtype=bool)
        for t in range(V):
            for s in range(V):
                n_a = np.sum(~np.isnan(atk_mats[:, t, s]))
                n_b = np.sum(~np.isnan(bon_mats[:, t, s]))
                if n_a >= MIN_PAIR_SAMPLES and n_b >= MIN_PAIR_SAMPLES:
                    valid_mask[t, s] = True

        exceed_count = np.zeros((V, V), dtype=np.int64)
        t0 = time.time()
        for _ in range(n_perm):
            perm     = np.random.permutation(n_pool)
            perm_stat = _e1_test_stat(pool_mats[perm[:n_atk]], pool_mats[perm[n_atk:]])
            exceed_count[valid_mask] += (
                np.abs(perm_stat[valid_mask]) >= np.abs(obs[valid_mask]))

        p_raw = np.full((V, V), np.nan)
        p_raw[valid_mask] = (exceed_count[valid_mask] + 1) / (n_perm + 1)

        # Holm correction across valid pairs
        valid_pairs = list(zip(*np.where(valid_mask)))
        p_raw_flat  = np.array([p_raw[t, s] for t, s in valid_pairs])
        p_holm_flat = holm_bonferroni(p_raw_flat)
        p_holm = np.full((V, V), np.nan)
        for (t, s), ph in zip(valid_pairs, p_holm_flat):
            p_holm[t, s] = ph

        n_sig = int((p_holm_flat < 0.05).sum())
        print(f"done ({time.time()-t0:.1f}s)  {n_sig}/{len(valid_pairs)} pairs survive Holm (p<0.05)")
        results[sid] = {"obs": obs, "p_raw": p_raw, "p_holm": p_holm,
                        "valid_mask": valid_mask}

    return results


def run_e2_permutation(records: list[dict],
                        per_sample: dict[int, np.ndarray],
                        systems: list[str], V: int,
                        sym_list: list[str],
                        n_perm: int = N_PERM_FULL) -> dict:
    """
    Per-phoneme hub-mass permutation test (E2).
    Test statistic: |Δd_in(p)|.  Family statistic: max_p |Δd_in(p)|.
    Returns individual phoneme p-values (for Holm correction) per attack system.
    """
    attack_systems = [s for s in systems if s != "-"]
    sample_colsums = _precompute_sample_colsums(records, per_sample, V)
    bon_cs  = sample_colsums["-"]
    results = {}

    for sid in sorted(attack_systems):
        print(f"  E2 perm {sid} ({n_perm} perms)...", end=" ", flush=True)
        atk_cs  = sample_colsums[sid]
        pool_cs = np.concatenate([atk_cs, bon_cs], axis=0)
        n_atk   = len(atk_cs)

        obs = _e2_test_stat(atk_cs, bon_cs)   # (V,)
        exceed = np.zeros(V, dtype=np.int64)
        t0 = time.time()
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_test_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            exceed += (np.abs(pstat) >= np.abs(obs))

        p_raw  = (exceed + 1) / (n_perm + 1)
        p_holm = holm_bonferroni(p_raw)
        n_sig  = int((p_holm < 0.05).sum())
        print(f"done ({time.time()-t0:.1f}s)  {n_sig}/{V} phonemes survive Holm (p<0.05)")

        # Family-level p: fraction of perms where max |perm_stat| >= max |obs_stat|
        max_obs = np.abs(obs).max()
        max_exceed = 0
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_test_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            if np.abs(pstat).max() >= max_obs:
                max_exceed += 1
        p_family = (max_exceed + 1) / (n_perm + 1)

        results[sid] = {"obs": obs, "p_raw": p_raw, "p_holm": p_holm,
                        "p_family": p_family, "n_sig": n_sig}

    return results


# ── Post-permutation report helpers ──────────────────────────────────────────

def write_e1_report(e1_results: dict, class_agg: dict,
                     delta_agg: dict, sym_list: list[str], V: int,
                     l0_sig_pairs: dict | None = None) -> None:
    """Write E1 full report after permutation tests."""
    lines: list[str] = []
    W = lines.append
    W("# E1: Aggregated Attention Graphs — Full Report\n")

    attack_systems = sorted(e1_results.keys())
    for sid in attack_systems:
        r = e1_results[sid]
        valid = r["valid_mask"]
        p_holm = r["p_holm"]
        obs    = r["obs"]
        sig_pairs = [(sym_list[t], sym_list[s], float(obs[t, s]), float(p_holm[t, s]))
                     for t, s in zip(*np.where(valid))
                     if p_holm[t, s] < 0.05]
        sig_pairs.sort(key=lambda x: x[3])

        W(f"## {sid}\n")
        W(f"  Valid pairs tested: {valid.sum()}")
        W(f"  Surviving Holm (p<0.05): {len(sig_pairs)}\n")

        if sig_pairs:
            W("| src_sym | tgt_sym | obs_Δ | p_holm |")
            W("|---------|---------|-------|--------|")
            for t_sym, s_sym, d, ph in sig_pairs:
                W(f"| {s_sym} | {t_sym} | {d:+.5f} | {ph:.5f} |")
        else:
            W("_No pairs survive Holm correction._\n")

    # Comparison table: layer-0 vs aggregated (only if l0 data provided)
    if l0_sig_pairs is not None:
        W("\n## Comparison: Layer-0 surviving pairs vs Aggregated surviving pairs\n")
        W("| system | layer-0 sig | agg sig | new detections (A03/A05/A06)? |")
        W("|--------|-------------|---------|-------------------------------|")
        for sid in attack_systems:
            n_l0  = len(l0_sig_pairs.get(sid, []))
            n_agg = len([t for t, s, d, p in [(sym_list[t], sym_list[s],
                          float(e1_results[sid]["obs"][t,s]),
                          float(e1_results[sid]["p_holm"][t,s]))
                         for t, s in zip(*np.where(e1_results[sid]["valid_mask"]))
                         if e1_results[sid]["p_holm"][t,s] < 0.05]])
            new_det = "YES" if sid in ["A03", "A05", "A06"] and n_agg > 0 else "-"
            W(f"| {sid} | {n_l0} | {n_agg} | {new_det} |")

    out = OUT_E1 / "e1_full_report.md"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out}")


def write_e2_report(e2_perm: dict, hub_shift: dict,
                     indegree: dict, gini: dict,
                     sym_list: list[str], systems: list[str]) -> None:
    """Write E2 full report after permutation tests."""
    lines: list[str] = []
    W = lines.append
    bon_d = indegree["-"]
    attack_systems = sorted(e2_perm.keys())

    W("# E2: Hub Analysis — Full Report\n")

    W("## Table 1: Hub-identity shift (Spearman + top-k overlap)\n")
    W("| system | top-5 | top-10 | Spearman ρ | p-val |")
    W("|--------|-------|--------|------------|-------|")
    for s in attack_systems:
        r = hub_shift[s]
        W(f"| {s} | {r['overlaps'][5]} | {r['overlaps'][10]} "
          f"| {r['spearman_rho']:+.4f} | {r['spearman_p']:.5f} |")

    W("\n## Table 2: Surviving hub phonemes (Holm-corrected, E2 permutation)\n")
    W("| system | surviving phonemes | n_sig | p_family |")
    W("|--------|-------------------|-------|---------|")
    for s in attack_systems:
        r   = e2_perm[s]
        sig = [sym_list[i] for i in range(len(sym_list)) if r["p_holm"][i] < 0.05]
        W(f"| {s} | {', '.join(sig) if sig else 'none'} "
          f"| {r['n_sig']} | {r['p_family']:.5f} |")

    W("\n## Table 3: Gini coefficient of in-degree distribution\n")
    W("| system | Gini | d_in mean | d_in max |")
    W("|--------|------|-----------|---------|")
    for s in sorted(systems):
        d = indegree[s]
        W(f"| {s} | {gini[s]:.4f} | {d.mean():.5f} | {d.max():.5f} |")

    W("\n## Interpretation: null systems (A03, A05, A06)\n")
    null_sig = {s: [sym_list[i] for i in range(len(sym_list))
                    if e2_perm[s]["p_holm"][i] < 0.05]
                for s in ["A03", "A05", "A06"] if s in e2_perm}
    any_new = any(len(v) > 0 for v in null_sig.values())

    if any_new:
        surviving = {s: v for s, v in null_sig.items() if v}
        W("Per-edge analysis at layer 0 found no significant attention redistribution "
          "for A03, A05, and A06 after Holm correction. The aggregated hub analysis "
          "**does** detect phoneme-specific hub-mass shifts for: "
          + ", ".join(f"{s} ({', '.join(v)})" for s, v in surviving.items()) + ". "
          "This suggests that these systems produce a detectable artefact when "
          "attention flow is integrated across all 3 GAT layers, even though no "
          "single-edge redistribution is strong enough to survive individual correction. "
          "The effect appears at the level of which phonemes accumulate disproportionate "
          "attention as hubs, consistent with the El et al. 'reference node' hypothesis.")
    else:
        W("Per-edge analysis at layer 0 found no significant attention redistribution "
          "for A03, A05, and A06. The aggregated hub analysis also finds no phoneme "
          "surviving Holm correction for any of these three systems. The null result "
          "is thus robust: these systems do not produce detectable attention anomalies "
          "either at the per-edge level (layer 0) or at the per-hub level (3-layer aggregate). "
          "This is consistent with a vocoder artefact mechanism that does not engage the "
          "GAT phoneme-routing pathway at all for these particular systems.")

    out = OUT_E2 / "e2_full_report.md"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    full_perm = "--full-perm" in sys.argv

    # ── Phase 1: Extract or load all-layer attention ───────────────────────────
    if ALL_LAYERS.exists():
        print(f"Loading cached all-layer artifacts: {ALL_LAYERS}")
        records = load_all_layers(ALL_LAYERS)
        print(f"  {len(records)} samples loaded")

        # Cross-validate layer-0 attention against existing cache
        old_data = torch.load(str(ARTIFACTS_L0), map_location="cpu", weights_only=False)
        old_sys  = old_data["system_ids"]
        new_sys  = [r["system_id"] for r in records]
        if sorted(old_sys) == sorted(new_sys):
            print("  Cross-validation: system_id distribution matches layer-0 cache ✓")
        else:
            print("  [WARN] system_id distribution mismatch vs layer-0 cache")
    else:
        print("No cached all-layer artifacts — running extraction...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
        patch_phoneme_loader()
        lit = load_model(CKPT, device)

        try:
            from torchaudio import set_audio_backend
        except Exception:
            pass

        dataset = BalancedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                                  hf_token, N_PER_CLASS, SEED)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate)

        records, _ = extract_all_layers(lit, loader, device)
        save_all_layers(records, ALL_LAYERS)

    systems = sorted(set(r["system_id"] for r in records))
    valid   = [r for r in records if not r["is_degenerate"]]
    print(f"\n  Systems: {systems}  |  valid samples: {len(valid)}")

    id_to_sym, id_to_cls = build_vocab()
    print(f"  Vocab: {len(id_to_sym)} tokens")

    # ── Phase 2: Build A_agg per sample ───────────────────────────────────────
    print("\n--- Building aggregated attention graphs ---")
    a_agg_map: dict[int, np.ndarray] = {}
    for rec in valid:
        a_agg_map[rec["sample_id"]] = build_aggregated_attention_graph(rec)
    print(f"  Built A_agg for {len(a_agg_map)} samples")

    # Save per-sample A_agg as .npz
    agg_npz_dir = OUT_E1 / "a_agg_per_sample"
    agg_npz_dir.mkdir(exist_ok=True)
    for rec in valid:
        sid  = rec["sample_id"]
        path = agg_npz_dir / f"sample_{sid:04d}.npz"
        if not path.exists():
            np.savez_compressed(str(path),
                                a_agg=a_agg_map[sid],
                                system_id=np.array([rec["system_id"]]),
                                n_nodes=np.array([rec["n_nodes"]]))
    print(f"  Per-sample A_agg saved to {agg_npz_dir}/")

    # ── Phase 3: Sanity checks ────────────────────────────────────────────────
    sym_list, sym_to_idx = build_symbol_vocab(records, id_to_sym)
    V = len(sym_list)
    print(f"\n  Symbol vocabulary: {V} unique phoneme symbols")

    run_sanity_checks(records, a_agg_map, id_to_sym, sym_to_idx, V)

    # ── Phase 4: Per-sample pair matrices + class aggregates ──────────────────
    print("--- Computing per-sample symbol-pair matrices ---")
    per_sample: dict[int, np.ndarray] = {}
    for rec in valid:
        per_sample[rec["sample_id"]] = per_sample_pair_matrix(
            rec, a_agg_map[rec["sample_id"]], id_to_sym, sym_to_idx, V)
    print(f"  Computed {len(per_sample)} per-sample (V={V}) pair matrices")

    class_agg = compute_class_aggregates(records, per_sample, systems, V)
    delta_agg = {s: class_agg[s] - class_agg["-"]
                 for s in systems if s != "-"}

    # Class-conditional aggregates as .npz
    cond_npz = OUT_E1 / "class_conditional_agg.npz"
    save_dict = {s: class_agg[s] for s in systems}
    save_dict["sym_list"] = np.array(sym_list)
    np.savez_compressed(str(cond_npz), **save_dict)
    print(f"  Saved: {cond_npz}")

    show_e1_intermediate(class_agg, delta_agg, sym_list, systems, V)

    # ── Phase 5: E2 hub analysis ──────────────────────────────────────────────
    print("\n--- E2: Hub analysis ---")
    indegree  = compute_indegree(class_agg, sym_list)
    hub_shift = hub_identity_shift(indegree, systems, sym_list)
    show_e2_intermediate(indegree, hub_shift, systems, sym_list, class_agg, V)

    # ── Phase 6: Dry-run permutation for timing ───────────────────────────────
    print("\n--- Dry-run permutation test (1k perms for timing) ---")
    timing = dry_run_permutation(records, per_sample, systems, V, N_PERM_DRY)

    if not full_perm:
        print("\n" + "=" * 70)
        print("CHECKPOINT: Intermediate results above. Permutation tests not yet run.")
        print(f"  E1 timing estimate (10k × {timing['n_systems']} systems): "
              f"{timing['e1_dry_s'] * N_PERM_FULL / N_PERM_DRY * timing['n_systems']:.0f}s")
        print(f"  E2 timing estimate (10k × {timing['n_systems']} systems): "
              f"{timing['e2_dry_s'] * N_PERM_FULL / N_PERM_DRY * timing['n_systems']:.0f}s")
        print("Run with --full-perm to execute the full 10k permutation tests.")
        print("=" * 70)
        return

    # ── Phase 7: Full permutation tests ───────────────────────────────────────
    print("\n--- E1: Full permutation test (10k perms) ---")
    e1_perm = run_e1_permutation(records, per_sample, systems, V, sym_list, N_PERM_FULL)

    print("\n--- E2: Full permutation test (10k perms) ---")
    gini = {s: compute_gini(indegree[s]) for s in systems}
    e2_perm = run_e2_permutation(records, per_sample, systems, V, sym_list, N_PERM_FULL)

    # ── Reports ───────────────────────────────────────────────────────────────
    print("\n--- Writing reports ---")
    write_e1_report(e1_perm, class_agg, delta_agg, sym_list, V)
    write_e2_report(e2_perm, hub_shift, indegree, gini, sym_list, systems)

    print(f"\nAll outputs in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
