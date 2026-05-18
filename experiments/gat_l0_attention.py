#!/usr/bin/env python3
"""
gat_l0_attention.py
====================
GAT layer 0 attention analysis: characterize how attention patterns differ
across bona fide and spoofed inputs.

Design vs prior scripts (gat_attention.py, gat_attention_by_system.py):
  - SpecAugment fully off: frozen frontend called without _mask_hidden_states.
  - Per-head disaggregated: (E, NH) attention kept throughout; never mean-reduced early.
  - Edge-type tracking: adjacent (tgt==src+1) vs phoneme-lookahead classified post-hoc,
    surviving torch.unique dedup in generate_edges_by_combine_and_split.
  - GAT layer 0 only.
  - Raw phoneme drilldown for the top-KL attack family.

Outputs → experiments/results/gat_l0_attention/
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
from scipy.stats import mannwhitneyu


def _holm_sidak(pvals: np.ndarray) -> np.ndarray:
    """Holm-Sidak correction; returns corrected p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    corrected = np.ones(n)
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        c = 1.0 - (1.0 - p) ** (n - i)
        c = max(c, running_max)
        running_max = c
        corrected[order[i]] = c
    return np.clip(corrected, 0.0, 1.0)

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
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "gat_l0_attention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT          = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
VOCAB_DIR     = REPO_ROOT / "vocab_phoneme"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # 149
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42
MIN_NODES      = 3    # samples with fewer phoneme nodes are degenerate → filtered

LANG_ORDER  = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL     = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]
C = len(CLASS_ORDER)


# ---------------------------------------------------------------------------
# Vocab / phoneme-class helpers  (same mapping as gat_attention.py)
# ---------------------------------------------------------------------------

def build_vocab() -> tuple[dict[int, str], dict[int, str]]:
    """Returns (id_to_symbol, id_to_class)."""
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
# Balanced dataset: N_PER_CLASS per system_id
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
# Phoneme-loader patch
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
    """
    Run frozen WavLM pipeline without _mask_hidden_states.
    Returns (clean_hidden_states, phoneme_ids) — both deterministic across calls.
    """
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
# Phoneme-ID capture (monkey-patch on encoder_and_GAT)
# ---------------------------------------------------------------------------

class PhonemeCapture:
    """
    Monkey-patches encoder_and_GAT to capture per-node phoneme IDs and sample indices.
    Attributes updated after each call:
      .node_phoneme_ids : (total_nodes,)
      .node_sample_idx  : (total_nodes,)
      .reduced_num_frames: (B,)
    """
    def __init__(self, gat_model):
        self.node_phoneme_ids  = None
        self.node_sample_idx   = None
        self.reduced_num_frames = None
        cap = self
        orig = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_inner, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            rids = result[2].detach().cpu()   # (B, Lmax) padded phoneme IDs
            rnf  = result[3].detach().cpu()   # (B,) phoneme counts
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


# ---------------------------------------------------------------------------
# Edge-type classification
# ---------------------------------------------------------------------------

def classify_edge_types(edge_index: torch.Tensor,
                         node_sample_idx: torch.Tensor) -> torch.Tensor:
    """
    Returns bool tensor (E,): True = adjacent edge (src→src+1 within same sample).

    Adjacent edges are generated as i→i+1 by get_adj_edges before torch.unique.
    In the flat global node space, after sample-offset remapping, this property is
    preserved: tgt == src + 1 AND same_sample(src, tgt).

    cross-sample adjacent pairs (last node of sample i → first of sample i+1) are
    impossible because generate_edges_by_combine_and_split uses -1 padding tokens
    between samples and masks those edges out before returning.
    """
    src = edge_index[0]
    tgt = edge_index[1]
    same_sample = node_sample_idx[src] == node_sample_idx[tgt]
    is_adjacent = same_sample & (tgt == src + 1)
    return is_adjacent


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_attn_sum(attn: torch.Tensor, edge_index: torch.Tensor, tol: float = 1e-3) -> None:
    """
    For each target node and each head, incoming attention weights must sum to 1.0.
    (neighborhood-aware softmax invariant, valid in eval mode with dropout p=0).
    """
    NH = attn.shape[1]
    tgt_nodes = edge_index[1]
    num_nodes = int(tgt_nodes.max().item()) + 1

    for h in range(NH):
        node_sum = torch.zeros(num_nodes)
        node_sum.scatter_add_(0, tgt_nodes, attn[:, h])
        # Only nodes that are actually targets
        active = node_sum > 1e-7
        if active.any():
            max_dev = (node_sum[active] - 1.0).abs().max().item()
            if max_dev > tol:
                raise AssertionError(
                    f"ATTN SUM CHECK FAILED head {h}: "
                    f"max deviation from 1.0 = {max_dev:.6f} (tol={tol})\n"
                    "Wrong tensor captured — check log_attention_weights flag."
                )
    print(f"  Attn-sum check PASSED ({NH} heads, tol={tol})")


def check_self_consistency(gat_model, hidden_states, phoneme_ids, num_f,
                            buf: dict, tol: float = 1e-6) -> None:
    """Run encoder_and_GAT twice on identical inputs; attention must be bit-identical."""
    with torch.no_grad():
        gat_model.encoder_and_GAT(hidden_states, num_f, phoneme_ids)
    attn1 = buf["attn"].clone() if buf["attn"] is not None else None

    with torch.no_grad():
        gat_model.encoder_and_GAT(hidden_states, num_f, phoneme_ids)
    attn2 = buf["attn"]

    if attn1 is None or attn2 is None:
        raise AssertionError("Self-consistency: attn buffer empty after forward pass")
    max_diff = (attn1 - attn2).abs().max().item()
    if max_diff > tol:
        raise AssertionError(
            f"SELF-CONSISTENCY FAILED: max attention diff = {max_diff:.2e} "
            "(encoder_and_GAT is not deterministic on fixed input)"
        )
    print(f"  Self-consistency check PASSED: max_diff = {max_diff:.2e}")


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_attention(lit, loader, device) -> list[dict]:
    """
    Run model over all batches with SpecAugment fully off.
    Returns list of per-sample dicts:
      sample_id, label, system_id,
      node_phoneme_ids (n,),
      edge_index (2, E) — sample-local 0-indexed,
      edge_type_adj (E,) bool,
      attn_l0 (E, NH),
      n_nodes, n_edges, is_degenerate
    """
    gat_model = lit.model
    gat_model.GAT.gat_net[0].log_attention_weights = True

    # Buffer for hook
    buf: dict = {"attn": None, "edge_index": None}

    def _hook_l0(module, inp, out):
        buf["edge_index"] = out[1].detach().cpu()
        if module.attention_weights is not None:
            buf["attn"] = module.attention_weights.squeeze(-1).detach().cpu()  # (E, NH)

    hook = gat_model.GAT.gat_net[0].register_forward_hook(_hook_l0)
    phoneme_cap = PhonemeCapture(gat_model)

    sanity_done = False
    records: list[dict] = []
    n_degenerate = 0
    total = len(loader.dataset)

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio     = batch["audio"].to(device)
                labels    = batch["label"].tolist()
                sys_ids   = batch["system_id"]
                B         = len(labels)
                num_f     = torch.full((B,), NF_PER_SAMPLE, device=device)

                # SpecAugment fully off — clean hidden_states, no masking
                hidden_states, phoneme_ids = run_frozen_frontend(audio, gat_model, device)

                # Self-consistency + attn-sum checks on first batch
                if not sanity_done:
                    check_self_consistency(gat_model, hidden_states, phoneme_ids, num_f, buf)

                with torch.no_grad():
                    gat_model.encoder_and_GAT(hidden_states, num_f, phoneme_ids)

                # Retrieve captured tensors
                edge_index_global = buf["edge_index"]   # (2, E_total)
                attn_global       = buf["attn"]          # (E_total, NH)

                if edge_index_global is None or attn_global is None:
                    print(f"  [WARN] batch {bi}: hook did not fire, skipping")
                    continue

                NH = attn_global.shape[1]

                # First-batch sanity: print shape + check attn sums
                if not sanity_done:
                    print(f"\n  [Sanity] GAT l0 attention shape: {attn_global.shape} "
                          f"(E={attn_global.shape[0]}, NH={NH})")
                    check_attn_sum(attn_global, edge_index_global)
                    sanity_done = True

                node_pids     = phoneme_cap.node_phoneme_ids   # (total_nodes,)
                node_samp_idx = phoneme_cap.node_sample_idx    # (total_nodes,)
                rnf           = phoneme_cap.reduced_num_frames  # (B,)

                # Compute per-sample node offsets
                node_offsets = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                base_id = len(records)

                for i in range(B):
                    n_nodes_i = int(rnf[i].item())
                    offset_i  = node_offsets[i]

                    # Edges belonging to sample i (by source node)
                    src_global = edge_index_global[0]
                    edge_mask  = (node_samp_idx[src_global] == i)

                    ei_global = edge_index_global[:, edge_mask]      # (2, E_i)
                    attn_i    = attn_global[edge_mask]                # (E_i, NH)
                    ei_local  = ei_global - offset_i                  # sample-local 0-indexed

                    n_edges_i = ei_local.shape[1]

                    # Edge type: adjacent = tgt_local == src_local + 1
                    is_adj = (ei_local[1] - ei_local[0]) == 1        # (E_i,) bool

                    pids_i = node_pids[node_samp_idx == i]            # (n_nodes_i,)

                    is_deg = n_nodes_i < MIN_NODES
                    if is_deg:
                        n_degenerate += 1

                    records.append({
                        "sample_id":       base_id + i,
                        "label":           labels[i],
                        "system_id":       sys_ids[i],
                        "node_phoneme_ids": pids_i.clone(),
                        "edge_index":      ei_local.clone(),
                        "edge_type_adj":   is_adj.clone(),
                        "attn_l0":         attn_i.clone(),
                        "n_nodes":         n_nodes_i,
                        "n_edges":         n_edges_i,
                        "is_degenerate":   is_deg,
                    })

                buf["attn"] = None
                buf["edge_index"] = None

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    print(f"  {min((bi + 1) * BATCH_SIZE, total)}/{total}")

    finally:
        hook.remove()

    valid   = [r for r in records if not r["is_degenerate"]]
    print(f"\n  Extracted {len(records)} samples  "
          f"({n_degenerate} degenerate [<{MIN_NODES} nodes] filtered out)")
    return records, n_degenerate


# ---------------------------------------------------------------------------
# 9-class aggregation  →  accum[system_id]: (C, C, NH),  count[system_id]: (C, C)
# ---------------------------------------------------------------------------

def aggregate_9class(records: list[dict], id_to_cls: dict[int, str],
                     systems: list[str]) -> tuple[dict, dict]:
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    NH = records[0]["attn_l0"].shape[1] if records else 6

    accum  = {s: np.zeros((C, C, NH), dtype=np.float64) for s in systems}
    counts = {s: np.zeros((C, C),    dtype=np.int64)   for s in systems}

    for rec in records:
        if rec["is_degenerate"]:
            continue
        sid  = rec["system_id"]
        pids = rec["node_phoneme_ids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"]          # (E, NH)

        src_nodes = ei[0]
        tgt_nodes = ei[1]

        for e in range(ei.shape[1]):
            sc = cls_idx[id_to_cls.get(int(pids[src_nodes[e]].item()), "Other")]
            tc = cls_idx[id_to_cls.get(int(pids[tgt_nodes[e]].item()), "Other")]
            accum[sid][sc, tc, :]  += attn[e].numpy()   # all heads
            counts[sid][sc, tc]    += 1

    return accum, counts


# ---------------------------------------------------------------------------
# Attention entropy per node, grouped by phoneme class
# ---------------------------------------------------------------------------

def compute_entropy(records: list[dict], id_to_cls: dict[int, str]) -> dict[str, dict[str, list[float]]]:
    """
    Returns {system_id: {phoneme_class: [per-sample mean entropy of nodes in that class]}}
    Entropy is computed per target node over its incoming attention distribution,
    averaged over heads, then averaged over nodes in the same phoneme class per sample.
    """
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    result: dict[str, dict[str, list[float]]] = {
        sid: {c: [] for c in CLASS_ORDER} for sid in set(r["system_id"] for r in records)
    }

    for rec in records:
        if rec["is_degenerate"]:
            continue
        sid  = rec["system_id"]
        pids = rec["node_phoneme_ids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"].float()   # (E, NH)
        NH   = attn.shape[1]
        n    = rec["n_nodes"]

        # Per target-node, per-head entropy over incoming edges
        tgt_nodes = ei[1]
        node_entropy = torch.zeros(n, NH)   # will accumulate

        # For each target node: gather incoming attn, compute entropy
        # Use scatter approach
        for t in range(n):
            mask = (tgt_nodes == t)
            if not mask.any():
                continue
            a_t = attn[mask]   # (k, NH)  incoming edges to node t
            # Entropy: -sum(a * log(a))
            log_a = torch.log(a_t + 1e-10)
            ent   = -(a_t * log_a).sum(0)   # (NH,)
            node_entropy[t] = ent

        node_entropy_mean = node_entropy.mean(1)   # (n,) mean over heads

        # Group by phoneme class, compute per-class mean entropy for this sample
        class_entropies: dict[str, list[float]] = defaultdict(list)
        for node_i in range(n):
            cls = id_to_cls.get(int(pids[node_i].item()), "Other")
            class_entropies[cls].append(float(node_entropy_mean[node_i].item()))

        for cls, vals in class_entropies.items():
            result[sid][cls].append(float(np.mean(vals)))

    return result


# ---------------------------------------------------------------------------
# KL divergence  (Laplace-smoothed, per head, then mean across heads)
# ---------------------------------------------------------------------------

def smooth_dist(accum_h: np.ndarray) -> np.ndarray:
    """Laplace-smooth (C,C) attention accumulator → probability distribution."""
    p = accum_h + 1.0
    return p / p.sum()


def kl_div(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P||Q). Both must be valid distributions (no zeros with Laplace smoothing)."""
    return float(np.sum(P * np.log(P / Q)))


def compute_kl_per_system(accum: dict, systems: list[str],
                           bonafide_key: str = "-") -> dict[str, dict]:
    """
    Returns {system_id: {head_kl: [kl_h for h in heads], mean_kl: float}}
    for all non-bonafide systems.
    """
    bon = accum[bonafide_key]   # (C, C, NH)
    NH  = bon.shape[2]
    kl_results = {}
    for sid in systems:
        if sid == bonafide_key:
            continue
        att = accum[sid]
        head_kls = []
        for h in range(NH):
            P = smooth_dist(att[:, :, h])
            Q = smooth_dist(bon[:, :, h])
            head_kls.append(kl_div(P, Q))
        kl_results[sid] = {"head_kl": head_kls, "mean_kl": float(np.mean(head_kls))}
    return kl_results


# ---------------------------------------------------------------------------
# Mann-Whitney U per phoneme class
# ---------------------------------------------------------------------------

def mann_whitney_per_class(entropy_by_sys: dict[str, dict[str, list[float]]],
                            bonafide_key: str = "-") -> list[dict]:
    """
    For each phoneme class and each attack system: compare entropy distributions
    (bonafide vs attack) using Mann-Whitney U, then apply Holm-Sidak correction
    across all (class, system) pairs.
    Returns list of dicts sorted by corrected p-value.
    """
    bon_ent = entropy_by_sys.get(bonafide_key, {})
    attack_systems = [s for s in entropy_by_sys if s != bonafide_key]
    rows = []

    for sys_id in attack_systems:
        for cls in CLASS_ORDER:
            bon_vals = bon_ent.get(cls, [])
            att_vals = entropy_by_sys[sys_id].get(cls, [])
            if len(bon_vals) < 3 or len(att_vals) < 3:
                continue
            stat, p = mannwhitneyu(att_vals, bon_vals, alternative="two-sided")
            delta_mean = float(np.mean(att_vals)) - float(np.mean(bon_vals))
            rows.append({
                "system_id":    sys_id,
                "phoneme_class": cls,
                "U_stat":       float(stat),
                "p_raw":        float(p),
                "delta_mean_entropy": delta_mean,
                "n_attack":     len(att_vals),
                "n_bonafide":   len(bon_vals),
            })

    if rows:
        p_corr = _holm_sidak(np.array([r["p_raw"] for r in rows]))
        for r, pc in zip(rows, p_corr):
            r["p_corrected"] = float(pc)
            r["significant"] = pc < 0.05
    else:
        for r in rows:
            r["p_corrected"] = 1.0
            r["significant"] = False

    rows.sort(key=lambda x: x["p_raw"])
    return rows


# ---------------------------------------------------------------------------
# Raw phoneme drilldown for top-KL attack system
# ---------------------------------------------------------------------------

def raw_phoneme_drilldown(records: list[dict], top_kl_sid: str,
                           id_to_sym: dict[int, str], id_to_cls: dict[int, str],
                           top_k: int = 30) -> list[dict]:
    """
    For top-KL system vs bonafide: sparse (src_pid, tgt_pid) attention accumulation.
    Returns top_k pairs ranked by absolute mean attention delta.
    """
    accum_bon = defaultdict(lambda: [0.0, 0])   # key → [sum, count]
    accum_att = defaultdict(lambda: [0.0, 0])

    for rec in records:
        if rec["is_degenerate"]:
            continue
        sid  = rec["system_id"]
        if sid not in (top_kl_sid, "-"):
            continue
        pids = rec["node_phoneme_ids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"].float().mean(1)   # (E,) mean over heads

        acc = accum_att if sid == top_kl_sid else accum_bon

        for e in range(ei.shape[1]):
            src_pid = int(pids[ei[0, e]].item())
            tgt_pid = int(pids[ei[1, e]].item())
            key = (src_pid, tgt_pid)
            acc[key][0] += float(attn[e].item())
            acc[key][1] += 1

    all_keys = set(accum_bon) | set(accum_att)
    rows = []
    for key in all_keys:
        src_pid, tgt_pid = key
        bon_mean = accum_bon[key][0] / accum_bon[key][1] if accum_bon[key][1] > 0 else 0.0
        att_mean = accum_att[key][0] / accum_att[key][1] if accum_att[key][1] > 0 else 0.0
        rows.append({
            "src_pid":      src_pid,
            "tgt_pid":      tgt_pid,
            "src_symbol":   id_to_sym.get(src_pid, "?"),
            "tgt_symbol":   id_to_sym.get(tgt_pid, "?"),
            "src_class":    id_to_cls.get(src_pid, "Other"),
            "tgt_class":    id_to_cls.get(tgt_pid, "Other"),
            "bon_mean_attn": bon_mean,
            "att_mean_attn": att_mean,
            "delta":         att_mean - bon_mean,
            "n_bon_edges":   accum_bon[key][1],
            "n_att_edges":   accum_att[key][1],
        })

    rows.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _row_norm(mat: np.ndarray) -> np.ndarray:
    rs = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)


def _draw_attention_wheel(ax, attn_mat_rn: np.ndarray, title: str,
                           threshold: float = 0.02) -> None:
    """
    Draw phoneme-class graph: 9 nodes in a circle, directed edges weighted by attention.
    attn_mat_rn: (C, C) row-normalised.
    """
    n = C
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    px = np.cos(angles)
    py = np.sin(angles)
    short = [c[:5] for c in CLASS_ORDER]

    cmap = _get_mpl().cm.get_cmap("tab10")
    node_colors = [cmap(i / n) for i in range(n)]

    # Draw edges
    for src in range(n):
        for tgt in range(n):
            w = attn_mat_rn[src, tgt]
            if w < threshold:
                continue
            dx = px[tgt] - px[src]
            dy = py[tgt] - py[src]
            ax.annotate("",
                xy=(px[tgt], py[tgt]),
                xytext=(px[src], py[src]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=max(0.3, w * 8),
                    color=node_colors[src],
                    alpha=min(1.0, w * 5 + 0.2),
                    connectionstyle="arc3,rad=0.15",
                ),
                annotation_clip=False,
            )

    # Draw nodes
    for i in range(n):
        ax.scatter(px[i], py[i], s=220, color=node_colors[i],
                   zorder=5, edgecolors="black", linewidths=0.5)
        offset = 0.18
        ax.text(px[i] * (1 + offset), py[i] * (1 + offset),
                short[i], ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def plot_grid_pdf(accum: dict, counts: dict, systems: list[str],
                  out_path: Path) -> None:
    """
    Grid PDF: rows = systems, cols = [attention wheel, 9×9 heatmap, Δ vs bonafide].
    """
    plt = _get_mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    spoof_systems = [s for s in systems if s != "-"]
    all_rows = ["-"] + spoof_systems   # bonafide first
    n_rows = len(all_rows)

    # Pre-compute row-normalised mean matrices
    rn: dict[str, np.ndarray] = {}
    for sid in all_rows:
        # Mean over heads then row-normalise
        mean_h = np.divide(accum[sid].sum(2), counts[sid],
                           out=np.zeros((C, C)), where=counts[sid] > 0)
        rn[sid] = _row_norm(mean_h)

    rn_bon  = rn["-"]
    vmax_abs = max(rn[sid].max() for sid in all_rows)
    dlim = max((rn[sid] - rn_bon).max() - (rn[sid] - rn_bon).min()
               for sid in spoof_systems) / 2 if spoof_systems else 0.1

    SHORT = [c[:5] for c in CLASS_ORDER]

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(n_rows, 3,
                                  figsize=(15, 4.5 * n_rows),
                                  gridspec_kw={"wspace": 0.35, "hspace": 0.4})
        if n_rows == 1:
            axes = axes[None, :]

        for row_i, sid in enumerate(all_rows):
            label_str = "Bonafide (-)" if sid == "-" else f"Spoof {sid}"
            mat  = rn[sid]
            delta = mat - rn_bon

            # Col 0: attention wheel
            _draw_attention_wheel(axes[row_i, 0], mat,
                                  title=f"{label_str}\nAttention graph")

            # Col 1: 9×9 heatmap
            ax = axes[row_i, 1]
            im = ax.imshow(mat, vmin=0, vmax=vmax_abs, cmap="Blues", aspect="auto")
            ax.set_xticks(range(C)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(C)); ax.set_yticklabels(SHORT, fontsize=7)
            ax.set_title(f"{label_str}\n9×9 heatmap (row-norm)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Col 2: delta vs bonafide
            ax = axes[row_i, 2]
            dlim_local = max(abs(delta).max(), 1e-6)
            im2 = ax.imshow(delta, vmin=-dlim_local, vmax=dlim_local,
                            cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(C)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(C)); ax.set_yticklabels(SHORT, fontsize=7)
            ax.set_title(f"Δ {label_str} − Bonafide", fontsize=9)
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            for r in range(C):
                for c in range(C):
                    color = "white" if abs(delta[r, c]) > dlim_local * 0.6 else "black"
                    ax.text(c, r, f"{delta[r,c]:+.3f}",
                            ha="center", va="center", fontsize=5, color=color)

        fig.suptitle(
            "GAT Layer 0 — phoneme-class attention by system\n"
            "Row-normalised  ·  SpecAugment off  ·  Per-head mean shown",
            fontsize=12, y=1.005)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_path}")


def plot_kl_bar(kl_results: dict[str, dict], out_path: Path) -> None:
    plt = _get_mpl()
    systems = sorted(kl_results, key=lambda s: -kl_results[s]["mean_kl"])
    mean_kls = [kl_results[s]["mean_kl"] for s in systems]
    # Per-head bars with mean overlay
    NH = len(kl_results[systems[0]]["head_kl"]) if systems else 6
    x  = np.arange(len(systems))

    fig, ax = plt.subplots(figsize=(max(6, len(systems) * 1.4), 5))
    width   = 0.8 / NH
    cmap    = plt.cm.get_cmap("Set2")
    for h in range(NH):
        vals = [kl_results[s]["head_kl"][h] for s in systems]
        ax.bar(x + h * width - (NH - 1) * width / 2, vals,
               width=width * 0.9, color=cmap(h / NH),
               alpha=0.75, label=f"head {h}")
    ax.plot(x, mean_kls, "ko-", markersize=6, linewidth=1.5, label="mean KL", zorder=10)
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("KL divergence  KL(attack ‖ bonafide)"); ax.set_xlabel("Attack system")
    ax.set_title("KL divergence from bonafide — GAT layer 0 attention\nper head + mean")
    ax.legend(fontsize=8, ncol=NH + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_kl_delta(accum: dict, top_kl_sid: str, out_path: Path) -> None:
    """Difference heatmap (top-KL attack − bonafide), row-normalised."""
    plt = _get_mpl()
    mean_h_top = np.divide(accum[top_kl_sid].sum(2),
                           accum[top_kl_sid].sum(2).sum() + 1e-10)
    # Recompute as proper row-norm matrices
    def rn_mean(sid):
        cnt = accum[sid].sum(2)   # (C,C) sum over heads
        mean_ = cnt / (cnt.sum() + 1e-10)   # global-norm for delta
        return mean_

    att_rn = _row_norm(np.divide(accum[top_kl_sid].sum(2),
                                  accum[top_kl_sid].sum((0,1,2)) / (C * C) + 1e-10))
    bon_rn = _row_norm(np.divide(accum["-"].sum(2),
                                  accum["-"].sum((0,1,2)) / (C * C) + 1e-10))
    # Simpler: just row-norm the head-mean attention
    def head_mean_rn(sid):
        cnt  = accum[sid].sum(2)     # sum over NH: (C,C)
        n_h  = accum[sid].shape[2]
        mean_mat = cnt / n_h         # not normalised yet
        # Use counts from 9-class aggregation — but we don't have them here
        # Use the accumulated sum directly (proportional to mean × count)
        return _row_norm(cnt)

    att_mat = head_mean_rn(top_kl_sid)
    bon_mat = head_mean_rn("-")
    delta   = att_mat - bon_mat
    dlim    = max(abs(delta).max(), 1e-6)
    SHORT   = [c[:5] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(delta, vmin=-dlim, vmax=dlim, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(C)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(C)); ax.set_yticklabels(SHORT, fontsize=9)
    ax.set_xlabel("Target phoneme class (receiver)", fontsize=10)
    ax.set_ylabel("Source phoneme class (sender)", fontsize=10)
    ax.set_title(
        f"Δ attention: {top_kl_sid} − Bonafide  (highest KL system)\n"
        "Row-normalised head-mean  ·  Red = more in spoof, Blue = more in bonafide",
        fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(C):
        for c in range(C):
            color = "white" if abs(delta[r, c]) > dlim * 0.55 else "black"
            ax.text(c, r, f"{delta[r,c]:+.3f}",
                    ha="center", va="center", fontsize=7.5, color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_raw_phoneme_drilldown(drilldown: list[dict], top_kl_sid: str,
                                out_path: Path) -> None:
    plt = _get_mpl()
    labels = [f"{r['src_symbol'][:6]}→{r['tgt_symbol'][:6]}" for r in drilldown]
    deltas = [r["delta"] for r in drilldown]
    colors = ["#E03030" if d > 0 else "#6EB5FF" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, max(5, len(drilldown) * 0.38)))
    y = range(len(drilldown))
    ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="grey", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ mean attention  ({top_kl_sid} − bonafide, head-mean)", fontsize=10)
    ax.set_title(
        f"Top-{len(drilldown)} raw phoneme pair shifts — {top_kl_sid} vs bonafide\n"
        "Red = more attended in spoof  ·  Blue = more attended in bonafide",
        fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(records: list[dict], path: Path) -> None:
    data = {
        "sample_ids":        [r["sample_id"]       for r in records],
        "labels":            [r["label"]            for r in records],
        "system_ids":        [r["system_id"]        for r in records],
        "n_nodes":           [r["n_nodes"]          for r in records],
        "n_edges":           [r["n_edges"]          for r in records],
        "is_degenerate":     [r["is_degenerate"]    for r in records],
        "node_phoneme_ids":  [r["node_phoneme_ids"] for r in records],
        "edge_index":        [r["edge_index"]       for r in records],
        "edge_type_adj":     [r["edge_type_adj"]    for r in records],
        "attn_l0":           [r["attn_l0"]          for r in records],
    }
    torch.save(data, path)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(systems, kl_results, top_kl_sid, mw_rows,
                 n_degenerate, n_total, n_valid,
                 out_path: Path) -> None:
    kl_table  = "| system | mean KL | " + " | ".join(f"head {h}" for h in range(
        len(next(iter(kl_results.values()))["head_kl"]))) + " |\n"
    kl_table += "|--------|---------|" + "---------|" * len(next(iter(kl_results.values()))["head_kl"]) + "\n"
    for sid in sorted(kl_results, key=lambda s: -kl_results[s]["mean_kl"]):
        heads = " | ".join(f"{v:.4f}" for v in kl_results[sid]["head_kl"])
        kl_table += f"| {sid} | {kl_results[sid]['mean_kl']:.4f} | {heads} |\n"

    sig_rows = [r for r in mw_rows if r["significant"]]
    mw_str = ""
    if sig_rows:
        mw_str = "| system | class | Δ entropy | U | p_raw | p_holm |\n"
        mw_str += "|--------|-------|-----------|---|-------|--------|\n"
        for r in sig_rows[:20]:
            mw_str += (f"| {r['system_id']} | {r['phoneme_class']} | "
                       f"{r['delta_mean_entropy']:+.4f} | {r['U_stat']:.0f} | "
                       f"{r['p_raw']:.4f} | {r['p_corrected']:.4f} |\n")
    else:
        mw_str = "_No significant differences after Holm-Sidak correction._"

    surprising = ""
    if top_kl_sid:
        surprising = (
            f"- **Top-KL system is {top_kl_sid}** (KL = "
            f"{kl_results[top_kl_sid]['mean_kl']:.4f}). "
            "Raw phoneme drilldown reveals which specific phoneme transitions shift most.\n"
        )
    if not sig_rows:
        surprising += "- Mann-Whitney found no significant per-class entropy differences after correction — "
        surprising += "attention structure may differ in distribution shape rather than class-level mean entropy.\n"

    report = textwrap.dedent(f"""
    # GAT Layer 0 Attention Analysis

    **Model**: robust_goat.ckpt  |  **Layer**: `lit.model.GAT.gat_net[0]`  |  **Heads**: per-head disaggregated
    **SpecAugment**: fully off (frozen frontend, no `_mask_hidden_states`)
    **Samples**: {n_total} total, {n_valid} valid ({n_degenerate} degenerate filtered, < {MIN_NODES} nodes)

    ---

    ## KL divergence  KL(attack ‖ bonafide)

    {kl_table}

    **Top-KL attack system**: {top_kl_sid}

    ---

    ## Mann-Whitney U — per-phoneme-class entropy shifts

    (Holm-Sidak corrected, α=0.05)

    {mw_str}

    ---

    ## What surprised me

    {surprising}

    ---

    ## Outputs

    | file | description |
    |------|-------------|
    | `attention_artifacts.pt` | Per-sample edge_index, attn (E,NH), pids, edge_type |
    | `attention_grid.pdf` | Rows=systems, cols=[graph wheel, 9×9 heatmap, Δ heatmap] |
    | `kl_divergence.png` | KL per system per head + mean |
    | `top_kl_delta_heatmap.png` | Δ attention heatmap for {top_kl_sid} |
    | `top_kl_raw_phonemes.csv` | Raw phoneme pair drilldown for {top_kl_sid} |
    | `per_class_stats.csv` | Mann-Whitney results per class |

    *Generated by experiments/gat_l0_attention.py*
    """).strip()

    out_path.write_text(report)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *a, **kw: None
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    id_to_sym, id_to_cls = build_vocab()
    print(f"Vocab: {len(id_to_sym)} tokens → {C} phoneme classes")

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = BalancedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                                hf_token, N_PER_CLASS, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate)

    patch_phoneme_loader()
    print(f"\nLoading: {CKPT}")
    lit = load_model(CKPT, device)

    # ── Extraction ─────────────────────────────────────────────────────────────
    print("\n--- Extracting GAT layer 0 attention ---")
    records, n_degenerate = extract_attention(lit, loader, device)

    save_artifacts(records, RESULTS_DIR / "attention_artifacts.pt")

    systems = sorted(set(r["system_id"] for r in records))
    valid   = [r for r in records if not r["is_degenerate"]]
    print(f"\nSystems: {systems}")

    # ── 9-class aggregation ────────────────────────────────────────────────────
    print("\n--- Aggregating attention by phoneme class ---")
    accum, counts = aggregate_9class(records, id_to_cls, systems)

    # ── KL divergence ──────────────────────────────────────────────────────────
    print("\n--- KL divergence ---")
    kl_results = compute_kl_per_system(accum, systems)
    top_kl_sid = max(kl_results, key=lambda s: kl_results[s]["mean_kl"]) if kl_results else None
    for sid in sorted(kl_results, key=lambda s: -kl_results[s]["mean_kl"]):
        head_str = "  ".join(f"h{h}={v:.4f}" for h, v in enumerate(kl_results[sid]["head_kl"]))
        print(f"  {sid}: mean_KL={kl_results[sid]['mean_kl']:.4f}  [{head_str}]")
    if top_kl_sid:
        print(f"  → Top-KL: {top_kl_sid}")

    # ── Entropy + Mann-Whitney ─────────────────────────────────────────────────
    print("\n--- Computing attention entropy per phoneme class ---")
    entropy_by_sys = compute_entropy(records, id_to_cls)
    mw_rows = mann_whitney_per_class(entropy_by_sys)
    sig = [r for r in mw_rows if r["significant"]]
    print(f"  {len(sig)} significant class-level entropy shifts (Holm p<0.05)")
    for r in sig[:5]:
        print(f"    {r['system_id']} / {r['phoneme_class']:14s}  "
              f"Δentropy={r['delta_mean_entropy']:+.4f}  p_holm={r['p_corrected']:.4f}")

    with open(RESULTS_DIR / "per_class_stats.csv", "w", newline="") as f:
        if mw_rows:
            w = csv.DictWriter(f, fieldnames=list(mw_rows[0].keys()))
            w.writeheader(); w.writerows(mw_rows)
    print(f"Saved: {RESULTS_DIR / 'per_class_stats.csv'}")

    # ── Raw phoneme drilldown for top-KL ──────────────────────────────────────
    if top_kl_sid:
        print(f"\n--- Raw phoneme drilldown for {top_kl_sid} ---")
        drilldown = raw_phoneme_drilldown(records, top_kl_sid, id_to_sym, id_to_cls)
        print(f"  Top-5 raw phoneme pair shifts:")
        for r in drilldown[:5]:
            print(f"    {r['src_symbol']:8s} → {r['tgt_symbol']:8s}  "
                  f"Δ={r['delta']:+.4f}  (bon={r['bon_mean_attn']:.4f}, "
                  f"att={r['att_mean_attn']:.4f})")
        csv_path = RESULTS_DIR / "top_kl_raw_phonemes.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(drilldown[0].keys()))
            w.writeheader(); w.writerows(drilldown)
        print(f"Saved: {csv_path}")
        plot_raw_phoneme_drilldown(drilldown, top_kl_sid,
                                   RESULTS_DIR / "top_kl_raw_phonemes.png")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_grid_pdf(accum, counts, systems, RESULTS_DIR / "attention_grid.pdf")
    if kl_results:
        plot_kl_bar(kl_results, RESULTS_DIR / "kl_divergence.png")
    if top_kl_sid:
        plot_top_kl_delta(accum, top_kl_sid, RESULTS_DIR / "top_kl_delta_heatmap.png")

    # ── Report ─────────────────────────────────────────────────────────────────
    write_report(systems, kl_results, top_kl_sid, mw_rows,
                 n_degenerate, len(records), len(valid),
                 RESULTS_DIR / "report.md")

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
