#!/usr/bin/env python3
"""
gat_e5.py
=========
E5: Replication study — pre-registered predictions P1–P4 on held-out checkpoint.

Pre-registered predictions (commit before training):
  P1 [primary]:   Per-system family hub-mass p-values rank similarly to original.
                  Spearman ρ ≥ 0.7 across 6 systems.
                  Original rank: A01 < A03 < A04 < A02 < A05 < A06.
                  Falsification: ρ ≤ 0.3.

  P2 [primary]:   h0/h4 are causally sufficient: ablating them produces
                  Δ-EER within 0.02 of ablating all 6 heads on A01/A03/A04.
                  Falsification: max |Δ(h0h4) - Δ(all6)| > 0.02 on any of the three.

  P3 [secondary]: A05/A06 Δ-EER under all-attention ablation is ≤ 0.
                  Falsification: A05 OR A06 shows Δ-EER ≥ +0.04 with CI excluding 0.

  P4 [secondary]: Gini direction agrees with original for ≥ 5/6 systems.
                  (ΔGini = Gini_attack - Gini_bonafide, same sign as in E2)
                  Falsification: ≤ 3/6 agreement.

Usage:
  venv/bin/python3 experiments/gat_e5.py --ckpt models/robust_goat_seed7.ckpt

Outputs → experiments/results/gat_e5/
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
REPO_ROOT   = Path(__file__).resolve().parents[1]
EXP_DIR     = Path(__file__).resolve().parent
OUT_BASE    = EXP_DIR / "results" / "gat_e5"
VOCAB_DIR   = REPO_ROOT / "vocab_phoneme"

for p in [OUT_BASE]:
    p.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Original E2 results (for P1 rank comparison) ──────────────────────────────
# Family p-values from gat_attn_graphs.py run with --full-perm (seed=42 checkpoint)
ORIGINAL_P_FAMILY = {
    "A01": 0.00010,
    "A02": 0.45555,
    "A03": 0.00250,
    "A04": 0.00820,
    "A05": 0.30577,
    "A06": 0.51565,
}
# Original rank (ascending p_family = descending significance):
# A01 < A03 < A04 < A02 < A05 < A06
ORIGINAL_RANK_ORDER = ["A01", "A03", "A04", "A02", "A05", "A06"]

# Original Gini (from E2 gini_indegree.csv)
ORIGINAL_GINI = {
    "-":   0.4242,
    "A01": 0.2384,
    "A02": 0.3809,
    "A03": 0.2384,
    "A04": 0.2673,
    "A05": 0.5627,
    "A06": 0.5180,
}
# Original ΔGini = Gini_attack - Gini_bonafide (sign = direction of hub concentration shift)
ORIGINAL_DELTA_GINI_SIGN = {
    s: np.sign(g - ORIGINAL_GINI["-"])
    for s, g in ORIGINAL_GINI.items() if s != "-"
}

# Original E4 ablation EERs (for P2/P3 context)
ORIGINAL_EER = {
    "baseline":      {"A01": 0.040, "A02": 0.100, "A03": 0.060,
                      "A04": 0.060, "A05": 0.100, "A06": 0.260},
    "attn_ablated":  {"A01": 0.060, "A02": 0.080, "A03": 0.160,
                      "A04": 0.100, "A05": 0.060, "A06": 0.200},
    "critical_only": {"A01": 0.060, "A02": 0.120, "A03": 0.160,
                      "A04": 0.100, "A05": 0.080, "A06": 0.260},
}

# ── Constants ─────────────────────────────────────────────────────────────────
CRITICAL_HEADS  = [0, 4]
SEED            = 42
N_PERM          = 10_000
N_BOOTSTRAP     = 1_000
NF_PER_SAMPLE   = 3 * 16_000 // 320 - 1
N_PER_CLASS     = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 8))
HF_DATASET      = "Bisher/ASVspoof_2019_LA"
CACHE_DIR       = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH   = REPO_ROOT / "secret.txt"
LANG_ORDER      = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL         = ["|", "</s>", "<s>", "<unk>", "<pad>"]
TARGET_SR       = 16_000
TARGET_SAMPLES  = 3 * TARGET_SR
MIN_NODES       = 3


# ── Shared helpers (vocab, holm, eer, gini) ───────────────────────────────────

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
    def _sym_to_cls(sym):
        if sym in SPECIAL or sym.isdigit(): return "Other"
        for substr, cls in _CAT_RULES:
            if substr in sym: return cls
        return "Other"
    total = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists(): continue
        for sym, _ in sorted(json.load(open(p)).items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id_to_sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    id_to_cls = {i: _sym_to_cls(s) for i, s in id_to_sym.items()}
    return id_to_sym, id_to_cls


def holm_bonferroni(pvals):
    n = len(pvals); order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * np.arange(n, 0, -1, dtype=float))
    for i in range(1, n): adj[i] = max(adj[i], adj[i-1])
    result = np.empty(n); result[order] = adj
    return np.clip(result, 0.0, 1.0)


def compute_eer(labels, scores):
    thresholds = np.unique(scores)
    n_bon = (labels == 0).sum(); n_sp = (labels == 1).sum()
    best_eer, best_diff = 1.0, float("inf")
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        far = fp / max(n_bon, 1); frr = fn / max(n_sp, 1)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff; best_eer = (far + frr) / 2
    return best_eer


def compute_gini(values):
    v = values[values > 0]
    if len(v) == 0: return 0.0
    v = np.sort(v); n = len(v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) / (n * v.sum())) - (n + 1) / n)


# ── Audio helpers ─────────────────────────────────────────────────────────────

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


# ── Dataset ───────────────────────────────────────────────────────────────────

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


# ── Model loading ─────────────────────────────────────────────────────────────

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


# ── Phoneme capture ───────────────────────────────────────────────────────────

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


# ── All-layer extraction ──────────────────────────────────────────────────────

def run_frozen_frontend(audio, gat_model, device):
    x = audio
    if x.ndim == 3 and x.size(1) == 1: x = x[:, 0, :]
    with torch.no_grad():
        feat1  = gat_model.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hs, _  = gat_model.transformer_in_phoneme_model.feature_projection(feat1)
        pf     = gat_model.transformer_in_phoneme_model.encoder(hs)[0]
        pl     = gat_model.phoneme_model.model.model.lm_head(pf)
        pids   = torch.argmax(pl, dim=-1)
    return hs, pids


def extract_all_layers(lit, loader, device):
    gm = lit.model; n_layers = 3
    for i in range(n_layers):
        gm.GAT.gat_net[i].log_attention_weights = True
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

                node_offsets = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                base_id = len(records)
                for i in range(B):
                    n_nodes_i = int(rnf[i].item()); off = node_offsets[i]
                    src_global = ei_global[0]
                    edge_mask  = (node_samp_idx[src_global] == i)
                    ei_local   = ei_global[:, edge_mask] - off
                    pids_i     = node_pids[node_samp_idx == i]
                    is_deg     = n_nodes_i < MIN_NODES
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


# ── A_agg + hub-mass pipeline ─────────────────────────────────────────────────

def sparse_to_dense(attn_mean, edge_index, N):
    A = np.zeros((N, N), dtype=np.float64)
    np.add.at(A, (edge_index[1], edge_index[0]), attn_mean)
    row_sums = A.sum(axis=1); no_in = np.where(row_sums < 1e-10)[0]
    A[no_in, no_in] = 1.0
    return A


def build_agg_graph(rec, head_subset=None):
    N = rec["n_nodes"]; ei = rec["edge_index"].numpy(); layers = []
    for li in range(3):
        attn_np = rec[f"attn_l{li}"].float().numpy()
        if head_subset is not None: attn_np = attn_np[:, head_subset]
        attn_mean = attn_np.mean(axis=1)
        layers.append(sparse_to_dense(attn_mean, ei, N))
    A_agg = layers[2] @ layers[1] @ layers[0]
    rs    = A_agg.sum(axis=1, keepdims=True)
    return A_agg / np.where(rs > 1e-10, rs, 1.0)


def build_symbol_vocab(records, id_to_sym):
    seen = set()
    for rec in records:
        if rec["is_degenerate"]: continue
        for pid in rec["node_phoneme_ids"].tolist():
            seen.add(id_to_sym.get(int(pid), "?"))
    sym_list = sorted(seen)
    return sym_list, {s: i for i, s in enumerate(sym_list)}


def per_sample_colsum(rec, A_agg, id_to_sym, sym_to_idx, V):
    pids    = rec["node_phoneme_ids"].numpy(); N = rec["n_nodes"]
    sym_ids = np.array([sym_to_idx.get(id_to_sym.get(int(p), "?"), 0)
                        for p in pids], dtype=np.int64)
    H = np.zeros((N, V)); H[np.arange(N), sym_ids] = 1.0
    pair_sum = H.T @ A_agg @ H
    counts   = H.sum(axis=0); outer = counts[:, None] * counts[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        pair_mean = np.where(outer > 0, pair_sum / outer, np.nan)
    return np.nansum(pair_mean, axis=0)


def _e2_stat(atk, bon):
    with np.errstate(all="ignore"):
        return np.nanmean(atk, axis=0) - np.nanmean(bon, axis=0)


def run_hub_mass_perm(sample_colsums, attack_systems, V, n_perm=N_PERM):
    bon_cs = sample_colsums["-"]; results = {}
    for sid in sorted(attack_systems):
        atk_cs  = sample_colsums[sid]
        pool_cs = np.concatenate([atk_cs, bon_cs]); n_atk = len(atk_cs)
        obs     = _e2_stat(atk_cs, bon_cs)
        exceed  = np.zeros(V, dtype=np.int64)
        t0      = time.time()
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            exceed += (np.abs(pstat) >= np.abs(obs))
        p_raw  = (exceed + 1) / (n_perm + 1)
        p_holm = holm_bonferroni(p_raw)
        n_sig  = int((p_holm < 0.05).sum())
        max_obs = np.abs(obs).max(); max_exceed = 0
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            if np.abs(pstat).max() >= max_obs: max_exceed += 1
        p_family = (max_exceed + 1) / (n_perm + 1)
        print(f"  {sid} p_family={p_family:.5f}  n_sig={n_sig}/{V}  "
              f"({time.time()-t0:.1f}s)")
        results[sid] = {"obs": obs, "p_raw": p_raw, "p_holm": p_holm,
                        "p_family": p_family, "n_sig": n_sig}
    return results


# ── E4-style ablation (all 3 layers) ─────────────────────────────────────────

def install_ablation_all_layers(gat_net, heads_to_zero):
    removers = []
    for layer in gat_net:
        orig_nas = layer.neighborhood_aware_softmax
        def _make_abl(orig, heads):
            def _abl(scores, trg_index, num_nodes):
                attn = orig(scores, trg_index, num_nodes)
                if not heads: return attn
                attn = attn.clone()
                for h in heads: attn[:, h, 0] = 0.0
                return attn
            return _abl
        layer.neighborhood_aware_softmax = _make_abl(orig_nas, heads_to_zero)
        def _make_rm(lay):
            def rm():
                if "neighborhood_aware_softmax" in lay.__dict__:
                    del lay.__dict__["neighborhood_aware_softmax"]
            return rm
        removers.append(_make_rm(layer))
    def remove_all():
        for r in removers: r()
    return remove_all


def run_eval_condition(lit, loader, device, heads_to_zero):
    gm = lit.model; remove_all = install_ablation_all_layers(gm.GAT.gat_net, heads_to_zero)
    records = []; sid_counter = 0
    try:
        with torch.no_grad():
            for batch in loader:
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)
                hs, pids = run_frozen_frontend(audio, gm, device)
                result   = gm.encoder_and_GAT(hs, num_f, pids)
                logits   = result[5].cpu()
                for i in range(B):
                    records.append({"sample_id": sid_counter + i,
                                    "label": labels[i], "system_id": sys_ids[i],
                                    "logit": float(logits[i].item())})
                sid_counter += B
    finally:
        remove_all()
        for idx, layer in enumerate(gm.GAT.gat_net):
            assert "neighborhood_aware_softmax" not in layer.__dict__, \
                f"Layer {idx} not restored!"
    return records


def per_system_eer(records, attack_systems):
    bon_recs = [r for r in records if r["system_id"] == "-"]
    result   = {}
    for sid in attack_systems:
        atk_recs = [r for r in records if r["system_id"] == sid]
        combined = bon_recs + atk_recs
        labels   = np.array([r["label"] for r in combined])
        scores   = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in combined])))
        result[sid] = compute_eer(labels, scores)
    return result


def bootstrap_eer_ci(records, attack_systems, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    bon_recs = [r for r in records if r["system_id"] == "-"]
    result   = {}
    for sid in attack_systems:
        atk_recs = [r for r in records if r["system_id"] == sid]
        combined = bon_recs + atk_recs; n = len(combined)
        boot_eers = []
        for _ in range(n_boot):
            idx  = rng.integers(0, n, size=n)
            boot = [combined[i] for i in idx]
            lbs  = np.array([r["label"] for r in boot])
            sc   = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in boot])))
            if lbs.sum() == 0 or lbs.sum() == len(lbs): continue
            boot_eers.append(compute_eer(lbs, sc))
        if boot_eers:
            result[sid] = (float(np.percentile(boot_eers, 2.5)),
                           float(np.percentile(boot_eers, 97.5)))
        else:
            result[sid] = (float("nan"), float("nan"))
    return result


# ── P1–P4 evaluation ─────────────────────────────────────────────────────────

def evaluate_predictions(hub_perm_results: dict,
                          ablation_eer: dict[str, dict],
                          gini_new: dict[str, float],
                          attack_systems: list[str]) -> dict:
    print("\n" + "=" * 70)
    print("PRE-REGISTERED PREDICTION OUTCOMES")
    print("=" * 70)

    results = {}

    # ── P1: Spearman ρ of family p-value rankings ─────────────────────────────
    sids = sorted(attack_systems)
    orig_pf = [ORIGINAL_P_FAMILY[s] for s in sids]
    new_pf  = [hub_perm_results[s]["p_family"] for s in sids]
    rho, pval_spear = spearmanr(orig_pf, new_pf)
    p1_pass = rho >= 0.7
    print(f"\nP1 [primary]: Spearman ρ on family p-values = {rho:+.4f}  "
          f"(p={pval_spear:.4f})  {'PASS ✓' if p1_pass else 'FAIL ✗'}")
    print(f"  Threshold: ρ ≥ 0.70  |  Falsification: ρ ≤ 0.30")
    print("  Original vs new p_family:")
    for s in sids:
        print(f"    {s}:  orig={ORIGINAL_P_FAMILY[s]:.5f}  new={hub_perm_results[s]['p_family']:.5f}")
    results["P1"] = {"rho": rho, "spearman_p": pval_spear, "pass": p1_pass}

    # ── P2: h0/h4 causally sufficient (Δ within 0.02 of all-6 on A01/A03/A04) ─
    detectable = ["A01", "A03", "A04"]
    delta_all6  = {s: ablation_eer["attn_ablated"][s]  - ablation_eer["baseline"][s]
                   for s in detectable}
    delta_h0h4  = {s: ablation_eer["critical_only"][s] - ablation_eer["baseline"][s]
                   for s in detectable}
    diffs       = {s: abs(delta_h0h4[s] - delta_all6[s]) for s in detectable}
    p2_pass     = all(diffs[s] <= 0.02 for s in detectable)
    print(f"\nP2 [primary]: h0/h4 causal sufficiency")
    print(f"  {'system':6s}  {'Δ(all-6)':>10s}  {'Δ(h0,h4)':>10s}  {'|diff|':>8s}  {'≤0.02?':>7s}")
    for s in detectable:
        ok = "✓" if diffs[s] <= 0.02 else "✗"
        print(f"  {s:6s}  {delta_all6[s]:+10.4f}  {delta_h0h4[s]:+10.4f}  "
              f"{diffs[s]:8.4f}  {ok}")
    print(f"  Result: {'PASS ✓' if p2_pass else 'FAIL ✗'}")
    results["P2"] = {"diffs": diffs, "delta_all6": delta_all6,
                     "delta_h0h4": delta_h0h4, "pass": p2_pass}

    # ── P3: A05/A06 Δ-EER ≤ 0 under all-attention ablation ────────────────────
    vc_systems = ["A05", "A06"]
    delta_vc = {s: ablation_eer["attn_ablated"][s] - ablation_eer["baseline"][s]
                for s in vc_systems}
    ci_vc    = {s: ablation_eer.get("ci_attn_ablated", {}).get(s, (None, None))
                for s in vc_systems}
    # P3 fails only if Δ ≥ +0.04 WITH CI excluding 0 (not just directional)
    p3_violations = []
    for s in vc_systems:
        ci = ablation_eer.get("ci_attn_ablated", {}).get(s, (None, None))
        if delta_vc[s] >= 0.04 and ci[0] is not None and ci[0] > 0.0:
            p3_violations.append(s)
    p3_pass = len(p3_violations) == 0
    print(f"\nP3 [secondary]: A05/A06 Δ-EER ≤ 0 under all-attention ablation")
    for s in vc_systems:
        ci = ablation_eer.get("ci_attn_ablated", {}).get(s, (None, None))
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci[0] is not None else "n/a"
        viol = " ← violation" if s in p3_violations else ""
        print(f"  {s}: Δ={delta_vc[s]:+.4f}  CI={ci_str}{viol}")
    print(f"  Result: {'PASS ✓' if p3_pass else 'FAIL ✗'}")
    print(f"  (Falsification: Δ ≥ +0.04 AND CI lo > 0; currently "
          f"{', '.join(p3_violations) if p3_violations else 'none'})")
    results["P3"] = {"delta_vc": delta_vc, "violations": p3_violations, "pass": p3_pass}

    # ── P4: Gini direction ≥ 5/6 systems ──────────────────────────────────────
    bon_gini_new = gini_new["-"]
    agreement = {}
    for s in sids:
        orig_sign = ORIGINAL_DELTA_GINI_SIGN[s]
        new_delta  = gini_new[s] - bon_gini_new
        new_sign   = np.sign(new_delta)
        agreement[s] = (orig_sign == new_sign or
                        (orig_sign == 0.0 and abs(new_delta) < 0.01))
    n_agree = sum(agreement.values())
    p4_pass = n_agree >= 5
    print(f"\nP4 [secondary]: Gini direction agreement ≥ 5/6 systems")
    print(f"  {'system':6s}  {'orig ΔGini sign':>16s}  {'new ΔGini':>10s}  {'agree?':>7s}")
    for s in sids:
        orig_s = ORIGINAL_DELTA_GINI_SIGN[s]
        new_d  = gini_new[s] - bon_gini_new
        ok_str = "✓" if agreement[s] else "✗"
        sign_str = "+" if orig_s > 0 else ("-" if orig_s < 0 else "0")
        print(f"  {s:6s}  {sign_str:>16s}  {new_d:+10.4f}  {ok_str}")
    print(f"  Agreement: {n_agree}/6  Result: {'PASS ✓' if p4_pass else 'FAIL ✗'}")
    results["P4"] = {"agreement": agreement, "n_agree": n_agree, "pass": p4_pass}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    for pred, res in results.items():
        status = "PASS ✓" if res["pass"] else "FAIL ✗"
        print(f"  {pred}:  {status}")
    primary_pass = results["P1"]["pass"] and results["P2"]["pass"]
    print(f"\nPrimary cluster (P1+P2): {'BOTH PASS → causal-circuit claim survives single-seed concern' if primary_pass else 'AT LEAST ONE FAIL → see decision rules'}")
    print("=" * 70)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to new checkpoint, e.g. models/robust_goat_seed7.ckpt")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip E4-style ablation (only run hub-mass + Gini)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    patch_phoneme_loader()
    print(f"Loading checkpoint: {ckpt_path}")
    lit = load_model(ckpt_path, device)
    print(f"  Device: {device}  ckpt: {ckpt_path.name}")

    dataset = BalancedDataset(HF_DATASET, "validation", CACHE_DIR,
                              hf_token, N_PER_CLASS, SEED)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate)

    systems        = sorted(set(r["system_id"] for r in
                                [dataset[i] for i in range(len(dataset))]))
    attack_systems = [s for s in systems if s != "-"]

    # ── Step 1: Extract all-layer attention ──────────────────────────────────
    print(f"\n--- Extracting all-layer attention from {ckpt_path.name} ---")
    records = extract_all_layers(lit, loader, device)
    valid   = [r for r in records if not r["is_degenerate"]]
    print(f"  Valid: {len(valid)}")

    id_to_sym, _ = build_vocab()
    sym_list, sym_to_idx = build_symbol_vocab(records, id_to_sym)
    V = len(sym_list)
    print(f"  Vocab V={V}")

    # ── Step 2: Hub-mass permutation (P1) ─────────────────────────────────────
    print(f"\n--- Hub-mass permutation ({N_PERM} perms) for P1 ---")
    sample_colsums: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in valid:
        A_agg  = build_agg_graph(rec)
        colsum = per_sample_colsum(rec, A_agg, id_to_sym, sym_to_idx, V)
        sample_colsums[rec["system_id"]].append(colsum)
    stacked = {s: np.stack(arrs) for s, arrs in sample_colsums.items()}
    hub_perm = run_hub_mass_perm(stacked, attack_systems, V)

    # ── Step 3: Gini (P4) ────────────────────────────────────────────────────
    print("\n--- Computing Gini for P4 ---")
    bon_colsum  = stacked["-"].mean(axis=0)   # (V,) mean colsum = d_in
    gini_new: dict[str, float] = {}
    gini_new["-"] = compute_gini(bon_colsum)
    print(f"  bonafide Gini: {gini_new['-']:.4f}  (original: {ORIGINAL_GINI['-']:.4f})")
    for sid in sorted(attack_systems):
        atk_colsum = stacked[sid].mean(axis=0)
        gini_new[sid] = compute_gini(atk_colsum)
        delta = gini_new[sid] - gini_new["-"]
        orig_delta = ORIGINAL_GINI[sid] - ORIGINAL_GINI["-"]
        print(f"  {sid}: Gini={gini_new[sid]:.4f}  Δ={delta:+.4f}  "
              f"(orig Δ={orig_delta:+.4f})")

    # ── Step 4: Ablation EER (P2, P3) ────────────────────────────────────────
    ablation_eer: dict[str, dict] = {}
    ci_attn_ablated: dict = {}

    if not args.no_ablation:
        print("\n--- Ablation conditions (P2, P3) ---")
        conditions = [
            ("baseline",      []),
            ("attn_ablated",  list(range(6))),
            ("critical_only", CRITICAL_HEADS),
        ]
        all_records: dict[str, list[dict]] = {}
        for cond_name, heads_to_zero in conditions:
            t0   = time.time()
            recs = run_eval_condition(lit, loader, device, heads_to_zero)
            all_records[cond_name] = recs
            total_eer = compute_eer(
                np.array([r["label"] for r in recs]),
                1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in recs]))))
            print(f"  {cond_name:16s}  overall EER={total_eer:.4f}  ({time.time()-t0:.1f}s)")

        for cond_name, _ in conditions:
            ablation_eer[cond_name] = per_system_eer(all_records[cond_name], attack_systems)

        ci_attn_ablated = bootstrap_eer_ci(all_records["attn_ablated"], attack_systems)
        ablation_eer["ci_attn_ablated"] = ci_attn_ablated

        print(f"\n  {'system':8s}  {'baseline':>10s}  {'attn_abl':>10s}  "
              f"{'crit_only':>10s}  {'Δ(abl)':>8s}  {'Δ(crit)':>8s}")
        for s in sorted(attack_systems):
            b  = ablation_eer["baseline"][s]
            aa = ablation_eer["attn_ablated"][s]
            co = ablation_eer["critical_only"][s]
            print(f"  {s:8s}  {b:10.4f}  {aa:10.4f}  {co:10.4f}  "
                  f"{aa-b:+8.4f}  {co-b:+8.4f}")
    else:
        print("\n  [Skipping ablation — use without --no-ablation for P2/P3]")
        for cond in ("baseline", "attn_ablated", "critical_only"):
            ablation_eer[cond] = {s: float("nan") for s in attack_systems}

    # ── P1–P4 evaluation ─────────────────────────────────────────────────────
    pred_results = evaluate_predictions(hub_perm, ablation_eer, gini_new, attack_systems)

    # ── Save report ───────────────────────────────────────────────────────────
    lines = [f"# E5: Replication Study — {ckpt_path.name}\n",
             "## Hub-mass family p-values\n",
             "| system | p_family (original) | p_family (new) |",
             "|--------|--------------------|--------------------|"]
    for s in sorted(attack_systems):
        lines.append(f"| {s} | {ORIGINAL_P_FAMILY[s]:.5f} | "
                     f"{hub_perm[s]['p_family']:.5f} |")

    lines += [f"\nSpearman ρ (P1) = {pred_results['P1']['rho']:+.4f}  "
              f"(p={pred_results['P1']['spearman_p']:.4f})\n",
              "## Ablation EER\n",
              "| system | baseline | attn_ablated | critical_only | Δ(abl) | Δ(crit) |",
              "|--------|---------|-------------|--------------|--------|---------|"]
    for s in sorted(attack_systems):
        b  = ablation_eer["baseline"].get(s, float("nan"))
        aa = ablation_eer["attn_ablated"].get(s, float("nan"))
        co = ablation_eer["critical_only"].get(s, float("nan"))
        lines.append(f"| {s} | {b:.4f} | {aa:.4f} | {co:.4f} | {aa-b:+.4f} | {co-b:+.4f} |")

    lines += ["\n## Gini\n",
              "| system | Gini (original) | Gini (new) | ΔGini new | direction agree? |",
              "|--------|----------------|------------|-----------|-----------------|"]
    for s in sorted(attack_systems):
        orig_d = ORIGINAL_GINI[s] - ORIGINAL_GINI["-"]
        new_d  = gini_new[s] - gini_new["-"]
        ok     = "✓" if pred_results["P4"]["agreement"][s] else "✗"
        lines.append(f"| {s} | {ORIGINAL_GINI[s]:.4f} ({orig_d:+.4f}) "
                     f"| {gini_new[s]:.4f} ({new_d:+.4f}) | {ok} |")

    lines += ["\n## Prediction outcomes\n"]
    for pred, res in pred_results.items():
        status = "PASS" if res["pass"] else "FAIL"
        lines.append(f"- **{pred}**: {status}")

    primary_pass = pred_results["P1"]["pass"] and pred_results["P2"]["pass"]
    lines.append(f"\n**Primary cluster (P1+P2): "
                 f"{'PASS — causal-circuit claim survives' if primary_pass else 'FAIL — see decision rules'}**")

    out = OUT_BASE / f"e5_report_{ckpt_path.stem}.md"
    out.write_text("\n".join(lines))
    print(f"\n  Saved: {out}")

    # Save raw CSV
    rows = []
    for s in sorted(attack_systems):
        rows.append({"system": s,
                     "p_family_orig":  ORIGINAL_P_FAMILY[s],
                     "p_family_new":   hub_perm[s]["p_family"],
                     "gini_orig":      ORIGINAL_GINI[s],
                     "gini_new":       gini_new[s],
                     "gini_agree":     pred_results["P4"]["agreement"][s],
                     "eer_baseline":   ablation_eer["baseline"].get(s, float("nan")),
                     "eer_attn_abl":   ablation_eer["attn_ablated"].get(s, float("nan")),
                     "eer_crit_only":  ablation_eer["critical_only"].get(s, float("nan")),
                     })
    csv_out = OUT_BASE / f"e5_raw_{ckpt_path.stem}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {csv_out}")
    print(f"\nAll E5 outputs in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
