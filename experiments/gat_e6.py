#!/usr/bin/env python3
"""
gat_e6.py
=========
E6: Seed-3 ranking replication — pre-registered H1/H2/H3 on a third independent
checkpoint. Hub-mass analysis only (no ablation, no Gini).

Pre-registered hypotheses (commit before training seed-3 checkpoint):

  H1 [primary]:   Spearman rank ρ of per-system family p-values between
                  seed 3 and seed 1 ≥ 0.9, AND between seed 3 and seed 2 ≥ 0.9.
                  Falsification: either pairwise ρ ≤ 0.6.

  H2 [primary]:   The three previously-significant systems (A01, A03, A04) all
                  have p_family < 0.05 on seed 3.  The three null systems
                  (A02, A05, A06) all have p_family > 0.05 on seed 3.
                  Falsification: any system flips category.

  H3 [secondary]: Raw p-value stability.  For each of A01, A03, A04 the seed-3
                  p-value lies within 5× of the seed-1 p-value (either direction).
                  Falsification: any of the three exceeds the 5× band.

Decision logic:
  H1 + H2 both pass → ranking-invariance claim is locked.
  H1 passes, H2 fails → ordering preserved but category boundary checkpoint-dependent.
  H1 fails → P1's two-seed agreement was a fluke; paper restructures around Gini / TTS-vs-VC.

Usage:
  venv/bin/python3 experiments/gat_e6.py --ckpt models/robust_goat_seed3.ckpt

  # Dry run first (1k perms) — required sanity check before full 10k
  venv/bin/python3 experiments/gat_e6.py --ckpt models/robust_goat_seed3.ckpt --dry-run

Outputs → experiments/results/gat_attn_graphs/e6/
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
REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR   = Path(__file__).resolve().parent
OUT_BASE  = EXP_DIR / "results" / "gat_attn_graphs" / "e6"
VOCAB_DIR = REPO_ROOT / "vocab_phoneme"

OUT_BASE.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Hardcoded prior-seed p_family values ─────────────────────────────────────
# Seed 1 (models/robust_goat.ckpt):  from gat_attn_graphs.py --full-perm (E2)
SEED1_P_FAMILY = {
    "A01": 0.00010,
    "A02": 0.45555,
    "A03": 0.00250,
    "A04": 0.00820,
    "A05": 0.30577,
    "A06": 0.51565,
}

# Seed 2 (models/robust_goat_seed7.ckpt):  from gat_e5.py run (E5)
SEED2_P_FAMILY = {
    "A01": 9.999000099990002e-05,
    "A02": 0.565943405659434,
    "A03": 0.0034996500349965005,
    "A04": 0.009099090090990901,
    "A05": 0.32496750324967505,
    "A06": 0.5821417858214178,
}

# Pre-registered category labels (H2)
SIGNIFICANT_SYSTEMS = {"A01", "A03", "A04"}   # p_family < 0.05 on seeds 1 & 2
NULL_SYSTEMS        = {"A02", "A05", "A06"}    # p_family > 0.05 on seeds 1 & 2

# Pre-registered sanity-check reference: seeds 1/2 training val-EER ≈ 0.05–0.08
# Halt if seed-3 overall EER > 0.20 or < 0.005 on the balanced 350-sample eval
EER_SANITY_LO = 0.005
EER_SANITY_HI = 0.200

# ── Constants (verbatim from gat_e5.py) ──────────────────────────────────────
SEED           = 42
N_PERM_FULL    = 10_000
N_PERM_DRY     = 1_000
NF_PER_SAMPLE  = 3 * 16_000 // 320 - 1
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
HF_DATASET     = "Bisher/ASVspoof_2019_LA"
CACHE_DIR      = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH  = REPO_ROOT / "secret.txt"
LANG_ORDER     = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL        = ["|", "</s>", "<s>", "<unk>", "<pad>"]
TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
MIN_NODES      = 3


# ── Shared helpers (verbatim from gat_e5.py) ─────────────────────────────────

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
    return id_to_sym


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


# ── Audio helpers (verbatim from gat_e5.py) ──────────────────────────────────

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


# ── Dataset (verbatim from gat_e5.py) ────────────────────────────────────────

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


# ── Model loading (verbatim from gat_e5.py) ───────────────────────────────────

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


# ── Phoneme capture (verbatim from gat_e5.py) ────────────────────────────────

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


# ── Frontend + extraction (verbatim from gat_e5.py) ─────────────────────────

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


# ── Sanity-check forward pass (logits only) ──────────────────────────────────

def sanity_check_eer(lit, loader, device):
    gm = lit.model; records = []; sid_counter = 0
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
                records.append({"label": labels[i], "system_id": sys_ids[i],
                                "logit": float(logits[i].item())})
            sid_counter += B
    labels_arr = np.array([r["label"] for r in records])
    scores_arr = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in records])))
    overall_eer = compute_eer(labels_arr, scores_arr)
    return overall_eer, records


# ── A_agg + hub-mass pipeline (verbatim from gat_e5.py) ─────────────────────

def sparse_to_dense(attn_mean, edge_index, N):
    A = np.zeros((N, N), dtype=np.float64)
    np.add.at(A, (edge_index[1], edge_index[0]), attn_mean)
    row_sums = A.sum(axis=1); no_in = np.where(row_sums < 1e-10)[0]
    A[no_in, no_in] = 1.0
    return A


def build_agg_graph(rec):
    N = rec["n_nodes"]; ei = rec["edge_index"].numpy(); layers = []
    for li in range(3):
        attn_np   = rec[f"attn_l{li}"].float().numpy()
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


def run_hub_mass_perm(sample_colsums, attack_systems, V, n_perm):
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


# ── H1/H2/H3 evaluation ───────────────────────────────────────────────────────

def evaluate_hypotheses(seed3_pf: dict[str, float],
                         attack_systems: list[str]) -> dict:
    sids = sorted(attack_systems)

    s1_pf = [SEED1_P_FAMILY[s] for s in sids]
    s2_pf = [SEED2_P_FAMILY[s] for s in sids]
    s3_pf = [seed3_pf[s]       for s in sids]

    rho_s3_s1, pval_s3_s1 = spearmanr(s3_pf, s1_pf)
    rho_s3_s2, pval_s3_s2 = spearmanr(s3_pf, s2_pf)
    rho_s1_s2, pval_s1_s2 = spearmanr(s1_pf, s2_pf)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED HYPOTHESIS OUTCOMES  (E6)")
    print("=" * 70)

    # ── H1 ────────────────────────────────────────────────────────────────────
    h1_pass = (rho_s3_s1 >= 0.9) and (rho_s3_s2 >= 0.9)
    print(f"\nH1 [primary]: Spearman ρ ≥ 0.9 for both seed-3 vs seed-1 and seed-3 vs seed-2")
    print(f"  Seed3 vs Seed1:  ρ = {rho_s3_s1:+.4f}  (p={pval_s3_s1:.4f})  "
          f"{'≥0.9 ✓' if rho_s3_s1 >= 0.9 else '<0.9 ✗'}")
    print(f"  Seed3 vs Seed2:  ρ = {rho_s3_s2:+.4f}  (p={pval_s3_s2:.4f})  "
          f"{'≥0.9 ✓' if rho_s3_s2 >= 0.9 else '<0.9 ✗'}")
    print(f"  Seed1 vs Seed2:  ρ = {rho_s1_s2:+.4f}  (p={pval_s1_s2:.4f})  [reference]")
    print(f"  Falsification: either ρ ≤ 0.6")
    print(f"  Result: {'PASS ✓' if h1_pass else 'FAIL ✗'}")

    # ── H2 ────────────────────────────────────────────────────────────────────
    flips = []
    for s in sids:
        was_sig = s in SIGNIFICANT_SYSTEMS
        is_sig  = seed3_pf[s] < 0.05
        if was_sig != is_sig:
            flips.append((s, was_sig, is_sig))
    h2_pass = len(flips) == 0
    print(f"\nH2 [primary]: Category membership preserved (sig/null split) on seed 3")
    print(f"  {'system':6s}  {'seed1/2 category':>20s}  {'seed3 p_family':>16s}  {'category':>12s}  {'flip?':>6s}")
    for s in sids:
        cat = "sig" if s in SIGNIFICANT_SYSTEMS else "null"
        s3  = seed3_pf[s]
        s3cat = "sig" if s3 < 0.05 else "null"
        flip = "✗ FLIP" if cat != s3cat else "✓"
        print(f"  {s:6s}  {cat:>20s}  {s3:>16.5f}  {s3cat:>12s}  {flip:>6s}")
    if flips:
        print(f"  Flipped systems: {', '.join(s for s,_,_ in flips)}")
    print(f"  Result: {'PASS ✓' if h2_pass else 'FAIL ✗'}")

    # ── H3 ────────────────────────────────────────────────────────────────────
    det = sorted(SIGNIFICANT_SYSTEMS)
    h3_violations = []
    print(f"\nH3 [secondary]: Seed-3 p-value within 5× of seed-1 for A01/A03/A04")
    print(f"  {'system':6s}  {'seed1':>10s}  {'seed3':>10s}  {'ratio':>8s}  {'≤5×?':>6s}")
    for s in det:
        p1 = SEED1_P_FAMILY[s]; p3 = seed3_pf[s]
        ratio = max(p1, p3) / max(min(p1, p3), 1e-10)
        ok = ratio <= 5.0
        if not ok: h3_violations.append(s)
        print(f"  {s:6s}  {p1:10.5f}  {p3:10.5f}  {ratio:8.2f}×  {'✓' if ok else '✗'}")
    h3_pass = len(h3_violations) == 0
    print(f"  Result: {'PASS ✓' if h3_pass else 'FAIL ✗'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'-'*70}")
    print(f"  H1 (ρ ≥ 0.9 both pairs):           {'PASS ✓' if h1_pass else 'FAIL ✗'}")
    print(f"  H2 (category membership stable):    {'PASS ✓' if h2_pass else 'FAIL ✗'}")
    print(f"  H3 (p-val within 5× of seed-1):     {'PASS ✓' if h3_pass else 'FAIL ✗'}")

    if h1_pass and h2_pass:
        verdict = ("H1 + H2 PASS → ranking-invariance is locked across three "
                   "independent training runs.  Write the paper.")
    elif h1_pass and not h2_pass:
        verdict = ("H1 PASS, H2 FAIL → ordering preserved but category boundary "
                   "is checkpoint-dependent near the 0.05 threshold.  Note in paper.")
    else:
        verdict = ("H1 FAIL → P1's two-seed agreement did not replicate.  "
                   "Paper restructures around Gini taxonomy and TTS-vs-VC split.")

    print(f"\n  Decision: {verdict}")
    print("=" * 70)

    return {
        "H1": {"rho_s3_s1": rho_s3_s1, "rho_s3_s2": rho_s3_s2,
               "rho_s1_s2": rho_s1_s2, "pass": h1_pass},
        "H2": {"flips": flips, "pass": h2_pass},
        "H3": {"violations": h3_violations, "pass": h3_pass},
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to seed-3 checkpoint, e.g. models/robust_goat_seed3.ckpt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1k-perm dry run only (sanity check before full 10k)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    n_perm = N_PERM_DRY if args.dry_run else N_PERM_FULL
    print(f"=== E6: Seed-3 Ranking Replication ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Permutations: {n_perm}{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  Seed       : {SEED}")

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    patch_phoneme_loader()
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    lit = load_model(ckpt_path, device)
    print(f"  Device: {device}")

    dataset = BalancedDataset(HF_DATASET, "validation", CACHE_DIR,
                              hf_token, N_PER_CLASS, SEED)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate)

    systems        = sorted(set(r["system_id"] for r in
                                [dataset[i] for i in range(len(dataset))]))
    attack_systems = [s for s in systems if s != "-"]
    print(f"  Attack systems: {attack_systems}")

    # ── Sanity check: overall EER ─────────────────────────────────────────────
    print(f"\n--- Sanity check: overall EER on {len(dataset)}-sample balanced eval ---")
    overall_eer, logit_records = sanity_check_eer(lit, loader, device)
    print(f"  Overall EER: {overall_eer:.4f}")
    print(f"  Reference range (seeds 1–2): approx 0.03–0.10")
    if overall_eer < EER_SANITY_LO or overall_eer > EER_SANITY_HI:
        print(f"  HALT: EER {overall_eer:.4f} outside sanity band "
              f"[{EER_SANITY_LO},{EER_SANITY_HI}] — model may not have converged.")
        sys.exit(2)
    print(f"  Sanity check PASSED.")

    # ── Extract all-layer attention ───────────────────────────────────────────
    print(f"\n--- Extracting all-layer attention ---")
    records = extract_all_layers(lit, loader, device)
    valid   = [r for r in records if not r["is_degenerate"]]
    print(f"  Valid: {len(valid)}")

    id_to_sym = build_vocab()
    sym_list, sym_to_idx = build_symbol_vocab(records, id_to_sym)
    V = len(sym_list)
    print(f"  Vocab V={V}")

    # ── Hub-mass permutation ──────────────────────────────────────────────────
    print(f"\n--- Hub-mass permutation ({n_perm} perms) ---")
    sample_colsums: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in valid:
        A_agg  = build_agg_graph(rec)
        colsum = per_sample_colsum(rec, A_agg, id_to_sym, sym_to_idx, V)
        sample_colsums[rec["system_id"]].append(colsum)
    stacked = {s: np.stack(arrs) for s, arrs in sample_colsums.items()}
    hub_perm = run_hub_mass_perm(stacked, attack_systems, V, n_perm)

    seed3_pf = {s: hub_perm[s]["p_family"] for s in attack_systems}

    # ── Per-system p-value comparison table ───────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-SYSTEM FAMILY p-VALUE: ALL THREE SEEDS")
    print(f"{'='*70}")
    print(f"  {'system':6s}  {'seed1':>10s}  {'seed2':>10s}  {'seed3':>10s}  {'rank ok?':>9s}")
    for s in sorted(attack_systems):
        p1 = SEED1_P_FAMILY[s]; p2 = SEED2_P_FAMILY[s]; p3 = seed3_pf[s]
        sig1 = "*" if p1 < 0.05 else " "
        sig2 = "*" if p2 < 0.05 else " "
        sig3 = "*" if p3 < 0.05 else " "
        cat_match = (p3 < 0.05) == (s in SIGNIFICANT_SYSTEMS)
        print(f"  {s:6s}  {p1:>9.5f}{sig1}  {p2:>9.5f}{sig2}  {p3:>9.5f}{sig3}  "
              f"{'✓' if cat_match else '✗':>9s}")
    print(f"  (* = p_family < 0.05)")

    # ── Apply pre-registered hypotheses ──────────────────────────────────────
    hyp_results = evaluate_hypotheses(seed3_pf, attack_systems)

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = ckpt_path.stem
    suffix = "_dry" if args.dry_run else ""

    # CSV
    rows = []
    for s in sorted(attack_systems):
        rows.append({
            "system":       s,
            "p_family_s1":  SEED1_P_FAMILY[s],
            "p_family_s2":  SEED2_P_FAMILY[s],
            "p_family_s3":  seed3_pf[s],
            "n_sig_s3":     hub_perm[s]["n_sig"],
            "cat_s1s2":     "sig" if s in SIGNIFICANT_SYSTEMS else "null",
            "cat_s3":       "sig" if seed3_pf[s] < 0.05 else "null",
            "cat_flip":     ("sig" if s in SIGNIFICANT_SYSTEMS else "null") !=
                            ("sig" if seed3_pf[s] < 0.05 else "null"),
        })
    csv_path = OUT_BASE / f"e6_raw_{stem}{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Markdown report
    rho_s3_s1 = hyp_results["H1"]["rho_s3_s1"]
    rho_s3_s2 = hyp_results["H1"]["rho_s3_s2"]
    rho_s1_s2 = hyp_results["H1"]["rho_s1_s2"]
    lines = [
        f"# E6: Seed-3 Ranking Replication — {stem}",
        f"n_perm={n_perm}{'  (DRY RUN)' if args.dry_run else ''}  seed={SEED}",
        "",
        "## Per-system family p-values",
        "",
        "| system | seed 1 | seed 2 | seed 3 | category flip? |",
        "|--------|--------|--------|--------|----------------|",
    ]
    for s in sorted(attack_systems):
        p1 = SEED1_P_FAMILY[s]; p2 = SEED2_P_FAMILY[s]; p3 = seed3_pf[s]
        flip = "yes" if ("sig" if s in SIGNIFICANT_SYSTEMS else "null") != \
                        ("sig" if p3 < 0.05 else "null") else "no"
        lines.append(f"| {s} | {p1:.5f} | {p2:.5f} | {p3:.5f} | {flip} |")

    lines += [
        "",
        "## Pairwise Spearman ρ matrix",
        "",
        "| pair       | ρ       |",
        "|------------|---------|",
        f"| seed3 vs seed1 | {rho_s3_s1:+.4f} |",
        f"| seed3 vs seed2 | {rho_s3_s2:+.4f} |",
        f"| seed1 vs seed2 | {rho_s1_s2:+.4f} |",
        "",
        "## Hypothesis outcomes",
        "",
    ]
    for h, res in hyp_results.items():
        if h == "verdict": continue
        status = "PASS" if res["pass"] else "FAIL"
        lines.append(f"- **{h}**: {status}")

    lines += ["", f"**Decision: {hyp_results['verdict']}**"]

    md_path = OUT_BASE / f"e6_report_{stem}{suffix}.md"
    md_path.write_text("\n".join(lines))

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {md_path}")
    print(f"\nAll E6 outputs in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
