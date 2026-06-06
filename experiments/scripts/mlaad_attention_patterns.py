#!/usr/bin/env python3
"""
mlaad_attention_patterns.py
============================
Attention pattern analysis on MLAAD checkpoints for the target heads
identified in Phase 2 (h2 = top-1, h4 = top-2).

Three analyses:
  1. cross_checkpoint     — compare mlaad_goat vs mlaad_robust_goat on the
                            same fixed in-distribution subset (entropy, L2 diffs,
                            comparison against ASVspoof numbers)
  2. per_attack_robust    — per-attack-system attention patterns on mlaad_robust_goat
                            (9×9 phoneme-class heatmaps per head)
  3. cross_language_stability — cosine similarity of per-head attention matrices
                            between in-distribution (EN) and cross-language (DE spoof)
                            for systems present in both test sets

Fixed subset: min(100, available) bonafide + min(30, available) per attack system,
              seed=42; identical example indices across checkpoints.

Usage:
    python experiments/scripts/mlaad_attention_patterns.py

Reads:
    experiments/results/mlaad/head_discovery/target_heads.json
    experiments/results/mlaad/baseline_eval/test_{in_distribution,cross_language}.json
    experiments/checkpoints/mlaad_{goat,robust_goat}-best-*.ckpt

Writes:
    experiments/results/mlaad/attention_patterns/
        cross_checkpoint/
            entropy_per_head.json
            l2_diff_matrices.json
            checkpoint_comparison.json
        per_attack_robust/
            aggregates.npz          (C,C,2) per attack_system × head
            summary_stats.csv
        cross_language_stability/
            cosine_similarity.json
            stability_summary.json
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
HEAD_DISC_DIR = EXP_DIR / "results" / "mlaad" / "head_discovery"
BASELINE_DIR  = EXP_DIR / "results" / "mlaad" / "baseline_eval"
OUT_BASE      = EXP_DIR / "results" / "mlaad" / "attention_patterns"
VOCAB_DIR     = PROJECT_ROOT / "vocab_phoneme"

for _p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from head_ablation import load_model, run_frozen_frontend, patch_phoneme_loader, NF_PER_SAMPLE
from utils.attention_hook import AttentionHook, PhonemeCapture

# ── Constants ─────────────────────────────────────────────────────────────────
SEED         = 42
N_BONAFIDE   = 100
N_PER_ATTACK = 30
BATCH_SIZE   = 8
MIN_NODES    = 3

LANG_ORDER  = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL     = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]
C           = len(CLASS_ORDER)

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


# ── JSON helper ───────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_Enc, indent=2))


# ── Vocab ─────────────────────────────────────────────────────────────────────

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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _best_or_last(stem: str) -> Path:
    candidates = sorted(CKPT_DIR.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else CKPT_DIR / f"{stem}.ckpt"


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_records(
    records: list[dict],
    n_bonafide: int,
    n_per_attack: int,
    seed: int,
) -> list[dict]:
    """
    Sample up to n_bonafide bonafide + up to n_per_attack per attack system.
    Returns records in a consistent random order (seeded).
    """
    rng = random.Random(seed)

    bonafide = [r for r in records if r["label"] == "bonafide"]
    rng.shuffle(bonafide)
    selected = bonafide[:n_bonafide]

    by_sys: dict[str, list[dict]] = {}
    for r in records:
        if r["label"] != "bonafide":
            by_sys.setdefault(r["attack_system"], []).append(r)
    for sid in sorted(by_sys):
        pool = list(by_sys[sid])
        rng.shuffle(pool)
        selected.extend(pool[:n_per_attack])

    rng.shuffle(selected)
    return selected


# ── Dataset ───────────────────────────────────────────────────────────────────

class MAALDAttnDataset(Dataset):
    def __init__(self, records: list[dict], processed_dir: Path) -> None:
        self.records      = records
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


def make_loader(records: list[dict]) -> DataLoader:
    ds = MAALDAttnDataset(records, PROCESSED_DIR)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, collate_fn=_collate)


# ── Attention extraction ──────────────────────────────────────────────────────

def extract_attention(
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    label: str = "",
) -> list[dict]:
    """
    Run AttentionHook + PhonemeCapture over all batches.
    Returns list of per-sample dicts with attn_l0 (E, NH), edge_index (2, E), etc.
    """
    print(f"\n  Extracting {label} ({ckpt_path.name})...")
    patch_phoneme_loader()
    lit = load_model(ckpt_path, device)
    gat_model = lit.model

    hook        = AttentionHook(gat_model, layer_idx=0).install()
    phoneme_cap = PhonemeCapture(gat_model)

    records:     list[dict] = []
    n_degenerate = 0
    total        = len(loader.dataset)

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hook.clear()
                hs, pids = run_frozen_frontend(audio, gat_model, device)
                gat_model.encoder_and_GAT(hs, num_f, pids)

                ei_g   = hook.edge_index
                attn_g = hook.attn
                if ei_g is None or attn_g is None:
                    print(f"  [WARN] batch {bi}: hook did not fire, skipping")
                    continue

                node_pids = phoneme_cap.node_phoneme_ids
                node_samp = phoneme_cap.node_sample_idx
                rnf       = phoneme_cap.reduced_num_frames

                offsets = [0]
                for i in range(B - 1):
                    offsets.append(offsets[-1] + int(rnf[i].item()))

                for i in range(B):
                    n_i  = int(rnf[i].item())
                    off  = offsets[i]
                    mask = (node_samp[ei_g[0]] == i)
                    ei_i = ei_g[:, mask] - off
                    a_i  = attn_g[mask]
                    pid_i = node_pids[node_samp == i]
                    deg   = n_i < MIN_NODES
                    if deg:
                        n_degenerate += 1
                    records.append({
                        "system_id": sys_ids[i],
                        "label":     labels[i],
                        "edge_index": ei_i.clone(),
                        "attn_l0":    a_i.clone(),
                        "node_pids":  pid_i.clone(),
                        "n_nodes":    n_i,
                        "is_deg":     deg,
                    })

                if (bi + 1) % 20 == 0 or (bi + 1) == len(loader):
                    print(f"    {min((bi+1)*BATCH_SIZE, total)}/{total}")
    finally:
        hook.remove()

    print(f"  Done: {len(records)} samples, {n_degenerate} degenerate")
    return records


# ── 9×9 aggregation ───────────────────────────────────────────────────────────

def aggregate_9x9(
    records: list[dict],
    id_to_cls: dict[int, str],
    head_idx: int,
    system_id: str,
) -> np.ndarray:
    """(C,C) attention matrix, edge-sum-normalized for (head, system_id)."""
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    accum   = np.zeros((C, C), dtype=np.float64)
    for rec in records:
        if rec["is_deg"] or rec["system_id"] != system_id:
            continue
        pids = rec["node_pids"]
        ei   = rec["edge_index"]
        attn = rec["attn_l0"][:, head_idx].numpy()
        for e in range(ei.shape[1]):
            sc = cls_idx[id_to_cls.get(int(pids[ei[0, e]].item()), "Other")]
            tc = cls_idx[id_to_cls.get(int(pids[ei[1, e]].item()), "Other")]
            accum[sc, tc] += float(attn[e])
    total = accum.sum()
    if total > 0:
        accum /= total
    return accum


# ── Per-sample entropy ────────────────────────────────────────────────────────

def mean_entropy(records: list[dict], head_idx: int, system_id: str) -> float:
    """Mean per-query-node attention entropy across samples matching system_id."""
    entropies = []
    for rec in records:
        if rec["is_deg"] or rec["system_id"] != system_id:
            continue
        attn_h = rec["attn_l0"][:, head_idx].float()
        tgt    = rec["edge_index"][1]
        for t in range(rec["n_nodes"]):
            mask = (tgt == t)
            if not mask.any():
                continue
            a   = attn_h[mask]
            ent = -(a * (a + 1e-10).log()).sum().item()
            entropies.append(ent)
    return float(np.mean(entropies)) if entropies else float("nan")


# ── Analysis 1: Cross-checkpoint comparison ───────────────────────────────────

def cross_checkpoint_comparison(
    records_a: list[dict],
    records_b: list[dict],
    name_a: str,
    name_b: str,
    id_to_cls: dict[int, str],
    target_heads: list[int],
    all_sids: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Analysis 1: Cross-checkpoint comparison ({name_a} vs {name_b}) ---")

    # Per-head per-system entropy comparison
    entropy_results: dict = {}
    for h in target_heads:
        entropy_results[f"h{h}"] = {}
        for sid in all_sids:
            ea = mean_entropy(records_a, h, sid)
            eb = mean_entropy(records_b, h, sid)
            entropy_results[f"h{h}"][sid] = {
                name_a: round(float(ea), 4) if not np.isnan(ea) else None,
                name_b: round(float(eb), 4) if not np.isnan(eb) else None,
                "delta":  round(float(eb - ea), 4) if not (np.isnan(ea) or np.isnan(eb)) else None,
            }

    _dump(entropy_results, out_dir / "entropy_per_head.json")
    print(f"  Wrote: {out_dir}/entropy_per_head.json")

    # L2 differences in 9×9 attention matrices per (head, system)
    l2_results: dict = {}
    for h in target_heads:
        l2_results[f"h{h}"] = {}
        for sid in all_sids:
            mat_a = aggregate_9x9(records_a, id_to_cls, h, sid)
            mat_b = aggregate_9x9(records_b, id_to_cls, h, sid)
            l2    = float(np.sqrt(((mat_a - mat_b) ** 2).sum()))
            cos   = _cosine_sim(mat_a.flatten(), mat_b.flatten())
            l2_results[f"h{h}"][sid] = {
                "l2_diff":       round(l2, 4),
                "cosine_sim":    round(cos, 4),
            }

    _dump(l2_results, out_dir / "l2_diff_matrices.json")
    print(f"  Wrote: {out_dir}/l2_diff_matrices.json")

    # High-level comparison summary
    for h in target_heads:
        all_l2   = [v["l2_diff"]    for v in l2_results[f"h{h}"].values() if v["l2_diff"] is not None]
        all_cos  = [v["cosine_sim"] for v in l2_results[f"h{h}"].values() if v["cosine_sim"] is not None]
        if all_l2:
            print(f"  h{h}: mean L2={np.mean(all_l2):.4f}  mean cosine_sim={np.mean(all_cos):.4f}")

    comparison = {
        "checkpoints": {name_a: None, name_b: None},
        "target_heads": target_heads,
        "n_systems": len(all_sids),
        "per_head_mean_l2": {
            f"h{h}": round(float(np.mean([
                v["l2_diff"] for v in l2_results[f"h{h}"].values()
            ])), 4)
            for h in target_heads
        },
        "per_head_mean_cosine_sim": {
            f"h{h}": round(float(np.mean([
                v["cosine_sim"] for v in l2_results[f"h{h}"].values()
            ])), 4)
            for h in target_heads
        },
        "note": (
            "L2 and cosine similarity of 9x9 phoneme-class attention matrices "
            "between the two checkpoints. High cosine_sim = similar routing pattern."
        ),
    }
    _dump(comparison, out_dir / "checkpoint_comparison.json")
    print(f"  Wrote: {out_dir}/checkpoint_comparison.json")


# ── Analysis 2: Per-attack patterns on robust_goat ────────────────────────────

def per_attack_analysis(
    records: list[dict],
    id_to_cls: dict[int, str],
    attack_sids: list[str],
    target_heads: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Analysis 2: Per-attack attention patterns ---")

    # Build aggregates dict: {f"{sid}_h{h}": (C,C)}
    aggregates = {}
    for sid in ["-"] + attack_sids:
        for h in target_heads:
            mat = aggregate_9x9(records, id_to_cls, h, sid)
            aggregates[f"{sid}_h{h}"] = mat

    # Save as npz
    np.savez(str(out_dir / "aggregates.npz"), **{
        k.replace("(", "").replace(")", "").replace(" ", "_").replace("/", "_"): v
        for k, v in aggregates.items()
    })
    print(f"  Wrote: {out_dir}/aggregates.npz")

    # Summary stats CSV
    cls_idx  = {c: i for i, c in enumerate(CLASS_ORDER)}
    csv_rows = []
    for sid in ["-"] + attack_sids:
        for h in target_heads:
            mat  = aggregates[f"{sid}_h{h}"]
            flat = mat.flatten()
            flat_sorted = np.sort(flat)[::-1]
            peak_i    = int(np.argmax(mat))
            csv_rows.append({
                "system_id":    sid,
                "head":         h,
                "top3_sparsity": round(float(flat_sorted[:3].sum()), 4),
                "top5_sparsity": round(float(flat_sorted[:5].sum()), 4),
                "peak_src":     CLASS_ORDER[peak_i // C],
                "peak_tgt":     CLASS_ORDER[peak_i  % C],
                "peak_magnitude": round(float(flat_sorted[0]), 4),
                "attn_variance":  round(float(mat.var()), 6),
                "mean_entropy":   round(float(mean_entropy(records, h, sid)), 4),
            })

    fields = list(csv_rows[0].keys())
    out_csv = out_dir / "summary_stats.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  Wrote: {out_csv}")

    # Simple heatmap figure: (2 heads) × (top-N systems + bonafide)
    _plot_heatmap_grid(aggregates, attack_sids, target_heads, out_dir / "heatmap_grid.png")


def _plot_heatmap_grid(aggregates, attack_sids, target_heads, out_path):
    n_show = min(10, len(attack_sids))
    cols   = ["-"] + attack_sids[:n_show]
    n_rows, n_cols = len(target_heads), len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.2))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("MLAAD GAT layer-0 attention (9×9 phoneme classes)", fontsize=8)

    for ri, h in enumerate(target_heads):
        for ci, sid in enumerate(cols):
            ax  = axes[ri][ci]
            mat = aggregates.get(f"{sid}_h{h}", np.zeros((C, C)))
            ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])
            label = "bonafide" if sid == "-" else sid[:12]
            if ri == 0:
                ax.set_title(label, fontsize=5, rotation=45, ha="right")
            if ci == 0:
                ax.set_ylabel(f"h{h}", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Analysis 3: Cross-language stability ─────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cross_language_stability(
    records_in: list[dict],
    records_xl: list[dict],
    id_to_cls: dict[int, str],
    attack_sids_in:  list[str],
    attack_sids_xl:  list[str],
    target_heads: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Analysis 3: Cross-language attention stability ---")

    common_sids = sorted(set(attack_sids_in) & set(attack_sids_xl))
    print(f"  Systems in both splits: {len(common_sids)}")

    cos_sims: dict = {}
    for h in target_heads:
        cos_sims[f"h{h}"] = {}
        for sid in common_sids:
            mat_in = aggregate_9x9(records_in, id_to_cls, h, sid)
            mat_xl = aggregate_9x9(records_xl, id_to_cls, h, sid)
            cos    = _cosine_sim(mat_in.flatten(), mat_xl.flatten())
            l2     = float(np.sqrt(((mat_in - mat_xl) ** 2).sum()))
            cos_sims[f"h{h}"][sid] = {
                "cosine_sim": round(cos, 4) if not np.isnan(cos) else None,
                "l2_diff":    round(l2, 4),
            }

    # Also compare bonafide (EN) between in_dist and xl (both have EN bonafide)
    for h in target_heads:
        mat_in_bon = aggregate_9x9(records_in, id_to_cls, h, "-")
        mat_xl_bon = aggregate_9x9(records_xl, id_to_cls, h, "-")
        cos = _cosine_sim(mat_in_bon.flatten(), mat_xl_bon.flatten())
        cos_sims[f"h{h}"]["bonafide"] = {
            "cosine_sim": round(cos, 4) if not np.isnan(cos) else None,
            "l2_diff":    round(float(np.sqrt(((mat_in_bon - mat_xl_bon) ** 2).sum())), 4),
            "note": "EN bonafide in both splits",
        }

    _dump(cos_sims, out_dir / "cosine_similarity.json")
    print(f"  Wrote: {out_dir}/cosine_similarity.json")

    # Stability summary
    stability_summary: dict = {"target_heads": target_heads, "n_common_systems": len(common_sids)}
    for h in target_heads:
        vals = [v["cosine_sim"] for v in cos_sims[f"h{h}"].values()
                if v["cosine_sim"] is not None and v.get("note") is None]
        if vals:
            stability_summary[f"h{h}_attack_mean_cosine_sim"] = round(float(np.mean(vals)), 4)
            stability_summary[f"h{h}_attack_min_cosine_sim"]  = round(float(np.min(vals)), 4)
            stability_summary[f"h{h}_attack_max_cosine_sim"]  = round(float(np.max(vals)), 4)
        print(f"  h{h}: mean cos_sim(in-dist vs cross-lang, {len(vals)} systems) "
              f"= {np.mean(vals) if vals else float('nan'):.4f}")

    stability_summary["interpretation"] = (
        "cosine_sim close to 1.0 = attention routing stable across languages; "
        "low cosine_sim = head learns language-specific phoneme patterns"
    )
    _dump(stability_summary, out_dir / "stability_summary.json")
    print(f"  Wrote: {out_dir}/stability_summary.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load target heads ─────────────────────────────────────────────────────
    target_json  = json.loads((HEAD_DISC_DIR / "target_heads.json").read_text())
    target_heads = target_json["target_heads"]   # e.g. [2, 4]
    print(f"\nTarget heads: {target_heads}")

    # ── Load test records ─────────────────────────────────────────────────────
    in_dist_all = json.loads((BASELINE_DIR / "test_in_distribution.json").read_text())
    xl_all      = json.loads((BASELINE_DIR / "test_cross_language.json").read_text())

    # Sample fixed subsets
    in_dist_sampled = sample_records(in_dist_all, N_BONAFIDE, N_PER_ATTACK, SEED)
    xl_sampled      = sample_records(xl_all,      N_BONAFIDE, N_PER_ATTACK, SEED)

    attack_sids_in = sorted(
        {r["attack_system"] for r in in_dist_sampled if r["label"] == "spoof"})
    attack_sids_xl = sorted(
        {r["attack_system"] for r in xl_sampled if r["label"] == "spoof"})

    print(f"\nSampled subsets:")
    in_bon_n  = sum(1 for r in in_dist_sampled if r["label"] == "bonafide")
    in_sp_n   = len(in_dist_sampled) - in_bon_n
    xl_bon_n  = sum(1 for r in xl_sampled if r["label"] == "bonafide")
    xl_sp_n   = len(xl_sampled) - xl_bon_n
    print(f"  in_distribution: {len(in_dist_sampled)} "
          f"({in_bon_n} bonafide + {in_sp_n} spoof, {len(attack_sids_in)} systems)")
    print(f"  cross_language:  {len(xl_sampled)} "
          f"({xl_bon_n} bonafide + {xl_sp_n} spoof, {len(attack_sids_xl)} systems)")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_robust = _best_or_last("mlaad_robust_goat")
    ckpt_goat   = _best_or_last("mlaad_goat")
    for p in (ckpt_robust, ckpt_goat):
        if not p.exists():
            print(f"ERROR: checkpoint not found: {p}")
            sys.exit(1)
    print(f"\nCheckpoints:")
    print(f"  robust: {ckpt_robust.name}")
    print(f"  goat:   {ckpt_goat.name}")

    # ── Build vocab ───────────────────────────────────────────────────────────
    id_to_cls = build_vocab()
    print(f"  Vocab: {len(id_to_cls)} phoneme IDs → 9 classes")

    # ── Build loaders (same records, used by all three analyses) ─────────────
    loader_in = make_loader(in_dist_sampled)
    loader_xl = make_loader(xl_sampled)

    # ── Extract attention — mlaad_robust_goat (in-dist) ──────────────────────
    records_robust_in = extract_attention(ckpt_robust, loader_in, device,
                                          label="mlaad_robust_goat/in_dist")
    # ── Extract attention — mlaad_goat (in-dist, same samples) ───────────────
    records_goat_in   = extract_attention(ckpt_goat, loader_in, device,
                                          label="mlaad_goat/in_dist")
    # ── Extract attention — mlaad_robust_goat (cross-lang) ───────────────────
    records_robust_xl = extract_attention(ckpt_robust, loader_xl, device,
                                          label="mlaad_robust_goat/cross_lang")

    # All systems present in each set
    all_sids_in = sorted({r["system_id"] for r in records_robust_in
                          if not r["is_deg"]})

    # ── Analysis 1: Cross-checkpoint ──────────────────────────────────────────
    cross_checkpoint_comparison(
        records_a  = records_goat_in,
        records_b  = records_robust_in,
        name_a     = "mlaad_goat",
        name_b     = "mlaad_robust_goat",
        id_to_cls  = id_to_cls,
        target_heads = target_heads,
        all_sids   = all_sids_in,
        out_dir    = OUT_BASE / "cross_checkpoint",
    )

    # ── Analysis 2: Per-attack on robust_goat ────────────────────────────────
    per_attack_analysis(
        records      = records_robust_in,
        id_to_cls    = id_to_cls,
        attack_sids  = attack_sids_in,
        target_heads = target_heads,
        out_dir      = OUT_BASE / "per_attack_robust",
    )

    # ── Analysis 3: Cross-language stability ──────────────────────────────────
    cross_language_stability(
        records_in     = records_robust_in,
        records_xl     = records_robust_xl,
        id_to_cls      = id_to_cls,
        attack_sids_in = attack_sids_in,
        attack_sids_xl = attack_sids_xl,
        target_heads   = target_heads,
        out_dir        = OUT_BASE / "cross_language_stability",
    )

    print(f"\n{'='*60}")
    print(f"MLAAD Attention Patterns complete.")
    print(f"Outputs: {OUT_BASE}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
