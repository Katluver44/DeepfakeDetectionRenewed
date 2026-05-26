#!/usr/bin/env python3
"""
per_system_head_ablation.py
============================
Per-attack-system breakdown of the GAT layer-0 head ablation experiment.

Matches head_ablation.py exactly:
  - checkpoint, dataset, seed, batch size, NF_PER_SAMPLE all copied verbatim
  - ablation mechanism: uniform mode on neighborhood_aware_softmax (post-softmax
    replace with 1/in-degree per target node) at GAT layer 0 only
  - control set: {1,2,3,5}  (ctrl_h1235 from head_ablation.py)

Four conditions per attack system S:
  clean           — no ablation
  suspect         — uniform-ablate {h0, h4}
  control         — uniform-ablate {h1, h2, h3, h5}
  all_heads       — uniform-ablate {h0..h5}

Bonafide (n=50) is always the negative class for per-system EER.

Outputs (same directory as this script):
  per_system_ablation.csv
  per_system_ablation_NOTES.md
  (summary table also printed to stdout)
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

# ── torch.load compat (mirrored from head_ablation.py) ───────────────────────
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
# Script lives at experiments/results/gat_l0_attention_followups/
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]          # …/results/…/  → experiments/ → repo root
RESULTS_DIR = SCRIPT_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT          = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # 149  (same as head_ablation.py)
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
SEED           = 42

SUSPECT_HEADS  = frozenset({0, 4})
CONTROL_HEADS  = frozenset({1, 2, 3, 5})
ALL_HEADS      = frozenset(range(6))

CONDITIONS = [
    ("clean",     frozenset(),     "uniform"),
    ("suspect",   SUSPECT_HEADS,   "uniform"),
    ("control",   CONTROL_HEADS,   "uniform"),
    ("all_heads", ALL_HEADS,       "uniform"),
]

ATTACK_SYSTEMS = ["A01", "A02", "A03", "A04", "A05", "A06"]

# ── Audio helpers (identical to head_ablation.py) ─────────────────────────────

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


# ── Balanced dataset (identical to head_ablation.py) ─────────────────────────

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


# ── Phoneme loader patch (identical to head_ablation.py) ──────────────────────

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


# ── Model loading (identical to head_ablation.py) ─────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit


# ── Frozen frontend (identical to head_ablation.py) ───────────────────────────

def run_frozen_frontend(audio: torch.Tensor, gat_model, device: torch.device):
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


# ── Ablation hook (identical to head_ablation.py) ─────────────────────────────

def install_ablation(layer0, heads: frozenset, mode: str):
    orig_nas = layer0.neighborhood_aware_softmax

    def _ablated_nas(scores_per_edge, trg_index, num_of_nodes):
        attn = orig_nas(scores_per_edge, trg_index, num_of_nodes)   # (E, NH, 1)
        if not heads:
            return attn
        attn = attn.clone()
        for h in sorted(heads):
            if mode == "uniform":
                degree = torch.zeros(num_of_nodes, dtype=attn.dtype, device=attn.device)
                ones   = torch.ones(len(trg_index), dtype=attn.dtype, device=attn.device)
                degree.scatter_add_(0, trg_index, ones)
                uniform_w = 1.0 / degree[trg_index].clamp(min=1.0)
                attn[:, h, 0] = uniform_w
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
        return attn

    layer0.neighborhood_aware_softmax = _ablated_nas

    def remove():
        if "neighborhood_aware_softmax" in layer0.__dict__:
            del layer0.__dict__["neighborhood_aware_softmax"]

    return remove


def sanity_layer_untouched(lit) -> None:
    gat_net = lit.model.GAT.gat_net
    for idx in (1, 2):
        if "neighborhood_aware_softmax" in gat_net[idx].__dict__:
            raise AssertionError(
                f"SANITY FAILED — layer {idx} has an instance-level override. "
                "Ablation leaked to wrong layer."
            )


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(lit, loader, device, heads: frozenset, mode: str) -> list[dict]:
    """Returns per-sample dicts: {sample_id, label, system_id, logit, pred}."""
    gat_model = lit.model
    layer0    = gat_model.GAT.gat_net[0]
    remove    = install_ablation(layer0, heads, mode)

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


# ── EER ───────────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
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


# ── Per-system metrics ────────────────────────────────────────────────────────

def per_system_metrics(
    clean_recs: list[dict],
    cond_recs:  list[dict],
    sid: str,
) -> dict:
    """
    Compute metrics for one (system, condition) cell.
    EER is computed on sid samples + bonafide samples only (n ≈ 100).
    Δlogit is relative to clean_recs.
    """
    # Index clean logits by sample_id for delta computation
    clean_by_id = {r["sample_id"]: r["logit"] for r in clean_recs}

    # Filter to sid + bonafide
    subset = [r for r in cond_recs if r["system_id"] in (sid, "-")]
    clean_sub = [r for r in clean_recs if r["system_id"] in (sid, "-")]

    if not subset:
        nan = float("nan")
        return dict(eer=nan,
                    mean_logit_spoof=nan, mean_logit_bonafide=nan,
                    delta_logit_spoof_mean=nan, delta_logit_spoof_std=nan,
                    delta_logit_bonafide_mean=nan, delta_logit_bonafide_std=nan,
                    n_spoof=0, n_bonafide=0)

    labels  = np.array([r["label"] for r in subset])
    logits  = np.array([r["logit"] for r in subset])
    scores  = 1.0 / (1.0 + np.exp(-logits))

    spoof_logits = logits[labels == 1]
    bon_logits   = logits[labels == 0]

    eer = compute_eer(labels, scores)

    # Δlogit vs clean
    delta_spoof = np.array([
        r["logit"] - clean_by_id[r["sample_id"]]
        for r in cond_recs if r["system_id"] == sid
    ])
    delta_bon = np.array([
        r["logit"] - clean_by_id[r["sample_id"]]
        for r in cond_recs if r["system_id"] == "-"
    ])

    return dict(
        eer=float(eer),
        mean_logit_spoof=float(spoof_logits.mean()) if len(spoof_logits) else float("nan"),
        mean_logit_bonafide=float(bon_logits.mean()) if len(bon_logits) else float("nan"),
        delta_logit_spoof_mean=float(delta_spoof.mean()) if len(delta_spoof) else float("nan"),
        delta_logit_spoof_std=float(delta_spoof.std())  if len(delta_spoof) else float("nan"),
        delta_logit_bonafide_mean=float(delta_bon.mean()) if len(delta_bon) else float("nan"),
        delta_logit_bonafide_std=float(delta_bon.std())  if len(delta_bon) else float("nan"),
        n_spoof=int((labels == 1).sum()),
        n_bonafide=int((labels == 0).sum()),
    )


# ── CSV output ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "system", "condition",
    "eer", "mean_logit_spoof", "mean_logit_bonafide",
    "delta_logit_spoof_mean", "delta_logit_spoof_std",
    "delta_logit_bonafide_mean", "delta_logit_bonafide_std",
    "n_spoof", "n_bonafide",
]

def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.5f}" if isinstance(row[k], float) else row[k])
                        for k in CSV_FIELDS})
    print(f"Saved: {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(table: dict[str, dict[str, dict]]) -> None:
    """
    table[system][condition] = metrics dict.
    Prints: system | clean | suspect | control | all_heads | susp-ctrl | susp-all
    """
    cnames = ["clean", "suspect", "control", "all_heads"]
    header = (f"{'System':8s} | {'clean':7s} | {'suspect':7s} | "
              f"{'control':7s} | {'all_hds':7s} | {'s-c':7s} | {'s-a':7s}")
    sep    = "-" * len(header)
    print("\n" + sep)
    print("Per-system EER by ablation condition  (s-c = suspect−control,  s-a = suspect−all_heads)")
    print(sep)
    print(header)
    print(sep)
    for sid in ATTACK_SYSTEMS:
        eers = {c: table[sid][c]["eer"] for c in cnames}
        sc   = eers["suspect"] - eers["control"]
        sa   = eers["suspect"] - eers["all_heads"]
        print(f"{sid:8s} | {eers['clean']:7.4f} | {eers['suspect']:7.4f} | "
              f"{eers['control']:7.4f} | {eers['all_heads']:7.4f} | "
              f"{sc:+7.4f} | {sa:+7.4f}")
    print(sep + "\n")


# ── Notes markdown ────────────────────────────────────────────────────────────

def write_notes(path: Path) -> None:
    import datetime
    lines = [
        "# Per-System Head Ablation — Run Metadata",
        "",
        f"- **Date**: {datetime.date.today().isoformat()}",
        f"- **Checkpoint**: `{CKPT.relative_to(REPO_ROOT)}`",
        f"- **Seed**: {SEED}",
        f"- **N per class**: {N_PER_CLASS}",
        f"- **Dataset split**: validation  (`{HF_DATASET}`)",
        f"- **Suspect heads**: {sorted(SUSPECT_HEADS)}",
        f"- **Control heads**: {sorted(CONTROL_HEADS)}  (ctrl_h1235 from head_ablation.py)",
        "- **Ablation mechanism**: post-softmax uniform replacement — for each ablated head h, "
        "sets `attn[:, h, 0] = 1 / in_degree(target_node)` for every edge, applied to "
        "`gat_net[0].neighborhood_aware_softmax` only; layers 1 and 2 are untouched; "
        "skip_proj at layer 0 is untouched.",
        "- **Conditions**: clean (no ablation), suspect ({h0,h4} uniform), "
        "control ({h1,h2,h3,h5} uniform), all_heads ({h0–h5} uniform)",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"REPO_ROOT: {REPO_ROOT}")

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    patch_phoneme_loader()
    print(f"Loading model: {CKPT}")
    lit = load_model(CKPT, device)
    print("  model loaded")

    print(f"Loading dataset (N_PER_CLASS={N_PER_CLASS}, SEED={SEED})...")
    dataset = BalancedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                              hf_token, N_PER_CLASS, SEED)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, collate_fn=collate,
                                          num_workers=0)

    # ── Run all four conditions ───────────────────────────────────────────────
    all_records: dict[str, list[dict]] = {}
    for cname, heads, mode in CONDITIONS:
        print(f"\n--- Condition: {cname}  heads={sorted(heads)}  mode={mode} ---")
        recs = run_eval(lit, loader, device, heads, mode)
        all_records[cname] = recs
        # Quick aggregate check
        logits_spoof = np.array([r["logit"] for r in recs if r["label"] == 1])
        logits_bon   = np.array([r["logit"] for r in recs if r["label"] == 0])
        print(f"  mean_logit_spoof={logits_spoof.mean():.4f}  "
              f"mean_logit_bon={logits_bon.mean():.4f}")

    # ── Verify sample alignment across conditions ─────────────────────────────
    ref_sids = [r["system_id"] for r in all_records["clean"]]
    for cname in ["suspect", "control", "all_heads"]:
        test_sids = [r["system_id"] for r in all_records[cname]]
        if test_sids != ref_sids:
            raise AssertionError(
                f"Sample order mismatch between clean and {cname} — "
                "check dataset determinism."
            )
    print("\n  Sample alignment check PASSED")

    # ── Compute per-system metrics ────────────────────────────────────────────
    table: dict[str, dict[str, dict]] = {}
    csv_rows: list[dict] = []

    for sid in ATTACK_SYSTEMS:
        table[sid] = {}
        for cname, _, _ in CONDITIONS:
            m = per_system_metrics(all_records["clean"], all_records[cname], sid)
            table[sid][cname] = m
            csv_rows.append({"system": sid, "condition": cname, **m})

    # ── Outputs ───────────────────────────────────────────────────────────────
    write_csv(csv_rows, RESULTS_DIR / "per_system_ablation.csv")
    write_notes(RESULTS_DIR / "per_system_ablation_NOTES.md")
    print_summary(table)


if __name__ == "__main__":
    main()
