#!/usr/bin/env python3
"""
gat_l0_compare.py
=================
Per-input cosine similarity between GAT layer 0 activations of goat.ckpt vs robust_goat.ckpt.

Hypothesis: despite near-zero weight cosine sim (-0.0075), functional representations at GAT
layer 0 may converge on similar attack-compressed information (target mean >= 0.8).

Design:
  - SpecAugment bypass: frozen frontend runs ONCE per batch; both models call encoder_and_GAT
    on identical masked_hidden_states. Eliminates masking noise; self-similarity == 1.0 exactly.
  - Per-sample representation: split (total_phonemes, 768) node tensor by reduced_num_frames,
    mean-pool to (768,) for cosine sim. Unpooled per-phoneme tensors also saved.
  - CKA: node-level (total_phonemes, 768) is authoritative (N >> d). Sample-level (N, 768)
    reported as coarse cross-check. Divergence between the two is itself informative.

Outputs (experiments/results/gat_l0_similarity/):
  activations_a.pt / activations_b.pt  — pooled (N,768), nodes (list), labels, system_ids
  per_sample_cos.csv                   — sample_id, cos_sim, label, system_id
  per_class_stats.csv                  — system_id, mean, std, n
  cos_sim_histogram.png
  report.md
"""
from __future__ import annotations

import csv
import io
import os
import sys
import random
import textwrap
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
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "gat_l0_similarity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT_A        = REPO_ROOT / "models" / "goat.ckpt"
CKPT_B        = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # 149
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 30))
SEED           = 42


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode(entry: dict) -> torch.Tensor:
    raw = entry.get("bytes")
    path = entry.get("path")
    arr, sr = (
        sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if raw is not None
        else sf.read(path, dtype="float32", always_2d=False)
    )
    w = torch.tensor(arr)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    elif w.ndim == 2:
        w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = T.Resample(sr, TARGET_SR)(w)
    return w


def _crop(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    if n < TARGET_SAMPLES:
        w = w.repeat(1, -(-TARGET_SAMPLES // n))
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s : s + TARGET_SAMPLES]


def _lbl(raw) -> int:
    if isinstance(raw, str):
        s = raw.strip().lower()
        return 0 if s in ("0", "bonafide", "real", "genuine") else 1
    return int(raw)


# ---------------------------------------------------------------------------
# Balanced dataset: N_PER_CLASS samples per system_id
# ---------------------------------------------------------------------------

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, n_per_class, seed=42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        ds = ds.cast_column("audio", HFAudio(decode=False))

        ex0 = ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )

        by_system: dict[str, list[int]] = defaultdict(list)
        for i in range(len(ds)):
            sid = ds[i].get("system_id", "unknown")
            by_system[sid].append(i)

        rng = random.Random(seed)
        selected: list[int] = []
        for sid, idxs in sorted(by_system.items()):
            rng.shuffle(idxs)
            selected.extend(idxs[:n_per_class])

        rng.shuffle(selected)
        self.ds = ds
        self.indices = selected
        self.label_key = self.label_key

        print(f"\nBalanced dataset: {len(selected)} samples across {len(by_system)} system IDs")
        for sid in sorted(by_system):
            cnt = sum(1 for i in selected if ds[i].get("system_id", "unknown") == sid)
            print(f"  {sid:10s}: {cnt}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        return {
            "audio":     _crop(_decode(ex["audio"])),
            "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
            "system_id": ex.get("system_id", "unknown"),
        }


def collate(batch):
    return {
        "audio":     torch.stack([b["audio"] for b in batch]),
        "label":     torch.stack([b["label"] for b in batch]),
        "system_id": [b["system_id"] for b in batch],
    }


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
            "microsoft/wavlm-base"
            if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h"
        )
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path),
                network_param=network_param,
                optim_param=optim_param,
                tokenizer=None,
                total_num_phonemes=total_num_phonemes,
                weights_only=False,
            ).cpu()
        return BaseModule(
            network_param, optim_param, tokenizer=None,
            total_num_phonemes=total_num_phonemes,
        )

    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Model loading + architecture check
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(
        PhonemeGAT=Namespace(
            backbone="wavlm", use_raw=False, use_GAT=True,
            n_edges=10, use_aug=True, use_pool=True, use_clip=True,
        )
    )
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True
    )
    lit.to(device)
    lit.eval()
    lit.freeze()
    return lit


def check_architecture(lit_a, lit_b):
    sd_a = lit_a.model.state_dict()
    sd_b = lit_b.model.state_dict()
    keys_a, keys_b = set(sd_a), set(sd_b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a or only_b:
        raise AssertionError(
            f"ARCHITECTURE MISMATCH — state dict keys differ.\n"
            f"  Only in goat:        {sorted(only_a)[:10]}\n"
            f"  Only in robust_goat: {sorted(only_b)[:10]}"
        )

    shape_mismatches = [
        (k, tuple(sd_a[k].shape), tuple(sd_b[k].shape))
        for k in keys_a if sd_a[k].shape != sd_b[k].shape
    ]
    if shape_mismatches:
        raise AssertionError(f"ARCHITECTURE MISMATCH — shape differences:\n{shape_mismatches}")

    print(f"  Architecture check PASSED: {len(keys_a)} parameters, all shapes match.")


# ---------------------------------------------------------------------------
# Shared frozen frontend (SpecAugment applied once, shared by both models)
# ---------------------------------------------------------------------------

def run_shared_frontend(
    audio: torch.Tensor,
    gat_model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the deterministic frozen pipeline + one SpecAugment draw.
    Returns (masked_hidden_states, phoneme_ids) to be shared by both models.
    """
    from phoneme_GAT.modules import _mask_hidden_states

    x = audio
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0, :]

    with torch.no_grad():
        feat1 = gat_model.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hidden_states, _ = gat_model.transformer_in_phoneme_model.feature_projection(feat1)
        phoneme_feat = gat_model.transformer_in_phoneme_model.encoder(hidden_states)[0]
        phoneme_logits = gat_model.phoneme_model.model.model.lm_head(phoneme_feat)
        phoneme_ids = torch.argmax(phoneme_logits, dim=-1)
        masked_hs = _mask_hidden_states(hidden_states, gat_model.transformer_in_phoneme_model)

    return masked_hs, phoneme_ids


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def self_similarity_check(gat_model, masked_hs, phoneme_ids, num_f, device):
    """
    Run model A vs itself. All per-sample cosine sims must be 1.0 ± 1e-5.
    Validates that the pipeline is deterministic and hook placement is correct.
    """
    buf = {"nodes": None}

    def _hook(module, inp, out):
        buf["nodes"] = out[0].detach().cpu()

    h = gat_model.GAT.gat_net[0].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            result = gat_model.encoder_and_GAT(masked_hs, num_f, phoneme_ids)
        nodes_1 = buf["nodes"].clone()
        buf["nodes"] = None

        with torch.no_grad():
            result2 = gat_model.encoder_and_GAT(masked_hs, num_f, phoneme_ids)
        nodes_2 = buf["nodes"]
    finally:
        h.remove()

    assert nodes_1 is not None and nodes_2 is not None, "Self-sim: hook did not fire"
    assert nodes_1.shape == nodes_2.shape, f"Self-sim: shape mismatch {nodes_1.shape} vs {nodes_2.shape}"

    rnf = result[3].cpu().tolist()
    splits_1 = torch.split(nodes_1, [int(n) for n in rnf], dim=0)
    splits_2 = torch.split(nodes_2, [int(n) for n in rnf], dim=0)

    cos_sims = [
        F.cosine_similarity(s1.mean(0, keepdim=True), s2.mean(0, keepdim=True)).item()
        for s1, s2 in zip(splits_1, splits_2)
    ]
    arr = np.array(cos_sims)

    if not np.allclose(arr, 1.0, atol=1e-5):
        raise AssertionError(
            f"SELF-SIMILARITY CHECK FAILED: mean={arr.mean():.8f}, min={arr.min():.8f}\n"
            "encoder_and_GAT is not deterministic given fixed inputs."
        )
    print(f"  Self-similarity check PASSED: mean={arr.mean():.10f}, min={arr.min():.10f}")


def permutation_test(
    pooled_a: np.ndarray, pooled_b: np.ndarray, cos_sims: np.ndarray
) -> dict | None:
    """
    If overall cos sim < 0.3, check whether neuron permutation closes the gap.
    If permuted ≈ unpermuted, representations agree up to neuron reordering → CKA is the metric.
    """
    overall = float(np.mean(cos_sims))
    if overall >= 0.3:
        return None

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(pooled_a.shape[1])
    perm_sims = F.cosine_similarity(
        torch.tensor(pooled_a[:, perm], dtype=torch.float32),
        torch.tensor(pooled_b, dtype=torch.float32),
        dim=-1,
    ).numpy()
    perm_mean = float(np.mean(perm_sims))

    conclusion = (
        "neuron_reordering"
        if abs(perm_mean - overall) < 0.05
        else "genuine_orthogonality"
    )
    print(f"\n  PERMUTATION TEST (cos sim < 0.3 triggered):")
    print(f"    original mean = {overall:.4f}  |  permuted mean = {perm_mean:.4f}")
    print(f"    conclusion    = {conclusion}")
    return {"original_mean": overall, "permuted_mean": perm_mean, "conclusion": conclusion}


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA on (N, D) representation matrices.
    CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    Uses the feature-space form (O(N*D^2)), efficient when D < N.
    """
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    XtY = Xc.T @ Yc         # (D, D)
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    numerator   = float(np.linalg.norm(XtY, "fro") ** 2)
    denominator = float(np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro"))
    return numerator / (denominator + 1e-10)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_activations(lit_a, lit_b, loader, device):
    """
    For each batch:
      1. Run shared frozen frontend once (via model A — both share identical frozen weights).
      2. Register hooks on gat_net[0] of both models.
      3. Call encoder_and_GAT on both with the same masked_hidden_states.
      4. Split node tensors by reduced_num_frames → per-sample records.

    Returns records_a, records_b: list of dicts
      {sample_id, pooled (768,), nodes (n_phonemes, 768), label, system_id}
    """
    gat_a = lit_a.model
    gat_b = lit_b.model

    buf = {"a": None, "b": None}

    def _hook_a(module, inp, out):
        buf["a"] = out[0].detach().cpu()

    def _hook_b(module, inp, out):
        buf["b"] = out[0].detach().cpu()

    h_a = gat_a.GAT.gat_net[0].register_forward_hook(_hook_a)
    h_b = gat_b.GAT.gat_net[0].register_forward_hook(_hook_b)

    records_a: list[dict] = []
    records_b: list[dict] = []
    self_sim_done = False
    total = len(loader.dataset)

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio  = batch["audio"].to(device)
                labels = batch["label"].tolist()
                sids   = batch["system_id"]
                B      = len(labels)
                num_f  = torch.full((B,), NF_PER_SAMPLE, device=device)

                masked_hs, phoneme_ids = run_shared_frontend(audio, gat_a, device)

                # Self-similarity check on first batch (before main extraction)
                if not self_sim_done:
                    self_similarity_check(gat_a, masked_hs, phoneme_ids, num_f, device)
                    self_sim_done = True

                result_a = gat_a.encoder_and_GAT(masked_hs, num_f, phoneme_ids)
                result_b = gat_b.encoder_and_GAT(masked_hs, num_f, phoneme_ids)

                nodes_a = buf["a"]
                nodes_b = buf["b"]

                # Sanity checks
                assert nodes_a is not None, f"Hook A did not fire at batch {bi}"
                assert nodes_b is not None, f"Hook B did not fire at batch {bi}"
                assert nodes_a.shape == nodes_b.shape, (
                    f"Shape mismatch at batch {bi}: {nodes_a.shape} vs {nodes_b.shape}"
                )

                rnf_a = result_a[3].cpu()
                rnf_b = result_b[3].cpu()
                assert torch.equal(rnf_a, rnf_b), (
                    f"reduced_num_frames mismatch at batch {bi}:\n  A={rnf_a}\n  B={rnf_b}\n"
                    "Shared input should produce identical phoneme segmentation."
                )

                rnf_list = [int(n) for n in rnf_a.tolist()]
                splits_a = torch.split(nodes_a, rnf_list, dim=0)
                splits_b = torch.split(nodes_b, rnf_list, dim=0)

                base_id = len(records_a)
                for i in range(B):
                    seg_a = splits_a[i].numpy()   # (n_phonemes, 768)
                    seg_b = splits_b[i].numpy()
                    records_a.append({
                        "sample_id": base_id + i,
                        "pooled":    seg_a.mean(0),
                        "nodes":     seg_a,
                        "label":     labels[i],
                        "system_id": sids[i],
                    })
                    records_b.append({
                        "sample_id": base_id + i,
                        "pooled":    seg_b.mean(0),
                        "nodes":     seg_b,
                        "label":     labels[i],
                        "system_id": sids[i],
                    })

                buf["a"] = None
                buf["b"] = None

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    done = min((bi + 1) * BATCH_SIZE, total)
                    print(f"  {done}/{total}")
    finally:
        h_a.remove()
        h_b.remove()

    assert len(records_a) == len(records_b) == total, (
        f"Record count mismatch: got {len(records_a)}, expected {total}"
    )
    return records_a, records_b


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_per_sample_cos(records_a, records_b) -> np.ndarray:
    pooled_a = np.stack([r["pooled"] for r in records_a]).astype(np.float32)
    pooled_b = np.stack([r["pooled"] for r in records_b]).astype(np.float32)
    return F.cosine_similarity(
        torch.tensor(pooled_a), torch.tensor(pooled_b), dim=-1
    ).numpy()


def per_class_stats(cos_sims, records) -> dict[str, dict]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for sim, rec in zip(cos_sims, records):
        by_class[rec["system_id"]].append(float(sim))
    return {
        sid: {
            "mean":   float(np.mean(v)),
            "std":    float(np.std(v)),
            "median": float(np.median(v)),
            "n":      len(v),
            "label":  records[next(i for i, r in enumerate(records) if r["system_id"] == sid)]["label"],
        }
        for sid, v in by_class.items()
    }


def node_matrices(records_a, records_b):
    """Stack all per-phoneme node features for node-level CKA."""
    all_a = np.concatenate([r["nodes"] for r in records_a], axis=0).astype(np.float32)
    all_b = np.concatenate([r["nodes"] for r in records_b], axis=0).astype(np.float32)
    return all_a, all_b


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histogram(cos_sims, records, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    system_ids = sorted(set(r["system_id"] for r in records))
    colors = plt.cm.tab20(np.linspace(0, 1, len(system_ids)))
    color_map = dict(zip(system_ids, colors))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-system overlapping histograms
    ax = axes[0]
    bins = np.linspace(-0.2, 1.0, 40)
    for sid in system_ids:
        mask = np.array([r["system_id"] == sid for r in records])
        vals = cos_sims[mask]
        lbl  = f"{sid} (n={mask.sum()}, μ={vals.mean():.3f})"
        ax.hist(vals, bins=bins, alpha=0.45, color=color_map[sid], label=lbl, density=True)
    ax.axvline(cos_sims.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"overall mean = {cos_sims.mean():.3f}")
    ax.set_xlabel("Cosine similarity (goat vs robust_goat)")
    ax.set_ylabel("Density")
    ax.set_title("GAT Layer 0 Activation Similarity\nPer-System Distribution")
    ax.legend(fontsize=7, loc="upper left")

    # Right: overall histogram + KDE-like
    ax2 = axes[1]
    ax2.hist(cos_sims, bins=bins, color="#4C72B0", alpha=0.7, density=True, label="all samples")
    ax2.axvline(cos_sims.mean(),   color="black",  linestyle="--", linewidth=1.2,
                label=f"mean = {cos_sims.mean():.3f}")
    ax2.axvline(np.median(cos_sims), color="gray", linestyle=":",  linewidth=1.0,
                label=f"median = {np.median(cos_sims):.3f}")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("Overall Distribution")
    ax2.legend(fontsize=9)

    fig.suptitle("goat vs robust_goat — GAT layer 0 activation cosine similarity", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Save activations
# ---------------------------------------------------------------------------

def save_activations(records, path: Path):
    data = {
        "pooled":     np.stack([r["pooled"] for r in records]).astype(np.float32),
        "nodes":      [torch.tensor(r["nodes"], dtype=torch.float32) for r in records],
        "labels":     np.array([r["label"]     for r in records], dtype=np.int32),
        "system_ids": [r["system_id"] for r in records],
        "sample_ids": np.array([r["sample_id"] for r in records], dtype=np.int32),
    }
    torch.save(data, path)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    cos_sims: np.ndarray,
    records: list,
    class_stats: dict,
    cka_node: float,
    cka_sample: float,
    n_nodes: int,
    perm_result: dict | None,
    out_path: Path,
):
    overall_mean = float(np.mean(cos_sims))
    overall_std  = float(np.std(cos_sims))
    overall_med  = float(np.median(cos_sims))
    p5           = float(np.percentile(cos_sims, 5))
    p95          = float(np.percentile(cos_sims, 95))
    n_total      = len(records)

    # Per-class table rows, sorted by mean descending
    rows = sorted(class_stats.items(), key=lambda x: -x[1]["mean"])
    table = "| system_id | label | n | mean | std | median |\n"
    table += "|-----------|-------|---|------|-----|--------|\n"
    for sid, s in rows:
        lbl = "bonafide" if s["label"] == 0 else "spoof"
        table += f"| {sid} | {lbl} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['median']:.4f} |\n"

    # Top/bottom 5
    top5_idx    = np.argsort(cos_sims)[-5:][::-1]
    bottom5_idx = np.argsort(cos_sims)[:5]

    def _fmt_samples(idxs):
        lines = []
        for i in idxs:
            lines.append(f"  - sample {int(i)}: cos={cos_sims[int(i)]:.4f}  "
                         f"label={'bonafide' if records[int(i)]['label']==0 else 'spoof'}  "
                         f"system={records[int(i)]['system_id']}")
        return "\n".join(lines)

    # Interpretation
    if overall_mean >= 0.8:
        interp = "Hypothesis SUPPORTED: mean >= 0.8. Models converge on functionally similar GAT layer 0 representations despite orthogonal weight spaces."
    elif overall_mean >= 0.5:
        interp = "Partial agreement: mean is moderate (0.5–0.8). Models share coarse representational geometry but differ in fine structure."
    elif overall_mean >= 0.3:
        interp = "Weak agreement: mean is low (0.3–0.5). Representations share limited functional overlap."
    else:
        interp = "Hypothesis REJECTED: mean < 0.3. Orthogonal weights produce orthogonal representations. See permutation test for neuron-reordering analysis."

    cka_note = ""
    if abs(cka_node - cka_sample) > 0.15:
        cka_note = (
            "\n\n**CKA divergence note**: node-level and sample-level CKA differ by "
            f"{abs(cka_node - cka_sample):.3f}. This suggests attack-relevant structure "
            "is localized to specific phonemes rather than spread uniformly across the utterance."
        )

    perm_section = ""
    if perm_result is not None:
        perm_section = textwrap.dedent(f"""
        ## Permutation test (triggered: overall cos sim < 0.3)

        | | mean cos sim |
        |---|---|
        | original     | {perm_result['original_mean']:.4f} |
        | feature-dim permuted | {perm_result['permuted_mean']:.4f} |

        **Conclusion**: {perm_result['conclusion']}
        """)

    report = textwrap.dedent(f"""
    # GAT Layer 0 Activation Similarity: goat vs robust_goat

    **Question**: Do the two checkpoints compute functionally similar representations at GAT layer 0,
    despite orthogonal weight spaces (weight cosine sim = -0.0075)?

    **Setup**: {n_total} samples (balanced by system_id, {N_PER_CLASS} per class).
    SpecAugment shared across both models — masking noise eliminated.
    Layer: `lit.model.GAT.gat_net[0]` output, mean-pooled over phonemes to (768,) per sample.

    ---

    ## Overall cosine similarity

    | metric | value |
    |--------|-------|
    | mean   | {overall_mean:.4f} |
    | std    | {overall_std:.4f} |
    | median | {overall_med:.4f} |
    | p5     | {p5:.4f} |
    | p95    | {p95:.4f} |

    **{interp}**

    ---

    ## CKA (Centered Kernel Alignment)

    | scope | N | CKA |
    |-------|---|-----|
    | node-level (authoritative) | {n_nodes} phonemes | {cka_node:.4f} |
    | sample-level (cross-check) | {n_total} samples | {cka_sample:.4f} |
    {cka_note}

    ---

    ## Per-system breakdown

    {table}
    ---

    ## Extreme samples

    **Top 5 (highest similarity)**:
    {_fmt_samples(top5_idx)}

    **Bottom 5 (lowest similarity)**:
    {_fmt_samples(bottom5_idx)}

    {perm_section}
    ---

    *Generated by experiments/gat_l0_compare.py*
    """).strip()

    out_path.write_text(report)
    print(f"Saved: {out_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *a, **kw: None
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = BalancedDataset(HF_DATASET, "validation", str(CACHE_DIR), hf_token, N_PER_CLASS, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate,
    )

    # Models
    patch_phoneme_loader()
    print(f"\nLoading goat:        {CKPT_A}")
    lit_a = load_model(CKPT_A, device)
    print(f"Loading robust_goat: {CKPT_B}")
    lit_b = load_model(CKPT_B, device)

    print("\n--- Architecture check ---")
    check_architecture(lit_a, lit_b)

    # Extraction
    print("\n--- Extracting GAT layer 0 activations ---")
    records_a, records_b = extract_activations(lit_a, lit_b, loader, device)

    # Save activations
    save_activations(records_a, RESULTS_DIR / "activations_a.pt")
    save_activations(records_b, RESULTS_DIR / "activations_b.pt")

    # Cosine similarity
    print("\n--- Computing cosine similarity ---")
    cos_sims = compute_per_sample_cos(records_a, records_b)
    class_stats = per_class_stats(cos_sims, records_a)

    print(f"\n  Overall: mean={cos_sims.mean():.4f}  std={cos_sims.std():.4f}  "
          f"median={np.median(cos_sims):.4f}  p5={np.percentile(cos_sims,5):.4f}  "
          f"p95={np.percentile(cos_sims,95):.4f}")
    print(f"\n  {'system_id':12s}  {'label':8s}  {'n':>4s}  {'mean':>7s}  {'std':>7s}")
    print(f"  {'-'*50}")
    for sid, s in sorted(class_stats.items(), key=lambda x: -x[1]["mean"]):
        lbl = "bonafide" if s["label"] == 0 else "spoof"
        print(f"  {sid:12s}  {lbl:8s}  {s['n']:4d}  {s['mean']:7.4f}  {s['std']:7.4f}")

    # Permutation test (only if cos sim < 0.3)
    pooled_a = np.stack([r["pooled"] for r in records_a]).astype(np.float32)
    pooled_b = np.stack([r["pooled"] for r in records_b]).astype(np.float32)
    perm_result = permutation_test(pooled_a, pooled_b, cos_sims)

    # CKA
    print("\n--- Computing CKA ---")
    all_nodes_a, all_nodes_b = node_matrices(records_a, records_b)
    print(f"  Node-level matrix shapes: A={all_nodes_a.shape}  B={all_nodes_b.shape}")
    cka_node   = linear_cka(all_nodes_a, all_nodes_b)
    cka_sample = linear_cka(pooled_a, pooled_b)
    print(f"  CKA node-level   (authoritative): {cka_node:.4f}")
    print(f"  CKA sample-level (cross-check):   {cka_sample:.4f}")
    if abs(cka_node - cka_sample) > 0.15:
        print("  NOTE: >0.15 divergence — attack structure may be phoneme-localized.")

    # Save CSVs
    csv_path = RESULTS_DIR / "per_sample_cos.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "cos_sim", "label", "system_id"])
        w.writeheader()
        for i, (rec, sim) in enumerate(zip(records_a, cos_sims)):
            w.writerow({"sample_id": rec["sample_id"], "cos_sim": f"{sim:.6f}",
                        "label": rec["label"], "system_id": rec["system_id"]})
    print(f"\nSaved: {csv_path}")

    stats_path = RESULTS_DIR / "per_class_stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system_id", "label", "n", "mean", "std", "median"])
        w.writeheader()
        for sid, s in sorted(class_stats.items()):
            w.writerow({"system_id": sid, "label": s["label"], "n": s["n"],
                        "mean": f"{s['mean']:.6f}", "std": f"{s['std']:.6f}",
                        "median": f"{s['median']:.6f}"})
    print(f"Saved: {stats_path}")

    # Histogram
    plot_histogram(cos_sims, records_a, RESULTS_DIR / "cos_sim_histogram.png")

    # Report
    write_report(
        cos_sims, records_a, class_stats,
        cka_node, cka_sample, all_nodes_a.shape[0],
        perm_result, RESULTS_DIR / "report.md",
    )

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
