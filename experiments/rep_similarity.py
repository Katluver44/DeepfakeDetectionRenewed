#!/usr/bin/env python3
"""
rep_similarity.py
=================
Inter-model interpretability comparison between two Phoneme_GAT checkpoints:
  - goat.ckpt        (baseline model)
  - robust_goat.ckpt (robustly trained model)

Question: Do robust training methods reshape internal feature geometry,
          or only adjust decision boundaries?

Hook insertion points:
  pre_encoder  — feature_projection output (frozen CNN → 768-d; sanity check, expect ~1.0)
  encoder_feat — trainable WavLM encoder output (first divergence point; B, T, 768)
  gat_pre      — phoneme-pooled encoder output, input to GAT (total_phonemes, 768)
  gat_l{0,1,2} — output of each of the 3 GATLayer modules (total_phonemes, 768)
  bilstm       — post-BiLSTM mean-pooled + L2-norm representation (B, 768)

Tensors compared:
  Frame-level (pre_encoder, encoder_feat): mean-pooled over T → (B, 768) → per-sample cosine sim
  Node-level  (gat_pre, gat_l*):           split by reduced_num_frames, mean per sample → (D,)
  Sample-level (bilstm):                   already (B, 768)

Also computes weight-space cosine similarity per parameter group (encoder, GAT, rnn, cls_head).

Outputs (experiments/results/rep_similarity/):
  per_sample_cosine.npz       — per-sample cosine sims for every layer (N, L)
  per_sample_cosine.csv       — human-readable per-row: sample_id, layer sims, label, system_id
  layer_stats.csv             — mean / std / median / p5 / p95 per layer
  weight_cosine.csv           — weight-space cosine sims per parameter group
  similarity_distribution.png — violin plot of per-sample sims by layer
  weight_vs_activation.png    — scatter: weight sim vs activation sim per layer group
"""
from __future__ import annotations

import csv
import io
import os
import sys
import types
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# ── torch.load compat (required for Namespace / tokenizer objects in ckpts) ──
_orig_torch_load = torch.load
def _patch(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patch

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rep_similarity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT_A        = REPO_ROOT / "models" / "goat.ckpt"
CKPT_B        = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR          # 48,000 samples = 3 seconds
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1   # WavLM frame count for 3-second clips
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
MAX_SAMPLES    = int(os.environ.get("N_SAMPLES", 500))
SEED           = 42
N_GAT_LAYERS   = 3

LAYER_ORDER = [
    "pre_encoder",
    "encoder_feat",
    "gat_pre",
    "gat_l0",
    "gat_l1",
    "gat_l2",
    "bilstm",
]


# ---------------------------------------------------------------------------
# Audio helpers (identical to linear_probe.py)
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
# Dataset
# ---------------------------------------------------------------------------

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, max_samples, seed=42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )
        rng = random.Random(seed)
        indices = list(range(len(self.ds)))
        rng.shuffle(indices)
        self.indices = indices[:max_samples]
        self.sys_ids = [self.ds[i].get("system_id", "unknown") for i in self.indices]
        print(f"Dataset: {len(self.indices)} samples loaded")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        return {
            "audio":     _crop(_decode(ex["audio"])),
            "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
            "system_id": self.sys_ids[idx],
        }


def collate(batch):
    return {
        "audio":     torch.stack([b["audio"] for b in batch]),
        "label":     torch.stack([b["label"] for b in batch]),
        "system_id": [b["system_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Phoneme-loader patch (identical to linear_probe.py)
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
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    """Load a Phoneme_GAT_lit checkpoint, freeze in eval mode."""
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


# ---------------------------------------------------------------------------
# Activation capture harness
# ---------------------------------------------------------------------------

class ActivationCapture:
    """
    Installs hooks on a Phoneme_GAT model to capture intermediate representations.

    Hook placement:
      pre_encoder  — forward hook on transformer_in_phoneme_model.feature_projection
                     (frozen CNN → proj; fires once per batch during the no_grad block)
      gat_pre      — forward pre-hook on GAT module
                     (captures node features just before the first GAT layer)
      gat_l{i}     — forward hook on each GAT.gat_net[i] layer
                     (captures post-attention node features after each layer)
      encoder_feat — monkey-patched encoder_and_GAT return[4] (B, T, 768)
      bilstm       — monkey-patched encoder_and_GAT return[0] (B, 768)
      _rnf         — monkey-patched encoder_and_GAT return[3] (B,) phoneme counts
                     (used to split flat node-level tensors into per-sample slices)
    """

    def __init__(self, gat_model):
        self._buf: dict[str, torch.Tensor | None] = {
            "pre_encoder":  None,
            "encoder_feat": None,
            "gat_pre":      None,
            **{f"gat_l{i}": None for i in range(N_GAT_LAYERS)},
            "bilstm":       None,
            "_rnf":         None,
        }
        self.store: dict[str, list] = defaultdict(list)
        self._hooks: list = []
        capture = self

        # 1. Frozen feature_projection → pre_encoder
        #    output is (hidden_states, extract_features); we take [0] → (B, T, 768)
        def _fp_hook(module, inp, out):
            capture._buf["pre_encoder"] = out[0].detach().cpu()

        self._hooks.append(
            gat_model.transformer_in_phoneme_model.feature_projection
            .register_forward_hook(_fp_hook)
        )

        # 2. GAT forward pre-hook → gat_pre
        #    inp[0] = (node_features, edge_index); node_features: (total_phonemes, 768)
        def _gat_pre_hook(module, inp):
            capture._buf["gat_pre"] = inp[0][0].detach().cpu()

        self._hooks.append(
            gat_model.GAT.register_forward_pre_hook(_gat_pre_hook)
        )

        # 3. Each GATLayer output → gat_l{i}
        #    out = (node_features, edge_index); node_features: (total_phonemes, D)
        for i, layer in enumerate(gat_model.GAT.gat_net):
            def _make_gat_hook(idx):
                def _h(module, inp, out):
                    capture._buf[f"gat_l{idx}"] = out[0].detach().cpu()
                return _h
            self._hooks.append(layer.register_forward_hook(_make_gat_hook(i)))

        # 4. Monkey-patch encoder_and_GAT to capture encoder_feat, bilstm, _rnf
        orig_fn = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig_fn(
                self_inner, hidden_states, num_frames, phoneme_ids,
                profiler=profiler, use_encoder=use_encoder,
                ground_truth_labels=ground_truth_labels,
            )
            # result: (final_hs (B,768), reduced_hs, reduced_pids, reduced_nf (B,), encoder_feat (B,T,768), logit)
            capture._buf["bilstm"]       = result[0].detach().cpu()
            capture._buf["encoder_feat"] = result[4].detach().cpu()
            capture._buf["_rnf"]         = result[3].detach().cpu()
            return result

        gat_model.encoder_and_GAT = types.MethodType(_patched, gat_model)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def collect_batch(self, labels: list[int], sids: list[str]) -> None:
        """
        Mean-pool captured tensors to per-sample vectors and append to self.store.

        Frame-level (pre_encoder, encoder_feat): mean over T axis.
        Node-level  (gat_pre, gat_l*):           split by _rnf, mean per sample.
        Sample-level (bilstm):                   already per-sample.
        """
        rnf = self._buf["_rnf"]
        if rnf is None:
            return
        B = len(labels)

        # sample-level
        if (v := self._buf["bilstm"]) is not None:
            for i in range(B):
                self.store["bilstm"].append((v[i].numpy(), labels[i], sids[i]))

        # frame-level
        for key in ("pre_encoder", "encoder_feat"):
            if (v := self._buf[key]) is not None:
                for i in range(B):
                    self.store[key].append((v[i].mean(0).numpy(), labels[i], sids[i]))

        # node-level — split flat tensor by per-sample phoneme counts
        cursor = 0
        for i in range(B):
            n = int(rnf[i].item())
            for key in ["gat_pre"] + [f"gat_l{li}" for li in range(N_GAT_LAYERS)]:
                if (v := self._buf[key]) is not None:
                    seg = v[cursor : cursor + n]        # (n, D)
                    self.store[key].append((seg.mean(0).numpy(), labels[i], sids[i]))
            cursor += n

        # clear buffers
        for k in self._buf:
            self._buf[k] = None


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    lit_model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    name: str = "model",
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Run one model over the full loader and collect per-sample layer representations.

    Returns:
        arrays  — dict: layer_name → float32 ndarray of shape (N, D)
        meta    — dict: "labels" (N,), "system_ids" (N,)
    """
    gat_model = lit_model.model
    capture   = ActivationCapture(gat_model)
    nf        = NF_PER_SAMPLE
    total     = len(loader.dataset)

    print(f"  Extracting activations [{name}]...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            audio  = batch["audio"].to(device)
            labels = batch["label"].tolist()
            sids   = batch["system_id"]
            B      = len(labels)
            num_f  = torch.full((B,), nf, device=device)

            gat_model(audio, num_f, profiler=None, use_aug=False, stage="eval")
            capture.collect_batch(labels, sids)

            if (bi + 1) % 20 == 0 or (bi + 1) == len(loader):
                print(f"    {min((bi + 1) * BATCH_SIZE, total)}/{total} samples")

    capture.remove_hooks()

    arrays: dict[str, np.ndarray] = {}
    for key, records in capture.store.items():
        arrays[key] = np.stack([r[0] for r in records]).astype(np.float32)

    meta = {
        "labels":     np.array([r[1] for r in capture.store["bilstm"]], dtype=np.int32),
        "system_ids": np.array([r[2] for r in capture.store["bilstm"]]),
    }
    return arrays, meta


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def compute_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between (N, D) arrays.  Returns (N,) in [-1, 1]."""
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    return F.cosine_similarity(a_t, b_t, dim=-1).numpy()


def layer_stats(cos: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(cos)),
        "std":    float(np.std(cos)),
        "median": float(np.median(cos)),
        "p5":     float(np.percentile(cos, 5)),
        "p95":    float(np.percentile(cos, 95)),
    }


# ---------------------------------------------------------------------------
# Weight-space similarity
# ---------------------------------------------------------------------------

def compute_weight_similarity(
    model_a, model_b
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Flatten and compare each parameter tensor between model_a and model_b.

    Parameter groups (by name prefix):
      encoder     — trainable WavLM encoder layers
      GAT         — graph attention layers
      rnn         — BiLSTM
      cls_head    — classification head
      phoneme_model — frozen backbone (expect ~1.0)

    Returns:
        per_param   — {param_name: cosine_sim}
        group_means — {group: mean_cosine_sim}
    """
    sd_a = {k: v.detach().cpu().float() for k, v in model_a.model.state_dict().items()}
    sd_b = {k: v.detach().cpu().float() for k, v in model_b.model.state_dict().items()}

    def _group(name: str) -> str:
        if name.startswith("encoder."):       return "encoder"
        if name.startswith("GAT."):           return "GAT"
        if name.startswith("rnn."):           return "rnn"
        if name.startswith("cls_head."):      return "cls_head"
        if name.startswith("phoneme_model."): return "phoneme_model"
        return "other"

    group_sims: dict[str, list[float]] = defaultdict(list)
    per_param:  dict[str, float]       = {}

    for key in sd_a:
        if key not in sd_b:
            continue
        a = sd_a[key].flatten()
        b = sd_b[key].flatten()
        if a.numel() < 2:
            continue
        sim = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        per_param[key] = sim
        group_sims[_group(key)].append(sim)

    group_means = {g: float(np.mean(v)) for g, v in group_sims.items()}
    return per_param, group_means


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_distributions(
    per_layer_cosines: dict[str, np.ndarray], out_path: Path
) -> None:
    """Violin plot of per-sample cosine sims across all captured layers."""
    plt = _get_mpl()
    layers = [l for l in LAYER_ORDER if l in per_layer_cosines]
    data   = [per_layer_cosines[l] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: violin
    ax = axes[0]
    parts = ax.violinplot(data, positions=range(len(layers)), showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.65)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("goat vs robust_goat\nPer-Sample Activation Similarity by Layer")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.7, label="Perfect (1.0)")
    ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=0.7, alpha=0.5, label="Orthogonal (0.0)")
    ax.legend(fontsize=8)

    # Right: mean sim across layers (trend line)
    ax2 = axes[1]
    means = [float(np.mean(per_layer_cosines[l])) for l in layers]
    ax2.plot(range(len(layers)), means, "o-", color="#4C72B0", linewidth=1.8, markersize=7)
    for i, (l, m) in enumerate(zip(layers, means)):
        ax2.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                     xytext=(5, 3), fontsize=8)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Cosine Similarity")
    ax2.set_title("Mean Similarity by Depth")
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_weight_vs_activation(
    group_means: dict[str, float],
    layer_means: dict[str, float],
    out_path: Path,
) -> None:
    """Scatter: weight-space sim vs activation sim for matched layer groups."""
    plt = _get_mpl()

    # Map each parameter group to its best matching activation layer
    group_to_act = {
        "encoder":     "encoder_feat",
        "GAT":         "gat_l2",
        "rnn":         "bilstm",
        "cls_head":    "bilstm",
        "phoneme_model": "pre_encoder",
    }

    xs, ys, labels = [], [], []
    for group, act_key in group_to_act.items():
        if act_key in layer_means and group in group_means:
            xs.append(group_means[group])
            ys.append(layer_means[act_key])
            labels.append(group)

    if not xs:
        print("  (no data for weight vs activation scatter)")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(range(len(xs)))
    for x, y, lbl, c in zip(xs, ys, labels, colors):
        ax.scatter(x, y, s=100, color=c, zorder=5, label=lbl)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=9)

    ax.set_xlabel("Weight-space cosine similarity (goat vs robust_goat)")
    ax.set_ylabel("Activation cosine similarity")
    ax.set_title("Weight Similarity vs Activation Similarity\n(goat vs robust_goat)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="identity line")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    # ── Device ────────────────────────────────────────────────────────────────
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except RuntimeError:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SimpleDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token, MAX_SAMPLES, SEED
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate,
    )

    # ── Load models ───────────────────────────────────────────────────────────
    patch_phoneme_loader()

    print(f"\nLoading goat:        {CKPT_A}")
    model_a = load_model(CKPT_A, device)
    print(f"Loading robust_goat: {CKPT_B}")
    model_b = load_model(CKPT_B, device)

    # ── Weight-space similarity (data-independent) ────────────────────────────
    print("\n--- Weight-space similarity ---")
    per_param_sims, group_means = compute_weight_similarity(model_a, model_b)
    print(f"  {'Group':20s}  {'Cosine Sim':>10s}")
    print(f"  {'-'*20}  {'-'*10}")
    for g, s in sorted(group_means.items(), key=lambda x: -x[1]):
        print(f"  {g:20s}  {s:10.4f}")

    with open(RESULTS_DIR / "weight_cosine.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "weight_cos_sim"])
        w.writeheader()
        w.writerows(
            [{"group": g, "weight_cos_sim": f"{s:.6f}"}
             for g, s in sorted(group_means.items())]
        )

    # ── Activation extraction ─────────────────────────────────────────────────
    print()
    arrays_a, meta = extract_activations(model_a, loader, device, name="goat")
    print()
    arrays_b, _    = extract_activations(model_b, loader, device, name="robust_goat")

    # ── Per-sample cosine similarity ──────────────────────────────────────────
    print("\n--- Activation cosine similarity (goat vs robust_goat) ---")
    print(f"  {'Layer':15s}  {'Mean':>7s}  {'Std':>7s}  {'Median':>7s}  "
          f"{'p5':>7s}  {'p95':>7s}")
    print(f"  {'-'*70}")

    per_layer_cosines: dict[str, np.ndarray] = {}
    stats_rows: list[dict] = []

    for layer in LAYER_ORDER:
        if layer not in arrays_a or layer not in arrays_b:
            print(f"  {layer:15s}  (not captured)")
            continue
        a, b = arrays_a[layer], arrays_b[layer]
        if a.shape != b.shape:
            print(f"  {layer:15s}  shape mismatch: {a.shape} vs {b.shape}")
            continue
        cos = compute_cosine(a, b)
        per_layer_cosines[layer] = cos
        s = layer_stats(cos)
        stats_rows.append({"layer": layer, **{k: f"{v:.6f}" for k, v in s.items()}})
        print(f"  {layer:15s}  {s['mean']:7.4f}  {s['std']:7.4f}  "
              f"{s['median']:7.4f}  {s['p5']:7.4f}  {s['p95']:7.4f}")

    # ── Save per-sample cosine NPZ ────────────────────────────────────────────
    save_dict = {
        "labels":     meta["labels"],
        "system_ids": meta["system_ids"],
    }
    for layer, cos in per_layer_cosines.items():
        save_dict[f"cos_{layer}"] = cos
    np.savez_compressed(RESULTS_DIR / "per_sample_cosine.npz", **save_dict)
    print(f"\nSaved: {RESULTS_DIR / 'per_sample_cosine.npz'}")

    # ── Save per-sample CSV ───────────────────────────────────────────────────
    N = len(meta["labels"])
    csv_path = RESULTS_DIR / "per_sample_cosine.csv"
    fieldnames = ["sample_id", "label", "system_id"] + [f"cos_{l}" for l in per_layer_cosines]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            row = {
                "sample_id": i,
                "label":     int(meta["labels"][i]),
                "system_id": meta["system_ids"][i],
            }
            for layer, cos in per_layer_cosines.items():
                row[f"cos_{layer}"] = f"{cos[i]:.6f}"
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # ── Save layer stats CSV ──────────────────────────────────────────────────
    stats_path = RESULTS_DIR / "layer_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer", "mean", "std", "median", "p5", "p95"]
        )
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"Saved: {stats_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_distributions(per_layer_cosines, RESULTS_DIR / "similarity_distribution.png")

    layer_means = {l: float(np.mean(v)) for l, v in per_layer_cosines.items()}
    plot_weight_vs_activation(
        group_means, layer_means, RESULTS_DIR / "weight_vs_activation.png"
    )

    # ── Global summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("GLOBAL SUMMARY")
    print("=" * 64)

    early_keys = [l for l in ("pre_encoder", "encoder_feat", "gat_pre") if l in layer_means]
    late_keys  = [l for l in ("gat_l1", "gat_l2", "bilstm") if l in layer_means]
    early_avg  = float(np.mean([layer_means[l] for l in early_keys])) if early_keys else None
    late_avg   = float(np.mean([layer_means[l] for l in late_keys]))  if late_keys  else None

    print("\nMean cosine similarity by depth region:")
    if early_avg is not None:
        print(f"  Early layers ({', '.join(early_keys)}): {early_avg:.4f}")
    if late_avg is not None:
        print(f"  Late layers  ({', '.join(late_keys)}): {late_avg:.4f}")

    if early_avg is not None and late_avg is not None:
        delta = early_avg - late_avg
        print()
        if delta > 0.08:
            print("  FINDING: Representations diverge with depth.")
            print("  → Robust training primarily shifts DECISION BOUNDARY layers,")
            print("    leaving early feature geometry largely intact.")
        elif delta < -0.08:
            print("  FINDING: Later representations are MORE similar than early ones.")
            print("  → Robust training reshapes EARLY FEATURE GEOMETRY,")
            print("    but the models converge to a shared late-stage representation.")
        else:
            print("  FINDING: Similarity is approximately uniform across depth.")
            print("  → Robust training causes GLOBAL representational shift")
            print("    with no clear early/late asymmetry.")

    print()
    ordered = [(l, layer_means[l]) for l in LAYER_ORDER if l in layer_means]
    if len(ordered) >= 2:
        sims  = [v for _, v in ordered]
        trend = "DECREASING" if sims[-1] < sims[0] else "INCREASING or FLAT"
        print(f"  Similarity trend (input → output): {trend}")
        print(f"    {ordered[0][0]} = {sims[0]:.4f}  →  {ordered[-1][0]} = {sims[-1]:.4f}")

    print("\n  Weight-space context:")
    for g in ("phoneme_model", "encoder", "GAT", "rnn", "cls_head"):
        if g in group_means:
            print(f"    {g:20s} weight sim = {group_means[g]:.4f}")

    print(f"\nAll outputs in: {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
