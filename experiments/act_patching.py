"""
act_patching.py
===============
Layer-0 GAT activation patching on correctly-classified spoof utterances.

For each phoneme node i in a spoof utterance x:
  Δ_i = L(x) − L_patch(x, i)

where L_patch is the spoof logit after replacing h_i^(0) with h̄_{c_i}^(0),
the bonafide mean for phoneme class c_i (e.g., all Sibilants, all Nasals, ...).

  Δ > 0  →  node was pushing toward "spoof" — patching to bonafide reduced logit
  Δ ≈ 0  →  node was not contributing
  Δ < 0  →  node was acting as a counterweight toward bonafide

Saves to experiments/results/act_patching/:
  bonafide_class_means.npz      — (9,768) mean h0 vectors per phoneme class
  act_patching_records.npz      — per-node (class_id, delta, logit_orig, system_id)
  act_patching_class_stats.csv  — per-class mean / std / count
  act_patching_violin.png       — Δ distribution per class (violin + mean bar)
  act_patching_by_system.png    — class × system mean-Δ heatmap
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch  # must be before matplotlib to avoid NumPy C-API conflict
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── torch.load compat ─────────────────────────────────────────────────────────
_orig_load = torch.load
def _patch(*a, **kw): kw.setdefault("weights_only", False); return _orig_load(*a, **kw)
torch.load = _patch
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

REPO_ROOT       = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUT_DIR         = EXPERIMENTS_DIR / "results" / "act_patching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT  = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
VOCAB_DIR     = REPO_ROOT / "vocab_phoneme"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
BATCH_SIZE     = 5      # utterances per extraction batch
PER_SYSTEM     = 500    # samples per system
SEED           = 42

# ---------------------------------------------------------------------------
# Phoneme class mapping  (same as gat_attention_by_system.py)
# ---------------------------------------------------------------------------

LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]

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


def sym_to_cls(sym: str) -> str:
    if sym in SPECIAL or sym.isdigit():
        return "Other"
    for sub, cls in _CAT_RULES:
        if sub in sym:
            return cls
    return "Other"


def build_id_to_class() -> dict[int, str]:
    total = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists():
            continue
        vocab = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id2sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    return {i: sym_to_cls(sym) for i, sym in id2sym.items()}


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
        return 0 if raw.strip().lower() in ("0", "bonafide", "real", "genuine") else 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SystemStratifiedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, per_system, seed=42):
        from collections import defaultdict as dd
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )
        by_sys = dd(list)
        for i in range(len(self.ds)):
            by_sys[self.ds[i].get("system_id", "unknown")].append(i)
        rng = random.Random(seed)
        self.indices, self.sys_ids = [], []
        for sid, idxs in sorted(by_sys.items()):
            rng.shuffle(idxs)
            chosen = idxs[:per_system] if per_system else idxs
            self.indices.extend(chosen)
            self.sys_ids.extend([sid] * len(chosen))
        print(f"Dataset '{split}': {len(self.indices)} samples  "
              f"({len(by_sys)} systems)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex  = self.ds[self.indices[idx]]
        wav = _crop(_decode(ex["audio"]))
        y   = _lbl(ex[self.label_key])
        sid = ex.get("system_id", "unknown")
        return {
            "audio":       wav,
            "label":       torch.tensor(y, dtype=torch.long),
            "system_id":   sid,
            "sample_rate": TARGET_SR,
        }


def collate_fn(batch):
    audio  = torch.stack([b["audio"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    sids   = [b["system_id"] for b in batch]
    return {"audio": audio, "label": labels, "system_id": sids,
            "sample_rate": TARGET_SR}


# ---------------------------------------------------------------------------
# phoneme loader patch (same as other scripts)
# ---------------------------------------------------------------------------

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = (
            "microsoft/wavlm-base" if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h"
        )
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None,
                total_num_phonemes=total_num_phonemes, weights_only=False,
            ).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)

    pm.load_phoneme_model  = _load
    mm.load_phoneme_model  = _load


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def run_and_capture(lit_model, audio: torch.Tensor, device: torch.device) -> dict:
    """
    Run one batch through the model, intercepting:
      - h0        : (N_total_nodes, 768) layer-0 GAT output
      - edge_index: (2, E) full-batch edge index
      - node_pids : (N_total_nodes,) phoneme ID per node
      - rnf       : (B,) number of nodes per sample
      - logit     : (B,) raw spoof logit
    """
    gat_model = lit_model.model
    store: dict = {}

    # Hook layer-0 output
    def _hook_l0(module, inp, output):
        store["h0"]         = output[0].detach()
        store["edge_index"] = output[1].detach()

    h0_hook = gat_model.GAT.gat_net[0].register_forward_hook(_hook_l0)

    # Patch encoder_and_GAT to capture node phoneme IDs
    _orig_eag = gat_model.encoder_and_GAT

    def _patched_eag(*args, **kwargs):
        result = _orig_eag(*args, **kwargs)
        _, _, reduced_pids, rnf, _, _ = result
        store["node_pids"] = torch.cat(
            [reduced_pids[i, : rnf[i]] for i in range(len(rnf))]
        ).detach()
        store["rnf"] = rnf.detach()
        return result

    gat_model.encoder_and_GAT = _patched_eag

    try:
        x = audio.squeeze(1) if audio.ndim == 3 else audio
        B = x.shape[0]
        num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)
        with torch.no_grad():
            out = gat_model(x.to(device), num_frames, use_aug=False, stage="eval")
        store["logit"] = out["logit"].detach()
    finally:
        h0_hook.remove()
        gat_model.encoder_and_GAT = _orig_eag

    return store


def extract_sample_subgraph(h0: torch.Tensor, edge_index: torch.Tensor,
                             rnf: torch.Tensor, b: int):
    """Extract h0 and edge_index for sample b from a packed batch."""
    offset = int(rnf[:b].sum())
    n      = int(rnf[b])
    h0_b   = h0[offset : offset + n]

    src, dst = edge_index
    mask     = ((src >= offset) & (src < offset + n) &
                (dst >= offset) & (dst < offset + n))
    ei_b     = edge_index[:, mask] - offset
    return h0_b, ei_b


def patch_utterance(h0_b: torch.Tensor, ei_b: torch.Tensor,
                    node_pids_b: torch.Tensor,
                    class_means: dict[str, torch.Tensor],
                    id2cls: dict[int, str],
                    gat_model, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """
    Batch-patch all N_b nodes of one utterance simultaneously.

    Returns
    -------
    deltas    : (N_b,)  Δ_i = L(x) − L_patch(x, i)   (nan if class mean missing)
    cls_names : [str]   phoneme class name per node
    """
    N = h0_b.shape[0]
    E = ei_b.shape[1] if ei_b.numel() > 0 else 0

    cls_names = [id2cls.get(int(node_pids_b[i]), "Other") for i in range(N)]

    # ── Build N copies of h0, one patched node each ──────────────────────────
    h0_copies = h0_b.unsqueeze(0).expand(N, -1, -1).clone()  # (N, N, 768)
    valid = torch.ones(N, dtype=torch.bool, device=device)
    for i in range(N):
        cn = cls_names[i]
        if cn in class_means:
            h0_copies[i, i] = class_means[cn]
        else:
            valid[i] = False

    if not valid.any():
        return torch.full((N,), float("nan"), device=device), cls_names

    # ── Meta-graph: N disconnected copies of the utterance's graph ───────────
    h0_flat = h0_copies.reshape(N * N, 768)   # (N*N, 768)

    # If the utterance has no graph edges (e.g. single phoneme segment),
    # add self-loops so the GAT aggregation doesn't get an empty edge list.
    if E == 0:
        self_idx = torch.arange(N, device=device)
        ei_b = torch.stack([self_idx, self_idx])  # (2, N)
        E = N

    offsets = (torch.arange(N, device=device) * N).view(N, 1, 1)  # (N,1,1)
    ei_rep  = ei_b.unsqueeze(0).expand(N, -1, -1) + offsets       # (N,2,E)
    ei_flat = ei_rep.reshape(2, N * E)                              # (2,N*E)

    # ── Run layers 1+2 ───────────────────────────────────────────────────────
    h1_flat, _ = gat_model.GAT.gat_net[1]((h0_flat, ei_flat))
    h2_flat, _ = gat_model.GAT.gat_net[2]((h1_flat, ei_flat))

    # Reshape to (N, N, 768) — N copies, N nodes each
    h2_copies = h2_flat.reshape(N, N, 768)

    # ── LSTM: (N copies, N nodes, 768) → (N, N, 768) ─────────────────────────
    lstm_out, _ = gat_model.rnn(h2_copies)

    # ── Mean pool, norm, classify → (N,) logits ──────────────────────────────
    pooled  = lstm_out.mean(dim=1)                     # (N, 768)
    normed  = gat_model.norm_feat(pooled)              # (N, 768)
    logits  = gat_model.cls_head(normed).squeeze(-1)   # (N,)

    # Δ_i = L_orig − L_patch (computed by caller; we just return logits_patched)
    deltas = torch.full((N,), float("nan"), device=device)
    deltas[valid] = logits[valid]   # store logits_patched (caller subtracts L_orig)
    return deltas, cls_names


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_violin(records_by_cls: dict[str, list[float]], out_path: Path) -> None:
    classes = [c for c in CLASS_ORDER if records_by_cls.get(c)]
    data    = [records_by_cls[c] for c in classes]
    means   = [float(np.mean(d)) for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin
    ax = axes[0]
    parts = ax.violinplot(data, showmedians=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Δ  (logit drop when node patched to bonafide)")
    ax.set_title("Per-node causal effect Δ by phoneme class", fontsize=11)
    for i, (pc, m) in enumerate(zip(parts["bodies"], means), 1):
        pc.set_alpha(0.7)
        ax.scatter(i, m, color="black", s=30, zorder=5)

    # Bar chart of means
    ax2 = axes[1]
    colours = ["#E03030" if m > 0 else "#4C72B0" for m in means]
    ax2.bar(range(len(classes)), means, color=colours, width=0.6, alpha=0.85)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Δ")
    ax2.set_title("Mean causal effect per class\n(red=spoof-pushing, blue=bonafide-pushing)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_by_system(records_cls_sys: dict, systems: list[str], out_path: Path) -> None:
    """Heatmap: class × system mean Δ."""
    classes = [c for c in CLASS_ORDER if any(records_cls_sys.get((c, s)) for s in systems)]
    mat     = np.zeros((len(classes), len(systems)))
    count   = np.zeros((len(classes), len(systems)), dtype=int)

    for ci, cls in enumerate(classes):
        for si, sys in enumerate(systems):
            vals = records_cls_sys.get((cls, sys), [])
            if vals:
                mat[ci, si]   = float(np.mean(vals))
                count[ci, si] = len(vals)

    fig, ax = plt.subplots(figsize=(9, 5))
    vabs = max(abs(mat.max()), abs(mat.min()), 0.01)
    im   = ax.imshow(mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean Δ")

    ax.set_xticks(range(len(systems))); ax.set_xticklabels(systems, fontsize=9)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9)
    ax.set_title("Mean causal effect Δ  (class × attack system)", fontsize=11)

    for ci in range(len(classes)):
        for si in range(len(systems)):
            n = count[ci, si]
            if n > 0:
                ax.text(si, ci, f"{mat[ci,si]:.3f}\n(n={n})",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(mat[ci, si]) > vabs * 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ── Device ───────────────────────────────────────────────────────────────
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except RuntimeError:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Phoneme class map ─────────────────────────────────────────────────────
    id2cls = build_id_to_class()
    print(f"Phoneme ID → class map: {len(id2cls)} entries")

    # ── Model ─────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True,
    ))
    print(f"Loading checkpoint: {DEFAULT_CKPT}")
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    gat_model = lit.model
    print(f"Model ready on {device}.\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SystemStratifiedDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token, PER_SYSTEM, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    # ── Phase 1: Bonafide class means ────────────────────────────────────────
    print("Phase 1: computing bonafide class means...")
    class_h0_acc = defaultdict(list)   # class_name → list of (768,) tensors on CPU

    for batch in loader:
        sids   = batch["system_id"]
        labels = batch["label"]
        bon_mask = [i for i, (s, l) in enumerate(zip(sids, labels))
                    if s == "-" and l.item() == 0]
        if not bon_mask:
            continue

        audio_bon = batch["audio"][bon_mask]
        cap = run_and_capture(lit, audio_bon, device)

        h0       = cap["h0"]         # (N_total, 768)
        node_pids = cap["node_pids"]  # (N_total,)
        rnf      = cap["rnf"]         # (B_bon,)

        cursor = 0
        for b in range(len(rnf)):
            n = int(rnf[b])
            for i in range(n):
                pid = int(node_pids[cursor + i])
                cls = id2cls.get(pid, "Other")
                class_h0_acc[cls].append(h0[cursor + i].cpu())
            cursor += n

    class_means: dict[str, torch.Tensor] = {}
    for cls, vecs in class_h0_acc.items():
        class_means[cls] = torch.stack(vecs).mean(0).to(device)
    print("  Classes with bonafide means:", sorted(class_means.keys()))
    print(f"  Total bonafide nodes used: {sum(len(v) for v in class_h0_acc.values())}")

    # Save bonafide means
    np.savez_compressed(
        OUT_DIR / "bonafide_class_means.npz",
        **{cls: np.array(mean.cpu().detach().tolist())
           for cls, mean in class_means.items()},
    )

    # ── Phase 2: Activation patching on spoof samples ────────────────────────
    print("\nPhase 2: activation patching on spoof samples...")

    # Records: parallel arrays
    rec_cls   = []   # str class name per node
    rec_delta = []   # float Δ_i
    rec_logit = []   # float L(x)
    rec_sid   = []   # str system_id
    rec_nvalid = 0
    rec_nskip  = 0

    SPOOF_SYSTEMS = ["A01", "A02", "A03", "A04", "A05", "A06"]

    for batch in loader:
        sids   = batch["system_id"]
        labels = batch["label"]

        spoof_mask = [i for i, (s, l) in enumerate(zip(sids, labels))
                      if s in SPOOF_SYSTEMS and l.item() == 1]
        if not spoof_mask:
            continue

        audio_sp = batch["audio"][spoof_mask]
        sids_sp  = [sids[i] for i in spoof_mask]

        cap = run_and_capture(lit, audio_sp, device)
        h0        = cap["h0"]          # (N_total, 768)
        edge_index = cap["edge_index"] # (2, E)
        node_pids  = cap["node_pids"]  # (N_total,)
        rnf        = cap["rnf"]        # (B_sp,)
        logits_orig = cap["logit"]     # (B_sp,)

        cursor = 0
        for b in range(len(rnf)):
            n        = int(rnf[b])
            logit_b  = float(logits_orig[b])
            sid_b    = sids_sp[b]

            # Only patch correctly-classified spoof samples (logit > 0 → predicted spoof)
            if logit_b <= 0:
                rec_nskip += 1
                cursor += n
                continue

            h0_b, ei_b     = extract_sample_subgraph(h0, edge_index, rnf, b)
            node_pids_b    = node_pids[cursor : cursor + n]

            with torch.no_grad():
                logits_patched, cls_names = patch_utterance(
                    h0_b, ei_b, node_pids_b, class_means, id2cls, gat_model, device)

            # Δ_i = L(x) - L_patch(x, i)
            for i in range(n):
                lp = float(logits_patched[i])
                if not np.isnan(lp):
                    delta_i = logit_b - lp
                    rec_cls.append(cls_names[i])
                    rec_delta.append(delta_i)
                    rec_logit.append(logit_b)
                    rec_sid.append(sid_b)

            rec_nvalid += 1
            cursor += n

    print(f"  Patched: {rec_nvalid} utterances  ({rec_nskip} skipped, wrong prediction)")
    print(f"  Total node records: {len(rec_delta)}")

    # ── Save raw records ──────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_DIR / "act_patching_records.npz",
        cls_names  = np.array(rec_cls),
        deltas     = np.array(rec_delta, dtype=np.float32),
        logit_origs= np.array(rec_logit, dtype=np.float32),
        system_ids = np.array(rec_sid),
    )

    # ── Phase 3: Aggregate and report ────────────────────────────────────────
    by_cls: dict[str, list[float]] = defaultdict(list)
    by_cls_sys: dict[tuple, list[float]] = defaultdict(list)
    for cls, delta, sid in zip(rec_cls, rec_delta, rec_sid):
        by_cls[cls].append(delta)
        by_cls_sys[(cls, sid)].append(delta)

    print(f"\n{'Class':<16}  {'Mean Δ':>9}  {'Std':>7}  {'n':>6}")
    print("─" * 44)
    stats_rows = []
    for cls in CLASS_ORDER:
        if not by_cls[cls]:
            continue
        d = np.array(by_cls[cls])
        print(f"  {cls:<14}  {d.mean():+.4f}   {d.std():.4f}   {len(d):6}")
        stats_rows.append({"class": cls, "mean_delta": f"{d.mean():.6f}",
                            "std_delta": f"{d.std():.6f}", "n": len(d)})

    with open(OUT_DIR / "act_patching_class_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "mean_delta", "std_delta", "n"])
        w.writeheader(); w.writerows(stats_rows)
    print(f"\nSaved: act_patching_class_stats.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_violin(by_cls, OUT_DIR / "act_patching_violin.png")
    plot_by_system(by_cls_sys, SPOOF_SYSTEMS, OUT_DIR / "act_patching_by_system.png")

    print(f"\nAll artefacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
