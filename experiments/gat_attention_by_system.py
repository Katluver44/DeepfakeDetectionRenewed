"""
gat_attention_by_system.py
==========================
Repeat the GAT class×class attention analysis from gat_attention.py but
stratify by ASVspoof attack system (A01–A06 in the validation split, plus
bonafide "-").

For each system we accumulate:
  • A 9×9 mean attention matrix  (row-normalised)
  • A 9×9 edge-count matrix      (raw pair counts → confidence proxy)

Outputs saved to experiments/results/:
  gat_by_system_heatmaps.png    — one row per system, 3 cols: normal | system | Δ
  gat_by_system_sibil.png       — sibilant self-attention + nasal→sibilant per system
  gat_by_system_counts.png      — edge count heatmaps (one per system)
  gat_by_system_summary.csv     — numeric table of key metrics per system
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# ── torch.load compat ────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

REPO_ROOT       = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR     = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT  = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"
VOCAB_DIR     = REPO_ROOT / "vocab_phoneme"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
BATCH_SIZE     = 10
SEED           = 42
PER_SYSTEM     = 500   # samples per system; set to None to use all
N_GAT_LAYERS   = 3

LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]
C = len(CLASS_ORDER)
SHORT = [c[:5] for c in CLASS_ORDER]


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def build_id_to_symbol():
    total = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists(): continue
        vocab = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    return {i: (e if i < 5 else e.split("-",1)[1]) for i,e in enumerate(total)}

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

def sym_to_cls(sym):
    if sym in SPECIAL or sym.isdigit(): return "Other"
    for sub, cls in _CAT_RULES:
        if sub in sym: return cls
    return "Other"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode(entry):
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else sf.read(path, dtype="float32", always_2d=False))
    w = torch.from_numpy(arr)
    if w.ndim == 1: w = w.unsqueeze(0)
    elif w.ndim == 2: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR: w = T.Resample(sr, TARGET_SR)(w)
    return w

def _crop(w):
    n = w.shape[-1]
    if n < TARGET_SAMPLES: w = w.repeat(1, -(-TARGET_SAMPLES // n))
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s:s+TARGET_SAMPLES]

def _lbl(raw):
    if isinstance(raw, str):
        s = raw.strip().lower()
        return 0 if s in ("0","bonafide","real","genuine") else 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset — stratified by system_id, with per-system limit
# ---------------------------------------------------------------------------

class SystemStratifiedDataset(torch.utils.data.Dataset):
    """
    Returns batches that include system_id so we can accumulate per-system.
    No 50/50 balancing — we keep all available samples per system up to
    per_system limit.
    """
    def __init__(self, hf_name, split, cache_dir, token, per_system, seed=42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))

        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower()=="key")),
            "label")

        # Group indices by system_id
        by_system: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.ds)):
            sid = self.ds[i].get("system_id", "unknown")
            by_system[sid].append(i)

        rng = random.Random(seed)
        self.indices   = []
        self.sys_ids   = []   # parallel to self.indices

        system_counts = {}
        for sid, idxs in sorted(by_system.items()):
            rng.shuffle(idxs)
            chosen = idxs[:per_system] if per_system else idxs
            self.indices.extend(chosen)
            self.sys_ids.extend([sid] * len(chosen))
            system_counts[sid] = len(chosen)

        print(f"\nValidation split — samples per system (capped at {per_system}):")
        for sid, cnt in sorted(system_counts.items()):
            lbl = "bonafide" if sid == "-" else "spoof"
            print(f"  {sid:6s}  ({lbl:8s})  {cnt:5d} samples")
        print(f"  {'TOTAL':6s}            {len(self.indices):5d} samples")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ex   = self.ds[self.indices[idx]]
        wave = _crop(_decode(ex["audio"]))
        y    = _lbl(ex[self.label_key])
        return {
            "audio":     wave,
            "label":     torch.tensor(y, dtype=torch.long),
            "system_id": self.sys_ids[idx],
            "sample_rate": TARGET_SR,
        }


def collate_fn(batch):
    """Custom collate so system_id (a string) survives DataLoader batching."""
    return {
        "audio":      torch.stack([b["audio"] for b in batch]),
        "label":      torch.stack([b["label"] for b in batch]),
        "system_id":  [b["system_id"] for b in batch],
        "sample_rate": batch[0]["sample_rate"],
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
            "microsoft/wavlm-base" if network_name.lower()=="wavlm"
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
# Capture hooks (same pattern as gat_attention.py)
# ---------------------------------------------------------------------------

class PhonemeAndEdgeCapture:
    def __init__(self, gat_model):
        self.node_phoneme_ids = None
        self.node_sample_idx  = None
        capture = self
        orig = gat_model.encoder_and_GAT.__func__
        import types
        def patched(self_inner, hidden_states, num_frames, phoneme_ids,
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
            capture.node_phoneme_ids = torch.cat(flat_ids)
            capture.node_sample_idx  = torch.cat(flat_samp)
            return result
        gat_model.encoder_and_GAT = types.MethodType(patched, gat_model)

    def register_edge_hook(self, gat_module):
        captured = {}
        def _hook(module, inp, out):
            captured["edge_index"] = out[1].detach().cpu()
        h = gat_module.gat_net[-1].register_forward_hook(_hook)
        return captured, h


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(edge_index, attn_w, node_pids, node_samp,
               system_ids_batch, id_to_class, cls_idx,
               accum: dict, counts: dict):
    """
    accum[system_id] += attention weights per (src_cls, tgt_cls)
    counts[system_id] += 1 per edge
    """
    src = edge_index[0]
    tgt = edge_index[1]
    w   = attn_w.mean(dim=1)   # average over heads → (E,)

    src_pids = node_pids[src]
    tgt_pids = node_pids[tgt]
    src_samp = node_samp[src]

    # Vectorised per-system accumulation
    for e in range(len(src)):
        si_idx  = int(src_samp[e].item())
        sid     = system_ids_batch[si_idx]
        sc      = id_to_class.get(int(src_pids[e].item()), "Other")
        tc      = id_to_class.get(int(tgt_pids[e].item()), "Other")
        r, c    = cls_idx[sc], cls_idx[tc]
        accum[sid][r, c]  += float(w[e].item())
        counts[sid][r, c] += 1


def mean_matrix(accum, counts):
    return np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)

def row_norm(mat):
    rs = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_system_heatmaps(systems_order, rn_matrices, rn_bonafide,
                              count_matrices, out_path, per_system):
    """
    One row per spoof system.  Columns: bonafide | system | Δ (system − bonafide)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    spoof_systems = [s for s in systems_order if s != "-"]
    n_rows = len(spoof_systems)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows))
    if n_rows == 1: axes = axes[None, :]

    vmax_abs = max(rn_bonafide.max(),
                   max(rn_matrices[s].max() for s in spoof_systems))
    delta_lim = max(
        abs((rn_matrices[s] - rn_bonafide)).max() for s in spoof_systems)

    for row_i, sid in enumerate(spoof_systems):
        mat_s  = rn_matrices[sid]
        delta  = mat_s - rn_bonafide
        cnts   = count_matrices[sid]

        for col_i, (mat, title, cmap, vmin, vmax) in enumerate([
            (rn_bonafide, "Bonafide (-)",       "Blues", 0,          vmax_abs),
            (mat_s,       f"Spoof {sid}",        "Reds",  0,          vmax_abs),
            (delta,       f"Δ  {sid} − Bonafide","RdBu_r",-delta_lim, delta_lim),
        ]):
            ax = axes[row_i, col_i]
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks(range(C)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(C)); ax.set_yticklabels(SHORT, fontsize=7)
            if row_i == 0: ax.set_title(title, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate delta cells
            if col_i == 2:
                dlim_local = abs(delta).max()
                for r in range(C):
                    for c in range(C):
                        n_pairs = int(cnts[r, c])
                        color   = "white" if abs(delta[r,c]) > dlim_local*0.5 else "black"
                        ax.text(c, r, f"{delta[r,c]:+.3f}\nn={n_pairs}",
                                ha="center", va="center", fontsize=5, color=color)

        axes[row_i, 0].set_ylabel(f"{sid}", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"GAT attention by phoneme class — per attack system\n"
        f"Row-normalised  ·  n≤{per_system} per system  ·  "
        f"Δ cell shows (system − bonafide) with edge count n",
        fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_count_heatmaps(systems_order, count_matrices, out_path):
    """Show raw edge counts per (src, tgt) class pair for every system."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(systems_order)
    cols = min(4, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    axes = np.array(axes).reshape(-1)

    for i, sid in enumerate(systems_order):
        cnts = count_matrices[sid]
        ax   = axes[i]
        im   = ax.imshow(np.log1p(cnts), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(C)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(C)); ax.set_yticklabels(SHORT, fontsize=7)
        ax.set_title(f"{'Bonafide' if sid=='-' else sid}  (log scale)", fontsize=9)
        for r in range(C):
            for c in range(C):
                ax.text(c, r, str(int(cnts[r,c])),
                        ha="center", va="center", fontsize=5,
                        color="white" if cnts[r,c] > cnts.max()*0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1+count)")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Edge pair counts per phoneme class pair (all GAT layers, log scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_sibilant_bars(systems_order, rn_matrices, rn_bonafide, count_matrices, out_path):
    """
    Two-panel bar chart:
      Left:  Sibilant → Sibilant attention  per system vs bonafide
      Right: Nasal → Sibilant attention     per system vs bonafide
    Error bar = 0 (we show raw mean; n annotated on each bar for context).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    si_idx = CLASS_ORDER.index("Sibilants")
    na_idx = CLASS_ORDER.index("Nasals")

    spoof_sys = [s for s in systems_order if s != "-"]
    x_labels  = ["Bonafide"] + spoof_sys
    bon_sib   = rn_bonafide[si_idx, si_idx]
    bon_nas   = rn_bonafide[na_idx, si_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (src_idx, src_name), bon_val in [
        (axes[0], (si_idx, "Sibilant → Sibilant (self)"), bon_sib),
        (axes[1], (na_idx, "Nasal → Sibilant"),           bon_nas),
    ]:
        vals   = [bon_val] + [rn_matrices[s][src_idx, si_idx] for s in spoof_sys]
        ns     = [count_matrices["-"][src_idx, si_idx]] + \
                 [count_matrices[s][src_idx, si_idx] for s in spoof_sys]
        colors = ["#6EB5FF"] + ["#E03030"] * len(spoof_sys)

        bars = ax.bar(x_labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(bon_val, color="#6EB5FF", linewidth=1.2, linestyle="--", alpha=0.7,
                   label="Bonafide level")

        for bar, n_val in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ax.get_ylim()[1]*0.01,
                    f"n={int(n_val)}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Row-normalised attention weight", fontsize=10)
        ax.set_title(src_name, fontsize=11)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Sibilant attention weights by attack system\n"
        "Light blue = bonafide  ·  Red = individual spoof systems",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    id_to_sym   = build_id_to_symbol()
    id_to_class = {i: sym_to_cls(s) for i,s in id_to_sym.items()}
    cls_idx     = {c: i for i,c in enumerate(CLASS_ORDER)}

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SystemStratifiedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                                        hf_token, PER_SYSTEM, SEED)
    systems_order = sorted(set(dataset.sys_ids))   # ['-', 'A01', ...]
    print(f"\nSystems found: {systems_order}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=True, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))

    print(f"\nLoading checkpoint: {DEFAULT_CKPT}")
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location="cpu", strict=True)
    lit.eval(); lit.freeze()
    gat_model = lit.model

    for layer in gat_model.GAT.gat_net:
        layer.log_attention_weights = True

    capture = PhonemeAndEdgeCapture(gat_model)
    edge_cache, edge_hook = capture.register_edge_hook(gat_model.GAT)

    # ── Per-system accumulators ────────────────────────────────────────────────
    accum  = {s: np.zeros((C, C), dtype=np.float64) for s in systems_order}
    counts = {s: np.zeros((C, C), dtype=np.int64)   for s in systems_order}

    total_edges = 0
    print()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            audio      = batch["audio"]
            labels     = batch["label"]
            system_ids = batch["system_id"]   # list of str, len=B
            B          = labels.shape[0]
            num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1)

            _ = gat_model(audio, num_frames, profiler=None, use_aug=False, stage="eval")

            ei = edge_cache.get("edge_index")
            if ei is None:
                continue
            total_edges += ei.shape[1]

            # Accumulate over all 3 GAT layers
            for layer in gat_model.GAT.gat_net:
                if layer.attention_weights is None: continue
                attn = layer.attention_weights.squeeze(-1).detach().cpu()  # (E, NH)
                accumulate(ei, attn,
                           capture.node_phoneme_ids, capture.node_sample_idx,
                           system_ids, id_to_class, cls_idx,
                           accum, counts)

            edge_cache.clear()

            if (bi + 1) % 20 == 0:
                done = (bi + 1) * BATCH_SIZE
                print(f"  {done:5d} / {len(dataset)}  utterances  "
                      f"({total_edges} edges)")

    edge_hook.remove()
    print(f"\nTotal edges: {total_edges}")

    # ── Compute mean matrices ─────────────────────────────────────────────────
    mean_mats = {s: mean_matrix(accum[s], counts[s]) for s in systems_order}
    rn_mats   = {s: row_norm(mean_mats[s])            for s in systems_order}
    rn_bon    = rn_mats["-"]

    # ── Console summary ───────────────────────────────────────────────────────
    si = cls_idx["Sibilants"]
    na = cls_idx["Nasals"]

    print(f"\n{'System':8s}  {'Sibil→Sibil':>12s}  {'Δ vs bon':>10s}  "
          f"{'Nas→Sibil':>11s}  {'Δ vs bon':>10s}  {'n(Sib→Sib)':>12s}  {'n(Nas→Sib)':>12s}")
    print("-" * 82)
    bon_ss = rn_bon[si, si]
    bon_ns = rn_bon[na, si]
    csv_rows = []
    for s in systems_order:
        ss  = rn_mats[s][si, si]
        ns_ = rn_mats[s][na, si]
        n_ss = int(counts[s][si, si])
        n_ns = int(counts[s][na, si])
        lbl  = "bonafide" if s == "-" else "spoof"
        print(f"{s:8s}  {ss:12.4f}  {ss-bon_ss:+10.4f}  "
              f"{ns_:11.4f}  {ns_-bon_ns:+10.4f}  {n_ss:12d}  {n_ns:12d}")
        csv_rows.append({
            "system_id": s, "label": lbl,
            "sibil_sibil": round(ss, 6),
            "sibil_sibil_delta": round(ss - bon_ss, 6),
            "nasal_sibil": round(ns_, 6),
            "nasal_sibil_delta": round(ns_ - bon_ns, 6),
            "n_sibil_sibil_edges": n_ss,
            "n_nasal_sibil_edges": n_ns,
            "total_edges_this_system": int(counts[s].sum()),
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "gat_by_system_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nSaved: {csv_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_per_system_heatmaps(
        systems_order, rn_mats, rn_bon, counts,
        RESULTS_DIR / "gat_by_system_heatmaps.png", PER_SYSTEM)

    plot_count_heatmaps(
        systems_order, counts,
        RESULTS_DIR / "gat_by_system_counts.png")

    plot_sibilant_bars(
        systems_order, rn_mats, rn_bon, counts,
        RESULTS_DIR / "gat_by_system_sibil.png")

    print(f"\nAll results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
