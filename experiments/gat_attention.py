"""
gat_attention.py
================
Extract per-edge attention weights from every GAT layer and build a
phoneme-class × phoneme-class attention matrix for bonafide vs deepfake.

For each directed edge (src_phoneme_node → tgt_phoneme_node) the GAT
computes a scalar attention weight (softmaxed over all incoming edges at
the target).  We read those weights, look up the IPA class of both
endpoints, and accumulate:

    M[label, src_class, tgt_class] += mean-head attention weight

After collecting all edges across all batches we produce:

  • gat_attention_heatmap.png   — 3-panel figure:
       normal (left) | deepfake (center) | Δ deepfake−normal (right)
       Each 9×9 cell: average attention weight (row-normalised per label)

  • gat_attention_topedges.png  — horizontal bar chart showing the
       top-15 class→class edge types ranked by |Δ| (deepfake − normal)

Saves to experiments/results/.

Convention:
  edge_index[0] = source node (message sender)
  edge_index[1] = target node (message receiver / aggregator)
  attention[e]  = how much src contributes when updating tgt
  → matrix rows = source class, columns = target class
"""
from __future__ import annotations

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
    from pandas import Series as _PandasSeries
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PandasSeries, _PTR])
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
N_SAMPLES      = 300   # more utterances → more edges → stable matrix estimates
BATCH_SIZE     = 10
SEED           = 42
N_GAT_LAYERS   = 3     # must match gat_config in modules.py
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]

CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]


# ---------------------------------------------------------------------------
# Vocab helpers  (shared with phoneme_analysis.py)
# ---------------------------------------------------------------------------

def build_id_to_symbol() -> dict[int, str]:
    total_phonemes: list[str] = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists():
            continue
        vocab: dict[str, int] = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total_phonemes.append(f"{lang}-{sym}")
    return {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total_phonemes)}


_CATEGORY_RULES: list[tuple[str, str]] = [
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

def symbol_to_class(sym: str) -> str:
    if sym in SPECIAL or sym.isdigit():
        return "Other"
    for substr, cls in _CATEGORY_RULES:
        if substr in sym:
            return cls
    return "Other"


# ---------------------------------------------------------------------------
# Audio / dataset helpers  (same as other scripts)
# ---------------------------------------------------------------------------

def _decode_audio_entry(entry):
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else sf.read(path, dtype="float32", always_2d=False))
    wave = torch.from_numpy(arr)
    if wave.ndim == 1: wave = wave.unsqueeze(0)
    elif wave.ndim == 2: wave = wave.mean(0, keepdim=True)
    if sr != TARGET_SR: wave = T.Resample(sr, TARGET_SR)(wave)
    return wave

def _crop_center(wave):
    n = wave.shape[-1]
    if n < TARGET_SAMPLES:
        wave = wave.repeat(1, -(-TARGET_SAMPLES // n))
    s = (wave.shape[-1] - TARGET_SAMPLES) // 2
    return wave[:, s : s + TARGET_SAMPLES]

def _label_to_int(raw):
    if isinstance(raw, str):
        s = raw.strip().lower()
        return 0 if s in ("0","bonafide","real","genuine") else 1
    return int(raw)


class SoundfileHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, limit, token, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")), "label")
        labels = [_label_to_int(self.ds[i][self.label_key]) for i in range(len(self.ds))]
        idx0 = [i for i,y in enumerate(labels) if y==0]
        idx1 = [i for i,y in enumerate(labels) if y==1]
        rng  = random.Random(seed); rng.shuffle(idx0); rng.shuffle(idx1)
        k = min(len(idx0), len(idx1), limit // 2)
        self.indices = idx0[:k] + idx1[:k]; rng.shuffle(self.indices)
        counts = Counter(_label_to_int(self.ds[i][self.label_key]) for i in self.indices)
        print(f"Holdout '{split}': {len(self.indices)} examples ({counts[0]} normal + {counts[1]} deepfake)")

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        wave = _crop_center(_decode_audio_entry(ex["audio"]))
        y = _label_to_int(ex[self.label_key])
        return {"audio": wave, "label": torch.tensor(y, dtype=torch.long), "sample_rate": TARGET_SR}


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
        return BaseModule(network_param, optim_param, tokenizer=None, total_num_phonemes=total_num_phonemes)

    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Capture hook: intercepts encoder_and_GAT to save phoneme IDs + edge_index
# ---------------------------------------------------------------------------

class PhonemeAndEdgeCapture:
    """
    Monkey-patches Phoneme_GAT.encoder_and_GAT to save after each call:
      .node_phoneme_ids  — flat tensor (total_nodes,) with phoneme ID per node
      .node_sample_idx   — flat tensor (total_nodes,) with sample index per node
      .edge_index        — (2, E) int64 edge tensor (same across all GAT layers)
    """
    def __init__(self, gat_model):
        self.node_phoneme_ids = None
        self.node_sample_idx  = None
        self.edge_index       = None
        capture = self
        orig = gat_model.encoder_and_GAT.__func__

        import types
        def patched(self_inner, hidden_states, num_frames, phoneme_ids,
                    profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_inner, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            # result = (hidden_states, reduced_hidden_states, reduced_phoneme_ids,
            #           reduced_num_frames, encoder_feat, logit)
            rids = result[2].detach().cpu()   # (B, Lmax)
            rnf  = result[3].detach().cpu()   # (B,)

            # Flatten phoneme IDs and build sample-index per node
            flat_ids  = []
            flat_samp = []
            for i in range(len(rnf)):
                n = int(rnf[i].item())
                flat_ids.append(rids[i, :n])
                flat_samp.append(torch.full((n,), i, dtype=torch.long))
            capture.node_phoneme_ids = torch.cat(flat_ids)   # (total_nodes,)
            capture.node_sample_idx  = torch.cat(flat_samp)  # (total_nodes,)
            return result

        gat_model.encoder_and_GAT = types.MethodType(patched, gat_model)

    def edge_index_from_gat(self, gat_module):
        """Read edge_index from the output of the last GAT layer (it passes through unchanged)."""
        # The GAT Sequential passes (features, edge_index) tuples between layers.
        # We hook the last layer's forward output to grab the edge_index.
        # Simpler: we register a one-shot hook here.
        captured = {}
        def _hook(module, inp, out):
            captured["edge_index"] = out[1].detach().cpu()
        h = gat_module.gat_net[-1].register_forward_hook(_hook)
        return captured, h  # caller must call h.remove() after the forward pass


# ---------------------------------------------------------------------------
# Accumulate attention matrices
# ---------------------------------------------------------------------------

def accumulate_attention(
    edge_index: torch.Tensor,          # (2, E)
    attn_weights: torch.Tensor,        # (E, NH) — mean over heads will be taken
    node_phoneme_ids: torch.Tensor,    # (total_nodes,)
    node_sample_idx: torch.Tensor,     # (total_nodes,)
    sample_labels: torch.Tensor,       # (B,)
    id_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    accum: np.ndarray,                 # (2, C, C) float — updated in place
    count: np.ndarray,                 # (2, C, C) int — updated in place
) -> None:
    """
    For every edge (src→tgt):
      label  = sample_labels[node_sample_idx[src]]
      w      = mean attention weight across heads
      Update accum[label, src_class, tgt_class] += w
              count[label, src_class, tgt_class] += 1
    """
    src_nodes = edge_index[0]  # (E,)
    tgt_nodes = edge_index[1]  # (E,)
    w = attn_weights.mean(dim=1)  # (E,) — average over heads

    src_pids  = node_phoneme_ids[src_nodes]   # (E,)
    tgt_pids  = node_phoneme_ids[tgt_nodes]   # (E,)
    src_samps = node_sample_idx[src_nodes]    # (E,)
    labels_e  = sample_labels[src_samps]      # (E,) — label of each edge's source sample

    for e in range(len(src_nodes)):
        lbl     = int(labels_e[e].item())
        src_cls = id_to_class.get(int(src_pids[e].item()), "Other")
        tgt_cls = id_to_class.get(int(tgt_pids[e].item()), "Other")
        si      = class_to_idx[src_cls]
        ti      = class_to_idx[tgt_cls]
        accum[lbl, si, ti] += float(w[e].item())
        count[lbl, si, ti] += 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _row_normalise(mat: np.ndarray) -> np.ndarray:
    """Row-normalise a 2-D matrix (each source row sums to 1)."""
    row_sums = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)


def plot_heatmaps(mean_normal: np.ndarray, mean_fake: np.ndarray,
                  classes: list[str], out_path: Path, n_samples: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm_n = _row_normalise(mean_normal)
    norm_f = _row_normalise(mean_fake)
    delta  = norm_f - norm_n

    # Shared colour scale for absolute matrices
    vmax_abs = max(norm_n.max(), norm_f.max())
    # Delta uses its own symmetric scale
    dlim = np.abs(delta).max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    short = [c[:4] for c in classes]   # abbreviated tick labels

    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], norm_n, "Normal (bonafide)", "Blues",   0, vmax_abs),
        (axes[1], norm_f, "Deepfake (spoof)",  "Reds",    0, vmax_abs),
        (axes[2], delta,  "Δ  (deepfake − normal)",  "RdBu_r", -dlim, dlim),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Target phoneme class  (node being updated)", fontsize=9)
        ax.set_ylabel("Source phoneme class  (message sender)", fontsize=9)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells with values for the delta panel only
        if mat is delta:
            for r in range(len(classes)):
                for c in range(len(classes)):
                    ax.text(c, r, f"{delta[r,c]:+.3f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if abs(delta[r,c]) > dlim * 0.5 else "black")

    fig.suptitle(
        f"GAT attention by phoneme class  ·  n={n_samples} utterances\n"
        "Row = source class (sender),  Column = target class (receiver)  ·  Row-normalised",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_top_edges(delta_matrix: np.ndarray, classes: list[str],
                   out_path: Path, top_k: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = len(classes)
    rows, cols, vals = [], [], []
    for r in range(C):
        for c in range(C):
            rows.append(r); cols.append(c); vals.append(delta_matrix[r, c])

    # Sort by absolute delta
    order   = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)[:top_k]
    labels  = [f"{classes[rows[i]][:5]}→{classes[cols[i]][:5]}" for i in order]
    values  = [vals[i] for i in order]
    colors  = ["#E03030" if v > 0 else "#6EB5FF" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.38)))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Δ attention weight  (deepfake − normal, row-normalised)", fontsize=10)
    ax.set_title(
        f"Top {top_k} class→class edge types by |Δ|\n"
        "Red = more attended in deepfake  ·  Blue = more attended in normal",
        fontsize=11,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    id_to_sym   = build_id_to_symbol()
    id_to_class = {i: symbol_to_class(s) for i, s in id_to_sym.items()}
    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    C = len(CLASS_ORDER)
    print(f"Vocab: {len(id_to_sym)} tokens  |  {C} phoneme classes")

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SoundfileHFDataset(HF_DATASET, "validation", str(CACHE_DIR),
                                   N_SAMPLES, hf_token, SEED)
    loader   = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0)

    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))

    print(f"\nLoading checkpoint: {DEFAULT_CKPT}")
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location="cpu", strict=True)
    lit.to(DEVICE)
    lit.eval(); lit.freeze()
    gat_model = lit.model
    print(f"Model ready on {DEVICE}.\n")

    # ── Enable attention logging on every GAT layer ──────────────────────────
    for layer in gat_model.GAT.gat_net:
        layer.log_attention_weights = True

    # ── Install phoneme + edge capture hook ───────────────────────────────────
    capture     = PhonemeAndEdgeCapture(gat_model)
    edge_capture, edge_hook = capture.edge_index_from_gat(gat_model.GAT)

    # ── Accumulators: shape (2, C, C) — index 0=normal, 1=deepfake ──────────
    # We accumulate per-layer separately, then average at the end.
    accum_layers = [np.zeros((2, C, C), dtype=np.float64) for _ in range(N_GAT_LAYERS)]
    count_layers = [np.zeros((2, C, C), dtype=np.int64)   for _ in range(N_GAT_LAYERS)]

    total_edges_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            audio  = batch["audio"].to(DEVICE)
            labels = batch["label"]   # (B,) — kept on CPU for accumulation
            B      = labels.shape[0]
            num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=DEVICE)

            _ = gat_model(audio, num_frames, profiler=None, use_aug=False, stage="eval")

            # edge_index: read from the hook we registered on the last GAT layer
            edge_index = edge_capture.get("edge_index")
            if edge_index is None:
                print(f"  [WARNING] edge_index not captured at batch {batch_idx}, skipping")
                continue

            E = edge_index.shape[1]
            total_edges_seen += E

            for layer_i, layer in enumerate(gat_model.GAT.gat_net):
                if layer.attention_weights is None:
                    continue
                # attention_weights shape: (E, NH, 1)
                attn = layer.attention_weights.squeeze(-1).detach().cpu()   # (E, NH)

                accumulate_attention(
                    edge_index, attn,
                    capture.node_phoneme_ids,
                    capture.node_sample_idx,
                    labels,
                    id_to_class, class_to_idx,
                    accum_layers[layer_i],
                    count_layers[layer_i],
                )

            # Reset edge capture for next batch
            edge_capture.clear()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx+1)*BATCH_SIZE} / {len(dataset)} utterances"
                      f"  ({total_edges_seen} edges so far)")

    edge_hook.remove()
    print(f"\nTotal edges processed: {total_edges_seen}")

    # ── Average across GAT layers ─────────────────────────────────────────────
    # Use count_layers to compute mean; stack layers then average
    mean_all = np.zeros((2, C, C), dtype=np.float64)
    cnt_all  = np.zeros((2, C, C), dtype=np.int64)
    for layer_i in range(N_GAT_LAYERS):
        mean_all += accum_layers[layer_i]
        cnt_all  += count_layers[layer_i]

    # Mean attention per (label, src_class, tgt_class)
    mean_matrix = np.divide(mean_all, cnt_all,
                             out=np.zeros_like(mean_all), where=cnt_all > 0)

    mean_normal = mean_matrix[0]   # (C, C)
    mean_fake   = mean_matrix[1]   # (C, C)
    delta       = _row_normalise(mean_fake) - _row_normalise(mean_normal)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Row-normalised attention matrix: Δ (deepfake − normal) ──────────────")
    header = f"{'':12s}" + "".join(f"{c[:5]:>8s}" for c in CLASS_ORDER)
    print(header)
    print("-" * len(header))
    for r, src in enumerate(CLASS_ORDER):
        row = f"{src[:12]:12s}" + "".join(f"{delta[r,c]:+8.4f}" for c in range(C))
        print(row)

    # ── Per-layer report (final layer only) ───────────────────────────────────
    last_mean = np.divide(accum_layers[-1], count_layers[-1],
                          out=np.zeros_like(accum_layers[-1]), where=count_layers[-1] > 0)
    last_n = _row_normalise(last_mean[0])
    last_f = _row_normalise(last_mean[1])
    last_d = last_f - last_n

    print("\n── Final GAT layer only — top outgoing Δ from Sibilants and Nasals ──────")
    for src_name in ["Sibilants", "Nasals"]:
        si = class_to_idx[src_name]
        print(f"\n  {src_name} (as source):")
        ranked = sorted(enumerate(last_d[si]), key=lambda x: abs(x[1]), reverse=True)
        for ti, dv in ranked[:5]:
            print(f"    → {CLASS_ORDER[ti]:15s}  Δ={dv:+.4f}  "
                  f"(normal={last_n[si,ti]:.4f}, fake={last_f[si,ti]:.4f})")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_heatmaps(mean_normal, mean_fake, CLASS_ORDER,
                  RESULTS_DIR / "gat_attention_heatmap.png", len(dataset))
    plot_top_edges(delta, CLASS_ORDER,
                   RESULTS_DIR / "gat_attention_topedges.png")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
