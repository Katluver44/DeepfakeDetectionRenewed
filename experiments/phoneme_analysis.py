"""
phoneme_analysis.py
====================
Project post-GAT node embeddings onto PC1 and stratify by phoneme class.

Each phoneme node in the GAT graph gets its own post-GAT embedding.
We project those onto the first principal component (fitted on the pooled
classification embeddings so PC1 already points toward the real/fake axis),
then ask: do different phoneme classes separate along that axis, and is the
signal concentrated in particular classes (e.g. sibilants, nasals)?

Outputs (saved to experiments/results/):
  phoneme_pc1_violin.png  — violin plot, one strip per phoneme class,
                            split red/blue by sample label
  phoneme_pc1_mean.png    — per-class mean PC1 score, real vs deepfake

Red   = deepfake / spoof  (label 1)
Light blue = normal / bonafide (label 0)
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
N_SAMPLES      = 200   # more samples for phoneme-level stats
BATCH_SIZE     = 10
SEED           = 42

# Language ordering used by Phonemer_Tokenizer_Recombination
LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]


# ---------------------------------------------------------------------------
# Build combined phoneme ID → symbol mapping
# ---------------------------------------------------------------------------

def build_id_to_symbol() -> dict[int, str]:
    """
    Reconstruct the exact ID→symbol table that Phonemer_Tokenizer_Recombination
    builds internally.  IDs 0-4 are special tokens; IDs 5+ are
    "<lang>-<ipa_symbol>" entries in lang order, skipping per-lang specials.
    Returns a plain symbol string (lang prefix stripped).
    """
    total_phonemes: list[str] = list(SPECIAL)  # indices 0-4

    for lang in LANG_ORDER:
        vocab_path = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not vocab_path.exists():
            continue
        with open(vocab_path) as f:
            vocab: dict[str, int] = json.load(f)
        # vocab is symbol→id; sort by id to preserve order
        for symbol, _idx in sorted(vocab.items(), key=lambda x: x[1]):
            if symbol not in SPECIAL:
                total_phonemes.append(f"{lang}-{symbol}")

    # Strip lang prefix for display
    id_to_sym = {}
    for idx, entry in enumerate(total_phonemes):
        if idx < 5:
            id_to_sym[idx] = entry          # keep special as-is
        else:
            id_to_sym[idx] = entry.split("-", 1)[1]  # drop "en-" etc.
    return id_to_sym


# ---------------------------------------------------------------------------
# Phoneme class categorisation
# ---------------------------------------------------------------------------

# Map IPA (and a few SAMPA-like) symbols to broad phonetic classes.
# Each entry is (substring_or_exact, class_name); checked in order.
_CATEGORY_RULES: list[tuple[str, str]] = [
    # sibilants first (order matters — check before generic fricatives)
    ("tʃ",  "Affricates"),
    ("dʒ",  "Affricates"),
    ("ʃ",   "Sibilants"),
    ("ʒ",   "Sibilants"),
    ("s",   "Sibilants"),
    ("z",   "Sibilants"),
    # nasals
    ("ŋ",   "Nasals"),
    ("n̩",   "Nasals"),
    ("nʲ",  "Nasals"),
    ("m̩",   "Nasals"),
    ("n",   "Nasals"),
    ("m",   "Nasals"),
    # plosives / stops
    ("ʔ",   "Stops"),
    ("ɡʲ",  "Stops"),
    ("ɡ",   "Stops"),
    ("p",   "Stops"),
    ("b",   "Stops"),
    ("t",   "Stops"),
    ("d",   "Stops"),
    ("k",   "Stops"),
    # fricatives (non-sibilant)
    ("θ",   "Fricatives"),
    ("ð",   "Fricatives"),
    ("ɬ",   "Fricatives"),
    ("ç",   "Fricatives"),
    ("x",   "Fricatives"),
    ("f",   "Fricatives"),
    ("v",   "Fricatives"),
    ("h",   "Fricatives"),
    # approximants / liquids
    ("ɹ",   "Approximants"),
    ("ɾ",   "Approximants"),
    ("ʁ",   "Approximants"),
    ("əl",  "Approximants"),
    ("l",   "Approximants"),
    ("r",   "Approximants"),
    ("w",   "Approximants"),
    ("j",   "Approximants"),
    # diphthongs (check before monophthongs so multi-char symbols match)
    ("aɪɚ", "Diphthongs"),
    ("aɪə", "Diphthongs"),
    ("oʊ",  "Diphthongs"),
    ("eɪ",  "Diphthongs"),
    ("aɪ",  "Diphthongs"),
    ("aʊ",  "Diphthongs"),
    ("ɔɪ",  "Diphthongs"),
    ("iə",  "Diphthongs"),
    # vowels (monophthongs & r-coloured)
    ("ɚ",   "Vowels"),
    ("ɜː",  "Vowels"),
    ("ɛɹ",  "Vowels"),
    ("ɪɹ",  "Vowels"),
    ("ɔːɹ", "Vowels"),
    ("ɑːɹ", "Vowels"),
    ("ʊɹ",  "Vowels"),
    ("oːɹ", "Vowels"),
    ("iː",  "Vowels"),
    ("uː",  "Vowels"),
    ("ɪː",  "Vowels"),
    ("ɛː",  "Vowels"),
    ("ɔː",  "Vowels"),
    ("ɑː",  "Vowels"),
    ("oː",  "Vowels"),
    ("ɪ",   "Vowels"),
    ("ɛ",   "Vowels"),
    ("æ",   "Vowels"),
    ("ʌ",   "Vowels"),
    ("ɑ",   "Vowels"),
    ("ɔ",   "Vowels"),
    ("ʊ",   "Vowels"),
    ("ə",   "Vowels"),
    ("ᵻ",   "Vowels"),
    ("ɐ",   "Vowels"),
    ("ɜ",   "Vowels"),
    ("a",   "Vowels"),
    ("e",   "Vowels"),
    ("i",   "Vowels"),
    ("o",   "Vowels"),
    ("u",   "Vowels"),
    ("ææ",  "Vowels"),
]

# Class display order for plots
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]


def symbol_to_class(sym: str) -> str:
    if sym in SPECIAL or sym.isdigit():
        return "Other"
    for substr, cls in _CATEGORY_RULES:
        if substr in sym:
            return cls
    return "Other"


# ---------------------------------------------------------------------------
# Audio helpers  (same as test_model.py / plot_embeddings.py)
# ---------------------------------------------------------------------------

def _decode_audio_entry(entry: dict) -> torch.Tensor:
    raw_bytes = entry.get("bytes")
    path      = entry.get("path")
    if raw_bytes is not None:
        arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    elif path is not None:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio entry has neither 'bytes' nor 'path'.")
    wave = torch.from_numpy(arr)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    elif wave.ndim == 2:
        wave = wave.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wave = T.Resample(sr, TARGET_SR)(wave)
    return wave


def _crop_center(wave: torch.Tensor) -> torch.Tensor:
    n = wave.shape[-1]
    if n < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // n)
        wave = wave.repeat(1, reps)
    start = (wave.shape[-1] - TARGET_SAMPLES) // 2
    return wave[:, start : start + TARGET_SAMPLES]


def _label_to_int(raw) -> int:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "bonafide", "real", "genuine"):
            return 0
        return 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SoundfileHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, limit, token, seed=42):
        from datasets import load_dataset, Audio as HFAudio

        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))

        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower() == "key")),
            "label",
        )

        labels = [_label_to_int(self.ds[i][self.label_key]) for i in range(len(self.ds))]

        idx0 = [i for i, y in enumerate(labels) if y == 0]
        idx1 = [i for i, y in enumerate(labels) if y == 1]
        rng  = random.Random(seed)
        rng.shuffle(idx0); rng.shuffle(idx1)
        k = min(len(idx0), len(idx1), limit // 2)

        self.indices = idx0[:k] + idx1[:k]
        rng.shuffle(self.indices)

        counts = Counter(_label_to_int(self.ds[i][self.label_key]) for i in self.indices)
        print(f"Holdout '{split}': {len(self.indices)} examples  "
              f"({counts[0]} normal  +  {counts[1]} deepfake)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex   = self.ds[self.indices[idx]]
        wave = _decode_audio_entry(ex["audio"])
        wave = _crop_center(wave)
        y    = _label_to_int(ex[self.label_key])
        return {"audio": wave, "label": torch.tensor(y, dtype=torch.long), "sample_rate": TARGET_SR}


# ---------------------------------------------------------------------------
# Phoneme-loader patch
# ---------------------------------------------------------------------------

def patch_phoneme_loader() -> None:
    import phoneme_GAT.modules as modules_mod
    import phoneme_GAT.phoneme_model as pm_mod
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

    pm_mod.load_phoneme_model     = _load
    modules_mod.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Hook to capture per-phoneme post-GAT embeddings
# ---------------------------------------------------------------------------

class PhonemeCapture:
    """
    Monkey-patches Phoneme_GAT.encoder_and_GAT to save, after each call:
      .last_reduced_hidden_states  — (total_phonemes_in_batch, 768) post-GAT
      .last_reduced_phoneme_ids    — (B, max_phonemes)
      .last_reduced_num_frames     — (B,)
    """

    def __init__(self, gat_model):
        self.model = gat_model
        self._orig_fn = gat_model.encoder_and_GAT.__func__
        capture = self

        def patched_encoder_and_GAT(self_inner, hidden_states, num_frames, phoneme_ids,
                                    profiler=None, use_encoder=True, ground_truth_labels=None):
            result = capture._orig_fn(
                self_inner, hidden_states, num_frames, phoneme_ids,
                profiler=profiler, use_encoder=use_encoder,
                ground_truth_labels=ground_truth_labels,
            )
            # result = (hidden_states, reduced_hidden_states, reduced_phoneme_ids,
            #           reduced_num_frames, encoder_feat, logit)
            capture.last_reduced_hidden_states = result[1].detach().cpu()
            capture.last_reduced_phoneme_ids   = result[2].detach().cpu()
            capture.last_reduced_num_frames    = result[3].detach().cpu()
            return result

        import types
        gat_model.encoder_and_GAT = types.MethodType(patched_encoder_and_GAT, gat_model)

        self.last_reduced_hidden_states = None
        self.last_reduced_phoneme_ids   = None
        self.last_reduced_num_frames    = None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_violin_plot(pc1_by_class: dict[str, dict[int, list[float]]],
                     out_path: Path, n_samples: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = [c for c in CLASS_ORDER if c in pc1_by_class]
    n_cls   = len(classes)
    fig, ax = plt.subplots(figsize=(max(10, n_cls * 1.4), 6))

    gap = 0.38
    for ci, cls in enumerate(classes):
        x_base = ci
        for label, color, offset in [(0, "#6EB5FF", -gap/2), (1, "#E03030", gap/2)]:
            vals = pc1_by_class[cls].get(label, [])
            if len(vals) < 4:
                if vals:
                    ax.scatter([x_base + offset] * len(vals), vals,
                               c=color, s=18, alpha=0.7, zorder=3)
                continue
            vp = ax.violinplot([vals], positions=[x_base + offset],
                               widths=gap * 0.85, showmedians=True,
                               showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.65)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(classes, fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("PC1 projection  (post-GAT embedding space)", fontsize=11)
    ax.set_title(
        f"PC1 distribution by phoneme class  |  n={n_samples} utterances\n"
        "Light blue = normal (bonafide)   ·   Red = deepfake (spoof)",
        fontsize=12,
    )
    # legend proxy
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#6EB5FF", label="Normal (bonafide)"),
        Patch(facecolor="#E03030", label="Deepfake (spoof)"),
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def make_mean_delta_plot(pc1_by_class: dict[str, dict[int, list[float]]],
                          out_path: Path, n_samples: int) -> None:
    """
    Bar chart: mean(PC1|deepfake) - mean(PC1|normal) per phoneme class.
    Larger positive delta → class shifted toward fake end of PC1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = [c for c in CLASS_ORDER if c in pc1_by_class]
    deltas  = []
    errors  = []
    for cls in classes:
        real  = pc1_by_class[cls].get(0, [])
        fake  = pc1_by_class[cls].get(1, [])
        if real and fake:
            delta = np.mean(fake) - np.mean(real)
            # pooled SEM for error bar
            sem = np.sqrt(np.var(fake) / max(len(fake), 1) + np.var(real) / max(len(real), 1))
        else:
            delta, sem = 0.0, 0.0
        deltas.append(delta)
        errors.append(sem)

    colors = ["#E03030" if d > 0 else "#6EB5FF" for d in deltas]

    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.3), 5))
    bars = ax.bar(classes, deltas, color=colors, alpha=0.8, yerr=errors,
                  error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "black"})
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Δ mean PC1  (deepfake − normal)", fontsize=11)
    ax.set_title(
        f"Per-class PC1 shift: deepfake vs normal  |  n={n_samples} utterances\n"
        "Positive = phoneme nodes pushed toward the deepfake pole of PC1",
        fontsize=12,
    )
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    # ── Vocab ────────────────────────────────────────────────────────────────
    id_to_sym = build_id_to_symbol()
    print(f"Combined vocab: {len(id_to_sym)} tokens")
    # Spot check
    for test_id in [0, 5, 96, 170, 686]:
        print(f"  ID {test_id:3d} → '{id_to_sym.get(test_id, '?')}'")

    # ── Dataset ──────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SoundfileHFDataset(
        hf_name=HF_DATASET, split="validation",
        cache_dir=str(CACHE_DIR), limit=N_SAMPLES,
        token=hf_token, seed=SEED,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit

    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=True, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True,
    ))

    print(f"\nLoading checkpoint: {DEFAULT_CKPT}")
    lit_model = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location="cpu", strict=True,
    )
    lit_model.eval()
    lit_model.freeze()
    gat_model = lit_model.model     # Phoneme_GAT instance
    print("Model ready.\n")

    # Install capture hook
    capture = PhonemeCapture(gat_model)

    # ── Collect post-GAT pooled embeddings (for PCA) and per-phoneme data ───
    pooled_embs: list[np.ndarray] = []
    pooled_labels: list[int]      = []

    # Per-phoneme: list of (embedding_768, phoneme_id, sample_label)
    phoneme_records: list[tuple[np.ndarray, int, int]] = []

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"]
            labels = batch["label"]
            B      = labels.shape[0]
            num_frames = torch.full((B,), TARGET_SAMPLES // 320 - 1)

            out = gat_model(audio, num_frames, profiler=None, use_aug=False, stage="eval")

            # Pooled (B, 768) for PCA fit
            pooled_embs.append(out["hidden_states"].numpy())
            pooled_labels.extend(labels.tolist())

            # Per-phoneme from the capture hook
            # reduced_hidden_states: (total_phonemes_in_batch, 768)
            # reduced_phoneme_ids:   (B, max_phonemes)
            # reduced_num_frames:    (B,)
            rhs  = capture.last_reduced_hidden_states   # (P, 768)
            rids = capture.last_reduced_phoneme_ids     # (B, Lmax)
            rnf  = capture.last_reduced_num_frames      # (B,)

            cursor = 0
            for i in range(B):
                n_phonemes = int(rnf[i].item())
                sample_embs  = rhs[cursor : cursor + n_phonemes]   # (n_phonemes, 768)
                sample_pids  = rids[i, :n_phonemes]                 # (n_phonemes,)
                sample_label = int(labels[i].item())
                for emb, pid in zip(sample_embs, sample_pids):
                    phoneme_records.append((emb.numpy(), int(pid.item()), sample_label))
                cursor += n_phonemes

    pooled_embs_np  = np.concatenate(pooled_embs, axis=0)     # (N, 768)
    pooled_labels_np = np.array(pooled_labels)

    print(f"Pooled embeddings : {pooled_embs_np.shape}")
    print(f"Phoneme records   : {len(phoneme_records)}")

    # ── Fit PCA on pooled embeddings so PC1 = classification axis ────────────
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(pooled_embs_np)
    var = pca.explained_variance_ratio_
    print(f"PCA variance: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")

    # ── Project per-phoneme embeddings onto PC1 ───────────────────────────────
    phoneme_embs   = np.stack([r[0] for r in phoneme_records])      # (P, 768)
    phoneme_pids   = np.array([r[1] for r in phoneme_records])      # (P,)
    phoneme_labels = np.array([r[2] for r in phoneme_records])      # (P,)

    pc1_coords = pca.transform(phoneme_embs)[:, 0]                  # (P,)

    # ── Assign phoneme class labels ───────────────────────────────────────────
    phoneme_classes = np.array([symbol_to_class(id_to_sym.get(pid, "?"))
                                 for pid in phoneme_pids])

    # Count & report
    class_counts = Counter(phoneme_classes)
    print("\nPhoneme class distribution (node count):")
    for cls in CLASS_ORDER:
        if cls in class_counts:
            print(f"  {cls:15s}: {class_counts[cls]:6d} nodes")

    # Organise into dict[class][label] = [pc1_values]
    pc1_by_class: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pc1, cls, lbl in zip(pc1_coords, phoneme_classes, phoneme_labels):
        pc1_by_class[cls][lbl].append(float(pc1))

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_violin_plot(
        pc1_by_class,
        out_path=RESULTS_DIR / "phoneme_pc1_violin.png",
        n_samples=len(dataset),
    )
    make_mean_delta_plot(
        pc1_by_class,
        out_path=RESULTS_DIR / "phoneme_pc1_mean_delta.png",
        n_samples=len(dataset),
    )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── PC1 mean ± std by class ─────────────────────────────────────────")
    print(f"{'Class':15s}  {'Normal mean':>12s}  {'Fake mean':>10s}  {'Δ (fake−normal)':>16s}  {'n nodes':>8s}")
    print("-" * 70)
    for cls in CLASS_ORDER:
        if cls not in pc1_by_class:
            continue
        real = pc1_by_class[cls].get(0, [])
        fake = pc1_by_class[cls].get(1, [])
        if not real or not fake:
            continue
        delta = np.mean(fake) - np.mean(real)
        n     = len(real) + len(fake)
        print(f"{cls:15s}  {np.mean(real):+12.4f}  {np.mean(fake):+10.4f}  {delta:+16.4f}  {n:8d}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
