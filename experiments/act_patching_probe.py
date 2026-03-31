"""
act_patching_probe.py
=====================
Layer-0 GAT activation patching measured against the 6-way multiclass vocoder probe.

For each phoneme node i in a spoof utterance x with true system s:

  ΔP_i[k] = P_base[k] − P_patched_i[k]   (k ∈ {A01,...,A06})

where P_base = probe(mean(h0))  and
      P_patched_i = probe( mean(h0) + (bonafide_mean[c_i] − h0[i]) / N )

Because patching one node only shifts the mean by 1/N, no additional model
forward pass is needed — the patched mean is computed analytically.

Interpretation:
  ΔP_i[s]  > 0  →  node i contributed to the correct vocoder ID (patching reduces it)
  ΔP_i[k≠s] < 0  →  node i was suppressing system k (patching increases that system)

The "shift matrix" S[c, k] = mean over all nodes of class c of ΔP_i[k]
answers: "when we make class-c nodes look bonafide, which vocoder does the probe
stop predicting, and which does it start predicting instead?"

Saves to experiments/results/act_patching_probe/:
  act_probe_records.npz          — per-node (cls, system_id, delta_true, delta_all_6)
  act_probe_class_stats.csv      — per-class: mean Δ_true, mean shift to each system
  act_probe_shift_matrix.png     — 9-class × 6-system heatmap of mean ΔP
  act_probe_true_system_delta.png — mean Δ_true bar chart per phoneme class, split by system
  act_probe_confusion_shift.png  — for each system: how does patching redistribute prob mass?
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
import torch                 # must precede scipy/numpy to avoid NumPy C-API conflict
import torchaudio.transforms as T
import numpy as np
from scipy.special import softmax
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUT_DIR = EXPERIMENTS_DIR / "results" / "act_patching_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT   = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET     = "Bisher/ASVspoof_2019_LA"
CACHE_DIR      = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH  = REPO_ROOT / "secret.txt"
VOCAB_DIR      = REPO_ROOT / "vocab_phoneme"
PROBE_PATH     = EXPERIMENTS_DIR / "results" / "multiclass_probe" / "probe_mc_gat_l0.npz"
BON_MEANS_PATH = EXPERIMENTS_DIR / "results" / "act_patching" / "bonafide_class_means.npz"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
BATCH_SIZE     = 8
PER_SYSTEM     = 500
SEED           = 42

LANG_ORDER = ["de","en","es","fr","it","pl","ru","uk","zh-CN"]
SPECIAL    = ["|","</s>","<s>","<unk>","<pad>"]
CLASS_ORDER = ["Vowels","Diphthongs","Approximants","Nasals",
               "Stops","Fricatives","Sibilants","Affricates","Other"]
SYSTEMS     = ["A01","A02","A03","A04","A05","A06"]

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


def build_id_to_class():
    total = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists(): continue
        vocab = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id2sym = {i: (e if i < 5 else e.split("-",1)[1]) for i,e in enumerate(total)}
    return {i: sym_to_cls(s) for i,s in id2sym.items()}


def _decode(entry):
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
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
    return w[:, s:s+TARGET_SAMPLES]


def _lbl(raw):
    if isinstance(raw, str):
        return 0 if raw.strip().lower() in ("0","bonafide","real","genuine") else 1
    return int(raw)


class SystemStratifiedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, per_system, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower()=="key")),
            "label")
        by_sys = defaultdict(list)
        for i in range(len(self.ds)):
            by_sys[self.ds[i].get("system_id","unknown")].append(i)
        rng = random.Random(seed)
        self.indices, self.sys_ids = [], []
        for sid, idxs in sorted(by_sys.items()):
            rng.shuffle(idxs)
            chosen = idxs[:per_system] if per_system else idxs
            self.indices.extend(chosen); self.sys_ids.extend([sid]*len(chosen))
        print(f"Dataset '{split}': {len(self.indices)} samples ({len(by_sys)} systems)")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        return {"audio": _crop(_decode(ex["audio"])),
                "label": torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
                "system_id": ex.get("system_id","unknown"), "sample_rate": TARGET_SR}


def collate_fn(batch):
    return {"audio": torch.stack([b["audio"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "system_id": [b["system_id"] for b in batch],
            "sample_rate": TARGET_SR}


def patch_phoneme_loader():
    import phoneme_GAT.modules as mm, phoneme_GAT.phoneme_model as pm
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
    pm.load_phoneme_model = mm.load_phoneme_model = _load


def capture_h0(lit, audio, device):
    """Run one batch; return h0 (N_total,768), node_pids (N_total,), rnf (B,)."""
    gat_model = lit.model
    store = {}

    def _hook(module, inp, out):
        store["h0"]    = out[0].detach().cpu()   # keep on CPU – we'll do probe math in numpy
        store["ei"]    = out[1].detach().cpu()

    hook = gat_model.GAT.gat_net[0].register_forward_hook(_hook)
    orig = gat_model.encoder_and_GAT

    def _patched(*args, **kwargs):
        res = orig(*args, **kwargs)
        _, _, rpids, rnf, _, _ = res
        store["node_pids"] = torch.cat(
            [rpids[i, :rnf[i]] for i in range(len(rnf))]).detach().cpu()
        store["rnf"] = rnf.detach().cpu()
        return res

    gat_model.encoder_and_GAT = _patched
    try:
        x = audio.squeeze(1) if audio.ndim == 3 else audio
        B = x.shape[0]
        nf = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)
        with torch.no_grad():
            gat_model(x.to(device), nf, use_aug=False, stage="eval")
    finally:
        hook.remove()
        gat_model.encoder_and_GAT = orig

    to_np = lambda t: np.array(t.tolist())
    return to_np(store["h0"]), to_np(store["node_pids"]), to_np(store["rnf"]).astype(int)


# ---------------------------------------------------------------------------
# Probe application (pure numpy — fast)
# ---------------------------------------------------------------------------

class MulticlassProbe:
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.coef      = d["coef"].astype(np.float64)      # (6, 768)
        self.intercept = d["intercept"].astype(np.float64) # (6,)
        self.mu        = d["scaler_mean"].astype(np.float64)
        self.sigma     = d["scaler_scale"].astype(np.float64)
        self.classes   = list(d["classes"])

    def proba(self, x: np.ndarray) -> np.ndarray:
        """x: (768,) or (M, 768) → (6,) or (M, 6) probabilities."""
        z = (x - self.mu) / self.sigma
        logits = z @ self.coef.T + self.intercept   # (..., 6)
        return softmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shift_matrix(mat: np.ndarray, row_labels: list[str],
                      col_labels: list[str], title: str,
                      out_path: Path) -> None:
    """mat: (n_classes, n_systems), positive = patching reduces that system's prob."""
    fig, ax = plt.subplots(figsize=(9, 6))
    vabs = max(abs(mat).max(), 1e-4)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean ΔP (positive = patching reduces system prob)")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Vocoder system"); ax.set_ylabel("Phoneme class patched")
    ax.set_title(title, fontsize=11)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(mat[i,j]) > vabs*0.55 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_true_delta_bars(stats: dict, out_path: Path) -> None:
    """Bar chart: mean Δ_true per class (averaged over all systems), sorted descending."""
    classes = [c for c in CLASS_ORDER if stats.get(c)]
    means   = [np.mean([v[0] for v in stats[c]]) for c in classes]
    counts  = [len(stats[c]) for c in classes]
    order   = np.argsort(means)[::-1]
    classes = [classes[i] for i in order]
    means   = [means[i] for i in order]
    counts  = [counts[i] for i in order]

    colours = ["#E03030" if m > 0 else "#4C72B0" for m in means]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(classes)), means, color=colours, width=0.6, alpha=0.87)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, m + (0.0005 if m >= 0 else -0.0005),
                f"n={n}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Δ_true  (reduction in true-system probability)")
    ax.set_title("Mean causal effect on true-vocoder confidence per phoneme class\n"
                 "(positive = patching reduces true-system probability)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_per_system_shifts(shift_by_sys: dict[str, np.ndarray],
                           classes: list[str], out_path: Path) -> None:
    """
    6-panel figure (one per true system).  Each panel is a bar chart across
    phoneme classes showing mean ΔP[true_system] for that system's nodes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()
    for si, sys in enumerate(SYSTEMS):
        ax = axes[si]
        mat = shift_by_sys.get(sys)   # (n_classes, n_nodes) jagged → list of arrays
        if mat is None or len(mat) == 0:
            ax.set_title(sys); continue
        cls_means = []
        cls_labels = []
        for cls in CLASS_ORDER:
            vals = mat.get(cls, [])
            if vals:
                cls_means.append(float(np.mean(vals)))
                cls_labels.append(cls)
        colours = ["#E03030" if m > 0 else "#4C72B0" for m in cls_means]
        ax.bar(range(len(cls_labels)), cls_means, color=colours, width=0.6, alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xticks(range(len(cls_labels)))
        ax.set_xticklabels(cls_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{sys} — mean Δ_true per class", fontsize=9)
        ax.set_ylabel("Mean Δ_true", fontsize=8)
    fig.suptitle("Per-system causal effect on true-vocoder probability",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except RuntimeError:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load probe + bonafide means ──────────────────────────────────────────
    probe = MulticlassProbe(PROBE_PATH)
    print(f"Probe loaded: {probe.classes}")

    bm_raw = np.load(BON_MEANS_PATH, allow_pickle=True)
    class_means = {k: bm_raw[k].astype(np.float64) for k in bm_raw}
    print(f"Bonafide class means: {sorted(class_means.keys())}")

    id2cls = build_id_to_class()

    # ── Model ────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    print(f"Loading checkpoint: {DEFAULT_CKPT}")
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    print(f"Model ready on {device}.\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SystemStratifiedDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token, PER_SYSTEM, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    # ── Records ───────────────────────────────────────────────────────────────
    rec_cls      = []   # str
    rec_sid      = []   # str  (true system)
    rec_dtrue    = []   # float  ΔP[true_system]
    rec_dall     = []   # (6,) float  ΔP across all 6 systems

    n_utt_used = 0
    n_utt_skip = 0

    for batch in loader:
        sids   = batch["system_id"]
        labels = batch["label"]

        spoof_idx = [i for i, (s, l) in enumerate(zip(sids, labels))
                     if s in SYSTEMS and l.item() == 1]
        if not spoof_idx:
            continue

        audio_sp = batch["audio"][spoof_idx]
        sids_sp  = [sids[i] for i in spoof_idx]

        h0, node_pids, rnf = capture_h0(lit, audio_sp, device)
        # h0: (N_total, 768) numpy float32

        cursor = 0
        for b, sid in enumerate(sids_sp):
            n    = rnf[b]
            h0_b = h0[cursor:cursor+n].astype(np.float64)   # (n, 768)
            pids = node_pids[cursor:cursor+n]                # (n,)

            # base probe prediction for this utterance
            h0_mean_base = h0_b.mean(0)        # (768,)
            p_base = probe.proba(h0_mean_base)  # (6,)

            # Skip if probe doesn't predict the true system as top-1
            true_idx = probe.classes.index(sid)
            if p_base.argmax() != true_idx:
                n_utt_skip += 1
                cursor += n
                continue

            n_utt_used += 1

            # Analytic patching: mean_patched_i = mean_base + (bm[c] - h0[i]) / n
            for i in range(n):
                cls = id2cls.get(int(pids[i]), "Other")
                bm  = class_means.get(cls)
                if bm is None:
                    cursor_i_skip = True
                    continue

                mean_patched = h0_mean_base + (bm - h0_b[i]) / n   # (768,)
                p_patched    = probe.proba(mean_patched)             # (6,)

                delta_all  = p_base - p_patched                     # (6,)
                delta_true = float(delta_all[true_idx])

                rec_cls.append(cls)
                rec_sid.append(sid)
                rec_dtrue.append(delta_true)
                rec_dall.append(delta_all)

            cursor += n

    print(f"Utterances used: {n_utt_used}  (skipped {n_utt_skip} — probe wrong top-1)")
    print(f"Node records: {len(rec_dtrue)}")

    rec_dall_arr  = np.stack(rec_dall).astype(np.float32)    # (M, 6)
    rec_dtrue_arr = np.array(rec_dtrue, dtype=np.float32)    # (M,)
    rec_cls_arr   = np.array(rec_cls)
    rec_sid_arr   = np.array(rec_sid)

    # ── Save raw records ──────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_DIR / "act_probe_records.npz",
        cls_names  = rec_cls_arr,
        system_ids = rec_sid_arr,
        delta_true = rec_dtrue_arr,
        delta_all  = rec_dall_arr,
    )
    print("Saved: act_probe_records.npz")

    # ── Aggregate: shift matrix (n_classes × n_systems) ──────────────────────
    shift_mat    = np.zeros((len(CLASS_ORDER), len(SYSTEMS)))
    shift_counts = np.zeros((len(CLASS_ORDER), len(SYSTEMS)), dtype=int)
    dtrue_by_cls_sys = defaultdict(lambda: defaultdict(list))  # for per-system plot

    for cls, sid, dtrue, dall in zip(rec_cls_arr, rec_sid_arr, rec_dtrue_arr, rec_dall_arr):
        ci = CLASS_ORDER.index(cls) if cls in CLASS_ORDER else -1
        if ci < 0: continue
        shift_mat[ci]    += dall          # sum ΔP[k] for each k
        shift_counts[ci] += 1
        dtrue_by_cls_sys[sid][cls].append(float(dtrue))

    # Normalize to mean
    for ci in range(len(CLASS_ORDER)):
        n = shift_counts[ci, 0]   # same count for all k within a class
        if n > 0:
            shift_mat[ci] /= n

    # ── Print summary ─────────────────────────────────────────────────────────
    valid_classes = [c for c in CLASS_ORDER if shift_counts[CLASS_ORDER.index(c), 0] > 0]
    print(f"\n{'Class':<16}  {'Δ_true':>8}  " +
          "  ".join(f"Δ_{s:>3}" for s in SYSTEMS))
    print("─" * (16 + 10 + 8*6))
    for cls in CLASS_ORDER:
        ci = CLASS_ORDER.index(cls)
        n  = shift_counts[ci, 0]
        if n == 0: continue
        row = shift_mat[ci]
        # Δ_true = average of shift_mat[ci][true_system] weighted by how many nodes
        # had each system — simpler: just look at the per-class mean
        mean_true_from_mat = np.mean([shift_mat[ci][probe.classes.index(sid)]
                                      for sid in rec_sid_arr[rec_cls_arr==cls]
                                      if sid in probe.classes])
        print(f"  {cls:<14}  {mean_true_from_mat:+.5f}  " +
              "  ".join(f"{v:+.4f}" for v in row) + f"  (n={n})")

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT_DIR / "act_probe_class_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n", "mean_delta_true"] + [f"mean_delta_{s}" for s in SYSTEMS])
        for cls in CLASS_ORDER:
            ci = CLASS_ORDER.index(cls)
            n  = int(shift_counts[ci, 0])
            if n == 0: continue
            # mean_delta_true: weighted mean across nodes (each knows its own true system)
            mask = rec_cls_arr == cls
            mean_true = float(rec_dtrue_arr[mask].mean()) if mask.any() else 0.0
            w.writerow([cls, n, f"{mean_true:.6f}"] +
                       [f"{shift_mat[ci, si]:.6f}" for si in range(len(SYSTEMS))])
    print("Saved: act_probe_class_stats.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_shift_matrix(
        shift_mat, CLASS_ORDER, SYSTEMS,
        title="Mean ΔP per (phoneme class, vocoder system)\n"
              "Positive = patching that class reduces system probability",
        out_path=OUT_DIR / "act_probe_shift_matrix.png",
    )

    # Aggregate Δ_true per class (pooling across systems, weighted by node count)
    stats_for_bar = defaultdict(list)
    for cls, dtrue, sid in zip(rec_cls_arr, rec_dtrue_arr, rec_sid_arr):
        stats_for_bar[cls].append((float(dtrue), sid))
    plot_true_delta_bars(stats_for_bar, OUT_DIR / "act_probe_true_system_delta.png")

    plot_per_system_shifts(
        dtrue_by_cls_sys, CLASS_ORDER,
        OUT_DIR / "act_probe_per_system.png",
    )

    print(f"\nAll artefacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
