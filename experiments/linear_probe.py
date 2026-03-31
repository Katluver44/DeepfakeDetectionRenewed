"""
linear_probe.py
===============
Train linear probes on representations extracted at different points in the
Phoneme_GAT pipeline to localise where classification-relevant structure lives.

Representations extracted (all mean-pooled to (N, D) per utterance):
  bilstm           — post-BiLSTM + mean pool (model's own classification input, 768-d)
  gat_l{0,1,2}     — post-GAT layer 0/1/2, before BiLSTM, mean-pooled over phoneme nodes (768-d)
  head_l{0-2}_h{0-5} — single attention head within each GAT layer, mean-pooled (128-d each)

Each representation gets a logistic-regression linear probe (L2, class-weighted).
Results are ranked by AUC.

Data:  validation split of Bisher/ASVspoof_2019_LA, 500 samples per system
       (same as gat_attention_by_system.py; model never trained on validation).
Split: stratified 80/20 probe-train / probe-test (within validation only).

Saves to experiments/results/linear_probe/:
  features_<name>.npz          — X (N,D), y (N,), system_ids (N,)
  probe_<name>.npz             — coef (D,), intercept (), full metrics dict
  probe_metrics_summary.csv    — one row per probe, ranked by AUC
  probe_ranking.png            — AUC bar chart, colour-coded by probe type
"""
from __future__ import annotations

import csv
import io
import json
import pickle
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# ── torch.load compat ────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _patch(*a, **kw): kw.setdefault("weights_only", False); return _orig_torch_load(*a, **kw)
torch.load = _patch
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

REPO_ROOT       = Path(__file__).resolve().parents[1]
RESULTS_DIR     = Path(__file__).resolve().parent / "results" / "linear_probe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT  = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
BATCH_SIZE     = 10
PER_SYSTEM     = 500
SEED           = 42
TEST_FRAC      = 0.20    # fraction held out for probe evaluation
N_GAT_LAYERS   = 3
N_HEADS        = 6
HEAD_DIM       = 128     # features per head per GAT layer


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _decode(entry):
    raw = entry.get("bytes"); path = entry.get("path")
    arr, sr = (sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
               if raw is not None else sf.read(path, dtype="float32", always_2d=False))
    w = torch.tensor(arr)  # torch.from_numpy fails w/ NumPy 2.x + torch built for NumPy 1.x
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
# Dataset — same as gat_attention_by_system.py
# ---------------------------------------------------------------------------
class SystemStratifiedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, per_system, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        ex0 = self.ds[0]
        self.label_key = next(
            (k for k in ex0 if k != "audio" and ("label" in k.lower() or k.lower()=="key")),
            "label")
        by_system = defaultdict(list)
        for i in range(len(self.ds)):
            by_system[self.ds[i].get("system_id","unknown")].append(i)
        rng = random.Random(seed)
        self.indices, self.sys_ids = [], []
        for sid, idxs in sorted(by_system.items()):
            rng.shuffle(idxs)
            chosen = idxs[:per_system] if per_system else idxs
            self.indices.extend(chosen); self.sys_ids.extend([sid]*len(chosen))
        print(f"Dataset: {len(self.indices)} samples  "
              f"({len(set(self.sys_ids))} systems, ≤{per_system} each)")

    def __len__(self): return len(self.indices)
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
# Phoneme-loader patch
# ---------------------------------------------------------------------------
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
    pm.load_phoneme_model = _load; mm.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Capture harness
# ---------------------------------------------------------------------------
class RepresentationCapture:
    """
    Installs hooks to capture per-batch representations at:
      - output of each GATLayer  (3 hooks)
      - output of linear_proj inside each GATLayer (for per-head slicing)
    Also monkey-patches encoder_and_GAT to capture reduced_num_frames.

    After each forward pass, call .collect(batch_labels, batch_sids) to
    accumulate into self.store[repr_name] = list of (feat_vec, label, sid).
    """
    def __init__(self, gat_model):
        import types
        self._rnf: torch.Tensor | None = None       # reduced_num_frames for current batch
        self._layer_out: list[torch.Tensor]  = [None]*N_GAT_LAYERS
        self._head_out:  list[torch.Tensor]  = [None]*N_GAT_LAYERS  # (N, NH*FOUT) pre-act
        self._bilstm_out: torch.Tensor | None = None

        self.store: dict[str, list] = defaultdict(list)
        self._hooks = []

        # --- patch encoder_and_GAT to grab rnf and post-bilstm reps ----------
        capture = self
        orig_fn = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig_fn(self_inner, hidden_states, num_frames, phoneme_ids,
                             profiler=profiler, use_encoder=use_encoder,
                             ground_truth_labels=ground_truth_labels)
            # result[3] = reduced_num_frames (B,)
            capture._rnf = result[3].detach().cpu()
            # result[0] = hidden_states (B, 768)  post-GAT+BiLSTM+pool+L2norm
            capture._bilstm_out = result[0].detach().cpu()
            return result

        gat_model.encoder_and_GAT = types.MethodType(_patched, gat_model)

        # --- hook each GATLayer forward output (post-activation) --------------
        for i, layer in enumerate(gat_model.GAT.gat_net):
            def _make_layer_hook(idx):
                def _h(module, inp, out):
                    # out = (node_features (N, 768), edge_index)
                    capture._layer_out[idx] = out[0].detach().cpu()
                return _h
            self._hooks.append(layer.register_forward_hook(_make_layer_hook(i)))

        # --- hook linear_proj inside each GATLayer (pre-attention, pre-act) --
        # linear_proj output shape: (N, NH*FOUT) = (N, 768)
        # We slice into 6 x 128 heads after capturing.
        for i, layer in enumerate(gat_model.GAT.gat_net):
            def _make_proj_hook(idx):
                def _h(module, inp, out):
                    capture._head_out[idx] = out.detach().cpu()  # (N, 768)
                return _h
            self._hooks.append(layer.linear_proj.register_forward_hook(_make_proj_hook(i)))

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def collect(self, labels: list[int], sids: list[str]) -> None:
        """Mean-pool node reps per sample and append to store."""
        if self._rnf is None: return
        rnf = self._rnf          # (B,)
        B   = len(labels)

        # --- post-BiLSTM (already pooled per sample) -------------------------
        if self._bilstm_out is not None:
            for i in range(B):
                self.store["bilstm"].append(
                    (np.array(self._bilstm_out[i].cpu().detach().tolist()), labels[i], sids[i]))

        # --- post-GAT per-layer (node-level → mean pool per sample) ----------
        cursor = 0
        for i in range(B):
            n = int(rnf[i].item())
            for li in range(N_GAT_LAYERS):
                if self._layer_out[li] is not None:
                    seg = self._layer_out[li][cursor : cursor + n]  # (n, 768)
                    self.store[f"gat_l{li}"].append(
                        (np.array(seg.mean(0).cpu().detach().tolist()), labels[i], sids[i]))

                if self._head_out[li] is not None:
                    seg = self._head_out[li][cursor : cursor + n]   # (n, 768)
                    seg_hd = seg.view(n, N_HEADS, HEAD_DIM)          # (n, 6, 128)
                    for h in range(N_HEADS):
                        self.store[f"head_l{li}_h{h}"].append(
                            (np.array(seg_hd[:, h, :].mean(0).cpu().detach().tolist()), labels[i], sids[i]))
            cursor += n


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
def run_probe(name: str, X: np.ndarray, y: np.ndarray,
              sids: np.ndarray, train_mask: np.ndarray) -> dict:
    """
    Fit LogisticRegression on X[train_mask] and evaluate on X[~train_mask].
    Returns metrics dict + saves .npz to RESULTS_DIR.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                  f1_score, precision_score, recall_score,
                                  average_precision_score)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_mask])
    X_te = scaler.transform(X[~train_mask])
    y_tr = y[train_mask]; y_te = y[~train_mask]

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="lbfgs", C=1.0, random_state=SEED)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "name":       name,
        "n_train":    int(train_mask.sum()),
        "n_test":     int((~train_mask).sum()),
        "n_features": X.shape[1],
        "auc":        float(roc_auc_score(y_te, proba)),
        "ap":         float(average_precision_score(y_te, proba)),
        "accuracy":   float(accuracy_score(y_te, preds)),
        "f1":         float(f1_score(y_te, preds, zero_division=0)),
        "precision":  float(precision_score(y_te, preds, zero_division=0)),
        "recall":     float(recall_score(y_te, preds, zero_division=0)),
    }

    # Coefficient norms (one per feature dimension) — proxy for which dims matter
    coef = clf.coef_[0]          # (D,)
    coef_l1 = float(np.abs(coef).sum())
    coef_max = float(np.abs(coef).max())
    metrics["coef_l1"] = coef_l1
    metrics["coef_max"] = coef_max

    # Save probe weights + scaler + features
    np.savez_compressed(
        RESULTS_DIR / f"probe_{name}.npz",
        coef=coef,
        intercept=clf.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        **{k: v for k, v in metrics.items()
           if isinstance(v, (int, float, str))},
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = SystemStratifiedDataset(HF_DATASET, "validation", str(CACHE_DIR),
                                        hf_token, PER_SYSTEM, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    print(f"Loading checkpoint: {DEFAULT_CKPT}")
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()   # verify driver actually works
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except RuntimeError:
        device = torch.device("cpu")
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(DEFAULT_CKPT), cfg=cfg, map_location=device, strict=True)
    lit.to(device)
    lit.eval(); lit.freeze()
    gat_model = lit.model
    print(f"Model ready on {device}.\n")

    # ── Install capture harness ────────────────────────────────────────────────
    capture = RepresentationCapture(gat_model)

    # ── Feature extraction loop ────────────────────────────────────────────────
    print("Extracting representations...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            audio  = batch["audio"].to(device)
            labels = batch["label"].tolist()
            sids   = batch["system_id"]
            B      = len(labels)
            nf     = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)

            _ = gat_model(audio, nf, profiler=None, use_aug=False, stage="eval")
            capture.collect(labels, sids)

            if (bi+1) % 50 == 0:
                print(f"  {(bi+1)*BATCH_SIZE}/{len(dataset)} samples done")

    capture.remove_hooks()
    print(f"Done. Representations collected: {list(capture.store.keys())}\n")

    # ── Build arrays + stratified train/test split ────────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit

    # We process all probes; collect results for ranking
    all_metrics: list[dict] = []

    repr_names = sorted(capture.store.keys(),
                        key=lambda n: (0 if n=="bilstm" else
                                       1 if n.startswith("gat") else 2,
                                       n))

    print(f"Training {len(repr_names)} linear probes...")
    print(f"{'Probe':25s}  {'AUC':>6s}  {'ACC':>6s}  {'F1':>6s}  {'D':>5s}")
    print("-" * 55)

    for name in repr_names:
        records = capture.store[name]
        X   = np.stack([r[0] for r in records]).astype(np.float32)
        y   = np.array([r[1] for r in records], dtype=np.int32)
        sid = np.array([r[2] for r in records])

        # Save raw features
        np.savez_compressed(
            RESULTS_DIR / f"features_{name}.npz",
            X=X, y=y, system_ids=sid)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                     random_state=SEED)
        train_idx, test_idx = next(sss.split(X, y))
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[train_idx] = True

        m = run_probe(name, X, y, sid, train_mask)
        all_metrics.append(m)
        print(f"  {name:25s}  {m['auc']:.4f}  {m['accuracy']:.4f}  "
              f"{m['f1']:.4f}  {m['n_features']:5d}")

    # ── Rank by AUC ───────────────────────────────────────────────────────────
    all_metrics.sort(key=lambda m: m["auc"], reverse=True)

    print(f"\n{'Rank':>5s}  {'Probe':25s}  {'AUC':>6s}  {'ACC':>6s}  {'F1':>6s}  {'D':>5s}")
    print("-" * 60)
    for rank, m in enumerate(all_metrics, 1):
        print(f"  {rank:3d}  {m['name']:25s}  {m['auc']:.4f}  {m['accuracy']:.4f}  "
              f"{m['f1']:.4f}  {m['n_features']:5d}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "probe_metrics_summary.csv"
    fieldnames = ["rank", "name", "auc", "ap", "accuracy", "f1",
                  "precision", "recall", "n_features",
                  "n_train", "n_test", "coef_l1", "coef_max"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rank, m in enumerate(all_metrics, 1):
            w.writerow({"rank": rank, **m})
    print(f"\nSaved: {csv_path.name}")

    # ── Plot: AUC ranking bar chart ───────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names  = [m["name"] for m in all_metrics]
    aucs   = [m["auc"]  for m in all_metrics]
    accs   = [m["accuracy"] for m in all_metrics]

    def _color(n):
        if n == "bilstm":       return "#2E86AB"   # blue  — BiLSTM
        if n.startswith("gat"): return "#A23B72"   # purple — GAT layer
        # heads: colour by layer
        li = int(n.split("_h")[0].split("gat_l" if "gat" in n else "head_l")[1][0])
        return ["#F18F01","#C73E1D","#3B1F2B"][li]

    colors = [_color(n) for n in names]
    n = len(names)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n*0.55), 9))

    for ax, vals, ylabel, title in [
        (axes[0], aucs, "AUC-ROC", "Linear probe AUC by representation"),
        (axes[1], accs, "Accuracy", "Linear probe Accuracy by representation"),
    ]:
        bars = ax.bar(range(n), vals, color=colors, alpha=0.88, edgecolor="black", linewidth=0.4)
        ax.axhline(0.5, color="grey", linewidth=0.9, linestyle="--", label="Chance (0.5)")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0.4, 1.02)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    legend_patches = [
        Patch(color="#2E86AB", label="BiLSTM output"),
        Patch(color="#A23B72", label="GAT layer (full, 768-d)"),
        Patch(color="#F18F01", label="GAT layer 0 head (128-d)"),
        Patch(color="#C73E1D", label="GAT layer 1 head (128-d)"),
        Patch(color="#3B1F2B", label="GAT layer 2 head (128-d)"),
    ]
    axes[0].legend(handles=legend_patches, fontsize=8, loc="lower right")
    fig.suptitle(
        f"Linear probe ranking — Bisher/ASVspoof_2019_LA validation\n"
        f"n={len(dataset)} utterances, {TEST_FRAC*100:.0f}% probe-test holdout, "
        f"stratified split, class-balanced LR",
        fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR.parent / "probe_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: probe_ranking.png")

    # ── Per-layer + per-head AUC summary ─────────────────────────────────────
    print("\n── Layer-level probes ─────────────────────────────────")
    for m in all_metrics:
        if m["name"] in ("bilstm","gat_l0","gat_l1","gat_l2"):
            print(f"  {m['name']:15s}  AUC={m['auc']:.4f}  ACC={m['accuracy']:.4f}")

    print("\n── Head AUC — ranked within each layer ────────────────")
    for li in range(N_GAT_LAYERS):
        head_ms = sorted(
            [m for m in all_metrics if m["name"].startswith(f"head_l{li}_")],
            key=lambda m: m["auc"], reverse=True)
        print(f"\n  Layer {li}:")
        for rank, m in enumerate(head_ms, 1):
            h = m["name"].split("_h")[1]
            print(f"    Head {h} (rank {rank})  AUC={m['auc']:.4f}  ACC={m['accuracy']:.4f}")

    print(f"\nAll probe artefacts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
