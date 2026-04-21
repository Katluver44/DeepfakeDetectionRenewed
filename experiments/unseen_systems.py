"""
unseen_systems.py
=================
Evaluate robust_goat.ckpt on ASVspoof 2019 LA test-split systems A07–A19
(never seen during training).  Also includes bonafide (-) and the six
training systems A01–A06 (from the validation split) for direct comparison.

For every system:
  - Accuracy, EER, AUC, min-DCF (p_target=0.05)
  - Post-GAT mean-pooled embeddings (gat_l0 feature, 768-d)

Visualisation:
  - PCA: scatter of all systems (first 200 samples per system for clarity)
  - t-SNE: same

Saves to experiments/results/unseen_systems/:
  metrics.csv               — per-system acc / EER / AUC / minDCF
  embeddings.npz            — X (N,768), system_ids (N,), labels (N,)
  pca_unseen.png            — PCA scatter, coloured by system
  tsne_unseen.png           — t-SNE scatter, coloured by system
"""
from __future__ import annotations

import io
import json
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

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
OUT_DIR = EXPERIMENTS_DIR / "results" / "unseen_systems"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT  = REPO_ROOT / "models" / "robust_goat.ckpt"
HF_DATASET    = "Bisher/ASVspoof_2019_LA"
CACHE_DIR     = REPO_ROOT / "data" / "asvspoof_2019_la"
HF_TOKEN_PATH = REPO_ROOT / "secret.txt"

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
BATCH_SIZE     = 16
PER_SYSTEM     = 500    # cap per system to keep runtime reasonable
SEED           = 42

# Training systems (from val split) + unseen test systems
TRAIN_SYSTEMS  = ["A01","A02","A03","A04","A05","A06"]
UNSEEN_SYSTEMS = ["A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]
BONAFIDE_ID    = "-"

# For PCA/t-SNE scatter: fixed colour per group
_CMAP_TRAIN   = plt.cm.Reds
_CMAP_UNSEEN  = plt.cm.Blues
_BONAFIDE_COL = "#2CA02C"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

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
        return 0 if raw.strip().lower() in ("0","bonafide","real","genuine","-") else 1
    return int(raw)


# ---------------------------------------------------------------------------
# Dataset — test split, stratified by system, with per-system cap
# ---------------------------------------------------------------------------

class TestSystemDataset(torch.utils.data.Dataset):
    def __init__(self, hf_name, split, cache_dir, token, systems, per_system, seed=42):
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir, token=token)
        self.ds = ds.cast_column("audio", HFAudio(decode=False))
        self.label_key = "key"   # ASVspoof test uses "key" field (bonafide/spoof)

        by_sys = defaultdict(list)
        for i in range(len(self.ds)):
            sid = self.ds[i].get("system_id", "?")
            if sid in systems:
                by_sys[sid].append(i)

        rng = random.Random(seed)
        self.indices, self.sys_ids = [], []
        sys_counts = {}
        for sid in sorted(by_sys):
            idxs = by_sys[sid]
            rng.shuffle(idxs)
            chosen = idxs[:per_system] if per_system else idxs
            self.indices.extend(chosen)
            self.sys_ids.extend([sid] * len(chosen))
            sys_counts[sid] = len(chosen)
        print(f"  {split} split: {len(self.indices)} samples from {len(sys_counts)} systems")
        for sid, cnt in sorted(sys_counts.items()):
            print(f"    {sid}: {cnt}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ex  = self.ds[self.indices[idx]]
        wav = _crop(_decode(ex["audio"]))
        y   = _lbl(ex[self.label_key])
        sid = ex.get("system_id", "?")
        return {"audio": wav, "label": torch.tensor(y, dtype=torch.long),
                "system_id": sid, "sample_rate": TARGET_SR}


def collate_fn(batch):
    return {"audio":     torch.stack([b["audio"] for b in batch]),
            "label":     torch.stack([b["label"] for b in batch]),
            "system_id": [b["system_id"] for b in batch],
            "sample_rate": TARGET_SR}


# ---------------------------------------------------------------------------
# Model loading
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
    pm.load_phoneme_model = mm.load_phoneme_model = _load


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate via linear interpolation."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_min_dcf(scores: np.ndarray, labels: np.ndarray,
                    p_target: float = 0.05, c_miss: float = 1.0,
                    c_fa: float = 1.0) -> float:
    """Minimum detection cost function (NIST convention)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(dcf.min() / norm)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2: return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_batch(lit, audio, device):
    """Return gat_l0 mean-pool (B,768), logits (B,)."""
    gat = lit.model
    store = {}

    def _hook(mod, inp, out):
        store["h0"]  = out[0]          # (N_total, 768)
    def _eag_patch(*args, **kwargs):
        res = gat._orig_eag(*args, **kwargs)
        _, _, rpids, rnf, _, _ = res
        store["rnf"] = rnf
        return res

    gat._orig_eag = gat.encoder_and_GAT
    gat.encoder_and_GAT = _eag_patch
    hk = gat.GAT.gat_net[0].register_forward_hook(_hook)
    try:
        x = audio.squeeze(1) if audio.ndim == 3 else audio
        B = x.shape[0]
        nf = torch.full((B,), TARGET_SAMPLES // 320 - 1, device=device)
        with torch.no_grad():
            out = gat(x.to(device), nf, use_aug=False, stage="eval")
    finally:
        hk.remove()
        gat.encoder_and_GAT = gat._orig_eag
        del gat._orig_eag

    h0  = store["h0"]          # (N_total, 768)
    rnf = store["rnf"]         # (B,)
    # mean pool per sample
    embs = []
    cursor = 0
    for b in range(B):
        n = int(rnf[b])
        embs.append(h0[cursor:cursor+n].mean(0))
        cursor += n
    embs = torch.stack(embs)   # (B, 768)

    logits = out["logit"]      # (B,)
    to_np = lambda t: np.array(t.cpu().detach().tolist())
    return to_np(embs), to_np(logits)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _system_colour(sid, train_systems, unseen_systems):
    if sid == BONAFIDE_ID:
        return _BONAFIDE_COL
    elif sid in train_systems:
        idx = train_systems.index(sid) / max(len(train_systems) - 1, 1)
        return _CMAP_TRAIN(0.35 + 0.55 * idx)
    else:
        idx = unseen_systems.index(sid) / max(len(unseen_systems) - 1, 1)
        return _CMAP_UNSEEN(0.35 + 0.55 * idx)


def scatter_plot(proj: np.ndarray, sids: np.ndarray,
                 train_systems, unseen_systems,
                 title: str, xlabel: str, ylabel: str,
                 out_path: Path) -> None:
    all_systems = [BONAFIDE_ID] + sorted(train_systems) + sorted(unseen_systems)
    fig, ax = plt.subplots(figsize=(12, 9))
    for sid in all_systems:
        mask = sids == sid
        if not mask.any(): continue
        col   = _system_colour(sid, train_systems, unseen_systems)
        label = sid if sid != BONAFIDE_ID else "Bonafide"
        marker = "^" if sid == BONAFIDE_ID else ("s" if sid in train_systems else "o")
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[col], s=18, alpha=0.65, label=label, marker=marker,
                   edgecolors="none")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=3, markerscale=1.5,
              loc="upper right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_metrics(df_rows: list[dict], out_path: Path) -> None:
    """Grouped bar chart: AUC and (1-EER) per system."""
    systems  = [r["system_id"] for r in df_rows]
    aucs     = [float(r["auc"]) for r in df_rows]
    one_eer  = [1 - float(r["eer"]) for r in df_rows]
    x = np.arange(len(systems))

    train_set  = set(TRAIN_SYSTEMS)
    unseen_set = set(UNSEEN_SYSTEMS)
    colours_auc = []
    for s in systems:
        if s == BONAFIDE_ID: colours_auc.append("#2CA02C")
        elif s in train_set: colours_auc.append("#D62728")
        else:                colours_auc.append("#1F77B4")

    fig, ax = plt.subplots(figsize=(16, 5))
    w = 0.38
    ax.bar(x - w/2, aucs,    width=w, color=colours_auc, alpha=0.85, label="AUC")
    ax.bar(x + w/2, one_eer, width=w, color=colours_auc, alpha=0.45,
           hatch="//", label="1 − EER")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUC  /  1−EER")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-system AUC and 1−EER\n"
                 "Red = trained (val split)  |  Blue = unseen (test split)  |  "
                 "Solid = AUC  |  Hatched = 1−EER", fontsize=10)

    # Add value labels
    for i, (a, e) in enumerate(zip(aucs, one_eer)):
        ax.text(i - w/2, a + 0.005, f"{a:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i + w/2, e + 0.005, f"{e:.2f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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

    # ── Model ─────────────────────────────────────────────────────────────────
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

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

    # ── Build combined dataset ─────────────────────────────────────────────────
    # Test split: bonafide + A07-A19 (unseen) + A01-A06 not in test, use val for comparison
    all_test_systems = [BONAFIDE_ID] + UNSEEN_SYSTEMS

    print("Loading test split (unseen systems + bonafide)...")
    test_ds = TestSystemDataset(
        HF_DATASET, "test", str(CACHE_DIR), hf_token,
        systems=set(all_test_systems), per_system=PER_SYSTEM, seed=SEED)

    print("\nLoading validation split (training systems A01-A06 + bonafide for reference)...")
    val_ds = TestSystemDataset(
        HF_DATASET, "validation", str(CACHE_DIR), hf_token,
        systems=set(TRAIN_SYSTEMS + [BONAFIDE_ID]), per_system=PER_SYSTEM, seed=SEED)

    # Combined loader
    combined_dataset = torch.utils.data.ConcatDataset([test_ds, val_ds])
    # Rebuild indices/sys_ids for combined
    all_sids_list = test_ds.sys_ids + val_ds.sys_ids

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader  = torch.utils.data.DataLoader(
        val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ── Extract embeddings + logits ────────────────────────────────────────────
    all_embs    = []
    all_logits  = []
    all_labels  = []
    all_sys_ids = []

    for name, loader in [("test", test_loader), ("val", val_loader)]:
        print(f"\nExtracting from {name} split...")
        n_done = 0
        for batch in loader:
            embs, logits = extract_batch(lit, batch["audio"], device)
            all_embs.append(embs)
            all_logits.append(logits)
            all_labels.extend(batch["label"].tolist())
            all_sys_ids.extend(batch["system_id"])
            n_done += len(logits)
            if n_done % 200 == 0:
                print(f"  {n_done} done...")
        print(f"  {name}: {n_done} samples")

    all_embs    = np.concatenate(all_embs, axis=0).astype(np.float32)
    all_logits  = np.concatenate(all_logits, axis=0).astype(np.float32)
    all_labels  = np.array(all_labels, dtype=np.int32)
    all_sys_ids = np.array(all_sys_ids)
    print(f"\nTotal: {len(all_embs)} embeddings")

    # Save embeddings
    np.savez_compressed(
        OUT_DIR / "embeddings.npz",
        X=all_embs, system_ids=all_sys_ids, labels=all_labels, logits=all_logits)
    print("Saved: embeddings.npz")

    # ── Per-system metrics ─────────────────────────────────────────────────────
    print(f"\n{'System':<8}  {'Split':<6}  {'N':>5}  {'Acc':>6}  {'AUC':>6}  {'EER':>6}  {'minDCF':>8}")
    print("─" * 56)

    metrics_rows = []
    for sid in sorted(set(all_sys_ids)):
        mask = all_sys_ids == sid
        y    = all_labels[mask]
        sc   = all_logits[mask]     # higher = more spoof
        pred = (sc > 0).astype(int)

        split = "val" if sid in TRAIN_SYSTEMS else "test"

        if sid == BONAFIDE_ID:
            # bonafide: all label=0, compute false acceptance rate
            acc   = float((pred == y).mean())
            auc   = float("nan"); eer = float("nan"); dcf = float("nan")
        else:
            # Need both bonafide and spoof scores for EER/AUC/DCF
            bon_mask   = all_sys_ids == BONAFIDE_ID
            # use the bonafide from the same split as reference
            if sid in TRAIN_SYSTEMS:
                bon_split_mask = bon_mask & np.isin(
                    np.arange(len(all_sys_ids)),
                    np.where(all_sys_ids == BONAFIDE_ID)[0][-PER_SYSTEM:])
            else:
                bon_split_mask = bon_mask & np.isin(
                    np.arange(len(all_sys_ids)),
                    np.where(all_sys_ids == BONAFIDE_ID)[0][:PER_SYSTEM])

            combined_sc  = np.concatenate([sc, all_logits[bon_split_mask]])
            combined_lbl = np.concatenate([y,  all_labels[bon_split_mask]])
            acc = float((pred == y).mean())
            auc = compute_auc(combined_sc, combined_lbl)
            eer = compute_eer(combined_sc, combined_lbl)
            dcf = compute_min_dcf(combined_sc, combined_lbl)

        row = {"system_id": sid, "split": split, "n": int(mask.sum()),
               "acc": f"{acc:.4f}", "auc": f"{auc:.4f}",
               "eer": f"{eer:.4f}", "min_dcf": f"{dcf:.4f}"}
        metrics_rows.append(row)
        print(f"  {sid:<6}  {split:<6}  {mask.sum():5}  "
              f"{acc:6.4f}  {auc:6.4f}  {eer:6.4f}  {dcf:8.4f}")

    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system_id","split","n","acc","auc","eer","min_dcf"])
        w.writeheader(); w.writerows(metrics_rows)
    print("\nSaved: metrics.csv")

    # ── PCA ───────────────────────────────────────────────────────────────────
    print("\nRunning PCA...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_embs)

    pca = PCA(n_components=2, random_state=SEED)
    pca_proj = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    print(f"  PCA variance: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")

    scatter_plot(
        pca_proj, all_sys_ids, TRAIN_SYSTEMS, UNSEEN_SYSTEMS,
        title=(f"PCA of GAT layer-0 embeddings — all systems\n"
               f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
               f"(▲=bonafide  ■=trained  ●=unseen)"),
        xlabel=f"PC1  ({var[0]*100:.1f}%)", ylabel=f"PC2  ({var[1]*100:.1f}%)",
        out_path=OUT_DIR / "pca_unseen.png")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("Running t-SNE (this may take a minute)...")
    from sklearn.manifold import TSNE

    # Subsample for t-SNE to keep it fast: 200 per system
    TSNE_PER_SYS = 200
    tsne_idx = []
    for sid in sorted(set(all_sys_ids)):
        idx = np.where(all_sys_ids == sid)[0]
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(idx, min(TSNE_PER_SYS, len(idx)), replace=False)
        tsne_idx.extend(chosen.tolist())
    tsne_idx = np.array(tsne_idx)

    X_tsne_in = X_scaled[tsne_idx]
    sids_tsne  = all_sys_ids[tsne_idx]

    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=SEED, init="pca", learning_rate="auto")
    tsne_proj = tsne.fit_transform(X_tsne_in)
    print(f"  t-SNE done ({len(tsne_idx)} points)")

    scatter_plot(
        tsne_proj, sids_tsne, TRAIN_SYSTEMS, UNSEEN_SYSTEMS,
        title="t-SNE of GAT layer-0 embeddings — all systems\n"
              "(▲=bonafide  ■=trained A01–A06  ●=unseen A07–A19)",
        xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
        out_path=OUT_DIR / "tsne_unseen.png")

    # ── Metrics bar chart ──────────────────────────────────────────────────────
    # Only spoof systems for the chart
    spoof_rows = [r for r in metrics_rows if r["system_id"] != BONAFIDE_ID]
    plot_metrics(spoof_rows, OUT_DIR / "metrics_bar.png")

    print(f"\nAll artefacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
