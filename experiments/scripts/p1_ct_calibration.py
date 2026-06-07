#!/usr/bin/env python3
"""
p1_ct_calibration.py
====================
P1 ablation: Inference-time score calibration using C and T features.

Extracts per-utterance C (WavLM L12 compactness, −rog) and T (WavLM L9 velocity entropy)
alongside raw robust_goat logits on the MLAAD test split, then fits LOO calibration:

    logit_cal(u) = logit_raw(u) + β₁·C(u) + β₂·T(u)

β is fitted via logistic regression on utterances from the other 62 systems (LOO-system).
EER is evaluated before and after calibration, stratified by C quartile.

Key question: Does C/T discriminate bonafide from spoof at the utterance level?
If yes, calibration improves EER. If not, calibration is a no-op (EER unchanged).

Outputs:
    experiments/results/mlaad/p1_ct_calibration/
        utterance_features.csv       — per-utterance raw_logit, C, T, label, system
        per_system_eer.csv           — before/after EER for 63 systems
        calibration_coefficients.csv — β₁, β₂ from LOO folds
        summary.md                   — narrative + diagnostic statistics
        figures/
            eer_before_after.png
            ct_bonafide_vs_spoof.png
            calibration_gain_vs_eer.png
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
TEST_JSON     = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
CKPT_PATH     = PROJECT_ROOT / "experiments" / "checkpoints" / "mlaad_robust_goat.ckpt"
OUT_DIR       = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─── torch.load compat ───────────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 16
TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR   # 48000
NF             = TARGET_SAMPLES // 320 - 1   # 149

MIN_SPOOF_SAMPLES = 5   # drop systems with fewer spoof utterances

# ─── Feature utilities ───────────────────────────────────────────────────────

def rog(frames: np.ndarray) -> float:
    """Radius of gyration: RMS distance from centroid. (T, D) → scalar."""
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c) ** 2, 1))))


def vel_entropy(frames: np.ndarray, stride: int = 1) -> float:
    """Adjacent-frame L2 velocity entropy. (T, D) → scalar."""
    f1, f2 = frames[:-stride], frames[stride:]
    vels = np.linalg.norm(f2 - f1, axis=1)
    n_bins = max(5, min(20, len(vels) // 4))
    hist, _ = np.histogram(vels, bins=n_bins)
    h = hist.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))


def compute_ct_batch(frames_L9_batch, frames_L12_batch) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute C and T for a batch of utterances.
    frames_L9_batch:  list of (T, 768) arrays
    frames_L12_batch: list of (T, 768) arrays
    Returns C_arr (B,), T_arr (B,)
    """
    C_vals = np.array([rog(f) for f in frames_L12_batch])
    T_vals = np.array([vel_entropy(f) for f in frames_L9_batch])
    return C_vals, T_vals


# ─── Dataset ─────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = torch.load(PROCESSED_DIR / rec["audio_path"])  # (1, 48000)
        y   = 0 if rec["label"] == "bonafide" else 1
        return {
            "audio":   wav,           # (1, 48000)
            "label":   y,
            "system":  rec["attack_system"],
            "sample_id": rec["sample_id"],
        }


def collate_fn(batch):
    audios   = torch.stack([b["audio"] for b in batch])
    labels   = [b["label"] for b in batch]
    systems  = [b["system"] for b in batch]
    sids     = [b["sample_id"] for b in batch]
    return {"audio": audios, "label": labels, "system": systems, "sample_id": sids}


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(ckpt_path: Path) -> tuple:
    from phoneme_GAT.modules import Phoneme_GAT_lit
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), map_location=DEVICE)
    lit.eval()
    lit.to(DEVICE)
    wavlm = lit.model.transformer_in_phoneme_model
    return lit, wavlm


# ─── Inference + feature extraction ─────────────────────────────────────────

@torch.no_grad()
def extract_all_features(lit, wavlm, records: list[dict]) -> pd.DataFrame:
    """
    For each utterance:
      - raw_logit from robust_goat
      - C = rog@L12 (NOT negated; lower = harder after correlation check)
      - T = vel_entropy@L9 (higher = harder)
    Returns DataFrame with columns: sample_id, system, label, raw_logit, C, T
    """
    ds = EvalDataset(records)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, collate_fn=collate_fn, pin_memory=True)

    rows = []
    for batch_idx, batch in enumerate(dl):
        audio   = batch["audio"].to(DEVICE)   # (B, 1, 48000)
        B       = audio.shape[0]
        num_frames = torch.full((B,), NF, device=DEVICE)

        # 1) robust_goat inference
        out    = lit.model(audio, num_frames, profiler=None, use_aug=False, stage="val")
        logits = out["logit"].cpu().float().numpy()

        # 2) WavLM hidden states for C and T
        audio_1d = audio[:, 0, :] if audio.ndim == 3 else audio   # (B, 48000)
        wout = wavlm(input_values=audio_1d, output_hidden_states=True)
        # hidden_states: tuple of 13 tensors, each (B, T', 768)
        # Index 0 = pre-transformer (L0), index 9 = L9, index 12 = L12
        hs_all = wout.hidden_states
        frames_L9  = [hs_all[9][i].cpu().numpy()  for i in range(B)]   # list of (T', 768)
        frames_L12 = [hs_all[12][i].cpu().numpy() for i in range(B)]

        C_arr, T_arr = compute_ct_batch(frames_L9, frames_L12)

        for i in range(B):
            rows.append({
                "sample_id": batch["sample_id"][i],
                "system":    batch["system"][i],
                "label":     batch["label"][i],
                "raw_logit": float(logits[i]),
                "C":         float(C_arr[i]),   # rog@L12 (lower = harder)
                "T":         float(T_arr[i]),   # vel_entropy@L9 (higher = harder)
            })

        if (batch_idx + 1) % 10 == 0:
            print(f"  batch {batch_idx+1}/{len(dl)}")

    return pd.DataFrame(rows)


# ─── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict | None:
    """
    Compute EER, AUC, accuracy, and balanced accuracy for a binary classification problem.
    Optimal threshold is chosen via Youden's J (max TPR - FPR).
    Returns dict with keys: eer, auc, acc, bal_acc, opt_threshold.
    """
    if len(np.unique(labels)) < 2:
        return None
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
        auc = float(roc_auc_score(labels, scores))
        j_idx   = int(np.argmax(tpr - fpr))
        opt_thr = float(thresholds[j_idx])
        preds   = (scores >= opt_thr).astype(int)
        acc     = float((preds == labels).mean())
        bal_acc = float(balanced_accuracy_score(labels, preds))
        return {"eer": eer, "auc": auc, "acc": acc, "bal_acc": bal_acc, "opt_threshold": opt_thr}
    except Exception:
        return None


def per_system_metrics(
    df: pd.DataFrame, score_col: str, min_n: int = MIN_SPOOF_SAMPLES
) -> dict[str, dict | None]:
    """EER/AUC/accuracy per spoof system: system spoof scores vs ALL bonafide scores."""
    bf_mask   = df["label"] == 0
    bf_scores = df.loc[bf_mask, score_col].values
    bf_labels = np.zeros(len(bf_scores), dtype=int)

    results = {}
    for sys, grp in df[~bf_mask].groupby("system"):
        sp_scores = grp[score_col].values
        sp_labels = np.ones(len(sp_scores), dtype=int)
        if len(sp_scores) < min_n:
            results[sys] = None
            continue
        labels_combined = np.concatenate([bf_labels, sp_labels])
        scores_combined = np.concatenate([bf_scores, sp_scores])
        results[sys] = compute_metrics(labels_combined, scores_combined)
    return results


# ─── LOO calibration ─────────────────────────────────────────────────────────

def loo_calibrate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of 63 spoof systems (held out), fit logistic regression on the
    other 62 systems' utterances + all bonafide, using features [raw_logit, C, T].
    Apply calibrated logit = β₀ + β₁·raw_logit + β₂·C + β₃·T to held-out system.
    Returns df with added column 'cal_logit'.
    """
    df = df.copy()
    df["cal_logit"] = np.nan

    spoof_systems = [s for s in df["system"].unique() if s != "bonafide"]
    bf_mask = df["label"] == 0
    bf_idx  = df.index[bf_mask]

    scaler = StandardScaler()
    # Fit scaler on all data once (for consistent normalization)
    feat_all = df[["raw_logit", "C", "T"]].values
    scaler.fit(feat_all)

    betas = []
    for sys in spoof_systems:
        held_out_mask  = df["system"] == sys
        train_mask     = ~held_out_mask  # all other systems + bonafide

        X_train = scaler.transform(df.loc[train_mask, ["raw_logit", "C", "T"]].values)
        y_train = df.loc[train_mask, "label"].values.astype(int)

        # Logistic regression: P(spoof) = sigmoid(β₀ + β₁·f₁ + β₂·f₂ + β₃·f₃)
        clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        clf.fit(X_train, y_train)

        # Calibrated logit = decision_function (log odds) for held-out + bonafide
        test_mask = held_out_mask | bf_mask
        X_test = scaler.transform(df.loc[test_mask, ["raw_logit", "C", "T"]].values)
        cal_logits = clf.decision_function(X_test)
        df.loc[test_mask, "cal_logit"] = cal_logits

        betas.append({
            "system":      sys,
            "intercept":   float(clf.intercept_[0]),
            "beta_logit":  float(clf.coef_[0, 0]),
            "beta_C":      float(clf.coef_[0, 1]),
            "beta_T":      float(clf.coef_[0, 2]),
        })
        print(f"  {sys[:40]:40s}  β_logit={clf.coef_[0,0]:.3f}  β_C={clf.coef_[0,1]:.3f}  β_T={clf.coef_[0,2]:.3f}")

    return df, pd.DataFrame(betas)


# ─── Plotting ────────────────────────────────────────────────────────────────

def make_figures(df_utt: pd.DataFrame, metrics_before: dict, metrics_after: dict, out_dir: Path):
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    systems = [s for s in metrics_before
               if metrics_before[s] is not None and metrics_after.get(s) is not None]

    def _arr(d, key):
        return np.array([d[s][key] for s in systems])

    eer_b  = _arr(metrics_before, "eer");  eer_a  = _arr(metrics_after, "eer")
    auc_b  = _arr(metrics_before, "auc");  auc_a  = _arr(metrics_after, "auc")
    acc_b  = _arr(metrics_before, "acc");  acc_a  = _arr(metrics_after, "acc")
    bal_b  = _arr(metrics_before, "bal_acc"); bal_a = _arr(metrics_after, "bal_acc")

    # 1) EER before vs after scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(eer_b.max(), eer_a.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.scatter(eer_b, eer_a, alpha=0.6, s=25, c="steelblue")
    ax.set_xlabel("EER (raw logit)"); ax.set_ylabel("EER (calibrated)")
    ax.set_title(f"P1 Calibration: EER before vs after  (N={len(systems)} systems)")
    delta_eer = eer_b - eer_a
    ax.text(0.05, 0.95,
            f"Improved: {(delta_eer>0).sum()}/{len(systems)}\nMean ΔEER={delta_eer.mean():.4f}",
            transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "eer_before_after.png", dpi=150); plt.close(fig)

    # 2) AUC before vs after scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    lim_lo = min(auc_b.min(), auc_a.min()) * 0.97
    lim_hi = max(auc_b.max(), auc_a.max()) * 1.01
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.5)
    ax.scatter(auc_b, auc_a, alpha=0.6, s=25, c="darkorange")
    ax.set_xlabel("AUC (raw logit)"); ax.set_ylabel("AUC (calibrated)")
    ax.set_title(f"P1 Calibration: AUC before vs after  (N={len(systems)} systems)")
    delta_auc = auc_a - auc_b
    ax.text(0.05, 0.05,
            f"Improved: {(delta_auc>0).sum()}/{len(systems)}\nMean ΔAUC={delta_auc.mean():+.4f}",
            transform=ax.transAxes, va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "auc_before_after.png", dpi=150); plt.close(fig)

    # 3) Balanced accuracy before vs after
    fig, ax = plt.subplots(figsize=(6, 6))
    lim_lo = min(bal_b.min(), bal_a.min()) * 0.97
    lim_hi = max(bal_b.max(), bal_a.max()) * 1.01
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.5)
    ax.scatter(bal_b, bal_a, alpha=0.6, s=25, c="forestgreen")
    ax.set_xlabel("Balanced Acc (raw logit)"); ax.set_ylabel("Balanced Acc (calibrated)")
    ax.set_title(f"P1 Calibration: Balanced accuracy before vs after  (N={len(systems)} systems)")
    delta_bal = bal_a - bal_b
    ax.text(0.05, 0.05,
            f"Improved: {(delta_bal>0).sum()}/{len(systems)}\nMean Δbal_acc={delta_bal.mean():+.4f}",
            transform=ax.transAxes, va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "bal_acc_before_after.png", dpi=150); plt.close(fig)

    # 4) C/T distributions: bonafide vs spoof
    bf = df_utt[df_utt["label"] == 0]
    sp = df_utt[df_utt["label"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, name in zip(axes, ["C", "T"], ["C (rog@L12)", "T (vel_entropy@L9)"]):
        ax.hist(bf[col], bins=30, alpha=0.6, label="Bonafide", color="green", density=True)
        ax.hist(sp[col], bins=30, alpha=0.6, label="Spoof",    color="red",   density=True)
        ax.set_xlabel(name); ax.set_ylabel("Density"); ax.legend()
        ax.set_title(f"{name}\nSpoof−Bonafide mean={sp[col].mean()-bf[col].mean():.4f}")
    fig.suptitle("P1: C and T distributions — bonafide vs spoof")
    fig.tight_layout(); fig.savefig(fig_dir / "ct_bonafide_vs_spoof.png", dpi=150); plt.close(fig)

    # 5) Calibration gain (ΔEER) vs system C quartile
    sys_stats = df_utt[df_utt["label"] == 1].groupby("system")["C"].mean().rename("mean_C")
    gain_df = pd.DataFrame({
        "system":   systems,
        "eer_b":    eer_b,
        "delta_eer": eer_b - eer_a,
        "delta_auc": auc_a - auc_b,
        "delta_bal": bal_a - bal_b,
    }).set_index("system").join(sys_stats).dropna()
    gain_df["C_quartile"] = pd.qcut(gain_df["mean_C"], 4,
                                     labels=["Q1\n(compact)", "Q2", "Q3", "Q4\n(spread)"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, ylabel, title in zip(
        axes,
        ["delta_eer", "delta_auc", "delta_bal"],
        ["Mean ΔEER (before−after)", "Mean ΔAUC (after−before)", "Mean Δbal_acc (after−before)"],
        ["ΔEER by C quartile", "ΔAUC by C quartile", "Δbal_acc by C quartile"],
    ):
        colors = ["lightblue", "steelblue", "royalblue", "navy"]
        gain_df.groupby("C_quartile", observed=False)[col].mean().plot(
            kind="bar", ax=ax, color=colors, edgecolor="black"
        )
        ax.axhline(0, color="red", lw=0.8, ls="--")
        ax.set_xlabel("System C quartile"); ax.set_ylabel(ylabel)
        ax.set_title(f"P1: {title}"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(fig_dir / "calibration_gain_vs_eer.png", dpi=150); plt.close(fig)

    print(f"Figures saved to {fig_dir}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def write_summary(df_utt, metrics_before, metrics_after, betas_df, out_dir: Path):
    systems = [s for s in metrics_before
               if metrics_before[s] is not None and metrics_after.get(s) is not None]

    def _arr(d, key):
        return np.array([d[s][key] for s in systems])

    eer_b  = _arr(metrics_before, "eer");  eer_a  = _arr(metrics_after, "eer")
    auc_b  = _arr(metrics_before, "auc");  auc_a  = _arr(metrics_after, "auc")
    acc_b  = _arr(metrics_before, "acc");  acc_a  = _arr(metrics_after, "acc")
    bal_b  = _arr(metrics_before, "bal_acc"); bal_a = _arr(metrics_after, "bal_acc")

    delta_eer = eer_b - eer_a   # positive = improved (lower EER)
    delta_auc = auc_a - auc_b   # positive = improved (higher AUC)
    delta_bal = bal_a - bal_b   # positive = improved (higher bal_acc)

    bf = df_utt[df_utt["label"] == 0]
    sp = df_utt[df_utt["label"] == 1]
    c_delta = sp["C"].mean() - bf["C"].mean()
    t_delta = sp["T"].mean() - bf["T"].mean()
    mean_beta_C = betas_df["beta_C"].mean()
    mean_beta_T = betas_df["beta_T"].mean()

    # C quartile calibration gain
    sys_c = df_utt[df_utt["label"] == 1].groupby("system")["C"].mean().rename("mean_C")
    gain_df = pd.DataFrame({
        "system": systems, "delta_eer": delta_eer, "delta_auc": delta_auc, "delta_bal": delta_bal,
    }).set_index("system").join(sys_c).dropna()
    gain_df["C_quartile"] = pd.qcut(gain_df["mean_C"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    q_eer = gain_df.groupby("C_quartile", observed=False)["delta_eer"].mean()
    q_auc = gain_df.groupby("C_quartile", observed=False)["delta_auc"].mean()
    q_bal = gain_df.groupby("C_quartile", observed=False)["delta_bal"].mean()

    hard_mask = eer_b >= np.percentile(eer_b, 75)

    lines = [
        "# P1: Inference-time Score Calibration using C and T",
        "",
        "## Setup",
        f"- **Model**: mlaad_robust_goat.ckpt",
        f"- **Test split**: {len(df_utt)} utterances ({(df_utt['label']==0).sum()} bonafide, {(df_utt['label']==1).sum()} spoof)",
        f"- **Systems**: {len(systems)} evaluated (min {MIN_SPOOF_SAMPLES} spoof samples)",
        f"- **Calibration**: LOO logistic regression on [raw_logit, C, T] (standardized)",
        "",
        "## Key Diagnostic: Do C/T Discriminate Bonafide vs Spoof?",
        "",
        "| Feature | Bonafide mean | Spoof mean | Δ(Spoof−Bonafide) |",
        "|---------|--------------|------------|-------------------|",
        f"| C (rog@L12) | {bf['C'].mean():.5f} | {sp['C'].mean():.5f} | {c_delta:+.5f} |",
        f"| T (vel_entropy@L9) | {bf['T'].mean():.5f} | {sp['T'].mean():.5f} | {t_delta:+.5f} |",
        "",
        "## Calibration Coefficients (LOO mean ± std)",
        "",
        f"| Coefficient | Mean | Std |",
        f"|------------|------|-----|",
        f"| β_logit | {betas_df['beta_logit'].mean():.4f} | {betas_df['beta_logit'].std():.4f} |",
        f"| β_C (rog@L12) | {mean_beta_C:.4f} | {betas_df['beta_C'].std():.4f} |",
        f"| β_T (vel_entropy@L9) | {mean_beta_T:.4f} | {betas_df['beta_T'].std():.4f} |",
        "",
        "## Results: All Metrics",
        "",
        f"| Metric | Before | After | Δ |",
        f"|--------|--------|-------|---|",
        f"| Mean EER (all {len(systems)} systems) | {eer_b.mean():.4f} | {eer_a.mean():.4f} | {delta_eer.mean():+.4f} |",
        f"| Median EER | {np.median(eer_b):.4f} | {np.median(eer_a):.4f} | {np.median(eer_b)-np.median(eer_a):+.4f} |",
        f"| Hard-quartile EER (≥75th pct) | {eer_b[hard_mask].mean():.4f} | {eer_a[hard_mask].mean():.4f} | {delta_eer[hard_mask].mean():+.4f} |",
        f"| Mean AUC | {auc_b.mean():.4f} | {auc_a.mean():.4f} | {delta_auc.mean():+.4f} |",
        f"| Hard-quartile AUC | {auc_b[hard_mask].mean():.4f} | {auc_a[hard_mask].mean():.4f} | {delta_auc[hard_mask].mean():+.4f} |",
        f"| Mean Acc (Youden thr) | {acc_b.mean():.4f} | {acc_a.mean():.4f} | {(acc_a-acc_b).mean():+.4f} |",
        f"| Mean Balanced Acc | {bal_b.mean():.4f} | {bal_a.mean():.4f} | {delta_bal.mean():+.4f} |",
        f"| Systems improved (EER) | {(delta_eer>0).sum()}/{len(systems)} | — | — |",
        "",
        "## Calibration Gain by System C Quartile",
        "",
        "| C Quartile | ΔEER | ΔAUC | Δbal_acc | Note |",
        "|-----------|------|------|----------|------|",
    ]
    for q in q_eer.index:
        note = "compact (harder)" if str(q) in ("Q1", "Q2") else "spread (easier)"
        lines.append(
            f"| {q} ({note}) | {q_eer[q]:+.4f} | {q_auc[q]:+.4f} | {q_bal[q]:+.4f} | |"
        )

    lines += ["", "## Interpretation", ""]

    if abs(c_delta) > 0.05:
        lines.append(
            f"C differs between bonafide ({bf['C'].mean():.3f}) and spoof ({sp['C'].mean():.3f}) "
            f"by {c_delta:+.4f} — utterance-level separation exists."
        )
    else:
        lines.append(
            f"C overlap is large (Δ={c_delta:+.4f}): C is primarily a system-level predictor."
        )

    if mean_beta_C < 0:
        lines.append(f"β_C={mean_beta_C:.4f} < 0: compact (low rog) → higher spoof score. ✓ Expected.")
    else:
        lines.append(f"β_C={mean_beta_C:.4f} > 0: spread (high rog) → higher spoof score. ✗ Unexpected.")

    gain = delta_eer.mean()
    if gain > 0.005:
        lines.append(
            f"\n**Conclusion: Calibration helps overall (ΔEER={gain:+.4f}, ΔAUC={delta_auc.mean():+.4f}). "
            "C/T carry utterance-level discriminative information beyond the raw model logit.**"
        )
    elif gain < -0.005:
        lines.append(
            f"\n**Conclusion: Calibration HURTS overall (ΔEER={gain:+.4f}). "
            "C/T not useful at utterance level.**"
        )
    else:
        lines.append(
            f"\n**Conclusion: Calibration is neutral (ΔEER={gain:+.4f}, ΔAUC={delta_auc.mean():+.4f}). "
            "C/T explain system-level hardness but not utterance-level discrimination.**"
        )

    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"Summary written to {out_dir / 'summary.md'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading records from {TEST_JSON}...")
    with open(TEST_JSON) as f:
        records = json.load(f)
    print(f"  {len(records)} records loaded.")

    print(f"\nLoading model from {CKPT_PATH}...")
    lit, wavlm = load_model(CKPT_PATH)
    print("  Model loaded.")

    # --- Step 1: Extract per-utterance features ---
    utt_csv = OUT_DIR / "utterance_features.csv"
    if utt_csv.exists():
        print(f"\nLoading cached features from {utt_csv}...")
        df_utt = pd.read_csv(utt_csv)
    else:
        print("\nExtracting per-utterance logits and C/T features...")
        df_utt = extract_all_features(lit, wavlm, records)
        df_utt.to_csv(utt_csv, index=False)
        print(f"  Saved to {utt_csv}")

    print(f"\n{'='*60}")
    print(f"Feature statistics:")
    print(f"  Bonafide utterances: {(df_utt['label']==0).sum()}")
    print(f"  Spoof utterances:    {(df_utt['label']==1).sum()}")
    print(f"  Systems:             {df_utt[df_utt['label']==1]['system'].nunique()}")
    bf = df_utt[df_utt["label"] == 0]
    sp = df_utt[df_utt["label"] == 1]
    for col in ["raw_logit", "C", "T"]:
        print(f"  {col}: bonafide={bf[col].mean():.4f}±{bf[col].std():.4f}, "
              f"spoof={sp[col].mean():.4f}±{sp[col].std():.4f}, "
              f"Δ={sp[col].mean()-bf[col].mean():+.4f}")

    # --- Step 2: Baseline metrics ---
    print(f"\n{'='*60}")
    print("Computing baseline metrics (raw logit)...")
    metrics_before = per_system_metrics(df_utt, "raw_logit")
    valid_systems = [s for s in metrics_before if metrics_before[s] is not None]
    m = metrics_before
    print(f"  Mean EER: {np.mean([m[s]['eer'] for s in valid_systems]):.4f}  "
          f"AUC: {np.mean([m[s]['auc'] for s in valid_systems]):.4f}  "
          f"bal_acc: {np.mean([m[s]['bal_acc'] for s in valid_systems]):.4f}  "
          f"({len(valid_systems)} systems)")

    # --- Step 3: LOO calibration ---
    print(f"\n{'='*60}")
    print("Running LOO calibration...")
    df_utt, betas_df = loo_calibrate(df_utt)
    betas_df.to_csv(OUT_DIR / "calibration_coefficients.csv", index=False)

    # --- Step 4: Calibrated metrics ---
    print(f"\n{'='*60}")
    print("Computing calibrated metrics...")
    metrics_after = per_system_metrics(df_utt, "cal_logit")

    # --- Step 5: Save per-system results ---
    sys_rows = []
    for sys in sorted(valid_systems):
        ma = metrics_after.get(sys)
        mb = metrics_before[sys]
        if ma is None:
            continue
        sys_rows.append({
            "system":       sys,
            "eer_before":   mb["eer"],
            "eer_after":    ma["eer"],
            "delta_eer":    mb["eer"] - ma["eer"],
            "auc_before":   mb["auc"],
            "auc_after":    ma["auc"],
            "delta_auc":    ma["auc"] - mb["auc"],
            "acc_before":   mb["acc"],
            "acc_after":    ma["acc"],
            "bal_acc_before": mb["bal_acc"],
            "bal_acc_after":  ma["bal_acc"],
        })
    sys_df = pd.DataFrame(sys_rows).sort_values("eer_before", ascending=False)
    sys_df.to_csv(OUT_DIR / "per_system_eer.csv", index=False)

    print(f"\n{'='*60}")
    before_arr = np.array([r["eer_before"] for r in sys_rows])
    after_arr  = np.array([r["eer_after"]  for r in sys_rows])
    auc_b_arr  = np.array([r["auc_before"] for r in sys_rows])
    auc_a_arr  = np.array([r["auc_after"]  for r in sys_rows])
    bal_b_arr  = np.array([r["bal_acc_before"] for r in sys_rows])
    bal_a_arr  = np.array([r["bal_acc_after"]  for r in sys_rows])
    delta_arr  = before_arr - after_arr
    hard_mask  = before_arr >= np.percentile(before_arr, 75)
    print(f"Results:")
    print(f"  EER before / after / Δ: {before_arr.mean():.4f} / {after_arr.mean():.4f} / {delta_arr.mean():+.4f}")
    print(f"  AUC before / after / Δ: {auc_b_arr.mean():.4f} / {auc_a_arr.mean():.4f} / {(auc_a_arr-auc_b_arr).mean():+.4f}")
    print(f"  bal_acc before / after:  {bal_b_arr.mean():.4f} / {bal_a_arr.mean():.4f}")
    print(f"  Systems improved (EER): {(delta_arr > 0).sum()}/{len(delta_arr)}")
    print(f"  Hard-quartile ΔEER: {delta_arr[hard_mask].mean():+.4f}")
    print(f"  Hard-quartile ΔAUC: {(auc_a_arr-auc_b_arr)[hard_mask].mean():+.4f}")

    # --- Step 6: Figures and summary ---
    make_figures(df_utt, metrics_before, metrics_after, OUT_DIR)
    write_summary(df_utt, metrics_before, metrics_after, betas_df, OUT_DIR)

    # Save updated utterance features
    df_utt.to_csv(utt_csv, index=False)
    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
