#!/usr/bin/env python3
"""
p8_cross_encoder_c_ensemble.py
==============================
P8 ablation: cross-encoder C ensemble (WavLM L12 + wav2vec2 L8).

E1 found comparable LOO R² for C in WavLM@L12 (+0.069) and wav2vec2@L8 (+0.073).
Averaging the two compactness estimates should reduce encoder-specific noise.
The roadmap's end goal is to inject C_ensemble via the P2 mechanism, but that
needs TWO frozen encoders inside the training loop (2× compute) and is flagged
"only worth doing after P2 is validated."

This script is the cheap GATE that decides whether that retraining is worth it:
it recomputes the P1 calibration diagnostic with C_ensemble in place of
C_wavlm and asks whether ensembling improves the system-level signal. It reuses
P1's per-utterance WavLM features (raw_logit, C_wavlm, T) and only runs the
*second* encoder (wav2vec2) over the test split to add C_w2v2.

Because rog scales differ across encoders, C from each is z-standardized before
averaging:  C_ensemble = (z(C_wavlm) + z(C_w2v2)) / 2.

Pipeline:
  1. Load P1 utterance_features.csv  (sample_id, raw_logit, C_wavlm, T, label, system).
  2. Extract wav2vec2 L8 rog per test utterance → C_w2v2 (joined by sample_id).
  3. LOO-system affine calibration with feature set {wavlm-only} vs {ensemble}.
  4. Per-system EER + C-quartile stratification for: raw, wavlm-cal, ensemble-cal.

Decision rule (diagnostic, not a trained model):
  GO (retrain with injection)  if ensemble-cal improves hard-quartile mean EER
                                  by ≥0.01 over wavlm-cal;
  NO-GO                          otherwise (ensembling does not add signal).

Outputs: experiments/results/mlaad/p8_cross_encoder_ensemble/
         {utterance_features_ensemble.csv, per_system_eer.csv, summary.md}

Usage:
    python p8_cross_encoder_c_ensemble.py            # full test split
    python p8_cross_encoder_c_ensemble.py --limit 200  # quick smoke
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _ablation_common as AC
from _ablation_common import (
    PROC_DIR, TEST_JSON, compute_eer, per_system_metrics_from_dict,
    load_system_ct, stratify_by_ct, stratified_markdown, EXP_DIR,
)

P1_FEATURES = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "utterance_features.csv"
OUT_DIR = EXP_DIR / "results" / "mlaad" / "p8_cross_encoder_ensemble"


def _rog(frames: np.ndarray) -> float:
    """rog of a (T, D) frame cloud."""
    c = frames.mean(axis=0, keepdims=True)
    return float(np.sqrt(((frames - c) ** 2).sum(axis=1).mean()))


@torch.no_grad()
def extract_w2v2_c(sample_ids, limit=None) -> dict[str, float]:
    """C_w2v2 = rog of wav2vec2 L8 frames, per sample_id."""
    from transformers import Wav2Vec2Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading facebook/wav2vec2-base ...", flush=True)
    w2v2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval().to(device)

    records = {r["sample_id"]: r for r in json.loads(TEST_JSON.read_text())}
    ids = list(sample_ids)[: limit] if limit else list(sample_ids)

    out = {}
    for i, sid in enumerate(ids):
        rec = records.get(sid)
        if rec is None:
            continue
        wav = torch.load(PROC_DIR / rec["audio_path"])  # (1, 48000) @ 16 kHz
        x = wav.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hs = w2v2(x, output_hidden_states=True).hidden_states  # 13 tensors (L0..L12)
        frames_l8 = hs[8][0].float().cpu().numpy()  # (T, 768)
        out[sid] = _rog(frames_l8)
        if (i + 1) % 200 == 0:
            print(f"  wav2vec2 {i+1}/{len(ids)}", flush=True)
    return out


def loo_calibrate(df: pd.DataFrame, c_col: str) -> np.ndarray:
    """Leave-one-system-out affine calibration on [raw_logit, C, T] (standardized).
    Returns calibrated scores aligned to df rows."""
    from sklearn.linear_model import LogisticRegression
    systems = df["system"].values
    feats = df[["raw_logit", c_col, "T"]].values.astype(float)
    y = df["label"].values.astype(int)
    cal = np.zeros(len(df))
    for s in np.unique(systems):
        te = systems == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            cal[te] = df["raw_logit"].values[te]
            continue
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-8
        lr = LogisticRegression(max_iter=1000)
        lr.fit((feats[tr] - mu) / sd, y[tr])
        cal[te] = lr.decision_function((feats[te] - mu) / sd)
    return cal


def per_system_metrics(df: pd.DataFrame, score_col: str) -> dict:
    return per_system_metrics_from_dict(
        df["label"].tolist(), df[score_col].tolist(), df["system"].tolist())


def _mean_metric(metrics: dict, key: str, ct: dict | None = None,
                 hard_only: bool = False) -> float:
    systems = [s for s in metrics if metrics.get(s) is not None]
    if hard_only:
        cvals = np.array([ct[s]["C"] for s in systems if s in ct])
        thr = np.quantile(cvals, 0.75)
        systems = [s for s in systems if s in ct and ct[s]["C"] >= thr]
    vals = [metrics[s][key] for s in systems if metrics[s].get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="limit #utterances (smoke test); default = full 1846")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not P1_FEATURES.exists():
        raise SystemExit(f"Missing {P1_FEATURES}; run p1_ct_calibration.py first.")
    df = pd.read_csv(P1_FEATURES)  # sample_id, system, label, raw_logit, C, T, ...
    if args.limit:
        df = df.groupby("system", group_keys=False).head(max(2, args.limit // df["system"].nunique()))

    # ── add wav2vec2 C, join by sample_id ────────────────────────────────────
    c_w2v2 = extract_w2v2_c(df["sample_id"].tolist(), limit=args.limit)
    df = df[df["sample_id"].isin(c_w2v2)].copy()
    df["C_w2v2"] = df["sample_id"].map(c_w2v2)

    # z-standardize each compactness, then average (encoders have different rog scales)
    zc_wavlm = (df["C"] - df["C"].mean()) / (df["C"].std() + 1e-8)
    zc_w2v2 = (df["C_w2v2"] - df["C_w2v2"].mean()) / (df["C_w2v2"].std() + 1e-8)
    df["C_ensemble"] = (zc_wavlm + zc_w2v2) / 2.0
    df["C_wavlm_z"] = zc_wavlm
    df.to_csv(OUT_DIR / "utterance_features_ensemble.csv", index=False)

    # ── calibrate with wavlm-only vs ensemble ────────────────────────────────
    df["cal_wavlm"] = loo_calibrate(df, "C_wavlm_z")
    df["cal_ensemble"] = loo_calibrate(df, "C_ensemble")

    m_raw = per_system_metrics(df, "raw_logit")
    m_wavlm = per_system_metrics(df, "cal_wavlm")
    m_ens = per_system_metrics(df, "cal_ensemble")

    ct = load_system_ct()

    def line(tag, m):
        return (f"| {tag} | {_mean_metric(m,'eer'):.4f} | {_mean_metric(m,'auc'):.4f} | "
                f"{_mean_metric(m,'bal_acc'):.4f} | {_mean_metric(m,'eer',ct,True):.4f} | "
                f"{_mean_metric(m,'auc',ct,True):.4f} |")

    rows = []
    for s in sorted(set(m_raw) | set(m_ens)):
        r = {"system": s, "C": ct.get(s, {}).get("C"), "T": ct.get(s, {}).get("T")}
        for tag, m in (("raw", m_raw), ("cal_wavlm", m_wavlm), ("cal_ensemble", m_ens)):
            mm = m.get(s) or {}
            for k in ("eer", "auc", "bal_acc"):
                r[f"{tag}_{k}"] = mm.get(k)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_system_metrics.csv", index=False)

    report = stratify_by_ct(m_ens, m_raw, ct)  # ensemble-cal vs raw, by C/T quartile

    # GO requires the hard-quartile EER to drop AND AUC to rise vs wavlm-only —
    # an EER-only gain that AUC contradicts is not credible.
    hq_eer_wavlm = _mean_metric(m_wavlm, "eer", ct, True)
    hq_eer_ens = _mean_metric(m_ens, "eer", ct, True)
    hq_auc_wavlm = _mean_metric(m_wavlm, "auc", ct, True)
    hq_auc_ens = _mean_metric(m_ens, "auc", ct, True)
    go = (hq_eer_ens <= hq_eer_wavlm - 0.01) and (hq_auc_ens >= hq_auc_wavlm)

    md = [
        "# P8: Cross-encoder C ensemble (WavLM L12 + wav2vec2 L8) — diagnostic gate",
        "",
        f"Utterances: {len(df)}  |  systems: {df['system'].nunique()}",
        "",
        "## Mean per-system metrics (overall | hard-C-quartile)",
        "| Score | Mean EER | Mean AUC | Mean bal_acc | HardQ EER | HardQ AUC |",
        "|-------|----------|----------|--------------|-----------|-----------|",
        line("raw_logit (no cal)", m_raw),
        line("+ wavlm-only cal", m_wavlm),
        line("+ ensemble cal", m_ens),
        "",
        f"**Decision: {'GO — ensemble adds signal (EER↓ and AUC↑ on hard quartile); proceed to P2-style injection with C_ensemble' if go else 'NO-GO — ensembling does not jointly improve hard-quartile EER and AUC over WavLM-only; not worth 2× inference'}**",
        f"(criterion: hard-quartile EER ≤ wavlm−0.01 AND AUC ≥ wavlm;  "
        f"ΔEER={hq_eer_ens - hq_eer_wavlm:+.4f}, ΔAUC={hq_auc_ens - hq_auc_wavlm:+.4f})",
        "",
        stratified_markdown(report, "Ensemble-calibrated metrics vs raw, by C/T quartile"),
        "",
        "## Note",
        "This is a calibration diagnostic, not a trained model — consistent with P1's",
        "finding that C/T are system-level (not utterance-level) signals. A GO here",
        "means the only code change for full P8 is computing C in modules_ct.py as the",
        "z-averaged (WavLM-L12, wav2vec2-L8) rog instead of WavLM-L12 alone.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:14]))
    print(f"\nsummary → {OUT_DIR/'summary.md'}")


if __name__ == "__main__":
    main()
