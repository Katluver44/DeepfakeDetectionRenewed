#!/usr/bin/env python3
"""
P2 — Causal minimal augmentation: does training with the CAUSALLY-IDENTIFIED channel
augmentation (reverb+MP3, from E13) beat broad/standard augmentation at equal budget?
====================================================================================
E10 ruled out additive noise as the ITW mechanism; E13 showed reverb + lossy codec
(MP3) are the causal drivers of ITW false positives, and that bandlimiting/noise are
not. The standard "robust" recipe here augments with NOISE+PITCH+REVERB. P2 tests the
augmentation-selection claim: finetuning from the same base, for the same #epochs and
same augmentation probability, the CAUSAL recipe (reverb+MP3) should generalize to ITW
better than the BROAD recipe (noise+pitch+reverb), which spends budget on noise/pitch
that E10/E13 show are not the ITW mechanism.

Conditions (all finetune mlaad_robust_goat, equal epochs/seeds/aug_prob):
  none    : aug_prob=0                      (finetune-only control)
  broad   : noise+pitch+reverb (the existing 'robust' recipe)
  causal  : reverb+MP3            (E13-identified channel effects)

Evaluation: ITW (EER, FPR@MLAAD-thr, balanced acc) + MLAAD test EER (in-domain).
Outputs -> experiments/results/p2_causal_aug/
"""
from __future__ import annotations
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT/"experiments"), str(SCRIPTS)):
    if _p not in sys.path: sys.path.insert(0, _p)
os.chdir(PROJECT_ROOT)
warnings.filterwarnings("ignore")

import px_common as C
from train_mlaad_adversarial import (HP, MAALDSplitDataset, set_seed,
                                      aug_noise, aug_pitch_up, aug_reverb, _match_len)

RES = C.EXP_DIR / "results" / "p2_causal_aug"; RES.mkdir(parents=True, exist_ok=True)
CKPT_DIR = C.EXP_DIR / "checkpoints"
BASE_CKPT = CKPT_DIR / "mlaad_robust_goat.ckpt"
PROC_DIR  = C.EXP_DIR / "data" / "mlaad_tiny_processed"
SEEDS   = [int(x) for x in os.environ.get("P2_SEEDS", "42,123").split(",")]
EPOCHS  = int(os.environ.get("P2_EPOCHS", 5))
CONDS   = os.environ.get("P2_CONDS", "none,broad,causal").split(",")
AUG_PROB = 0.35

# ── MP3 augmentation (the causal ingredient missing from the standard recipe) ────
from train_mlaad_adversarial import TARGET_SR
def aug_mp3(wav: torch.Tensor) -> torch.Tensor:
    x = wav.squeeze(0).cpu().numpy().astype(np.float32)
    try:
        y = C.deg_mp3(x, bitrate=16)
    except Exception:
        return wav
    y = C._fix(y); y = C.match_rms(y, x)
    return torch.from_numpy(y).unsqueeze(0).to(wav.dtype)

AUG_SETS = {
    "none":   ([], []),
    "broad":  ([aug_noise, aug_pitch_up, aug_reverb], [0.50, 0.30, 0.20]),
    "causal": ([aug_reverb, aug_mp3], [0.50, 0.50]),
}

# ── evaluation on ITW + MLAAD (reuse px_common) ──────────────────────────────────
print("[eval-data] loading ITW + MLAAD eval waveforms ...", flush=True)
ITW = C.load_itw_waves(); W_itw, y_itw = ITW["waves"], ITW["labels"]
ML  = C.load_mlaad_waves(n_per_class=600)
mb, ms = ML["bona"], ML["spoof"]

def evaluate_model(gm):
    lb = C.detector_logits(gm, mb); ls = C.detector_logits(gm, ms)
    y_ml = np.r_[np.zeros(len(lb)), np.ones(len(ls))]; s_ml = np.r_[lb, ls]
    ml_eer, thr = C.compute_eer(y_ml, s_ml)
    li = C.detector_logits(gm, W_itw)
    itw_eer, _ = C.compute_eer(y_itw, li)
    m = C.metrics_at_threshold(y_itw, li, thr)
    return {"mlaad_eer": ml_eer, "itw_eer": itw_eer, "itw_FPR": m["FPR"],
            "itw_FNR": m["FNR"], "itw_acc": m["acc"], "itw_bal": m["bal_acc"]}

# ── train one (condition, seed) ──────────────────────────────────────────────────
def train_one(cond, seed):
    from argparse import Namespace
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from phoneme_GAT.modules import Phoneme_GAT_lit
    from callbacks import EER_Callback
    from callbacks_rational import BinaryACC_Callback, BinaryAUC_Callback
    set_seed(seed)
    fns, wts = AUG_SETS[cond]
    splits = PROC_DIR / "splits"
    tr = MAALDSplitDataset(splits/"train.json", PROC_DIR, mode="train", balance=True, seed=seed,
                           aug_prob=(AUG_PROB if fns else 0.0), aug_fns=fns, aug_weights=wts)
    va = MAALDSplitDataset(splits/"val.json", PROC_DIR, mode="eval", balance=True, seed=seed)
    trdl = DataLoader(tr, batch_size=HP["batch_size"], shuffle=True, num_workers=3,
                      pin_memory=True, drop_last=True, persistent_workers=True)
    vadl = DataLoader(va, batch_size=HP["batch_size"], shuffle=False, num_workers=2,
                      pin_memory=True, persistent_workers=True)
    model = Phoneme_GAT_lit.load_from_checkpoint(str(BASE_CKPT),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    stem = f"p2_{cond}_seed{seed}"
    cbs = [EER_Callback(batch_key="label", output_key="logit"),
           BinaryACC_Callback(batch_key="label", output_key="logit"),
           BinaryAUC_Callback(batch_key="label", output_key="logit"),
           ModelCheckpoint(dirpath=str(CKPT_DIR), filename=stem+"-{epoch:02d}-{val-eer:.4f}",
                           monitor="val-eer", mode="min", save_last=True)]
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=EPOCHS, precision="bf16-mixed",
                         logger=CSVLogger(save_dir=str(RES/"logs"), name=stem, version=""),
                         callbacks=cbs, log_every_n_steps=20, enable_progress_bar=False,
                         deterministic=False)
    t0 = time.time(); trainer.fit(model, trdl, vadl); dt = time.time()-t0
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model.model)
    metrics.update({"cond": cond, "seed": seed, "train_s": round(dt, 1)})
    print(f"  [{cond} s{seed}] {dt:.0f}s  ITW-EER={metrics['itw_eer']:.3f} "
          f"ITW-FPR={metrics['itw_FPR']:.3f} MLAAD-EER={metrics['mlaad_eer']:.3f}", flush=True)
    return metrics

# ── reference: un-finetuned robust_goat ──────────────────────────────────────────
print("[ref] scoring un-finetuned robust_goat ...", flush=True)
gm0 = C.load_detector()
ref = evaluate_model(gm0); ref.update({"cond": "robust_goat (ref)", "seed": -1, "train_s": 0})
print(f"  ref: ITW-EER={ref['itw_eer']:.3f} ITW-FPR={ref['itw_FPR']:.3f} MLAAD-EER={ref['mlaad_eer']:.3f}")
del gm0; torch.cuda.empty_cache()

all_rows = [ref]
for cond in CONDS:
    for seed in SEEDS:
        try:
            all_rows.append(train_one(cond, seed))
        except Exception as e:
            print(f"  [ERROR {cond} s{seed}] {e}", flush=True)
        torch.cuda.empty_cache()
        import json as _j
        (RES/"p2_runs_partial.json").write_text(_j.dumps(all_rows, indent=2))

import pandas as pd
df = pd.DataFrame(all_rows); df.to_csv(RES / "p2_runs.csv", index=False)
agg = (df[df.seed >= 0].groupby("cond")[["mlaad_eer","itw_eer","itw_FPR","itw_FNR","itw_bal"]]
       .agg(["mean","std"]))
agg.to_csv(RES / "p2_aggregate.csv")

def m(cond, k):
    s = df[df.cond == cond][k]; return (s.mean(), s.std())
causal_fpr, _ = m("causal","itw_FPR"); broad_fpr, _ = m("broad","itw_FPR")
causal_eer, _ = m("causal","itw_eer"); broad_eer, _ = m("broad","itw_eer")
causal_mle, _ = m("causal","mlaad_eer"); broad_mle, _ = m("broad","mlaad_eer")
passed = (causal_fpr < broad_fpr - 0.01) and (causal_mle <= broad_mle + 0.02)
verdict = "SUPPORTED" if passed else "PARTIAL / NOT supported"

L = ["# P2 — Causal minimal augmentation (reverb+MP3) vs broad augmentation",
     "", f"## Verdict: **{verdict}**", "",
     f"Finetune mlaad_robust_goat, {EPOCHS} epochs, seeds {SEEDS}, aug_prob={AUG_PROB}. "
     "Conditions differ ONLY in the augmentation set.", "",
     "| condition | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-bal |",
     "|---|---|---|---|---|---|"]
for cond in ["robust_goat (ref)","none","broad","causal"]:
    sub = df[df.cond == cond]
    if not len(sub): continue
    def f(k): return f"{sub[k].mean():.3f}" + (f"±{sub[k].std():.3f}" if len(sub)>1 else "")
    L.append(f"| {cond} | {f('mlaad_eer')} | {f('itw_eer')} | {f('itw_FPR')} | {f('itw_FNR')} | {f('itw_bal')} |")
L += ["", f"**causal vs broad**: ITW-FPR {causal_fpr:.3f} vs {broad_fpr:.3f} "
      f"(Δ={causal_fpr-broad_fpr:+.3f}); ITW-EER {causal_eer:.3f} vs {broad_eer:.3f}; "
      f"MLAAD-EER {causal_mle:.3f} vs {broad_mle:.3f}.", "",
      "## Reading",
      "- If causal (reverb+MP3) lowers ITW false positives more than broad (noise+pitch+reverb) "
      "at equal budget, augmentation selection guided by the CAUSAL channel diagnosis beats "
      "generic augmentation — a mechanism-driven training recommendation.",
      "", "## Files: p2_runs.csv, p2_aggregate.csv"]
(RES / "p2_summary.md").write_text("\n".join(L))
print("\n".join(L)); print(f"\n[P2] done -> {RES}")
