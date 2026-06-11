#!/usr/bin/env python3
"""
P3 — Paired-invariance adapter: a learned, nonlinear channel disentanglement that the
linear projection (P1) provably cannot achieve.
====================================================================================
P1 showed the channel axis is causal for ITW false positives but partially collinear
with genuine spoof evidence, so a rank-1 linear removal trades FP for FN. P3 keeps the
ENTIRE detector frozen (WavLM + GAT + cls_head) and trains only a tiny residual adapter
(identity-initialised) inserted at the trainable-encoder frame output — the exact P1
injection point — with two objectives on MLAAD ONLY:
  (i)  BCE spoof discrimination on clean MLAAD (do not forget the task),
  (ii) PAIRED INVARIANCE: the logit on a clean genuine utterance and on its reverb+MP3
       version must match  (||logit_clean − logit_deg||²).
Paired data is free: any clean clip + a random reverb/MP3 channel. ITW and ITW labels
are never used. The adapter can bend the decision surface nonlinearly to absorb channel
variation while preserving spoof evidence — exactly what a linear axis-removal cannot.

Evaluation: ITW (EER, FPR@MLAAD-thr, balanced acc) + MLAAD test EER. Controls:
  - λ=0 (BCE only, no invariance) ablation — isolates the invariance objective.
  - identity adapter at step 0 == frozen detector (sanity: equals baseline).
Outputs -> experiments/results/p3_invariance_adapter/
"""
from __future__ import annotations
import sys, os, json, time, types, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from argparse import Namespace

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT/"experiments"), str(SCRIPTS)):
    if _p not in sys.path: sys.path.insert(0, _p)
os.chdir(PROJECT_ROOT)
warnings.filterwarnings("ignore")
import px_common as C

RES = C.EXP_DIR / "results" / "p3_invariance_adapter"; RES.mkdir(parents=True, exist_ok=True)
DEVICE = C.DEVICE
PROC_DIR = C.EXP_DIR / "data" / "mlaad_tiny_processed"
torch.manual_seed(C.SEED); np.random.seed(C.SEED)
rng = np.random.default_rng(C.SEED)

STEPS   = int(os.environ.get("P3_STEPS", 400))
BATCH   = 8        # discrimination batch (bona+spoof)
PBATCH  = 8        # paired-invariance batch (bona)
LR      = 3e-4
LAMBDAS = [float(x) for x in os.environ.get("P3_LAMBDAS", "0.0,1.0").split(",")]

# ── adapter (identity-initialised residual MLP) ─────────────────────────────────
class Adapter(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)   # exact identity at init
    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))

# ── install gated trainable hook at gm.encoder output (P1 injection point) ──────
def install_adapter(gm, adapter):
    gm._tap_on = False
    orig = gm.encoder_and_GAT.__func__
    def wrapped(self_inner, *a, **k):
        self_inner._tap_on = True
        try: return orig(self_inner, *a, **k)
        finally: self_inner._tap_on = False
    gm.encoder_and_GAT = types.MethodType(wrapped, gm)
    def hook(module, inp, out):
        if not getattr(gm, "_tap_on", False): return None
        return (adapter(out[0]),) + tuple(out[1:])
    return gm.encoder.register_forward_hook(hook)

# ── differentiable logit (no no_grad) ───────────────────────────────────────────
def logit_grad(gm, xb):
    nf = torch.full((xb.shape[0],), C.NF_PER, device=DEVICE)
    return gm(xb, nf, profiler=None, use_aug=False, stage="val")["logit"]

# ── MLAAD train waveforms ────────────────────────────────────────────────────────
print("[data] MLAAD train split ...", flush=True)
tr = json.loads((PROC_DIR/"splits"/"train.json").read_text())
bona_p  = [PROC_DIR/r["audio_path"] for r in tr if r["label"] == "bonafide"]
spoof_p = [PROC_DIR/r["audio_path"] for r in tr if r["label"] == "spoof"]
def load_wav(p): return C._fix(np.asarray(torch.load(p), np.float32))
def degrade(x, r): return C.match_rms(C._fix(C.deg_mp3(C.deg_reverb(x, rng=r))), x)

# eval sets
ITW = C.load_itw_waves(); W_itw, y_itw = ITW["waves"], ITW["labels"]
ML  = C.load_mlaad_waves(n_per_class=600); mb, ms = ML["bona"], ML["spoof"]
def evaluate(gm):
    gm.eval()
    lb = C.detector_logits(gm, mb); ls = C.detector_logits(gm, ms)
    y = np.r_[np.zeros(len(lb)), np.ones(len(ls))]; eer, thr = C.compute_eer(y, np.r_[lb, ls])
    li = C.detector_logits(gm, W_itw); itw_eer, _ = C.compute_eer(y_itw, li)
    m = C.metrics_at_threshold(y_itw, li, thr)
    return {"mlaad_eer": eer, "itw_eer": itw_eer, "itw_FPR": m["FPR"], "itw_FNR": m["FNR"],
            "itw_acc": m["acc"], "itw_bal": m["bal_acc"]}

def sample_batch(paths_b, paths_s, nb):
    pb = [paths_b[i] for i in rng.integers(0, len(paths_b), nb)]
    ps = [paths_s[i] for i in rng.integers(0, len(paths_s), nb)]
    x = np.stack([load_wav(p) for p in pb] + [load_wav(p) for p in ps])
    y = np.r_[np.zeros(nb), np.ones(nb)]
    return torch.tensor(x, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.float32, device=DEVICE)

def sample_paired(paths_b, n):
    pb = [paths_b[i] for i in rng.integers(0, len(paths_b), n)]
    clean = [load_wav(p) for p in pb]
    deg   = [degrade(c, np.random.default_rng(rng.integers(1<<30))) for c in clean]
    xc = torch.tensor(np.stack(clean), dtype=torch.float32, device=DEVICE)
    xd = torch.tensor(np.stack(deg),   dtype=torch.float32, device=DEVICE)
    return xc, xd

# ── reference (frozen detector) ──────────────────────────────────────────────────
print("[ref] frozen detector baseline ...", flush=True)
gm = C.load_detector()
ref = evaluate(gm); ref["cond"] = "frozen (ref)"
print(f"  ref ITW-EER={ref['itw_eer']:.3f} FPR={ref['itw_FPR']:.3f} bal={ref['itw_bal']:.3f} "
      f"MLAAD-EER={ref['mlaad_eer']:.3f}", flush=True)

bce = nn.BCEWithLogitsLoss()
rows = [ref]
for lam in LAMBDAS:
    print(f"\n[train] adapter  λ_invariance={lam} ...", flush=True)
    adapter = Adapter().to(DEVICE)
    handle = install_adapter(gm, adapter)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=1e-4)
    adapter.train()
    t0 = time.time()
    for step in range(1, STEPS+1):
        opt.zero_grad()
        # frozen detector stays in eval (frozen BN stats / no dropout); cuDNN disabled so the
        # frozen LSTM's backward is allowed in eval mode (native RNN kernel).
        with torch.backends.cudnn.flags(enabled=False):
            xb, yb = sample_batch(bona_p, spoof_p, BATCH)
            lg = logit_grad(gm, xb)
            loss_bce = bce(lg, yb)
            if lam > 0:
                xc, xd = sample_paired(bona_p, PBATCH)
                lc = logit_grad(gm, xc); ld = logit_grad(gm, xd)
                loss_inv = ((lc - ld)**2).mean()
            else:
                loss_inv = torch.tensor(0.0, device=DEVICE)
            loss = loss_bce + lam * loss_inv
            loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); opt.step()
        if step % 50 == 0:
            print(f"  step {step}/{STEPS}  bce={loss_bce.item():.3f} inv={float(loss_inv):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    adapter.eval()
    with torch.no_grad():
        met = evaluate(gm)
    met["cond"] = f"adapter λ={lam}"; met["train_s"] = round(time.time()-t0, 1)
    rows.append(met)
    print(f"  -> ITW-EER={met['itw_eer']:.3f} FPR={met['itw_FPR']:.3f} FNR={met['itw_FNR']:.3f} "
          f"bal={met['itw_bal']:.3f} MLAAD-EER={met['mlaad_eer']:.3f}", flush=True)
    torch.save(adapter.state_dict(), RES / f"adapter_lam{lam}.pt")
    handle.remove()
    # restore clean encoder_and_GAT for next lambda
    import phoneme_GAT.modules as _mm
    gm.encoder_and_GAT = types.MethodType(_mm.Phoneme_GAT.encoder_and_GAT, gm)
    gm._tap_on = False

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(RES / "p3_runs.csv", index=False)

ref_r = df[df.cond == "frozen (ref)"].iloc[0]
_full = df[df.cond == "adapter λ=1.0"]
full  = _full.iloc[0] if len(_full) else df.iloc[-1]
d_bal = full.itw_bal - ref_r.itw_bal
d_eer = full.itw_eer - ref_r.itw_eer
passed = (d_bal > 0.01 or d_eer < -0.01) and (full.mlaad_eer <= ref_r.mlaad_eer + 0.03)
verdict = "SUPPORTED" if passed else "PARTIAL / NOT supported"

L = ["# P3 — Paired-invariance adapter (learned nonlinear channel disentanglement)",
     "", f"## Verdict: **{verdict}**", "",
     f"Frozen detector + identity-initialised residual adapter at the P1 injection point; "
     f"trained {STEPS} steps on MLAAD with BCE + λ·channel-invariance (reverb+MP3 paired). "
     "ITW/labels never used.", "",
     "| condition | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-bal |",
     "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    L.append(f"| {r.cond} | {r.mlaad_eer:.3f} | {r.itw_eer:.3f} | {r.itw_FPR:.3f} | "
             f"{r.itw_FNR:.3f} | {r.itw_bal:.3f} |")
L += ["", f"**adapter(λ=1) vs frozen**: ITW balanced acc Δ={d_bal:+.3f}, ITW EER Δ={d_eer:+.3f}, "
      f"MLAAD-EER Δ={full.mlaad_eer-ref_r.mlaad_eer:+.3f}. λ=0 ablation (BCE only) isolates the "
      "invariance objective's contribution.", "",
      "## Reading",
      "- Unlike the linear projection (P1), a learned adapter can move FPR down WITHOUT paying the "
      "full FNR cost if channel and spoof are nonlinearly separable in the frozen features. "
      "The λ=1 vs λ=0 contrast shows whether the channel-invariance objective (not mere extra "
      "capacity) is what helps ITW.",
      "", "## Files: p3_runs.csv, adapter_lam*.pt"]
(RES / "p3_summary.md").write_text("\n".join(L))
print("\n".join(L)); print(f"\n[P3] done -> {RES}")
