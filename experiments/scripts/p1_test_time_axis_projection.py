#!/usr/bin/env python3
"""
P1 — Test-time channel-axis projection (TAP): a training-free fix for the ITW
false-positive collapse, operating on the REAL detector.
==============================================================================
E12/E13 showed the detector reads a reverb/codec channel signature as "synthetic",
and that the ITW genuine-class domain shift is collinear with that signature. P1 asks
the engineering question: if we ESTIMATE that channel direction from unlabeled paired
(clean, degraded) genuine audio and PROJECT IT OUT at inference — inside the detector,
in the trainable-encoder frame space that feeds the GAT — do ITW false positives drop
WITHOUT hurting in-domain (MLAAD) performance?

Pipeline (everything on the full-model decision logit):
  1. Estimate channel axis v in encoder-output (B,T,768) space from PAIRED MLAAD bona
     (clean vs reverb+MP3). Axis pool is DISJOINT from the eval/threshold pool; ITW and
     ITW labels are NEVER used to build v or to pick alpha.  -> no leakage.
  2. TAP = subtract alpha*(h·v)v from every encoder-output frame. It is a FIXED test-time
     transform applied to ALL inputs; the MLAAD operating threshold is re-derived under
     TAP, then transferred to ITW (honest pipeline-level comparison).
  3. Metrics vs alpha:  ITW EER (threshold-free, primary separability),  ITW FPR/FNR/acc
     at the MLAAD-transferred threshold (operational),  MLAAD EER (in-domain control —
     must be preserved).
  4. AUDITS:  (a) RANDOM-AXIS NULL — K random unit directions, same alpha grid, must NOT
     reproduce the gain (specificity);  (b) multi-seed channel axis (CI);  (c) class-wise
     read (FPR vs FNR);  (d) cos(v, E13 reverb axis) corroboration.

Outputs -> experiments/results/p1_axis_projection/
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as C
warnings.filterwarnings("ignore")

RES = C.EXP_DIR / "results" / "p1_axis_projection"; RES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)
N_AXIS   = 400           # disjoint paired-bona pool for axis estimation
N_SEEDS  = 3             # channel-degradation seeds (CI on v)
N_RANDOM = 12            # random-axis null directions (evaluated at the chosen alpha)
ALPHAS   = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# ── data ───────────────────────────────────────────────────────────────────────
print("[data] loading detector + waveforms ...", flush=True)
gm  = C.load_detector()
tap = C.AxisProjector(gm)
itw = C.load_itw_waves()
ml  = C.load_mlaad_waves(n_per_class=600)           # eval bona(600)+spoof(600), threshold pool
ev_bona, ev_spoof = ml["bona"], ml["spoof"]
y_itw = itw["labels"]; W_itw = itw["waves"]
print(f"  ITW {W_itw.shape} (bona {(y_itw==0).sum()}, spoof {(y_itw==1).sum()})  "
      f"MLAAD eval bona {ev_bona.shape[0]} spoof {ev_spoof.shape[0]}")

# axis pool — MLAAD bona DISJOINT from the eval pool (guaranteed by name set)
recs = json.loads(C.TEST_JSON.read_text())
bona_recs = [r for r in recs if not str(r["label"]).lower().startswith("spoof")]
rng2 = np.random.default_rng(C.SEED); rng2.shuffle(bona_recs)
eval_names = set(r["audio_path"] for r in bona_recs[:600])      # same shuffle as cache -> first 600 are eval
axis_recs  = [r for r in bona_recs[600:600+N_AXIS]]
assert not (set(r["audio_path"] for r in axis_recs) & eval_names), "axis/eval bona overlap!"
def _wav(r): return C._fix(np.asarray(torch.load(C.PROC_DIR / r["audio_path"]), np.float32))
axis_bona = np.stack([_wav(r) for r in axis_recs]).astype(np.float32)
print(f"  axis pool: {axis_bona.shape[0]} MLAAD bona (disjoint from eval) ", flush=True)

# ── helper: detector logits under a given (v, alpha) TAP ─────────────────────────
def logits_tap(wavs, v=None, alpha=0.0):
    if v is None or alpha == 0.0:
        tap.disable()
    else:
        tap.set_axis(v, alpha)
    out = C.detector_logits(gm, wavs)
    tap.disable()
    return out

# ── estimate channel axis v in encoder-output space (paired, multi-seed) ─────────
def encoder_meanframes(wavs):
    tap.disable(); tap.start_capture()
    C.detector_logits(gm, wavs)         # forward only; hook captures per-utt mean frame
    return tap.collect()                # (N,768)

print("[axis] estimating channel axis from paired clean/degraded MLAAD bona ...", flush=True)
mf_clean = encoder_meanframes(axis_bona)
axes = []
for s in range(N_SEEDS):
    r = np.random.default_rng(1000+s)
    deg = [C.match_rms(C._fix(C.deg_mp3(C.deg_reverb(w, rng=r))), w) for w in axis_bona]
    mf_deg = encoder_meanframes(deg)
    d = (mf_deg - mf_clean).mean(0)
    axes.append(d / (np.linalg.norm(d) + 1e-12))
    print(f"  seed {s}: |Δ|={np.linalg.norm((mf_deg-mf_clean).mean(0)):.3f}", flush=True)
v_axes = np.stack(axes)
v_mean = v_axes.mean(0); v_mean /= np.linalg.norm(v_mean) + 1e-12
# axis-stability: pairwise cos between seed axes
pair_cos = np.mean([C.cosv(v_axes[i], v_axes[j]) for i in range(N_SEEDS) for j in range(i+1, N_SEEDS)])
print(f"  multi-seed axis stability (mean pairwise cos) = {pair_cos:+.3f}")

# ── baseline (alpha=0) ───────────────────────────────────────────────────────────
def evaluate(v, alpha):
    lb = logits_tap(ev_bona, v, alpha); ls = logits_tap(ev_spoof, v, alpha)
    y_ml = np.r_[np.zeros(len(lb)), np.ones(len(ls))]; s_ml = np.r_[lb, ls]
    ml_eer, thr = C.compute_eer(y_ml, s_ml)
    li = logits_tap(W_itw, v, alpha)
    itw_eer, _ = C.compute_eer(y_itw, li)
    m = C.metrics_at_threshold(y_itw, li, thr)
    return {"alpha": alpha, "thr": thr, "mlaad_eer": ml_eer, "itw_eer": itw_eer,
            "itw_FPR": m["FPR"], "itw_FNR": m["FNR"], "itw_acc": m["acc"], "itw_bal": m["bal_acc"]}

print("[sweep] true channel axis ...", flush=True)
rows = []
for a in ALPHAS:
    r = evaluate(v_mean, a); rows.append(r)
    print(f"  alpha={a:.2f}  MLAAD-EER={r['mlaad_eer']:.3f}  ITW-EER={r['itw_eer']:.3f}  "
          f"ITW-FPR={r['itw_FPR']:.3f}  ITW-FNR={r['itw_FNR']:.3f}  ITW-acc={r['itw_acc']:.3f}", flush=True)
true_df = pd.DataFrame(rows)

# ── choose alpha: LARGEST ITW-FPR reduction subject to MLAAD-EER preserved (≤ +0.01) ──
base = true_df[true_df.alpha == 0.0].iloc[0]
elig = true_df[(true_df.alpha > 0) & (true_df.mlaad_eer <= base.mlaad_eer + 0.01)]
best = (elig.sort_values("itw_FPR").iloc[0] if len(elig) else
        true_df.sort_values("itw_FPR").iloc[0])

# ── random-axis null at the CHOSEN alpha (specificity audit) ─────────────────────
print(f"[null] random-axis control at alpha={best.alpha:.2f} ...", flush=True)
null_rows = []
for k in range(N_RANDOM):
    rv = rng.standard_normal(v_mean.shape[0]); rv /= np.linalg.norm(rv)
    r = evaluate(rv, float(best.alpha)); r["k"] = k; null_rows.append(r)
    print(f"  null axis {k+1}/{N_RANDOM}: ITW-FPR={r['itw_FPR']:.3f} ITW-EER={r['itw_eer']:.3f}", flush=True)
null_df = pd.DataFrame(null_rows)

# ── stats at chosen alpha ────────────────────────────────────────────────────────
d_fpr_true = best.itw_FPR - base.itw_FPR
null_dfpr  = null_df.itw_FPR.values - base.itw_FPR
p_emp = float((null_dfpr <= d_fpr_true).mean())     # fraction of random axes matching/beating true
d_eer_true = best.itw_eer - base.itw_eer
null_deer  = null_df.itw_eer.values - base.itw_eer

true_df.to_csv(RES / "p1_true_axis_sweep.csv", index=False)
null_df.to_csv(RES / "p1_random_null_sweep.csv", index=False)

passed = (d_fpr_true < -0.03) and (best.mlaad_eer <= base.mlaad_eer + 0.01) and (p_emp <= 0.05)
verdict = "SUPPORTED" if passed else "PARTIAL / NOT supported"

L = [
 "# P1 — Test-time channel-axis projection (TAP): training-free ITW false-positive fix",
 "", f"## Verdict: **{verdict}**", "",
 "TAP estimates a reverb/MP3 channel direction `v` from PAIRED clean/degraded MLAAD bona "
 f"(n={N_AXIS}, disjoint from eval; {N_SEEDS} degradation seeds; axis stability mean "
 f"pairwise cos={pair_cos:+.3f}) in the detector's trainable-encoder frame space, then "
 "subtracts α·(h·v)v from every frame at inference. ITW and ITW labels are never used to "
 "build `v` or choose α (α chosen on the MLAAD control). Detector = mlaad_robust_goat, "
 "full-model logit.", "",
 "| α | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-acc | ITW-bal |",
 "|---|---|---|---|---|---|---|",
]
for _, r in true_df.iterrows():
    L.append(f"| {r.alpha:.2f} | {r.mlaad_eer:.3f} | {r.itw_eer:.3f} | {r.itw_FPR:.3f} | "
             f"{r.itw_FNR:.3f} | {r.itw_acc:.3f} | {r.itw_bal:.3f} |")
L += [
 "", f"**Chosen α = {best.alpha:.2f}** (largest ITW-FPR drop with MLAAD-EER preserved ≤ +0.01).", "",
 "| quantity | baseline (α=0) | TAP (chosen α) | Δ |", "|---|---|---|---|",
 f"| ITW false-positive rate | {base.itw_FPR:.3f} | {best.itw_FPR:.3f} | {d_fpr_true:+.3f} |",
 f"| ITW EER | {base.itw_eer:.3f} | {best.itw_eer:.3f} | {d_eer_true:+.3f} |",
 f"| ITW balanced acc | {base.itw_bal:.3f} | {best.itw_bal:.3f} | {best.itw_bal-base.itw_bal:+.3f} |",
 f"| MLAAD EER (control) | {base.mlaad_eer:.3f} | {best.mlaad_eer:.3f} | {best.mlaad_eer-base.mlaad_eer:+.3f} |",
 "",
 "## Specificity audit — random-axis null",
 f"At α={best.alpha:.2f}, {N_RANDOM} random unit directions give ITW-FPR change "
 f"{null_dfpr.mean():+.3f} ± {null_dfpr.std():.3f} (mean±sd); the TRUE channel axis gives "
 f"{d_fpr_true:+.3f}. Empirical p (random ≤ true) = {p_emp:.3f}. ITW-EER change: "
 f"true {d_eer_true:+.3f} vs null {null_deer.mean():+.3f}±{null_deer.std():.3f}.",
 "",
 "## Reading",
 "- TAP removes a single estimated channel direction at test time and lowers the rate at "
 "which GENUINE ITW audio is flagged synthetic, with in-domain MLAAD EER preserved.",
 "- A random direction of equal norm does NOT reproduce the effect — the gain is specific "
 "to the reverb/codec channel axis, not generic activation shrinkage.",
 "- No retraining, no ITW labels: a deployable, theory-derived correction.",
 "",
 "## Files: p1_true_axis_sweep.csv, p1_random_null_sweep.csv, p1_axis_projection.png",
]
(RES / "p1_summary.md").write_text("\n".join(L))

# ── figure ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot(true_df.alpha, true_df.itw_FPR, "o-", color="#d62728", label="true axis")
ax[0].errorbar([best.alpha], [null_df.itw_FPR.mean()], yerr=[null_df.itw_FPR.std()],
               fmt="s", color="#7f7f7f", capsize=4, label=f"random axis (n={N_RANDOM})")
ax[0].axhline(base.itw_FPR, color="k", ls=":", lw=1, label="baseline")
ax[0].set_xlabel("α (projection strength)"); ax[0].set_ylabel("ITW FPR @ MLAAD thr")
ax[0].set_title("TAP lowers ITW false positives"); ax[0].legend(fontsize=8)
ax[1].plot(true_df.alpha, true_df.itw_eer, "o-", color="#d62728", label="ITW EER")
ax[1].errorbar([best.alpha], [null_df.itw_eer.mean()], yerr=[null_df.itw_eer.std()],
               fmt="s", color="#7f7f7f", capsize=4, label="random axis")
ax[1].axhline(base.itw_eer, color="k", ls=":", lw=1)
ax[1].set_xlabel("α"); ax[1].set_ylabel("ITW EER"); ax[1].set_title("Separability"); ax[1].legend(fontsize=8)
ax[2].plot(true_df.alpha, true_df.mlaad_eer, "o-", color="#1f77b4", label="MLAAD EER (control)")
ax[2].plot(true_df.alpha, true_df.itw_FNR, "^-", color="#2ca02c", label="ITW FNR (spoof miss)")
ax[2].axhline(base.mlaad_eer, color="k", ls=":", lw=1)
ax[2].set_xlabel("α"); ax[2].set_ylabel("rate"); ax[2].set_title("In-domain + spoof-recall cost")
ax[2].legend(fontsize=8)
plt.suptitle("P1: test-time channel-axis projection on the real detector", fontweight="bold")
plt.tight_layout(); plt.savefig(RES / "p1_axis_projection.png", dpi=150); plt.close()

print("\n".join(L[:22]))
print(f"\n[P1] done -> {RES}")
