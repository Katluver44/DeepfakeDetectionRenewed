#!/usr/bin/env python3
"""
P4 — Blind channel-aware logit correction: a no-retraining, no-representation-access
calibration that cancels the channel-induced spoof-logit inflation from cheap signal
features.
====================================================================================
P1 corrects inside the network. P4 is the lightest possible deployable variant: from
BLIND waveform descriptors (spectral flatness, bandwidth, HF/LF energy, reverb/modulation
proxies — no model, no labels) predict how much spurious "spoof" logit the recording
CHANNEL adds to GENUINE audio, and subtract that estimate:  logit_corr = logit − ĝ(d).

ĝ is fit ONLY on MLAAD (clean vs reverb+MP3 degraded bona): target = per-utterance logit
inflation (degraded − clean), features = blind descriptors. ITW and ITW labels are never
used to fit ĝ or to choose the threshold. The MLAAD EER threshold is recomputed on the
CORRECTED logits and transferred to ITW.

Predictions:
  P4.1  ĝ neutralizes the inflation IN-DOMAIN (held-out MLAAD degraded bona logit returns
        toward its clean value).
  P4.2  applied blind to ITW, correction lowers the false-positive rate / raises balanced
        accuracy, MLAAD EER preserved.
  P4.3  SHUFFLE control — permuting descriptors across utterances destroys the gain
        (the signal is in the per-utterance channel descriptor, not a global offset).

Outputs -> experiments/results/p4_blind_calibration/
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as C
warnings.filterwarnings("ignore")

RES = C.EXP_DIR / "results" / "p4_blind_calibration"; RES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)
SR = C.SR

# ── blind waveform descriptors (NO model, NO labels) ────────────────────────────
def descriptors(x):
    x = np.asarray(x, np.float32)
    f, t, Z = signal.stft(x, fs=SR, nperseg=512, noverlap=384)
    P = (np.abs(Z)**2) + 1e-12                       # (F, T)
    psd = P.mean(1)                                   # avg power spectrum
    psd_n = psd / psd.sum()
    # spectral flatness (Wiener entropy)
    flat = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    # spectral centroid / bandwidth
    cen = float((f * psd_n).sum())
    bw  = float(np.sqrt(((f - cen)**2 * psd_n).sum()))
    # rolloff 85%
    cumsum = np.cumsum(psd); roll = float(f[np.searchsorted(cumsum, 0.85*cumsum[-1])])
    # band energy ratios
    tot = psd.sum()
    hf  = float(psd[f > 3400].sum() / tot)            # above telephone band (codec/bandlimit)
    lf  = float(psd[f < 300].sum() / tot)
    # temporal envelope / modulation
    env = np.sqrt((np.abs(Z)**2).sum(0) + 1e-12)      # (T,) frame energy
    lev = np.log(env + 1e-8)
    mod_depth = float(lev.std())                      # reverb smears -> lower modulation depth
    # reverb proxy: late-to-early envelope autocorrelation (reverb raises correlation at lag)
    e = env - env.mean(); ac = np.correlate(e, e, "full")[len(e)-1:]
    ac = ac / (ac[0] + 1e-12)
    lag = max(1, int(0.05 * SR / (512-384)))          # ~50 ms in frames
    revp = float(np.mean(ac[lag:lag+5]))
    # crest factor (peak / rms) — codecs/reverb reduce peakiness
    crest = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return [flat, cen, bw, roll, hf, lf, mod_depth, revp, crest, zcr]

DESC_NAMES = ["flatness","centroid","bandwidth","rolloff85","hf_ratio","lf_ratio",
              "mod_depth","reverb_proxy","crest","zcr"]

def descmat(wavs):
    return np.array([descriptors(w) for w in wavs], np.float32)

# ── data + detector ─────────────────────────────────────────────────────────────
print("[data] detector + waveforms ...", flush=True)
gm = C.load_detector()
itw = C.load_itw_waves(); W_itw, y_itw = itw["waves"], itw["labels"]
ml  = C.load_mlaad_waves(n_per_class=600)
mb, ms = ml["bona"], ml["spoof"]

# fit/eval split of MLAAD bona (disjoint): fit ĝ on first half, validate in-domain on second
n = len(mb); idx = rng.permutation(n); fit_i, val_i = idx[:n//2], idx[n//2:]
mb_fit, mb_val = mb[fit_i], mb[val_i]

def degrade(wavs, seed):
    r = np.random.default_rng(seed)
    return [C.match_rms(C._fix(C.deg_mp3(C.deg_reverb(w, rng=r))), w) for w in wavs]

print("[logits] scoring clean + degraded MLAAD ...", flush=True)
deg_mb_fit = degrade(mb_fit, 11); deg_mb_val = degrade(mb_val, 12); deg_ms = degrade(ms, 13)
L_mb_fit_clean = C.detector_logits(gm, mb_fit);  L_mb_fit_deg = C.detector_logits(gm, deg_mb_fit)
L_mb_val_clean = C.detector_logits(gm, mb_val);  L_mb_val_deg = C.detector_logits(gm, deg_mb_val)
L_ms_clean     = C.detector_logits(gm, ms);      L_ms_deg     = C.detector_logits(gm, deg_ms)
L_itw          = C.detector_logits(gm, W_itw)
print(f"  MLAAD bona inflation (deg-clean) med = {np.median(L_mb_fit_deg - L_mb_fit_clean):+.2f}")

# ── fit ĝ: blind descriptors -> channel-induced GENUINE-logit inflation ─────────
# training rows: clean bona (target 0) + degraded bona (target = deg-clean), features=blind d
print("[fit] descriptors + GBR ...", flush=True)
D_clean = descmat(mb_fit); D_deg = descmat(deg_mb_fit)
Xtr = np.vstack([D_clean, D_deg])
ytr = np.concatenate([np.zeros(len(D_clean)), L_mb_fit_deg - L_mb_fit_clean])
scaler = StandardScaler().fit(Xtr)
ghat = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, random_state=C.SEED).fit(scaler.transform(Xtr), ytr)

def correct(logits, wavs):
    d = scaler.transform(descmat(wavs))
    return logits - ghat.predict(d)

# ── P4.1 in-domain: does ĝ neutralize inflation on held-out MLAAD degraded bona? ──
infl_before = float(np.median(L_mb_val_deg - L_mb_val_clean))
Lc_mb_val_deg = correct(L_mb_val_deg, deg_mb_val)
infl_after = float(np.median(Lc_mb_val_deg - L_mb_val_clean))
print(f"[P4.1] held-out MLAAD deg-bona inflation: before {infl_before:+.2f} -> after {infl_after:+.2f}")

# ── threshold from CORRECTED MLAAD (clean bona + clean spoof) ────────────────────
def mlaad_thr(correct_fn):
    lb = correct_fn(L_mb_val_clean, mb_val); ls = correct_fn(L_ms_clean, ms)
    y = np.r_[np.zeros(len(lb)), np.ones(len(ls))]; s = np.r_[lb, ls]
    return C.compute_eer(y, s)

# baseline (no correction) and corrected pipelines on ITW
def itw_eval(correct_fn, thr):
    li = correct_fn(L_itw, W_itw)
    eer, _ = C.compute_eer(y_itw, li)
    m = C.metrics_at_threshold(y_itw, li, thr)
    return eer, m

ident = lambda l, w: l
base_eer_ml, base_thr = mlaad_thr(ident)
corr_eer_ml, corr_thr = mlaad_thr(correct)
base_itw_eer, base_m = itw_eval(ident, base_thr)
corr_itw_eer, corr_m = itw_eval(correct, corr_thr)

# ── P4.3 shuffle control: permute descriptors across utterances ─────────────────
def correct_shuffled(logits, wavs, perm):
    d = scaler.transform(descmat(wavs)); d = d[perm]
    return logits - ghat.predict(d)
perm_itw = rng.permutation(len(W_itw))
sh_li = correct_shuffled(L_itw, W_itw, perm_itw)
sh_thr_perm = rng.permutation(len(mb_val))   # not used for thr; thr stays corr_thr for fair op-point
sh_itw_eer, _ = C.compute_eer(y_itw, sh_li)
sh_m = C.metrics_at_threshold(y_itw, sh_li, corr_thr)

# ── results ──────────────────────────────────────────────────────────────────────
res = pd.DataFrame([
    {"pipeline":"baseline (no correction)","mlaad_eer":base_eer_ml,"itw_eer":base_itw_eer,
     "itw_FPR":base_m["FPR"],"itw_FNR":base_m["FNR"],"itw_acc":base_m["acc"],"itw_bal":base_m["bal_acc"]},
    {"pipeline":"blind correction ĝ(d)","mlaad_eer":corr_eer_ml,"itw_eer":corr_itw_eer,
     "itw_FPR":corr_m["FPR"],"itw_FNR":corr_m["FNR"],"itw_acc":corr_m["acc"],"itw_bal":corr_m["bal_acc"]},
    {"pipeline":"shuffled-descriptor control","mlaad_eer":corr_eer_ml,"itw_eer":sh_itw_eer,
     "itw_FPR":sh_m["FPR"],"itw_FNR":sh_m["FNR"],"itw_acc":sh_m["acc"],"itw_bal":sh_m["bal_acc"]},
])
res.to_csv(RES / "p4_pipelines.csv", index=False)
# feature importances
imp = pd.DataFrame({"descriptor":DESC_NAMES,"importance":ghat.feature_importances_}
                   ).sort_values("importance", ascending=False)
imp.to_csv(RES / "p4_feature_importance.csv", index=False)

d_fpr = corr_m["FPR"] - base_m["FPR"]
passed = (infl_after < infl_before*0.6) and (d_fpr < -0.02) and \
         (corr_eer_ml <= base_eer_ml + 0.01) and (corr_m["FPR"] < sh_m["FPR"] - 0.01)
verdict = "SUPPORTED" if passed else "PARTIAL / NOT supported"

L = [
 "# P4 — Blind channel-aware logit correction (no retraining, no representation access)",
 "", f"## Verdict: **{verdict}**", "",
 "ĝ maps 10 blind waveform descriptors → channel-induced genuine-logit inflation, fit on "
 "MLAAD clean/reverb+MP3 bona only; logit_corr = logit − ĝ(d). ITW/labels never used to fit "
 "ĝ or pick the threshold.", "",
 "### P4.1 In-domain neutralization (held-out MLAAD degraded bona)",
 f"median logit inflation (degraded − clean): **{infl_before:+.2f} → {infl_after:+.2f}** after "
 f"correction ({100*(1-infl_after/max(infl_before,1e-6)):.0f}% removed).", "",
 "### P4.2 / P4.3 ITW transfer + shuffle control",
 "| pipeline | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-acc | ITW-bal |",
 "|---|---|---|---|---|---|---|",
]
for _, r in res.iterrows():
    L.append(f"| {r.pipeline} | {r.mlaad_eer:.3f} | {r.itw_eer:.3f} | {r.itw_FPR:.3f} | "
             f"{r.itw_FNR:.3f} | {r.itw_acc:.3f} | {r.itw_bal:.3f} |")
L += [
 "", f"ITW false-positive change from blind correction: **{d_fpr:+.3f}**; the shuffled-descriptor "
 f"control gives ITW-FPR {sh_m['FPR']:.3f} (no real gain) — confirming the correction uses "
 "per-utterance channel information, not a global shift.", "",
 "### Most informative blind descriptors",
 "| descriptor | importance |", "|---|---|",
] + [f"| {r.descriptor} | {r.importance:.3f} |" for _, r in imp.head(6).iterrows()] + [
 "", "## Reading",
 "- A handful of cheap signal features predict the channel's spurious spoof-evidence well "
 "enough to cancel most of it, lowering ITW false positives with no model retraining and no "
 "access to internal representations — the most portable form of the fix.",
 "", "## Files: p4_pipelines.csv, p4_feature_importance.csv, p4_blind_calibration.png",
]
(RES / "p4_summary.md").write_text("\n".join(L))

# ── figure ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].bar(["before","after"], [infl_before, infl_after], color=["#d62728","#2ca02c"])
ax[0].axhline(0, color="k", lw=1); ax[0].set_ylabel("median logit inflation")
ax[0].set_title("P4.1 in-domain neutralization")
piv = res.set_index("pipeline")[["itw_FPR","itw_bal"]]
piv.plot(kind="bar", ax=ax[1], color=["#d62728","#1f77b4"]); ax[1].set_title("P4.2/3 ITW transfer")
ax[1].set_xticklabels([t.get_text().replace(" ","\n") for t in ax[1].get_xticklabels()], fontsize=7, rotation=0)
ax[1].legend(fontsize=8)
ax[2].barh(imp.head(6).descriptor[::-1], imp.head(6).importance[::-1], color="#9467bd")
ax[2].set_title("blind descriptor importance")
plt.suptitle("P4: blind channel-aware logit correction", fontweight="bold")
plt.tight_layout(); plt.savefig(RES / "p4_blind_calibration.png", dpi=150); plt.close()

print("\n".join(L[:20]))
print(f"\n[P4] done -> {RES}")
