#!/usr/bin/env python3
"""
E10 (Test 1) — Is T's failure on In-The-Wild a background-noise artifact?
========================================================================

Hypothesis (story3): T = vel_entropy@L9 stops predicting spoof on ITW *because*
ITW clips carry uncontrolled background noise / random acoustic artifacts that
inflate and scramble the L9 frame-to-frame velocity entropy. On clean MLAAD T
carries (weak) signal (rho(T,logit)=+0.150); on ITW it is dead (rho=+0.004,
p=0.84; spoof-vs-bona Cliff's d=-0.077). This script is the *observational* test:
it does NOT re-synthesize audio. It reuses the per-sample T / C / detector logits
already in outputs/e5_itw_utterance_ct.csv and adds a per-sample no-reference SNR
proxy (WADA-SNR) computed on the identical 3 s center-crop that T was measured on.

Tests
-----
  (a) Noise-driven?      Spearman rho(T, SNR) and rho(C, SNR).  Hypothesis: T
                         strongly negative (noisier -> higher T), C weak.
  (b) Stratified signal  Split samples at median SNR (clean vs noisy half). In
                         each half: rho(T, logit) on spoof + spoof-vs-bona Cliff's
                         d on T. Hypothesis: T discriminates in the CLEAN half,
                         is dead in the NOISY half. C run identically as control.
  (c) Partial corr       partial Spearman rho(T,logit | SNR), rho(T,label | SNR).
                         Does any T signal survive controlling for noise?
  (d) Per-quintile trend rho(T,logit) and Cliff's d(T) across SNR quintiles.

C is carried through every test as a noise-robustness control: a clean
dissociation = C keeps its (weak) signal under noise-stratification while T's
collapses with SNR.

Outputs -> experiments/results/e10_t_noise/
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE   = Path(__file__).resolve().parents[2]
OUT    = BASE / "outputs"
RESDIR = BASE / "experiments" / "results" / "e10_t_noise"
RESDIR.mkdir(parents=True, exist_ok=True)

ITW_CSV = OUT / "e5_itw_utterance_ct.csv"
RW_DIR  = (BASE / "data" / "in_the_wild" / "downloads" / "extracted" /
           "c3c93f2f54ac2d261fa7010629351505bd6e05597ea22fd4a35c92dda590a3bf" /
           "release_in_the_wild")

TARGET_SR  = 16_000
TARGET_LEN = 3 * TARGET_SR          # 48000 — same center-crop T was measured on
LOGIT_COL  = "logit_robust_goat"    # in-distribution headline detector (matches E5/E9)

# ── WADA-SNR (Kim & Stern, ICASSP 2008) ─────────────────────────────────────────
# Core statistic v3 = log(E|x|) - E[log|x|] is *monotone increasing* in SNR, so all
# rank-based stats below (Spearman, median split, partial Spearman) are EXACT using
# v3 and independent of the dB lookup. The canonical 121-pt table is used ONLY to
# attach a human-readable dB value (db in [-20,100]); treat dB as display-scale.
_WADA_DB = np.arange(-20, 101)
_WADA_G = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006,
    0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272,
    0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373,
    0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
    0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985,
    0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 ,
    1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995,
    1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313,
    1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619,
    1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
    1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915,
    1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517,
    1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
    1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134,
    1.60826199, 1.61004547, 1.61192472, 1.61359585, 1.61534225, 1.61688905,
    1.61838795, 1.61985213, 1.62135253, 1.62268673, 1.62390685, 1.62513166,
    1.62632042, 1.6274023 , 1.62842952, 1.62945603, 1.6303307 , 1.63128026,
    1.63204102])

def wada_stat_and_db(wav: np.ndarray):
    """Return (v3 statistic, dB estimate). v3 is the exact rank carrier."""
    eps = 1e-10
    w = wav.astype(np.float64)
    m = np.abs(w).max()
    if m < eps:
        return np.nan, np.nan
    w = w / m
    aw = np.abs(w)
    aw[aw < eps] = eps
    v1 = max(eps, aw.mean())
    v2 = np.log(aw).mean()
    v3 = np.log(v1) - v2                       # monotone increasing in SNR
    if v3 <= _WADA_G[0]:
        idx = 0
    elif v3 >= _WADA_G[-1]:
        idx = len(_WADA_G) - 1
    else:
        idx = int(np.where(_WADA_G >= v3)[0].min())
    return float(v3), float(_WADA_DB[idx])

def center_crop_pad(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if len(wav) < TARGET_LEN:
        reps = -(-TARGET_LEN // len(wav))
        wav = np.tile(wav, reps)
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN]

# ── stats helpers ───────────────────────────────────────────────────────────────
def cliffs_delta(a, b):
    """Cliff's d for a vs b via MW-U (a>b => positive). Returns (delta, p_two)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * U / (len(a) * len(b)) - 1.0), float(p)

def partial_spearman(x, y, z):
    """Spearman partial corr of x,y controlling z (rank-linear residualization)."""
    x, y, z = map(lambda v: rankdata(np.asarray(v, float)), (x, y, z))
    Z = np.c_[np.ones_like(z), z]
    def resid(a):
        coef, *_ = np.linalg.lstsq(Z, a, rcond=None)
        return a - Z @ coef
    rx, ry = resid(x), resid(y)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)

# ═════════════════════════════════════════════════════════════════════════════════
print("[E10] loading per-sample C/T/logits from E5 ...", flush=True)
df = pd.read_csv(ITW_CSV)
assert LOGIT_COL in df.columns, f"{LOGIT_COL} missing in {ITW_CSV}"
df["sample_id"] = df["sample_id"].astype(str)
print(f"  rows={len(df)}  spoof={(df.label=='spoof').sum()} bona={(df.label=='bona-fide').sum()}")

print("[E10] computing WADA-SNR on identical 3s center-crops ...", flush=True)
v3s, dbs, ok = [], [], []
for i, sid in enumerate(df["sample_id"].tolist()):
    p = RW_DIR / f"{sid}.wav"
    try:
        arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if sr != TARGET_SR:                    # ITW is 16k; guard anyway
            import torchaudio.transforms as TAT, torch
            arr = TAT.Resample(sr, TARGET_SR)(torch.tensor(arr).unsqueeze(0)).squeeze(0).numpy()
        v3, db = wada_stat_and_db(center_crop_pad(np.asarray(arr)))
        v3s.append(v3); dbs.append(db); ok.append(True)
    except Exception as e:
        v3s.append(np.nan); dbs.append(np.nan); ok.append(False)
    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{len(df)}", flush=True)
df["wada_v3"] = v3s
df["wada_snr_db"] = dbs
df = df[np.isfinite(df["wada_v3"])].reset_index(drop=True)
print(f"  usable rows with SNR: {len(df)} ({np.mean(ok)*100:.1f}% loaded)")
df.to_csv(RESDIR / "e10_itw_ct_snr.csv", index=False)

SNR  = df["wada_v3"].values                    # rank carrier
SNRdb = df["wada_snr_db"].values
T    = df["T"].values
C    = df["C"].values
logit = df[LOGIT_COL].values
is_spoof = (df["label"] == "spoof").values
y_label  = is_spoof.astype(int)                # 1=spoof

# ── (a) noise-driven? ────────────────────────────────────────────────────────────
def sp(a, b):
    r = stats.spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)
rTS, pTS = sp(T, SNR)
rCS, pCS = sp(C, SNR)
rTS_sp, pTS_sp = sp(T[is_spoof], SNR[is_spoof])
rCS_sp, pCS_sp = sp(C[is_spoof], SNR[is_spoof])

# ── (b) stratified signal: clean vs noisy half (median split on SNR) ─────────────
med = np.median(SNR)
clean = SNR >= med                              # high SNR
noisy = SNR <  med
def discrim(mask, name):
    m_sp = mask & is_spoof
    m_bo = mask & (~is_spoof)
    rTl, pTl = sp(T[m_sp], logit[m_sp])
    rCl, pCl = sp(C[m_sp], logit[m_sp])
    dT, pdT = cliffs_delta(T[m_sp], T[m_bo])    # spoof vs bona on T
    dC, pdC = cliffs_delta(C[m_sp], C[m_bo])
    return {"stratum": name, "n_spoof": int(m_sp.sum()), "n_bona": int(m_bo.sum()),
            "snr_db_med": float(np.median(SNRdb[mask])),
            "rho_T_logit": rTl, "p_T_logit": pTl,
            "rho_C_logit": rCl, "p_C_logit": pCl,
            "cliff_T_spoofVbona": dT, "p_cliff_T": pdT,
            "cliff_C_spoofVbona": dC, "p_cliff_C": pdC}
strat = [discrim(np.ones_like(clean), "ALL"),
         discrim(clean, "CLEAN (hi-SNR)"),
         discrim(noisy, "NOISY (lo-SNR)")]

# ── (c) partial correlations controlling SNR ─────────────────────────────────────
prTl, ppTl = partial_spearman(T[is_spoof], logit[is_spoof], SNR[is_spoof])
prCl, ppCl = partial_spearman(C[is_spoof], logit[is_spoof], SNR[is_spoof])
prTy, ppTy = partial_spearman(T, y_label, SNR)      # T->label | SNR (all samples)
prCy, ppCy = partial_spearman(C, y_label, SNR)
# raw (uncontrolled) label corr for comparison
rrTy, prTy_ = sp(T, y_label)
rrCy, prCy_ = sp(C, y_label)

# ── (d) per-quintile trend ───────────────────────────────────────────────────────
q = np.quantile(SNR, [0, .2, .4, .6, .8, 1.0])
qbin = np.clip(np.digitize(SNR, q[1:-1]), 0, 4)
quint = []
for b in range(5):
    m = qbin == b
    m_sp = m & is_spoof; m_bo = m & (~is_spoof)
    rTl, _ = sp(T[m_sp], logit[m_sp])
    dT, _ = cliffs_delta(T[m_sp], T[m_bo])
    dC, _ = cliffs_delta(C[m_sp], C[m_bo])
    quint.append({"quintile": b + 1, "snr_db_med": float(np.median(SNRdb[m])),
                  "n_spoof": int(m_sp.sum()), "rho_T_logit": rTl,
                  "cliff_T": dT, "cliff_C": dC, "mean_T": float(T[m].mean())})
quint_df = pd.DataFrame(quint)
quint_df.to_csv(RESDIR / "e10_snr_quintile_trend.csv", index=False)

# ═════════════════════════════════════════════════════════════════════════════════
# Verdict
# Hypothesis SUPPORTED if: (1) T is materially noise-driven (|rho(T,SNR)|>=0.2),
# AND (2) T's discriminativity is materially stronger in the CLEAN half than NOISY
# half (|cliff_T| or |rho_T_logit| larger in clean), AND (3) C is comparatively
# robust (its clean-vs-noisy gap << T's). Reported transparently regardless.
clean_s, noisy_s = strat[1], strat[2]
T_clean_strength = max(abs(clean_s["cliff_T_spoofVbona"]), abs(clean_s["rho_T_logit"]))
T_noisy_strength = max(abs(noisy_s["cliff_T_spoofVbona"]), abs(noisy_s["rho_T_logit"]))
C_clean_strength = max(abs(clean_s["cliff_C_spoofVbona"]), abs(clean_s["rho_C_logit"]))
C_noisy_strength = max(abs(noisy_s["cliff_C_spoofVbona"]), abs(noisy_s["rho_C_logit"]))
cond_noise_driven = abs(rTS) >= 0.20
cond_clean_recovers = T_clean_strength > T_noisy_strength + 0.03
cond_C_robust = (C_clean_strength - C_noisy_strength) < (T_clean_strength - T_noisy_strength)
supported = cond_noise_driven and cond_clean_recovers
verdict = "SUPPORTED" if supported else ("PARTIAL" if (cond_noise_driven or cond_clean_recovers) else "NOT supported")

# ── figures ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(11, 4.3))
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(SNRdb[is_spoof], T[is_spoof], s=6, alpha=.25, c="#d62728", label="spoof")
ax1.scatter(SNRdb[~is_spoof], T[~is_spoof], s=6, alpha=.25, c="#2ca02c", label="bona")
ax1.set_xlabel("WADA-SNR (dB, display)"); ax1.set_ylabel("T = vel_entropy@L9")
ax1.set_title(f"T vs SNR   Spearman rho={rTS:+.3f} (p={pTS:.1e})")
ax1.legend(markerscale=2, fontsize=8)
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(SNRdb[is_spoof], C[is_spoof], s=6, alpha=.25, c="#d62728")
ax2.scatter(SNRdb[~is_spoof], C[~is_spoof], s=6, alpha=.25, c="#2ca02c")
ax2.set_xlabel("WADA-SNR (dB, display)"); ax2.set_ylabel("C = -rog@L12")
ax2.set_title(f"C vs SNR   Spearman rho={rCS:+.3f} (p={pCS:.1e})")
plt.suptitle("E10: is T (vs C) noise-driven on ITW?", fontweight="bold")
plt.tight_layout(); plt.savefig(RESDIR / "e10_ct_vs_snr.png", dpi=150); plt.close()

plt.figure(figsize=(11, 4.3))
ax = plt.subplot(1, 2, 1)
ax.plot(quint_df["snr_db_med"], quint_df["cliff_T"], "o-", c="#d62728", label="T spoof-vs-bona")
ax.plot(quint_df["snr_db_med"], quint_df["cliff_C"], "s-", c="#1f77b4", label="C spoof-vs-bona")
ax.axhline(0, c="k", lw=.6); ax.set_xlabel("SNR quintile (median dB)")
ax.set_ylabel("Cliff's d (spoof vs bona)")
ax.set_title("Discriminativity vs SNR"); ax.legend(fontsize=8)
ax = plt.subplot(1, 2, 2)
ax.plot(quint_df["snr_db_med"], quint_df["rho_T_logit"], "o-", c="#d62728", label="rho(T,logit)")
ax.axhline(0, c="k", lw=.6); ax.set_xlabel("SNR quintile (median dB)")
ax.set_ylabel("rho(T, detector logit) | spoof")
ax.set_title("Does T predict where it's cleaner?"); ax.legend(fontsize=8)
plt.suptitle("E10: T signal across SNR quintiles", fontweight="bold")
plt.tight_layout(); plt.savefig(RESDIR / "e10_snr_quintile_trend.png", dpi=150); plt.close()

# ── summary ──────────────────────────────────────────────────────────────────────
L = ["# E10 (Test 1) — Is T's ITW failure a background-noise artifact?", "",
     f"Per-sample WADA-SNR added to E5's {len(df)} ITW clips (detector: {LOGIT_COL}). "
     "Observational test; no re-synthesis. SNR rank carrier = WADA v3; dB is display-scale.", "",
     f"## Verdict: **{verdict}**",
     f"- (1) T noise-driven (|rho(T,SNR)|>=0.20): **{cond_noise_driven}**  (rho={rTS:+.3f})",
     f"- (2) T recovers in clean half (strength {T_clean_strength:.3f} > noisy {T_noisy_strength:.3f}+0.03): **{cond_clean_recovers}**",
     f"- (3) C comparatively noise-robust (gap smaller than T's): **{cond_C_robust}**", "",
     "## (a) Is T just measuring noise?  Spearman vs WADA-SNR",
     "| factor | all samples | spoof only |", "|---|---|---|",
     f"| **T** | {rTS:+.3f} (p={pTS:.1e}) | {rTS_sp:+.3f} (p={pTS_sp:.1e}) |",
     f"| C (control) | {rCS:+.3f} (p={pCS:.1e}) | {rCS_sp:+.3f} (p={pCS_sp:.1e}) |", "",
     "## (b) Stratified discriminativity — clean (hi-SNR) vs noisy (lo-SNR) half",
     "| stratum | n_sp | med dB | rho(T,logit) | Cliff d T(sp>bo) | rho(C,logit) | Cliff d C(sp>bo) |",
     "|---|---|---|---|---|---|---|"]
for s in strat:
    L.append(f"| {s['stratum']} | {s['n_spoof']} | {s['snr_db_med']:.0f} | "
             f"{s['rho_T_logit']:+.3f} (p={s['p_T_logit']:.1e}) | {s['cliff_T_spoofVbona']:+.3f} (p={s['p_cliff_T']:.1e}) | "
             f"{s['rho_C_logit']:+.3f} (p={s['p_C_logit']:.1e}) | {s['cliff_C_spoofVbona']:+.3f} (p={s['p_cliff_C']:.1e}) |")
L += ["", "## (c) Partial correlation — does signal survive controlling for SNR?",
      "| relation | raw Spearman | partial | Spearman (control SNR) |", "|---|---|---|---|",
      f"| T -> logit (spoof) | {strat[0]['rho_T_logit']:+.3f} | {prTl:+.3f} (p={ppTl:.1e}) | |",
      f"| C -> logit (spoof) | {strat[0]['rho_C_logit']:+.3f} | {prCl:+.3f} (p={ppCl:.1e}) | |",
      f"| T -> label (all)   | {rrTy:+.3f} | {prTy:+.3f} (p={ppTy:.1e}) | |",
      f"| C -> label (all)   | {rrCy:+.3f} | {prCy:+.3f} (p={ppCy:.1e}) | |", "",
      "## (d) Per-quintile trend (see e10_snr_quintile_trend.csv / .png)",
      "| quintile | med dB | n_sp | rho(T,logit) | Cliff d T | Cliff d C | mean T |",
      "|---|---|---|---|---|---|---|"]
for _, r in quint_df.iterrows():
    L.append(f"| Q{int(r['quintile'])} | {r['snr_db_med']:.0f} | {int(r['n_spoof'])} | "
             f"{r['rho_T_logit']:+.3f} | {r['cliff_T']:+.3f} | {r['cliff_C']:+.3f} | {r['mean_T']:.3f} |")
L += ["", "## Reading",
      "- If T is noise-driven AND only discriminates in the clean half (while C holds), the "
      "hypothesis is supported: ITW's T-failure is a background-noise/artifact effect, not a "
      "statement that temporal dynamics are irrelevant to spoofing.",
      "- If T is dead in BOTH halves and barely tracks SNR, noise is NOT the explanation — "
      "T's ITW failure is intrinsic (domain shift / representation mismatch), echoing E9 where "
      "T was not an isolable causal lever.",
      "- This is the cheap observational test; the causal confirmation is Test 2 "
      "(inject controlled noise into clean MLAAD and watch T's signal collapse).", "",
      "## Files",
      "- `e10_itw_ct_snr.csv` — per-sample C/T/logits + WADA SNR",
      "- `e10_snr_quintile_trend.csv`, `e10_ct_vs_snr.png`, `e10_snr_quintile_trend.png`"]
(RESDIR / "e10_summary.md").write_text("\n".join(L))

print("\n".join(L[:40]))
print(f"\n[E10] done -> {RESDIR}")
