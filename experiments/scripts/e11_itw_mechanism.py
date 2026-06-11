#!/usr/bin/env python3
"""
E11 — Mechanistic decomposition: WHY is In-The-Wild (ITW) harder than MLAAD?
===========================================================================

Adjudicates competing mechanisms for ITW difficulty rather than assuming one:
  M1 bona-fide domain shift  — real ITW recordings flagged as fake (false positives)
  M2 spoof realism / quality — incl. the user's "less-advanced TTS in some cases"
  M3 off-manifold rep. shift — whole ITW off the training distribution (E5)
  M4 recording/speaker confound — degraded real audio looks synthetic

Decisive diagnostic = error decomposition (FP vs FN). "Less-advanced TTS" predicts
crude spoofs are EASY => low FN; FP-dominated difficulty means the spoofs are not the
problem (bona-fide domain shift). ITW has NO TTS-source labels, so the spoof-quality
arm is probed via LATENT structure (clustering + logit bimodality).

All inputs already on disk (no GPU). MLAAD-trained detector (mlaad_robust_goat) is the
in-domain anchor; p1 raw_logit is verified to come from the SAME checkpoint as ITW's
logit_mlaad_robust_goat. Sign convention: higher logit = more spoof.

Outputs -> experiments/results/e11_itw_mechanism/
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RES     = BASE / "experiments" / "results" / "e11_itw_mechanism"
RES.mkdir(parents=True, exist_ok=True)
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from _ablation_common import compute_eer  # noqa: E402

ITW_CSV = BASE / "outputs" / "e5_itw_utterance_ct.csv"
SNR_CSV = BASE / "experiments" / "results" / "e10_t_noise" / "e10_itw_ct_snr.csv"
MLA_REF = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "utterance_features.csv"
EMB_NPZ = BASE / "outputs" / "e5_itw_embeddings.npz"
ML_DET  = "logit_mlaad_robust_goat"   # MLAAD-trained (in-domain anchor)
RG_DET  = "logit_robust_goat"         # ASVspoof-trained (cross-domain)

# ── helpers ──────────────────────────────────────────────────────────────────────
def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * U / (len(a) * len(b)) - 1.0), float(p)

def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean()-b.mean())/s) if s > 0 else float("nan")

def eer_threshold(labels, scores):
    """Threshold where FPR == FNR (the EER operating point)."""
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)

def youden_threshold(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    return float(thr[int(np.argmax(tpr - fpr))])

def decompose(labels, scores, thr):
    """At fixed thr (pred=spoof if score>=thr) return FP/FN counts, rates, error shares."""
    labels = np.asarray(labels); pred = (np.asarray(scores) >= thr).astype(int)
    sp = labels == 1; bo = labels == 0
    fp = int((pred[bo] == 1).sum()); tn = int((pred[bo] == 0).sum())
    fn = int((pred[sp] == 0).sum()); tp = int((pred[sp] == 1).sum())
    n_err = fp + fn
    return {"thr": thr, "n_spoof": int(sp.sum()), "n_bona": int(bo.sum()),
            "FP": fp, "FN": fn, "TP": tp, "TN": tn,
            "FPR_bona->spoof": fp / max(bo.sum(), 1),
            "FNR_spoof->bona": fn / max(sp.sum(), 1),
            "acc": (tp + tn) / len(labels),
            "FP_share_of_error": fp / max(n_err, 1),
            "FN_share_of_error": fn / max(n_err, 1)}

def bimodality_coeff(x):
    """Sarle's bimodality coefficient; >0.555 suggests bimodal/multimodal."""
    x = np.asarray(x); n = len(x)
    g = stats.skew(x); k = stats.kurtosis(x, fisher=True)
    return float((g**2 + 1) / (k + 3.0 * (n-1)**2 / ((n-2)*(n-3))))

def gmm_bic_ncomp(x, kmax=6):
    x = np.asarray(x).reshape(-1, 1)
    bics = []
    for k in range(1, kmax+1):
        g = GaussianMixture(k, random_state=SEED, n_init=3).fit(x)
        bics.append(g.bic(x))
    return int(np.argmin(bics) + 1), bics

# ═════════════════════════════════════════════════════════════════════════════════
print("[E11] loading existing artifacts (no GPU) ...", flush=True)
itw = pd.read_csv(ITW_CSV)
itw["sample_id"] = itw["sample_id"].astype(str)
snr = pd.read_csv(SNR_CSV)[["sample_id", "wada_snr_db"]]
snr["sample_id"] = snr["sample_id"].astype(str)
itw = itw.merge(snr, on="sample_id", how="left")
itw["y"] = (itw["label"] == "spoof").astype(int)
mla = pd.read_csv(MLA_REF)            # MLAAD in-domain ref (same detector), label 0/1, raw_logit
print(f"  ITW: {len(itw)} (spoof={itw.y.sum()} bona={(itw.y==0).sum()})  "
      f"MLAAD ref: {len(mla)} (spoof={(mla.label==1).sum()} bona={(mla.label==0).sum()})")

# embeddings aligned: emb[0:3000]=ITW-spoof, [3000:6000]=ITW-bona (csv-label order), [6000:]=MLAAD-spoof
z = np.load(EMB_NPZ, allow_pickle=True)
E12, E9 = z["emb_L12"], z["emb_L9"]
itw_sp = itw[itw.y == 1].reset_index(drop=True)   # aligns to E12[0:3000]
itw_bo = itw[itw.y == 0].reset_index(drop=True)    # aligns to E12[3000:6000]
emb_sp12, emb_bo12, emb_mla12 = E12[:3000], E12[3000:6000], E12[6000:]
emb_sp9,  emb_bo9              = E9[:3000],  E9[3000:6000]
assert len(itw_sp) == 3000 and len(itw_bo) == 3000

REP = ["# E11 — Why is In-The-Wild harder than MLAAD? (mechanistic decomposition)", ""]

# ═════════════════════════════════════════════════════════════════════════════════
# Part A — error decomposition (the adjudicator)
# ═════════════════════════════════════════════════════════════════════════════════
print("[A] error decomposition ...", flush=True)
# MLAAD detector: thresholds from the MLAAD in-domain reference, transferred to ITW
thr_eer_ml, mlaad_eer = eer_threshold(mla.label.values, mla.raw_logit.values)
thr_you_ml = youden_threshold(mla.label.values, mla.raw_logit.values)
A = {}
A["MLAAD_indomain@EERthr"]  = decompose(mla.label.values, mla.raw_logit.values, thr_eer_ml)
A["ITW@MLAAD_EERthr(transfer)"] = decompose(itw.y.values, itw[ML_DET].values, thr_eer_ml)
A["ITW@MLAAD_Youden(transfer)"] = decompose(itw.y.values, itw[ML_DET].values, thr_you_ml)
itw_ml_eer_thr, _ = eer_threshold(itw.y.values, itw[ML_DET].values)
A["ITW@own_EERthr(MLAADdet)"]   = decompose(itw.y.values, itw[ML_DET].values, itw_ml_eer_thr)
# robust_goat (ASVspoof-trained): no MLAAD anchor -> threshold-free + ITW-internal EER thr
itw_rg_eer_thr, _ = eer_threshold(itw.y.values, itw[RG_DET].values)
A["ITW@own_EERthr(robust_goat)"] = decompose(itw.y.values, itw[RG_DET].values, itw_rg_eer_thr)

itw_eer_ml = compute_eer(itw.y.values, itw[ML_DET].values)
itw_eer_rg = compute_eer(itw.y.values, itw[RG_DET].values)
itw_auc_ml = roc_auc_score(itw.y.values, itw[ML_DET].values)
itw_auc_rg = roc_auc_score(itw.y.values, itw[RG_DET].values)
mla_auc    = roc_auc_score(mla.label.values, mla.raw_logit.values)
print(f"  ITW EER mlaad={itw_eer_ml:.3f} robust_goat={itw_eer_rg:.3f} | MLAAD ref EER={mlaad_eer:.3f}")

REP += ["## Part A — Error decomposition (FP = real flagged fake; FN = spoof passed as real)",
        f"AUC: ITW(mlaad_det)={itw_auc_ml:.3f}  ITW(robust_goat)={itw_auc_rg:.3f}  MLAAD-indomain={mla_auc:.3f}",
        f"EER: ITW(mlaad_det)={itw_eer_ml:.3f}  ITW(robust_goat)={itw_eer_rg:.3f}  MLAAD-indomain={mlaad_eer:.3f}",
        "",
        "| scenario | thr | FPR (bona→spoof) | FNR (spoof→bona) | acc | FP share | FN share |",
        "|---|---|---|---|---|---|---|"]
for k, d in A.items():
    REP.append(f"| {k} | {d['thr']:.2f} | {d['FPR_bona->spoof']:.3f} | {d['FNR_spoof->bona']:.3f} | "
               f"{d['acc']:.3f} | {d['FP_share_of_error']:.2f} | {d['FN_share_of_error']:.2f} |")
pd.DataFrame(A).T.to_csv(RES / "e11_A_error_decomposition.csv")

# ═════════════════════════════════════════════════════════════════════════════════
# Part B — which class moved? (score space, MLAAD detector)
# ═════════════════════════════════════════════════════════════════════════════════
print("[B] which class moved ...", flush=True)
itw_bona_lg = itw_bo[ML_DET].values; itw_spoof_lg = itw_sp[ML_DET].values
mla_bona_lg = mla[mla.label == 0].raw_logit.values; mla_spoof_lg = mla[mla.label == 1].raw_logit.values
B = []
for name, a, b in [("bona: ITW vs MLAAD", itw_bona_lg, mla_bona_lg),
                    ("spoof: ITW vs MLAAD", itw_spoof_lg, mla_spoof_lg)]:
    dlt, p = cliff(a, b); d = cohen_d(a, b); ks, ksp = stats.ks_2samp(a, b)
    # displacement toward boundary: signed median shift relative to MLAAD EER thr
    B.append({"comparison": name, "itw_med": float(np.median(a)), "mlaad_med": float(np.median(b)),
              "median_shift": float(np.median(a)-np.median(b)), "cliff": dlt, "cohen_d": d,
              "ks": ks, "ks_p": ksp,
              "itw_frac_above_thr": float((a >= thr_eer_ml).mean()),
              "mlaad_frac_above_thr": float((b >= thr_eer_ml).mean())})
REP += ["", "## Part B — Which class moved? (logit space, MLAAD detector; thr={:.2f})".format(thr_eer_ml),
        "| comparison | ITW med | MLAAD med | shift | Cliff δ | Cohen d | KS | ITW>thr | MLAAD>thr |",
        "|---|---|---|---|---|---|---|---|---|"]
for r in B:
    REP.append(f"| {r['comparison']} | {r['itw_med']:.2f} | {r['mlaad_med']:.2f} | {r['median_shift']:+.2f} | "
               f"{r['cliff']:+.3f} | {r['cohen_d']:+.3f} | {r['ks']:.3f} | {r['itw_frac_above_thr']:.3f} | "
               f"{r['mlaad_frac_above_thr']:.3f} |")
pd.DataFrame(B).to_csv(RES / "e11_B_class_displacement.csv", index=False)
# note: ITW-bona frac above thr = its false-positive rate; vs MLAAD-bona = the bona shift magnitude
bona_shift = abs(B[0]["median_shift"]); spoof_shift = abs(B[1]["median_shift"])

# ═════════════════════════════════════════════════════════════════════════════════
# Part C — latent spoof heterogeneity ("less-advanced TTS in some cases")
# ═════════════════════════════════════════════════════════════════════════════════
print("[C] latent spoof tiers ...", flush=True)
Xsp = StandardScaler().fit_transform(np.hstack([emb_sp12, emb_sp9]))   # 3000 x 1536
# choose k by GMM-BIC over embeddings (cap small for interpretability)
emb_bic = []
for k in range(1, 7):
    g = GaussianMixture(k, random_state=SEED, n_init=2, covariance_type="diag").fit(Xsp)
    emb_bic.append((k, g.bic(Xsp)))
k_emb = min(emb_bic, key=lambda t: t[1])[0]
k_use = max(2, min(k_emb, 4))
km = KMeans(k_use, random_state=SEED, n_init=10).fit(Xsp)
itw_sp = itw_sp.copy(); itw_sp["cluster"] = km.labels_
bona_pool_ml = itw_bona_lg                      # shared negatives for within-EER
clstats = []
for c in range(k_use):
    m = itw_sp.cluster == c
    lg_ml = itw_sp.loc[m, ML_DET].values; lg_rg = itw_sp.loc[m, RG_DET].values
    y = np.r_[np.ones(m.sum()), np.zeros(len(bona_pool_ml))]
    s = np.r_[lg_ml, bona_pool_ml]
    eer = compute_eer(y, s)
    clstats.append({"cluster": c, "n": int(m.sum()),
                    "med_logit_mlaad": float(np.median(lg_ml)),
                    "med_logit_robust_goat": float(np.median(lg_rg)),
                    "within_EER_vs_bona": float(eer) if eer is not None else np.nan,
                    "mean_C": float(itw_sp.loc[m, "C"].mean()),
                    "mean_T": float(itw_sp.loc[m, "T"].mean()),
                    "mean_SNR_db": float(itw_sp.loc[m, "wada_snr_db"].mean())})
cl_df = pd.DataFrame(clstats).sort_values("within_EER_vs_bona")
cl_df.to_csv(RES / "e11_C_spoof_clusters.csv", index=False)

# bimodality of ITW-spoof detector logits
nmode_ml, bics_ml = gmm_bic_ncomp(itw_spoof_lg)
bc_ml = bimodality_coeff(itw_spoof_lg)
nmode_rg, _ = gmm_bic_ncomp(itw_sp[RG_DET].values)
bc_rg = bimodality_coeff(itw_sp[RG_DET].values)

# aggregate realism: within-dataset spoof-vs-bona separability
auc_itw_ml = roc_auc_score(itw.y.values, itw[ML_DET].values)
auc_itw_rg = roc_auc_score(itw.y.values, itw[RG_DET].values)
auc_mla    = roc_auc_score(mla.label.values, mla.raw_logit.values)
# is there a reliably EASY spoof tier? (cluster with low within-EER and high logit)
easy_cluster = cl_df.iloc[0]
hard_cluster = cl_df.iloc[-1]
crude_tier = (easy_cluster["within_EER_vs_bona"] < 0.15) and (easy_cluster["n"] >= 0.10 * 3000)

REP += ["", "## Part C — Latent spoof heterogeneity (no source labels ⇒ inferred tiers)",
        f"k-means k={k_use} on WavLM L12+L9 (GMM-BIC suggested {k_emb}). within-EER = spoof-cluster "
        "vs the ITW bona pool (lower = easier to detect).",
        "| cluster | n | med logit (mlaad) | med logit (rg) | within-EER | mean C | mean T | mean SNR |",
        "|---|---|---|---|---|---|---|---|"]
for _, r in cl_df.iterrows():
    REP.append(f"| {int(r.cluster)} | {int(r.n)} | {r.med_logit_mlaad:+.2f} | {r.med_logit_robust_goat:+.2f} | "
               f"{r.within_EER_vs_bona:.3f} | {r.mean_C:.2f} | {r.mean_T:.3f} | {r.mean_SNR_db:.0f} |")
REP += ["",
        f"- ITW-spoof logit modality: mlaad GMM-BIC n_modes={nmode_ml} (bimod.coeff={bc_ml:.3f}); "
        f"robust_goat n_modes={nmode_rg} (bc={bc_rg:.3f}).  (>0.555 ⇒ multimodal)",
        f"- spoof-vs-bona separability (AUC): ITW(mlaad)={auc_itw_ml:.3f} ITW(robust_goat)={auc_itw_rg:.3f} "
        f"vs MLAAD-indomain={auc_mla:.3f}.  Lower ITW AUC ⇒ ITW spoofs are MORE bona-like (more realistic).",
        f"- crude/easy tier present? **{crude_tier}** "
        f"(easiest cluster within-EER={easy_cluster['within_EER_vs_bona']:.3f}, n={int(easy_cluster['n'])})."]

# ═════════════════════════════════════════════════════════════════════════════════
# Part D — off-manifold / generalization gap (vs MLAAD reference cloud)
# ═════════════════════════════════════════════════════════════════════════════════
print("[D] off-manifold gap ...", flush=True)
sc = StandardScaler().fit(np.vstack([emb_mla12, emb_sp12, emb_bo12]))
Z_mla = sc.transform(emb_mla12); Z_sp = sc.transform(emb_sp12); Z_bo = sc.transform(emb_bo12)
nn = NearestNeighbors(n_neighbors=10).fit(Z_mla)
d_sp = nn.kneighbors(Z_sp)[0].mean(1); d_bo = nn.kneighbors(Z_bo)[0].mean(1)
d_mla_self = nn.kneighbors(Z_mla)[0][:, 1:].mean(1)   # exclude self
# does distance-to-reference predict per-sample difficulty (lower logit = harder for spoof)?
rho_sp = stats.spearmanr(d_sp, itw_sp[ML_DET].values).correlation
REP += ["", "## Part D — Off-manifold gap (kNN dist to MLAAD-spoof reference cloud, WavLM-L12)",
        f"- mean kNN dist: MLAAD-spoof(self)={d_mla_self.mean():.2f}  ITW-spoof={d_sp.mean():.2f}  "
        f"ITW-bona={d_bo.mean():.2f}  ⇒ both ITW classes sit "
        f"{'OFF' if min(d_sp.mean(), d_bo.mean()) > 1.3*d_mla_self.mean() else 'near'} the reference manifold.",
        f"- ρ(dist-to-ref, spoof logit) = {rho_sp:+.3f} (more off-manifold ⇒ "
        f"{'harder' if rho_sp<0 else 'not clearly harder'})."]

# ═════════════════════════════════════════════════════════════════════════════════
# Part E — recording-quality confound on bona-fide (mechanism for M1)
# ═════════════════════════════════════════════════════════════════════════════════
print("[E] recording-quality confound ...", flush=True)
bo = itw_bo.copy()
bo["is_FP"] = (bo[ML_DET].values >= thr_eer_ml).astype(int)   # real flagged as fake
q = pd.qcut(bo["wada_snr_db"], 5, labels=False, duplicates="drop")
bo["snr_q"] = q
fp_by_snr = bo.groupby("snr_q").agg(med_snr=("wada_snr_db", "median"),
                                    FP_rate=("is_FP", "mean"), n=("is_FP", "size")).reset_index()
rho_bona_snr = stats.spearmanr(bo["wada_snr_db"], bo[ML_DET]).correlation   # logit vs SNR
# speaker concentration of difficulty (FP rate per speaker)
spk = bo.groupby("speaker").agg(FP_rate=("is_FP", "mean"), n=("is_FP", "size"))
spk = spk[spk.n >= 20].sort_values("FP_rate", ascending=False)
fp_by_snr.to_csv(RES / "e11_E_bona_fp_by_snr.csv", index=False)
REP += ["", "## Part E — Recording-quality confound on bona-fide (FP = real flagged fake)",
        f"- ρ(ITW-bona logit, SNR) = {rho_bona_snr:+.3f} (negative ⇒ noisier real audio scored MORE spoof-like).",
        "| SNR quintile | med dB | FP rate | n |", "|---|---|---|---|"]
for _, r in fp_by_snr.iterrows():
    REP.append(f"| Q{int(r.snr_q)+1} | {r.med_snr:.0f} | {r.FP_rate:.3f} | {int(r.n)} |")
REP += [f"- bona FP-rate concentrated in {int((spk.FP_rate>0.5).sum())}/{len(spk)} speakers (>50% FP); "
        f"top speaker FP={spk.FP_rate.iloc[0]:.2f} ({spk.index[0]})."]

# ═════════════════════════════════════════════════════════════════════════════════
# Verdicts
# ═════════════════════════════════════════════════════════════════════════════════
fp_share = A["ITW@MLAAD_EERthr(transfer)"]["FP_share_of_error"]
fn_share = A["ITW@MLAAD_EERthr(transfer)"]["FN_share_of_error"]
M1 = "SUPPORTED" if fp_share >= 0.6 else ("PARTIAL" if fp_share >= 0.45 else "weak")
# spoof realism: ITW spoofs more bona-like than MLAAD spoofs (lower AUC) AND no easy crude tier
M2_realistic = (auc_itw_ml < auc_mla - 0.02)
user_hyp = "SUPPORTED" if crude_tier else ("NOT supported" if M2_realistic else "inconclusive")
M3 = "SUPPORTED" if min(d_sp.mean(), d_bo.mean()) > 1.3*d_mla_self.mean() else "weak"
M4 = "SUPPORTED" if (rho_bona_snr < -0.1) else "weak"

REP = REP[:2] + [
    "## Verdict (mechanisms adjudicated)",
    f"- **M1 bona-fide domain shift: {M1}** — FP share of ITW error = {fp_share:.2f} "
    f"(FN share {fn_share:.2f}); bona logit shift vs MLAAD = {bona_shift:+.2f} (Cliff {B[0]['cliff']:+.3f}).",
    f"- **M2 spoof realism: {'ITW spoofs MORE bona-like' if M2_realistic else 'comparable'}** — "
    f"spoof-vs-bona AUC ITW={auc_itw_ml:.3f} vs MLAAD={auc_mla:.3f}; spoof logit shift={spoof_shift:+.2f}.",
    f"- **User's 'less-advanced TTS in some cases': {user_hyp}** — latent crude/easy tier present="
    f"{crude_tier}; ITW-spoof logits multimodal={'yes' if nmode_ml>1 else 'no'}.",
    f"- **M3 off-manifold shift: {M3}** — ITW kNN dist {min(d_sp.mean(),d_bo.mean()):.2f} vs "
    f"MLAAD-self {d_mla_self.mean():.2f}.",
    f"- **M4 recording-quality (bona): {M4}** — ρ(bona logit, SNR)={rho_bona_snr:+.3f}.",
    "",
    "**Bottom line:** "
    + ("ITW difficulty is dominated by FALSE POSITIVES on real audio (bona-fide domain shift), "
       "not by spoof quality — the 'less-advanced TTS' framing targets the wrong class."
       if fp_share >= 0.55 else
       "ITW difficulty has a substantial false-negative (spoof) component; see Part C for whether "
       "a crude TTS tier exists."),
    ""] + REP[2:]

# ═════════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════════
print("[fig] ...", flush=True)
# Fig 1: 4-cloud logit distributions (MLAAD detector) + transferred threshold
plt.figure(figsize=(8, 5))
bins = np.linspace(min(itw[ML_DET].min(), mla.raw_logit.min()),
                   max(itw[ML_DET].max(), mla.raw_logit.max()), 60)
for v, c, lab in [(mla_bona_lg, "#2ca02c", "MLAAD bona"), (mla_spoof_lg, "#7f7f7f", "MLAAD spoof"),
                  (itw_bona_lg, "#1f77b4", "ITW bona"), (itw_spoof_lg, "#d62728", "ITW spoof")]:
    plt.hist(v, bins=bins, density=True, histtype="step", lw=2, color=c, label=lab)
plt.axvline(thr_eer_ml, c="k", ls="--", label=f"transferred thr={thr_eer_ml:.1f}")
plt.xlabel("detector logit (higher = more spoof)"); plt.ylabel("density")
plt.title("E11-A: 4-cloud logits — ITW bona shifts INTO the spoof zone?")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(RES / "e11_logit_clouds.png", dpi=150); plt.close()

# Fig 2: FP/FN decomposition
plt.figure(figsize=(7, 4.2))
scen = ["MLAAD_indomain@EERthr", "ITW@MLAAD_EERthr(transfer)", "ITW@own_EERthr(robust_goat)"]
xs = np.arange(len(scen))
fps = [A[s]["FPR_bona->spoof"] for s in scen]; fns = [A[s]["FNR_spoof->bona"] for s in scen]
plt.bar(xs-0.2, fps, 0.4, label="FPR (bona→spoof)", color="#1f77b4")
plt.bar(xs+0.2, fns, 0.4, label="FNR (spoof→bona)", color="#d62728")
plt.xticks(xs, ["MLAAD\nin-domain", "ITW\n(MLAAD det)", "ITW\n(robust_goat)"], fontsize=8)
plt.ylabel("rate"); plt.title("E11-A: false-positive vs false-negative balance")
plt.legend(); plt.tight_layout(); plt.savefig(RES / "e11_fp_fn.png", dpi=150); plt.close()

# Fig 3: spoof clusters in PCA space + per-cluster within-EER
P = PCA(2, random_state=SEED).fit_transform(Xsp)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for c in range(k_use):
    m = km.labels_ == c
    ax[0].scatter(P[m, 0], P[m, 1], s=6, alpha=.4, label=f"c{c} (EER {cl_df.set_index('cluster').loc[c,'within_EER_vs_bona']:.2f})")
ax[0].set_title("ITW-spoof clusters (WavLM L12+L9 PCA)"); ax[0].legend(fontsize=7, markerscale=2)
ax[1].bar(cl_df["cluster"].astype(str), cl_df["within_EER_vs_bona"], color="#9467bd")
ax[1].axhline(itw_eer_ml, c="k", ls=":", label=f"overall ITW EER {itw_eer_ml:.2f}")
ax[1].set_xlabel("cluster"); ax[1].set_ylabel("within-EER vs bona pool")
ax[1].set_title("Is any spoof tier reliably EASY?"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(RES / "e11_spoof_clusters.png", dpi=150); plt.close()

# Fig 4: bona FP-rate vs SNR
plt.figure(figsize=(6.5, 4.2))
plt.plot(fp_by_snr.med_snr, fp_by_snr.FP_rate, "o-", color="#1f77b4")
plt.xlabel("ITW bona SNR quintile (median dB)"); plt.ylabel("false-positive rate (real flagged fake)")
plt.title(f"E11-E: noisier real audio → more false positives  (ρ_logit,SNR={rho_bona_snr:+.2f})")
plt.tight_layout(); plt.savefig(RES / "e11_bona_fp_vs_snr.png", dpi=150); plt.close()

(RES / "e11_summary.md").write_text("\n".join(REP))
print("\n".join(REP[:40]))
print(f"\n[E11] done -> {RES}")
