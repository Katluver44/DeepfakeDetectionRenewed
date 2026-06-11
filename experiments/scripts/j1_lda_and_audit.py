#!/usr/bin/env python3
"""
J1 — "Is this just LDA?" + leakage/error audit
===============================================
Part A (LDA question). The centroid-difference axis is the naive-Bayes / nearest-
centroid direction. Adversarial question: is the contribution just "ensemble with a
linear classifier"? We compare, under the SAME system-disjoint 5-fold protocol:
    w_mean   : bona-standardized centroid difference (ours)
    w_lda    : shrinkage-LDA direction (within-class whitened)
    w_logreg : full 768-dim L2 logistic probe (not 1-D)
    w_rand   : random directions (null)
on three criteria:
    (1) utterance-level AUC/EER (classifier quality),
    (2) per-system HARDNESS prediction (does position/spread along the direction
        predict which systems beat the detector? the geometry law),
    (3) fusion gain with the detector.
If w_mean ~ w_logreg on (2)-(3), the geometry is genuinely ~1-D and the law is not
"just a better classifier". We also test whether the law holds along the DETECTOR's
own score direction (ridge from embeddings to logit) — i.e. is hardness about THIS
axis or any discriminative direction?

Part B (audit).
    B1  split-half: features from half of each system's utterances, hardness from
        the other half (kills same-utterance finite-sample coupling).
    B2  per-fold EER for I7 fusion (kills cross-fold affine concatenation artifacts).
    B3  label-orientation + bootstrap sanity checks.
    B4  bona leakage check in I3: re-run system-level s_along with bona ALSO
        fold-disjoint for standardization.

Outputs -> experiments/results/j1_lda_audit/
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "j1_lda_audit"
OUT.mkdir(parents=True, exist_ok=True)
I3 = px.EXP_DIR / "results" / "i3_position_geometry"

recs = json.loads(px.TEST_JSON.read_text())
ok = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
recs = [recs[i] for i in ok]
labels = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
systems = np.array([f'{r["attack_system"]}|{r["language"]}' if labels[i] else "bona"
                    for i, r in enumerate(recs)])
X = np.load(I3 / "embeddings.npz")["X12"]
logit_seeds = {s: np.load(I3 / f"logits_{s}.npy") for s in ["main", "s42", "s1024"]}
mean_logit = np.mean([logit_seeds[s] for s in logit_seeds], 0)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

sys_units = sorted(set(systems) - {"bona"})
rng.shuffle(sys_units)
folds = np.array_split(np.array(sys_units), 5)
bona_idx = np.where(labels == 0)[0]
bona_folds = np.array_split(rng.permutation(bona_idx), 5)

def fold_masks(k):
    te_sys = set(folds[k]); te_b = set(bona_folds[k].tolist())
    te = np.array([(systems[i] in te_sys) or (i in te_b) for i in range(len(X))])
    return ~te, te

def run_scorer(make_score):
    """make_score(tr_mask) -> callable(X_subset)->scores. Returns oof scores."""
    s = np.full(len(X), np.nan)
    for k in range(5):
        tr, te = fold_masks(k)
        f = make_score(tr)
        s[te] = f(X[te])
    return s

def mk_wmean(tr):
    b = tr & (labels == 0); sp = tr & (labels == 1)
    mu = X[b].mean(0); sd = X[b].std(0) + 1e-9
    w = ((X[sp]-mu)/sd).mean(0) - ((X[b]-mu)/sd).mean(0); w /= np.linalg.norm(w)
    return lambda Q: ((Q - mu) / sd) @ w

def mk_lda(tr):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X[tr], labels[tr])
    return lambda Q: lda.decision_function(Q)

def mk_logreg(tr):
    b = tr & (labels == 0)
    mu = X[b].mean(0); sd = X[b].std(0) + 1e-9
    lr = LogisticRegression(C=0.1, max_iter=2000)
    lr.fit((X[tr]-mu)/sd, labels[tr])
    return lambda Q: lr.decision_function((Q-mu)/sd)

def mk_rand(tr):
    b = tr & (labels == 0)
    mu = X[b].mean(0); sd = X[b].std(0) + 1e-9
    w = rng.standard_normal(X.shape[1]); w /= np.linalg.norm(w)
    return lambda Q: ((Q - mu) / sd) @ w

def mk_detaxis(tr):
    """detector's own score direction: ridge embeddings -> mean logit."""
    b = tr & (labels == 0)
    mu = X[b].mean(0); sd = X[b].std(0) + 1e-9
    r = Ridge(alpha=100.0).fit((X[tr]-mu)/sd, mean_logit[tr])
    w = r.coef_ / np.linalg.norm(r.coef_)
    return lambda Q: ((Q - mu) / sd) @ w

SCORERS = {"w_mean": mk_wmean, "w_lda": mk_lda, "w_logreg": mk_logreg,
           "w_rand": mk_rand, "w_detaxis": mk_detaxis}
oof = {n: run_scorer(f) for n, f in SCORERS.items()}

# (1) classifier quality
print("[A1] out-of-fold classifier quality:")
a1 = {}
for n, s in oof.items():
    a1[n] = {"AUC": roc_auc_score(labels, s), "EER": eer_of(labels, s)}
    print(f"  {n:>10}: AUC={a1[n]['AUC']:.4f} EER={a1[n]['EER']:.4f}")

# (2) hardness law along each direction
big = [s for s in sys_units if (systems == s).sum() >= 8]
def hard_shared(s):
    return np.mean([1 - roc_auc_score(
        np.r_[np.zeros((labels == 0).sum()), np.ones((systems == s).sum())],
        np.r_[lg[labels == 0], lg[systems == s]]) for lg in logit_seeds.values()])
H = np.array([hard_shared(s) for s in big])
N = len(big)
def loo(Xf, y):
    Xf = np.asarray(Xf, float).reshape(N, -1)
    Xs = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    pred = np.empty(N)
    for i in range(N):
        m = np.ones(N, bool); m[i] = False
        pred[i] = Ridge(alpha=1.0).fit(Xs[m], y[m]).predict(Xs[i:i+1])[0]
    return 1 - ((y-pred)**2).sum() / ((y-y.mean())**2).sum()
print("\n[A2] hardness law (position+spread along direction -> shared hardness):")
a2 = {}
for n, s in oof.items():
    mean_pos = np.array([np.median(s[systems == u]) for u in big])
    sd_pos   = np.array([np.std(s[systems == u]) for u in big])
    rho, p = stats.spearmanr(mean_pos, H)
    r2 = loo(np.c_[mean_pos, sd_pos], H)
    a2[n] = {"rho_pos": float(rho), "p": float(p), "loo_pos_sd": float(r2)}
    print(f"  {n:>10}: rho(pos,H)={rho:+.3f} (p={p:.4f})  LOO(pos+sd)={r2:+.3f}")

# (3) fusion gain per scorer (lambda=1.5 fixed, z within train folds)
print("\n[A3] fusion gain with detector (3-seed mean dEER):")
a3 = {}
for n in SCORERS:
    dEs = []
    for sn, lg in logit_seeds.items():
        f = np.full(len(X), np.nan)
        for k in range(5):
            tr, te = fold_masks(k)
            sc = oof[n]
            zl = (lg[te] - lg[tr].mean()) / (lg[tr].std() + 1e-12)
            zp = (sc[te] - sc[tr].mean()) / (sc[tr].std() + 1e-12)
            f[te] = zl + 1.5 * zp
        dEs.append(eer_of(labels, f) - eer_of(labels, lg))
    a3[n] = float(np.mean(dEs))
    print(f"  {n:>10}: dEER={a3[n]:+.4f}")

# cosine between directions (global fit, for reference)
def gw(make):
    f = make(np.ones(len(X), bool))
    # recover direction by probing unit vectors is costly; instead refit explicit w
    return None
b_all = labels == 0
mu = X[b_all].mean(0); sd = X[b_all].std(0) + 1e-9
Zs = (X - mu) / sd
w_mean_g = Zs[labels == 1].mean(0) - Zs[b_all].mean(0); w_mean_g /= np.linalg.norm(w_mean_g)
lda_g = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, labels)
w_lda_g = lda_g.coef_[0] / np.linalg.norm(lda_g.coef_[0])
lr_g = LogisticRegression(C=0.1, max_iter=2000).fit(Zs, labels)
w_lr_g = lr_g.coef_[0] / np.linalg.norm(lr_g.coef_[0])
print(f"\n[A4] cos(w_mean, w_lda)={float(w_mean_g @ w_lda_g):+.3f}  "
      f"cos(w_mean, w_logreg)={float(w_mean_g @ w_lr_g):+.3f}")

# ─── Part B: audits ───────────────────────────────────────────────────────────
print("\n[B1] split-half: features from half utterances, hardness from other half")
sa = np.full(len(X), np.nan)
for s in big + ["bona"]:
    m = systems == s
    sp_excl = (labels == 1) & (systems != s)
    bm = labels == 0
    mu2 = X[bm].mean(0); sd2 = X[bm].std(0) + 1e-9
    w = ((X[sp_excl]-mu2)/sd2).mean(0) - ((X[bm]-mu2)/sd2).mean(0); w /= np.linalg.norm(w)
    sa[m] = ((X[m]-mu2)/sd2) @ w
rh = []
for rep in range(20):
    r2_ = np.random.default_rng(rep)
    pos_h, sd_h, hh = [], [], []
    for u in big:
        idx = np.where(systems == u)[0]
        idx = r2_.permutation(idx)
        fa, fb = idx[:len(idx)//2], idx[len(idx)//2:]
        if len(fa) < 3 or len(fb) < 3: continue
        pos_h.append(np.median(sa[fa])); sd_h.append(np.std(sa[fa]))
        h = np.mean([1 - roc_auc_score(
            np.r_[np.zeros((labels==0).sum()), np.ones(len(fb))],
            np.r_[lg[labels==0], lg[fb]]) for lg in logit_seeds.values()])
        hh.append(h)
    rho, p = stats.spearmanr(pos_h, hh)
    rh.append(rho)
print(f"  rho(pos_halfA, hard_halfB) over 20 splits: {np.mean(rh):+.3f} ± {np.std(rh):.3f}")

print("\n[B2] I7 per-fold EER (concatenation artifact check):")
i7 = np.load(px.EXP_DIR / "results" / "i7_axis_fusion" / "i7_scores.npz")
fl = i7["labels"]
for sn in ["main", "s42", "s1024"]:
    fu = i7[f"fused_{sn}"]; lg = i7[f"logit_{sn}"]
    per_fold = []
    for k in range(5):
        tr, te = fold_masks(k)
        per_fold.append(eer_of(fl[te], fu[te]) - eer_of(fl[te], lg[te]))
    print(f"  {sn}: per-fold dEER = {np.round(per_fold, 4)}  (mean {np.mean(per_fold):+.4f})")

print("\n[B3] label orientation: median logit bona vs spoof:",
      float(np.median(mean_logit[labels==0])), float(np.median(mean_logit[labels==1])))

print("\n[B4] bona-fold-disjoint standardization for the system-level law:")
sa2 = np.full(len(X), np.nan)
for k in range(5):
    tr, te = fold_masks(k)
    bm = tr & (labels == 0); sp = tr & (labels == 1)
    mu2 = X[bm].mean(0); sd2 = X[bm].std(0) + 1e-9
    w = ((X[sp]-mu2)/sd2).mean(0) - ((X[bm]-mu2)/sd2).mean(0); w /= np.linalg.norm(w)
    sa2[te] = ((X[te]-mu2)/sd2) @ w
pos2 = np.array([np.median(sa2[systems == u]) for u in big])
sd2_ = np.array([np.std(sa2[systems == u]) for u in big])
rho2, p2 = stats.spearmanr(pos2, H)
print(f"  rho(pos,H)={rho2:+.3f} (p={p2:.4f})  LOO(pos+sd)={loo(np.c_[pos2, sd2_], H):+.3f}")

(OUT / "j1_stats.json").write_text(json.dumps(
    {"A1_classifier": a1, "A2_hardness_law": a2, "A3_fusion_dEER": a3,
     "A4_cos": {"mean_lda": float(w_mean_g @ w_lda_g), "mean_logreg": float(w_mean_g @ w_lr_g)},
     "B1_splithalf_rho": [float(x) for x in rh],
     "B3_median_logits": [float(np.median(mean_logit[labels==0])),
                          float(np.median(mean_logit[labels==1]))],
     "B4_bona_disjoint": {"rho": float(rho2), "p": float(p2)}}, indent=2))
print(f"\n[J1] done -> {OUT}")
