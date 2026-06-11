#!/usr/bin/env python3
"""
I7 — Decision-boundary geometry made actionable: frozen-axis score fusion
==========================================================================
I3 showed per-system hardness is governed by position/spread along a linear
natural<->synthetic axis in FROZEN pretrained WavLM space. If true, the frozen
axis projection carries decision evidence the fine-tuned detector under-uses,
and score fusion  s = z(logit) + lam * z(s_along)  should yield large EER gains.

Honesty protocol (system-disjoint, no leakage):
  * 5-fold split over spoof systems; bona split into the same folds.
  * For each fold: axis w = centroid(spoof_trainfold) - centroid(bona_trainfold),
    standardization stats and z-scaling fit on train fold; lam selected on train
    fold; fold-test utterances scored with that axis/lam.
  * Repeated with 3 detector seeds; paired utterance bootstrap on dEER/dAUC.
  * Per-system breakdown by baseline-hardness quartile (where do gains come from?).

Also reports the axis ALONE (is the linear frozen probe already better than the
fine-tuned GAT detector?).

Outputs -> experiments/results/i7_axis_fusion/
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

OUT = px.EXP_DIR / "results" / "i7_axis_fusion"
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

from sklearn.metrics import roc_auc_score, roc_curve
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

# ─── 5-fold system-disjoint axis projection ───────────────────────────────────
sys_units = sorted(set(systems) - {"bona"})
rng.shuffle(sys_units)
folds = np.array_split(np.array(sys_units), 5)
bona_idx = np.where(labels == 0)[0]
bona_folds = np.array_split(rng.permutation(bona_idx), 5)

s_along = np.full(len(X), np.nan)
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
fused = {sn: np.full(len(X), np.nan) for sn in logit_seeds}
lam_pick = {sn: [] for sn in logit_seeds}

for k in range(5):
    te_sys = set(folds[k]); te_bona = set(bona_folds[k].tolist())
    te_mask = np.array([(systems[i] in te_sys) or (i in te_bona) for i in range(len(X))])
    tr_mask = ~te_mask
    trb = tr_mask & (labels == 0); trs = tr_mask & (labels == 1)
    mu = X[trb].mean(0); sd = X[trb].std(0) + 1e-9
    Ztr_s = (X[trs] - mu) / sd; Ztr_b = (X[trb] - mu) / sd
    w = Ztr_s.mean(0) - Ztr_b.mean(0); w /= np.linalg.norm(w)
    proj_tr = ((X[tr_mask] - mu) / sd) @ w
    proj_te = ((X[te_mask] - mu) / sd) @ w
    s_along[te_mask] = proj_te
    # z-stats on train fold
    pm, ps = proj_tr.mean(), proj_tr.std() + 1e-12
    for sn, lg in logit_seeds.items():
        lm, ls = lg[tr_mask].mean(), lg[tr_mask].std() + 1e-12
        ztr_l = (lg[tr_mask] - lm) / ls; ztr_p = (proj_tr - pm) / ps
        ytr = labels[tr_mask]
        best = max(LAMS, key=lambda L: roc_auc_score(ytr, ztr_l + L * ztr_p))
        lam_pick[sn].append(best)
        fused[sn][te_mask] = (lg[te_mask] - lm) / ls + best * (proj_te - pm) / ps

print("[I7] lambda picked per fold:", {k: v for k, v in lam_pick.items()})

# ─── headline metrics ─────────────────────────────────────────────────────────
rows = []
for sn, lg in logit_seeds.items():
    rows.append({"seed": sn, "scorer": "detector", "EER": eer_of(labels, lg),
                 "AUC": roc_auc_score(labels, lg)})
    rows.append({"seed": sn, "scorer": "fused", "EER": eer_of(labels, fused[sn]),
                 "AUC": roc_auc_score(labels, fused[sn])})
rows.append({"seed": "-", "scorer": "axis_alone", "EER": eer_of(labels, s_along),
             "AUC": roc_auc_score(labels, s_along)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "i7_headline.csv", index=False)
print(df.to_string(index=False))

# paired utterance bootstrap on across-seed mean dEER / dAUC
N = len(labels); B = 2000
bidx = np.random.default_rng(SEED).integers(0, N, size=(B, N))
dE = np.empty(B); dA = np.empty(B)
for b in range(B):
    ix = bidx[b]; yb = labels[ix]
    if yb.min() == yb.max(): dE[b] = dA[b] = np.nan; continue
    dE[b] = np.mean([eer_of(yb, fused[sn][ix]) - eer_of(yb, logit_seeds[sn][ix])
                     for sn in logit_seeds])
    dA[b] = np.mean([roc_auc_score(yb, fused[sn][ix]) - roc_auc_score(yb, logit_seeds[sn][ix])
                     for sn in logit_seeds])
dE = dE[np.isfinite(dE)]; dA = dA[np.isfinite(dA)]
point_E = float(np.mean([eer_of(labels, fused[sn]) - eer_of(labels, logit_seeds[sn])
                         for sn in logit_seeds]))
point_A = float(np.mean([roc_auc_score(labels, fused[sn]) - roc_auc_score(labels, logit_seeds[sn])
                         for sn in logit_seeds]))
pE = float(min(1, 2 * min((dE <= 0).mean(), (dE >= 0).mean())))
pA = float(min(1, 2 * min((dA <= 0).mean(), (dA >= 0).mean())))
print(f"\n[boot] dEER={point_E:+.4f} [{np.percentile(dE,2.5):+.4f},{np.percentile(dE,97.5):+.4f}] p={pE:.4f}")
print(f"       dAUC={point_A:+.4f} [{np.percentile(dA,2.5):+.4f},{np.percentile(dA,97.5):+.4f}] p={pA:.4f}")

# ─── per-system gains by baseline-hardness quartile ───────────────────────────
def sys_auc(sc, s):
    bl = sc[labels == 0]; sl = sc[systems == s]
    return roc_auc_score(np.r_[np.zeros(len(bl)), np.ones(len(sl))], np.r_[bl, sl])
big = [s for s in sys_units if (systems == s).sum() >= 8]
baseh = {s: np.mean([1 - sys_auc(logit_seeds[sn], s) for sn in logit_seeds]) for s in big}
q = pd.qcut(pd.Series(baseh), 4, labels=["Q1_easy", "Q2", "Q3", "Q4_hard"])
qrows = []
for qq in ["Q1_easy", "Q2", "Q3", "Q4_hard"]:
    ss = [s for s in big if q[s] == qq]
    bA = float(np.mean([[sys_auc(logit_seeds[sn], s) for s in ss] for sn in logit_seeds]))
    fA = float(np.mean([[sys_auc(fused[sn], s) for s in ss] for sn in logit_seeds]))
    qrows.append({"quartile": qq, "AUC_detector": bA, "AUC_fused": fA, "delta": fA - bA})
qdf = pd.DataFrame(qrows)
qdf.to_csv(OUT / "i7_quartiles.csv", index=False)
print("\n" + qdf.round(4).to_string(index=False))

np.savez_compressed(OUT / "i7_scores.npz", labels=labels, s_along=s_along,
                    **{f"fused_{sn}": fused[sn] for sn in fused},
                    **{f"logit_{sn}": logit_seeds[sn] for sn in logit_seeds})
(OUT / "i7_stats.json").write_text(json.dumps(
    {"dEER": point_E, "dEER_ci": [float(np.percentile(dE, 2.5)), float(np.percentile(dE, 97.5))],
     "p_EER": pE, "dAUC": point_A,
     "dAUC_ci": [float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))], "p_AUC": pA,
     "lambdas": {k: list(map(float, v)) for k, v in lam_pick.items()}}, indent=2))
print(f"\n[I7] done -> {OUT}")
