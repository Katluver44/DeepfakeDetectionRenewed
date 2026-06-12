#!/usr/bin/env python3
"""Audit A9 — Is "hardness" itself a reliable measurement?

Hardness = 1-AUC of a system's utterances vs the SHARED bona pool. Two issues:
  1. shared bona pool -> hardness values across systems are statistically
     dependent (a few weird bona utterances shift every system's hardness);
  2. systems have as few as 8 utterances -> heavy measurement noise, which
     caps any predictor's attainable R².

Tests:
  1. Split-half reliability of MLAAD system hardness (within-system utterance
     halves) -> Spearman-Brown; ceiling for predictor rho/R².
  2. Bona-pool bootstrap: resample bona pool, recompute hardness ranks;
     how much do system ranks co-move? Does sd_along's rho survive bona
     resampling?
  3. min-utts threshold sensitivity: rho(sd_along, hardness) at
     min_utts in {8, 12, 16, 25}.
  4. seed-disagreement: cross-seed hardness correlation (already in i3 logs)
     recomputed as another reliability bound.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import audit_common as ac

OUT = ac.OUT_ROOT / "audit9_hardness_reliability"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}

d = ac.load_mlaad()
labels, systems, X, logits = d["labels"], d["systems"], d["X"], d["logits"]
sys_list = ac.sys_list_of(systems, labels)
feat, s_along_utt, _ = ac.loso_axis_features(X, labels, systems, sys_list)
H = ac.hardness_table(labels, systems, logits, sys_list)
y = H.loc[feat.index, "shared"].values
bona = labels == 0
bidx = np.where(bona)[0]
mean_logit = np.mean([logits[s] for s in logits], 0)

def hard_idx(lg, sysidx, bpool):
    yv = np.r_[np.zeros(len(bpool)), np.ones(len(sysidx))]
    return 1 - roc_auc_score(yv, np.r_[lg[bpool], lg[sysidx]])

# ── 1. split-half reliability ────────────────────────────────────────────────
rels = []
for rep in range(50):
    rr = np.random.default_rng(rep)
    h1l, h2l = [], []
    for s in feat.index:
        ix = rr.permutation(np.where(systems == s)[0])
        a, b = ix[: len(ix) // 2], ix[len(ix) // 2:]
        h1l.append(np.mean([hard_idx(lg, a, bidx) for lg in logits.values()]))
        h2l.append(np.mean([hard_idx(lg, b, bidx) for lg in logits.values()]))
    rels.append(stats.spearmanr(h1l, h2l)[0])
r_sh = float(np.mean(rels))
rel_full = 2 * r_sh / (1 + r_sh)
res["split_half"] = {"rho_half": r_sh, "spearman_brown_full": rel_full,
                     "rho_ceiling": float(np.sqrt(rel_full)),
                     "r2_ceiling": rel_full}
print(f"MLAAD hardness split-half: {r_sh:.3f} -> full-sample reliability≈{rel_full:.3f}; "
      f"R² ceiling≈{rel_full:.3f}; sd_along's 0.277 uses "
      f"{0.277/rel_full:.1%} of explainable variance")

# ── 2. bona-pool bootstrap ───────────────────────────────────────────────────
sys_idx = {s: np.where(systems == s)[0] for s in feat.index}
rhos = []
rank_sd = []
hmat = []
for b in range(1000):
    bb = rng.choice(bidx, len(bidx), replace=True)
    hv = np.array([np.mean([hard_idx(lg, sys_idx[s], bb) for lg in logits.values()])
                   for s in feat.index])
    hmat.append(hv)
    rhos.append(stats.spearmanr(feat["sd_along"].values, hv)[0])
hmat = np.array(hmat)
rank_stability = np.mean([stats.spearmanr(hmat[i], y)[0] for i in range(len(hmat))])
res["bona_bootstrap"] = {
    "sd_along_rho_median": float(np.median(rhos)),
    "sd_along_rho_ci": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
    "hardness_rank_stability": float(rank_stability)}
print(f"bona-pool bootstrap: sd_along rho median={np.median(rhos):+.3f} "
      f"CI=[{np.percentile(rhos,2.5):+.3f},{np.percentile(rhos,97.5):+.3f}]; "
      f"hardness rank stability={rank_stability:.3f}")

# ── 3. min-utts threshold sensitivity ────────────────────────────────────────
thr_rows = []
for thr in [8, 12, 16, 25]:
    sl = ac.sys_list_of(systems, labels, min_utts=thr)
    if len(sl) < 10:
        thr_rows.append({"min_utts": thr, "n_systems": len(sl), "rho": np.nan,
                         "p": np.nan, "loso_r2": np.nan})
        continue
    f2, _, _ = ac.loso_axis_features(X, labels, systems, sl)
    H2 = ac.hardness_table(labels, systems, logits, sl)
    y2 = H2.loc[f2.index, "shared"].values
    rho, p = stats.spearmanr(f2["sd_along"], y2)
    r2 = ac.loo_r2(f2["sd_along"].values, y2)
    thr_rows.append({"min_utts": thr, "n_systems": len(sl), "rho": float(rho),
                     "p": float(p), "loso_r2": float(r2)})
T = pd.DataFrame(thr_rows)
T.to_csv(OUT / "min_utts_sensitivity.csv", index=False)
print(T.round(4).to_string(index=False))
res["min_utts_sensitivity"] = T.to_dict("records")

# ── 4. cross-seed hardness agreement ─────────────────────────────────────────
cs = H[["main", "s42", "s1024"]].corr(method="spearman")
res["cross_seed_hardness"] = cs.round(4).to_dict()
print("cross-seed hardness agreement:")
print(cs.round(3).to_string())

(OUT / "audit9_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(rhos, bins=40)
axes[0].axvline(0.597, color="crimson", ls="--", label="point estimate")
axes[0].set_xlabel("rho(sd_along, hardness) under bona-pool bootstrap")
axes[0].legend(); axes[0].set_title("shared-bona-pool dependence")
axes[1].errorbar(T.min_utts, T.rho, marker="o")
ax2 = axes[1].twinx(); ax2.plot(T.min_utts, T.n_systems, "s--", color="gray", alpha=0.6)
ax2.set_ylabel("n systems", color="gray")
axes[1].set_xlabel("min utts per system"); axes[1].set_ylabel("rho(sd_along, hardness)")
axes[1].set_title("threshold sensitivity")
fig.tight_layout()
fig.savefig(OUT / "audit9_reliability.png", dpi=150)
print(f"done -> {OUT}")
