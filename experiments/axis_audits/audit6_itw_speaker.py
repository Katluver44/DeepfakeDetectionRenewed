#!/usr/bin/env python3
"""Audit A6 — ITW per-speaker hardness claims.

Claims under test (report §3.3):
  "Per-speaker hardness: s_orth rho = +0.534 (p = 0.003) under WavLM-GAT;
   vel_entropy ns."
Red-team angles:
  1. n: the report's corpus description says 58 speakers, but the analysis
     used only spoof speakers with >=10 utts. Recount.
  2. Multiplicity: 7 predictors were tested (i5_stats.speaker_level); apply
     Holm/BH within the family. Also rog12 (rho=-0.561, p=0.0015) was
     STRONGER than s_orth and went unreported — selective reporting check.
  3. Stability: bootstrap CI (speakers), jackknife leave-one-speaker.
  4. Confounds: n_utts per speaker, RMS; s_orth vs rog12 collinearity —
     are they the same signal? Partial correlations.
  5. Hardness reliability at n>=10 utts/speaker: split-half.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import audit_common as ac

OUT = ac.OUT_ROOT / "audit6_itw_speaker"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}

utt = pd.read_csv(ac.RES / "i5_itw_transfer" / "utt_table.csv")
spk = pd.read_csv(ac.RES / "i5_itw_transfer" / "speaker_table.csv")
n_spoof_speakers_all = utt[utt.label == 1].speaker.nunique()
res["counts"] = {"spoof_speakers_in_subset": int(n_spoof_speakers_all),
                 "speakers_analyzed_ge10": int(len(spk)),
                 "claimed_in_report": 58}
print(f"speakers: {n_spoof_speakers_all} spoof speakers in subset, "
      f"{len(spk)} analyzed (>=10 utts); report says '58 speakers'")

# ── 2. family-wise correction & the unreported rog12 ─────────────────────────
fam_feats = ["ve12", "ve9", "s_along", "s_orth", "rog12", "vmean12", "vmean0"]
rows = []
for c in fam_feats:
    rho, p = stats.spearmanr(spk[c], spk["hard"])
    rows.append({"feature": c, "rho": float(rho), "p": float(p)})
F = pd.DataFrame(rows)
order = np.argsort(F.p.values)
holm = np.empty(len(F))
hv = np.minimum.accumulate((F.p.values[order] * (len(F) - np.arange(len(F))))[::-1])[::-1]
holm[order] = np.clip(hv, 0, 1)
F["p_holm"] = holm
F["q_bh"] = ac.bh_fdr(F.p.values)
F.to_csv(OUT / "speaker_family_corrected.csv", index=False)
print(F.round(4).to_string(index=False))
res["family"] = F.to_dict("records")

# ── 3. bootstrap + jackknife for s_orth and rog12 ────────────────────────────
for c in ["s_orth", "rog12"]:
    ci = ac.bootstrap_spearman_ci(spk[c].values, spk["hard"].values, seed=1)
    jk = []
    for i in range(len(spk)):
        m = np.ones(len(spk), bool); m[i] = False
        jk.append(stats.spearmanr(spk[c].values[m], spk["hard"].values[m])[0])
    res[f"{c}_stability"] = {"boot_ci": ci, "jackknife_min": float(np.min(jk)),
                            "jackknife_max": float(np.max(jk))}
    print(f"{c}: boot CI [{ci[0]:+.3f},{ci[1]:+.3f}], jackknife range "
          f"[{np.min(jk):+.3f},{np.max(jk):+.3f}]")

# ── 4. confounds / collinearity ──────────────────────────────────────────────
def partial(x, yv, z):
    z = np.asarray(z, float).reshape(len(x), -1)
    rx = x - LinearRegression().fit(z, x).predict(z)
    ry = yv - LinearRegression().fit(z, yv).predict(z)
    return stats.spearmanr(rx, ry)

conf = {}
conf["s_orth_vs_rog12_r"] = float(stats.spearmanr(spk["s_orth"], spk["rog12"])[0])
conf["s_orth_given_rog12"] = list(map(float, partial(spk.s_orth.values, spk.hard.values,
                                                     spk.rog12.values)))
conf["rog12_given_s_orth"] = list(map(float, partial(spk.rog12.values, spk.hard.values,
                                                     spk.s_orth.values)))
conf["s_orth_given_n"] = list(map(float, partial(spk.s_orth.values, spk.hard.values,
                                                 spk.n.values)))
conf["s_orth_given_rms"] = list(map(float, partial(spk.s_orth.values, spk.hard.values,
                                                   spk.rms.values)))
res["confounds"] = conf
print("s_orth vs rog12 collinearity rho:", round(conf["s_orth_vs_rog12_r"], 3))
print("s_orth|rog12:", [round(v, 4) for v in conf["s_orth_given_rog12"]],
      " rog12|s_orth:", [round(v, 4) for v in conf["rog12_given_s_orth"]])
print("s_orth|n_utts:", [round(v, 4) for v in conf["s_orth_given_n"]],
      " s_orth|rms:", [round(v, 4) for v in conf["s_orth_given_rms"]])

# ── 5. hardness split-half reliability at speaker level ──────────────────────
bona_l = utt[utt.label == 0].logit.values
def hard_of(g_logits):
    yv = np.r_[np.zeros(len(bona_l)), np.ones(len(g_logits))]
    return 1 - roc_auc_score(yv, np.r_[bona_l, g_logits])

rels = []
for rep in range(50):
    rr = np.random.default_rng(rep)
    h1l, h2l = [], []
    for s in spk.speaker:
        g = utt[(utt.label == 1) & (utt.speaker == s)].logit.values
        gp = rr.permutation(g)
        a, b = gp[: len(gp) // 2], gp[len(gp) // 2:]
        h1l.append(hard_of(a)); h2l.append(hard_of(b))
    rels.append(stats.spearmanr(h1l, h2l)[0])
rel_sh = float(np.mean(rels))
rel_full = 2 * rel_sh / (1 + rel_sh)
res["hardness_reliability"] = {"split_half_rho": rel_sh, "spearman_brown_full": rel_full,
                               "max_explainable_rho": float(np.sqrt(rel_full))}
print(f"speaker-hardness split-half reliability: {rel_sh:.3f} "
      f"(SB full≈{rel_full:.3f}; predictor rho ceiling≈{np.sqrt(rel_full):.3f})")

(OUT / "audit6_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(spk.s_orth, spk.hard, label="s_orth", alpha=0.8)
axes[0].set_xlabel("s_orth (median, speaker)"); axes[0].set_ylabel("hardness")
axes[0].set_title("reported: s_orth ρ=+0.534")
axes[1].scatter(spk.rog12, spk.hard, color="tab:red", alpha=0.8)
axes[1].set_xlabel("rog12 (median, speaker)")
axes[1].set_title("UNREPORTED stronger predictor: rog12 ρ=-0.561")
fig.tight_layout()
fig.savefig(OUT / "audit6_itw.png", dpi=150)
print(f"done -> {OUT}")
