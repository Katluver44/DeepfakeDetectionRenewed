#!/usr/bin/env python3
"""Audit A3 — ASVspoof2021 prospective prediction (J4/J5): does P3 survive?

Claims under test:
  "Position (s_along) along a dataset-internal LDA axis achieves rho=+0.599
   (p=0.031) in a prospective pre-registered prediction on ASVspoof 2021,
   replicating for AASIST (rho=+0.643, p=0.018)."

Red-team angles:
  1. n=13 attacks: exact permutation p (scipy's asymptotic p is unreliable).
  2. Forking paths: 4 predictors were registered, P1 was PRIMARY and failed;
     P3 is a secondary promoted to headline. Holm/Bonferroni across the family.
  3. Sequential testing: J2 (earlier prospective on WaveFake/VCC2020/MLAAD-
     unseen) failed before J4 was designed -> family is larger than 4.
  4. Replication independence: if WavLM and AASIST per-attack hardness are
     themselves correlated, the AASIST "replication" is not independent
     evidence. Quantify.
  5. AASIST hardness floor: EER=0.073, hardness values ~0.000-0.049 with only
     40 utts/attack; bootstrap attack-level hardness to get rank stability,
     propagate into rho CI.
  6. Leave-one-attack-out jackknife for both detectors.
  7. Top-3 hit rates: hypergeometric baseline.
  8. Pre-registration audit: timestamp gap between registration and scoring.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import audit_common as ac

OUT = ac.OUT_ROOT / "audit3_asvspoof_prospective"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

J4 = pd.read_csv(ac.RES / "j4_asvspoof21" / "j4_table.csv").set_index("attack")
J5 = pd.read_csv(ac.RES / "j5_aasist" / "asvspoof21_aasist.csv").set_index("attack")
res = {}

# ── 1+2. exact permutation p + family-wise corrections ──────────────────────
preds = ["P1", "P2", "P3", "P4"]
fam = []
for det, col, df in [("wavlm_gat", "hard_actual", J4), ("aasist_zs", "hard_aasist", J5)]:
    for p_ in preds:
        rho, pperm = ac.spearman_perm_p(df[p_].values, df[col].values,
                                        n_perm=200000, seed=7)
        fam.append({"detector": det, "pred": p_, "rho": float(rho), "p_perm": float(pperm)})
fam = pd.DataFrame(fam)
# Holm within each detector's 4-predictor family, and across all 8
for scope, grp in [("within_detector", None), ("all8", "all")]:
    if scope == "within_detector":
        out = []
        for det in fam.detector.unique():
            sub = fam[fam.detector == det].copy()
            order = np.argsort(sub.p_perm.values)
            holm = np.minimum.accumulate((sub.p_perm.values[order] *
                                          (len(sub) - np.arange(len(sub))))[::-1])[::-1]
            hv = np.empty(len(sub)); hv[order] = np.clip(holm, 0, 1)
            sub["p_holm_within"] = hv
            out.append(sub)
        fam = pd.concat(out)
    else:
        order = np.argsort(fam.p_perm.values)
        holm = np.minimum.accumulate((fam.p_perm.values[order] *
                                      (len(fam) - np.arange(len(fam))))[::-1])[::-1]
        hv = np.empty(len(fam)); hv[order] = np.clip(holm, 0, 1)
        fam["p_holm_all8"] = hv
fam["q_bh_all8"] = ac.bh_fdr(fam.p_perm.values)
fam.to_csv(OUT / "family_corrected_pvalues.csv", index=False)
print(fam.round(4).to_string(index=False))
res["family"] = fam.to_dict("records")

# ── 3. sequential-testing context (J2) ───────────────────────────────────────
j2 = json.loads((ac.RES / "j2_prospective" / "j2_results.json").read_text())
prior_tests = {k: v for k, v in j2.items() if isinstance(v, dict) and "rho" in v}
res["j2_prior_attempts"] = prior_tests
n_prior = len(prior_tests)
print(f"\nJ2 prior prospective tests (all ns): {n_prior}")
# effective family if J2 counted: 8 + prior
p3w = float(fam[(fam.detector == 'wavlm_gat') & (fam.pred == 'P3')].p_perm.iloc[0])
res["p3_bonferroni_with_j2"] = min(1.0, p3w * (8 + n_prior))

# ── 4. replication independence ──────────────────────────────────────────────
rho_dd, p_dd = stats.spearmanr(J4["hard_actual"], J5["hard_aasist"])
res["wavlm_vs_aasist_hardness"] = {"rho": float(rho_dd), "p": float(p_dd)}
print(f"\nWavLM vs AASIST per-attack hardness agreement: rho={rho_dd:+.3f} (p={p_dd:.3f})")

# ── 5. AASIST hardness floor: bootstrap rank stability ───────────────────────
z = np.load(ac.RES / "j4_asvspoof21" / "waves_sel.npz", allow_pickle=True)
atts = z["atts"]
sc = np.load(ac.RES / "j5_aasist" / "aasist_scores.npz")["s21"]
labels21 = (atts != "bonafide").astype(int)
# orient so spoof scores higher (the convention hardness=1-AUC assumes)
if np.median(sc[labels21 == 1]) < np.median(sc[labels21 == 0]):
    sc = -sc
bona = labels21 == 0
attacks = [f"A{i:02d}" for i in range(7, 20)]
from sklearn.metrics import roc_auc_score

def hard_vec(score, bidx, aidx_dict):
    out = {}
    for a, ai in aidx_dict.items():
        yv = np.r_[np.zeros(len(bidx)), np.ones(len(ai))]
        sv = np.r_[score[bidx], score[ai]]
        out[a] = 1 - roc_auc_score(yv, sv)
    return out

bidx = np.where(bona)[0]
aidx = {a: np.where(atts == a)[0] for a in attacks}
h0 = hard_vec(sc, bidx, aidx)
assert abs(h0["A10"] - J5.loc["A10", "hard_aasist"]) < 1e-9, "fidelity check failed"

B = 3000
rhos_b = []
wav_logits = np.load(ac.RES / "j4_asvspoof21" / "logits.npz")
wl_mean = np.mean([wav_logits[k] for k in wav_logits.files], 0)
rhos_w = []
for b in range(B):
    bb = rng.choice(bidx, len(bidx), replace=True)
    aa = {a: rng.choice(ai, len(ai), replace=True) for a, ai in aidx.items()}
    hb = hard_vec(sc, bb, aa)
    rhos_b.append(stats.spearmanr(J5["P3"].values, [hb[a] for a in attacks])[0])
    hw = hard_vec(wl_mean, bb, aa)
    rhos_w.append(stats.spearmanr(J4["P3"].values, [hw[a] for a in attacks])[0])
res["p3_rho_bootstrap"] = {
    "aasist": {"median": float(np.median(rhos_b)),
               "ci": [float(np.percentile(rhos_b, 2.5)), float(np.percentile(rhos_b, 97.5))],
               "frac_below_0": float(np.mean(np.array(rhos_b) <= 0))},
    "wavlm": {"median": float(np.median(rhos_w)),
              "ci": [float(np.percentile(rhos_w, 2.5)), float(np.percentile(rhos_w, 97.5))],
              "frac_below_0": float(np.mean(np.array(rhos_w) <= 0))}}
print("\nP3 rho under measurement bootstrap (bona+attack resampling):")
print(json.dumps(res["p3_rho_bootstrap"], indent=2))

# ── 6. leave-one-attack-out jackknife ────────────────────────────────────────
jk = []
for det, col, df in [("wavlm_gat", "hard_actual", J4), ("aasist_zs", "hard_aasist", J5)]:
    for a in attacks:
        sub = df.drop(a)
        r, p = stats.spearmanr(sub["P3"], sub[col])
        jk.append({"detector": det, "left_out": a, "rho": float(r), "p": float(p)})
jk = pd.DataFrame(jk)
jk.to_csv(OUT / "loao_jackknife.csv", index=False)
for det in ["wavlm_gat", "aasist_zs"]:
    s = jk[jk.detector == det]
    res[f"loao_{det}"] = {"rho_min": float(s.rho.min()), "rho_max": float(s.rho.max()),
                          "n_p_above_05": int((s.p > 0.05).sum())}
    print(f"LOAO {det}: rho in [{s.rho.min():+.3f},{s.rho.max():+.3f}]; "
          f"{int((s.p>0.05).sum())}/13 folds have p>0.05")

# ── 7. top-3 hits baseline ───────────────────────────────────────────────────
from scipy.stats import hypergeom
# P(>=2 hits of 3 picks among 13, 3 truly-hard) under random picking
p_ge2 = hypergeom.sf(1, 13, 3, 3)
res["top3_hits"] = {"p_ge2_random": float(p_ge2),
                    "note": "P1,P2,P3 all predicted the same top-3 set {A17,A18,A19}"}
print(f"\nP(>=2/3 top-3 hits by chance) = {p_ge2:.4f}")

# ── 8. prereg timeline ───────────────────────────────────────────────────────
import os, time
reg = json.loads((ac.RES / "j4_asvspoof21" / "j4_preregistered_predictions.json").read_text())
t_reg = reg["timestamp"]
t_logits = time.strftime("%Y-%m-%d %H:%M:%S",
                         time.localtime(os.path.getmtime(ac.RES / "j4_asvspoof21" / "logits.npz")))
res["prereg"] = {"registered": t_reg, "logits_written": t_logits,
                 "gap_seconds": os.path.getmtime(ac.RES / "j4_asvspoof21" / "logits.npz")
                                - time.mktime(time.strptime(t_reg, "%Y-%m-%d %H:%M:%S")),
                 "note": "registration and scoring in the same script run; "
                         "predictions cannot use detector scores by construction, "
                         "but there is no external commitment device"}
print(f"prereg gap: {res['prereg']['gap_seconds']:.0f}s")

(OUT / "audit3_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].bar(np.arange(8), fam.p_perm, color=["tab:blue"]*4 + ["tab:orange"]*4)
axes[0].axhline(0.05, color="r", ls="--")
axes[0].set_xticks(np.arange(8))
axes[0].set_xticklabels([f"{r.detector[:3]}\n{r.pred}" for r in fam.itertuples()], fontsize=8)
axes[0].set_ylabel("permutation p")
axes[0].set_title("8-test family (blue=WavLM, orange=AASIST)")
axes[1].hist(rhos_w, bins=40, alpha=0.6, label="WavLM")
axes[1].hist(rhos_b, bins=40, alpha=0.6, label="AASIST")
axes[1].axvline(0, color="k")
axes[1].set_xlabel("P3 rho under measurement bootstrap")
axes[1].legend(); axes[1].set_title("hardness measurement noise -> rho")
for det, marker in [("wavlm_gat", "o"), ("aasist_zs", "s")]:
    s = jk[jk.detector == det]
    axes[2].scatter(range(13), s.rho, marker=marker, label=det)
axes[2].axhline(0.05, lw=0); axes[2].axhline(0, color="k", lw=0.5)
axes[2].set_xticks(range(13)); axes[2].set_xticklabels(attacks, rotation=45, fontsize=7)
axes[2].set_ylabel("P3 rho (attack left out)"); axes[2].legend()
axes[2].set_title("leave-one-attack-out")
fig.tight_layout()
fig.savefig(OUT / "audit3_asvspoof.png", dpi=150)
print(f"done -> {OUT}")
