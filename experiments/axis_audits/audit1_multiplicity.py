#!/usr/bin/env python3
"""Audit A1 — Global multiplicity / p-hacking audit.

The I/J series tested MANY predictor-hardness correlations across corpora and
detectors, but every reported p-value is uncorrected and the paper's headline
table (table2_hardness_law.md) presents only winners. We:
  1. Enumerate every recorded predictor->hardness test in the artifacts
     (i2 univariate battery, i3, i4, i5, j4, j5, j6 stats JSONs).
  2. Apply BH-FDR globally and per-family; report which headline claims
     survive q<0.05.
  3. Check the i2 battery's own FDR column: does ANY battery feature pass?
  4. Selection-inference flag: sd_along was selected on the same MLAAD
     systems/detectors used to declare it the winner (no internal replication
     split). We run a split-half "discovery/confirmation" simulation: pick the
     best of {all candidate features} on a random half of systems, measure its
     rho on the other half — distribution of confirmed effect sizes.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import audit_common as ac

OUT = ac.OUT_ROOT / "audit1_multiplicity"
OUT.mkdir(parents=True, exist_ok=True)

tests = []
def add(family, predictor, dataset, detector, rho, p, headline=False):
    tests.append(dict(family=family, predictor=predictor, dataset=dataset,
                      detector=detector, rho=rho, p=p, headline=headline))

# i2 univariate battery (30 features)
uni = pd.read_csv(ac.RES / "i2_geometry_battery" / "univariate.csv")
for r in uni.itertuples():
    add("i2_battery", r.feature, "MLAAD", "wavlm_gat", r.spearman, r.p_spearman)

# i3 (s_along, s_orth across 4 hardness variants)
i3 = json.loads((ac.RES / "i3_position_geometry" / "i3_stats.json").read_text())
for h, v in i3["per_seed"].items():
    add("i3_position", "s_along", "MLAAD", f"wavlm_gat_{h}", v["rho_along"], v["p_along"])
    add("i3_position", "s_orth", "MLAAD", f"wavlm_gat_{h}", v["rho_orth"], v["p_orth"])

# sd_along (A2 recomputation; the headline)
a2 = json.loads((ac.OUT_ROOT / "audit2_sdalong_claim" / "audit2_results.json").read_text())
add("i3_position", "sd_along", "MLAAD", "wavlm_gat",
    a2["headline"]["spearman_rho"], a2["headline"]["spearman_p"], headline=True)

# i4 ASVspoof(2019-era) per-attack
i4 = json.loads((ac.RES / "i4_asvspoof_position" / "i4_stats.json").read_text())
for f, v in i4["per_attack"].items():
    add("i4_asvspoof", f, "ASVspoof_i4", "wavlm_gat", v["rho"], v["p"],
        headline=(f == "s_along"))

# i5 ITW speaker level
i5 = json.loads((ac.RES / "i5_itw_transfer" / "i5_stats.json").read_text())
for f, v in i5["speaker_level"].items():
    add("i5_itw", f, "ITW", "wavlm_gat", v["rho"], v["p"], headline=(f == "s_orth"))

# j4/j5 prospective (use exact permutation p from A3)
a3 = json.loads((ac.OUT_ROOT / "audit3_asvspoof_prospective" / "audit3_results.json").read_text())
for r in a3["family"]:
    add("j4j5_prospective", r["pred"], "ASVspoof21", r["detector"],
        r["rho"], r["p_perm"], headline=(r["pred"] == "P3"))

# j2 earlier prospective attempts
j2 = json.loads((ac.RES / "j2_prospective" / "j2_results.json").read_text())
for k, v in j2.items():
    if isinstance(v, dict) and "rho" in v and k != "detector_family_agreement":
        add("j2_prospective", k, "J2_targets", "wavlm_gat", v["rho"], v["p"])

# j5 H2 (AASIST-ZS on MLAAD), j5 H5 (AASIST on ITW), j6 law (AASIST-FT)
j5 = json.loads((ac.RES / "j5_aasist" / "j5_results.json").read_text())
for f, v in j5["H2"].items():
    add("j5_aasist_zs", f, "MLAAD", "aasist_zs", v["rho"], v["p"])
for f, v in j5["H5"].items():
    add("j5_aasist_itw", f, "ITW", "aasist_zs", v["rho"], v["p"])
j6 = json.loads((ac.RES / "j6_aasist_mlaad" / "j6_results.json").read_text())
for f, v in j6["law"].items():
    if f == "loo_pos_sd":
        continue
    add("j6_aasist_ft", f, "MLAAD", "aasist_ft", v["rho"], v["p"],
        headline=(f in ("sd_along", "vel_entropy_L12")))

T = pd.DataFrame(tests)
T["q_bh_global"] = ac.bh_fdr(T.p.values)
T["q_bh_family"] = np.nan
for fam_name, grp in T.groupby("family"):
    T.loc[grp.index, "q_bh_family"] = ac.bh_fdr(grp.p.values)
T["sig_unc"] = T.p < 0.05
T["sig_q05_global"] = T.q_bh_global < 0.05
T = T.sort_values("p")
T.to_csv(OUT / "master_test_table.csv", index=False)

print(f"total recorded tests: {len(T)}; uncorrected p<0.05: {int(T.sig_unc.sum())}; "
      f"global BH q<0.05: {int(T.sig_q05_global.sum())}")
print("\nheadline claims after global BH:")
print(T[T.headline][["family", "predictor", "dataset", "detector", "rho", "p",
                     "q_bh_global", "q_bh_family"]].round(4).to_string(index=False))
print("\ni2 battery features passing their own FDR<0.05:",
      int((uni.q_fdr < 0.05).sum()))

# ── 4. split-half discovery/confirmation simulation ─────────────────────────
sysdf = pd.read_csv(ac.RES / "i2_geometry_battery" / "system_table.csv")
i3sys = pd.read_csv(ac.RES / "i3_position_geometry" / "system_position.csv", index_col=0)
feat_cols = [c for c in sysdf.columns
             if c not in ("system", "language", "n_utt", "hardness", "median_logit")
             and sysdf[c].dtype != object]
M = sysdf.set_index("system")
M = M.join(i3sys[["s_along", "sd_along", "s_orth"]], how="inner")
ycol = "hard" if "hard" in M.columns else "hard_shared"
if ycol not in M.columns:
    M = M.join(i3sys[["hard_shared"]])
    ycol = "hard_shared"
cands = [c for c in feat_cols + ["s_along", "sd_along", "s_orth"] if c in M.columns]
rng = np.random.default_rng(0)
disc = []
n = len(M)
for rep in range(2000):
    idx = rng.permutation(n)
    h1, h2 = idx[: n // 2], idx[n // 2:]
    best, best_r = None, 0
    for c in cands:
        r = abs(stats.spearmanr(M[c].values[h1], M[ycol].values[h1])[0])
        if r > best_r:
            best, best_r = c, r
    r2_, p2_ = stats.spearmanr(M[best].values[h2], M[ycol].values[h2])
    disc.append({"winner": best, "rho_discovery": best_r, "rho_confirm": abs(r2_),
                 "p_confirm": p2_})
disc = pd.DataFrame(disc)
disc.to_csv(OUT / "splithalf_discovery_confirmation.csv", index=False)
win_rate = disc.winner.value_counts(normalize=True)
print("\nsplit-half winner frequency (top5):")
print(win_rate.head().round(3).to_string())
print(f"sd_along wins {win_rate.get('sd_along', 0):.1%} of discovery halves; "
      f"median confirmation rho when it wins = "
      f"{disc[disc.winner=='sd_along'].rho_confirm.median():.3f}; "
      f"confirmation p<0.05 rate = "
      f"{(disc[disc.winner=='sd_along'].p_confirm < 0.05).mean():.1%}")

summary = {
    "n_tests_recorded": int(len(T)),
    "n_sig_uncorrected": int(T.sig_unc.sum()),
    "n_sig_bh_global": int(T.sig_q05_global.sum()),
    "headline_after_global_bh": T[T.headline][["predictor", "dataset", "detector",
                                               "p", "q_bh_global"]].to_dict("records"),
    "i2_battery_fdr_survivors": int((uni.q_fdr < 0.05).sum()),
    "splithalf": {"winner_freq": win_rate.to_dict(),
                  "sd_along_confirm_median_rho": float(
                      disc[disc.winner == "sd_along"].rho_confirm.median()),
                  "sd_along_confirm_p05_rate": float(
                      (disc[disc.winner == "sd_along"].p_confirm < 0.05).mean())}}
(OUT / "audit1_results.json").write_text(json.dumps(summary, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].scatter(np.arange(len(T)), T.p.values, s=14, c=np.where(T.headline, "crimson", "grey"))
axes[0].axhline(0.05, color="r", ls="--", lw=0.8)
axes[0].set_yscale("log"); axes[0].set_xlabel("test rank"); axes[0].set_ylabel("p (uncorrected)")
axes[0].set_title(f"all {len(T)} recorded tests (red = paper headline)")
axes[1].hist(disc.rho_confirm, bins=30)
axes[1].axvline(0.597, color="crimson", ls="--", label="sd_along full-sample ρ")
axes[1].set_xlabel("confirmation-half |ρ| of discovery winner")
axes[1].legend(); axes[1].set_title("winner's-curse check")
fig.tight_layout()
fig.savefig(OUT / "audit1_multiplicity.png", dpi=150)
print(f"done -> {OUT}")
