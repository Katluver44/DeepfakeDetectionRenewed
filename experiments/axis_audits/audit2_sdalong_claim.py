#!/usr/bin/env python3
"""Audit A2 — Is sd_along really a stable predictor of MLAAD hardness?

The paper's headline: "sd_along achieves leave-one-system-out R² = 0.273 on
MLAAD (p = 0.0005)". That number is HARDCODED in build_package.py (line 317)
and appears in no results JSON, so first we recompute it from the raw
embeddings + logits. Then we stress it:
  1. exact recomputation of sd_along-alone LOSO R² + permutation p
  2. Spearman rho per detector seed + bootstrap CI
  3. jackknife: leave-one-system-out rho / R² (single-system influence)
  4. confounds: n_utts (std estimation noise), waveform RMS, language,
     vel_entropy partials
  5. estimator sensitivity: ridge alpha, mean vs median pooling, Pearson vs
     Spearman, log-hardness
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import audit_common as ac

OUT = ac.OUT_ROOT / "audit2_sdalong_claim"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

d = ac.load_mlaad()
labels, systems, X, logits = d["labels"], d["systems"], d["X"], d["logits"]
sys_list = ac.sys_list_of(systems, labels)
print(f"n_systems={len(sys_list)}")

H = ac.hardness_table(labels, systems, logits, sys_list)
feat, s_along_utt, _ = ac.loso_axis_features(X, labels, systems, sys_list)
feat = feat.loc[H.index] if set(feat.index) == set(H.index) else feat
y = H.loc[feat.index, "shared"].values

# fidelity check vs the original i3 output
orig = pd.read_csv(ac.RES / "i3_position_geometry" / "system_position.csv", index_col=0)
common = [s for s in feat.index if s in orig.index]
fid = {c: float(np.corrcoef(feat.loc[common, c], orig.loc[common, c])[0, 1])
       for c in ["s_along", "sd_along", "s_orth"]}
fid["hard_shared_r"] = float(np.corrcoef(y, orig.loc[feat.index, "hard_shared"])[0, 1])
print("fidelity vs i3 originals:", fid)

res = {"n_systems": len(sys_list), "fidelity": fid}

# ── 1. the headline number ───────────────────────────────────────────────────
r2_sd = ac.loo_r2(feat["sd_along"].values, y)
p_sd = ac.perm_p_loo(feat["sd_along"].values, y, r2_sd, n_perm=4000, seed=1)
rho_sd, p_rho = stats.spearmanr(feat["sd_along"], y)
ci = ac.bootstrap_spearman_ci(feat["sd_along"].values, y, seed=2)
print(f"sd_along alone: LOSO R2={r2_sd:+.3f} (perm p={p_sd:.4f}) | "
      f"spearman rho={rho_sd:+.3f} (p={p_rho:.5f}) CI=[{ci[0]:+.3f},{ci[1]:+.3f}]")
res["headline"] = {"claimed_r2": 0.273, "claimed_p": 0.0005,
                   "recomputed_loso_r2": float(r2_sd), "perm_p": float(p_sd),
                   "spearman_rho": float(rho_sd), "spearman_p": float(p_rho),
                   "rho_boot_ci": ci}

# triplet model the report also cites (R2≈0.31)
ve = orig.loc[feat.index, "vel_entropy_L12"].values
r2_trip = ac.loo_r2(np.c_[feat["s_along"].values, ve, feat["sd_along"].values], y)
res["triplet_loso_r2"] = float(r2_trip)
print(f"triplet (s_along, vel_entropy, sd_along) LOSO R2={r2_trip:+.3f}")

# ── 2. per-seed stability ────────────────────────────────────────────────────
per_seed = {}
for sn in ["main", "s42", "s1024"]:
    ys = H.loc[feat.index, sn].values
    rho, p = stats.spearmanr(feat["sd_along"], ys)
    per_seed[sn] = {"rho": float(rho), "p": float(p),
                    "loso_r2": float(ac.loo_r2(feat["sd_along"].values, ys))}
    print(f"  seed {sn}: rho={rho:+.3f} (p={p:.4f}) R2={per_seed[sn]['loso_r2']:+.3f}")
res["per_seed"] = per_seed

# ── 3. jackknife influence ───────────────────────────────────────────────────
n = len(y)
jk = []
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    rho_i, p_i = stats.spearmanr(feat["sd_along"].values[m], y[m])
    r2_i = ac.loo_r2(feat["sd_along"].values[m], y[m])
    jk.append({"left_out": feat.index[i], "rho": rho_i, "p": p_i, "loso_r2": r2_i})
jk = pd.DataFrame(jk)
jk.to_csv(OUT / "jackknife_leave_one_system.csv", index=False)
res["jackknife"] = {"rho_min": float(jk.rho.min()), "rho_max": float(jk.rho.max()),
                    "r2_min": float(jk.loso_r2.min()), "r2_max": float(jk.loso_r2.max()),
                    "n_p_above_05": int((jk.p > 0.05).sum()),
                    "worst_system": jk.loc[jk.rho.idxmin(), "left_out"]}
print(f"jackknife rho range [{jk.rho.min():+.3f},{jk.rho.max():+.3f}], "
      f"R2 range [{jk.loso_r2.min():+.3f},{jk.loso_r2.max():+.3f}], "
      f"{int((jk.p>0.05).sum())}/{n} folds lose p<0.05")

# also: drop the k most influential systems simultaneously
infl = (jk.rho - stats.spearmanr(feat["sd_along"], y)[0]).abs().sort_values(ascending=False)
for k in [2, 5]:
    drop = jk.loc[infl.index[:k], "left_out"].tolist()
    m = ~feat.index.isin(drop)
    rho_k, p_k = stats.spearmanr(feat["sd_along"].values[m], y[m])
    res[f"drop_top{k}_influential"] = {"dropped": drop, "rho": float(rho_k), "p": float(p_k)}
    print(f"  drop {k} most influential: rho={rho_k:+.3f} (p={p_k:.4f})")

# ── 4. confounds ─────────────────────────────────────────────────────────────
# 4a. n_utts: sd estimated from few utterances is noisy; hardness from few
#     utterances is also noisy — shared-n artifact?
rho_n_sd = stats.spearmanr(feat["n_utts"], feat["sd_along"])
rho_n_h = stats.spearmanr(feat["n_utts"], y)
print(f"n_utts vs sd_along rho={rho_n_sd[0]:+.3f} (p={rho_n_sd[1]:.3f}); "
      f"n_utts vs hardness rho={rho_n_h[0]:+.3f} (p={rho_n_h[1]:.3f})")

def partial_spearman(x, yv, z):
    z = np.asarray(z, float).reshape(len(x), -1)
    rx = x - LinearRegression().fit(z, x).predict(z)
    ry = yv - LinearRegression().fit(z, yv).predict(z)
    return stats.spearmanr(rx, ry)

conf = {"n_utts_vs_sd": list(map(float, rho_n_sd)),
        "n_utts_vs_hard": list(map(float, rho_n_h))}
pr = partial_spearman(feat["sd_along"].values, y, feat["n_utts"].values)
conf["sd_given_nutts"] = list(map(float, pr))
print(f"sd_along | n_utts: rho={pr[0]:+.3f} (p={pr[1]:.4f})")

# 4b. RMS
rms = orig.loc[feat.index, "rms"].values
pr = partial_spearman(feat["sd_along"].values, y, rms)
conf["sd_given_rms"] = list(map(float, pr))
print(f"sd_along | rms: rho={pr[0]:+.3f} (p={pr[1]:.4f})")

# 4c. language (en vs non-en) — group structure
lang = np.array([s.split("|")[-1] for s in feat.index])
en = (lang == "en").astype(float)
pr = partial_spearman(feat["sd_along"].values, y, en)
conf["sd_given_english"] = list(map(float, pr))
# within-language analysis
within = {}
for lg_ in ["en", "de", "fr", "it", "es"]:
    m = lang == lg_
    if m.sum() >= 6:
        r, p = stats.spearmanr(feat["sd_along"].values[m], y[m])
        within[lg_] = {"n": int(m.sum()), "rho": float(r), "p": float(p)}
conf["within_language"] = within
print("within-language rho:", {k: round(v["rho"], 3) for k, v in within.items()})

# 4d. vel_entropy partial (is sd_along independent signal?)
pr1 = partial_spearman(feat["sd_along"].values, y, ve)
pr2 = partial_spearman(ve, y, feat["sd_along"].values)
conf["sd_given_ve"] = list(map(float, pr1))
conf["ve_given_sd"] = list(map(float, pr2))
print(f"sd_along|ve rho={pr1[0]:+.3f} (p={pr1[1]:.4f}); ve|sd rho={pr2[0]:+.3f} (p={pr2[1]:.4f})")
res["confounds"] = conf

# ── 5. estimator sensitivity ─────────────────────────────────────────────────
sens = {}
for alpha in [0.1, 1.0, 10.0]:
    sens[f"ridge_alpha_{alpha}"] = float(ac.loo_r2(feat["sd_along"].values, y, alpha=alpha))
# IQR instead of std for spread
iqr = []
for s in feat.index:
    m = systems == s
    iqr.append(np.subtract(*np.percentile(s_along_utt[m], [75, 25])))
rho_iqr, p_iqr = stats.spearmanr(iqr, y)
sens["iqr_spread_rho"] = [float(rho_iqr), float(p_iqr)]
# log hardness
rho_log, p_log = stats.spearmanr(feat["sd_along"], np.log(y + 1e-3))
sens["log_hardness_rho"] = [float(rho_log), float(p_log)]
res["sensitivity"] = sens
print("sensitivity:", {k: (round(v, 3) if isinstance(v, float) else [round(x, 4) for x in v])
                       for k, v in sens.items()})

feat.assign(hard_shared=y).to_csv(OUT / "system_features_recomputed.csv")
(OUT / "audit2_results.json").write_text(json.dumps(res, indent=2, default=str))

# figure: scatter + jackknife band
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(feat["sd_along"], y, s=22, alpha=0.75)
ax.set_xlabel("sd_along (recomputed, LOSO axis)")
ax.set_ylabel("hardness (1-AUC, 3-seed mean)")
ax.set_title(f"recomputed: LOSO R²={r2_sd:+.3f}, ρ={rho_sd:+.3f}\n"
             f"claimed: R²=0.273, p=0.0005")
ax = axes[1]
ax.hist(jk.rho, bins=20)
ax.axvline(rho_sd, color="k", ls="--", label="full-sample ρ")
ax.set_xlabel("jackknife ρ (leave one system out)")
ax.set_title("single-system influence on ρ")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "audit2_sdalong.png", dpi=150)
print(f"done -> {OUT}")
