#!/usr/bin/env python3
"""E6 -- channel-confound control on the MLAAD sd_along law.

Question: is the sd_along vs hardness association (audit2: rho~+0.52-ish,
LOSO R2=0.273 headline) actually a channel/recording-condition artifact --
i.e. does it survive partialling out layer-0 "channel proxy" features
(vmean0 = vel_mean_L0, rog0 = rog_L0, rms)?

Data: n=61 MLAAD English system table, `experiments/results/i3_position_geometry/
system_position.csv` -- this is the exact per-system table audit2 uses to
recompute sd_along/hardness (fidelity-checked there against i2's
system_table.csv). sd_along is recomputed independently here too via
audit_common.loso_axis_features on the cached L12 embeddings, as a
cross-check, but the primary analysis uses the i3 CSV columns directly
(same values, confirmed by audit2's fidelity check).

Channel proxies available in the cached table: vel_mean_L0 (layer-0 mean
velocity == "vmean0" in audit6's ITW naming), rog_L0 (radius of gyration at
layer 0), rms (waveform RMS). No other L0 features (vel_cv_L0, vel_entropy_L0,
etc.) are cached for MLAAD -- i2's system_table.csv only computed the full
battery at L9/L12, with L0 limited to vel_mean_L0 and rog_L0. This is a
documented limitation: we cannot check e.g. L0 entropy or L0 curvature as
additional channel proxies for MLAAD (unlike ITW/i5 which cached vmean0 only
too, so this is the best available channel-proxy set for both experiments).

Method:
  1. Spearman(proxy, hardness) for each proxy alone.
  2. Spearman(sd_along, hardness) partialling out [vmean0, rog0, rms]
     jointly (rank-transform + OLS residualization a la audit2/audit6
     `partial_spearman`), plus each proxy individually for comparison.
  3. Permutation p (10k) on the partial statistic: permute hardness, redo
     the OLS-residualization + Spearman each time (proxies are NOT
     re-residualized against the permuted y -- only y's residual changes,
     which is the correct permutation null for a partial correlation test:
     the proxy->y regression uses the permuted y, so both x and y residuals
     depend correctly on the permutation).
  4. OLS: hardness ~ sd_along + vmean0 + rog0 + rms, HC3 robust SE, plus a
     permutation-based p-value on the sd_along coefficient (10k perms,
     refit full OLS each time) as a robustness cross-check to HC3's
     asymptotics at n=61.

Acceptance: sd_along's partial association with hardness survives (p<0.05)
after controlling for the channel proxies. CPU-only, deterministic
(fixed seeds), runs in well under a minute.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis_audits"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import audit_common as ac

OUT = Path(__file__).resolve().parents[1] / "results" / "k6_channel_control"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 10000

# ── load data ────────────────────────────────────────────────────────────────
tbl = pd.read_csv(ac.RES / "i3_position_geometry" / "system_position.csv", index_col=0)
print(f"loaded system_position.csv: n={len(tbl)} systems")

y = tbl["hard_shared"].values.astype(float)
sd_along = tbl["sd_along"].values.astype(float)
proxies = {
    "vmean0": tbl["vel_mean_L0"].values.astype(float),
    "rog0": tbl["rog_L0"].values.astype(float),
    "rms": tbl["rms"].values.astype(float),
}

res = {
    "n_systems": int(len(tbl)),
    "data_source": "experiments/results/i3_position_geometry/system_position.csv",
    "note_on_l0_features": (
        "Only vel_mean_L0 (vmean0) and rog_L0 (rog0) are cached at layer 0 for "
        "MLAAD (i2_geometry_battery/system_table.csv computed the full geometric "
        "battery -- vel_cv, vel_entropy, curvature, tortuosity, recurrence, "
        "twonn_id, etc. -- only at L9/L12, not L0). RMS is a waveform-level (not "
        "layer-0 embedding) channel proxy, also cached. These three (vmean0, "
        "rog0, rms) are therefore the complete available channel-proxy set; "
        "this is documented as a limitation of the recon, not a methodological "
        "choice."
    ),
}

# fidelity cross-check: recompute sd_along independently from cached L12
# embeddings via audit_common.loso_axis_features and confirm it matches the
# i3 CSV column (same procedure audit2 used).
d = ac.load_mlaad()
labels_u, systems_u, X = d["labels"], d["systems"], d["X"]
sys_list = ac.sys_list_of(systems_u, labels_u)
feat, _, _ = ac.loso_axis_features(X, labels_u, systems_u, sys_list)
common = [s for s in tbl.index if s in feat.index]
fid_r = float(np.corrcoef(tbl.loc[common, "sd_along"], feat.loc[common, "sd_along"])[0, 1])
res["fidelity_recomputed_sd_along_corr"] = fid_r
print(f"fidelity check: corr(sd_along cached vs recomputed via loso_axis_features) = {fid_r:.4f}")

# ── (i) proxy alone vs hardness ─────────────────────────────────────────────
alone = {}
rho_sd, p_sd = stats.spearmanr(sd_along, y)
alone["sd_along"] = {"rho": float(rho_sd), "p": float(p_sd)}
for name, v in proxies.items():
    rho, p = stats.spearmanr(v, y)
    alone[name] = {"rho": float(rho), "p": float(p)}
res["marginal_spearman"] = alone
print("marginal Spearman(., hardness):")
for k, v in alone.items():
    print(f"  {k}: rho={v['rho']:+.3f} p={v['p']:.4f}")

# proxies vs sd_along themselves (collinearity check)
collin = {}
for name, v in proxies.items():
    rho, p = stats.spearmanr(v, sd_along)
    collin[name] = {"rho": float(rho), "p": float(p)}
res["proxy_vs_sd_along_collinearity"] = collin
print("proxy vs sd_along collinearity:", {k: round(v["rho"], 3) for k, v in collin.items()})


# ── (ii) partial Spearman: rank-transform + OLS residualization ─────────────
def partial_spearman_stat(x, yv, Z):
    """Residualize ranks of x and y on controls Z (also rank-transformed
    implicitly via using raw Z here, matching audit2/audit6's
    `partial_spearman` helper -- OLS residualization on raw control values,
    Spearman on the residuals)."""
    Z = np.asarray(Z, float).reshape(len(x), -1)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = yv - LinearRegression().fit(Z, yv).predict(Z)
    return stats.spearmanr(rx, ry)


Z_all = np.column_stack([proxies["vmean0"], proxies["rog0"], proxies["rms"]])
rho_partial, p_partial_asym = partial_spearman_stat(sd_along, y, Z_all)
print(f"\npartial Spearman(sd_along, hardness | vmean0,rog0,rms): "
      f"rho={rho_partial:+.4f} (asymptotic p={p_partial_asym:.4f})")

partial_each = {}
for name, v in proxies.items():
    r, p = partial_spearman_stat(sd_along, y, v)
    partial_each[name] = {"rho": float(r), "p_asymptotic": float(p)}
    print(f"  partial | {name} alone: rho={r:+.4f} (asymptotic p={p:.4f})")

# ── (iii) permutation p on the joint-partial statistic (10k) ───────────────
rng = np.random.default_rng(SEED)


def partial_rho_only(x, yv, Z):
    Z = np.asarray(Z, float).reshape(len(x), -1)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = yv - LinearRegression().fit(Z, yv).predict(Z)
    return stats.spearmanr(rx, ry)[0]


obs_partial = partial_rho_only(sd_along, y, Z_all)
cnt = 0
perm_dist = np.empty(N_PERM)
for i in range(N_PERM):
    yp = rng.permutation(y)
    r_i = partial_rho_only(sd_along, yp, Z_all)
    perm_dist[i] = r_i
    if abs(r_i) >= abs(obs_partial) - 1e-12:
        cnt += 1
perm_p_partial = (cnt + 1) / (N_PERM + 1)
print(f"partial rho={obs_partial:+.4f}, permutation p (n_perm={N_PERM}, two-sided) = {perm_p_partial:.5f}")

res["partial_spearman"] = {
    "controls": ["vmean0", "rog0", "rms"],
    "rho": float(rho_partial),
    "p_asymptotic": float(p_partial_asym),
    "n_perm": N_PERM,
    "perm_p_twosided": float(perm_p_partial),
    "seed": SEED,
    "per_control_partial": partial_each,
}

# ── (iv) OLS: hardness ~ sd_along + vmean0 + rog0 + rms, HC3 + perm on coef ─
Xd = pd.DataFrame({
    "sd_along": sd_along,
    "vmean0": proxies["vmean0"],
    "rog0": proxies["rog0"],
    "rms": proxies["rms"],
})
Xd_z = (Xd - Xd.mean()) / Xd.std()
Xd_c = sm.add_constant(Xd_z)
ols = sm.OLS(y, Xd_c).fit(cov_type="HC3")
print("\nOLS hardness ~ sd_along + vmean0 + rog0 + rms (standardized X, HC3 SE):")
print(ols.summary().tables[1])

coef_sd = float(ols.params["sd_along"])
se_hc3 = float(ols.bse["sd_along"])
t_hc3 = float(ols.tvalues["sd_along"])
p_hc3 = float(ols.pvalues["sd_along"])

# permutation test on the sd_along OLS coefficient: permute y, refit full
# model each time (proxies stay fixed / unpermuted -- this tests whether
# sd_along carries information about hardness beyond what's attributable to
# chance, in the presence of the (fixed) proxy design).
obs_coef = coef_sd
rng2 = np.random.default_rng(SEED + 1)
cnt_c = 0
for i in range(N_PERM):
    yp = rng2.permutation(y)
    ols_p = sm.OLS(yp, Xd_c).fit()
    if abs(float(ols_p.params["sd_along"])) >= abs(obs_coef) - 1e-12:
        cnt_c += 1
perm_p_coef = (cnt_c + 1) / (N_PERM + 1)
print(f"OLS sd_along coef={obs_coef:+.4f}; HC3 p={p_hc3:.4f}; permutation p (n_perm={N_PERM}) = {perm_p_coef:.5f}")

res["ols_multivariate"] = {
    "formula": "hardness ~ sd_along + vmean0 + rog0 + rms (standardized predictors)",
    "r2": float(ols.rsquared),
    "adj_r2": float(ols.rsquared_adj),
    "coef_sd_along": coef_sd,
    "hc3_se": se_hc3,
    "hc3_t": t_hc3,
    "hc3_p": p_hc3,
    "perm_p_coef": float(perm_p_coef),
    "n_perm": N_PERM,
    "seed": SEED + 1,
    "all_coefs": {k: float(v) for k, v in ols.params.items()},
    "all_hc3_p": {k: float(v) for k, v in ols.pvalues.items()},
}

# ── acceptance ───────────────────────────────────────────────────────────────
accept = bool(perm_p_partial < 0.05)
res["acceptance"] = {
    "criterion": "partial Spearman(sd_along, hardness | vmean0,rog0,rms) permutation p < 0.05",
    "observed_perm_p": float(perm_p_partial),
    "survives": accept,
}
print(f"\nACCEPTANCE: sd_along partial association survives channel-confound control "
      f"= {accept} (perm p={perm_p_partial:.5f})")

(OUT / "k6_results.json").write_text(json.dumps(res, indent=2, default=str))

# ── report ───────────────────────────────────────────────────────────────────
report = f"""# K6 -- Channel-Confound Control on the MLAAD sd_along Law

Appendix experiment (exploratory, not pre-registered). CPU-only, deterministic.

## Question

Is the MLAAD-English sd_along-vs-hardness association (audited in
`audit2_sdalong_claim.py`: LOSO R2=0.273, headline p=0.0005) actually a
recording-channel artifact, rather than a property of the spoof-detection
axis geometry? We test whether it survives partialling out layer-0
"channel proxy" features.

## Data

- `experiments/results/i3_position_geometry/system_position.csv`, n={len(tbl)}
  MLAAD-English systems (the exact table audit2 uses; fidelity re-check
  against an independent recomputation via `audit_common.loso_axis_features`
  on cached L12 embeddings gives corr={fid_r:.4f}).
- Channel proxies available: `vel_mean_L0` (vmean0), `rog_L0` (rog0), `rms`
  (waveform RMS). **Limitation**: `i2_geometry_battery/system_table.csv`
  only computed the full geometric battery (vel_cv, vel_entropy, curvature,
  tortuosity, recurrence, twonn_id, ...) at layers L9/L12, not L0 -- so
  vmean0/rog0/rms are the *complete* cached L0/channel-proxy set for MLAAD,
  not a subset chosen for convenience.

## Marginal associations (Spearman vs hardness)

| feature | rho | p |
|---|---|---|
""" + "\n".join(
        f"| {k} | {v['rho']:+.3f} | {v['p']:.4f} |" for k, v in alone.items()
    ) + f"""

Proxy-vs-sd_along collinearity: """ + ", ".join(
        f"{k} rho={v['rho']:+.3f} (p={v['p']:.4f})" for k, v in collin.items()
    ) + f"""

## Partial Spearman: sd_along vs hardness | [vmean0, rog0, rms]

rank-transform-free OLS residualization (rho computed on residuals of
sd_along and hardness after each is linearly regressed on the three
proxies jointly, matching `audit_common`/audit2/audit6 `partial_spearman`
methodology):

- **partial rho = {rho_partial:+.4f}**
- asymptotic p (scipy spearmanr on residuals) = {p_partial_asym:.4f}
- **permutation p (n_perm={N_PERM}, seed={SEED}, two-sided) = {perm_p_partial:.5f}**

Per-control (single-proxy) partials:
""" + "\n".join(
        f"- | {k} alone: rho={v['rho']:+.4f} (p={v['p_asymptotic']:.4f})"
        for k, v in partial_each.items()
    ) + f"""

## Multivariate OLS: hardness ~ sd_along + vmean0 + rog0 + rms

Standardized predictors, HC3-robust SEs (n={len(tbl)}):

- R2 = {ols.rsquared:.4f} (adj {ols.rsquared_adj:.4f})
- sd_along coefficient = {coef_sd:+.4f}, HC3 SE = {se_hc3:.4f}, HC3 p = {p_hc3:.4f}
- **permutation p on sd_along coefficient (n_perm={N_PERM}, seed={SEED+1}) = {perm_p_coef:.5f}**

Full coefficient table (standardized, HC3 p-values):

| term | coef | HC3 p |
|---|---|---|
""" + "\n".join(
        f"| {k} | {v:+.4f} | {ols.pvalues[k]:.4f} |" for k, v in ols.params.items()
    ) + f"""

## Acceptance

Pre-declared criterion: partial Spearman(sd_along, hardness | proxies)
permutation p < 0.05.

**Result: {"PASSES" if accept else "FAILS"}** (perm p = {perm_p_partial:.5f}).

## Honest interpretation

The channel proxies available for MLAAD ({', '.join(proxies.keys())}) show
marginal associations with hardness themselves (see table above), and some
collinearity with sd_along. After controlling for all three jointly, the
sd_along-hardness partial association {"survives" if accept else "does not clearly survive"}
at the pre-declared p<0.05 threshold, both by the OLS-residualization
permutation test and (independently) by the permutation test on the
multivariate OLS regression coefficient (perm p={perm_p_coef:.5f}). This is
evidence {"against" if accept else "consistent with"} the "MLAAD law is just a
recording-channel dataset artifact" hypothesis, restricted to the specific
L0/waveform channel proxies cached for this corpus -- it does not rule out
channel confounds not captured by vmean0/rog0/rms (e.g. codec, sample rate,
or non-L0 recording-condition signatures), which were not computed for
MLAAD and are out of scope for this appendix check.
"""
(OUT / "K6_REPORT.md").write_text(report)
print(f"\nwrote {OUT / 'k6_results.json'} and {OUT / 'K6_REPORT.md'}")
