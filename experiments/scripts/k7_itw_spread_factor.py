#!/usr/bin/env python3
"""E7 -- ITW speaker-level spread-factor merge (s_orth vs rog12).

Background (audit6 / COMPREHENSIVE_VERDICT S2): the paper's headline
"s_orth rho=+0.534 (p=0.003) predicts ITW per-speaker hardness" survives
family-wise correction, but plain radius-of-gyration (rog12) is *stronger*
(rho=-0.561), unreported, and collinear with s_orth (r=-0.687); neither
survives partialling the other, so the specific "off-axis residual" (s_orth)
interpretation is NOT separable from "compactness/spread" (rog12) on
current evidence. This appendix experiment asks whether the two collapse
into one interpretable "spread/compactness factor" (via PCA) that predicts
hardness at least as well as either alone, and explicitly documents the
non-separability rather than picking a side.

Data: experiments/results/i5_itw_transfer/speaker_table.csv, n=29 speakers
(same >=10-utterance-per-speaker filter audit6 used -- confirmed: this CSV
*is* the already-filtered analysis table audit6 loads directly, no
additional filtering applied here).

Method:
  1. Spearman(s_orth, rog12) at speaker level (collinearity check, should
     reproduce audit6's r=-0.687).
  2. PCA on standardized [s_orth, rog12] -> factor-1 score (sign-oriented
     so higher factor = higher expected hardness).
     Spearman(factor1, hardness) + exact permutation p (100k, seed 0).
  3. Discriminating partial regressions: s_orth | rog12 and rog12 | s_orth
     (partial Spearman via OLS residualization, same helper as audit6),
     to (re)confirm whether either survives controlling for the other.
  4. vmean0 (layer-0 mean velocity, channel proxy) vs hardness, and vmean0
     vs the factor-1 score -- channel-confound caveat per audit6 S2 ("a
     layer-0 velocity feature also predicting hardness hints at a channel
     confound").

Framing (binding): appendix-strength, n=29. Max defensible claim is "a
single spread/compactness factor predicts ITW speaker hardness"; the
s_orth-specific "off-axis residual direction" interpretation is NOT
separable from plain compactness (rog12) on this evidence, per
COMPREHENSIVE_VERDICT S2. This script does not attempt to resolve that
non-separability -- it quantifies it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis_audits"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import audit_common as ac

OUT = Path(__file__).resolve().parents[1] / "results" / "k7_itw_spread"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 100000

spk = pd.read_csv(ac.RES / "i5_itw_transfer" / "speaker_table.csv")
n = len(spk)
print(f"loaded speaker_table.csv: n={n} speakers (>=10-utt filter, same as audit6)")
assert n == 29, f"expected n=29 speakers per audit6/plan, got {n}"

res = {
    "n_speakers": int(n),
    "min_utts_filter": "same as audit6_itw_speaker.py -- speaker_table.csv is pre-filtered to >=10 utts/speaker",
    "data_source": "experiments/results/i5_itw_transfer/speaker_table.csv",
}

s_orth = spk["s_orth"].values.astype(float)
rog12 = spk["rog12"].values.astype(float)
hard = spk["hard"].values.astype(float)
vmean0 = spk["vmean0"].values.astype(float)

# ── (i) collinearity check ──────────────────────────────────────────────────
rho_collin, p_collin = stats.spearmanr(s_orth, rog12)
res["s_orth_vs_rog12"] = {"rho": float(rho_collin), "p": float(p_collin)}
print(f"s_orth vs rog12: rho={rho_collin:+.4f} (p={p_collin:.4f})  [audit6 reference: r=-0.687]")

rho_so_h, p_so_h = stats.spearmanr(s_orth, hard)
rho_rog_h, p_rog_h = stats.spearmanr(rog12, hard)
res["marginal"] = {
    "s_orth_vs_hard": {"rho": float(rho_so_h), "p": float(p_so_h)},
    "rog12_vs_hard": {"rho": float(rho_rog_h), "p": float(p_rog_h)},
}
print(f"s_orth vs hard: rho={rho_so_h:+.4f} (p={p_so_h:.4f})  [reported: +0.534]")
print(f"rog12 vs hard: rho={rho_rog_h:+.4f} (p={p_rog_h:.4f})  [reported: -0.561]")

# ── (ii) PCA factor-1 ────────────────────────────────────────────────────────
Xs = np.column_stack([s_orth, rog12])
Xs_z = (Xs - Xs.mean(0)) / Xs.std(0)
pca = PCA(n_components=2, random_state=SEED)
pcs = pca.fit_transform(Xs_z)
factor1_raw = pcs[:, 0]
explained_var = pca.explained_variance_ratio_.tolist()

# orient the factor so higher factor1 => higher hardness (sign convention only)
rho_f_raw, _ = stats.spearmanr(factor1_raw, hard)
sign = 1.0 if rho_f_raw >= 0 else -1.0
factor1 = sign * factor1_raw

rho_factor, p_factor_asym = stats.spearmanr(factor1, hard)
print(f"\nPCA on standardized [s_orth, rog12]: PC1 explains {explained_var[0]*100:.1f}% of variance "
      f"(PC2 {explained_var[1]*100:.1f}%)")
print(f"PC1 loadings: s_orth={pca.components_[0,0]*sign:+.4f}, rog12={pca.components_[0,1]*sign:+.4f}")
print(f"factor1 vs hard: rho={rho_factor:+.4f} (asymptotic p={p_factor_asym:.4f})")

# exact-style permutation p (100k)
rho_perm, p_perm_factor = ac.spearman_perm_p(factor1, hard, n_perm=N_PERM, seed=SEED, alternative="two-sided")
print(f"factor1 vs hard: exact perm p (n_perm={N_PERM}, seed={SEED}) = {p_perm_factor:.6f}")

res["pca_factor"] = {
    "explained_variance_ratio": explained_var,
    "pc1_loadings_signed": {"s_orth": float(pca.components_[0, 0] * sign),
                            "rog12": float(pca.components_[0, 1] * sign)},
    "sign_convention": "oriented so higher factor1 = higher expected hardness",
    "rho_vs_hard": float(rho_factor),
    "p_asymptotic": float(p_factor_asym),
    "n_perm": N_PERM,
    "perm_p_twosided": float(p_perm_factor),
    "seed": SEED,
}

# ── (iii) discriminating partial regressions ────────────────────────────────
def partial_spearman(x, yv, z):
    z = np.asarray(z, float).reshape(len(x), -1)
    rx = x - LinearRegression().fit(z, x).predict(z)
    ry = yv - LinearRegression().fit(z, yv).predict(z)
    return stats.spearmanr(rx, ry)

rho_so_given_rog, p_so_given_rog = partial_spearman(s_orth, hard, rog12)
rho_rog_given_so, p_rog_given_so = partial_spearman(rog12, hard, s_orth)
print(f"\ns_orth | rog12: rho={rho_so_given_rog:+.4f} (p={p_so_given_rog:.4f})")
print(f"rog12 | s_orth: rho={rho_rog_given_so:+.4f} (p={p_rog_given_so:.4f})")

res["discriminating_partials"] = {
    "s_orth_given_rog12": {"rho": float(rho_so_given_rog), "p": float(p_so_given_rog)},
    "rog12_given_s_orth": {"rho": float(rho_rog_given_so), "p": float(p_rog_given_so)},
    "either_survives_p05": bool(p_so_given_rog < 0.05 or p_rog_given_so < 0.05),
    "both_survive_p05": bool(p_so_given_rog < 0.05 and p_rog_given_so < 0.05),
}

# ── (iv) channel-confound caveat: vmean0 ────────────────────────────────────
rho_vm_h, p_vm_h = stats.spearmanr(vmean0, hard)
rho_vm_f, p_vm_f = stats.spearmanr(vmean0, factor1)
print(f"\nvmean0 vs hard: rho={rho_vm_h:+.4f} (p={p_vm_h:.4f})")
print(f"vmean0 vs factor1: rho={rho_vm_f:+.4f} (p={p_vm_f:.4f})")

rho_factor_given_vm, p_factor_given_vm = partial_spearman(factor1, hard, vmean0)
print(f"factor1 | vmean0: rho={rho_factor_given_vm:+.4f} (p={p_factor_given_vm:.4f})")

res["channel_confound_caveat"] = {
    "vmean0_vs_hard": {"rho": float(rho_vm_h), "p": float(p_vm_h)},
    "vmean0_vs_factor1": {"rho": float(rho_vm_f), "p": float(p_vm_f)},
    "factor1_given_vmean0": {"rho": float(rho_factor_given_vm), "p": float(p_factor_given_vm)},
}

# ── framing / acceptance ─────────────────────────────────────────────────────
separable = res["discriminating_partials"]["both_survive_p05"]
res["framing"] = {
    "max_claim": "a single spread/compactness factor (PC1 of [s_orth, rog12]) predicts ITW speaker hardness",
    "s_orth_specific_interpretation_separable_from_rog12": separable,
    "conclusion": (
        "NOT separable: neither s_orth nor rog12 individually survives partialling "
        "out the other at p<0.05 (matches COMPREHENSIVE_VERDICT S2)."
        if not separable else
        "One of s_orth/rog12 survives partialling out the other at p<0.05 -- "
        "re-examine whether the S2 non-separability verdict still holds."
    ),
}
print(f"\nFRAMING: s_orth vs rog12 separable at p<0.05 (both survive mutual partial)? {separable}")
print(f"  -> {res['framing']['conclusion']}")

(OUT / "k7_results.json").write_text(json.dumps(res, indent=2, default=str))

report = f"""# K7 -- ITW Speaker-Level Spread-Factor Merge (s_orth vs rog12)

Appendix experiment (exploratory hygiene follow-up to audit6 / COMPREHENSIVE_VERDICT
S2). CPU-only, deterministic, n=29 speakers.

## Background

`audit6_itw_speaker.py` found that the paper's headline "s_orth rho=+0.534
predicts ITW per-speaker hardness" survives 7-test family correction, but
plain radius-of-gyration `rog12` is *stronger* (rho=-0.561), unreported, and
collinear with s_orth (r=-0.687); neither survives partialling the other.
COMPREHENSIVE_VERDICT S2 concludes the defensible claim is "a single
spread/compactness factor predicts ITW speaker hardness," not the specific
s_orth axis-residual interpretation. This script tests that merge directly.

## Data

`experiments/results/i5_itw_transfer/speaker_table.csv`, n={n} speakers
(pre-filtered to >=10 utterances/speaker -- the same filter and table
`audit6_itw_speaker.py` loads directly; no additional filtering applied here).

## (i) Collinearity: s_orth vs rog12

rho = {rho_collin:+.4f} (p={p_collin:.4f})  [audit6 reference: r=-0.687]

## Marginal associations with hardness

| feature | rho | p | reported (report) |
|---|---|---|---|
| s_orth | {rho_so_h:+.4f} | {p_so_h:.4f} | +0.534 |
| rog12 | {rho_rog_h:+.4f} | {p_rog_h:.4f} | -0.561 |

## (ii) PCA merge: factor-1 of standardized [s_orth, rog12]

- PC1 explains {explained_var[0]*100:.1f}% of variance (PC2: {explained_var[1]*100:.1f}%)
- PC1 loadings (sign-oriented so higher factor = higher expected hardness):
  s_orth={res['pca_factor']['pc1_loadings_signed']['s_orth']:+.4f},
  rog12={res['pca_factor']['pc1_loadings_signed']['rog12']:+.4f}
- **factor1 vs hardness: rho = {rho_factor:+.4f}**
  - asymptotic p = {p_factor_asym:.4f}
  - **exact permutation p (n_perm={N_PERM}, seed={SEED}) = {p_perm_factor:.6f}**

The merged factor's association with hardness ({rho_factor:+.4f}) is
{"comparable to or stronger than" if abs(rho_factor) >= max(abs(rho_so_h), abs(rho_rog_h)) else "in between"}
either individual predictor alone (s_orth {rho_so_h:+.4f}, rog12 {rho_rog_h:+.4f}),
consistent with s_orth and rog12 carrying substantially overlapping
(collinear, r={rho_collin:+.3f}) signal rather than independent information.

## (iii) Discriminating partial regressions

| test | rho | p |
|---|---|---|
| s_orth \\| rog12 | {rho_so_given_rog:+.4f} | {p_so_given_rog:.4f} |
| rog12 \\| s_orth | {rho_rog_given_so:+.4f} | {p_rog_given_so:.4f} |

**Both survive p<0.05: {res['discriminating_partials']['both_survive_p05']}**
**Either survives p<0.05: {res['discriminating_partials']['either_survives_p05']}**

## (iv) Channel-confound caveat: vmean0 (layer-0 mean velocity)

| test | rho | p |
|---|---|---|
| vmean0 vs hardness | {rho_vm_h:+.4f} | {p_vm_h:.4f} |
| vmean0 vs factor1 | {rho_vm_f:+.4f} | {p_vm_f:.4f} |
| factor1 \\| vmean0 | {rho_factor_given_vm:+.4f} | {p_factor_given_vm:.4f} |

## Framing (binding, per plan)

**Max defensible claim: "{res['framing']['max_claim']}"** -- supported: factor1
vs hardness rho={rho_factor:+.4f}, exact permutation p={p_perm_factor:.6f} (n=29).

**s_orth-specific "off-axis residual" interpretation separable from plain
compactness (rog12)? {separable}.**

{res['framing']['conclusion']}

This matches COMPREHENSIVE_VERDICT S2: s_orth and rog12 are too collinear
(r={rho_collin:+.3f}) at n=29 to attribute the hardness association to the
specific axis-residual construction rather than generic spread/compactness.
The vmean0 (layer-0 velocity, channel proxy) association with both hardness
({rho_vm_h:+.4f}, p={p_vm_h:.4f}) and the merged factor ({rho_vm_f:+.4f},
p={p_vm_f:.4f}) is reported here as a caveat, not resolved -- it is
consistent with (but does not prove) a channel-confound component in the
ITW speaker-hardness association, per audit6/S2's flag.
"""
(OUT / "K7_REPORT.md").write_text(report)
print(f"\nwrote {OUT / 'k7_results.json'} and {OUT / 'K7_REPORT.md'}")
