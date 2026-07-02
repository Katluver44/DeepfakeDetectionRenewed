# K6 -- Channel-Confound Control on the MLAAD sd_along Law

Appendix experiment (exploratory, not pre-registered). CPU-only, deterministic.

## Question

Is the MLAAD-English sd_along-vs-hardness association (audited in
`audit2_sdalong_claim.py`: LOSO R2=0.273, headline p=0.0005) actually a
recording-channel artifact, rather than a property of the spoof-detection
axis geometry? We test whether it survives partialling out layer-0
"channel proxy" features.

## Data

- `experiments/results/i3_position_geometry/system_position.csv`, n=61
  MLAAD-English systems (the exact table audit2 uses; fidelity re-check
  against an independent recomputation via `audit_common.loso_axis_features`
  on cached L12 embeddings gives corr=0.9997).
- Channel proxies available: `vel_mean_L0` (vmean0), `rog_L0` (rog0), `rms`
  (waveform RMS). **Limitation**: `i2_geometry_battery/system_table.csv`
  only computed the full geometric battery (vel_cv, vel_entropy, curvature,
  tortuosity, recurrence, twonn_id, ...) at layers L9/L12, not L0 -- so
  vmean0/rog0/rms are the *complete* cached L0/channel-proxy set for MLAAD,
  not a subset chosen for convenience.

## Marginal associations (Spearman vs hardness)

| feature | rho | p |
|---|---|---|
| sd_along | +0.589 | 0.0000 |
| vmean0 | -0.197 | 0.1289 |
| rog0 | -0.425 | 0.0006 |
| rms | -0.433 | 0.0005 |

Proxy-vs-sd_along collinearity: vmean0 rho=-0.393 (p=0.0017), rog0 rho=-0.446 (p=0.0003), rms rho=-0.335 (p=0.0084)

## Partial Spearman: sd_along vs hardness | [vmean0, rog0, rms]

rank-transform-free OLS residualization (rho computed on residuals of
sd_along and hardness after each is linearly regressed on the three
proxies jointly, matching `audit_common`/audit2/audit6 `partial_spearman`
methodology):

- **partial rho = +0.5792**
- asymptotic p (scipy spearmanr on residuals) = 0.0000
- **permutation p (n_perm=10000, seed=0, two-sided) = 0.00010**

Per-control (single-proxy) partials:
- | vmean0 alone: rho=+0.5453 (p=0.0000)
- | rog0 alone: rho=+0.5672 (p=0.0000)
- | rms alone: rho=+0.5810 (p=0.0000)

## Multivariate OLS: hardness ~ sd_along + vmean0 + rog0 + rms

Standardized predictors, HC3-robust SEs (n=61):

- R2 = 0.4027 (adj 0.3601)
- sd_along coefficient = +0.0723, HC3 SE = 0.0243, HC3 p = 0.0029
- **permutation p on sd_along coefficient (n_perm=10000, seed=1) = 0.00030**

Full coefficient table (standardized, HC3 p-values):

| term | coef | HC3 p |
|---|---|---|
| const | +0.1962 | 0.0000 |
| sd_along | +0.0723 | 0.0029 |
| vmean0 | +0.0268 | 0.2743 |
| rog0 | -0.0395 | 0.1224 |
| rms | -0.0146 | 0.4088 |

## Acceptance

Pre-declared criterion: partial Spearman(sd_along, hardness | proxies)
permutation p < 0.05.

**Result: PASSES** (perm p = 0.00010).

## Honest interpretation

The channel proxies available for MLAAD (vmean0, rog0, rms) show
marginal associations with hardness themselves (see table above), and some
collinearity with sd_along. After controlling for all three jointly, the
sd_along-hardness partial association survives
at the pre-declared p<0.05 threshold, both by the OLS-residualization
permutation test and (independently) by the permutation test on the
multivariate OLS regression coefficient (perm p=0.00030). This is
evidence against the "MLAAD law is just a
recording-channel dataset artifact" hypothesis, restricted to the specific
L0/waveform channel proxies cached for this corpus -- it does not rule out
channel confounds not captured by vmean0/rog0/rms (e.g. codec, sample rate,
or non-L0 recording-condition signatures), which were not computed for
MLAAD and are out of scope for this appendix check.
