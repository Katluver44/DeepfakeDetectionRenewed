# E12 — Geometric theory of ITW difficulty: a DIRECTIONAL covariate shift

## Verdict: **SUPPORTED (directional, not novelty)**  (4/4)

Frozen WavLM (mean-pooled L12⊕L9), standardized on the MLAAD training domain. Spoof axis taken three ways: w_mean = class-centroid difference (robust), w_ridge = the detector's own logit axis recovered by ridge, w_lda = whitened LDA (perfectly separates MLAAD ⇒ overfit). Δμ = domain-shift mean differences ITW−MLAAD, per class.

| # | test | quantity | value | pass |
|---|---|---|---|---|
| A1 | linear natural↔synthetic axis exists & tracks detector | AUC(w_mean) / ρ(w_mean·x, logit) [honest] | 0.889 / +0.365 | True |
| A2 | REAL-class shift aligns w/ that axis (bona-specific) | cos(Δμ_bona, w_mean) [null95=0.050] | **+0.445** | True |
|    | (spoof shift is NOT aligned) | cos(Δμ_spoof, w_mean) | -0.068 | |
| A3 | ITW-bona slides onto spoof side | slide along w_mean (bona→spoof gaps) | **+1.06** | True |
|    | (spoof slide) | ITW-spoof slide | +0.88 | |
| A4 | NOT a novelty effect (falsifier) | ρ(Maha-to-bona, logit) | **-0.299** | True |
|    | (and real audio is the MOST off-manifold) | med Maha: ITW-bona / ITW-spoof / MLAAD-bona | 52 / 45 / 44 | |
|    | (caveat: frozen linear probe ≠ GAT) | CV ρ(ridge,logit); cos(w_ridge,w_mean) | +0.58; +0.06 | |

## The theory
Frozen mean-pooled WavLM carries a low-dimensional **natural↔synthetic contrast axis** w_mean (centroid difference) that separates clean-natural from clean-synthetic speech at AUC 0.89 — the same separability the GAT detector achieves in-domain — and whose projection tracks the detector's logit out-of-sample (ρ=+0.36). In the clean training domain this axis is effectively a **recording-channel / processing signature** (vocoder smoothness, band-limiting, spectral regularity), because channel is confounded with the synthetic/natural label. (A frozen linear probe is only a rough proxy for the GAT — CV ρ(ridge,logit)=+0.58, and the probe's own axis is not collinear with w_mean — so we anchor the geometry on the robust centroid axis, not the probe.)

ITW's real-world recordings impose a covariate shift on GENUINE speech whose direction is **aligned with that same axis** — cos(Δμ_bona, w_mean)=+0.44, 9× the random baseline, and **specific to the real class** (the spoof shift is orthogonal, cos=-0.07). So genuine ITW audio slides 106% of a full bona→spoof gap onto the spoof side, the classes collapse, and the MLAAD-calibrated threshold reads almost all real audio as fake (E11).

**Crucially it is DIRECTIONAL, not radial.** The rival 'detector = one-class novelty model of the bona manifold' is falsified: distance-from-bona (Mahalanobis) does NOT drive the spoof logit (ρ=-0.30, wrong sign), and ITW *real* audio is actually the *furthest* from the training-bona Gaussian (Maha 52 > spoof 45) yet is not maximally 'spoof'. The detector keys on a SPECIFIC channel direction, not generic outlierness.

**One line:** ITW is hard because the recording-channel shift of real-world audio is collinear with the WavLM subspace the detector mistakes for 'synthetic' — a directional domain–decision alignment, not novelty and not spoof sophistication.

## Files: e12_geometry.csv, e12_projection.png, e12_maha_logit.png