# E9 (re-run) — Is C causal for detection? Multi-seed significance test

Direct intervention on the detector's **frozen** WavLM L12 representation (C = −rog@L12), applied equally to spoof+bona. ASVspoof eval (A07–A19), 800 spoof + 800 bona. Detector: **robust_GOAT × 3 seeds** (s1, s3, s7). 16 alphas spanning [0.5, 4.0] (doubled density vs original).

The **shift control** translates the representation by an equal ‖Δh‖ while leaving rog/velocities (hence C and T) mathematically unchanged. The **C-specific effect** is the paired difference ΔAUC = AUC_scale − AUC_shift at matched energy; significance is a paired utterance bootstrap (B=2000) of the across-seed mean, plus 3/3 seed-sign consistency.

## Headline significance (moderate regime)
- **More compact (α∈[0.5, 0.625, 0.75, 0.875]):** ΔAUC = -0.0573 (95% CI [-0.0627, -0.0516], p=0.0000) ⇒ compaction **specifically worsens** beyond matched energy.
- **Less compact (α∈[1.125, 1.25, 1.375, 1.5]):** ΔAUC = +0.0016 (95% CI [-0.0007, +0.0039], p=0.166) ⇒ in the *moderate* expansion regime the C-specific benefit is **NOT significant** (CI straddles 0) and is **not seed-robust** (seed s3 dissents, signs `[1,-1,1]`). A significant C-specific *improvement* only emerges at **strong** expansion (α≥1.75: ΔAUC up to +0.049, p≈0) — but s3 still dissents there, so that arm is seed-dependent and partly rides generic shift-control collapse, not a clean moderate-regime effect.
- Baseline (α=1): EER=0.076, AUC=0.978, C(rog)=14.85.

## C-scale sweep (mean ± SD across seeds)
| α | C (rog) | EER (scale) | AUC (scale) | EER (shift) | AUC (shift) | ΔAUC [95% CI] | p | seed signs |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 7.5 | 0.278±0.009 | 0.788±0.009 | 0.085±0.014 | 0.973±0.007 | -0.1851 [-0.205,-0.165]*** | 0.0000 | [-1, -1, -1] |
| 0.625 | 9.3 | 0.129±0.020 | 0.947±0.010 | 0.076±0.008 | 0.978±0.005 | -0.0313 [-0.038,-0.025]*** | 0.0000 | [-1, -1, -1] |
| 0.75 | 11.2 | 0.094±0.004 | 0.969±0.003 | 0.072±0.004 | 0.979±0.002 | -0.0101 [-0.013,-0.007]*** | 0.0000 | [-1, -1, -1] |
| 0.875 | 13.0 | 0.077±0.009 | 0.977±0.004 | 0.074±0.006 | 0.980±0.002 | -0.0026 [-0.005,-0.000]* | 0.0230 | [-1, -1, -1] |
| 1.0 | 14.9 | 0.076±0.013 | 0.978±0.006 | 0.076±0.011 | 0.979±0.004 | -0.0004 [-0.002,+0.002] | 0.7180 | [1, -1, -1] |
| 1.125 | 16.8 | 0.074±0.008 | 0.979±0.006 | 0.072±0.008 | 0.980±0.003 | -0.0009 [-0.003,+0.001] | 0.3860 | [-1, -1, 1] |
| 1.25 | 18.6 | 0.074±0.007 | 0.979±0.004 | 0.078±0.003 | 0.979±0.001 | +0.0001 [-0.003,+0.003] | 0.9620 | [-1, -1, 1] |
| 1.375 | 20.5 | 0.070±0.007 | 0.981±0.004 | 0.075±0.005 | 0.978±0.004 | +0.0024 [-0.001,+0.006] | 0.1450 | [1, -1, 1] |
| 1.5 | 22.4 | 0.074±0.008 | 0.979±0.005 | 0.081±0.021 | 0.974±0.008 | +0.0047 [+0.001,+0.009]* | 0.0150 | [1, -1, 1] |
| 1.75 | 26.1 | 0.078±0.009 | 0.977±0.004 | 0.097±0.029 | 0.965±0.017 | +0.0123 [+0.007,+0.018]*** | 0.0000 | [1, -1, 1] |
| 2.0 | 29.8 | 0.087±0.002 | 0.973±0.003 | 0.104±0.025 | 0.955±0.021 | +0.0184 [+0.011,+0.026]*** | 0.0000 | [1, -1, 1] |
| 2.25 | 33.4 | 0.094±0.006 | 0.970±0.005 | 0.116±0.029 | 0.949±0.022 | +0.0216 [+0.013,+0.030]*** | 0.0000 | [1, -1, 1] |
| 2.5 | 37.3 | 0.099±0.010 | 0.964±0.005 | 0.124±0.026 | 0.938±0.025 | +0.0262 [+0.017,+0.035]*** | 0.0000 | [1, -1, 1] |
| 3.0 | 44.7 | 0.125±0.012 | 0.951±0.011 | 0.157±0.050 | 0.914±0.044 | +0.0378 [+0.026,+0.050]*** | 0.0000 | [1, -1, 1] |
| 3.5 | 52.2 | 0.138±0.023 | 0.940±0.017 | 0.172±0.066 | 0.893±0.063 | +0.0477 [+0.034,+0.062]*** | 0.0000 | [1, -1, 1] |
| 4.0 | 59.5 | 0.149±0.033 | 0.930±0.019 | 0.189±0.067 | 0.881±0.068 | +0.0492 [+0.035,+0.064]*** | 0.0000 | [1, -1, 1] |

## T sweep (mean ± SD across seeds) — secondary
| intervention | param | measured T | C@L9 (rog) | EER | AUC |
|---|---|---|---|---|---|
| T_jitter | 0.5 | 2.678±0.007 | 5.06±0.00 | 0.075±0.004 | 0.978±0.003 |
| T_jitter | 1.0 | 2.639±0.011 | 5.74±0.01 | 0.085±0.012 | 0.973±0.005 |
| T_jitter | 2.0 | 2.656±0.027 | 7.85±0.04 | 0.137±0.013 | 0.940±0.005 |
| T_jitter | 4.0 | 2.635±0.024 | 13.39±0.02 | 0.330±0.030 | 0.734±0.034 |
| T_smooth | 3.0 | 2.616±0.028 | 4.12±0.00 | 0.098±0.019 | 0.968±0.009 |
| T_smooth | 5.0 | 2.682±0.014 | 3.60±0.02 | 0.163±0.052 | 0.922±0.043 |
| T_smooth | 9.0 | 2.681±0.035 | 2.98±0.01 | 0.367±0.046 | 0.719±0.057 |
| T_smooth | 15.0 | 2.600±0.013 | 2.47±0.01 | 0.396±0.030 | 0.663±0.036 |

## Verdict
- **The causal compaction arm is now statistically significant and seed-robust.** Making the
  representation *more compact* specifically worsens detection beyond its matched-energy shift
  control: the moderate-compaction regime (α∈[0.5,0.875]) gives ΔAUC = −0.057, 95% CI
  [−0.063, −0.052], p<0.001 from a paired utterance bootstrap (B=2000), and **all 3 seeds agree
  in sign at every compaction α** (one-sided sign consistency 3/3). This is the headline result
  the re-run was meant to establish: **`more compact → harder` is causal, not just correlational.**
- **The expansion arm does NOT replicate cleanly.** In the moderate expansion regime
  (α∈[1.125,1.5]) the C-specific benefit is small and **not significant** (ΔAUC = +0.0016,
  CI [−0.0007, +0.0039], p=0.17), and seed s3 shows the opposite sign throughout expansion. The
  C-specific *improvement* reaches significance only under **strong** expansion (α≥1.75), but it
  is seed-dependent (s3 still dissents) and there the matched shift control is itself collapsing,
  so it is not clean moderate-regime evidence. **Honest reading:** the multi-seed test
  *strengthens* the compaction direction but *weakens* the earlier single-seed claim that moderate
  expansion specifically improves detection — that was within seed-to-seed noise.
- **Net:** C is causal with polarity `more compact → harder`; the rigorous, replicated evidence
  is the compaction (α<1) arm. The expansion arm is directionally consistent on 2/3 seeds but is
  not a significant, seed-robust effect in the clean moderate regime.
- T (smooth/jitter at L9) again moves L9 compactness rather than vel-entropy (measured T stays
  2.60–2.68 vs C@L9 swinging 2.5→13.4 with EER); not an isolable causal lever (see T table).

## Files
- `e9_causal_C_seeds.csv`, `e9_causal_C_aggregate.csv`, `e9_causal_T_seeds.csv`
- `e9_deltaAUC_bootstrap.csv`, `e9_stats.json`
- figures: `e9_causal_C_aggregate.png`, `e9_deltaAUC.png`