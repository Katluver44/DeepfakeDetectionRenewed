# Causal Interpretation — Temporal Smoothness Intervention

## Verdict: PARTIALLY SUPPORTED (asymmetric evidence)

## Evidence

Baseline EER: 0.3597 (phoneme_var=0.41023, frame_dist=16.37)

**Smoothing (mean Δ EER = +0.0957, mean Δ phoneme_var = −0.167):**

  Smoothing consistently and monotonically degrades detection as temporal variance
  decreases. EER traces a clean curve: MA k=3 (0.421) → MA k=5 (0.481) → MA k=10 (0.571).
  Exponential smoothing replicates the direction. This is the causal direction predicted
  by the hypothesis: reducing temporal variance impairs the detector.

**Jitter — two distinct regimes:**

  Gaussian noise (σ ∈ {0.01, 0.05, 0.1}): EER ≈ 0.358–0.360 — effectively unchanged.
  The model is robust to small additive i.i.d. noise; the discriminative signal is not
  disrupted by noise alone.

  Local permutation (k ∈ {3, 5, 10}): EER increases to 0.431–0.472, counter to the
  hypothesis. Critically, this is a CONFOUNDED manipulation: permutation disrupts the
  frame ordering while the phoneme_ids used to build the graph remain from the unperturbed
  input. The frame content and the graph structure are therefore mismatched, degrading the
  model for reasons unrelated to temporal variance. These conditions do not cleanly test
  the jitter hypothesis.

## Causal vs Correlational Assessment

The smoothing arm of the experiment establishes a causal link between temporal embedding
variance and detection difficulty. Artificially reducing phoneme segment variance — by
either moving average or exponential smoothing — degrades the detector's ability to separate
bonafide from spoof, with EER increasing by up to +0.21 absolute at the most extreme
smoothing level (MA k=10, EER=0.571). The effect is monotonic in both the manipulation
parameter (k or α) and the measured variance, providing strong evidence that temporal
smoothness is a mechanistically relevant factor, not merely a correlate of system type.

The jitter arm does not cleanly support the complementary prediction. Gaussian noise is
neutral (detector is robust to i.i.d. perturbations), and local permutation is confounded.
A cleaner jitter test would require recomputing phoneme_ids from the permuted audio,
which would require re-running the WavLM encoder.

**Conclusion:** The observational correlation (ρ=−0.48 between phoneme_var and EER, n=63
systems) reflects a genuine causal mechanism on the smoothing side. Systems that produce
temporally uniform WavLM representations evade detection not merely because of dataset or
architecture confounds, but because the GAT-based detector exploits temporal variance as a
discriminative signal. However, adding noise does not improve detection, suggesting the
relevant signal is structured temporal consistency rather than raw variance magnitude.
