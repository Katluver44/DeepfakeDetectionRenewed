# PREDICTOR_SPEC.md — Frozen Pre-Registration for the Genuinely Held-Out ASVspoof21 Confirmation

**Status: FROZEN AT COMMIT TIME. This document contains NO results. No held-out
data has been scored. It exists to be committed and externally timestamped
BEFORE any scoring occurs.**

**Author's declaration:** the single predictor specified below (P3) is the
ONLY predictor that will ever be reported for this held-out condition. No
other predictor (P1, P2, P4, or any variant, transform, subset, or detector
substitution not listed under "Nuisance parameters" below) will be computed,
tried, or reported for this confirmation. If P3 fails, the confirmation
fails; there is no fallback predictor and no second attempt on this
condition.

---

## 1. Background (why this document exists)

Audit 3 (`experiments/axis_audits/audits_outputs/audit3_asvspoof_prospective/AUDIT3_SUMMARY.md`,
folded into `experiments/axis_audits/audits_outputs/COMPREHENSIVE_VERDICT.md`
§2 S1) found that the existing ASVspoof 2021 "prospective" result is
**promising but not confirmatory**:

- Four predictors (P1–P4) were registered in `j4_asvspoof21_prospective.py`;
  the declared **primary**, P1, failed (ρ=0.25, ns). The reported headline
  (ρ≈0.60/0.64) is P3, a promoted secondary.
- n = 13 attacks (A07–A19, codec = `none` only).
- Exact permutation p = 0.034 (WavLM) / 0.020 (AASIST) uncorrected; after
  Holm within the 4-predictor family: 0.134 / 0.082; across the 8-test family:
  0.235 / 0.164. Nothing survives family-wise correction.
- The "pre-registration" JSON was written 63 seconds before detector scoring,
  by the same script, with no external commitment device.
- In P3's favor: WavLM and AASIST per-attack hardness are only weakly
  correlated (ρ=0.21) — a quasi-independent replication — and P3's ρ is
  stable under measurement-noise bootstrap (never crosses 0) and
  leave-one-attack-out jackknife (ρ stays in [0.49, 0.75]).

The audit's recommended decisive next step: **one genuinely out-of-sample
condition, scored with a single pre-committed predictor (P3), externally
timestamped before scoring.** This document is that pre-commitment.

---

## 2. The single predictor under test: P3

P3 is copied **verbatim in construction** from
`experiments/scripts/j4_asvspoof21_prospective.py` (lines ~107–144) and its
independent recomputation in `experiments/axis_audits/audit3_asvspoof_prospective.py`.
It is **not modified, tuned, or re-derived** for the held-out condition —
only re-applied to the held-out corpus's own embeddings and labels.

### 2.1 Inputs

- Frozen WavLM-base (`microsoft/wavlm-base`, HuggingFace `transformers`),
  hidden_states[12] (layer 12 of 12), mean-pooled over time
  (`.mean(1)`) → 768-d embedding `X` per utterance. This is the exact encoder
  and layer used throughout the I/J experiment series (`px_common.load_frozen_wavlm`).
- Ground-truth bona/spoof labels and attack/system identifiers of the
  **held-out corpus itself** (see §3), taken from corpus metadata. The axis
  construction never uses any detector output — only these ground-truth
  labels and the WavLM embeddings.

### 2.2 Construction (corpus-internal, leave-one-attack-out LDA axis)

For a held-out corpus with bona set `B` and attack systems `a ∈ A`:

1. Compute `X` (768-d WavLM-L12 mean-pooled embeddings) for every held-out
   utterance.
2. For each attack `a`:
   a. Define the **training set** `tr = B ∪ {utterances of every attack
      except a}` (leave-one-attack-out: `a`'s own utterances are excluded
      from fitting the axis/discriminant that will score `a`).
   b. Fit `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr",
      shrinkage="auto")` on `X[tr]` with binary labels (0=bona, 1=spoof) —
      this is the "shrinkage-LDA direction" referred to as P3 in J4/Audit 3.
   c. Score attack `a`'s utterances with `lda.decision_function(X[attack==a])`
      to get `pos_lda` for each utterance of `a`.
3. Per attack, take `pos_lda_a = median(pos_lda over that attack's utterances)`.
4. Convert to a predicted-hardness score via a rank-preserving z-score and
   sign flip (matching J4's convention that higher predicted value = harder):
   `P3_a = -zscore(pos_lda_a)` computed across all attacks in the held-out
   condition (z-score uses the mean/std of `pos_lda` over the `A`-length
   vector of per-attack medians, i.e. `(v - mean(v)) / (std(v) + 1e-12)`).
5. `P3_a` for `a = 1 … |A|` is the full predicted-hardness ranking. No other
   quantity derived from `pos_lda`, `sd_lda`, or any other axis
   (`pos_int`/centroid axis = P1, `pos_int + sd_int` = P2, MLAAD-transfer
   axis = P4) is part of P3 and none of P1/P2/P4 will be scored or reported
   for the held-out condition.

### 2.3 Actual (ground-truth) hardness

For each attack `a`, actual hardness is defined exactly as in J4/Audit 3:
`hardness_a = 1 − AUROC(labels = {bona=0, attack a=1}, scores = detector
score)`, using the **held-out corpus's own bona pool** (not any other
corpus's bona) as the negative class. If multiple detector seeds are used,
`hardness_a` is the mean of `1-AUROC` across seeds (matching J4's 3-seed
`mlaad_robust_goat` mean).

**Detector identity is a nuisance parameter, not part of P3.** The detector
used to compute actual hardness must be pre-specified before scoring (see
§4) but is not part of the P3 predictor itself, which is detector-free by
construction (built only from WavLM embeddings + ground-truth labels).

---

## 3. The held-out condition to be scored

Per the Phase-1 data-recon findings (`PHASE1_REPORT.md`), the following
condition is available and has **not** been touched by J4/Audit 3 or any
other prior script in this repository:

> **ASVspoof 2021 DF (DeepFake) evaluation partition**, HuggingFace dataset
> `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` (611,829 trials; 22,617
> bonafide / 589,212 spoof), restricted to
> **`notes.source == "asvspoof"`** (the subset built from the same underlying
> ASVspoof attack systems A07–A19 as the LA track, but re-processed through
> the DF pipeline — i.e. different post-processing/codec exposure and a
> disjoint utterance set from the LA `codec=='none'` subset J4 scored) —
> **or**, as a secondary acceptable choice if the `source=="asvspoof"` subset
> is too small per-attack, the full DF attack roster restricted to
> **`notes.codec == "nocodec"`**.

Exactly one of these two condition definitions must be chosen and locked
*before* any scoring script is executed, and recorded as such in
`results_heldout.json`'s `condition` field. The choice must be made on
feasibility grounds only (minimum utterances/attack, see §4.4) — never
on any peek at hardness or correlation.

This condition satisfies "genuinely held out" because:
- No utterance from `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` was
  downloaded, embedded, or scored by J4, J5, J2, or any audit script.
- The LA `codec=='none'` utterances scored by J4 are a disjoint utterance
  set (different `path`/`utterance_id` namespace: `LA_E_*` vs `DF_E_*`).
- The DF partition additionally exposes new attack systems (HUB-\*, SPO-\*,
  Task1/2-team\*, VCC2018/VCC2020-sourced) that were never seen in any prior
  I/J experiment or MLAAD training data referenced in this repo.

If, at scoring time, this condition turns out to be infeasible (see
blockers in `PHASE1_REPORT.md`), the fallback specified there — ASVspoof
2021 **LA**, restricted to `notes.codec != "none"` (i.e. `alaw`, `ulaw`,
`g722`, `opus`, `pstn`, `gsm` — genuinely unscored codec conditions on the
same attacks J4 already used for `codec=='none'`) — may be substituted, but
**only one of these two conditions, chosen before scoring, will be used**;
they are not both scored and the more favorable one reported.

---

## 4. Pre-specified statistical test

- **Test statistic:** Spearman rank correlation ρ between the per-attack
  vector `P3_a` (predicted hardness) and the per-attack vector `hardness_a`
  (actual, 1-AUROC-based), computed over all attacks in the locked held-out
  condition with ≥ `N_MIN_UTTS_PER_ATTACK` utterances (§4.4).
- **Direction:** one-sided, pre-specified as **ρ > 0** (P3 predicts higher
  hardness for attacks it ranks as harder). This matches the sign
  convention used throughout J4/Audit 3.
- **p-value:** exact permutation p-value, one-sided (fraction of label
  permutations of the actual-hardness vector giving permuted ρ ≥ observed ρ),
  computed with a fixed, pre-specified number of permutations
  (200,000, matching `audit_common.spearman_perm_p`'s `n_perm` used in Audit
  3) and a fixed seed (`seed=7`, reused from Audit 3 for consistency).
- **No other test statistic** (Pearson r, AUC-of-ranks, top-k hit rate,
  R², etc.) will be computed or reported as a primary result for this
  condition. Descriptive companions (e.g. a scatter plot, the raw per-attack
  table) may be produced but do not constitute additional hypothesis tests.

### 4.1 Acceptance criterion (decided in advance)

The confirmation is declared **SUCCESSFUL** if and only if:

1. one-sided exact permutation p < 0.05, **and**
2. ρ > 0 (consistent with the pre-specified direction — condition 1 already
   implies this under a one-sided test, stated separately for clarity).

The confirmation is declared **FAILED** if p ≥ 0.05. There is no "promising
but not significant" middle category for this document's purposes — Audit 3
already established that framing is exhausted for this predictor; this is
the single, final, binary confirmatory test on new data. No family-wise
correction is applied here because there is exactly one test being run (n=1
predictor × 1 condition × 1 statistic); this is the entire point of running
only P3 on only one new condition.

### 4.2 What will be reported regardless of outcome

- The observed ρ and exact permutation p.
- The per-attack table (`P3_a`, `hardness_a`, n_utts).
- Whether the acceptance criterion was met.
- A restatement that this was the sole pre-registered test — win or lose.

A failed confirmation will be reported as a failed confirmation. It will
**not** be treated as license to try P1, P2, P4, a different detector, a
different codec condition, or a different DF subset after the fact. Any
such follow-up exploration must be labeled exploratory and must not be
merged into this confirmation's reported result.

### 4.3 Explicit no-scoring statement

**As of this document's commit, NO scoring of any held-out data has
occurred.** No embeddings have been computed on `ASVspoof2021_DF` (or the
LA non-'none' codec fallback) for this confirmation, no LDA axis has been
fit on held-out labels, no detector has been run on held-out audio for this
purpose, and no ρ or p-value for the held-out condition exists anywhere in
this repository as of this commit. `score_heldout.py` (§5) is written but
deliberately gated and has not been executed.

### 4.4 Feasibility parameter (pre-specified, not data-dependent)

`N_MIN_UTTS_PER_ATTACK = 20` — an attack system is included in the held-out
confirmation only if it has ≥ 20 utterances in the locked condition (half of
J4's 40/attack, chosen for feasibility given DF's more fragmented per-attack
counts observed during Phase-1 recon, not chosen after looking at hardness
results). If fewer than 8 attacks meet this threshold under the primary
condition definition (`source=="asvspoof"`), the secondary condition
definition (`codec=="nocodec"`, full DF roster) is used instead, per §3.
This substitution rule is mechanical and pre-specified; it does not depend
on any hardness or correlation computation.

---

## 5. Scope limitation

This document governs exactly one confirmatory test: P3 vs. one held-out
ASVspoof21 condition, as defined above. It does not extend to, and must not
be cited in support of, any other predictor, corpus, or detector comparison
in this repository.
