# Experiment 4: System-level Generalization to ASVspoof A01–A06

**N = 6 ASVspoof 2019 LA attack systems · 167 utterances/system · robust_goat inference**

---

## Setup

- **Dataset**: ASVspoof 2019 LA train split, systems A01–A06 (167 sampled per system, 1002 bonafide)
- **WavLM features**: L9 (vel_entropy) and L12 (rog) extracted per utterance, averaged per system
- **Inference**: `models/robust_goat.ckpt` — the same model trained on MLAAD
- **EER convention**: higher EER = harder to detect (more dangerous spoof system)

---

## Per-system Results

| System | C (−rog@L12) | T (vel_entropy@L9) | EER (robust_goat) | Notes |
|--------|-------------|------------------|------------------|-------|
| A01 | −10.857 | 2.753 | **0.038** | Easiest to detect (Merlin+WORLD TTS) |
| A02 | −10.939 | 2.801 | 0.102 | Merlin+WORLD (v2) |
| A03 | −10.813 | 2.767 | 0.120 | Tacotron neural TTS |
| A04 | −10.670 | 2.712 | 0.067 | Tacotron2 neural TTS |
| A05 | −10.953 | 2.767 | 0.094 | Voice conversion (Griffin-Lim) |
| A06 | −10.927 | 2.780 | **0.257** | **Hardest** (voice conversion / AutoVC) |

### EER ordering (hardest first): A06 > A03 > A02 > A05 > A04 > A01

---

## Correlations with EER (N=6 systems)

| Metric | Pearson r | Spearman ρ | p (two-tailed) | MLAAD reference |
|--------|----------|------------|--------------|----------------|
| **C** | **−0.380** | **−0.200** | 0.458 | r=+0.329, ρ=+0.293 |
| **T** | **+0.451** | **+0.771** | 0.370 | r=+0.276, ρ=+0.330 |

Note: MLAAD reference used residual hardness as outcome; ASVspoof uses raw EER.
In MLAAD, higher C (more compact) → harder. In ASVspoof, r(C, EER) = −0.380 (reversed direction).
In MLAAD, higher T (more irregular) → harder. In ASVspoof, r(T, EER) = +0.451 (consistent direction).

---

## Rank Analysis

T rank (1=most irregular, 6=least) vs EER rank (1=hardest, 6=easiest):

| System | EER rank | C rank | T rank | T|EER match? |
|--------|---------|--------|--------|------------|
| A06 | 1 (hardest) | 4 | **2** | Near ✓ |
| A03 | 2 | 2 | **3** | ✓ |
| A02 | 3 | 5 | **1** | Off by 2 |
| A05 | 4 | 6 | **3** | Off by 1 |
| A04 | 5 | **1** | **6** | Off by 1 (C reversed) |
| A01 | 6 (easiest) | 3 | **5** | Near ✓ |

Spearman ρ(T rank, EER rank) = +0.771 — the hardest systems (A06, A03) have higher T, and the
easiest (A01) has lower T. A02 (highest T) being ranked 3rd in hardness is the main discrepancy.

Spearman ρ(C rank, EER rank) = −0.200 — C is NOT predictive of ASVspoof EER in the expected direction.

---

## Why C Fails to Generalize

The C values across A01–A06 span only **0.28 units** (10.670–10.953), compared to
MLAAD where the range is ~3 units across 63 systems. The six ASVspoof systems cluster
tightly in C-space — there is essentially no signal to exploit.

This happens because A01–A06 are all trained on similar data (LJ Speech / VCTK) with similar
TTS architectures (neural or WORLD vocoder), so their deep WavLM representations have similar
compactness. In MLAAD, the 63 systems span a much wider diversity of TTS architectures and
training corpora, creating the larger C spread that enables the correlation.

**Verdict for C**: The ASVspoof result is **inconclusive**, not contradictory. The signal is too
small relative to noise at N=6 to test directional consistency.

---

## Why T Shows Directional Consistency

T spans only **0.089 units** across A01–A06 (2.712–2.801), but the Spearman rank correlation
ρ=+0.771 is still notable: the hardest systems (A06, A03) happen to have higher T, and the
easiest (A01) has lower T.

However: with N=6, the minimum detectable Spearman ρ at p=0.05 (one-tailed) is approximately
0.83 (Spearman critical value). ρ=+0.771 does NOT reach significance. This is an encouraging trend,
not a confirmed result.

The directional consistency with MLAAD (T↑ → harder in both datasets) is the primary positive signal.

---

## Overall E4 Verdict

| Test | C | T |
|------|---|---|
| Direction consistent with MLAAD? | **NO** (r=−0.380, reversed) | **YES** (r=+0.451, consistent) |
| Spearman ρ > 0.6? | **NO** (ρ=−0.200) | **YES** (ρ=+0.771) |
| Statistically significant (p<0.05)? | **NO** | **NO** |
| Spread sufficient to test? | **NO** (range=0.28) | **NO** (range=0.089) |

**Summary**: Neither C nor T demonstrates statistically confirmed generalization to ASVspoof.
T shows a directionally consistent trend (ρ=+0.771) that is encouraging but not significant given N=6.
C's apparent reversal is likely noise given the tiny spread. The ASVspoof A01–A06 systems are too
homogeneous in their WavLM representations to provide a meaningful test of the C/T framework.

A more informative test would require: (a) more than 6 systems, (b) systems spanning different TTS
architectures, and/or (c) a larger per-system utterance sample to reduce measurement noise.
