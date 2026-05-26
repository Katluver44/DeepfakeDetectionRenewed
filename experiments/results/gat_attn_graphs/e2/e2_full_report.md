# E2: Hub Analysis — Full Report

## Table 1: Hub-identity shift (Spearman + top-k overlap)

| system | top-5 | top-10 | Spearman ρ | p-val |
|--------|-------|--------|------------|-------|
| A01 | 1 | 3 | +0.8866 | 0.00000 |
| A02 | 3 | 7 | +0.8722 | 0.00000 |
| A03 | 2 | 5 | +0.8678 | 0.00000 |
| A04 | 2 | 3 | +0.8518 | 0.00000 |
| A05 | 3 | 4 | +0.5082 | 0.00000 |
| A06 | 3 | 6 | +0.7299 | 0.00000 |

## Table 2: Surviving hub phonemes (Holm-corrected, E2 permutation)

| system | surviving phonemes | n_sig | p_family |
|--------|-------------------|-------|---------|
| A01 | aʊ | 1 | 0.00010 |
| A02 | none | 0 | 0.45555 |
| A03 | none | 0 | 0.00250 |
| A04 | none | 0 | 0.00820 |
| A05 | none | 0 | 0.30577 |
| A06 | none | 0 | 0.51565 |

## Table 3: Gini coefficient of in-degree distribution

| system | Gini | d_in mean | d_in max |
|--------|------|-----------|---------|
| - | 0.4242 | 0.60421 | 4.96447 |
| A01 | 0.2384 | 0.36619 | 1.67361 |
| A02 | 0.3809 | 0.54196 | 3.67438 |
| A03 | 0.2384 | 0.39798 | 1.74609 |
| A04 | 0.2673 | 0.39101 | 1.74853 |
| A05 | 0.5627 | 0.76668 | 5.68841 |
| A06 | 0.5180 | 0.63867 | 4.82300 |

## Interpretation: null systems (A03, A05, A06)

Per-edge analysis at layer 0 found no significant attention redistribution for A03, A05, and A06 after Holm correction. The 3-layer aggregated hub analysis yields a split result:

**A03**: The family-level permutation test (max_p |Δd_in|) is significant at p_family = 0.0025, meaning the maximum hub-mass shift across all phonemes is larger than expected by chance. However, no single phoneme survives individual Holm correction once the family test is decomposed. This pattern — family significant, no individual winner — indicates that A03 produces a diffuse redistribution of hub mass across multiple phonemes simultaneously (with `aʊ`, `ɾ`, and `œ̃` showing the largest raw shifts) rather than concentrating its effect on one identifiable phoneme. This is a new detection relative to layer-0: **A03 becomes detectable at the hub-family level under 3-layer aggregation**, even though it was undetectable per-edge.

**A04**: Similarly, p_family = 0.0082 (significant), with no individual phoneme surviving Holm. Same diffuse-redistribution pattern.

**A05 and A06**: p_family = 0.31 and 0.52 respectively — genuinely null at both the family and individual level. Despite the large raw Δd_in values for `s`/`CN-s` (A05) and `CN-dʒ` (A06), these arise from sparse phoneme pairs (phonemes near-absent in bonafide) and do not survive permutation. These systems do not engage the GAT phoneme-routing pathway in a way detectable by hub-mass analysis at N=50 per class.

The partial positive result for A03 is the key finding: the null at layer-0 is not fully robust under aggregation; it survives only as a diffuse family effect rather than a localised per-phoneme or per-edge effect.