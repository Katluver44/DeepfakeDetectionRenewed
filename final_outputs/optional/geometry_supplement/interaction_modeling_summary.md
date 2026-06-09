# Interaction Modelling Summary

## Setup

- N = 63 systems
- 8 main features: cosine_sim_entropy, cosine_centroid, cosine_rank_vol, mean_gini_mean, knn_entropy_norm, var_entropy_mean, phoneme_var_mean, frame_mean_dist_mean
- 28 pairwise interaction terms (C(8,2))
- Evaluation: Leave-one-out R² throughout

## Overall Result: Do Interactions Help?

| Model | LOO R² (Lasso) | LOO R² (RF) |
|---|---|---|
| Main effects only | −0.062 | +0.194 |
| Main + all pairwise interactions | −0.266 | +0.110 |
| ΔR² from adding interactions | −0.204 | −0.084 |

Interactions **hurt** prediction in both Lasso and RF. The Lasso suffers severe overfitting
(−0.266 vs −0.062). The RF loses 0.084 LOO R² points. Permutation test (n=200): p > 0.70
(observed ΔR² = −0.084 is below the null distribution median; essentially zero probability
of being considered "significant").

**Verdict: Interactions do NOT add meaningful predictive power beyond main effects.**

## Explicitly Requested Interactions (LOO Ridge)

| Interaction | R²_main | R²_+inter | ΔR² | β direction |
|---|---|---|---|---|
| smoothness × cluster entropy | −0.045 | −0.075 | −0.031 | − (98% of folds) |
| cluster entropy × attention | −0.027 | −0.029 | −0.003 | + (100% of folds) |
| smoothness × attention | −0.079 | −0.081 | −0.002 | − (98% of folds) |

All three requested interactions slightly *hurt* prediction. The β directions are highly
consistent across LOO folds (indicating the interactions are not zero), but they have the
wrong sign or magnitude to improve LOO R².

Interpretation: Adding the interaction term `smoothness × cluster entropy` to the main-effects
model actually absorbs variance from the main effects rather than adding new variance, due to
correlation between the interaction term and the main effects (multicollinearity).

## Feature Stability (Lasso Selection Rate)

Features selected in >20% of LOO folds:

| Feature | Type | Sel.% | Direction |
|---|---|---|---|
| cosine_centroid×cosine_rank_vol | interaction | 90.5% | − |
| mean_gini_mean×knn_entropy_norm | interaction | 90.5% | + |
| var_entropy_mean | main | 90.5% | − |
| cosine_sim_entropy×mean_gini_mean | interaction | 88.9% | + |
| var_entropy_mean×frame_mean_dist_mean | interaction | 69.8% | − |
| mean_gini_mean×phoneme_var_mean | interaction | 68.3% | + |
| mean_gini_mean | main | 68.3% | + |
| cosine_centroid×phoneme_var_mean | interaction | 66.7% | + |
| cosine_sim_entropy×cosine_rank_vol | interaction | 66.7% | + |

Several interaction terms are consistently selected by Lasso, but their inclusion does not
improve LOO R². This is a hallmark of multicollinearity: interaction terms are correlated
with their constituent main effects, so the Lasso selects them as compressed representations
of the same variance structure rather than as independent predictors.

## Interpretation

Interactions do NOT add meaningful predictive power beyond main effects. The permutation test
confirms this is consistent with sampling noise. Residual hardness is better described by
additive main effects than by multiplicative interactions among the tested features. The Lasso
stability of certain interaction terms (cosine_sim_entropy×mean_gini_mean at 88.9%) reflects
shared variance between H3 and H2 features, not a genuine synergistic effect.

The most stable main effect is var_entropy_mean (90.5% selection, negative direction), consistent
with prior findings that hard systems have LOWER variance in attention entropy across utterances —
they are more rigidly concentrated in their attention patterns.
