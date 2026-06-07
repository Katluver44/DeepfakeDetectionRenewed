# Orthogonalization Summary

N = 63 systems  |  K = 5 factors  |  1000 permutations


## Factor Definitions

| Code | Feature | Direction |
|------|---------|-----------|
| S | frame_mean_dist_mean | rougher = harder |
| C | −rog_L12 | compact = harder |
| T | vel_entropy_L9 | irregular = harder |
| A | mean_gini_mean | concentrated = harder |
| E | cosine_sim_entropy | diverse = harder |

## Factor Intercorrelations

```
               S         C         T         A         E
       S   1.000    -0.250    -0.198     0.052    -0.528
       C  -0.250     1.000     0.114     0.015     0.300
       T  -0.198     0.114     1.000    -0.109     0.204
       A   0.052     0.015    -0.109     1.000    -0.108
       E  -0.528     0.300     0.204    -0.108     1.000
```

VIF: S=1.42, C=1.12, T=1.07, A=1.02, E=1.48

## Variance Partitioning (Residual Hardness, OLS)

R²(all 5 factors) = 0.2316

| Factor | Raw R² | Unique R² | Shapley | LOO Shapley | Perm p |
|--------|--------|-----------|---------|-------------|--------|
| S | 0.0038 | 0.0181 | 0.0083 | 0.0085 ± 0.0002 | 0.288 |
| C | 0.1079 | 0.0663 | 0.0845 | 0.0846 ± 0.0011 | 0.037 |
| T | 0.0763 | 0.0604 | 0.0657 | 0.0658 ± 0.0009 | 0.043 |
| A | 0.0239 | 0.0363 | 0.0310 | 0.0314 ± 0.0009 | 0.132 |
| E | 0.0588 | 0.0312 | 0.0421 | 0.0423 ± 0.0007 | 0.148 |

## Variance Partitioning (Observed EER, OLS)

R²(all 5 factors) = 0.3446

| Factor | Raw R² | Unique R² | Shapley |
|--------|--------|-----------|---------|
| S | 0.1355 | 0.0118 | 0.0622 |
| C | 0.1415 | 0.0485 | 0.0853 |
| T | 0.0930 | 0.0434 | 0.0614 |
| A | 0.0143 | 0.0304 | 0.0239 |
| E | 0.1945 | 0.0523 | 0.1118 |

## LOO R² (Cross-validated)

Full model LOO R² (residual) = 0.0484
Full model LOO R² (EER)      = 0.1910

| Factor | LOO unique (resid) | LOO unique (EER) |
|--------|-------------------|-----------------|
| S | -0.0128 | -0.0129 |
| C | 0.0557 | 0.0244 |
| T | 0.0374 | 0.0173 |
| A | -0.0105 | -0.0080 |
| E | -0.0069 | 0.0283 |

## PCA Factor Collapse Test

| PC | Eigenvalue | Var % | Cum % | r(resid) | p | r(EER) | p |
|----|-----------|-------|-------|---------|---|-------|---|
| PC1 | 1.894 | 37.3% | 37.3% | -0.286 | 0.0233 | -0.527 | 0.0000 |
| PC2 | 1.058 | 20.8% | 58.1% | -0.165 | 0.1950 | -0.188 | 0.1401 |
| PC3 | 0.867 | 17.1% | 75.2% | 0.238 | 0.0603 | 0.158 | 0.2169 |
| PC4 | 0.790 | 15.6% | 90.7% | 0.189 | 0.1387 | 0.034 | 0.7939 |
| PC5 | 0.472 | 9.3% | 100.0% | -0.174 | 0.1718 | -0.075 | 0.5575 |

Kaiser criterion (λ>1): **2 component(s)**

## Robustness Checks

Outlier removal (N=58) R²(all 5) = 0.1732
Alt proxies (neg_var_entropy, phoneme_var) R²(all 5) = 0.2169
