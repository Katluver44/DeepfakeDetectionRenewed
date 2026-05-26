# E6: Seed-3 Ranking Replication — robust_goat_seed3
n_perm=10000  seed=42

## Per-system family p-values

| system | seed 1 | seed 2 | seed 3 | category flip? |
|--------|--------|--------|--------|----------------|
| A01 | 0.00010 | 0.00010 | 0.00010 | no |
| A02 | 0.45555 | 0.56594 | 0.65803 | no |
| A03 | 0.00250 | 0.00350 | 0.00410 | no |
| A04 | 0.00820 | 0.00910 | 0.01010 | no |
| A05 | 0.30577 | 0.32497 | 0.34767 | no |
| A06 | 0.51565 | 0.58214 | 0.57704 | no |

## Pairwise Spearman ρ matrix

| pair       | ρ       |
|------------|---------|
| seed3 vs seed1 | +0.9429 |
| seed3 vs seed2 | +0.9429 |
| seed1 vs seed2 | +1.0000 |

## Hypothesis outcomes

- **H1**: PASS
- **H2**: PASS
- **H3**: PASS

**Decision: H1 + H2 PASS → ranking-invariance is locked across three independent training runs.  Write the paper.**