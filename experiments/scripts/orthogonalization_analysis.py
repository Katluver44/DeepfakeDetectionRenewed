#!/usr/bin/env python3
"""
Task 6: Orthogonalization Analysis

Test whether the five confirmed hardness factors are genuinely independent,
partially redundant, or dominated by fewer latent variables.

Five factors (all oriented: higher = harder):
  S: Temporal smoothness  → frame_mean_dist_mean     (note: ~zero residual corr)
  C: Deep compactness     → -rog_L12                 (negated)
  T: Traj irregularity    → vel_entropy_L9
  A: Attention collapse   → mean_gini_mean
  E: Multi-cluster entropy→ cosine_sim_entropy

Primary outcome: residual hardness (observed_eer - ensemble_pred_eer)
Secondary outcome: observed_eer (shows full factor system)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import factorial
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
OUT  = BASE / 'outputs'
FIG  = OUT  / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

FULL_TABLE  = BASE / 'experiments/results/mlaad/residual_hardness_2/full_feature_table.csv'
LAYER_TABLE = OUT  / 'sig_layer_features.csv'

# ─── 1. Load and merge ────────────────────────────────────────────────────────
full  = pd.read_csv(FULL_TABLE)
layer = pd.read_csv(LAYER_TABLE)[['system','rog_L12','vel_entropy_L9']]
df    = full.merge(layer, on='system', how='inner')
print(f"Merged N={len(df)} systems\n")

y     = df['residual'].values          # primary outcome
y_eer = df['observed_eer'].values      # secondary outcome

# ─── 2. Factor definitions ────────────────────────────────────────────────────
# Oriented so that higher = harder (more positive correlation with hardness)
factor_data = {
    'S': df['frame_mean_dist_mean'].values,        # rougher = harder for EER
    'C': -df['rog_L12'].values,                    # compact = harder → negate rog
    'T': df['vel_entropy_L9'].values,              # irregular = harder
    'A': df['mean_gini_mean'].values,              # concentrated = harder
    'E': df['cosine_sim_entropy'].values,          # diverse = harder
}
FACTOR_NAMES   = list(factor_data.keys())
FACTOR_LABELS  = {
    'S': 'Temporal\nSmoothness\n(frame_mean_dist)',
    'C': 'Deep\nCompactness\n(−rog@L12)',
    'T': 'Traj\nIrregularity\n(vel_entropy@L9)',
    'A': 'Attention\nCollapse\n(mean_gini)',
    'E': 'Multi-cluster\nEntropy\n(cosine_sim)',
}
X_raw  = np.column_stack([factor_data[k] for k in FACTOR_NAMES])
N, K   = X_raw.shape
print(f"Factor matrix: {N} × {K}")

# Alternative proxies for robustness checks
ALT_A = df['var_entropy_mean'].values         # lower = harder → will negate
ALT_S = df['phoneme_var_mean'].values         # alternative S proxy

# ─── Utility functions ────────────────────────────────────────────────────────

def ols_r2_subset(X, y, cols):
    if not cols:
        return 0.0
    Xs = StandardScaler().fit_transform(X[:, cols])
    return max(0.0, LinearRegression().fit(Xs, y).score(Xs, y))

def compute_all_r2(X, y):
    """OLS R² for every non-empty subset of columns (2^K − 1 fits)."""
    K = X.shape[1]
    r2 = {frozenset(): 0.0}
    for sz in range(1, K+1):
        for sub in combinations(range(K), sz):
            r2[frozenset(sub)] = ols_r2_subset(X, y, list(sub))
    return r2

def shapley_values(X, y, r2=None):
    """Exact Shapley (= general dominance) values; sum to R²(all K)."""
    K = X.shape[1]
    if r2 is None:
        r2 = compute_all_r2(X, y)
    sv = np.zeros(K)
    for i in range(K):
        others = [j for j in range(K) if j != i]
        for sz in range(K):       # coalition size without i
            for coal in combinations(others, sz):
                S  = frozenset(coal)
                Si = S | {i}
                w  = factorial(sz) * factorial(K - sz - 1) / factorial(K)
                sv[i] += w * (r2[Si] - r2[S])
    return sv

def semi_partial_r2(X, y, r2=None):
    """Unique effect: R²(all) − R²(all \ {i})."""
    K = X.shape[1]
    if r2 is None:
        r2 = compute_all_r2(X, y)
    full   = frozenset(range(K))
    full_r2 = r2[full]
    sp = np.zeros(K)
    for i in range(K):
        sp[i] = full_r2 - r2[full - {i}]
    return sp, full_r2

def loo_r2_ridge(X_feat, y, alpha=1.0):
    """LOO R² with Ridge regression."""
    n  = len(y)
    ps = np.zeros(n)
    for i in range(n):
        idx = list(range(n))
        idx.pop(i)
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(X_feat[idx])
        Xte  = sc.transform(X_feat[[i]])
        ps[i] = Ridge(alpha=alpha).fit(Xtr, y[idx]).predict(Xte)[0]
    ss_res = np.sum((y - ps)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res / ss_tot

def loo_r2_ols(X_feat, y):
    """LOO R² with OLS."""
    n  = len(y)
    ps = np.zeros(n)
    for i in range(n):
        idx = list(range(n))
        idx.pop(i)
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(X_feat[idx])
        Xte  = sc.transform(X_feat[[i]])
        ps[i] = LinearRegression().fit(Xtr, y[idx]).predict(Xte)[0]
    ss_res = np.sum((y - ps)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res / ss_tot

def vif(X):
    """Variance Inflation Factor for each column."""
    K  = X.shape[1]
    Xs = StandardScaler().fit_transform(X)
    vifs = np.zeros(K)
    for i in range(K):
        others = np.delete(Xs, i, axis=1)
        r2_i   = LinearRegression().fit(others, Xs[:, i]).score(others, Xs[:, i])
        vifs[i] = 1.0 / max(1.0 - r2_i, 1e-9)
    return vifs

def partial_corr(X, y):
    """Partial correlation of each column with y, controlling for all others."""
    K   = X.shape[1]
    Xs  = StandardScaler().fit_transform(X)
    pcs = np.zeros(K)
    for i in range(K):
        others = np.delete(Xs, i, axis=1)
        # Residualise y and x_i against others
        ry = y - LinearRegression().fit(others, y).predict(others)
        rx = Xs[:, i] - LinearRegression().fit(others, Xs[:, i]).predict(others)
        pcs[i] = stats.pearsonr(ry, rx)[0]
    return pcs

def residualise_factor(X, col):
    """Residualise factor col against all other factors; return residuals."""
    others = np.delete(X, col, axis=1)
    Xo     = StandardScaler().fit_transform(others)
    xi     = X[:, col]
    xi_hat = LinearRegression().fit(Xo, xi).predict(Xo)
    return xi - xi_hat

def raw_corr(X, y):
    """Pearson r between each factor and y."""
    K = X.shape[1]
    r = np.zeros(K)
    for i in range(K):
        r[i] = stats.pearsonr(X[:, i], y)[0]
    return r

def spearman_corr(X, y):
    K = X.shape[1]
    r = np.zeros(K)
    for i in range(K):
        r[i] = stats.spearmanr(X[:, i], y)[0]
    return r

# ─── 3. Factor intercorrelations ──────────────────────────────────────────────
print("=" * 70)
print("SECTION 3: FACTOR INTERCORRELATIONS")
print("=" * 70)

Xs = StandardScaler().fit_transform(X_raw)
corr_mat = np.corrcoef(Xs.T)

print("\nPearson correlation matrix (raw orientation):")
hdr = "        " + "  ".join(f"{n:>8}" for n in FACTOR_NAMES)
print(hdr)
for i, ni in enumerate(FACTOR_NAMES):
    row = f"{ni:>8}" + "  ".join(f"{corr_mat[i,j]:>8.3f}" for j in range(K))
    print(row)

vifs = vif(X_raw)
print("\nVIF per factor:")
for i, n in enumerate(FACTOR_NAMES):
    print(f"  {n}: VIF = {vifs[i]:.2f}")

# Save factor correlation matrix
corr_df = pd.DataFrame(corr_mat, index=FACTOR_NAMES, columns=FACTOR_NAMES)

# ─── 4. Raw & partial correlations with hardness ─────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: RAW AND PARTIAL CORRELATIONS WITH RESIDUAL HARDNESS")
print("=" * 70)

raw_r     = raw_corr(X_raw, y)
raw_rho   = spearman_corr(X_raw, y)
part_r    = partial_corr(X_raw, y)

# Residualised correlations
resid_r   = np.zeros(K)
for i in range(K):
    res_i    = residualise_factor(X_raw, i)
    resid_r[i] = stats.pearsonr(res_i, y)[0]

print(f"\n{'Factor':<10} {'Raw r':>8} {'Raw ρ':>8} {'Partial r':>10} {'Resid r':>9} {'p(raw)':>8}")
print("-" * 60)
for i, n in enumerate(FACTOR_NAMES):
    p_raw = stats.pearsonr(X_raw[:,i], y)[1]
    print(f"{n:<10} {raw_r[i]:>8.3f} {raw_rho[i]:>8.3f} {part_r[i]:>10.3f} {resid_r[i]:>9.3f} {p_raw:>8.4f}")

# ─── 5. Variance partitioning (OLS) ──────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: VARIANCE PARTITIONING (OLS, both outcomes)")
print("=" * 70)

# For residual hardness (y)
r2_resid = compute_all_r2(X_raw, y)
sp_resid, full_resid = semi_partial_r2(X_raw, y, r2_resid)
sv_resid  = shapley_values(X_raw, y, r2_resid)

# For observed EER (y_eer)
r2_eer_d = compute_all_r2(X_raw, y_eer)
sp_eer, full_eer = semi_partial_r2(X_raw, y_eer, r2_eer_d)
sv_eer    = shapley_values(X_raw, y_eer, r2_eer_d)

raw_r2_resid = np.array([r2_resid[frozenset([i])] for i in range(K)])
raw_r2_eer   = np.array([r2_eer_d[frozenset([i])] for i in range(K)])

print("\n--- Residual hardness (primary outcome) ---")
print(f"R²(all 5 factors) = {full_resid:.4f}")
print(f"\n{'Factor':<10} {'Raw R²':>8} {'Unique R²':>10} {'Shapley':>10} {'Shared':>9}")
print("-" * 55)
for i, n in enumerate(FACTOR_NAMES):
    shared = raw_r2_resid[i] - sp_resid[i]
    print(f"{n:<10} {raw_r2_resid[i]:>8.4f} {sp_resid[i]:>10.4f} {sv_resid[i]:>10.4f} {shared:>9.4f}")
print(f"{'TOTAL':<10} {'':>8} {sp_resid.sum():>10.4f} {sv_resid.sum():>10.4f}")

print("\n--- Observed EER (secondary outcome) ---")
print(f"R²(all 5 factors) = {full_eer:.4f}")
print(f"\n{'Factor':<10} {'Raw R²':>8} {'Unique R²':>10} {'Shapley':>10} {'Shared':>9}")
print("-" * 55)
for i, n in enumerate(FACTOR_NAMES):
    shared = raw_r2_eer[i] - sp_eer[i]
    print(f"{n:<10} {raw_r2_eer[i]:>8.4f} {sp_eer[i]:>10.4f} {sv_eer[i]:>10.4f} {shared:>9.4f}")
print(f"{'TOTAL':<10} {'':>8} {sp_eer.sum():>10.4f} {sv_eer.sum():>10.4f}")

# ─── 6. Cross-validated unique effects (LOO) ─────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: LOO R² — UNIQUE EFFECTS (CROSS-VALIDATED)")
print("=" * 70)

full_loo_resid = loo_r2_ridge(X_raw, y)
full_loo_eer   = loo_r2_ridge(X_raw, y_eer)

sp_loo_resid = np.zeros(K)
sp_loo_eer   = np.zeros(K)
for i in range(K):
    cols_excl = [j for j in range(K) if j != i]
    sp_loo_resid[i] = full_loo_resid - loo_r2_ridge(X_raw[:, cols_excl], y)
    sp_loo_eer[i]   = full_loo_eer   - loo_r2_ridge(X_raw[:, cols_excl], y_eer)

print(f"\nFull model LOO R² (residual)  = {full_loo_resid:.4f}")
print(f"Full model LOO R² (EER)       = {full_loo_eer:.4f}")
print(f"\n{'Factor':<10} {'LOO unique (resid)':>18} {'LOO unique (EER)':>18}")
print("-" * 50)
for i, n in enumerate(FACTOR_NAMES):
    print(f"{n:<10} {sp_loo_resid[i]:>18.4f} {sp_loo_eer[i]:>18.4f}")

# ─── 7. Permutation tests for unique effects ──────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: PERMUTATION TESTS FOR UNIQUE EFFECTS (n=1000)")
print("=" * 70)

N_PERM = 1000
perm_sp = np.zeros((N_PERM, K))
full_all = frozenset(range(K))

# Use OLS for permutation (faster than ridge)
for perm_idx in range(N_PERM):
    yp = np.random.permutation(y)
    r2p = compute_all_r2(X_raw, yp)
    fr2p = r2p[full_all]
    for i in range(K):
        perm_sp[perm_idx, i] = fr2p - r2p[full_all - {i}]

perm_pvals = np.zeros(K)
for i in range(K):
    perm_pvals[i] = np.mean(perm_sp[:, i] >= sp_resid[i])

print(f"\n{'Factor':<10} {'Observed':>10} {'Perm mean':>10} {'Perm SD':>10} {'p-value':>10}")
print("-" * 55)
for i, n in enumerate(FACTOR_NAMES):
    print(f"{n:<10} {sp_resid[i]:>10.4f} {perm_sp[:,i].mean():>10.4f} "
          f"{perm_sp[:,i].std():>10.4f} {perm_pvals[i]:>10.4f}")

# ─── 8. PCA / Factor Collapse Test ───────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: PCA — FACTOR COLLAPSE TEST")
print("=" * 70)

pca   = PCA()
Xstd  = StandardScaler().fit_transform(X_raw)
pca.fit(Xstd)

eigenvalues = pca.explained_variance_
explained   = pca.explained_variance_ratio_
loadings    = pca.components_.T   # (K, K): factor × component

print("\nEigenvalues (correlation matrix scale):")
print("  PC" + "  ".join(f"{i+1:>8}" for i in range(K)))
print("  EV" + "  ".join(f"{eigenvalues[i]:>8.3f}" for i in range(K)))
print("  %V" + "  ".join(f"{100*explained[i]:>8.1f}" for i in range(K)))
print(f"  Cum  " + "  ".join(f"{100*explained[:i+1].sum():>7.1f}%" for i in range(K)))

print("\nLoadings matrix (rows=factors, cols=PCs):")
hdr2 = f"{'':>8}" + "".join(f"  PC{i+1:>3}" for i in range(K))
print(hdr2)
for j, n in enumerate(FACTOR_NAMES):
    row = f"{n:>8}" + "".join(f"  {loadings[j,i]:>6.3f}" for i in range(K))
    print(row)

# Count PCs with eigenvalue > 1 (Kaiser criterion)
n_kaiser = np.sum(eigenvalues > 1)
print(f"\nKaiser criterion (λ>1): {n_kaiser} components")

# PC scores vs. hardness
scores  = pca.transform(Xstd)
pc_corr_resid = [stats.pearsonr(scores[:, i], y) for i in range(K)]
pc_corr_eer   = [stats.pearsonr(scores[:, i], y_eer) for i in range(K)]

print("\nPC scores vs. outcomes:")
print(f"{'':>6}  {'r(resid)':>10}  {'p':>8}  {'r(EER)':>10}  {'p':>8}")
for i in range(K):
    print(f"PC{i+1:<4} {pc_corr_resid[i][0]:>10.3f}  {pc_corr_resid[i][1]:>8.4f}"
          f"  {pc_corr_eer[i][0]:>10.3f}  {pc_corr_eer[i][1]:>8.4f}")

# ─── 9. Robustness checks ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9: ROBUSTNESS CHECKS")
print("=" * 70)

# 9a. Outlier removal (top 5 by |residual|)
idx_sorted = np.argsort(np.abs(y))
idx_trim   = idx_sorted[:-5]
X_trim     = X_raw[idx_trim]
y_trim     = y[idx_trim]
print(f"\n9a. Outlier removal (N={len(idx_trim)} after removing 5 most extreme):")
r2_trim   = compute_all_r2(X_trim, y_trim)
sv_trim   = shapley_values(X_trim, y_trim, r2_trim)
sp_trim, full_trim = semi_partial_r2(X_trim, y_trim, r2_trim)
print(f"R²(all 5) trimmed = {full_trim:.4f}")
for i, n in enumerate(FACTOR_NAMES):
    print(f"  {n}: Shapley = {sv_trim[i]:.4f}  Unique = {sp_trim[i]:.4f}")

# 9b. Alternative factor proxies (swap A → neg var_entropy; S → phoneme_var_mean)
print("\n9b. Alternative proxies (A=neg_var_entropy, S=phoneme_var_mean):")
X_alt = X_raw.copy()
X_alt[:, 0] = ALT_S          # alt S
X_alt[:, 3] = -ALT_A         # alt A (negated so higher = harder)
r2_alt  = compute_all_r2(X_alt, y)
sv_alt  = shapley_values(X_alt, y, r2_alt)
sp_alt, full_alt = semi_partial_r2(X_alt, y, r2_alt)
print(f"R²(all 5) alt = {full_alt:.4f}")
for i, n in enumerate(FACTOR_NAMES):
    print(f"  {n}: Shapley = {sv_alt[i]:.4f}  Unique = {sp_alt[i]:.4f}")

# 9c. LOO Shapley (cross-validated Shapley values)
print("\n9c. LOO-CV Shapley values (each fold: N-1 samples):")
sv_loo_list = []
for fold in range(N):
    mask = [j for j in range(N) if j != fold]
    r2_f  = compute_all_r2(X_raw[mask], y[mask])
    sv_f  = shapley_values(X_raw[mask], y[mask], r2_f)
    sv_loo_list.append(sv_f)
sv_loo_arr  = np.array(sv_loo_list)
sv_loo_mean = sv_loo_arr.mean(0)
sv_loo_se   = sv_loo_arr.std(0) / np.sqrt(N)
print(f"{'Factor':<10} {'LOO Shapley':>12} {'SE':>8}")
for i, n in enumerate(FACTOR_NAMES):
    print(f"{n:<10} {sv_loo_mean[i]:>12.4f} {sv_loo_se[i]:>8.4f}")

# ─── 10. FIGURES ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10: GENERATING FIGURES")
print("=" * 70)

short_names = ['S\nSmooth', 'C\nCompact', 'T\nTraj', 'A\nAttn', 'E\nEntropy']

# Fig 1: Factor intercorrelation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(K)); ax.set_xticklabels(short_names, fontsize=9)
ax.set_yticks(range(K)); ax.set_yticklabels(short_names, fontsize=9)
for i in range(K):
    for j in range(K):
        v = corr_mat[i, j]
        clr = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color=clr)
ax.set_title('Factor Intercorrelation Matrix', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'orthog_factor_correlations.png', dpi=150)
plt.close()
print("  Saved: orthog_factor_correlations.png")

# Fig 2: Variance partitioning (Shapley, residual outcome)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (sv, full_r2, yrng, title) in zip(
        axes,
        [(sv_resid, full_resid, y, 'Residual Hardness'),
         (sv_eer,   full_eer,   y_eer, 'Observed EER')]):
    colors = ['#4878D0','#EE854A','#6ACC65','#D65F5F','#B47CC7']
    bars = ax.bar(short_names, sv, color=colors, edgecolor='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Shapley value (portion of R²)')
    ax.set_title(f'Shapley Variance Partition\nOutcome: {title}\n(Total R²={full_r2:.3f})', fontsize=10)
    for b, v in zip(bars, sv):
        if abs(v) > 0.001:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(FIG / 'orthog_variance_partition.png', dpi=150)
plt.close()
print("  Saved: orthog_variance_partition.png")

# Fig 3: Raw vs. partial vs. residualized correlations
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(K)
width = 0.28
ax.bar(x_pos - width, raw_r,    width, label='Raw r', color='#4878D0', alpha=0.85)
ax.bar(x_pos,         part_r,   width, label='Partial r', color='#EE854A', alpha=0.85)
ax.bar(x_pos + width, resid_r,  width, label='Residualised r', color='#6ACC65', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Pearson r with residual hardness')
ax.set_title('Raw / Partial / Residualised Correlations with Residual Hardness', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / 'orthog_partial_correlations.png', dpi=150)
plt.close()
print("  Saved: orthog_partial_correlations.png")

# Fig 4: PCA scree + loadings heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Scree
ax1.bar(range(1, K+1), eigenvalues, color='#4878D0', alpha=0.8, edgecolor='black')
ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='λ=1 (Kaiser)')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue')
ax1.set_title(f'Scree Plot\n({n_kaiser} PC(s) with λ>1)', fontsize=11)
ax1.legend()
ax1_twin = ax1.twinx()
ax1_twin.plot(range(1, K+1), np.cumsum(explained)*100, 'o-', color='#D65F5F', linewidth=1.5)
ax1_twin.axhline(80, color='grey', linestyle=':', linewidth=1)
ax1_twin.set_ylabel('Cumulative variance (%)', color='#D65F5F')
ax1_twin.tick_params(axis='y', colors='#D65F5F')

# Loadings heatmap
im2 = ax2.imshow(loadings.T, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im2, ax=ax2, label='Loading')
ax2.set_xticks(range(K)); ax2.set_xticklabels(short_names, fontsize=9)
ax2.set_yticks(range(K)); ax2.set_yticklabels([f'PC{i+1}\n({100*explained[i]:.0f}%)' for i in range(K)], fontsize=9)
for i in range(K):
    for j in range(K):
        v = loadings[j, i]
        clr = 'white' if abs(v) > 0.6 else 'black'
        ax2.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=clr)
ax2.set_title('PCA Loadings Matrix', fontsize=11)
plt.tight_layout()
plt.savefig(FIG / 'orthog_pca.png', dpi=150)
plt.close()
print("  Saved: orthog_pca.png")

# Fig 5: LOO Shapley with error bars
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#4878D0','#EE854A','#6ACC65','#D65F5F','#B47CC7']
ax.bar(short_names, sv_loo_mean, yerr=2*sv_loo_se, color=colors,
       edgecolor='black', linewidth=0.7, capsize=4)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('LOO Shapley value')
ax.set_title('Cross-validated (LOO) Shapley Values\n(± 2 SE)', fontsize=11)
plt.tight_layout()
plt.savefig(FIG / 'orthog_loo_shapley.png', dpi=150)
plt.close()
print("  Saved: orthog_loo_shapley.png")

# ─── 11. Save CSV outputs ─────────────────────────────────────────────────────

# 11a. variance_partition.csv
vp_rows = []
for i, n in enumerate(FACTOR_NAMES):
    vp_rows.append({
        'factor': n,
        'raw_r_resid':       raw_r[i],
        'raw_rho_resid':     raw_rho[i],
        'partial_r_resid':   part_r[i],
        'residualised_r':    resid_r[i],
        'raw_r2_resid':      raw_r2_resid[i],
        'unique_r2_resid':   sp_resid[i],
        'shapley_resid':     sv_resid[i],
        'shapley_loo_resid': sv_loo_mean[i],
        'shapley_loo_se':    sv_loo_se[i],
        'perm_pval':         perm_pvals[i],
        'vif':               vifs[i],
        'raw_r2_eer':        raw_r2_eer[i],
        'unique_r2_eer':     sp_eer[i],
        'shapley_eer':       sv_eer[i],
        'loo_unique_resid':  sp_loo_resid[i],
        'loo_unique_eer':    sp_loo_eer[i],
    })
vp_df = pd.DataFrame(vp_rows)
vp_df.to_csv(OUT / 'variance_partition.csv', index=False)
print(f"\nSaved: variance_partition.csv")

# 11b. factor_collapse_results.csv
fc_rows = []
for i in range(K):
    row = {'component': f'PC{i+1}', 'eigenvalue': eigenvalues[i],
           'variance_explained': explained[i],
           'cumulative_variance': explained[:i+1].sum(),
           'r_with_residual': pc_corr_resid[i][0],
           'p_residual': pc_corr_resid[i][1],
           'r_with_eer': pc_corr_eer[i][0],
           'p_eer': pc_corr_eer[i][1]}
    for j, n in enumerate(FACTOR_NAMES):
        row[f'loading_{n}'] = loadings[j, i]
    fc_rows.append(row)
fc_df = pd.DataFrame(fc_rows)
fc_df.to_csv(OUT / 'factor_collapse_results.csv', index=False)
print("Saved: factor_collapse_results.csv")

# ─── 12. orthogonalization_summary.md ─────────────────────────────────────────
summary_lines = [
    "# Orthogonalization Summary\n",
    f"N = {N} systems  |  K = {K} factors  |  1000 permutations\n",
    "\n## Factor Definitions\n",
    "| Code | Feature | Direction |",
    "|------|---------|-----------|",
    "| S | frame_mean_dist_mean | rougher = harder |",
    "| C | −rog_L12 | compact = harder |",
    "| T | vel_entropy_L9 | irregular = harder |",
    "| A | mean_gini_mean | concentrated = harder |",
    "| E | cosine_sim_entropy | diverse = harder |",
    "\n## Factor Intercorrelations\n",
    "```",
    hdr,
]
for i, ni in enumerate(FACTOR_NAMES):
    row = f"{ni:>8}" + "  ".join(f"{corr_mat[i,j]:>8.3f}" for j in range(K))
    summary_lines.append(row)
summary_lines += [
    "```",
    f"\nVIF: " + ", ".join(f"{n}={vifs[i]:.2f}" for i, n in enumerate(FACTOR_NAMES)),
    "\n## Variance Partitioning (Residual Hardness, OLS)\n",
    f"R²(all 5 factors) = {full_resid:.4f}\n",
    "| Factor | Raw R² | Unique R² | Shapley | LOO Shapley | Perm p |",
    "|--------|--------|-----------|---------|-------------|--------|",
]
for i, n in enumerate(FACTOR_NAMES):
    summary_lines.append(
        f"| {n} | {raw_r2_resid[i]:.4f} | {sp_resid[i]:.4f} | "
        f"{sv_resid[i]:.4f} | {sv_loo_mean[i]:.4f} ± {sv_loo_se[i]:.4f} | "
        f"{perm_pvals[i]:.3f} |"
    )
summary_lines += [
    "\n## Variance Partitioning (Observed EER, OLS)\n",
    f"R²(all 5 factors) = {full_eer:.4f}\n",
    "| Factor | Raw R² | Unique R² | Shapley |",
    "|--------|--------|-----------|---------|",
]
for i, n in enumerate(FACTOR_NAMES):
    summary_lines.append(
        f"| {n} | {raw_r2_eer[i]:.4f} | {sp_eer[i]:.4f} | {sv_eer[i]:.4f} |"
    )
summary_lines += [
    "\n## LOO R² (Cross-validated)\n",
    f"Full model LOO R² (residual) = {full_loo_resid:.4f}",
    f"Full model LOO R² (EER)      = {full_loo_eer:.4f}\n",
    "| Factor | LOO unique (resid) | LOO unique (EER) |",
    "|--------|-------------------|-----------------|",
]
for i, n in enumerate(FACTOR_NAMES):
    summary_lines.append(
        f"| {n} | {sp_loo_resid[i]:.4f} | {sp_loo_eer[i]:.4f} |"
    )
summary_lines += [
    "\n## PCA Factor Collapse Test\n",
    "| PC | Eigenvalue | Var % | Cum % | r(resid) | p | r(EER) | p |",
    "|----|-----------|-------|-------|---------|---|-------|---|",
]
for i in range(K):
    summary_lines.append(
        f"| PC{i+1} | {eigenvalues[i]:.3f} | {100*explained[i]:.1f}% | "
        f"{100*explained[:i+1].sum():.1f}% | "
        f"{pc_corr_resid[i][0]:.3f} | {pc_corr_resid[i][1]:.4f} | "
        f"{pc_corr_eer[i][0]:.3f} | {pc_corr_eer[i][1]:.4f} |"
    )
summary_lines += [
    f"\nKaiser criterion (λ>1): **{n_kaiser} component(s)**",
    "\n## Robustness Checks\n",
    f"Outlier removal (N={len(idx_trim)}) R²(all 5) = {full_trim:.4f}",
    f"Alt proxies (neg_var_entropy, phoneme_var) R²(all 5) = {full_alt:.4f}",
]
with open(OUT / 'orthogonalization_summary.md', 'w') as f:
    f.write("\n".join(summary_lines) + "\n")
print("Saved: orthogonalization_summary.md")

print("\n" + "=" * 70)
print("ALL SECTIONS COMPLETE")
print("=" * 70)
