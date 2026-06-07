"""
65% Analysis — Part 1: Interaction Modelling
=============================================
Tests whether pairwise interactions among the top predictors of residual
hardness add unique predictive power beyond main effects.

Explicitly tests the three requested interactions:
  A) smoothness × cluster entropy   (phoneme_var_mean × cosine_sim_entropy)
  B) cluster entropy × attention     (cosine_sim_entropy × mean_gini_mean)
  C) smoothness × attention          (phoneme_var_mean × mean_gini_mean)

Also runs a full interaction Lasso across all C(8,2)=28 pairwise terms.
Evaluation: LOO R² throughout.  Results are saved to outputs/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE    = Path("/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed")
OUT     = BASE / "outputs"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "experiments/results/mlaad/residual_hardness_2/full_feature_table.csv")
print(f"Loaded {len(df)} systems with {len(df.columns)} columns")

# ── 1. Select top 8 features ──────────────────────────────────────────────────
# Chosen by |Pearson r| or |Spearman ρ| from prior analysis, ensuring
# representation across all four hypothesis groups.
TOP_FEATS = [
    "cosine_sim_entropy",   # H3 — strongest: ρ=+0.332**
    "cosine_centroid",      # H4 — r=−0.253*
    "cosine_rank_vol",      # H1 — ρ=−0.228†
    "mean_gini_mean",       # H2 — ρ=+0.224†
    "knn_entropy_norm",     # H1 — r=+0.219†
    "var_entropy_mean",     # H2 — r=−0.202
    "phoneme_var_mean",     # BASE temporal
    "frame_mean_dist_mean", # BASE temporal
]

# Verify all features exist and are non-null
TOP_FEATS = [f for f in TOP_FEATS if f in df.columns and df[f].notna().sum() >= 55]
print(f"Using {len(TOP_FEATS)} top features: {TOP_FEATS}")

df_clean = df.dropna(subset=TOP_FEATS + ["residual"]).copy().reset_index(drop=True)
N = len(df_clean)
y = df_clean["residual"].values
print(f"Clean dataset: N={N}")


# ── 2. LOO helper functions ────────────────────────────────────────────────────
def loo_r2(X: np.ndarray, y: np.ndarray, model_fn) -> float:
    preds = np.empty(N)
    for i in range(N):
        tr = np.delete(np.arange(N), i)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        m = model_fn()
        m.fit(Xtr, y[tr])
        preds[i] = m.predict(Xte)[0]
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

def ridge_fn():  return Ridge(alpha=1.0)
def rf_fn():
    return RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                 min_samples_leaf=3, random_state=42)


# ── 3. Explicit individual interaction tests ──────────────────────────────────
REQUESTED = [
    ("smoothness × cluster entropy",   "phoneme_var_mean",     "cosine_sim_entropy"),
    ("cluster entropy × attention",    "cosine_sim_entropy",   "mean_gini_mean"),
    ("smoothness × attention",         "phoneme_var_mean",     "mean_gini_mean"),
]

print("\n" + "="*75)
print("EXPLICIT INTERACTION TESTS (LOO Ridge: main effects vs. +interaction)")
print("="*75)
print(f"\n{'Interaction':<35} {'R²_main':>9} {'R²_+inter':>11} {'ΔR²':>8} {'β consistency':>15}")
print("─"*80)

explicit_rows = []

for label, f1, f2 in REQUESTED:
    if f1 not in df_clean.columns or f2 not in df_clean.columns:
        print(f"  {label}: feature missing, skip")
        continue

    x1 = df_clean[f1].values
    x2 = df_clean[f2].values

    X_main  = np.column_stack([x1, x2])
    X_inter = np.column_stack([x1, x2, x1 * x2])

    r2_main  = loo_r2(X_main,  y, ridge_fn)
    r2_inter = loo_r2(X_inter, y, ridge_fn)

    # Stability of interaction coefficient across LOO folds
    inter_coefs = []
    for i in range(N):
        tr = np.delete(np.arange(N), i)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_inter[tr])
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr, y[tr])
        inter_coefs.append(ridge.coef_[2])  # interaction coef

    inter_coefs = np.array(inter_coefs)
    pct_pos   = float(np.mean(inter_coefs > 0))
    beta_mean = float(np.mean(inter_coefs))
    direction = "+" if pct_pos > 0.5 else "−"
    consistency = f"{direction} ({max(pct_pos, 1-pct_pos)*100:.0f}%)"

    print(f"  {label:<35} {r2_main:>+8.3f}  {r2_inter:>+9.3f}  {(r2_inter-r2_main):>+8.3f}  {consistency}")
    explicit_rows.append(dict(interaction=label, f1=f1, f2=f2,
                              r2_main=r2_main, r2_inter=r2_inter,
                              delta_r2=r2_inter - r2_main,
                              beta_mean=beta_mean, pct_positive=pct_pos))

explicit_df = pd.DataFrame(explicit_rows)


# ── 4. Full interaction Lasso ─────────────────────────────────────────────────
# Generate standardised main effects and all pairwise interaction terms.
# LassoCV (inner 5-fold) selects alpha within each LOO fold.

inter_pairs = list(combinations(range(len(TOP_FEATS)), 2))
inter_labels = [f"{TOP_FEATS[i]}×{TOP_FEATS[j]}" for i, j in inter_pairs]
print(f"\n{len(inter_pairs)} pairwise interactions generated")

X_main_raw = df_clean[TOP_FEATS].values  # (N, 8)

def lasso_loo_with_stability(X_full, y, n_cv=5):
    """
    Returns: (preds, coef_positive_fraction) where coef_positive_fraction
    has shape (n_features,) = fraction of LOO folds with positive coef.
    """
    preds = np.empty(len(y))
    coef_mat = np.zeros((len(y), X_full.shape[1]))

    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_full[tr])
        Xte = sc.transform(X_full[[i]])
        las = LassoCV(cv=n_cv, max_iter=10000, random_state=42)
        las.fit(Xtr, y[tr])
        preds[i] = las.predict(Xte)[0]
        coef_mat[i] = las.coef_

    return preds, coef_mat

print("Running full interaction Lasso (LOO) — this may take 2-3 min …")

# Main effects only (Lasso)
preds_main_lasso, coef_main = lasso_loo_with_stability(X_main_raw, y)
r2_main_lasso = float(1 - np.sum((y - preds_main_lasso)**2) / np.sum((y - y.mean())**2))

# Compute interaction features (raw, unstandardised — Lasso normalises internally)
X_inter_cols = []
for i, j in inter_pairs:
    X_inter_cols.append(X_main_raw[:, i] * X_main_raw[:, j])
X_inter_raw = np.column_stack(X_inter_cols)
X_full_raw  = np.hstack([X_main_raw, X_inter_raw])

preds_full_lasso, coef_full = lasso_loo_with_stability(X_full_raw, y)
r2_full_lasso = float(1 - np.sum((y - preds_full_lasso)**2) / np.sum((y - y.mean())**2))

print(f"\nLasso LOO R²: main-only={r2_main_lasso:+.3f}  main+interactions={r2_full_lasso:+.3f}  "
      f"Δ={r2_full_lasso - r2_main_lasso:+.3f}")

# RF comparison
r2_main_rf = loo_r2(X_main_raw, y, rf_fn)
r2_full_rf = loo_r2(X_full_raw, y, rf_fn)
print(f"RF   LOO R²: main-only={r2_main_rf:+.3f}  main+interactions={r2_full_rf:+.3f}  "
      f"Δ={r2_full_rf - r2_main_rf:+.3f}")

# ── 5. Feature stability analysis ────────────────────────────────────────────
# Fraction of LOO folds with non-zero coefficient (for Lasso — measures stability)
all_labels = TOP_FEATS + inter_labels
feat_stability = (np.abs(coef_full) > 1e-8).mean(axis=0)  # shape (36,)
feat_sign_pos  = (coef_full > 0).mean(axis=0)

stability_df = pd.DataFrame({
    "feature": all_labels,
    "is_interaction": ["×" in l for l in all_labels],
    "selection_rate": feat_stability,
    "pct_positive": feat_sign_pos,
    "mean_coef": coef_full.mean(axis=0),
}).sort_values("selection_rate", ascending=False)

print("\n" + "="*75)
print("LASSO FEATURE STABILITY (fraction of LOO folds with non-zero coef)")
print("="*75)
print(f"\n  {'Feature':<40} {'Sel.%':>7} {'Dir.':>6} {'Mean β':>10}")
print("  " + "─"*65)
for _, row in stability_df.iterrows():
    tag = " [inter]" if row["is_interaction"] else "        "
    if row["selection_rate"] > 0.05:  # show only features selected in >5% of folds
        direction = "+" if row["pct_positive"] > 0.5 else "−"
        print(f"  {row['feature']:<40}{tag}  {row['selection_rate']*100:>5.1f}%  "
              f"{direction}   {row['mean_coef']:>+9.4f}")

print(f"\n  (Features with <5% selection rate omitted — "
      f"{(stability_df['selection_rate'] <= 0.05).sum()} features)")


# ── 6. Permutation test for interaction importance ────────────────────────────
print("\nRunning permutation test for interaction block significance (n_perm=500) …")

# Observed: how much do interactions improve over main effects?
obs_delta_rf = r2_full_rf - r2_main_rf

perm_deltas = []
rng = np.random.RandomState(0)
for _ in range(500):
    y_perm = rng.permutation(y)
    # permute just within the delta: shuffle y, recompute RF with and without
    r2_m = loo_r2(X_main_raw, y_perm, rf_fn)
    r2_f = loo_r2(X_full_raw, y_perm, rf_fn)
    perm_deltas.append(r2_f - r2_m)

perm_deltas = np.array(perm_deltas)
perm_p = float(np.mean(perm_deltas >= obs_delta_rf))
print(f"  Observed ΔR²(RF) from adding interactions: {obs_delta_rf:+.3f}")
print(f"  Permutation p-value: {perm_p:.3f}")


# ── 7. Figures ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Interaction Modelling: LOO Evaluation", fontsize=13, fontweight="bold")

# Panel A: Explicit interaction ΔR²
ax = axes[0]
labels = [r["interaction"].replace("×", "\n×\n") for r in explicit_rows]
deltas = [r["delta_r2"] for r in explicit_rows]
colors = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]
bars = ax.barh(range(len(labels)), deltas, color=colors, alpha=0.8, height=0.55)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("ΔR² (LOO Ridge: +interaction vs. main effects)", fontsize=10)
ax.set_title("A — Explicit Interaction ΔR²", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Panel B: Lasso stability plot (top interactions)
ax = axes[1]
top30 = stability_df.head(30)
y_pos = np.arange(len(top30))
bar_colors = ["#E91E63" if row["is_interaction"] else "#2196F3"
              for _, row in top30.iterrows()]
ax.barh(y_pos, top30["selection_rate"] * 100, color=bar_colors, alpha=0.8, height=0.7)
ax.set_yticks(y_pos)
labels30 = [l[:35] for l in top30["feature"]]
ax.set_yticklabels(labels30, fontsize=7)
ax.set_xlabel("Selection rate (% of LOO folds, Lasso)", fontsize=10)
ax.set_title("B — Feature Stability (top 30)", fontsize=11, fontweight="bold")
ax.axvline(50, color="gray", lw=0.8, ls="--", alpha=0.6)
patch_inter = plt.Rectangle((0,0),1,1, fc="#E91E63", alpha=0.8, label="interaction")
patch_main  = plt.Rectangle((0,0),1,1, fc="#2196F3", alpha=0.8, label="main effect")
ax.legend(handles=[patch_main, patch_inter], fontsize=9)
ax.grid(axis="x", alpha=0.3)

# Panel C: Permutation distribution
ax = axes[2]
ax.hist(perm_deltas, bins=30, color="#90A4AE", alpha=0.8, edgecolor="white")
ax.axvline(obs_delta_rf, color="#F44336", lw=2, label=f"Observed ΔR²={obs_delta_rf:+.3f}")
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("ΔR²(RF) from adding interactions (permuted y)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(f"C — Permutation Test (p={perm_p:.3f})", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "interaction_panels.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {FIG_DIR}/interaction_panels.png")


# ── 8. Save summary markdown ──────────────────────────────────────────────────
explicit_df.to_csv(OUT / "interaction_explicit_tests.csv", index=False, float_format="%.5f")
stability_df.to_csv(OUT / "interaction_stability.csv",    index=False, float_format="%.5f")

with open(OUT / "interaction_modeling_summary.md", "w") as f:
    f.write("# Interaction Modelling Summary\n\n")
    f.write("## Setup\n")
    f.write(f"- N = {N} systems\n")
    f.write(f"- {len(TOP_FEATS)} main features: {', '.join(TOP_FEATS)}\n")
    f.write(f"- {len(inter_pairs)} pairwise interaction terms\n")
    f.write("- Evaluation: Leave-one-out R² throughout\n\n")

    f.write("## Overall Result: Do Interactions Help?\n\n")
    f.write(f"| Model | LOO R² (Lasso) | LOO R² (RF) |\n")
    f.write(f"|---|---|---|\n")
    f.write(f"| Main effects only | {r2_main_lasso:+.3f} | {r2_main_rf:+.3f} |\n")
    f.write(f"| Main + all pairwise interactions | {r2_full_lasso:+.3f} | {r2_full_rf:+.3f} |\n")
    f.write(f"| ΔR² from interactions | {r2_full_lasso-r2_main_lasso:+.3f} | {r2_full_rf-r2_main_rf:+.3f} |\n")
    f.write(f"\nPermutation test (RF ΔR²): observed={obs_delta_rf:+.3f}, p={perm_p:.3f}\n\n")

    f.write("## Explicitly Requested Interactions\n\n")
    f.write(f"| Interaction | R²_main | R²_+inter | ΔR² | Consistency |\n")
    f.write(f"|---|---|---|---|---|\n")
    for r in explicit_rows:
        dir_s = f"+{r['pct_positive']*100:.0f}%" if r['pct_positive'] > 0.5 \
                else f"−{(1-r['pct_positive'])*100:.0f}%"
        f.write(f"| {r['interaction']} | {r['r2_main']:+.3f} | {r['r2_inter']:+.3f} | "
                f"{r['delta_r2']:+.3f} | {dir_s} |\n")

    f.write("\n## Feature Stability (Lasso Selection Rate)\n\n")
    f.write("Features selected in >20% of LOO folds:\n\n")
    f.write(f"| Feature | Type | Sel.% | Direction |\n")
    f.write(f"|---|---|---|---|\n")
    for _, row in stability_df[stability_df["selection_rate"] > 0.20].iterrows():
        t = "interaction" if row["is_interaction"] else "main"
        d = f"+{row['pct_positive']*100:.0f}%" if row["pct_positive"] > 0.5 \
            else f"−{(1-row['pct_positive'])*100:.0f}%"
        f.write(f"| {row['feature']} | {t} | {row['selection_rate']*100:.1f}% | {d} |\n")

    f.write("\n## Interpretation\n\n")
    if abs(r2_full_rf - r2_main_rf) < 0.05 and perm_p > 0.05:
        interp = ("Interactions do NOT add meaningful predictive power beyond main effects. "
                  "The permutation test confirms this is consistent with sampling noise. "
                  "Residual hardness is better described by additive main effects than by "
                  "multiplicative interactions among the tested features.")
    elif perm_p < 0.05:
        interp = ("Interactions add statistically significant predictive power (permutation p<0.05). "
                  "See stability table for which interaction terms survive Lasso regularization most "
                  "consistently.")
    else:
        interp = ("Interactions show marginal improvement in RF but not Lasso, and the permutation "
                  f"test does not reach significance (p={perm_p:.3f}). Signal is weak and possibly "
                  "outlier-driven. Treat with caution.")
    f.write(interp + "\n")

print(f"\nSummary saved → {OUT}/interaction_modeling_summary.md")
print("\n✓ Part 1 (Interaction Modelling) complete.")
