"""
Smoothing Augmentation Analysis
--------------------------------
Tests whether ΔEER (augmented - baseline) is mechanistically predictable
from baseline temporal structure (phoneme_var_mean, frame_mean_dist_mean).
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/experiments/results/mlaad/smoothing_augmentation"
HIGHER_ORDER_DIR = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/experiments/results/mlaad/higher_order_mechanism_analysis"
EXTREME_DIR = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/experiments/results/mlaad/extreme_system_analysis"

# ── 1. Load per-system baseline and smoothaug EER ────────────────────────────
with open(f"{RESULTS_DIR}/per_seed_metrics.json") as f:
    seed_data = json.load(f)

records = []
for seed_str, sd in seed_data.items():
    seed = int(seed_str)
    for sys_name, baseline_eer in sd["baseline"]["per_sys"].items():
        smoothaug_eer = sd["smoothaug"]["per_sys"].get(sys_name)
        if smoothaug_eer is None:
            continue
        records.append({
            "system": sys_name,
            "seed": seed,
            "baseline_eer": baseline_eer,
            "smoothaug_eer": smoothaug_eer,
            "delta_eer": smoothaug_eer - baseline_eer,
        })

per_seed_df = pd.DataFrame(records)

# Mean across seeds per system
per_sys = (
    per_seed_df.groupby("system")
    .agg(
        baseline_eer_mean=("baseline_eer", "mean"),
        smoothaug_eer_mean=("smoothaug_eer", "mean"),
        delta_eer_mean=("delta_eer", "mean"),
        delta_eer_std=("delta_eer", "std"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)

# ── 2. Load temporal metrics ─────────────────────────────────────────────────
phoneme_df = pd.read_csv(f"{HIGHER_ORDER_DIR}/phoneme_variance.csv")
phoneme_df = phoneme_df.rename(columns={"system_id": "system"})

frame_df = pd.read_csv(f"{HIGHER_ORDER_DIR}/frame_variance.csv")
frame_df = frame_df.rename(columns={"system_id": "system"})

wavlm_df = pd.read_csv(f"{EXTREME_DIR}/wavlm_distances.csv")
wavlm_df = wavlm_df.rename(columns={"attack_system": "system"})
group_map = wavlm_df.set_index("system")["group"].to_dict()

# ── 3. Merge ─────────────────────────────────────────────────────────────────
df = per_sys.merge(
    phoneme_df[["system", "phoneme_var_mean", "phoneme_var_median"]],
    on="system", how="inner"
).merge(
    frame_df[["system", "frame_mean_dist_mean"]],
    on="system", how="inner"
)
df["group"] = df["system"].map(group_map).fillna("mid")

print(f"Systems with full data: {len(df)}")

# ── 4. Correlations ──────────────────────────────────────────────────────────
feature_cols = ["phoneme_var_mean", "phoneme_var_median", "frame_mean_dist_mean", "baseline_eer_mean"]
feature_labels = {
    "phoneme_var_mean": "phoneme_var_mean",
    "phoneme_var_median": "phoneme_var_median",
    "frame_mean_dist_mean": "frame_mean_dist_mean",
    "baseline_eer_mean": "baseline EER",
}

print("\n" + "="*70)
print("CORRELATION ANALYSIS: ΔEER vs. temporal + baseline features")
print("="*70)
print(f"\n{'Feature':<25} {'Pearson r':>10} {'p-value':>10} {'Spearman ρ':>12} {'p-value':>10}")
print("-"*70)

corr_results = {}
for col in feature_cols:
    mask = df[col].notna() & df["delta_eer_mean"].notna()
    x = df.loc[mask, col].values
    y = df.loc[mask, "delta_eer_mean"].values
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    corr_results[col] = {"pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp}
    sig_p = "*" if pp < 0.05 else ""
    sig_s = "*" if sp < 0.05 else ""
    print(f"{feature_labels[col]:<25} {pr:>+9.3f}{sig_p}  {pp:>9.4f}  {sr:>+11.3f}{sig_s}  {sp:>9.4f}")

print("\n* p < 0.05")

# ── 5. Quartile stratification by phoneme_var_mean ───────────────────────────
q25 = df["phoneme_var_mean"].quantile(0.25)
q75 = df["phoneme_var_mean"].quantile(0.75)

def assign_quartile(v):
    if v <= q25:
        return "Low (≤Q1)"
    elif v >= q75:
        return "High (≥Q3)"
    else:
        return "Middle (Q1–Q3)"

df["pvar_group"] = df["phoneme_var_mean"].apply(assign_quartile)
order = ["Low (≤Q1)", "Middle (Q1–Q3)", "High (≥Q3)"]

print("\n" + "="*70)
print("QUARTILE STRATIFICATION: phoneme_var_mean vs. ΔEER")
print(f"  Q25={q25:.4f}  Q75={q75:.4f}")
print("="*70)
print(f"\n{'Group':<18} {'N':>5} {'Mean ΔEER':>12} {'Median ΔEER':>13} {'Std ΔEER':>12}")
print("-"*65)
for g in order:
    sub = df[df["pvar_group"] == g]["delta_eer_mean"]
    print(f"{g:<18} {len(sub):>5} {sub.mean():>+11.4f}  {sub.median():>+11.4f}  {sub.std():>11.4f}")

# ── 6. Regression: ΔEER ~ phoneme_var_mean + frame_mean_dist_mean + baseline_eer ──
print("\n" + "="*70)
print("LINEAR REGRESSION: ΔEER ~ temporal + baseline features")
print("="*70)

features = ["phoneme_var_mean", "frame_mean_dist_mean", "baseline_eer_mean"]
X = df[features].values
y = df["delta_eer_mean"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression().fit(X_scaled, y)
y_pred = reg.predict(X_scaled)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\n  R² = {r2:.3f}")
print(f"\n  {'Feature':<28} {'Std. Coef.':>12}  {'Sign'}  {'Interpretation'}")
print("  " + "-"*75)
interpret = {
    "phoneme_var_mean": "Higher pvar → more ΔEER" if reg.coef_[0] > 0 else "Higher pvar → less ΔEER (improvement)",
    "frame_mean_dist_mean": "Higher frame dist → more ΔEER" if reg.coef_[1] > 0 else "Higher frame dist → less ΔEER",
    "baseline_eer_mean": "Higher baseline → more ΔEER" if reg.coef_[2] > 0 else "Higher baseline → less ΔEER",
}
for feat, coef in zip(features, reg.coef_):
    sign = "+" if coef > 0 else "−"
    print(f"  {feat:<28} {coef:>+11.4f}  [{sign}]  {interpret[feat]}")
print(f"\n  Intercept (unstd): {reg.intercept_:+.4f}")

# F-test for overall regression significance
from scipy.stats import f as f_dist
n, k = X.shape
f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
f_p = 1 - f_dist.cdf(f_stat, k, n - k - 1)
print(f"\n  F({k}, {n-k-1}) = {f_stat:.2f}, p = {f_p:.4f}")

# ── 7. System-level sorted table ─────────────────────────────────────────────
print("\n" + "="*70)
print("SYSTEM TABLE sorted by ΔEER (negative = improvement)")
print("="*70)
display = df[[
    "system", "group", "pvar_group",
    "phoneme_var_mean", "frame_mean_dist_mean",
    "baseline_eer_mean", "smoothaug_eer_mean", "delta_eer_mean"
]].sort_values("delta_eer_mean").reset_index(drop=True)

print(f"\n{'#':>3}  {'System':<32} {'Grp':>5}  {'PVar':>6}  {'FrmDst':>7}  "
      f"{'Base':>6}  {'Aug':>6}  {'ΔEER':>7}")
print("-"*85)
for i, row in display.iterrows():
    pvar_str = f"{row['phoneme_var_mean']:.4f}"
    fdst_str = f"{row['frame_mean_dist_mean']:.2f}"
    marker = " ◄" if abs(row["delta_eer_mean"]) > 0.07 else ""
    print(f"{i+1:>3}  {row['system']:<32} {row['group']:>5}  {pvar_str}  {fdst_str}  "
          f"{row['baseline_eer_mean']:>6.3f}  {row['smoothaug_eer_mean']:>6.3f}  "
          f"{row['delta_eer_mean']:>+7.4f}{marker}")

# ── 8. Hypothesis test: low vs high pvar ΔEER ────────────────────────────────
print("\n" + "="*70)
print("HYPOTHESIS TEST: Low pvar vs High pvar ΔEER")
print("="*70)
low_group = df[df["pvar_group"] == "Low (≤Q1)"]["delta_eer_mean"]
high_group = df[df["pvar_group"] == "High (≥Q3)"]["delta_eer_mean"]
mid_group = df[df["pvar_group"] == "Middle (Q1–Q3)"]["delta_eer_mean"]

t_stat, t_p = stats.ttest_ind(low_group, high_group, equal_var=False)
mwu_stat, mwu_p = stats.mannwhitneyu(low_group, high_group, alternative="two-sided")

print(f"\n  Low pvar mean ΔEER : {low_group.mean():+.4f}  (n={len(low_group)})")
print(f"  Mid pvar mean ΔEER : {mid_group.mean():+.4f}  (n={len(mid_group)})")
print(f"  High pvar mean ΔEER: {high_group.mean():+.4f}  (n={len(high_group)})")
print(f"\n  Welch t-test (low vs high): t={t_stat:.3f}, p={t_p:.4f}")
print(f"  Mann-Whitney U     (low vs high): U={mwu_stat:.0f}, p={mwu_p:.4f}")

# ── 9. Plots ─────────────────────────────────────────────────────────────────
group_colors = {"easy": "#2196F3", "mid": "#FF9800", "hard": "#F44336"}
group_markers = {"easy": "o", "mid": "s", "hard": "^"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Smoothing Augmentation: ΔEER vs. Baseline Temporal Structure",
             fontsize=13, fontweight="bold", y=1.01)

def annotate_key(ax, row):
    key_systems = {"ZipVoice", "OuteTTS", "Index-TTS-1.5", "Supertonic",
                   "Metavoice-1B", "Spark-TTS-0.5B", "kokoro", "Veena", "mars5"}
    s = row["system"]
    if any(k.lower() in s.lower() for k in key_systems):
        ax.annotate(s, (row["x"], row["delta_eer_mean"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color="#333")

# Plot 1: phoneme_var_mean vs ΔEER
ax = axes[0]
for grp, sub in df.groupby("group"):
    color = group_colors.get(grp, "#888")
    marker = group_markers.get(grp, "o")
    ax.scatter(sub["phoneme_var_mean"], sub["delta_eer_mean"],
               c=color, marker=marker, s=55, alpha=0.82, zorder=3,
               edgecolors="white", linewidths=0.4, label=grp.capitalize())

# regression line
x_range = np.linspace(df["phoneme_var_mean"].min(), df["phoneme_var_mean"].max(), 100)
slope, intercept, r, p, _ = stats.linregress(df["phoneme_var_mean"], df["delta_eer_mean"])
ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.5, alpha=0.7,
        label=f"OLS (r={r:+.2f}, p={p:.3f})")

# quartile shading
ax.axvspan(df["phoneme_var_mean"].min(), q25, alpha=0.07, color="#2196F3", label="Low quartile")
ax.axvspan(q75, df["phoneme_var_mean"].max(), alpha=0.07, color="#F44336", label="High quartile")
ax.axhline(0, color="gray", lw=0.8, ls=":")

# annotate key systems
for _, row in df.iterrows():
    row["x"] = row["phoneme_var_mean"]
    annotate_key(ax, row)

ax.set_xlabel("phoneme_var_mean (WavLM space)", fontsize=11)
ax.set_ylabel("ΔEER (augmented − baseline)", fontsize=11)
ax.set_title("phoneme_var_mean vs. ΔEER", fontsize=11)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Plot 2: frame_mean_dist_mean vs ΔEER
ax = axes[1]
for grp, sub in df.groupby("group"):
    color = group_colors.get(grp, "#888")
    marker = group_markers.get(grp, "o")
    ax.scatter(sub["frame_mean_dist_mean"], sub["delta_eer_mean"],
               c=color, marker=marker, s=55, alpha=0.82, zorder=3,
               edgecolors="white", linewidths=0.4, label=grp.capitalize())

slope2, intercept2, r2_corr, p2, _ = stats.linregress(df["frame_mean_dist_mean"], df["delta_eer_mean"])
x_range2 = np.linspace(df["frame_mean_dist_mean"].min(), df["frame_mean_dist_mean"].max(), 100)
ax.plot(x_range2, slope2 * x_range2 + intercept2, "k--", lw=1.5, alpha=0.7,
        label=f"OLS (r={r2_corr:+.2f}, p={p2:.3f})")
ax.axhline(0, color="gray", lw=0.8, ls=":")

for _, row in df.iterrows():
    row["x"] = row["frame_mean_dist_mean"]
    annotate_key(ax, row)

ax.set_xlabel("frame_mean_dist_mean (WavLM L2)", fontsize=11)
ax.set_ylabel("ΔEER (augmented − baseline)", fontsize=11)
ax.set_title("frame_mean_dist_mean vs. ΔEER", fontsize=11)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_scatter = f"{RESULTS_DIR}/smoothing_aug_scatter.png"
plt.savefig(out_scatter, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nScatter plots saved → {out_scatter}")

# ── 10. Save system table CSV ─────────────────────────────────────────────────
out_csv = f"{RESULTS_DIR}/system_delta_eer_table.csv"
display.to_csv(out_csv, index=False, float_format="%.5f")
print(f"System table saved   → {out_csv}")

print("\n" + "="*70)
print("MECHANISTIC INTERPRETATION")
print("="*70)
print("""
FINDING SUMMARY:
  The correlation between phoneme_var_mean and ΔEER determines whether
  augmentation benefit is temporally structured.

  Negative r(phoneme_var_mean, ΔEER) → high-variability systems IMPROVE more
  Positive r(phoneme_var_mean, ΔEER) → low-variability systems IMPROVE more

  Negative r(frame_mean_dist_mean, ΔEER) → high frame-distance systems improve more

  Regression coefficient signs reveal which features are load-bearing predictors.
""")
