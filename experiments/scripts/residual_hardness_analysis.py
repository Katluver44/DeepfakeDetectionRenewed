"""
Residual Hardness Analysis
---------------------------
Fits smoothness-only models predicting system EER, then computes
residual hardness = observed - predicted to identify systems that are
harder/easier than temporal smoothness alone can explain.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

RES = "/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/experiments/results/mlaad"
OUT = f"{RES}/residual_hardness"
import os; os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

# Canonical EER: mean over all robust_goat seeds in e6 ranking lock
e6 = pd.read_csv(f"{RES}/e6_ranking_lock/per_seed_per_attack_eer.csv")
robust_e6 = e6[e6["condition"] == "robust_goat"]
seeds_present = robust_e6["seed"].unique()
print(f"robust_goat seeds in e6: {sorted(seeds_present)}")
mean_eer = (
    robust_e6.groupby("attack_system")["eer"]
    .agg(observed_eer="mean", eer_std="std", n_seeds="count")
    .reset_index()
    .rename(columns={"attack_system": "system"})
)

# Temporal smoothness features
phoneme = pd.read_csv(f"{RES}/higher_order_mechanism_analysis/phoneme_variance.csv")
phoneme = phoneme.rename(columns={"system_id": "system"})

frame = pd.read_csv(f"{RES}/higher_order_mechanism_analysis/frame_variance.csv")
frame = frame.rename(columns={"system_id": "system"})

# Additional features (graph topology, attention entropy, concentration)
entropy = pd.read_csv(f"{RES}/higher_order_mechanism_analysis/attention_entropy.csv")
entropy = entropy.rename(columns={"system_id": "system"})

sparsity = pd.read_csv(f"{RES}/higher_order_mechanism_analysis/graph_sparsity.csv")
sparsity = sparsity.rename(columns={"system_id": "system"})

concentration = pd.read_csv(f"{RES}/higher_order_mechanism_analysis/graph_concentration.csv")
concentration = concentration.rename(columns={"system_id": "system"})

# WavLM cluster distances
wavlm = pd.read_csv(f"{RES}/extreme_system_analysis/wavlm_distances.csv")
wavlm = wavlm.rename(columns={"attack_system": "system"})

# ════════════════════════════════════════════════════════════════════════════
# 2. MERGE INTO ONE DATAFRAME
# ════════════════════════════════════════════════════════════════════════════
df = mean_eer.copy()
df = df.merge(phoneme[["system","phoneme_var_mean","phoneme_var_median","n_segs_mean"]], on="system", how="inner")
df = df.merge(frame[["system","frame_mean_dist_mean","frame_var_dist_mean"]], on="system", how="inner")
df = df.merge(entropy, on="system", how="left")
df = df.merge(sparsity, on="system", how="left")
df = df.merge(concentration, on="system", how="left")
df = df.merge(wavlm[["system","group","cosine_centroid","l2_centroid","per_utt_cosine_mean","per_utt_cosine_std"]], on="system", how="left")

print(f"Systems with full data: {len(df)}")
print(f"Observed EER range: [{df['observed_eer'].min():.3f}, {df['observed_eer'].max():.3f}]")

# Colour/marker coding
GROUP_COLOR = {"easy": "#2196F3", "mid": "#FF9800", "hard": "#F44336"}
GROUP_MARKER = {"easy": "o", "mid": "s", "hard": "^"}

# ════════════════════════════════════════════════════════════════════════════
# 3. TASK 1 — SMOOTHNESS-ONLY HARDNESS MODELS (LOO CV)
# ════════════════════════════════════════════════════════════════════════════
SMOOTH_FEATS = ["phoneme_var_mean", "phoneme_var_median", "frame_mean_dist_mean"]

X_smooth = df[SMOOTH_FEATS].values
y = df["observed_eer"].values
n = len(y)

loo = LeaveOneOut()
scaler_s = StandardScaler()

def loo_predict(estimator, X, y, scaler=None):
    preds = np.zeros(n)
    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        if scaler is not None:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        clone = type(estimator)(**estimator.get_params())
        clone.fit(Xtr, ytr)
        preds[test_idx] = clone.predict(Xte)
    return preds

# Linear Regression
lr = LinearRegression()
lr_preds = loo_predict(lr, X_smooth, y, scaler=StandardScaler())
lr_r2 = r2_score(y, lr_preds)
lr_mae = mean_absolute_error(y, lr_preds)

# Full fit for coefficients
scaler_s.fit(X_smooth)
lr.fit(scaler_s.transform(X_smooth), y)
lr_coef = dict(zip(SMOOTH_FEATS, lr.coef_))

# Lasso (with CV alpha, LOO)
# Run full LassoCV first to pick alpha, then LOO with that alpha
from sklearn.linear_model import Lasso
lasso_cv = LassoCV(cv=5, max_iter=5000).fit(scaler_s.transform(X_smooth), y)
best_alpha = lasso_cv.alpha_
lasso_fixed = Lasso(alpha=best_alpha, max_iter=5000)
lasso_preds = loo_predict(lasso_fixed, X_smooth, y, scaler=StandardScaler())
lasso_r2 = r2_score(y, lasso_preds)
lasso_mae = mean_absolute_error(y, lasso_preds)
lasso_coef = dict(zip(SMOOTH_FEATS, lasso_cv.coef_))

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
rf_preds = loo_predict(rf, X_smooth, y, scaler=None)
rf_r2 = r2_score(y, rf_preds)
rf_mae = mean_absolute_error(y, rf_preds)
rf.fit(X_smooth, y)
rf_importances = dict(zip(SMOOTH_FEATS, rf.feature_importances_))

print("\n" + "="*70)
print("TASK 1 — SMOOTHNESS-ONLY HARDNESS MODELS (Leave-One-Out CV)")
print("="*70)
print(f"\n{'Model':<22} {'LOO R²':>8}  {'LOO MAE':>9}")
print("-"*42)
print(f"{'Linear Regression':<22} {lr_r2:>8.3f}  {lr_mae:>9.4f}")
print(f"{'Lasso (α={:.4f})':<22} {lasso_r2:>8.3f}  {lasso_mae:>9.4f}".format(best_alpha))
print(f"{'Random Forest':<22} {rf_r2:>8.3f}  {rf_mae:>9.4f}")

print("\n  Linear Regression coefficients (standardised features):")
for feat, coef in lr_coef.items():
    print(f"    {feat:<28} {coef:>+8.4f}")

print("\n  Lasso coefficients:")
for feat, coef in lasso_coef.items():
    print(f"    {feat:<28} {coef:>+8.4f}")

print("\n  Random Forest feature importances:")
for feat, imp in sorted(rf_importances.items(), key=lambda x: -x[1]):
    print(f"    {feat:<28} {imp:>8.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. TASK 2 — COMPUTE RESIDUAL HARDNESS (use LR as primary; ensemble average)
# ════════════════════════════════════════════════════════════════════════════
# Use ensemble mean prediction for robustness
df["pred_eer_lr"] = lr_preds
df["pred_eer_lasso"] = lasso_preds
df["pred_eer_rf"] = rf_preds
df["pred_eer_ensemble"] = (lr_preds + lasso_preds + rf_preds) / 3
df["residual"] = df["observed_eer"] - df["pred_eer_ensemble"]
df["residual_lr"] = df["observed_eer"] - df["pred_eer_lr"]

# ════════════════════════════════════════════════════════════════════════════
# 5. TASK 3 — RANK SYSTEMS BY RESIDUAL
# ════════════════════════════════════════════════════════════════════════════
sorted_df = df.sort_values("residual").reset_index(drop=True)

print("\n" + "="*70)
print("TASK 3 — SYSTEMS RANKED BY RESIDUAL HARDNESS (ensemble LOO prediction)")
print("="*70)
print("\n  Positive residual  = harder than temporal smoothness predicts")
print("  Negative residual  = easier than temporal smoothness predicts")
print(f"\n{'#':>3}  {'System':<32} {'Grp':>5}  {'Obs':>6}  {'Pred':>6}  "
      f"{'Resid':>7}  {'PVar':>6}  {'FDist':>6}")
print("-"*88)

for i, row in sorted_df.iterrows():
    flag = ""
    if abs(row["residual"]) > 0.08:
        flag = " ◄"
    print(f"{i+1:>3}  {row['system']:<32} {row['group']:>5}  "
          f"{row['observed_eer']:>6.3f}  {row['pred_eer_ensemble']:>6.3f}  "
          f"{row['residual']:>+7.4f}{flag}  "
          f"{row['phoneme_var_mean']:>6.4f}  {row['frame_mean_dist_mean']:>6.2f}")

print("\n--- TOP 10 HARDER-THAN-EXPECTED ---")
top10_hard = sorted_df.tail(10)[::-1]
for _, row in top10_hard.iterrows():
    print(f"  {row['system']:<32}  resid={row['residual']:>+.4f}  "
          f"obs={row['observed_eer']:.3f}  pred={row['pred_eer_ensemble']:.3f}  "
          f"pvar={row['phoneme_var_mean']:.4f}  fdist={row['frame_mean_dist_mean']:.2f}")

print("\n--- TOP 10 EASIER-THAN-EXPECTED ---")
top10_easy = sorted_df.head(10)
for _, row in top10_easy.iterrows():
    print(f"  {row['system']:<32}  resid={row['residual']:>+.4f}  "
          f"obs={row['observed_eer']:.3f}  pred={row['pred_eer_ensemble']:.3f}  "
          f"pvar={row['phoneme_var_mean']:.4f}  fdist={row['frame_mean_dist_mean']:.2f}")

# ════════════════════════════════════════════════════════════════════════════
# 6. TASK 4 — CLUSTER ANALYSIS ON RESIDUAL-HARD VS RESIDUAL-EASY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 4 — CLUSTER ANALYSIS: residual-hard vs residual-easy")
print("="*70)

CLUSTER_FEATS = [
    "phoneme_var_mean", "frame_mean_dist_mean", "frame_var_dist_mean",
    "mean_entropy_mean", "var_entropy_mean", "mean_gini_mean",
    "per_utt_cosine_mean", "per_utt_cosine_std", "cosine_centroid",
]
cluster_df = df[["system","group","residual","observed_eer"] + CLUSTER_FEATS].dropna()

RESID_THRESH = 0.07
hard_res = cluster_df[cluster_df["residual"] >  RESID_THRESH]
easy_res  = cluster_df[cluster_df["residual"] < -RESID_THRESH]
mid_res   = cluster_df[(cluster_df["residual"] >= -RESID_THRESH) &
                        (cluster_df["residual"] <=  RESID_THRESH)]

print(f"\n  Residual threshold: ±{RESID_THRESH}")
print(f"  Residual-hard systems (resid > +{RESID_THRESH}): {len(hard_res)}")
print(f"  Residual-easy systems (resid < -{RESID_THRESH}): {len(easy_res)}")
print(f"  Neutral systems:                               {len(mid_res)}")

def group_profile(subset, label, feats):
    print(f"\n  {label} — feature profile (mean ± std):")
    for feat in feats:
        vals = subset[feat].dropna()
        print(f"    {feat:<30} {vals.mean():>8.4f} ± {vals.std():.4f}")

group_profile(hard_res, "RESIDUAL-HARD", CLUSTER_FEATS)
group_profile(easy_res,  "RESIDUAL-EASY",  CLUSTER_FEATS)
group_profile(mid_res,   "NEUTRAL",         CLUSTER_FEATS)

# Welch t-tests: hard vs easy for each feature
print("\n  Welch t-test: residual-hard vs residual-easy systems")
print(f"  {'Feature':<30} {'t':>7}  {'p':>8}  {'Direction'}")
print("  " + "-"*65)
for feat in CLUSTER_FEATS:
    hvals = hard_res[feat].dropna()
    evals = easy_res[feat].dropna()
    if len(hvals) < 2 or len(evals) < 2:
        continue
    t, p = stats.ttest_ind(hvals, evals, equal_var=False)
    direction = "hard>easy" if hvals.mean() > evals.mean() else "easy>hard"
    sig = "*" if p < 0.10 else ""
    print(f"  {feat:<30} {t:>+7.3f}  {p:>8.4f}{sig}  {direction}")

print("\n  * p < 0.10")

# Spotlight specific systems
spotlight = {
    "OuteTTS":        "Hypothesis: residual-hard",
    "MatchaTTS":      "Hypothesis: residual-hard",
    "Metavoice-1B":   "Hypothesis: residual-hard",
    "Nari Dia2":      "Hypothesis: residual-hard",
    "ZipVoice":       "Hypothesis: residual-easy",
    "Kitten-TTS-Nano-0.2": "Hypothesis: residual-easy",
    "orpheus-tts-0.1-finetune": "Check: easy group but high pvar",
}
print("\n  SPOTLIGHT SYSTEMS:")
print(f"  {'System':<32} {'Obs EER':>8}  {'Pred EER':>9}  {'Residual':>9}  {'PVar':>7}  {'FDist':>7}  {'Note'}")
print("  " + "-"*100)
for sname, note in spotlight.items():
    row = df[df["system"].str.lower() == sname.lower()]
    if len(row) == 0:
        row = df[df["system"].str.contains(sname.split("-")[0], case=False, na=False)]
    if len(row) == 0:
        print(f"  {sname}: not found")
        continue
    row = row.iloc[0]
    print(f"  {row['system']:<32} {row['observed_eer']:>8.3f}  {row['pred_eer_ensemble']:>9.3f}  "
          f"{row['residual']:>+9.4f}  {row['phoneme_var_mean']:>7.4f}  "
          f"{row['frame_mean_dist_mean']:>7.2f}  {note}")

# ════════════════════════════════════════════════════════════════════════════
# 7. VARIANCE EXPLANATION BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VARIANCE EXPLANATION BREAKDOWN")
print("="*70)
total_var = np.var(y)
resid_var_lr    = np.var(df["residual_lr"].values)
resid_var_ens   = np.var(df["residual"].values)

frac_explained_lr  = 1 - resid_var_lr  / total_var
frac_explained_ens = 1 - resid_var_ens / total_var

# Pearson correlations for each smoothness feature alone
print(f"\n  Total EER variance (across {n} systems): {total_var:.5f}")
print(f"\n  Fraction explained by smoothness features (LOO R²):")
print(f"    Linear Regression: {lr_r2:>.3f}  ({lr_r2*100:.1f}% of variance)")
print(f"    Lasso:             {lasso_r2:>.3f}  ({lasso_r2*100:.1f}% of variance)")
print(f"    Random Forest:     {rf_r2:>.3f}  ({rf_r2*100:.1f}% of variance)")

pr_pvar, _  = stats.pearsonr(df["phoneme_var_mean"],   y)
pr_pmed, _  = stats.pearsonr(df["phoneme_var_median"], y)
pr_fdist, _ = stats.pearsonr(df["frame_mean_dist_mean"], y)
print(f"\n  Individual r² contributions (bivariate):")
print(f"    phoneme_var_mean:     r={pr_pvar:.3f}  r²={pr_pvar**2:.3f}")
print(f"    phoneme_var_median:   r={pr_pmed:.3f}  r²={pr_pmed**2:.3f}")
print(f"    frame_mean_dist_mean: r={pr_fdist:.3f}  r²={pr_fdist**2:.3f}")

unexplained_ens = (1 - (lr_r2 + lasso_r2 + rf_r2)/3) * 100
print(f"\n  >>> Temporal smoothness explains ~{(lr_r2+lasso_r2+rf_r2)/3*100:.0f}% of EER variance")
print(f"  >>> ~{unexplained_ens:.0f}% of hardness variance remains unexplained")

# ════════════════════════════════════════════════════════════════════════════
# 8. TASK 5 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

# Color by residual magnitude
def residual_color(r):
    if r > 0.10: return "#D32F2F"
    elif r > 0.05: return "#F57C00"
    elif r < -0.10: return "#1565C0"
    elif r < -0.05: return "#1976D2"
    else: return "#78909C"

df["rcolor"] = df["residual"].apply(residual_color)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel A: Predicted vs Observed ──────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for grp, sub in df.groupby("group"):
    ax_a.scatter(sub["pred_eer_ensemble"], sub["observed_eer"],
                 c=sub["rcolor"], marker=GROUP_MARKER.get(grp,"o"),
                 s=60, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)

lims = [min(df["pred_eer_ensemble"].min(), df["observed_eer"].min()) - 0.01,
        max(df["pred_eer_ensemble"].max(), df["observed_eer"].max()) + 0.01]
ax_a.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="y = x (perfect)")
ax_a.set_xlim(lims); ax_a.set_ylim(lims)
ax_a.set_xlabel("Predicted EER (smoothness model)", fontsize=10)
ax_a.set_ylabel("Observed EER", fontsize=10)
ax_a.set_title(f"Predicted vs. Observed EER\n(ensemble LOO R²={((lr_r2+lasso_r2+rf_r2)/3):.2f})", fontsize=10)
ax_a.grid(True, alpha=0.3)

key_label = {"ZipVoice", "OuteTTS", "MatchaTTS", "Metavoice-1B",
             "Nari Dia2", "orpheus-tts-0.1-finetune", "sesame_csm", "vixTTS"}
for _, row in df.iterrows():
    if any(k.lower() in row["system"].lower() for k in key_label):
        ax_a.annotate(row["system"].replace("Metavoice-1B","Metavoice"),
                      (row["pred_eer_ensemble"], row["observed_eer"]),
                      xytext=(4, 3), textcoords="offset points", fontsize=7)

patches = [
    mpatches.Patch(color="#D32F2F", label="Resid > +0.10"),
    mpatches.Patch(color="#F57C00", label="Resid +0.05..+0.10"),
    mpatches.Patch(color="#78909C", label="Resid ±0.05"),
    mpatches.Patch(color="#1976D2", label="Resid −0.05..−0.10"),
    mpatches.Patch(color="#1565C0", label="Resid < −0.10"),
]
ax_a.legend(handles=patches, fontsize=7, framealpha=0.9)

# ── Panel B: Residual distribution ──────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
resids = df["residual"].values
ax_b.hist(resids, bins=18, color="#546E7A", alpha=0.8, edgecolor="white", linewidth=0.6)
ax_b.axvline(0, color="k", lw=1.5, ls="--")
ax_b.axvline(resids.mean(), color="#E53935", lw=1.5, ls="-", label=f"mean={resids.mean():+.3f}")
ax_b.axvline(np.median(resids), color="#43A047", lw=1.5, ls="-", label=f"median={np.median(resids):+.3f}")
_, norm_p = stats.normaltest(resids)
ax_b.set_xlabel("Residual Hardness", fontsize=10)
ax_b.set_ylabel("Count", fontsize=10)
ax_b.set_title(f"Residual Distribution\n(D'Agostino–Pearson p={norm_p:.3f})", fontsize=10)
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)

# ── Panel C: Residual vs phoneme variance ──────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
for grp, sub in df.groupby("group"):
    ax_c.scatter(sub["phoneme_var_mean"], sub["residual"],
                 c=sub["rcolor"], marker=GROUP_MARKER.get(grp,"o"),
                 s=60, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4,
                 label=grp.capitalize())
slope_c, intercept_c, r_c, p_c, _ = stats.linregress(df["phoneme_var_mean"], df["residual"])
xr = np.linspace(df["phoneme_var_mean"].min(), df["phoneme_var_mean"].max(), 100)
ax_c.plot(xr, slope_c*xr+intercept_c, "k--", lw=1.4, alpha=0.7,
          label=f"OLS r={r_c:+.2f} (p={p_c:.3f})")
ax_c.axhline(0, color="gray", lw=0.8, ls=":")
ax_c.set_xlabel("phoneme_var_mean", fontsize=10)
ax_c.set_ylabel("Residual Hardness", fontsize=10)
ax_c.set_title("Residual vs. phoneme_var_mean", fontsize=10)
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3)
for _, row in df.iterrows():
    if any(k.lower() in row["system"].lower() for k in key_label):
        ax_c.annotate(row["system"][:10], (row["phoneme_var_mean"], row["residual"]),
                      xytext=(4, 3), textcoords="offset points", fontsize=7)

# ── Panel D: Residual vs frame distance ─────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
for grp, sub in df.groupby("group"):
    ax_d.scatter(sub["frame_mean_dist_mean"], sub["residual"],
                 c=sub["rcolor"], marker=GROUP_MARKER.get(grp,"o"),
                 s=60, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4,
                 label=grp.capitalize())
slope_d, intercept_d, r_d, p_d, _ = stats.linregress(df["frame_mean_dist_mean"], df["residual"])
xr2 = np.linspace(df["frame_mean_dist_mean"].min(), df["frame_mean_dist_mean"].max(), 100)
ax_d.plot(xr2, slope_d*xr2+intercept_d, "k--", lw=1.4, alpha=0.7,
          label=f"OLS r={r_d:+.2f} (p={p_d:.3f})")
ax_d.axhline(0, color="gray", lw=0.8, ls=":")
ax_d.set_xlabel("frame_mean_dist_mean", fontsize=10)
ax_d.set_ylabel("Residual Hardness", fontsize=10)
ax_d.set_title("Residual vs. frame_mean_dist_mean", fontsize=10)
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)
for _, row in df.iterrows():
    if any(k.lower() in row["system"].lower() for k in key_label):
        ax_d.annotate(row["system"][:10], (row["frame_mean_dist_mean"], row["residual"]),
                      xytext=(4, 3), textcoords="offset points", fontsize=7)

# ── Panel E: Horizontal bar chart sorted by residual ────────────────────────
ax_e = fig.add_subplot(gs[1, 1:])
top_n = 20  # top 10 + bottom 10
extremes = pd.concat([sorted_df.head(10), sorted_df.tail(10)]).drop_duplicates("system")
extremes_sorted = extremes.sort_values("residual")
colors_bar = [residual_color(r) for r in extremes_sorted["residual"]]
bars = ax_e.barh(range(len(extremes_sorted)), extremes_sorted["residual"],
                 color=colors_bar, edgecolor="white", linewidth=0.5, alpha=0.9)
ax_e.set_yticks(range(len(extremes_sorted)))
ax_e.set_yticklabels(extremes_sorted["system"], fontsize=8)
ax_e.axvline(0, color="k", lw=1.2, ls="--")
ax_e.set_xlabel("Residual Hardness (observed − predicted)", fontsize=10)
ax_e.set_title("Top/Bottom 10 Systems by Residual Hardness", fontsize=10)
ax_e.grid(True, alpha=0.3, axis="x")

fig.suptitle("Residual Hardness Analysis — Beyond Temporal Smoothness",
             fontsize=14, fontweight="bold", y=1.01)

out_fig = f"{OUT}/residual_hardness_panels.png"
plt.savefig(out_fig, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")

# ════════════════════════════════════════════════════════════════════════════
# 9. ADDITIONAL DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("RESIDUAL CORRELATIONS WITH ALL FEATURES (secondary mechanism search)")
print("="*70)

extra_feats = [
    "frame_var_dist_mean", "mean_entropy_mean", "var_entropy_mean",
    "mean_gini_mean", "per_utt_cosine_mean", "per_utt_cosine_std",
    "cosine_centroid", "n_segs_mean"
]
print(f"\n  {'Feature':<30} {'Pearson r':>10}  {'p':>8}  {'Spearman ρ':>11}  {'p':>8}")
print("  " + "-"*72)
for feat in extra_feats:
    sub = df[["residual", feat]].dropna()
    if len(sub) < 5: continue
    pr, pp = stats.pearsonr(sub["residual"], sub[feat])
    sr, sp = stats.spearmanr(sub["residual"], sub[feat])
    sig = "*" if pp < 0.10 else ""
    print(f"  {feat:<30} {pr:>+10.3f}  {pp:>8.4f}{sig}  {sr:>+11.3f}  {sp:>8.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 10. SAVE MASTER TABLE
# ════════════════════════════════════════════════════════════════════════════
out_cols = [
    "system", "group", "observed_eer", "pred_eer_lr", "pred_eer_lasso",
    "pred_eer_rf", "pred_eer_ensemble", "residual",
    "phoneme_var_mean", "phoneme_var_median", "frame_mean_dist_mean",
    "frame_var_dist_mean", "per_utt_cosine_mean", "cosine_centroid",
    "mean_entropy_mean", "var_entropy_mean", "mean_gini_mean",
]
save_df = sorted_df[out_cols].copy()
save_df.to_csv(f"{OUT}/residual_hardness_table.csv", index=False, float_format="%.5f")
print(f"\nMaster table saved → {OUT}/residual_hardness_table.csv")

# ════════════════════════════════════════════════════════════════════════════
# 11. RESIDUAL HARDNESS vs GROUP
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("RESIDUAL BY DIFFICULTY GROUP")
print("="*70)
for grp in ["easy", "mid", "hard"]:
    sub = df[df["group"] == grp]["residual"]
    print(f"  {grp:<6}  n={len(sub):>2}  mean={sub.mean():>+.4f}  "
          f"median={sub.median():>+.4f}  std={sub.std():.4f}")
