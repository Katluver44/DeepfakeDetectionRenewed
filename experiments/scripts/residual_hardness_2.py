"""
Residual Hardness Analysis 2: Four Competing Hypotheses
========================================================
Tests four structural mechanisms beyond temporal smoothness:

  H1  Local neighbourhood instability
      kNN switch rate under noise, mutual-NN rate, cosine-rank volatility,
      kNN distance entropy
  H2  Graph construction sensitivity
      GAT attention entropy variance, Gini concentration, edge density,
      max attention entropy
  H3  Representation anisotropy collapse
      Covariance condition number, cosine similarity matrix entropy,
      eigenvalue decay exponent, effective rank / pca_var1
  H4  WavLM-GAT alignment gap
      Per-utterance phoneme cosine mean, cosine std, cosine centroid,
      attention/cosine ratio

Approach: incremental LOO R² attribution (Ridge + RF),
individual Pearson / Spearman correlations, ranked hypothesis table,
4-panel visualisation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR     = Path("/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed")
RESIDUAL_DIR = BASE_DIR / "experiments/results/mlaad/residual_hardness"
MANIFOLD_DIR = BASE_DIR / "experiments/results/mlaad/manifold_geometry"
HIGHER_DIR   = BASE_DIR / "experiments/results/mlaad/higher_order_mechanism_analysis"
EXTREME_DIR  = BASE_DIR / "experiments/results/mlaad/extreme_system_analysis"
OUT_DIR      = BASE_DIR / "experiments/results/mlaad/residual_hardness_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load residual hardness table (already has H2/H4 features merged) ──────
resid_df = pd.read_csv(RESIDUAL_DIR / "residual_hardness_table.csv")
print(f"Residual table: {len(resid_df)} systems, {len(resid_df.columns)} columns")
# Columns present: system, group, observed_eer, pred_eer_*, residual,
#   phoneme_var_mean, phoneme_var_median, frame_mean_dist_mean,
#   per_utt_cosine_mean, cosine_centroid, mean_entropy_mean,
#   var_entropy_mean, mean_gini_mean

# ── 2. Load extra features not yet in residual table ─────────────────────────
# H2 extras
attn = pd.read_csv(HIGHER_DIR / "attention_entropy.csv").rename(
    columns={"system_id": "system"})
graph = pd.read_csv(HIGHER_DIR / "graph_sparsity.csv").rename(
    columns={"system_id": "system"})
conc  = pd.read_csv(HIGHER_DIR / "graph_concentration.csv").rename(
    columns={"system_id": "system"})

# H4 extras
wavlm = pd.read_csv(EXTREME_DIR / "wavlm_distances.csv").rename(
    columns={"attack_system": "system"})

# Manifold metrics (carry effective_rank, pca_var1, knn_density)
manif = pd.read_csv(MANIFOLD_DIR / "manifold_metrics.csv")

# Merge extras into base
df = resid_df.copy()
df = df.merge(attn[["system", "max_entropy_mean"]],         on="system", how="left")
df = df.merge(graph[["system", "edge_density_mean", "avg_degree_mean"]], on="system", how="left")
df = df.merge(wavlm[["system", "per_utt_cosine_std"]],      on="system", how="left")
df = df.merge(manif[["system", "effective_rank", "pca_var1", "knn_density", "intrinsic_dim"]], on="system", how="left")

# ── 3. Compute H1 and H3 features from utterance embeddings ──────────────────
print("\nLoading utterance embeddings …")
npz = np.load(MANIFOLD_DIR / "system_embeddings.npz", allow_pickle=True)
system_embs = {}
for k in npz.files:
    arr = npz[k]
    if arr.ndim == 2 and arr.shape[1] == 768:
        system_embs[k] = arr.astype(np.float32)
print(f"  {len(system_embs)} systems loaded")


def h1_features(embs: np.ndarray, k: int = 5,
                n_perturb: int = 15, noise_frac: float = 0.03) -> dict:
    """
    Local neighbourhood instability features.
      knn_entropy_norm   : mean normalised entropy of kNN distance distribution
      mutual_nn_rate     : fraction of k-NN edges that are bidirectional
      neighbor_switch_rate: mean Jaccard distance of kNN sets under noise
      cosine_rank_vol    : 1 - mean Spearman corr of cosine rankings under noise
    """
    N = len(embs)
    k_eff = min(k, N - 1)
    if k_eff < 2:
        return dict(knn_entropy_norm=np.nan, mutual_nn_rate=np.nan,
                    neighbor_switch_rate=np.nan, cosine_rank_vol=np.nan)

    rng = np.random.RandomState(42)
    scale = float(np.std(embs)) * noise_frac

    cmat = squareform(pdist(embs, metric="cosine"))  # (N, N)
    np.fill_diagonal(cmat, np.inf)
    knn_idx = np.argsort(cmat, axis=1)[:, :k_eff]   # (N, k_eff)

    # --- knn_entropy_norm ---
    knn_dists = np.sort(cmat, axis=1)[:, :k_eff].clip(0)
    inv = 1.0 / (knn_dists + 1e-8)
    probs = inv / inv.sum(axis=1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    knn_entropy_norm = float(np.mean(ent / np.log(k_eff)))

    # --- mutual_nn_rate ---
    knn_sets = [set(knn_idx[i]) for i in range(N)]
    mutual, total = 0, 0
    for i in range(N):
        for j in knn_sets[i]:
            total += 1
            if i in knn_sets[j]:
                mutual += 1
    mutual_nn_rate = float(mutual / max(total, 1))

    # --- neighbor_switch_rate and cosine_rank_vol (perturbation-based) ---
    np.fill_diagonal(cmat, 0)  # restore diagonal for rank comparisons
    switch_buf, rank_buf = [], []
    for _ in range(n_perturb):
        noise = rng.randn(*embs.shape).astype(np.float32) * scale
        em_n = embs + noise
        cmat_n = squareform(pdist(em_n, metric="cosine"))
        np.fill_diagonal(cmat_n, np.inf)
        knn_n = np.argsort(cmat_n, axis=1)[:, :k_eff]
        np.fill_diagonal(cmat_n, 0)

        for i in range(N):
            orig_s = set(knn_idx[i])
            noisy_s = set(knn_n[i])
            u = len(orig_s | noisy_s)
            if u:
                switch_buf.append(1 - len(orig_s & noisy_s) / u)

        # Spearman of full cosine ranking (excl. self)
        mask = np.ones(N, dtype=bool)
        for i in range(N):
            mask[:] = True
            mask[i] = False
            if mask.sum() > 2:
                r, _ = stats.spearmanr(cmat[i][mask], cmat_n[i][mask])
                if not np.isnan(r):
                    rank_buf.append(r)

    neighbor_switch_rate = float(np.mean(switch_buf)) if switch_buf else np.nan
    cosine_rank_vol      = float(1.0 - np.mean(rank_buf)) if rank_buf else np.nan

    return dict(knn_entropy_norm=knn_entropy_norm,
                mutual_nn_rate=mutual_nn_rate,
                neighbor_switch_rate=neighbor_switch_rate,
                cosine_rank_vol=cosine_rank_vol)


def h3_features(embs: np.ndarray) -> dict:
    """
    Anisotropy collapse features.
      condition_number_log : log10( λ_max / λ_min ) of covariance
      cosine_sim_entropy   : entropy of pairwise cosine similarity histogram
      eigenvalue_decay_exp : power-law exponent of sorted eigenvalue spectrum
      sv_entropy           : entropy of normalised eigenvalues (= log eff_rank)
    """
    N = len(embs)
    if N < 3:
        return dict(condition_number_log=np.nan, cosine_sim_entropy=np.nan,
                    eigenvalue_decay_exp=np.nan, sv_entropy=np.nan)

    X = embs - embs.mean(axis=0)
    try:
        sv = np.linalg.svd(X, compute_uv=False)
        ev = (sv ** 2) / max(N - 1, 1)
        ev = ev[ev > 1e-10]
    except np.linalg.LinAlgError:
        return dict(condition_number_log=np.nan, cosine_sim_entropy=np.nan,
                    eigenvalue_decay_exp=np.nan, sv_entropy=np.nan)

    if len(ev) < 2:
        return dict(condition_number_log=np.nan, cosine_sim_entropy=np.nan,
                    eigenvalue_decay_exp=np.nan, sv_entropy=np.nan)

    # Condition number
    cond_log = float(np.log10(ev[0] / ev[-1] + 1))

    # Cosine similarity matrix entropy
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    en = embs / (norms + 1e-10)
    cs_mat = en @ en.T
    idx = np.triu_indices(N, k=1)
    cs_vals = cs_mat[idx]
    nbins = max(10, min(30, len(cs_vals) // 3))
    hist, _ = np.histogram(cs_vals, bins=nbins, range=(-1, 1))
    hist = hist.astype(float) + 1e-8
    hist /= hist.sum()
    cs_ent = float(-np.sum(hist * np.log(hist)))

    # Eigenvalue decay exponent (power-law fit on sorted spectrum)
    if len(ev) >= 3:
        ranks = np.arange(1, len(ev) + 1, dtype=float)
        slope, _, _, _, _ = stats.linregress(np.log(ranks), np.log(ev + 1e-10))
        ev_decay = float(-slope)   # positive = steeper decay = more collapsed
    else:
        ev_decay = np.nan

    # Eigenvalue entropy (same as log(effective_rank))
    evn = ev / ev.sum()
    sv_ent = float(-np.sum(evn * np.log(evn + 1e-12)))

    return dict(condition_number_log=cond_log,
                cosine_sim_entropy=cs_ent,
                eigenvalue_decay_exp=ev_decay,
                sv_entropy=sv_ent)


print("Computing H1 + H3 features per system …")
h1_rows, h3_rows = [], []
for sys_name, embs in system_embs.items():
    if len(embs) < 3:
        print(f"  skip {sys_name}: only {len(embs)} utterances")
        continue
    h1_rows.append({"system": sys_name, **h1_features(embs)})
    h3_rows.append({"system": sys_name, **h3_features(embs)})
    if len(h1_rows) % 15 == 0:
        print(f"  {len(h1_rows)} / {len(system_embs)} done")

h1_df = pd.DataFrame(h1_rows)
h3_df = pd.DataFrame(h3_rows)
print(f"  H1 ready: {len(h1_df)} systems  |  H3 ready: {len(h3_df)} systems")

# Merge into master frame
df = df.merge(h1_df, on="system", how="left")
df = df.merge(h3_df, on="system", how="left")

# H4 composite: attention spread relative to phoneme cosine distance
df["cosine_attention_ratio"] = (
    df["mean_entropy_mean"] / (df["per_utt_cosine_mean"] + 1e-6)
)
# low per_utt + low entropy = "collapsed" (hard); high = "alive" (easy)
# Sign: high ratio could mean either large attn or small cos → need both to interpret
df["utt_cos_x_entropy"] = df["per_utt_cosine_mean"] * df["mean_entropy_mean"]

print(f"\nMaster frame: {len(df)} systems, {len(df.columns)} columns")

# ── 4. Feature group definitions ─────────────────────────────────────────────
BASE_FEATS = ["phoneme_var_mean", "frame_mean_dist_mean"]

H1_FEATS = [
    "knn_entropy_norm", "mutual_nn_rate", "neighbor_switch_rate", "cosine_rank_vol",
    "knn_density", "intrinsic_dim",          # manifold-computed H1 proxies
]
H2_FEATS = [
    "var_entropy_mean", "max_entropy_mean", "mean_gini_mean", "edge_density_mean",
]
H3_FEATS = [
    "condition_number_log", "cosine_sim_entropy", "eigenvalue_decay_exp", "sv_entropy",
    "effective_rank", "pca_var1",             # manifold-computed H3 proxies
]
H4_FEATS = [
    "per_utt_cosine_mean", "per_utt_cosine_std", "cosine_centroid",
    "cosine_attention_ratio", "utt_cos_x_entropy",
]

HYPS = {
    "H1 Neighbourhood instability": H1_FEATS,
    "H2 Graph sensitivity":         H2_FEATS,
    "H3 Anisotropy collapse":       H3_FEATS,
    "H4 WavLM–GAT alignment":       H4_FEATS,
}

def available(feats, frame, min_valid=40):
    return [f for f in feats
            if f in frame.columns and frame[f].notna().sum() >= min_valid]

H1_A = available(H1_FEATS, df)
H2_A = available(H2_FEATS, df)
H3_A = available(H3_FEATS, df)
H4_A = available(H4_FEATS, df)

print("\nFeature availability after merging:")
for label, feats in [("H1", H1_A), ("H2", H2_A), ("H3", H3_A), ("H4", H4_A)]:
    print(f"  {label}: {feats}")

# ── 5. Individual feature correlations with residual hardness ─────────────────
y_full = df["residual"].values

print("\n" + "="*80)
print("INDIVIDUAL FEATURE CORRELATIONS WITH RESIDUAL HARDNESS")
print("="*80)

corr_rows = []
for hyp_label, feats in [
    ("BASE", BASE_FEATS), ("H1", H1_A), ("H2", H2_A), ("H3", H3_A), ("H4", H4_A)
]:
    print(f"\n{'─'*80}")
    print(f" {hyp_label}")
    print(f"  {'Feature':<30} {'Pearson r':>10} {'p':>8}  {'Spearman ρ':>10} {'p':>8}")
    print(f"  {'─'*68}")
    for feat in feats:
        sub = df[["residual", feat]].dropna()
        if len(sub) < 10:
            continue
        pr, pp = stats.pearsonr(sub[feat], sub["residual"])
        sr, sp = stats.spearmanr(sub[feat], sub["residual"])
        sp_p = ("**" if pp < 0.01 else "*" if pp < 0.05 else "†" if pp < 0.10 else " ")
        ss_p = ("**" if sp < 0.01 else "*" if sp < 0.05 else "†" if sp < 0.10 else " ")
        print(f"  {feat:<30} {pr:>+9.3f}{sp_p}  {pp:>7.4f}  {sr:>+9.3f}{ss_p}  {sp:>7.4f}")
        corr_rows.append(dict(hypothesis=hyp_label, feature=feat,
                              pearson_r=pr, pearson_p=pp,
                              spearman_r=sr, spearman_p=sp,
                              n=len(sub)))

corr_df = pd.DataFrame(corr_rows)

# ── 6. LOO CV ablation ────────────────────────────────────────────────────────
def loo_r2(X: np.ndarray, y: np.ndarray, model_fn) -> float:
    loo = LeaveOneOut()
    preds = np.empty(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = model_fn()
        m.fit(Xtr, y[tr])
        preds[te] = m.predict(Xte)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)

def ridge_fn():  return Ridge(alpha=1.0)
def rf_fn():
    return RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                 min_samples_leaf=3, random_state=42)

# Work on the clean subset where ALL feature groups have data
all_needed = BASE_FEATS + H1_A + H2_A + H3_A + H4_A + ["residual"]
df_clean = df.dropna(subset=all_needed).copy().reset_index(drop=True)
print(f"\nSystems with complete data for ablation: {len(df_clean)}")

y = df_clean["residual"].values

print("\n" + "="*80)
print("LOO R² ABLATION")
print("="*80)
print(f"\n{'Specification':<52} {'Ridge R²':>10} {'RF R²':>10}  {'ΔRidge':>8} {'ΔRF':>8}")
print("─" * 92)

ablation_rows = []

def run_ablation(label, feats, base_ridge=None, base_rf=None):
    X = df_clean[feats].values
    r2_ri = loo_r2(X, y, ridge_fn)
    r2_rf = loo_r2(X, y, rf_fn)
    d_ri = r2_ri - base_ridge if base_ridge is not None else float("nan")
    d_rf = r2_rf - base_rf   if base_rf   is not None else float("nan")
    d_ri_s = f"{d_ri:>+8.3f}" if not np.isnan(d_ri) else "       —"
    d_rf_s = f"{d_rf:>+8.3f}" if not np.isnan(d_rf) else "       —"
    print(f"{label:<52} {r2_ri:>+9.3f}   {r2_rf:>+9.3f}  {d_ri_s} {d_rf_s}")
    ablation_rows.append(dict(spec=label, ridge_r2=r2_ri, rf_r2=r2_rf,
                              delta_ridge=d_ri, delta_rf=d_rf))
    return r2_ri, r2_rf

b_ri, b_rf = run_ablation("Base (temporal only)", BASE_FEATS)

ranking_data = []
for hyp_label, hyp_feats in [
    ("H1 Neighbourhood instability", H1_A),
    ("H2 Graph sensitivity",         H2_A),
    ("H3 Anisotropy collapse",       H3_A),
    ("H4 WavLM–GAT alignment",       H4_A),
]:
    # standalone
    run_ablation(f"  {hyp_label[:35]} standalone", hyp_feats)
    # incremental
    comb_ri, comb_rf = run_ablation(
        f"  Base + {hyp_label[:35]}", BASE_FEATS + hyp_feats,
        base_ridge=b_ri, base_rf=b_rf
    )
    ranking_data.append(dict(
        hypothesis=hyp_label,
        standalone_r2_ridge=ablation_rows[-2]["ridge_r2"],
        standalone_r2_rf=ablation_rows[-2]["rf_r2"],
        incremental_delta_ridge=comb_ri - b_ri,
        incremental_delta_rf=comb_rf - b_rf,
        mean_delta=(comb_ri - b_ri + comb_rf - b_rf) / 2,
    ))
    print()

run_ablation("All four hypotheses + base", BASE_FEATS + H1_A + H2_A + H3_A + H4_A,
             base_ridge=b_ri, base_rf=b_rf)

# ── 7. Hypothesis ranking ─────────────────────────────────────────────────────
rank_df = pd.DataFrame(ranking_data).sort_values("mean_delta", ascending=False)

print("\n" + "="*80)
print("HYPOTHESIS RANKING (by mean incremental LOO R²)")
print("="*80)
print(f"\n{'#':<3} {'Hypothesis':<32} {'ΔR²(Ridge)':>12} {'ΔR²(RF)':>10} {'Mean ΔR²':>10} {'Max |r|':>10} {'N p<.10':>8}")
print("─"*88)

for rank, row in enumerate(rank_df.itertuples(), 1):
    hyp_key = row.hypothesis.split()[0]  # H1/H2/H3/H4
    hyp_corrs = corr_df[corr_df["hypothesis"] == hyp_key]
    max_r     = hyp_corrs["pearson_r"].abs().max() if len(hyp_corrs) else 0
    n_sig     = int((hyp_corrs["pearson_p"] < 0.10).sum())
    print(f"{rank:<3} {row.hypothesis:<32} {row.incremental_delta_ridge:>+11.3f}  "
          f"{row.incremental_delta_rf:>+9.3f}  {row.mean_delta:>+9.3f}  "
          f"{max_r:>9.3f}  {n_sig:>7}")

# ── 8. Best feature per hypothesis ────────────────────────────────────────────
print("\n" + "="*80)
print("BEST FEATURES PER HYPOTHESIS (by |Pearson r|)")
print("="*80)
best_feats = {}
for hyp_label in ["H1", "H2", "H3", "H4"]:
    sub = corr_df[corr_df["hypothesis"] == hyp_label].copy()
    if sub.empty:
        continue
    sub["abs_r"] = sub["pearson_r"].abs()
    best = sub.nlargest(3, "abs_r")[["feature", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]]
    best_feats[hyp_label] = best.iloc[0]["feature"]
    print(f"\n{hyp_label}:")
    print(best.to_string(index=False))

# ── 9. Residual explanation summary ──────────────────────────────────────────
print("\n" + "="*80)
print("RESIDUAL EXPLANATION SUMMARY")
print("="*80)
print(f"  Base (temporal)  Ridge={b_ri:+.3f}  RF={b_rf:+.3f}")
for row in ablation_rows:
    if "Base +" in row["spec"]:
        d_r = row["delta_ridge"]
        d_f = row["delta_rf"]
        bar = "▓" * max(0, int(abs(d_r) * 50))
        print(f"  {row['spec']:<52} ΔRidge={d_r:+.3f}  ΔRF={d_f:+.3f}  {bar}")

# ── 10. Visualisation ─────────────────────────────────────────────────────────
HYP_COLORS = {
    "H1 Neighbourhood instability": "#4CAF50",
    "H2 Graph sensitivity":         "#2196F3",
    "H3 Anisotropy collapse":       "#FF9800",
    "H4 WavLM–GAT alignment":       "#E91E63",
}
GROUP_COLORS = {"easy": "#2196F3", "mid": "#FF9800", "hard": "#F44336"}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Residual Hardness Analysis 2: Hypothesis Attribution",
             fontsize=13, fontweight="bold", y=1.01)

# ── Panel A: Incremental LOO R² bar chart ─────────────────────────────────────
ax = axes[0, 0]
hyp_labels = [r["hypothesis"] for r in ranking_data]
x_pos = np.arange(len(hyp_labels))
widths = 0.35
delta_ri = [r["incremental_delta_ridge"] for r in ranking_data]
delta_rf = [r["incremental_delta_rf"]    for r in ranking_data]
colors   = [HYP_COLORS.get(h, "#888") for h in hyp_labels]

bars_ri = ax.bar(x_pos - widths/2, delta_ri, widths, color=colors, alpha=0.85,
                 edgecolor="white", linewidth=0.5, label="Ridge ΔR²")
bars_rf = ax.bar(x_pos + widths/2, delta_rf, widths, color=colors, alpha=0.45,
                 edgecolor="white", linewidth=0.5, hatch="///", label="RF ΔR²")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x_pos)
short_labels = ["H1\nNeighb.", "H2\nGraph", "H3\nAniso.", "H4\nAlign."]
ax.set_xticklabels([short_labels[["H1","H2","H3","H4"].index(h.split()[0])] for h in hyp_labels],
                   fontsize=10)
ax.set_ylabel("Incremental LOO R² (beyond temporal base)", fontsize=10)
ax.set_title("A — Incremental LOO R² by Hypothesis", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ── Panel B: Correlation dotplot (all features, coloured by hypothesis) ───────
ax = axes[0, 1]
all_corr_plot = corr_df[corr_df["hypothesis"] != "BASE"].copy()
all_corr_plot = all_corr_plot.sort_values("pearson_r")
hyp_color_map = {"H1": "#4CAF50", "H2": "#2196F3", "H3": "#FF9800", "H4": "#E91E63"}

y_pos = np.arange(len(all_corr_plot))
colors_dot = [hyp_color_map.get(row["hypothesis"], "#888")
              for _, row in all_corr_plot.iterrows()]
ax.barh(y_pos, all_corr_plot["pearson_r"], color=colors_dot, alpha=0.75, height=0.65)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(-0.05, color="gray", lw=0.5, ls="--", alpha=0.5)
ax.axvline(+0.05, color="gray", lw=0.5, ls="--", alpha=0.5)

feat_labels = all_corr_plot["feature"].str.replace("_", " ").str.replace("mean", "μ")
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_labels, fontsize=7)
ax.set_xlabel("Pearson r with residual hardness", fontsize=10)
ax.set_title("B — Feature Correlations (coloured by hypothesis)", fontsize=11, fontweight="bold")

patches = [mpatches.Patch(color=c, label=h) for h, c in hyp_color_map.items()]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.3)

# Mark significance
for i, (_, row) in enumerate(all_corr_plot.iterrows()):
    if row["pearson_p"] < 0.05:
        ax.text(row["pearson_r"] + (0.005 if row["pearson_r"] >= 0 else -0.005),
                i, "*", va="center", ha="left" if row["pearson_r"] >= 0 else "right",
                fontsize=10, color="black")
    elif row["pearson_p"] < 0.10:
        ax.text(row["pearson_r"] + (0.005 if row["pearson_r"] >= 0 else -0.005),
                i, "†", va="center", ha="left" if row["pearson_r"] >= 0 else "right",
                fontsize=9, color="#555")

# ── Panel C: Best H4 feature scatter (per_utt_cosine_mean) ───────────────────
ax = axes[1, 0]
key_systems = {"Kitten-TTS-Nano-0.2", "VoxCPM-1.5", "Index-TTS-1.5",
               "OuteTTS", "ZipVoice", "Veena", "orpheus-tts-0.1-finetune",
               "Marvis-TTS", "Indri-TTS-0.1"}
best_h4 = "per_utt_cosine_mean"
sub = df_clean.dropna(subset=[best_h4, "residual"])
for grp, gsub in sub.groupby("group"):
    ax.scatter(gsub[best_h4], gsub["residual"],
               c=GROUP_COLORS.get(grp, "#888"), s=55, alpha=0.82,
               edgecolors="white", linewidths=0.4, label=grp.capitalize(), zorder=3)
slope, intercept, r, p, _ = stats.linregress(sub[best_h4], sub["residual"])
xr = np.linspace(sub[best_h4].min(), sub[best_h4].max(), 100)
ax.plot(xr, slope * xr + intercept, "k--", lw=1.5, alpha=0.7,
        label=f"OLS (r={r:+.2f}, p={p:.3f})")
for _, row in sub.iterrows():
    if row["system"] in key_systems:
        ax.annotate(row["system"][:18], (row[best_h4], row["residual"]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color="#333")
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.set_xlabel("per_utt_cosine_mean (phoneme-node cosine distance)", fontsize=10)
ax.set_ylabel("Residual hardness (observed − predicted EER)", fontsize=10)
ax.set_title("C — Best H4 feature: per-utterance phoneme cosine mean", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ── Panel D: Best H2 feature scatter (var_entropy_mean) ──────────────────────
ax = axes[1, 1]
best_h2 = "var_entropy_mean"
sub2 = df_clean.dropna(subset=[best_h2, "residual"])
for grp, gsub in sub2.groupby("group"):
    ax.scatter(gsub[best_h2], gsub["residual"],
               c=GROUP_COLORS.get(grp, "#888"), s=55, alpha=0.82,
               edgecolors="white", linewidths=0.4, label=grp.capitalize(), zorder=3)
slope2, intercept2, r2c, p2c, _ = stats.linregress(sub2[best_h2], sub2["residual"])
xr2 = np.linspace(sub2[best_h2].min(), sub2[best_h2].max(), 100)
ax.plot(xr2, slope2 * xr2 + intercept2, "k--", lw=1.5, alpha=0.7,
        label=f"OLS (r={r2c:+.2f}, p={p2c:.3f})")
for _, row in sub2.iterrows():
    if row["system"] in key_systems:
        ax.annotate(row["system"][:18], (row[best_h2], row["residual"]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color="#333")
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.set_xlabel("var_entropy_mean (attention entropy variance across utterances)", fontsize=10)
ax.set_ylabel("Residual hardness (observed − predicted EER)", fontsize=10)
ax.set_title("D — Best H2 feature: attention entropy variance", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out_fig = OUT_DIR / "hypothesis_panels.png"
plt.savefig(out_fig, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")

# ── 11. Save tables ───────────────────────────────────────────────────────────
corr_df.to_csv(OUT_DIR / "hypothesis_correlations.csv", index=False, float_format="%.5f")
pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "incremental_r2_table.csv", index=False, float_format="%.5f")
rank_df.to_csv(OUT_DIR / "hypothesis_ranking.csv", index=False, float_format="%.5f")

# Full per-system table
sys_out = df_clean[["system", "group", "observed_eer", "residual"] +
                    BASE_FEATS + H1_A + H2_A + H3_A + H4_A].copy()
sys_out = sys_out.sort_values("residual", ascending=False)
sys_out.to_csv(OUT_DIR / "full_feature_table.csv", index=False, float_format="%.5f")

print(f"Tables saved → {OUT_DIR}/")
print("\n✓ Done.")
