"""
Manifold Geometry Analysis
---------------------------
Extracts utterance-level WavLM embeddings (post_wavlm: mean over T frames of
feature_projection output, 768-dim) for every spoof system's utterances, then
computes manifold geometry statistics per system and tests whether compactness
predicts residual hardness beyond temporal smoothness.

Architecture note:
  audio → [CNN feature_extractor] → [feature_projection] ← post_wavlm here
                                   → [12-layer WavLM transformer] → ...

Outputs → experiments/results/mlaad/manifold_geometry/
"""

from __future__ import annotations

import json
import sys
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPTS_DIR.parents[1]
EXP_DIR       = PROJECT_ROOT / "experiments"
CKPT_DIR      = EXP_DIR / "checkpoints"
PROCESSED_DIR = EXP_DIR / "data" / "mlaad_tiny_processed"
INDIST_JSON   = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
RES_DIR       = EXP_DIR / "results" / "mlaad"
OUT_DIR       = RES_DIR / "manifold_geometry"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ─── Load model and extract post_wavlm features ───────────────────────────────
def load_model():
    """Load pretrained WavLM-base directly via transformers.
    post_wavlm features use only the frozen CNN + feature_projection
    (before the 12-layer transformer), so pretrained weights are correct.
    """
    from transformers import WavLMForCTC
    print("  Loading microsoft/wavlm-base (frozen CNN + feature_projection)")
    model = WavLMForCTC.from_pretrained("microsoft/wavlm-base")
    model = model.wavlm  # WavLMModel: feature_extractor + feature_projection + encoder
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def extract_post_wavlm(audio_tensor: torch.Tensor, wavlm_model) -> np.ndarray:
    """
    audio_tensor: (N, L) float32 at 16kHz
    Returns (N, 768) mean-pooled post-CNN feature_projection embeddings.
    feature_projection returns (hidden_states, norm_hidden_states) tuple;
    we take index [0] (the 768-dim projected representation).
    """
    wav = audio_tensor.to(DEVICE)
    fe_out  = wavlm_model.feature_extractor(wav)   # (N, 512, T')
    fe_out  = fe_out.transpose(1, 2)               # (N, T', 512)
    fp_out  = wavlm_model.feature_projection(fe_out)  # tuple: [(N,T',768), (N,T',512)]
    hidden  = fp_out[0]                            # (N, T', 768)
    pooled  = hidden.mean(dim=1)                   # (N, 768)
    return pooled.cpu().float().numpy()


def extract_all_embeddings(records: list[dict], phoneme_model) -> dict[str, np.ndarray]:
    """Returns {system_id: (n_utts, 768)}."""
    spoof = [r for r in records if r["label"] == "spoof"]
    by_sys: dict[str, list] = defaultdict(list)
    for rec in spoof:
        by_sys[rec["attack_system"]].append(rec)

    all_embs: dict[str, list] = defaultdict(list)

    # process in batches
    items = [(sys_id, rec) for sys_id, recs in by_sys.items() for rec in recs]
    n     = len(items)
    for start in range(0, n, BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        audios, sys_ids = [], []
        for sys_id, rec in batch:
            wav = torch.load(PROCESSED_DIR / rec["audio_path"])  # (L,)
            # Pad/trim to 3s @ 16kHz = 48000 samples
            target = 48000
            if wav.shape[-1] < target:
                wav = F.pad(wav, (0, target - wav.shape[-1]))
            else:
                wav = wav[:target]
            audios.append(wav)
            sys_ids.append(sys_id)

        audio_batch = torch.stack(audios)  # (B, 48000)
        embs = extract_post_wavlm(audio_batch, phoneme_model)
        for sid, emb in zip(sys_ids, embs):
            all_embs[sid].append(emb)

        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  Extracted {min(start+BATCH_SIZE, n)}/{n} utterances", flush=True)

    return {k: np.stack(v) for k, v in all_embs.items()}


# ─── Manifold geometry metrics ────────────────────────────────────────────────
def effective_rank(cov_eigvals: np.ndarray) -> float:
    """Effective rank via entropy of normalised eigenvalue distribution."""
    ev = np.maximum(cov_eigvals, 0)
    total = ev.sum()
    if total < 1e-12:
        return 1.0
    p = ev / total
    p = p[p > 1e-12]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def intrinsic_dim_twonn(X: np.ndarray) -> float:
    """Two-NN intrinsic dimensionality estimator (Facco et al. 2017)."""
    n = X.shape[0]
    if n < 4:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)  # (n, 3): col 0=self if included
    # take 1st and 2nd nearest neighbours (skip self at index 0)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = (r1 > 1e-12)
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1.0]
    if len(mu) < 2:
        return float("nan")
    return float(1.0 / np.mean(np.log(mu)))


def knn_density(X: np.ndarray, k: int = 3) -> float:
    """Mean kNN inverse distance as density proxy (higher = denser)."""
    n = X.shape[0]
    if n <= k:
        k = max(1, n - 1)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    knn_d = dists[:, 1:k+1].mean(axis=1)  # skip self
    return float(1.0 / (knn_d.mean() + 1e-8))


def log_det_cov(eigvals: np.ndarray) -> float:
    """Log-determinant of the non-zero-eigenvalue subspace."""
    ev = eigvals[eigvals > 1e-6]
    if len(ev) == 0:
        return float("nan")
    return float(np.sum(np.log(ev)))


def compute_manifold_metrics(X: np.ndarray) -> dict:
    """X: (n, 768) utterance embeddings for one system."""
    n, d = X.shape
    centroid = X.mean(axis=0)

    # Distance to centroid
    dists_to_ctr = np.linalg.norm(X - centroid, axis=1)  # (n,)

    # Pairwise distances (upper triangle only for speed)
    if n > 1:
        pw_dists = cdist(X, X, metric="euclidean")
        triu = pw_dists[np.triu_indices(n, k=1)]
        avg_pairwise = float(triu.mean()) if len(triu) > 0 else float("nan")
    else:
        avg_pairwise = float("nan")

    # Cosine distances to centroid
    ctr_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    X_norm   = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos_sims  = X_norm @ ctr_norm
    cos_dists = 1 - cos_sims

    # Covariance statistics (use regularized empirical cov)
    if n > 1:
        Xc = X - centroid
        cov = (Xc.T @ Xc) / (n - 1)  # (768, 768)
        cov_trace = float(np.trace(cov))
        # PCA eigenvalues (faster: use SVD of centred matrix)
        _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        eigvals = sv**2 / (n - 1)
        eff_rank = effective_rank(eigvals)
        logdet   = log_det_cov(eigvals)
        # PCA explained variance
        total_var = eigvals.sum()
        if total_var > 1e-12:
            cum_var = np.cumsum(eigvals) / total_var
            pca_k80 = int(np.searchsorted(cum_var, 0.80)) + 1
            pca_k95 = int(np.searchsorted(cum_var, 0.95)) + 1
            pca_var1 = float(eigvals[0] / total_var)
        else:
            pca_k80, pca_k95, pca_var1 = 1, 1, 1.0
    else:
        cov_trace = float("nan")
        eff_rank = 1.0
        logdet = float("nan")
        pca_k80 = pca_k95 = 1
        pca_var1 = 1.0
        eigvals = np.array([])

    # Intrinsic dimensionality (two-NN)
    id_twonn = intrinsic_dim_twonn(X) if n >= 4 else float("nan")

    # kNN density
    knn_d = knn_density(X, k=min(3, n-1)) if n >= 2 else float("nan")

    return {
        # Distance to centroid (L2)
        "dist_to_ctr_mean":   float(dists_to_ctr.mean()),
        "dist_to_ctr_median": float(np.median(dists_to_ctr)),
        "dist_to_ctr_var":    float(dists_to_ctr.var()),
        # Cosine to centroid
        "cos_dist_to_ctr_mean": float(cos_dists.mean()),
        "cos_dist_to_ctr_var":  float(cos_dists.var()),
        # Pairwise
        "avg_pairwise_dist": avg_pairwise,
        # Covariance
        "cov_trace":  cov_trace,
        "log_det_cov": logdet,
        # Dimensionality
        "effective_rank": eff_rank,
        "pca_k80":    float(pca_k80),
        "pca_k95":    float(pca_k95),
        "pca_var1":   pca_var1,
        # Density / compactness
        "knn_density":    knn_d,
        "intrinsic_dim":  id_twonn,
        "n_utts":         n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("MANIFOLD GEOMETRY ANALYSIS — WavLM Representation Compactness")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(INDIST_JSON) as f:
        records = json.load(f)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\nLoading WavLM model...")
    wavlm_model = load_model()
    print(f"  Device: {DEVICE}")

    # ── Extract embeddings ─────────────────────────────────────────────────────
    emb_cache = OUT_DIR / "system_embeddings.npz"
    if emb_cache.exists():
        print("\nLoading cached embeddings...")
        d = np.load(str(emb_cache), allow_pickle=True)
        sys_embs = {k: d[k] for k in d.files}
    else:
        print("\nExtracting post_wavlm embeddings...")
        sys_embs = extract_all_embeddings(records, wavlm_model)

        np.savez(str(emb_cache), **sys_embs)
        print(f"  Cached → {emb_cache}")

    print(f"\n  Systems with embeddings: {len(sys_embs)}")
    for k, v in sorted(sys_embs.items())[:5]:
        print(f"  {k}: {v.shape}")

    # ── Compute manifold metrics ───────────────────────────────────────────────
    print("\nComputing manifold geometry metrics per system...")
    metrics_list = []
    for sys_id in sorted(sys_embs.keys()):
        X = sys_embs[sys_id]
        m = compute_manifold_metrics(X)
        m["system"] = sys_id
        metrics_list.append(m)
        print(f"  {sys_id:<35} n={m['n_utts']:>2}  "
              f"dist_ctr={m['dist_to_ctr_mean']:.3f}  "
              f"eff_rank={m['effective_rank']:.2f}  "
              f"knn_d={m['knn_density']:.4f}")

    manifold_df = pd.DataFrame(metrics_list)
    manifold_df.to_csv(OUT_DIR / "manifold_metrics.csv", index=False, float_format="%.5f")
    print(f"\nManifold metrics saved → {OUT_DIR / 'manifold_metrics.csv'}")

    # ── Load residual hardness table ───────────────────────────────────────────
    residual_df = pd.read_csv(RES_DIR / "residual_hardness" / "residual_hardness_table.csv")
    temporal_df = pd.read_csv(RES_DIR / "higher_order_mechanism_analysis" / "phoneme_variance.csv")
    temporal_df = temporal_df.rename(columns={"system_id": "system"})
    frame_df    = pd.read_csv(RES_DIR / "higher_order_mechanism_analysis" / "frame_variance.csv")
    frame_df    = frame_df.rename(columns={"system_id": "system"})

    df = residual_df.merge(manifold_df, on="system", how="inner")
    df = df.merge(temporal_df[["system","phoneme_var_mean","phoneme_var_median"]], on="system", how="left")
    df = df.merge(frame_df[["system","frame_mean_dist_mean"]], on="system", how="left")
    print(f"\n  Systems in merged dataset: {len(df)}")

    MANIFOLD_FEATS = [
        "dist_to_ctr_mean", "dist_to_ctr_median", "dist_to_ctr_var",
        "cos_dist_to_ctr_mean", "cos_dist_to_ctr_var",
        "avg_pairwise_dist", "cov_trace", "log_det_cov",
        "effective_rank", "pca_k80", "pca_k95", "pca_var1",
        "knn_density", "intrinsic_dim",
    ]
    # Smoothness features may come with _x/_y suffixes after merge; detect them
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        return None
    pvar_col  = find_col(df, ["phoneme_var_mean", "phoneme_var_mean_x", "phoneme_var_mean_y"])
    pvmed_col = find_col(df, ["phoneme_var_median", "phoneme_var_median_x"])
    fdst_col  = find_col(df, ["frame_mean_dist_mean", "frame_mean_dist_mean_x"])
    SMOOTH_FEATS = [c for c in [pvar_col, pvmed_col, fdst_col] if c is not None]
    print(f"  Smooth feature columns found: {SMOOTH_FEATS}")

    # ════════════════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Correlation with Residual Hardness
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ANALYSIS 1 — MANIFOLD METRICS vs. RESIDUAL HARDNESS")
    print("="*70)
    print(f"\n{'Feature':<28} {'Pearson r':>10}  {'p':>8}  {'Spearman ρ':>11}  {'p':>8}")
    print("-"*72)

    corr_results = []
    for feat in MANIFOLD_FEATS:
        sub = df[["residual", feat]].dropna()
        if len(sub) < 5:
            continue
        pr, pp = stats.pearsonr(sub["residual"], sub[feat])
        sr, sp = stats.spearmanr(sub["residual"], sub[feat])
        sig = "*" if pp < 0.05 else ("†" if pp < 0.10 else "")
        print(f"{feat:<28} {pr:>+10.3f}  {pp:>8.4f}{sig}  {sr:>+11.3f}  {sp:>8.4f}")
        corr_results.append({"feature": feat, "pearson_r": pr, "pearson_p": pp,
                              "spearman_r": sr, "spearman_p": sp})

    print("\n* p<0.05  † p<0.10")

    # ════════════════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Residual-Hard vs Residual-Easy Comparison
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ANALYSIS 2 — RESIDUAL-HARD vs. RESIDUAL-EASY COMPARISON")
    print("="*70)

    q75 = df["residual"].quantile(0.75)
    q25 = df["residual"].quantile(0.25)
    hard_grp = df[df["residual"] >= q75]
    easy_grp = df[df["residual"] <= q25]
    print(f"\n  Threshold: hard ≥ Q75={q75:+.4f}, easy ≤ Q25={q25:+.4f}")
    print(f"  n_hard={len(hard_grp)}, n_easy={len(easy_grp)}")

    print(f"\n  {'Feature':<28} {'Hard mean':>10}  {'Easy mean':>10}  "
          f"{'Cohen d':>8}  {'Perm p':>8}  {'Direction'}")
    print("  " + "-"*82)

    comparison_results = []
    for feat in MANIFOLD_FEATS:
        h = hard_grp[feat].dropna().values
        e = easy_grp[feat].dropna().values
        if len(h) < 2 or len(e) < 2:
            continue
        # Cohen's d
        pooled_std = np.sqrt((h.var(ddof=1) + e.var(ddof=1)) / 2)
        d_val = (h.mean() - e.mean()) / (pooled_std + 1e-12)
        # Permutation test
        obs = abs(h.mean() - e.mean())
        combined = np.concatenate([h, e])
        rng = np.random.default_rng(42)
        count = 0
        N_PERM = 5000
        for _ in range(N_PERM):
            perm = rng.permutation(combined)
            count += abs(perm[:len(h)].mean() - perm[len(h):].mean()) >= obs
        perm_p = count / N_PERM
        direction = "hard<easy" if h.mean() < e.mean() else "hard>easy"
        sig = "*" if perm_p < 0.05 else ("†" if perm_p < 0.10 else "")
        print(f"  {feat:<28} {h.mean():>10.4f}  {e.mean():>10.4f}  "
              f"{d_val:>+8.3f}  {perm_p:>8.4f}{sig}  {direction}")
        comparison_results.append({"feature": feat, "hard_mean": h.mean(),
                                    "easy_mean": e.mean(), "cohen_d": d_val,
                                    "perm_p": perm_p})

    print("\n* p<0.05  † p<0.10")

    # ════════════════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Predictive Modeling
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ANALYSIS 3 — PREDICTIVE MODELING: residual hardness")
    print("="*70)

    # Drop rows with any NaN in key features
    feat_cols_manifold = [f for f in MANIFOLD_FEATS if df[f].notna().sum() >= 40]
    feat_cols_smooth   = SMOOTH_FEATS
    feat_cols_combined = feat_cols_manifold + feat_cols_smooth

    y_target = "residual"
    sub_m = df[feat_cols_manifold + [y_target]].dropna()
    sub_s = df[feat_cols_smooth   + [y_target]].dropna()
    sub_c = df[feat_cols_combined + [y_target]].dropna()

    loo = LeaveOneOut()

    def run_loo(X, y, label, feat_names):
        n = len(y)
        results = {}
        for name, estimator in [
            ("LinearRegression", LinearRegression()),
            ("Lasso",            Lasso(alpha=0.01, max_iter=5000)),
            ("RandomForest",     RandomForestRegressor(n_estimators=300, max_depth=4,
                                                       random_state=42)),
        ]:
            preds = np.zeros(n)
            for tr, te in loo.split(X):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr])
                Xte = sc.transform(X[te])
                clone = type(estimator)(**estimator.get_params())
                clone.fit(Xtr, y[tr])
                preds[te] = clone.predict(Xte)
            r2  = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            results[name] = {"r2": r2, "mae": mae}
        return results

    print("\n  Target: residual hardness")
    print(f"\n  {'Model set':<20} {'Model':<20} {'LOO R²':>8}  {'LOO MAE':>9}")
    print("  " + "-"*60)

    all_model_results = {}
    for label, sub, feats in [
        ("manifold-only", sub_m, feat_cols_manifold),
        ("smoothness-only", sub_s, feat_cols_smooth),
        ("combined",       sub_c, feat_cols_combined),
    ]:
        X = sub[feats].values
        y = sub[y_target].values
        res = run_loo(X, y, label, feats)
        all_model_results[label] = res
        for model_name, vals in res.items():
            print(f"  {label:<20} {model_name:<20} {vals['r2']:>8.3f}  {vals['mae']:>9.4f}")

    # Feature importance from combined RF
    sub_c_full = df[feat_cols_combined + [y_target]].dropna()
    X_c = sub_c_full[feat_cols_combined].values
    y_c = sub_c_full[y_target].values
    sc_c = StandardScaler()
    rf_c = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=42)
    rf_c.fit(sc_c.fit_transform(X_c), y_c)
    importances = pd.Series(rf_c.feature_importances_, index=feat_cols_combined)
    importances = importances.sort_values(ascending=False)

    print("\n  Random Forest feature importance (combined model):")
    for feat, imp in importances.items():
        marker = " ◄ [manifold]" if feat in feat_cols_manifold else " ◄ [smooth]"
        print(f"    {feat:<30} {imp:.4f}{marker}")

    # ════════════════════════════════════════════════════════════════════════════
    # SPOTLIGHT: Key systems
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SPOTLIGHT SYSTEMS")
    print("="*70)
    spotlight = ["FireRedTTS-2.0", "Index-TTS-1.5", "Spark-TTS-0.5B",
                 "OuteTTS", "VoxCPM-1.5", "ZipVoice",
                 "Kitten-TTS-Nano-0.2", "orpheus-tts-0.1-finetune",
                 "Indri-TTS-0.1", "Metavoice-1B"]
    print(f"\n  {'System':<32} {'Resid':>7}  {'dist_ctr':>9}  {'avg_pw':>8}  "
          f"{'cov_tr':>8}  {'eff_rk':>7}  {'knn_d':>8}  {'id':>6}")
    print("  " + "-"*95)
    for s in spotlight:
        row = df[df["system"].str.lower() == s.lower()]
        if len(row) == 0:
            row = df[df["system"].str.contains(s.split("-")[0], case=False, na=False)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        print(f"  {row['system']:<32} {row['residual']:>+7.4f}  "
              f"{row['dist_to_ctr_mean']:>9.4f}  "
              f"{row['avg_pairwise_dist']:>8.4f}  "
              f"{row['cov_trace']:>8.2f}  "
              f"{row['effective_rank']:>7.2f}  "
              f"{row['knn_density']:>8.4f}  "
              f"{row['intrinsic_dim']:>6.2f}")

    # ════════════════════════════════════════════════════════════════════════════
    # Hypothesis test: compact manifold → residual hard?
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: compact manifold ↔ residual hardness")
    print("="*70)

    compact_metrics = ["dist_to_ctr_mean", "avg_pairwise_dist", "cov_trace",
                       "effective_rank", "knn_density"]
    print(f"\n  Note: dist_to_ctr_mean and avg_pairwise_dist measure SPREAD.")
    print(f"  Compact manifold hypothesis predicts NEGATIVE r with residual hardness:")
    print(f"  (smaller spread → harder residual → higher residual)\n")

    for feat in compact_metrics:
        sub = df[[feat, "residual"]].dropna()
        pr, pp = stats.pearsonr(sub[feat], sub["residual"])
        sr, sp = stats.spearmanr(sub[feat], sub["residual"])
        direction = "compact→hard ✓" if pr < 0 else "compact→easy ✗"
        sig = "*" if pp < 0.05 else ("†" if pp < 0.10 else "")
        print(f"  {feat:<28} r={pr:>+.3f} (p={pp:.4f}{sig})  ρ={sr:>+.3f}  {direction}")

    # ════════════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ════════════════════════════════════════════════════════════════════════════
    GROUP_COLOR  = {"easy": "#2196F3", "mid": "#FF9800", "hard": "#F44336"}
    GROUP_MARKER = {"easy": "o", "mid": "s", "hard": "^"}
    KEY_SYSTEMS  = {"FireRedTTS-2.0", "Index-TTS-1.5", "Spark-TTS-0.5B", "OuteTTS",
                    "VoxCPM-1.5", "ZipVoice", "Kitten-TTS-Nano-0.2",
                    "orpheus-tts-0.1-finetune", "Metavoice-1B", "Indri-TTS-0.1"}

    def scatter_panel(ax, x_col, y_col, df, xlabel, ylabel, title, annotate=True):
        for grp, sub in df.groupby("group"):
            ax.scatter(sub[x_col], sub[y_col],
                       c=GROUP_COLOR.get(grp, "#888"),
                       marker=GROUP_MARKER.get(grp, "o"),
                       s=55, alpha=0.82, zorder=3,
                       edgecolors="white", linewidths=0.4,
                       label=grp.capitalize())
        mask = df[[x_col, y_col]].notna().all(axis=1)
        xs, ys = df.loc[mask, x_col], df.loc[mask, y_col]
        if len(xs) > 2:
            slope, intercept, r, p, _ = stats.linregress(xs, ys)
            xr = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xr, slope*xr+intercept, "k--", lw=1.4, alpha=0.7,
                    label=f"OLS r={r:+.2f} (p={p:.3f})")
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        if annotate:
            for _, row in df.iterrows():
                if any(k.lower() in row["system"].lower()
                       for k in KEY_SYSTEMS) and pd.notna(row.get(x_col)):
                    label = row["system"][:12]
                    ax.annotate(label, (row[x_col], row[y_col]),
                                xytext=(4, 3), textcoords="offset points", fontsize=6.5)

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel A: dist_to_ctr_mean vs residual
    scatter_panel(fig.add_subplot(gs[0, 0]),
                  "dist_to_ctr_mean", "residual", df,
                  "Mean dist. to centroid (L2)", "Residual Hardness",
                  "Centroid Distance vs. Residual Hardness")

    # Panel B: cov_trace vs residual
    scatter_panel(fig.add_subplot(gs[0, 1]),
                  "cov_trace", "residual", df,
                  "Covariance Trace", "Residual Hardness",
                  "Covariance Trace vs. Residual Hardness")

    # Panel C: effective_rank vs residual
    scatter_panel(fig.add_subplot(gs[0, 2]),
                  "effective_rank", "residual", df,
                  "Effective Rank", "Residual Hardness",
                  "Effective Rank vs. Residual Hardness")

    # Panel D: avg_pairwise_dist vs residual
    scatter_panel(fig.add_subplot(gs[1, 0]),
                  "avg_pairwise_dist", "residual", df,
                  "Avg. Pairwise Distance", "Residual Hardness",
                  "Pairwise Distance vs. Residual Hardness")

    # Panel E: knn_density vs residual
    scatter_panel(fig.add_subplot(gs[1, 1]),
                  "knn_density", "residual", df,
                  "kNN Density (higher=denser)", "Residual Hardness",
                  "kNN Density vs. Residual Hardness")

    # Panel F: PCA projection of systems coloured by residual
    ax_pca = fig.add_subplot(gs[1, 2])
    manifold_feat_matrix = df[feat_cols_manifold].dropna()
    pca_idx = manifold_feat_matrix.index
    pca = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(StandardScaler().fit_transform(manifold_feat_matrix.values))
    resid_vals = df.loc[pca_idx, "residual"].values
    sc_pca = ax_pca.scatter(Xpca[:, 0], Xpca[:, 1], c=resid_vals,
                             cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                             s=65, alpha=0.85, edgecolors="white", linewidths=0.4)
    plt.colorbar(sc_pca, ax=ax_pca, label="Residual Hardness", shrink=0.85)
    for i, (idx, row) in enumerate(df.loc[pca_idx].iterrows()):
        if any(k.lower() in row["system"].lower() for k in KEY_SYSTEMS):
            ax_pca.annotate(row["system"][:12],
                            (Xpca[list(pca_idx).index(idx), 0],
                             Xpca[list(pca_idx).index(idx), 1]),
                            xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
    ax_pca.set_title("PCA of Manifold Features\n(colored by Residual Hardness)", fontsize=9)
    ax_pca.grid(True, alpha=0.25)

    fig.suptitle("Manifold Geometry Analysis — Does Representation Compactness\n"
                 "Explain Residual Hardness Beyond Temporal Smoothness?",
                 fontsize=12, fontweight="bold", y=1.01)

    out_fig = OUT_DIR / "manifold_geometry_panels.png"
    plt.savefig(str(out_fig), dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out_fig}")

    # ─── Summary table ───────────────────────────────────────────────────────
    summary_df = df[["system","group","observed_eer","residual",
                      "dist_to_ctr_mean","avg_pairwise_dist",
                      "cov_trace","effective_rank","knn_density","intrinsic_dim",
                      pvar_col, fdst_col]].sort_values("residual")
    summary_df.to_csv(OUT_DIR / "residual_manifold_summary.csv", index=False, float_format="%.4f")
    print(f"Summary table → {OUT_DIR / 'residual_manifold_summary.csv'}")

    # ─── Print variance explanation summary ──────────────────────────────────
    print("\n" + "="*70)
    print("VARIANCE EXPLANATION SUMMARY")
    print("="*70)
    print(f"\n  Smoothness-only  (LR/Lasso/RF mean): "
          f"{np.mean([v['r2'] for v in all_model_results['smoothness-only'].values()]):.3f}")
    print(f"  Manifold-only    (LR/Lasso/RF mean): "
          f"{np.mean([v['r2'] for v in all_model_results['manifold-only'].values()]):.3f}")
    print(f"  Combined         (LR/Lasso/RF mean): "
          f"{np.mean([v['r2'] for v in all_model_results['combined'].values()]):.3f}")

    best_manifold = max(all_model_results["manifold-only"].items(), key=lambda x: x[1]["r2"])
    best_combined = max(all_model_results["combined"].items(),       key=lambda x: x[1]["r2"])
    print(f"\n  Best manifold-only model: {best_manifold[0]}  R²={best_manifold[1]['r2']:.3f}")
    print(f"  Best combined model:      {best_combined[0]}  R²={best_combined[1]['r2']:.3f}")


if __name__ == "__main__":
    main()
