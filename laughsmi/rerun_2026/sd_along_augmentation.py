"""Does the laughter augmentation affect sd_along (the paper's headline hardness
predictor) -- and, since the paper stresses that on few-system corpora like
ASVspoof2019 it's actually POSITION s_along (rho=-0.71) that's operative, not
spread sd_along (which only maps to hardness on 61-system MLAAD) -- what
happens to s_along too?

Convention (matches experiments/axis_audits/audit_common.py::loso_axis_features
and rerun_2026/wavlm_space_analysis.py, both already-established in this repo):
  - layer-12 WavLM-base (detector backbone), mean-pooled, 3s-cropped, RAW on disk.
  - standardize each 768-d coordinate by the BASE bona-fide mean/SD.
  - w = centroid(spoof_base) - centroid(bona_base) in standardized space, unit norm.
  - per utterance: d = z - mu_bona; s_along = d . w; s_orth = ||d - (d.w) w||.
  - per system: s_along = mean(proj), sd_along = std(proj), s_orth = mean(orth).
    (audit_common's loso_axis_features uses median for s_along/s_orth reporting;
    we report BOTH mean and median for s_along here since mean is the paper's
    literal "mean projection" definition in Setup/3.1, and note they agree in
    direction; sd_along is unambiguously std of projections.)

Three analyses, per the task brief:
  (1) Per-attack-system paired base-vs-aug (restricted to augmented==1 spoof
      files only, for apples-to-apples pairing) with Wilcoxon signed-rank
      across systems, for both s_along and sd_along (also s_orth for context).
  (2) Pooled (whole-set, ignoring system identity) base-vs-aug spoof s_along /
      sd_along / s_orth, with bootstrap CIs -- since per-system n is thin
      (5-19 utterances/system; audit_common's own MIN_UTTS=8 threshold drops
      several systems), pooled is the primary robustness check.
  (3) LOSO axis reconstruction (paper's actual w-estimation procedure: when
      scoring system S, rebuild w from bona pool + every OTHER (non-S) spoof
      system, all drawn from the BASE set) -- repeated at layers 0, 9, 12 for
      robustness.

All geometry uses the BASE-bona-fide standardization throughout (do not
re-standardize using aug-set statistics -- w and the coordinate frame must
stay fixed across base/aug so deltas are meaningful).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path

RERUN = Path(__file__).resolve().parent
MIN_UTTS = 8  # audit_common.py convention


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def standardize_and_axis(X, bona_base_mask, spoof_base_mask):
    mu_b = X[bona_base_mask].mean(0)
    sd_b = X[bona_base_mask].std(0) + 1e-9
    Z = (X - mu_b) / sd_b
    mu_bona = Z[bona_base_mask].mean(0)
    w = unit(Z[spoof_base_mask].mean(0) - mu_bona)
    return Z, mu_bona, w


def project(Z, mu_bona, w):
    d = Z - mu_bona
    a = d @ w
    orth = np.linalg.norm(d - np.outer(a, w), axis=1)
    return a, orth


def per_system_stats(a, orth, systems, sys_list):
    rows = []
    for s in sys_list:
        m = systems == s
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "system": s, "n_utts": n,
            "s_along_mean": float(np.mean(a[m])),
            "s_along_median": float(np.median(a[m])),
            "sd_along": float(np.std(a[m])),
            "s_orth": float(np.mean(orth[m])),
        })
    return pd.DataFrame(rows).set_index("system")


def bootstrap_ci(x, fn, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        vals.append(fn(x[ix]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def loso_features(Z, labels, systems, sys_list, bona_key="bona"):
    """Paper's LOSO w-estimation: rebuild w from bona pool + every OTHER
    spoof system (all within the BASE set) when scoring system s."""
    bona = labels == 0
    mu_bona = Z[bona].mean(0)
    a_out = np.full(len(Z), np.nan)
    orth_out = np.full(len(Z), np.nan)
    for s in list(sys_list) + [bona_key]:
        m = systems == s
        if m.sum() == 0:
            continue
        sp_excl = (labels == 1) & (systems != s)
        w = unit(Z[sp_excl].mean(0) - mu_bona)
        d = Z[m] - mu_bona
        a = d @ w
        a_out[m] = a
        orth_out[m] = np.linalg.norm(d - np.outer(a, w), axis=1)
    return a_out, orth_out


def analyze_layer(npz, meta, layer, base_idx_by_name, aug_spoof_df, out_prefix, verbose=True):
    X = npz[f"layer{layer}"]
    base_mask_all = (meta["set"] == "base").values
    bona_base_mask = base_mask_all & (meta["label"] == "bona-fide").values
    spoof_base_mask = base_mask_all & (meta["label"] == "spoof").values

    Z, mu_bona, w = standardize_and_axis(X, bona_base_mask, spoof_base_mask)

    # --- fixed-w (non-LOSO) projections for base spoof systems ---
    base_df = meta[meta["set"] == "base"].reset_index(drop=False)
    base_systems = base_df["system_id"].values
    base_labels = (base_df["label"] == "spoof").astype(int).values
    a_base_all, orth_base_all = project(Z[base_df["index"].values], mu_bona, w)

    sys_list = sorted(base_df.loc[base_df["label"] == "spoof", "system_id"].unique())

    base_spoof_mask_local = base_labels == 1
    base_spoof_stats = per_system_stats(
        a_base_all[base_spoof_mask_local], orth_base_all[base_spoof_mask_local],
        base_systems[base_spoof_mask_local], sys_list)

    # --- aug spoof, restricted to augmented==1, using SAME w (base-derived, fixed) ---
    aug_idx = aug_spoof_df["index"].values
    a_aug, orth_aug = project(Z[aug_idx], mu_bona, w)
    aug_systems = aug_spoof_df["system_id"].values
    aug_spoof_stats = per_system_stats(a_aug, orth_aug, aug_systems, sys_list)

    # paired per-system table (only systems present in BOTH, restricted to
    # augmented==1 aug files so counts are apples-to-apples with what aug
    # actually altered)
    # For a fair "apples-to-apples" base comparison, also compute base stats
    # restricted to the SAME basenames as the augmented==1 aug files (i.e. the
    # exact base counterparts of the aug-augmented files), not all base spoof.
    aug_basenames = set(aug_spoof_df["basename"])
    base_matched_mask = base_df["basename"].isin(aug_basenames).values & (base_labels == 1)
    base_matched_stats = per_system_stats(
        a_base_all[base_matched_mask], orth_base_all[base_matched_mask],
        base_systems[base_matched_mask], sys_list)

    merged = base_matched_stats.join(aug_spoof_stats, lsuffix="_base", rsuffix="_aug", how="inner")
    merged["delta_s_along_mean"] = merged["s_along_mean_aug"] - merged["s_along_mean_base"]
    merged["delta_s_along_median"] = merged["s_along_median_aug"] - merged["s_along_median_base"]
    merged["delta_sd_along"] = merged["sd_along_aug"] - merged["sd_along_base"]
    merged["delta_s_orth"] = merged["s_orth_aug"] - merged["s_orth_base"]
    merged["thin"] = (merged["n_utts_base"] < MIN_UTTS) | (merged["n_utts_aug"] < MIN_UTTS)

    # paired Wilcoxon across systems (all systems, note thin-data caveat)
    def paired_wilcoxon(col_base, col_aug, systems_subset=None):
        d = merged if systems_subset is None else merged.loc[systems_subset]
        if len(d) < 3:
            return np.nan, np.nan, len(d)
        try:
            stat, p = stats.wilcoxon(d[col_base], d[col_aug])
        except ValueError:
            stat, p = np.nan, np.nan
        return stat, p, len(d)

    wil_along_mean = paired_wilcoxon("s_along_mean_base", "s_along_mean_aug")
    wil_along_med = paired_wilcoxon("s_along_median_base", "s_along_median_aug")
    wil_sd_along = paired_wilcoxon("sd_along_base", "sd_along_aug")
    wil_orth = paired_wilcoxon("s_orth_base", "s_orth_aug")

    ok_systems = merged.index[~merged["thin"]]
    wil_along_mean_ok = paired_wilcoxon("s_along_mean_base", "s_along_mean_aug", ok_systems)
    wil_sd_along_ok = paired_wilcoxon("sd_along_base", "sd_along_aug", ok_systems)

    # --- pooled (whole-set) analysis ---
    a_base_matched = a_base_all[base_matched_mask]
    orth_base_matched = orth_base_all[base_matched_mask]
    pooled = {
        "layer": layer,
        "n_base_matched": int(base_matched_mask.sum()),
        "n_aug": len(aug_idx),
        "pooled_s_along_mean_base": float(np.mean(a_base_matched)),
        "pooled_s_along_mean_aug": float(np.mean(a_aug)),
        "pooled_s_along_median_base": float(np.median(a_base_matched)),
        "pooled_s_along_median_aug": float(np.median(a_aug)),
        "pooled_sd_along_base": float(np.std(a_base_matched)),
        "pooled_sd_along_aug": float(np.std(a_aug)),
        "pooled_s_orth_mean_base": float(np.mean(orth_base_matched)),
        "pooled_s_orth_mean_aug": float(np.mean(orth_aug)),
    }
    pooled["delta_pooled_s_along_mean"] = pooled["pooled_s_along_mean_aug"] - pooled["pooled_s_along_mean_base"]
    pooled["delta_pooled_sd_along"] = pooled["pooled_sd_along_aug"] - pooled["pooled_sd_along_base"]
    pooled["delta_pooled_s_orth_mean"] = pooled["pooled_s_orth_mean_aug"] - pooled["pooled_s_orth_mean_base"]

    # bootstrap CIs (utterance-level resample within group) for pooled deltas
    # and for the mean/std themselves
    ci_along_base = bootstrap_ci(a_base_matched, np.mean)
    ci_along_aug = bootstrap_ci(a_aug, np.mean)
    ci_sd_base = bootstrap_ci(a_base_matched, np.std)
    ci_sd_aug = bootstrap_ci(a_aug, np.std)
    # bootstrap CI of the delta directly via resampling both groups jointly (paired by basename)
    rng = np.random.default_rng(0)
    n_pair = min(len(a_base_matched), len(a_aug))
    # a_base_matched and a_aug are not index-aligned by basename in general
    # (matched_mask preserves base_df row order, aug_spoof_df its own order) --
    # align explicitly by basename for a paired bootstrap of the delta.
    base_a_map = dict(zip(base_df.loc[base_matched_mask, "basename"], a_base_matched))
    aug_a_map = dict(zip(aug_spoof_df["basename"], a_aug))
    common_names = sorted(set(base_a_map) & set(aug_a_map))
    ab = np.array([base_a_map[n] for n in common_names])
    aa = np.array([aug_a_map[n] for n in common_names])
    delta_pair = aa - ab
    ci_delta_along_mean = bootstrap_ci(delta_pair, np.mean)
    # bootstrap CI for delta of sd_along (std computed within each resample of the paired set)
    def delta_sd(idx_vals):
        return np.std(aa[idx_vals]) - np.std(ab[idx_vals])
    rng2 = np.random.default_rng(1)
    n = len(ab)
    deltas_sd = []
    for _ in range(10000):
        ix = rng2.integers(0, n, n)
        deltas_sd.append(np.std(aa[ix]) - np.std(ab[ix]))
    ci_delta_sd_along = (float(np.percentile(deltas_sd, 2.5)), float(np.percentile(deltas_sd, 97.5)))
    # paired t-test / wilcoxon at utterance level for s_along delta
    wil_utt = stats.wilcoxon(ab, aa) if len(ab) >= 3 else (np.nan, np.nan)

    pooled.update({
        "ci95_s_along_mean_base": ci_along_base, "ci95_s_along_mean_aug": ci_along_aug,
        "ci95_sd_along_base": ci_sd_base, "ci95_sd_along_aug": ci_sd_aug,
        "n_paired_utts_bootstrap": len(ab),
        "ci95_delta_s_along_mean_paired": ci_delta_along_mean,
        "ci95_delta_sd_along_paired": ci_delta_sd_along,
        "utt_wilcoxon_stat_along": float(wil_utt[0]) if not np.isnan(wil_utt[0]) else np.nan,
        "utt_wilcoxon_p_along": float(wil_utt[1]) if not np.isnan(wil_utt[1]) else np.nan,
    })

    # --- LOSO axis reconstruction ---
    loso_labels = base_labels  # 1 spoof / 0 bona over base_df order
    loso_systems_full = np.where(loso_labels == 1, base_systems, "bona")
    a_loso, orth_loso = loso_features(Z[base_df["index"].values], loso_labels, loso_systems_full, sys_list)
    loso_spoof_stats = per_system_stats(a_loso[base_spoof_mask_local], orth_loso[base_spoof_mask_local],
                                         base_systems[base_spoof_mask_local], sys_list)
    # also LOSO-project the aug files: for each aug file's system s, use the
    # LOSO-w built for s (bona + all OTHER base systems, excluding s), applied
    # to the aug embedding.
    a_aug_loso = np.full(len(aug_idx), np.nan)
    orth_aug_loso = np.full(len(aug_idx), np.nan)
    bona_mask_full = base_labels == 0
    mu_bona_full = Z[base_df["index"].values][bona_mask_full].mean(0)
    for i, (idx, sysid) in enumerate(zip(aug_idx, aug_systems)):
        sp_excl = (loso_labels == 1) & (base_systems != sysid)
        w_s = unit(Z[base_df["index"].values][sp_excl].mean(0) - mu_bona_full)
        d = Z[idx] - mu_bona_full
        a = float(d @ w_s)
        a_aug_loso[i] = a
        orth_aug_loso[i] = float(np.linalg.norm(d - a * w_s))
    aug_loso_stats = per_system_stats(a_aug_loso, orth_aug_loso, aug_systems, sys_list)

    loso_merged = loso_spoof_stats.join(aug_loso_stats, lsuffix="_base", rsuffix="_aug", how="inner")
    loso_merged["delta_s_along_mean"] = loso_merged["s_along_mean_aug"] - loso_merged["s_along_mean_base"]
    loso_merged["delta_sd_along"] = loso_merged["sd_along_aug"] - loso_merged["sd_along_base"]
    if len(loso_merged) >= 3:
        loso_wil_along = stats.wilcoxon(loso_merged["s_along_mean_base"], loso_merged["s_along_mean_aug"])
        loso_wil_sd = stats.wilcoxon(loso_merged["sd_along_base"], loso_merged["sd_along_aug"])
    else:
        loso_wil_along = (np.nan, np.nan)
        loso_wil_sd = (np.nan, np.nan)

    if verbose:
        print(f"\n===== layer {layer} =====")
        print(f"per-system paired table ({len(merged)} systems):")
        print(merged[["n_utts_base", "n_utts_aug", "s_along_mean_base", "s_along_mean_aug",
                       "delta_s_along_mean", "sd_along_base", "sd_along_aug", "delta_sd_along", "thin"]].round(4))
        print(f"\nWilcoxon (fixed base-w), all {len(merged)} systems:")
        print(f"  s_along_mean: stat={wil_along_mean[0]}, p={wil_along_mean[1]}")
        print(f"  s_along_median: stat={wil_along_med[0]}, p={wil_along_med[1]}")
        print(f"  sd_along: stat={wil_sd_along[0]}, p={wil_sd_along[1]}")
        print(f"  s_orth: stat={wil_orth[0]}, p={wil_orth[1]}")
        print(f"Wilcoxon restricted to n_utts>={MIN_UTTS} both sides, {len(ok_systems)} systems:")
        print(f"  s_along_mean: stat={wil_along_mean_ok[0]}, p={wil_along_mean_ok[1]}")
        print(f"  sd_along: stat={wil_sd_along_ok[0]}, p={wil_sd_along_ok[1]}")
        print("\nPooled (whole spoof set, matched basenames):")
        for k, v in pooled.items():
            print(f"  {k}: {v}")
        print(f"\nLOSO per-system ({len(loso_merged)} systems):")
        print(loso_merged[["s_along_mean_base", "s_along_mean_aug", "delta_s_along_mean",
                            "sd_along_base", "sd_along_aug", "delta_sd_along"]].round(4))
        print(f"LOSO Wilcoxon: s_along p={loso_wil_along[1]}, sd_along p={loso_wil_sd[1]}")

    return {
        "layer": layer, "w": w, "mu_bona": mu_bona,
        "per_system_table": merged,
        "wilcoxon": {
            "s_along_mean_all": wil_along_mean, "s_along_median_all": wil_along_med,
            "sd_along_all": wil_sd_along, "s_orth_all": wil_orth,
            "s_along_mean_ok": wil_along_mean_ok, "sd_along_ok": wil_sd_along_ok,
        },
        "pooled": pooled,
        "loso_per_system_table": loso_merged,
        "loso_wilcoxon": {"s_along": loso_wil_along, "sd_along": loso_wil_sd},
    }


def main():
    npz = np.load(RERUN / "embeddings" / "wavlm_base_embeddings.npz")
    meta = pd.read_csv(RERUN / "embeddings" / "wavlm_base_meta.csv")

    base_df_full = meta[meta["set"] == "base"].reset_index(drop=False)
    base_idx_by_name = dict(zip(base_df_full["basename"], base_df_full["index"]))

    aug_df = meta[meta["set"] == "aug"].reset_index(drop=False)
    aug_spoof_df = aug_df[(aug_df["label"] == "spoof") & (aug_df["augmented"].astype(str) == "1")].reset_index(drop=True)

    results = {}
    for layer in [12, 9, 0]:
        results[layer] = analyze_layer(npz, meta, layer, base_idx_by_name, aug_spoof_df, out_prefix=None)

    (RERUN / "tables").mkdir(parents=True, exist_ok=True)

    # main per-system table: layer 12, fixed-w (headline result)
    main_tab = results[12]["per_system_table"].reset_index().rename(columns={"index": "system"})
    main_tab.to_csv(RERUN / "tables" / "sd_along_augmentation.csv", index=False)
    print(f"\nwrote tables/sd_along_augmentation.csv ({len(main_tab)} systems, layer 12)")

    # pooled summary across layers, fixed-w and LOSO
    pooled_rows = []
    for layer in [12, 9, 0]:
        r = results[layer]
        row = dict(r["pooled"])
        row["wilcoxon_s_along_mean_all_p"] = r["wilcoxon"]["s_along_mean_all"][1]
        row["wilcoxon_sd_along_all_p"] = r["wilcoxon"]["sd_along_all"][1]
        row["wilcoxon_s_along_mean_ok_p"] = r["wilcoxon"]["s_along_mean_ok"][1]
        row["wilcoxon_sd_along_ok_p"] = r["wilcoxon"]["sd_along_ok"][1]
        row["loso_s_along_p"] = r["loso_wilcoxon"]["s_along"][1]
        row["loso_sd_along_p"] = r["loso_wilcoxon"]["sd_along"][1]
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(RERUN / "tables" / "sd_along_augmentation_pooled_summary.csv", index=False)
    print(f"wrote tables/sd_along_augmentation_pooled_summary.csv ({len(pooled_df)} layers)")

    # LOSO per-system table, all layers, stacked
    loso_rows = []
    for layer in [12, 9, 0]:
        t = results[layer]["loso_per_system_table"].reset_index().rename(columns={"index": "system"})
        t["layer"] = layer
        loso_rows.append(t)
    loso_df = pd.concat(loso_rows, ignore_index=True)
    loso_df.to_csv(RERUN / "tables" / "sd_along_augmentation_loso.csv", index=False)
    print(f"wrote tables/sd_along_augmentation_loso.csv")

    # ---- figure: per-system s_along and sd_along, base vs aug (layer 12, fixed-w) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RERUN / "figures").mkdir(parents=True, exist_ok=True)
    t = main_tab.sort_values("s_along_mean_base")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(t))
    ax = axes[0]
    ax.plot(x, t["s_along_mean_base"], "o-", color="#C1443C", label="base spoof")
    ax.plot(x, t["s_along_mean_aug"], "o-", color="#E8A33D", label="aug spoof (+laughter)")
    ax.set_xticks(x); ax.set_xticklabels(t["system"], rotation=45)
    ax.set_ylabel("s_along (mean projection onto w)")
    ax.set_title("Per-system s_along: base vs aug")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)

    ax = axes[1]
    ax.plot(x, t["sd_along_base"], "o-", color="#C1443C", label="base spoof")
    ax.plot(x, t["sd_along_aug"], "o-", color="#E8A33D", label="aug spoof (+laughter)")
    ax.set_xticks(x); ax.set_xticklabels(t["system"], rotation=45)
    ax.set_ylabel("sd_along (std of projection onto w)")
    ax.set_title("Per-system sd_along: base vs aug")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RERUN / "figures" / "sd_along_augmentation_per_system.png", dpi=150)
    plt.close(fig)
    print("wrote figures/sd_along_augmentation_per_system.png")

    # pooled bar figure
    p12 = results[12]["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    ax = axes[0]
    ax.bar(["base", "aug"], [p12["pooled_s_along_mean_base"], p12["pooled_s_along_mean_aug"]],
           color=["#C1443C", "#E8A33D"])
    ax.set_ylabel("pooled s_along (mean proj. onto w)")
    ax.set_title(f"Pooled s_along (n_base={p12['n_base_matched']}, n_aug={p12['n_aug']})")
    ax = axes[1]
    ax.bar(["base", "aug"], [p12["pooled_sd_along_base"], p12["pooled_sd_along_aug"]],
           color=["#C1443C", "#E8A33D"])
    ax.set_ylabel("pooled sd_along (std of proj. onto w)")
    ax.set_title("Pooled sd_along")
    fig.tight_layout()
    fig.savefig(RERUN / "figures" / "sd_along_augmentation_pooled.png", dpi=150)
    plt.close(fig)
    print("wrote figures/sd_along_augmentation_pooled.png")


if __name__ == "__main__":
    main()
