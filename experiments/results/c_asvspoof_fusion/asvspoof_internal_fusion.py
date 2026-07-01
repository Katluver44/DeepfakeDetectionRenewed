#!/usr/bin/env python3
"""
Workstream C — ASVspoof-internal-axis fusion DEMO (WavLM-GAT / robust_goat)
============================================================================
CPU-only. No torch, no GPU, no new detector inference: everything is loaded
from cached .npz/.csv artifacts already produced by the I/J experiment series.

WHAT THIS IS
------------
A second-corpus replication of the I7 "frozen-axis score fusion" recipe
(see experiments/scripts/i7_axis_fusion.py, verdict P5, EER 0.272->0.163 on
MLAAD), run on ASVspoof data instead of MLAAD.

  Detector : robust_goat  (checkpoints/robust_goat*.ckpt)
             *** SAME ARCHITECTURE as the MLAAD detector: WavLM-L12 encoder +
             phoneme-GAT head ("WavLM-GAT"). The ONLY difference from the
             MLAAD detector used in I7 is the TRAINING DOMAIN: robust_goat is
             trained on ASVspoof/robust data, mlaad_robust_goat is trained on
             MLAAD. This is the SAME-ARCHITECTURE / DIFFERENT-TRAINING-DOMAIN
             cell of the verdict's cross-detector-agreement matrix (audit7:
             "aasist_zs~robust_goat, both ASVspoof-trained" is the OTHER
             same-domain pair; robust_goat itself IS WavLM-GAT, not a
             different architecture).
  Corpus   : ASVspoof 2019 LA eval, attacks A07-A19 (13 systems), the exact
             1600-utterance (800 bona / 800 spoof) selection used by
             experiments/scripts/i1_geometry_causal_decomp.py and replicated
             by i4_asvspoof_position.py. Embeddings are frozen WavLM-L12
             mean-pooled (768-d), identical extraction recipe to I3/I7.

WHY THIS CORPUS/CACHE AND NOT experiments/results/j4_asvspoof21/logits.npz
---------------------------------------------------------------------------
j4_asvspoof21/logits.npz scores ASVspoof-2021-LA-clean audio with the
MLAAD-TRAINED detector (checkpoints/mlaad_robust_goat*.ckpt) -- that is the
*opposite* cross-domain direction (MLAAD-architecture-and-domain evaluated
out-of-domain on ASVspoof), not what we want here. Similarly,
j2_prospective/robustgoat_mlaad_logits.npz is robust_goat (ASVspoof-trained)
scored OUT-of-domain on MLAAD. Neither pairs the ASVspoof-*trained* robust_goat
detector with ASVspoof-corpus embeddings at the per-utterance level.

The one cache that does is experiments/results/i1_geometry_causal_decomp/
i1_logits.npz (robust_goat 3 seeds' *baseline* per-utterance logits, keyed
"s1__baseline"/"s3__baseline"/"s7__baseline") matched 1:1 by construction
with experiments/results/i4_asvspoof_position/features.npz (X12 = frozen
WavLM-L12 embeddings for the SAME 1600-utterance selection -- i4's script
asserts `lz["labels"] == labels` against this exact selection). Both derive
from the same deterministic seeded selection over the "Bisher/as_vspoof_2019_la"
HF dataset (system_id column only, no audio download needed -- the arrow
cache is already on disk under data/asvspoof_2019_la/, used here in
HF_DATASETS_OFFLINE=1 mode with the project venv's `datasets` package, purely
to recover which of the 1600 cached embeddings/logits belongs to which
attack system; no network access, no new GPU compute).

This script reconstructs that `attacks` array once (deterministic, verified
byte-identical against experiments/results/i4_asvspoof_position/attack_table.csv
per-attack hardness, max abs diff 8.9e-17) and caches it locally so re-runs
never need `datasets`/network again.

WHY THIS IS A DEMO, NOT A SECOND TEST OF THE HARDNESS LAW
-----------------------------------------------------------
n_systems = 13 (A07-A19). The MLAAD hardness law (verdict P1) was established
and stress-tested at n=61 systems; 13 systems is far too few to power a
LOSO-R^2 / permutation test of the sd_along->hardness law (see verdict S3 for
what happens to that law even at n~20-60 on a *different* detector). This
script does NOT attempt to re-test the law. It only asks the narrower,
adequately-powered question I7 asked on MLAAD: "does fusing the frozen
internal-axis projection with the fine-tuned detector's logit reduce EER,
in a system-disjoint / no-leakage evaluation?" -- a fusion demo, evaluated at
the utterance level (n=1600), not a system-level regression (n=13).

METHOD (mirrors experiments/scripts/i7_axis_fusion.py exactly)
-----------------------------------------------------------------
  1. 5-fold, SYSTEM-DISJOINT splits over the 13 attacks (bona split into
     independent random folds, same recipe as I7).
  2. Per fold: standardize embeddings on the TRAIN fold's bona pool; axis
     w = centroid(train-fold spoof) - centroid(train-fold bona), unit-norm.
     This is an UNSUPERVISED axis (only uses bona/spoof labels, no detector
     supervision, no held-out-fold information) -- identical construction to
     I7/I3/I4's LOAO axis.
  3. lambda in {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3} selected by train-fold AUC
     of z(logit) + lambda*z(axis_proj); fold-test utterances scored with
     that fold's fitted axis/lambda/standardization (no leakage).
  4. Repeated for the 3 robust_goat seeds (s1, s3, s7).
  5. Report: detector-alone EER, axis-alone EER, fused EER (per seed and
     mean), per-fold dEER (fused - baseline), and cos(w_ASVspoof_internal,
     w_MLAAD) to document the axis-rotation phenomenon (verdict P3): the
     ASVspoof-internal axis must NOT be assumed to equal the MLAAD axis.

Outputs -> experiments/results/c_asvspoof_fusion/
  - asvspoof_fusion_results.json
  - asvspoof_fusion_per_fold.csv
  - asvspoof_fusion_scores.npz  (raw scores, for independent re-audit)
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]           # repo root
RES = BASE / "experiments" / "results"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

I1DIR = RES / "i1_geometry_causal_decomp"
I4DIR = RES / "i4_asvspoof_position"
I3DIR = RES / "i3_position_geometry"          # for MLAAD axis (rotation check)
J4DIR = RES / "j4_asvspoof21"                 # for ASVspoof-2021 embeddings (rotation check, secondary corpus)
DATA_CACHE = BASE / "data" / "asvspoof_2019_la"
ATTACKS_CACHE = OUT / "_asvspoof2019_attacks_cache.npy"

EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]
SEED_NAMES = ["s1", "s3", "s7"]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_FOLDS = 5


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


# ─── 1. load cached embeddings + robust_goat logits (ASVspoof 2019 LA) ────────
print("[C] loading cached ASVspoof 2019 LA (i4) embeddings + (i1) robust_goat logits ...")
feat = np.load(I4DIR / "features.npz")
X = feat["X12"].astype(np.float64)                      # (1600, 768) frozen WavLM-L12

lz = np.load(I1DIR / "i1_logits.npz")
labels = lz["labels"]                                    # (1600,) 0=bona,1=spoof
logit_seeds = {s: lz[f"{s}__baseline"] for s in SEED_NAMES}   # robust_goat, 3 seeds
assert X.shape[0] == labels.shape[0] == 1600, "unexpected utterance count in cached arrays"

# ─── 2. recover per-utterance attack system id (deterministic, offline, CPU) ──
if ATTACKS_CACHE.exists():
    attacks = np.load(ATTACKS_CACHE, allow_pickle=True)
    print(f"  loaded cached attack-id array -> {ATTACKS_CACHE.name}")
else:
    print("  reconstructing attack-id array from the cached HF dataset metadata "
          "(offline, system_id column only, no audio download, no GPU) ...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("Bisher/as_vspoof_2019_la", cache_dir=str(DATA_CACHE),
                       trust_remote_code=True)["test"]
    sysids = ds["system_id"]
    rr = np.random.default_rng(42)   # identical seed/recipe to i1_geometry_causal_decomp.py
    spoof_idx = [i for i, s in enumerate(sysids) if s in EVAL_ATTACKS]
    bona_idx = [i for i, s in enumerate(sysids) if s == "-"]
    sel = sorted(rr.choice(spoof_idx, 800, replace=False).tolist() +
                 rr.choice(bona_idx, 800, replace=False).tolist())
    rec_labels = np.array([0 if sysids[i] == "-" else 1 for i in sel])
    assert np.array_equal(rec_labels, labels), \
        "reconstructed selection does not match cached i1_logits.npz labels"
    attacks = np.array([sysids[i] for i in sel])
    np.save(ATTACKS_CACHE, attacks)
    print(f"  verified against i1_logits.npz labels (exact match) -> cached to {ATTACKS_CACHE.name}")

sys_units = sorted(set(attacks) - {"-"})
n_systems = len(sys_units)
print(f"[C] n_systems (attacks) = {n_systems}: {sys_units}")
assert 6 <= n_systems <= 13, "expected 6-13 ASVspoof attack systems for this demo"

# sanity cross-check vs i4_asvspoof_position/attack_table.csv per-attack hardness
ref_table = pd.read_csv(I4DIR / "attack_table.csv")
def _hard(lg, a):
    bl = lg[labels == 0]; sl = lg[attacks == a]
    return 1 - roc_auc_score(np.r_[np.zeros(len(bl)), np.ones(len(sl))], np.r_[bl, sl])
mean_logit_all_seeds = np.mean([logit_seeds[s] for s in SEED_NAMES], axis=0)
chk = pd.DataFrame({"attack": sys_units,
                     "hard_check": [np.mean([_hard(logit_seeds[s], a) for s in SEED_NAMES])
                                    for a in sys_units]})
chk = chk.merge(ref_table[["attack", "hard"]], on="attack")
max_diff = float((chk["hard_check"] - chk["hard"]).abs().max())
print(f"[C] sanity check vs i4_asvspoof_position/attack_table.csv: max|Δhard| = {max_diff:.2e} "
      f"(should be ~0)")
assert max_diff < 1e-8, "attack-id reconstruction failed sanity check against i4 cache"

# ─── 3. ASVspoof-INTERNAL axis (unsupervised centroid difference, full data) ──
# Used only for the rotation-documentation cosine below; the FUSION folds each
# refit their own train-fold-only axis (see loop below) to avoid leakage.
bona_all = labels == 0
mu_full = X[bona_all].mean(0); sd_full = X[bona_all].std(0) + 1e-9
Z_full = (X - mu_full) / sd_full
w_internal_full = Z_full[labels == 1].mean(0) - Z_full[bona_all].mean(0)
w_internal_full /= np.linalg.norm(w_internal_full)

# ─── 4. MLAAD axis (I3/I7 construction) for the rotation check ────────────────
cos_report = {}
if (I3DIR / "embeddings.npz").exists():
    recs_path = RES / "mlaad" / "baseline_eval" / "test_in_distribution.json"
    wave_cache = BASE / "outputs" / "px_wave_cache" / "i2_full_test_waves.npz"
    if recs_path.exists() and wave_cache.exists():
        recs = json.loads(recs_path.read_text())
        ok = np.load(wave_cache, allow_pickle=True)["ok_idx"]
        recs = [recs[i] for i in ok]
        mlab = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
        MX = np.load(I3DIR / "embeddings.npz")["X12"].astype(np.float64)
        mu_ml = MX[mlab == 0].mean(0); sd_ml = MX[mlab == 0].std(0) + 1e-9
        Zml = (MX - mu_ml) / sd_ml
        w_mlaad = Zml[mlab == 1].mean(0) - Zml[mlab == 0].mean(0)
        w_mlaad /= np.linalg.norm(w_mlaad)
        cos_report["cos_internal_vs_mlaad_own_frame"] = float(w_internal_full @ w_mlaad)
        # also report in the MLAAD standardization frame (I7/audit4's primary frame)
        Z_asv_mlframe = (X - mu_ml) / sd_ml
        w_internal_mlframe = (Z_asv_mlframe[labels == 1].mean(0)
                               - Z_asv_mlframe[bona_all].mean(0))
        w_internal_mlframe /= np.linalg.norm(w_internal_mlframe)
        cos_report["cos_internal_vs_mlaad_mlaad_frame"] = float(w_internal_mlframe @ w_mlaad)
    else:
        cos_report["note"] = "MLAAD raw records/wave cache not found; cos vs MLAAD axis skipped"
else:
    cos_report["note"] = "i3_position_geometry/embeddings.npz not found; cos vs MLAAD axis skipped"

# secondary: cos vs the ASVspoof-2021 embeddings' own internal axis (different
# ASVspoof corpus/year, sanity check that "ASVspoof-internal" axes agree across
# ASVspoof corpora even though they rotate away from MLAAD)
if (J4DIR / "embeddings.npz").exists() and (J4DIR / "waves_sel.npz").exists():
    z21 = np.load(J4DIR / "waves_sel.npz", allow_pickle=True)
    y21 = (z21["atts"] != "bonafide").astype(int)
    X21 = np.load(J4DIR / "embeddings.npz")["X12"].astype(np.float64)
    mu21 = X21[y21 == 0].mean(0); sd21 = X21[y21 == 0].std(0) + 1e-9
    Z21 = (X21 - mu21) / sd21
    w21 = Z21[y21 == 1].mean(0) - Z21[y21 == 0].mean(0)
    w21 /= np.linalg.norm(w21)
    cos_report["cos_internal_2019LA_vs_internal_2021LA"] = float(w_internal_full @ w21)

print("[C] axis-rotation documentation (cos of ASVspoof-internal axis vs other corpora):")
for k, v in cos_report.items():
    print(f"    {k}: {v}")

# ─── 5. I7-style fusion: 5-fold system(attack)-disjoint, lambda from train fold
print(f"\n[C] running I7-style {N_FOLDS}-fold system-disjoint axis fusion "
      f"(detector = robust_goat / WavLM-GAT, ASVspoof-trained) ...")
rng = np.random.default_rng(SEED)
units_shuffled = np.array(sys_units)
rng.shuffle(units_shuffled)
folds = np.array_split(units_shuffled, N_FOLDS)
bona_idx_arr = np.where(bona_all)[0]
bona_folds = np.array_split(rng.permutation(bona_idx_arr), N_FOLDS)

s_along = np.full(len(X), np.nan)
fused = {sn: np.full(len(X), np.nan) for sn in SEED_NAMES}
lam_pick = {sn: [] for sn in SEED_NAMES}
fold_rows = []

for k in range(N_FOLDS):
    te_sys = set(folds[k].tolist())
    te_bona = set(bona_folds[k].tolist())
    te_mask = np.array([(attacks[i] in te_sys) or (i in te_bona) for i in range(len(X))])
    tr_mask = ~te_mask
    trb = tr_mask & bona_all
    trs = tr_mask & (labels == 1)

    mu = X[trb].mean(0); sd = X[trb].std(0) + 1e-9
    Ztr_s = (X[trs] - mu) / sd
    Ztr_b = (X[trb] - mu) / sd
    w = Ztr_s.mean(0) - Ztr_b.mean(0)
    w /= np.linalg.norm(w)

    proj_tr = ((X[tr_mask] - mu) / sd) @ w
    proj_te = ((X[te_mask] - mu) / sd) @ w
    s_along[te_mask] = proj_te
    pm, ps = proj_tr.mean(), proj_tr.std() + 1e-12

    row = {"fold": k, "n_te": int(te_mask.sum()), "test_systems": ",".join(sorted(te_sys))}
    for sn in SEED_NAMES:
        lg = logit_seeds[sn]
        lm, ls = lg[tr_mask].mean(), lg[tr_mask].std() + 1e-12
        ztr_l = (lg[tr_mask] - lm) / ls
        ztr_p = (proj_tr - pm) / ps
        ytr = labels[tr_mask]
        best = max(LAMS, key=lambda L: roc_auc_score(ytr, ztr_l + L * ztr_p))
        lam_pick[sn].append(best)
        fused[sn][te_mask] = (lg[te_mask] - lm) / ls + best * (proj_te - pm) / ps
        row[f"lambda_{sn}"] = best
        row[f"EER_detector_{sn}"] = eer_of(labels[te_mask], lg[te_mask])
        row[f"EER_fused_{sn}"] = eer_of(labels[te_mask], fused[sn][te_mask])
        row[f"dEER_{sn}"] = row[f"EER_fused_{sn}"] - row[f"EER_detector_{sn}"]
    row["EER_axis_alone"] = eer_of(labels[te_mask], s_along[te_mask])
    fold_rows.append(row)

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "asvspoof_fusion_per_fold.csv", index=False)
print("\n[C] per-fold results:")
print(fold_df.round(4).to_string(index=False))

print("\n[C] lambda picked per fold per seed:", lam_pick)

# ─── 6. headline metrics (concatenated out-of-fold scores, whole-corpus EER) ──
headline_rows = []
for sn in SEED_NAMES:
    headline_rows.append({"seed": sn, "scorer": "detector",
                           "EER": eer_of(labels, logit_seeds[sn]),
                           "AUC": roc_auc_score(labels, logit_seeds[sn])})
    headline_rows.append({"seed": sn, "scorer": "fused",
                           "EER": eer_of(labels, fused[sn]),
                           "AUC": roc_auc_score(labels, fused[sn])})
headline_rows.append({"seed": "-", "scorer": "axis_alone",
                       "EER": eer_of(labels, s_along),
                       "AUC": roc_auc_score(labels, s_along)})
headline_df = pd.DataFrame(headline_rows)
print("\n[C] headline (out-of-fold, whole-corpus EER):")
print(headline_df.round(4).to_string(index=False))

detector_alone_eer = float(np.mean([eer_of(labels, logit_seeds[sn]) for sn in SEED_NAMES]))
fused_eer = float(np.mean([eer_of(labels, fused[sn]) for sn in SEED_NAMES]))
axis_alone_eer = eer_of(labels, s_along)

# ─── 7. paired utterance bootstrap on mean dEER (I7-style honesty check) ──────
N = len(labels); B = 2000
bidx = np.random.default_rng(SEED).integers(0, N, size=(B, N))
dE = np.empty(B)
for b in range(B):
    ix = bidx[b]; yb = labels[ix]
    if yb.min() == yb.max():
        dE[b] = np.nan
        continue
    dE[b] = np.mean([eer_of(yb, fused[sn][ix]) - eer_of(yb, logit_seeds[sn][ix])
                      for sn in SEED_NAMES])
dE = dE[np.isfinite(dE)]
point_dE = fused_eer - detector_alone_eer
p_dE = float(min(1, 2 * min((dE <= 0).mean(), (dE >= 0).mean())))
ci_dE = [float(np.percentile(dE, 2.5)), float(np.percentile(dE, 97.5))]
print(f"\n[C] bootstrap dEER (fused - detector, mean over seeds) = {point_dE:+.4f} "
      f"[{ci_dE[0]:+.4f},{ci_dE[1]:+.4f}] p={p_dE:.4f}  (utterance-level paired bootstrap, B=2000)")

# ─── 8. save everything ────────────────────────────────────────────────────────
np.savez_compressed(OUT / "asvspoof_fusion_scores.npz",
                     labels=labels, attacks=attacks, s_along=s_along,
                     **{f"fused_{sn}": fused[sn] for sn in SEED_NAMES},
                     **{f"logit_{sn}": logit_seeds[sn] for sn in SEED_NAMES})

results = {
    "demo_scope_caveat": (
        "This is a fusion/robustness DEMO on n_systems=%d ASVspoof attacks, NOT a "
        "powered test of the system-hardness law (verdict P1, established/stress-tested "
        "at n=61 MLAAD systems). Utterance-level fusion metrics (n=1600) are "
        "well-powered; any system-level (n=13) regression would not be." % n_systems
    ),
    "detector_architecture": "WavLM-GAT",
    "detector_training_domain": "ASVspoof/robust (robust_goat.ckpt + 2 seeds: "
                                 "robust_goat_seed3.ckpt, robust_goat_seed7.ckpt)",
    "detector_note": (
        "robust_goat IS the WavLM-GAT architecture (frozen WavLM-L12 encoder + "
        "phoneme-GAT head), identical architecture to the MLAAD detector used in "
        "I7 (mlaad_robust_goat*.ckpt). The only difference is TRAINING DOMAIN "
        "(ASVspoof vs MLAAD) -- this experiment is the same-architecture / "
        "different-training-domain case, consistent with the verdict's "
        "cross-detector-agreement classification (audit7: aasist_zs~robust_goat "
        "is the ASVspoof-domain same-domain pair; robust_goat itself is WavLM-GAT, "
        "not a distinct architecture)."
    ),
    "corpus": "ASVspoof 2019 LA eval, attacks A07-A19 (robust_goat's native "
              "training/eval domain), 1600 utts (800 bona / 800 spoof), reproduced "
              "from experiments/results/i4_asvspoof_position/features.npz (frozen "
              "WavLM-L12 embeddings) + experiments/results/i1_geometry_causal_decomp/"
              "i1_logits.npz (robust_goat 3-seed baseline logits on the identical "
              "1600-utterance selection).",
    "n_systems": n_systems,
    "systems": sys_units,
    "n_utts": int(len(labels)),
    "n_bona": int(bona_all.sum()),
    "n_spoof": int((labels == 1).sum()),
    "axis_construction": "unsupervised centroid difference: w = mean(z(spoof_trainfold)) "
                          "- mean(z(bona_trainfold)), unit-normalized, standardization "
                          "(mu,sd) fit on train-fold bona pool only -- identical "
                          "construction to I7 (experiments/scripts/i7_axis_fusion.py) "
                          "and I3/I4's LOAO axis. Estimated INTERNALLY on ASVspoof "
                          "2019 LA train folds (never transferred from MLAAD).",
    "fusion_method": "s_fused = z(detector_logit) + lambda * z(axis_projection), "
                      "lambda in {0,0.25,0.5,0.75,1,1.5,2,3} selected by train-fold AUC, "
                      f"{N_FOLDS}-fold attack(system)-disjoint splits, bona split into "
                      "independent random folds, all standardization/lambda fit on "
                      "train folds only, evaluated out-of-fold.",
    "detector_alone_eer": detector_alone_eer,
    "detector_alone_eer_per_seed": {sn: eer_of(labels, logit_seeds[sn]) for sn in SEED_NAMES},
    "axis_alone_eer": axis_alone_eer,
    "fused_eer": fused_eer,
    "fused_eer_per_seed": {sn: eer_of(labels, fused[sn]) for sn in SEED_NAMES},
    "dEER_mean_over_seeds": point_dE,
    "dEER_bootstrap_ci95": ci_dE,
    "dEER_bootstrap_p": p_dE,
    "per_fold_dEER": fold_df.to_dict("records"),
    "lambdas_per_fold": {sn: list(map(float, v)) for sn, v in lam_pick.items()},
    "cos_internal_vs_mlaad": cos_report,
    "provenance": {
        "embeddings": str((I4DIR / "features.npz").relative_to(BASE)),
        "detector_logits": str((I1DIR / "i1_logits.npz").relative_to(BASE)),
        "attack_id_reconstruction": (
            "deterministic reproduction of experiments/scripts/"
            "i1_geometry_causal_decomp.py's seeded (seed=42) selection over "
            "HF dataset Bisher/as_vspoof_2019_la (system_id column only, offline "
            "mode, cached arrow files under data/asvspoof_2019_la/, no audio "
            "download, no GPU); verified byte-identical (max|Δhard|=%.2e) against "
            "experiments/results/i4_asvspoof_position/attack_table.csv" % max_diff
        ),
        "reference_method": "experiments/scripts/i7_axis_fusion.py "
                             "(verdict P5, MLAAD EER 0.272->0.163, audited clean "
                             "in experiments/axis_audits/audit5_fusion_claims.py)",
        "mlaad_axis_source": str((I3DIR / "embeddings.npz").relative_to(BASE)),
        "asvspoof2021_axis_source (secondary rotation check)": str((J4DIR / "embeddings.npz").relative_to(BASE)) if (J4DIR / "embeddings.npz").exists() else None,
    },
}
(OUT / "asvspoof_fusion_results.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\n[C] done -> {OUT}")
print(f"\n[C] HEADLINE: detector_alone={detector_alone_eer:.4f}  "
      f"axis_alone={axis_alone_eer:.4f}  fused={fused_eer:.4f}  "
      f"(dEER={point_dE:+.4f}, n_systems={n_systems})")
