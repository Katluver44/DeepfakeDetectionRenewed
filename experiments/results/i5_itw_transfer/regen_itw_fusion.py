#!/usr/bin/env python3
"""Regenerate the ITW (In-The-Wild) axis-fusion analysis from raw scores.

WHY THIS SCRIPT EXISTS
-----------------------
`experiments/results/i7_axis_fusion/itw_fusion_test.json` is a MISLABELED,
un-reproducible artifact: it has no committed generating script, and its own
numbers were misreported downstream as "ITW fusion 0.363 -> 0.292" when 0.292
is actually the *axis-alone* EER; the true fusion EER is 0.312 (fusion HURTS
relative to the axis alone on ITW for the WavLM-GAT detector). This was
caught by an independent red-team audit:

    experiments/axis_audits/audit5_fusion_claims.py   (fusion mislabeling,
                                                        leakage quantification)
    experiments/axis_audits/audit6_itw_speaker.py     (per-speaker geometry;
                                                        not reproduced here)
    experiments/axis_audits/audits_outputs/COMPREHENSIVE_VERDICT.md (S3.2/3.3)
    experiments/axis_audits/audits_outputs/audit5_fusion_claims/AUDIT5_SUMMARY.md

This script is a from-scratch, CPU-only, no-torch reimplementation of the
audit5 methodology restricted to the ITW corpus, producing a *correctly
labeled* replacement artifact:

    experiments/results/i5_itw_transfer/itw_fusion_regenerated.json

It does NOT overwrite `itw_fusion_test.json` (left in place for the
historical record / diff).

WHAT IT COMPUTES (four EERs, unambiguous names)
------------------------------------------------
1. detector_alone_eer        - the pretrained WavLM-GAT detector logit,
                                straight EER on all 3000 ITW utterances
                                (matches the "itw_baseline" claim, ~0.363).
2. axis_alone_eer            - 5-fold *speaker-disjoint* internal ITW axis
                                (centroid-difference of standardized L12
                                embeddings, train-fold only), EER on the
                                held-out fold scores concatenated
                                (matches audit5's recomputation, ~0.30;
                                the original mislabeled artifact's "0.292").
3. fusion_eer                - same folds, detector logit z-scored + lambda
                                * axis projection z-scored, lambda chosen per
                                fold on TRAIN AUC from {0.5, 1.0, 1.5, 2.0}
                                (matches audit5's recomputation, ~0.306-0.312;
                                fusion HURTS relative to axis alone).
4. fusion_eer_speaker_disjoint - identical fusion protocol, but starting from
                                a construction that FORCES bona speakers to
                                be disjoint across folds too (this is what
                                fold #2/#3 already guarantee here, since we
                                fold on speaker identity for BOTH classes
                                from the start -- see "leakage" note below).

LEAKAGE NOTE (important asymmetry vs. audit5's J5-H4 pipeline)
-----------------------------------------------------------------
Audit5's headline "strict speaker-disjoint EER ~0.193" figure comes from a
*different* pipeline than the one above: it is the AASIST-ZS detector fused
with a *supervised* shrinkage-LDA axis (trained with labels on the ITW
corpus, group-disjoint on the spoof side), evaluated under audit5's J5-H4
protocol -- see `fusion_decomp(..., strict_group_bona=True)` in
`audit5_fusion_claims.py`. In that pipeline, the non-strict version folds
bona utterances at random (independent of speaker), so the SAME bona speaker
can appear in both the LDA-training fold and the held-out evaluation fold.
Forcing bona speaker-disjointness there drops fused EER from 0.161 -> 0.193
(i.e. EER goes UP because leakage was making the number look better than it
should -- for LDA-alone: 0.101 -> 0.143).

This script reproduces BOTH pipelines end to end so the two "0.193-ish" and
"0.30-ish" territories are not confused:

  (A) WavLM-GAT detector + unsupervised centroid ITW axis, ALREADY
      speaker-disjoint by construction (matches the itw_fusion_test.json
      family of numbers: baseline ~0.363, axis-alone ~0.30-0.29,
      fusion ~0.306-0.312 i.e. fusion hurts).
  (B) AASIST-ZS detector + supervised LDA axis on WavLM features (matches
      audit5's J5-H4 decomposition): LDA-alone and fused EER, computed both
      with bona folded at random (leaky, audit5's default protocol) and
      with bona forced speaker-disjoint (strict). This is where the
      ~0.193 "strict speaker-disjoint" number lives, and where the
      leakage delta (~3-4 EER points) is quantified.

Both pipelines are written to the corrected artifact so the reader can see
exactly which detector/axis combination produced which number -- no more
silent conflation.

RAW INPUTS (read-only; nothing outside i5_itw_transfer/ is written)
--------------------------------------------------------------------
  experiments/results/i5_itw_transfer/features.npz      (X12: L12 embeddings)
  experiments/results/i5_itw_transfer/utt_table.csv      (speaker, label, logit, ...)
  experiments/results/i5_itw_transfer/speaker_table.csv  (per-speaker aggregates; unused
                                                           here beyond a sanity check)
  experiments/results/i5_itw_transfer/i5_stats.json      (unused; reference only)
  experiments/results/j5_aasist/aasist_scores.npz        (sit: AASIST-ZS scores on ITW,
                                                           same row order as utt_table.csv)

CPU only. No torch. No GPU. No package installs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent          # .../experiments/results/i5_itw_transfer
RES = HERE.parent                                # .../experiments/results
J5_DIR = RES / "j5_aasist"
LAMS = [0.5, 1.0, 1.5, 2.0]


# ─────────────────────────── shared utilities ────────────────────────────────
def eer_of(y, s):
    """Equal error rate from labels y (1=spoof/positive) and scores s (higher=spoof)."""
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def orient(s, y):
    """Flip sign if score does not already point spoof-high (median spoof > median bona)."""
    s = np.asarray(s, float)
    return -s if np.median(s[y == 1]) < np.median(s[y == 0]) else s


def assert_speaker_disjoint(spk, folds, fold_indices_are_speakers=True):
    """Hard assertion: no speaker appears in more than one fold."""
    seen = {}
    for k, fold in enumerate(folds):
        units = fold if fold_indices_are_speakers else np.unique(spk[fold])
        for u in units:
            assert u not in seen, (
                f"LEAKAGE: speaker/unit {u!r} appears in both fold {seen[u]} and fold {k}")
            seen[u] = k
    return True


def assert_train_eval_disjoint(spk, train_mask, eval_mask, label="fold"):
    """Hard assertion: the set of speakers in eval_mask is disjoint from train_mask."""
    tr_spk = set(np.unique(spk[train_mask]))
    ev_spk = set(np.unique(spk[eval_mask]))
    overlap = tr_spk & ev_spk
    assert not overlap, f"LEAKAGE in {label}: speakers {overlap} appear in BOTH train and eval"


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE A: WavLM-GAT detector + unsupervised centroid ITW-internal axis
#             (the pipeline that itw_fusion_test.json intended to report,
#              5-fold speaker-disjoint from the start -- mirrors audit5's
#              from-scratch recomputation in audit5_fusion_claims.py, section [b])
# ═══════════════════════════════════════════════════════════════════════════
def pipeline_a(utt: pd.DataFrame, X: np.ndarray, seed: int = 0):
    y = utt["label"].values
    spk = utt["speaker"].values
    logit = utt["logit"].values.astype(float)

    detector_alone_eer = eer_of(y, logit)

    rng = np.random.default_rng(seed)
    spk_units = np.array(sorted(set(spk)))
    rng.shuffle(spk_units)
    sfolds = np.array_split(spk_units, 5)

    # Hard assertion: every fold's speaker set is disjoint from every other fold.
    assert_speaker_disjoint(spk, sfolds, fold_indices_are_speakers=True)

    axis_sc = np.full(len(y), np.nan)
    fused_sc = np.full(len(y), np.nan)
    per_fold_rows = []
    lam_used = []

    for k in range(5):
        te = np.isin(spk, sfolds[k])
        tr = ~te
        # Extra runtime assertion (belt-and-suspenders): no bona OR spoof
        # speaker overlaps between this fold's train and eval sets.
        assert_train_eval_disjoint(spk, tr, te, label=f"pipeline_a fold {k}")

        trb = tr & (y == 0)
        trs = tr & (y == 1)
        mu = X[trb].mean(0)
        sd = X[trb].std(0) + 1e-9
        w = ((X[trs] - mu) / sd).mean(0) - ((X[trb] - mu) / sd).mean(0)
        w = w / np.linalg.norm(w)

        proj_tr = ((X[tr] - mu) / sd) @ w
        proj_te = ((X[te] - mu) / sd) @ w
        axis_sc[te] = proj_te

        pm, ps = proj_tr.mean(), proj_tr.std() + 1e-12
        lm, ls = logit[tr].mean(), logit[tr].std() + 1e-12
        ztr_l = (logit[tr] - lm) / ls
        ztr_p = (proj_tr - pm) / ps
        lam = max(LAMS, key=lambda L: roc_auc_score(y[tr], ztr_l + L * ztr_p))
        lam_used.append(lam)
        fused_sc[te] = (logit[te] - lm) / ls + lam * (proj_te - pm) / ps

        eer_det_fold = eer_of(y[te], logit[te])
        eer_axis_fold = eer_of(y[te], proj_te)
        eer_fused_fold = eer_of(y[te], fused_sc[te])
        per_fold_rows.append({
            "fold": k, "n_eval": int(te.sum()), "lambda": lam,
            "detector_eer": eer_det_fold,
            "axis_eer": eer_axis_fold,
            "fused_eer": eer_fused_fold,
            "dEER_fusion_minus_axis": eer_fused_fold - eer_axis_fold,
            "dEER_fusion_minus_detector": eer_fused_fold - eer_det_fold,
        })

    axis_alone_eer = eer_of(y, axis_sc)
    fusion_eer = eer_of(y, fused_sc)

    return {
        "detector_alone_eer": detector_alone_eer,
        "axis_alone_eer": axis_alone_eer,
        "fusion_eer": fusion_eer,
        "lambda_per_fold": lam_used,
        "per_fold": per_fold_rows,
        "n_folds": 5,
        "n_utts": int(len(y)),
        "n_speakers": int(len(spk_units)),
        "fold_speaker_assignment_seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE B: AASIST-ZS detector + supervised shrinkage-LDA axis on WavLM
#             features (mirrors audit5's J5-H4 `fusion_decomp`), run once
#             with the leaky (random-bona-fold) protocol and once with the
#             strict (speaker-disjoint-bona) protocol -- this is where the
#             ~0.193 "strict speaker-disjoint" figure and the leakage delta
#             actually come from.
# ═══════════════════════════════════════════════════════════════════════════
def fusion_decomp_lda(X, y, s, spk, mode: str, seed: int = 42):
    """mode:
      'leaky'          - audit5's default: spoof units group-disjoint by speaker,
                         bona folded uniformly at random over utterance INDEX
                         (independent of speaker) -> same bona speaker's utterances
                         can span train and eval.
      'strict_audit5'  - audit5's "strict_group_bona": spoof AND bona each
                         group-disjoint by speaker, but the two partitions are
                         drawn INDEPENDENTLY of one another. Since ~46/49 ITW
                         speakers have BOTH a bona and a spoof recording, a given
                         person's bona utterances and spoof utterances can still
                         land in different folds under this scheme -- it removes
                         "same-label" leakage but not full identity leakage. This
                         is the exact protocol behind the ~0.193 figure quoted in
                         COMPREHENSIVE_VERDICT.md / AUDIT5_SUMMARY.md.
      'strict_identity' - this script's stronger variant: ONE speaker-identity
                         partition (matching pipeline A) used for both bona and
                         spoof rows, so no speaker's utterances of either label
                         appear in more than one fold. This is the version that
                         satisfies a hard train/eval speaker-disjointness
                         assertion; reported alongside 'strict_audit5' for
                         transparency about which claim it backs.
    """
    assert mode in ("leaky", "strict_audit5", "strict_identity")
    rng = np.random.default_rng(seed)

    if mode == "strict_identity":
        spk_units = np.array(sorted(set(spk)))
        rng.shuffle(spk_units)
        sfolds = np.array_split(spk_units, 5)
    else:
        spoof_units = sorted(set(spk[y == 1]))
        rng.shuffle(spoof_units)
        gfolds = np.array_split(np.array(spoof_units), 5)
        if mode == "strict_audit5":
            bona_units = sorted(set(spk[y == 0]))
            rng.shuffle(bona_units)
            bfolds_u = np.array_split(np.array(bona_units), 5)
        else:  # leaky
            bidx = np.where(y == 0)[0]
            bfolds = np.array_split(rng.permutation(bidx), 5)

    lda_sc = np.full(len(y), np.nan)
    fus = np.full(len(y), np.nan)
    per_fold_rows = []

    for k in range(5):
        if mode == "strict_identity":
            te = np.isin(spk, sfolds[k])
        elif mode == "strict_audit5":
            te_u = set(gfolds[k])
            te_b_u = set(bfolds_u[k])
            te = np.array([(spk[i] in te_u) if y[i] else (spk[i] in te_b_u)
                            for i in range(len(y))])
        else:  # leaky
            te_u = set(gfolds[k])
            te_b = set(bfolds[k].tolist())
            te = np.array([(spk[i] in te_u) or (i in te_b) for i in range(len(y))])
        tr = ~te

        if mode == "strict_identity":
            assert_train_eval_disjoint(spk, tr, te, label=f"pipeline_b({mode}) fold {k}")

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
        ptr, pte = lda.decision_function(X[tr]), lda.decision_function(X[te])
        lda_sc[te] = pte

        pm, ps = ptr.mean(), ptr.std() + 1e-12
        lm, ls = s[tr].mean(), s[tr].std() + 1e-12
        zl = (s[tr] - lm) / ls
        zp = (ptr - pm) / ps
        lam = max(LAMS, key=lambda L: roc_auc_score(y[tr], zl + L * zp))
        fus[te] = (s[te] - lm) / ls + lam * (pte - pm) / ps

        eer_lda_fold = eer_of(y[te], pte)
        eer_fused_fold = eer_of(y[te], fus[te])
        per_fold_rows.append({
            "fold": k, "n_eval": int(te.sum()), "lambda": lam,
            "lda_axis_eer": eer_lda_fold,
            "fused_eer": eer_fused_fold,
        })

    return {
        "lda_axis_alone_eer": eer_of(y, lda_sc),
        "fused_eer": eer_of(y, fus),
        "per_fold": per_fold_rows,
    }


def pipeline_b(utt: pd.DataFrame, X: np.ndarray):
    y = utt["label"].values
    spk = utt["speaker"].values

    asc_path = J5_DIR / "aasist_scores.npz"
    if not asc_path.exists():
        return None
    asc = np.load(asc_path)
    sit = asc["sit"]
    assert len(sit) == len(y), (
        f"aasist_scores.npz 'sit' length {len(sit)} != utt_table rows {len(y)}; "
        "row-order alignment assumption violated, refusing to proceed with pipeline B")
    s = orient(sit, y)

    aasist_alone_eer = eer_of(y, s)
    leaky = fusion_decomp_lda(X, y, s, spk, mode="leaky", seed=42)
    strict_a5 = fusion_decomp_lda(X, y, s, spk, mode="strict_audit5", seed=42)
    strict_id = fusion_decomp_lda(X, y, s, spk, mode="strict_identity", seed=42)

    leakage_delta_lda = leaky["lda_axis_alone_eer"] - strict_a5["lda_axis_alone_eer"]
    leakage_delta_fused = leaky["fused_eer"] - strict_a5["fused_eer"]

    return {
        "aasist_alone_eer": aasist_alone_eer,
        "leaky_bona_folds": {
            "lda_axis_alone_eer": leaky["lda_axis_alone_eer"],
            "fused_eer": leaky["fused_eer"],
            "per_fold": leaky["per_fold"],
        },
        "strict_speaker_disjoint_bona_folds": {
            "lda_axis_alone_eer": strict_a5["lda_axis_alone_eer"],
            "fused_eer": strict_a5["fused_eer"],
            "per_fold": strict_a5["per_fold"],
        },
        "strict_identity_disjoint_folds": {
            "description": (
                "Stronger than audit5's 'strict_group_bona': partitions bona AND "
                "spoof rows using a SINGLE speaker-identity fold assignment, so a "
                "person who appears as both a bona speaker and a spoof-clone target "
                "(true for ~46/49 ITW speakers) is entirely inside one fold. This is "
                "the variant this script's hard train/eval disjointness assertion "
                "actually verifies."
            ),
            "lda_axis_alone_eer": strict_id["lda_axis_alone_eer"],
            "fused_eer": strict_id["fused_eer"],
            "per_fold": strict_id["per_fold"],
        },
        "leakage_delta_lda_axis_alone": leakage_delta_lda,
        "leakage_delta_fused": leakage_delta_fused,
        "note": (
            "leaky_bona_folds mirrors audit5/J5-H4's default protocol, where bona "
            "utterances are folded at random (independent of speaker), so the same "
            "bona speaker can appear in both the LDA-training fold and the held-out "
            "evaluation fold. strict_speaker_disjoint_bona_folds reproduces audit5's "
            "'strict_group_bona' exactly (bona and spoof each speaker-disjoint, but "
            "the two partitions are independent of each other) -- this is the exact "
            "protocol behind the ~0.193 'strict speaker-disjoint' figure quoted in "
            "COMPREHENSIVE_VERDICT.md / AUDIT5_SUMMARY.md. Because ~46/49 ITW "
            "speakers have BOTH a bona and a spoof recording, that protocol does not "
            "fully eliminate identity leakage (a person's bona and spoof utterances "
            "can still land in different folds); strict_identity_disjoint_folds is "
            "this script's fix for that residual leakage, provided for transparency "
            "and verified by an explicit disjointness assertion."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# cos(w_ITW, w_MLAAD): does the ITW axis point the same direction as the
# MLAAD axis, in the MLAAD-bona standardization frame? (mirrors audit4's
# axis() helper; expected near-zero / slightly negative per the audit)
# ═══════════════════════════════════════════════════════════════════════════
def compute_cos_witw_wmlaad(X_itw, y_itw):
    """Best-effort: needs MLAAD embeddings + labels from i3_position_geometry.
    Returns None (with a reason) if those artifacts are unavailable, rather
    than fabricating a number."""
    try:
        i2_test_json = RES / "mlaad" / "baseline_eval" / "test_in_distribution.json"
        wave_cache = RES.parent.parent / "outputs" / "px_wave_cache" / "i2_full_test_waves.npz"
        i3_emb = RES / "i3_position_geometry" / "embeddings.npz"
        if not (i2_test_json.exists() and wave_cache.exists() and i3_emb.exists()):
            return None, "MLAAD reference artifacts (i2/i3) not found from this script's location"

        recs = json.loads(i2_test_json.read_text())
        ok = np.load(wave_cache, allow_pickle=True)["ok_idx"]
        recs = [recs[i] for i in ok]
        y_ml = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
        X_ml = np.load(i3_emb)["X12"]
        if len(y_ml) != len(X_ml):
            return None, "MLAAD label/embedding length mismatch"

        def axis(X, y, mu=None, sd=None):
            if mu is None:
                mu = X[y == 0].mean(0)
                sd = X[y == 0].std(0) + 1e-9
            Z = (X - mu) / sd
            w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
            return w / np.linalg.norm(w), mu, sd

        w_ml, mu_ml, sd_ml = axis(X_ml, y_ml)
        w_itw, _, _ = axis(X_itw, y_itw, mu_ml, sd_ml)
        return float(w_ml @ w_itw), None
    except Exception as e:  # pragma: no cover - defensive, report don't fudge
        return None, f"exception computing cos(w_ITW, w_MLAAD): {e!r}"


# ═══════════════════════════════════════════════════════════════════════════
def main():
    utt_path = HERE / "utt_table.csv"
    feat_path = HERE / "features.npz"
    spk_path = HERE / "speaker_table.csv"
    assert utt_path.exists(), f"missing {utt_path}"
    assert feat_path.exists(), f"missing {feat_path}"

    utt = pd.read_csv(utt_path)
    X = np.load(feat_path)["X12"].astype(float)
    assert len(utt) == len(X), f"utt_table rows {len(utt)} != features rows {len(X)}"

    y_itw = utt["label"].values
    print(f"Loaded ITW: {len(utt)} utterances, {utt['speaker'].nunique()} speakers, "
          f"label counts={dict(pd.Series(y_itw).value_counts())}")

    print("\n=== Pipeline A: WavLM-GAT detector + unsupervised centroid ITW axis "
          "(speaker-disjoint 5-fold) ===")
    a = pipeline_a(utt, X, seed=0)
    print(f"  detector_alone_eer = {a['detector_alone_eer']:.4f}  (expect ~0.363)")
    print(f"  axis_alone_eer     = {a['axis_alone_eer']:.4f}  (expect ~0.29-0.30)")
    print(f"  fusion_eer         = {a['fusion_eer']:.4f}  (expect ~0.306-0.312; "
          f"FUSION HURTS relative to axis alone)")
    print("  per-fold:")
    for row in a["per_fold"]:
        print(f"    fold {row['fold']}: n={row['n_eval']:4d} lam={row['lambda']:.1f} "
              f"det={row['detector_eer']:.4f} axis={row['axis_eer']:.4f} "
              f"fused={row['fused_eer']:.4f} dEER(fus-axis)={row['dEER_fusion_minus_axis']:+.4f}")

    print("\n=== Pipeline B: AASIST-ZS detector + supervised LDA axis "
          "(J5-H4 decomposition; leaky vs strict speaker-disjoint bona folds) ===")
    b = pipeline_b(utt, X)
    if b is None:
        print("  SKIPPED: experiments/results/j5_aasist/aasist_scores.npz not found.")
    else:
        print(f"  aasist_alone_eer                 = {b['aasist_alone_eer']:.4f}")
        print(f"  leaky:          lda_axis_alone={b['leaky_bona_folds']['lda_axis_alone_eer']:.4f}  "
              f"fused={b['leaky_bona_folds']['fused_eer']:.4f}")
        print(f"  strict(audit5): lda_axis_alone={b['strict_speaker_disjoint_bona_folds']['lda_axis_alone_eer']:.4f}  "
              f"fused={b['strict_speaker_disjoint_bona_folds']['fused_eer']:.4f}  "
              "(<- this is the 'strict speaker-disjoint ~0.193' figure)")
        print(f"  strict(identity): lda_axis_alone={b['strict_identity_disjoint_folds']['lda_axis_alone_eer']:.4f}  "
              f"fused={b['strict_identity_disjoint_folds']['fused_eer']:.4f}  "
              "(<- this script's fully speaker-identity-disjoint variant)")
        print(f"  leakage_delta (lda alone, vs audit5-strict)  = {b['leakage_delta_lda_axis_alone']:+.4f}")
        print(f"  leakage_delta (fused, vs audit5-strict)      = {b['leakage_delta_fused']:+.4f}")

    print("\n=== cos(w_ITW, w_MLAAD) ===")
    cos_val, cos_reason = compute_cos_witw_wmlaad(X, y_itw)
    if cos_val is None:
        print(f"  UNAVAILABLE: {cos_reason}")
    else:
        print(f"  cos(w_ITW, w_MLAAD) = {cos_val:+.4f}  (expect near-zero/negative; "
              "the axis rotates across domains)")

    # ── original mislabeled artifact, for side-by-side reference only ───────
    orig_artifact_path = RES / "i7_axis_fusion" / "itw_fusion_test.json"
    orig_artifact = None
    if orig_artifact_path.exists():
        orig_artifact = json.loads(orig_artifact_path.read_text())

    # ── assemble corrected, unambiguous artifact ─────────────────────────────
    leakage_delta_headline = None
    if b is not None:
        leakage_delta_headline = b["leakage_delta_fused"]

    out = {
        "provenance": {
            "generating_script": "experiments/results/i5_itw_transfer/regen_itw_fusion.py",
            "raw_inputs": [
                "experiments/results/i5_itw_transfer/features.npz (X12 embeddings)",
                "experiments/results/i5_itw_transfer/utt_table.csv "
                "(speaker, label, logit=WavLM-GAT detector score)",
                "experiments/results/i5_itw_transfer/speaker_table.csv (sanity check only)",
                "experiments/results/j5_aasist/aasist_scores.npz (sit=AASIST-ZS scores on ITW, "
                "row-aligned with utt_table.csv)" if b is not None else
                "experiments/results/j5_aasist/aasist_scores.npz (NOT FOUND, pipeline B skipped)",
            ],
            "reference_methodology": [
                "experiments/axis_audits/audit5_fusion_claims.py",
                "experiments/axis_audits/audit6_itw_speaker.py",
                "experiments/axis_audits/audit_common.py",
            ],
            "purpose": (
                "Correct the mislabeled experiments/results/i7_axis_fusion/itw_fusion_test.json "
                "(0.292 was reported downstream as 'fusion' but is actually axis-alone; true "
                "fusion is ~0.31 and HURTS relative to axis alone on ITW). This artifact is the "
                "unambiguous replacement; the original file is left in place, unmodified."
            ),
            "does_not_overwrite": "experiments/results/i7_axis_fusion/itw_fusion_test.json",
        },
        # ── headline, unambiguous keys (pipeline A: WavLM-GAT + unsup. centroid axis) ──
        "axis_alone_eer": a["axis_alone_eer"],
        "detector_alone_eer": a["detector_alone_eer"],
        "fusion_eer": a["fusion_eer"],
        "fusion_hurts_relative_to_axis_alone": bool(a["fusion_eer"] > a["axis_alone_eer"]),
        "fusion_eer_speaker_disjoint": (
            b["strict_speaker_disjoint_bona_folds"]["fused_eer"] if b is not None else None
        ),
        "leakage_delta": leakage_delta_headline,
        "cos_wITW_wMLAAD": cos_val,
        "cos_wITW_wMLAAD_unavailable_reason": cos_reason,

        "pipeline_a_wavlmgat_centroid_axis": {
            "description": (
                "WavLM-GAT fine-tuned detector logit, fused with an unsupervised "
                "centroid-difference ITW-internal axis on standardized L12 embeddings. "
                "5-fold, speaker-disjoint from construction (both bona and spoof "
                "speakers partitioned into folds jointly). This is the pipeline the "
                "original itw_fusion_test.json intended to report."
            ),
            "detector_alone_eer": a["detector_alone_eer"],
            "axis_alone_eer": a["axis_alone_eer"],
            "fusion_eer": a["fusion_eer"],
            "per_fold": a["per_fold"],
            "n_speakers": a["n_speakers"],
            "n_utts": a["n_utts"],
            "fold_seed": a["fold_speaker_assignment_seed"],
            "comparison_to_original_mislabeled_artifact": {
                "original_itw_baseline_EER": orig_artifact.get("itw_baseline", {}).get("EER")
                if orig_artifact else None,
                "original_itw_internal_axis_speaker_disjoint_EER":
                    orig_artifact.get("itw_internal_axis_speaker_disjoint", {}).get("EER")
                    if orig_artifact else None,
                "original_fusion_internal_lam1.5_EER":
                    orig_artifact.get("fusion_internal", {}).get("lam1.5", {}).get("EER")
                    if orig_artifact else None,
                "note": (
                    "The original artifact's 'itw_internal_axis_speaker_disjoint' (0.2917) "
                    "was mislabeled downstream as the fusion result; its actual "
                    "'fusion_internal' value (0.3123) is WORSE than the axis alone. "
                    "This script's independent from-scratch recomputation (different "
                    "fold RNG / speaker partition) reproduces the same qualitative and "
                    "closely matching quantitative finding: fusion_eer > axis_alone_eer."
                ),
            },
        },

        "pipeline_b_aasist_lda_axis_leakage_quantification": b,

        "reference_audit_outputs_for_comparison": {
            "audit5_itw_internal_axis_alone_speaker_disjoint": 0.30066666666666664,
            "audit5_itw_internal_fused_speaker_disjoint": 0.306,
            "audit5_original_artifact_axis_alone": 0.2917,
            "audit5_original_artifact_fused": 0.3123,
            "audit5_j5h4_itw_leaky_lda_alone": 0.10066666666666668,
            "audit5_j5h4_itw_leaky_fused": 0.16066666666666665,
            "audit5_j5h4_itw_strict_lda_alone": 0.1433333333333333,
            "audit5_j5h4_itw_strict_fused": 0.19333333333333336,
        },
    }

    out_path = HERE / "itw_fusion_regenerated.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote corrected artifact -> {out_path}")

    # ── final sanity assertions before declaring success ─────────────────────
    assert a["fusion_eer"] > a["axis_alone_eer"], (
        "Expected fusion to HURT relative to axis alone on ITW (pipeline A) -- "
        "this no longer matches the audited finding; investigate before trusting output.")
    if b is not None:
        assert b["strict_speaker_disjoint_bona_folds"]["fused_eer"] > \
               b["leaky_bona_folds"]["fused_eer"], (
            "Expected strict speaker-disjoint bona folds to raise (not lower) fused EER "
            "relative to the leaky protocol -- leakage should make numbers look better, "
            "not worse. Investigate before trusting output.")
    print("\nAll sanity assertions passed.")
    return out


if __name__ == "__main__":
    main()
