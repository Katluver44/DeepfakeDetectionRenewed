#!/usr/bin/env python3
"""score_heldout.py — GATED scoring harness for the ASVspoof21 held-out P3 confirmation.

============================================================================
 DO NOT RUN THIS BEFORE PREDICTOR_SPEC.md HAS BEEN COMMITTED AND EXTERNALLY
 TIMESTAMPED. Running this script constitutes scoring the held-out data and
 destroys the scientific value of the confirmation described there.
============================================================================

This script implements EXACTLY the predictor and test specified in
`PREDICTOR_SPEC.md` (same directory), which was itself copied verbatim in
construction from `experiments/scripts/j4_asvspoof21_prospective.py`
(P3 = corpus-internal leave-one-attack-out shrinkage-LDA axis) and
`experiments/axis_audits/audit3_asvspoof_prospective.py` (exact permutation
p-value machinery). It computes ONLY P3 — no P1/P2/P4 — against ONE
pre-specified held-out condition (ASVspoof 2021 DF, source=='asvspoof'
subset, or the codec!='none' LA fallback; see PREDICTOR_SPEC.md section 3).

CPU-only. No GPU required. Requires: numpy, pandas, scipy, sklearn, torch
(CPU), transformers, huggingface_hub, pyarrow, soundfile — same environment
as the rest of experiments/ (use the repo venv:
/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/venv/bin/python).

Usage (only after the spec commit is tagged):
    HF_DATASETS_OFFLINE=0 DHELDOUT_UNLOCK=1 \
        venv/bin/python experiments/results/d_heldout/score_heldout.py

Output: experiments/results/d_heldout/results_heldout.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import collections
from pathlib import Path

import numpy as np
import pandas as pd

# ─── frozen constants (copied from PREDICTOR_SPEC.md — do not tune) ─────────
N_MIN_UTTS_PER_ATTACK = 20         # §4.4 — feasibility threshold, pre-specified
N_PERM = 200_000                    # §4 — exact permutation count, matches Audit 3
PERM_SEED = 7                       # §4 — matches audit_common.spearman_perm_p usage in Audit 3
ONE_SIDED_DIRECTION = "greater"     # rho > 0 is the pre-specified direction
ALPHA = 0.05                        # §4.1 acceptance criterion

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))

DF_REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF"
LA_REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"
SR = 16000
TLEN = 48000  # matches px_common.TLEN; only relevant if a full detector forward pass is added


def _fix_len(y: np.ndarray, tlen: int = TLEN) -> np.ndarray:
    """Pad/crop to fixed length the same way px_common._fix does (centered pad, no random crop)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= tlen:
        start = (len(y) - tlen) // 2
        return y[start:start + tlen]
    pad = tlen - len(y)
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right), mode="constant")


# ─── §3 condition selection (mechanical, pre-specified, NOT data-dependent on hardness) ──
def load_condition_metadata():
    """Fetch per-row `notes` metadata (no audio) for DF, decide primary vs fallback
    condition per the pre-specified §4.4 rule, and return (repo, filter_fn, rows_meta).
    This function may download parquet shards to read `notes`/`label` columns, but does
    NOT decode audio and does NOT touch any detector or embedding.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    info = api.dataset_info(DF_REPO)
    shard_files = sorted(s.rfilename for s in info.siblings
                          if s.rfilename.startswith("data/test-") and s.rfilename.endswith(".parquet"))

    per_attack_counts = collections.Counter()
    rows = []  # (shard, row_group, row_idx_in_rg, path, label, attack_id, codec, source)
    for shard_rel in shard_files:
        fp = hf_hub_download(DF_REPO, shard_rel, repo_type="dataset")
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["path", "label", "notes"])
            paths = t.column("path").to_pylist()
            labels = t.column("label").to_pylist()
            notes = [json.loads(n) for n in t.column("notes").to_pylist()]
            for i, (p, lb, nt) in enumerate(zip(paths, labels, notes)):
                a = nt.get("attack_id")
                src = nt.get("source")
                codec = nt.get("codec")
                rows.append({"shard": shard_rel, "row_group": rg, "row_idx": i,
                             "path": p, "label": lb, "attack_id": a,
                             "source": src, "codec": codec})
                if src == "asvspoof" and a and a != "-":
                    per_attack_counts[a] += 1
        # Early stop once we have enough evidence either way (feasibility check only,
        # not a hardness peek): keep scanning until every A07-A19 attack has >= N_MIN or
        # we've scanned all shards.
        asv_attacks = [f"A{i:02d}" for i in range(7, 20)]
        if all(per_attack_counts[a] >= N_MIN_UTTS_PER_ATTACK for a in asv_attacks):
            break

    asv_attacks = [f"A{i:02d}" for i in range(7, 20)]
    n_ok = sum(1 for a in asv_attacks if per_attack_counts[a] >= N_MIN_UTTS_PER_ATTACK)
    primary_feasible = n_ok >= 8  # §4.4 rule

    if primary_feasible:
        condition_name = "asvspoof21_df_source_asvspoof"
        keep = lambda r: r["source"] == "asvspoof" and r["attack_id"] in asv_attacks  # noqa: E731
    else:
        condition_name = "asvspoof21_la_codec_not_none_fallback"
        keep = None  # handled separately in the LA branch of main()

    return condition_name, per_attack_counts, rows


# ─── §2 P3 construction — verbatim from j4_asvspoof21_prospective.py ─────────
def compute_p3(X: np.ndarray, labels: np.ndarray, attack_of: np.ndarray, attacks: list[str]):
    """
    X: (N, 768) frozen WavLM-L12 mean-pooled embeddings.
    labels: (N,) 0=bona, 1=spoof.
    attack_of: (N,) attack id string per row ('bonafide' for bona rows).
    attacks: list of attack ids to score (the held-out roster).
    Returns: DataFrame indexed by attack with column 'P3' (predicted hardness, higher=harder).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    bona = labels == 0
    pos_lda = np.full(len(X), np.nan)
    for a in attacks:
        m = attack_of == a
        excl = (labels == 1) & (attack_of != a)   # leave-one-attack-out
        tr = bona | excl
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr])
        pos_lda[m] = lda.decision_function(X[m])

    rows = []
    for a in attacks:
        m = attack_of == a
        rows.append({"attack": a, "pos_lda": float(np.median(pos_lda[m])), "n_utts": int(m.sum())})
    P = pd.DataFrame(rows).set_index("attack")

    def zr(v):
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)

    P["P3"] = -zr(P["pos_lda"].values)
    return P


# ─── actual hardness: 1-AUROC vs held-out corpus's own bona pool ─────────────
def compute_actual_hardness(scores: np.ndarray, labels: np.ndarray, attack_of: np.ndarray,
                             attacks: list[str]):
    from sklearn.metrics import roc_auc_score

    bona_idx = np.where(labels == 0)[0]
    out = {}
    for a in attacks:
        a_idx = np.where(attack_of == a)[0]
        y = np.r_[np.zeros(len(bona_idx)), np.ones(len(a_idx))]
        s = np.r_[scores[bona_idx], scores[a_idx]]
        out[a] = 1 - roc_auc_score(y, s)
    return out


# ─── exact one-sided permutation test on Spearman rho ────────────────────────
def spearman_perm_p_one_sided(x, y, n_perm=N_PERM, seed=PERM_SEED):
    from scipy import stats
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs, _ = stats.spearmanr(x, y)
    cnt = 0
    for _ in range(n_perm):
        r, _ = stats.spearmanr(x, rng.permutation(y))
        cnt += r >= obs - 1e-12
    return float(obs), float((cnt + 1) / (n_perm + 1))


def main():
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[score_heldout] starting at {t0}")
    print("[score_heldout] STEP 1: resolve held-out condition (metadata only, no audio, no scoring)")
    condition_name, per_attack_counts, df_rows_meta = load_condition_metadata()
    print(f"[score_heldout] condition selected: {condition_name}")
    print(f"[score_heldout] per-attack ASVspoof-A07-A19 counts in DF/source==asvspoof: "
          f"{dict(per_attack_counts)}")

    # NOTE: the remaining steps (audio fetch, WavLM embedding, P3 construction,
    # detector scoring for actual hardness, and the permutation test) are the
    # actual "scoring" this harness exists to gate. They are implemented below
    # so the script is complete and ready to run, but intentionally are not
    # exercised until this file is invoked with DHELDOUT_UNLOCK=1 (see __main__).

    import soundfile as sf
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(SCRIPTS))
    import px_common as px  # noqa: E402

    device = torch.device("cpu")  # CPU-only harness, per Phase-1 constraint

    asv_attacks = [f"A{i:02d}" for i in range(7, 20)]
    if condition_name == "asvspoof21_df_source_asvspoof":
        keep_rows = [r for r in df_rows_meta if r["source"] == "asvspoof"
                     and r["attack_id"] in asv_attacks]
        # subsample per attack + bona to a manageable, pre-specified quota (mirrors J4's
        # N_PER_ATTACK/N_BONA quota mechanism; not a data-dependent choice)
        N_PER_ATTACK, N_BONA = 40, 400
        quota = {**{a: N_PER_ATTACK for a in asv_attacks}, "bonafide": N_BONA}
        sel = {a: [] for a in asv_attacks + ["bonafide"]}
        bona_rows = [r for r in df_rows_meta if r["label"] == 0]
        candidate_rows = keep_rows + bona_rows
        waves, atts = [], []
        for r in candidate_rows:
            key = r["attack_id"] if r["label"] == 1 else "bonafide"
            if key not in sel or len(sel[key]) >= quota.get(key, 0):
                continue
            fp = hf_hub_download(DF_REPO, r["shard"], repo_type="dataset")
            pf = pq.ParquetFile(fp)
            t = pf.read_row_group(r["row_group"], columns=["audio"])
            ab = t.column("audio")[r["row_idx"]].as_py()["bytes"]
            y, sr = sf.read(io.BytesIO(ab), dtype="float32")
            assert sr == SR
            waves.append(_fix_len(y))
            atts.append(key)
            sel[key].append(r["path"])
            if all(len(sel[k]) >= quota.get(k, 0) for k in quota):
                break
        waves = np.stack(waves).astype(np.float32)
        atts = np.array(atts)
    else:
        raise NotImplementedError(
            "LA codec!='none' fallback path: implement analogous quota-based fetch from "
            f"{LA_REPO} filtered to notes.codec != 'none', restricted to attacks "
            f"{asv_attacks}, before running. Not pre-fetched because the primary DF "
            "condition was expected to be feasible; see PREDICTOR_SPEC.md §3."
        )

    labels = (atts != "bonafide").astype(int)
    print(f"[score_heldout] fetched {len(waves)} held-out utterances "
          f"({int((labels == 0).sum())} bona, {int(labels.sum())} spoof)")

    print("[score_heldout] STEP 2: WavLM-L12 embeddings (frozen, CPU)")
    wl = px.load_frozen_wavlm(device)
    X = []
    with torch.no_grad():
        for b in range(0, len(waves), 8):
            xb = torch.as_tensor(waves[b:b + 8], dtype=torch.float32, device=device)
            X.append(wl(xb, output_hidden_states=True).hidden_states[12].mean(1).float().cpu().numpy())
    X = np.concatenate(X)
    del wl

    print("[score_heldout] STEP 3: construct P3 (corpus-internal LOAO shrinkage-LDA)")
    attacks_present = [a for a in asv_attacks if (atts == a).sum() >= N_MIN_UTTS_PER_ATTACK]
    P = compute_p3(X, labels, atts, attacks_present)

    print("[score_heldout] STEP 4: score detector for actual hardness (CPU)")
    gm = px.load_detector(device=device)
    scores = px.detector_logits(gm, waves, device=device)
    del gm
    hard = compute_actual_hardness(scores, labels, atts, attacks_present)
    P["hard_actual"] = [hard[a] for a in P.index]

    print("[score_heldout] STEP 5: single pre-specified test — Spearman rho(P3, hard_actual), one-sided")
    rho, p_perm = spearman_perm_p_one_sided(P["P3"].values, P["hard_actual"].values)
    success = bool(p_perm < ALPHA and rho > 0)

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_file": "PREDICTOR_SPEC.md",
        "condition": condition_name,
        "n_attacks": len(attacks_present),
        "attacks": attacks_present,
        "n_min_utts_per_attack": N_MIN_UTTS_PER_ATTACK,
        "n_perm": N_PERM,
        "perm_seed": PERM_SEED,
        "rho": rho,
        "p_one_sided_exact_perm": p_perm,
        "alpha": ALPHA,
        "acceptance_criterion": "p < 0.05 AND rho > 0 (one-sided, pre-specified)",
        "confirmation_result": "SUCCESS" if success else "FAILED",
        "per_attack_table": P.reset_index().to_dict("records"),
        "note": "This is the ONLY test run for this confirmation. No P1/P2/P4, no other "
                "detector, no other condition was scored under this pre-registration.",
    }
    out_path = HERE / "results_heldout.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[score_heldout] rho={rho:+.3f} p={p_perm:.5f} -> {result['confirmation_result']}")
    print(f"[score_heldout] wrote {out_path}")


if __name__ == "__main__":
    if os.environ.get("DHELDOUT_UNLOCK") != "1":
        print("SCORING IS GATED — run only after the spec is committed & tagged")
        print("(set environment variable DHELDOUT_UNLOCK=1 to proceed; see PREDICTOR_SPEC.md)")
        sys.exit(1)
    main()
