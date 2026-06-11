#!/usr/bin/env python3
"""
J4 — Pre-registered prospective hardness prediction on ASVspoof 2021 LA (clean cond.)
======================================================================================
Lesson from J2: cross-corpus axis transfer fails under domain rotation; the law is
domain-internal. Refined methodology, registered BEFORE detector scoring:
  P1 (primary): corpus-INTERNAL leave-one-attack-out centroid axis; pred = -z(pos)
  P2: -z(pos) + z(sd)   (position + spread)
  P3: corpus-internal LOAO shrinkage-LDA direction; pred = -z(pos_lda)
  P4 (rotation control): MLAAD-axis transfer; pred = -z(pos_mlaad)
Axes use ground-truth bona/spoof labels of the 2021 corpus (metadata), never any
detector output. Evaluation detector: mlaad_robust_goat x 3 seeds — a detector
family that has NEVER seen ASVspoof; per-attack hardness vs the corpus-internal
bona pool. 13 attacks (A07–A19), codec condition 'none' only, 40 utts/attack,
~400 bonafide.

Outputs -> experiments/results/j4_asvspoof21/
"""
from __future__ import annotations
import io, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "j4_asvspoof21"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = px.DEVICE
N_PER_ATTACK, N_BONA = 40, 400
ATTACKS = [f"A{i:02d}" for i in range(7, 20)]
REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"
N_SHARDS = 5

# ─── metadata (official keys) ─────────────────────────────────────────────────
meta = pd.read_csv("/tmp/keys/LA/CM/trial_metadata.txt", sep=r"\s+", header=None,
                   names=["spk", "utt", "codec", "tx", "attack", "key", "trim", "phase"])
meta = meta[meta.codec == "none"]
att_of = dict(zip(meta.utt, meta.attack))
print(f"[J4] clean-condition trials: {len(meta)} "
      f"(bona={int((meta.attack=='bonafide').sum())})")

# ─── selective audio fetch: first N_SHARDS shards, filter to clean condition ──
import soundfile as sf
from huggingface_hub import hf_hub_download
CACHE = OUT / "waves_sel.npz"
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    waves, atts = z["waves"].astype(np.float32), z["atts"]
else:
    import pyarrow.parquet as pq
    sel_rows = {a: [] for a in ATTACKS + ["bonafide"]}
    quota = {**{a: N_PER_ATTACK for a in ATTACKS}, "bonafide": N_BONA}
    waves_l, atts_l = [], []
    for sh in range(N_SHARDS):
        fp = hf_hub_download(REPO, f"data/test-{sh:05d}-of-00024.parquet",
                             repo_type="dataset")
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["path", "audio"])
            paths = t.column("path").to_pylist()
            for i, p in enumerate(paths):
                uid = p.replace(".flac", "")
                a = att_of.get(uid)
                if a is None or len(sel_rows[a]) >= quota[a]:
                    continue
                ab = t.column("audio")[i].as_py()["bytes"]
                y, sr = sf.read(io.BytesIO(ab), dtype="float32")
                assert sr == px.SR
                waves_l.append(px._fix(y)); atts_l.append(a)
                sel_rows[a].append(uid)
        got = {k: len(v) for k, v in sel_rows.items()}
        print(f"  shard {sh}: bona={got['bonafide']} "
              f"min_attack={min(got[a] for a in ATTACKS)}", flush=True)
        if got["bonafide"] >= N_BONA and all(got[a] >= N_PER_ATTACK for a in ATTACKS):
            break
    waves = np.stack(waves_l).astype(np.float16)
    atts = np.array(atts_l)
    np.savez_compressed(CACHE, waves=waves, atts=atts)
    waves = waves.astype(np.float32)
labels = (atts != "bonafide").astype(int)
print(f"  selected {len(waves)} utts "
      f"({int((labels==0).sum())} bona, {int(labels.sum())} spoof)")

# ─── embeddings ───────────────────────────────────────────────────────────────
EMB = OUT / "embeddings.npz"
if EMB.exists():
    X = np.load(EMB)["X12"]
else:
    wl = px.load_frozen_wavlm(DEVICE)
    X = []
    with torch.no_grad():
        for b in range(0, len(waves), 8):
            xb = torch.as_tensor(waves[b:b+8], dtype=torch.float32, device=DEVICE)
            X.append(wl(xb, output_hidden_states=True).hidden_states[12]
                     .mean(1).float().cpu().numpy())
    X = np.concatenate(X)
    np.savez_compressed(EMB, X12=X)
    del wl; torch.cuda.empty_cache()

# ─── PHASE A: corpus-internal axes + REGISTERED predictions ───────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
bona = labels == 0
pos_int = np.full(len(X), np.nan); pos_lda = np.full(len(X), np.nan)
mu_b = X[bona].mean(0); sd_b = X[bona].std(0) + 1e-9
Z = (X - mu_b) / sd_b
for a in ATTACKS:
    m = atts == a
    excl = (labels == 1) & (atts != a)
    w = Z[excl].mean(0) - Z[bona].mean(0); w /= np.linalg.norm(w)
    pos_int[m] = Z[m] @ w
    tr = bona | excl
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr])
    pos_lda[m] = lda.decision_function(X[m])
# MLAAD-axis transfer (rotation control)
I3 = px.EXP_DIR / "results" / "i3_position_geometry"
MX = np.load(I3 / "embeddings.npz")["X12"]
recs = json.loads(px.TEST_JSON.read_text())
mok = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
mlab = np.array([1 if str(recs[i]["label"]).lower().startswith("spoof") else 0 for i in mok])
mmu = MX[mlab == 0].mean(0); msd = MX[mlab == 0].std(0) + 1e-9
Zm = (MX - mmu) / msd
wm = Zm[mlab == 1].mean(0) - Zm[mlab == 0].mean(0); wm /= np.linalg.norm(wm)
pos_mlaad = ((X - mmu) / msd) @ wm

def zr(v): return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
rows = []
for a in ATTACKS:
    m = atts == a
    rows.append({"attack": a,
                 "pos_int": float(np.median(pos_int[m])), "sd_int": float(np.std(pos_int[m])),
                 "pos_lda": float(np.median(pos_lda[m])),
                 "pos_mlaad": float(np.median(pos_mlaad[m]))})
P = pd.DataFrame(rows).set_index("attack")
P["P1"] = -zr(P["pos_int"].values)
P["P2"] = -zr(P["pos_int"].values) + zr(P["sd_int"].values)
P["P3"] = -zr(P["pos_lda"].values)
P["P4"] = -zr(P["pos_mlaad"].values)
top3 = {k: P[k].sort_values(ascending=False).index[:3].tolist()
        for k in ["P1", "P2", "P3", "P4"]}
reg = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
       "note": "registered BEFORE any detector scoring of ASVspoof2021 audio",
       "primary": "P1 (corpus-internal LOAO centroid axis, -z(pos))",
       "predicted_hardest_top3": top3,
       "table": P.reset_index().to_dict("records")}
(OUT / "j4_preregistered_predictions.json").write_text(json.dumps(reg, indent=2))
print("\n[J4-A] REGISTERED predictions. Predicted hardest top-3:")
for k, v in top3.items(): print(f"  {k}: {v}")

# ─── PHASE B: score 3 MLAAD seeds ─────────────────────────────────────────────
SEED_CKPTS = {
    "main":  px.EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
    "s42":   px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt",
    "s1024": px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt",
}
LOG = OUT / "logits.npz"
if LOG.exists():
    lz = np.load(LOG); logits = {k: lz[k] for k in lz.files}
else:
    logits = {}
    for sn, ck in SEED_CKPTS.items():
        print(f"[J4-B] scoring under {sn} ...", flush=True)
        gm = px.load_detector(ckpt=ck)
        logits[sn] = px.detector_logits(gm, waves)
        del gm; torch.cuda.empty_cache()
    np.savez_compressed(LOG, **logits)

# ─── PHASE C: evaluation ──────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score
from scipy import stats
H = []
for a in ATTACKS:
    m = atts == a
    h = np.mean([1 - roc_auc_score(np.r_[np.zeros(bona.sum()), np.ones(m.sum())],
                                   np.r_[lg[bona], lg[m]]) for lg in logits.values()])
    H.append(h)
P["hard_actual"] = H
print("\n[J4-C] actual hardness (MLAAD detector, 3 seeds, vs 2021 clean bona):")
print(P[["pos_int", "P1", "P2", "P3", "P4", "hard_actual"]]
      .sort_values("hard_actual", ascending=False).round(3).to_string())
res = {}
for k in ["P1", "P2", "P3", "P4"]:
    rho, p = stats.spearmanr(P[k], P["hard_actual"])
    hit3 = len(set(top3[k]) & set(P["hard_actual"].sort_values(ascending=False).index[:3]))
    res[k] = {"rho": float(rho), "p": float(p), "top3_hits": int(hit3)}
    print(f"  {k}: rho={rho:+.3f} (p={p:.4f})  top3 hits={hit3}/3")
overall_eer, _ = px.compute_eer(labels, np.mean([logits[s] for s in logits], 0))
res["overall_eer_meanlogit"] = float(overall_eer)
P.to_csv(OUT / "j4_table.csv")
(OUT / "j4_results.json").write_text(json.dumps(res, indent=2))
print(f"\n  overall EER (mean logit) on 2021-clean subset: {overall_eer:.3f}")
print(f"[J4] done -> {OUT}")
