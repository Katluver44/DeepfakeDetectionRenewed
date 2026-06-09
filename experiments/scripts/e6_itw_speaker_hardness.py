#!/usr/bin/env python3
"""
E6 ITW Speaker: Does C/T predict ITW difficulty at the *speaker* level?
=======================================================================

E5 ITW found a dissociation: ITW is genuinely hard and C tracks hardness within
ITW at the per-sample level (rho(C,logit)=-0.16), yet ITW spoof does NOT occupy
the MLAAD hard manifold region (kNN-Q4 frac 0.085 << 0.25) -> looks like domain
shift, not hard-manifold overlap.

The loose end: that per-sample rho was never resolved by *granularity*. MLAAD's
C/T story was validated per-SYSTEM. ITW's natural unit is the SPEAKER (54 target
identities). This experiment asks whether speakers whose spoofs are more compact
(higher C) / burstier (higher T) get detected worse -- the apples-to-apples
per-speaker analogue of the MLAAD per-system analysis -- and decomposes E5's
per-sample rho into between- vs within-speaker components.

  C (compactness)      = -rog@L12     (higher = harder)
  T (temporal entropy) = vel_entropy@L9 (higher = harder)

Pure analysis over outputs/e5_itw_utterance_ct.csv (no GPU / no WavLM):
  1. Per-speaker aggregation: mean C/T + per-speaker EER (shared bona pool,
     MLAAD-analogue) for both checkpoints; within-speaker EER (secondary).
  2. Speaker-level hardness vs C/T: Spearman + quartile stratification, with the
     MLAAD per-system reference rho overlaid.
  3. Between- vs within-speaker decomposition + C-variance ICC.
  4. Figures + outputs/e6_itw_speaker_summary.md.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUT     = BASE / "outputs"
FIG     = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ITW_CSV      = OUT / "e5_itw_utterance_ct.csv"
BASELINE_EER = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "per_system_eer.csv"

CKPTS = ["mlaad_robust_goat", "robust_goat"]   # primary first (MLAAD-analogue comparator)
PRIMARY = "mlaad_robust_goat"
MIN_N = 5

from _ablation_common import (compute_eer, per_system_eer_from_dict,  # noqa: E402
                              load_system_ct, _quartile_buckets)

print(f"[E6 ITW Speaker] base={BASE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Load the E5 per-sample table
# ═══════════════════════════════════════════════════════════════════════════════
assert ITW_CSV.exists(), f"missing {ITW_CSV} (run e5 first)"
df = pd.read_csv(ITW_CSV)
logit_cols = {c: f"logit_{c}" for c in CKPTS}
have_ckpts = [c for c in CKPTS if logit_cols[c] in df.columns]
assert have_ckpts, f"no logit_* columns in {ITW_CSV}"
print(f"[0] loaded {len(df)} rows | speakers={df['speaker'].nunique()} | "
      f"checkpoints={have_ckpts}")

is_spoof = df["label"] == "spoof"
sp = df[is_spoof].copy()
bf = df[~is_spoof].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-speaker aggregation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Per-speaker aggregation ...")

# per-speaker mean C/T + counts (over spoof rows)
agg = sp.groupby("speaker").agg(
    mean_C=("C", "mean"), mean_T=("T", "mean"), n_spoof=("C", "size")).reset_index()
n_bona_per = bf.groupby("speaker").size()
agg["n_bona"] = agg["speaker"].map(n_bona_per).fillna(0).astype(int)

# Per-speaker EER, shared bona pool (MLAAD-analogue), both checkpoints.
# per_system_eer_from_dict scores each spoof group against the SHARED bona pool.
for c in have_ckpts:
    lc = logit_cols[c]
    labels  = df["label"].map({"spoof": 1, "bona-fide": 0}).tolist()
    logits  = df[lc].tolist()
    # grouping key: speaker for spoof, "__bona__" sentinel for bona (pooled)
    keys = [s if l == "spoof" else "__bona__"
            for s, l in zip(df["speaker"], df["label"])]
    eer_by_spk = per_system_eer_from_dict(labels, logits, keys, min_n=MIN_N)
    agg[f"eer_{c}"] = agg["speaker"].map(eer_by_spk)

# Within-speaker EER (secondary): speaker spoof vs that speaker's own bona-fide.
def within_speaker_eer(spk, lc):
    s_log = sp.loc[sp["speaker"] == spk, lc].values
    b_log = bf.loc[bf["speaker"] == spk, lc].values
    if len(s_log) < MIN_N or len(b_log) < MIN_N:
        return np.nan
    y = np.r_[np.ones(len(s_log)), np.zeros(len(b_log))]
    x = np.r_[s_log, b_log]
    e = compute_eer(y, x)
    return e if e is not None else np.nan

for c in have_ckpts:
    agg[f"within_eer_{c}"] = agg["speaker"].apply(lambda s: within_speaker_eer(s, logit_cols[c]))

# Keep speakers with enough spoof for a stable EER
spk_df = agg[agg["n_spoof"] >= MIN_N].copy().reset_index(drop=True)
spk_df = spk_df.sort_values(f"eer_{PRIMARY}", ascending=False).reset_index(drop=True)
spk_df.to_csv(OUT / "e6_itw_speaker_ct.csv", index=False)
print(f"    speakers with n_spoof>={MIN_N}: {len(spk_df)} "
      f"(of {agg['speaker'].nunique()} total)")
print(f"    saved e6_itw_speaker_ct.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Speaker-level hardness vs C/T (Spearman + quartile stratification)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Speaker-level C/T -> EER ...")

def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan"), float("nan"), int(m.sum())
    r = stats.spearmanr(x[m], y[m])
    return float(r.correlation), float(r.pvalue), int(m.sum())

speaker_corr = {}   # {ckpt: {"C": (rho,p,n), "T": (rho,p,n)}}
for c in have_ckpts:
    e = spk_df[f"eer_{c}"].values
    speaker_corr[c] = {
        "C": spear(spk_df["mean_C"].values, e),
        "T": spear(spk_df["mean_T"].values, e),
    }
    rC, pC, _ = speaker_corr[c]["C"]
    rT, pT, _ = speaker_corr[c]["T"]
    print(f"    [{c}] rho(meanC,EER)={rC:+.3f}(p={pC:.2g})  "
          f"rho(meanT,EER)={rT:+.3f}(p={pT:.2g})")

# Quartile stratification on speakers (Q4 = highest C/T = predicted hardest).
spk_ct = {row["speaker"]: {"C": row["mean_C"], "T": row["mean_T"]}
          for _, row in spk_df.iterrows()}
spk_eer = {row["speaker"]: row[f"eer_{PRIMARY}"] for _, row in spk_df.iterrows()}

def quartile_eer(axis):
    buckets = _quartile_buckets(list(spk_ct.keys()), spk_ct, axis)
    rows = []
    for qi, members in enumerate(buckets):
        evals = [spk_eer[s] for s in members if np.isfinite(spk_eer.get(s, np.nan))]
        rows.append({"quartile": f"Q{qi+1}", "n": len(members),
                     "mean_eer": float(np.mean(evals)) if evals else float("nan")})
    return rows

q_C = quartile_eer("C")
q_T = quartile_eer("T")
print(f"    C-quartile EER: " + " ".join(f"{r['quartile']}={r['mean_eer']:.3f}" for r in q_C))
print(f"    T-quartile EER: " + " ".join(f"{r['quartile']}={r['mean_eer']:.3f}" for r in q_T))

# MLAAD per-system reference rho (extraction-free consistency anchor).
ct = load_system_ct()
base = pd.read_csv(BASELINE_EER)
eer_col = "eer_before" if "eer_before" in base.columns else "eer"
sys_eer = dict(zip(base["system"], base[eer_col]))
common = [s for s in sys_eer if s in ct]
mla_C = np.array([ct[s]["C"] for s in common])
mla_T = np.array([ct[s]["T"] for s in common])
mla_e = np.array([sys_eer[s] for s in common])
mla_rC = spear(mla_C, mla_e)
mla_rT = spear(mla_T, mla_e)
# MLAAD quartile-EER profile (for the figure overlay)
mla_ct = {s: ct[s] for s in common}
mla_eer_d = {s: sys_eer[s] for s in common}
def mla_quartile_eer(axis):
    out = []
    for members in _quartile_buckets(common, mla_ct, axis):
        out.append(float(np.mean([mla_eer_d[s] for s in members])))
    return out
mla_qC = mla_quartile_eer("C")
mla_qT = mla_quartile_eer("T")
print(f"    MLAAD(per-system, n={len(common)}) rho(C,EER)={mla_rC[0]:+.3f}(p={mla_rC[1]:.2g})  "
      f"rho(T,EER)={mla_rT[0]:+.3f}(p={mla_rT[1]:.2g})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Between- vs within-speaker decomposition (interprets E5's per-sample rho)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Between- vs within-speaker decomposition ...")

# valid speakers = those kept in spk_df (n_spoof>=MIN_N)
valid_spk = set(spk_df["speaker"])
sp_v = sp[sp["speaker"].isin(valid_spk)].copy()

decomp = {}
for c in have_ckpts:
    lc = logit_cols[c]
    # pooled per-sample (for reference / E5 cross-check)
    pooled = spear(sp_v["C"].values, sp_v[lc].values)
    # between-speaker: speaker means of C vs speaker means of logit
    g = sp_v.groupby("speaker").agg(mC=("C", "mean"), mL=(lc, "mean"))
    between = spear(g["mC"].values, g["mL"].values)
    # within-speaker: subtract each speaker's mean from C and logit, pooled Spearman
    spk_mean_C = sp_v.groupby("speaker")["C"].transform("mean")
    spk_mean_L = sp_v.groupby("speaker")[lc].transform("mean")
    within = spear((sp_v["C"] - spk_mean_C).values, (sp_v[lc] - spk_mean_L).values)
    decomp[c] = {"pooled": pooled, "between": between, "within": within}
    print(f"    [{c}] pooled rho(C,logit)={pooled[0]:+.3f}  "
          f"between={between[0]:+.3f}  within={within[0]:+.3f}")

# C-variance ICC: fraction of spoof-C variance that is between-speaker.
groups = [g["C"].values for _, g in sp_v.groupby("speaker")]
grand = sp_v["C"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = float(((sp_v["C"] - grand) ** 2).sum())
icc_C = float(ss_between / ss_total) if ss_total > 0 else float("nan")
# same for T (informative)
grandT = sp_v["T"].mean()
groupsT = [g["T"].values for _, g in sp_v.groupby("speaker")]
ss_between_T = sum(len(g) * (g.mean() - grandT) ** 2 for g in groupsT)
ss_total_T = float(((sp_v["T"] - grandT) ** 2).sum())
icc_T = float(ss_between_T / ss_total_T) if ss_total_T > 0 else float("nan")
print(f"    C variance between-speaker (ICC-like)={icc_C:.3f}  T={icc_T:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Figures ...")
RED, BLU = "#d62728", "#1f77b4"

# Fig 1: per-speaker mean-C/T vs EER (primary checkpoint)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
e = spk_df[f"eer_{PRIMARY}"].values
sizes = 20 + 4 * np.sqrt(spk_df["n_spoof"].values)
for ax, axis, col, lab in [(axes[0], "mean_C", RED, "C = -rog@L12"),
                           (axes[1], "mean_T", BLU, "T = vel_entropy@L9")]:
    x = spk_df[axis].values
    ax.scatter(x, e, s=sizes, alpha=0.6, c=col, edgecolors="k", linewidths=0.3)
    m = np.isfinite(x) & np.isfinite(e)
    if m.sum() >= 2:
        b1, b0 = np.polyfit(x[m], e[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, b1 * xs + b0, "k--", lw=1)
    key = "C" if axis == "mean_C" else "T"
    rho, p, n = speaker_corr[PRIMARY][key]
    ax.set_xlabel(lab); ax.set_ylabel(f"speaker EER ({PRIMARY})")
    ax.set_title(f"{key}: rho={rho:+.3f} (p={p:.2g}, n={n})")
plt.suptitle("E6 ITW Speaker: per-speaker C/T vs detection EER\n"
             "(marker size ∝ n_spoof; +rho ⇒ harder speakers are more compact/bursty)",
             fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e6_itw_speaker_ct_eer.png", dpi=150); plt.close()

# Fig 2: quartile-EER bars, ITW speakers vs MLAAD systems
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
qx = np.arange(4); w = 0.38
for ax, itw_q, mla_q, axis in [(axes[0], q_C, mla_qC, "C"), (axes[1], q_T, mla_qT, "T")]:
    itw_vals = [r["mean_eer"] for r in itw_q]
    ax.bar(qx - w/2, itw_vals, w, color=RED, alpha=0.8, label="ITW speakers")
    ax.bar(qx + w/2, mla_q, w, color="#7f7f7f", alpha=0.8, label="MLAAD systems")
    ax.set_xticks(qx); ax.set_xticklabels([f"Q{i+1}" for i in range(4)])
    ax.set_xlabel(f"{axis}-quartile (Q4 = highest {axis} = predicted hardest)")
    ax.set_ylabel("mean EER"); ax.set_title(f"{axis} stratification"); ax.legend(fontsize=8)
plt.suptitle("E6 ITW Speaker: EER by C/T quartile — ITW (per-speaker) vs MLAAD (per-system)",
             fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e6_itw_speaker_quartile_eer.png", dpi=150); plt.close()

# Fig 3: within- vs between-speaker decomposition (primary checkpoint)
lc = logit_cols[PRIMARY]
g = sp_v.groupby("speaker").agg(mC=("C", "mean"), mL=(lc, "mean"))
cC = (sp_v["C"] - sp_v.groupby("speaker")["C"].transform("mean")).values
cL = (sp_v[lc] - sp_v.groupby("speaker")[lc].transform("mean")).values
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].scatter(cC, cL, s=8, alpha=0.3, c=RED)
wr = decomp[PRIMARY]["within"]
axes[0].set_xlabel("C − speaker-mean(C)"); axes[0].set_ylabel("logit − speaker-mean(logit)")
axes[0].set_title(f"WITHIN-speaker: rho={wr[0]:+.3f} (p={wr[1]:.2g})")
axes[1].scatter(g["mC"], g["mL"], s=40, alpha=0.7, c=BLU, edgecolors="k", linewidths=0.3)
br = decomp[PRIMARY]["between"]
axes[1].set_xlabel("speaker-mean C"); axes[1].set_ylabel("speaker-mean logit")
axes[1].set_title(f"BETWEEN-speaker: rho={br[0]:+.3f} (p={br[1]:.2g})")
plt.suptitle(f"E6 ITW Speaker: decomposing E5's per-sample rho(C,logit) "
             f"[{PRIMARY}]\n(pooled={decomp[PRIMARY]['pooled'][0]:+.3f}; "
             f"C between-speaker variance ICC={icc_C:.2f})", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "e6_itw_within_between.png", dpi=150); plt.close()
print("    figures saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary + verdict
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Writing summary ...")

# Verdict: hardness-consistent sign in EER-space is POSITIVE (higher C/T -> higher EER).
def axis_supported(key):
    rho, p, _ = speaker_corr[PRIMARY][key]
    q = q_C if key == "C" else q_T
    q4_gt_q1 = (np.isfinite(q[3]["mean_eer"]) and np.isfinite(q[0]["mean_eer"])
                and q[3]["mean_eer"] > q[0]["mean_eer"])
    mla_rho = (mla_rC if key == "C" else mla_rT)[0]
    sign_match = np.isfinite(rho) and np.isfinite(mla_rho) and (np.sign(rho) == np.sign(mla_rho))
    return (np.isfinite(rho) and p < 0.05 and rho > 0 and q4_gt_q1 and sign_match)

c_sup, t_sup = axis_supported("C"), axis_supported("T")
axis_verdict = ("BOTH C and T" if c_sup and t_sup else "C only" if c_sup
                else "T only" if t_sup else "NEITHER")
supported = c_sup or t_sup

prho = decomp[PRIMARY]
between_dom = (np.isfinite(prho["between"][0]) and np.isfinite(prho["within"][0])
               and abs(prho["between"][0]) > abs(prho["within"][0]))

if supported:
    closing = (f"_RESCUE: at speaker granularity, C/T **does** predict ITW detection "
               f"difficulty ({axis_verdict}), with the same sign as MLAAD's per-system "
               f"effect. E5's per-sample dissociation was partly a **granularity artifact** "
               f"— the C/T mechanism operates between speakers even though the ITW spoof "
               f"population does not overlap MLAAD's absolute hard region._")
else:
    closing = (f"_REINFORCES domain-shift verdict: even at speaker granularity, C/T does "
               f"not predict ITW detection difficulty (verdict {axis_verdict}). E5's "
               f"per-sample rho(C,logit) is "
               f"{'between-speaker' if between_dom else 'within-speaker'}-dominated and does "
               f"not aggregate into a speaker-level hardness signal — ITW difficulty remains "
               f"best explained by domain shift, not C/T hard-manifold overlap._")

L = ["# E6 ITW Speaker — Does C/T predict ITW difficulty at the speaker level?", "",
     f"Source: `outputs/e5_itw_utterance_ct.csv` (no re-extraction). "
     f"Speakers with n_spoof≥{MIN_N}: **{len(spk_df)}**. "
     f"Primary checkpoint: `{PRIMARY}` (MLAAD-analogue comparator).",
     "Sign convention: C/T higher = harder ⇒ a **positive** rho with EER means the "
     "mechanism transfers (this is EER-space; E5 reported logit-space, opposite sign).", "",
     f"## Verdict: speaker-level mechanism **{'SUPPORTED' if supported else 'NOT supported'}** "
     f"({axis_verdict})", "",
     "## 1. Per-speaker C/T → EER (Spearman across speakers)",
     "| checkpoint | rho(meanC, EER) | rho(meanT, EER) |", "|---|---|---|"]
for c in have_ckpts:
    rC, pC, nC = speaker_corr[c]["C"]
    rT, pT, _ = speaker_corr[c]["T"]
    L.append(f"| {c} | {rC:+.3f} (p={pC:.2g}, n={nC}) | {rT:+.3f} (p={pT:.2g}) |")
L.append(f"| **MLAAD (per-system ref, n={len(common)})** | {mla_rC[0]:+.3f} (p={mla_rC[1]:.2g}) | "
         f"{mla_rT[0]:+.3f} (p={mla_rT[1]:.2g}) |")

L += ["", "## 2. Quartile stratification (primary ckpt; Q4 = highest C/T = predicted hardest)",
      "| axis | Q1 | Q2 | Q3 | Q4 | Q4>Q1? |", "|---|---|---|---|---|---|"]
for axis, q in [("C (ITW speakers)", q_C), ("T (ITW speakers)", q_T)]:
    vals = [r["mean_eer"] for r in q]
    flag = "yes" if (np.isfinite(vals[3]) and np.isfinite(vals[0]) and vals[3] > vals[0]) else "no"
    L.append(f"| {axis} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {flag} |")
for axis, q in [("C (MLAAD systems)", mla_qC), ("T (MLAAD systems)", mla_qT)]:
    flag = "yes" if q[3] > q[0] else "no"
    L.append(f"| {axis} | " + " | ".join(f"{v:.3f}" for v in q) + f" | {flag} |")

L += ["", "## 3. Between- vs within-speaker decomposition of E5's per-sample rho(C, logit)",
      "| checkpoint | pooled (per-sample) | between-speaker | within-speaker |",
      "|---|---|---|---|"]
for c in have_ckpts:
    d = decomp[c]
    L.append(f"| {c} | {d['pooled'][0]:+.3f} | {d['between'][0]:+.3f} (p={d['between'][1]:.2g}) | "
             f"{d['within'][0]:+.3f} (p={d['within'][1]:.2g}) |")
L += ["",
      f"- C variance that is between-speaker (ICC-like): **{icc_C:.3f}**  |  T: {icc_T:.3f}",
      f"- Dominant component (primary ckpt): **{'between-speaker' if between_dom else 'within-speaker'}**",
      "", closing, "",
      "## Figures",
      "- `figures/e6_itw_speaker_ct_eer.png` — per-speaker C/T vs EER scatter (+trend, +rho)",
      "- `figures/e6_itw_speaker_quartile_eer.png` — EER by C/T quartile, ITW vs MLAAD",
      "- `figures/e6_itw_within_between.png` — within- vs between-speaker decomposition",
      "", "## Outputs",
      "- `e6_itw_speaker_ct.csv` — per-speaker mean C/T, n, EER (both ckpts, shared-pool + within)"]
(OUT / "e6_itw_speaker_summary.md").write_text("\n".join(L))
print(f"    summary -> {OUT/'e6_itw_speaker_summary.md'}")
print("\n[E6 ITW Speaker] done.")
