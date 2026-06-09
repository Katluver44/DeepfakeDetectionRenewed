#!/usr/bin/env python3
"""Generate the 7 'missing figures' from the paper plan (paper_plan_story3.html).
All outputs -> outputs/figures/.  Reproducible; reads only existing result files.
Each figure is wrapped in try/except so one failure does not block the others.
"""
import sys, traceback
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[1]
EXP  = BASE / "experiments"
OUT  = BASE / "outputs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXP / "scripts"))
plt.rcParams.update({"figure.dpi":140, "savefig.dpi":150, "font.size":11,
                     "axes.titleweight":"bold", "axes.spines.top":False, "axes.spines.right":False})
RED="#d62728"; BLUE="#1f5fbf"; GREEN="#1a7f4b"; AMBER="#b9770a"; PURPLE="#6b3fa0"; GREY="#9aa3ad"; INK="#1a1d24"
done=[]

def save(fig, name):
    p = OUT / name; fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  [ok] {name}"); done.append(name)

# ───────────────────────────────────────────────────────────────────────────
# Fig 1 — Architecture & C/T extraction pipeline schematic (SVG)
# ───────────────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 9.2)); ax.set_xlim(0,10); ax.set_ylim(0,15); ax.axis("off")
    def box(y, h, text, fc, ec, tc="#1a1d24", x=2.2, w=5.6, fs=11, bold=True):
        ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.08,rounding_size=0.12",
                    fc=fc, ec=ec, lw=1.6))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal")
    def arrow(y0,y1,x=5.0):
        ax.add_patch(FancyArrowPatch((x,y0),(x,y1), arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444"))
    def tap(y, label, color, side="right"):
        xx = 7.8 if side=="right" else 2.2
        ax.add_patch(FancyArrowPatch((7.8 if side=="right" else 2.2, y),
                                     (9.0 if side=="right" else 1.0, y),
                                     arrowstyle="-|>", mutation_scale=14, lw=2.0, color=color))
        ax.text(9.05 if side=="right" else 0.95, y, label, ha="left" if side=="right" else "right",
                va="center", fontsize=10.5, color=color, fontweight="bold")

    FROZ="#eef2f8"; FROZE="#9bb4d6"; TRN="#fdeee0"; TRNE="#d8a566"; HEAD="#e9f6ee"; HEADE="#7cc69b"
    box(13.4,1.0,"Raw audio  (16 kHz, 3 s)", "#ffffff", "#bbb", fs=10.5, bold=False)
    arrow(13.4,13.0)
    box(12.0,1.0,"WavLM feature extractor   [FROZEN]", FROZ, FROZE)
    arrow(12.0,11.6)
    box(9.6,2.0,"WavLM encoder   [FROZEN]\n12 transformer layers\nCTC phoneme head → phoneme IDs", FROZ, FROZE)
    # taps for C and T
    tap(11.05, "T = vel_entropy @ L9", PURPLE, "left")
    tap(10.15, "C = −rog @ L12", RED, "right")
    arrow(9.6, 9.2)
    box(8.0,1.0,"Adaptive phoneme pooling\n(consecutive same-ID frames averaged)", "#ffffff", "#bbb", fs=10, bold=False)
    arrow(8.0,7.6)
    box(6.0,1.6,"GAT   [3 layers · 6 heads · skip]   [TRAINED]", TRN, TRNE)
    arrow(6.0,5.6)
    box(4.4,1.0,"BiLSTM   [2 layers]   [TRAINED]", TRN, TRNE)
    arrow(4.4,4.0)
    box(2.8,1.0,"Mean-pool + L2 norm → utterance embedding", "#ffffff", "#bbb", fs=10, bold=False)
    arrow(2.8,2.4)
    box(1.2,1.0,"Binary spoof / bonafide logit", HEAD, HEADE)

    ax.text(5.0,14.85,"Detector architecture & C/T extraction points", ha="center", fontsize=13.5, fontweight="bold", color=INK)
    # legend
    ax.add_patch(FancyBboxPatch((0.2,0.05),3.0,0.55, boxstyle="round,pad=0.05", fc=FROZ, ec=FROZE))
    ax.text(1.7,0.32,"frozen WavLM", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((3.5,0.05),3.0,0.55, boxstyle="round,pad=0.05", fc=TRN, ec=TRNE))
    ax.text(5.0,0.32,"trained GAT+BiLSTM", ha="center", va="center", fontsize=9)
    ax.text(8.4,0.32,"C / T read from frozen reps", ha="center", va="center", fontsize=9, color="#555")
    fig.tight_layout(); fig.savefig(OUT/"fig1_architecture_ct_pipeline.png", bbox_inches="tight")
    save(fig, "fig1_architecture_ct_pipeline.svg")

# ───────────────────────────────────────────────────────────────────────────
# Fig 2 — Objective hardness agreement across seeds
# ───────────────────────────────────────────────────────────────────────────
def fig_seed_agreement():
    df = pd.read_csv(EXP/"results/mlaad/e6_ranking_lock/per_seed_per_attack_eer.csv")
    cond = "robust_goat" if "robust_goat" in df.condition.unique() else df.condition.unique()[0]
    d = df[df.condition==cond]
    piv = d.pivot_table(index="attack_system", columns="seed", values="eer", aggfunc="mean")
    piv = piv.dropna(axis=0, how="any")
    seeds = list(piv.columns)
    # pairwise Spearman across seeds
    rhos=[]
    for i in range(len(seeds)):
        for j in range(i+1,len(seeds)):
            rhos.append(stats.spearmanr(piv[seeds[i]], piv[seeds[j]]).correlation)
    mean_rho=float(np.mean(rhos))
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order]

    fig, axes = plt.subplots(1,2, figsize=(13,5.4), gridspec_kw={"width_ratios":[1.15,1]})
    # left: each attack's EER across seeds (lines), sorted by mean -> stable ordering
    ax=axes[0]
    x=np.arange(len(piv))
    for s in seeds:
        ax.plot(x, piv[s].values, marker="o", ms=3, lw=1, alpha=.55, label=f"seed {s}")
    ax.plot(x, piv.mean(axis=1).values, color=INK, lw=2.6, label="mean")
    ax.set_xlabel(f"MLAAD system (sorted by mean EER)   n={len(piv)}")
    ax.set_ylabel("per-system EER")
    ax.set_title(f"Hardness ordering is stable across {len(seeds)} seeds")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    # right: scatter two most-separated seeds + rho
    ax=axes[1]
    s0,s1=seeds[0],seeds[-1]
    ax.scatter(piv[s0], piv[s1], s=34, color=BLUE, alpha=.8, edgecolor="w")
    lim=[min(piv[s0].min(),piv[s1].min())-.02, max(piv[s0].max(),piv[s1].max())+.02]
    ax.plot(lim,lim, "--", color=GREY, lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"per-system EER  (seed {s0})"); ax.set_ylabel(f"per-system EER  (seed {s1})")
    r=stats.spearmanr(piv[s0],piv[s1]).correlation
    ax.set_title("Seed-vs-seed per-system EER")
    ax.text(.05,.93, f"Spearman ρ = {r:+.2f}\nmean pairwise ρ = {mean_rho:+.2f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", fc="#eef4fc", ec="#cfe0f5"))
    fig.suptitle("Deepfake hardness is objective: systems agree on what is hard across random seeds",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_hardness_seed_agreement.png")

# ───────────────────────────────────────────────────────────────────────────
# Fig 3 — Unified C → hardness across corpora (MLAAD + ITW)
# ───────────────────────────────────────────────────────────────────────────
def fig_crosscorpus():
    ml = pd.read_csv(BASE/"outputs/sig_layer_features.csv")
    ml["C"] = -ml["rog_L12"]
    itw = pd.read_csv(BASE/"outputs/e6_itw_speaker_ct.csv")
    # primary checkpoint (matches E6 summary headline ρ≈+0.44)
    itw_eer = "eer_mlaad_robust_goat" if "eer_mlaad_robust_goat" in itw.columns else itw.columns[-1]
    fig, axes = plt.subplots(1,2, figsize=(12.4,5.3))
    def panel(ax, x, y, xlab, ylab, title, color):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        ax.scatter(x,y,s=34,color=color,alpha=.75,edgecolor="w")
        b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),50)
        ax.plot(xs,b*xs+a,"--",color=INK,lw=1.6)
        rho=stats.spearmanr(x,y).correlation
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.text(.05,.93,f"Spearman ρ = {rho:+.2f}\nn = {len(x)}",transform=ax.transAxes,
                va="top",fontsize=11,bbox=dict(boxstyle="round",fc="#f6f8fb",ec="#cfd6e0"))
    panel(axes[0], ml["C"].values, ml["residual"].values,
          "C = −rog@L12  (→ more compact)", "residual hardness (↑ harder)",
          "MLAAD — 63 TTS/VC systems", BLUE)
    panel(axes[1], itw["mean_C"].values, itw[itw_eer].values,
          "mean C  (→ more compact)", "per-speaker EER (↑ harder)",
          "In-the-Wild — 45 speakers", GREEN)
    fig.suptitle("More compact (higher C) → harder, consistently across corpora",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_C_hardness_crosscorpus.png")

# ───────────────────────────────────────────────────────────────────────────
# Fig 4 — C is causal (E9): C-scale vs matched-energy shift control
# ───────────────────────────────────────────────────────────────────────────
def fig_e9():
    df = pd.read_csv(BASE/"outputs/ct_causal_intervention/e9_causal_C.csv")
    models=[m for m in ["robust_GOAT","GOAT"] if m in df.model.unique()]
    fig, axes = plt.subplots(1,len(models), figsize=(6.6*len(models),5.4), squeeze=False)
    for ax,m in zip(axes[0],models):
        sc=df[(df.model==m)&(df.intervention=="C_scale")].sort_values("measured_C")
        sh=df[(df.model==m)&(df.intervention=="shift_control")].sort_values("measured_C")
        baseC=df[(df.model==m)&(df.alpha==1.0)&(df.intervention=="C_scale")]["measured_C"].iloc[0]
        # moderate regime band (alpha 0.75..1.5)
        lo=sc[sc.alpha==0.75]["measured_C"].iloc[0]; hi=sc[sc.alpha==1.5]["measured_C"].iloc[0]
        ax.axvspan(lo,hi,color="#eaf4ec",label="moderate regime (clean)")
        ax.plot(sc["measured_C"],sc["EER"],"o-",color=RED,lw=2.2,label="C-scale (changes C)")
        ax.plot(sh["measured_C"],sh["EER"],"s--",color=BLUE,lw=1.8,alpha=.85,label="shift control (C fixed)")
        ax.axvline(baseC,color=GREY,ls=":",lw=1.2); ax.axhline(sc[sc.alpha==1.0]["EER"].iloc[0],color=GREY,ls=":",lw=.8)
        ax.annotate("more compact\n→ harder", xy=(lo, sc[sc.alpha==0.75]["EER"].iloc[0]),
                    xytext=(lo-7, .17), fontsize=9, color=RED, ha="center",
                    arrowprops=dict(arrowstyle="->",color=RED))
        ax.set_xlabel("measured C (rog@L12)  →  less compact"); ax.set_ylabel("EER")
        ax.set_title(f"{m}")
        ax.legend(fontsize=8.5, loc="upper center")
    fig.suptitle("C is causal: scaling compactness moves EER beyond a matched-energy shift control",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_e9_causal_C_main.png")

# ───────────────────────────────────────────────────────────────────────────
# Fig 5 — Actionability on hardest systems (P1 + P5)
# ───────────────────────────────────────────────────────────────────────────
def fig_actionability():
    from _ablation_common import load_baseline_metrics, load_system_ct
    base=load_baseline_metrics(); ct=load_system_ct()
    import glob
    p5=pd.concat([pd.read_csv(f) for f in glob.glob(str(EXP/"results/mlaad/p5_hardness_reweight/per_system_metrics_seed*.csv"))])
    p5=p5.groupby("system").mean(numeric_only=True)
    rows=[]
    for s,r in p5.iterrows():
        if s in base and s in ct:
            rows.append((s, ct[s]["C"], r["eer"]-base[s]["eer"], r["auc"]-base[s]["auc"]))
    d=pd.DataFrame(rows,columns=["system","C","d_eer","d_auc"]).dropna()
    # C quartiles (Q4 = most compact = highest C)
    d["Cq"]=pd.qcut(d["C"],4,labels=["Q1","Q2","Q3","Q4"])
    d=d.sort_values("d_eer")
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={"width_ratios":[1.3,1]})
    # left: per-system ΔEER sorted, color by C quartile, highlight Q4
    ax=axes[0]
    colors={"Q1":GREY,"Q2":GREY,"Q3":AMBER,"Q4":RED}
    ax.barh(np.arange(len(d)), d["d_eer"], color=[colors[q] for q in d["Cq"]], height=.8)
    ax.axvline(0,color=INK,lw=1)
    ax.set_yticks([]); ax.set_xlabel("ΔEER vs baseline  (←  improvement)")
    ax.set_ylabel(f"63 MLAAD systems (sorted)")
    ax.set_title("P5 hardness-reweight: per-system ΔEER")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED,label="C-Q4 (most compact = hardest)"),
                       Patch(color=AMBER,label="C-Q3"), Patch(color=GREY,label="C-Q1/Q2")],
              fontsize=8.5, loc="lower right", framealpha=.95)
    # right: grouped bar overall vs C-Q4 for P1 and P5 (ΔEER, ΔAUC)
    ax=axes[1]
    q4=d[d.Cq=="Q4"]
    p5_overall_eer=d["d_eer"].mean(); p5_q4_eer=q4["d_eer"].mean()
    p5_overall_auc=d["d_auc"].mean(); p5_q4_auc=q4["d_auc"].mean()
    # P1 numbers from its summary (verified): overall EER -0.0047; use per_system file for Q4
    p1=pd.read_csv(EXP/"results/mlaad/p1_ct_calibration/per_system_eer.csv")
    p1=p1.merge(pd.DataFrame([(s,ct[s]["C"]) for s in ct],columns=["system","C"]),on="system",how="left").dropna(subset=["C"])
    p1["Cq"]=pd.qcut(p1["C"],4,labels=["Q1","Q2","Q3","Q4"])
    p1_overall_eer=-p1["delta_eer"].mean(); p1_q4_eer=-p1[p1.Cq=="Q4"]["delta_eer"].mean()  # delta_eer=before-after (improvement +)
    labels=["P1 overall","P1 C-Q4","P5 overall","P5 C-Q4"]
    vals=[p1_overall_eer,p1_q4_eer,p5_overall_eer,p5_q4_eer]
    cols=[GREY,RED,GREY,RED]
    ax.bar(np.arange(4), vals, color=cols)
    ax.axhline(0,color=INK,lw=1)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("ΔEER  (negative = better)")
    ax.set_title("Overall vs hardest-quartile (C-Q4) gains")
    pad=(max(vals)-min(vals))*0.12
    ax.set_ylim(min(vals)-pad*2.2, max(vals)+pad*1.6)
    for i,v in enumerate(vals):
        ax.text(i, v - pad*0.5 if v<0 else v + pad*0.4, f"{v:+.3f}", ha="center",
                va="top" if v<0 else "bottom", fontsize=9.5, fontweight="bold")
    ax.text(0.5,0.97,"P5 reweighting targets C-Q4; P1 calibration lifts the average",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555", style="italic")
    fig.suptitle("Hardness reweighting (P5) lowers EER on the hardest (C-Q4) systems; calibration (P1) improves the average",
                 fontsize=12, fontweight="bold")
    save(fig, "fig_actionability_hardsystems.png")

# ───────────────────────────────────────────────────────────────────────────
# Fig 6 — Story 1 mechanism composite
# ───────────────────────────────────────────────────────────────────────────
def fig_story1():
    mc=pd.read_csv(EXP/"results/multiclass_probe/multiclass_summary.csv").set_index("probe")
    lp=pd.read_csv(EXP/"results/linear_probe/probe_metrics_summary.csv").set_index("name")
    layers=["gat_l0","gat_l1","gat_l2","bilstm"]
    mc_acc=[mc.loc[l,"acc"] for l in layers]
    # binary AUC per layer = mean over that layer's 6 heads (clean monotone trend)
    def layer_auc(pref):
        hs=[f"{pref}_h{i}" for i in range(6) if f"{pref}_h{i}" in lp.index]
        return np.mean([lp.loc[h,"auc"] for h in hs])
    bin_auc=[layer_auc("head_l0"),layer_auc("head_l1"),layer_auc("head_l2"),lp.loc["bilstm","auc"]]
    ap=pd.read_csv(EXP/"results/act_patching/act_patching_class_stats.csv")
    ap=ap[ap["class"]!="Other"].sort_values("mean_delta")

    fig,axes=plt.subplots(1,2,figsize=(13,5.3))
    ax=axes[0]; x=np.arange(len(layers))
    l1,=ax.plot(x, mc_acc, "o-", color=PURPLE, lw=2.4, ms=8, label="system-ID probe acc (↓ destroyed)")
    ax.set_ylabel("multiclass system-ID accuracy", color=PURPLE)
    ax.tick_params(axis="y", labelcolor=PURPLE)
    ax.set_xticks(x); ax.set_xticklabels(["GAT L0","GAT L1","GAT L2","BiLSTM"])
    ax.set_title("Fingerprint destruction ↔ detection creation")
    for xi,v in zip(x,mc_acc): ax.text(xi,v+.006,f"{v:.2f}",ha="center",color=PURPLE,fontsize=9)
    ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    l2,=ax2.plot(x, bin_auc, "s--", color=GREEN, lw=2.4, ms=7, label="binary detection AUC (↑ built)")
    ax2.set_ylabel("binary spoof-detection AUC", color=GREEN); ax2.tick_params(axis="y", labelcolor=GREEN)
    for xi,v in zip(x,bin_auc): ax2.text(xi,v-.004,f"{v:.3f}",ha="center",va="top",color=GREEN,fontsize=8.5)
    ax.legend(handles=[l1,l2], fontsize=9, loc="center left")

    ax=axes[1]
    cols=[RED if c in ("Sibilants","Affricates") else (GREEN if c in ("Diphthongs","Vowels") else GREY) for c in ap["class"]]
    ax.barh(np.arange(len(ap)), ap["mean_delta"], color=cols)
    ax.set_yticks(np.arange(len(ap))); ax.set_yticklabels(ap["class"])
    ax.set_xlabel("mean causal Δ on binary spoof logit (activation patching)")
    ax.set_title("Phoneme-class causal contribution")
    ax.set_xlim(1.6, 2.0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN,label="vowels / diphthongs (drive decision)"),
                       Patch(color=RED,label="sibilants / affricates (weak)")], fontsize=8.5, loc="lower right")
    fig.suptitle("Story 1 mechanism: GAT abstracts away system identity while deep layers build a generic detector",
                 fontsize=12, fontweight="bold")
    save(fig, "fig_story1_mechanism.png")

# ───────────────────────────────────────────────────────────────────────────
# Fig 7 — Smoothing is a dose-dependent lever
# ───────────────────────────────────────────────────────────────────────────
def fig_smoothing():
    from _ablation_common import load_baseline_metrics
    import glob
    base=load_baseline_metrics()
    p4=pd.concat([pd.read_csv(f) for f in glob.glob(str(EXP/"results/mlaad/p4_smoothing_aug/per_system_metrics_seed*.csv"))])
    p4=p4.groupby("system").mean(numeric_only=True)
    rows=[]
    for s,r in p4.iterrows():
        if s in base:
            rows.append((s, r["eer"]-base[s]["eer"], r["auc"]-base[s]["auc"]))
    d=pd.DataFrame(rows,columns=["system","d_eer","d_auc"]).sort_values("d_eer").reset_index(drop=True)
    both=d[(d.d_eer<0)&(d.d_auc>0)]
    fig,ax=plt.subplots(figsize=(9.6,6.2))
    colors=[GREEN if (e<0 and a>0) else (AMBER if e<0 else GREY) for e,a in zip(d.d_eer,d.d_auc)]
    ax.bar(np.arange(len(d)), d["d_eer"], color=colors, width=.9)
    ax.axhline(0,color=INK,lw=1)
    ax.set_xlabel("63 MLAAD systems (sorted by ΔEER)"); ax.set_ylabel("ΔEER vs baseline  (negative = improved)")
    ax.set_title("Uniform smoothing augmentation is a dose-dependent lever:\nit genuinely helps the subset that needs it, and overcorrects the rest")
    # annotate top improvers
    top=d.head(4)
    for i,(_,r) in enumerate(top.iterrows()):
        ax.annotate(f"{r['system']}  {r['d_eer']:+.3f}", xy=(i, r["d_eer"]),
                    xytext=(i+5, r["d_eer"]-.02), fontsize=8.5, color=GREEN,
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN,label=f"improved on EER+AUC ({len(both)} systems)"),
                       Patch(color=AMBER,label="improved EER only"),
                       Patch(color=GREY,label="overcorrected (worse)")], fontsize=9, loc="lower right")
    save(fig, "fig_smoothing_dose_dependent.png")

# ── run all ──
fns=[("architecture",fig_architecture),("seed_agreement",fig_seed_agreement),
     ("crosscorpus",fig_crosscorpus),("e9_causal",fig_e9),("actionability",fig_actionability),
     ("story1",fig_story1),("smoothing",fig_smoothing)]
print(f"Generating figures -> {OUT}")
for nm,fn in fns:
    try: fn()
    except Exception as e:
        print(f"  [FAIL] {nm}: {e}"); traceback.print_exc()
print(f"\nDONE: {len(done)}/7 figures: {done}")
