#!/usr/bin/env python3
"""Build final_outputs/paper_plan_story3.pdf — mirrors the website plan, with the
generated figures embedded and a verified related-work list (26 refs)."""
import os
from pathlib import Path
from PIL import Image
from fpdf import FPDF

BASE = Path(__file__).resolve().parents[1]
FIG  = BASE/"outputs"/"figures"
PDF_OUT = BASE/"final_outputs"/"paper_plan_story3.pdf"
FONTS = "/usr/share/matplotlib/mpl-data/fonts/ttf"

INK=(26,29,36); BLUE=(13,59,102); ACC=(31,95,191); MUT=(91,100,114)
GREEN=(26,127,75); RED=(179,38,30); PURPLE=(107,63,160); AMBER=(150,95,10); LINE=(210,216,224)

class Doc(FPDF):
    def header(self):
        if self.page_no()==1: return
        self.set_y(8); self.set_font("DV","",8); self.set_text_color(*MUT)
        self.cell(0,5,"Objective Hardness Geometry in Deepfake Speech Detectors — Paper Plan",align="L")
        self.cell(0,5,f"p.{self.page_no()}",align="R"); self.ln(7)
        self.set_draw_color(*LINE); self.set_line_width(0.2); self.line(15,18,195,18)
    def footer(self):
        self.set_y(-12); self.set_font("DV","I",7); self.set_text_color(*MUT)
        self.cell(0,5,"Planning document — grounded in repository results · E8 excluded · C:T = 70:30",align="C")

pdf=Doc(orientation="P",unit="mm",format="A4")
pdf.add_font("DV","",f"{FONTS}/DejaVuSans.ttf")
pdf.add_font("DV","B",f"{FONTS}/DejaVuSans-Bold.ttf")
pdf.add_font("DV","I",f"{FONTS}/DejaVuSans-Oblique.ttf")
pdf.set_auto_page_break(True,margin=16)
pdf.set_margins(15,16,15)
EPW=180

# Force every multi_cell to return the cursor to the left margin on the next line.
# (fpdf2's default new_x=RIGHT corrupts x in looped helpers like bullet()/refgroup().)
from fpdf.enums import XPos, YPos
_mc = pdf.multi_cell
def _safe_mc(*a, **k):
    k.setdefault("new_x", XPos.LMARGIN); k.setdefault("new_y", YPos.NEXT)
    return _mc(*a, **k)
pdf.multi_cell = _safe_mc

def h1(t):
    if pdf.get_y()>250: pdf.add_page()
    pdf.ln(2); pdf.set_font("DV","B",15); pdf.set_text_color(*BLUE)
    pdf.multi_cell(EPW,7,t); pdf.set_draw_color(*ACC); pdf.set_line_width(0.4)
    pdf.line(15,pdf.get_y()+0.5,195,pdf.get_y()+0.5); pdf.ln(2.5); pdf.set_text_color(*INK)
def h2(t):
    pdf.ln(1.5); pdf.set_font("DV","B",11.5); pdf.set_text_color(*ACC)
    pdf.multi_cell(EPW,5.5,t); pdf.ln(0.5); pdf.set_text_color(*INK)
def body(t,size=9.5,gap=1.2):
    pdf.set_font("DV","",size); pdf.set_text_color(*INK); pdf.multi_cell(EPW,4.7,t); pdf.ln(gap)
def bullet(items,size=9.5):
    pdf.set_font("DV","",size); pdf.set_text_color(*INK)
    for it in items:
        x=pdf.get_x(); pdf.set_x(x+3); pdf.multi_cell(EPW-3,4.6,"•  "+it);
    pdf.ln(1)
def italic(t,size=8.5):
    pdf.set_font("DV","I",size); pdf.set_text_color(*MUT); pdf.multi_cell(EPW,4.3,t); pdf.ln(1); pdf.set_text_color(*INK)

def table(rows,widths,size=8.3,header=True,aligns=None):
    pdf.set_font("DV","",size)
    if aligns is None: aligns=["LEFT"]*len(widths)
    with pdf.table(width=sum(widths),col_widths=widths,text_align=aligns,
                   first_row_as_headings=header,line_height=4.6,
                   borders_layout="MINIMAL") as t:
        for r in rows:
            t.row([str(c) for c in r])
    pdf.ln(2)

def figure(name,caption,max_w=180,max_h=205):
    p=FIG/name
    im=Image.open(p); pw,ph=im.size; ar=ph/pw
    w=max_w; h=w*ar
    if h>max_h: h=max_h; w=h/ar
    if pdf.get_y()+h+12 > (pdf.h-pdf.b_margin): pdf.add_page()
    x=(pdf.w-w)/2; pdf.image(str(p),x=x,w=w); pdf.ln(1)
    pdf.set_font("DV","I",8); pdf.set_text_color(*MUT); pdf.multi_cell(EPW,4,caption); pdf.ln(2.5); pdf.set_text_color(*INK)

# ═══════════════ COVER ═══════════════
pdf.add_page()
pdf.ln(14)
pdf.set_fill_color(*BLUE); pdf.rect(0,pdf.get_y(),210,46,style="F")
pdf.set_y(pdf.get_y()+8); pdf.set_text_color(255,255,255); pdf.set_font("DV","B",19)
pdf.multi_cell(EPW,8,"Objective Hardness Geometry in\nDeepfake Speech Detectors",align="C")
pdf.set_font("DV","",11); pdf.multi_cell(EPW,6,"Paper Plan & Evidence Map",align="C")
pdf.ln(8); pdf.set_text_color(*INK)
pdf.set_font("DV","",10)
pdf.multi_cell(EPW,5,"A planning document for deciding what the paper is — not a draft. "
    "Primary narrative: Story 3. Figures generated from repository results are embedded throughout.",align="C")
pdf.ln(4)
for k,v in [("Primary contribution","Story 3 — geometric hardness factor C (causal, actionable)"),
            ("Factor weighting","C : T = 70 : 30  (T real but fragile; cautious)"),
            ("ASVspoof role","Causal validation via E9 only"),
            ("Excluded","E8 (confounded ASVspoof correlation)"),
            ("Story 1","Supporting mechanism + supplement"),
            ("Venue framing","Dual: Speech (Interspeech/ICASSP) + ML-interp (NeurIPS/ICLR)")]:
    pdf.set_x(28); pdf.set_font("DV","B",9.5); pdf.cell(45,5.5,k+":")
    pdf.set_font("DV","",9.5); pdf.multi_cell(110,5.5,v)
pdf.ln(6); pdf.set_font("DV","I",8.5); pdf.set_text_color(*MUT)
pdf.multi_cell(EPW,4.5,"Generated 2026-06-09 · companion to outputs/paper_plan_story3.html · "
    "supporting assets organised under final_outputs/{mandatory,optional}/",align="C")

# ═══════════════ 1 EXEC SUMMARY ═══════════════
pdf.add_page(); h1("1 · Executive Summary")
body("Some text-to-speech / voice-conversion systems are systematically hard for detectors — and that "
     "hardness is not random. We argue it has an objective geometric origin: the compactness of a system's "
     "speech trajectory in deep self-supervised (WavLM) space. We name this factor C and show it (i) predicts "
     "per-system hardness, (ii) is causal under direct intervention, and (iii) is actionable — targeting it "
     "lowers EER on the hardest systems.")
h2("Factor definitions")
bullet(["C (compactness) = −rog@L12 — negative radius of gyration of the WavLM layer-12 frame cloud. "
        "Higher C = more compact. Primary factor.",
        "T (trajectory irregularity) = vel_entropy@L9 — entropy of frame-to-frame velocity at layer 9. "
        "Secondary (70/30), fragile."])
h2("What's strong (lead with these)")
bullet(["Hardness is system-intrinsic — systems agree on what is hard across seeds (ρ ≈ 0.79–0.84).",
        "C ↔ hardness on MLAAD (63 systems): ρ(C,resid) = +0.29; more compact → harder.",
        "C is causal (E9): scaling compactness moves EER through baseline, beating a matched-energy shift control.",
        "C transfers to in-the-wild at the speaker level: ρ(C,EER) = +0.44 (45 speakers).",
        "C is actionable: hardness reweighting (P5) lowers the hardest (C-Q4) systems' EER by 0.023 (AUC +0.026)."])
h2("What's fragile (handle carefully)")
bullet(["T is fragile — scale-dependent, not independently manipulable in E9, likely distorted in in-the-wild audio.",
        "Encoder scope — C validates on WavLM & wav2vec2 but FAILS on HuBERT. Never claim \"all SSL.\"",
        "Coverage — C+T explain ≈ 9% of residual-hardness variance (LOO). Frame as \"actionable factors,\" "
        "not \"we explain hardness.\""])
italic("Verdict on Story 1 (GAT-hierarchy interpretability): mechanistically rich but narrow (ASVspoof-only, "
       "language-invariance untested). Belongs as a compressed supporting mechanism + supplement — not a pillar.")

# ═══════════════ 2 NARRATIVE REC ═══════════════
h1("2 · Narrative Recommendation")
body("Keep Story 3 as the spine. The call on every candidate sub-story:")
table([["Sub-story","Placement","Support","Completes vs distracts"],
 ["Objective hardness agreement (seeds)","MAIN","STRONG","Completes — establishes the phenomenon is real"],
 ["C & T as geometric factors (MLAAD)","MAIN","STRONG (C)","Completes — the core contribution"],
 ["C causality via intervention (E9)","MAIN","STRONG","Completes — keystone above correlation"],
 ["Actionability (P1 / P5)","MAIN","STRONG","Completes — the 'so what'"],
 ["In-the-wild support for C","MAIN","SUPPORTING","Completes — external validity"],
 ["Temporal invariance / smoothing","APPENDIX (+1 fig)","MEDIUM","Completes — motivates T; hardness targetable"],
 ["Story 1 — GAT hierarchy","APPENDIX (+1 fig)","MEDIUM","Completes — a 'why' mechanism, if compressed"],
 ["Story 2 — language overfitting","OMIT","WEAK","Distracts — one language pair; other paper"],
 ["Story 4 — taxonomy fragility","OMIT","WEAK","Distracts — negative replication; other paper"]],
 widths=[52,30,26,72])
h2("Reframe required: the smoothing sub-story")
body("The earlier framing (\"smoothing improved detection\") is wrong — but it is also not a failure. "
     "Smoothing-as-augmentation (P4) is a dose-dependent per-system lever: applied uniformly the mean regresses "
     "(EER Δ=+0.037), yet 8 systems improve on both EER and AUC (MegaTTS3 −0.134, Resemble.ai −0.078, "
     "FireRedTTS-2.0 −0.072). It overcorrects systems that don't need it and helps those that do ⇒ hardness is "
     "targetable. Separately, smoothing-as-intervention establishes causality (MA k=3→k=10: EER 0.421→0.571). "
     "Do not write \"smoothing uniformly improved detection,\" and do not write \"smoothing augmentation failed.\"")

# ═══════════════ 3 STORY 3 FRAMING ═══════════════
h1("3 · Story 3 as the Primary Framing")
body("Thesis. Deepfake-detection hardness has an objective, measurable geometric origin in deep SSL space: "
     "systems whose layer-12 WavLM trajectories are more compact (C) are systematically harder to detect; this "
     "factor is causal and can be acted on. A secondary dynamic factor (T) correlates with hardness but is fragile.")
h2("Evidence arc (paper spine)")
bullet(["Objective — hardness rankings agree across seeds → not model noise.",
        "Geometric correlate — C predicts per-system hardness on MLAAD; T adds a weaker, correlated signal.",
        "Independent — C ⊥ T (r≈+0.11; separate PCA components; positive unique partials).",
        "Causal — intervening on C inside the frozen detector moves EER beyond a matched-energy shift control (E9).",
        "Actionable — C/T-guided calibration (P1) and reweighting (P5) lower hardest-system EER.",
        "External validity — C transfers to in-the-wild at the speaker level; T treated cautiously there."])
italic("Dual-venue note — Speech: foreground EER tables, benchmarks, actionability. "
       "ML/interp: foreground the causal-intervention design (E9) and representation-geometry framing.")
figure("fig1_architecture_ct_pipeline.png",
       "Figure 1. Detector architecture and the C/T extraction points: a frozen WavLM encoder feeds a trained "
       "phoneme-GAT + BiLSTM; C is read as −rog at layer 12, T as velocity-entropy at layer 9.", max_h=170)

# ═══════════════ 4 STORY 1 HONEST REVIEW ═══════════════
pdf.add_page(); h1("4 · Story 1 Honest Review")
body("Core contribution if primary: a hierarchical functional dissociation in a phoneme-GAT — early layers "
     "linearly encode vocoder/system identity (multiclass probe 91.8%@gat_l0 → 75%@bilstm) while binary "
     "detection strengthens with depth; activation patching shows vowels/diphthongs drive the binary decision "
     "(Δ≈+1.95) whereas sibilants/nasals drive system-ID (Δ≈+0.004).")
table([["Question","Answer"],
 ["Strong enough alone?","No — ASVspoof A01–A06 only (6 systems); tiny causal shifts (±0.004); language-invariance untested."],
 ["Best role","Mechanistic explanation supporting Story 3 (why compact systems resist separation) + supplement."],
 ["Genuinely novel","Fingerprint-destruction ↔ detection-creation layer trade-off; dual-subspace phoneme-class causal dissociation."],
 ["Speculative now","Language-invariance; cross-dataset generality; that destruction is 'purposeful'."],
 ["Experiments to add","Probe hierarchy + activation patching on MLAAD by language; intermediate-layer (L1) patching."],
 ["Into main paper","≤ 1 composite figure (probe hierarchy + phoneme-class causal bars)."]],
 widths=[40,140])
body("Final verdict: KEEP AS SUPPORTING EVIDENCE (compressed mechanism section for Story 3); bulk to supplement.",size=9.5)
figure("fig_story1_mechanism.png",
       "Figure 2. Story 1 mechanism. Left: system-ID probe accuracy falls L0→BiLSTM while binary detection AUC "
       "rises (fingerprint destruction ↔ detection creation). Right: phoneme-class causal contribution — "
       "vowels/diphthongs drive the binary decision; sibilants are causally weak.", max_h=95)

# ═══════════════ 5 CORE EVIDENCE FIGURES ═══════════════
pdf.add_page(); h1("5 · Core Evidence Figures (main paper)")
figure("fig_hardness_seed_agreement.png",
       "Figure 3. Hardness is objective. Per-system EER ordering is stable across random seeds (left); "
       "seed-vs-seed Spearman ρ = +0.84, mean pairwise ρ = +0.79 (right).", max_h=88)
figure("fig_C_hardness_crosscorpus.png",
       "Figure 4. More compact (higher C) → harder, consistently across corpora: MLAAD systems ρ=+0.29 and "
       "in-the-wild speakers ρ=+0.44. (ASVspoof's opposite correlation, E8, is excluded.)", max_h=92)
figure("fig_e9_causal_C_main.png",
       "Figure 5. C is causal. Scaling compactness inside the frozen detector (red) moves EER through baseline "
       "and beats the matched-energy shift control (blue) in the clean moderate regime; both detectors agree. "
       "More compact → harder.", max_h=92)
figure("fig_actionability_hardsystems.png",
       "Figure 6. Actionability. Hardness reweighting (P5) lowers EER on the hardest C-Q4 systems (−0.023); "
       "calibration (P1) improves the average but not specifically C-Q4 (shown honestly).", max_h=92)
figure("fig_smoothing_dose_dependent.png",
       "Figure 7. Smoothing augmentation is a dose-dependent lever: 8 systems genuinely improve on EER+AUC "
       "(MegaTTS3 −0.134) while the rest are overcorrected — evidence that hardness is targetable. (Appendix.)", max_h=120)

# ═══════════════ 6 CLAIMS STRENGTH ═══════════════
pdf.add_page(); h1("6 · Claims Strength")
h2("Core claims — STRONG evidence")
table([["Claim","Basis"],
 ["Hardness is system-intrinsic across seeds","seed ranking agreement, ρ≈0.79–0.84 (5 seeds)"],
 ["C correlates with per-system hardness (MLAAD)","ρ(C,resid)=+0.29; r≈+0.35, LOO R²≈0.069 (63 systems)"],
 ["C is causal within the detector","E9: moderate regime EER 0.067→0.090 (compact) / →0.056 (expanded), beats shift control"],
 ["C is actionable on hardest systems","P5 C-Q4 EER −0.023, AUC +0.026; T-Q4 EER −0.020, AUC +0.027"],
 ["C and T are independent factors","r(C,T)≈+0.11; separate PCA components; positive unique partials"]],
 widths=[70,110])
h2("Supporting claims")
table([["Claim","Basis"],
 ["C transfers to in-the-wild (speaker level)","ρ(C,EER)=+0.44, n=45; mostly between-speaker"],
 ["Temporal smoothness is a hardness factor","smoothing intervention MA k=3→k=10: EER 0.421→0.571"],
 ["Smoothing is a dose-dependent lever","P4: 8 systems improve on EER+AUC (MegaTTS3 −0.134)"],
 ["C generalizes across encoders (partial)","WavLM r=+0.348, wav2vec2 r=+0.346"],
 ["Adversarial training exploits C","robustness-mechanism analysis (ΔEER vs C; head entropy collapse)"]],
 widths=[70,110])
h2("Claims to avoid overstating")
table([["Tempting claim","Say instead"],
 ["T is a robust universal factor","T is correlated but fragile; near-constant in ASVspoof → neither proven nor refuted there"],
 ["C generalizes across all SSL encoders","C validates on WavLM & wav2vec2; fails on HuBERT"],
 ["C predicts hardness on ASVspoof (corr.)","Omit — ASVspoof = causal E9 only; E8 excluded"],
 ["Smoothing improves detection","Removing temporal variance hurts; augmentation is a targeted lever"],
 ["The GAT hierarchy is universal","Demonstrated on ASVspoof A01–A06; generality untested"],
 ["We explain deepfake hardness","We identify actionable factors; ≈65% residual unexplained"]],
 widths=[62,118])
italic("Six reviewer danger zones — keep claims tight: cross-dataset · cross-encoder (HuBERT) · in-the-wild (codec→T) "
       "· temporal smoothing · GAT-mechanism generality · explained variance.")

# ═══════════════ 7 SUPPORTING ASSETS MAP ═══════════════
pdf.add_page(); h1("7 · Supporting Assets — final_outputs/")
body("Every figure, result and summary is grouped by sub-narrative under final_outputs/. MANDATORY = main-paper "
     "evidence; OPTIONAL = supplement.")
h2("mandatory/  (7 sub-narratives)")
table([["Folder","Key contents"],
 ["00_architecture","fig1 schematic (svg+png), research.md"],
 ["01_objective_hardness_seed_ranking","fig_hardness_seed_agreement, seed_stability_summary.json, per-seed EER csv"],
 ["02_geometry_C_and_T","C→hardness, orthogonality, layerwise-R², E1 cross-encoder; final_report.md, orthog report"],
 ["03_causality_E9","fig_e9_causal_C_main, e9_summary.md, e9_causal_C.csv"],
 ["04_actionability_P1_P5","fig_actionability, P1/P5 summaries + per-system csvs"],
 ["05_in_the_wild_C","e6_itw_speaker_ct_eer, E5/E6 ITW summaries + speaker CT csv"],
 ["06_mechanism_story1_composite","fig_story1_mechanism + probe/patching stats csvs"]],
 widths=[60,120])
h2("optional/  (9 sub-narratives)")
table([["Folder","Key contents"],
 ["temporal_smoothing","dose-dependent + intervention figs; p4 summary; causal_interpretation.md"],
 ["geometry_supplement","PCA / variance-partition / Shapley / interaction / E2 / E3 + summaries"],
 ["causality_replication_and_T","E9 GOAT replication; E9 T-intervention figs + csv"],
 ["in_the_wild_supplement","ITW manifold / distributions / within-between / quartiles"],
 ["actionability_supplement","P1 extra panels; P3/P6/P7/P8 summaries"],
 ["robustness_mechanism_E7","ΔEER-vs-C, quartile-C, AUC-vs-C + robustness summary"],
 ["manifold_hardness_landscape","manifold geometry, extreme-system, residual-hardness panels"],
 ["story1_full_mechanism","probe rankings, activation patching, attention-KL, PC1, etc."],
 ["e4_asvspoof_inconclusive","E4 A01–A06 figure + summary (low usefulness)"]],
 widths=[60,120])

# ═══════════════ 8 RELATED WORK ═══════════════
pdf.add_page(); h1("8 · Related Work — Mandatory Citations")
body("26 works a credible related-work section must engage, grouped by theme. Bibliographic details verified "
     "against arXiv / proceedings (June 2026).")
def refgroup(title, refs):
    h2(title); pdf.set_font("DV","",8.7); pdf.set_text_color(*INK)
    for i,r in refs:
        x=pdf.get_x(); pdf.set_x(x+2); pdf.multi_cell(EPW-2,4.4,f"[{i}]  {r}")
    pdf.ln(1)
refgroup("Self-supervised speech encoders",[
 (1,"Baevski et al., 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS."),
 (2,"Hsu et al., 2021. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM TASLP."),
 (3,"Chen et al., 2022. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE JSTSP."),
 (4,"Babu et al., 2022. XLS-R: Self-Supervised Cross-Lingual Speech Representation Learning at Scale. Interspeech."),
])
refgroup("SSL probing & layer-wise analysis",[
 (5,"Yang et al., 2021. SUPERB: Speech Processing Universal PERformance Benchmark. Interspeech."),
 (6,"Pasad, Chou & Livescu, 2021. Layer-wise Analysis of a Self-Supervised Speech Representation Model. IEEE ASRU."),
 (7,"Pasad, Shi & Livescu, 2023. Comparative Layer-wise Analysis of Self-Supervised Speech Models. ICASSP."),
 (8,"Alain & Bengio, 2017. Understanding Intermediate Layers Using Linear Classifier Probes. ICLR Workshop."),
 (9,"Hewitt & Liang, 2019. Designing and Interpreting Probes with Control Tasks. EMNLP."),
])
refgroup("Anti-spoofing corpora & detectors",[
 (10,"Wang et al., 2020. ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech. Computer Speech & Language."),
 (11,"Yamagishi et al., 2021. ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection. ASVspoof Workshop."),
 (12,"Müller et al., 2022. Does Audio Deepfake Detection Generalize? (In-the-Wild dataset). Interspeech. arXiv:2203.16263."),
 (13,"Müller et al., 2024. MLAAD: The Multi-Language Audio Anti-Spoofing Dataset. IJCNN. arXiv:2401.09512."),
 (14,"Jung et al., 2022. AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. ICASSP."),
 (15,"Tak et al., 2021. End-to-End Anti-Spoofing with RawNet2. ICASSP."),
 (16,"Tak et al., 2022. ASV Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation. Odyssey. arXiv:2202.12233."),
 (17,"Tak et al., 2022. RawBoost: A Raw Data Boosting and Augmentation Method Applied to ASV Anti-Spoofing. ICASSP."),
])
refgroup("Graph attention & mechanistic interpretability",[
 (18,"Veličković et al., 2018. Graph Attention Networks. ICLR."),
 (19,"Vig et al., 2020. Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS. arXiv:2004.12265."),
 (20,"Meng et al., 2022. Locating and Editing Factual Associations in GPT (ROME / activation patching). NeurIPS. arXiv:2202.05262."),
])
refgroup("Representation geometry & instance hardness",[
 (21,"Papyan, Han & Donoho, 2020. Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training. PNAS. arXiv:2008.08186."),
 (22,"Ansuini et al., 2019. Intrinsic Dimension of Data Representations in Deep Neural Networks. NeurIPS. arXiv:1905.12784."),
 (23,"Smith, Martinez & Giraud-Carrier, 2014. An Instance Level Analysis of Data Complexity (instance hardness). Machine Learning."),
 (24,"Bengio et al., 2009. Curriculum Learning. ICML."),
 (25,"Shrivastava, Gupta & Girshick, 2016. Training Region-based Object Detectors with Online Hard Example Mining (OHEM). CVPR."),
 (26,"Swayamdipta et al., 2020. Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. EMNLP."),
])
italic("Venue tilt — Speech: foreground refs 10–17 (ASVspoof/AASIST/RawBoost). ML/interp: foreground refs 6–9, 18–22.")

# ═══════════════ 9 FINAL REC ═══════════════
h1("9 · Final Recommendation")
body("Write one paper, Story 3, organized around C — a geometric hardness factor that is objective "
     "(seed-agreement), causal (E9 beats a matched-energy control) and actionable (P1/P5) — with T as a 70/30 "
     "fragile companion we caveat but do not denounce.")
bullet(["Corpus roles: MLAAD = correlational + actionability backbone; ASVspoof = causal keystone via E9 only "
        "(E8 excluded); in-the-wild = external validity for C.",
        "Story 1 → one compressed mechanism section + supplement (supporting, not a pillar).",
        "Drop Stories 2 & 4 to a future paper.",
        "Guard the six danger zones; the seven mandatory-folder sub-narratives are the main-paper backbone."])

pdf.output(str(PDF_OUT))
print("wrote", PDF_OUT, f"({PDF_OUT.stat().st_size//1024} KB, {pdf.page_no()} pages)")
