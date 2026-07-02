# Final submission — MOSS@COLM 2026, Track 1 (Small-Scale Frontier)

**Title:** "Low-Cost Geometric Directions in Frozen Speech Models: Where a
Training-Free Deepfake-Hardness Signal Helps — and Where It Does Not"

## Build
```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Sections 1–5 (Introduction → Limitations and Conclusion) occupy the 4-page main
body. The Reproducibility statement and References follow on page 5+, which the
MOSS/COLM CfP does not count toward the 4-page limit.

## Format
- Official COLM 2026 template (`colm2026_conference.sty/.bst`, `fancyhdr.sty`,
  `natbib.sty`). Double-blind, line numbers on, Palatino.
- MOSS Track 1 compute: frozen ~95M encoder (WavLM-base, 94.7M) + ~13M head
  (« 3B params); largest run ~1e16 FLOPs (« 1e20). Stated in the Reproducibility
  statement.

## Structure
1. Introduction
2. Setup — Detector, Corpora and hardness, Objective hardness (seed-convergence),
   Compute. Architecture diagram: `figs/fig1_architecture.pdf`.
3. Experiments
   - 3.1 The axis and its spread (frozen L12 embeddings, why layer 12, axis
     construction, why LOSO, sd_along). Concept diagram: `figs/fig1_concept.pdf`.
   - 3.2 Results: the law (`fig2a`) and its rotation (`fig2b`).
   - 3.3 Verification (independent recomputation, global Benjamini–Hochberg,
     K6 channel-confound control, Spearman–Brown reliability).
4. Axis fusion improves performance under detector headroom
   (`fig_fusion_asvspoof`, `fig_fusion_mlaad`, `fig3_quartile_mechanism`).
5. Limitations and conclusion.

## Figures
- `figs/fig1_architecture.pdf` — WavLM-GAT architecture (original redraw).
- `figs/fig1_concept.pdf` — natural–synthetic axis visualization (regen:
  `figs_src/fig_concept.py`).
- `figs/fig2a_law_scatter.pdf` — the hardness law scatter (from `paper/figures/`).
- `figs/fig2b_axis_rotation.pdf` — cross-corpus cosine matrix (from `paper/figures/`).
- `figs/fig_fusion_asvspoof.pdf`, `figs/fig_fusion_mlaad.pdf` — split fusion
  panels (regen: `figs_src/fig_fusion.py`).
- `figs/fig3_quartile_mechanism.pdf` — fusion gain by difficulty quartile
  (from `paper/figures/`).

Every plotted number is read from committed artifacts under
`experiments/results/` and `experiments/paper_regen/regenerated_numbers.json`;
the figure scripts assert against those values (never hand-typed).
