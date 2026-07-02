# MOSS@COLM 2026 — Track 1 (Small-Scale Frontier) submission

Title: "Low-Cost Geometric Directions in Frozen Speech Models: Where a
Training-Free Deepfake-Hardness Signal Helps — and Where It Does Not"

## Build
    pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
Main body = 4 pages; references follow (citations are unlimited and do not count
toward the 4-page main-body limit, per the MOSS/COLM CfP). Compiles on Overleaf.

## Formatting status
- Built on the OFFICIAL COLM 2026 template you provided (colm2026_conference.sty
  is byte-identical to the official file). Line numbers are on via the official
  mechanism (`\usepackage{lineno}` + `\linenumbers`, as in the official shell).
- Palatino is enforced throughout (the earlier `\usepackage{times}` — a Palatino
  override / template violation — was removed).
- NOTE on venue: MOSS Track 1 nominally uses the MOSS style file
  ("style_file_MOSS2025" on the CfP's Google Drive, which I could not download
  here). The COLM 2026 style is the closest official proxy and is single-column
  Palatino as MOSS requires; if the MOSS .sty is mandatory, drop it beside
  main.tex and swap the one `\usepackage[submission]{colm2026_conference}` line
  (remove the two `lineno` lines if the MOSS style already numbers lines).

## Files
    main.tex                       4-page manuscript (official COLM 2026 style)
    references.bib                 bibliography (Zhang et al. authorship corrected)
    figs/fig1_architecture.pdf     WavLM-GAT architecture — ORIGINAL redraw (see note)
    figs/fig2_results.pdf          law / axis-rotation / headroom-lever
    make_figs.py + fig1a_points.json   figure regeneration (real MLAAD scatter data)
    colm2026_conference.sty/.bst, fancyhdr.sty, natbib.sty, math_commands.tex
                                   (official COLM 2026 template files)

## Figure-license note
The architecture diagram is an ORIGINAL redraw of the pipeline described in
Zhang et al. (2025), arXiv:2412.12619, NOT a copy of their figure. That paper is
under arXiv's non-exclusive distribution license (authors retain copyright), so
their figure may not be reproduced without permission; a redraw + citation is the
copyright-clean path.

## Scope / naming
- ITW and ASVspoof-2021 content removed completely.
- Axis rotation shown MLAAD <-> ASVspoof-2019 only.
- "goat" checkpoints named professionally: weak = "WavLM-GAT (500 utt)",
  full = "WavLM-GAT (full)".
- Numbers reconciled to experiments/paper_regen/regenerated_numbers.json.

## FLOP / compute (MOSS Track 1 requirement)
Frozen ~95M encoder + ~13M trainable head (<< 3B params). Largest single
training run ~1e16 FLOPs (<< 1e20 soft cap). Reported in the Reproducibility
statement.
