# Paper figures

Every figure regenerates from committed scripts; no number is hand-typed. Each
script loads values from the artifact files listed below and asserts them
against the paper/roadmap values before drawing. Run any one with
`python3 paper/figures/<script>.py`; it writes both `.pdf` (vector) and `.png`.

| Figure | Script | Data artifacts | Paper claim |
|---|---|---|---|
| Fig 1 — WavLM-GAT concept | `fig1_concept.py` | (schematic; no data) | §2 detector architecture; L12 geometry tap |
| Fig 2a — the hardness law | `fig2a_law_scatter.py` | `i3_position_geometry/{system_position.csv,utt_position.csv}`; reliability 0.863 (audit9) | §3 ρ=0.59, p<1e-6, LOSO R²=0.277, ~32% of ceiling |
| Fig 2b — the axis rotates | `fig2b_axis_rotation.py` | `audit4_axis_rotation/audit4_results.json`; `c_asvspoof_fusion/asvspoof_fusion_results.json` | §4 within-corpus ~0.93; cross-corpus ≈−0.08 & negative; chance ±0.07 |
| Fig 2c — headroom | `fig2c_headroom.py` | `e_mini_goat_fusion/…results.json`; `i7_axis_fusion/{i7_headline.csv,i7_stats.json}`; `c_asvspoof_fusion/…results.json` | §5 weak ΔEER −0.030, mid −0.109, strong ≈0 |
| Fig 3 — quartile mechanism (new) | `fig3_quartile_mechanism.py` | `i7_axis_fusion/i7_quartiles.csv` | §3/§5 fusion gain grows +0.036→+0.220 easy→hard |

## Changes made vs the original figures

- **Fig 2b** was redrawn (roadmap bug #3): the old panel annotated "chance 0.08"
  while the *observed* cross-corpus cosine is ≈−0.08 and the true 768-d chance
  floor is ≈0.03. New form is a 4×4 cosine matrix (MLAAD, ASVspoof19,
  ASVspoof21, ITW) with within-corpus split-half reliability on the diagonal and
  the ±0.07 chance band marked on the colour scale. Two cells (ASV19 diagonal,
  ASV19↔ITW) have no committed cross-cosine and are hatched `n/a`, not invented.
  (Layout was also fixed: the in-figure caption that overlapped the title/colorbar
  was removed — that prose belongs in the LaTeX caption.)
- **Fig 2c** was redrawn (roadmap bug #4): the old three bars labelled
  "WavLM-GAT (500 utt)/(full)/(MLAAD)" mixed detector strength and corpus on one
  axis, with a y-axis reading EER but annotations reading ΔEER. New form is three
  regimes (weak / mid / strong), each a paired bar (detector alone vs + fusion),
  y-axis = EER, ΔEER + 95% CI annotated per regime.
- **Fig 2a** reliability-band label moved to the caption to declutter the point cloud.
- **Fig 3** is new: the law-as-mechanism plot (fusion gain concentrates on the
  hardest quartile), which the paper describes but never showed.

## Number corrections / confirmations for the paper text

- **Strong-detector EER 0.078 is CORRECT** — not a bug. `c_asvspoof_fusion`
  `detector_alone_eer` = 0.078125 (mean over 3 seeds). The 0.071 that looked like a
  mismatch is just the single-seed s1 value (`detector_alone_eer_per_seed.s1` =
  0.07125); the mean is the right number to report. Fig 2c uses the mean.
- Fig 2a: `hard_shared` gives Spearman ρ = **0.589** (paper rounds to 0.60; within
  tolerance). Hardest system = Spark-TTS-0.5B (0.601), easiest = Kitten-TTS-Nano-0.1 (0.012).
- Fig 2b: reliabilities MLAAD 0.926 / ASV21 0.981 / ITW 0.982; cross MLAAD↔ASV19
  −0.078, MLAAD↔ASV21 −0.207, MLAAD↔ITW +0.103, ASV19↔ASV21 +0.747, ITW↔ASV21 −0.238;
  chance E|cos|=0.029, 95th pct 0.070.

## Not produced

- **ITW adaptation-budget figure** — intentionally omitted per instruction.
- **LaTeX source for the title kerning bug** ("T raining-Free") — only `main (1).pdf`
  is in the repo; no `.tex` source was found, so the kerning fix must be applied
  wherever the source lives (not in this repo tree as of this pass).
