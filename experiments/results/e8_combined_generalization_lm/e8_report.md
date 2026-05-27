# E8: Combined TTS+VC Generalization Test

## Datasets
- **MLAAD** (English subset, 64 TTS systems): mueller91/MLAAD-tiny
  - License: CC-BY-NC-4.0
  - Citation: Müller et al. 2024, arXiv:2401.09512
- **VCC2020 Task 1** (intra-lingual English VC, T01–T33): Zenodo 4433173
  - License: ODbL (audio)
  - Citation: Yi et al. 2020, DOI 10.21437/VCC_BC.2020-14

## Exclusion notes
- MLAAD: English subset only (other languages confound phoneme-recognizer analysis).
  Language identified via directory path `fake/en/{system}/`.
- VCC2020: Task 1 (intra-lingual English) only. Task 2 (cross-lingual: Finnish/
  German/Mandarin targets) excluded to avoid language confound.

## Headline metrics (200-sample combined set)

| metric | value | 95% CI |
|--------|-------|--------|
| Balanced accuracy | 0.4861 | [0.4306, 0.5350] |
| TTS recall (MLAAD) | 0.8611 | [0.7727, 0.9306] |
| VC recall (VCC2020) | 0.1111 | [0.0429, 0.1884] |
| TTS precision | 0.4921 | - |
| VC precision | 0.4444 | - |
| F1 TTS | 0.6263 | - |
| F1 VC | 0.1778 | - |
| AUC | 0.4828 | [0.3895, 0.5815] |

## Hypothesis outcomes

- **H1**: FAIL
- **H2**: FAIL
- **H3**: FAIL
- **H4**: PASS
- **H5**: FALSIFIED

**Decision: H5 FALSIFIED → re-run with length-matched subsampling. If still positive, claim survives.**