# Table 2 — Hardness Law Correlations

| Predictor | Dataset | Detector | Effect | p-value |
| --- | --- | --- | --- | --- |
| sd_along | MLAAD | WavLM-GAT | +0.316 (LOSO R²) | 0.0005 |
| sd_along | MLAAD | AASIST-FT | +0.349 (ρ) | 0.0058 |
| sd_along | MLAAD | AASIST-ZS | +0.240 (ρ) | 0.0627 |
| vel_entropy_L12 | MLAAD | WavLM-GAT | +0.079 (LOSO R²) | 0.030 |
| vel_entropy_L12 | MLAAD | AASIST-FT | +0.342 (ρ) | 0.0070 |
| s_along | ASVspoof21 | WavLM-GAT | -0.714 (ρ) | 0.0061 |
| P3 (LDA axis proj) | ASVspoof21 | WavLM-GAT | +0.599 (ρ) | 0.0306 |
| P3 (LDA axis proj) | ASVspoof21 | AASIST-ZS | +0.643 (ρ) | 0.0178 |
| s_orth | ITW (speaker) | WavLM-GAT | +0.534 (ρ) | 0.0028 |
