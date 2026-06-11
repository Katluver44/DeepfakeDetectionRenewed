# Table 4 — Architecture-General vs Detector-Conditional Features

| Feature | Type | Effect Size | p-value | Notes |
| --- | --- | --- | --- | --- |
| sd_along (spread) | General | ρ≈+0.35 | p<0.01 | Both MLAAD-GAT and AASIST-FT |
| vel_entropy (T) | General | ρ≈+0.34 | p<0.01 | AASIST-FT (fails cross-domain) |
| s_along (mean position) | Conditional | ns for AASIST | p>0.1 | WavLM-GAT only in-domain |
| Fusion gain (axis) | General | ΔEER up to −0.33 | p<0.001 | Both architectures under shift |
| P3 LDA predictor | General | ρ≈+0.60–0.64 | p<0.05 | Both MLAAD-GAT and AASIST-ZS on ASVspoof21 |
| ITW speaker law (s_orth) | Conditional | ρ=+0.53 (WavLM only) | p=0.003 | Fails for AASIST |
| Rotation law (cos=0.05) | General | Domain-agnostic | — | MLAAD and ITW axes near-orthogonal |
