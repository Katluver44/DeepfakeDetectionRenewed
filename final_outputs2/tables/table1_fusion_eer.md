# Table 1 — Axis Fusion EER Results

| Dataset | Detector | EER_before | EER_after_fusion | ΔEER | p |
| --- | --- | --- | --- | --- | --- |
| MLAAD | WavLM-GAT (mean) | 0.272 | 0.163 | −0.109 | <0.0001 |
| MLAAD | WavLM-GAT (axis alone) | — | 0.187 | — | — |
| MLAAD | AASIST (zero-shot) | 0.376 | 0.116 | −0.260 | <0.0001 |
| MLAAD | AASIST (fine-tuned) | 0.200 | 0.136 | −0.064 | <0.05 |
| ITW | WavLM-GAT | 0.363 | 0.292 | −0.071 | 0.10 |
| ITW | AASIST (zero-shot) | 0.486 | 0.161 | −0.325 | <0.0001 |
| ITW | AASIST (fine-tuned) | 0.486 | 0.192 | −0.294 | <0.001 |
| ASVspoof21 | WavLM-GAT | 0.320 | — | — | — |
| ASVspoof21 | AASIST (zero-shot) | 0.073 | 0.076 | +0.003 | ns |
