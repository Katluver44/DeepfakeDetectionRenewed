# Audit 4 — Axis Rotation Across Domains (cos(w_MLAAD, w_ITW) = 0.05)

**Claims under test.** "Discriminative axes are nearly orthogonal across recording domains (cos(w_MLAAD, w_ITW) = 0.05), explaining why out-of-domain transfer fails"; "cos(w_MLAAD, w_ASVspoof21) ≈ 0.36".

**Red-team concerns.** (i) In 768-d, *random* unit vectors have E|cos| ≈ 0.029 — "near-orthogonal" is the default for any two unrelated directions, so the claim is only meaningful if each axis is itself reliably estimated. (ii) The 0.0525 artifact (`itw_fusion_test.json`) has **no committed generating script**. (iii) Cosines depend on the standardization frame.

**Method.** Recomputed all three cross-corpus cosines in two frames (MLAAD-bona standardized; raw). Random-direction null (20k pairs). Split-half axis reliability within each corpus (50 splits) + Spearman–Brown disattenuation. ITW axis convergence vs sample size.

**Results** (`audit4_results.json`, `audit4_rotation.png`, `itw_axis_convergence.csv`).

| pair | MLAAD frame | raw frame |
|---|---|---|
| MLAAD ↔ ITW | **+0.103** | +0.027 |
| MLAAD ↔ ASVspoof21 | **−0.207** | −0.140 |
| ITW ↔ ASVspoof21 | −0.238 | −0.118 |

- Null: E|cos| = 0.029, 95th pct = 0.070. The ITW cosine (0.027–0.103, claimed 0.05) is **at or barely above chance level for unrelated directions** depending on frame.
- Axis reliability is excellent: split-half cos = 0.926 (MLAAD), 0.982 (ITW), 0.981 (ASV21); ITW axis converges by n≈800/class. **So the near-orthogonality is real rotation, not estimation noise** — disattenuated cosines barely change.
- **The "≈0.36" MLAAD↔ASVspoof21 claim is contradicted**: the actual cosine is *negative* (−0.14 to −0.21). The 0.362 in the report is cos(w_mean, w_lda) *within MLAAD* from J1 — an unrelated quantity apparently transplanted. Note the negative cosine is *consistent* with J4's finding that the transferred MLAAD axis P4 predicts ASVspoof21 hardness with the wrong sign (ρ=−0.37).

**Verdict.** The qualitative rotation law (corpus-internal axes are required; transfer fails) is **VERIFIED and actually stronger than claimed** — the MLAAD axis is not merely orthogonal to but slightly *anti-aligned* with the ASVspoof21 axis. The specific numbers need repair: 0.05 is frame-dependent (report the frame and the chance floor E|cos|=0.029), and "cos(w_MLAAD, w_ASVspoof21) ≈ 0.36" must be deleted or corrected to its true negative value. The missing generating script for `itw_fusion_test.json` is a reproducibility gap.
