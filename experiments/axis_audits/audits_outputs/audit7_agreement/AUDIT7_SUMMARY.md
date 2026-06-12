# Audit 7 — Cross-Detector Hardness Agreement

**Claim under test.** "Same-domain pairs agree ρ ≈ 0.55; cross-domain pairs ρ ≈ 0.30–0.36 regardless of architecture — hardness is training-domain-conditional, not architecture-conditional" (§3.4).

**Method.** Recomputed the 4×4 matrix from per-system hardness CSVs (n=61 MLAAD systems); 3,000-draw system bootstrap CIs for each pair; shared-bootstrap difference tests (same-domain pair minus each cross-domain pair); training-domain bookkeeping for all four detectors.

**Results** (`pairwise_agreement_ci.csv`, `samedomain_vs_cross_difference.csv`, `audit7_agreement.png`).

| pair | type | ρ | 95% CI |
|---|---|---|---|
| AASIST-FT ~ WavLM-GAT | same-domain (MLAAD/MLAAD) | 0.553 | [0.32, 0.73] |
| AASIST-ZS ~ RobustGoat | **same-domain (ASVspoof/ASVspoof)** | 0.544 | [0.33, 0.72] |
| AASIST-FT ~ AASIST-ZS | same-arch, cross-domain | 0.308 | [0.05, 0.54] |
| AASIST-FT ~ RobustGoat | cross | 0.360 | [0.08, 0.60] |
| WavLM-GAT ~ AASIST-ZS | cross | 0.320 | [0.06, 0.55] |
| WavLM-GAT ~ RobustGoat | cross | 0.301 | [0.04, 0.54] |

- **The matrix is verified, and the story is actually stronger than the paper noticed:** there are *two* same-domain pairs, not one. AASIST-ZS and RobustGoat are both ASVspoof-trained, and their agreement (0.544) is nearly identical to the MLAAD same-domain pair (0.553), while all four cross-domain pairs cluster at 0.30–0.36. Both same-domain pairs are *cross-architecture* (raw-waveform AASIST vs WavLM-based), which is precisely the paper's thesis. The paper presents 0.544 as an unexplained cell.
- The same-architecture, cross-domain pair (AASIST-FT ~ AASIST-ZS, 0.308) sits at the bottom of the range — directly supporting "domain over architecture".
- **Statistical caveat:** no individual same-vs-cross difference reaches significance at n=61 (Δρ ≈ 0.19–0.25, bootstrap p = 0.06–0.22). The pattern (2/2 same-domain pairs above all 4/4 cross-domain pairs) is consistent and the joint pattern is unlikely under exchangeability, but the paper should not imply the 0.55-vs-0.30 gap is individually significant.

**Verdict. VERIFIED, with a framing upgrade and a significance caveat.** The domain-conditional hypothesis is supported by two independent same-domain/cross-architecture pairs; the paper should claim the pattern, cite both pairs, give CIs, and avoid implying a formally significant separation at this sample size.
