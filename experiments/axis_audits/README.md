# axis_audits — Red-Team Audit Suite for final_outputs2

Independent recomputation + adversarial stress-testing of every claim in
`final_outputs2` (geometric axis / hardness law, I1–I7 & J1–J6). All audits
load only cached raw artifacts (embeddings, logits, scores under
`experiments/results/`) — no GPU, no original analysis code paths.

Run any audit standalone: `python3 auditN_*.py` (order matters only in that
A1 and A10 read the JSON outputs of A2/A3/A4/A5/A8 — run those first).

| script | question |
|---|---|
| `audit1_multiplicity.py` | global p-hacking audit: 83 recorded tests, BH-FDR, winner's-curse split-half simulation |
| `audit2_sdalong_claim.py` | verify the (hardcoded) headline sd_along LOSO R²=0.273; jackknife, confounds, estimator sensitivity |
| `audit3_asvspoof_prospective.py` | J4/J5 "pre-registered" P3 claim: exact perm p, Holm, LOAO, replication independence, prereg timeline |
| `audit4_axis_rotation.py` | cos(w_MLAAD,w_ITW)=0.05: random-768d null, split-half axis reliability, frame sensitivity |
| `audit5_fusion_claims.py` | I7 verification; ITW fusion mislabel; J5-H4 decomposition (LDA-alone vs fused) + bona-speaker-leak quantification |
| `audit6_itw_speaker.py` | ITW s_orth claim: family correction, collinearity with unreported rog12, reliability ceiling |
| `audit7_agreement.py` | cross-detector agreement: CIs, same-vs-cross significance, second same-domain pair |
| `audit8_i1_causal.py` | I1 contrasts at seed-level inference; artifact-share arithmetic |
| `audit9_hardness_reliability.py` | hardness metric: split-half reliability, bona-pool bootstrap, min-utts sensitivity |
| `audit10_consistency.py` | claim-by-claim report-vs-artifact table (8 verified / 6 mislabeled / 5 misleading / 2 contradicted) |

Outputs: `audits_outputs/auditN_*/` (each has `AUDIT*_SUMMARY.md` + CSVs +
figures + results JSON), plus the roll-up `audits_outputs/COMPREHENSIVE_VERDICT.md`
and `audits_outputs/comprehensive_verdict.pdf` (rebuild with
`build_verdict_pdf.py`).

Headline: the core sd_along→hardness law, axis rotation, domain-conditional
agreement, and I7 fusion are solid; the ASVspoof21 "prospective" claims fail
multiplicity correction; the domain-shift "zero-training fusion" gains are a
supervised LDA probe in disguise; and the report layer contains two
contradicted numbers and six mislabels (see verdict §3).
