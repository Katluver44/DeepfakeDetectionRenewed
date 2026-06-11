# P3 — Paired-invariance adapter (learned nonlinear channel disentanglement)

## Verdict: **PARTIAL / NOT supported**

Frozen detector + identity-initialised residual adapter at the P1 injection point; trained 400 steps on MLAAD with BCE + λ·channel-invariance (reverb+MP3 paired). ITW/labels never used.

| condition | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-bal |
|---|---|---|---|---|---|
| frozen (ref) | 0.254 | 0.378 | 0.669 | 0.145 | 0.593 |
| adapter λ=0.0 | 0.287 | 0.358 | 0.708 | 0.107 | 0.593 |
| adapter λ=1.0 | 0.313 | 0.454 | 0.703 | 0.208 | 0.545 |

**adapter(λ=1) vs frozen**: ITW balanced acc Δ=-0.048, ITW EER Δ=+0.076, MLAAD-EER Δ=+0.059. λ=0 ablation (BCE only) isolates the invariance objective's contribution.

## Reading
- Unlike the linear projection (P1), a learned adapter can move FPR down WITHOUT paying the full FNR cost if channel and spoof are nonlinearly separable in the frozen features. The λ=1 vs λ=0 contrast shows whether the channel-invariance objective (not mere extra capacity) is what helps ITW.

## Files: p3_runs.csv, adapter_lam*.pt