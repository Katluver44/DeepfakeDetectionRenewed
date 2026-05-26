# E4: Per-System EER Under Attention Ablation (All 3 Layers)

## Conditions

- **baseline**: no ablation
- **attn_ablated**: zero ALL 6 heads at ALL 3 layers (skip path only)
- **critical_only**: zero h0,h4 at ALL 3 layers

## Results

| system | EER baseline | EER attn_ablated | EER critical_only | Δ(attn_abl) | Δ(crit_only) |
|--------|-------------|-----------------|------------------|------------|-------------|
| A01 | 0.0400 | 0.0600 | 0.0600 | +0.0200 | +0.0200 |
| A02 | 0.1000 | 0.0800 | 0.1200 | -0.0200 | +0.0200 |
| A03 | 0.0600 | 0.1600 | 0.1600 | +0.1000 | +0.1000 |
| A04 | 0.0600 | 0.1000 | 0.1000 | +0.0400 | +0.0400 |
| A05 | 0.1000 | 0.0600 | 0.0800 | -0.0400 | -0.0200 |
| A06 | 0.2600 | 0.2000 | 0.2600 | -0.0600 | +0.0000 |

## Pre-specified predictions

- A01/A03/A04 degrade under attn_ablated (engage GAT routing path)
- A05/A06 show small degradation (skip-route hypothesis: skip_proj carries signal even without attention)
- critical_only degrades proportionally to attn_ablated for h0/h4-mediated systems
