# Per-System Head Ablation — Run Metadata

- **Date**: 2026-05-19
- **Checkpoint**: `models/robust_goat.ckpt`
- **Seed**: 42
- **N per class**: 50
- **Dataset split**: validation  (`Bisher/ASVspoof_2019_LA`)
- **Suspect heads**: [0, 4]
- **Control heads**: [1, 2, 3, 5]  (ctrl_h1235 from head_ablation.py)
- **Ablation mechanism**: post-softmax uniform replacement — for each ablated head h, sets `attn[:, h, 0] = 1 / in_degree(target_node)` for every edge, applied to `gat_net[0].neighborhood_aware_softmax` only; layers 1 and 2 are untouched; skip_proj at layer 0 is untouched.
- **Conditions**: clean (no ablation), suspect ({h0,h4} uniform), control ({h1,h2,h3,h5} uniform), all_heads ({h0–h5} uniform)
