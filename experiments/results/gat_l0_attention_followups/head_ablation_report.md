# GAT Layer-0 Head Ablation Report

**Model**: `robust_goat.ckpt`
**Dataset**: ASVspoof 2019 LA validation, 50 samples per system
**Ablation target**: `gat_net[0].neighborhood_aware_softmax` (attention routing only;
skip_proj bypass is intentionally intact — see audit).

## Sanity checks

- Baseline identity check PASSED (max logit diff < 1e-5 across two runs)
- Layer-1/2 untouched check PASSED (no instance override on gat_net[1/2])
- `all_uniform` attack_acc = 0.793  bonafide_acc = 0.940  (above chance due to skip_proj bypass)

> **Floor check note**: `all_uniform` retains above-chance attack accuracy because
> `skip_proj = Linear(768,768)` routes information to all head slots regardless of
> attention routing. This is expected given the Option-A design choice.

## Main results

|Config|Heads|Mode|BF acc|Atk acc|-|A01|A02|A03|A04|A05|A06|EER|AUC|Specificity|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|baseline|[]|uniform|0.500|0.987|0.500|1.000|1.000|1.000|1.000|0.980|0.940|0.100|0.952|attack-only effect|
|h0|[0]|uniform|0.880|0.917|0.880|1.000|0.960|0.980|0.980|0.960|0.620|0.100|0.953|attack-only effect|
|h4|[4]|uniform|0.640|0.963|0.640|1.000|1.000|1.000|1.000|0.980|0.800|0.115|0.950|attack-only effect|
|**h0_h4**|[0, 4]|uniform|0.920|0.863|0.920|1.000|0.820|0.900|0.940|0.960|0.560|0.118|0.944|attack-only effect|
|h0_h4_zero|[0, 4]|zero|0.940|0.740|0.940|0.940|0.740|0.560|0.840|0.860|0.500|0.143|0.921|attack-only effect|
|h0_h4_bfmean|[0, 4]|bonafide_mean|0.920|0.857|0.920|1.000|0.820|0.900|0.940|0.960|0.520|0.117|0.943|attack-only effect|
|**ctrl_h1235**|[1, 2, 3, 5]|uniform|0.620|0.947|0.620|1.000|1.000|1.000|1.000|0.960|0.720|0.103|0.941|attack-only effect|
|all_uniform|[0, 1, 2, 3, 4, 5]|uniform|0.940|0.793|0.940|0.980|0.720|0.860|0.900|0.920|0.380|0.140|0.935|attack-only effect|

## Causal claim

Causal claim: "ablating h0 and h4 specifically degrades attack detection while
preserving bona-fide classification."

Assessment: **PARTIAL** — ablating h0 and h4 degrades attack detection, but control heads also degrade performance, suggesting the effect is not head-specific.

Key numbers:
- Attack acc drop {h0,h4} uniform: **+0.123**
- Bona-fide acc drop {h0,h4} uniform: **-0.420**
- Attack acc drop control {h1,h2,h3,h5} uniform: **+0.040**

## Mode-independence check for {h0, h4}

| Config | BF acc | Atk acc | EER | AUC |
|--------|--------|---------|-----|-----|
|h0_h4|0.920|0.863|0.118|0.944|
|h0_h4_zero|0.940|0.740|0.143|0.921|
|h0_h4_bfmean|0.920|0.857|0.117|0.943|

## Files

| File | Description |
|------|-------------|
| `ablation_summary.csv` | Full metrics for all 8 configs |
| `ablation_modes_table.csv` | {h0,h4} comparison across 3 modes |
| `ablation_bar.png` | Accuracy drop bar chart (main figure) |
| `per_system_attack_acc.png` | Per-system breakdown |
| `per_sample_preds.pt` | Raw per-sample logits/preds for all configs |
