# E3: Head-Decomposed Hub-Mass — Critical vs Control

## Family p-value table

| system | p_family (critical h0,h4) | p_family (control h1-3,5) | verdict |
|--------|--------------------------|--------------------------|---------|
| A01 | **0.00010** | **0.00060** | both |
| A02 | 0.34957 | 0.49835 | null |
| A03 | **0.00120** | **0.00190** | both |
| A04 | **0.00770** | **0.00960** | both |
| A05 | 0.38566 | 0.26937 | null |
| A06 | 0.46355 | 0.54175 | null |

## Interpretation

critical heads = h0, h4 (identified as high-KL heads in prior gat_l0_attention analysis)
control heads  = h1, h2, h3, h5

Outcomes:
- **critical only**: signal mediated by h0/h4 across all 3 layers
- **both**: graph-wide redistribution not specific to critical heads
- **control only**: unexpected — critical heads not carrying the hub signal
- **null**: system does not produce detectable hub-mass shift in either subset
