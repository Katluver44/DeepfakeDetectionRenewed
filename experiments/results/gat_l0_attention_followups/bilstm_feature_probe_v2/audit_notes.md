# Audit Notes — bilstm_feature_probe_v2

## Architectural Diagram

```
ARCHITECTURE DIAGRAM — Phoneme_GAT (WavLM backbone)
=====================================================

 Input: raw audio (B, L=48000)
    │
    ▼  [FROZEN] transformer_in_phoneme_model.feature_extractor
         CNN conv stack → (B, T=149, 512)
    │
    ▼  [FROZEN] transformer_in_phoneme_model.feature_projection
         Linear(512 → 768) → (B, T=149, 768)
    │
    ◆◆◆◆◆  ← post_wavlm extraction point (mean over T=149 frames → (B, 768))
    │
    ▼  [FROZEN*] gat_model.encoder   (* alias of phoneme_model.encoder,
         12-layer WavLM transformer    phoneme_model.requires_grad_(False) was
         encoder → (B, T=149, 768)     called; weights confirmed identical
                                        between checkpoints by sha256 comparison)
    │
    ▼  [non-param] reduce_feat
         Adaptive phoneme pooling: avg frames with same CTC phoneme ID
         (B, T=149, 768) + phoneme_ids → (total_phonemes, 768)
    │
    ◆◆◆◆◆  ← pre_gat extraction point (mean per sample over phonemes → (B, 768))
    │
    ▼  [TRAINABLE] gat_model.GAT
         3-layer, 6-head GAT (GATLayer × 3)
         (total_phonemes, 768) → (total_phonemes, 768)
    │
    ▼  split into per-sample sequences, pad
    │
    ▼  [TRAINABLE] gat_model.rnn
         nn.LSTM(768 → 384, num_layers=2, bidirectional=True)
         → (B, max_N, 768), mean-pool → (B, 768)
    │
    ◆◆◆◆◆  ← post_gat extraction point (BiLSTM mean-pool → (B, 768)) [v1 cache]
    │
    ▼  [non-param] norm_feat: L2 normalize per sample
    │
    ▼  [TRAINABLE] cls_head
         Linear(768,768) → BatchNorm1d → ReLU → Dropout(0.1) → Linear(768,1)
    │
    ▼  logit (B,)

Trainable components per extraction stage:
  post_wavlm → (before any trainable component)
  pre_gat    → encoder is FROZEN (weights identical, verified)
             → reduce_feat has no parameters
  post_gat   → GAT [trainable], BiLSTM [trainable]

Frozen components:
  feature_extractor, feature_projection — part of phoneme_model (requires_grad_(False))
  encoder — alias of phoneme_model.encoder (same object); frozen same way
  phoneme_model (CTC model for phoneme IDs) — fully frozen

Weight diff between goat.ckpt and robust_goat.ckpt (verified at startup):
  encoder   : max_diff=0.000000  → IDENTICAL (frozen confirmed)
  GAT       : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)
  BiLSTM    : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)
  cls_head  : max_diff>0         → DIFFERENT (trainable, checkpoint-specific)

AUDIT VERDICT (v1 bilstm_feature_probe.py):
  v1 extraction point  : post_gat (BiLSTM mean-pool after GAT)
  extraction_correct   : false  — v1 extracted POST-GAT, not pre-GAT
  v1_claim_valid       : true   — "pre-GAT features identical across ckpts" is correct
                                   because encoder is confirmed frozen (identical weights)
  implication          : v1 results are valid but not pre-GAT;
                          v2 adds the pre-GAT extraction to complete the decomposition
```

## Extraction Point in v1 (`bilstm_feature_probe.py`)

The v1 script label: `POOLING_SCHEME = "bilstm_mean_pool_post_gat"`

Key code in `extract_bilstm_batch`:
```python
hidden_states = gat_model.encoder(hidden_states)[0]      # WavLM encoder
reduced_hs, reduced_nf, reduced_pids = reduce_feat(...)   # phoneme pooling
reduced_hs, _ = gat_model.GAT((reduced_hs, edge_index))  # ← GAT applied
lstm_out, _ = gat_model.rnn(padded)                       # ← BiLSTM applied
feat = lstm_out[i, :rnf_list[i], :].mean(0)              # mean-pool → extracted here
```

**Extraction was AFTER GAT and AFTER BiLSTM** — post-gat, not pre-gat.

## Frozen vs Trainable Components

| Component | Status | Evidence |
|---|---|---|
| feature_extractor | FROZEN | part of `phoneme_model`; `phoneme_model.requires_grad_(False)` |
| feature_projection | FROZEN | same as above |
| encoder | FROZEN (alias) | `self.encoder = self.transformer_in_phoneme_model.encoder` (alias, not copy); `phoneme_model.requires_grad_(False)` covers it |
| reduce_feat | non-parametric | pure function, no weights |
| GAT | TRAINABLE | separate `nn.Module`; in `configure_optimizers` with `lr=1e-4` |
| rnn (BiLSTM) | TRAINABLE | `nn.LSTM`; in optimizer |
| norm_feat | non-parametric | L2 divide, no weights |
| cls_head | TRAINABLE | `nn.Sequential(Linear, BN, ReLU, Dropout, Linear)` |

**Note on `self.encoder`**: The constructor assigns
`self.encoder = self.transformer_in_phoneme_model.encoder`
(a reference alias, NOT a `deepcopy`). The commented-out deepcopy code:
```python
#self.encoder = deepcopy(self.transformer_in_phoneme_model.encoder)
#self.encoder.requires_grad_(False) #originally true but they got sum bs going on
#self.encoder.train()
```
shows the original intent was a trainable copy; it was commented out and replaced with
an alias. Because `phoneme_model.requires_grad_(False)` was called, and `self.encoder`
is the same Python object as `phoneme_model`'s encoder submodule, the encoder parameters
have `requires_grad=False`. They appear in `configure_optimizers` under `"model.encoder"`,
but since `requires_grad=False`, no gradients flow and weights are never updated.

## Weight Comparison (empirical)

At script startup, both checkpoint `state_dict`s were loaded on CPU and compared:

| Component | n_params | max_abs_diff | Identical? |
|---|---|---|---|
| encoder | 89,785,712 | 0.000000 | True |
| GAT | 3,545,856 | 0.204896 | False |
| BiLSTM | 7,090,176 | 0.147423 | False |
| cls_head | 594,433 | 0.160878 | False |

The encoder is **confirmed identical** between checkpoints.

## Verdict

```
extraction_correct (was v1 pre-GAT?): False
v1_claim_valid ("pre-GAT features identical"): True
```

- The v1 label said `post_gat` and it **was** post-GAT — the label is accurate.
- The v1 rationale said "pre-GAT features identical because WavLM is frozen" — this
  claim is **empirically confirmed** by the weight comparison above.
- The v1 extraction point captured trainable-component output (GAT + BiLSTM), which is
  correct for measuring the effect of those components.
- **Why v2 is still needed**: v1 could only measure the combined GAT+BiLSTM effect.
  v2 adds pre-GAT extraction to confirm the features entering the GAT are truly identical,
  and to establish the absolute discriminability baseline at that stage.
