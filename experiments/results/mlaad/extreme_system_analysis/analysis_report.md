# MLAAD Extreme System Analysis

## Extreme Groups

**Hard systems** (≥7/9 seed-condition top-quartile slots):

| System | Mean EER | WavLM dist | Phoneme KL |
|--------|----------|-----------|-----------|
| FireRedTTS-2.0 | 0.540 | 0.0334 | 0.0906 |
| Index-TTS-1.5 | 0.537 | 0.0470 | 0.0849 |
| Spark-TTS-0.5B | 0.511 | 0.0400 | 0.0648 |
| VoxCPM-1.5 | 0.471 | 0.0162 | 0.0744 |
| Higgs-Audio-V2 | 0.442 | 0.0154 | 0.0497 |
| ZipVoice | 0.418 | 0.0513 | 0.1368 |
| OuteTTS | 0.405 | 0.0146 | 0.0411 |
| griffin_lim | 0.401 | 0.0618 | 0.0602 |

**Easy systems** (≥7/9 slots):

| System | Mean EER | WavLM dist | Phoneme KL |
|--------|----------|-----------|-----------|
| orpheus-tts-0.1-finetune | 0.068 | 0.0445 | 0.1399 |
| Kitten-TTS-Nano-0.1 | 0.072 | 0.0721 | 0.0698 |
| Veena | 0.097 | 0.0864 | 0.0656 |
| Supertonic | 0.104 | 0.0286 | 0.0726 |
| kokoro | 0.113 | 0.0286 | 0.0559 |
| Kitten-TTS-Nano-0.2 | 0.113 | 0.0820 | 0.0926 |
| Ringg Squirrel TTS v1.0 | 0.115 | 0.0339 | 0.0560 |
| DeepGram | 0.115 | 0.0278 | 0.1209 |
| facebook_mms-tts-eng | 0.130 | 0.0632 | 0.0416 |
| Indri-TTS-0.1 | 0.135 | 0.0373 | 0.0652 |

## Correlation with EER (all 63 systems)

| Predictor | Spearman ρ | p | Pearson r | p |
|-----------|-----------|---|----------|---|
| WavLM cosine dist | -0.222 | 0.080 | -0.268 | 0.034 |
| Phoneme KL div    | -0.185 | 0.147 | -0.202 | 0.113 |

## Hard vs Easy Group Comparison

### WavLM cosine distance to bonafide
Hard mean: 0.0350 | Easy mean: 0.0504 | Δ=-0.0155 | Cohen d=-0.722 | perm p=0.147

### Phoneme attention KL from bonafide
Hard mean: 0.0753 | Easy mean: 0.0780 | Δ=-0.0027 | Cohen d=-0.094 | perm p=0.853

## Interpretation

- WavLM distance does NOT significantly predict EER (ρ=-0.222, p=0.080). Embedding proximity to bonafide does not explain detection difficulty.
- Phoneme KL divergence does NOT significantly predict EER (ρ=-0.185, p=0.147). Attention-level differences from bonafide do not explain difficulty.
- Hard vs easy WavLM distance difference is **not significant** (perm p=0.147).
- Hard vs easy phoneme KL difference is **not significant** (perm p=0.853).
- **Notable violations of the KL hypothesis:** Higgs-Audio-V2 (hard, KL=0.0497 — very low), OuteTTS (hard, KL=0.0411 — very low), orpheus-tts-0.1-finetune (easy, KL=0.1399 — very high), DeepGram (easy, KL=0.1209 — very high)
