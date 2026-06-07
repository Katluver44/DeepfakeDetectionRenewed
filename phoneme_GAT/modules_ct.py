"""
modules_ct.py
=============
P2 ablation: Phoneme_GAT_CT — extends Phoneme_GAT to inject per-utterance
C (WavLM L12 compactness, rog) and T (WavLM L9 velocity entropy) as auxiliary
features at the classification head.

Architecture change (minimal):
  cls_head input: 768 → 768 + n_ct  (default n_ct=2: C and T)
  All other components unchanged.

The C/T features are extracted from WavLM hidden states during the frozen
forward pass (no additional model). New dimensions initialized to zero so that
loading a robust_goat.ckpt checkpoint is safe (zero init = equivalent to baseline
at start of fine-tuning).

Usage in training:
    from phoneme_GAT.modules_ct import Phoneme_GAT_CT_lit
    model = Phoneme_GAT_CT_lit(cfg)
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from phoneme_GAT.modules import Phoneme_GAT, Phoneme_GAT_lit


# ─── C/T computation (GPU, batched) ─────────────────────────────────────────

def _rog_batch(frames_batch: torch.Tensor) -> torch.Tensor:
    """
    Radius of gyration per sample.
    frames_batch: (B, T, D)
    Returns: (B,) float32
    """
    c = frames_batch.mean(dim=1, keepdim=True)              # (B, 1, D)
    sq_dist = ((frames_batch - c) ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
    return torch.sqrt(sq_dist)


def _vel_entropy_batch(frames_batch: torch.Tensor, n_bins: int = 20) -> torch.Tensor:
    """
    Adjacent-frame L2 velocity entropy per sample.
    frames_batch: (B, T, D)
    Returns: (B,) float32
    """
    B, T, D = frames_batch.shape
    vels = (frames_batch[:, 1:, :] - frames_batch[:, :-1, :]).norm(dim=-1)  # (B, T-1)
    entropies = torch.zeros(B, device=frames_batch.device, dtype=torch.float32)
    for i in range(B):
        v = vels[i]
        n = max(5, min(n_bins, len(v) // 4))
        v_min, v_max = v.min(), v.max()
        if (v_max - v_min).item() < 1e-8:
            entropies[i] = 0.0
            continue
        hist = torch.histc(v, bins=n, min=v_min.item(), max=(v_max + 1e-6).item())
        h = hist.float() + 1e-8
        h = h / h.sum()
        entropies[i] = -(h * torch.log(h)).sum()
    return entropies


# ─── CT-augmented model ───────────────────────────────────────────────────────

class Phoneme_GAT_CT(Phoneme_GAT):
    """
    Phoneme_GAT extended with C/T feature injection at the classification head.

    n_ct controls how many scalar features to inject:
      n_ct=1: C only (rog@L12)
      n_ct=2: C and T (rog@L12, vel_entropy@L9)      ← default (P2)
      n_ct=3: C, T, and −T_window11 (P3)
    """

    def __init__(self, backbone: str = 'wavlm', use_raw: int = 0,
                 use_GAT: int = 1, n_edges: int = 10, n_ct: int = 2):
        super().__init__(backbone=backbone, use_raw=use_raw,
                         use_GAT=use_GAT, n_edges=n_edges)
        self.n_ct = n_ct

        # Replace cls_head with wider input (768 + n_ct → 768 → 1)
        self.cls_head = nn.Sequential(
            nn.Linear(768 + n_ct, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
        )
        # Zero-init new weight columns so warm-starting from base checkpoint is safe:
        # the new dimensions produce zero contribution at epoch 0 ≡ baseline.
        with torch.no_grad():
            self.cls_head[0].weight[:, 768:] = 0.0

    def load_base_state_dict(self, state_dict: dict, strict: bool = False):
        """
        Load weights from a base robust_goat checkpoint (which has cls_head[0] 768×768).
        The new n_ct columns in cls_head[0].weight default to zero.
        """
        # Filter out cls_head[0].weight and bias if shapes mismatch
        own_sd = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in own_sd and own_sd[k].shape != v.shape:
                print(f"  [CT] Skipping {k}: checkpoint {v.shape} vs model {own_sd[k].shape}")
                continue
            filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=strict)
        if missing:
            print(f"  [CT] Missing keys (will use random init): {missing[:5]}")
        return missing, unexpected

    def _extract_wavlm_hidden(self, x: torch.Tensor):
        """
        Run the frozen WavLM with output_hidden_states=True.
        x: (B, L) raw audio waveform
        Returns: (hidden_L0, frames_L9, frames_L12, phoneme_feat)
          hidden_L0:    (B, T', 768) — pre-transformer features (for encoder input)
          frames_L9:    (B, T', 768) — layer 9 output
          frames_L12:   (B, T', 768) — layer 12 output (= phoneme_feat)
          phoneme_feat: (B, T', 768) — alias for frames_L12
        """
        # The full WavLM model returns hidden_states tuple with 13 elements:
        #   index 0 = L0 (feature_projection output, before encoder)
        #   index k = output of transformer layer k (k=1..12)
        wavlm_out = self.transformer_in_phoneme_model(
            input_values=x, output_hidden_states=True
        )
        hs = wavlm_out.hidden_states  # tuple of 13 tensors
        return hs[0], hs[9], hs[12], wavlm_out.last_hidden_state

    def encoder_and_GAT_ct(
        self, hidden_states, num_frames, phoneme_ids,
        ct_features=None, profiler=None, use_encoder=True, ground_truth_labels=None
    ):
        """
        encoder_and_GAT with optional C/T feature injection at cls_head.
        ct_features: (B, n_ct) float32 tensor, or None (falls back to base behaviour)
        """
        import pytorch_lightning as pl
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        # Apply trainable encoder
        with profiler.profile("generate encoder features"):
            if use_encoder:
                hidden_states = self.encoder(hidden_states)[0]
            encoder_feat = hidden_states

        num_frames = num_frames.to(hidden_states.device)

        # Adaptive phoneme pooling
        from phoneme_GAT.modules import reduce_feat, generate_edges_by_combine_and_split
        with profiler.profile("reduce hidden states"):
            reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids = reduce_feat(
                hidden_states, num_frames, phoneme_ids
            )

        if self.use_GAT:
            with profiler.profile("generate edges"):
                with torch.no_grad():
                    reduced_num_frames = reduced_num_frames.to(hidden_states.device)
                    edge_index = generate_edges_by_combine_and_split(
                        reduced_num_frames, reduced_phoneme_ids, N=self.n_edges
                    ).to(reduced_hidden_states.device)

            with profiler.profile("generate GAT logits"):
                reduced_hidden_states, edge_index = self.GAT((reduced_hidden_states, edge_index))

            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)
            padded_batch  = torch.nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
            output, _     = self.rnn(padded_batch)
            hidden_states = [output[i, : reduced_num_frames[i], :] for i in range(len(reduced_num_frames))]
            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])
        else:
            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)
            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])

        # Normalize (stored for return)
        hidden_normed = self.norm_feat(hidden_states)

        # Inject C/T at cls_head
        if ct_features is not None and self.n_ct > 0:
            cls_input = torch.cat([hidden_normed, ct_features], dim=1)  # (B, 768+n_ct)
        else:
            cls_input = hidden_normed

        logit = self.cls_head(cls_input)

        return (
            hidden_normed,
            reduced_hidden_states,
            reduced_phoneme_ids,
            reduced_num_frames,
            encoder_feat,
            logit.squeeze(-1),
        )

    def __call__(self, x, num_frames, profiler=None, use_aug=True,
                 ground_truth_labels=None, stage="train"):
        """
        Full forward pass. Identical to Phoneme_GAT.__call__ except:
          - WavLM called with output_hidden_states=True (once, no extra cost)
          - C and T computed per sample from L12/L9 hidden states
          - ct_features concatenated to pooled rep before cls_head
        """
        import pytorch_lightning as pl
        from phoneme_GAT.modules import _mask_hidden_states
        from phoneme_GAT.utils.augmentation import aug_hidden_states as func_aug_hidden_states

        x = self.check_input(x)
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        # ── Frozen extraction with L9/L12 hidden states ──────────────────────
        with profiler.profile("generate phoneme features"):
            with torch.no_grad():
                hidden_L0, frames_L9, frames_L12, phoneme_feat = self._extract_wavlm_hidden(x)
                phoneme_logits = self.phoneme_model.model.model.lm_head(phoneme_feat)
                phoneme_ids    = torch.argmax(phoneme_logits, dim=-1)

        # ── Compute C and T per sample ────────────────────────────────────────
        with torch.no_grad():
            C = _rog_batch(frames_L12)           # (B,) — lower = harder
            T = _vel_entropy_batch(frames_L9)    # (B,) — higher = harder
            ct_raw = torch.stack([C, T], dim=1)  # (B, 2)
            # Normalize with running stats to keep values O(1)
            ct_features = ct_raw  # model learns the scale; no external normalization

        if self.n_ct == 1:
            ct_features = C.unsqueeze(1)
        elif self.n_ct == 2:
            ct_features = torch.stack([C, T], dim=1)
        elif self.n_ct == 3:
            # P3: also include −T_window11 (medium-range velocity, sign-reversed)
            T_w11 = _vel_entropy_window11_batch(frames_L9)
            ct_features = torch.stack([C, T, -T_w11], dim=1)

        # ── SpecAugment masking ───────────────────────────────────────────────
        masked_hidden_states = _mask_hidden_states(hidden_L0, self.transformer_in_phoneme_model)
        org_hidden_states = masked_hidden_states

        # ── Main forward pass ─────────────────────────────────────────────────
        with profiler.profile("generate normal logit"):
            (
                hidden_states,
                reduced_hidden_states,
                reduced_phoneme_ids,
                reduced_num_frames,
                encoder_feat,
                logit,
            ) = self.encoder_and_GAT_ct(
                masked_hidden_states, num_frames, phoneme_ids,
                ct_features=ct_features,
                profiler=profiler,
                ground_truth_labels=ground_truth_labels,
            )

        # ── RPSA augmentation (training only) ────────────────────────────────
        aug_labels, aug_logit, aug_frame_logit, phoneme_cls_logit, phoneme_cls_label = \
            None, None, None, None, None
        if stage == "train" and use_aug:
            with profiler.profile("generate augmenation features"):
                aug_feat, aug_labels, aug_num_frames, aug_phoneme_ids = func_aug_hidden_states(
                    org_hidden_states, num_frames, phoneme_ids, N=5
                )

            if aug_feat.shape[1] <= 200:
                with profiler.profile("generate augmenation logit"):
                    # Augmented pass reuses same ct_features (same utterances)
                    (
                        aug_hidden_states,
                        aug_reduced_hidden_states,
                        aug_reduced_phoneme_ids,
                        aug_reduced_num_frames,
                        aug_encoder_feat,
                        aug_logit,
                    ) = self.encoder_and_GAT_ct(
                        aug_feat, aug_num_frames, aug_phoneme_ids,
                        ct_features=ct_features,
                        use_encoder=False,
                    )

                    aug_frame_logit = self.aug_cls_head(aug_reduced_hidden_states).squeeze(-1)
                    phoneme_cls_logit = self.phoneme_cls_head(aug_reduced_hidden_states)
                    phoneme_cls_label = torch.concat(
                        [aug_reduced_phoneme_ids[i, :_len]
                         for i, _len in enumerate(aug_reduced_num_frames)]
                    )

        return {
            "logit":              logit,
            "hidden_states":      hidden_states,
            "phoneme_feat":       phoneme_feat,
            "encoder_feat":       encoder_feat,
            "phoneme_cls_logit":  phoneme_cls_logit,
            "phoneme_cls_label":  phoneme_cls_label,
            "aug_logit":          aug_logit,
            "aug_frame_logit":    aug_frame_logit,
            "aug_labels":         aug_labels,
            "C":                  C.detach(),
            "T":                  T.detach(),
        }


def _vel_entropy_window11_batch(frames_batch: torch.Tensor) -> torch.Tensor:
    """
    Window-11 velocity entropy: entropy of ||frames[t+11] - frames[t]|| distribution.
    P3 extension. frames_batch: (B, T, D). Returns (B,).
    """
    B, T, D = frames_batch.shape
    stride = 11
    if T <= stride:
        return torch.zeros(B, device=frames_batch.device)
    vels = (frames_batch[:, stride:, :] - frames_batch[:, :-stride, :]).norm(dim=-1)
    entropies = torch.zeros(B, device=frames_batch.device, dtype=torch.float32)
    for i in range(B):
        v = vels[i]
        n = max(5, min(20, len(v) // 4))
        v_min, v_max = v.min(), v.max()
        if (v_max - v_min).item() < 1e-8:
            continue
        hist = torch.histc(v, bins=n, min=v_min.item(), max=(v_max + 1e-6).item())
        h = hist.float() + 1e-8
        h = h / h.sum()
        entropies[i] = -(h * torch.log(h)).sum()
    return entropies


# ─── Lightning wrapper ────────────────────────────────────────────────────────

class Phoneme_GAT_CT_lit(Phoneme_GAT_lit):
    """
    Lightning wrapper for Phoneme_GAT_CT. Inherits all training logic from
    Phoneme_GAT_lit; only the underlying model is replaced.
    """

    def __init__(self, cfg=None, args=None, n_ct: int = 2, **kwargs):
        super().__init__(cfg=cfg, args=args, **kwargs)
        # Replace the model with the CT variant
        self.model = Phoneme_GAT_CT(
            backbone=cfg.PhonemeGAT.backbone,
            use_raw=cfg.PhonemeGAT.use_raw,
            use_GAT=cfg.PhonemeGAT.use_GAT,
            n_edges=cfg.PhonemeGAT.n_edges,
            n_ct=n_ct,
        )

    @classmethod
    def load_from_base_checkpoint(cls, base_ckpt_path: str, cfg, n_ct: int = 2):
        """
        Load weights from a base robust_goat checkpoint (768-dim cls_head),
        skipping the mismatched cls_head[0] weight/bias (new dims zero-inited).
        """
        lit = cls(cfg=cfg, n_ct=n_ct)

        # Load base checkpoint state dict
        ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        # Lightning checkpoints store state dict under 'state_dict' key
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        # Strip Lightning module prefix if present
        cleaned = {}
        for k, v in sd.items():
            # Keys like "model.cls_head.0.weight" → "cls_head.0.weight"
            new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned[new_k] = v

        lit.model.load_base_state_dict(cleaned, strict=False)
        print(f"Loaded base weights from {base_ckpt_path} (cls_head expanded to {768+n_ct}→768)")
        return lit
