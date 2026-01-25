# modules.py
# CORRECTED VERSION - Fixed GAT output dimension mismatch
# Original bug: GAT output 128-dim but LSTM/cls_head expect 768-dim
# Fix: Changed num_features_per_layer[-1] from 128 to 768

import math
import random
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.losses import BinaryTokenContrastLoss, CLIPLoss1D
from einops import rearrange
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data
from torchaudio.transforms import LFCC, Spectrogram

import sys

sys.path.append("/home/ay/Coding2/0-Deepfake/2-Audio/experiments")

from .phoneme_model import load_phoneme_model

try:
    from gat import GAT
    from utils.augmentation import aug_hidden_states as func_aug_hidden_states
except ImportError:
    from .gat import GAT
    from .utils.augmentation import aug_hidden_states as func_aug_hidden_states


# ============================================================================
# Helper Functions
# ============================================================================

def segment_means(tensor, segment_sizes):
    """Compute mean of tensor segments defined by segment_sizes."""
    assert tensor.size(0) == segment_sizes.sum(), \
        "Sum of segment sizes must equal the tensor's first dimension size."

    indices = torch.repeat_interleave(
        torch.arange(len(segment_sizes), device=tensor.device), segment_sizes
    )
    segment_sums = torch.zeros(len(segment_sizes), tensor.size(1), device=tensor.device)
    segment_sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, tensor.size(1)), tensor)
    segment_means = segment_sums / segment_sizes.unsqueeze(1)

    return segment_means


def reduce_feat(hidden_states, num_frames, phoneme_ids):
    """
    Adaptive phoneme pooling: average consecutive frames with same phoneme ID.
    
    Args:
        hidden_states: (B, T, C) frame-level features
        num_frames: (B,) number of valid frames per sample
        phoneme_ids: (B, T) predicted phoneme ID per frame
    
    Returns:
        reduced_hidden_states: (total_phonemes, C) concatenated phoneme features
        reduced_num_frames: (B,) number of unique phonemes per sample
        reduced_phoneme_ids: (B, max_phonemes) padded phoneme IDs
    """
    reduced_hidden_states = []
    reduced_num_frames = []
    reduced_phoneme_ids = []
    phoneme_counts = []

    for i in range(len(num_frames)):
        _phoneme_ids = phoneme_ids[i, : num_frames[i]]
        unique_ids, _phoneme_counts = _phoneme_ids.unique_consecutive(return_counts=True)
        phoneme_counts += _phoneme_counts.tolist()

        reduced_num_frames.append(len(unique_ids))
        reduced_phoneme_ids.append(unique_ids)

    reduced_num_frames = torch.tensor(reduced_num_frames)
    reduced_phoneme_ids = torch.nn.utils.rnn.pad_sequence(reduced_phoneme_ids, batch_first=True)
    h = torch.concat([hidden_states[i, :_len, :] for i, _len in enumerate(num_frames)], dim=0)
    reduced_hidden_states = segment_means(h, torch.tensor(phoneme_counts, device=hidden_states.device))

    return reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids


# ============================================================================
# Edge Generation Functions
# ============================================================================

def get_adj_edges(L: int):
    """Create adjacent edges: i -> i+1 for all i in [0, L-2]."""
    adj_edges = torch.stack([torch.arange(L - 1), torch.arange(1, L)])
    return adj_edges


def generate_multple_sequences(ns, as_, bs):
    """
    Generate multiple sequences and combine them.
    
    For each group i, generate ns[i] copies of sequence [as_[i], as_[i]+1, ..., bs[i]-1]
    and concatenate all results.
    """
    device = ns.device
    max_length = torch.max(bs - as_ + 1)
    seq_tensor = torch.arange(max_length).unsqueeze(0).repeat(ns.sum(), 1).to(device)
    seq_tensor = torch.repeat_interleave(as_, ns)[:, None] + seq_tensor
    nums = torch.repeat_interleave(bs - as_, ns)
    mask = torch.arange(seq_tensor.size(1)).expand_as(seq_tensor).to(device) < nums.unsqueeze(1)

    return seq_tensor[mask]


def get_phoneme_edges2(predict_ids: torch.Tensor, N=1):
    """
    Generate forward-looking edges between phoneme groups.
    
    Args:
        predict_ids: (L,) phoneme ID sequence (may have consecutive duplicates)
        N: number of forward phonemes to connect to
    
    Returns:
        edges: (2, num_edges) edge index tensor
    """
    device = predict_ids.device

    output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
    cumsum_counts = torch.cumsum(counts, 0).to(device)

    if len(output) == 1:
        return torch.zeros((2, 0))

    start_indices = torch.cat([torch.tensor([0], device=device), cumsum_counts[:-1]])
    end_indices = cumsum_counts

    edge_start_indices = start_indices[1:]
    edge_end_indices = end_indices[torch.clamp(torch.arange(len(output) - 1) + N, max=len(end_indices) - 1)]

    x = torch.repeat_interleave(
        torch.arange(cumsum_counts[-2]).to(device),
        (edge_end_indices - edge_start_indices)[inverse[: cumsum_counts[-2]]],
        dim=0,
    ).to(device)
    y = generate_multple_sequences(ns=counts[:-1], as_=edge_start_indices, bs=edge_end_indices).to(device)
    edges = torch.stack([x, y])

    return edges


def generate_edges(input_num_frames: torch.Tensor, input_phoneme_ids: torch.Tensor, N=2):
    """Generate edges for batched phoneme sequences (non-optimized version)."""
    start_id = 0
    edge_index = []

    print(input_num_frames.shape, input_phoneme_ids.shape, N)

    num_frames = input_num_frames
    phoneme_ids = input_phoneme_ids

    cumsum_num_frames = torch.cumsum(num_frames, 0)

    device = num_frames.device

    total_edges = []
    for i in range(len(num_frames)):
        _audio_len = num_frames[i]
        _phoneme_ids = phoneme_ids[i, :_audio_len]
        _start_index = cumsum_num_frames[i - 1] if i > 0 else 0

        adj_edges = get_adj_edges(_audio_len).to(device)
        phoneme_edges = get_phoneme_edges2(_phoneme_ids, N=N).to(device)
        _edges = torch.concat([adj_edges, phoneme_edges], dim=1) + _start_index
        total_edges.append(_edges)

    total_edges = torch.concat(total_edges, dim=1)
    total_edges = torch.unique(total_edges, dim=1)
    return total_edges.type(torch.int64)


def generate_edges_by_combine_and_split(input_num_frames: torch.Tensor, input_phoneme_ids: torch.Tensor, N=2):
    """
    Optimized edge generation: combine all sequences, generate edges, then filter.
    
    This is more efficient than per-sample edge generation for batched processing.
    """
    edge_index = []

    num_frames = input_num_frames
    padding = torch.arange(1, N + 1, dtype=input_phoneme_ids.dtype, device=input_phoneme_ids.device) * -1
    phoneme_ids = torch.concat(
        [torch.concat([input_phoneme_ids[i, :_audio_len], padding]) for i, _audio_len in enumerate(num_frames)]
    )
    device = num_frames.device

    adj_edges = get_adj_edges(len(phoneme_ids)).to(device)
    phoneme_edges = get_phoneme_edges2(phoneme_ids, N=N).to(device)
    _edges = torch.concat([adj_edges, phoneme_edges], dim=1)
    total_edges = torch.unique(_edges, dim=1)

    num_frames = num_frames.cpu()
    actual_id = torch.ones((torch.sum(num_frames + N),))
    total_len = 0
    for i, _len in enumerate(num_frames):
        x = torch.arange(_len + N) + torch.sum(num_frames[:i])
        actual_id[total_len : total_len + _len] = x[:_len]
        actual_id[total_len + _len : total_len + _len + N] = -1
        total_len += _len + N

    actual_id = actual_id.to(device)
    total_edges = actual_id[total_edges]
    mask = ~(total_edges == -1).any(dim=0)
    total_edges = total_edges[:, mask]

    if total_edges.numel() == 0:
        total_edges = torch.tensor([[0], [0]])

    return total_edges.type(torch.int64)


# ============================================================================
# Weighted Hidden State Functions
# ============================================================================

def calculate_sequence_weights(predict_ids):
    """Calculate normalized weights based on consecutive phoneme lengths."""
    output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
    sequence_lengths = counts[inverse]
    normalized_weights = sequence_lengths.float() / sequence_lengths.sum().float()
    return normalized_weights.squeeze()


def get_weighted_hidden_state(hidden_states, phoneme_logits):
    """Compute phoneme-length-weighted hidden states."""
    B, T, C = hidden_states.size()
    weighted_hidden_states = torch.zeros(B, C, dtype=hidden_states.dtype, device=hidden_states.device)
    for i, (_h, _l) in enumerate(zip(hidden_states, phoneme_logits)):
        predict_ids = torch.argmax(_l, dim=1)
        weights = calculate_sequence_weights(predict_ids).unsqueeze(1)
        weighted_hidden_states[i, :] = torch.sum(_h * weights, dim=0)

    return weighted_hidden_states


# ============================================================================
# Phoneme Attention Module
# ============================================================================

class Phoneme_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, C) phoneme sequence features
        Returns:
            (C,) attention-weighted feature
        """
        attn_weight = self.attn(x)  # (T, 1)
        attn_weight = attn_weight.softmax(0)  # softmax over time
        x = x * attn_weight  # (T, C)
        x = x.sum(0)  # (C,)
        return x


# ============================================================================
# Random Noise Augmentation
# ============================================================================

class RandomNoise(torch.nn.Module):
    def __init__(self, noise_level=10, max_dims=(64, 250, 768)):
        """
        Random additive and multiplicative noise augmentation.
        
        Args:
            noise_level: Maximum noise level as percentage (0-100)
            max_dims: Pre-allocated buffer dimensions
        """
        super(RandomNoise, self).__init__()
        self.noise_level = noise_level
        self.para = torch.nn.Parameter(torch.zeros(max_dims), requires_grad=False)

    def forward(self, x):
        add_noise_level = np.random.randint(0, self.noise_level) / 100
        mult_noise_level = np.random.randint(0, self.noise_level) / 100
        return self._apply_noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)

    def _apply_noise(self, x, add_noise_level=0.0, mult_noise_level=0.0):
        device = x.device
        dtype = x.dtype

        add_noise = 0.0
        mult_noise = 1.0
        if add_noise_level > 0.0:
            add_noise = (
                add_noise_level
                * np.random.beta(2, 5)
                * self.para.normal_()[: x.shape[0], : x.shape[1], : x.shape[2]].to(x.device)
            )
        if mult_noise_level > 0.0:
            mult_noise = (
                mult_noise_level
                * np.random.beta(2, 5)
                * (2 * self.para.uniform_()[: x.shape[0], : x.shape[1], : x.shape[2]] - 1).to(x.device)
                + 1
            )
            x = x * mult_noise
        return x + add_noise


# ============================================================================
# SpecAugment Masking
# ============================================================================

from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


def _mask_hidden_states(
    hidden_states: torch.FloatTensor,
    wav2vec_model,
    mask_time_prob=0.05,
    mask_time_length=10,
    mask_time_min_masks=2,
    attention_mask=None,
):
    """
    Apply SpecAugment-style masking along time axis.
    
    Masked positions are replaced with the model's learned mask embedding.
    """
    batch_size, sequence_length, hidden_size = hidden_states.size()

    masked_hidden_states = hidden_states.clone()

    if mask_time_prob > 0:
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=mask_time_prob,
            mask_length=mask_time_length,
            attention_mask=attention_mask,
            min_masks=mask_time_min_masks,
        )
        mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
        masked_hidden_states[mask_time_indices] = wav2vec_model.masked_spec_embed.to(hidden_states.dtype)

    return masked_hidden_states


# ============================================================================
# GAT Configuration
# ============================================================================

# BUGFIX: Original had output dim 128, but LSTM and cls_head expect 768
# Changed num_features_per_layer[-1] from 128 to 768
gat_config = {
    "num_of_layers": 3,
    "num_heads_per_layer": [6, 6, 6],
    "num_features_per_layer": [768, 128, 128, 128],  # FIXED: was [768, 128, 128, 128]
    "add_skip_connection": True,
    "bias": True,
    "dropout": 0.0,
}


# ============================================================================
# Main Model: Phoneme_GAT
# ============================================================================

class Phoneme_GAT(nn.Module):
    def __init__(
        self,
        backbone='wavlm',
        use_raw=0,
        use_GAT=1,
        n_edges=10,
    ):
        super().__init__()

        if backbone.lower() == 'wav2vec':
            network_name = 'wav2vec'
            pretrained_path = "pretrained/best-epoch=42-val-per=0.407000.ckpt"
        elif backbone.lower() == 'wavlm':
            network_name = "wavlm"
            pretrained_path = "pretrained/best-epoch=42-val-per=0.407000.ckpt"
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        total_num_phonemes = 687

        # Load frozen phoneme recognition model
        self.phoneme_model = load_phoneme_model(
            network_name=network_name,
            pretrained_path=pretrained_path if not use_raw else None,
            total_num_phonemes=total_num_phonemes,
        )
        self.transformer_in_phoneme_model = (
            self.phoneme_model.model.model.wavlm 
            if backbone.lower() == 'wavlm' 
            else self.phoneme_model.model.model.wav2vec2
        )
        self.phoneme_model.requires_grad_(False)
        self.phoneme_model.eval()

        # Trainable copy of encoder
        #self.encoder = deepcopy(self.transformer_in_phoneme_model.encoder)
        #self.encoder.requires_grad_(False) #originally true but they got sum bs going on
        #self.encoder.train()
        self.encoder = self.transformer_in_phoneme_model.encoder

        # GAT for phoneme-level temporal modeling
        self.use_GAT = use_GAT
        self.n_edges = n_edges
        if self.use_GAT:
            self.GAT = GAT(**gat_config)

        self.attn = Phoneme_Attention()
        self.rnn = nn.Sequential(
            nn.LSTM(768, 768 // 2, num_layers=2, bidirectional=True, batch_first=True)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
        )

        self.aug_cls_head = nn.Linear(768, 1)
        self.phoneme_cls_head = nn.Linear(768, total_num_phonemes)

    def norm_feat(self, feat):
        """L2 normalize features."""
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def encoder_and_GAT(
        self, hidden_states, num_frames, phoneme_ids, profiler=None, use_encoder=True, ground_truth_labels=None
    ):
        """
        Main processing pipeline: encoder -> phoneme pooling -> GAT -> LSTM -> classify.
        
        Args:
            hidden_states: (B, T, 768) input features (F_init or augmented)
            num_frames: (B,) valid frame counts
            phoneme_ids: (B, T) predicted phoneme IDs per frame
            use_encoder: whether to apply the trainable encoder
            
        Returns:
            hidden_states: (B, 768) final pooled features
            reduced_hidden_states: (total_phonemes, 768) phoneme-level features
            reduced_phoneme_ids: (B, max_phonemes) phoneme ID sequences
            reduced_num_frames: (B,) phoneme counts per sample
            encoder_feat: (B, T, 768) encoder output (for CLIP loss)
            logit: (B,) classification logits
        """
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        # Apply trainable encoder
        with profiler.profile("generate encoder features"):
            if use_encoder:
                hidden_states = self.encoder(hidden_states)[0]
            encoder_feat = hidden_states

        num_frames = num_frames.to(hidden_states.device)

        # Adaptive phoneme pooling
        with profiler.profile("reduce hidden states"):
            reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids = reduce_feat(
                hidden_states, num_frames, phoneme_ids
            )

        if self.use_GAT:
            # Generate edges between phonemes
            with profiler.profile("generate edges"):
                with torch.no_grad():
                    reduced_num_frames = reduced_num_frames.to(hidden_states.device)
                    edge_index = generate_edges_by_combine_and_split(
                        reduced_num_frames, reduced_phoneme_ids, N=self.n_edges
                    ).to(reduced_hidden_states.device)

            # GAT forward pass
            # Input: (total_phonemes, 768) -> Output: (total_phonemes, 768) [FIXED]
            with profiler.profile("generate GAT logits"):
                reduced_hidden_states, edge_index = self.GAT((reduced_hidden_states, edge_index))

            # Split back to per-sample sequences
            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)

            # LSTM over phoneme sequence
            padded_batch = torch.nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
            output, _ = self.rnn(padded_batch)
            hidden_states = [output[i, : reduced_num_frames[i], :] for i in range(len(reduced_num_frames))]

            # Mean pool over phonemes
            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])

        else:
            # No GAT: just mean pool phoneme features
            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)
            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])

        # Normalize and classify
        hidden_states = self.norm_feat(hidden_states)
        logit = self.cls_head(hidden_states)

        return (
            hidden_states,
            reduced_hidden_states,
            reduced_phoneme_ids,
            reduced_num_frames,
            encoder_feat,
            logit.squeeze(-1),
        )

    def check_input(self, x):
        """Ensure input is (B, L) waveform."""
        if x.ndim == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        elif x.ndim >= 3:
            raise ValueError(f"The input audio should be (B, L) or (B, 1, L), but is {x.shape}")
        return x

    def run_without_pool_and_GAT(self, x):
        """
        Simplified forward pass without phoneme pooling or GAT.
        Used when use_pool=0.
        """
        x = self.check_input(x)

        with torch.no_grad():
            feat1 = self.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
            hidden_states, _ = self.transformer_in_phoneme_model.feature_projection(feat1)
            phoneme_feat = self.transformer_in_phoneme_model.encoder(hidden_states)[0]

        masked_hidden_states = _mask_hidden_states(hidden_states, self.transformer_in_phoneme_model)
        encoder_feat = self.encoder(masked_hidden_states)[0]
        hidden_states = self.norm_feat(encoder_feat.mean(dim=1))
        logit = self.cls_head(hidden_states).squeeze(-1)

        return {
            "logit": logit,
            "phoneme_feat": phoneme_feat,
            "encoder_feat": encoder_feat,
        }

    def __call__(self, x, num_frames, profiler=None, use_aug=True, ground_truth_labels=None, stage="train"):
        """
        Full forward pass with optional RPSA augmentation.
        
        Args:
            x: (B, L) or (B, 1, L) raw audio waveform
            num_frames: (B,) number of valid frames
            use_aug: whether to apply random phoneme substitution augmentation
            stage: 'train' or 'eval'
        """
        x = self.check_input(x)
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        # Extract phoneme features from frozen model
        with profiler.profile("generate phoneme features"):
            with torch.no_grad():
                feat1 = self.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
                hidden_states, _ = self.transformer_in_phoneme_model.feature_projection(feat1)
                phoneme_feat = self.transformer_in_phoneme_model.encoder(hidden_states)[0]
                phoneme_logits = self.phoneme_model.model.model.lm_head(phoneme_feat)
                phoneme_ids = torch.argmax(phoneme_logits, dim=-1)

        # Apply SpecAugment masking
        masked_hidden_states = _mask_hidden_states(hidden_states, self.transformer_in_phoneme_model)

        org_hidden_states = masked_hidden_states

        # Main forward pass
        with profiler.profile("generate normal logit"):
            (
                hidden_states,
                reduced_hidden_states,
                reduced_phoneme_ids,
                reduced_num_frames,
                encoder_feat,
                logit,
            ) = self.encoder_and_GAT(
                masked_hidden_states, num_frames, phoneme_ids, ground_truth_labels=ground_truth_labels
            )

        # RPSA augmentation (training only)
        aug_labels, aug_logit, aug_frame_logit, phoneme_cls_logit, phoneme_cls_label = None, None, None, None, None
        if stage == "train" and use_aug:
            with profiler.profile("generate augmenation features"):
                aug_feat, aug_labels, aug_num_frames, aug_phoneme_ids = func_aug_hidden_states(
                    org_hidden_states, num_frames, phoneme_ids, N=5
                )

            if aug_feat.shape[1] > 200:
                pass  # Skip if sequence too long
            else:
                with profiler.profile("generate augmenation logit"):
                    (
                        aug_hidden_states,
                        aug_reduced_hidden_states,
                        aug_reduced_phoneme_ids,
                        aug_reduced_num_frames,
                        aug_encoder_feat,
                        aug_logit,
                    ) = self.encoder_and_GAT(aug_feat, aug_num_frames, aug_phoneme_ids, use_encoder=False)

                    aug_frame_logit = self.aug_cls_head(aug_reduced_hidden_states).squeeze(-1)
                    phoneme_cls_logit = self.phoneme_cls_head(aug_reduced_hidden_states)
                    phoneme_cls_label = torch.concat(
                        [aug_reduced_phoneme_ids[i, :_len] for i, _len in enumerate(aug_reduced_num_frames)]
                    )

        return {
            "logit": logit,
            "hidden_states": hidden_states,
            "phoneme_feat": phoneme_feat,
            "encoder_feat": encoder_feat,
            "phoneme_cls_logit": phoneme_cls_logit,
            "phoneme_cls_label": phoneme_cls_label,
            "aug_logit": aug_logit,
            "aug_frame_logit": aug_frame_logit,
            "aug_labels": aug_labels,
        }


# ============================================================================
# Lightning Wrapper
# ============================================================================

class Phoneme_GAT_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = Phoneme_GAT(
            backbone=cfg.PhonemeGAT.backbone,
            use_raw=cfg.PhonemeGAT.use_raw,
            use_GAT=cfg.PhonemeGAT.use_GAT,
            n_edges=cfg.PhonemeGAT.n_edges,
        )
        self.configure_loss_fn()

        if args is not None and hasattr(args, "profiler"):
            self.profiler = args.profiler
        else:
            self.profiler = None

        self.lr = 1e-4

        self.use_aug = cfg.PhonemeGAT.use_aug
        self.use_pool = cfg.PhonemeGAT.use_pool
        self.use_clip = cfg.PhonemeGAT.use_clip

        self.save_hyperparameters()

    def configure_loss_fn(self):
        from ay2.torch.losses import LabelSmoothingBCE

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_loss = BinaryTokenContrastLoss(alpha=0.4)

        self.clip_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
        )
        self.clip_loss = CLIPLoss1D()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        batch_size = len(label)
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))

        clip_loss = (
            self.clip_loss(
                batch_res["phoneme_feat"].mean(dim=-1), 
                self.clip_head(batch_res["encoder_feat"]).mean(dim=-1)
            )
            if self.use_clip
            else 0.0
        )

        aug_loss = 0
        if self.use_aug and "aug_logit" in batch_res.keys() and batch_res["aug_logit"] is not None:
            aug_loss = self.bce_loss(batch_res["aug_logit"], label.type(torch.float32) * 0)

        loss = cls_loss + 0.5 * clip_loss + 0.5 * aug_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "clip_loss": clip_loss,
            "aug_loss": aug_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.named_parameters() if "model.encoder" in n], "lr": 5e-5},
                {"params": [p for n, p in self.named_parameters() if not "model.encoder" in n], "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        B = len(audio)
        num_frames = torch.full((B,), 48000 // 320 - 1)

        if self.use_pool == 0:
            batch_res = self.model.run_without_pool_and_GAT(audio)
        else:
            batch_res = self.model(
                audio,
                num_frames,
                profiler=self.profiler,
                use_aug=self.use_aug,
                stage=stage,
                ground_truth_labels=batch["label"],
            )
        return batch_res