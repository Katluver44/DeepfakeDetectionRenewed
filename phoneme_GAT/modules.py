# modules.py
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.losses import BinaryTokenContrastLoss, CLIPLoss1D
from .phoneme_model import load_phoneme_model
from .gat import GAT
from .utils.augmentation import aug_hidden_states as func_aug_hidden_states
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


# ----------------------
# Helper functions
# ----------------------

def segment_means(tensor, segment_sizes):
    assert tensor.size(0) == segment_sizes.sum()
    indices = torch.repeat_interleave(torch.arange(len(segment_sizes), device=tensor.device), segment_sizes)
    segment_sums = torch.zeros(len(segment_sizes), tensor.size(1), device=tensor.device)
    segment_sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, tensor.size(1)), tensor)
    return segment_sums / segment_sizes.unsqueeze(1)


def reduce_feat(hidden_states, num_frames, phoneme_ids):
    reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids = [], [], []
    phoneme_counts = []
    for i in range(len(num_frames)):
        _phoneme_ids = phoneme_ids[i, : num_frames[i]]
        unique_ids, _counts = _phoneme_ids.unique_consecutive(return_counts=True)
        phoneme_counts += _counts.tolist()
        reduced_num_frames.append(len(unique_ids))
        reduced_phoneme_ids.append(unique_ids)

    reduced_num_frames = torch.tensor(reduced_num_frames)
    reduced_phoneme_ids = torch.nn.utils.rnn.pad_sequence(reduced_phoneme_ids, batch_first=True)
    h = torch.concat([hidden_states[i, :_len, :] for i, _len in enumerate(num_frames)], dim=0)
    reduced_hidden_states = segment_means(h, torch.tensor(phoneme_counts, device=hidden_states.device))
    return reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids


def get_adj_edges(L: int):
    return torch.stack([torch.arange(L - 1), torch.arange(1, L)])


def _mask_hidden_states(hidden_states, wav2vec_model, mask_time_prob=0.05, mask_time_length=10, mask_time_min_masks=2, attention_mask=None):
    """SpecAugment along time axis."""
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


# ----------------------
# Attention & Noise
# ----------------------

class Phoneme_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weight = self.attn(x).softmax(1)
        return (x * attn_weight).sum(0)


class RandomNoise(nn.Module):
    def __init__(self, noise_level=10, max_dims=(64, 250, 768)):
        super().__init__()
        self.noise_level = noise_level
        self.para = nn.Parameter(torch.zeros(max_dims), requires_grad=False)

    def forward(self, x):
        add_noise_level = np.random.randint(0, self.noise_level) / 100
        mult_noise_level = np.random.randint(0, self.noise_level) / 100
        return self._apply_noise(x, add_noise_level, mult_noise_level)

    def _apply_noise(self, x, add_noise_level=0.0, mult_noise_level=0.0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * np.random.beta(2, 5) * self.para.normal_()[: x.shape[0], : x.shape[1], : x.shape[2]].to(x.device)
        else:
            add_noise = 0.0
        if mult_noise_level > 0.0:
            mult_noise = (
                mult_noise_level * np.random.beta(2, 5) *
                (2 * self.para.uniform_()[: x.shape[0], : x.shape[1], : x.shape[2]] - 1).to(x.device) + 1
            )
            x = x * mult_noise
        return x + add_noise


# ----------------------
# Phoneme GAT Model
# ----------------------

gat_config = {
    "num_of_layers": 3,
    "num_heads_per_layer": [6, 6, 6],
    "num_features_per_layer": [768, 128, 128, 128],
    "add_skip_connection": True,
    "bias": True,
    "dropout": 0.0,
}


class Phoneme_GAT(nn.Module):
    def __init__(self, backbone="wavlm", use_raw=0, use_GAT=1, n_edges=10):
        super().__init__()

        if backbone.lower() == "wav2vec":
            network_name = "wav2vec"
            pretrained_path = "/workspace/PLFD-ADD/pretrained/best-epoch=49-val-per=0.362394.ckpt"
        elif backbone.lower() == "wavlm":
            network_name = "wavlm"
            pretrained_path = "/workspace/PLFD-ADD/pretrained/best-epoch=42-val-per=0.407000.ckpt"
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        total_num_phonemes = 687

        self.phoneme_model = load_phoneme_model(
            network_name=network_name,
            pretrained_path=None if use_raw else pretrained_path,
            total_num_phonemes=total_num_phonemes,
        )
        if backbone.lower() == "wavlm":
            self.transformer_in_phoneme_model = self.phoneme_model.model.model.wavlm
        else:
            self.transformer_in_phoneme_model = self.phoneme_model.model.model.wav2vec2

        self.phoneme_model.requires_grad_(False).eval()

        self.encoder = deepcopy(self.transformer_in_phoneme_model.encoder)
        self.encoder.requires_grad_(True).train()

        self.use_GAT = use_GAT
        self.n_edges = n_edges
        if self.use_GAT:
            self.GAT = GAT(**gat_config)
        self.attn = Phoneme_Attention()
        self.rnn = nn.LSTM(768, 384, num_layers=2, bidirectional=True, batch_first=True)

        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
        )

        self.aug_cls_head = nn.Linear(768, 1)
        self.phoneme_cls_head = nn.Linear(768, total_num_phonemes)

    def check_input(self, x):
        if x.ndim == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        elif x.ndim >= 3:
            raise ValueError(f"The input audio should be (B, L) or (B, 1, L), but is {x.shape}")
        return x

    def run_without_pool_and_GAT(self, x):
        x = self.check_input(x)
        with torch.no_grad():
            feat1 = self.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
            hidden_states, _ = self.transformer_in_phoneme_model.feature_projection(feat1)
            phoneme_feat = self.transformer_in_phoneme_model.encoder(hidden_states)[0]
        masked_hidden_states = _mask_hidden_states(hidden_states, self.transformer_in_phoneme_model)
        encoder_feat = self.encoder(masked_hidden_states)[0]
        hidden_states = encoder_feat.mean(dim=1)
        logit = self.cls_head(hidden_states).squeeze(-1)
        return {"logit": logit, "phoneme_feat": phoneme_feat, "encoder_feat": encoder_feat}

    def forward(self, x, num_frames, profiler=None, use_aug=True, ground_truth_labels=None, stage="train"):
        return self.run_without_pool_and_GAT(x)


# ----------------------
# Lightning Wrapper
# ----------------------

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
        self.profiler = getattr(args, "profiler", None)
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
        self.clip_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(0.1), nn.Linear(768, 768))
        self.clip_loss = CLIPLoss1D()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))
        clip_loss = (
            self.clip_loss(
                batch_res["phoneme_feat"].mean(dim=-1), self.clip_head(batch_res["encoder_feat"]).mean(dim=-1)
            )
            if self.use_clip
            else 0.0
        )
        aug_loss = 0.0
        if self.use_aug and "aug_logit" in batch_res and batch_res["aug_logit"] is not None:
            aug_loss = self.bce_loss(batch_res["aug_logit"], label.type(torch.float32) * 0)
        loss = cls_loss + 0.5 * clip_loss + 0.5 * aug_loss
        return {"loss": loss, "cls_loss": cls_loss, "clip_loss": clip_loss, "aug_loss": aug_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.named_parameters() if "model.encoder" in n], "lr": 5e-5},
                {"params": [p for n, p in self.named_parameters() if not "model.encoder" in n], "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio = batch["audio"]
        num_frames = torch.full((len(audio),), 48000 // 320 - 1)
        if self.use_pool == 0:
            batch_res = self.model.run_without_pool_and_GAT(audio)
        else:
            batch_res = self.model(audio, num_frames, profiler=self.profiler, use_aug=self.use_aug, stage=stage, ground_truth_labels=batch["label"])
        return batch_res
