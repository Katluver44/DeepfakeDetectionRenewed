# %load_ext autoreload
# %autoreload 2

# +
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

# +
import sys

sys.path.append("/home/ay/Coding2/0-Deepfake/2-Audio/experiments")
# -

from phoneme_model import load_phoneme_model

try:
    from gat import GAT
    from utils.augmentation import aug_hidden_states as func_aug_hidden_states
except ImportError:
    from .gat import GAT
    from .utils.augmentation import aug_hidden_states as func_aug_hidden_states


# +
def segment_means(tensor, segment_sizes):
    # print(tensor.shape, segment_sizes, segment_sizes.sum())

    assert tensor.size(0) == segment_sizes.sum(), "Sum of segment sizes must equal the tensor's first dimension size."

    # Create an indices tensor that maps each row in the tensor to its corresponding segment
    indices = torch.repeat_interleave(torch.arange(len(segment_sizes), device=tensor.device), segment_sizes)

    # Create a tensor to hold the sum of each segment
    segment_sums = torch.zeros(len(segment_sizes), tensor.size(1), device=tensor.device)

    # Scatter and sum the inputs into the segment_sums tensor
    segment_sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, tensor.size(1)), tensor)

    # Calculate the mean of each segment
    segment_means = segment_sums / segment_sizes.unsqueeze(1)

    return segment_means


def reduce_feat(hidden_states, num_frames, phoneme_ids):
    reduced_hidden_states = []
    reduced_num_frames = []
    reduced_phoneme_ids = []

    phoneme_counts = []

    for i in range(len(num_frames)):
        _phoneme_ids = phoneme_ids[i, : num_frames[i]]
        # _h = hidden_states[i, : num_frames[i]]
        unique_ids, _phoneme_counts = _phoneme_ids.unique_consecutive(return_counts=True)
        phoneme_counts += _phoneme_counts.tolist()

        reduced_num_frames.append(len(unique_ids))
        reduced_phoneme_ids.append(unique_ids)

    reduced_num_frames = torch.tensor(reduced_num_frames)
    reduced_phoneme_ids = torch.nn.utils.rnn.pad_sequence(reduced_phoneme_ids, batch_first=True)
    h = torch.concat([hidden_states[i, :_len, :] for i, _len in enumerate(num_frames)], dim=0)
    reduced_hidden_states = segment_means(h, torch.tensor(phoneme_counts, device=hidden_states.device))

    return reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids


# +
def get_adj_edges(L: int):
    adj_edges = torch.stack([torch.arange(L - 1), torch.arange(1, L)])
    return adj_edges


def generate_multple_sequences(ns, as_, bs):
    """
    quickly generate n1 sequences that ranging from a1 to b1,
            generate n2 sequences that ranging from a2 to b2,
            generate n3 sequences that ranging from a3 to b3,
            ....
    and finally combine these sequences


    ```python
    ns = torch.tensor([5, 3, 4])
    as_ = torch.tensor([1, 11, 16])
    bs = torch.tensor([10, 15, 20])
    ```
    > tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9,
               1,  2,  3,  4,  5,  6,  7,  8,  9,
               1,  2,  3,  4,  5,  6,  7,  8,  9,
               1,  2,  3,  4,  5,  6,  7,  8,  9,
               1,  2,  3,  4,  5,  6,  7,  8,  9,
               11, 12, 13, 14,
               11, 12, 13, 14,
               11, 12, 13, 14,
               16, 17, 18, 19,
               16, 17, 18, 19,
               16, 17, 18, 19,
               16, 17, 18, 19])
    Args:
        ns: the repeat number of sequences
        as_: the start number for each seq
        bs: the end number for each seq

    Returns:
        tensor: a 1D tensor for the combined seq.
    """
    # print(ns, as_, bs)
    device = ns.device
    # The maximum value in bs determines the tensor width for uniformity
    max_length = torch.max(bs - as_ + 1)
    # Generate a tensor where each row is a sequence from 0 to max_length
    seq_tensor = torch.arange(max_length).unsqueeze(0).repeat(ns.sum(), 1).to(device)
    seq_tensor = torch.repeat_interleave(as_, ns)[:, None] + seq_tensor
    nums = torch.repeat_interleave(bs - as_, ns)
    mask = torch.arange(seq_tensor.size(1)).expand_as(seq_tensor).to(device) < nums.unsqueeze(1)

    return seq_tensor[mask]


def get_phoneme_edges2(predict_ids: torch.Tensor, N=1):
    """
    Args:
        predict_ids: a tensor with shape of (L,) that represents the phoneme id for each audio frame.
        N: the number of looking forward phonemes
    Returns:
        torch.Tensor: the edges with shape of (2, n_edges)
    """

    device = predict_ids.device

    output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
    cumsum_counts = torch.cumsum(counts, 0).to(device)
    # print(output, inverse, counts, cumsum_counts)
    if len(output) == 1:
        return torch.zeros((2, 0))
    # both start and end are length L
    start_indices = torch.cat([torch.tensor([0], device=device), cumsum_counts[:-1]])
    end_indices = cumsum_counts
    # print("start", start_indices, "end", end_indices)

    edge_start_indices = start_indices[1:]
    edge_end_indices = end_indices[torch.clamp(torch.arange(len(output) - 1) + N, max=len(end_indices) - 1)]
    # print(edge_start_indices, edge_end_indices)

    # print(edge_end_indices.device)
    x = torch.repeat_interleave(
        torch.arange(cumsum_counts[-2]).to(device),
        (edge_end_indices - edge_start_indices)[inverse[: cumsum_counts[-2]]],
        dim=0,
    ).to(device)
    y = generate_multple_sequences(ns=counts[:-1], as_=edge_start_indices, bs=edge_end_indices).to(device)
    edges = torch.stack([x, y])
    # print(x.shape, y.shape, edges.shape)
    return edges


def generate_edges(input_num_frames: torch.Tensor, input_phoneme_ids: torch.Tensor, N=2):
    start_id = 0
    edge_index = []

    print(input_num_frames.shape, input_phoneme_ids.shape, N)

    # num_frames = input_num_frames.cpu()
    num_frames = input_num_frames
    # phoneme_ids = input_phoneme_ids.cpu()
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

    # print(num_frames, input_phoneme_ids.shape, total_edges.shape, e-s, input_phoneme_ids.device, input_num_frames.device)

    # print(total_edges.shape, input_num_frames, input_phoneme_ids.shape)

    if total_edges.numel() == 0:  # （2， 0）
        total_edges = torch.tensor([[0], [0]])

    return total_edges.type(torch.int64)


# +
def calculate_sequence_weights(predict_ids):
    """
    Calculate the normalized weights for each position in a sequence based on
    the sequence lengths of consecutive identical elements.
    """

    output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
    sequence_lengths = counts[inverse]
    normalized_weights = sequence_lengths.float() / sequence_lengths.sum().float()
    return normalized_weights.squeeze()


def get_weighted_hidden_state(hidden_states, phoneme_logits):
    """
    Computes weighted hidden states efficiently when predict_ids is 1D.
    """
    B, T, C = hidden_states.size()
    weighted_hidden_states = torch.zeros(B, C, dtype=hidden_states.dtype, device=hidden_states.device)
    for i, (_h, _l) in enumerate(zip(hidden_states, phoneme_logits)):
        predict_ids = torch.argmax(_l, dim=1)  # Get most likely phoneme ID sequence from logits
        weights = calculate_sequence_weights(predict_ids).unsqueeze(1)
        weighted_hidden_states[i, :] = torch.sum(_h * weights, dim=0)

    return weighted_hidden_states


# -

# # model

# ##  Phoneme Attention


class Phoneme_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (T, C)

        Returns:
            a tensor: (C)
        """
        attn_weight = self.attn(x)  # (T, 1)
        attn_weight = attn_weight.softmax(1)  # (T, 1)
        x = x * attn_weight  # (T, C)
        x = x.sum(0)  # (C)
        return x


# ## random noise


class RandomNoise(torch.nn.Module):
    def __init__(self, noise_level=10, max_dims=(64, 250, 768)):
        """
        Initializes the RandomNoise module.

        Args:
            noise_level (int): The maximum noise level (0 to 10) as a percentage.
        """
        super(RandomNoise, self).__init__()
        self.noise_level = noise_level
        self.para = torch.nn.Parameter(torch.zeros(max_dims), requires_grad=False)

    def forward(self, x):
        """
        Applies random noise to the input tensor x.

        Args:
            x (torch.Tensor): The input tensor to which noise will be added.

        Returns:
            torch.Tensor: The input tensor with added random noise.
        """
        add_noise_level = np.random.randint(0, self.noise_level) / 100
        mult_noise_level = np.random.randint(0, self.noise_level) / 100
        return self._apply_noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)

    def _apply_noise(self, x, add_noise_level=0.0, mult_noise_level=0.0):
        """
        Applies additive and multiplicative noise to the input tensor x.

        Args:
            x (torch.Tensor): The input tensor to which noise will be added.
            add_noise_level (float): The level of additive noise to apply.
            mult_noise_level (float): The level of multiplicative noise to apply.

        Returns:
            torch.Tensor: The input tensor with noise applied.
        """
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


# ## Mask

# +
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
    Masks extracted features along time axis and/or along feature axis according to
    [SpecAugment](https://arxiv.org/abs/1904.08779).
    """

    # generate indices & apply SpecAugment along time axis
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
        # print(mask_time_indices.nonzero())

    return masked_hidden_states


# -

# ## Model

gat_config = {
    # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
    "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
    "num_heads_per_layer": [6, 6, 6],  # other values may give even better results from the reported ones
    # "num_features_per_layer": [768, 128, 128, 768],  # the first number is actually input dim
    "num_features_per_layer": [768, 128, 128, 128],  # the first number is actually input dim
    "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
    "bias": True,  # bias doesn't matter that much
    "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
}


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
            network_name='wav2vec'
            # pretrained_path=None
            # pretrained_path = "/home/ay/data/DATA/1-model_save/01-phoneme/phoneme_recongition/version_4/checkpoints/best-epoch=57-val-per=0.268208.ckpt"
            # pretrained_path = "/home/ay/data/DATA/1-model_save/01-phoneme/phoneme_recongition/version_0/checkpoints/best-epoch=28-val-per=0.278449.ckpt"
            # pretrained_path = "/home/ay/data/DATA/1-model_save/01-phoneme/phoneme_recongition/version_5/checkpoints/best-epoch=49-val-per=0.273750.ckpt"
            # pretrained_path = "/home/ay/data/best-epoch=49-val-per=0.440167.ckpt"
            # pretrained_path = "/home/ay/data/best-epoch=32-val-per=0.398863.ckpt"
            pretrained_path = "/home/ay/data/best-epoch=49-val-per=0.362394.ckpt"
        ## wavlm
        elif backbone.lower() == 'wavlm':
            network_name = "wavlm"
            # pretrained_path = "/home/ay/data/phonemes/wavlm/best-epoch=13-val-per=0.575074.ckpt"
            # pretrained_path = "/home/ay/data/phonemes/wavlm/best-epoch=14-val-per=0.547916.ckpt"
            # pretrained_path = "/home/ay/data/phonemes/wavlm/best-epoch=19-val-per=0.489741.ckpt"
            # pretrained_path = "/home/ay/data/phonemes/wavlm/best-epoch=30-val-per=0.436879.ckpt"
            pretrained_path = "/home/ay/data/phonemes/wavlm/best-epoch=42-val-per=0.407000.ckpt"
            # pretrained_path = "/home/ay/data/phonemes/wavlm/freeze_feature_extractor/best-epoch=44-val-per=0.406913.ckpt"
        total_num_phonemes = 687  ## 198, or 687

        self.phoneme_model = load_phoneme_model(
            network_name=network_name,
            pretrained_path=pretrained_path if not use_raw else None,
            total_num_phonemes=total_num_phonemes,
        )
        self.transformer_in_phoneme_model = self.phoneme_model.model.model.wavlm if backbone.lower() == 'wavlm' else self.phoneme_model.model.model.wav2vec2
        self.phoneme_model.requires_grad_(False)
        self.phoneme_model.eval()

        self.encoder = deepcopy(self.transformer_in_phoneme_model.encoder)
        # self.encoder = load_phoneme_model(
        #     network_name=network_name,
        #     pretrained_path=None,
        #     total_num_phonemes=total_num_phonemes,
        # ).model.model.wavlm.encoder
        self.encoder.requires_grad_(True)
        self.encoder.train()

        self.use_GAT = use_GAT
        self.n_edges = n_edges
        if self.use_GAT:
            self.GAT = GAT(**gat_config)
        self.attn = Phoneme_Attention()
        self.rnn = nn.Sequential(
            # nn.Linear(768, 768),
            # nn.BatchNorm1d(768),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.1),
            nn.LSTM(768, 768 // 2, num_layers=2, bidirectional=True, batch_first=True)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
            # nn.utils.parametrizations.weight_norm(nn.Linear(768, 1))
        )
        # self.cls_head = nn.utils.parametrizations.weight_norm(nn.Linear(768, 1))
        # self.cls_head = nn.Linear(768, 1)

        # self.random_noise = RandomNoise(noise_level=10)

        self.aug_cls_head = nn.Linear(768, 1)
        self.phoneme_cls_head = nn.Linear(768, total_num_phonemes)

    def norm_feat(self, feat):
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def encoder_and_GAT(
        self, hidden_states, num_frames, phoneme_ids, profiler=None, use_encoder=True, ground_truth_labels=None
    ):
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        with profiler.profile("generate encoder features"):
            if use_encoder:
                hidden_states = self.encoder(hidden_states)[0]
            encoder_feat = hidden_states

        num_frames = num_frames.to(hidden_states.device)
        with profiler.profile("reduce hidden states"):
            reduced_hidden_states, reduced_num_frames, reduced_phoneme_ids = reduce_feat(
                hidden_states, num_frames, phoneme_ids
            )
            # print(reduced_hidden_states.shape, reduced_num_frames)

        if self.use_GAT:
            with profiler.profile("generate edges"):
                with torch.no_grad():
                    reduced_num_frames = reduced_num_frames.to(hidden_states.device)
                    edge_index = generate_edges_by_combine_and_split(
                        reduced_num_frames, reduced_phoneme_ids, N=self.n_edges
                    ).to(reduced_hidden_states.device)

            ## logit is with shape of (sum(T), 2), we will convert it to (B, 2)
            with profiler.profile("generate GAT logits"):
                reduced_hidden_states, edge_index = self.GAT((reduced_hidden_states, edge_index))

            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)  # B tensors with (T_i, C)

            padded_batch = torch.nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
            output, _ = self.rnn(padded_batch)
            hidden_states = [output[i, : reduced_num_frames[i], :] for i in range(len(reduced_num_frames))]

            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])  # -> (B, C)
            # hidden_states = torch.stack([self.attn(seg) for seg in hidden_states])  # -> (B, C)

        else:
            hidden_states = torch.split(reduced_hidden_states, list(reduced_num_frames), 0)  # B tensors with (T_i, C)
            hidden_states = torch.stack([seg.mean(0) for seg in hidden_states])  # -> (B, C)

        # logit = hidden_states
        hidden_states = self.norm_feat(hidden_states)
        logit = self.cls_head(hidden_states)

        # pred = (torch.sigmoid(logit) + 0.5).int()
        # for i, n in enumerate(reduced_num_frames):
        #     with open("text.txt", "a") as f:
        #         if pred[i] !=  ground_truth_labels[i]:
        #             print(
        #                 i,
        #                 "n_phonemes",
        #                 n.item(),
        #                 "phoneme_id",
        #                 phoneme_ids[i].unique_consecutive(),
        #                 "label",
        #                 ground_truth_labels[i].item(),
        #                 "logit",
        #                 logit[i].item(),
        #                 file=f,
        #             )

        return (
            hidden_states,
            reduced_hidden_states,
            reduced_phoneme_ids,
            reduced_num_frames,
            encoder_feat,
            logit.squeeze(-1),
        )

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
            hidden_states, _ = self.transformer_in_phoneme_model.feature_projection(
                feat1
            )  # (B, T, C), extract_features is the layer norm of feat1
            phoneme_feat = self.transformer_in_phoneme_model.encoder(hidden_states)[0]
        masked_hidden_states = _mask_hidden_states(hidden_states, self.transformer_in_phoneme_model)  ##(B, T, C)
        encoder_feat = self.encoder(masked_hidden_states)[0]  # (B, T, C)
        hidden_states = self.norm_feat(encoder_feat.mean(dim=1))  # (B, C)
        logit = self.cls_head(hidden_states).squeeze(-1)  # (B,)
        return {
            "logit": logit,
            "phoneme_feat": phoneme_feat,
            "encoder_feat": encoder_feat,
        }

    def __call__(self, x, num_frames, profiler=None, use_aug=True, ground_truth_labels=None, stage="train"):
        x = self.check_input(x)
        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        with profiler.profile("generate phoneme features"):
            with torch.no_grad():
                # output = self.phoneme_model.model.model(x, output_hidden_states=True)
                feat1 = self.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
                hidden_states, _ = self.transformer_in_phoneme_model.feature_projection(
                    feat1
                )  # (B, T, C), extract_features is the layer norm of feat1
                phoneme_feat = self.transformer_in_phoneme_model.encoder(hidden_states)[0]
                phoneme_logits = self.phoneme_model.model.model.lm_head(phoneme_feat)  # (B, T, P)
                phoneme_ids = torch.argmax(phoneme_logits, dim=-1)  # shape changes from (B, T, P) -> (B,T)

        # in org config of wav2vec2, mask_time_prob=0.05, mask_feature_prob=0
        masked_hidden_states = _mask_hidden_states(hidden_states, self.transformer_in_phoneme_model)  # (B, T, C)

        org_hidden_states = masked_hidden_states
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

        aug_labels, aug_logit, aug_frame_logit, phoneme_cls_logit, phoneme_cls_label = None, None, None, None, None
        if stage == "train" and use_aug:
            with profiler.profile("generate augmenation features"):
                aug_feat, aug_labels, aug_num_frames, aug_phoneme_ids = func_aug_hidden_states(
                    org_hidden_states, num_frames, phoneme_ids, N=5
                )

            if aug_feat.shape[1] > 200:
                pass
            else:
                # aug_feat = self.random_noise(aug_feat)
                with profiler.profile("generate augmenation logit"):
                    # print("phoneme lengths", num_frames, aug_num_frames)
                    (
                        aug_hidden_states,
                        aug_reduced_hidden_states,
                        aug_reduced_phoneme_ids,
                        aug_reduced_num_frames,
                        aug_encoder_feat,
                        aug_logit,
                    ) = self.encoder_and_GAT(aug_feat, aug_num_frames, aug_phoneme_ids, use_encoder=False)
                    aug_frame_logit = self.aug_cls_head(aug_reduced_hidden_states).squeeze(-1)
                    phoneme_cls_logit = self.phoneme_cls_head(aug_reduced_hidden_states)  # (sum_Len, n_phonemes)
                    phoneme_cls_label = torch.concat(
                        [aug_reduced_phoneme_ids[i, :_len] for i, _len in enumerate(aug_reduced_num_frames)]
                    )  # (B, max_len) -> (sum_Len)

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


# +
# model = Phoneme_GAT()
# x = torch.randn(3, 48000)
# T = 150
# B = 3
# num_frames = torch.randint(3, T, (B,))
# logit = model(x, num_frames)
# print(logit.shape)
# -

# # lit model


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

        # self.lr = 0.00016
        # self.lr = 1.12e-6
        self.lr = 1e-4  ## loss coe 0.5
        # self.lr = 2.5e-4
        # self.lr = 1.5e-6

        self.use_aug = cfg.PhonemeGAT.use_aug
        self.use_pool = cfg.PhonemeGAT.use_pool
        self.use_clip = cfg.PhonemeGAT.use_clip

        self.save_hyperparameters()

    def configure_loss_fn(self):
        from ay2.torch.losses import LabelSmoothingBCE

        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = LabelSmoothingBCE(label_smoothing=0.05)
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

        # phoneme_cls_loss = (
        #     self.ce_loss(batch_res["phoneme_cls_logit"], batch_res["phoneme_cls_label"].long())
        #     if batch_res["phoneme_cls_logit"] is not None
        #     else 0
        # )
        # contrast_loss = self.contrast_loss(batch_res["hidden_states"], label)
        clip_loss = (
            self.clip_loss(
                batch_res["phoneme_feat"].mean(dim=-1), self.clip_head(batch_res["encoder_feat"]).mean(dim=-1)
            )
            if self.use_clip
            else 0.0
        )

        aug_loss = 0
        if self.use_aug and "aug_logit" in batch_res.keys() and batch_res["aug_logit"] is not None:
            aug_loss = self.bce_loss(batch_res["aug_logit"], label.type(torch.float32) * 0)
        # aug_frame_loss = 0
        # if batch_res["aug_labels"] is not None and batch_res["aug_frame_logit"] is not None and self.use_aug:
        #     aug_frame_loss = self.bce_loss(
        #         batch_res["aug_frame_logit"], batch_res["aug_labels"].type(torch.float32) * 0
        #     )

        loss = (
            cls_loss
            + 0.5 * clip_loss
            + 0.5 * aug_loss
            # + 0.5 * aug_frame_loss
        )

        return {
            "loss": loss,
            # "phoneme_cls_loss": phoneme_cls_loss,
            "cls_loss": cls_loss,
            "clip_loss": clip_loss,
            # "contrast_loss": contrast_loss,
            "aug_loss": aug_loss,
            # "aug_frame_loss": aug_frame_loss,
        }

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.named_parameters() if "model.encoder" in n], "lr": 5e-5},
                {"params": [p for n, p in self.named_parameters() if not "model.encoder" in n], "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    # def configure_optimizers(self):
    #     from ay2.torch.optim.selective_weight_decay import (
    #         Optimizers_with_selective_weight_decay,
    #         Optimizers_with_selective_weight_decay_for_modulelist,
    #     )
    #     optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
    #         [self],
    #         optimizer="Adam",
    #         lr=self.lr,
    #         weight_decay=0.00001,
    #     )

    #     return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # print(batch['language'])

        B = len(audio)

        num_frames = torch.full((B,), 48000 // 320 - 1)

        if self.use_pool == 0:
            # use frame-level feature for classification
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
