# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import time

import numpy as np
import torch
from torch import Tensor


# -

# :::{note}
# 在音素识别时，一个音素可能会跨多个时间帧，因此需要合并连续且相同的音素，构造为一个音素，从而预测整句话的音素，并与ground truth进行比较。
# :::
#
# 那使用音素特征识别时，那么也许合并连续且相同的音素特征帧。设音素特征为`(B, T, 768)`，那么合并后的特征为`(B, T', 768)`。

def segment_means(x: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
    """
    Args:
      tensor: torch.Tensor: a 2D tensor with shape of `(L, C)`
      segment_sizes: torch.Tensor: a 1D tensor that its sum is equal to the length `L` of tensor

    Returns:
        torch.Tenosr: the tensor with reduce length `(L', C)`, where $L'=len(segment_sizes)$
    """
    assert x.size(0) == segment_sizes.sum(), "Sum of segment sizes must equal the tensor's first dimension size."

    # Create an indices tensor that maps each row in the tensor to its corresponding segment
    indices = torch.repeat_interleave(torch.arange(len(segment_sizes), device=x.device), segment_sizes)

    # Create a tensor to hold the sum of each segment
    segment_sums = torch.zeros(len(segment_sizes), x.size(1), device=x.device)

    # Scatter and sum the inputs into the segment_sums tensor
    segment_sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, x.size(1)), x)

    # Calculate the mean of each segment
    segment_means = segment_sums / segment_sizes.unsqueeze(1)

    return segment_means


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# ## 验证
# x = torch.randn(5, 1)
# segments = torch.tensor([1, 2, 2])
# res = segment_means(x, segments)
# print(x, "\n", res)
# -

def reduce_feat_by_phonemes(hidden_states: Tensor, audio_lengths: Tensor, phoneme_ids: Tensor, debug: bool = False) -> Tensor:
    """

    For each audio, combine continuous phonemes to reduce the temporal dimension of audio features.
    For example, the audio with 10 frames, which phoneme ids will change from
    ```python
    [0, 0, 0, 1, 1, 1, 2, 2, 0, 0] -> [0, 1, 2, 0]
    ```
    and the hidden_states will also changed by this way.


    Args:
      hidden_states:Tensor: a 3D tensor with shape of (B, T, C), where T is audio frames
      audio_lengths:Tensor: a 1D tensor with shape of (B,) that represent the legnth of each audio
      phoneme_ids:Tensor: a 2D tensor with shape of (B, T) that represents the phoneme ids in each audio frame
      debug:bool: determine whether to debug tensor info.

    Returns:
        the reduced hidden_states with shape (B*L', C), reduced audio lengths, reduced phoneme ids.
    """

    reduced_hidden_states = []
    reduced_audio_lengths = []
    reduced_phoneme_ids = []
    phoneme_counts = []

    
    if debug:
        s = time.time()
        print('reduce feat input:', hidden_states.shape, audio_lengths.shape, phoneme_ids.shape)

    for i in range(len(audio_lengths)):
        _phoneme_ids = phoneme_ids[i, : audio_lengths[i]]
        unique_ids, _phoneme_counts = _phoneme_ids.unique_consecutive(return_counts=True)
        phoneme_counts += _phoneme_counts.tolist()

        reduced_audio_lengths.append(len(unique_ids))
        reduced_phoneme_ids.append(unique_ids)

    reduced_audio_lengths = torch.tensor(reduced_audio_lengths)
    reduced_phoneme_ids = torch.nn.utils.rnn.pad_sequence(reduced_phoneme_ids, batch_first=True)
    h = torch.concat([hidden_states[i, :_len, :] for i, _len in enumerate(audio_lengths)], dim=0)
    reduced_hidden_states = segment_means(h, torch.tensor(phoneme_counts, device=hidden_states.device))

    if debug:
        e = time.time()
        print('reduce feat output:',  reduced_hidden_states.shape,reduced_audio_lengths.shape, reduced_phoneme_ids.shape)
        print('reduce feat time:', e - s)
    return reduced_hidden_states, reduced_audio_lengths, reduced_phoneme_ids

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# B = 64
# T = 149
# hidden_states = torch.randn(B, T, 768).cuda().requires_grad_(True)
# phoneme_ids = torch.randint(0, 5, (B, T)).cuda()
# audio_lengths = torch.randint(140, 149, (B,)).cuda()
#
# reduced_hidden_states, reduced_audio_lengths, reduced_phoneme_ids = reduce_feat_by_phonemes(
#     hidden_states, audio_lengths, phoneme_ids, debug=1
# )
# print(reduced_audio_lengths.shape, reduced_phoneme_ids.shape, reduced_hidden_states.shape)
