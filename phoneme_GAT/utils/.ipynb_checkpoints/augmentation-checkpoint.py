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
import random
import time

import numpy as np
import torch
from ay2.tools import TimerContextManager


# -

# 每个audio的frames个数都是$T$，为了避免unique_consecutive把不同的audio的phoneme给合并了，给每个audio padding了一个frame，因此每个audio的frames个数变成了$T+1$。
#
# 设对batch的phoneme ids进行unique_consecutive之后，共有N个phonemes，那么`get_phonme_id_mapping`的作用就是计算出，对于reduced的第$i$个phoneme，其对应的：
# 1. 音频样本的id，$0<=id<B$
# 2. 在音频样本中，该phoneme对应哪些音频frames，$s:e$
# 3. 它是音频样本的第几个phoneme

def get_phonme_id_mapping(cumsum_counts, T):
    """
    id_to_index_range is a tensor with (L, 4), where each row (a, b, c, d)
    - a: the audio sample id
    - b, c: the phoneme id range in the audio frames, 1 phoneme may contain multiple frames
    - d: the n-th phoneme in current audio sample
    """
    id_to_index_range = torch.zeros(len(cumsum_counts), 4, dtype=torch.int32, device=cumsum_counts.device)
    id_to_index_range[:, 0] = (cumsum_counts - 1) // (T + 1)
    id_to_index_range[1:, 1] = cumsum_counts[:-1] % (T + 1)
    id_to_index_range[:, 2] = (cumsum_counts - 1) % (T + 1) + 1

    _, nums = id_to_index_range[:, 0].unique_consecutive(return_counts=True)
    id_to_index_range[:, 2] = torch.concat([torch.arange(x) for x in nums])

    # _n = 0
    # for i in range(1, len(cumsum_counts)):
    #     if id_to_index_range[i, 0] != id_to_index_range[i-1, 0]:
    #         _n = 0
    #     id_to_index_range[i, 3] = _n
    #     _n += 1

    # with open('text.txt', 'w') as f:
    # for i in range(len(cumsum_counts)):
    # print(id_to_index_range[i], file=f)
    return id_to_index_range


# 测试

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# # B, T = 64, 149
# B, T = 3, 10
# phoneme_ids = torch.randint(0, 2, (B, T)).type(torch.int64).cuda()
# phoneme_ids = torch.concat([phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device) * -1], dim=-1)
# reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
# # print(phoneme_ids)
# %time phoneme_id_to_index_range = get_phonme_id_mapping(cumsum_counts=torch.cumsum(counts, 0), T=T)
# -

# 对于一个batch，计算每个音频样本$i$，在reduced的phoneme ids中，phoneme id的起始和结束的index范围是多少：

def get_sample_index_range_in_phonemes(samples):
    change_indices = torch.nonzero(samples[1:] != samples[:-1], as_tuple=False)[:, 0] + 1
    # Start indices for each unique value
    start_indices = torch.cat((torch.tensor([0], device=change_indices.device), change_indices))
    # End indices for each unique value
    ## 这里减1是为了忽略那个padding的frame
    end_indices = torch.cat((change_indices - 1, torch.tensor([len(samples) - 1], device=change_indices.device)))
    # Create a dictionary to store index ranges
    index_ranges = {
        value.item(): (start.item(), end.item())
        for value, start, end in zip(samples[start_indices], start_indices, end_indices)
    }
    return index_ranges

# + [markdown] tags=["style-solution"]
# 测试

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# %time index_ranges = get_sample_index_range_in_phonemes(phoneme_id_to_index_range[:, 0])
# print(index_ranges)
# -



# 查找每个phoneme同类的index
def find_same_phoneme_ids(reduced_phoneme_ids):
    same_phoneme_id_index = {}
    for i, x in enumerate(reduced_phoneme_ids.cpu().numpy()):
        same_phoneme_id_index.setdefault(x, []).append(i)
    return same_phoneme_id_index


# 测试

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# %time same_phoneme_id_index = find_same_phoneme_ids(reduced_phoneme_ids)
# print(same_phoneme_id_index)
# -

# # 整合　

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# B, T = 64, 149
# # B, T = 3, 10
#
# hidden_states = torch.randn(B, T, 768).cuda()
# audio_lengths = torch.randint(T - 10, T, (B,)).cuda()
# phoneme_ids = torch.randint(0, 10, (B, T)).type(torch.int64).cuda()
# languages = torch.randint(0, 3, (B,)).cuda()

# + editable=true slideshow={"slide_type": ""}
def aug_hidden_states(hidden_states, audio_lengths, phoneme_ids, languages=None, p=0.2, *args, **kwargs):
    B,T = hidden_states.shape[0], hidden_states.shape[1]
    
    hidden_states = torch.concat([hidden_states, torch.ones(B, 1, 768, device=hidden_states.device)], dim=1)
    phoneme_ids = torch.concat([phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device) * -1], dim=-1)

    reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
    phoneme_id_to_index_range = get_phonme_id_mapping(cumsum_counts=torch.cumsum(counts, 0), T=T)
    sample_index_ranges = get_sample_index_range_in_phonemes(phoneme_id_to_index_range[:, 0])
    same_phoneme_id_index = find_same_phoneme_ids(reduced_phoneme_ids)

    labels = torch.zeros_like(reduced_phoneme_ids)
    
    
    with TimerContextManager(debug=False):
        _phoneme_ids = reduced_phoneme_ids.cpu().numpy()
        L = len(_phoneme_ids)
        replaced_index = np.arange(L)
        probs = np.random.rand(L)
        filtered_index = np.nonzero(_phoneme_ids < 4)[0]

        choices = np.random.randint(100, 1000, L)

        for i in filtered_index:
            if probs[i] <= p:
                # replaced_index[i] = random.choice(filter_list)
                x = _phoneme_ids[i]
                _list = same_phoneme_id_index[x]
                _len = len(_list)
                _replace = choices[i] % _len
                if _list[_replace] == i:
                    replaced_index[i] = _list[(_replace + 1) % _len]
                else:
                    replaced_index[i] = _list[_replace]
                # print(i, x, replaced_index[i])

    labels[replaced_index!=np.arange(L)] = 1
    
    split_hidden_states = list(torch.split(hidden_states.view(-1, 768), tuple(counts.cpu().numpy())))
    splits = [[]] * B
    for i in range(B):
        s, e = index_ranges[i]
        # print(s, e, replaced_index[s:e])
        splits[i] = torch.concat([split_hidden_states[j] for j in replaced_index[s:e]])
    aug_hidden_states = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)
    # print(aug_hidden_states.shape)
    return aug_hidden_states, labels

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# aug_hidden_states(hidden_states, audio_lengths, phoneme_ids, p=0.2)

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# B, T = 64, 149
# # B, T = 3, 10
#
# hidden_states = torch.randn(B, T, 768).cuda()
# audio_lengths = torch.randint(T - 10, T, (B,)).cuda()
# phoneme_ids = torch.randint(0, 10, (B, T)).type(torch.int64).cuda()
# languages = torch.randint(0, 3, (B,)).cuda()
#
#
# hidden_states = torch.concat([hidden_states, torch.ones(B, 1, 768, device=hidden_states.device)], dim=1)
# phoneme_ids = torch.concat([phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device) * -1], dim=-1)
#
#
# reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
# phoneme_id_to_index_range = get_phonme_id_mapping(cumsum_counts=torch.cumsum(counts, 0), T=T)
#
# %time index_ranges = get_sample_index_range_in_phonemes(phoneme_id_to_index_range[:, 0])
# print(reduced_phoneme_ids, index_ranges)
