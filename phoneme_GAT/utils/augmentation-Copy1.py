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

import torch
import numpy as np
import time
from ay2.tools import TimerContextManager
import random


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# B = 64
# T = 149
#
# hidden_states= torch.randn(B, T, 768).cuda()
# audio_lengths = torch.randint(T-10, T, (B,)).cuda()
# phoneme_ids = torch.randint(0, 10, (B, T)).type(torch.int64).cuda()
# languages = torch.randint(0, 3, (B,)).cuda()

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# ## add -1 in each audio, 避免unique_consecutive把不同的audio的phoneme给合并了　
# phoneme_ids = torch.concat([ phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device)*-1], dim=-1)
# hidden_states = torch.concat([ hidden_states, torch.ones(B, 1, 768, device=hidden_states.device)], dim=1)
#
# ### 尽管phoneme_ids是2D，但是unique_consecutive会把它拉伸为1D
# ### note!!!, inverse is 2D
# reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
# reduced_phoneme_ids = reduced_phoneme_ids.type(torch.int64)
# cumsum_counts = torch.cumsum(counts, 0)
#
# ### id_pairs[i, j] denotes whether the phoneme i, j are same
# id_pairs = (reduced_phoneme_ids[:, None] == reduced_phoneme_ids[None, :])
#
# #### languages_pairs[i, j] denotes whether the samples i, j are of sample language
# languages_pairs = ( languages[:, None] == languages[None, ...]) 
# -

# # Step 1

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
    id_to_index_range[:, 0] = (cumsum_counts-1) // (T+1)
    id_to_index_range[1:, 1] = cumsum_counts[:-1] % (T+1)
    id_to_index_range[:, 2] = (cumsum_counts-1) % (T+1) + 1

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


# + [markdown] tags=["style-solution"]
# 测试

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# B = 64
# T = 149
# phoneme_ids = torch.randint(0, 2, (B, T)).type(torch.int64).cuda()
# phoneme_ids = torch.concat([ phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device)*-1], dim=-1)
# reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
# # print(phoneme_ids)
# %time phoneme_id_to_index_range = get_phonme_id_mapping(cumsum_counts=torch.cumsum(counts, 0), T=T)
# -

# ## Step 2

# 对于一个batch，计算每个音频样本$i$，在reduced的phoneme ids中，phoneme id的起始和结束的index范围是多少：

def get_sample_index_range_in_phonemes(samples):
    change_indices = torch.nonzero(samples[1:] != samples[:-1], as_tuple=False)[:, 0] + 1
    # Start indices for each unique value
    start_indices = torch.cat((torch.tensor([0], device=change_indices.device), change_indices))
    # End indices for each unique value
    ## 这里减1是为了忽略那个padding的frame　
    end_indices = torch.cat((change_indices - 1, torch.tensor([len(samples) - 1], device=change_indices.device)))
    # Create a dictionary to store index ranges
    index_ranges = {value.item(): (start.item(), end.item()) for value, start, end in zip(samples[start_indices], start_indices, end_indices)}
    return index_ranges


# + [markdown] tags=["style-solution"]
# 测试

# + tags=["style-solution"]
B, T = 64, 149
# B, T = 3, 10

hidden_states= torch.randn(B, T, 768).cuda()
audio_lengths = torch.randint(T-10, T, (B,)).cuda()
phoneme_ids = torch.randint(0, 10, (B, T)).type(torch.int64).cuda()
languages = torch.randint(0, 3, (B,)).cuda()
hidden_states = torch.concat([ hidden_states, torch.ones(B, 1, 768, device=hidden_states.device)], dim=1)
phoneme_ids = torch.concat([ phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device)*-1], dim=-1)
reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
# print(phoneme_ids)
phoneme_id_to_index_range = get_phonme_id_mapping(cumsum_counts=torch.cumsum(counts, 0), T=T)

# %time index_ranges = get_sample_index_range_in_phonemes(phoneme_id_to_index_range[:, 0])
print(reduced_phoneme_ids, index_ranges)
# -

hidden_states.shape, torch.sum(counts)

reduced_phoneme_ids


# ## 查找每个phoneme同类的index　

def find_same_phoneme_ids(reduced_phoneme_ids):
    same_phoneme_id_index = {}
    for i, x in enumerate(reduced_phoneme_ids.cpu().numpy()):
        same_phoneme_id_index.setdefault(x, []).append(i)
    return same_phoneme_id_index


# + tags=["style-solution"]
# %time same_phoneme_id_index = find_same_phoneme_ids(reduced_phoneme_ids)
print(same_phoneme_id_index)
# -

np.nonzero(reduced_phoneme_ids.cpu().numpy()<=4)[0]



p = 5
with TimerContextManager():
    _phoneme_ids = reduced_phoneme_ids.cpu().numpy()
    L = len(_phoneme_ids)
    replaced_index = np.arange(L)
    probs = np.random.rand(L)
    filtered_index = np.nonzero(_phoneme_ids<4)[0]

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
            print(i, x, same_phoneme_id_index[x], replaced_index[i])

replaced_index

[[]]*10

split_hidden_states = list(torch.split(hidden_states.view(-1, 768), tuple(counts.cpu().numpy())))
splits = [[]] * B
for i in range(B):
    s, e = index_ranges[i]
    print(s, e, replaced_index[s:e])
    splits[i] = torch.concat([split_hidden_states[j] for j in replaced_index[s:e]])
    # print(splits[i].shape)
aug_hidden_states = torch.nn.utils.rnn.pad_sequence(splits,batch_first=True)

aug_hidden_states.shape

sum(tuple(counts.cpu().numpy()))

np.random.rand(10)

x = [1, 2, 6]
x.remove(1)
x

# +
# def find_first_larger_element(X, y):
#     """
#     Finds the first element in sorted tensor X that is larger than y.
    
#     Args:
#     - X (torch.Tensor): Sorted 1D tensor.
#     - y (float or int): Value to compare against.
    
#     Returns:
#     - int: Index of the first element in X that is larger than y.
#       Returns -1 if no such element exists.
#     """
#     greater_mask = X > y
#     indices = torch.nonzero(greater_mask).squeeze()
    
#     if indices.numel() > 0:
#         return indices[0].item()
#     else:
#         return len(X)
        
# def find_phoneme_ids_index(sample_id, cumsum_counts, T):

#     # each sample has T+1 unreduced phoneme ids
#     s = sample_id * (T + 1)
#     e = s + (T + 1)
#     s = find_first_larger_element(cumsum_counts, s)
#     e = find_first_larger_element(cumsum_counts, e) - 1
#     return s, e


# def get_sample_phoneme_ids_indices(cumsum_counts, B, T):
#     sample_phoneme_ids_index = {}
#     for i in range(B):
#         s, e = find_phoneme_ids_index(i, cumsum_counts, T)
#         sample_phoneme_ids_index[i] = (s, e)
#         # print(i, s, e, reduced_phoneme_ids[s:e])
#     return sample_phoneme_ids_index

# # %time sample_phoneme_ids_index = get_sample_phoneme_ids_indices(cumsum_counts, B, T)
# -

# # Step 3　



# + editable=true slideshow={"slide_type": ""}
def aug_hidden_states(hidden_states,audio_lengths ,phoneme_ids, languages=None, N=5):


    with TimerContextManager('init'):
        B,T = hidden_states.shape[0], hidden_states.shape[1]
        ## add -1 in each audio, 避免unique_consecutive把不同的audio的phoneme给合并了　
        phoneme_ids = torch.concat([ phoneme_ids, torch.ones(B, 1, device=phoneme_ids.device)*-1], dim=-1)
        hidden_states = torch.concat([ hidden_states, torch.ones(B, 1, 768, device=hidden_states.device)], dim=1)
        
        ### 尽管phoneme_ids是2D，但是unique_consecutive会把它拉伸为1D
        ### note!!!, inverse is 2D
        reduced_phoneme_ids, inverse, counts = phoneme_ids.unique_consecutive(return_inverse=True, return_counts=True)
        reduced_phoneme_ids = reduced_phoneme_ids.type(torch.int64)
        cumsum_counts = torch.cumsum(counts, 0)
        
        ### id_pairs[i, j] denotes whether the phoneme i, j are same
        id_pairs = (reduced_phoneme_ids[:, None] == reduced_phoneme_ids[None, :])
        
        #### languages_pairs[i, j] denotes whether the samples i, j are of sample language
        # languages_pairs = ( languages[:, None] == languages[None, ...]) 
    
    
        labels = torch.zeros_like(reduced_phoneme_ids)
    
    with TimerContextManager('compute mapping'):
        id_to_index_range = get_phonme_id_mapping(cumsum_counts.cpu(), T)
        # sample_phoneme_ids_index = get_sample_phoneme_ids_indices(cumsum_counts, B,T)
        sample_phoneme_ids_index = get_sample_index_range_in_phonemes(id_to_index_range[:, 0])
        
        
    split_hidden_states = list(torch.split(hidden_states.view(-1, 768), tuple(counts.cpu().numpy())))
    split_hidden_states_org = list(torch.split(hidden_states.view(-1, 768), tuple(counts.cpu().numpy())))
    
    i = 1
    splits = []
    for i in range(B):
        start_index, end_index = sample_phoneme_ids_index[i]
        _split = split_hidden_states[start_index:end_index]
        x_phoneme_ids = reduced_phoneme_ids[start_index:end_index]
    
        # print('sample', i, start_index, end_index)
        
        find_substitution = 0
        with TimerContextManager(f'sample {i}'):
            for j in (np.random.permutation(end_index - start_index) + start_index): # 遍历phoneme　
                if reduced_phoneme_ids[j] >= 0 and reduced_phoneme_ids[j] < 5:
                    continue
                _id = reduced_phoneme_ids[j]
                same_ids = id_pairs[j].nonzero().squeeze()
                # print('phonme_id', _id)
                if same_ids.ndim == 0:
                    continue
                success = 0
                for k in torch.randperm(len(same_ids)):
                    current_phoneme_id = same_ids[k]
                    current_sample, current_sample_s, current_sample_e, current_sample_phoneme  = id_to_index_range[current_phoneme_id]
                    # print(current_sample)
                    if current_sample == i:
                        continue
                    ### replace
                    start_index2, end_index2 = sample_phoneme_ids_index[current_sample.item()]
                    # print(i, start_index, end_index, 'source', j-start_index, _id, 'target', current_sample, start_index2, end_index2,current_sample_phoneme, reduced_phoneme_ids[current_phoneme_id])
                    _split[j-start_index] = split_hidden_states_org[start_index2:end_index2][current_sample_phoneme]
                    success = 1
    
                    labels[j] = 1
                    
                    break
                
                find_substitution += success
                if find_substitution >= N:
                    break
            splits.append(_split)

    with TimerContextManager(f'final process'):
        for i in range(B):
            splits[i] = torch.concat(splits[i])
            # print(splits[i].shape)
        aug_hidden_states = torch.nn.utils.rnn.pad_sequence(splits,batch_first=True)
        labels = labels[reduced_phoneme_ids!=-1]
    return aug_hidden_states, labels

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# B = 64
# T = 149
#
# hidden_states= torch.randn(B, T, 768).cuda()
# audio_lengths = torch.randint(T-10, T, (B,)).cuda()
# phoneme_ids = torch.randint(0, 10, (B, T)).type(torch.int64).cuda()
# languages = torch.randint(0, 3, (B,)).cuda()

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# aug_hidden_states(hidden_states, audio_lengths, phoneme_ids, languages)
