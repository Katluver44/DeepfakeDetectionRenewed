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


# 设，一个audio的所有帧（T）对应的phoneme ids是，并给出了audio长度为10：
# ```python　
# predict_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
# audio_lengths = [10]
# ```
#
# 那么，建立单项边时，有两个操作：
# 1. 添加所有邻接边：`0->1, 1->2, 2->3, ...`
# 2. 对于一个phoneme，对接下来的N个phoneme都建立边：
#     - 以第1个node为例（index从0开始），设N=1，那么会新加边：`1->2, 1->3, 1->4`
#     - 以第3个node为例（index从0开始），设N=1，那么会新加边：`3->5,6,7,8,9`

# # Step 1 

# 使用arange可以很快地生成所有邻接边。

def get_adj_edges(L: int, use_np=False):
    if use_np:
        adj_edges = np.stack([np.arange(L - 1), np.arange(1, L)])
    else:
        adj_edges = torch.stack([torch.arange(L - 1), torch.arange(1, L)])
    return adj_edges


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# L = 149
# adj_edges = get_adj_edges(L, use_np=0)
# adj_edges_np = get_adj_edges(L, use_np=1)
# print(adj_edges, adj_edges.numpy()-adj_edges_np)
# -

# ## Step 2

# 使用 `unique_consecutive`可以快速地查找所有不同的phoneme id，并定位其index范围。

# ### 遍历　

# 下面这种遍历长度L的方法，耗时太长了

# #### torch

# + editable=true slideshow={"slide_type": ""} tags=["style-student", "active-ipynb"]
# # predict_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
# predict_ids = torch.randint(0, 3, (149,))
#
# N = 10
#
# output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
# cumsum_counts = torch.cumsum(counts, 0)
# # output, inverse, conuts, cumsum_counts
#
# edges = []
# for i in range(L):
#     unique_id = inverse[i]  # 0, 1, 2, 3,
#     unique_id_end_index = cumsum_counts[unique_id]
#     if unique_id == len(output) - 1:
#         break
#     next_id = min(len(output) - 1, unique_id + N)
#     next_end_index = cumsum_counts[next_id]
#     _edges = torch.stack(
#         [torch.full((next_end_index - unique_id_end_index,), i), torch.arange(unique_id_end_index, next_end_index)]
#     )
#     edges.append(_edges)
# edges = torch.concat(edges, 1)
# -

edges


# #### numpy 

# 首先，由于numpy中没有unique_consecutive这个函数，因此需要先实现他。

# + editable=true slideshow={"slide_type": ""}
def unique_consecutive(x: np.ndarray):

    output = [x[0]]
    inverse = np.zeros_like(x)
    counts = [1]

    for i in range(1, len(x)):
        if x[i] == output[-1]:
            counts[-1] += 1
        else:
            output.append(x[i])
            counts.append(1)
        inverse[i] = len(output) -1
    return np.array(output), inverse, np.array(counts)

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# predict_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
# output, inverse, counts = predict_ids.unique_consecutive(return_inverse=True, return_counts=True)
# output, inverse, counts

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# ## 可以看到，和tensor版本的输出结果是一样的
# unique_consecutive(predict_ids.numpy())

# + editable=true slideshow={"slide_type": ""} tags=["style-student", "active-ipynb"]
# L = 149 * 64
# predict_ids = torch.randint(0, 3, (L,)).numpy()
# N = 10
#
# output, inverse, counts = unique_consecutive(predict_ids)
# cumsum_counts = np.cumsum(counts, 0)
# # output, inverse, conuts, cumsum_counts
#
# edges = []
# for i in range(L):
#     unique_id = inverse[i]  # 0, 1, 2, 3,
#     unique_id_end_index = cumsum_counts[unique_id]
#     if unique_id == len(output) - 1:
#         break
#     next_id = min(len(output) - 1, unique_id + N)
#     next_end_index = cumsum_counts[next_id]
#     _edges = np.stack(
#         [np.full((next_end_index - unique_id_end_index,), i), np.arange(unique_id_end_index, next_end_index)]
#     )
#     edges.append(_edges)
# edges = np.concatenate(edges, 1)
