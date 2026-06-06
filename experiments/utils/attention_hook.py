"""
attention_hook.py
=================
Reusable utilities for extracting GAT layer-0 attention weights and
phoneme-node metadata during a forward pass.

Classes
-------
AttentionHook
    Context manager / callable that registers a forward hook on
    gat_net[layer_idx] and captures attention tensors.

PhonemeCapture
    Monkey-patches encoder_and_GAT to capture per-node phoneme IDs
    and per-node sample indices.
"""
from __future__ import annotations

import types
from typing import Optional

import torch


class AttentionHook:
    """
    Registers a forward hook on GAT layer ``layer_idx`` and captures
    the post-softmax attention weights and edge index after each call.

    Usage::

        with AttentionHook(gat_model, layer_idx=0) as hook:
            with torch.no_grad():
                gat_model.encoder_and_GAT(hidden_states, num_f, phoneme_ids)
            attn      = hook.attn        # (E, NH)
            edge_idx  = hook.edge_index  # (2, E)
            hook.clear()

    Or imperatively::

        hook = AttentionHook(gat_model).install()
        ...
        hook.remove()

    Attributes
    ----------
    attn : Tensor | None
        Shape ``(E, NH)`` — post-softmax attention weights, squeezed from
        ``attention_weights`` which has shape ``(E, NH, 1)`` inside GATLayer.
    edge_index : Tensor | None
        Shape ``(2, E)`` — global node indices for source and target.
    """

    def __init__(self, gat_model, layer_idx: int = 0) -> None:
        self._layer = gat_model.GAT.gat_net[layer_idx]
        self._handle: Optional[object] = None
        self.attn: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None

    def install(self) -> "AttentionHook":
        self._layer.log_attention_weights = True
        cap = self

        def _hook(module, inp, out):
            cap.edge_index = out[1].detach().cpu()
            if module.attention_weights is not None:
                cap.attn = module.attention_weights.squeeze(-1).detach().cpu()

        self._handle = self._layer.register_forward_hook(_hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        self.attn = None
        self.edge_index = None

    def __enter__(self) -> "AttentionHook":
        return self.install()

    def __exit__(self, *args) -> None:
        self.remove()


class PhonemeCapture:
    """
    Monkey-patches ``gat_model.encoder_and_GAT`` to capture per-node
    phoneme IDs and batch-local sample indices after each call.

    The patch is installed in ``__init__`` and is permanent (not scoped).
    Re-create the object on a fresh model if you need a clean state.

    Attributes updated after each ``encoder_and_GAT`` call
    -------------------------------------------------------
    node_phoneme_ids : Tensor, shape ``(total_nodes,)``
    node_sample_idx  : Tensor, shape ``(total_nodes,)``   (0-indexed, batch-local)
    reduced_num_frames : Tensor, shape ``(B,)``
    """

    def __init__(self, gat_model) -> None:
        self.node_phoneme_ids: Optional[torch.Tensor] = None
        self.node_sample_idx: Optional[torch.Tensor] = None
        self.reduced_num_frames: Optional[torch.Tensor] = None

        cap = self
        orig = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_inner, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            rids = result[2].detach().cpu()   # (B, Lmax) padded phoneme IDs
            rnf  = result[3].detach().cpu()   # (B,) phoneme counts per sample
            flat_ids, flat_samp = [], []
            for i in range(len(rnf)):
                n = int(rnf[i].item())
                flat_ids.append(rids[i, :n])
                flat_samp.append(torch.full((n,), i, dtype=torch.long))
            cap.node_phoneme_ids   = torch.cat(flat_ids)
            cap.node_sample_idx    = torch.cat(flat_samp)
            cap.reduced_num_frames = rnf
            return result

        gat_model.encoder_and_GAT = types.MethodType(_patched, gat_model)
