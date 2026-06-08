"""
modules_ablations.py
====================
Loss-side C/T ablations (P5–P7) from the roadmap. Each is a thin subclass of
Phoneme_GAT_lit that overrides ONLY calcuate_loss — the underlying model
(Phoneme_GAT) is unchanged, so a robust_goat checkpoint warm-starts exactly
(no cls_head shape change, unlike the P2 CT variant in modules_ct.py).

  P5  Phoneme_GAT_Reweight_lit  — per-sample BCE weighted by system hardness
  P6  Phoneme_GAT_Diversity_lit — auxiliary −rog spread loss on bonafide frames
  P7  Phoneme_GAT_CRank_lit     — pairwise margin ranking of spoof logits by
                                  per-sample compactness (−rog@L12)

Auxiliary terms are gated on self.training so validation EER/loss reflect the
plain classifier (the monitored metric stays comparable to baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phoneme_GAT.modules import Phoneme_GAT_lit


def _rog_per_sample(frames: torch.Tensor) -> torch.Tensor:
    """Radius of gyration per sample. frames: (B, T, D) → (B,)."""
    c = frames.mean(dim=1, keepdim=True)
    return ((frames - c) ** 2).sum(dim=-1).mean(dim=-1).clamp_min(1e-12).sqrt()


class _FinetuneOptMixin:
    """Honors self.lr for warm-start fine-tuning (the base configure_optimizers
    hardcodes 5e-5/1e-4 and ignores self.lr). Single AdamW group at self.lr —
    matching the P2 intent LR_ENCODER == LR_HEAD."""

    def configure_optimizers(self):
        lr = float(getattr(self, "lr", 5e-5))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.num_training_batches = self.trainer.num_training_batches
        return [opt]


# ─── P5: system-hardness-weighted BCE ────────────────────────────────────────

class Phoneme_GAT_Reweight_lit(_FinetuneOptMixin, Phoneme_GAT_lit):
    """
    Re-weights the per-utterance classification loss by a per-system hardness
    weight (top-C/T-quartile systems get up-weighted). Bonafide samples and
    systems absent from the weight table get weight 1.0. Weights are clipped to
    `max_weight`× the mean to keep training stable.
    """

    def configure_ablation(self, system_weights: dict[str, float], max_weight: float = 3.0):
        self.system_weights = dict(system_weights)
        self.max_weight = float(max_weight)
        self._bce_none = nn.BCEWithLogitsLoss(reduction="none")
        return self

    def _sample_weights(self, batch, device) -> torch.Tensor:
        systems = batch.get("attack_system", None)
        n = len(batch["label"])
        if systems is None:
            return torch.ones(n, device=device)
        w = torch.tensor([self.system_weights.get(s, 1.0) for s in systems],
                         dtype=torch.float32, device=device)
        return w.clamp(max=self.max_weight)

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        if not self.training:
            return super().calcuate_loss(batch_res, batch)

        w = self._sample_weights(batch, batch_res["logit"].device)
        per_sample = self._bce_none(batch_res["logit"], label.type(torch.float32))
        cls_loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)

        clip_loss = (
            self.clip_loss(batch_res["phoneme_feat"].mean(dim=-1),
                           self.clip_head(batch_res["encoder_feat"]).mean(dim=-1))
            if self.use_clip else 0.0
        )
        aug_loss = 0
        if self.use_aug and batch_res.get("aug_logit") is not None:
            aug_loss = self.bce_loss(batch_res["aug_logit"], label.type(torch.float32) * 0)

        loss = cls_loss + 0.5 * clip_loss + 0.5 * aug_loss
        return {"loss": loss, "cls_loss": cls_loss,
                "clip_loss": clip_loss, "aug_loss": aug_loss}


# ─── P6: bonafide representation-diversity regularizer ───────────────────────

class Phoneme_GAT_Diversity_lit(_FinetuneOptMixin, Phoneme_GAT_lit):
    """
    Adds an auxiliary loss that *maximizes* the radius of gyration (spread) of
    bonafide encoder frames, widening the C-axis gap between bonafide and
    synthetic representations. total += lambda_div * (−rog(bonafide frames)).
    """

    def configure_ablation(self, lambda_div: float = 0.01):
        self.lambda_div = float(lambda_div)
        return self

    def calcuate_loss(self, batch_res, batch):
        base = super().calcuate_loss(batch_res, batch)
        if not self.training:
            return base

        label = batch["label"]
        feat = batch_res.get("encoder_feat")  # (B, T', 768)
        div_loss = torch.zeros((), device=base["loss"].device)
        if feat is not None and feat.dim() == 3:
            bf_mask = (label == 0)
            if bf_mask.sum() > 1:
                rog = _rog_per_sample(feat[bf_mask]).mean()
                div_loss = -rog  # maximize spread
        base["loss"] = base["loss"] + self.lambda_div * div_loss
        base["div_loss"] = div_loss.detach()
        return base


# ─── P7: compactness-ordered pairwise ranking ────────────────────────────────

class Phoneme_GAT_CRank_lit(_FinetuneOptMixin, Phoneme_GAT_lit):
    """
    Augments the CLIP objective with a pairwise margin-ranking loss over the
    spoof samples in a batch: more-compact (higher −rog@L12) spoof utterances —
    which the mechanism says are harder — should receive *higher* spoof logits.
    Per-sample compactness is read from phoneme_feat (L12), already in batch_res.

    Speculative / noisy per the roadmap; lambda_rank defaults small.
    """

    def configure_ablation(self, lambda_rank: float = 0.1, margin: float = 0.0):
        self.lambda_rank = float(lambda_rank)
        self.margin = float(margin)
        return self

    def calcuate_loss(self, batch_res, batch):
        base = super().calcuate_loss(batch_res, batch)
        if not self.training:
            return base

        label = batch["label"]
        logit = batch_res["logit"]
        pf = batch_res.get("phoneme_feat")  # (B, T', 768) = L12 frames
        rank_loss = torch.zeros((), device=base["loss"].device)

        sp_mask = (label == 1)
        if pf is not None and sp_mask.sum() > 1:
            comp = -_rog_per_sample(pf[sp_mask])      # compactness, higher = harder
            sp_logit = logit[sp_mask]
            # All ordered pairs (i, j) with comp_i > comp_j: want logit_i > logit_j.
            ci = comp.unsqueeze(1) - comp.unsqueeze(0)       # (n, n)
            li = sp_logit.unsqueeze(1) - sp_logit.unsqueeze(0)
            pair_mask = (ci > 0).float()
            denom = pair_mask.sum().clamp_min(1.0)
            # hinge: penalize when logit gap < margin for compactness-ordered pairs
            rank_loss = (torch.relu(self.margin - li) * pair_mask).sum() / denom

        base["loss"] = base["loss"] + self.lambda_rank * rank_loss
        base["rank_loss"] = rank_loss.detach()
        return base
