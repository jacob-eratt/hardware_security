"""Soft-label cross-entropy with masked ways (spec §6.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(masked_logits: torch.Tensor, victim_counts: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy against ``victim_counts`` normalized to a distribution.

    Masked ways carry ``-inf`` logits and zero target mass; ``torch.where`` drops them
    so ``0 * -inf`` never produces NaN.
    """
    counts = victim_counts.float()
    p_target = counts / counts.sum(-1, keepdim=True).clamp_min(1e-12)
    log_p = F.log_softmax(masked_logits.float(), dim=-1)
    per_way = torch.where(p_target > 0, p_target * log_p, torch.zeros_like(log_p))
    loss = -per_way.sum(-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
