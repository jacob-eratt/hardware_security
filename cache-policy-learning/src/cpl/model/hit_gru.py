"""Streaming hit/miss predictor: embed access IDs -> GRU -> per-access P(hit).

Given a sequence of accesses to one cache set, predict at each position whether
that access hits (served by cache) or misses (DRAM). A plain causal GRU: the
output at step t depends on accesses 0..t, so it predicts the current access's
outcome from the running history -- it is NOT told prior hit/miss outcomes, so
it must infer residency from the access pattern. One sigmoid output per step.
"""
from __future__ import annotations

import torch
from torch import nn

NUM_BLOCKS = 32  # block ids 0..31; the embedding's padding index is NUM_BLOCKS


class SeqHitGRU(nn.Module):
    def __init__(self, num_blocks: int = NUM_BLOCKS, d_emb: int = 32, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.pad = num_blocks
        self.embed = nn.Embedding(num_blocks + 1, d_emb, padding_idx=self.pad)
        self.gru = nn.GRU(d_emb, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, accesses: torch.Tensor) -> torch.Tensor:
        """``accesses`` [B, T] long (pad entries = self.pad). Returns logits [B, T]."""
        out, _ = self.gru(self.embed(accesses))
        return self.head(out).squeeze(-1)


def build_hit_model(cfg: dict) -> SeqHitGRU:
    return SeqHitGRU(
        num_blocks=cfg.get("num_blocks", NUM_BLOCKS),
        d_emb=cfg.get("d_emb", 32),
        hidden=cfg.get("hidden", 128),
        layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.0),
    )
