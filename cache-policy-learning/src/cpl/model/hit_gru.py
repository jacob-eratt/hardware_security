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
                 layers: int = 2, dropout: float = 0.0, n_set_ids: int = 0) -> None:
        super().__init__()
        self.pad = num_blocks
        self.embed = nn.Embedding(num_blocks + 1, d_emb, padding_idx=self.pad)
        self.gru = nn.GRU(d_emb, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)
        # multi-set data: the global set index is embedded and added to the line
        # embedding, so the baseline gets the same set information as the slot model
        self.set_embed = nn.Embedding(n_set_ids, d_emb) if n_set_ids else None

    def forward(self, accesses: torch.Tensor, sets: torch.Tensor | None = None,
                set_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``accesses`` [B, T] long (pad entries = self.pad). ``set_ids`` [B, T] global set
        index (multi-set data; ``sets``, the local slot, is already in the line ids).
        Returns logits [B, T]."""
        x = self.embed(accesses)
        if self.set_embed is not None:
            x = x + self.set_embed(set_ids)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def build_hit_model(cfg: dict) -> SeqHitGRU:
    return SeqHitGRU(
        num_blocks=cfg.get("num_blocks", NUM_BLOCKS),
        d_emb=cfg.get("d_emb", 32),
        hidden=cfg.get("hidden", 128),
        layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.0),
        n_set_ids=cfg.get("n_set_ids", 0),
    )


def gru_from_ckpt(ck: dict) -> SeqHitGRU:
    """Rebuild a cpl.train_hits checkpoint (old checkpoints: single-set defaults)."""
    a = ck["args"]
    m = build_hit_model({"hidden": a["hidden"], "layers": a["layers"],
                         "num_blocks": a.get("num_blocks", NUM_BLOCKS), "n_set_ids": a.get("n_set_ids", 0)})
    m.load_state_dict(ck["model"])
    return m
