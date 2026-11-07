"""GRU victim classifier (spec §6.2)."""
from __future__ import annotations

import torch
from torch import nn

from cpl.data.tokens import PAD, VOCAB_SIZE, W_MAX


def way_mask(W: torch.Tensor | int, batch: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[batch, W_MAX]`` mask that is True for invalid ways (``way >= W``)."""
    if not torch.is_tensor(W):
        W = torch.full((batch,), int(W), device=device)
    return torch.arange(W_MAX, device=device)[None, :] >= W.to(device).long()[:, None]


class VictimGRU(nn.Module):
    """Embedding -> GRU -> hidden at NEW -> LayerNorm -> Linear(W_MAX), ways >= W masked to -inf."""

    def __init__(
        self,
        d_emb: int = 32,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        vocab_size: int = VOCAB_SIZE,
        w_max: int = W_MAX,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_emb, padding_idx=PAD)
        self.gru = nn.GRU(d_emb, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, w_max)

    def hidden_at_new(self, tokens: torch.Tensor, new_pos: torch.Tensor) -> torch.Tensor:
        """Last-layer GRU output at the ``NEW`` position, shape ``[B, hidden]``."""
        out, _ = self.gru(self.embed(tokens))
        idx = new_pos.long().view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)

    def forward(self, tokens: torch.Tensor, new_pos: torch.Tensor, W: torch.Tensor | int) -> torch.Tensor:
        """Return masked logits ``[B, W_MAX]``; entries for ways ``>= W`` are ``-inf``."""
        logits = self.head(self.norm(self.hidden_at_new(tokens, new_pos)))
        return logits.masked_fill(way_mask(W, tokens.size(0), tokens.device), float("-inf"))


def build_model(cfg: dict) -> VictimGRU:
    """Construct a ``VictimGRU`` from the ``model`` section of a config."""
    return VictimGRU(
        d_emb=cfg.get("d_emb", 32),
        hidden=cfg.get("hidden", 128),
        layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.0),
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
