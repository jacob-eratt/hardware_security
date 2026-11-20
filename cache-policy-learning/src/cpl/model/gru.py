"""GRU victim classifier (spec §6.2)."""
from __future__ import annotations

import torch
from torch import nn

from cpl.data.tokens import PAD, VOCAB_SIZE, W_MAX


def way_mask(bound: torch.Tensor | int, n_classes: int, batch: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[batch, n_classes]`` mask, True for invalid classes (``class >= bound``).

    ``bound`` is the number of valid victim classes per sample: the associativity ``W``
    for hits-only data (victim is one of the W ways), or the block-pool size for miss
    data (victim can be any resident block). ``n_classes`` is the model's output width.
    """
    if not torch.is_tensor(bound):
        bound = torch.full((batch,), int(bound), device=device)
    return torch.arange(n_classes, device=device)[None, :] >= bound.to(device).long()[:, None]


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

    def forward(self, tokens: torch.Tensor, new_pos: torch.Tensor, bound: torch.Tensor | int) -> torch.Tensor:
        """Return masked logits ``[B, n_classes]``; entries for classes ``>= bound`` are ``-inf``.

        ``bound`` is the per-sample victim span (``W`` for hits-only, pool size for misses).
        """
        logits = self.head(self.norm(self.hidden_at_new(tokens, new_pos)))
        return logits.masked_fill(way_mask(bound, logits.size(-1), tokens.size(0), tokens.device), float("-inf"))


def build_model(cfg: dict) -> VictimGRU:
    """Construct a ``VictimGRU`` from the ``model`` section of a config."""
    return VictimGRU(
        d_emb=cfg.get("d_emb", 32),
        hidden=cfg.get("hidden", 128),
        layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.0),
        w_max=cfg.get("w_max", W_MAX),   # victim-class count; widen for miss data
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
