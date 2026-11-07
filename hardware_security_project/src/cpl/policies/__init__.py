"""Single-set cache replacement policy simulators."""

from __future__ import annotations

from .base import LinearFillPolicy, ReplacementPolicy, run_experiment
from .brrip import BRRIP
from .fifo import FIFO
from .lru import LRU
from .oracle import lru_victim_from_recency, plru_victim_from_recency
from .plru import TreePLRU
from .srrip import SRRIP

POLICIES: dict[str, type[LinearFillPolicy]] = {
    "plru": TreePLRU,
    "lru": LRU,
    "fifo": FIFO,
    "srrip": SRRIP,
    "brrip": BRRIP,
}


def make_policy(name: str, W: int, seed: int | None = 0) -> LinearFillPolicy:
    """Construct a policy by name. `seed` is only used by stochastic policies (brrip)."""
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
    if cls is BRRIP:
        return BRRIP(W, seed=seed)
    return cls(W)


__all__ = [
    "BRRIP",
    "FIFO",
    "LRU",
    "POLICIES",
    "SRRIP",
    "LinearFillPolicy",
    "ReplacementPolicy",
    "TreePLRU",
    "lru_victim_from_recency",
    "make_policy",
    "plru_victim_from_recency",
    "run_experiment",
]
