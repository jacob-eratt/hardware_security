"""BRRIP (optional, stochastic comparison policy, spec §3.4)."""

from __future__ import annotations

import numpy as np

from .srrip import RRPV_LONG, RRPV_MAX, SRRIP

BIMODAL_P = 1 / 32


class BRRIP(SRRIP):
    """Like SRRIP but inserts with RRPV=3, except with probability 1/32 inserts with RRPV=2.

    Stochastic: the RNG is explicitly seeded. `reset()` does NOT reseed, so repeated
    experiments on one instance draw fresh randomness; use `seed=` for reproducibility.
    """

    name = "brrip"

    def __init__(self, W: int, seed: int | None = 0, p: float = BIMODAL_P) -> None:
        self.rng = np.random.default_rng(seed)
        self.p = p
        super().__init__(W)

    def _insert_rrpv(self) -> int:
        return RRPV_LONG if self.rng.random() < self.p else RRPV_MAX
