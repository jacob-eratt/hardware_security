"""DRRIP follower set with a forced mode (Jaleel et al., ISCA 2010).

DRRIP duels SRRIP against BRRIP using leader sets and a global PSEL counter; follower
sets use whichever sub-policy PSEL currently favours. In single-set simulation the
global state is not modelled: the active sub-policy is given explicitly as ``mode``
(the ``forced_mode`` column of the dataset). Both modes share SRRIP's hit promotion and
victim search and differ only in the insertion RRPV:

- ``srrip``: insert with RRPV = 2 (deterministic)
- ``brrip``: insert with RRPV = 3, except with probability ``p`` (default 1/32) RRPV = 2

With hits-only histories the two modes pick the same victim ~97% of the time at W = 16,
so DRRIP datasets use histories that also contain misses (see ``run_miss_experiment``).
"""

from __future__ import annotations

import numpy as np

from .brrip import BIMODAL_P
from .srrip import RRPV_LONG, RRPV_MAX, SRRIP

DRRIP_MODES = ("srrip", "brrip")


class DRRIP(SRRIP):
    """DRRIP follower set whose sub-policy is fixed to ``mode``."""

    name = "drrip"

    def __init__(self, W: int, mode: str = "srrip", seed: int | None = 0, p: float = BIMODAL_P) -> None:
        if mode not in DRRIP_MODES:
            raise ValueError(f"mode must be one of {DRRIP_MODES}, got {mode!r}")
        self.mode = mode
        self.p = p
        self.rng = np.random.default_rng(seed)
        super().__init__(W)

    def _insert_rrpv(self) -> int:
        if self.mode == "srrip":
            return RRPV_LONG
        return RRPV_LONG if self.rng.random() < self.p else RRPV_MAX
