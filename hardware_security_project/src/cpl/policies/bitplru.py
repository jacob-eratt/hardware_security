"""Bit-PLRU (a.k.a. MRU-bit / NRU-style PLRU; comparison policy).

State: one "recently used" bit per way. Every hit or fill sets the way's bit; if that
makes all W bits 1, every OTHER bit is cleared (the accessed way keeps its 1). The
victim is the lowest-index way whose bit is 0.
"""

from __future__ import annotations

import numpy as np

from .base import LinearFillPolicy


class BitPLRU(LinearFillPolicy):
    name = "bitplru"

    def _reset_state(self) -> None:
        self.bits: list[int] = [0] * self.W

    def _use(self, w: int) -> None:
        self.bits[w] = 1
        if all(self.bits):
            self.bits = [0] * self.W
            self.bits[w] = 1

    _on_hit = _use
    _on_insert = _use

    def _victim_way(self) -> int:
        return self.bits.index(0)          # always exists: a reset keeps exactly one 1

    def state(self) -> np.ndarray:
        """Reference bit per way, shape (W,), int8 (-1 for empty ways)."""
        return np.array([v if b is not None else -1 for v, b in zip(self.bits, self.ways)], dtype=np.int8)
