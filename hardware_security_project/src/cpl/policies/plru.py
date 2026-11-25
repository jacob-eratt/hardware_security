"""Tree-PLRU, the target policy (spec §3.3).

State: W-1 bits in heap order (node n -> children 2n+1, 2n+2; leaves = ways 0..W-1).
bit 0 = victim search goes left, bit 1 = goes right. Touch(w) points every node on
the path to w *away* from w. Reset: all bits 0, all ways empty.
"""

from __future__ import annotations

import numpy as np

from .base import LinearFillPolicy


class TreePLRU(LinearFillPolicy):
    """Reference tree-PLRU. Source of truth for all faster implementations."""

    name = "plru"

    def __init__(self, W: int) -> None:
        if not isinstance(W, int) or W < 2 or (W & (W - 1)) != 0:
            raise ValueError(f"tree-PLRU needs a power-of-two W >= 2, got {W!r}")
        super().__init__(W)

    def _reset_state(self) -> None:
        self.bits: list[int] = [0] * (self.W - 1)

    def _touch(self, w: int) -> None:
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            right = w >= mid
            self.bits[node] = 0 if right else 1
            node = 2 * node + (2 if right else 1)
            lo, hi = (mid, hi) if right else (lo, mid)

    _on_hit = _touch
    _on_insert = _touch

    def victim_way(self) -> int:
        """Follow the bits from the root to a leaf (does not modify state)."""
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.bits[node]:
                node, lo = 2 * node + 2, mid
            else:
                node, hi = 2 * node + 1, mid
        return lo

    def _victim_way(self) -> int:
        return self.victim_way()

    def state(self) -> np.ndarray:
        """Tree bits in heap order, shape (W-1,), int8."""
        return np.array(self.bits, dtype=np.int8)
