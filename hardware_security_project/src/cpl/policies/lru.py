"""True LRU (comparison policy, spec §3.4)."""

from __future__ import annotations

import numpy as np

from .base import LinearFillPolicy


class LRU(LinearFillPolicy):
    """Evict the block whose most recent access (fills count) is oldest."""

    name = "lru"

    def _reset_state(self) -> None:
        self._clock = 0
        self._stamp: list[int] = [-1] * self.W  # last-access time per way, -1 = empty

    def _use(self, w: int) -> None:
        self._stamp[w] = self._clock
        self._clock += 1

    _on_hit = _use
    _on_insert = _use

    def _victim_way(self) -> int:
        return min(range(self.W), key=self._stamp.__getitem__)

    def state(self) -> np.ndarray:
        """Recency rank per way: 0 = MRU, W-1 = LRU, -1 = empty. Shape (W,), int8."""
        valid = [w for w in range(self.W) if self._stamp[w] >= 0]
        order = sorted(valid, key=lambda w: -self._stamp[w])
        ranks = np.full(self.W, -1, dtype=np.int8)
        for r, w in enumerate(order):
            ranks[w] = r
        return ranks
