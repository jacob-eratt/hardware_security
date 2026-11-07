"""FIFO (comparison policy, spec §3.4).

Note: with hits-only histories after a fill, FIFO always evicts b0. This is expected,
not a finding.
"""

from __future__ import annotations

import numpy as np

from .base import LinearFillPolicy


class FIFO(LinearFillPolicy):
    """Evict the block inserted earliest; hits do not change state."""

    name = "fifo"

    def _reset_state(self) -> None:
        self._clock = 0
        self._inserted: list[int] = [-1] * self.W  # insertion time per way, -1 = empty

    def _on_hit(self, w: int) -> None:
        pass

    def _on_insert(self, w: int) -> None:
        self._inserted[w] = self._clock
        self._clock += 1

    def _victim_way(self) -> int:
        return min(range(self.W), key=self._inserted.__getitem__)

    def state(self) -> np.ndarray:
        """Insertion-age rank per way: 0 = newest, W-1 = oldest (next victim), -1 = empty."""
        valid = [w for w in range(self.W) if self._inserted[w] >= 0]
        order = sorted(valid, key=lambda w: -self._inserted[w])
        ranks = np.full(self.W, -1, dtype=np.int8)
        for r, w in enumerate(order):
            ranks[w] = r
        return ranks
