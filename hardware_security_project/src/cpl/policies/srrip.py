"""SRRIP-HP with M = 2 bits (Jaleel et al., ISCA 2010; comparison policy, spec §3.4)."""

from __future__ import annotations

import numpy as np

from .base import LinearFillPolicy

RRPV_MAX = 3  # 2**M - 1 with M = 2
RRPV_LONG = 2  # "long" re-reference insertion value


class SRRIP(LinearFillPolicy):
    """Insert with RRPV=2, hit sets RRPV=0, evict lowest-index way with RRPV=3 (aging all if none)."""

    name = "srrip"

    def _reset_state(self) -> None:
        self.rrpv: list[int] = [RRPV_MAX] * self.W  # value for empty ways is irrelevant

    def _insert_rrpv(self) -> int:
        return RRPV_LONG

    def _on_hit(self, w: int) -> None:
        self.rrpv[w] = 0

    def _on_insert(self, w: int) -> None:
        self.rrpv[w] = self._insert_rrpv()

    def _victim_way(self) -> int:
        while True:
            for w in range(self.W):
                if self.rrpv[w] == RRPV_MAX:
                    return w
            self.rrpv = [v + 1 for v in self.rrpv]

    def state(self) -> np.ndarray:
        """RRPV per way, shape (W,), int8 (-1 for empty ways)."""
        return np.array(
            [v if b is not None else -1 for v, b in zip(self.rrpv, self.ways)], dtype=np.int8
        )
