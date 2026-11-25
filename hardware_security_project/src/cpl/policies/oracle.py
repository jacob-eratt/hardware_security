"""Independent, recency-only PLRU oracle (spec §3.6).

With hits-only histories after a full linear fill, the PLRU victim depends only on the
order of each block's most recent access: each node points away from whichever half was
accessed most recently.
"""

from __future__ import annotations

from collections.abc import Sequence


def plru_victim_from_recency(W: int, history: Sequence[int]) -> int:
    """PLRU victim for `history` (phase-1, all IDs < W) computed from recency alone."""
    last = {b: i for i, b in enumerate(range(W))}  # fill counts as access at time i
    for t, b in enumerate(history):
        last[int(b)] = W + t
    lo, hi = 0, W
    while hi - lo > 1:
        mid = (lo + hi) // 2
        left_recent = max(last[b] for b in range(lo, mid))
        right_recent = max(last[b] for b in range(mid, hi))
        if left_recent > right_recent:
            lo = mid  # left half used more recently -> victim on the right
        else:
            hi = mid
    return lo


def lru_victim_from_recency(W: int, history: Sequence[int]) -> int:
    """LRU victim for `history` (phase-1): the block with the oldest last access."""
    last = {b: b for b in range(W)}
    for t, b in enumerate(history):
        last[int(b)] = W + t
    return min(range(W), key=last.__getitem__)
