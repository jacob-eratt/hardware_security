"""Bit-PLRU: hand-worked vectors and an independent set-based reformulation."""
from __future__ import annotations

import numpy as np

from cpl.policies import BitPLRU, make_policy


def test_hand_vector_w4():
    p = BitPLRU(4)
    for b in range(4):                       # fill ways 0..3; filling way 3 sets the last bit
        assert p.access(b) is None
    assert p.state().tolist() == [0, 0, 0, 1]
    assert p.access(4) == 0                  # lowest 0-bit way is 0 -> evict block 0
    assert p.state().tolist() == [1, 0, 0, 1]
    assert p.access(1) is None               # hit way 1
    assert p.state().tolist() == [1, 1, 0, 1]
    assert p.access(5) == 2                  # way 2 evicted; setting it fills all bits -> reset
    assert p.state().tolist() == [0, 0, 1, 0]
    assert p.access(6) == 4                  # way 0 (holding block 4) is the lowest 0
    assert p.state().tolist() == [1, 0, 1, 0]


def _independent(W, seq):
    """Set-based reformulation: the 1-bits are the ways used since the last 'epoch' began;
    an epoch ends when every way has been used, and the new epoch starts with that way."""
    ways, epoch, out = [None] * W, set(), []
    for b in seq:
        if b in ways:
            w = ways.index(b); ev = None
        elif None in ways:
            w = ways.index(None); ways[w] = b; ev = None
        else:
            w = min(set(range(W)) - epoch); ev = ways[w]; ways[w] = b
        epoch = {w} if (epoch | {w}) == set(range(W)) else epoch | {w}
        out.append(ev)
    return out


def test_matches_independent_reformulation():
    rng = np.random.default_rng(0)
    for W in (4, 8, 16):
        for _ in range(200):
            seq = rng.integers(0, W + 8, size=120).tolist()
            p = make_policy("bitplru", W)
            assert [p.access(b) for b in seq] == _independent(W, seq)
