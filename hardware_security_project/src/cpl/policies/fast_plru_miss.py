"""Vectorized tree-PLRU simulator for experiments whose histories may miss.

Simulates many `run_miss_experiment` calls on `TreePLRU` at once: reset -> linear fill
b0..b(W-1) -> history (hits and misses) -> non-resident trigger. On a miss the tree
victim is evicted, the new block takes its way and that way is touched (spec §3.3).
`TreePLRU` is the source of truth; `tests/test_plru_miss.py` checks this module against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fast import is_pow2


@dataclass
class PLRUMissResult:
    victim_way: np.ndarray   # (n,) int8, way evicted by the trigger
    n_misses: np.ndarray     # (n,) int16, misses inside the history (trigger excluded)
    bits: np.ndarray         # (n, W-1) int8, tree bits just before the trigger


def _touch(bits: np.ndarray, rows: np.ndarray, way: np.ndarray, W: int) -> None:
    """Point every node on the root-to-`way` path away from `way` (rows given by index)."""
    node = np.zeros(len(rows), np.int64)
    lo = np.zeros(len(rows), np.int64)
    hi = np.full(len(rows), W, np.int64)
    while (hi - lo > 1).any():
        mid = (lo + hi) // 2
        right = way >= mid
        bits[rows, node] = np.where(right, 0, 1)
        node = 2 * node + np.where(right, 2, 1)
        lo, hi = np.where(right, mid, lo), np.where(right, hi, mid)


def _victim(bits: np.ndarray, rows: np.ndarray, W: int) -> np.ndarray:
    """Follow the tree bits from the root: 0 = left, 1 = right."""
    node = np.zeros(len(rows), np.int64)
    lo = np.zeros(len(rows), np.int64)
    hi = np.full(len(rows), W, np.int64)
    while (hi - lo > 1).any():
        mid = (lo + hi) // 2
        go_right = bits[rows, node] == 1
        node = 2 * node + np.where(go_right, 2, 1)
        lo, hi = np.where(go_right, mid, lo), np.where(go_right, hi, mid)
    return lo


def simulate_plru_misses(hist: np.ndarray, lens: np.ndarray, W: int) -> PLRUMissResult:
    """Run one tree-PLRU miss experiment per row of ``hist`` (pad = -1)."""
    if not is_pow2(W):
        raise ValueError("tree-PLRU needs power-of-two W")
    n = len(lens)
    lens = np.asarray(lens)
    rows = np.arange(n)
    bits = np.zeros((n, W - 1), np.int8)
    blocks = np.tile(np.arange(W, dtype=np.int16), (n, 1))
    for w in range(W):  # linear fill touches every way in order
        _touch(bits, rows, np.full(n, w), W)
    misses = np.zeros(n, np.int16)
    for t in range(hist.shape[1] if n else 0):
        act = lens > t
        if not act.any():
            break
        b = hist[:, t].astype(np.int16)
        eq = blocks == b[:, None]
        hit = eq.any(axis=1) & act
        h = np.flatnonzero(hit)
        if len(h):
            _touch(bits, h, eq[h].argmax(axis=1), W)
        m = np.flatnonzero(act & ~hit)
        if len(m):
            v = _victim(bits, m, W)
            blocks[m, v] = b[m]
            _touch(bits, m, v, W)
            misses[m] += 1
    snapshot = bits.copy()
    victim = _victim(bits, rows, W)
    return PLRUMissResult(victim_way=victim.astype(np.int8), n_misses=misses, bits=snapshot)
