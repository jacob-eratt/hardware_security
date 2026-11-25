"""Vectorized NumPy simulator for DRRIP experiments whose histories may miss.

Simulates many `run_miss_experiment` calls on `DRRIP` at once: reset -> linear fill
b0..b(W-1) -> history (hits and misses) -> non-resident trigger. Each row has its own
mode (``brrip[i]`` True = BRRIP insertion, False = SRRIP insertion). The reference
class `DRRIP` is the source of truth; `tests/test_drrip.py` checks this module against it.

Histories are a right-padded int matrix (pad = -1) plus lengths. Block IDs may be any
value in [0, 127].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brrip import BIMODAL_P
from .srrip import RRPV_LONG, RRPV_MAX

TRIGGER = -2  # never resident


@dataclass
class MissResult:
    victim_way: np.ndarray   # (n,) int8, way evicted by the trigger
    n_misses: np.ndarray     # (n,) int16, misses inside the history (trigger excluded)


def _insert_values(brrip: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """Insertion RRPV per row: 2 for SRRIP rows; 3, or 2 with probability p, for BRRIP rows."""
    val = np.full(len(brrip), RRPV_LONG, dtype=np.int8)
    k = int(brrip.sum())
    if k:
        val[brrip] = np.where(rng.random(k) < p, RRPV_LONG, RRPV_MAX)
    return val


def _evict_insert(blocks: np.ndarray, rrpv: np.ndarray, idx: np.ndarray, new_block: np.ndarray,
                  brrip: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """Miss on rows ``idx``: age until some RRPV is 3, evict the lowest such way, insert."""
    r = rrpv[idx]
    r += (RRPV_MAX - r.max(axis=1))[:, None].astype(np.int8)
    v = (r == RRPV_MAX).argmax(axis=1)
    r[np.arange(len(idx)), v] = _insert_values(brrip[idx], rng, p)
    rrpv[idx] = r
    blocks[idx, v] = new_block
    return v


def simulate_drrip_misses(
    hist: np.ndarray,
    lens: np.ndarray,
    W: int,
    brrip: np.ndarray,
    rng: np.random.Generator,
    p: float = BIMODAL_P,
) -> MissResult:
    """Run one DRRIP miss experiment per row of ``hist``; see module docstring."""
    n = len(lens)
    brrip = np.asarray(brrip, dtype=bool)
    lens = np.asarray(lens)
    blocks = np.tile(np.arange(W, dtype=np.int16), (n, 1))
    # linear fill: every way is inserted once, in order
    rrpv = np.empty((n, W), dtype=np.int8)
    for w in range(W):
        rrpv[:, w] = _insert_values(brrip, rng, p)
    misses = np.zeros(n, dtype=np.int16)
    for t in range(hist.shape[1] if n else 0):
        act = lens > t
        if not act.any():
            break
        b = hist[:, t].astype(np.int16)
        eq = blocks == b[:, None]
        hit = eq.any(axis=1) & act
        rrpv[eq & hit[:, None]] = 0
        miss = np.flatnonzero(act & ~hit)
        if len(miss):
            _evict_insert(blocks, rrpv, miss, b[miss], brrip, rng, p)
            misses[miss] += 1
    victim = _evict_insert(blocks, rrpv, np.arange(n), np.full(n, TRIGGER, np.int16), brrip, rng, p)
    return MissResult(victim_way=victim.astype(np.int8), n_misses=misses)
