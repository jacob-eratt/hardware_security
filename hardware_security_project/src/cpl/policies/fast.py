"""Vectorized NumPy simulator for phase-1 (hits-only) experiments.

Simulates many `run_experiment` calls at once: reset -> linear fill b0..b(W-1) ->
history -> trigger bW. The reference classes (`TreePLRU`, `LRU`, ...) remain the
source of truth; this module is tested against them in `tests/test_fast_sim.py`.

Histories are passed as a right-padded int matrix (pad value -1) plus lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .srrip import RRPV_LONG

FAST_POLICIES = ("plru", "lru", "fifo", "srrip")
PAD = -1


@dataclass
class BatchResult:
    """Victims per policy (int8, shape (n,)) and PLRU tree bits before the trigger."""

    victims: dict[str, np.ndarray]
    plru_bits: np.ndarray | None  # (n, W-1) int8, or None if W is not a power of two


def is_pow2(W: int) -> bool:
    return W >= 2 and (W & (W - 1)) == 0


def simulate_batch(
    hist: np.ndarray,
    lens: np.ndarray,
    W: int,
    policies: tuple[str, ...] = FAST_POLICIES,
) -> BatchResult:
    """Run phase-1 experiments for every row of `hist`.

    Args:
        hist: (n, Lmax) integer matrix; row i holds the history in its first lens[i]
            entries, every later entry is PAD (-1).
        lens: (n,) history lengths.
        W: associativity.
        policies: subset of FAST_POLICIES to simulate. "plru" requires power-of-two W.

    Returns:
        BatchResult with the victim block ID of each requested policy and, if "plru"
        was requested, the tree bits immediately before the trigger access.
    """
    hist = np.asarray(hist)
    lens = np.asarray(lens)
    if hist.ndim != 2 or lens.shape != (hist.shape[0],):
        raise ValueError("hist must be (n, Lmax) and lens (n,)")
    unknown = set(policies) - set(FAST_POLICIES)
    if unknown:
        raise ValueError(f"fast simulator does not support {sorted(unknown)}")
    n, Lmax = hist.shape
    if (lens < 0).any() or (lens > Lmax).any():
        raise ValueError("lens out of range")
    valid = np.arange(Lmax)[None, :] < lens[:, None]
    if ((hist < 0) | (hist >= W))[valid].any():
        raise ValueError(f"phase-1 histories must only contain block IDs in [0, {W})")
    if (hist[~valid] != PAD).any():
        raise ValueError("entries beyond each row's length must be PAD (-1)")

    want_plru = "plru" in policies
    if want_plru and not is_pow2(W):
        raise ValueError(f"tree-PLRU needs a power-of-two W, got {W}")

    rows = np.arange(n)
    # Fill prefix b0..b(W-1) followed by the history. Fills are into empty ways, and
    # linear fill puts b_i in way i, so block ID == way index throughout phase 1.
    seq = np.concatenate([np.broadcast_to(np.arange(W, dtype=hist.dtype), (n, W)), hist], axis=1)

    k = W.bit_length() - 1 if want_plru else 0
    bits = np.zeros((n, W - 1), dtype=np.int8) if want_plru else None
    last = np.full((n, W), -1, dtype=np.int32)          # LRU: last access time per way
    rrpv = np.full((n, W), RRPV_LONG, dtype=np.int8)    # SRRIP: all fills insert RRPV=2

    for t in range(seq.shape[1]):
        w = seq[:, t]
        if t < W:  # fill step: every row active
            r, ww = rows, w.astype(np.intp)
        else:
            m = w >= 0
            if not m.any():
                break
            r, ww = rows[m], w[m].astype(np.intp)
            rrpv[r, ww] = 0  # SRRIP hit promotion
        last[r, ww] = t
        if want_plru:
            # Touch: node at depth d on the path to w is (2^d - 1) + (w >> (k - d)); set
            # its bit to point away from w (1 if w is in its left half, 0 if right).
            for d in range(k):
                node = ((1 << d) - 1) + (ww >> (k - d))
                go_right = (ww >> (k - 1 - d)) & 1
                bits[r, node] = 1 - go_right

    victims: dict[str, np.ndarray] = {}
    if want_plru:
        node = np.zeros(n, dtype=np.intp)
        for _ in range(k):
            node = 2 * node + 1 + bits[rows, node]
        victims["plru"] = (node - (W - 1)).astype(np.int8)
    if "lru" in policies:
        victims["lru"] = last.argmin(axis=1).astype(np.int8)
    if "fifo" in policies:
        # Insertion order is the fill order and hits never change it -> always b0.
        victims["fifo"] = np.zeros(n, dtype=np.int8)
    if "srrip" in policies:
        # Aging adds the same amount to every way until one reaches RRPV_MAX, so the
        # victim is the lowest-index way holding the current maximum RRPV.
        victims["srrip"] = (rrpv == rrpv.max(axis=1, keepdims=True)).argmax(axis=1).astype(np.int8)
    return BatchResult(victims=victims, plru_bits=bits)


def pad_histories(histories: list[list[int]], min_width: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of histories into the (padded int8 matrix, lengths) form."""
    lens = np.array([len(h) for h in histories], dtype=np.int64)
    width = max(int(lens.max()) if len(lens) else 0, min_width)
    out = np.full((len(histories), width), PAD, dtype=np.int8)
    for i, h in enumerate(histories):
        out[i, : len(h)] = h
    return out, lens
