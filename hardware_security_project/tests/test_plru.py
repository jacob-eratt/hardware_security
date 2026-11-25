"""Tree-PLRU: spec §3.5 test vectors, §3.6 oracle agreement, and interface behaviour."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cpl.policies import (
    LRU,
    TreePLRU,
    lru_victim_from_recency,
    plru_victim_from_recency,
    run_experiment,
)


# --------------------------------------------------------------------------------------
# Verbatim copy of the reference implementation in spec §3.3, used as an extra
# independent check that our class (which adds the full interface) did not drift.
class _SpecTreePLRU:
    def __init__(self, W):
        assert W >= 2 and (W & (W - 1)) == 0, "tree-PLRU needs power-of-two W"
        self.W = W
        self.reset()

    def reset(self):
        self.bits = [0] * (self.W - 1)
        self.ways = [None] * self.W

    def _touch(self, w):
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            right = w >= mid
            self.bits[node] = 0 if right else 1
            node = 2 * node + (2 if right else 1)
            lo, hi = (mid, hi) if right else (lo, mid)

    def victim_way(self):
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.bits[node]:
                node, lo = 2 * node + 2, mid
            else:
                node, hi = 2 * node + 1, mid
        return lo

    def access(self, block):
        if block in self.ways:
            self._touch(self.ways.index(block))
            return None
        w = self.ways.index(None) if None in self.ways else self.victim_way()
        evicted, self.ways[w] = self.ways[w], block
        self._touch(w)
        return evicted


# --------------------------------------------------------------------------------------
# §3.5 required test vectors: (W, history, plru_victim, bits_before_trigger | None, lru_victim)
VECTORS = [
    (4, [], 0, [0, 0, 0], 0),
    (4, [0], 2, [1, 1, 0], 1),
    (4, [0, 1, 2, 3], 0, [0, 0, 0], 0),
    (4, [3, 2, 1, 0], 3, [1, 1, 1], 3),
    (4, [1, 1, 1], 2, [1, 0, 0], 0),
    (4, [2, 0, 3], 1, [0, 1, 0], 1),
    (8, [], 0, [0] * 7, 0),
    (16, [], 0, [0] * 15, 0),
    (16, [0, 5, 9], 2, None, 1),
    (16, [15, 14, 13, 12, 11, 10, 9, 8], 0, None, 0),
    (16, [3, 3, 3, 7, 0, 12], 4, None, 1),
]


@pytest.mark.parametrize("W,history,plru_v,bits,lru_v", VECTORS)
def test_spec_vectors(W, history, plru_v, bits, lru_v):
    victim, state = run_experiment(TreePLRU(W), history)
    assert victim == plru_v
    assert state.shape == (W - 1,)
    assert state.dtype == np.int8
    if bits is not None:
        assert state.tolist() == bits
    assert run_experiment(LRU(W), history)[0] == lru_v
    # the oracles agree on these vectors too
    assert plru_victim_from_recency(W, history) == plru_v
    assert lru_victim_from_recency(W, history) == lru_v


def _random_histories(n: int, seed: int, Ws=(4, 8, 16), L_max: int = 64):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        W = int(rng.choice(Ws))
        L = int(rng.integers(0, L_max + 1))
        yield W, rng.integers(0, W, size=L).tolist()


def test_oracle_agreement_random():
    """§3.6: TreePLRU and the recency oracle agree on >= 10,000 random histories."""
    sims = {W: TreePLRU(W) for W in (4, 8, 16)}
    n = 0
    for W, h in _random_histories(12_000, seed=0):
        v, _ = run_experiment(sims[W], h)
        assert v == plru_victim_from_recency(W, h), (W, h)
        n += 1
    assert n >= 10_000


def test_oracle_agreement_w16_20k():
    """The spec states agreement was checked on 20,000 W=16 histories; reproduce it."""
    sim = TreePLRU(16)
    for W, h in _random_histories(20_000, seed=1, Ws=(16,), L_max=64):
        assert run_experiment(sim, h)[0] == plru_victim_from_recency(W, h), h


@pytest.mark.parametrize("W,L_max", [(4, 8), (8, 5), (16, 3)])
def test_oracle_agreement_exhaustive(W, L_max):
    """Every possible history up to L_max (not a sample) matches the recency oracle."""
    sim, n = TreePLRU(W), 0
    for L in range(L_max + 1):
        for h in itertools.product(range(W), repeat=L):
            assert run_experiment(sim, list(h))[0] == plru_victim_from_recency(W, h), h
            n += 1
    assert n == sum(W**L for L in range(L_max + 1))


def test_w2_plru_equals_lru_exhaustive():
    """With one tree bit, PLRU remembers exactly the MRU way, so it must equal LRU."""
    for L in range(13):
        for h in itertools.product(range(2), repeat=L):
            assert run_experiment(TreePLRU(2), list(h))[0] == run_experiment(LRU(2), list(h))[0], h


def test_trace_w4_bits_after_each_access():
    """Hand-checkable trace for W=4, history [2, 0, 3] (bits = [root, left node, right node])."""
    p = TreePLRU(4)
    expected_fill = [[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]
    for b, bits in zip(range(4), expected_fill):
        p.access(b)
        assert p.state().tolist() == bits, f"after fill b{b}"
    for b, bits, victim in [(2, [0, 0, 1], 0), (0, [1, 1, 1], 3), (3, [0, 1, 0], 1)]:
        p.access(b)
        assert p.state().tolist() == bits, f"after hit b{b}"
        assert p.victim_way() == victim
    assert p.access(4) == 1


def test_matches_verbatim_spec_reference():
    """Victim and bits identical to the spec's reference class, incl. miss sequences."""
    rng = np.random.default_rng(2)
    for _ in range(3_000):
        W = int(rng.choice([2, 4, 8, 16]))
        ours, ref = TreePLRU(W), _SpecTreePLRU(W)
        # arbitrary sequences over 2W blocks -> mixes hits, fills and evictions
        for b in rng.integers(0, 2 * W, size=int(rng.integers(0, 80))).tolist():
            assert ours.access(b) == ref.access(b)
            assert ours.resident() == ref.ways
            assert ours.state().tolist() == ref.bits


def test_reset_and_fill():
    p = TreePLRU(8)
    assert p.resident() == [None] * 8
    assert p.state().tolist() == [0] * 7
    for b in range(8):
        assert p.access(b) is None  # fill into empty way
    assert p.resident() == list(range(8))  # linear fill: b_i in way i
    assert p.access(3) is None  # hit
    p.reset()
    assert p.resident() == [None] * 8 and p.state().tolist() == [0] * 7


def test_miss_sequence_w4():
    """Hand-derived: after fill (bits 000), new blocks 4,5,6,7 evict ways 0,2,1,3."""
    p = TreePLRU(4)
    for b in range(4):
        p.access(b)
    assert [p.access(b) for b in (4, 5, 6, 7)] == [0, 2, 1, 3]
    assert p.resident() == [4, 6, 5, 7]
    assert p.state().tolist() == [0, 0, 0]  # last touch was way 3


def test_touch_points_away():
    """After touching way w, the victim is never w (for W >= 2)."""
    for W in (2, 4, 8, 16):
        p = TreePLRU(W)
        for b in range(W):
            p.access(b)
        for w in range(W):
            p.access(w)
            assert p.victim_way() != w


def test_state_is_a_copy():
    p = TreePLRU(4)
    s = p.state()
    s[:] = 1
    assert p.state().tolist() == [0, 0, 0]


def test_clone_is_independent():
    p = TreePLRU(16)
    for b in range(16):
        p.access(b)
    p.access(5)
    q = p.clone()
    assert q.resident() == p.resident() and np.array_equal(q.state(), p.state())
    q.access(0)
    q.access(99)
    assert p.resident() == list(range(16))
    assert not np.array_equal(q.state(), p.state())


@pytest.mark.parametrize("W", [0, 1, 3, 6, 12, 11])
def test_rejects_non_power_of_two(W):
    with pytest.raises(ValueError):
        TreePLRU(W)


def test_run_experiment_rejects_non_hit_history():
    with pytest.raises(AssertionError):
        run_experiment(TreePLRU(4), [0, 4])  # block 4 is not resident -> not phase 1
    with pytest.raises(AssertionError):
        run_experiment(TreePLRU(4), [-1])


def test_run_experiment_is_repeatable_on_same_instance():
    p = TreePLRU(16)
    h = [3, 1, 4, 1, 5, 9, 2, 6]
    assert run_experiment(p, h)[0] == run_experiment(p, h)[0]
