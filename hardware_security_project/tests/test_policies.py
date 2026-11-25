"""LRU / FIFO / SRRIP / BRRIP unit tests and cross-policy properties (spec §3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cpl.policies import (
    BRRIP,
    FIFO,
    LRU,
    POLICIES,
    SRRIP,
    ReplacementPolicy,
    TreePLRU,
    lru_victim_from_recency,
    make_policy,
    plru_victim_from_recency,
    run_experiment,
)

NON_TREE = ["lru", "fifo", "srrip", "brrip", "bitplru"]


def _filled(policy):
    for b in range(policy.W):
        assert policy.access(b) is None
    return policy


# ---------------------------------------------------------------- generic interface
@pytest.mark.parametrize("name", sorted(POLICIES))
@pytest.mark.parametrize("W", [4, 8, 16])
def test_interface_and_linear_fill(name, W):
    p = make_policy(name, W)
    assert isinstance(p, ReplacementPolicy)
    assert p.W == W and p.resident() == [None] * W
    _filled(p)
    assert p.resident() == list(range(W))
    for b in range(W):
        assert p.access(b) is None  # all hits
    ev = p.access(W)
    assert ev is not None and 0 <= ev < W
    assert W in p.resident() and ev not in p.resident()
    assert isinstance(p.state(), np.ndarray)


@pytest.mark.parametrize("name", NON_TREE)
@pytest.mark.parametrize("W", [2, 3, 11, 12])
def test_non_tree_policies_support_any_W(name, W):
    p = _filled(make_policy(name, W))
    rng = np.random.default_rng(W)
    for b in rng.integers(0, 3 * W, size=200).tolist():
        before = p.resident()
        ev = p.access(b)
        after = p.resident()
        if b in before:
            assert ev is None and after == before
        else:
            assert ev in before
            assert sum(x != y for x, y in zip(before, after)) == 1
        assert len(set(after)) == W  # no duplicates, set stays full


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_clone_independent(name):
    p = _filled(make_policy(name, 8))
    p.access(3)
    q = p.clone()
    for b in range(100, 110):
        q.access(b)
    assert p.resident() == list(range(8))
    assert q.resident() != p.resident()


def test_make_policy_unknown():
    with pytest.raises(ValueError):
        make_policy("mru", 8)


# ---------------------------------------------------------------- LRU
def test_lru_matches_recency_oracle():
    rng = np.random.default_rng(0)
    sims = {W: LRU(W) for W in (4, 8, 11, 16)}
    for _ in range(5_000):
        W = int(rng.choice(list(sims)))
        h = rng.integers(0, W, size=int(rng.integers(0, 65))).tolist()
        assert run_experiment(sims[W], h)[0] == lru_victim_from_recency(W, h)


def test_lru_miss_sequence_and_state():
    p = _filled(LRU(4))
    assert p.state().tolist() == [3, 2, 1, 0]  # way 0 is LRU, way 3 MRU
    p.access(0)
    assert p.state().tolist() == [0, 3, 2, 1]
    assert [p.access(b) for b in (4, 5, 6)] == [1, 2, 3]
    assert p.access(7) == 0


# ---------------------------------------------------------------- FIFO
def test_fifo_always_evicts_b0_on_hit_histories():
    rng = np.random.default_rng(0)
    for W in (4, 8, 16):
        p = FIFO(W)
        for _ in range(500):
            h = rng.integers(0, W, size=int(rng.integers(0, 65))).tolist()
            assert run_experiment(p, h)[0] == 0


def test_fifo_miss_sequence():
    p = _filled(FIFO(4))
    p.access(0)  # hit does not protect b0
    assert [p.access(b) for b in (4, 5, 6, 7, 8)] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- SRRIP
@pytest.mark.parametrize(
    "history,victim,rrpv_before",
    [
        ([], 0, [2, 2, 2, 2]),
        ([0], 1, [0, 2, 2, 2]),
        ([3], 0, [2, 2, 2, 0]),
        ([1, 2, 3], 0, [2, 0, 0, 0]),
        ([0, 1, 2, 3], 0, [0, 0, 0, 0]),
        ([0, 2], 1, [0, 2, 0, 2]),
    ],
)
def test_srrip_hand_vectors(history, victim, rrpv_before):
    v, state = run_experiment(SRRIP(4), history)
    assert v == victim
    assert state.tolist() == rrpv_before


def test_srrip_aging_and_insert():
    p = _filled(SRRIP(4))
    p.access(0)  # [0,2,2,2]
    assert p.access(4) == 1  # age -> [1,3,3,3], evict way 1, insert RRPV 2
    assert p.state().tolist() == [1, 2, 3, 3]
    assert p.access(5) == 2  # way 2 already at 3 -> no aging
    assert p.state().tolist() == [1, 2, 2, 3]
    assert p.access(6) == 3
    assert p.access(7) == 4  # age -> [2,3,3,3]; lowest-index RRPV 3 is way 1 (block 4)
    assert p.resident() == [0, 7, 5, 6]  # hit block b0 survived the scan
    assert p.state().tolist() == [2, 2, 3, 3]


def test_srrip_state_marks_empty_ways():
    p = SRRIP(4)
    p.access(0)
    assert p.state().tolist() == [2, -1, -1, -1]


# ---------------------------------------------------------------- BRRIP
def test_brrip_seeded_determinism():
    seq = np.random.default_rng(5).integers(0, 40, size=2_000).tolist()
    a, b, c = BRRIP(8, seed=7), BRRIP(8, seed=7), BRRIP(8, seed=8)
    ra = [a.access(x) for x in seq]
    rb = [b.access(x) for x in seq]
    rc = [c.access(x) for x in seq]
    assert ra == rb
    assert ra != rc


def test_brrip_insertion_rate():
    p = BRRIP(4, seed=0)
    draws = np.array([p._insert_rrpv() for _ in range(64_000)])
    frac_long = float((draws == 2).mean())
    assert set(np.unique(draws).tolist()) <= {2, 3}
    assert abs(frac_long - 1 / 32) < 0.004


def test_brrip_clone_copies_rng():
    p = _filled(BRRIP(8, seed=3))
    q = p.clone()
    seq = list(range(100, 300))
    assert [p.access(x) for x in seq] == [q.access(x) for x in seq]


# ---------------------------------------------------------------- cross-policy stats
def test_plru_lru_disagreement_w16():
    """Spec §1/§7.3: PLRU and LRU disagree on ~77% of random W=16 histories (L in [0,32])."""
    rng = np.random.default_rng(0)
    n, disagree = 10_000, 0
    for _ in range(n):
        h = rng.integers(0, 16, size=int(rng.integers(0, 33))).tolist()
        disagree += plru_victim_from_recency(16, h) != lru_victim_from_recency(16, h)
    assert 0.72 < disagree / n < 0.82, disagree / n


def test_plru_victim_skew_w16():
    """Spec §4.4: victim distribution skewed to way 0 (~15%) for L in [0,32]; ~uniform at L=64."""
    rng = np.random.default_rng(0)
    sim = TreePLRU(16)
    short = np.bincount(
        [run_experiment(sim, rng.integers(0, 16, size=int(rng.integers(0, 33))).tolist())[0]
         for _ in range(10_000)],
        minlength=16,
    ) / 10_000
    assert 0.12 < short[0] < 0.18
    assert short[0] == short.max()
    assert short[1::2].mean() < short[0::2].mean()  # odd ways are rarer
    long = np.bincount(
        [run_experiment(sim, rng.integers(0, 16, size=64).tolist())[0] for _ in range(10_000)],
        minlength=16,
    ) / 10_000
    assert np.abs(long - 1 / 16).max() < 0.02
