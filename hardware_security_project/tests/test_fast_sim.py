"""The vectorized simulator must match the reference classes exactly (spec §4.5, §10)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cpl.policies import FIFO, LRU, SRRIP, TreePLRU, run_experiment
from cpl.policies.fast import pad_histories, simulate_batch

REF = {"plru": TreePLRU, "lru": LRU, "fifo": FIFO, "srrip": SRRIP}


def _random(n, Ws, L_max, seed):
    rng = np.random.default_rng(seed)
    out = {W: [] for W in Ws}
    for _ in range(n):
        W = int(rng.choice(Ws))
        out[W].append(rng.integers(0, W, size=int(rng.integers(0, L_max + 1))).tolist())
    return out


def _check(W, hists, policies):
    hist, lens = pad_histories(hists)
    res = simulate_batch(hist, lens, W, policies=policies)
    sims = {p: REF[p](W) for p in policies}
    for i, h in enumerate(hists):
        for p in policies:
            v, state = run_experiment(sims[p], h)
            assert res.victims[p][i] == v, (p, W, h)
            if p == "plru":
                assert res.plru_bits[i].tolist() == state.tolist(), (W, h)


def test_plru_matches_reference_100k():
    """>= 100k random histories: victim and pre-trigger bits identical to TreePLRU."""
    groups = _random(100_000, (4, 8, 16), 64, seed=0)
    assert sum(map(len, groups.values())) >= 100_000
    for W, hists in groups.items():
        _check(W, hists, ("plru",))


def test_other_policies_match_reference():
    for W, hists in _random(30_000, (4, 8, 16), 64, seed=1).items():
        _check(W, hists, ("lru", "fifo", "srrip"))


@pytest.mark.parametrize("W", [3, 11, 12])
def test_non_power_of_two_W(W):
    _check(W, _random(3_000, (W,), 64, seed=W)[W], ("lru", "fifo", "srrip"))


def test_exhaustive_w4_all_policies():
    hists = [list(h) for L in range(7) for h in itertools.product(range(4), repeat=L)]
    _check(4, hists, ("plru", "lru", "fifo", "srrip"))


def test_long_histories():
    for W, hists in _random(500, (4, 16), 512, seed=2).items():
        _check(W, hists, ("plru", "lru", "srrip"))


def test_empty_batch_and_empty_histories():
    res = simulate_batch(np.zeros((0, 5), np.int8), np.zeros(0, np.int64), 8)
    assert all(len(v) == 0 for v in res.victims.values())
    res = simulate_batch(np.full((3, 0), -1, np.int8), np.zeros(3, np.int64), 8)
    assert res.victims["plru"].tolist() == [0, 0, 0]
    assert res.plru_bits.tolist() == [[0] * 7] * 3


def test_input_validation():
    hist, lens = pad_histories([[0, 4]])
    with pytest.raises(ValueError):
        simulate_batch(hist, lens, 4)  # block 4 not resident
    with pytest.raises(ValueError):
        simulate_batch(np.array([[1, 2]], np.int8), np.array([1]), 4)  # tail not PAD
    with pytest.raises(ValueError):
        simulate_batch(np.full((1, 2), -1, np.int8), np.array([0]), 6, policies=("plru",))
    with pytest.raises(ValueError):
        simulate_batch(np.full((1, 2), -1, np.int8), np.array([0]), 4, policies=("brrip",))
