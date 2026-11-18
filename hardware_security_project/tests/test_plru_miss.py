"""Vectorized tree-PLRU miss simulator vs the reference TreePLRU."""

from __future__ import annotations

import numpy as np
import pytest

from cpl.data.histories import draw_random
from cpl.policies import TreePLRU, run_miss_experiment
from cpl.policies.fast import simulate_batch
from cpl.policies.fast_plru_miss import simulate_plru_misses


@pytest.mark.parametrize("W", [4, 8, 16])
def test_matches_reference_with_misses(W):
    rng = np.random.default_rng(W)
    n = 3000
    hist, lens = draw_random(W, n, (0, 48), rng, pool_range=(W, min(2 * W, 32)))
    res = simulate_plru_misses(hist, lens, W)
    for i in range(n):
        h = hist[i, : lens[i]].tolist()
        pol = TreePLRU(W)
        way, _ = run_miss_experiment(pol, h)
        assert res.victim_way[i] == way, (i, h)


@pytest.mark.parametrize("W", [4, 16])
def test_hits_only_matches_phase1_simulator(W):
    hist, lens = draw_random(W, 2000, (0, 32), np.random.default_rng(7))
    res = simulate_plru_misses(hist, lens, W)
    ref = simulate_batch(hist, lens, W, policies=("plru",))
    assert (res.victim_way == ref.victims["plru"]).all()
    assert (res.bits == ref.plru_bits).all()
    assert (res.n_misses == 0).all()


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        simulate_plru_misses(np.zeros((1, 1), np.int8), np.array([1]), 12)
