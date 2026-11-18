"""DRRIP reference policy, miss experiments and the vectorized miss simulator."""

from __future__ import annotations

import numpy as np
import pytest

from cpl.data.histories import draw_random
from cpl.policies import BRRIP, DRRIP, SRRIP, run_experiment, run_miss_experiment
from cpl.policies.fast_rrip import simulate_drrip_misses


def _ref(W, history, mode, p, seed=0):
    return run_miss_experiment(DRRIP(W, mode=mode, seed=seed, p=p), history)[0]


def test_srrip_mode_matches_srrip():
    rng = np.random.default_rng(0)
    for _ in range(300):
        h = rng.integers(0, 24, size=rng.integers(0, 40)).tolist()
        assert _ref(16, h, "srrip", 1 / 32) == run_miss_experiment(SRRIP(16), h)[0]


def test_brrip_mode_matches_brrip_distribution_free_cases():
    # p = 0 and p = 1 make BRRIP deterministic, so DRRIP must match BRRIP exactly
    rng = np.random.default_rng(1)
    for p in (0.0, 1.0):
        for _ in range(200):
            h = rng.integers(0, 24, size=rng.integers(0, 40)).tolist()
            assert _ref(16, h, "brrip", p) == run_miss_experiment(BRRIP(16, p=p), h)[0]


def test_bad_mode():
    with pytest.raises(ValueError):
        DRRIP(16, mode="lru")


def test_miss_experiment_hits_only_agrees_with_run_experiment():
    rng = np.random.default_rng(2)
    for _ in range(200):
        h = rng.integers(0, 8, size=rng.integers(0, 20)).tolist()
        assert run_miss_experiment(SRRIP(8), h)[0] == run_experiment(SRRIP(8), h)[0]


@pytest.mark.parametrize("W", [4, 8, 16])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_fast_matches_reference_deterministic(W, p):
    """SRRIP rows always, and BRRIP rows with p in {0, 1}, are deterministic: exact match."""
    rng = np.random.default_rng(W)
    n = 3000
    hist, lens = draw_random(W, n, (0, 48), rng, pool_range=(W, min(2 * W, 32)))
    brrip = rng.random(n) < 0.5
    fast = simulate_drrip_misses(hist, lens, W, brrip, np.random.default_rng(0), p=p).victim_way
    for i in range(n):
        h = hist[i, : lens[i]].tolist()
        mode = "brrip" if brrip[i] else "srrip"
        assert fast[i] == _ref(W, h, mode, p), (i, h, mode)


def test_fast_counts_misses():
    hist = np.array([[0, 16, 17, 16, -1]], np.int8)
    res = simulate_drrip_misses(hist, np.array([4]), 16, np.array([False]), np.random.default_rng(0))
    assert res.n_misses.tolist() == [2]


def test_fast_brrip_stochastic_distribution():
    """With p = 1/32 the fast and reference victim distributions agree statistically."""
    W, R = 8, 4000
    h = [0, 9, 3, 10, 1, 9, 12, 2]
    fast = simulate_drrip_misses(np.tile(np.array(h, np.int8), (R, 1)), np.full(R, len(h)), W,
                                 np.ones(R, bool), np.random.default_rng(3)).victim_way
    pol = DRRIP(W, mode="brrip", seed=4)
    ref = np.array([run_miss_experiment(pol, h)[0] for _ in range(R)])
    pf, pr = np.bincount(fast, minlength=W) / R, np.bincount(ref, minlength=W) / R
    assert np.abs(pf - pr).max() < 0.04, (pf, pr)


def test_pool_range_none_keeps_phase1_stream():
    a = draw_random(16, 100, (0, 32), np.random.default_rng(5))
    b = draw_random(16, 100, (0, 32), np.random.default_rng(5), pool_range=None)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    h, L = draw_random(16, 1000, (0, 32), np.random.default_rng(6), pool_range=(16, 24))
    assert h.max() <= 23 and (h[np.arange(32)[None, :] < L[:, None]] >= 0).all()
