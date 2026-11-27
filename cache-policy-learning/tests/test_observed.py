"""Observed-outcome prediction (cpl.observed, eval_sim --observed).

  1. if the revealed outcomes are the model's own free-running predictions, nothing is
     corrected and the run is identical to free-running;
  2. no leakage: the prediction for access t never depends on the outcome of t or later;
  3. the baselines' correction obeys the same two rules.
"""
from __future__ import annotations

import numpy as np
import torch

from cpl.eval_sim import lru, model_predict, tree_plru
from cpl.model.slot_cache import build_slot_model
from cpl.observed import ObservedChooser


def _streams(n=8, T=120, pool=24, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, pool, T).astype(np.int64) for _ in range(n)]


def _model():
    torch.manual_seed(0)
    return build_slot_model({"use_ages": True}).eval()


def test_own_outcomes_equal_free_running():
    m, acc = _model(), _streams()
    h_free, c_free = model_predict(m, acc, contents=True)
    st = {}
    h_obs, c_obs = model_predict(m, acc, contents=True, obs=h_free, stats=st)
    assert st["corrections"] == 0
    for a, b, x, y in zip(h_free, h_obs, c_free, c_obs):
        assert np.array_equal(a, b) and np.array_equal(x, y)


@torch.no_grad()
def test_no_leakage_model():
    m, acc = _model(), _streams(n=4)
    rng = np.random.default_rng(1)
    A = torch.from_numpy(np.stack(acc)); B, T = A.shape
    obs = rng.random((B, T)) < 0.5
    base = m(A, chooser=ObservedChooser(A.numpy(), obs, np.ones((B, T), bool), m.N))
    for t in (0, 17, 60, T - 1):
        o2 = obs.copy(); o2[:, t:] = ~o2[:, t:]            # change the outcome of t and later
        alt = m(A, chooser=ObservedChooser(A.numpy(), o2, np.ones((B, T), bool), m.N))
        assert torch.equal(base[:, :t + 1], alt[:, :t + 1])
        assert t == T - 1 or not torch.equal(base, alt)    # later outcomes are used


def test_baselines_observed():
    W = 16
    for a in _streams(n=6, seed=3):
        h, c = tree_plru(a, W)
        h2, c2 = tree_plru(a, W, obs=h)                    # consistent outcomes: identical
        assert np.array_equal(h, h2) and np.array_equal(c, c2)
        obs = np.random.default_rng(4).random(len(a)) < 0.5
        base, _ = lru(a, W, obs=obs)
        for t in (5, 40, 100):
            o2 = obs.copy(); o2[t:] = ~o2[t:]
            alt, _ = lru(a, W, obs=o2)
            assert np.array_equal(base[:t + 1], alt[:t + 1])


@torch.no_grad()
def test_observed_training_residency_matches_observed_eval():
    """--observed-train with no exploration drives residency exactly like observed-outcome
    evaluation (own choices + the same corrections), so training and test see the same
    states."""
    from cpl.train_hindsight import HindsightChooser
    m, acc = _model(), _streams(n=6)
    A = torch.from_numpy(np.stack(acc)); B, T = A.shape
    obs = np.random.default_rng(2).random((B, T)) < 0.6
    M = np.ones((B, T), bool)
    ref = m(A, chooser=ObservedChooser(A.numpy(), obs, M, m.N))
    hs = HindsightChooser(A, torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(M.astype(np.float32)),
                          explore=0.0, conf_weight=False, n_blocks=m.N, own_choice=True)
    assert torch.equal(m(A, chooser=hs), ref)
