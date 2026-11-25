"""The simulated ground-truth generator must record exactly what the cache did.

Checks (on a small generated dataset):
  1. bookkeeping: each access changes the contents exactly as hit/miss/evicted/
     filled_way say (hit -> unchanged; miss -> one way overwritten with the block).
  2. independent tree-PLRU: victims recomputed WITHOUT the simulator's bit logic,
     from per-way recency (each node points away from the more recently used half;
     empty ways filled lowest-index first), agree at every access.
  3. LRU: the evicted block is always the least recently used one.
  4. independent sets really are independent (different seeds -> different streams).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpl.data.sim_stream_dataset import build


def _load(tmp_path, policy, n=300, seed=3, W=16, pool=24):
    out = tmp_path / f"{policy}_{seed}"
    build(policy, W, pool, n, (32, 96), seed, False, 0.1, out)
    return pd.read_parquet(out / "samples.parquet"), W


@pytest.mark.parametrize("policy", ["plru", "lru", "fifo", "srrip", "bitplru"])
def test_bookkeeping_consistent(tmp_path, policy):
    df, W = _load(tmp_path, policy)
    for _, r in df.iterrows():
        acc = np.asarray(r["accesses"]); T = len(acc)
        C = np.asarray(r["contents"]).reshape(T, W)
        prev = np.full(W, -1)
        for t in range(T):
            b = acc[t]
            if r["p_hit"][t] == 1.0:
                assert b in prev and (C[t] == prev).all() and r["filled_way"][t] == -1
            else:
                assert b not in prev
                w = r["filled_way"][t]
                changed = np.nonzero(C[t] != prev)[0].tolist()
                assert changed == [w] and C[t][w] == b
                assert r["evicted"][t] == prev[w]            # -1 when the way was empty
                if (prev == -1).any():                       # empties filled lowest-index first
                    assert w == int(np.nonzero(prev == -1)[0][0])
            prev = C[t]
        assert len(np.asarray(r["policy_state"])) % T == 0


def test_tree_plru_matches_independent_recency_oracle(tmp_path):
    df, W = _load(tmp_path, "plru", n=400)
    for _, r in df.iterrows():
        acc = np.asarray(r["accesses"]); T = len(acc)
        ways = [-1] * W; last_use = [-1] * W               # last time each WAY was touched
        for t in range(T):
            b = int(acc[t])
            if b in ways:
                w = ways.index(b)
                assert r["p_hit"][t] == 1.0
            else:
                assert r["p_hit"][t] == 0.0
                if -1 in ways:
                    w = ways.index(-1)
                else:
                    lo, hi = 0, W
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if max(last_use[lo:mid]) > max(last_use[mid:hi]):
                            lo = mid
                        else:
                            hi = mid
                    w = lo
                assert r["filled_way"][t] == w
                assert r["evicted"][t] == ways[w]
                ways[w] = b
            last_use[w] = t


def test_lru_evicts_least_recently_used(tmp_path):
    df, W = _load(tmp_path, "lru")
    for _, r in df.iterrows():
        last = {}
        for t, b in enumerate(np.asarray(r["accesses"]).tolist()):
            ev = int(r["evicted"][t])
            if ev >= 0:
                resident = set(np.asarray(r["contents"]).reshape(-1, W)[t - 1].tolist())
                assert ev == min(resident, key=lambda x: last[x])
            last[b] = t


def test_independent_seeds_differ(tmp_path):
    a, _ = _load(tmp_path, "plru", n=50, seed=1)
    b, _ = _load(tmp_path, "plru", n=50, seed=2)
    assert sum(np.array_equal(x, y) for x, y in zip(a["accesses"], b["accesses"])) == 0
