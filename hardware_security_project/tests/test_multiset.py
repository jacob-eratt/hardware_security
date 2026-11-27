"""Multi-set simulators and the multi-set stream generator (architecture_spec.md §6).

  1. leader sets: complement select gives exactly n_leaders of each kind, one of each
     per constituency, disjoint.
  2. independence: in MultiSetCache every set behaves exactly like the single-set
     simulator run on that set's own accesses.
  3. DRRIP sub-policies against an independent reference written here (SRRIP / BRRIP
     with the 1-in-32 throttle counter): SRRIP leaders always behave as SRRIP, BRRIP
     leaders as BRRIP, followers as whichever side PSEL favours.
  4. PSEL trajectory recomputed from the recorded hits and set roles.
  5. generator bookkeeping per set, line-id layout, determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpl.data.sim_multiset_dataset import build, make_stream
from cpl.data.sim_stream_dataset import simulate_stream
from cpl.policies.multiset import (BRRIP_LEADER, FOLLOWER, SRRIP_LEADER, DRRIPCache, MultiSetCache,
                                   leader_roles)

W, S, POOL = 16, 1024, 24


# ---- independent reference: SRRIP-HP mechanics, insertion value from a callback ------
def ref_rrip(stream, insert_value):
    """Returns (hits, contents per step). insert_value() is called once per insertion."""
    ways, rrpv, hits, cont = [None] * W, [3] * W, [], []
    for b in stream:
        if b in ways:
            w = ways.index(b); rrpv[w] = 0; hits.append(True)
        else:
            hits.append(False)
            if None in ways:
                w = ways.index(None)
            else:
                while 3 not in rrpv:
                    rrpv = [v + 1 for v in rrpv]
                w = rrpv.index(3)
            ways[w] = b; rrpv[w] = insert_value()
        cont.append(list(ways))
    return hits, cont


def srrip_ref(stream):
    return ref_rrip(stream, lambda: 2)


def brrip_counter_ref(stream):
    n = [0]
    def ins():
        v = 2 if n[0] % 32 == 31 else 3
        n[0] += 1
        return v
    return ref_rrip(stream, ins)


def run_cache(cache, s, stream):
    hits, cont = [], []
    for b in stream:
        h, _, _ = cache.access(s, b); hits.append(h); cont.append(cache.resident(s))
    return hits, cont


# ---- 1 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("S_,n", [(1024, 32), (2048, 32), (512, 16), (64, 8)])
def test_leader_roles(S_, n):
    r = leader_roles(S_, n)
    assert (r == SRRIP_LEADER).sum() == n and (r == BRRIP_LEADER).sum() == n
    k = S_ // n
    for c in range(n):
        block = r[c * k:(c + 1) * k]
        assert (block == SRRIP_LEADER).sum() == 1 and (block == BRRIP_LEADER).sum() == 1


# ---- 2 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("policy", ["plru", "lru", "fifo", "srrip", "bitplru"])
def test_sets_are_independent(policy):
    rng = np.random.default_rng(7)
    for _ in range(40):
        lines, sets = make_stream(rng, S, POOL, (1, 4), (32, 96))
        cache = MultiSetCache(policy, S, W)
        got = {}
        for b, s in zip(lines.tolist(), sets.tolist()):
            h, _, _ = cache.access(s, b)
            got.setdefault(s, []).append(h)
        for s in got:
            sub = lines[sets == s].tolist()
            ref = simulate_stream(policy, W, sub)["p_hit"] > 0
            assert got[s] == ref.tolist()


# ---- 3 -----------------------------------------------------------------------------------
def _one_set_stream(rng, n=400):
    return rng.integers(0, POOL, n).tolist()


@pytest.mark.parametrize("psel", [0, 511, 512, 1023])
def test_drrip_leaders_ignore_psel(psel):
    roles = leader_roles(S, 32)
    s_lead, b_lead = int(np.nonzero(roles == SRRIP_LEADER)[0][3]), int(np.nonzero(roles == BRRIP_LEADER)[0][5])
    rng = np.random.default_rng(psel)
    st = _one_set_stream(rng)
    assert run_cache(DRRIPCache(S, W, psel_init=psel), s_lead, st) == srrip_ref(st)
    assert run_cache(DRRIPCache(S, W, psel_init=psel), b_lead, st) == brrip_counter_ref(st)


@pytest.mark.parametrize("psel,ref", [(0, srrip_ref), (511, srrip_ref), (512, brrip_counter_ref), (1023, brrip_counter_ref)])
def test_drrip_follower_follows_psel(psel, ref):
    roles = leader_roles(S, 32)
    f = int(np.nonzero(roles == FOLLOWER)[0][100])
    st = _one_set_stream(np.random.default_rng(psel + 1))
    cache = DRRIPCache(S, W, psel_init=psel)
    assert run_cache(cache, f, st) == ref(st)
    assert cache.psel == psel                       # follower misses never move PSEL


class _RefSet:
    """Step-wise SRRIP-mechanics set for the shared-counter reference."""
    def __init__(self):
        self.ways, self.rrpv = [None] * W, [3] * W

    def access(self, b, insert_value):
        if b in self.ways:
            self.rrpv[self.ways.index(b)] = 0; return True
        if None in self.ways:
            w = self.ways.index(None)
        else:
            while 3 not in self.rrpv:
                self.rrpv = [v + 1 for v in self.rrpv]
            w = self.rrpv.index(3)
        self.ways[w] = b; self.rrpv[w] = insert_value(); return False


def test_bimodal_counter_is_global_across_sets():
    """Two BRRIP-mode followers interleaved must match a reference with ONE shared
    throttle counter (and must differ from two private counters)."""
    roles = leader_roles(S, 32)
    f1, f2 = [int(x) for x in np.nonzero(roles == FOLLOWER)[0][:2]]
    rng = np.random.default_rng(3)
    acc = [(f1 if rng.random() < 0.5 else f2, int(rng.integers(0, POOL))) for _ in range(1500)]
    cache = DRRIPCache(S, W, psel_init=1023)
    got = [cache.access(s, b)[0] for s, b in acc]

    def ref(shared):
        sets = {f1: _RefSet(), f2: _RefSet()}; n = {f1: 0, f2: 0, "all": 0}
        def ins_for(s):
            def ins():
                key = "all" if shared else s
                v = 2 if n[key] % 32 == 31 else 3; n[key] += 1; return v
            return ins
        return [sets[s].access(b, ins_for(s)) for s, b in acc]

    assert got == ref(shared=True)
    assert got != ref(shared=False)                 # the test can tell the two apart


# ---- 4, 5: generator ---------------------------------------------------------------------
def _load(tmp_path, policy, n=200, seed=5, **kw):
    out = tmp_path / f"{policy}_{seed}_{'_'.join(map(str, kw.values()))}"
    meta = build(policy, S, W, POOL, n, (1, 4), (32, 96), seed, False, 0.1, out, **kw)
    return pd.read_parquet(out / "samples.parquet"), meta


@pytest.mark.parametrize("psel_init", ["mid", "uniform"])
def test_psel_trajectory(tmp_path, psel_init):
    df, meta = _load(tmp_path, "drrip", psel_init=psel_init)
    for _, r in df.iterrows():
        role, hit, psel = np.asarray(r["set_role"]), np.asarray(r["p_hit"]) > 0, np.asarray(r["psel"])
        p = psel[0] - (1 if (not hit[0] and role[0] == SRRIP_LEADER) else 0) + (1 if (not hit[0] and role[0] == BRRIP_LEADER) else 0)
        if psel_init == "mid":
            assert p == 512
        for t in range(len(hit)):
            if not hit[t] and role[t] == SRRIP_LEADER:
                p = min(p + 1, 1023)
            elif not hit[t] and role[t] == BRRIP_LEADER:
                p = max(p - 1, 0)
            assert psel[t] == p


def test_follower_insertion_matches_mode(tmp_path):
    """On a follower miss, the inserted way's RRPV is 2 under SRRIP mode, and 3 (or the
    throttled 2) under BRRIP mode; mode read from PSEL before the access."""
    df, _ = _load(tmp_path, "drrip", n=300, psel_init="mid")
    n_brrip = n_srrip = 0
    for _, r in df.iterrows():
        T = len(r["accesses"]); st = np.asarray(r["policy_state"]).reshape(T, W)
        psel, role, fw = np.asarray(r["psel"]), np.asarray(r["set_role"]), np.asarray(r["filled_way"])
        prev = 512
        for t in range(T):
            if role[t] == FOLLOWER and fw[t] >= 0:
                v = st[t][fw[t]]
                if prev >= 512:
                    assert v in (2, 3); n_brrip += 1
                else:
                    assert v == 2; n_srrip += 1
            prev = psel[t]
    assert n_brrip > 0 and n_srrip > 0                      # both modes exercised


@pytest.mark.parametrize("policy", ["drrip", "plru", "lru"])
def test_generator_bookkeeping(tmp_path, policy):
    df, meta = _load(tmp_path, policy)
    assert meta["max_line_id"] < 4 * POOL
    for _, r in df.iterrows():
        acc, sets = np.asarray(r["accesses"]), np.asarray(r["sets"]); T = len(acc)
        C = np.asarray(r["contents"]).reshape(T, W)
        assert len(np.unique(sets)) == r["n_sets"]
        # line-id layout: each set owns one block of POOL ids, slots in order of first touch
        slot_of_set = {}
        for t in range(T):
            slot_of_set.setdefault(int(sets[t]), len(slot_of_set))
            assert acc[t] // POOL == slot_of_set[int(sets[t])]
        prev = {}
        for t in range(T):
            s, b = int(sets[t]), int(acc[t]); pv = prev.get(s, np.full(W, -1))
            if r["p_hit"][t] == 1.0:
                assert b in pv and (C[t] == pv).all() and r["filled_way"][t] == -1
            else:
                assert b not in pv
                w = r["filled_way"][t]
                assert np.nonzero(C[t] != pv)[0].tolist() == [w] and C[t][w] == b
                assert r["evicted"][t] == pv[w]
                if (pv == -1).any():
                    assert w == int(np.nonzero(pv == -1)[0][0])
            prev[s] = C[t]
        if policy != "drrip":
            assert (np.asarray(r["set_role"]) == 0).all() and (np.asarray(r["psel"]) == -1).all()


def test_generator_deterministic(tmp_path):
    a, _ = _load(tmp_path / "a", "drrip", n=50, seed=9)
    b, _ = _load(tmp_path / "b", "drrip", n=50, seed=9)
    c, _ = _load(tmp_path / "c", "drrip", n=50, seed=10)
    same = lambda x, y: all((np.asarray(p) == np.asarray(q)).all() if len(p) == len(q) else False
                            for p, q in zip(x["accesses"], y["accesses"]))
    assert same(a, b) and not same(a, c)
    assert all((np.asarray(p) == np.asarray(q)).all() for p, q in zip(a["p_hit"], b["p_hit"]))


def test_leaders_appear(tmp_path):
    _, meta = _load(tmp_path, "drrip", n=600)
    sh = meta["access_share_by_role"]
    assert sh["srrip_leader"] > 0 and sh["brrip_leader"] > 0
