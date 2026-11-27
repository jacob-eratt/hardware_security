"""Simulated MULTI-SET stream dataset with full ground truth (architecture_spec.md §6, §8.1).

The multi-set counterpart of ``cpl.data.sim_stream_dataset``. Each stream touches k
sets (k uniform in [sets_min, sets_max]), sampled uniformly without replacement from
all S sets of the simulated cache (so leader sets appear at their natural rate, as on
hardware). Every touched set gets its own pool of ``pool`` congruent lines and its own
sub-stream of length uniform in [len_min, len_max] of uniform draws from that pool,
exactly like one single-set stream. The k sub-streams are then interleaved in a random
order that keeps each sub-stream's own order. All sets start empty (as after the
hardware engine's flush); for DRRIP, PSEL starts at ``psel_init`` (see below).

Line ids are stream-local: the j-th set touched by a stream (local set slot j, in order
of first appearance) owns lines j*pool .. j*pool+pool-1.

Columns (per stream; arrays are per access):
  accesses     line id                                             (model input)
  sets         global set index 0..S-1 of the access               (model input: stands
               in for the set decoder's output until Unit 1 exists)
  p_hit        1.0 hit / 0.0 miss                                  (training label)
  set_role     0 follower, 1 SRRIP leader, 2 BRRIP leader          (answer key; always 0
               for non-DRRIP policies)
  evicted      line evicted by this access, -1 if none             (answer key)
  filled_way   way the line was placed in on a miss, -1 on a hit   (answer key)
  contents     line in each way of the ACCESSED set after the      (answer key)
               access, -1 = empty; flattened [T * W]
  policy_state accessed set's policy state after the access,       (answer key)
               flattened [T * state_dim]
  psel         DRRIP PSEL after the access (-1 for other policies)  (answer key)
  n_sets, hist_len, split

Models must only ever train on ``accesses``, ``sets`` and ``p_hit``.

``--psel-init``: "uniform" (default) draws PSEL for each stream uniformly from its full
range, because on hardware the global state at the start of a measurement is unknown;
"mid" starts every stream at the threshold, so dueling flips the follower mode within
streams; an integer fixes it.

    python -m cpl.data.sim_multiset_dataset --policy drrip --n 100000 --seed 1 \
        --val-frac 0.05 --out data/processed/sim_drrip_ms
    python -m cpl.data.sim_multiset_dataset --policy drrip --n 8000 --seed 2 --all-test \
        --out data/processed/sim_drrip_ms_test
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import cpl
from cpl.policies.multiset import DRRIPCache, make_multiset


def make_stream(rng: np.random.Generator, S: int, pool: int, sets_range: tuple[int, int],
                len_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """One stream: (line ids [T], global set index [T])."""
    k = int(rng.integers(sets_range[0], sets_range[1] + 1))
    chosen = rng.choice(S, size=k, replace=False)
    lens = rng.integers(len_range[0], len_range[1] + 1, size=k)
    subs = [rng.integers(0, pool, int(n)) for n in lens]
    order = rng.permutation(np.repeat(np.arange(k), lens))        # interleave, keep sub-orders
    ptr = np.zeros(k, np.int64)
    lines = np.empty(len(order), np.int64); sets = np.empty(len(order), np.int64)
    first_seen: dict[int, int] = {}
    for t, j in enumerate(order):
        slot = first_seen.setdefault(int(j), len(first_seen))       # local slot = order of first touch
        lines[t] = slot * pool + subs[j][ptr[j]]; ptr[j] += 1
        sets[t] = chosen[j]
    return lines, sets


def simulate(cache, lines: np.ndarray, sets: np.ndarray, W: int) -> dict[str, np.ndarray]:
    T = len(lines)
    out = {"p_hit": np.zeros(T, np.float32), "evicted": np.full(T, -1, np.int16),
           "filled_way": np.full(T, -1, np.int8), "contents": np.full((T, W), -1, np.int16),
           "set_role": np.zeros(T, np.int8), "psel": np.full(T, -1, np.int16)}
    states = []
    for t in range(T):
        s, b = int(sets[t]), int(lines[t])
        hit, ev, w = cache.access(s, b)
        out["p_hit"][t] = float(hit)
        if not hit:
            out["filled_way"][t] = w
        if ev is not None:
            out["evicted"][t] = ev
        out["contents"][t] = [-1 if x is None else x for x in cache.resident(s)]
        out["set_role"][t] = cache.role(s)
        if isinstance(cache, DRRIPCache):
            out["psel"][t] = cache.psel
        states.append(np.asarray(cache.state(s), np.int8).ravel())
    out["contents"] = out["contents"].ravel()
    out["policy_state"] = np.concatenate(states)
    return out


def build(policy: str, S: int, W: int, pool: int, n: int, sets_range: tuple[int, int],
          len_range: tuple[int, int], seed: int, all_test: bool, val_frac: float, out_dir: Path,
          psel_init: str = "uniform", bimodal: str = "counter", n_leaders: int = 32,
          psel_bits: int = 10) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    split = np.array(["test"] * n) if all_test else np.where(rng.random(n) < val_frac, "val", "train")
    keys = ("p_hit", "evicted", "filled_way", "contents", "policy_state", "set_role", "psel")
    cols = {k: [] for k in ("accesses", "sets", *keys)}
    psel_max = (1 << psel_bits) - 1
    for i in range(n):
        lines, sets = make_stream(rng, S, pool, sets_range, len_range)
        if policy == "drrip":
            if psel_init == "uniform":
                p0 = int(rng.integers(0, psel_max + 1))
            elif psel_init == "mid":
                p0 = 1 << (psel_bits - 1)
            else:
                p0 = int(psel_init)
            cache = make_multiset("drrip", S, W, seed=seed * 1_000_003 + i, n_leaders=n_leaders,
                                  psel_bits=psel_bits, psel_init=p0, bimodal=bimodal)
        else:
            cache = make_multiset(policy, S, W, seed=seed * 1_000_003 + i)
        g = simulate(cache, lines, sets, W)
        cols["accesses"].append(lines); cols["sets"].append(sets.astype(np.int16))
        for k in keys:
            cols[k].append(g[k])
    T0 = len(cols["accesses"][0])
    state_dim = len(cols["policy_state"][0]) // T0
    table = pa.table({
        **{k: pa.array(cols[k]) for k in ("accesses", "sets", *keys)},
        "n_sets": pa.array(np.array([len(np.unique(s)) for s in cols["sets"]], np.int8)),
        "hist_len": pa.array(np.array([len(a) for a in cols["accesses"]], np.int16)),
        "split": pa.array(split),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")
    hits = np.concatenate(cols["p_hit"]); roles = np.concatenate(cols["set_role"])
    meta = {
        "generator": "cpl.data.sim_multiset_dataset", "code_version": cpl.__version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "sim", "policy": policy, "S": S, "W": W, "pool": pool, "n": n,
        "sets_range": list(sets_range), "len_range_per_set": list(len_range), "seed": seed,
        "all_test": all_test, "start_state": "empty", "state_dim": state_dim,
        "max_line_id": int(max(a.max() for a in cols["accesses"])),
        "splits": {s: int((split == s).sum()) for s in ("train", "val", "test") if (split == s).any()},
        "overall_hit_rate": round(float(hits.mean()), 4),
        "access_share_by_role": {r: round(float((roles == i).mean()), 4)
                                 for i, r in enumerate(("follower", "srrip_leader", "brrip_leader"))},
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if policy == "drrip":
        meta["drrip"] = {"n_leaders": n_leaders, "psel_bits": psel_bits, "psel_init": psel_init,
                         "bimodal": bimodal, "leader_rule": "complement select",
                         "psel_rule": "SRRIP-leader miss +1, BRRIP-leader miss -1, followers BRRIP iff PSEL >= 2**(bits-1)"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="drrip, or any cpl.policies single-set name")
    ap.add_argument("--S", type=int, default=1024, help="sets in the simulated cache")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--pool", type=int, default=24, help="congruent lines per touched set")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sets-min", type=int, default=1)
    ap.add_argument("--sets-max", type=int, default=4)
    ap.add_argument("--len-min", type=int, default=32, help="per-set sub-stream length")
    ap.add_argument("--len-max", type=int, default=96)
    ap.add_argument("--psel-init", default="uniform", help="uniform | mid | <int>")
    ap.add_argument("--bimodal", default="counter", choices=["counter", "rng"])
    ap.add_argument("--leaders", type=int, default=32, help="leader sets per sub-policy")
    ap.add_argument("--psel-bits", type=int, default=10)
    ap.add_argument("--seed", type=int, required=True, help="use a different seed for each independent set")
    ap.add_argument("--all-test", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = build(a.policy, a.S, a.W, a.pool, a.n, (a.sets_min, a.sets_max), (a.len_min, a.len_max),
                 a.seed, a.all_test, a.val_frac, Path(a.out), a.psel_init, a.bimodal, a.leaders, a.psel_bits)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
