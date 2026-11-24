"""Simulated per-access stream dataset with FULL ground truth (no hardware).

Same stream shape as the hardware generator (cpl.data.hw_stream_dataset): each
stream is a sequence of accesses to block ids 0..pool-1 that all map to one W-way
set, starting from an EMPTY set (a known state). Streams are replayed through a
reference simulator from cpl.policies, and every access records:

  accesses     block id accessed                                   (model input)
  p_hit        1.0 hit / 0.0 miss                                  (training label;
               the same column name/semantics as the hardware data)
  evicted      block evicted by this access, -1 if none            (answer key)
  filled_way   way the block was placed in on a miss, -1 on a hit  (answer key)
  contents     block in each way AFTER the access, -1 = empty;     (answer key)
               flattened row-major [T * W]
  policy_state the policy's internal state AFTER the access,       (answer key)
               flattened [T * S] (e.g. the W-1 tree-PLRU bits)
  split        train / val / test

Models must only ever train on ``accesses`` and ``p_hit`` (hardware-equivalent);
the other columns are for grading and diagnostics.

    python -m cpl.data.sim_stream_dataset --policy plru --n 100000 --seed 1 \
        --val-frac 0.05 --out data/processed/sim_plru_stream
    python -m cpl.data.sim_stream_dataset --policy plru --n 8000 --seed 2 --all-test \
        --out data/processed/sim_plru_stream_test
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
from cpl.policies import make_policy


def simulate_stream(policy_name: str, W: int, stream: list[int]) -> dict[str, np.ndarray]:
    """Replay one stream from an empty set; return per-access ground truth."""
    p = make_policy(policy_name, W)
    T = len(stream)
    hit = np.zeros(T, np.float32)
    evicted = np.full(T, -1, np.int16)
    filled = np.full(T, -1, np.int8)
    contents = np.full((T, W), -1, np.int16)
    states = []
    for t, b in enumerate(stream):
        before = p.resident()
        if b in before:
            hit[t] = 1.0
        ev = p.access(b)
        after = p.resident()
        if not hit[t]:
            filled[t] = after.index(b)
        if ev is not None:
            evicted[t] = ev
        contents[t] = [-1 if x is None else x for x in after]
        states.append(np.asarray(p.state(), np.int8).ravel())
    return {"p_hit": hit, "evicted": evicted, "filled_way": filled,
            "contents": contents.ravel(), "policy_state": np.concatenate(states)}


def build(policy: str, W: int, pool: int, n: int, len_range: tuple[int, int], seed: int,
          all_test: bool, val_frac: float, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lo, hi = len_range
    streams = [rng.integers(0, pool, int(rng.integers(lo, hi + 1))).tolist() for _ in range(n)]
    if all_test:
        split = np.array(["test"] * n)
    else:
        split = np.where(rng.random(n) < val_frac, "val", "train")
    cols = {k: [] for k in ("p_hit", "evicted", "filled_way", "contents", "policy_state")}
    for s in streams:
        g = simulate_stream(policy, W, s)
        for k in cols:
            cols[k].append(g[k])
    state_dim = len(cols["policy_state"][0]) // len(streams[0])
    table = pa.table({
        "accesses": pa.array([np.asarray(s, np.int64) for s in streams]),
        "p_hit": pa.array(cols["p_hit"]),
        "evicted": pa.array(cols["evicted"]),
        "filled_way": pa.array(cols["filled_way"]),
        "contents": pa.array(cols["contents"]),
        "policy_state": pa.array(cols["policy_state"]),
        "hist_len": pa.array(np.array([len(s) for s in streams], np.int16)),
        "split": pa.array(split),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")
    hits = np.concatenate(cols["p_hit"])
    meta = {
        "generator": "cpl.data.sim_stream_dataset", "code_version": cpl.__version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "sim", "policy": policy, "W": W, "pool": pool, "n": n,
        "len_range": list(len_range), "seed": seed, "all_test": all_test,
        "start_state": "empty", "state_dim": state_dim,
        "splits": {s: int((split == s).sum()) for s in ("train", "val", "test") if (split == s).any()},
        "overall_hit_rate": round(float(hits.mean()), 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="cpl.policies name, e.g. plru, lru, fifo, srrip")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--len-min", type=int, default=32)
    ap.add_argument("--len-max", type=int, default=96)
    ap.add_argument("--seed", type=int, required=True, help="use a different seed for each independent set")
    ap.add_argument("--all-test", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = build(a.policy, a.W, a.pool, a.n, (a.len_min, a.len_max), a.seed, a.all_test, a.val_frac, Path(a.out))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
