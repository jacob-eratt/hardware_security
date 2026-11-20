"""Measure per-access hit/miss streams on real hardware -> a black-box cache dataset.

The task: given a sequence of accesses that all map to one cache set, predict
whether each access hits (served by this core's L1/L2) or misses (goes to DRAM).
We measure the ground-truth hit/miss vector on the Celeron by timing every access
(fast = hit, slow = miss), repeated R times to get a per-access hit probability.
No policy is assumed or simulated -- the goal is a model that reproduces the real
cache's hit/miss behavior, judged purely by predictive accuracy on held-out data.

    # training/val/test (hash-split) set:
    python -m cpl.data.hw_stream_dataset --config configs/hw_l2_celeron.yaml \
        --pool 24 --n 20000 --out data/processed/hw_l2_celeron_stream
    # a fresh, independent test set (different seed):
    python -m cpl.data.hw_stream_dataset --config configs/hw_l2_celeron.yaml \
        --pool 24 --n 4000 --seed 777 --all-test --out data/processed/hw_l2_celeron_stream_test

Output ``samples.parquet`` columns: ``accesses`` (block ids into the congruent
pool), ``p_hit`` (measured hit probability per access), ``hit_counts``, ``n_repeats``,
``hist_len``, ``split``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import cpl
from cpl.data.generate import _code_hash, load_config
from cpl.data.hw_measure import ensure_engine

log = logging.getLogger("cpl.hw_stream_dataset")


def gen_streams(n: int, pool: int, len_range: tuple[int, int],
                rng: np.random.Generator) -> list[list[int]]:
    lo, hi = len_range
    out = []
    for _ in range(n):
        L = int(rng.integers(lo, hi + 1))
        out.append(rng.integers(0, pool, size=L).tolist())
    return out


def split_of_stream(accesses: list[int], all_test: bool) -> str:
    if all_test:
        return "test"
    u = int(hashlib.sha1(",".join(map(str, accesses)).encode()).hexdigest()[:8], 16) / 16**8
    return "train" if u < 0.90 else ("val" if u < 0.95 else "test")


def run_engine_stream(exe: Path, cfg: dict[str, Any], pool: int, hist_file: Path, res_file: Path) -> None:
    hw = cfg["hw"]
    cmd = [
        str(exe), "--stream",
        "--core", str(hw.get("core", 0)),
        "--ways", str(cfg["W"]),
        "--pool", str(pool),
        "--repeats", str(hw.get("repeats", 16)),
        "--buf-mb", str(hw.get("buf_mb", 256)),
        "--in", str(hist_file), "--out", str(res_file),
    ]
    if hw.get("threshold"):
        cmd += ["--threshold", str(int(hw["threshold"]))]
    if hw.get("min_detect") is not None:
        cmd += ["--min-detect", str(float(hw["min_detect"]))]
    if hw.get("force"):
        cmd += ["--force"]
    log.info("engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build(cfg: dict[str, Any], pool: int, n: int, len_range: tuple[int, int],
          out_dir: Path, seed: int | None = None, all_test: bool = False) -> dict:
    t0 = time.perf_counter()
    W = int(cfg["W"])
    if pool <= W:
        raise ValueError(f"--pool must exceed W={W} so the stream can miss")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = ensure_engine()
    rng = np.random.default_rng(seed if seed is not None else int(cfg.get("seed", 1234)))

    streams = gen_streams(n, pool, len_range, rng)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    hist_file, res_file = tmp / "streams.txt", tmp / "res.txt"
    hist_file.write_text(f"{n}\n" + "\n".join(
        f"{len(s)} " + " ".join(map(str, s)) for s in streams) + "\n", encoding="utf-8")
    run_engine_stream(exe, cfg, pool, hist_file, res_file)

    R = int(cfg["hw"].get("repeats", 16))
    rows = [ln.split() for ln in res_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != n:
        raise RuntimeError(f"engine returned {len(rows)} streams, expected {n}")

    width = max(len(s) for s in streams)
    acc = np.full((n, width), -1, np.int8)
    hitc = np.zeros((n, width), np.int16)
    lens = np.zeros(n, np.int16)
    splits = []
    for i, (s, r) in enumerate(zip(streams, rows)):
        L = len(s)
        acc[i, :L] = s
        hitc[i, :L] = [int(x) for x in r][:L]
        lens[i] = L
        splits.append(split_of_stream(s, all_test))

    def _list(mat, dtype):
        valid = np.arange(width)[None, :] < lens[:, None]
        offs = pa.array(np.concatenate([[0], np.cumsum(lens, dtype=np.int64)]).astype(np.int32))
        return pa.ListArray.from_arrays(offs, pa.array(mat[valid].astype(dtype[0]), type=dtype[1]))

    p_hit = (hitc.astype(np.float32) / max(R, 1))
    target = cfg.get("target", "hw_stream")
    table = pa.table({
        "stream_id": pa.array([f"{target}_{i:08d}" for i in range(n)], pa.string()),
        "source": pa.array(["hw"] * n, pa.string()),
        "W": pa.array(np.full(n, W, np.int8)),
        "pool": pa.array(np.full(n, pool, np.int16)),
        "accesses": _list(acc, (np.int8, pa.int8())),
        "hit_counts": _list(hitc, (np.int16, pa.int16())),
        "p_hit": _list(p_hit, (np.float32, pa.float32())),
        "hist_len": pa.array(lens),
        "n_repeats": pa.array(np.full(n, R, np.int16)),
        "split": pa.array(splits, pa.string()),
    })
    pq.write_table(table, out_dir / "samples.parquet")

    sp = np.array(splits)
    overall_hit = float(p_hit[np.arange(width)[None, :] < lens[:, None]].mean())
    per_split = {s: int((sp == s).sum()) for s in ("train", "val", "test") if (sp == s).any()}
    meta = {
        "generator": "cpl.data.hw_stream_dataset", "code_version": cpl.__version__,
        "code_hash": _code_hash(), "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "hw", "W": W, "pool": pool, "repeats": R, "len_range": list(len_range),
        "seed": seed, "all_test": all_test, "target": target, "n": n,
        "splits": per_split, "overall_hit_rate": round(overall_hit, 4),
        "mean_access_consistency": round(float(
            (np.abs(p_hit - (p_hit > 0.5))[np.arange(width)[None, :] < lens[:, None]] < 0.25).mean()), 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d streams, %d accesses) splits=%s hit_rate=%.3f",
             out_dir / "samples.parquet", n, int(lens.sum()), per_split, overall_hit)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--n", type=int, default=20000, help="number of access streams")
    ap.add_argument("--len-min", type=int, default=32)
    ap.add_argument("--len-max", type=int, default=96)
    ap.add_argument("--seed", type=int, default=None, help="override seed (use a fresh one for the test set)")
    ap.add_argument("--all-test", action="store_true", help="put every stream in the 'test' split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build(load_config(args.config), args.pool, args.n, (args.len_min, args.len_max),
          Path(args.out), args.seed, args.all_test)


if __name__ == "__main__":
    main()
