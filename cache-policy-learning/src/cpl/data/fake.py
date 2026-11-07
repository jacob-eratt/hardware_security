"""Placeholder dataset generator so the GRU pipeline can be developed before steps 1-2 land.

Writes ``samples.parquet`` + ``meta.json`` in the §5 schema (``meta.json`` has ``"fake": true``).
Label rules:

- ``plru``   : stand-in PLRU labels from the recency rule of spec §3.6 (no TreePLRU dependency)
- ``lru``    : least-recently-used block (easy, learnable sanity target)
- ``random`` : uniform random labels (pipeline plumbing only; model should stay at chance)

Replace with ``python -m cpl.data.generate`` once the real simulator and generator exist.

    python -m cpl.data.fake --config configs/sim_plru_w16.yaml --n 120000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cpl.data.tokens import W_MAX
from cpl.utils.seed import named_rng

HW_NULL_COLUMNS = ("cache_level", "set_index", "slice", "set_role", "forced_mode", "run_id")


def history_id(W: int, history: Sequence[int]) -> str:
    """Deterministic 16-hex-digit ID of a history (spec §4.5)."""
    return hashlib.sha1((f"W{W}:" + ",".join(map(str, history))).encode()).hexdigest()[:16]


def split_of(hid: str) -> str:
    """Hash-based train/val/test assignment (spec §4.5)."""
    u = int(hid, 16) / 16**16
    return "train" if u < 0.90 else ("val" if u < 0.95 else "test")


def _last_access(W: int, history: Sequence[int]) -> list[int]:
    last = list(range(W))
    for t, b in enumerate(history):
        last[b] = W + t
    return last


def victim_plru_recency(W: int, history: Sequence[int]) -> int:
    """Hits-only tree-PLRU victim via the recency rule of spec §3.6."""
    last = _last_access(W, history)
    lo, hi = 0, W
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if max(last[lo:mid]) > max(last[mid:hi]):
            lo = mid
        else:
            hi = mid
    return lo


def victim_lru(W: int, history: Sequence[int]) -> int:
    """Hits-only LRU victim (fill counts as an access)."""
    last = _last_access(W, history)
    return int(np.argmin(last))


def _rules(rng: np.random.Generator) -> dict[str, Callable[[int, Sequence[int]], int]]:
    return {
        "plru": victim_plru_recency,
        "lru": victim_lru,
        "random": lambda W, h: int(rng.integers(W)),
    }


def random_histories(n: int, W: int, len_range: tuple[int, int], rng: np.random.Generator) -> list[list[int]]:
    """``n`` histories with L ~ U[lo, hi] and elements ~ U[0, W)."""
    lens = rng.integers(len_range[0], len_range[1] + 1, size=n)
    return [rng.integers(0, W, size=int(L)).tolist() for L in lens]


def _dedup(rows: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
    seen: set[str] = set()
    out = []
    for hid, h in rows:
        if len(h) > 3 and hid in seen:
            continue
        seen.add(hid)
        out.append((hid, h))
    return out


def build_fake(
    W: int,
    target: str,
    seed: int,
    n_main: int,
    n_long: int,
    label_rule: str = "plru",
    main_range: tuple[int, int] = (0, 32),
    long_range: tuple[int, int] = (33, 128),
) -> tuple[pa.Table, dict]:
    """Build the fake samples table and its meta dict."""
    if label_rule not in ("plru", "lru", "random"):
        raise ValueError(f"unknown label rule {label_rule}")
    if label_rule == "plru" and W & (W - 1):
        raise ValueError("plru rule needs power-of-two W")
    label_fn = _rules(named_rng(seed, "fake_labels"))[label_rule]
    rows: list[tuple[str, str, list[int]]] = []
    main = [(history_id(W, h), h) for h in random_histories(n_main, W, main_range, named_rng(seed, "fake_main"))]
    by_split: dict[str, list[tuple[str, list[int]]]] = {"train": [], "val": [], "test": []}
    for hid, h in main:
        by_split[split_of(hid)].append((hid, h))
    train_ids = {hid for hid, _ in by_split["train"]}
    for split, items in by_split.items():
        rows += [(split, hid, h) for hid, h in _dedup(items)]
    if n_long:
        long = [(history_id(W, h), h) for h in random_histories(n_long, W, long_range, named_rng(seed, "fake_test_long"))]
        rows += [("test_long", hid, h) for hid, h in _dedup(long) if hid not in train_ids]
    n = len(rows)
    labels = np.array([label_fn(W, h) for _, _, h in rows], dtype=np.int8)
    counts = np.zeros((n, W_MAX), dtype=np.int32)
    counts[np.arange(n), labels] = 1
    cols: dict[str, pa.Array] = {
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n)], pa.string()),
        "target": pa.array([target] * n, pa.string()),
        "source": pa.array(["sim"] * n, pa.string()),
        "policy": pa.array([label_rule] * n, pa.string()),
        "W": pa.array(np.full(n, W, np.int8)),
    }
    for c in HW_NULL_COLUMNS:
        cols[c] = pa.nulls(n)
    cols |= {
        "history_id": pa.array([hid for _, hid, _ in rows], pa.string()),
        "history": pa.array([h for _, _, h in rows], pa.list_(pa.int8())),
        "hist_len": pa.array(np.array([len(h) for _, _, h in rows], np.int16)),
        "gen_type": pa.array(["random"] * n, pa.string()),
        "victim_counts": pa.array(counts.tolist(), pa.list_(pa.int32(), W_MAX)),
        "n_valid_repeats": pa.array(np.ones(n, np.int16)),
        "label_hard": pa.array(labels),
        "label_entropy": pa.array(np.zeros(n, np.float32)),
        "sim_victims": pa.nulls(n, pa.struct([(p, pa.int8()) for p in ("plru", "lru", "fifo", "srrip")])),
        "plru_bits": pa.nulls(n, pa.list_(pa.int8(), W_MAX - 1)),
        "split": pa.array([s for s, _, _ in rows], pa.string()),
    }
    table = pa.table(cols)
    split_counts = {str(s): int(c) for s, c in zip(*np.unique([r[0] for r in rows], return_counts=True))}
    meta = {
        "fake": True,
        "label_rule": label_rule,
        "target": target,
        "W": W,
        "seed": seed,
        "n_main": n_main,
        "n_long": n_long,
        "len_ranges": {"main": list(main_range), "test_long": list(long_range)},
        "counts": split_counts,
        "victim_distribution": {
            s: np.bincount(labels[[r[0] == s for r in rows]], minlength=W).tolist() for s in split_counts
        },
    }
    return table, meta


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="default: data/fake/<target>")
    ap.add_argument("--n", type=int, default=120_000, help="histories with L in train range (split ~90/5/5)")
    ap.add_argument("--n-long", type=int, default=5_000, help="test_long histories")
    ap.add_argument("--label-rule", choices=["plru", "lru", "random"], default="plru")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    W, target, seed = int(cfg["W"]), cfg["target"], int(cfg["seed"])
    ranges = cfg.get("data", {}).get("len_ranges", {})
    out = Path(args.out_dir or f"data/fake/{target}")
    t0 = time.time()
    table, meta = build_fake(
        W, target, seed, args.n, args.n_long, args.label_rule,
        tuple(ranges.get("train", (0, 32))), tuple(ranges.get("test_long", (33, 128))),
    )
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "samples.parquet")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fake] wrote {table.num_rows} samples to {out} in {time.time() - t0:.1f}s  counts={meta['counts']}")
    for s, dist in meta["victim_distribution"].items():
        print(f"[fake]   {s:<10} victim dist: {dist}")


if __name__ == "__main__":
    main()
