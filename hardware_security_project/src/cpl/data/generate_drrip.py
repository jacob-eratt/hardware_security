"""Build a DRRIP dataset with a forced mode per sample: samples.parquet + meta.json.

    python -m cpl.data.generate_drrip --config configs/sim_drrip_w16.yaml [--scale 0.01]

Every history appears twice, once per mode (``forced_mode`` = "srrip" / "brrip"), in
the same split, so the two modes never leak across splits. Histories draw blocks from
a pool of P ~ U[pool_range] block IDs; ids >= W miss, which is what makes the modes
behave differently (hits-only histories give the same victim ~97% of the time).

Labels: SRRIP mode is deterministic, so its ``victim_counts`` are R x one-hot. BRRIP
mode is stochastic, so each history is simulated R times with independent insertion
draws and ``victim_counts`` holds the R outcomes (soft labels, spec §6.3).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import cpl
from cpl.data.generate import (
    N_BITS_MAX,
    SIM_VICTIM_FIELDS,
    W_MAX,
    _code_hash,
    _fixed_list,
    _list_array,
    check_no_overlap,
    load_config,
)
from cpl.data.histories import HASH_SPLITS, HistoryBatch, generate_random_split
from cpl.policies.drrip import DRRIP_MODES
from cpl.policies.fast_rrip import simulate_drrip_misses
from cpl.utils.seed import derive_rng

log = logging.getLogger("cpl.generate_drrip")

SPLIT_ORDER = ("train", "val", "test", "test_long", "test_xlong")
DEFAULT_LEN_RANGES = {"train": [0, 32], "test_long": [33, 128], "test_xlong": [129, 512]}
CHUNK = 20_000


def generate_histories(cfg: dict[str, Any], scale: float) -> tuple[dict[str, HistoryBatch], dict]:
    W, seed, data = int(cfg["W"]), int(cfg["seed"]), cfg["data"]
    lr = {**DEFAULT_LEN_RANGES, **(data.get("len_ranges") or {})}
    pool = tuple(data["pool_range"])
    batches, stats, used = {}, {}, set()
    for name in SPLIT_ORDER:
        size = int(round(int(data["sizes"].get(name, 0)) * scale))
        if size <= 0:
            continue
        L = tuple(lr.get(name, lr["train"]))
        t0 = time.perf_counter()
        rng = derive_rng(seed, "drrip", "split", name)
        exclude = frozenset() if name in HASH_SPLITS else used
        batch, drawn = generate_random_split(W, name, size, L, rng, exclude, pool_range=pool)
        batches[name] = batch
        used.update(batch.ids)
        stats[name] = {"requested": size, "generated": len(batch), "candidates_drawn": drawn,
                       "L_range": list(L), "seconds": round(time.perf_counter() - t0, 2)}
        log.info("histories %-10s n=%-8d L=%s (%.1fs)", name, len(batch), L, stats[name]["seconds"])
    check_no_overlap(batches)
    return batches, stats


def simulate_counts(batch: HistoryBatch, repeats: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Victim counts (n, W_MAX) per mode, plus SRRIP-mode misses inside the history."""
    n, W = len(batch), batch.W
    out = {m: np.zeros((n, W_MAX), np.int32) for m in DRRIP_MODES}
    misses = np.zeros(n, np.int16)
    for s in range(0, n, CHUNK):
        h, L = batch.hist[s:s + CHUNK], batch.lens[s:s + CHUNK]
        k = len(L)
        rs = simulate_drrip_misses(h, L, W, np.zeros(k, bool), rng)
        out["srrip"][s + np.arange(k), rs.victim_way] = repeats
        misses[s:s + k] = rs.n_misses
        rb = simulate_drrip_misses(np.repeat(h, repeats, 0), np.repeat(L, repeats), W, np.ones(k * repeats, bool), rng)
        row = np.repeat(np.arange(k), repeats)
        np.add.at(out["brrip"], (s + row, rb.victim_way.astype(np.int64)), 1)
    return {**out, "misses": misses}


def entropy_bits(counts: np.ndarray) -> np.ndarray:
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(1).astype(np.float32)


def build_dataset(cfg: dict[str, Any], out_dir: str | Path | None = None, scale: float = 1.0) -> dict:
    t_start = time.perf_counter()
    W, target, data = int(cfg["W"]), cfg["target"], cfg["data"]
    R = int(data.get("repeats", 16))
    out_dir = Path(out_dir or data["out_dir"])
    batches, gen_stats = generate_histories(cfg, scale)

    parts, per_split = [], {}
    for name, b in batches.items():
        t0 = time.perf_counter()
        sim = simulate_counts(b, R, derive_rng(int(cfg["seed"]), "drrip", "sim", name))
        n = len(b)
        hist_new = np.where(b.hist >= W, b.hist, -1)
        n_new = np.array([len(set(r[r >= 0].tolist())) for r in hist_new], np.int16)
        hard = {m: sim[m].argmax(1).astype(np.int8) for m in DRRIP_MODES}
        for m in DRRIP_MODES:
            parts.append(dict(split=name, mode=m, batch=b, counts=sim[m], hard=hard[m],
                              srrip=hard["srrip"], misses=sim["misses"], n_new=n_new))
        per_split[name] = {
            **gen_stats[name],
            "sim_seconds": round(time.perf_counter() - t0, 2),
            "mean_misses_in_history_srrip": round(float(sim["misses"].mean()), 3),
            "frac_hits_only": round(float((n_new == 0).mean()), 4),
            "mode_victim_agreement": round(float((hard["srrip"] == hard["brrip"]).mean()), 4),
            **{f"{m}_victim_frac": np.round(np.bincount(hard[m], minlength=W)[:W] / max(n, 1), 4).tolist()
               for m in DRRIP_MODES},
            "brrip_label_consistency": round(float((sim["brrip"].max(1) / R).mean()), 4),
            "brrip_mean_entropy_bits": round(float(entropy_bits(sim["brrip"]).mean()), 4),
        }
        log.info("simulated %-10s n=%d x2 modes  mode-agreement=%.3f  brrip consistency=%.3f  (%.1fs)",
                 name, n, per_split[name]["mode_victim_agreement"], per_split[name]["brrip_label_consistency"],
                 per_split[name]["sim_seconds"])

    table = _table(cfg, parts, R)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")
    meta = {
        "generator": "cpl.data.generate_drrip",
        "code_version": cpl.__version__,
        "code_hash": _code_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": int(cfg["seed"]),
        "scale": scale,
        "config": cfg,
        "W": W,
        "target": target,
        "policy": "drrip",
        "modes": list(DRRIP_MODES),
        "repeats": R,
        "label_space": "way index evicted by the trigger",
        "n_samples": table.num_rows,
        "counts_per_split": {k: 2 * len(b) for k, b in batches.items()},
        "splits": per_split,
        "timing_seconds": {"total": round(time.perf_counter() - t_start, 2)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows) in %.1fs", out_dir / "samples.parquet", table.num_rows, meta["timing_seconds"]["total"])
    return meta


def _table(cfg: dict[str, Any], parts: list[dict], R: int) -> pa.Table:
    W, target = int(cfg["W"]), cfg["target"]
    n_tot = sum(len(p["batch"]) for p in parts)
    width = max(p["batch"].hist.shape[1] for p in parts)
    hist = np.full((n_tot, width), -1, np.int8)
    lens = np.zeros(n_tot, np.int16)
    counts = np.zeros((n_tot, W_MAX), np.int32)
    hard, srrip_v = np.zeros(n_tot, np.int8), np.zeros(n_tot, np.int8)
    misses, n_new = np.zeros(n_tot, np.int16), np.zeros(n_tot, np.int16)
    ids, split_col, mode_col = [], [], []
    o = 0
    for p in parts:
        b, n = p["batch"], len(p["batch"])
        hist[o:o + n, :b.hist.shape[1]] = b.hist
        lens[o:o + n] = b.lens
        counts[o:o + n] = p["counts"]
        hard[o:o + n], srrip_v[o:o + n] = p["hard"], p["srrip"]
        misses[o:o + n], n_new[o:o + n] = p["misses"], p["n_new"]
        ids += b.ids
        split_col += [p["split"]] * n
        mode_col += [p["mode"]] * n
        o += n
    assert ((hard >= 0) & (hard < W)).all()
    none = np.full(n_tot, -1, np.int8)
    null = lambda typ: pa.nulls(n_tot, type=typ)  # noqa: E731
    return pa.table({
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n_tot)], pa.string()),
        "target": pa.array([target] * n_tot, pa.string()),
        "source": pa.array(["sim"] * n_tot, pa.string()),
        "policy": pa.array(["drrip"] * n_tot, pa.string()),
        "W": pa.array(np.full(n_tot, W, np.int8)),
        "cache_level": null(pa.int8()),
        "set_index": null(pa.int32()),
        "slice": null(pa.int16()),
        "set_role": pa.array(["follower"] * n_tot, pa.string()),
        "forced_mode": pa.array(mode_col, pa.string()),
        "run_id": null(pa.string()),
        "history_id": pa.array(ids, pa.string()),
        "history": _list_array(hist, lens),
        "hist_len": pa.array(lens),
        "gen_type": pa.array(["random_miss"] * n_tot, pa.string()),
        "victim_counts": _fixed_list(counts, pa.int32()),
        "n_valid_repeats": pa.array(np.full(n_tot, R, np.int16)),
        "label_hard": pa.array(hard),
        "label_entropy": pa.array(entropy_bits(counts)),
        "sim_victims": pa.StructArray.from_arrays(
            [pa.array(srrip_v if f == "srrip" else none) for f in SIM_VICTIM_FIELDS], names=list(SIM_VICTIM_FIELDS)),
        "plru_bits": _fixed_list(np.full((n_tot, N_BITS_MAX), -1, np.int8), pa.int8()),
        "split": pa.array(split_col, pa.string()),
        # DRRIP-specific analysis columns
        "n_new_blocks": pa.array(n_new),
        "n_misses_srrip": pa.array(misses),
    })


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build_dataset(load_config(args.config), args.out_dir, args.scale)


if __name__ == "__main__":
    main()
