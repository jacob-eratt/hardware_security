"""Measure real cache evictions and build a training dataset (hardware phase).

This is the hardware counterpart of ``cpl.data.generate``: instead of computing
the victim from a known policy it drives the C engine ``measure/cacheprobe.c`` to
OBSERVE which way a real cache set evicts, then writes ``samples.parquet`` +
``meta.json`` in the exact schema the simulator uses, so the GRU trainer and
``VictimDataset`` consume hardware and simulated data identically.

    python -m cpl.data.hw_measure --config configs/hw_l2_celeron.yaml [--scale 0.01]

Pipeline:
  1. Generate histories with the shared, split-stable generator (cpl.data.histories),
     so a history_id never crosses the train/val/test boundary.
  2. Hand them to cacheprobe.exe, which finds a congruent address pool (an eviction
     set) for one L2 set and measures R eviction trials per history.
  3. Aggregate the R per-trial victims into ``victim_counts`` (soft labels under
     measurement noise, spec section 6.3) and write the dataset.

Only addresses from our own buffer are ever touched; see the header of
``measure/cacheprobe.c`` for the method and assumptions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
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
from cpl.data.generate import (
    SIM_VICTIM_FIELDS,
    W_MAX,
    _code_hash,
    _fixed_list,
    _list_array,
    check_no_overlap,
    load_config,
)
from cpl.data.histories import (
    HASH_SPLITS,
    HistoryBatch,
    generate_random_split,
    history_id,
    split_of,
)
from cpl.policies.fast import is_pow2, simulate_batch
from cpl.utils.seed import derive_rng

log = logging.getLogger("cpl.hw_measure")

SPLIT_ORDER = ("train", "val", "test", "test_long", "test_xlong")
DEFAULT_LEN_RANGES = {"train": [0, 32], "test_long": [33, 128], "test_xlong": [129, 512]}
MEASURE_DIR = Path(cpl.__file__).resolve().parents[2] / "measure"


# --------------------------------------------------------------------- engine
def ensure_engine() -> Path:
    """Return the path to cacheprobe.exe, compiling it from .c if needed."""
    exe = MEASURE_DIR / "cacheprobe.exe"
    src = MEASURE_DIR / "cacheprobe.c"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    if not src.exists():
        raise FileNotFoundError(f"measurement engine source missing: {src}")
    cc = _find_cc()
    cmd = [cc, "-O2", "-march=native", "-o", str(exe), str(src)]
    log.info("compiling engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe


def _find_cc() -> str:
    import shutil

    for name in ("gcc", "cc", "clang"):
        if (p := shutil.which(name)):
            return p
    # common MSYS2 location on Windows
    for cand in (r"C:\msys64\ucrt64\bin\gcc.exe", r"C:\msys64\mingw64\bin\gcc.exe"):
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("no C compiler found (install MSYS2 gcc or set PATH)")


# --------------------------------------------------------------------- histories
def generate_histories(cfg: dict[str, Any], scale: float) -> tuple[dict[str, HistoryBatch], dict]:
    W, seed, data = int(cfg["W"]), int(cfg["seed"]), cfg["data"]
    lr = {**DEFAULT_LEN_RANGES, **(data.get("len_ranges") or {})}
    pool = tuple(data["pool_range"]) if data.get("pool_range") else None
    batches, stats, used = {}, {}, set()
    for name in SPLIT_ORDER:
        size = int(round(int(data["sizes"].get(name, 0)) * scale))
        if size <= 0:
            continue
        L = tuple(lr.get(name, lr["train"]))
        t0 = time.perf_counter()
        rng = derive_rng(seed, "hw", "split", name)
        exclude = frozenset() if name in HASH_SPLITS else used
        batch, drawn = generate_random_split(W, name, size, L, rng, exclude, pool_range=pool)
        batches[name] = batch
        used.update(batch.ids)
        stats[name] = {"requested": size, "generated": len(batch), "candidates_drawn": drawn,
                       "L_range": list(L), "seconds": round(time.perf_counter() - t0, 2)}
        log.info("histories %-10s n=%-7d L=%s (%.1fs)", name, len(batch), L, stats[name]["seconds"])
    check_no_overlap(batches)
    return batches, stats


# --------------------------------------------------------------------- measurement I/O
def _write_hist_file(path: Path, batch: HistoryBatch) -> None:
    """Flat text: 'N' then one 'L i0 i1 ...' line per history (engine protocol)."""
    lines = [str(len(batch))]
    for row, L in zip(batch.hist.tolist(), batch.lens.tolist()):
        lines.append(f"{L} " + " ".join(str(int(x)) for x in row[:L]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_result_file(path: Path, n: int, R: int) -> np.ndarray:
    """Parse the engine output into an (n, R) int array of per-trial victim ways."""
    rows = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != n:
        raise RuntimeError(f"engine returned {len(rows)} rows, expected {n}")
    out = np.full((n, R), -1, np.int64)
    for i, r in enumerate(rows):
        vals = [int(x) for x in r][:R]
        out[i, : len(vals)] = vals
    return out


def _counts_from_victims(victims: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(n,R) per-trial victims -> (victim_counts [n,16], n_valid_repeats [n]).

    Trials that detected no eviction (-1) or an out-of-range way are dropped.
    """
    n = victims.shape[0]
    counts = np.zeros((n, W_MAX), np.int32)
    valid = (victims >= 0) & (victims < W)
    for i in range(n):
        ways = victims[i][valid[i]]
        if ways.size:
            counts[i, : W] += np.bincount(ways, minlength=W)[:W]
    return counts, valid.sum(1).astype(np.int16)


def run_engine(exe: Path, cfg: dict[str, Any], batch: HistoryBatch, tmp: Path) -> np.ndarray:
    hw = cfg["hw"]
    hist_file, res_file = tmp / "hist.txt", tmp / "res.txt"
    _write_hist_file(hist_file, batch)
    pool = int(hw.get("pool", cfg["W"]))
    cmd = [
        str(exe),
        "--core", str(hw.get("core", 0)),
        "--ways", str(cfg["W"]),
        "--pool", str(pool),
        "--repeats", str(hw.get("repeats", 16)),
        "--buf-mb", str(hw.get("buf_mb", 256)),
        "--in", str(hist_file),
        "--out", str(res_file),
    ]
    if hw.get("threshold"):
        cmd += ["--threshold", str(int(hw["threshold"]))]
    if hw.get("l1_fill"):
        cmd += ["--l1-fill", str(int(hw["l1_fill"]))]
    if hw.get("min_detect") is not None:
        cmd += ["--min-detect", str(float(hw["min_detect"]))]
    if hw.get("force"):
        cmd += ["--force"]
    if hw.get("sim"):
        cmd += ["--sim"]
    if hw.get("verbose"):
        cmd += ["--verbose"]
    log.info("engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _read_result_file(res_file, len(batch), int(hw.get("repeats", 16)))


# --------------------------------------------------------------------- table
def entropy_bits(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, counts / np.maximum(tot, 1), 0.0)
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(1).astype(np.float32)


def _sim_victims_for(batch: HistoryBatch, W: int) -> dict[str, np.ndarray]:
    """What each known policy WOULD evict, for the section 7.3 matching table.

    Only meaningful for hits-only histories (every block id < W); otherwise -1.
    """
    out = {p: np.full(len(batch), -1, np.int8) for p in SIM_VICTIM_FIELDS}
    hits_only = (batch.hist < W) | (batch.hist < 0)
    mask = hits_only.all(1)
    if mask.any():
        pols = SIM_VICTIM_FIELDS if is_pow2(W) else tuple(p for p in SIM_VICTIM_FIELDS if p != "plru")
        r = simulate_batch(batch.hist[mask], batch.lens[mask], W, policies=pols)
        for p, v in r.victims.items():
            out[p][mask] = v
    return out


def build_table(cfg: dict[str, Any], parts: list[dict]) -> pa.Table:
    W, target = int(cfg["W"]), cfg["target"]
    n_tot = sum(len(p["batch"]) for p in parts)
    width = max((p["batch"].hist.shape[1] for p in parts), default=1)

    hist = np.full((n_tot, width), -1, np.int8)
    lens = np.zeros(n_tot, np.int16)
    counts = np.zeros((n_tot, W_MAX), np.int32)
    nvalid = np.zeros(n_tot, np.int16)
    simv = {p: np.full(n_tot, -1, np.int8) for p in SIM_VICTIM_FIELDS}
    ids, split_col, gen_col = [], [], []
    o = 0
    for p in parts:
        b, n = p["batch"], len(p["batch"])
        hist[o:o + n, : b.hist.shape[1]] = b.hist
        lens[o:o + n] = b.lens
        counts[o:o + n] = p["counts"]
        nvalid[o:o + n] = p["nvalid"]
        for k in SIM_VICTIM_FIELDS:
            simv[k][o:o + n] = p["sim"][k]
        ids += b.ids
        split_col += [p["split"]] * n
        gen_col += ["hw_random"] * n
        o += n

    label = counts.argmax(1).astype(np.int8)
    label[counts.sum(1) == 0] = -1  # no valid trial -> no label
    bits = np.full((n_tot, W_MAX - 1), -1, np.int8)
    null = lambda typ: pa.nulls(n_tot, type=typ)  # noqa: E731
    return pa.table({
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n_tot)], pa.string()),
        "target": pa.array([target] * n_tot, pa.string()),
        "source": pa.array(["hw"] * n_tot, pa.string()),
        "policy": pa.array(["unknown"] * n_tot, pa.string()),
        "W": pa.array(np.full(n_tot, W, np.int8)),
        "cache_level": pa.array(np.full(n_tot, int(cfg["hw"].get("cache_level", 2)), np.int8)),
        "set_index": null(pa.int32()),
        "slice": null(pa.int16()),
        "set_role": null(pa.string()),
        "forced_mode": null(pa.string()),
        "run_id": pa.array([cfg["hw"].get("run_id", "run0")] * n_tot, pa.string()),
        "history_id": pa.array(ids, pa.string()),
        "history": _list_array(hist, lens),
        "hist_len": pa.array(lens),
        "gen_type": pa.array(gen_col, pa.string()),
        "victim_counts": _fixed_list(counts, pa.int32()),
        "n_valid_repeats": pa.array(nvalid),
        "label_hard": pa.array(label),
        "label_entropy": pa.array(entropy_bits(counts)),
        "sim_victims": pa.StructArray.from_arrays(
            [pa.array(simv[p]) for p in SIM_VICTIM_FIELDS], names=list(SIM_VICTIM_FIELDS)),
        "plru_bits": _fixed_list(bits, pa.int8()),
        "split": pa.array(split_col, pa.string()),
    })


# --------------------------------------------------------------------- stats
def split_summary(W: int, counts: np.ndarray, nvalid: np.ndarray, lens: np.ndarray,
                  simv: dict[str, np.ndarray]) -> dict[str, Any]:
    label = counts.argmax(1)
    labeled = counts.sum(1) > 0
    n = max(int(labeled.sum()), 1)
    dist = np.bincount(label[labeled], minlength=W)[:W]
    agree = {}
    for p, v in simv.items():
        m = labeled & (v >= 0)
        agree[p] = round(float((v[m] == label[m]).mean()), 4) if m.any() else None
    return {
        "n": int(len(counts)),
        "n_labeled": int(labeled.sum()),
        "frac_undetected": round(float((~labeled).mean()), 4),
        "mean_valid_repeats": round(float(nvalid.mean()), 2),
        "mean_label_consistency": round(float((counts.max(1)[labeled] / np.maximum(counts.sum(1)[labeled], 1)).mean()), 4)
        if labeled.any() else 0.0,
        "hist_len": {"min": int(lens.min()), "max": int(lens.max()), "mean": round(float(lens.mean()), 2)}
        if len(lens) else None,
        "victim_frac": [round(x / n, 4) for x in dist.tolist()],
        "majority_class_acc": round(float(dist.max()) / n, 4),
        "agreement_with_label": agree,
    }


# --------------------------------------------------------------------- driver
def build_dataset(cfg: dict[str, Any], out_dir: str | Path | None = None, scale: float = 1.0) -> dict:
    t_start = time.perf_counter()
    W, target = int(cfg["W"]), cfg["target"]
    out_dir = Path(out_dir or cfg["data"]["out_dir"])
    exe = ensure_engine()
    batches, gen_stats = generate_histories(cfg, scale)

    tmp = out_dir / "_measure_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts, per_split, t_measure = [], {}, 0.0
    for name, b in batches.items():
        t0 = time.perf_counter()
        victims = run_engine(exe, cfg, b, tmp)
        t_measure += time.perf_counter() - t0
        counts, nvalid = _counts_from_victims(victims, W)
        sim = _sim_victims_for(b, W)
        parts.append({"split": name, "batch": b, "counts": counts, "nvalid": nvalid, "sim": sim})
        per_split[name] = {**gen_stats[name], **split_summary(W, counts, nvalid, b.lens, sim)}
        log.info("%-10s labeled=%d/%d  consistency=%.3f  majority=%.3f", name,
                 per_split[name]["n_labeled"], per_split[name]["n"],
                 per_split[name]["mean_label_consistency"], per_split[name]["majority_class_acc"])

    table = build_table(cfg, parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")

    meta = {
        "generator": "cpl.data.hw_measure",
        "code_version": cpl.__version__,
        "code_hash": _code_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": int(cfg["seed"]),
        "scale": scale,
        "source": "hw",
        "config": cfg,
        "W": W,
        "target": target,
        "policy": "unknown",
        "n_samples": table.num_rows,
        "counts_per_split": {k: len(b) for k, b in batches.items()},
        "splits": per_split,
        "timing_seconds": {
            "histories": round(sum(s["seconds"] for s in gen_stats.values()), 2),
            "measurement": round(t_measure, 2),
            "total": round(time.perf_counter() - t_start, 2),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows) in %.1fs", out_dir / "samples.parquet", table.num_rows,
             meta["timing_seconds"]["total"])
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="override data.out_dir")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every split size (e.g. 0.01 for a smoke run)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build_dataset(load_config(args.config), args.out_dir, args.scale)


if __name__ == "__main__":
    main()
