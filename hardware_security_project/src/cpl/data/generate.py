"""Build a simulated dataset: samples.parquet + meta.json + histories.parquet (spec §4-5).

    python -m cpl.data.generate --config configs/sim_plru_w16.yaml [--scale 0.01] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import cpl
from cpl.data.histories import HASH_SPLITS, HistoryBatch, generate_diagnostic, generate_random_split
from cpl.policies.fast import FAST_POLICIES, BatchResult, is_pow2, simulate_batch
from cpl.utils.seed import derive_rng

log = logging.getLogger("cpl.generate")

W_MAX = 16
N_BITS_MAX = W_MAX - 1
SIM_VICTIM_FIELDS = ("plru", "lru", "fifo", "srrip")  # fixed struct layout (spec §5)
SPLIT_ORDER = ("train", "val", "test", "test_long", "test_xlong", "test_diag")

DEFAULT_LEN_RANGES = {
    "train": [0, 32],
    "test_long": [33, 128],
    "test_xlong": [129, 512],
    "test_diag": [0, 32],
}


# ----------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SplitPlan:
    name: str
    size: int
    L_range: tuple[int, int]
    kind: str  # "hash" | "random" | "diagnostic"


def plan_splits(cfg: dict[str, Any], scale: float = 1.0) -> list[SplitPlan]:
    """Resolve sizes / length ranges for each split. Splits with size 0 are skipped."""
    data = cfg["data"]
    sizes = data["sizes"]
    lr = {**DEFAULT_LEN_RANGES, **(data.get("len_ranges") or {})}
    plans = []
    for name in SPLIT_ORDER:
        size = int(round(int(sizes.get(name, 0)) * scale))
        if size <= 0:
            continue
        # val / test use the train length range unless given explicitly
        L = lr.get(name, lr["train"])
        kind = "hash" if name in HASH_SPLITS else ("diagnostic" if name == "test_diag" else "random")
        plans.append(SplitPlan(name, size, (int(L[0]), int(L[1])), kind))
    return plans


# ----------------------------------------------------------------------- generation
def generate_histories(cfg: dict[str, Any], scale: float = 1.0) -> tuple[dict[str, HistoryBatch], dict]:
    """Generate the histories of every split. Returns (batches, per-split generation stats).

    Hash splits never overlap by construction. Separately generated splits drop any
    history whose ID already appears in an earlier split, so every history_id belongs
    to exactly one split.
    """
    W = int(cfg["W"])
    seed = int(cfg["seed"])
    data = cfg["data"]
    batches: dict[str, HistoryBatch] = {}
    stats: dict[str, dict] = {}
    used_ids: set[str] = set()
    for plan in plan_splits(cfg, scale):
        t0 = time.perf_counter()
        rng = derive_rng(seed, "split", plan.name)
        exclude = frozenset() if plan.kind == "hash" else used_ids
        if plan.kind == "diagnostic":
            batch, drawn = generate_diagnostic(
                data.get("diag_policies", ["plru", "lru"]),
                plan.size,
                W,
                plan.L_range,
                rng,
                require=data.get("diag_require", "any"),
                exclude_ids=exclude,
            )
        else:
            batch, drawn = generate_random_split(W, plan.name, plan.size, plan.L_range, rng, exclude)
        batches[plan.name] = batch
        used_ids.update(batch.ids)
        stats[plan.name] = {
            "requested": plan.size,
            "generated": len(batch),
            "candidates_drawn": drawn,
            "L_range": list(plan.L_range),
            "gen_type": batch.gen_type,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        log.info("generated %-10s n=%-8d drawn=%-9d L=%s (%.1fs)",
                 plan.name, len(batch), drawn, plan.L_range, stats[plan.name]["seconds"])
    check_no_overlap(batches)
    return batches, stats


def check_no_overlap(batches: dict[str, HistoryBatch]) -> None:
    """Leakage check (spec §10): history_id sets of different splits are disjoint."""
    names = list(batches)
    sets = {k: set(b.ids) for k, b in batches.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            assert not inter, f"history_id overlap between {a} and {b}: {len(inter)} e.g. {next(iter(inter))}"


def simulate(batch: HistoryBatch) -> BatchResult:
    pols = FAST_POLICIES if is_pow2(batch.W) else tuple(p for p in FAST_POLICIES if p != "plru")
    return simulate_batch(batch.hist, batch.lens, batch.W, policies=pols)


# ----------------------------------------------------------------------- table building
def _list_array(hist: np.ndarray, lens: np.ndarray) -> pa.Array:
    valid = np.arange(hist.shape[1])[None, :] < lens[:, None]
    values = pa.array(hist[valid].astype(np.int8), type=pa.int8())
    offsets = pa.array(np.concatenate([[0], np.cumsum(lens, dtype=np.int64)]).astype(np.int32))
    return pa.ListArray.from_arrays(offsets, values)


def _fixed_list(mat: np.ndarray, typ: pa.DataType) -> pa.Array:
    n, k = mat.shape
    return pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1), type=typ), k)


def build_table(cfg: dict[str, Any], batches: dict[str, HistoryBatch],
                results: dict[str, BatchResult]) -> pa.Table:
    """Assemble samples.parquet (spec §5) for all splits, in SPLIT_ORDER."""
    W = int(cfg["W"])
    target = cfg["target"]
    policy = cfg["policy"]
    if policy not in SIM_VICTIM_FIELDS:
        raise ValueError(f"label policy {policy!r} not supported by the fast simulator")
    if cfg.get("noise"):
        raise NotImplementedError("synthetic noise (spec §8.1) is not implemented yet; set noise: null")

    names = list(batches)
    n_tot = sum(len(batches[k]) for k in names)
    width = max(batches[k].hist.shape[1] for k in names)

    hist = np.full((n_tot, width), -1, dtype=np.int8)
    lens = np.zeros(n_tot, dtype=np.int16)
    ids: list[str] = []
    split_col: list[str] = []
    gen_col: list[str] = []
    victims = {p: np.full(n_tot, -1, dtype=np.int8) for p in SIM_VICTIM_FIELDS}
    bits = np.full((n_tot, N_BITS_MAX), -1, dtype=np.int8)
    o = 0
    for k in names:
        b, r = batches[k], results[k]
        n = len(b)
        hist[o:o + n, : b.hist.shape[1]] = b.hist
        lens[o:o + n] = b.lens
        ids += b.ids
        split_col += [k] * n
        gen_col += [b.gen_type] * n
        for p, v in r.victims.items():
            victims[p][o:o + n] = v
        if r.plru_bits is not None:
            bits[o:o + n, : W - 1] = r.plru_bits
        o += n

    label = victims[policy]
    assert (label >= 0).all() and (label < W).all(), "every sample must have a valid victim"
    counts = np.zeros((n_tot, W_MAX), dtype=np.int32)
    counts[np.arange(n_tot), label] = 1

    null = lambda typ: pa.nulls(n_tot, type=typ)  # noqa: E731
    cols = {
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n_tot)], pa.string()),
        "target": pa.array([target] * n_tot, pa.string()),
        "source": pa.array(["sim"] * n_tot, pa.string()),
        "policy": pa.array([policy] * n_tot, pa.string()),
        "W": pa.array(np.full(n_tot, W, np.int8)),
        # hardware-only fields (null in simulation)
        "cache_level": null(pa.int8()),
        "set_index": null(pa.int32()),
        "slice": null(pa.int16()),
        "set_role": null(pa.string()),
        "forced_mode": null(pa.string()),
        "run_id": null(pa.string()),
        "history_id": pa.array(ids, pa.string()),
        "history": _list_array(hist, lens),
        "hist_len": pa.array(lens.astype(np.int16)),
        "gen_type": pa.array(gen_col, pa.string()),
        "victim_counts": _fixed_list(counts, pa.int32()),
        "n_valid_repeats": pa.array(np.ones(n_tot, np.int16)),
        "label_hard": pa.array(label.astype(np.int8)),
        "label_entropy": pa.array(np.zeros(n_tot, np.float32)),
        "sim_victims": pa.StructArray.from_arrays(
            [pa.array(victims[p]) for p in SIM_VICTIM_FIELDS], names=list(SIM_VICTIM_FIELDS)
        ),
        "plru_bits": _fixed_list(bits, pa.int8()),
        "split": pa.array(split_col, pa.string()),
    }
    return pa.table(cols)


# ----------------------------------------------------------------------- stats / meta
def split_summary(W: int, label: np.ndarray, lens: np.ndarray, ids: list[str],
                  victims: dict[str, np.ndarray]) -> dict[str, Any]:
    dist = np.bincount(label, minlength=W)[:W]
    n = max(len(label), 1)
    return {
        "n": int(len(label)),
        "n_unique_histories": len(set(ids)),
        "hist_len": {"min": int(lens.min()), "max": int(lens.max()), "mean": round(float(lens.mean()), 2)}
        if len(lens) else None,
        "victim_counts": dist.tolist(),
        "victim_frac": [round(x / n, 4) for x in dist.tolist()],
        "majority_class_acc": round(float(dist.max()) / n, 4),
        "agreement_with_label": {p: round(float((v == label).mean()), 4) for p, v in victims.items()},
    }


def _code_hash() -> str:
    """sha1 over the simulator + data-generation sources, recorded for provenance."""
    root = Path(cpl.__file__).parent
    h = hashlib.sha1()
    for sub in ("policies", "data"):
        for p in sorted((root / sub).glob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def write_histories(path: Path, table: pa.Table) -> int:
    """Merge this dataset's unique histories into the shared histories.parquet."""
    cols = ["history_id", "W", "history", "hist_len", "gen_type"]
    t = table.select(cols)
    if path.exists():
        t = pa.concat_tables([pq.read_table(path).select(cols).cast(t.schema), t])
    df = t.to_pandas().drop_duplicates("history_id", keep="first")
    out = pa.Table.from_pandas(df, schema=t.schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, path)
    return out.num_rows


def build_dataset(cfg: dict[str, Any], out_dir: str | Path | None = None, scale: float = 1.0,
                  histories_path: str | Path | None = "data/histories/histories.parquet") -> dict:
    """Generate, simulate, validate and write a dataset. Returns the meta dict."""
    t_start = time.perf_counter()
    W = int(cfg["W"])
    out_dir = Path(out_dir or cfg["data"]["out_dir"])
    batches, gen_stats = generate_histories(cfg, scale)

    t0 = time.perf_counter()
    results = {k: simulate(b) for k, b in batches.items()}
    sim_seconds = time.perf_counter() - t0

    table = build_table(cfg, batches, results)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")
    n_hist = write_histories(Path(histories_path), table) if histories_path else None

    policy = cfg["policy"]
    per_split = {}
    for k, b in batches.items():
        v = results[k].victims
        per_split[k] = {**gen_stats[k], **split_summary(W, v[policy], b.lens, b.ids, v)}
        log.info("%-10s n=%-8d majority=%.3f  victim%%=%s", k, per_split[k]["n"],
                 per_split[k]["majority_class_acc"],
                 " ".join(f"{100 * x:.1f}" for x in per_split[k]["victim_frac"]))

    meta = {
        "generator": "cpl.data.generate",
        "code_version": cpl.__version__,
        "code_hash": _code_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": int(cfg["seed"]),
        "scale": scale,
        "config": cfg,
        "W": W,
        "target": cfg["target"],
        "policy": policy,
        "n_samples": table.num_rows,
        "counts_per_split": {k: len(b) for k, b in batches.items()},
        "splits": per_split,
        "timing_seconds": {
            "histories": round(sum(s["seconds"] for s in gen_stats.values()), 2),
            "simulation": round(sim_seconds, 2),
            "total": round(time.perf_counter() - t_start, 2),
        },
        "histories_parquet_rows": n_hist,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s (%d rows) in %.1fs", out_dir / "samples.parquet", table.num_rows,
             meta["timing_seconds"]["total"])
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="override data.out_dir")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every split size (e.g. 0.01 for a smoke run)")
    ap.add_argument("--histories", default="data/histories/histories.parquet",
                    help="shared histories table to merge into ('' to skip)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build_dataset(load_config(args.config), args.out_dir, args.scale, args.histories or None)


if __name__ == "__main__":
    main()
