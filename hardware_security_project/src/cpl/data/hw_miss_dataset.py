"""Build a hardware miss-history dataset with SOFT victim labels, for training.

Unlike cpl.data.hw_discriminate (which keeps only histories with one clean,
reproducible victim and discards ~97%), this keeps EVERY miss history and stores
the full measured victim distribution (soft ``victim_counts``). Training a GRU on
these soft labels lets the model average the cache's per-run nondeterminism out
across many histories and learn whatever deterministic miss-insertion signal
exists -- then the learned model can be compared to the simulated policies.

    python -m cpl.data.hw_miss_dataset --config configs/hw_l2_celeron.yaml \
        --pool 24 --n 40000 --out data/processed/hw_l2_celeron_miss

Output schema matches cpl.data.dataset.VictimDataset, with two miss-specific
fields: ``victim_span`` (= pool; the number of valid victim classes) and a
``victim_counts`` width of ``--nv`` (default 32) so a miss-block victim fits. Set
the trainer's ``model.w_max`` to the same ``--nv``.
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
from cpl.data.generate import W_MAX, _code_hash, _fixed_list, _list_array, load_config
from cpl.data.histories import history_id, split_of
from cpl.data.hw_discriminate import read_counts, run_engine
from cpl.data.hw_measure import ensure_engine
from cpl.policies import make_policy, run_miss_experiment

log = logging.getLogger("cpl.hw_miss_dataset")

SIM_FIELDS = ("lru", "srrip", "fifo", "plru")


def gen_histories(n: int, W: int, pool: int, len_range: tuple[int, int],
                  rng: np.random.Generator) -> tuple[list[list[int]], list[str]]:
    lo, hi = len_range
    hists, splits = [], []
    for _ in range(n):
        L = int(rng.integers(lo, hi + 1))
        h = rng.integers(0, pool, size=L).tolist()
        hists.append(h)
        splits.append(split_of(history_id(W, h)))  # hash split: no leakage across sets
    return hists, splits


def sim_victims(hists: list[list[int]], W: int, pool: int) -> dict[str, np.ndarray]:
    """Trigger's evicted block per reference policy (-1 if a policy can't run)."""
    out = {p: np.full(len(hists), -1, np.int8) for p in SIM_FIELDS}
    for i, h in enumerate(hists):
        for p in SIM_FIELDS:
            try:
                out[p][i] = run_miss_experiment(make_policy(p, W), h, trigger=pool)[1]
            except Exception:  # noqa: BLE001
                out[p][i] = -1
    return out


def build(cfg: dict[str, Any], pool: int, n: int, len_range: tuple[int, int],
          out_dir: Path, nv: int = 32, l1_fill: int = 16) -> dict:
    t0 = time.perf_counter()
    W = int(cfg["W"])
    if not (W < pool <= nv <= 32):
        raise ValueError(f"need W({W}) < pool({pool}) <= nv({nv}) <= 32")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = ensure_engine()
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))

    hists, splits = gen_histories(n, W, pool, len_range, rng)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    hist_file, res_file = tmp / "hist.txt", tmp / "res.txt"
    hist_file.write_text(f"{len(hists)}\n" + "\n".join(
        f"{len(h)} " + " ".join(map(str, h)) for h in hists) + "\n", encoding="utf-8")
    run_engine(exe, cfg, pool, hist_file, res_file, l1_fill)

    R = int(cfg["hw"].get("repeats", 16))
    counts_pool = read_counts(res_file, len(hists), pool)       # (n, pool)
    keep = counts_pool.sum(1) > 0                               # drop histories w/ no eviction ever
    sims = sim_victims(hists, W, pool)

    n_tot = int(keep.sum())
    width = max(len(h) for h in hists)
    hist_mat = np.full((n_tot, width), -1, np.int8)
    lens = np.zeros(n_tot, np.int16)
    counts = np.zeros((n_tot, nv), np.int32)
    split_col, simv = [], {p: np.full(n_tot, -1, np.int8) for p in SIM_FIELDS}
    o = 0
    for i, h in enumerate(hists):
        if not keep[i]:
            continue
        hist_mat[o, : len(h)] = h
        lens[o] = len(h)
        counts[o, :pool] = counts_pool[i]
        split_col.append(splits[i])
        for p in SIM_FIELDS:
            simv[p][o] = sims[p][i]
        o += 1
    label = counts.argmax(1).astype(np.int8)

    target = cfg.get("target", "hw_miss")
    null = lambda typ: pa.nulls(n_tot, type=typ)  # noqa: E731
    table = pa.table({
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n_tot)], pa.string()),
        "target": pa.array([target] * n_tot, pa.string()),
        "source": pa.array(["hw"] * n_tot, pa.string()),
        "policy": pa.array(["unknown"] * n_tot, pa.string()),
        "W": pa.array(np.full(n_tot, W, np.int8)),
        "victim_span": pa.array(np.full(n_tot, pool, np.int16)),
        "cache_level": pa.array(np.full(n_tot, int(cfg["hw"].get("cache_level", 2)), np.int8)),
        "forced_mode": null(pa.string()),
        "history": _list_array(hist_mat, lens),
        "hist_len": pa.array(lens),
        "gen_type": pa.array(["hw_miss"] * n_tot, pa.string()),
        "victim_counts": _fixed_list(counts, pa.int32()),
        "n_valid_repeats": pa.array(np.full(n_tot, R, np.int16)),
        "label_hard": pa.array(label),
        "sim_victims": pa.StructArray.from_arrays(
            [pa.array(simv[p]) for p in SIM_FIELDS], names=list(SIM_FIELDS)),
        "split": pa.array(split_col, pa.string()),
    })
    pq.write_table(table, out_dir / "samples.parquet")

    # agreement of each policy's victim with the hard (argmax) measured victim
    sp = np.array(split_col)
    per_split = {}
    for s in ("train", "val", "test"):
        m = sp == s
        if not m.any():
            continue
        agree = {}
        for p in SIM_FIELDS:
            mm = m & (np.concatenate([simv[p]]) >= 0)
            agree[p] = round(float((simv[p][mm] == label[mm]).mean()), 4) if mm.any() else None
        per_split[s] = {"n": int(m.sum()),
                        "mean_top_count_frac": round(float((counts[m].max(1) / R).mean()), 4),
                        "agreement_hard": agree}
    meta = {
        "generator": "cpl.data.hw_miss_dataset",
        "code_version": cpl.__version__, "code_hash": _code_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "hw", "W": W, "pool": pool, "nv": nv, "repeats": R,
        "len_range": list(len_range), "target": target,
        "n_generated": n, "n_kept": n_tot, "frac_no_eviction": round(float((~keep).mean()), 4),
        "splits": per_split,
        "note": "soft victim_counts over pool blocks; set model.w_max = nv to train.",
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d/%d kept) splits=%s", out_dir / "samples.parquet", n_tot, n,
             {s: per_split[s]["n"] for s in per_split})
    for s, v in per_split.items():
        log.info("  %-5s top-count-frac=%.3f agreement_hard=%s", s, v["mean_top_count_frac"], v["agreement_hard"])
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--n", type=int, default=40000, help="histories to measure (before dropping no-eviction)")
    ap.add_argument("--len-min", type=int, default=16)
    ap.add_argument("--len-max", type=int, default=40)
    ap.add_argument("--nv", type=int, default=32, help="victim_counts width (set trainer model.w_max to match)")
    ap.add_argument("--l1-fill", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build(load_config(args.config), args.pool, args.n, (args.len_min, args.len_max),
          Path(args.out), args.nv, args.l1_fill)


if __name__ == "__main__":
    main()
