"""Build a tree-PLRU dataset on histories that may miss: samples.parquet + meta.json.

    python -m cpl.data.generate_plru_miss --config configs/sim_plru_miss_w16.yaml [--scale 0.01]

Histories come from the same generator, seed and pool as the DRRIP dataset
(`generate_drrip.generate_histories`), so with the same config values both datasets
contain exactly the same histories and differ only in the policy that labels them.
PLRU is deterministic, so every label is one-hot.
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
from cpl.data.generate import N_BITS_MAX, SIM_VICTIM_FIELDS, W_MAX, _code_hash, _fixed_list, _list_array, load_config
from cpl.data.generate_drrip import CHUNK, generate_histories
from cpl.policies.fast_plru_miss import simulate_plru_misses
from cpl.policies.fast_rrip import simulate_drrip_misses
from cpl.utils.seed import derive_rng

log = logging.getLogger("cpl.generate_plru_miss")


def build_dataset(cfg: dict[str, Any], out_dir: str | Path | None = None, scale: float = 1.0) -> dict:
    t_start = time.perf_counter()
    W, target = int(cfg["W"]), cfg["target"]
    out_dir = Path(out_dir or cfg["data"]["out_dir"])
    batches, gen_stats = generate_histories(cfg, scale)

    cols: dict[str, list] = {k: [] for k in ("hist", "lens", "ids", "split", "plru", "srrip", "bits", "misses", "n_new")}
    per_split = {}
    for name, b in batches.items():
        n = len(b)
        plru = np.zeros(n, np.int8)
        srrip = np.zeros(n, np.int8)
        bits = np.zeros((n, W - 1), np.int8)
        misses = np.zeros(n, np.int16)
        for s in range(0, n, CHUNK):
            h, L = b.hist[s:s + CHUNK], b.lens[s:s + CHUNK]
            r = simulate_plru_misses(h, L, W)
            plru[s:s + len(L)], bits[s:s + len(L)], misses[s:s + len(L)] = r.victim_way, r.bits, r.n_misses
            srrip[s:s + len(L)] = simulate_drrip_misses(h, L, W, np.zeros(len(L), bool), derive_rng(0, "unused")).victim_way
        n_new = np.array([len(set(r[r >= W].tolist())) for r in b.hist], np.int16)
        for k, v in (("hist", b.hist), ("lens", b.lens), ("ids", b.ids), ("split", [name] * n), ("plru", plru),
                     ("srrip", srrip), ("bits", bits), ("misses", misses), ("n_new", n_new)):
            cols[k].append(v)
        per_split[name] = {
            **gen_stats[name],
            "mean_misses_in_history": round(float(misses.mean()), 3),
            "frac_hits_only": round(float((n_new == 0).mean()), 4),
            "victim_frac": np.round(np.bincount(plru, minlength=W)[:W] / max(n, 1), 4).tolist(),
            "majority_class_acc": round(float(np.bincount(plru, minlength=W).max() / max(n, 1)), 4),
            "agreement_with_srrip_mode": round(float((plru == srrip).mean()), 4),
        }
        log.info("%-10s n=%-7d misses/hist=%.2f  majority=%.3f  PLRU=SRRIP on %.3f", name, n,
                 per_split[name]["mean_misses_in_history"], per_split[name]["majority_class_acc"],
                 per_split[name]["agreement_with_srrip_mode"])

    width = max(h.shape[1] for h in cols["hist"])
    hist = np.concatenate([np.pad(h, ((0, 0), (0, width - h.shape[1])), constant_values=-1) for h in cols["hist"]])
    lens = np.concatenate(cols["lens"]).astype(np.int16)
    label = np.concatenate(cols["plru"])
    n_tot = len(label)
    counts = np.zeros((n_tot, W_MAX), np.int32)
    counts[np.arange(n_tot), label] = 1
    bits = np.full((n_tot, N_BITS_MAX), -1, np.int8)
    bits[:, :W - 1] = np.concatenate(cols["bits"])
    srrip = np.concatenate(cols["srrip"])
    none = np.full(n_tot, -1, np.int8)
    null = lambda typ: pa.nulls(n_tot, type=typ)  # noqa: E731
    victims = {"plru": label, "srrip": srrip}
    table = pa.table({
        "sample_id": pa.array([f"{target}_{i:08d}" for i in range(n_tot)], pa.string()),
        "target": pa.array([target] * n_tot, pa.string()),
        "source": pa.array(["sim"] * n_tot, pa.string()),
        "policy": pa.array(["plru"] * n_tot, pa.string()),
        "W": pa.array(np.full(n_tot, W, np.int8)),
        "cache_level": null(pa.int8()),
        "set_index": null(pa.int32()),
        "slice": null(pa.int16()),
        "set_role": null(pa.string()),
        "forced_mode": null(pa.string()),
        "run_id": null(pa.string()),
        "history_id": pa.array([i for ids in cols["ids"] for i in ids], pa.string()),
        "history": _list_array(hist, lens),
        "hist_len": pa.array(lens),
        "gen_type": pa.array(["random_miss"] * n_tot, pa.string()),
        "victim_counts": _fixed_list(counts, pa.int32()),
        "n_valid_repeats": pa.array(np.ones(n_tot, np.int16)),
        "label_hard": pa.array(label),
        "label_entropy": pa.array(np.zeros(n_tot, np.float32)),
        "sim_victims": pa.StructArray.from_arrays(
            [pa.array(victims.get(f, none)) for f in SIM_VICTIM_FIELDS], names=list(SIM_VICTIM_FIELDS)),
        "plru_bits": _fixed_list(bits, pa.int8()),
        "split": pa.array([s for ss in cols["split"] for s in ss], pa.string()),
        "n_new_blocks": pa.array(np.concatenate(cols["n_new"])),
        "n_misses_plru": pa.array(np.concatenate(cols["misses"])),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "samples.parquet")
    meta = {
        "generator": "cpl.data.generate_plru_miss",
        "code_version": cpl.__version__,
        "code_hash": _code_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": int(cfg["seed"]),
        "scale": scale,
        "config": cfg,
        "W": W,
        "target": target,
        "policy": "plru",
        "histories_shared_with": "generate_drrip (same seed, pool_range, sizes and len_ranges)",
        "label_space": "way index evicted by the trigger",
        "n_samples": n_tot,
        "counts_per_split": {k: len(b) for k, b in batches.items()},
        "splits": per_split,
        "timing_seconds": {"total": round(time.perf_counter() - t_start, 2)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows) in %.1fs", out_dir / "samples.parquet", n_tot, meta["timing_seconds"]["total"])
    return meta


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
