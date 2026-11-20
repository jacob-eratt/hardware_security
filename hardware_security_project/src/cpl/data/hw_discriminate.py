"""Distinguish LRU from SRRIP (and other policies) on real hardware.

Hits-only histories can't separate LRU from SRRIP (they predict the same victim),
so this experiment uses histories WITH MISSES and keeps only the *diagnostic* ones
where the candidate policies disagree, then measures those on hardware and reports
which policy the cache actually follows.

    python -m cpl.data.hw_discriminate --config configs/hw_l2_celeron.yaml \
        --pool 20 --n-diag 1500 --out runs/hw_discriminate

Pipeline:
  1. Draw random miss-bearing histories (block IDs in [0, pool); IDs >= W miss).
  2. For each, compute the trigger's evicted BLOCK for LRU/SRRIP/FIFO/PLRU with the
     reference simulators (cpl.policies.run_miss_experiment). Keep histories where
     LRU != SRRIP (the ones that actually tell them apart).
  3. Measure those on hardware in --miss mode (engine reports the evicted block).
  4. Tally, over the diagnostic set, how often the hardware matched each policy.

Hardware params (core, repeats, buf_mb, l1_fill, threshold, min_detect) are read from
the same config used for cpl.data.hw_measure.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cpl.data.generate import load_config
from cpl.data.hw_measure import ensure_engine
from cpl.policies import make_policy, run_miss_experiment

log = logging.getLogger("cpl.hw_discriminate")

CANDIDATE_POLICIES = ("lru", "srrip", "fifo", "plru")


def predicted_victim_block(history: list[int], W: int, trigger: int, names=CANDIDATE_POLICIES) -> dict[str, int]:
    """Evicted block ID of the trigger access, per reference policy."""
    out: dict[str, int] = {}
    for nm in names:
        pol = make_policy(nm, W)
        try:
            _, evicted = run_miss_experiment(pol, history, trigger=trigger)
        except Exception:  # noqa: BLE001  (policy e.g. plru needs power-of-two W)
            evicted = -1
        out[nm] = int(evicted)
    return out


def generate_diagnostic(n: int, W: int, pool: int, len_range: tuple[int, int],
                        rng: np.random.Generator) -> tuple[list[list[int]], list[dict[str, int]]]:
    """Random miss histories where LRU and SRRIP evict *different* blocks."""
    lo, hi = len_range
    hists: list[list[int]] = []
    preds: list[dict[str, int]] = []
    drawn = 0
    while len(hists) < n:
        L = int(rng.integers(lo, hi + 1))
        h = rng.integers(0, pool, size=L).tolist()
        drawn += 1
        lru = make_policy("lru", W)
        srrip = make_policy("srrip", W)
        v_lru = run_miss_experiment(lru, h, trigger=pool)[1]
        v_srrip = run_miss_experiment(srrip, h, trigger=pool)[1]
        if v_lru != v_srrip:
            hists.append(h)
            preds.append(predicted_victim_block(h, W, pool))
        if drawn > 2000 * n + 100000:
            log.warning("stopped after %d draws with only %d/%d diagnostic histories", drawn, len(hists), n)
            break
    log.info("kept %d diagnostic histories from %d draws (%.0f%% disagree)",
             len(hists), drawn, 100 * len(hists) / max(drawn, 1))
    return hists, preds


def write_hist_file(path: Path, hists: list[list[int]]) -> None:
    lines = [str(len(hists))]
    for h in hists:
        lines.append(f"{len(h)} " + " ".join(str(int(x)) for x in h))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_results(path: Path, n: int, R: int) -> np.ndarray:
    rows = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != n:
        raise RuntimeError(f"engine returned {len(rows)} rows, expected {n}")
    out = np.full((n, R), -2, np.int64)  # -2 = missing, -1 = engine 'no victim'
    for i, r in enumerate(rows):
        v = [int(x) for x in r][:R]
        out[i, : len(v)] = v
    return out


def modal_victim(trials: np.ndarray) -> tuple[int, float, int]:
    """(modal block, consistency, n_valid) from one history's per-trial victims."""
    valid = trials[trials >= 0]
    if valid.size == 0:
        return -1, 0.0, 0
    block, count = Counter(valid.tolist()).most_common(1)[0]
    return int(block), count / valid.size, int(valid.size)


def run_engine(exe: Path, cfg: dict[str, Any], pool: int, hist_file: Path, res_file: Path) -> None:
    hw = cfg["hw"]
    cmd = [
        str(exe), "--miss",
        "--core", str(hw.get("core", 0)),
        "--ways", str(cfg["W"]),
        "--pool", str(pool),
        "--repeats", str(hw.get("repeats", 16)),
        "--buf-mb", str(hw.get("buf_mb", 256)),
        "--in", str(hist_file), "--out", str(res_file),
    ]
    if hw.get("threshold"):
        cmd += ["--threshold", str(int(hw["threshold"]))]
    if hw.get("l1_fill"):
        cmd += ["--l1-fill", str(int(hw["l1_fill"]))]
    if hw.get("min_detect") is not None:
        cmd += ["--min-detect", str(float(hw["min_detect"]))]
    if hw.get("force"):
        cmd += ["--force"]
    log.info("engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def discriminate(cfg: dict[str, Any], pool: int, n_diag: int, len_range: tuple[int, int],
                 out_dir: Path, min_consistency: float = 0.5) -> dict:
    t0 = time.perf_counter()
    W = int(cfg["W"])
    if pool <= W:
        raise ValueError(f"--pool must exceed W={W} so some accesses miss")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = ensure_engine()
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))

    hists, preds = generate_diagnostic(n_diag, W, pool, len_range, rng)
    hist_file, res_file = out_dir / "hist.txt", out_dir / "res.txt"
    write_hist_file(hist_file, hists)
    run_engine(exe, cfg, pool, hist_file, res_file)

    R = int(cfg["hw"].get("repeats", 16))
    trials = read_results(res_file, len(hists), R)

    # tally, over confidently-labelled diagnostic histories, which policy the HW matched
    tallies = {p: 0 for p in CANDIDATE_POLICIES}
    hw_lru = hw_srrip = neither = used = undetected = 0
    rows = []
    for h, pred, tr in zip(hists, preds, trials):
        block, consist, nvalid = modal_victim(tr)
        if block < 0 or consist < min_consistency:
            undetected += 1
            continue
        used += 1
        for p in CANDIDATE_POLICIES:
            if pred[p] == block:
                tallies[p] += 1
        hw_lru += pred["lru"] == block
        hw_srrip += pred["srrip"] == block
        if pred["lru"] != block and pred["srrip"] != block:
            neither += 1
        rows.append({"hist_len": len(h), "hw_block": block, "consistency": round(consist, 3),
                     "pred": pred})

    denom = max(used, 1)
    report = {
        "generator": "cpl.data.hw_discriminate",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "W": W, "pool": pool, "repeats": R, "len_range": list(len_range),
        "n_diagnostic": len(hists),
        "n_used": used, "n_undetected_or_noisy": undetected,
        "min_consistency": min_consistency,
        "agreement_on_diagnostic": {p: round(tallies[p] / denom, 4) for p in CANDIDATE_POLICIES},
        "hw_matches_lru": round(hw_lru / denom, 4),
        "hw_matches_srrip": round(hw_srrip / denom, 4),
        "hw_matches_neither": round(neither / denom, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "report.json").write_text(json.dumps({**report, "samples": rows[:200]}, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info("DIAGNOSTIC RESULT (histories where LRU != SRRIP)")
    log.info("  used %d / %d (dropped %d noisy/undetected)", used, len(hists), undetected)
    for p in CANDIDATE_POLICIES:
        log.info("  hardware matches %-6s on %5.1f%%", p, 100 * report["agreement_on_diagnostic"][p])
    log.info("  -> LRU %.1f%%  vs  SRRIP %.1f%%  (neither %.1f%%)",
             100 * report["hw_matches_lru"], 100 * report["hw_matches_srrip"],
             100 * report["hw_matches_neither"])
    log.info("=" * 60)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="same config as hw_measure (for hw params + W)")
    ap.add_argument("--pool", type=int, default=24, help="number of distinct blocks (> W so some miss)")
    ap.add_argument("--n-diag", type=int, default=1200, help="diagnostic histories to measure")
    ap.add_argument("--len-min", type=int, default=16)
    ap.add_argument("--len-max", type=int, default=40,
                    help="longer histories => more misses => LRU/SRRIP diverge more")
    ap.add_argument("--min-consistency", type=float, default=0.5,
                    help="drop histories whose modal victim is below this repeat fraction")
    ap.add_argument("--out", default="runs/hw_discriminate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    discriminate(load_config(args.config), args.pool, args.n_diag,
                 (args.len_min, args.len_max), Path(args.out), args.min_consistency)


if __name__ == "__main__":
    main()
