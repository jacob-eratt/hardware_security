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
# (per-trial Counter no longer needed: victims are denoised in the engine)
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


def generate_control(n: int, W: int, pool: int, len_range: tuple[int, int],
                     rng: np.random.Generator) -> tuple[list[list[int]], list[int]]:
    """Hits-only control histories (block IDs in [0, W), so no misses) and their LRU
    victim. Run through the SAME miss-path detector, these isolate measurement
    soundness from policy identity: the main dataset already showed the L2 is LRU on
    hits-only traffic, so the miss-path must reproduce LRU here (~0.95) if it works."""
    lo, hi = len_range
    hists, lru_v = [], []
    for _ in range(n):
        L = int(rng.integers(lo, hi + 1))
        h = rng.integers(0, W, size=L).tolist()
        hists.append(h)
        lru_v.append(int(run_miss_experiment(make_policy("lru", W), h, trigger=pool)[1]))
    return hists, lru_v


def write_hist_file(path: Path, hists: list[list[int]]) -> None:
    lines = [str(len(hists))]
    for h in hists:
        lines.append(f"{len(h)} " + " ".join(str(int(x)) for x in h))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_counts(path: Path, n: int, pool: int) -> np.ndarray:
    """Per-history soft victim counts (engine miss mode), shape (n, pool)."""
    rows = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(rows) != n:
        raise RuntimeError(f"engine returned {len(rows)} rows, expected {n}")
    M = np.zeros((n, pool), np.int64)
    for i, r in enumerate(rows):
        v = [int(x) for x in r][:pool]
        M[i, : len(v)] = v
    return M


def victim_from_counts(row: np.ndarray, R: int) -> int:
    """Clean single victim from a soft count row: argmax drop, accepted only if it is
    large (>= R/2) and clearly beats the runner-up; else -1 (ambiguous/nondeterministic)."""
    order = np.argsort(row)[::-1]
    best = int(row[order[0]])
    second = int(row[order[1]]) if len(order) > 1 else 0
    if 2 * best < R or best - second < R // 4:
        return -1
    return int(order[0])


def run_engine(exe: Path, cfg: dict[str, Any], pool: int, hist_file: Path, res_file: Path,
               l1_fill: int = 16) -> None:
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
    # L1 draining is essential in miss mode: the trigger's victim may still be in
    # L1, so without it the probe reads an L1 hit and the eviction is masked.
    if l1_fill > 0:
        cmd += ["--l1-fill", str(l1_fill)]
    if hw.get("min_detect") is not None:
        cmd += ["--min-detect", str(float(hw["min_detect"]))]
    if hw.get("force"):
        cmd += ["--force"]
    log.info("engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def discriminate(cfg: dict[str, Any], pool: int, n_diag: int, len_range: tuple[int, int],
                 out_dir: Path, n_control: int = 400, l1_fill: int = 16) -> dict:
    t0 = time.perf_counter()
    W = int(cfg["W"])
    if pool <= W:
        raise ValueError(f"--pool must exceed W={W} so some accesses miss")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = ensure_engine()
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))

    diag, preds = generate_diagnostic(n_diag, W, pool, len_range, rng)
    ctrl, ctrl_lru = generate_control(n_control, W, pool, len_range, rng)
    log.info("control: %d hits-only histories (measurement sanity check)", len(ctrl))

    all_hists = ctrl + diag                       # one engine run; control first
    hist_file, res_file = out_dir / "hist.txt", out_dir / "res.txt"
    write_hist_file(hist_file, all_hists)
    run_engine(exe, cfg, pool, hist_file, res_file, l1_fill)

    R = int(cfg["hw"].get("repeats", 16))
    counts = read_counts(res_file, len(all_hists), pool)
    vic = np.array([victim_from_counts(counts[i], R) for i in range(len(all_hists))], np.int64)
    ctrl_vic, diag_vic = vic[: len(ctrl)], vic[len(ctrl):]

    # --- control: hits-only, HW vs trusted LRU (does the miss-path detector work?) ---
    c_used = c_match = 0
    for lv, block in zip(ctrl_lru, ctrl_vic):
        if block < 0:
            continue
        c_used += 1
        c_match += int(block == lv)
    control_match = c_match / max(c_used, 1)

    # --- diagnostic: which policy the HW follows where LRU != SRRIP ---
    tallies = {p: 0 for p in CANDIDATE_POLICIES}
    hw_lru = hw_srrip = neither = used = undetected = 0
    rows = []
    for h, pred, block in zip(diag, preds, diag_vic):
        if block < 0:            # -1 none, -2 ambiguous (>1 block flipped)
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
        rows.append({"hist_len": len(h), "hw_block": int(block), "pred": pred})

    denom = max(used, 1)
    report = {
        "generator": "cpl.data.hw_discriminate",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "W": W, "pool": pool, "repeats": R, "len_range": list(len_range),
        "control": {"n": len(ctrl), "n_used": c_used,
                    "hw_matches_lru": round(control_match, 4)},
        "n_diagnostic": len(diag),
        "n_used": used, "n_undetected_or_noisy": undetected,
        "agreement_on_diagnostic": {p: round(tallies[p] / denom, 4) for p in CANDIDATE_POLICIES},
        "hw_matches_lru": round(hw_lru / denom, 4),
        "hw_matches_srrip": round(hw_srrip / denom, 4),
        "hw_matches_neither": round(neither / denom, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "report.json").write_text(json.dumps({**report, "samples": rows[:200]}, indent=2), encoding="utf-8")

    log.info("=" * 64)
    log.info("CONTROL (hits-only; miss-path detector must reproduce LRU if sound)")
    log.info("  HW matches LRU on %.1f%%  (%d/%d clean)   [sound if >~85%%]",
             100 * control_match, c_match, c_used)
    log.info("-" * 64)
    log.info("DIAGNOSTIC (histories where LRU != SRRIP)")
    log.info("  used %d / %d (dropped %d)", used, len(diag), undetected)
    for p in CANDIDATE_POLICIES:
        log.info("  HW matches %-6s on %5.1f%%", p, 100 * report["agreement_on_diagnostic"][p])
    log.info("  -> LRU %.1f%%  vs  SRRIP %.1f%%  (neither %.1f%%)",
             100 * report["hw_matches_lru"], 100 * report["hw_matches_srrip"],
             100 * report["hw_matches_neither"])
    log.info("=" * 64)
    if control_match < 0.85:
        log.warning("control LOW: the miss-path victim measurement is unreliable, so the "
                    "diagnostic numbers are not trustworthy (not a policy finding).")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="same config as hw_measure (for hw params + W)")
    ap.add_argument("--pool", type=int, default=24, help="number of distinct blocks (> W so some miss)")
    ap.add_argument("--n-diag", type=int, default=1200, help="diagnostic histories to measure")
    ap.add_argument("--len-min", type=int, default=16)
    ap.add_argument("--len-max", type=int, default=40,
                    help="longer histories => more misses => LRU/SRRIP diverge more")
    ap.add_argument("--n-control", type=int, default=400,
                    help="hits-only control histories (sanity check for the miss-path detector)")
    ap.add_argument("--l1-fill", type=int, default=16,
                    help="L1-drain addresses before probing (essential in miss mode; 0 disables)")
    ap.add_argument("--out", default="runs/hw_discriminate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    discriminate(load_config(args.config), args.pool, args.n_diag,
                 (args.len_min, args.len_max), Path(args.out), args.n_control, args.l1_fill)


if __name__ == "__main__":
    main()
