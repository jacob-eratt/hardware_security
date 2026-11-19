# Hardware measurement (eviction sets + eviction labeling)

This is the hardware counterpart of the NumPy simulator in
`src/cpl/policies/`. Instead of computing the victim from an assumed policy, it
**observes** which way a real cache set evicts, so the GRU can be trained on
evictions measured from the machine itself (e.g. the Intel Celeron L2).

Only memory from one buffer we allocate ourselves is ever accessed; this
reverse-engineers the policy of a machine we own and mounts no attack (see the
report's "Reverse-Engineering Setting").

## Files
- `cacheprobe.c` — the measurement engine: finds a congruent-address pool (an
  eviction set) for one cache set by timing, then for each access history runs
  R fill/replay/trigger/probe trials and reports the evicted way per trial.
- `../src/cpl/data/hw_measure.py` — Python driver: generates histories with the
  shared split-stable generator, runs the engine, and writes
  `samples.parquet` + `meta.json` in the **same schema the simulator uses**, so
  the trainer consumes hardware and simulated data identically.

## Build
```
gcc -O2 -march=native -o cacheprobe.exe cacheprobe.c
```
The Python driver compiles it automatically if missing (it finds MSYS2 gcc).

## Run
```
# full pipeline (build dataset from real measurements)
python -m cpl.data.hw_measure --config configs/hw_l2_celeron.yaml --scale 0.01

# self-test: replace measurement with a software tree-PLRU oracle.
#   set hw.sim: true in the config. A --sim dataset MUST have
#   agreement_with_label[plru] == 1.0, which checks the whole pipeline
#   (tokenization, counts, splits, schema) without any hardware.
```

## Tuning on the target machine (important)
Cycle-accurate eviction measurement needs a quiet environment. The engine
prints calibration and eviction-set diagnostics to stderr; use them:

- **`--ways W`** must equal the real L2 associativity. Check with
  `Get-CimInstance Win32_CacheMemory` or CPUID leaf 4.
- **Pin to an idle core** (`hw.core`). Close background load. On an SMT part the
  sibling logical core shares the L2 and adds noise — prefer a part/core without
  an active sibling.
- **`--buf-mb`** larger makes a full eviction set easier to find.
- **`--repeats`** more trials per history → cleaner `victim_counts` under noise.
- **`--min-detect`** is the victim-detection rate required when validating the
  eviction set (default 0.6). The engine prints the rate it achieves; if it
  cannot reach the target it retries with fresh candidates, then errors.
- **`--force`** proceeds with the best-effort set even below `--min-detect`
  (expect noisier labels) — useful while tuning.
- **`--l1-fill N`** accesses N extra lines before probing to push the set out of
  the (smaller) L1 so the probe reads L2 residency.
- **`--threshold T`** overrides the auto hit/miss threshold (in TSC cycles) if
  the calibrated `hit`/`miss` distributions printed at startup overlap.

The `victim_counts` field holds the per-trial outcomes, so measurement noise
becomes soft labels exactly as in the simulator's noise model (spec §6.3, §8.1).
