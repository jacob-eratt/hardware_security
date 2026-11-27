"""Diagnose a slot model's eviction decisions against simulator ground truth.

Runs on the VALIDATION split (never the test set) of a simulated dataset that carries
full ground truth (cpl.data.sim_stream_dataset). The model runs free (no labels).

Because model and simulator both fill empty ways lowest-index first, their way
layouts are identical until the model's first wrong eviction. So for every decision
taken from the SAME state (contents equal way-for-way), we can say whether the model
evicted the same way, and for tree-PLRU, at which tree level it went wrong:
  level 0 = wrong half, 1 = wrong quarter, 2 = wrong pair, 3 = wrong way within the pair.

Reports:
  * same-state decision accuracy, overall and by eviction index (how many full-set
    misses into the stream the decision is)
  * where the FIRST wrong decision of each stream falls (eviction index, tree level)
  * recency rank of the true vs chosen victim at first errors (0 = most recent)

    python -m cpl.analyze_sim --data-dir <sim dataset> --slot runs/x/best.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpl.eval_sim import model_predict
from cpl.model.slot_cache import slot_from_ckpt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.split == "test":
        raise SystemExit("diagnostics must not use the test split")
    meta = json.loads((Path(a.data_dir) / "meta.json").read_text()); W = int(meta["W"])
    df = pd.read_parquet(Path(a.data_dir) / "samples.parquet")
    df = df[df["split"] == a.split].reset_index(drop=True)
    acc = [np.asarray(x, np.int64) for x in df["accesses"]]
    tc = [np.asarray(c).reshape(len(x), W) for x, c in zip(acc, df["contents"])]
    tfill = [np.asarray(f) for f in df["filled_way"]]
    tev = [np.asarray(e) for e in df["evicted"]]

    ck = torch.load(a.slot, map_location="cpu", weights_only=False)
    if "sets_range" in meta:
        raise SystemExit("analyze_sim is single-set only (tree-level diagnostics)")
    m = slot_from_ckpt(ck)
    _, mc = model_predict(m, acc, contents=True)

    by_idx = {}                                   # eviction index -> [correct, total]
    first = []                                    # (eviction index, level, true rank, chosen rank)
    naming_mismatch = 0
    for a_, T_, F_, E_, M_ in zip(acc, tc, tfill, tev, mc):
        prev_t = np.full(W, -1); prev_m = np.full(W, -1); k = 0; last = {}
        diverged = False
        for t in range(len(a_)):
            b = int(a_[t])
            if E_[t] >= 0:                                         # full-set miss in truth
                if not diverged and (prev_t == prev_m).all():
                    tw = int(F_[t])
                    changed = np.nonzero(M_[t] != prev_m)[0]
                    mw = int(changed[0]) if len(changed) else -1
                    ok = mw == tw
                    c = by_idx.setdefault(k, [0, 0]); c[0] += ok; c[1] += 1
                    if not ok:
                        level = int(np.log2(W)) - 1 - int(np.floor(np.log2(tw ^ mw))) if mw >= 0 else -1
                        res = sorted((x for x in prev_t.tolist() if x >= 0), key=lambda x: -last[x])
                        first.append((k, level, res.index(int(prev_t[tw])),
                                      res.index(int(prev_m[mw])) if mw >= 0 else -1))
                        diverged = True
                k += 1
            last[b] = t
            if not diverged and set(T_[t].tolist()) == set(M_[t].tolist()) and not (T_[t] == M_[t]).all():
                naming_mismatch += 1
            prev_t, prev_m = T_[t], M_[t]

    tot = sum(v[1] for v in by_idx.values()); cor = sum(v[0] for v in by_idx.values())
    n = len(acc)
    print(f"streams={n:,}  same-state decisions={tot:,}  decision acc={cor / tot:.4f}  "
          f"streams with a wrong decision={len(first):,} ({len(first) / n:.3f})  naming mismatches={naming_mismatch}")
    print("\ndecision accuracy by eviction index (k-th full-set miss in the stream):")
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 999)]:
        c = [by_idx[k] for k in by_idx if lo <= k <= hi]
        if c:
            cc, tt = sum(x[0] for x in c), sum(x[1] for x in c)
            print(f"  {lo:>3}-{hi:<3}  acc={cc / tt:.4f}  decisions={tt:,}")
    levels = Counter(f[1] for f in first)
    names = {0: "wrong half (root)", 1: "wrong quarter", 2: "wrong pair", 3: "wrong way in right pair"}
    print("\nfirst wrong decision per stream, by tree level:")
    for lv in sorted(levels):
        print(f"  {names.get(lv, lv):<26} {levels[lv]:>6}  ({levels[lv] / len(first):.3f})")
    ks = Counter(min(f[0], 32) for f in first)
    print("\nfirst wrong decision, eviction index:", dict(sorted(ks.items())))
    tr = Counter(f[2] for f in first); cr = Counter(f[3] for f in first)
    print("recency rank of TRUE victim at first error  :", dict(sorted(tr.items())))
    print("recency rank of CHOSEN victim at first error:", dict(sorted(cr.items())))
    if a.out:
        Path(a.out).write_text(json.dumps({
            "split": a.split, "decision_acc": cor / tot, "decisions": tot,
            "streams_with_error": len(first), "streams": n,
            "acc_by_eviction_index": {str(k): v[0] / v[1] for k, v in sorted(by_idx.items())},
            "first_error_levels": {names.get(k, str(k)): v for k, v in levels.items()},
            "first_error_true_rank": tr, "first_error_chosen_rank": cr}, indent=2, default=int))


if __name__ == "__main__":
    main()
