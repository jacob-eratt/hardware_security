"""Grade predictors on a simulated test set that carries full ground truth.

Metrics (all on the independent test set, models free-running):
  hit/miss acc (all)          -- fraction of accesses predicted correctly
  hit/miss acc (non-first)    -- excluding first touches, which always miss
  acc by reuse distance       -- reuse distance = distinct other lines since the
                                 previous access to this line (0-7 / 8-15 / 16+)
  exact-stream rate           -- fraction of streams with EVERY access right
  contents acc                -- after each access, is the predicted SET of resident
                                 lines exactly the true one? (way positions ignored:
                                 they are unobservable naming; only models that track
                                 contents report this)
  contents exact-stream rate  -- fraction of streams whose contents are right at every step

Baselines (fixed policies run as predictors, empty ways filled lowest-index first):
  tree-PLRU (independent recency implementation) -- MUST score 1.0 everywhere when the
                                                   data is tree-PLRU: a check on the
                                                   metric code itself
  LRU, FIFO, random replacement (mean of 3 seeds)

    python -m cpl.eval_sim --test-dir <sim test set> [--gru runs/x/best.pt] [--slot runs/y/best.pt] \
        --out runs/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpl.model.hit_gru import NUM_BLOCKS, build_hit_model
from cpl.model.slot_cache import build_slot_model


def load_truth(test_dir):
    df = pd.read_parquet(Path(test_dir) / "samples.parquet")
    meta = json.loads((Path(test_dir) / "meta.json").read_text())
    W = int(meta["W"])
    acc = [np.asarray(a, np.int64) for a in df["accesses"]]
    hit = [np.asarray(p) > 0.5 for p in df["p_hit"]]
    cont = [np.asarray(c).reshape(len(a), W) for a, c in zip(acc, df["contents"])]
    return acc, hit, cont, W, meta


def reuse_distance(a):
    last, rd = {}, np.full(len(a), -1)
    for t, b in enumerate(a.tolist()):
        if b in last:
            rd[t] = len(set(a[last[b] + 1:t].tolist()))
        last[b] = t
    return rd


# ---- baseline predictors: return (hits [T] bool, contents [T,W]) ----------------------
def _run(a, W, victim_fn, on_access=None):
    ways, hits, cont = [-1] * W, [], []
    st = {}
    for t, b in enumerate(a.tolist()):
        if b in ways:
            w = ways.index(b); hits.append(True)
        else:
            hits.append(False)
            w = ways.index(-1) if -1 in ways else victim_fn(ways, st, t)
            ways[w] = b
            st.setdefault("ins", {})[w] = t
        st.setdefault("use", {})[w] = t
        cont.append(list(ways))
    return np.array(hits), np.array(cont)


def tree_plru(a, W):
    def victim(ways, st, t):
        use = st["use"]; lo, hi = 0, W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if max(use.get(w, -1) for w in range(lo, mid)) > max(use.get(w, -1) for w in range(mid, hi)):
                lo = mid
            else:
                hi = mid
        return lo
    return _run(a, W, victim)


def lru(a, W):
    return _run(a, W, lambda ways, st, t: min(range(W), key=lambda w: st["use"][w]))


def fifo(a, W):
    return _run(a, W, lambda ways, st, t: min(range(W), key=lambda w: st["ins"][w]))


def random_repl(a, W, rng):
    return _run(a, W, lambda ways, st, t: int(rng.integers(W)))


def metrics(pred_hits, truth_hits, acc, pred_cont=None, truth_cont=None):
    ph, th = np.concatenate(pred_hits), np.concatenate(truth_hits)
    rd = np.concatenate([reuse_distance(a) for a in acc])
    ok = ph == th
    r = {"acc_all": float(ok.mean()),
         "acc_non_first": float(ok[rd >= 0].mean()),
         "acc_rd_0_7": float(ok[(rd >= 0) & (rd <= 7)].mean()),
         "acc_rd_8_15": float(ok[(rd >= 8) & (rd <= 15)].mean()),
         "acc_rd_16_plus": float(ok[rd >= 16].mean()),
         "exact_stream": float(np.mean([(p == t).all() for p, t in zip(pred_hits, truth_hits)]))}
    if pred_cont is not None:
        per_step, per_stream = [], []
        for pc, tc in zip(pred_cont, truth_cont):
            same = [set(x[x >= 0].tolist()) == set(y[y >= 0].tolist()) for x, y in zip(pc, tc)]
            per_step += same; per_stream.append(all(same))
        r["contents_acc"] = float(np.mean(per_step))
        r["contents_exact_stream"] = float(np.mean(per_stream))
    return r


@torch.no_grad()
def model_predict(model, acc, bs=1000, contents=False):
    model.eval()
    T = max(map(len, acc)); hits, conts = [], []
    for s in range(0, len(acc), bs):
        chunk = acc[s:s + bs]
        A = torch.full((len(chunk), T), NUM_BLOCKS, dtype=torch.long)
        for i, a in enumerate(chunk):
            A[i, :len(a)] = torch.from_numpy(a)
        if contents:
            lg, C = model(A, return_contents=True)
        else:
            lg, C = model(A), None
        for i, a in enumerate(chunk):
            hits.append((lg[i, :len(a)] > 0).numpy())
            if contents:
                conts.append(C[i, :len(a)].numpy())
    return hits, (conts if contents else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--gru", default=None, help="plain GRU checkpoint (cpl.train_hits)")
    ap.add_argument("--slot", default=None, help="slot model checkpoint (cpl.train_hindsight)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    acc, th, tc, W, meta = load_truth(a.test_dir)
    rows = {}
    if meta.get("policy") == "plru":
        h, c = zip(*[tree_plru(x, W) for x in acc]); rows["tree-PLRU (truth check)"] = metrics(h, th, acc, c, tc)
    for name, fn in [("LRU", lru), ("FIFO", fifo)]:
        h, c = zip(*[fn(x, W) for x in acc]); rows[name] = metrics(h, th, acc, c, tc)
    rs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        h, c = zip(*[random_repl(x, W, rng) for x in acc]); rs.append(metrics(h, th, acc, c, tc))
    rows["random replacement"] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    if a.gru:
        ck = torch.load(a.gru, map_location="cpu", weights_only=False)
        m = build_hit_model({"hidden": ck["args"]["hidden"], "layers": ck["args"]["layers"]})
        m.load_state_dict(ck["model"]); h, _ = model_predict(m, acc)
        rows["plain GRU"] = metrics(h, th, acc)
    if a.slot:
        ck = torch.load(a.slot, map_location="cpu", weights_only=False)
        m = build_slot_model({"d_way": ck["args"]["d_way"], "d_glob": ck["args"]["d_glob"],
                          "use_ages": ck["args"].get("use_ages", False)})
        m.load_state_dict(ck["model"]); h, c = model_predict(m, acc, contents=True)
        rows["three-unit model"] = metrics(h, th, acc, c, tc)
    keys = ["acc_all", "acc_non_first", "acc_rd_0_7", "acc_rd_8_15", "acc_rd_16_plus",
            "exact_stream", "contents_acc", "contents_exact_stream"]
    print(f"{'predictor':<26}" + "".join(f"{k:>15}" for k in keys))
    for name, r in rows.items():
        print(f"{name:<26}" + "".join(f"{r[k]:>15.4f}" if k in r else f"{'-':>15}" for k in keys))
    if a.out:
        Path(a.out).write_text(json.dumps({"test_dir": str(a.test_dir), "meta": meta, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
