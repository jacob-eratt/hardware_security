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

``--observed``: observed-outcome prediction (cpl.observed) instead of free-running. Every
predictor, baselines included, predicts access t from residency corrected by the true
outcomes of accesses < t. Contents metrics are then after corrections. ``--split`` grades
only one split of a dataset (e.g. ``val`` of a training set, for diagnostics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpl.model.hit_gru import NUM_BLOCKS, gru_from_ckpt
from cpl.model.slot_cache import slot_from_ckpt
from cpl.observed import ObservedChooser


def load_truth(test_dir, split=None):
    """Per stream: line ids, true hits, true contents of the ACCESSED set after each
    access [T, W], and the global set of each access (all 0 for single-set data)."""
    df = pd.read_parquet(Path(test_dir) / "samples.parquet")
    meta = json.loads((Path(test_dir) / "meta.json").read_text())
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    W = int(meta["W"])
    acc = [np.asarray(a, np.int64) for a in df["accesses"]]
    hit = [np.asarray(p) > 0.5 for p in df["p_hit"]]
    cont = [np.asarray(c).reshape(len(a), W) for a, c in zip(acc, df["contents"])]
    sets = ([np.asarray(x, np.int64) for x in df["sets"]] if "sets" in df.columns
            else [np.zeros(len(a), np.int64) for a in acc])
    return acc, hit, cont, W, meta, sets


def reuse_distance(a, s=None):
    """Distinct other lines OF THE SAME SET since the previous access to this line
    (-1 = first touch). ``s`` = set per access (default: one set)."""
    if s is None:
        s = np.zeros(len(a), np.int64)
    last, rd = {}, np.full(len(a), -1)
    for t, b in enumerate(a.tolist()):
        if b in last:
            seg = slice(last[b] + 1, t)
            rd[t] = len(set(a[seg][s[seg] == s[t]].tolist()))
        last[b] = t
    return rd


# ---- baseline predictors: return (hits [T] bool, contents of the accessed set [T,W]) --
def _run(a, W, victim_fn, sets=None, obs=None):
    """Each set (``sets``, default one set) runs its own copy of the policy. ``obs`` (true
    hits) switches on observed-outcome correction (cpl.observed): a true miss on a line
    the policy holds re-fetches it into the same way; a true hit on a line it lacks
    inserts it like a miss."""
    if sets is None:
        sets = np.zeros(len(a), np.int64)
    ways_of, st_of, hits, cont = {}, {}, [], []
    for t, (b, s) in enumerate(zip(a.tolist(), sets.tolist())):
        ways = ways_of.setdefault(s, [-1] * W); st = st_of.setdefault(s, {})
        if b in ways:
            w = ways.index(b); hits.append(True)
            if obs is not None and not obs[t]:
                st.setdefault("ins", {})[w] = t
        else:
            hits.append(False)
            w = ways.index(-1) if -1 in ways else victim_fn(ways, st, t)
            ways[w] = b
            st.setdefault("ins", {})[w] = t
        st.setdefault("use", {})[w] = t
        cont.append(list(ways))
    return np.array(hits), np.array(cont)


def tree_plru(a, W, sets=None, obs=None):
    def victim(ways, st, t):
        use = st["use"]; lo, hi = 0, W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if max(use.get(w, -1) for w in range(lo, mid)) > max(use.get(w, -1) for w in range(mid, hi)):
                lo = mid
            else:
                hi = mid
        return lo
    return _run(a, W, victim, sets, obs)


def lru(a, W, sets=None, obs=None):
    return _run(a, W, lambda ways, st, t: min(range(W), key=lambda w: st["use"][w]), sets, obs)


def fifo(a, W, sets=None, obs=None):
    return _run(a, W, lambda ways, st, t: min(range(W), key=lambda w: st["ins"][w]), sets, obs)


def random_repl(a, W, rng, sets=None, obs=None):
    return _run(a, W, lambda ways, st, t: int(rng.integers(W)), sets, obs)


def metrics(pred_hits, truth_hits, acc, pred_cont=None, truth_cont=None, sets=None):
    ph, th = np.concatenate(pred_hits), np.concatenate(truth_hits)
    rd = np.concatenate([reuse_distance(a, None if sets is None else s) for a, s in zip(acc, sets or [None] * len(acc))])
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


def _local_slots(s):
    _, first = np.unique(s, return_index=True)
    order = {int(v): k for k, v in enumerate(s[np.sort(first)])}       # slot = order of first touch
    return np.array([order[int(v)] for v in s], np.int64)


@torch.no_grad()
def model_predict(model, acc, bs=1000, contents=False, sets=None, pad=NUM_BLOCKS, obs=None, stats=None):
    """Free-running predictions. ``sets`` (global set per access) is passed to the
    model as local slots + set ids; ``pad`` must be the model's num_blocks."""
    model.eval()
    T = max(map(len, acc)); hits, conts = [], []
    for s in range(0, len(acc), bs):
        chunk = acc[s:s + bs]
        A = torch.full((len(chunk), T), pad, dtype=torch.long)
        for i, a in enumerate(chunk):
            A[i, :len(a)] = torch.from_numpy(a)
        kw = {}
        if sets is not None:
            Sl = torch.zeros((len(chunk), T), dtype=torch.long); G = torch.zeros((len(chunk), T), dtype=torch.long)
            for i, g in enumerate(sets[s:s + bs]):
                Sl[i, :len(g)] = torch.from_numpy(_local_slots(g)); G[i, :len(g)] = torch.from_numpy(g)
            kw = {"sets": Sl, "set_ids": G}
        if obs is not None:                     # observed-outcome prediction (slot model only)
            H = np.zeros(A.shape, bool); Mk = np.zeros(A.shape, bool)
            for i, o in enumerate(obs[s:s + bs]):
                H[i, :len(o)] = o; Mk[i, :len(o)] = True
            ch = ObservedChooser(A.numpy(), H, Mk, pad); kw["chooser"] = ch
        if contents:
            lg, C = model(A, return_contents=True, **kw)
        else:
            lg, C = model(A, **kw), None
        if obs is not None and stats is not None:
            stats["corrections"] = stats.get("corrections", 0) + ch.n_corr
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
    ap.add_argument("--observed", action="store_true", help="observed-outcome prediction (cpl.observed)")
    ap.add_argument("--split", default=None, help="grade only this split (e.g. val)")
    a = ap.parse_args(argv)
    acc, th, tc, W, meta, st = load_truth(a.test_dir, a.split)
    ob = th if a.observed else None
    print(f"[eval] {len(acc)} streams, mode={'observed outcomes' if a.observed else 'free-running'}"
          + (f", split={a.split}" if a.split else ""))
    ms = st if "sets_range" in meta else None   # set info goes to models only for multi-set data
    rows = {}
    if meta.get("policy") == "plru":
        h, c = zip(*[tree_plru(x, W, s, o) for x, s, o in zip(acc, st, ob or [None] * len(acc))]); rows["tree-PLRU (truth check)"] = metrics(h, th, acc, c, tc, st)
    for name, fn in [("LRU", lru), ("FIFO", fifo)]:
        h, c = zip(*[fn(x, W, s, o) for x, s, o in zip(acc, st, ob or [None] * len(acc))]); rows[name] = metrics(h, th, acc, c, tc, st)
    rs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        h, c = zip(*[random_repl(x, W, rng, s, o) for x, s, o in zip(acc, st, ob or [None] * len(acc))]); rs.append(metrics(h, th, acc, c, tc, st))
    rows["random replacement"] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    if a.gru and a.observed:
        raise SystemExit("--observed is not implemented for the plain GRU (it has no residency to correct)")
    if a.gru:
        ck = torch.load(a.gru, map_location="cpu", weights_only=False)
        m = gru_from_ckpt(ck); h, _ = model_predict(m, acc, sets=ms, pad=m.pad)
        rows["plain GRU"] = metrics(h, th, acc, sets=st)
    if a.slot:
        ck = torch.load(a.slot, map_location="cpu", weights_only=False)
        stats = {}
        m = slot_from_ckpt(ck); h, c = model_predict(m, acc, contents=True, sets=ms, pad=m.N, obs=ob, stats=stats)
        rows["three-unit model"] = metrics(h, th, acc, c, tc, st)
        if a.observed:
            rows["three-unit model"]["corrections_per_access"] = stats["corrections"] / sum(map(len, acc))
    keys = ["acc_all", "acc_non_first", "acc_rd_0_7", "acc_rd_8_15", "acc_rd_16_plus",
            "exact_stream", "contents_acc", "contents_exact_stream"]
    print(f"{'predictor':<26}" + "".join(f"{k:>15}" for k in keys))
    for name, r in rows.items():
        print(f"{name:<26}" + "".join(f"{r[k]:>15.4f}" if k in r else f"{'-':>15}" for k in keys))
    if a.out:
        Path(a.out).write_text(json.dumps({"test_dir": str(a.test_dir), "split": a.split, "observed": a.observed,
                                           "meta": meta, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
