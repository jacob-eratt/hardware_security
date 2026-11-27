"""DRRIP val-only error diagnostic for runs/sim_drrip_ms_capC (free-running model).

Per access: error rate by set role x follower mode x hit/miss truth, error direction.
Per stream: the FIRST step where the model's contents of the accessed set diverge from
the truth (later errors cascade from it), classified by role, mode, and what kind of
victim mistake it was, using the true RRPV state just before the access.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from cpl.eval_sim import model_predict  # noqa: E402
from cpl.model.slot_cache import slot_from_ckpt  # noqa: E402

D = "../hardware_security_project/data/processed/sim_drrip_ms"
df = pd.read_parquet(f"{D}/samples.parquet")
df = df[df["split"] == "val"].reset_index(drop=True)
W = 16
acc = [np.asarray(a, np.int64) for a in df["accesses"]]
sets = [np.asarray(s, np.int64) for s in df["sets"]]
hit = [np.asarray(p) > 0.5 for p in df["p_hit"]]
cont = [np.asarray(c).reshape(len(a), W) for a, c in zip(acc, df["contents"])]
rrpv = [np.asarray(c).reshape(len(a), W) for a, c in zip(acc, df["policy_state"])]
role = [np.asarray(r) for r in df["set_role"]]
psel = [np.asarray(p, np.int64) for p in df["psel"]]
ev = [np.asarray(e, np.int64) for e in df["evicted"]]
fw = [np.asarray(f, np.int64) for f in df["filled_way"]]
print(f"val streams: {len(df)}, accesses: {sum(map(len, acc))}")

ck = torch.load("runs/sim_drrip_ms_capC/best.pt", map_location="cpu", weights_only=False)
torch.set_num_threads(4)
m = slot_from_ckpt(ck)
ph, pc = model_predict(m, acc, contents=True, sets=sets, pad=m.N)

ROLE = {0: "follower", 1: "SRRIP leader", 2: "BRRIP leader"}


def mode_of(i, t):
    """Insertion mode in force at access t (follower: from PSEL before the access)."""
    r = int(role[i][t])
    if r == 1:
        return "SRRIP"
    if r == 2:
        return "BRRIP"
    p = psel[i][t - 1] if t > 0 else psel[i][0]
    return "BRRIP" if p >= 512 else "SRRIP"


# ---- per-access errors --------------------------------------------------------------
tab = Counter(); err = Counter()
first_touch_rows = 0
for i in range(len(acc)):
    seen = set()
    for t in range(len(acc[i])):
        b = int(acc[i][t])
        if b not in seen:                       # first touch: always a miss, free
            seen.add(b); continue
        key = (ROLE[int(role[i][t])], mode_of(i, t), "hit" if hit[i][t] else "miss")
        tab[key] += 1
        if ph[i][t] != hit[i][t]:
            err[key] += 1
print("\n== per-access errors (non-first touches) ==")
print(f"{'role':14s} {'mode':6s} {'truth':5s} {'n':>8s} {'err':>7s} {'rate':>7s}")
for k in sorted(tab):
    print(f"{k[0]:14s} {k[1]:6s} {k[2]:5s} {tab[k]:8d} {err[k]:7d} {err[k] / tab[k]:7.4f}")
for mo in ("SRRIP", "BRRIP"):
    n = sum(v for k, v in tab.items() if k[1] == mo); e = sum(v for k, v in err.items() if k[1] == mo)
    print(f"  all roles, mode {mo}: n={n} err rate={e / n:.4f}")
print("  (truth=hit errors: model evicted a line the cache kept; truth=miss: model kept a line the cache evicted)")

# ---- first divergence per stream ------------------------------------------------------
fd = Counter(); n_div = 0; at_frac = []
kind = Counter(); rare_before = Counter(); div_mode_streams = Counter(); stream_mode = Counter()
for i in range(len(acc)):
    T = len(acc[i])
    m0 = "BRRIP" if psel[i][0] >= 512 else "SRRIP"
    stream_mode[m0] += 1
    same = [set(x[x >= 0].tolist()) == set(y[y >= 0].tolist()) for x, y in zip(pc[i], cont[i])]
    if all(same):
        continue
    t = same.index(False); n_div += 1; at_frac.append(t / T)
    div_mode_streams[m0] += 1
    s = int(sets[i][t]); mo = mode_of(i, t)
    fd[(ROLE[int(role[i][t])], mo)] += 1
    prev = [u for u in range(t) if sets[i][u] == s]
    if hit[i][t] or not prev:
        kind["not a full-set miss (unexpected)"] += 1; continue
    u = prev[-1]
    before = cont[i][u]; r_before = rrpv[i][u]
    true_v = int(ev[i][t]); model_now = set(pc[i][t][pc[i][t] >= 0].tolist())
    model_v = [x for x in before.tolist() if x >= 0 and x not in model_now]
    if true_v < 0 or len(model_v) != 1:
        kind[f"other (true_v={true_v}, model victims={len(model_v)})"] += 1; continue
    model_v = model_v[0]
    rv = {int(x): int(r) for x, r in zip(before, r_before)}
    mx = max(rv.values())
    if rv[model_v] == mx:
        kind["tie: model's victim had the max RRPV too (wrong way among the oldest)"] += 1
    else:
        kind[f"model's victim RRPV {rv[model_v]} < max {mx} (did not track RRPVs)"] += 1
    # did this set see a rare (RRPV 2) insertion while in BRRIP mode before t?
    rare = any((not hit[i][v]) and mode_of(i, v) == "BRRIP" and rrpv[i][v][fw[i][v]] == 2
               for v in prev)
    rare_before[("BRRIP" if mo == "BRRIP" else "SRRIP", rare)] += 1

print(f"\n== first divergence of contents: {n_div}/{len(acc)} streams diverge "
      f"(contents exact-stream {1 - n_div / len(acc):.4f}) ==")
print("  stream start mode -> fraction of streams that diverge:")
for k in sorted(stream_mode):
    print(f"    {k}: {div_mode_streams[k]}/{stream_mode[k]} = {div_mode_streams[k] / stream_mode[k]:.3f}")
print(f"  position of first divergence (fraction of stream): median {np.median(at_frac):.2f}")
print("  role/mode at first divergence:")
for k, v in fd.most_common():
    print(f"    {k[0]:14s} {k[1]:6s} {v:5d}  ({v / n_div:.3f})")
print("  kind of victim mistake:")
for k, v in kind.most_common():
    print(f"    {v:5d}  ({v / n_div:.3f})  {k}")
print("  was there an earlier rare RRPV-2 BRRIP insertion in that set? (mode at divergence, rare)")
for k, v in sorted(rare_before.items()):
    print(f"    {k}: {v}")
