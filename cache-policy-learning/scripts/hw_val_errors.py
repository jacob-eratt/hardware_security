"""Break down hardware VAL-split errors of recipe C (and the plain GRU, for comparison).
Val split only; the independent test set is not touched."""
import sys
import numpy as np
import torch
from cpl.train_hits import load_streams
from cpl.model.slot_cache import build_slot_model
from cpl.model.hit_gru import build_hit_model

D = "../hardware_security_project/data/processed/hw_l2_celeron_stream_v2"
val = load_streams(D, "val")
A, P, M, L = val["A"], val["P"], val["M"] > 0, val["lens"]
lab = P > 0.5

ck = torch.load("runs/hw_celeron_capC/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
slot = build_slot_model({"d_way": a["d_way"], "d_glob": a["d_glob"], "use_ages": a["use_ages"]})
slot.load_state_dict(ck["model"]); slot.eval()
ckg = torch.load("runs/hit_celeron_v2/best.pt", map_location="cpu", weights_only=False)
gru = build_hit_model({"hidden": ckg["args"]["hidden"], "layers": ckg["args"]["layers"]})
gru.load_state_dict(ckg["model"]); gru.eval()


@torch.no_grad()
def predict(model, bs=1000):
    out = np.zeros(A.shape, bool)
    for s in range(0, len(A), bs):
        T = int(L[s:s + bs].max())
        out[s:s + bs, :T] = (model(torch.from_numpy(A[s:s + bs, :T])) > 0).numpy()
    return out


ps, pg = predict(slot), predict(gru)

# per-access features
B, T = A.shape
rd = np.full((B, T), -2)          # -1 = first touch
pos = np.tile(np.arange(T), (B, 1))
ndist = np.zeros((B, T), int)     # distinct lines seen so far (incl. this one)
for i in range(B):
    last, seen = {}, set()
    for t in range(L[i]):
        x = int(A[i, t]); seen.add(x); ndist[i, t] = len(seen)
        rd[i, t] = len(set(A[i, last[x] + 1:t].tolist())) if x in last else -1
        last[x] = t
conf = np.abs(2 * P - 1)

es, eg = (ps != lab) & M, (pg != lab) & M
n = M.sum()
print(f"val accesses={n:,}  streams={B:,}")
print(f"recipe C errors={es.sum():,} ({es.sum()/n:.4f})   GRU errors={eg.sum():,} ({eg.sum()/n:.4f})")
print(f"both wrong={(es&eg).sum():,}   only C wrong={(es&~eg).sum():,}   only GRU wrong={(eg&~es).sum():,}")

print("\n== error type (recipe C) ==")
print(f"  predicted MISS, label HIT : {(es & lab).sum():,}")
print(f"  predicted HIT,  label MISS: {(es & ~lab).sum():,}")

print("\n== label confidence of the access (|2p-1|; 1 = all 64 repeats agree) ==")
for lo, hi in [(0, .5), (.5, .9), (.9, .99), (.99, 1.01)]:
    sel = M & (conf >= lo) & (conf < hi)
    print(f"  conf [{lo:.2f},{hi:.2f}): accesses={sel.sum():>8,}  C err={es[sel].mean():.4f} ({es[sel].sum():>5,})  GRU err={eg[sel].mean():.4f}")

print("\n== by reuse distance (distinct other lines since last access to this line) ==")
for name, sel in [("first touch", rd == -1), ("0-3", (rd >= 0) & (rd <= 3)), ("4-7", (rd >= 4) & (rd <= 7)),
                  ("8-11", (rd >= 8) & (rd <= 11)), ("12-14", (rd >= 12) & (rd <= 14)), ("15", rd == 15),
                  ("16-17", (rd >= 16) & (rd <= 17)), ("18+", rd >= 18)]:
    sel = sel & M
    print(f"  {name:12s} accesses={sel.sum():>8,}  label hit-rate={lab[sel].mean():.3f}  "
          f"C err={es[sel].mean():.4f} ({es[sel].sum():>5,})  GRU err={eg[sel].mean():.4f}")

print("\n== first touches: label says HIT? (model always predicts miss: set starts empty) ==")
ft = (rd == -1) & M
print(f"  first touches={ft.sum():,}  labelled hit={(lab & ft).sum():,}  ({lab[ft].mean():.4f})")

print("\n== by position in stream ==")
for lo, hi in [(0, 16), (16, 32), (32, 48), (48, 64), (64, 96)]:
    sel = M & (pos >= lo) & (pos < hi)
    print(f"  t [{lo:>2},{hi:>2}): accesses={sel.sum():>8,}  C err={es[sel].mean():.4f} ({es[sel].sum():>5,})  GRU err={eg[sel].mean():.4f}")

print("\n== before vs after the set first needs an eviction (distinct lines > 16) ==")
for name, sel in [("<=16 distinct so far (no eviction yet)", ndist <= 16), (">16 distinct (evictions happening)", ndist > 16)]:
    sel = sel & M
    print(f"  {name:40s} accesses={sel.sum():>8,}  C err={es[sel].mean():.4f} ({es[sel].sum():>5,})  GRU err={eg[sel].mean():.4f}")

print("\n== errors per stream (recipe C) ==")
per = es.sum(1)
for k in [0, 1, 2, 3, 5, 10]:
    print(f"  streams with >= {k:>2} errors: {(per >= k).mean():.4f}" if k else f"  streams with 0 errors: {(per == 0).mean():.4f}")
print(f"  share of errors in the worst 10% of streams: {np.sort(per)[::-1][:B//10].sum()/max(per.sum(),1):.3f}")

print("\n== does the first error in a stream cascade? (errors after the first, per stream with >=1 error) ==")
first = np.where(per > 0, es.argmax(1), -1)
after = [es[i, first[i] + 1:L[i]].sum() for i in range(B) if first[i] >= 0]
print(f"  mean further errors after the first: {np.mean(after):.2f}   median: {np.median(after):.0f}")

print("\n== L1 filtering test: at the same reuse distance, does it matter whether this line's")
print("   PREVIOUS access was likely an L1 hit (its own reuse distance <= 5, L1D is 6-way)? ==")
prev_rd = np.full((B, T), -9)
for i in range(B):
    last = {}
    for t in range(L[i]):
        x = int(A[i, t])
        if x in last: prev_rd[i, t] = rd[i, last[x]]
        last[x] = t
for name, sel0 in [("rd 12-14", (rd >= 12) & (rd <= 14)), ("rd 15", rd == 15), ("rd 16-17", (rd >= 16) & (rd <= 17))]:
    for pn, ps_ in [("prev access L1-hit-likely (prev rd 0-5)", (prev_rd >= 0) & (prev_rd <= 5)),
                    ("prev access went to L2   (prev rd >5 or first)", (prev_rd > 5) | (prev_rd == -1))]:
        sel = sel0 & ps_ & M
        print(f"  {name:9s} {pn:48s} n={sel.sum():>6,}  label hit-rate={lab[sel].mean():.3f}  C err={es[sel].mean():.4f}")
