"""Train the slot cache model with HINDSIGHT victim candidates (no victim labels needed).

The hardware only gives hit/miss per access, but those labels constrain evictions:
  * line x is accessed again and HITS   -> x was not evicted in between (protected)
  * line x is accessed again and MISSES -> x was evicted somewhere in between (must go,
                                           before that next access = its deadline)
  * line x is never accessed again, or a way is empty / holds a pre-stream line
                                         -> evicting it is free
At each miss the model may only evict an *allowed* way: protected lines are excluded,
and a choice is allowed only if every remaining deadline can still be met (each miss
evicts one line; earliest-deadline-first check). The model picks its best-scoring
allowed way (Gumbel exploration on non-empty ways early on), so training residency
always agrees with the labels, and the loss is the partial-label loss
-log P(victim in allowed set). Empty ways are filled lowest-index first (the model's
fill-order naming, cpl.model.slot_cache); such fills are not a learned decision and
carry no loss. If a canonical fill is ever disallowed by the labels it is counted as
an inconsistency (on simulated data the count MUST be 0).

Evaluation is the real test and uses no labels: the model runs free (hard victims)
and is scored on hit/miss exactly like cpl.train_hits.

    python -m cpl.train_hindsight --data-dir <stream dataset> --test-dir <independent set> \
        --epochs 15 --run-dir runs/hindsight
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cpl.model.hit_gru import NUM_BLOCKS
from cpl.model.slot_cache import build_slot_model
from cpl.train_hits import base_rate_baseline, evaluate, iterate, load_streams
from cpl.utils.seed import set_seed

N = NUM_BLOCKS
FAR = 1 << 30


def next_access(A: np.ndarray, L: np.ndarray) -> np.ndarray:
    """nxt[i, t] = index of the next access to the same line after t (FAR if none)."""
    B, T = A.shape
    nxt = np.full((B, T), FAR, np.int64)
    for i in range(B):
        seen = {}
        for t in range(int(L[i]) - 1, -1, -1):
            b = int(A[i, t])
            nxt[i, t] = seen.get(b, FAR)
            seen[b] = t
    return nxt


def iterate_with_victims(data, bs, rng, with_victims):
    """Like cpl.train_hits.iterate (shuffled), also yielding true victim ways when asked."""
    idx = rng.permutation(len(data["lens"]))
    for s in range(0, len(idx), bs):
        j = idx[s : s + bs]
        T = int(data["lens"][j].max())
        F = data["F"][j, :T] if with_victims else None
        yield (torch.from_numpy(data["A"][j, :T]), torch.from_numpy(data["P"][j, :T]),
               torch.from_numpy(data["M"][j, :T]), F)


class HindsightChooser:
    """Per-batch victim chooser + partial-label loss, called by the model each step."""

    def __init__(self, A, P, M, explore: float, conf_weight: bool):
        self.A = A.numpy(); self.M = M.numpy() > 0
        self.lab = P.numpy() > 0.5                                    # majority label
        self.conf = np.abs(2 * P.numpy() - 1) if conf_weight else np.ones_like(self.lab, float)
        self.L = self.M.sum(1)
        self.nxt = next_access(self.A, self.L)
        miss = (~self.lab) & self.M
        self.cm = np.cumsum(miss, 1)                                  # misses in [0, t]
        B = self.A.shape[0]
        self.last = np.full((B, N + 1), -1, np.int64)                 # last access time per line
        self.explore = explore
        self.losses, self.n_dec = [], 0
        self.n_incons = 0

    def _misses_between(self, i, t, d):                               # misses in (t, d)
        return int(self.cm[i, d - 1] - self.cm[i, t]) if d - 1 > t else 0

    def _allowed(self, i, t, lines):
        W = len(lines)
        free, must = [], []                                           # must: (deadline, way)
        for w in range(W):
            x = int(lines[w])
            if x == N or self.last[i, x] < 0:
                free.append(w); continue
            d = int(self.nxt[i, self.last[i, x]])
            if d >= FAR or d >= self.L[i]:
                free.append(w)
            elif not self.lab[i, d]:
                must.append((d, w))
            # else: next access hits -> protected
        must.sort()

        def feasible(ms):
            return all(k + 1 <= self._misses_between(i, t, d) for k, (d, _) in enumerate(ms))

        if feasible(must):
            return free + [w for _, w in must]
        return [w for j, (_, w) in enumerate(must) if feasible(must[:j] + must[j + 1:])]

    def __call__(self, t, way_lines, score):
        B, W = score.shape
        lines = way_lines.cpu().numpy()
        v = np.zeros((B, W), np.float32); ph = np.zeros(B, np.float32)
        allowed = np.zeros((B, W), bool); use = np.zeros(B, np.float32)
        sc = score.detach().cpu().numpy()
        if self.explore > 0:
            sc = sc + self.explore * -np.log(-np.log(np.random.rand(B, W).clip(1e-9, 1 - 1e-9)))
        for i in range(B):
            if not self.M[i, t]:
                continue
            b = int(self.A[i, t])
            where = np.nonzero(lines[i] == b)[0]
            if self.lab[i, t]:
                if len(where):
                    ph[i] = 1.0                                       # consistent hit
                else:                                                 # label hit, model says absent
                    self.n_incons += 1
                    w = int(np.argmax(sc[i])); v[i, w] = 1.0
            else:
                if len(where):                                        # label miss, model says present
                    self.n_incons += 1
                    v[i, int(where[0])] = 1.0                         # re-fetch into the same way
                elif (lines[i] == N).any():                           # fill-order naming
                    w = int(np.nonzero(lines[i] == N)[0][0])
                    if w not in self._allowed(i, t, lines[i]):
                        self.n_incons += 1
                    v[i, w] = 1.0
                else:
                    al = self._allowed(i, t, lines[i])
                    if not al:
                        self.n_incons += 1
                        al = list(range(W))
                    elif len(al) < W:
                        allowed[i, al] = True; use[i] = self.conf[i, t]
                    w = al[int(np.argmax(sc[i, al]))]; v[i, w] = 1.0
            self.last[i, b] = t
        if use.any():
            al = torch.from_numpy(allowed).to(score.device)
            u = torch.from_numpy(use).to(score.device)
            lp = torch.logsumexp(score.masked_fill(~al, -1e9), -1) - torch.logsumexp(score, -1)
            self.losses.append(-(lp * u).sum()); self.n_dec += float(use.sum())
        dev = score.device
        return torch.from_numpy(v).to(dev), torch.from_numpy(ph).to(dev)


class VictimLabelChooser:
    """CAPACITY TESTS ON SIMULATED DATA ONLY: supervise each full-set miss with the
    simulator's true victim way (``filled_way``). Residency then follows the true
    history exactly (fill-order naming makes model ways = simulator ways). Loss:
    cross-entropy of the score over ways against the true way."""

    def __init__(self, P, M, F):
        self.lab = P.numpy() > 0.5; self.M = M.numpy() > 0; self.F = F
        self.losses, self.n_dec, self.n_incons = [], 0, 0

    def __call__(self, t, way_lines, score):
        B, W = score.shape
        lines = way_lines.cpu().numpy()
        v = np.zeros((B, W), np.float32); ph = np.zeros(B, np.float32)
        tgt = np.full(B, -1, np.int64)
        for i in range(B):
            if not self.M[i, t]:
                continue
            if self.lab[i, t]:
                ph[i] = 1.0
                continue
            w = int(self.F[i, t])
            if (lines[i] == N).any():                     # empty fill: not a decision
                if w != int(np.nonzero(lines[i] == N)[0][0]):
                    self.n_incons += 1
            else:
                tgt[i] = w
            v[i, w] = 1.0
        sel = tgt >= 0
        if sel.any():
            st = torch.from_numpy(sel); tg = torch.from_numpy(tgt[sel])
            self.losses.append(torch.nn.functional.cross_entropy(score[st], tg, reduction="sum"))
            self.n_dec += int(sel.sum())
        return torch.from_numpy(v), torch.from_numpy(ph)


def load_filled_way(data_dir, split):
    """filled_way per stream (simulated datasets only), padded with -1, aligned with load_streams."""
    import pandas as pd
    df = pd.read_parquet(Path(data_dir) / "samples.parquet")
    if "filled_way" not in df.columns:
        raise SystemExit("--victim-labels needs a simulated dataset with ground truth (filled_way)")
    df = df[df["split"] == split].reset_index(drop=True)
    T = int(df["filled_way"].map(len).max())
    F = np.full((len(df), T), -1, np.int64)
    for i, f in enumerate(df["filled_way"]):
        F[i, :len(f)] = f
    return F


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--run-dir", default="runs/hindsight")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-way", type=int, default=32)
    ap.add_argument("--d-glob", type=int, default=32)
    ap.add_argument("--explore", type=float, default=1.0, help="Gumbel exploration at epoch 0, fades to 0")
    ap.add_argument("--conf-weight", action="store_true",
                    help="weight each decision by label confidence |2p-1| (for noisy hardware labels)")
    ap.add_argument("--use-ages", action="store_true", help="give the model per-way line ages (recency/insertion ranks)")
    ap.add_argument("--victim-labels", action="store_true",
                    help="CAPACITY TEST, simulated data only: supervise with the true victims instead of hindsight")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    set_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    train = load_streams(args.data_dir, "train"); val = load_streams(args.data_dir, "val")
    if args.max_train and len(train["lens"]) > args.max_train:
        train = {k: v[: args.max_train] for k, v in train.items()}
    if args.victim_labels:
        train["F"] = load_filled_way(args.data_dir, "train")
        if args.max_train:
            train["F"] = train["F"][: args.max_train]
        assert train["F"].shape[0] == len(train["lens"])
    model = build_slot_model({"d_way": args.d_way, "d_glob": args.d_glob, "use_ages": args.use_ages})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hindsight] params={n_params:,} train_streams={len(train['lens']):,} "
          f"train_access={int(train['M'].sum()):,}", flush=True)

    best = {"acc": -1.0}
    for ep in range(args.epochs):
        t0 = time.time()
        explore = args.explore * max(0.0, 1 - ep / max(args.epochs - 1, 1))
        incons = decs = 0; tot = 0.0
        for A, P, M, F in iterate_with_victims(train, args.batch_size, rng, args.victim_labels):
            ch = (VictimLabelChooser(P, M, F) if args.victim_labels
                  else HindsightChooser(A, P, M, explore, args.conf_weight))
            model(A, chooser=ch)
            if not ch.losses:
                continue
            loss = torch.stack(ch.losses).sum() / max(ch.n_dec, 1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            incons += ch.n_incons; decs += ch.n_dec; tot += float(loss) * ch.n_dec
        ev = evaluate(model, val, "cpu")                              # free-running, no labels
        print(f"[ep {ep:>2}] explore={explore:.2f} loss={tot / max(decs, 1):.4f} "
              f"inconsistent_steps={incons} val_acc(free-running)={ev['acc']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ev["acc"] > best["acc"]:
            best = ev
            torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "best.pt")

    summary = {"params": n_params, "best_val": best, "base_rate_val": base_rate_baseline(train, val)}
    if args.test_dir:
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu")["model"])
        test = load_streams(args.test_dir, None)
        summary["independent_test"] = evaluate(model, test, "cpu")
        summary["base_rate_test"] = base_rate_baseline(train, test)
        print(f"[TEST on fresh set] acc={summary['independent_test']['acc']:.4f} "
              f"(base-rate acc={summary['base_rate_test']['acc']:.4f})", flush=True)
    (run_dir / "hit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best_val_acc={best['acc']:.4f} -> {run_dir / 'hit_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
