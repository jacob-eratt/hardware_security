"""Train the streaming hit/miss black-box cache model (cpl.model.hit_gru).

    python -m cpl.train_hits --data-dir data/processed/hw_l2_celeron_stream \
        [--test-dir data/processed/hw_l2_celeron_stream_test] [--epochs 20]

Trains a GRU to predict P(hit) per access against the measured hit rate (soft
BCE). Reports per-access accuracy and log-loss on val/test, plus a base-rate
baseline for context. With --test-dir, the final accuracy is on an INDEPENDENT
freshly-measured Celeron dataset -- the real test of "does the black box model
the cache."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpl.model.hit_gru import NUM_BLOCKS, build_hit_model
from cpl.utils.seed import set_seed


def data_geometry(data_dir: str | Path) -> dict:
    """Line-id range, local set slots and global set count of a stream dataset.

    Single-set data (no ``sets`` column): ids < NUM_BLOCKS, one slot, no set ids, which
    keeps every single-set model and checkpoint exactly as before. Multi-set data
    (cpl.data.sim_multiset_dataset): ids < pool * sets_max, sets_max local slots, S
    global sets. ``num_blocks`` is also the padding id."""
    meta = json.loads((Path(data_dir) / "meta.json").read_text())
    if "sets_range" not in meta:
        return {"num_blocks": NUM_BLOCKS, "s_max": 1, "n_set_ids": 0}
    return {"num_blocks": int(meta["pool"]) * int(meta["sets_range"][1]),
            "s_max": int(meta["sets_range"][1]), "n_set_ids": int(meta["S"])}


def load_streams(data_dir: str | Path, split: str | None = None, num_blocks: int = NUM_BLOCKS):
    """Padded arrays A (line ids, pad = num_blocks), P (p_hit), M (mask), lens; for
    multi-set data also Sl (local set slot, by order of first touch) and G (global set
    index). Single-set data gets no Sl/G."""
    df = pd.read_parquet(Path(data_dir) / "samples.parquet")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    if df.empty:
        return None
    acc = [np.asarray(a, np.int64) for a in df["accesses"]]
    ph = [np.asarray(p, np.float32) for p in df["p_hit"]]
    lens = np.fromiter((len(a) for a in acc), np.int64, len(acc))
    T = int(lens.max())
    n = len(acc)
    if max(int(a.max()) for a in acc) >= num_blocks:
        raise ValueError(f"line ids reach {max(int(a.max()) for a in acc)} but num_blocks={num_blocks}")
    A = np.full((n, T), num_blocks, np.int64)   # pad id
    P = np.zeros((n, T), np.float32)
    M = np.zeros((n, T), np.float32)            # valid-position mask
    for i, (a, p) in enumerate(zip(acc, ph)):
        A[i, : len(a)] = a
        P[i, : len(p)] = p
        M[i, : len(a)] = 1.0
    out = {"A": A, "P": P, "M": M, "lens": lens}
    if "sets" in df.columns:
        Sl = np.zeros((n, T), np.int64); G = np.zeros((n, T), np.int64)
        for i, s in enumerate(df["sets"]):
            s = np.asarray(s, np.int64)
            _, first = np.unique(s, return_index=True)
            order = {int(v): k for k, v in enumerate(s[np.sort(first)])}   # slot = order of first touch
            Sl[i, : len(s)] = [order[int(v)] for v in s]
            G[i, : len(s)] = s
        out["Sl"], out["G"] = Sl, G
    return out


def batch_sets(data, j, T) -> dict:
    """Model keyword arguments carrying set information for rows ``j`` (empty if single-set)."""
    if "Sl" not in data:
        return {}
    return {"sets": torch.from_numpy(data["Sl"][j, :T]), "set_ids": torch.from_numpy(data["G"][j, :T])}


def iterate(data, bs, shuffle, rng):
    """Yields (A, P, M, kw); ``kw`` holds the set tensors for multi-set data, else {}."""
    idx = np.arange(len(data["lens"]))
    if shuffle:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), bs):
        j = idx[s : s + bs]
        T = int(data["lens"][j].max())
        yield (torch.from_numpy(data["A"][j, :T]), torch.from_numpy(data["P"][j, :T]),
               torch.from_numpy(data["M"][j, :T]), batch_sets(data, j, T))


@torch.no_grad()
def evaluate(model, data, device, bs=1024):
    model.eval()
    tot_m = correct = 0.0
    ll = 0.0
    for A, P, M, kw in iterate(data, bs, False, None):
        A, P, M = A.to(device), P.to(device), M.to(device)
        logits = model(A, **{k: v.to(device) for k, v in kw.items()})
        prob = torch.sigmoid(logits)
        pred_hit = (prob > 0.5).float()
        meas_hit = (P > 0.5).float()
        correct += float(((pred_hit == meas_hit) * M).sum())
        ll += float((torch.nn.functional.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), P, reduction="none") * M).sum())
        tot_m += float(M.sum())
    model.train()
    return {"acc": correct / max(tot_m, 1), "logloss": ll / max(tot_m, 1), "n_access": int(tot_m)}


def base_rate_baseline(train, data):
    """Accuracy/log-loss of always predicting the train hit-rate (context for the model)."""
    m = train["M"] > 0
    r = float((train["P"][m]).mean())
    mm = data["M"] > 0
    meas = data["P"][mm]
    acc = float(((meas > 0.5) == (r > 0.5)).mean())
    ll = float(-(meas * np.log(max(r, 1e-6)) + (1 - meas) * np.log(max(1 - r, 1e-6))).mean())
    return {"hit_rate": round(r, 4), "acc": round(acc, 4), "logloss": round(ll, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--test-dir", default=None, help="independent fresh dataset for final test")
    ap.add_argument("--run-dir", default="runs/hit_model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load a model and evaluate (e.g. clean model on noisy data)")
    ap.add_argument("--load", default=None, help="checkpoint to evaluate (default: <run-dir>/best.pt)")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    geo = data_geometry(args.data_dir)
    args.num_blocks, args.n_set_ids = geo["num_blocks"], geo["n_set_ids"]
    train = load_streams(args.data_dir, "train", geo["num_blocks"])
    val = load_streams(args.data_dir, "val", geo["num_blocks"])
    if train is None:
        raise SystemExit("no 'train' split in --data-dir")
    model = build_hit_model({"hidden": args.hidden, "layers": args.layers,
                             "num_blocks": args.num_blocks, "n_set_ids": args.n_set_ids}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hit] params={n_params:,} train_streams={len(train['lens']):,} "
          f"train_access={int(train['M'].sum()):,} device={device}", flush=True)

    best = {"acc": -1.0}
    if args.eval_only:
        ckpt = args.load or (run_dir / "best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        best = evaluate(model, val if val is not None else train, device)
        print(f"[eval-only] loaded {ckpt}  val_acc={best['acc']:.4f} logloss={best['logloss']:.4f}")
    else:
        for ep in range(args.epochs):
            for A, P, M, kw in iterate(train, args.batch_size, True, rng):
                A, P, M = A.to(device), P.to(device), M.to(device)
                logits = model(A, **{k: v.to(device) for k, v in kw.items()})
                loss = (bce(logits, P) * M).sum() / M.sum().clamp_min(1)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ev = evaluate(model, val, device) if val is not None else evaluate(model, train, device)
            tag = "val" if val is not None else "train"
            print(f"[ep {ep:>2}] loss={float(loss):.4f} {tag}_acc={ev['acc']:.4f} {tag}_logloss={ev['logloss']:.4f}", flush=True)
            if ev["acc"] > best["acc"]:
                best = ev
                torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "best.pt")

    summary = {"params": n_params, "best_val": best,
               "base_rate_val": base_rate_baseline(train, val if val is not None else train)}
    if args.test_dir:
        if not args.eval_only:
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
        test = load_streams(args.test_dir, None, geo["num_blocks"])   # whole fresh set
        te = evaluate(model, test, device)
        summary["independent_test"] = te
        summary["base_rate_test"] = base_rate_baseline(train, test)
        print(f"[TEST on fresh set] acc={te['acc']:.4f} logloss={te['logloss']:.4f} "
              f"(base-rate acc={summary['base_rate_test']['acc']:.4f})")
    (run_dir / "hit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best_val_acc={best['acc']:.4f}  summary -> {run_dir/'hit_summary.json'}")


if __name__ == "__main__":
    main()
