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


def load_streams(data_dir: str | Path, split: str | None = None):
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
    A = np.full((n, T), NUM_BLOCKS, np.int64)   # pad id
    P = np.zeros((n, T), np.float32)
    M = np.zeros((n, T), np.float32)            # valid-position mask
    for i, (a, p) in enumerate(zip(acc, ph)):
        A[i, : len(a)] = a
        P[i, : len(p)] = p
        M[i, : len(a)] = 1.0
    return {"A": A, "P": P, "M": M, "lens": lens}


def iterate(data, bs, shuffle, rng):
    idx = np.arange(len(data["lens"]))
    if shuffle:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), bs):
        j = idx[s : s + bs]
        T = int(data["lens"][j].max())
        yield (torch.from_numpy(data["A"][j, :T]), torch.from_numpy(data["P"][j, :T]),
               torch.from_numpy(data["M"][j, :T]))


@torch.no_grad()
def evaluate(model, data, device, bs=1024):
    model.eval()
    tot_m = correct = 0.0
    ll = 0.0
    for A, P, M in iterate(data, bs, False, None):
        A, P, M = A.to(device), P.to(device), M.to(device)
        logits = model(A)
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
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    train = load_streams(args.data_dir, "train")
    val = load_streams(args.data_dir, "val")
    if train is None:
        raise SystemExit("no 'train' split in --data-dir")
    model = build_hit_model({"hidden": args.hidden, "layers": args.layers}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hit] params={n_params:,} train_streams={len(train['lens']):,} "
          f"train_access={int(train['M'].sum()):,} device={device}")

    best = {"acc": -1.0}
    for ep in range(args.epochs):
        for A, P, M in iterate(train, args.batch_size, True, rng):
            A, P, M = A.to(device), P.to(device), M.to(device)
            logits = model(A)
            loss = (bce(logits, P) * M).sum() / M.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ev = evaluate(model, val, device) if val is not None else evaluate(model, train, device)
        tag = "val" if val is not None else "train"
        print(f"[ep {ep:>2}] loss={float(loss):.4f} {tag}_acc={ev['acc']:.4f} {tag}_logloss={ev['logloss']:.4f}")
        if ev["acc"] > best["acc"]:
            best = ev
            torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "best.pt")

    summary = {"params": n_params, "best_val": best,
               "base_rate_val": base_rate_baseline(train, val if val is not None else train)}
    if args.test_dir:
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
        test = load_streams(args.test_dir, None)   # whole fresh set
        te = evaluate(model, test, device)
        summary["independent_test"] = te
        summary["base_rate_test"] = base_rate_baseline(train, test)
        print(f"[TEST on fresh set] acc={te['acc']:.4f} logloss={te['logloss']:.4f} "
              f"(base-rate acc={summary['base_rate_test']['acc']:.4f})")
    (run_dir / "hit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best_val_acc={best['acc']:.4f}  summary -> {run_dir/'hit_summary.json'}")


if __name__ == "__main__":
    main()
