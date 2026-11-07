"""Train the GRU victim classifier (spec §6.4).

    python -m cpl.train --config configs/sim_plru_w16.yaml
    python -m cpl.train --config configs/sim_plru_w16.yaml --data-dir data/fake/sim_plru_W16

Writes ``runs/<run_name>/``: ``config.yaml``, ``metrics.jsonl``, ``best.pt``, ``last.pt``, ``train_summary.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch
import yaml

from cpl.data.dataset import VictimDataset, load_samples, load_split, to_device
from cpl.model.gru import VictimGRU, build_model, count_parameters
from cpl.model.losses import soft_cross_entropy
from cpl.utils.seed import named_rng, set_seed

TRAIN_DEFAULTS = {
    "batch_size": 512,
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "min_lr": 1e-5,
    "max_epochs": 20,
    "max_steps": None,
    "grad_clip": 1.0,
    "eval_every": 2000,
    "curriculum": {"start_len": 8, "step": 8, "max_len": 32},
    "early_stop_acc": 0.999,
    "early_stop_patience": 2,
    "device": "auto",
}
COST_THRESHOLDS = (0.99, 0.999)


def resolve_device(name: str) -> torch.device:
    """``auto`` -> cuda if available else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def curriculum_len(epoch: int, cur: dict | None) -> int | None:
    """Max history length for ``epoch`` (0-based); ``None`` disables the filter."""
    if not cur:
        return None
    return min(cur["start_len"] + cur["step"] * epoch, cur["max_len"])


def lr_lambda_factory(lr: float, min_lr: float, warmup: int, total: int):
    """Linear warmup then cosine decay from ``lr`` to ``min_lr``, as a LambdaLR multiplier."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return (min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))) / lr
    return f


@torch.no_grad()
def evaluate(model: VictimGRU, ds: VictimDataset, device: torch.device, batch_size: int = 2048) -> dict:
    """Victim accuracy and mean NLL over a dataset."""
    model.eval()
    correct, nll, n = 0, 0.0, 0
    for b in ds.iter_batches(batch_size):
        b = to_device(b, device)
        logits = model(b["tokens"], b["new_pos"], b["W"])
        correct += int((logits.argmax(-1) == b["label"]).sum())
        nll += float(soft_cross_entropy(logits, b["victim_counts"], reduction="sum"))
        n += len(b["label"])
    model.train()
    return {"acc": correct / max(n, 1), "nll": nll / max(n, 1), "n": n}


def load_config(path: str | Path, overrides: argparse.Namespace) -> dict:
    """Load YAML config, fill train defaults and apply CLI overrides."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["train"] = {**copy.deepcopy(TRAIN_DEFAULTS), **(cfg.get("train") or {})}
    cfg.setdefault("model", {})
    t = cfg["train"]
    for key in ("batch_size", "max_epochs", "max_steps", "eval_every", "device", "lr"):
        val = getattr(overrides, key, None)
        if val is not None:
            t[key] = val
    if overrides.no_curriculum:
        t["curriculum"] = None
    if overrides.seed is not None:
        cfg["seed"] = overrides.seed
    cfg["data_dir"] = overrides.data_dir or cfg.get("data", {}).get("out_dir")
    if not cfg["data_dir"]:
        raise ValueError("no data directory: set data.out_dir in the config or pass --data-dir")
    return cfg


def train(cfg: dict, run_dir: Path) -> dict:
    """Run one training job described by a resolved config; returns the summary dict."""
    t = cfg["train"]
    seed = int(cfg["seed"])
    set_seed(seed)
    device = resolve_device(t["device"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    df = load_samples(cfg["data_dir"])
    train_ds, val_ds = load_split(df, "train"), load_split(df, "val")
    test_ds = load_split(df, "test") if (df["split"] == "test").any() else None
    del df
    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    bs = int(t["batch_size"])
    epoch_idx = [train_ds.indices(curriculum_len(e, t["curriculum"])) for e in range(t["max_epochs"])]
    total_steps = sum(math.ceil(len(i) / bs) for i in epoch_idx)
    if t["max_steps"]:
        total_steps = min(total_steps, int(t["max_steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(t["lr"], t["min_lr"], t["warmup_steps"], total_steps))
    print(f"[train] device={device} params={n_params:,} train={len(train_ds):,} val={len(val_ds):,} "
          f"total_steps={total_steps:,} run_dir={run_dir}")
    rng = named_rng(seed, "train_shuffle")
    metrics_f = (run_dir / "metrics.jsonl").open("w")
    step, seen, best_acc, streak, stop = 0, 0, -1.0, 0, False
    cost: dict[str, dict | None] = {f"{x:g}": None for x in COST_THRESHOLDS}
    loss_sum, loss_n = 0.0, 0
    t0 = time.time()

    def do_eval(epoch: int, max_len: int | None) -> None:
        nonlocal best_acc, streak, stop, loss_sum, loss_n
        ev = evaluate(model, val_ds, device)
        elapsed = time.time() - t0
        rec = {
            "step": step, "epoch": epoch, "samples_seen": seen, "elapsed_s": round(elapsed, 2),
            "train_loss": loss_sum / max(loss_n, 1), "val_acc": ev["acc"], "val_nll": ev["nll"],
            "lr": sched.get_last_lr()[0], "curriculum_max_len": max_len,
        }
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        loss_sum, loss_n = 0.0, 0
        for thr in COST_THRESHOLDS:
            key = f"{thr:g}"
            if cost[key] is None and ev["acc"] >= thr:
                cost[key] = {"step": step, "samples_seen": seen, "elapsed_s": round(elapsed, 2)}
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save({"model": model.state_dict(), "config": cfg, "step": step, "val_acc": best_acc}, run_dir / "best.pt")
        streak = streak + 1 if ev["acc"] >= t["early_stop_acc"] else 0
        stop = stop or streak >= t["early_stop_patience"]
        print(f"[eval] step={step:>6} ep={epoch:>2} len<={max_len} loss={rec['train_loss']:.4f} "
              f"val_acc={ev['acc']:.5f} val_nll={ev['nll']:.4f} lr={rec['lr']:.2e} t={elapsed:.0f}s")

    model.train()
    epoch = 0
    for epoch in range(t["max_epochs"]):
        max_len = curriculum_len(epoch, t["curriculum"])
        for b in train_ds.iter_batches(bs, epoch_idx[epoch], shuffle=True, rng=rng):
            b = to_device(b, device)
            loss = soft_cross_entropy(model(b["tokens"], b["new_pos"], b["W"]), b["victim_counts"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            seen += len(b["label"])
            loss_sum += loss.item() * len(b["label"])
            loss_n += len(b["label"])
            if step % t["eval_every"] == 0:
                do_eval(epoch, max_len)
            if stop or step >= total_steps:
                break
        if not stop and step % t["eval_every"]:
            do_eval(epoch, max_len)
        if stop or step >= total_steps:
            break
    metrics_f.close()
    train_time = time.time() - t0
    torch.save({"model": model.state_dict(), "config": cfg, "step": step}, run_dir / "last.pt")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    summary = {
        "params": n_params,
        "device": str(device),
        "steps": step,
        "epochs_run": epoch + 1,
        "samples_seen": seen,
        "train_time_s": round(train_time, 2),
        "early_stopped": stop,
        "best_val_acc": best_acc,
        "cost_to_val_acc": cost,
        "best_val": evaluate(model, val_ds, device),
        "best_test": evaluate(model, test_ds, device) if test_ds is not None else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    test_acc = summary["best_test"]["acc"] if summary["best_test"] else float("nan")
    print(f"[done] steps={step} time={train_time:.0f}s best_val_acc={best_acc:.5f} test_acc={test_acc:.5f}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None, help="override data.out_dir (e.g. data/fake/sim_plru_W16)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-epochs", dest="max_epochs", type=int, default=None)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-curriculum", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args)
    name = args.run_name or f"{cfg['target']}_{datetime.now():%Y%m%d_%H%M%S}"
    train(cfg, Path(args.runs_dir) / name)


if __name__ == "__main__":
    main()
