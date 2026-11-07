"""Torch dataset, tokenization and padding for victim samples (spec §6.1).

Batches are right-padded with PAD and carry ``new_pos`` (index of NEW) per sample.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cpl.data.tokens import PAD, W_MAX, encode

Batch = dict[str, torch.Tensor]


def encode_padded(histories: Sequence[Sequence[int]], Ws: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize and right-pad many histories. Returns ``(tokens uint8 [N, T], new_pos int32 [N])``."""
    lens = np.fromiter((len(h) for h in histories), dtype=np.int64, count=len(histories))
    Ws = np.asarray(Ws, dtype=np.int64)
    new_pos = (Ws + lens + 2).astype(np.int32)
    T = int(new_pos.max()) + 1 if len(new_pos) else 1
    tokens = np.full((len(histories), T), PAD, dtype=np.uint8)
    for i, (h, w) in enumerate(zip(histories, Ws)):
        seq = encode(h, int(w))
        tokens[i, : len(seq)] = seq
    return tokens, new_pos


def collate(samples: Sequence[dict]) -> Batch:
    """Collate raw samples ``{history, W, victim_counts}`` into a padded batch."""
    histories = [s["history"] for s in samples]
    Ws = [int(s["W"]) for s in samples]
    tokens, new_pos = encode_padded(histories, Ws)
    counts = np.stack([np.asarray(s["victim_counts"], dtype=np.float32) for s in samples])
    return {
        "tokens": torch.from_numpy(tokens.astype(np.int64)),
        "new_pos": torch.from_numpy(new_pos.astype(np.int64)),
        "W": torch.tensor(Ws, dtype=torch.long),
        "victim_counts": torch.from_numpy(counts),
        "label": torch.from_numpy(counts.argmax(-1).astype(np.int64)),
        "hist_len": torch.tensor([len(h) for h in histories], dtype=torch.long),
    }


class VictimDataset(Dataset):
    """Pre-tokenized samples from one split of ``samples.parquet``.

    ``__getitem__`` returns a single raw sample (usable with ``collate``); ``batch`` and
    ``iter_batches`` slice the pre-tokenized arrays directly, which is much faster.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        histories = [np.asarray(h, dtype=np.int64) for h in df["history"]]
        self.W = df["W"].to_numpy(dtype=np.int64)
        if (self.W > W_MAX).any():
            raise ValueError(f"W > {W_MAX} not supported")
        self.histories = histories
        self.hist_len = np.fromiter((len(h) for h in histories), dtype=np.int64, count=len(histories))
        self.victim_counts = np.stack([np.asarray(v, dtype=np.float32) for v in df["victim_counts"]]) if len(df) else np.zeros((0, W_MAX), np.float32)
        if "label_hard" in df:
            self.label = df["label_hard"].to_numpy(dtype=np.int64)
        else:
            self.label = self.victim_counts.argmax(-1).astype(np.int64)
        self.tokens, self.new_pos = encode_padded(histories, self.W)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int) -> dict:
        return {"history": self.histories[i], "W": int(self.W[i]), "victim_counts": self.victim_counts[i]}

    def batch(self, idx: np.ndarray) -> Batch:
        """Batch for row indices ``idx``, trimmed to the longest sequence in the batch."""
        new_pos = self.new_pos[idx]
        T = int(new_pos.max()) + 1
        return {
            "tokens": torch.from_numpy(self.tokens[idx, :T].astype(np.int64)),
            "new_pos": torch.from_numpy(new_pos.astype(np.int64)),
            "W": torch.from_numpy(self.W[idx]),
            "victim_counts": torch.from_numpy(self.victim_counts[idx]),
            "label": torch.from_numpy(self.label[idx]),
            "hist_len": torch.from_numpy(self.hist_len[idx]),
        }

    def indices(self, max_len: int | None = None) -> np.ndarray:
        """Row indices, optionally restricted to ``hist_len <= max_len`` (curriculum)."""
        if max_len is None:
            return np.arange(len(self))
        return np.flatnonzero(self.hist_len <= max_len)

    def iter_batches(
        self,
        batch_size: int,
        idx: np.ndarray | None = None,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[Batch]:
        """Yield batches over ``idx`` (default: all rows)."""
        idx = self.indices() if idx is None else np.asarray(idx)
        if shuffle:
            idx = (rng or np.random.default_rng()).permutation(idx)
        for s in range(0, len(idx), batch_size):
            yield self.batch(idx[s : s + batch_size])


def load_samples(data_dir: str | Path) -> pd.DataFrame:
    """Read ``samples.parquet`` from a processed data directory."""
    return pd.read_parquet(Path(data_dir) / "samples.parquet")


def load_split(df: pd.DataFrame, split: str) -> VictimDataset:
    """Build a ``VictimDataset`` for one split of a samples dataframe."""
    sub = df[df["split"] == split].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"split '{split}' is empty or missing")
    return VictimDataset(sub)


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor in a batch to ``device``."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

