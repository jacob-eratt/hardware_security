"""History generation, history IDs and split assignment (spec §4.2-4.5)."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from cpl.policies.fast import PAD, simulate_batch

SPLIT_TRAIN_MAX = 0.90  # u < 0.90 -> train
SPLIT_VAL_MAX = 0.95    # 0.90 <= u < 0.95 -> val, else test
HASH_SPLITS = ("train", "val", "test")
DEDUP_MIN_LEN = 4       # within a split, duplicates are allowed only for L <= 3


# ----------------------------------------------------------------------- IDs & splits
def history_id(W: int, history: Iterable[int]) -> str:
    """16-hex-char ID: sha1("W{W}:" + comma-joined block IDs)[:16]  (spec §4.5)."""
    s = f"W{W}:" + ",".join(map(str, history))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def id_to_unit(hid: str) -> float:
    """Deterministic value in [0, 1) derived from a history ID."""
    return int(hid, 16) / 16**16


def split_of(hid: str) -> str:
    """Hash split for a history ID: train (90%) / val (5%) / test (5%)."""
    u = id_to_unit(hid)
    return "train" if u < SPLIT_TRAIN_MAX else ("val" if u < SPLIT_VAL_MAX else "test")


# ----------------------------------------------------------------------- containers
@dataclass
class HistoryBatch:
    """A set of phase-1 histories stored as a right-padded int8 matrix (pad = -1)."""

    W: int
    hist: np.ndarray       # (n, Lmax) int8
    lens: np.ndarray       # (n,) int16
    ids: list[str]
    gen_type: str          # "random" | "diagnostic"

    def __len__(self) -> int:
        return len(self.ids)

    def to_lists(self) -> list[list[int]]:
        return [row[:L] for row, L in zip(self.hist.tolist(), self.lens.tolist())]


def _ids_for(W: int, hist: np.ndarray, lens: np.ndarray) -> list[str]:
    prefix = f"W{W}:"
    return [
        hashlib.sha1((prefix + ",".join(map(str, row[:L]))).encode()).hexdigest()[:16]
        for row, L in zip(hist.tolist(), lens.tolist())
    ]


def draw_random(
    W: int, n: int, L_range: Sequence[int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n histories: L ~ U[lo, hi], each element ~ U[0, W). Returns (padded, lens)."""
    lo, hi = int(L_range[0]), int(L_range[1])
    if not 0 <= lo <= hi:
        raise ValueError(f"bad L_range {L_range}")
    lens = rng.integers(lo, hi + 1, size=n)
    hist = rng.integers(0, W, size=(n, hi), dtype=np.int8)
    hist[np.arange(hi)[None, :] >= lens[:, None]] = PAD
    return hist, lens.astype(np.int16)


CandidateFilter = Callable[[np.ndarray, np.ndarray], np.ndarray]


def collect(
    W: int,
    n: int,
    L_range: Sequence[int],
    rng: np.random.Generator,
    *,
    accept_id: Callable[[str], bool] | None = None,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
    candidate_filter: CandidateFilter | None = None,
    gen_type: str = "random",
    batch_size: int = 8_192,
    max_draws: int | None = None,
) -> tuple[HistoryBatch, int]:
    """Draw random candidates until n histories are accepted.

    A candidate is kept if (in order): `candidate_filter` (vectorized, e.g. a
    diagnostic condition) passes, `accept_id(id)` is true (e.g. hash split == "val"),
    its ID is not in `exclude_ids`, and it is not a duplicate within this collection
    (duplicates are allowed for L <= 3). Output order is the draw order, so the result
    is a deterministic function of the generator state. Candidates are drawn in fixed
    size batches, so asking for more rows extends the result without changing its
    prefix.

    Returns (batch, number_of_candidates_drawn). Warns and returns fewer than n rows
    if `max_draws` is exhausted.
    """
    max_draws = max_draws if max_draws is not None else max(1_000 * n, 10_000_000)
    width = int(L_range[1])
    hists: list[np.ndarray] = []
    lens_out: list[np.ndarray] = []
    ids_out: list[str] = []
    seen: set[str] = set()
    drawn = 0
    while len(ids_out) < n and drawn < max_draws:
        m = min(batch_size, max_draws - drawn)
        hist, lens = draw_random(W, m, L_range, rng)
        drawn += m
        if candidate_filter is not None:
            keep = candidate_filter(hist, lens)
            hist, lens = hist[keep], lens[keep]
        ids = _ids_for(W, hist, lens)
        take: list[int] = []
        need = n - len(ids_out)
        for i, (hid, L) in enumerate(zip(ids, lens.tolist())):
            if accept_id is not None and not accept_id(hid):
                continue
            if hid in exclude_ids:
                continue
            if L >= DEDUP_MIN_LEN:
                if hid in seen:
                    continue
                seen.add(hid)
            take.append(i)
            ids_out.append(hid)
            if len(take) == need:
                break
        hists.append(hist[take])
        lens_out.append(lens[take])
    if len(ids_out) < n:
        warnings.warn(
            f"collect: only {len(ids_out)}/{n} histories accepted after {drawn} draws "
            f"(W={W}, L_range={tuple(L_range)}, gen_type={gen_type})",
            stacklevel=2,
        )
    hist = np.concatenate(hists) if hists else np.full((0, width), PAD, np.int8)
    lens = np.concatenate(lens_out) if lens_out else np.zeros(0, np.int16)
    return HistoryBatch(W=W, hist=hist, lens=lens, ids=ids_out, gen_type=gen_type), drawn


# ----------------------------------------------------------------------- generators
def generate_random_split(
    W: int,
    split: str,
    n: int,
    L_range: Sequence[int],
    rng: np.random.Generator,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[HistoryBatch, int]:
    """Random histories for one split.

    For the hash splits (train/val/test) a candidate is kept only if its hash maps to
    `split`, so each split has its own RNG stream yet no history can be in two of them.
    Other splits (test_long, test_xlong, ...) keep any candidate not in `exclude_ids`.
    """
    accept = (lambda hid: split_of(hid) == split) if split in HASH_SPLITS else None
    return collect(W, n, L_range, rng, accept_id=accept, exclude_ids=exclude_ids)


def diagnostic_filter(policies: Sequence[str], W: int, require: str) -> CandidateFilter:
    """Vectorized condition on candidates: the given policies' victims disagree.

    require="any": not all victims equal (with two policies: PLRU(h) != LRU(h)).
    require="all_pairwise": every pair of policies gives a different victim.
    """
    if len(policies) < 2:
        raise ValueError("need at least two policies to define disagreement")
    if require not in ("any", "all_pairwise"):
        raise ValueError(f"require must be 'any' or 'all_pairwise', got {require!r}")
    policies = tuple(policies)

    def keep(hist: np.ndarray, lens: np.ndarray) -> np.ndarray:
        v = simulate_batch(hist, lens, W, policies=policies).victims
        vs = [v[p] for p in policies]
        if require == "any":
            return np.logical_or.reduce([vs[0] != x for x in vs[1:]])
        ok = np.ones(len(lens), dtype=bool)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                ok &= vs[i] != vs[j]
        return ok

    return keep


def generate_diagnostic(
    policies: Sequence[str],
    n: int,
    W: int,
    L_range: Sequence[int],
    rng: np.random.Generator,
    require: str = "any",
    max_tries: int | None = None,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[HistoryBatch, int]:
    """Random histories on which `policies` disagree (spec §4.3); gen_type="diagnostic".

    Warns and returns fewer than n histories if `max_tries` candidates are exhausted.
    """
    return collect(
        W,
        n,
        L_range,
        rng,
        exclude_ids=exclude_ids,
        candidate_filter=diagnostic_filter(policies, W, require),
        gen_type="diagnostic",
        max_draws=max_tries,
    )
