"""Common replacement-policy interface and the shared experiment driver (spec §3.1-3.2)."""

from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ReplacementPolicy(Protocol):
    """Interface every simulated single-set replacement policy implements."""

    W: int

    def reset(self) -> None:
        """Empty the set and put policy state at its reset value."""
        ...

    def access(self, block: int) -> int | None:
        """Access `block`. Return the evicted block ID, or None on a hit or a fill into an empty way."""
        ...

    def resident(self) -> list[int | None]:
        """Block ID held in each way (None = empty)."""
        ...

    def state(self) -> np.ndarray:
        """Snapshot of policy-internal state (copy; safe to keep)."""
        ...

    def clone(self) -> "ReplacementPolicy":
        """Independent deep copy."""
        ...


class LinearFillPolicy:
    """Base class implementing the access/fill/evict skeleton shared by all policies.

    Empty ways are filled lowest-index first (spec §3.1 "linear insertion order").
    Subclasses implement `_reset_state`, `_on_hit`, `_on_insert` and `_victim_way`.
    """

    name: str = "base"

    def __init__(self, W: int) -> None:
        if not isinstance(W, int) or W < 2:
            raise ValueError(f"W must be an int >= 2, got {W!r}")
        self.W = W
        self.reset()

    # ---- subclass hooks -------------------------------------------------
    def _reset_state(self) -> None:
        raise NotImplementedError

    def _on_hit(self, w: int) -> None:
        raise NotImplementedError

    def _on_insert(self, w: int) -> None:
        """Called after a block is placed in way w (fill into empty way or replacement)."""
        raise NotImplementedError

    def _victim_way(self) -> int:
        """Choose the way to evict; called only when the set is full. May update state."""
        raise NotImplementedError

    # ---- public interface ----------------------------------------------
    def reset(self) -> None:
        """Empty the set and reset policy state."""
        self.ways: list[int | None] = [None] * self.W
        self._reset_state()

    def access(self, block: int) -> int | None:
        """Access `block`; return the evicted block ID or None (hit / fill into empty way)."""
        if block is None or int(block) < 0:
            raise ValueError(f"block IDs must be non-negative ints, got {block!r}")
        if block in self.ways:
            self._on_hit(self.ways.index(block))
            return None
        if None in self.ways:
            w = self.ways.index(None)
            self.ways[w] = block
            self._on_insert(w)
            return None
        w = self._victim_way()
        evicted = self.ways[w]
        self.ways[w] = block
        self._on_insert(w)
        return evicted

    def resident(self) -> list[int | None]:
        """Block ID in each way (None = empty)."""
        return list(self.ways)

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def clone(self) -> "LinearFillPolicy":
        """Deep copy (including any RNG state)."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(W={self.W}, ways={self.ways}, state={self.state().tolist()})"


def run_experiment(policy: ReplacementPolicy, history: list[int]) -> tuple[int, np.ndarray]:
    """reset -> fill b0..b(W-1) -> apply history -> snapshot state -> access bW.

    Phase 1 semantics: every history element must be a block ID in [0, W), so every
    history access is a hit. Returns (victim_block_id, state_snapshot_before_trigger).
    """
    W = policy.W
    policy.reset()
    for b in range(W):
        ev = policy.access(b)
        assert ev is None, f"fill of b{b} unexpectedly evicted b{ev}"
    assert policy.resident() == list(range(W)), f"fill not linear: {policy.resident()}"

    for b in history:
        b = int(b)
        assert 0 <= b < W, f"phase-1 history element {b} not in [0, {W})"
        ev = policy.access(b)
        assert ev is None, f"history access to b{b} was not a hit (evicted b{ev})"

    snapshot = np.array(policy.state(), copy=True)
    before = policy.resident()
    victim = policy.access(W)
    after = policy.resident()
    assert victim is not None and victim in before, f"trigger bW did not evict a resident block ({victim})"
    changed = [w for w in range(W) if before[w] != after[w]]
    assert len(changed) == 1 and after[changed[0]] == W and before[changed[0]] == victim, (
        f"trigger must replace exactly one block: before={before} after={after}"
    )
    return int(victim), snapshot


def run_miss_experiment(policy: ReplacementPolicy, history: list[int], trigger: int | None = None) -> tuple[int, int]:
    """reset -> fill b0..b(W-1) -> apply history -> access a non-resident trigger block.

    Unlike `run_experiment`, history elements may be any non-negative block ID, so
    accesses to blocks outside the set cause misses and evictions mid-history.
    Returns (victim_way, evicted_block) for the trigger access.
    """
    W = policy.W
    policy.reset()
    for b in range(W):
        assert policy.access(b) is None, f"fill of b{b} unexpectedly evicted a block"
    for b in history:
        policy.access(int(b))
    before = policy.resident()
    if trigger is None:
        trigger = max([W, *map(int, history)]) + 1
    assert trigger not in before, f"trigger b{trigger} is already resident"
    evicted = policy.access(trigger)
    assert evicted is not None and evicted in before
    return before.index(evicted), int(evicted)
