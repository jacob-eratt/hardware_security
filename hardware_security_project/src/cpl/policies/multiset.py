"""Multi-set cache simulators (architecture_spec.md §6: DRRIP with set dueling, multi-set).

``MultiSetCache`` holds S independent sets that all run the same single-set policy
from ``cpl.policies`` (LRU, tree-PLRU, SRRIP, ...). Sets share no state, so each set
behaves exactly like the single-set simulator run on that set's own accesses.

``DRRIPCache`` is DRRIP with set dueling (Jaleel et al., ISCA 2010): a few *leader*
sets always run SRRIP, a few always run BRRIP, and one global saturating counter PSEL
records which leader group misses more. All other (*follower*) sets use the sub-policy
PSEL currently favours. This is the global, cross-set state that the replacement
unit's global state ``g`` and the set identity ``e(a)`` exist to capture.

Conventions (MUST be kept stable; datasets record them in meta.json):
  * Sub-policies share SRRIP's hit promotion (RRPV = 0) and victim search (lowest-index
    way with RRPV = 3, aging all ways until one exists); they differ only in the
    insertion RRPV: SRRIP inserts 2; BRRIP inserts 3, except 1 in 32 insertions get 2.
  * Leader sets, "complement select": the S sets are split into ``n_leaders``
    constituencies of k = S / n_leaders consecutive sets. In constituency c, the set at
    offset ``c % k`` is an SRRIP leader and the set at offset ``(k - 1 - c) % k`` is a
    BRRIP leader.
  * PSEL has ``psel_bits`` bits (default 10, range 0..1023). Every miss in an SRRIP
    leader increments it, every miss in a BRRIP leader decrements it (both saturate).
    Fills into empty ways count as misses. Followers use BRRIP when PSEL >= 2**(bits-1)
    (the MSB is set), else SRRIP. PSEL is updated after the missing line is inserted.
  * BRRIP's 1-in-32 insertion, ``bimodal``:
      "counter" (default): deterministic. One global counter of BRRIP insertions; every
                 32nd one (counter % 32 == 31) inserts with RRPV = 2. Real designs use
                 such a throttle counter as well as random draws.
      "rng":     each BRRIP insertion draws RRPV = 2 with probability 1/32 from a seeded
                 RNG. Outcomes are then not predictable from addresses alone.
"""
from __future__ import annotations

import numpy as np

from . import make_policy
from .srrip import RRPV_LONG, RRPV_MAX, SRRIP

FOLLOWER, SRRIP_LEADER, BRRIP_LEADER = 0, 1, 2
BIMODAL_PERIOD = 32


class MultiSetCache:
    """S independent W-way sets running one single-set policy; sets start empty."""

    def __init__(self, policy: str, S: int, W: int, seed: int | None = 0) -> None:
        self.policy, self.S, self.W, self.seed = policy, S, W, seed
        self._sets: dict[int, object] = {}

    def _set(self, s: int):
        if not 0 <= s < self.S:
            raise ValueError(f"set index {s} out of range [0, {self.S})")
        if s not in self._sets:
            self._sets[s] = self._make_set(s)
        return self._sets[s]

    def _make_set(self, s: int):
        return make_policy(self.policy, self.W, seed=None if self.seed is None else self.seed + s)

    def access(self, s: int, block: int) -> tuple[bool, int | None, int]:
        """Access ``block`` in set ``s``. Returns (hit, evicted block or None, way of block)."""
        p = self._set(s)
        hit = block in p.resident()
        ev = p.access(block)
        return hit, ev, p.resident().index(block)

    def resident(self, s: int) -> list[int | None]:
        return self._set(s).resident()

    def state(self, s: int) -> np.ndarray:
        return np.asarray(self._set(s).state())

    def role(self, s: int) -> int:
        return FOLLOWER


class _DuelSet(SRRIP):
    """One set of a DRRIPCache: SRRIP mechanics, insertion RRPV chosen by the owner."""

    name = "drrip_set"

    def __init__(self, W: int, owner: "DRRIPCache", s: int) -> None:
        self.owner, self.s = owner, s
        super().__init__(W)

    def _insert_rrpv(self) -> int:
        return self.owner._insert_rrpv(self.s)


class DRRIPCache(MultiSetCache):
    """DRRIP with set dueling over S sets (see module docstring for every convention)."""

    def __init__(self, S: int, W: int, n_leaders: int = 32, psel_bits: int = 10,
                 psel_init: int | None = None, bimodal: str = "counter", seed: int | None = 0) -> None:
        super().__init__("drrip", S, W, seed)
        if S % n_leaders:
            raise ValueError(f"S={S} must be a multiple of n_leaders={n_leaders}")
        if bimodal not in ("counter", "rng"):
            raise ValueError(f"bimodal must be 'counter' or 'rng', got {bimodal!r}")
        self.n_leaders, self.psel_bits, self.bimodal = n_leaders, psel_bits, bimodal
        self.psel_max = (1 << psel_bits) - 1
        self.psel_mid = 1 << (psel_bits - 1)
        self.psel = self.psel_mid if psel_init is None else int(psel_init)
        if not 0 <= self.psel <= self.psel_max:
            raise ValueError(f"psel_init {psel_init} outside [0, {self.psel_max}]")
        self.roles = leader_roles(S, n_leaders)
        self.brrip_inserts = 0
        self.rng = np.random.default_rng(seed)

    def _make_set(self, s: int):
        return _DuelSet(self.W, self, s)

    def role(self, s: int) -> int:
        return int(self.roles[s])

    def mode(self, s: int) -> str:
        """Sub-policy set ``s`` uses right now: 'srrip' or 'brrip'."""
        r = self.roles[s]
        if r == SRRIP_LEADER:
            return "srrip"
        if r == BRRIP_LEADER:
            return "brrip"
        return "brrip" if self.psel >= self.psel_mid else "srrip"

    def _insert_rrpv(self, s: int) -> int:
        if self.mode(s) == "srrip":
            return RRPV_LONG
        if self.bimodal == "counter":
            long_ = self.brrip_inserts % BIMODAL_PERIOD == BIMODAL_PERIOD - 1
            self.brrip_inserts += 1
        else:
            long_ = self.rng.random() < 1.0 / BIMODAL_PERIOD
        return RRPV_LONG if long_ else RRPV_MAX

    def access(self, s: int, block: int) -> tuple[bool, int | None, int]:
        hit, ev, w = super().access(s, block)
        if not hit:
            r = self.roles[s]
            if r == SRRIP_LEADER:
                self.psel = min(self.psel + 1, self.psel_max)
            elif r == BRRIP_LEADER:
                self.psel = max(self.psel - 1, 0)
        return hit, ev, w


def leader_roles(S: int, n_leaders: int) -> np.ndarray:
    """Role of every set (FOLLOWER / SRRIP_LEADER / BRRIP_LEADER), complement select."""
    k = S // n_leaders
    roles = np.full(S, FOLLOWER, np.int8)
    for c in range(n_leaders):
        a, b = c * k + c % k, c * k + (k - 1 - c) % k
        if a == b:
            raise ValueError(f"S={S}, n_leaders={n_leaders}: SRRIP and BRRIP leader collide in constituency {c}")
        roles[a], roles[b] = SRRIP_LEADER, BRRIP_LEADER
    return roles


def make_multiset(policy: str, S: int, W: int, seed: int | None = 0, **drrip_kw) -> MultiSetCache:
    """``policy`` = 'drrip' (set dueling, ``drrip_kw`` passed on) or any single-set name."""
    if policy == "drrip":
        return DRRIPCache(S, W, seed=seed, **drrip_kw)
    if drrip_kw:
        raise ValueError(f"DRRIP options {sorted(drrip_kw)} given for policy {policy!r}")
    return MultiSetCache(policy, S, W, seed)
