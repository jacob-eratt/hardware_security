"""Fixed token vocabulary and sequence layout (spec §6.1).

Sequence for one sample: ``FILL b0 .. b(W-1) SEP h1 .. hL NEW`` (length ``W + L + 3``).
With a mode/policy token (spec §8.4): ``FILL MODE b0 .. b(W-1) SEP h1 .. hL NEW`` (length ``W + L + 4``).
"""
from __future__ import annotations

from collections.abc import Sequence

PAD = 0
FILL = 1
SEP = 2
NEW = 3
BLOCK_BASE = 4
N_BLOCK_TOKENS = 32
RESERVED_BASE = BLOCK_BASE + N_BLOCK_TOKENS
N_RESERVED = 12
VOCAB_SIZE = RESERVED_BASE + N_RESERVED
W_MAX = 16

assert VOCAB_SIZE == 48

# policy / forced-mode tokens, reserved IDs 36-39 (spec §8.4)
POLICY_TOKENS = {"plru": 36, "lru": 37, "srrip": 38, "brrip": 39}


def block_token(block: int) -> int:
    """Token ID for block ``b_block``."""
    if not 0 <= block < N_BLOCK_TOKENS:
        raise ValueError(f"block id {block} outside [0, {N_BLOCK_TOKENS})")
    return BLOCK_BASE + block


def mode_token(mode: str) -> int:
    """Token ID for a policy / forced-mode name."""
    try:
        return POLICY_TOKENS[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(POLICY_TOKENS)}") from None


def seq_len(W: int, L: int, mode: bool = False) -> int:
    """Length of the token sequence for associativity ``W`` and history length ``L``."""
    return W + L + 3 + int(mode)


def new_position(W: int, L: int, mode: bool = False) -> int:
    """Index of the ``NEW`` token in the sequence."""
    return W + L + 2 + int(mode)


def encode(history: Sequence[int], W: int, mode: str | None = None) -> list[int]:
    """Tokenize one history into ``FILL [MODE] b0..b(W-1) SEP h1..hL NEW``."""
    if not 2 <= W <= W_MAX:
        raise ValueError(f"W={W} outside [2, {W_MAX}]")
    prefix = [FILL] if mode is None else [FILL, mode_token(mode)]
    return [*prefix, *(block_token(b) for b in range(W)), SEP, *(block_token(int(h)) for h in history), NEW]
