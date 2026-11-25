"""Seeding helpers (spec §10)."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def _stable_int(name: str) -> int:
    """Process-independent 32-bit integer for a string (Python's hash() is salted)."""
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def derive_rng(seed: int, *names: str) -> np.random.Generator:
    """Independent generator for (master seed, name...), e.g. derive_rng(1234, "val").

    Changing how many draws one split makes never affects another split's stream.
    """
    return np.random.default_rng([int(seed), *(_stable_int(n) for n in names)])


def seed_everything(seed: int) -> None:
    """Seed Python `random`, NumPy's legacy global RNG and torch (CPU + CUDA) if present."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
