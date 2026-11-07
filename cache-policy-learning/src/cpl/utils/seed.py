"""Reproducibility helpers (spec §10)."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def named_rng(seed: int, name: str) -> np.random.Generator:
    """Independent NumPy generator derived from the master seed and a stream name."""
    tag = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([seed, tag])
