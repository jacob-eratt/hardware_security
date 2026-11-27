"""Observed-outcome prediction: predict access t given the TRUE hit/miss of accesses < t.

Free-running evaluation (the default everywhere) gives a model only the addresses. That
is the right test for fixed policies, but an adaptive policy's hidden global state (e.g.
DRRIP's PSEL, whose value at the start of a measurement is unknown) cannot be inferred
from addresses alone. On hardware every access's outcome is measured anyway, so the
realistic online task is: having seen the outcomes so far, predict the next one.

Protocol (same for the model and for the baselines in cpl.eval_sim):
  1. predict access t from the current residency (built from outcomes < t only);
  2. the true outcome of t is revealed and residency is made consistent with it:
       true hit,  line absent  -> insert it (lowest empty way, else the predictor's own victim)
       true miss, line present -> the true cache evicted it earlier: re-fetch into the same way
       true miss, line absent  -> the predictor's own choice (lowest empty way, else its victim)
These are the corrections cpl.train_hindsight applies at inconsistent steps; no future
information is used, so the prediction at t never depends on the outcome of t or later.
"""
from __future__ import annotations

import numpy as np
import torch


class ObservedChooser:
    """``chooser`` for SlotCacheModel.forward: applies the protocol above."""

    def __init__(self, A: np.ndarray, hits: np.ndarray, M: np.ndarray, n_blocks: int):
        self.A, self.hit, self.M, self.N = A, hits.astype(bool), M.astype(bool), n_blocks
        self.n_corr = 0                                   # steps where the outcome contradicted residency

    def __call__(self, t, way_lines, score):
        lines = way_lines.cpu().numpy(); sc = score.detach().cpu().numpy()
        B, W = lines.shape
        v = np.zeros((B, W), np.float32); ph = np.zeros(B, np.float32)
        for i in range(B):
            if not self.M[i, t]:
                continue
            b = int(self.A[i, t]); where = np.nonzero(lines[i] == b)[0]
            if self.hit[i, t] and len(where):
                ph[i] = 1.0
                continue
            if len(where):                                # true miss, line present
                self.n_corr += 1; w = int(where[0])
            else:
                if self.hit[i, t]:                        # true hit, line absent
                    self.n_corr += 1
                empty = np.nonzero(lines[i] == self.N)[0]
                w = int(empty[0]) if len(empty) else int(np.argmax(sc[i]))
            v[i, w] = 1.0
        dev = score.device
        return torch.from_numpy(v).to(dev), torch.from_numpy(ph).to(dev)
