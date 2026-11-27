"""Multi-set plumbing of the three-unit model and of hindsight training (spec §3.4, §5).

Needs a small multi-set dataset from hardware_security_project (generated on the fly):
  1. local set slots are isolated: with no global state and no set identity, a
     multi-set stream gives exactly the predictions of running each set's sub-stream
     on its own.
  2. per-set hindsight: driving residency with the simulator's TRUE victims, the true
     victim is always among the ways the hindsight chooser allows (so the constraints
     never exclude the truth), and no step is inconsistent.
  3. the set identity changes the model's output (it is really wired in).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cpl.model.slot_cache import build_slot_model
from cpl.train_hindsight import HindsightChooser, load_filled_way
from cpl.train_hits import batch_sets, data_geometry, load_streams

HSP = Path(__file__).resolve().parents[2] / "hardware_security_project"


@pytest.fixture(scope="module")
def ms_data(tmp_path_factory):
    out = {}
    for policy in ("drrip", "plru"):
        d = tmp_path_factory.mktemp(policy)
        subprocess.run([sys.executable, "-m", "cpl.data.sim_multiset_dataset", "--policy", policy,
                        "--n", "120", "--seed", "21", "--val-frac", "0.0", "--psel-init", "mid",
                        "--out", str(d)], check=True, cwd=HSP, env={"PYTHONPATH": str(HSP / "src"),
                                                                  "SYSTEMROOT": "C:\\Windows", "PATH": ""},
                       capture_output=True)
        out[policy] = d
    return out


def _model(geo, **kw):
    torch.manual_seed(0)
    return build_slot_model({"use_ages": True, "num_blocks": geo["num_blocks"], "sets": geo["s_max"], **kw}).eval()


@torch.no_grad()
def test_slots_isolated_without_global_state(ms_data):
    geo = data_geometry(ms_data["drrip"])
    data = load_streams(ms_data["drrip"], "train", geo["num_blocks"])
    m = _model(geo, use_global=False)
    for i in range(30):
        L = int(data["lens"][i]); A = torch.from_numpy(data["A"][i:i + 1, :L])
        Sl = torch.from_numpy(data["Sl"][i:i + 1, :L])
        joint = m(A, sets=Sl)[0]
        for j in Sl.unique().tolist():
            sel = (Sl[0] == j)
            alone = m(A[:, sel])[0]                   # that set's sub-stream as a single-set stream
            assert torch.equal(joint[sel], alone)


def _allowed_under_true_victims(geo, A, P, M, F, Sl):
    """Drive residency with the true victims; return {(stream, t): allowed ways} at every
    full-set miss, asserting the true victim is always allowed."""
    hs = HindsightChooser(A, P, M, explore=0.0, conf_weight=False, Sl=Sl, n_blocks=geo["num_blocks"])
    out = {}

    def chooser(t, way_lines, score):
        lines = way_lines.numpy(); B, W = lines.shape
        v = np.zeros((B, W), np.float32); ph = np.zeros(B, np.float32)
        for i in range(B):
            if not hs.M[i, t]:
                continue
            b = int(hs.A[i, t])
            if hs.lab[i, t]:
                assert b in lines[i]; ph[i] = 1.0
            else:
                w = int(F[i, t])
                if not (lines[i] == hs.N).any():          # full-set miss: a real decision
                    al = hs._allowed(i, t, lines[i])
                    assert w in al
                    out[(i, t)] = sorted(al)
                v[i, w] = 1.0
            hs.last[i, b] = t
        return torch.from_numpy(v), torch.from_numpy(ph)

    with torch.no_grad():
        _model(geo)(A, chooser=chooser, **({} if Sl is None else {"sets": Sl}))
    return out


@pytest.mark.parametrize("policy", ["drrip", "plru"])
def test_hindsight_is_per_set(ms_data, policy):
    """True victim always allowed, AND the allowed set at each decision equals the one
    computed on that set's sub-stream alone (counting misses of other sets would make
    the constraints looser and fail this)."""
    geo = data_geometry(ms_data[policy])
    data = load_streams(ms_data[policy], "train", geo["num_blocks"])
    F = load_filled_way(ms_data[policy], "train")
    idx = np.arange(len(data["lens"])); T = int(data["lens"].max())
    A, P, M = (torch.from_numpy(data[k][idx, :T]) for k in ("A", "P", "M"))
    Sl = batch_sets(data, idx, T)["sets"]
    joint = _allowed_under_true_victims(geo, A, P, M, F, Sl)
    assert len(joint) > 500
    n_cmp = 0
    for i in range(40):
        L = int(data["lens"][i])
        for j in Sl[i, :L].unique().tolist():
            pos = np.nonzero(Sl[i, :L].numpy() == j)[0]
            sub = lambda X: torch.from_numpy(np.asarray(X)[i:i + 1, pos])
            alone = _allowed_under_true_victims(geo, sub(data["A"]), sub(data["P"]), sub(data["M"]),
                                                np.asarray(F)[i:i + 1, pos], None)
            for (_, tt), al in alone.items():
                assert joint[(i, int(pos[tt]))] == al; n_cmp += 1
    assert n_cmp > 100


@torch.no_grad()
def test_set_identity_is_used(ms_data):
    geo = data_geometry(ms_data["drrip"])
    data = load_streams(ms_data["drrip"], "train", geo["num_blocks"])
    m = _model(geo, n_set_ids=geo["n_set_ids"])
    idx = np.arange(40); T = int(data["lens"][idx].max())
    A = torch.from_numpy(data["A"][idx, :T]); kw = batch_sets(data, idx, T)
    a = m(A, **kw)
    b = m(A, sets=kw["sets"], set_ids=(kw["set_ids"] + 1) % geo["n_set_ids"])
    assert not torch.equal(a, b)
    with pytest.raises(ValueError):
        m(A, sets=kw["sets"])
