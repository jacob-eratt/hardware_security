"""History IDs, split assignment, generators and the end-to-end dataset writer (spec §4-5)."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cpl.data.generate import build_dataset, generate_histories
from cpl.data.histories import (
    collect,
    draw_random,
    generate_diagnostic,
    generate_random_split,
    history_id,
    split_of,
)
from cpl.policies import LRU, SRRIP, TreePLRU, run_experiment
from cpl.utils.seed import derive_rng


# ----------------------------------------------------------------------- IDs and splits
def test_history_id_formula():
    for W, h in [(16, [0, 5, 9]), (16, []), (4, [3, 3, 1]), (8, list(range(8)) * 4)]:
        expected = hashlib.sha1(("W%d:" % W + ",".join(str(x) for x in h)).encode()).hexdigest()[:16]
        assert history_id(W, h) == expected
        assert history_id(W, np.array(h, dtype=np.int8)) == expected  # numpy ints too
    assert history_id(4, [1, 2]) != history_id(8, [1, 2])  # W is part of the key
    assert history_id(16, [1, 12]) != history_id(16, [11, 2])  # separator matters


def test_split_thresholds_and_fractions():
    # u = int(id, 16) / 16**16; 0.9 -> 0xe666..., 0.95 -> 0xf333...
    assert split_of("0" * 16) == "train"
    assert split_of("e600000000000000") == "train"
    assert split_of("e700000000000000") == "val"
    assert split_of("f300000000000000") == "val"
    assert split_of("f400000000000000") == "test"
    assert split_of("f" * 16) == "test"
    rng = np.random.default_rng(0)
    splits = [split_of(history_id(16, rng.integers(0, 16, 10).tolist())) for _ in range(40_000)]
    frac = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert abs(frac["train"] - 0.90) < 0.01 and abs(frac["val"] - 0.05) < 0.005


def test_derive_rng_independent_and_stable():
    a = derive_rng(1234, "split", "val").integers(0, 1 << 30, 5)
    b = derive_rng(1234, "split", "val").integers(0, 1 << 30, 5)
    c = derive_rng(1234, "split", "test").integers(0, 1 << 30, 5)
    d = derive_rng(1235, "split", "val").integers(0, 1 << 30, 5)
    assert (a == b).all() and not (a == c).all() and not (a == d).all()


# ----------------------------------------------------------------------- generators
def test_draw_random_shape_and_padding():
    hist, lens = draw_random(8, 5_000, (3, 20), np.random.default_rng(0))
    assert hist.shape == (5_000, 20) and lens.min() >= 3 and lens.max() <= 20
    assert set(np.unique(lens)) == set(range(3, 21))
    for row, L in zip(hist, lens):
        assert ((row[:L] >= 0) & (row[:L] < 8)).all() and (row[L:] == -1).all()


def test_collect_dedup_rules():
    # W=2, L in [0,6]: tiny history space -> plenty of duplicates
    b, _ = collect(2, 400, (0, 6), np.random.default_rng(0), max_draws=200_000)
    long_ids = [i for i, L in zip(b.ids, b.lens) if L >= 4]
    assert len(long_ids) == len(set(long_ids))  # no duplicates for L >= 4
    short_ids = [i for i, L in zip(b.ids, b.lens) if L <= 3]
    assert len(short_ids) > len(set(short_ids))  # duplicates kept for L <= 3
    assert b.ids == [history_id(2, h) for h in b.to_lists()]


def test_collect_gives_up_with_warning():
    with pytest.warns(UserWarning, match="only"):
        b, drawn = collect(2, 1_000, (4, 4), np.random.default_rng(0), max_draws=5_000, batch_size=1_000)
    assert len(b) == 16 and drawn == 5_000  # only 2^4 distinct histories of length 4


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_hash_split_membership_and_determinism(split):
    b1, _ = generate_random_split(16, split, 2_000, (0, 32), derive_rng(7, "split", split))
    b2, _ = generate_random_split(16, split, 2_000, (0, 32), derive_rng(7, "split", split))
    assert len(b1) == 2_000
    assert all(split_of(i) == split for i in b1.ids)
    assert b1.ids == b2.ids and np.array_equal(b1.hist, b2.hist)


def test_non_hash_split_respects_exclusion():
    rng = np.random.default_rng(0)
    excl = {history_id(4, [])}
    b, _ = generate_random_split(4, "test_long", 300, (0, 1), rng, exclude_ids=excl)
    assert history_id(4, []) not in b.ids and set(b.lens.tolist()) == {1}


@pytest.mark.parametrize("require", ["any", "all_pairwise"])
def test_generate_diagnostic(require):
    pols = ["plru", "lru"] if require == "any" else ["plru", "lru", "srrip"]
    b, _ = generate_diagnostic(pols, 1_000, 16, (0, 32), np.random.default_rng(0), require=require)
    assert len(b) == 1_000 and b.gen_type == "diagnostic"
    ref = {"plru": TreePLRU(16), "lru": LRU(16), "srrip": SRRIP(16)}
    for h in b.to_lists()[:300]:
        v = [run_experiment(ref[p], h)[0] for p in pols]
        assert len(set(v)) == len(v)  # with 2 policies "any" == "differ"


def test_generate_diagnostic_max_tries_warns():
    # PLRU == LRU for W=2, so no diagnostic history exists
    with pytest.warns(UserWarning):
        b, _ = generate_diagnostic(["plru", "lru"], 10, 2, (0, 8), np.random.default_rng(0), max_tries=10_000)
    assert len(b) == 0


# ----------------------------------------------------------------------- end-to-end
def _small_cfg(W=16):
    return {
        "seed": 99,
        "target": f"sim_plru_W{W}",
        "policy": "plru",
        "W": W,
        "data": {
            "out_dir": "unused",
            "sizes": {"train": 4_000, "val": 400, "test": 400, "test_long": 300,
                      "test_xlong": 100, "test_diag": 300},
            "len_ranges": {"train": [0, 32], "test_long": [33, 128],
                           "test_xlong": [129, 512], "test_diag": [0, 32]},
            "diag_require": "any",
            "diag_policies": ["plru", "lru"],
        },
        "noise": None,
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    meta = build_dataset(_small_cfg(), d / "out", histories_path=d / "hist" / "histories.parquet")
    return d, meta, pq.read_table(d / "out" / "samples.parquet").to_pandas()


def test_schema_and_types(built):
    d, meta, df = built
    t = pq.read_table(d / "out" / "samples.parquet")
    expected = ["sample_id", "target", "source", "policy", "W", "cache_level", "set_index", "slice",
                "set_role", "forced_mode", "run_id", "history_id", "history", "hist_len", "gen_type",
                "victim_counts", "n_valid_repeats", "label_hard", "label_entropy", "sim_victims",
                "plru_bits", "split"]
    assert t.column_names == expected
    s = t.schema
    assert s.field("W").type == pa.int8() and s.field("hist_len").type == pa.int16()
    assert s.field("label_hard").type == pa.int8() and s.field("label_entropy").type == pa.float32()
    assert s.field("history").type.equals(pa.list_(pa.int8()))
    assert s.field("victim_counts").type.equals(pa.list_(pa.int32(), 16))
    assert s.field("plru_bits").type.equals(pa.list_(pa.int8(), 15))
    assert [f.name for f in s.field("sim_victims").type] == ["plru", "lru", "fifo", "srrip"]
    for c in ["cache_level", "set_index", "slice", "set_role", "forced_mode", "run_id"]:
        assert df[c].isna().all()
    assert (df["source"] == "sim").all() and (df["policy"] == "plru").all()
    assert df["sample_id"].iloc[0] == "sim_plru_W16_00000000" and df["sample_id"].is_unique


def test_split_sizes_and_lengths(built):
    _, meta, df = built
    cfg = _small_cfg()
    assert meta["counts_per_split"] == cfg["data"]["sizes"]
    assert df["split"].value_counts().to_dict() == cfg["data"]["sizes"]
    rng_of = {"train": (0, 32), "val": (0, 32), "test": (0, 32), "test_long": (33, 128),
              "test_xlong": (129, 512), "test_diag": (0, 32)}
    for s, (lo, hi) in rng_of.items():
        L = df.loc[df.split == s, "hist_len"]
        assert L.min() >= lo and L.max() <= hi
    assert (df.loc[df.split == "test_diag", "gen_type"] == "diagnostic").all()
    assert (df.loc[df.split != "test_diag", "gen_type"] == "random").all()


def test_no_leakage(built):
    _, _, df = built
    owners = df.groupby("history_id")["split"].nunique()
    assert (owners == 1).all()
    hs = df[df.split.isin(["train", "val", "test"])]
    assert (hs["history_id"].map(split_of) == hs["split"]).all()
    assert (df["history_id"] == [history_id(16, h) for h in df["history"]]).all()


def test_labels_match_reference_simulator(built):
    _, _, df = built
    sims = {"plru": TreePLRU(16), "lru": LRU(16), "srrip": SRRIP(16)}
    sample = df.sample(1_500, random_state=0)
    for _, r in sample.iterrows():
        h = r["history"].tolist()
        v, bits = run_experiment(sims["plru"], h)
        assert r["label_hard"] == v
        assert r["victim_counts"].tolist() == [int(i == v) for i in range(16)]
        assert r["plru_bits"].tolist() == bits.tolist()
        sv = r["sim_victims"]
        assert sv["plru"] == v and sv["fifo"] == 0
        assert sv["lru"] == run_experiment(sims["lru"], h)[0]
        assert sv["srrip"] == run_experiment(sims["srrip"], h)[0]
        if r["split"] == "test_diag":
            assert sv["plru"] != sv["lru"]
    assert (df["label_entropy"] == 0).all() and (df["n_valid_repeats"] == 1).all()


def test_dedup_in_written_data(built):
    _, _, df = built
    long = df[df.hist_len >= 4]
    assert not long.duplicated(["split", "history_id"]).any()


def test_meta_and_histories_table(built):
    d, meta, df = built
    on_disk = json.loads((d / "out" / "meta.json").read_text())
    assert on_disk["counts_per_split"] == meta["counts_per_split"]
    assert on_disk["seed"] == 99 and "victim_frac" in on_disk["splits"]["train"]
    h = pq.read_table(d / "hist" / "histories.parquet").to_pandas()
    assert list(h.columns) == ["history_id", "W", "history", "hist_len", "gen_type"]
    assert h["history_id"].is_unique and set(h["history_id"]) == set(df["history_id"])


def test_histories_table_merges(tmp_path):
    hp = tmp_path / "histories.parquet"
    m16 = build_dataset(_small_cfg(16), tmp_path / "a", scale=0.1, histories_path=hp)
    m4 = build_dataset(_small_cfg(4), tmp_path / "b", scale=0.1, histories_path=hp)
    h = pq.read_table(hp).to_pandas()
    assert set(h["W"]) == {4, 16} and h["history_id"].is_unique
    assert m4["histories_parquet_rows"] == len(h) > m16["histories_parquet_rows"]


def test_reproducible(tmp_path):
    a = build_dataset(_small_cfg(), tmp_path / "a", scale=0.25, histories_path=None)
    b = build_dataset(_small_cfg(), tmp_path / "b", scale=0.25, histories_path=None)
    ta = pq.read_table(tmp_path / "a" / "samples.parquet")
    tb = pq.read_table(tmp_path / "b" / "samples.parquet")
    assert ta.equals(tb)
    assert a["splits"]["train"]["victim_counts"] == b["splits"]["train"]["victim_counts"]


def test_split_streams_independent():
    """Changing one split's size must not change the histories of the others (spec §10)."""
    cfg = _small_cfg()
    bigger = copy.deepcopy(cfg)
    bigger["data"]["sizes"]["val"] = 900
    a, _ = generate_histories(cfg)
    b, _ = generate_histories(bigger)
    for s in ("train", "test", "test_long", "test_xlong"):
        assert a[s].ids == b[s].ids, s
    assert b["val"].ids[:400] == a["val"].ids


def test_noise_not_yet_supported(tmp_path):
    cfg = _small_cfg()
    cfg["noise"] = {"eps": 0.05, "repeats": 5, "label": "soft"}
    with pytest.raises(NotImplementedError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_dataset(cfg, tmp_path, scale=0.05, histories_path=None)
