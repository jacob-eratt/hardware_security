import pyarrow as pa
import torch

from cpl.data.dataset import VictimDataset
from cpl.data.fake import build_fake, history_id, split_of, victim_lru, victim_plru_recency
from cpl.model.gru import VictimGRU
from cpl.model.losses import soft_cross_entropy
from cpl.utils.seed import set_seed


def test_fake_rule_vectors():
    cases = [(4, [], 0, 0), (4, [0], 2, 1), (4, [1, 1, 1], 2, 0), (4, [2, 0, 3], 1, 1), (16, [0, 5, 9], 2, 1), (16, [3, 3, 3, 7, 0, 12], 4, 1)]
    for W, h, plru, lru in cases:
        assert victim_plru_recency(W, h) == plru
        assert victim_lru(W, h) == lru


def test_fake_schema_and_splits():
    table, meta = build_fake(16, "sim_plru_W16", 1, 3000, 200)
    assert meta["fake"] is True
    assert table.schema.field("victim_counts").type == pa.list_(pa.int32(), 16)
    assert table.schema.field("history").type == pa.list_(pa.int8())
    df = table.to_pandas()
    for _, r in df.head(200).iterrows():
        if r["split"] != "test_long":
            assert split_of(history_id(16, list(r["history"]))) == r["split"]
    ids = {s: set(df.loc[df["split"] == s, "history_id"]) for s in df["split"].unique()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["train"] & ids["test_long"]


def test_overfit_256():
    set_seed(0)
    table, _ = build_fake(16, "sim_plru_W16", 0, 400, 0)
    df = table.to_pandas()
    df = df[df["hist_len"] > 0].head(256).reset_index(drop=True)
    ds = VictimDataset(df)
    b = ds.batch(ds.indices())
    model = VictimGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = soft_cross_entropy(model(b["tokens"], b["new_pos"], b["W"]), b["victim_counts"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(b["tokens"], b["new_pos"], b["W"]).argmax(-1) == b["label"]).float().mean().item()
    assert acc == 1.0
