import numpy as np
import pytest

from cpl.data.dataset import collate, encode_padded
from cpl.data.tokens import FILL, NEW, PAD, SEP, VOCAB_SIZE, block_token, encode, new_position, seq_len


def test_vocab_constants():
    assert (PAD, FILL, SEP, NEW) == (0, 1, 2, 3)
    assert block_token(0) == 4 and block_token(31) == 35
    assert VOCAB_SIZE == 48
    with pytest.raises(ValueError):
        block_token(32)


@pytest.mark.parametrize("W", [4, 8, 16])
@pytest.mark.parametrize("history", [[], [0], [3, 1, 2, 0, 3]])
def test_sequence_layout(W, history):
    seq = encode(history, W)
    L = len(history)
    assert len(seq) == seq_len(W, L) == W + L + 3
    assert seq[0] == FILL
    assert seq[1 : W + 1] == [4 + i for i in range(W)]
    assert seq[W + 1] == SEP
    assert seq[W + 2 : W + 2 + L] == [4 + h for h in history]
    assert seq[new_position(W, L)] == NEW == seq[-1]


def test_padding_and_new_pos():
    hs = [[], [1, 2, 3], [0] * 10]
    tokens, new_pos = encode_padded(hs, [4, 4, 4])
    assert tokens.shape == (3, 4 + 10 + 3)
    for i, h in enumerate(hs):
        assert tokens[i, new_pos[i]] == NEW
        assert new_pos[i] == 4 + len(h) + 2
        assert (tokens[i, new_pos[i] + 1 :] == PAD).all()


def test_collate():
    vc = np.zeros(16, np.int32)
    vc[2] = 1
    b = collate([{"history": [1, 1], "W": 4, "victim_counts": vc}, {"history": [], "W": 8, "victim_counts": vc}])
    assert b["tokens"].shape == (2, 8 + 0 + 3)
    assert b["new_pos"].tolist() == [8, 10]
    assert b["tokens"][0, 8].item() == NEW and b["tokens"][1, 10].item() == NEW
    assert (b["tokens"][0, 9:] == PAD).all()
    assert b["label"].tolist() == [2, 2]
