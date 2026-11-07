import torch

from cpl.data.dataset import collate
from cpl.data.tokens import PAD, W_MAX
from cpl.model.gru import VictimGRU, count_parameters
from cpl.model.losses import soft_cross_entropy


def _batch(Ws, lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for W, L in zip(Ws, lens):
        h = torch.randint(0, W, (L,), generator=g).tolist()
        vc = torch.zeros(W_MAX, dtype=torch.int32)
        vc[torch.randint(0, W, (1,), generator=g)] = 1
        samples.append({"history": h, "W": W, "victim_counts": vc.numpy()})
    return collate(samples)


def test_param_count():
    n = count_parameters(VictimGRU())
    assert 150_000 <= n <= 175_000


def test_forward_shapes_and_mask():
    model = VictimGRU()
    b = _batch([4, 8, 16, 4], [0, 5, 32, 12])
    logits = model(b["tokens"], b["new_pos"], b["W"])
    assert logits.shape == (4, W_MAX)
    probs = logits.softmax(-1)
    for i, W in enumerate(b["W"].tolist()):
        assert torch.all(probs[i, W:] == 0)
        assert torch.isfinite(logits[i, :W]).all()
    assert torch.allclose(probs.sum(-1), torch.ones(4))


def test_int_W_mask():
    model = VictimGRU()
    b = _batch([8, 8], [3, 7])
    logits = model(b["tokens"], b["new_pos"], 8)
    assert torch.isinf(logits[:, 8:]).all() and torch.isfinite(logits[:, :8]).all()


def test_gather_uses_new_pos_not_last_step():
    torch.manual_seed(0)
    model = VictimGRU().eval()
    b = _batch([4, 4], [2, 20])
    short = {k: v[:1] for k, v in b.items()}
    T = int(short["new_pos"].item()) + 1
    with torch.no_grad():
        padded = model(b["tokens"][:1], b["new_pos"][:1], b["W"][:1])
        trimmed = model(short["tokens"][:, :T], short["new_pos"], short["W"])
    assert (b["tokens"][0, T:] == PAD).all()
    assert torch.allclose(padded, trimmed, atol=1e-6)


def test_loss_finite_and_grads():
    model = VictimGRU()
    b = _batch([4, 16, 8], [1, 9, 30])
    loss = soft_cross_entropy(model(b["tokens"], b["new_pos"], b["W"]), b["victim_counts"])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_soft_labels():
    logits = torch.tensor([[0.0, 0.0, float("-inf"), float("-inf")]])
    counts = torch.tensor([[3, 1, 0, 0]])
    loss = soft_cross_entropy(logits, counts)
    expected = -(0.75 * torch.log(torch.tensor(0.5)) + 0.25 * torch.log(torch.tensor(0.5)))
    assert torch.allclose(loss, expected)
