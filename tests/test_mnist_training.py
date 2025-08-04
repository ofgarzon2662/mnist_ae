"""Basic unit tests for the mnist_ae package.
Run with::

    pytest -q

The tests are intentionally lightweight so they run quickly on CPU-only
machines (CI, laptops).  We create synthetic tensors instead of downloading
MNIST.
"""
from pathlib import Path

import torch
import pytest

from mnist_ae import mnist_training as mt


def test_get_default_device():
    """`get_default_device` should return a torch.device."""
    dev = mt.get_default_device()
    assert isinstance(dev, torch.device)
    # On a CPU-only box this should still pass
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_mynet_output_shape():
    """Forward pass produces (batch, 10) logits."""
    model = mt.MyNet()
    xb = torch.randn(8, 1, 28, 28)  # mini-batch of fake images
    out = model(xb)
    assert out.shape == (8, 10)
    # log-softmax => rows sum to 0 in probability space => exp + sum == 1
    probs = out.exp()
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


@pytest.fixture(scope="module")
def tiny_dataloaders():
    """In-memory dataloaders with 32 random samples (no disk IO)."""
    rng = torch.Generator().manual_seed(0)
    xs = torch.randn(32, 1, 28, 28, generator=rng)
    ys = torch.randint(0, 10, (32,), generator=rng)
    ds = torch.utils.data.TensorDataset(xs, ys)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    return dl, dl  # train, test


def test_train_epoch_smoke(tiny_dataloaders):
    """`train_epoch` runs without error and returns scalar loss."""
    train_dl, _ = tiny_dataloaders
    model = mt.MyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = mt.train_epoch(model, train_dl, opt, device=torch.device("cpu"))
    assert isinstance(loss, float) and loss > 0


def test_evaluate_accuracy(tiny_dataloaders):
    """`evaluate` returns accuracy between 0 and 1."""
    _, test_dl = tiny_dataloaders
    model = mt.MyNet()
    acc = mt.evaluate(model, test_dl, device=torch.device("cpu"))
    assert 0.0 <= acc <= 1.0
