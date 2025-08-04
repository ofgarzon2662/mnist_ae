"""Basic unit tests for the mnist_ae package.
Run with::

    pytest -q
"""
from pathlib import Path
import sys
import types
import argparse

import torch
import pytest

from mnist_ae import mnist_training as mt

# ------------------------------------------------------------------------------------
# Simple unit checks -----------------------------------------------------------------
# ------------------------------------------------------------------------------------

def test_get_default_device():
    dev = mt.get_default_device()
    assert isinstance(dev, torch.device)


def test_mynet_output_shape():
    model = mt.MyNet()
    xb = torch.randn(8, 1, 28, 28)
    out = model(xb)
    assert out.shape == (8, 10)
    assert torch.allclose(out.exp().sum(1), torch.ones(8), atol=1e-4)


def test_mynet_flat_features():
    model = mt.MyNet(num_filters=16, kernel_size=3)
    # internal sanity: linear1 should take (C*H*W) features
    conv_out = 28 - 3 + 1         # formula from code comment
    pooled = (conv_out - 3) // 2 + 1
    expected_in = 16 * pooled * pooled
    assert model.linear1.in_features == expected_in


# ------------------------------------------------------------------------------------
# Training & eval on synthetic data ---------------------------------------------------
# ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_loaders():
    xs = torch.randn(32, 1, 28, 28)
    ys = torch.randint(0, 10, (32,))
    ds = torch.utils.data.TensorDataset(xs, ys)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    return dl, dl


def test_train_and_eval(tiny_loaders):
    tr, te = tiny_loaders
    model = mt.MyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = mt.train_epoch(model, tr, opt, torch.device("cpu"))
    assert loss > 0
    acc = mt.evaluate(model, te, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


# ------------------------------------------------------------------------------------
# Smoke-test the CLI entry point ------------------------------------------------------
# ------------------------------------------------------------------------------------

def test_main_smoke(monkeypatch, tmp_path, tiny_loaders):
    """Run `main()` end-to-end but patched to avoid disk+network IO."""

    # 1. Redirect get_dataloaders -> our tiny ones
    monkeypatch.setattr(mt, "get_dataloaders", lambda *a, **k: tiny_loaders)

    # 2. Avoid actually writing a .pth file
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)

    # 3. Pretend CUDA is unavailable (to stay on CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # 4. Run main() with custom argv
    test_args = ["prog", "--epochs", "1", "--batch_size", "8"]
    monkeypatch.setattr(sys, "argv", test_args)

    mt.main()  # should run without error
