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


def test_set_random_seed_repeats_random_sequence(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mt.set_random_seed(2026)
    first = torch.rand(5)
    mt.set_random_seed(2026)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_set_random_seed_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        mt.set_random_seed(-1)


def test_notebook_equivalent_settings_match_original_notebook():
    args = argparse.Namespace()

    returned = mt.apply_notebook_equivalent_settings(args)

    assert returned is args
    assert vars(args) == mt.NOTEBOOK_EQUIVALENT_SETTINGS


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


def test_train_epoch_respects_max_batches(tiny_loaders):
    tr, _ = tiny_loaders
    model = mt.MyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    step_counter = {"count": 0}
    orig_step = opt.step

    def counted_step(*args, **kwargs):
        step_counter["count"] += 1
        return orig_step(*args, **kwargs)

    opt.step = counted_step
    loss = mt.train_epoch(model, tr, opt, torch.device("cpu"), max_batches=2)

    assert loss > 0
    assert step_counter["count"] == 2


def test_evaluate_respects_max_batches():
    class AlwaysZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.size(0), 10), device=x.device)

    xs = torch.randn(32, 1, 28, 28)
    ys = torch.tensor([0] * 8 + [1] * 24)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs, ys), batch_size=8, shuffle=False
    )
    model = AlwaysZeroModel()

    full_acc = mt.evaluate(model, dl, torch.device("cpu"))
    quick_acc = mt.evaluate(model, dl, torch.device("cpu"), max_batches=1)

    assert pytest.approx(full_acc, rel=1e-6) == 0.25
    assert pytest.approx(quick_acc, rel=1e-6) == 1.0


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
    test_args = [
        "prog",
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--max_train_batches",
        "1",
        "--max_test_batches",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    mt.main()  # should run without error
