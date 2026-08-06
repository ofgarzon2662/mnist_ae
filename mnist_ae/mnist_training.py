#| default_exp mnist_training
# (put this as the first code cell in the notebook)

#| export
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


NOTEBOOK_EQUIVALENT_SETTINGS = {
    "epochs": 10,
    "lr": 0.01,
    "batch_size": 256,
    "max_train_batches": 4,
    "max_test_batches": 1,
    "num_filters": 16,
    "kernel_size": 3,
    "seed": 777,
    "shuffle": False,
}


#─────────────────────────────── Model ────────────────────────────────#
#| export
class MyNet(nn.Module):
    "A tiny CNN for 28×28 MNIST images."
    def __init__(self, num_filters: int = 32, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, stride=1)
        self.pool  = nn.MaxPool2d(3, 2)

        conv_out   = 28 - kernel_size + 1           # 28 → conv → conv_out
        pooled     = (conv_out - 3) // 2 + 1        # conv_out → pool → pooled
        flat_dim   = num_filters * pooled * pooled  # C × H × W

        self.linear1 = nn.Linear(flat_dim, 16)
        self.linear2 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N,1,28,28)
        x = F.relu(self.conv1(x))                                # → (N,C,H,W)
        x = self.pool(x)                                         # → (N,C,H',W')
        x = torch.flatten(x, 1)                                  # → (N, flat)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


#────────────────────────── Data utilities ───────────────────────────#
#| export
def get_default_device(verbose: bool = False) -> torch.device:
    "Return `cuda` if available, else `cpu`."
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if verbose:
        print(f"Using {dev}")
    return dev


#| export
def set_random_seed(seed: int = 777) -> None:
    "Seed the PyTorch random-number generators used by this program."
    if seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#| export
def apply_notebook_equivalent_settings(
    args: argparse.Namespace,
) -> argparse.Namespace:
    "Apply the settings used by the original demonstration notebook."
    for name, value in NOTEBOOK_EQUIVALENT_SETTINGS.items():
        setattr(args, name, value)
    return args


#| export
def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str | Path = "data",
    shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    "Return (train_loader, test_loader) for MNIST."
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    tr_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tfm)
    te_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    pin = torch.cuda.is_available()
    tr_dl = torch.utils.data.DataLoader(
        tr_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    te_dl = torch.utils.data.DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False
    )
    return tr_dl, te_dl


#──────────────────────── Train / Eval loops ────────────────────────#
#| export
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        seen += xb.size(0)
    return running / max(seen, 1)


#| export
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct, total = 0, 0
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / max(total, 1)


#────────────────────────────── CLI ────────────────────────────────#
#| export
def main() -> None:
    "Command-line entry point: trains the model and prints metrics."
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches",  type=int, default=None)
    p.add_argument("--num_filters", type=int, default=32)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--notebook-equivalent", action="store_true")
    p.set_defaults(shuffle=True)
    args = p.parse_args()

    if args.notebook_equivalent:
        apply_notebook_equivalent_settings(args)

    set_random_seed(args.seed)
    device = get_default_device(verbose=True)

    model = MyNet(
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
    ).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_dl, te_dl = get_dataloaders(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            model, tr_dl, opt, device, max_batches=args.max_train_batches
        )
        acc     = evaluate(model, te_dl, device, max_batches=args.max_test_batches)
        print(f"epoch {ep:2d} | {time.time()-t0:4.1f}s | "
              f"loss {tr_loss:.4f} | acc {acc*100:.2f}%")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Weights saved to mnist_cnn.pth")


#────────────────────────── Re-exports ───────────────────────────#
__all__ = [
    "NOTEBOOK_EQUIVALENT_SETTINGS",
    "MyNet",
    "get_default_device",
    "set_random_seed",
    "apply_notebook_equivalent_settings",
    "get_dataloaders",
    "train_epoch",
    "evaluate",
    "main",
]

if __name__ == "__main__":
    main()
