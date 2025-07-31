# mnist_ae – From Notebook to Python Package

This guide walks you **step-by-step** through turning the `01_mnist_intro.ipynb` notebook into a distributable Python package that you can install anywhere (even on TSCC).  It assumes you already know how to run a Jupyter notebook, and that you have **Python ≥ 3.8** available (Python 3.11 recommended).

---

## 1  Set up a clean Python environment

### Windows ( PowerShell or cmd )
```powershell
:: create & activate a virtual-env in the project root
python -m venv .venv
.venv\Scripts\activate          # cmd
# or
.\.venv\Scripts\Activate.ps1    # PowerShell
```

### macOS / Linux ( bash / zsh )
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip & install build-time tools:
```bash
pip install --upgrade pip nbdev build wheel twine
```

> 🗒️ **Why a venv?**  Keeping build tools isolated avoids polluting your base Python and makes the process reproducible.

---

## 2  Run & explore the notebook

```bash
jupyter notebook nbs/01_mnist_intro.ipynb
```

Execute a few cells to verify the model trains as expected (each epoch should take only a few seconds on CPU).

---

## 3  Export code with nbdev

nbdev turns specially-marked cells into a Python module.  The two directives you need to know are:

* `#| default_exp mnist_training` – appears once, tells nbdev *which module file* to create (`mnist_training.py`).
* `#| export` – placed on any cell whose code you want included in the library.

The **intro notebook already contains** these directives, so exporting is a one-liner:

```bash
nbdev_export            # generates mnist_ae/mnist_training.py
```

(Optional) update metadata in `settings.ini` – package name, version, runtime requirements, author, etc.  nbdev will read this file when we build the wheel.

---

## 4  Build the wheel (binary package)

```bash
python -m build --wheel        # produces dist/mnist_ae-0.0.1-py3-none-any.whl
```

The file inside `dist/` is a **portable package** that can be installed with `pip install <file>.whl` on any machine that has Python ≥ the minimum you set.

---

## 5  Publish to (Test)PyPI  
*(skip if you only need a local wheel)*

1. Create an account on [pypi.org](https://pypi.org) (and on [test.pypi.org](https://test.pypi.org) for dry-runs).
2. Generate an **API token**:  *Settings → API tokens → New token*.
3. Upload:

```bash
# one-time: store credentials safely or export as env-vars
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-********************************"

# upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# if everything looks good, push to the real PyPI
python -m twine upload dist/*
```

Once published, anyone can install with
```bash
pip install mnist_ae      # replace with the final project name
```

---

## 6  Install & run on TSCC (or any HPC)

```bash
# inside a job script or interactive srun session
module load python/3.11 cuda/12.4              # adjust to cluster versions
python -m venv ~/mnist_env && source ~/mnist_env/bin/activate

# install GPU-matched PyTorch first (example for CUDA 12.4)
pip install torch==2.3.0+cu124 torchvision==0.18.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# now install your package from PyPI (or from a wheel file you copied over)
pip install mnist_ae            # or pip install ~/dist/mnist_ae-0.0.1-py3-none-any.whl

# launch training
python -m mnist_ae.mnist_training --epochs 5 --batch_size 256
```

---

## Appendix – Common commands (Windows vs Unix)

| Task                     | Windows (PowerShell)                         | macOS / Linux (bash)            |
|--------------------------|----------------------------------------------|---------------------------------|
| Activate venv            | `.\.venv\Scripts\Activate.ps1`              | `source .venv/bin/activate`     |
| Deactivate venv          | `deactivate`                                 | `deactivate`                    |
| Upgrade pip              | `python -m pip install --upgrade pip`        | `pip install --upgrade pip`     |
| Run nbdev export         | `nbdev_export`                               | `nbdev_export`                  |
| Build wheel              | `python -m build --wheel`                    | `python -m build --wheel`       |
| Upload with twine        | `python -m twine upload dist/*`              | same                            |
| Install wheel            | `pip install dist\mnist_ae-*.whl`            | `pip install dist/mnist_ae-*.whl` |

That’s it!  You’ve gone from a Jupyter notebook to a published, pip-installable Python package 🎉
