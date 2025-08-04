# mnist_ae – From Notebook to Python Package

This guide walks you **step-by-step** through turning the `CIML25_MNIST_Intro_v6.ipynb` notebook into a distributable Python package that you can install anywhere (even on TSCC).  It assumes you already know how to run a Jupyter notebook, and that you have **Python ≥ 3.8** available (Python 3.11 recommended).

## 0  Clone the repository

```bash
git clone https://github.com/<your-username>/mnist_ae.git
cd mnist_ae
```

Feel free to fork the project first if you want your own remote.

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

### Install project requirements (to run the notebook)
The notebook itself depends on **PyTorch** and **torchvision** (plus NumPy, etc.).  The easiest way is to use the pinned list that comes with the repo:

```bash
pip install -r requirements.txt      # installs CPU wheels by default
```

If you already have GPU-enabled PyTorch, feel free to skip this step or install only the libraries you miss:

```bash
pip install torch torchvision
```

> 🗒️ **Why a venv?**  Keeping build tools isolated avoids polluting your base Python and makes the process reproducible.

---

## 1½  Place the notebook in `nbs/`

If your starting file is `CIML25_MNIST_Intro_v6.ipynb`, move (or copy) it into the `nbs/` directory **and** rename it to the more compact `01_mnist_intro.ipynb` so nbdev can pick it up.

### Windows
```powershell
move CIML25_MNIST_Intro_v6.ipynb nbs\01_mnist_intro.ipynb
```

### macOS / Linux
```bash
mv CIML25_MNIST_Intro_v6.ipynb nbs/01_mnist_intro.ipynb
```

> nbdev scans all notebooks inside `nbs/`. The numeric prefix (`01_`, `02_`, …) also sets the order of the generated documentation.

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

### 3½  Sync metadata & version (optional but recommended)

Before building, open `settings.ini` and update:

```
version      = 0.0.2        # bump each release
requirements = torch torchvision   # runtime deps only
```

Then run

```bash
nbdev_prepare      # sync settings → pyproject.toml, tag version, install git hooks
```

### Inspect what nbdev generated
`nbdev_prepare` rewrites `pyproject.toml`, regenerates type stubs, and may reformat your code. **Open the `mnist_ae/` folder** and look at the newly-created or updated modules.

**Recommendations:**
1. **Do *not* mark long training loops or plotting cells with `#| export`.**  Keep exploratory code in the notebook; only export reusable library functions and models. Heavy loops inside the package will run every time someone imports it and can waste GPU/CPU hours.
2. The exported file can be a single, monolithic script – notebooks aren’t always written with clean architecture in mind.  After export, audit the code (or ask an advanced LLM, o3 from ChatGPT is recommended, as well as Gemini2.5 or any other reasoning model) and refactor it into small, SOLID-compliant modules.

Use this starter prompt to guide the refactor:
```text
You are a senior Python engineer. Rewrite the file `mnist_ae/mnist_training.py` so that:
• Each class/function has one clear responsibility (Single-Responsibility Principle).
• Related functionality is grouped into modules (e.g. data, model, training, cli).
• Internal helpers are made private (_prefix).
• No global execution at import-time; provide a `main()` entry point.
• Add type hints and docstrings.
Return the full, refactored code as a valid Python package structure.
```

**What is SOLID?**  It’s a set of five design guidelines for maintainable OO code:

* **S — Single Responsibility:** each module/class/function does one job.
* **O — Open/Closed:** code is open for extension but closed for modification.
* **L — Liskov Substitution:** derived classes can stand in for their base without breaking behaviour.
* **I — Interface Segregation:** prefer many small, specific interfaces over one large general-purpose interface.
* **D — Dependency Inversion:** depend on abstractions (interfaces), not concrete implementations.

Spend some time on this step; clean structure pays off later.

---

---

## 4  Build the wheel (binary package)

```bash
python -m build --wheel        # produces dist/mnist_ae-0.0.1-py3-none-any.whl
```

The file inside `dist/` is a **portable package** that can be installed with `pip install <file>.whl` on any machine that has Python ≥ the minimum you set.

### 4½  Test the wheel locally

### 4¾  Run unit tests from source
If you’re working from the cloned repo rather than the installed wheel, install the package in *editable* mode so Python can find it:

```bash
pip install -e .[dev]   # or just `pip install -e .` if you skipped dev extras
pytest --cov=mnist_ae -q  # run tests **and** show coverage %
```

If `mnist_ae` is not importable you’ll get a `ModuleNotFoundError`; the editable install (or adding the repo root to `PYTHONPATH`) solves that.
```bash
pip install --force-reinstall dist/mnist_ae-*.whl
python -m mnist_ae.mnist_training --epochs 1 --batch_size 128  # quick sanity run
```

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
module load python3 cuda            # adjust to cluster versions
python -m venv ~/mnist_env && source ~/mnist_env/bin/activate


# now install your package from PyPI
pip install mnist_ae

# (alternative) install a local wheel -- You'd have to scp your local *.whl to TSCC.
# pip install ~/dist/mnist_ae-0.0.1-py3-none-any.whl

# launch training
python -m mnist_ae.mnist_training --epochs 5 --batch_size 256
```

Check the time it takes for these 5 epocs and compare to your local run. Spot any significant difference?

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
