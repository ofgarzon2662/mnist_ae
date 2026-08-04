# mnist_ae – From Notebook to Python Package

This guide walks you **step-by-step** through turning the `CIML25_MNIST_Intro_v6.ipynb` notebook into a distributable Python package that you can install anywhere, including SDSC Expanse. It assumes you already know how to run a Jupyter notebook and have **Python ≥ 3.9** available (Python 3.11 recommended).
## Introductory Video
https://github.com/user-attachments/assets/791bf691-32d1-458a-a816-d76942a65b64
## 0.1  Clone the repository
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
2. The exported file can be a single, monolithic script – notebooks aren’t always written with clean architecture in mind.  After export, audit the code (or ask an advanced LLM such as Codex Terra, or Sonnet5) and refactor it into small, SOLID-compliant modules.

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
python -m build --wheel        # produces an installable wheel under dist/
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

## 6  Install and run on Expanse

These instructions use the Summer Institute 2026 GPU reservation, the provided PyTorch/CUDA Singularity container, and the published [`mnist-ae` package](https://pypi.org/project/mnist-ae/).

### 6.1  Log in to Expanse

Run this command from your local computer, replacing `<username>` with your Expanse username:

```bash
ssh <username>@login.expanse.sdsc.edu
```

Do not run the training program directly on the login node. Computational work must run through Slurm on a compute node.

### 6.2  Get the Summer Institute helper script

Clone the Summer Institute repository on the Expanse login node:

```bash
cd ~
git clone https://github.com/sdsc/sdsc-summer-institute-2026.git
cd ~/sdsc-summer-institute-2026
```

If you already downloaded the repository, update it instead:

```bash
cd ~/sdsc-summer-institute-2026
git pull
```

### 6.3  Request an interactive GPU node

Choose either method below. Using the Summer Institute script is recommended because it contains the appropriate Slurm account, reservation, QoS, GPU, CPU, memory, and time settings.

#### Option A: Use the Summer Institute script

```bash
cd ~/sdsc-summer-institute-2026
bash srun-gpu.sh
```

The script is located in the repository root and is fully commented so you can inspect the Slurm options before running it.

#### Option B: Run a short Slurm request manually

Alternatively, request a 30-minute interactive GPU session directly:

```bash
srun --account=sdp173 \
     --reservation=si26gpu \
     --partition=gpu-shared \
     --qos=gpu-shared-eot \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=96G \
     --gpus=1 \
     --time=00:30:00 \
     --pty bash -l
```

After using either method, wait until the prompt changes from a login-node hostname to a compute-node hostname similar to:

```text
username@exp-1-60
```

Confirm that you are inside a Slurm GPU job:

```bash
hostname
echo "$SLURM_JOB_ID"
nvidia-smi
```

A nonempty Slurm job ID and successful `nvidia-smi` output indicate that you are running inside a GPU job.

> The `si26gpu` reservation and `gpu-shared-eot` QoS are available only during the scheduled Summer Institute reservation. If the script's requested time extends beyond the reservation's remaining time, use Option B to request a shorter session.

### 6.4  Create a persistent working directory

Run these commands on the allocated compute node:

```bash
mkdir -p ~/mnist_ae_expanse
cd ~/mnist_ae_expanse
```

The MNIST dataset and trained model will be saved in this directory.

### 6.5  Enter the PyTorch/CUDA container

Expanse's default system Python is too old for this package. Do not load Python or CUDA manually. Instead, use the provided Singularity container:

```bash
module load singularitypro

singularity shell --nv \
    --bind /expanse,/scratch,/cm \
    /cm/shared/examples/sdsc/si/2026/ptl-cuda-12-1.sif
```

The prompt should change to:

```text
Singularity>
```

Run all remaining Python commands inside this container.

### 6.6  Verify Python, PyTorch, and GPU access

```bash
python3 --version

python3 -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

`CUDA available` must print:

```text
True
```

### 6.7  Install the package from PyPI

For reproducibility, install the exact version used during the Summer Institute:

```bash
python3 -m pip install --user --no-deps mnist-ae==2.0.0
```

The `--no-deps` option is intentional. The container already provides GPU-enabled PyTorch and torchvision; allowing `pip` to reinstall them could replace the compatible CUDA environment.

Verify the installed version:

```bash
python3 -c "from importlib.metadata import version; \
print('mnist-ae:', version('mnist-ae'))"
```

Expected result:

```text
mnist-ae: 2.0.0
```

### 6.8  Run a short validation test

This run uses one epoch, four training batches, and one test batch. It verifies package installation, GPU access, data downloading, training, evaluation, and model saving:

```bash
python3 -m mnist_ae.mnist_training \
    --epochs 1 \
    --batch_size 256 \
    --max_train_batches 4 \
    --max_test_batches 1
```

### 6.9  Run the workshop training example

```bash
python3 -m mnist_ae.mnist_training \
    --epochs 10 \
    --batch_size 256 \
    --max_train_batches 100 \
    --max_test_batches 10
```

Increasing the number of batches is more important than increasing the number of epochs alone. With a batch size of 256, four training batches contain only 1,024 images.

### 6.10  Run the complete MNIST workflow

To use the entire training and test datasets, omit both batch-limit arguments:

```bash
python3 -m mnist_ae.mnist_training \
    --epochs 10 \
    --batch_size 256
```

The run produces:

```text
data/MNIST/
mnist_cnn.pth
```

Confirm the saved model:

```bash
ls -lh mnist_cnn.pth
```

During the initial dataset download, some URLs may return `HTTP Error 404`. This is harmless if the program subsequently downloads the same file from a PyTorch backup server.

### 6.11  Exit and release the GPU

First leave the Singularity container:

```bash
exit
```

Then leave the compute node and release the Slurm allocation:

```bash
exit
```

Confirm that you are back on the Expanse login node:

```bash
hostname
```

For additional details, see the [official SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html).

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
