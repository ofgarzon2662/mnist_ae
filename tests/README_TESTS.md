# Running the test-suite for **mnist_ae**

This mini-guide shows how to:

1.  create an isolated Python environment;
2.  install the package **and** the testing tools;
3.  execute the tests locally with code-coverage;
4.  rely on GitHub Actions to run the same checks automatically for every new push / PR.

---

## 1  Set-up an environment

Any recent Python (≥ 3.9) works; the examples below use 3.11.

### Using the standard library `venv` (recommended for quick hacks)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Using Conda / Mamba
```bash
conda create -n mnist_ae python=3.11
conda activate mnist_ae
```

---

## 2  Install dependencies

The test-suite needs **pytest** and **pytest-cov** in addition to the
package itself.

```bash
python -m pip install --upgrade pip
pip install -e .[dev]            # editable install + dev extras (if defined)
# or explicitly:
# pip install -e . pytest pytest-cov
```

> The editable (`-e`) install ensures that tests import the *local source*
rather than a wheel downloaded from PyPI.

---

## 3  Run the tests locally

```bash
pytest -q                                # quick run

# With coverage (terminal + XML report)
pytest --cov=mnist_ae --cov-report=term --cov-report=xml -q
```

By default the suite is CPU-only and finishes in a few seconds.

If you only care about the coverage percentage you can make the run fail
when it drops below 75 %:

```bash
pytest --cov=mnist_ae --cov-fail-under=75 -q
```

The XML file (`coverage.xml`) can be consumed by external tools such as
Codecov or SonarQube.

---

## 4  What exactly do these tests check?

The suite is intentionally **lightweight**—each test runs entirely on CPU
with synthetic tensors, so CI finishes in seconds.

Key checks include:

| Test | Purpose |
|------|---------|
| `test_get_default_device` | Ensures `get_default_device()` returns a valid `torch.device` and respects CUDA availability. |
| `test_mynet_output_shape` | Verifies that a forward pass through `MyNet` returns a tensor of shape `(batch, 10)` and that the logits form a proper log-softmax distribution. |
| `test_train_and_eval` | Runs **one mini-epoch** on random data to make sure the training loop executes and returns finite loss/accuracy values. |
| `test_main_smoke` | Monkey-patches data loading & file I/O, then executes `main()` to ensure the CLI wiring works without downloading MNIST or touching disk. |

They *do not* run a full training loop on 60 000 MNIST images—that would
be wasteful in CI and on student laptops.

### Why these minimal guarantees matter

1. **Import sanity:** the module can be imported without side-effects.
2. **API stability:** core functions/classes keep their signatures and
   expected tensor shapes.
3. **Fast feedback:** you’ll know within seconds if a refactor broke the
   public API or basic maths, long before running an expensive training
   script.

---

## 5  Continuous Integration with GitHub Actions

A workflow file lives in `.github/workflows/test-coverage.yml` and does
this on every push **to the `develop` branch whose commit message
contains the word “Deploy”**:

1.  Checks out the repo.
2.  Installs Python 3.11.
3.  Installs dependencies (`pip install -e . pytest pytest-cov`).
4.  Runs `pytest --cov … --cov-fail-under=75`.
5.  Prints a success message if coverage ≥ 75 %.

If the threshold is not met the job fails, blocking the future PR.

### Triggering the workflow manually
1. Push to `develop` with a commit message that includes “Deploy”.
2. Or click *Actions → Test & Coverage → Run workflow* in the GitHub UI.

### Raising or lowering the bar
Edit the `--cov-fail-under` value inside `test-coverage.yml` to change
the required percentage.

---

Happy testing! 🎉
