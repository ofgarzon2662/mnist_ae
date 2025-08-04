# ✍️ Exercise – CI Workflow for Feature Branches

In this exercise you will design a **GitHub Actions** workflow that
orchestrates a typical *feature → develop → master* flow while enforcing a
minimum level of test coverage.

Scenario
--------
* The repository has two permanent branches: `develop` and `master`.
* Each new piece of work is carried out on a branch named
  `feature/<something>` (e.g. `feature/fine_tuning`).
* Whenever you push to a **feature** branch **and** the commit message
  contains the word `Deploy`, a workflow must:

1.  *Synchronise* – fast-forward the feature branch with the latest commits
    from `develop` (equivalent to `git merge origin/develop`).
2.  *Test* – create a fresh Python 3.11 environment, install the project in
    editable mode, and run `pytest --cov=mnist_ae --cov-fail-under=75`.
3.  *Integrate* – if, and only if, all tests pass **and** coverage ≥ 75 %,
    merge the feature branch **back into `develop`**.
4.  *Inform* – leave a short success message in the Action log so the team
    knows the branch is now part of `develop` (or print an error summary
    otherwise).

Your task is to write the workflow file that automates those four steps.

Hints
-----
* Use the Action trigger `on: push` with a branch filter
  `feature/**`.
* Gate the job with
  `if: contains(github.event.head_commit.message, 'Deploy')`.
* Remember to set `permissions: contents: write` for the job that pushes.

Skeleton workflow
-----------------
```yaml
# .github/workflows/feature-ci.yml
name: Feature branch CI + auto-merge

on:
  push:
    branches: [ "feature/**" ]

jobs:
  test-and-merge:
    if: contains(github.event.head_commit.message, 'Deploy')
    runs-on: ubuntu-latest
    permissions:
      contents: write   # allow pushing merges

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0            # we need full history for merges

      # ------------------------------------------------------------------
      # 1. Fast-forward feature with latest develop
      # ------------------------------------------------------------------
      - name: Merge develop into feature/*
        run: |
          git config user.name  "github-actions"
          git config user.email "actions@github.com"
          git fetch origin develop
          git merge --no-edit origin/develop
          git push origin HEAD:${{ github.ref }}

      # ------------------------------------------------------------------
      # 2. Set up Python & install dependencies
      # ------------------------------------------------------------------
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies & test tools
        run: |
          python -m pip install --upgrade pip
          grep -v "^pywin32" requirements.txt > req.txt
          pip install [ ... ]
          pip install [ ... ]

      # ------------------------------------------------------------------
      # 3. Run test-suite + coverage gate
      # ------------------------------------------------------------------
      - name: Run unit tests
        run: [ ... ]

      # ------------------------------------------------------------------
      # 4. Merge feature → develop (only if all previous steps succeeded)
      # ------------------------------------------------------------------
      - name: Merge to develop
        if: success()
        run: |
          git fetch origin develop
          git checkout develop
          [ ... ]
          [ ... ]

      - name: Done
        if: success()
        run: echo "✅ CI passed & coverage ≥ 75 %. Branch merged into develop."
```

> **Challenge** – Fill the Gaps for this new GH Action to implement CI from feature/* to develop branch

