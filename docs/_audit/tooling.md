# Tooling Inventory (Task C.8)

**Audited:** 2026-04-26
**Repo HEAD:** `83d21ef`
**Python venv:** `.venv` (Python 3.12.12, uv 0.9.8)

## Declared in `pyproject.toml`

The current `pyproject.toml` declares only ONE dev tool:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.2",
]
```

No optional `dev` group under `[project.optional-dependencies]`; only `biocentral`
and `extreme` are listed there (runtime extras, not dev tools).

No `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`, `[tool.coverage]`, or `[tool.black]`
sections exist. The only `[tool.*]` block is `[tool.uv]` (`package = false`).

No tool config files exist in the repo:
- no `.pre-commit-config.yaml`
- no `mypy.ini` / `ruff.toml` / `.ruff.toml`
- no `setup.cfg` / `pytest.ini`

## Per-tool status

| Tool | Declared in pyproject? | Installed in venv? | Runnable? | Action |
|------|-----------------------|--------------------|-----------|--------|
| ruff           | no  | no  | no  | install (D.4) |
| mypy           | no  | no  | no  | install (D.4) — lenient mode |
| pytest         | yes (`>=9.0.2`) | yes (9.0.2) | yes | none |
| pytest-cov     | no  | no  | no  | install (D.4) |
| pre-commit     | no  | no  | no  | install (D.4) — minimal config |
| jupyter        | no  | no  | no  | install (D.4) — needed for `experiments/45_disorder_forensics.ipynb` |
| jupytext       | no  | no  | no  | install (D.4) — pair `.ipynb` ↔ `.py` for diff/review |
| marp-cli (npm) | n/a | no (npm not installed globally) | no | install (D.4) — blocks slide deck for H.1 |

### How "runnable?" was tested

```bash
uv run --no-sync <tool> --version
```

Only `pytest 9.0.2` returned a version string. All other Python tools failed with
`No such file or directory`. `marp` is not on `PATH`; `npx` IS available
(`/opt/homebrew/bin/npx`, v11.6.2), so the install path for marp-cli is
`npm install -g @marp-team/marp-cli` (or `npx --yes @marp-team/marp-cli`).

### `.venv/bin/` evidence

The venv contains only the binaries supplied by runtime deps (`pytest`, `optuna`,
`transformers`, `torchrun`, `hf`, etc.) — no linter, no formatter, no type
checker, no pre-commit hooks, no notebook server. Only `pytest` (and the alias
`py.test`) is present from the dev side.

## Severity classification

- [GREEN] **`pytest`** — declared, installed, runs (813/813 tests pass, see C.2).
- [YELLOW] **`ruff`** — missing. Not blocking immediate work; the codebase has zero
  declared linter and is internally consistent enough that a Rost-lab inspector
  would notice the absence rather than an actual style bug.
- [YELLOW] **`mypy`** — missing. Same reasoning as `ruff`. No type-stub strategy
  in the repo at all (no `py.typed` marker, no inline `from typing import ...`
  hygiene enforcement).
- [YELLOW] **`pytest-cov`** — missing. Coverage is not measured. For a research
  repo this is acceptable, but any "we have N tests" claim is undercut by
  having no coverage figure.
- [YELLOW] **`pre-commit`** — missing. No hooks installed; no `.pre-commit-config.yaml`.
  Visible to anyone who clones, but not blocking.
- [YELLOW] **`jupyter` + `jupytext`** — missing. The repo contains
  `experiments/45_disorder_forensics.ipynb` (a 1-cell forensics notebook). It
  WILL fail to render in CI / GitHub if anyone tries to re-execute it. Worth
  installing for D.4, but not on the talk's critical path.
- [RED] **`marp-cli`** — missing AND blocks Task H.1 (slide deck production).
  This is the only tool whose absence has a documented downstream blocker in
  the lab-talk-prep plan. Install via `npm install -g @marp-team/marp-cli`
  in D.4 (or invoke as `npx --yes @marp-team/marp-cli`).

## Distribution

1 GREEN / 6 YELLOW / 1 RED.

The single RED is `marp-cli` because the Phase H slide deck cannot be produced
without it. All other missing tools are nice-to-have for repo professionalism;
none would cause a concrete result/claim in the talk to be wrong.

## Recommendations for D.4

1. Install `marp-cli` first (unblocks H.1).
2. Add a `[dependency-groups]` `dev` block in pyproject.toml with
   `ruff`, `mypy`, `pytest-cov`, `pre-commit`, `jupyter`, `jupytext`.
3. Add a minimal `[tool.ruff]` section (line-length 100, target-version py312,
   default rule selection) and `[tool.mypy]` (lenient mode: `ignore_missing_imports`,
   no `--strict`). Goal is "tools exist and run", not "every line clean".
4. Optional: `.pre-commit-config.yaml` with trailing-whitespace, end-of-file-fixer,
   `check-yaml`, `check-added-large-files` (max 500KB), and `ruff format --check`.
   No mypy in pre-commit (too slow for hooks).

## Notes for the talk

A Rost-lab inspector running `ls .pre-commit-config.yaml mypy.ini` will see they
are absent. The honest framing is: this is a research repo, the protocol
rigour lives in `experiments/43_rigorous_benchmark/` (BCa CIs, GridSearchCV,
cluster bootstrap — all audited GREEN in C.4), not in style enforcement. D.4
will close the gap before talk day so the optics match the substance.
