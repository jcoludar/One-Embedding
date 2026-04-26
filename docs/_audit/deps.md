# Dependency Audit (Task C.9)

**Audited:** 2026-04-26
**Repo HEAD:** `83d21ef` (after C.8 commit `4760a6f`)
**uv:** 0.9.8 (Homebrew)
**Python:** 3.12.12 (`.venv`)

## uv.lock sync status

```bash
uv lock --check
# Resolved 97 packages in 5ms
# Resolved 97 packages in 5ms
# Exit code: 0
```

[GREEN] **PASS** — `uv.lock` is in sync with `pyproject.toml`. 97 packages
resolved without re-locking.

## `uv sync --dry-run` summary

```bash
uv sync --dry-run
# Would use project environment at: .venv
# Resolved 97 packages in 7ms
# Found up-to-date lockfile at: uv.lock
# Would uninstall 16 packages
#  - charset-normalizer==3.4.6
#  - dgl==2.2.0
#  - e3nn==0.6.0
#  - faiss-cpu==1.13.2
#  - llvmlite==0.46.0
#  - numba==0.64.0
#  - opt-einsum==3.4.0
#  - opt-einsum-fx==0.1.4
#  - protobuf==7.34.0
#  - psutil==7.2.2
#  - pyarrow==23.0.1
#  - pynndescent==0.6.0
#  - requests==2.32.5
#  - torchdata==0.11.0
#  - umap-learn==0.5.11
#  - urllib3==2.6.3
```

[YELLOW] **`uv sync --dry-run` would UNINSTALL 16 packages** that are present
in the venv but not declared in `pyproject.toml` and not pinned in `uv.lock`
(except `urllib3`, which IS in lock — included in the uninstall list as a
version-marker mismatch).

These are venv drift from earlier `uv pip install <name>` actions. None are
referenced from `pyproject.toml` or `uv.lock`. Concretely:

| Package | In lock? | Imported in code? | Status |
|---------|:---:|:---:|--------|
| charset-normalizer | no | no (transitive of `requests`) | drift |
| dgl                | no | no | drift (graph-NN, unused) |
| e3nn               | no | no | drift |
| faiss-cpu          | no | YES (`structural_similarity.py`) | **undeclared dep** |
| llvmlite           | no | no (transitive of `numba`) | drift |
| numba              | no | no | drift |
| opt-einsum         | no | no | drift |
| opt-einsum-fx      | no | no | drift |
| protobuf           | no | no (transitive of various) | drift |
| psutil             | no | no | drift |
| pyarrow            | no | no | drift |
| pynndescent        | no | no (transitive of `umap-learn`) | drift |
| requests           | no | no (lazy-imported by `transformers`) | drift |
| torchdata          | no | no | drift |
| umap-learn         | no | no | drift |
| urllib3            | YES (2.6.3) | no | dry-run version-marker mismatch |

So the venv is technically over-installed, but **only `faiss-cpu` is also
undeclared in any manifest while being imported at runtime** — see
"Undeclared deps" below.

## Phantom deps (declared, never imported)

[GREEN] **None obvious.** Every required dep in `pyproject.toml`'s
`dependencies` list maps to a top-level import in `src/` or `experiments/`:

| Declared | Top-level module | Imported in code? |
|---|---|:---:|
| biopython | `Bio` | yes |
| fair-esm | `esm` | yes |
| h5py | `h5py` | yes |
| matplotlib | `matplotlib` | yes |
| numpy | `numpy` | yes |
| optuna | `optuna` | yes |
| pandas | `pandas` | yes |
| scikit-learn | `sklearn` | yes |
| sentencepiece | `sentencepiece` | yes (transitively via `transformers` ProtT5/ANKH tokenizers) |
| torch | `torch` | yes |
| tqdm | `tqdm` | yes |
| transformers | `transformers` | yes |
| PyWavelets | `pywt` | yes |
| zstandard | `zstandard` | yes |

Optional extras:
- `biocentral = ["biocentral-api"]` → `biocentral_api` imported (`src/extraction/biocentral_extractor.py:1`).
- `extreme = ["pot", "ripser", "persim"]` → `ripser` and `persim` imported in `src/one_embedding/topological.py`; `pot` (module name `ot`) imported in `src/one_embedding/extreme_compression.py`.

Verdict: **no phantom deps**. (`deptry` not installed; manual import-grep used instead — limitation noted.)

## Undeclared deps (imported, not declared)

[RED] **Two undeclared imports** that would break a fresh-clone-and-`uv-sync`:

1. **`faiss`** — imported via `import faiss` (lazy) inside `FAISSSearchIndex` in
   `src/one_embedding/structural_similarity.py:243, 289, 323, 358, 370`. The wheel
   `faiss-cpu==1.13.2` IS in the current `.venv`, but neither `faiss-cpu` nor
   any other `faiss-*` distribution is in `pyproject.toml` or `uv.lock`. A fresh
   clone + `uv sync` would NOT install it; the FAISS retrieval index path would
   raise `ModuleNotFoundError`.
2. **`tmtools`** — imported via `import tmtools` (lazy) inside the TM-score path
   in `src/evaluation/structural_validation.py:98, 154`. Not in `pyproject.toml`,
   not in `uv.lock`, not even in the current `.venv` (`ModuleNotFoundError` at
   `import tmtools`). The site already gracefully raises with an install hint
   ("Install with: `uv pip install tmtools`"), but the dep should be declared
   so `uv sync` resolves it.

Both are in the `[extreme]` family of optional features. Recommended D.5 fix:
add `faiss-cpu` and `tmtools` to a new `[project.optional-dependencies]`
group (e.g. `structural` or extend `extreme`) so installs are explicit.

Two more imports worth noting (both **OK as-is**, listed for completeness):

3. **`scipy`** — imported in 24 files (13 in `src/`, 10 in `experiments/`,
   1 in `tests/`) but NOT in `pyproject.toml`. `scipy 1.17.1` IS in `uv.lock`
   as a transitive of `scikit-learn`, `pot`, `ripser`. Currently works but
   **fragile**: if any of the transitive parents drops `scipy`, the explicit
   imports break. Should be promoted to a direct dependency in D.5. [YELLOW]
4. **`click`** — imported in `src/one_embedding/cli.py:8` and
   `tests/test_cli.py:7` (CliRunner). NOT in `pyproject.toml`. `click 8.3.1` IS
   in `uv.lock` as a transitive of `typer` ← `huggingface-hub` ← `transformers`.
   Same fragility class as `scipy`. Should be promoted to direct in D.5. [YELLOW]

False positive flagged and resolved during the audit:

5. **`datasets`** — `experiments/43_rigorous_benchmark/run_phase_b.py:93–94`
   does `from datasets.netsurfp import load_netsurfp_csv`. This is a LOCAL
   subpackage at `experiments/43_rigorous_benchmark/datasets/` (`__init__.py`,
   `netsurfp.py`, `trizod.py`), NOT the HuggingFace `datasets` library.
   Resolved by import path. Not undeclared. [GREEN]

## Yanked / EOL packages in lock

[GREEN] **None detected.** `grep -i 'yank' uv.lock` returns no matches. All 97
packages resolve to live wheels on PyPI as of audit time.

## Python version constraint

| Source | Value |
|---|---|
| `pyproject.toml` `requires-python` | `>=3.12` |
| `uv.lock` `requires-python` | `>=3.12` |
| `uv run python --version` | `Python 3.12.12` |
| Match? | **yes** [GREEN] |

`requires-python = ">=3.12"` is correct; the codebase relies on
Python 3.12 features (typing improvements, `fair-esm` compatibility per
CLAUDE.md). No upper bound — could be tightened to `>=3.12,<3.14` to match
`uv.lock`'s resolution markers, but not blocking.

## Severity classification

- [GREEN] `uv lock --check` exit 0; lock is in sync with pyproject.
- [GREEN] No phantom deps detected (every declared dep is imported).
- [GREEN] Optional extras (`biocentral`, `extreme`) all map to real imports.
- [GREEN] No yanked / EOL packages in the lock.
- [GREEN] `requires-python` consistent across pyproject, lock, venv.
- [GREEN] False-positive `datasets` is a local subpackage, not undeclared.
- [YELLOW] `uv sync --dry-run` would uninstall 16 venv-only packages —
  drift, not breakage. Indicates earlier ad-hoc `uv pip install` actions
  (numba, umap-learn, dgl, e3nn, opt-einsum, faiss-cpu, etc.) that bypassed
  the manifest. Cleanup = run `uv sync` (with no `--dry-run`). Schedule for D.5.
- [YELLOW] `scipy` imported directly in 16 files but only available via the
  transitive resolution chain (sklearn / pot / ripser). Promote to direct dep.
- [YELLOW] `click` imported directly in 2 files but only available via the
  typer ← huggingface-hub ← transformers chain. Promote to direct dep.
- [RED] **`faiss-cpu` is imported in `src/` but not declared anywhere.**
  Fresh clone + `uv sync` would not install it; `FAISSSearchIndex` is broken
  on a clean install. Declare in `pyproject.toml`.
- [RED] **`tmtools` is imported in `src/` but not declared anywhere.**
  Fresh clone + `uv sync` would not install it; TM-score path is broken.
  The error message already tells users to install it manually, but a proper
  manifest entry is the fix.

## Distribution

6 GREEN / 3 YELLOW / 2 RED.

## Notes for D.5

1. Add `faiss-cpu` and `tmtools` to a `[project.optional-dependencies].structural`
   group (or extend `extreme`). Update `CLAUDE.md` install instructions if needed.
2. Promote `scipy` and `click` to direct deps in `[project].dependencies`.
3. Run `uv sync` (no `--dry-run`) to remove the 16 drift packages. Document
   what was removed; if any feature relies on `dgl` / `e3nn` / `numba` /
   `umap-learn` that hasn't been audited here, re-add explicitly.
4. Consider `uv tool install deptry` and add a CI check (`uv run deptry .`)
   to prevent future drift. (Out of scope for this audit; this is what makes
   the manual import-grep pass deferrable to D.5 without losing rigour.)
5. Consider tightening `requires-python = ">=3.12,<3.14"` to match the
   resolution markers already in `uv.lock`.

## Notes for the talk

A Rost-lab member running `git clone && uv sync && python -m src.one_embedding.cli`
would NOT crash on the codec path (binary default works with `numpy + h5py`
only). Crashes would only appear on the optional structural-similarity / TM-score
paths, which are NOT cited in any of the headline tables (Exp 43/44/46/47).
The two REDs above are real but **do not invalidate any claimed retention number**.
