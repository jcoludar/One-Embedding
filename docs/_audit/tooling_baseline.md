# Tooling baseline (D.4) — 2026-04-27

## Versions
ruff 0.15.12
mypy 1.20.2 (compiled: yes)
pytest 9.0.2
pre-commit 4.6.0
Selected Jupyter core packages...
IPython          : 9.13.0
ipykernel        : 7.2.0
ipywidgets       : 8.1.8
jupyter_client   : 8.8.0
jupyter_core     : 5.9.1
jupyter_server   : 2.17.0
deptry 0.25.1
marp-cli: 

## ruff check (baseline — not yet fixed)
13 | |     svd_spectrum,
14 | | )
   | |_^
15 |
16 |   RNG = np.random.RandomState(42)
   |
help: Organize imports

Found 659 errors.
[*] 482 fixable with the `--fix` option (44 hidden fixes can be enabled with the `--unsafe-fixes` option).

## mypy (baseline — not yet fixed)
src/compressors/attention_pool.py:80: error: Incompatible types in assignment (expression has type "Sequential", variable has type "Linear")  [assignment]
src/compressors/attention_pool.py:145: error: Value of type "Tensor | Module" is not indexable  [index]
src/evaluation/benchmark_suite.py:113: error: Incompatible types in assignment (expression has type "dict[str, float]", target has type "str")  [assignment]
src/evaluation/benchmark_suite.py:115: error: Incompatible types in assignment (expression has type "dict[str, object]", target has type "str")  [assignment]
src/evaluation/benchmark_suite.py:118: error: Incompatible types in assignment (expression has type "dict[str, float]", target has type "str")  [assignment]
src/evaluation/benchmark_suite.py:122: error: Incompatible types in assignment (expression has type "dict[str, Any]", target has type "str")  [assignment]
src/evaluation/benchmark_suite.py:133: error: Incompatible types in assignment (expression has type "dict[str, float]", target has type "str")  [assignment]
src/evaluation/benchmark_suite.py:146: error: Incompatible types in assignment (expression has type "dict[str, float]", target has type "str")  [assignment]
src/extraction/prot_t5_extractor.py:28: error: Argument 1 to "__call__" of "_Wrapped" has incompatible type "device"; expected "PreTrainedModel"  [arg-type]
Found 167 errors in 28 files (checked 86 source files)

## pytest collect-only
ERROR tools/reference/VespaG/tests - ModuleNotFoundError: No module named 'ja...
ERROR tools/reference/evolocity/tests/test_basic.py
ERROR tools/reference/vcmsa/tests/vcmsa_test.py
!!!!!!!!!!!!!!!!!!! Interrupted: 7 errors during collection !!!!!!!!!!!!!!!!!!!!
819 tests collected, 7 errors in 5.56s

## deptry
Scanning 162 files...

[1m[32mSuccess! No dependency issues found.[m
