"""Import-smoke for every Exp-43 entry-point script.

A module that imports a name which no longer exists fails at import time — the
script is dead on arrival before it can do any work. This guards against that
class of regression (it would have caught both the config CROSS_CHECK_* removal
and the run_phase_b dead-import drift).
"""

import importlib
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"
sys.path.insert(0, str(EXP.parents[1]))   # repo root (for `src` imports)
sys.path.insert(0, str(EXP))              # experiment dir (for config/runners)


@pytest.mark.parametrize("module", [
    "config",
    "runners.localization",
    "runners.per_residue",
    "runners.protein_level",
    "run_v2_validation",
    "run_phase_a1",
    "run_phase_b",
    "run_phase_c",
    "run_phase_d",
    "run_abtt_stability",
])
def test_module_imports(module):
    importlib.import_module(module)
