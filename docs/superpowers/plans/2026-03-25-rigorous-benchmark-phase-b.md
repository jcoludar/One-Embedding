# Phase B: Cross-Dataset Validation + Multi-PLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-validate SS3/SS8 on TS115+CASP12, cross-validate disorder on TriZOD348, run ESM2 multi-PLM validation on CB513+SCOPe — all with paired bootstrap retention CIs.

**Architecture:** Single new script `experiments/43_rigorous_benchmark/run_phase_b.py` that reuses all Phase A1 infrastructure (runners, probes, bootstrap CIs). New dataset loaders for TS115/CASP12 and TriZOD. Embedding extraction for TS115/CASP12 (~136 proteins).

**Tech Stack:** Existing Phase A1 modules + ProtT5 extraction via `src.one_embedding.extract`

---

### Task 1: TS115/CASP12 dataset loaders + embedding extraction

**Files:**
- Create: `experiments/43_rigorous_benchmark/datasets/netsurfp.py`

- [ ] **Step 1: Write the dataset loader**

TS115.csv and CASP12.csv have the same format as CB513.csv: `input,dssp3,dssp8,disorder,cb513_mask`. Reuse `load_cb513_csv` logic.

```python
"""NetSurfP cross-validation datasets: TS115 and CASP12.

Same CSV format as CB513 (input, dssp3, dssp8, disorder, cb513_mask).
These are independent test sets for cross-validating SS3/SS8 results.
"""
import csv
from pathlib import Path
import numpy as np


def load_netsurfp_csv(csv_path: Path) -> tuple[dict, dict, dict]:
    """Load TS115 or CASP12 CSV.

    Returns:
        (sequences, ss3_labels, ss8_labels) — same format as CB513.
    """
    csv_path = Path(csv_path)
    sequences, ss3_labels, ss8_labels = {}, {}, {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pid = f"{csv_path.stem}_{i}"
            sequences[pid] = row["input"]
            ss3_labels[pid] = row["dssp3"]
            ss8_labels[pid] = row["dssp8"]
    return sequences, ss3_labels, ss8_labels
```

- [ ] **Step 2: Extract ProtT5 embeddings for TS115 + CASP12**

Create a small extraction script. ~136 proteins total, should take ~1 minute on M3 Max.

```python
# experiments/43_rigorous_benchmark/extract_netsurfp_embeddings.py
"""Extract ProtT5 embeddings for TS115 and CASP12 proteins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import h5py
from experiments.exp43_datasets_netsurfp import load_netsurfp_csv

# This will use the project's ProtT5 extractor
from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings

DATA = Path(__file__).resolve().parents[2] / "data"

for dataset in ["TS115", "CASP12"]:
    csv_path = DATA / "per_residue_benchmarks" / f"{dataset}.csv"
    out_path = DATA / "residue_embeddings" / f"prot_t5_xl_{dataset.lower()}.h5"

    if out_path.exists():
        print(f"  {dataset}: already extracted at {out_path}")
        continue

    sequences, _, _ = load_netsurfp_csv(csv_path)
    print(f"  {dataset}: extracting {len(sequences)} proteins...")
    extract_prot_t5_embeddings(sequences, str(out_path))
    print(f"  {dataset}: saved to {out_path}")
```

- [ ] **Step 3: Commit**

```bash
git add experiments/43_rigorous_benchmark/datasets/netsurfp.py
git commit -m "feat(exp43): TS115/CASP12 dataset loader for SS3/SS8 cross-validation"
```

---

### Task 2: TriZOD dataset loader

**Files:**
- Create: `experiments/43_rigorous_benchmark/datasets/trizod.py`

- [ ] **Step 1: Write the loader**

TriZOD data is already loaded by the existing `load_chezod_seth` function pattern, but we need to handle the TriZOD-specific format. Check what exists in the project.

```python
"""TriZOD dataset loader for disorder cross-validation.

TriZOD348 is an independent test set for validating CheZOD disorder results.
Embeddings already extracted: data/residue_embeddings/prot_t5_xl_trizod.h5
Split: data/benchmark_suite/splits/trizod_predefined.json (train: 5438, test: 348)
"""
import json
from pathlib import Path
import numpy as np
import h5py


def load_trizod(
    embeddings_path: Path,
    split_path: Path,
    scores_dir: Path = None,
) -> dict:
    """Load TriZOD embeddings and split.

    Returns dict with 'embeddings', 'train_ids', 'test_ids'.
    Scores must be loaded separately from the TriZOD data files.
    """
    # Load embeddings
    embeddings = {}
    with h5py.File(str(embeddings_path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)

    # Load split
    with open(split_path) as f:
        split = json.load(f)

    return {
        "embeddings": embeddings,
        "train_ids": split["train_ids"],
        "test_ids": split["test_ids"],
    }
```

- [ ] **Step 2: Commit**

```bash
git add experiments/43_rigorous_benchmark/datasets/trizod.py
git commit -m "feat(exp43): TriZOD dataset loader for disorder cross-validation"
```

---

### Task 3: Phase B benchmark script

**Files:**
- Create: `experiments/43_rigorous_benchmark/run_phase_b.py`

This is the main script. It runs 4 benchmark groups:

**Group 1: SS3/SS8 cross-validation** (train on CB513, test on CB513/TS115/CASP12)
- Use same CB513 training set, but test on all three datasets
- Report Q3/Q8 for raw and compressed on each test set
- Compute retention with paired bootstrap CIs
- Check cross-dataset consistency (Rule 14)

**Group 2: Disorder cross-validation** (train on CheZOD1174, test on CheZOD117 + TriZOD348)
- Report pooled Spearman rho for both test sets
- Compute retention with paired bootstrap CIs
- Check cross-dataset consistency

**Group 3: ESM2 multi-PLM** (same tasks on ESM2 embeddings)
- SS3/SS8 on ESM2 CB513 (1280d raw vs 768d compressed)
- Retrieval on ESM2 SCOPe 5K

**Group 4: Cross-check summary**
- Compare retention across datasets and PLMs
- Flag any divergence > 3pp

The script should:
- Import all Phase A1 infrastructure
- Handle missing files gracefully
- Report every metric with CI
- Report every retention with paired bootstrap CI
- Save results JSON

- [ ] **Step 1: Write the script**

(The implementer should write this following the run_phase_a1.py pattern but covering the 4 groups above. Key details:
- Load CB513 raw embeddings + labels (already done in A1, reuse)
- Load TS115/CASP12 via `datasets.netsurfp.load_netsurfp_csv`
- For TS115/CASP12 embeddings: extract if missing, then load from H5
- Compress using Codec fitted on SCOPe 5K corpus (same as A1)
- Train probes on CB513 train set, evaluate on CB513 test + TS115 + CASP12
- For TriZOD: load via split, use CheZOD1174 for training, TriZOD348 for testing
- For ESM2: same pattern but with ESM2 embedding files
- For ESM2 compression: Codec(d_out=768) fitted on ESM2 SCOPe corpus
- Use `paired_bootstrap_retention` for all retention numbers
- Use `check_cross_dataset_consistency` for divergence checks)

- [ ] **Step 2: Test the script**

Run: `uv run python experiments/43_rigorous_benchmark/run_phase_b.py`

- [ ] **Step 3: Commit**

```bash
git add experiments/43_rigorous_benchmark/run_phase_b.py
git commit -m "feat(exp43): Phase B cross-validation + multi-PLM benchmarks"
```

---

### Task 4: Run Phase B and document results

- [ ] **Step 1: Run TS115/CASP12 embedding extraction**

Run: `uv run python experiments/43_rigorous_benchmark/extract_netsurfp_embeddings.py`

- [ ] **Step 2: Run Phase B benchmarks**

Run: `uv run python experiments/43_rigorous_benchmark/run_phase_b.py`

- [ ] **Step 3: Review cross-dataset consistency**

Check: Do SS3 retention numbers agree within 3pp across CB513/TS115/CASP12?
Check: Do disorder retention numbers agree within 3pp across CheZOD/TriZOD?

- [ ] **Step 4: Commit results**

```bash
git add data/benchmarks/rigorous_v1/phase_b_results.json
git commit -m "data(exp43): Phase B cross-validation results — SS3/SS8/disorder/ESM2"
```

---

## Summary

| Task | What | Key files |
|------|------|-----------|
| 1 | TS115/CASP12 loader + extraction | `datasets/netsurfp.py` |
| 2 | TriZOD loader | `datasets/trizod.py` |
| 3 | Phase B benchmark script | `run_phase_b.py` |
| 4 | Run + document | results JSON |

Total: **4 tasks, ~4 commits**.
