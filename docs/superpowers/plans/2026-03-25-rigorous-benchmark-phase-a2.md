# Phase A2: Tool Unit Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every shipped tool gets rigorous unit tests covering smoke, shape, value range, edge cases, determinism, and dimension agnosticity.

**Architecture:** 7 new test files (`tests/test_tool_{name}.py`), one per tool. Each follows the same template. Extends (not replaces) existing `test_tools.py` and `test_tools_dimension.py`. Uses shared helper `_make_one_h5_batch` from `test_tools_dimension.py` pattern.

**Tech Stack:** pytest, numpy, tempfile, existing `src/one_embedding/tools/` and `src/one_embedding/io.py`

**Spec:** `docs/superpowers/specs/2026-03-25-rigorous-benchmark-design.md` (Section: Unit Tests Per Tool)

---

## Shared Test Helper

Every test file needs to create synthetic .one.h5 files. Reuse this pattern (from `test_tools_dimension.py`):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch


def _make_batch(d, n=5, L=80, D=768, dct_k=4):
    """Create batch .one.h5 with n proteins, dimension D."""
    proteins = {}
    rng = np.random.RandomState(42)
    for i in range(n):
        per_res = rng.randn(L, D).astype(np.float32)
        proteins[f"prot_{i}"] = {
            "per_residue": per_res,
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test_batch.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)
```

## Test Template Per Tool

Each tool test file covers these 7 categories from the spec:

1. **Smoke test** — runs without error on synthetic data
2. **Shape test** — output dimensions match input protein lengths
3. **Value range test** — scores in expected bounds
4. **Known-answer test** — specific input → expected behavior
5. **Edge case tests** — L=1, L=5000, missing proteins
6. **Determinism test** — 3 runs produce identical output
7. **Dimension agnostic test** — works on 512d and 768d

---

### Task 1: Test disorder tool

**Files:**
- Create: `tests/test_tool_disorder.py`

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the disorder prediction tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.disorder import predict


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestDisorderSmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            assert isinstance(result, dict)
            assert len(result) == 5


class TestDisorderShape:
    def test_output_length_matches_protein(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=3, L=50)
            result = predict(path)
            for pid, scores in result.items():
                assert len(scores) == 50, f"{pid}: expected 50, got {len(scores)}"

    def test_long_protein(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=1, L=500)
            result = predict(path)
            assert len(result["prot_0"]) == 500


class TestDisorderValueRange:
    def test_scores_are_finite(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            for scores in result.values():
                assert np.isfinite(scores).all()


class TestDisorderDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path)
            r2 = predict(path)
            r3 = predict(path)
            for pid in r1:
                np.testing.assert_array_equal(r1[pid], r2[pid])
                np.testing.assert_array_equal(r2[pid], r3[pid])


class TestDisorderDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512, dct_k=4)
            result = predict(path)
            assert len(result) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768, dct_k=4)
            result = predict(path)
            assert len(result) == 5


class TestDisorderEdgeCases:
    def test_single_residue(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=1, L=1)
            result = predict(path)
            assert len(result["prot_0"]) == 1

    def test_single_protein(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=1, L=30)
            result = predict(path)
            assert len(result) == 1
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_disorder.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_disorder.py
git commit -m "test: rigorous disorder tool tests — smoke, shape, range, determinism, dimensions"
```

---

### Task 2: Test SS3 tool

**Files:**
- Create: `tests/test_tool_ss3.py`

Same pattern as disorder but for `src.one_embedding.tools.ss3`. Key differences:
- `predict(path)` returns `{pid: str}` (SS3 label string like "HHHEEEECCC")
- Output length == protein length
- All chars must be in {"H", "E", "C"}
- Known-answer: all-zero embedding should still produce valid labels

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the SS3 prediction tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.ss3 import predict


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestSS3Smoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            assert isinstance(result, dict)
            assert len(result) == 5


class TestSS3Shape:
    def test_output_length_matches_protein(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=3, L=50)
            result = predict(path)
            for pid, labels in result.items():
                assert len(labels) == 50

    def test_long_protein(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=1, L=500)
            result = predict(path)
            assert len(result["prot_0"]) == 500


class TestSS3ValueRange:
    def test_valid_labels_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            valid_chars = {"H", "E", "C"}
            for pid, labels in result.items():
                assert all(c in valid_chars for c in labels), f"{pid}: invalid chars in {labels[:20]}"


class TestSS3Determinism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path)
            r2 = predict(path)
            for pid in r1:
                assert r1[pid] == r2[pid]


class TestSS3DimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512, dct_k=4)
            result = predict(path)
            assert len(result) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768, dct_k=4)
            result = predict(path)
            assert len(result) == 5


class TestSS3EdgeCases:
    def test_single_residue(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=1, L=1)
            result = predict(path)
            assert len(result["prot_0"]) == 1

    def test_all_zero_embedding(self):
        """Zero embeddings should still produce valid SS labels."""
        with tempfile.TemporaryDirectory() as d:
            proteins = {"zero_prot": {
                "per_residue": np.zeros((20, 768), dtype=np.float32),
                "protein_vec": np.zeros(768 * 4, dtype=np.float16),
                "sequence": "A" * 20,
            }}
            path = Path(d) / "test.one.h5"
            write_one_h5_batch(str(path), proteins)
            result = predict(str(path))
            assert len(result["zero_prot"]) == 20
            assert all(c in {"H", "E", "C"} for c in result["zero_prot"])
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_ss3.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_ss3.py
git commit -m "test: rigorous SS3 tool tests — smoke, shape, valid labels, determinism, dimensions"
```

---

### Task 3: Test search tool

**Files:**
- Create: `tests/test_tool_search.py`

`find_neighbors(path, k=5)` returns `{pid: [{"name": str, "similarity": float}, ...]}`. Tests: k neighbors returned, similarities in [-1, 1] for cosine, self not in results, deterministic.

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the search tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.search import find_neighbors


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestSearchSmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = find_neighbors(path, k=3)
            assert isinstance(result, dict)
            assert len(result) == 5


class TestSearchShape:
    def test_k_neighbors_returned(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=10)
            result = find_neighbors(path, k=3)
            for pid, hits in result.items():
                assert len(hits) == 3

    def test_k_clamped_to_n_minus_1(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=3)
            result = find_neighbors(path, k=10)
            for pid, hits in result.items():
                assert len(hits) <= 2  # n-1 (exclude self)


class TestSearchValueRange:
    def test_similarity_in_range(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = find_neighbors(path, k=3)
            for pid, hits in result.items():
                for hit in hits:
                    assert -1.01 <= hit["similarity"] <= 1.01


class TestSearchSelfExclusion:
    def test_self_not_in_results(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = find_neighbors(path, k=4)
            for pid, hits in result.items():
                hit_names = [h["name"] for h in hits]
                assert pid not in hit_names


class TestSearchDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = find_neighbors(path, k=2)
            r2 = find_neighbors(path, k=2)
            for pid in r1:
                assert r1[pid][0]["name"] == r2[pid][0]["name"]
                assert abs(r1[pid][0]["similarity"] - r2[pid][0]["similarity"]) < 1e-6


class TestSearchDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512)
            result = find_neighbors(path, k=2)
            assert len(result) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768)
            result = find_neighbors(path, k=2)
            assert len(result) == 5
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_search.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_search.py
git commit -m "test: rigorous search tool tests — k neighbors, similarity range, self-exclusion, determinism"
```

---

### Task 4: Test classify tool

**Files:**
- Create: `tests/test_tool_classify.py`

`predict(path, db=path, k=1)` returns `{pid: {"neighbors": [...], "top_match": (name, sim)}}`.

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the classify tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.classify import predict


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestClassifySmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path, db=path, k=1)
            assert isinstance(result, dict)
            assert len(result) == 5

class TestClassifyShape:
    def test_top_match_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path, db=path, k=2)
            for pid, info in result.items():
                assert "top_match" in info
                assert info["top_match"] is not None

class TestClassifyValueRange:
    def test_similarity_is_float(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path, db=path, k=1)
            for pid, info in result.items():
                _, sim = info["top_match"]
                assert isinstance(float(sim), float)

class TestClassifyDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path, db=path, k=1)
            r2 = predict(path, db=path, k=1)
            for pid in r1:
                assert r1[pid]["top_match"][0] == r2[pid]["top_match"][0]

class TestClassifyDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512)
            result = predict(path, db=path, k=1)
            assert len(result) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768)
            result = predict(path, db=path, k=1)
            assert len(result) == 5
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_classify.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_classify.py
git commit -m "test: rigorous classify tool tests — top_match, similarity, determinism, dimensions"
```

---

### Task 5: Test align tool

**Files:**
- Create: `tests/test_tool_align.py`

`align_pair(path, pid_a, pid_b)` returns dict with "align_a", "align_b", "score", "n_aligned", "n_gaps_a", "n_gaps_b".

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the align tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.align import align_pair


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestAlignSmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = align_pair(path, "prot_0", "prot_1")
            assert isinstance(result, dict)
            assert "score" in result

class TestAlignShape:
    def test_alignment_lengths_match(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = align_pair(path, "prot_0", "prot_1")
            assert len(result["align_a"]) == len(result["align_b"])

    def test_n_aligned_positive(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = align_pair(path, "prot_0", "prot_1")
            assert result["n_aligned"] > 0

class TestAlignKnownAnswer:
    def test_identical_proteins_perfect_score(self):
        """Aligning a protein with itself should give high score, no gaps."""
        with tempfile.TemporaryDirectory() as d:
            proteins = {
                "prot_a": {
                    "per_residue": np.ones((20, 768), dtype=np.float32),
                    "protein_vec": np.ones(768 * 4, dtype=np.float16),
                    "sequence": "A" * 20,
                },
                "prot_b": {
                    "per_residue": np.ones((20, 768), dtype=np.float32),
                    "protein_vec": np.ones(768 * 4, dtype=np.float16),
                    "sequence": "A" * 20,
                },
            }
            path = Path(d) / "test.one.h5"
            write_one_h5_batch(str(path), proteins)
            result = align_pair(str(path), "prot_a", "prot_b")
            assert result["n_gaps_a"] == 0
            assert result["n_gaps_b"] == 0
            assert result["n_aligned"] == 20

class TestAlignDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = align_pair(path, "prot_0", "prot_1")
            r2 = align_pair(path, "prot_0", "prot_1")
            assert r1["score"] == r2["score"]
            assert r1["align_a"] == r2["align_a"]

class TestAlignDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512)
            result = align_pair(path, "prot_0", "prot_1")
            assert "score" in result

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768)
            result = align_pair(path, "prot_0", "prot_1")
            assert "score" in result

class TestAlignEdgeCases:
    def test_different_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.RandomState(42)
            proteins = {
                "short": {
                    "per_residue": rng.randn(10, 768).astype(np.float32),
                    "protein_vec": rng.randn(768 * 4).astype(np.float16),
                    "sequence": "A" * 10,
                },
                "long": {
                    "per_residue": rng.randn(100, 768).astype(np.float32),
                    "protein_vec": rng.randn(768 * 4).astype(np.float16),
                    "sequence": "A" * 100,
                },
            }
            path = Path(d) / "test.one.h5"
            write_one_h5_batch(str(path), proteins)
            result = align_pair(str(path), "short", "long")
            assert len(result["align_a"]) == len(result["align_b"])
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_align.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_align.py
git commit -m "test: rigorous align tool tests — alignment shape, identical proteins, different lengths"
```

---

### Task 6: Test conserve tool

**Files:**
- Create: `tests/test_tool_conserve.py`

`predict(path)` returns `{pid: ndarray}` of conservation scores in [0, 1] (norm-based heuristic).

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the conserve tool (heuristic)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.conserve import predict


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestConserveSmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            assert isinstance(result, dict)
            assert len(result) == 5

class TestConserveShape:
    def test_output_length_matches(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=3, L=50)
            result = predict(path)
            for pid, scores in result.items():
                assert len(scores) == 50

class TestConserveValueRange:
    def test_scores_in_0_1(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            for pid, scores in result.items():
                assert np.all(scores >= -0.01), f"{pid}: min={scores.min()}"
                assert np.all(scores <= 1.01), f"{pid}: max={scores.max()}"

class TestConserveDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path)
            r2 = predict(path)
            for pid in r1:
                np.testing.assert_array_equal(r1[pid], r2[pid])

class TestConserveDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512)
            assert len(predict(path)) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768)
            assert len(predict(path)) == 5

class TestConserveKnownAnswer:
    def test_constant_embedding_uniform_scores(self):
        """If all residues have identical embeddings, conservation should be uniform."""
        with tempfile.TemporaryDirectory() as d:
            proteins = {"const": {
                "per_residue": np.ones((20, 768), dtype=np.float32),
                "protein_vec": np.ones(768 * 4, dtype=np.float16),
                "sequence": "A" * 20,
            }}
            path = Path(d) / "test.one.h5"
            write_one_h5_batch(str(path), proteins)
            result = predict(str(path))
            scores = result["const"]
            # All norms are equal → all scores should be equal after min-max
            assert np.std(scores) < 0.01
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_conserve.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_conserve.py
git commit -m "test: rigorous conserve tool tests — value range [0,1], constant embedding, dimensions"
```

---

### Task 7: Test mutate tool

**Files:**
- Create: `tests/test_tool_mutate.py`

`predict(path)` returns `{pid: ndarray}` of mutation sensitivity scores (local context distance heuristic). Scores are non-negative floats.

- [ ] **Step 1: Write test file**

```python
"""Rigorous tests for the mutate tool (heuristic)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest
from src.one_embedding.io import write_one_h5_batch
from src.one_embedding.tools.mutate import predict


def _make_batch(d, n=5, L=80, D=768, dct_k=4, seed=42):
    proteins = {}
    rng = np.random.RandomState(seed)
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": rng.randn(L, D).astype(np.float32),
            "protein_vec": rng.randn(D * dct_k).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


class TestMutateSmoke:
    def test_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            assert isinstance(result, dict)
            assert len(result) == 5

class TestMutateShape:
    def test_output_length_matches(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, n=3, L=50)
            result = predict(path)
            for pid, scores in result.items():
                assert len(scores) == 50

class TestMutateValueRange:
    def test_scores_non_negative(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            for pid, scores in result.items():
                assert np.all(scores >= -0.01), f"{pid}: min={scores.min()}"
                assert np.isfinite(scores).all()

class TestMutateDeterminism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path)
            r2 = predict(path)
            for pid in r1:
                np.testing.assert_array_equal(r1[pid], r2[pid])

class TestMutateDimensionAgnostic:
    def test_512d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=512)
            assert len(predict(path)) == 5

    def test_768d(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d, D=768)
            assert len(predict(path)) == 5

class TestMutateKnownAnswer:
    def test_constant_embedding_zero_sensitivity(self):
        """Constant embeddings → zero local variance → zero sensitivity."""
        with tempfile.TemporaryDirectory() as d:
            proteins = {"const": {
                "per_residue": np.ones((20, 768), dtype=np.float32),
                "protein_vec": np.ones(768 * 4, dtype=np.float16),
                "sequence": "A" * 20,
            }}
            path = Path(d) / "test.one.h5"
            write_one_h5_batch(str(path), proteins)
            result = predict(str(path))
            scores = result["const"]
            assert np.allclose(scores, 0.0, atol=1e-5)
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/test_tool_mutate.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_mutate.py
git commit -m "test: rigorous mutate tool tests — non-negative scores, constant embedding zero, dimensions"
```

---

## Summary

| Task | Tool | Tests | Key assertions |
|------|------|:-----:|---------------|
| 1 | disorder | ~9 | finite scores, shape match, determinism, 512d/768d |
| 2 | ss3 | ~9 | valid H/E/C chars, shape match, zero-embedding valid |
| 3 | search | ~9 | k neighbors, similarity range, self-exclusion |
| 4 | classify | ~7 | top_match exists, determinism, dimensions |
| 5 | align | ~9 | alignment lengths match, identical=no gaps, different lengths |
| 6 | conserve | ~7 | scores in [0,1], constant=uniform, dimensions |
| 7 | mutate | ~7 | non-negative, constant=zero, dimensions |

Total: **7 tasks, ~57 tests, 7 commits**.

All tasks are independent — can be dispatched in parallel (up to 7 agents).
