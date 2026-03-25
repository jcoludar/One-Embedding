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
