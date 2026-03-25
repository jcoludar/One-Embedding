"""Rigorous tests for the SS3 prediction tool.

Note: ss3.predict() returns {pid: np.ndarray of int} where values are
0=Helix, 1=Strand, 2=Coil — NOT strings. Tests are written accordingly.
"""
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
        """All predicted labels must be in {0, 1, 2} (Helix, Strand, Coil)."""
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            result = predict(path)
            valid_values = {0, 1, 2}
            for pid, labels in result.items():
                unique = set(np.unique(labels).tolist())
                assert unique.issubset(valid_values), f"{pid}: invalid values {unique}"


class TestSS3Determinism:
    def test_same_input_same_output(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_batch(d)
            r1 = predict(path)
            r2 = predict(path)
            for pid in r1:
                np.testing.assert_array_equal(r1[pid], r2[pid])


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
            valid_values = {0, 1, 2}
            unique = set(np.unique(result["zero_prot"]).tolist())
            assert unique.issubset(valid_values)
