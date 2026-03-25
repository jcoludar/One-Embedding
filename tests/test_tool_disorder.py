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
                # CNN float32 may have tiny rounding differences across calls; use allclose
                np.testing.assert_allclose(r1[pid], r2[pid], atol=1e-5)
                np.testing.assert_allclose(r2[pid], r3[pid], atol=1e-5)


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
