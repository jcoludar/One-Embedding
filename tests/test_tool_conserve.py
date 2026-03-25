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
