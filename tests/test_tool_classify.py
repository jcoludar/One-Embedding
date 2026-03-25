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
