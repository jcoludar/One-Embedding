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
