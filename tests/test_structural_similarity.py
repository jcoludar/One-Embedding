"""Tests for structural similarity search."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.structural_similarity import (
    cosine_similarity_matrix,
    cosine_to_tm_score,
    StructuralSearchIndex,
)


class TestCosineSimilarityMatrix:
    def test_shape(self):
        vecs = np.random.randn(10, 512).astype(np.float32)
        mat = cosine_similarity_matrix(vecs)
        assert mat.shape == (10, 10)

    def test_diagonal_ones(self):
        vecs = np.random.randn(5, 64).astype(np.float32)
        mat = cosine_similarity_matrix(vecs)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_symmetric(self):
        vecs = np.random.randn(8, 64).astype(np.float32)
        mat = cosine_similarity_matrix(vecs)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)


class TestCosineToTmScore:
    def test_identity_mapping(self):
        sims = np.array([0.0, 0.5, 1.0])
        tm = cosine_to_tm_score(sims)
        np.testing.assert_allclose(tm, [0.0, 0.5, 1.0])

    def test_clipping(self):
        sims = np.array([-0.5, 1.5])
        tm = cosine_to_tm_score(sims)
        assert tm[0] == 0.0
        assert tm[1] == 1.0

    def test_calibrated(self):
        sims = np.array([0.5])
        tm = cosine_to_tm_score(sims, a=2.0, b=-0.5)
        assert abs(tm[0] - 0.5) < 1e-6


class TestStructuralSearchIndex:
    def _make_index(self):
        rng = np.random.RandomState(42)
        vectors = {f"prot_{i}": rng.randn(512).astype(np.float32) for i in range(20)}
        idx = StructuralSearchIndex()
        idx.build(vectors)
        return idx, vectors

    def test_search_returns_k(self):
        idx, vectors = self._make_index()
        results = idx.search(vectors["prot_0"], k=5)
        assert len(results) == 5

    def test_self_is_most_similar(self):
        idx, vectors = self._make_index()
        results = idx.search(vectors["prot_0"], k=1)
        assert results[0]["name"] == "prot_0"
        assert results[0]["similarity"] > 0.99

    def test_search_batch(self):
        idx, vectors = self._make_index()
        queries = {"q0": vectors["prot_0"], "q1": vectors["prot_1"]}
        results = idx.search_batch(queries, k=3)
        assert len(results) == 2
        assert len(results["q0"]) == 3

    def test_calibrate_tm_score(self):
        idx, vectors = self._make_index()
        pairs = [("prot_0", "prot_1"), ("prot_2", "prot_3")]
        true_tm = [0.8, 0.3]
        idx.calibrate_tm_score(pairs, true_tm)
        assert idx.tm_a != 1.0 or idx.tm_b != 0.0

    def test_all_vs_all(self):
        idx, _ = self._make_index()
        mat = idx.all_vs_all()
        assert mat.shape == (20, 20)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_save_load_roundtrip(self, tmp_path):
        idx, vectors = self._make_index()
        path = str(tmp_path / "index.npz")
        idx.save(path)
        loaded = StructuralSearchIndex.load(path)
        r1 = idx.search(vectors["prot_5"], k=3)
        r2 = loaded.search(vectors["prot_5"], k=3)
        assert r1[0]["name"] == r2[0]["name"]

    def test_min_score_filter(self):
        idx, vectors = self._make_index()
        results = idx.search(vectors["prot_0"], k=20, min_score=0.99)
        # Only self should pass such a high threshold
        assert len(results) <= 2

    def test_build_not_called_raises(self):
        idx = StructuralSearchIndex()
        with pytest.raises(ValueError):
            idx.search(np.random.randn(512))
