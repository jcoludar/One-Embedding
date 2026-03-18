"""Tests for embedding-based family classifier."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.classifier import EmbeddingClassifier


class TestEmbeddingClassifier:
    def _make_clustered_data(self, rng, n_families=3, n_per_family=10, d=512):
        """Create well-separated clusters for testing."""
        vectors = {}
        labels = {}
        for fam_idx in range(n_families):
            center = rng.randn(d) * 5
            for i in range(n_per_family):
                name = f"fam{fam_idx}_prot{i}"
                vectors[name] = (center + rng.randn(d) * 0.5).astype(np.float32)
                labels[name] = f"family_{fam_idx}"
        return vectors, labels

    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng)
        clf = EmbeddingClassifier(metric="cosine", k=1)
        clf.fit(vectors, labels)
        # Query with a vector near family_0's cluster
        query = vectors["fam0_prot0"] + rng.randn(512).astype(np.float32) * 0.1
        result = clf.predict(query)
        assert result["label"] == "family_0"
        assert result["confidence"] == 1.0

    def test_k3_majority_vote(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng, n_families=3, n_per_family=20)
        clf = EmbeddingClassifier(metric="cosine", k=3)
        clf.fit(vectors, labels)
        query = vectors["fam1_prot0"] + rng.randn(512).astype(np.float32) * 0.1
        result = clf.predict(query)
        assert result["label"] == "family_1"

    def test_euclidean_metric(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng)
        clf = EmbeddingClassifier(metric="euclidean", k=1)
        clf.fit(vectors, labels)
        query = vectors["fam2_prot0"]
        result = clf.predict(query)
        assert result["label"] == "family_2"

    def test_predict_batch(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng)
        clf = EmbeddingClassifier(k=1)
        clf.fit(vectors, labels)
        queries = {f"q{i}": vectors[f"fam{i}_prot0"] for i in range(3)}
        results = clf.predict_batch(queries)
        assert len(results) == 3
        assert results["q0"]["label"] == "family_0"

    def test_evaluate_perfect_clusters(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng, n_families=3, n_per_family=10)
        # Use half for reference, half for test
        ref_vecs = {k: v for k, v in vectors.items() if "prot0" in k or "prot1" in k or "prot2" in k or "prot3" in k or "prot4" in k}
        ref_labels = {k: v for k, v in labels.items() if k in ref_vecs}
        test_vecs = {k: v for k, v in vectors.items() if k not in ref_vecs}
        test_labels = {k: v for k, v in labels.items() if k in test_vecs}

        clf = EmbeddingClassifier(k=1)
        clf.fit(ref_vecs, ref_labels)
        result = clf.evaluate(test_vecs, test_labels)
        assert result["accuracy"] > 0.9

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng, n_families=2, n_per_family=5)
        clf = EmbeddingClassifier(metric="cosine", k=3)
        clf.fit(vectors, labels)

        path = str(tmp_path / "clf.npz")
        clf.save(path)
        loaded = EmbeddingClassifier.load(path)

        query = vectors["fam0_prot0"]
        assert clf.predict(query)["label"] == loaded.predict(query)["label"]

    def test_predict_without_fit_raises(self):
        clf = EmbeddingClassifier()
        with pytest.raises(ValueError):
            clf.predict(np.random.randn(512))

    def test_neighbors_returned(self):
        rng = np.random.RandomState(42)
        vectors, labels = self._make_clustered_data(rng)
        clf = EmbeddingClassifier(k=3)
        clf.fit(vectors, labels)
        result = clf.predict(vectors["fam0_prot0"])
        assert len(result["neighbors"]) == 3
        assert all(len(n) == 3 for n in result["neighbors"])  # (name, label, dist)
