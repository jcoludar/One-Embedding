"""Tests for protein-level benchmark runner with fair baselines."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult
from runners.protein_level import compute_protein_vectors, run_retrieval_benchmark


class TestComputeProteinVectors:

    def test_dct_k4_pooling(self):
        embeddings = {"p1": np.random.randn(50, 64).astype(np.float32)}
        vecs = compute_protein_vectors(embeddings, method="dct_k4", dct_k=4)
        assert "p1" in vecs
        assert vecs["p1"].shape == (64 * 4,)

    def test_mean_pooling(self):
        embeddings = {"p1": np.random.randn(50, 64).astype(np.float32)}
        vecs = compute_protein_vectors(embeddings, method="mean")
        assert vecs["p1"].shape == (64,)

    def test_unknown_method_raises(self):
        embeddings = {"p1": np.random.randn(50, 64).astype(np.float32)}
        with pytest.raises(ValueError, match="Unknown"):
            compute_protein_vectors(embeddings, method="invalid")


class TestRetrievalBenchmark:

    def test_returns_cosine_and_euclidean(self):
        rng = np.random.RandomState(42)
        vectors = {}
        metadata = []
        for fam in range(5):
            center = rng.randn(32).astype(np.float32)
            for j in range(10):
                pid = f"fam{fam}_p{j}"
                vectors[pid] = center + rng.randn(32).astype(np.float32) * 0.1
                metadata.append({"id": pid, "family": f"fam{fam}"})

        result = run_retrieval_benchmark(vectors=vectors, metadata=metadata, label_key="family", n_bootstrap=100)
        assert "ret1_cosine" in result
        assert "ret1_euclidean" in result
        assert isinstance(result["ret1_cosine"], MetricResult)
        assert isinstance(result["ret1_euclidean"], MetricResult)

    def test_well_separated_clusters_high_ret1(self):
        rng = np.random.RandomState(42)
        vectors = {}
        metadata = []
        for fam in range(3):
            center = np.zeros(32, dtype=np.float32)
            center[fam * 10:(fam + 1) * 10] = 10.0  # Very separated
            for j in range(10):
                pid = f"fam{fam}_p{j}"
                vectors[pid] = center + rng.randn(32).astype(np.float32) * 0.01
                metadata.append({"id": pid, "family": f"fam{fam}"})

        result = run_retrieval_benchmark(vectors=vectors, metadata=metadata, n_bootstrap=100)
        assert result["ret1_cosine"].value > 0.9
        assert result["ret1_euclidean"].value > 0.9
