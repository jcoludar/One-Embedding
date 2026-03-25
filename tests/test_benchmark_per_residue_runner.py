"""Tests for the per-residue benchmark runners (SS3, SS8, disorder).

Uses synthetic data (random embeddings, d=32, 20 proteins) to verify that
all runners return the correct types, shapes, and keys without requiring
real PLM embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult


class TestStackResidues:
    def test_basic_stacking(self):
        from runners.per_residue import _stack_residues, SS3_MAP

        embeddings = {"p1": np.ones((10, 4)), "p2": np.ones((5, 4)) * 2}
        labels = {"p1": "HHHHHEEEECC"[:10], "p2": "HHEEC"}
        X, y = _stack_residues(embeddings, labels, ["p1", "p2"], max_len=512, label_map=SS3_MAP)
        assert X.shape == (15, 4)
        assert len(y) == 15
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_truncation(self):
        from runners.per_residue import _stack_residues

        embeddings = {"p1": np.ones((100, 4))}
        labels = {"p1": "H" * 100}
        X, y = _stack_residues(embeddings, labels, ["p1"], max_len=50, label_map={"H": 0})
        assert X.shape == (50, 4)

    def test_nan_filtering_for_regression(self):
        from runners.per_residue import _stack_residues

        embeddings = {"p1": np.ones((10, 4))}
        scores = {"p1": np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])}
        X, y = _stack_residues(embeddings, scores, ["p1"], max_len=512)
        assert X.shape == (8, 4)  # 2 NaN removed
        assert len(y) == 8


class TestSS3Benchmark:
    def test_returns_metric_result(self):
        from runners.per_residue import run_ss3_benchmark

        rng = np.random.RandomState(42)
        n_proteins = 20
        embeddings = {f"p{i}": rng.randn(50, 32).astype(np.float32) for i in range(n_proteins)}
        labels = {
            f"p{i}": "".join(rng.choice(["H", "E", "C"], size=50))
            for i in range(n_proteins)
        }
        result = run_ss3_benchmark(
            embeddings=embeddings,
            labels=labels,
            train_ids=[f"p{i}" for i in range(16)],
            test_ids=[f"p{i}" for i in range(16, 20)],
            C_grid=[1.0],
            seeds=[42],
            n_bootstrap=100,
        )
        assert isinstance(result["q3"], MetricResult)
        assert "per_class_acc" in result
        assert "class_balance" in result


class TestDisorderBenchmark:
    def test_returns_metric_result(self):
        from runners.per_residue import run_disorder_benchmark

        rng = np.random.RandomState(42)
        n_proteins = 20
        embeddings = {f"p{i}": rng.randn(50, 32).astype(np.float32) for i in range(n_proteins)}
        # Make scores somewhat correlated with embeddings for testability
        scores = {f"p{i}": rng.randn(50) for i in range(n_proteins)}
        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=[f"p{i}" for i in range(16)],
            test_ids=[f"p{i}" for i in range(16, 20)],
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=100,
        )
        assert isinstance(result["spearman_rho"], MetricResult)
        assert "best_alpha" in result
