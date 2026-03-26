"""Tests for disorder evaluation methodology -- Nature-level rigor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from scipy.stats import spearmanr
from rules import MetricResult
from metrics.statistics import cluster_bootstrap_ci


class TestClusterBootstrapCI:

    def test_returns_metric_result(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(50):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=2000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_ci_width_reasonable(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(100):
            n = rng.randint(50, 200)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.5
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho)

        result = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=5000, seed=42)
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width < 0.10

    def test_uses_bca_for_large_n(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(50):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=2000, seed=42)
        assert result.ci_method == "bca"
        assert result.value > 0.5

    def test_uses_percentile_for_small_n(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(10):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=2000, seed=42)
        assert result.ci_method == "percentile"

    def test_deterministic_with_seed(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(50):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        r1 = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=2000, seed=99)
        r2 = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=2000, seed=99)
        assert r1.value == r2.value
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_n_equals_cluster_count(self):
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(30):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman, n_bootstrap=500, seed=42)
        assert result.n == 30


class TestDisorderBenchmarkRewrite:
    """Verify the rewritten run_disorder_benchmark returns correct keys and types."""

    @pytest.fixture
    def synthetic_disorder_data(self):
        """Create synthetic disorder data: 30 proteins, d=32."""
        rng = np.random.RandomState(42)
        n_proteins = 30
        embeddings = {}
        scores = {}
        for i in range(n_proteins):
            L = rng.randint(20, 80)
            embeddings[f"p{i}"] = rng.randn(L, 32).astype(np.float32)
            # Z-scores: correlated with first embedding dimension for some signal
            scores[f"p{i}"] = embeddings[f"p{i}"][:, 0] * 3.0 + rng.randn(L) * 2.0
        train_ids = [f"p{i}" for i in range(24)]
        test_ids = [f"p{i}" for i in range(24, 30)]
        return embeddings, scores, train_ids, test_ids

    def test_returns_all_required_keys(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=200,
        )
        required_keys = [
            "pooled_spearman_rho",
            "auc_roc",
            "per_protein_spearman_rho",
            "spearman_rho",
            "best_alpha",
            "n_train_residues",
            "n_test_residues",
            "n_test_proteins",
            "disorder_threshold",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_pooled_spearman_rho_is_metric_result(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=200,
        )
        assert isinstance(result["pooled_spearman_rho"], MetricResult)
        assert result["pooled_spearman_rho"].ci_lower <= result["pooled_spearman_rho"].value
        assert result["pooled_spearman_rho"].value <= result["pooled_spearman_rho"].ci_upper

    def test_auc_roc_is_metric_result(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=200,
        )
        assert isinstance(result["auc_roc"], MetricResult)
        assert 0.0 <= result["auc_roc"].value <= 1.0

    def test_backward_compat_spearman_rho_alias(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=200,
        )
        # spearman_rho should be the same object as per_protein_spearman_rho
        assert result["spearman_rho"] is result["per_protein_spearman_rho"]

    def test_disorder_threshold_default(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42],
            n_bootstrap=200,
        )
        assert result["disorder_threshold"] == 8.0

    def test_multi_seed_averages_predictions(self, synthetic_disorder_data):
        from runners.per_residue import run_disorder_benchmark
        embeddings, scores, train_ids, test_ids = synthetic_disorder_data

        result = run_disorder_benchmark(
            embeddings=embeddings,
            scores=scores,
            train_ids=train_ids,
            test_ids=test_ids,
            alpha_grid=[1.0],
            seeds=[42, 43, 44],
            n_bootstrap=200,
        )
        # With 3 seeds, should still return valid MetricResults
        assert isinstance(result["pooled_spearman_rho"], MetricResult)
        assert isinstance(result["auc_roc"], MetricResult)
        assert isinstance(result["per_protein_spearman_rho"], MetricResult)
