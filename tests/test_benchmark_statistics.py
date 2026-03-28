"""Tests for bootstrap CI and multi-seed statistics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult
from metrics.statistics import (
    bootstrap_ci, multi_seed_summary, averaged_multi_seed,
    paired_bootstrap_retention, paired_bootstrap_metric,
    paired_cluster_bootstrap_retention,
)


class TestBootstrapCI:

    def test_returns_metric_result(self):
        scores = {f"q{i}": float(i % 2) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=1000, seed=42)
        assert isinstance(result, MetricResult)

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(500)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=42)
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_narrow_ci_for_low_variance(self):
        scores = {f"q{i}": 0.95 for i in range(1000)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=1000, seed=42)
        assert result.ci_upper - result.ci_lower < 0.01

    def test_wide_ci_for_high_variance(self):
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.choice([0.0, 1.0])) for i in range(50)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=42)
        assert result.ci_upper - result.ci_lower > 0.05


class TestBCaBootstrap:
    """BCa (bias-corrected and accelerated) bootstrap CI tests."""

    def test_bca_returns_metric_result(self):
        """n=200 uniform data, verify ci_method=='bca'."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=1000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.ci_method == "bca"

    def test_bca_ci_contains_observed(self):
        """n=500, verify ci_lower <= value <= ci_upper."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(500)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=42)
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_bca_asymmetric_for_skewed_data(self):
        """n=200 exponential data, BCa should produce asymmetric CI for right-skewed data."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.exponential(scale=1.0)) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=10000, seed=42)
        upper_dist = result.ci_upper - result.value
        lower_dist = result.value - result.ci_lower
        # BCa should produce asymmetric CIs for right-skewed data
        assert upper_dist > lower_dist

    def test_fallback_to_percentile_for_small_n(self):
        """n=10, verify ci_method=='percentile' (fallback)."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(10)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=1000, seed=42)
        assert result.ci_method == "percentile"

    def test_bca_deterministic_with_seed(self):
        """Same scores + seed should produce identical CIs."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(100)}
        r1 = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=99)
        r2 = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=99)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.value == r2.value


class TestMultiSeedSummary:

    def test_returns_metric_result_with_seeds(self):
        seed_results = [
            MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=100),
            MetricResult(value=0.94, ci_lower=0.92, ci_upper=0.96, n=100),
            MetricResult(value=0.96, ci_lower=0.94, ci_upper=0.98, n=100),
        ]
        result = multi_seed_summary(seed_results)
        assert isinstance(result, MetricResult)
        assert result.seeds_mean is not None
        assert result.seeds_std is not None

    def test_median_seed_selected(self):
        seed_results = [
            MetricResult(value=0.90, ci_lower=0.88, ci_upper=0.92, n=100),
            MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=100),
            MetricResult(value=1.00, ci_lower=0.98, ci_upper=1.02, n=100),
        ]
        result = multi_seed_summary(seed_results)
        assert result.value == 0.95
        assert result.ci_lower == 0.93
        assert result.ci_upper == 0.97


class TestPairedBootstrapRetention:
    """Retention = compressed/raw as a percentage, with paired bootstrap CI."""

    def test_perfect_retention(self):
        """Identical scores → 100% retention with tight CI."""
        scores = {f"p{i}": 0.8 for i in range(200)}
        result = paired_bootstrap_retention(scores, scores, n_bootstrap=1000, seed=42)
        assert abs(result.value - 100.0) < 0.01
        assert result.ci_upper - result.ci_lower < 1.0

    def test_retention_ci_contains_point(self):
        rng = np.random.RandomState(42)
        raw = {f"p{i}": float(rng.rand() * 0.5 + 0.5) for i in range(200)}
        comp = {k: v * 0.95 for k, v in raw.items()}
        result = paired_bootstrap_retention(raw, comp, n_bootstrap=5000, seed=42)
        assert result.ci_lower <= result.value <= result.ci_upper
        assert 90 < result.value < 100  # ~95% retention

    def test_paired_resampling_gives_tighter_ci(self):
        """Paired bootstrap should give narrower CI than unpaired for correlated data."""
        rng = np.random.RandomState(42)
        raw = {f"p{i}": float(rng.rand() * 0.3 + 0.7) for i in range(100)}
        # Compressed = raw * 0.98 + small noise (highly correlated)
        comp = {k: v * 0.98 + rng.randn() * 0.001 for k, v in raw.items()}
        result = paired_bootstrap_retention(raw, comp, n_bootstrap=5000, seed=42)
        ci_width = result.ci_upper - result.ci_lower
        # Paired CI on correlated data should be quite narrow
        assert ci_width < 5.0  # Less than 5pp wide

    def test_handles_no_common_ids(self):
        raw = {"a": 1.0, "b": 2.0}
        comp = {"c": 1.0, "d": 2.0}
        result = paired_bootstrap_retention(raw, comp)
        assert result.n == 0


class TestPairedBootstrapMetric:
    """Paired CIs on two systems simultaneously."""

    def test_returns_two_metric_results(self):
        rng = np.random.RandomState(42)
        raw = {f"p{i}": float(rng.rand()) for i in range(100)}
        comp = {f"p{i}": float(rng.rand()) for i in range(100)}
        r_raw, r_comp = paired_bootstrap_metric(raw, comp, n_bootstrap=1000, seed=42)
        assert isinstance(r_raw, MetricResult)
        assert isinstance(r_comp, MetricResult)

    def test_same_n(self):
        raw = {f"p{i}": float(i) for i in range(50)}
        comp = {f"p{i}": float(i * 2) for i in range(50)}
        r_raw, r_comp = paired_bootstrap_metric(raw, comp, n_bootstrap=1000)
        assert r_raw.n == r_comp.n == 50


class TestAveragedMultiSeed:
    """Tests for averaged_multi_seed (Bouthillier et al. 2021)."""

    def test_averaged_seed_returns_metric_result(self):
        """3 seed dicts of 100 items, verify MetricResult with seeds_mean/seeds_std."""
        rng = np.random.RandomState(42)
        seed_scores = [
            {f"p{i}": float(rng.rand()) for i in range(100)}
            for _ in range(3)
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=1000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.seeds_mean is not None
        assert result.seeds_std is not None
        assert result.n == 100
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_averaged_reduces_variance(self):
        """3 noisy copies of base scores, averaged result close to base mean."""
        rng = np.random.RandomState(42)
        base = {f"p{i}": float(rng.rand()) for i in range(100)}
        base_mean = np.mean(list(base.values()))
        # Add noise to each seed
        seed_scores = []
        for s in range(3):
            rng_s = np.random.RandomState(100 + s)
            noisy = {k: v + rng_s.randn() * 0.05 for k, v in base.items()}
            seed_scores.append(noisy)
        result = averaged_multi_seed(seed_scores, n_bootstrap=1000, seed=42)
        # Averaged across 3 noisy seeds should be close to base mean
        assert abs(result.value - base_mean) < 0.05

    def test_averaged_ci_is_bca(self):
        """Verify ci_method == 'bca' for n=100."""
        rng = np.random.RandomState(42)
        seed_scores = [
            {f"p{i}": float(rng.rand()) for i in range(100)}
            for _ in range(3)
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=1000, seed=42)
        assert result.ci_method == "bca"

    def test_averaged_reports_per_seed_values(self):
        """3 dicts with known values (0.9, 0.95, 1.0), verify seeds_mean ~= 0.95."""
        seed_scores = [
            {f"p{i}": 0.9 for i in range(100)},
            {f"p{i}": 0.95 for i in range(100)},
            {f"p{i}": 1.0 for i in range(100)},
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=1000, seed=42)
        assert abs(result.seeds_mean - 0.95) < 1e-10
        # seeds_std should be std of [0.9, 0.95, 1.0]
        expected_std = float(np.std([0.9, 0.95, 1.0], ddof=1))
        assert abs(result.seeds_std - expected_std) < 1e-10
        # The averaged per-item value should be 0.95 (each item averages to 0.95)
        assert abs(result.value - 0.95) < 1e-10


class TestPairedClusterBootstrapRetention:
    """Retention CI for pooled metrics (e.g., pooled Spearman rho)."""

    def test_perfect_retention(self):
        """Identical clusters → 100% retention."""
        clusters = {
            f"p{i}": {"y_true": np.array([1.0, 2.0, 3.0]), "y_pred": np.array([1.1, 2.1, 3.1])}
            for i in range(50)
        }
        def stat_fn(data):
            return float(np.mean([np.mean(d["y_pred"]) for d in data]))
        result = paired_cluster_bootstrap_retention(clusters, clusters, stat_fn, n_bootstrap=500, seed=42)
        assert abs(result.value - 100.0) < 0.01
        assert result.ci_upper - result.ci_lower < 1.0

    def test_partial_retention(self):
        """Compressed = 90% of raw → ~90% retention."""
        rng = np.random.RandomState(42)
        raw = {}
        comp = {}
        for i in range(50):
            y_true = rng.rand(20)
            y_pred_raw = y_true + rng.randn(20) * 0.1
            y_pred_comp = y_true * 0.9 + rng.randn(20) * 0.1
            raw[f"p{i}"] = {"y_true": y_true, "y_pred": y_pred_raw}
            comp[f"p{i}"] = {"y_true": y_true, "y_pred": y_pred_comp}

        def stat_fn(data):
            return float(np.mean(np.concatenate([d["y_pred"] for d in data])))

        result = paired_cluster_bootstrap_retention(raw, comp, stat_fn, n_bootstrap=1000, seed=42)
        assert result.ci_lower <= result.value <= result.ci_upper
        assert result.n == 50

    def test_no_common_ids(self):
        raw = {"a": {"y_true": np.array([1.0]), "y_pred": np.array([1.0])}}
        comp = {"b": {"y_true": np.array([1.0]), "y_pred": np.array([1.0])}}
        def stat_fn(data):
            return 1.0
        result = paired_cluster_bootstrap_retention(raw, comp, stat_fn)
        assert result.n == 0


class TestRetrievalBCa:
    """Verify retrieval Ret@1 scores (binary 0/1) get BCa CI automatically."""

    def test_retrieval_scores_get_bca(self):
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.choice([0.0, 1.0])) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=2000, seed=42)
        assert result.ci_method == "bca"
        assert result.ci_lower <= result.value <= result.ci_upper
