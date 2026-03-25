"""Tests for bootstrap CI and multi-seed statistics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult
from metrics.statistics import (
    bootstrap_ci, multi_seed_summary,
    paired_bootstrap_retention, paired_bootstrap_metric,
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
