"""Tests for bootstrap CI and multi-seed statistics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult
from metrics.statistics import bootstrap_ci, multi_seed_summary


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
