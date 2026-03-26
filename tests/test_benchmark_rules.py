"""Tests for golden rule assertions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import pytest
import numpy as np
from rules import (
    MetricResult, check_protein_level_comparison, check_per_residue_comparison,
    check_no_leakage, check_class_balance, check_cross_dataset_consistency,
)


class TestRule1FairComparison:
    """Rule 1: Raw and compressed must use matching config."""

    def test_matching_config_passes(self):
        check_protein_level_comparison(
            pooling_method="dct_k4",
            raw_pooling="dct_k4",
        )

    def test_mismatched_pooling_fails(self):
        with pytest.raises(AssertionError, match="pooling"):
            check_protein_level_comparison(
                pooling_method="dct_k4",
                raw_pooling="mean",
            )

    def test_per_residue_matching_passes(self):
        check_per_residue_comparison(
            probe_type="logistic_regression",
            raw_probe_type="logistic_regression",
            hp_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            raw_hp_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        )

    def test_per_residue_mismatched_probe_fails(self):
        with pytest.raises(AssertionError, match="probe"):
            check_per_residue_comparison(
                probe_type="ridge",
                raw_probe_type="logistic_regression",
                hp_grid={"alpha": [1.0]},
                raw_hp_grid={"C": [1.0]},
            )


class TestRule2NoLeakage:
    """Rule 2: Zero overlap between train and test."""

    def test_no_overlap_passes(self):
        check_no_leakage(["a", "b", "c"], ["d", "e", "f"])

    def test_overlap_fails(self):
        with pytest.raises(AssertionError, match="leakage"):
            check_no_leakage(["a", "b", "c"], ["c", "d", "e"])


class TestRule4StatisticalSignificance:
    """Rule 4: Every metric must have CI."""

    def test_metric_result_has_ci(self):
        r = MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=1000)
        assert r.ci_lower < r.value < r.ci_upper
        assert r.ci_method == "percentile"

    def test_metric_result_rejects_no_ci(self):
        with pytest.raises(ValueError):
            MetricResult(value=0.95, ci_lower=None, ci_upper=None, n=1000)

    def test_metric_result_bca(self):
        r = MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=1000, ci_method="bca")
        assert r.ci_method == "bca"


class TestRule6ClassBalance:
    """Rule 6: Flag imbalanced classes."""

    def test_balanced_passes(self):
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        result = check_class_balance(labels)
        assert not result["imbalanced"]

    def test_imbalanced_flags(self):
        labels = np.array([0]*400 + [1]*50 + [2]*50)
        result = check_class_balance(labels)
        assert result["imbalanced"]
        assert result["max_ratio"] > 3.0


class TestRule14CrossCheck:
    """Rule 14: Cross-dataset consistency tiers."""

    def test_consistent_results_ok(self):
        result = check_cross_dataset_consistency({"CB513": 98.8, "TS115": 97.5})
        assert result["status"] == "ok"

    def test_divergent_results_warn(self):
        result = check_cross_dataset_consistency({"CB513": 98.8, "TS115": 94.5})
        assert result["status"] == "warn"

    def test_very_divergent_results_block(self):
        result = check_cross_dataset_consistency({"CB513": 98.8, "TS115": 90.0})
        assert result["status"] == "block"
