"""Tests for CV-tuned linear probes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from probes.linear import train_classification_probe, train_regression_probe


class TestClassificationProbe:

    def _make_data(self, n=500, d=64, n_classes=3, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randint(0, n_classes, size=n)
        for c in range(n_classes):
            X[y == c] += c * 0.5
        return X, y

    def test_returns_predictions_and_best_C(self):
        X, y = self._make_data()
        result = train_classification_probe(
            X_train=X[:400], y_train=y[:400],
            X_test=X[400:], y_test=y[400:],
            C_grid=[0.1, 1.0, 10.0], cv_folds=3, seed=42,
        )
        assert "predictions" in result
        assert "best_C" in result
        assert "accuracy" in result
        assert len(result["predictions"]) == 100

    def test_cv_selects_best_C(self):
        X, y = self._make_data(n=1000)
        result = train_classification_probe(
            X_train=X[:800], y_train=y[:800],
            X_test=X[800:], y_test=y[800:],
            C_grid=[0.001, 0.01, 0.1, 1.0, 10.0], cv_folds=3, seed=42,
        )
        assert result["best_C"] in [0.001, 0.01, 0.1, 1.0, 10.0]

    def test_per_class_accuracy(self):
        X, y = self._make_data()
        result = train_classification_probe(
            X_train=X[:400], y_train=y[:400],
            X_test=X[400:], y_test=y[400:],
            C_grid=[1.0], cv_folds=3, seed=42,
        )
        assert "per_class_acc" in result
        assert len(result["per_class_acc"]) == 3


class TestRegressionProbe:

    def _make_data(self, n=500, d=64, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        w = rng.randn(d)
        y = X @ w + rng.randn(n) * 0.1
        return X, y.astype(np.float64)

    def test_returns_predictions_and_best_alpha(self):
        X, y = self._make_data()
        result = train_regression_probe(
            X_train=X[:400], y_train=y[:400],
            X_test=X[400:], y_test=y[400:],
            alpha_grid=[0.1, 1.0, 10.0], cv_folds=3, seed=42,
        )
        assert "predictions" in result
        assert "best_alpha" in result
        assert "spearman_rho" in result
        assert len(result["predictions"]) == 100

    def test_spearman_is_high_for_clean_data(self):
        X, y = self._make_data()
        result = train_regression_probe(
            X_train=X[:400], y_train=y[:400],
            X_test=X[400:], y_test=y[400:],
            alpha_grid=[0.1, 1.0, 10.0], cv_folds=3, seed=42,
        )
        assert result["spearman_rho"] > 0.9
