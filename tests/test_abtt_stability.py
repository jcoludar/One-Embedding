"""Tests for ABTT cross-corpus stability metrics.

Tests principal_angles, subspace_similarity, and cross_corpus_stability_report
using synthetic data only (no real protein embeddings needed).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from the Exp 43 benchmark directory
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark")
)

from metrics.abtt_stability import (
    cross_corpus_stability_report,
    principal_angles,
    subspace_similarity,
)


# ---------------------------------------------------------------------------
# TestPrincipalAngles
# ---------------------------------------------------------------------------
class TestPrincipalAngles:
    """Tests for the Bjorck & Golub (1973) principal angles computation."""

    def test_identical_subspaces_zero_angles(self):
        """Same (3, 100) matrix should give angles approximately 0."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 100))
        angles = principal_angles(A, A)
        np.testing.assert_allclose(angles, 0.0, atol=1e-6)

    def test_orthogonal_subspaces_90_degrees(self):
        """First 3 rows of eye(100) vs rows 3:6 should give angles of pi/2."""
        I = np.eye(100)
        A = I[:3]   # rows 0, 1, 2
        B = I[3:6]  # rows 3, 4, 5
        angles = principal_angles(A, B)
        np.testing.assert_allclose(angles, np.pi / 2, atol=1e-10)

    def test_returns_sorted_angles(self):
        """Angles should be sorted in ascending order."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 100))
        B = rng.standard_normal((3, 100))
        angles = principal_angles(A, B)
        assert len(angles) == 3
        assert np.all(np.diff(angles) >= -1e-12), "Angles must be sorted ascending"

    def test_angles_between_0_and_pi_half(self):
        """All angles must lie in [0, pi/2]."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((3, 100))
        B = rng.standard_normal((3, 100))
        angles = principal_angles(A, B)
        assert np.all(angles >= -1e-12)
        assert np.all(angles <= np.pi / 2 + 1e-12)


# ---------------------------------------------------------------------------
# TestSubspaceSimilarity
# ---------------------------------------------------------------------------
class TestSubspaceSimilarity:
    """Tests for the subspace similarity metric (mean cos^2 of principal angles)."""

    def test_identical_is_one(self):
        """Identical subspaces should have similarity 1.0."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((3, 100))
        sim = subspace_similarity(A, A)
        assert abs(sim - 1.0) < 1e-10

    def test_orthogonal_is_zero(self):
        """Orthogonal subspaces should have similarity 0.0."""
        I = np.eye(100)
        A = I[:3]
        B = I[3:6]
        sim = subspace_similarity(A, B)
        assert abs(sim) < 1e-10

    def test_between_zero_and_one(self):
        """Random subspaces should give similarity strictly between 0 and 1."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((3, 100))
        B = rng.standard_normal((3, 100))
        sim = subspace_similarity(A, B)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# TestCrossCorpusStability
# ---------------------------------------------------------------------------
class TestCrossCorpusStability:
    """Tests for the full cross-corpus stability report."""

    def test_same_distribution_high_similarity(self):
        """Corpora from the same distribution with dominant top-3 eigenvalues
        should yield min_similarity > 0.95 and conclusion 'stable'."""
        rng = np.random.default_rng(42)
        D = 50

        # Build a covariance with 3 dominant eigenvalues (100x larger)
        eigenvalues = np.ones(D)
        eigenvalues[:3] = 100.0
        # Random orthogonal basis
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        cov = Q @ np.diag(eigenvalues) @ Q.T

        # Cholesky factor for sampling
        L = np.linalg.cholesky(cov)

        corpora = {}
        for name in ["corpus_a", "corpus_b", "corpus_c"]:
            samples = rng.standard_normal((10_000, D)) @ L.T
            corpora[name] = samples

        report = cross_corpus_stability_report(corpora, k=3, seed=42)

        assert report["min_similarity"] > 0.95, (
            f"Expected min_similarity > 0.95, got {report['min_similarity']:.4f}"
        )
        assert report["conclusion"] == "stable"

    def test_different_distributions_low_similarity(self):
        """Corpora from very different distributions should yield
        min_similarity < 0.90 and conclusion 'unstable'."""
        rng = np.random.default_rng(7)
        D = 50

        # Corpus A: dominant variance in first 3 dimensions
        corpus_a = rng.standard_normal((10_000, D))
        corpus_a[:, :3] *= 100.0

        # Corpus B: dominant variance in dimensions 20-22
        corpus_b = rng.standard_normal((10_000, D))
        corpus_b[:, 20:23] *= 100.0

        corpora = {"structured": corpus_a, "uniform": corpus_b}

        report = cross_corpus_stability_report(corpora, k=3, seed=42)

        assert report["min_similarity"] < 0.90, (
            f"Expected min_similarity < 0.90, got {report['min_similarity']:.4f}"
        )
        assert report["conclusion"] == "unstable"
