"""Tests for path geometry transforms.

Verifies mathematical properties: displacement losslessness, signature
algebraic structure, curvature invariants, eigenvalue ordering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.path_transforms import (
    curvature_enriched,
    discrete_curvature,
    displacement_decode,
    displacement_dct,
    displacement_encode,
    displacement_magnitude,
    gyration_eigenspectrum,
    inverse_displacement_dct,
    lag_cross_covariance_eigenvalues,
    path_signature_depth2,
    path_signature_depth3,
    path_statistics,
    shape_descriptors,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def matrix_small(rng):
    """(50, 16) — small path for signature tests."""
    return rng.randn(50, 16).astype(np.float32)


@pytest.fixture
def matrix_large(rng):
    """(200, 1024) — realistic protein size."""
    return rng.randn(200, 1024).astype(np.float32)


# ── Displacement Encoding ────────────────────────────────────────


class TestDisplacement:
    def test_encode_shape(self, matrix_large):
        dx = displacement_encode(matrix_large)
        assert dx.shape == (199, 1024)

    def test_lossless_roundtrip(self, matrix_large):
        dx = displacement_encode(matrix_large)
        x0 = matrix_large[0]
        recovered = displacement_decode(dx, x0)
        np.testing.assert_allclose(recovered, matrix_large, atol=1e-4)

    def test_zero_displacements_for_constant_path(self):
        constant = np.ones((10, 8), dtype=np.float32) * 3.14
        dx = displacement_encode(constant)
        np.testing.assert_allclose(dx, 0, atol=1e-7)

    def test_linear_path_constant_displacement(self):
        t = np.linspace(0, 1, 20)[:, np.newaxis]
        direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        path = (t * direction).astype(np.float32)
        dx = displacement_encode(path)
        # All displacements should be identical
        for i in range(1, len(dx)):
            np.testing.assert_allclose(dx[i], dx[0], atol=1e-5)


class TestDisplacementDCT:
    def test_roundtrip(self, matrix_large):
        K = 16
        D = matrix_large.shape[1]
        L = matrix_large.shape[0]
        x0 = matrix_large[0]

        coeffs = displacement_dct(matrix_large, K=K)
        recovered = inverse_displacement_dct(coeffs, D=D, target_len=L, x0=x0)
        assert recovered.shape == matrix_large.shape

        # Not lossless (K < L-1) but should have some signal
        cos_sims = []
        for i in range(L):
            n1 = np.linalg.norm(matrix_large[i])
            n2 = np.linalg.norm(recovered[i])
            if n1 > 1e-8 and n2 > 1e-8:
                cos_sims.append(np.dot(matrix_large[i], recovered[i]) / (n1 * n2))
        mean_cos = np.mean(cos_sims)
        assert mean_cos > 0.0  # Some positive correlation

    def test_full_K_lossless(self):
        """With K = L-1, displacement DCT should be lossless."""
        rng = np.random.RandomState(42)
        matrix = rng.randn(20, 8).astype(np.float32)
        K = 19  # L - 1
        D = 8
        x0 = matrix[0]

        coeffs = displacement_dct(matrix, K=K)
        recovered = inverse_displacement_dct(coeffs, D=D, target_len=20, x0=x0)
        np.testing.assert_allclose(recovered, matrix, atol=1e-4)


# ── Path Signatures ──────────────────────────────────────────────


class TestPathSignature:
    def test_depth2_shape(self, matrix_small):
        sig = path_signature_depth2(matrix_small)
        d = 16
        expected_len = 1 + d + d * d  # 1 + 16 + 256 = 273
        assert sig.shape == (expected_len,)

    def test_depth3_shape(self, matrix_small):
        sig = path_signature_depth3(matrix_small)
        d = 16
        expected_len = 1 + d + d * d + d * d * d  # 1 + 16 + 256 + 4096 = 4369
        assert sig.shape == (expected_len,)

    def test_depth2_first_element_is_one(self, matrix_small):
        sig = path_signature_depth2(matrix_small)
        assert abs(sig[0] - 1.0) < 1e-6

    def test_depth2_displacement_matches(self, matrix_small):
        """Depth-1 component should equal total displacement."""
        sig = path_signature_depth2(matrix_small)
        d = 16
        sig1 = sig[1 : 1 + d]
        total_displacement = matrix_small[-1] - matrix_small[0]
        np.testing.assert_allclose(sig1, total_displacement, atol=1e-4)

    def test_depth3_includes_depth2(self, matrix_small):
        """Depth-3 signature should agree with depth-2 for shared components."""
        sig2 = path_signature_depth2(matrix_small)
        sig3 = path_signature_depth3(matrix_small)
        # First 1+d+d^2 elements should match
        n = len(sig2)
        np.testing.assert_allclose(sig3[:n], sig2, atol=1e-3)

    def test_constant_path_zero_signature(self):
        """Constant path has zero displacement and zero area."""
        path = np.ones((30, 4), dtype=np.float32) * 2.0
        sig = path_signature_depth2(path)
        # sig[0] = 1, everything else = 0
        assert abs(sig[0] - 1.0) < 1e-6
        np.testing.assert_allclose(sig[1:], 0.0, atol=1e-6)

    def test_reversed_path_depth2_antisymmetric(self):
        """Reversing a path negates depth-1, changes depth-2 structure."""
        rng = np.random.RandomState(42)
        path = rng.randn(20, 4).astype(np.float32)
        sig_fwd = path_signature_depth2(path)
        sig_rev = path_signature_depth2(path[::-1])
        d = 4
        # Depth-1: reversed path has negated displacement
        np.testing.assert_allclose(sig_fwd[1 : 1 + d], -sig_rev[1 : 1 + d], atol=1e-4)

    def test_different_orderings_different_signatures(self):
        """Two paths with same points but different order should differ."""
        rng = np.random.RandomState(42)
        points = rng.randn(20, 4).astype(np.float32)
        shuffled = points[rng.permutation(20)]
        sig1 = path_signature_depth2(points)
        sig2 = path_signature_depth2(shuffled)
        # Should NOT be equal (signatures capture order)
        assert not np.allclose(sig1, sig2, atol=1e-2)


# ── Cross-Covariance ────────────────────────────────────────────


class TestCrossCovariance:
    def test_shape(self, matrix_large):
        eigvals = lag_cross_covariance_eigenvalues(matrix_large, k=64)
        assert eigvals.shape == (64,)

    def test_nonnegative(self, matrix_large):
        # Singular values are non-negative
        eigvals = lag_cross_covariance_eigenvalues(matrix_large, k=64)
        assert np.all(eigvals >= -1e-6)

    def test_descending_order(self, matrix_large):
        eigvals = lag_cross_covariance_eigenvalues(matrix_large, k=64)
        # Singular values should be non-increasing
        assert np.all(np.diff(eigvals) <= 1e-6)

    def test_zero_padded_if_small(self):
        """If L < k, should zero-pad."""
        small = np.random.randn(5, 8).astype(np.float32)
        eigvals = lag_cross_covariance_eigenvalues(small, k=64)
        assert eigvals.shape == (64,)
        # Most values should be zero (only min(L,D)=5 non-trivial SVs)
        assert np.sum(eigvals > 1e-6) <= 8


# ── Curvature ────────────────────────────────────────────────────


class TestCurvature:
    def test_shape(self, matrix_large):
        curv = discrete_curvature(matrix_large)
        assert curv.shape == (198,)  # L - 2

    def test_nonnegative(self, matrix_large):
        curv = discrete_curvature(matrix_large)
        assert np.all(curv >= 0)

    def test_straight_line_zero_curvature(self):
        """Straight line path has zero curvature."""
        t = np.linspace(0, 1, 50)[:, np.newaxis]
        direction = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        path = (t * direction).astype(np.float32)
        curv = discrete_curvature(path)
        np.testing.assert_allclose(curv, 0.0, atol=1e-5)

    def test_displacement_magnitude_shape(self, matrix_large):
        speed = displacement_magnitude(matrix_large)
        assert speed.shape == (199,)  # L - 1
        assert np.all(speed >= 0)


class TestCurvatureEnriched:
    def test_shape(self, matrix_large):
        enriched = curvature_enriched(matrix_large)
        assert enriched.shape == (200, 1027)  # D + 3

    def test_original_preserved(self, matrix_large):
        enriched = curvature_enriched(matrix_large)
        np.testing.assert_allclose(enriched[:, :1024], matrix_large, atol=1e-6)

    def test_position_fraction(self, matrix_large):
        enriched = curvature_enriched(matrix_large)
        pos_frac = enriched[:, -1]
        np.testing.assert_allclose(pos_frac[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(pos_frac[-1], 1.0, atol=1e-6)


# ── Gyration / Shape ────────────────────────────────────────────


class TestGyration:
    def test_eigenspectrum_shape(self, matrix_large):
        eig = gyration_eigenspectrum(matrix_large, k=64)
        assert eig.shape == (64,)

    def test_nonnegative(self, matrix_large):
        eig = gyration_eigenspectrum(matrix_large, k=64)
        assert np.all(eig >= -1e-6)

    def test_descending(self, matrix_large):
        eig = gyration_eigenspectrum(matrix_large, k=64)
        assert np.all(np.diff(eig) <= 1e-6)

    def test_shape_descriptors_length(self, matrix_large):
        sd = shape_descriptors(matrix_large)
        assert sd.shape == (5,)

    def test_radius_of_gyration_positive(self, matrix_large):
        sd = shape_descriptors(matrix_large)
        assert sd[0] > 0  # R_g


# ── Path Statistics ──────────────────────────────────────────────


class TestPathStatistics:
    def test_output_shape(self, matrix_large):
        stats = path_statistics(matrix_large)
        assert stats.ndim == 1
        assert len(stats) >= 20  # Should have ~25-30 features

    def test_all_finite(self, matrix_large):
        stats = path_statistics(matrix_large)
        assert np.all(np.isfinite(stats))

    def test_short_protein(self):
        """Should handle very short proteins gracefully."""
        short = np.random.randn(3, 8).astype(np.float32)
        stats = path_statistics(short)
        assert np.all(np.isfinite(stats))
