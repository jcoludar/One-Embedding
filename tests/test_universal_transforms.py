"""Tests for universal training-free pooling transforms."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.universal_transforms import (
    power_mean_pool,
    norm_weighted_mean,
    kernel_mean_embedding,
    svd_spectrum,
)

RNG = np.random.RandomState(42)


@pytest.fixture
def matrix_20x64():
    return RNG.randn(20, 64).astype(np.float32)


class TestPowerMean:
    def test_p1_equals_mean(self, matrix_20x64):
        """p=1 should degenerate to arithmetic mean."""
        result = power_mean_pool(matrix_20x64, p=1.0)
        expected = matrix_20x64.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_output_shape(self, matrix_20x64):
        assert power_mean_pool(matrix_20x64, p=3.0).shape == (64,)

    def test_output_dtype(self, matrix_20x64):
        assert power_mean_pool(matrix_20x64, p=2.0).dtype == np.float32

    def test_constant_matrix(self):
        """Constant matrix → power mean equals the constant."""
        const = np.full((10, 32), 3.0, dtype=np.float32)
        result = power_mean_pool(const, p=2.0)
        np.testing.assert_allclose(result, 3.0, atol=1e-5)


class TestNormWeighted:
    def test_output_shape(self, matrix_20x64):
        assert norm_weighted_mean(matrix_20x64).shape == (64,)

    def test_output_dtype(self, matrix_20x64):
        assert norm_weighted_mean(matrix_20x64).dtype == np.float32

    def test_equal_norms_equals_mean(self):
        """If all residues have equal norms, norm-weighted = mean."""
        # Create matrix where each row has the same norm
        matrix = RNG.randn(10, 32).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / norms  # all unit norm
        result = norm_weighted_mean(matrix)
        expected = matrix.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestKernelME:
    def test_output_shape(self, matrix_20x64):
        result = kernel_mean_embedding(matrix_20x64, D_out=128)
        assert result.shape == (128,)

    def test_output_dtype(self, matrix_20x64):
        result = kernel_mean_embedding(matrix_20x64, D_out=128)
        assert result.dtype == np.float32

    def test_deterministic(self, matrix_20x64):
        r1 = kernel_mean_embedding(matrix_20x64, seed=42)
        r2 = kernel_mean_embedding(matrix_20x64, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self, matrix_20x64):
        r1 = kernel_mean_embedding(matrix_20x64, seed=42)
        r2 = kernel_mean_embedding(matrix_20x64, seed=99)
        assert not np.allclose(r1, r2)


class TestSVDSpectrum:
    def test_output_shape(self, matrix_20x64):
        result = svd_spectrum(matrix_20x64, k=8)
        assert result.shape == (8,)

    def test_sorted_descending(self, matrix_20x64):
        result = svd_spectrum(matrix_20x64, k=8)
        assert np.all(result[:-1] >= result[1:])

    def test_nonnegative(self, matrix_20x64):
        result = svd_spectrum(matrix_20x64, k=8)
        assert np.all(result >= 0)

    def test_padding_when_k_exceeds_rank(self):
        """k > min(L,D) should pad with zeros."""
        matrix = RNG.randn(3, 4).astype(np.float32)
        result = svd_spectrum(matrix, k=8)
        assert result.shape == (8,)
        assert result[3] == 0.0  # only 3 nonzero singular values
