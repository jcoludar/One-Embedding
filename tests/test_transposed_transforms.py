"""Tests for transposed matrix view transforms — mathematical correctness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.transposed_transforms import (
    channel_resample,
    channel_statistics,
    per_protein_svd,
    zero_pad_flatten,
)


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def random_matrix():
    """Random (L, D) matrix."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 256).astype(np.float32)


@pytest.fixture
def short_matrix():
    """Short protein (L=5, D=64)."""
    rng = np.random.RandomState(99)
    return rng.randn(5, 64).astype(np.float32)


@pytest.fixture
def wide_matrix():
    """Wide protein with large L."""
    rng = np.random.RandomState(7)
    return rng.randn(600, 128).astype(np.float32)


# -- channel_resample --------------------------------------------------------


class TestChannelResample:
    def test_output_shape(self, random_matrix):
        result = channel_resample(random_matrix, l_out=64)
        assert result.shape == (256, 64)
        assert result.dtype == np.float32

    def test_fixed_size_regardless_of_L(self):
        """Different protein lengths produce the same output shape."""
        rng = np.random.RandomState(42)
        for L in [10, 50, 100, 500]:
            mat = rng.randn(L, 32).astype(np.float32)
            result = channel_resample(mat, l_out=64)
            assert result.shape == (32, 64), f"Failed for L={L}"

    def test_identity_when_L_equals_l_out(self):
        """When L == l_out, resampling should approximately preserve values."""
        rng = np.random.RandomState(42)
        mat = rng.randn(64, 32).astype(np.float32)
        result = channel_resample(mat, l_out=64)
        # Result is (D, l_out) = (32, 64), original transposed is (32, 64)
        np.testing.assert_allclose(result, mat.T, atol=1e-5)

    def test_different_l_out_values(self, random_matrix):
        """Supports various output sizes."""
        for l_out in [16, 32, 128, 256]:
            result = channel_resample(random_matrix, l_out=l_out)
            assert result.shape == (256, l_out)

    def test_preserves_channel_means(self, random_matrix):
        """Resampling should roughly preserve the mean of each channel."""
        result = channel_resample(random_matrix, l_out=64)
        original_means = random_matrix.mean(axis=0)  # (D,)
        resampled_means = result.mean(axis=1)  # (D,)
        np.testing.assert_allclose(resampled_means, original_means, atol=0.15)


# -- per_protein_svd ---------------------------------------------------------


class TestPerProteinSVD:
    def test_output_shape_k1(self, random_matrix):
        result = per_protein_svd(random_matrix, k=1)
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_output_shape_k4(self, random_matrix):
        result = per_protein_svd(random_matrix, k=4)
        assert result.shape == (256 * 4,)
        assert result.dtype == np.float32

    def test_k1_correlates_with_mean(self):
        """Top-1 SVD component should correlate with the mean direction."""
        # Use a matrix with clear non-zero mean so SVD top component aligns
        rng = np.random.RandomState(42)
        mat = rng.randn(100, 256).astype(np.float32) + 2.0  # shift away from zero
        result = per_protein_svd(mat, k=1)
        mean_vec = mat.mean(axis=0)
        # Cosine similarity (sign-invariant)
        cos_sim = abs(np.dot(result, mean_vec)) / (
            np.linalg.norm(result) * np.linalg.norm(mean_vec) + 1e-12
        )
        # With non-zero mean, top singular vector should align with mean
        assert cos_sim > 0.5

    def test_handles_short_protein_L_less_than_k(self, short_matrix):
        """When L < k, output should be zero-padded."""
        D = short_matrix.shape[1]  # 64
        k = 10  # L=5 < k=10
        result = per_protein_svd(short_matrix, k=k)
        assert result.shape == (D * k,)
        # Components beyond min(D, L)=5 should be zero
        trailing = result[D * 5:]
        np.testing.assert_allclose(trailing, 0.0, atol=1e-7)

    def test_single_residue(self):
        """Single residue protein: only k=1 component is non-zero."""
        mat = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # (1, 4)
        result = per_protein_svd(mat, k=3)
        assert result.shape == (12,)
        # Only first 4 values should be non-zero (k=1 component)
        assert np.linalg.norm(result[:4]) > 0
        np.testing.assert_allclose(result[4:], 0.0, atol=1e-7)

    def test_different_proteins_different_vectors(self):
        """Different inputs produce different SVD vectors."""
        rng = np.random.RandomState(42)
        mat1 = rng.randn(80, 64).astype(np.float32)
        mat2 = rng.randn(120, 64).astype(np.float32) + 5.0
        v1 = per_protein_svd(mat1, k=2)
        v2 = per_protein_svd(mat2, k=2)
        assert not np.allclose(v1, v2)


# -- channel_statistics ------------------------------------------------------


class TestChannelStatistics:
    def test_output_shape_default(self, random_matrix):
        result = channel_statistics(random_matrix)
        # Default: 6 stats * 256 channels
        assert result.shape == (256 * 6,)
        assert result.dtype == np.float32

    def test_subset_stats(self, random_matrix):
        """Custom subset of statistics."""
        result = channel_statistics(random_matrix, stats=["mean", "std"])
        assert result.shape == (256 * 2,)

    def test_single_stat(self, random_matrix):
        result = channel_statistics(random_matrix, stats=["max"])
        assert result.shape == (256,)

    def test_mean_stat_matches_numpy(self, random_matrix):
        """The mean component should exactly match X.mean(axis=0)."""
        result = channel_statistics(random_matrix, stats=["mean"])
        expected = random_matrix.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_std_stat_matches_numpy(self, random_matrix):
        """The std component should match X.std(axis=0)."""
        result = channel_statistics(random_matrix, stats=["std"])
        expected = random_matrix.std(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_min_max_ordering(self, random_matrix):
        """Min should be <= max for every channel."""
        result = channel_statistics(random_matrix, stats=["min", "max"])
        D = random_matrix.shape[1]
        mins = result[:D]
        maxs = result[D:]
        assert np.all(mins <= maxs)

    def test_mean_in_full_output(self, random_matrix):
        """Mean component is first D values in default output."""
        result = channel_statistics(random_matrix)
        D = random_matrix.shape[1]
        expected_mean = random_matrix.mean(axis=0)
        np.testing.assert_allclose(result[:D], expected_mean, atol=1e-5)

    def test_unknown_stat_raises(self, random_matrix):
        with pytest.raises(ValueError, match="Unknown statistic"):
            channel_statistics(random_matrix, stats=["median"])

    def test_constant_matrix_zero_std(self):
        mat = np.ones((50, 32), dtype=np.float32) * 3.0
        result = channel_statistics(mat, stats=["std", "skew", "kurtosis"])
        # Std of constant signal is 0
        std_part = result[:32]
        np.testing.assert_allclose(std_part, 0.0, atol=1e-6)


# -- zero_pad_flatten --------------------------------------------------------


class TestZeroPadFlatten:
    def test_output_shape(self, random_matrix):
        result = zero_pad_flatten(random_matrix, l_max=512)
        D = random_matrix.shape[1]
        assert result.shape == (D * 512,)
        assert result.dtype == np.float32

    def test_truncation(self, wide_matrix):
        """When L > l_max, truncates to first l_max residues."""
        D = wide_matrix.shape[1]  # 128
        l_max = 100
        result = zero_pad_flatten(wide_matrix, l_max=l_max)
        assert result.shape == (D * l_max,)
        # Reshape back to (D, l_max) and transpose to compare
        reconstructed = result.reshape(D, l_max).T  # (l_max, D)
        np.testing.assert_allclose(reconstructed, wide_matrix[:l_max], atol=1e-5)

    def test_padding_is_zero(self, short_matrix):
        """When L < l_max, padding positions should be zero."""
        L, D = short_matrix.shape  # (5, 64)
        l_max = 20
        result = zero_pad_flatten(short_matrix, l_max=l_max)
        # Reshape to (D, l_max)
        reshaped = result.reshape(D, l_max)
        # For each channel, positions L:l_max should be zero
        padding = reshaped[:, L:]
        np.testing.assert_allclose(padding, 0.0, atol=1e-7)

    def test_exact_length(self):
        """When L == l_max, no padding or truncation needed."""
        rng = np.random.RandomState(42)
        mat = rng.randn(64, 32).astype(np.float32)
        result = zero_pad_flatten(mat, l_max=64)
        assert result.shape == (32 * 64,)
        reconstructed = result.reshape(32, 64).T
        np.testing.assert_allclose(reconstructed, mat, atol=1e-5)

    def test_content_preserved(self, short_matrix):
        """Original data should be recoverable from the padded/flattened output."""
        L, D = short_matrix.shape
        l_max = 50
        result = zero_pad_flatten(short_matrix, l_max=l_max)
        reshaped = result.reshape(D, l_max)
        # Original data in first L positions of each channel
        recovered = reshaped[:, :L].T  # (L, D)
        np.testing.assert_allclose(recovered, short_matrix, atol=1e-5)

    def test_different_l_max_values(self, random_matrix):
        """Supports various l_max values."""
        D = random_matrix.shape[1]
        for l_max in [10, 50, 200, 1024]:
            result = zero_pad_flatten(random_matrix, l_max=l_max)
            assert result.shape == (D * l_max,)
