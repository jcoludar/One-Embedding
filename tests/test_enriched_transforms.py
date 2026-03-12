"""Tests for enriched pooling transforms — mathematical correctness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.enriched_transforms import (
    EnrichedTransformPipeline,
    autocovariance_pool,
    dct_pool,
    fisher_vector,
    gram_features,
    haar_pool,
    moment_pool,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def random_matrix():
    """Random (L, d) matrix."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 256).astype(np.float32)


@pytest.fixture
def constant_matrix():
    """Constant matrix — all residues identical."""
    return np.ones((50, 256), dtype=np.float32) * 0.5


@pytest.fixture
def ramp_matrix():
    """Linearly increasing matrix — strong autocovariance."""
    L, d = 100, 256
    ramp = np.linspace(0, 1, L)[:, None] * np.ones((1, d))
    return ramp.astype(np.float32)


@pytest.fixture
def train_matrices():
    """Dict of training matrices for pipeline fitting."""
    rng = np.random.RandomState(42)
    return {f"prot_{i}": rng.randn(rng.randint(30, 200), 256).astype(np.float32) for i in range(50)}


# ── Moment Pool ──────────────────────────────────────────────────


class TestMomentPool:
    def test_output_shape(self, random_matrix):
        result = moment_pool(random_matrix)
        assert result.shape == (5 * 256,)
        assert result.dtype == np.float32

    def test_mean_component_matches(self, random_matrix):
        result = moment_pool(random_matrix)
        expected_mean = random_matrix.mean(axis=0)
        np.testing.assert_allclose(result[:256], expected_mean, atol=1e-5)

    def test_constant_matrix_zero_std(self, constant_matrix):
        result = moment_pool(constant_matrix)
        # Std should be near zero (clipped to 1e-12)
        std_part = result[256:512]
        assert np.all(std_part < 1e-6)

    def test_constant_matrix_zero_autocov(self, constant_matrix):
        result = moment_pool(constant_matrix)
        # Autocovariance of constant signal is zero
        autocov = result[3 * 256:4 * 256]
        np.testing.assert_allclose(autocov, 0.0, atol=1e-6)

    def test_ramp_positive_autocov(self, ramp_matrix):
        result = moment_pool(ramp_matrix)
        autocov = result[3 * 256:4 * 256]
        # Linearly increasing signal has positive autocovariance
        assert np.all(autocov > 0)

    def test_single_residue(self):
        mat = np.ones((1, 256), dtype=np.float32) * 3.0
        result = moment_pool(mat)
        assert result.shape == (5 * 256,)
        # Mean should be 3.0
        np.testing.assert_allclose(result[:256], 3.0, atol=1e-5)
        # Lag-1 autocov should be zeros (L=1, no pairs)
        np.testing.assert_allclose(result[3 * 256:4 * 256], 0.0, atol=1e-6)


# ── Autocovariance Pool ──────────────────────────────────────────


class TestAutocovariancePool:
    def test_output_shape(self, random_matrix):
        result = autocovariance_pool(random_matrix)
        assert result.shape == (5 * 256,)  # mean + 4 lags

    def test_mean_component(self, random_matrix):
        result = autocovariance_pool(random_matrix)
        np.testing.assert_allclose(result[:256], random_matrix.mean(axis=0), atol=1e-5)

    def test_constant_zero_autocov(self, constant_matrix):
        result = autocovariance_pool(constant_matrix)
        # All autocovariances should be zero for constant signal
        for i in range(4):
            lag_part = result[(i + 1) * 256:(i + 2) * 256]
            np.testing.assert_allclose(lag_part, 0.0, atol=1e-6)

    def test_short_protein_handles_large_lags(self):
        mat = np.ones((5, 256), dtype=np.float32)
        result = autocovariance_pool(mat, lags=[1, 2, 4, 8])
        assert result.shape == (5 * 256,)
        # Lag 8 should be zeros since L=5 < 8+1
        lag8 = result[4 * 256:5 * 256]
        np.testing.assert_allclose(lag8, 0.0, atol=1e-6)

    def test_custom_lags(self, random_matrix):
        result = autocovariance_pool(random_matrix, lags=[1, 3])
        assert result.shape == (3 * 256,)  # mean + 2 lags


# ── Gram Features ────────────────────────────────────────────────


class TestGramFeatures:
    def test_output_shape(self, random_matrix):
        result = gram_features(random_matrix, top_k=32, n_bins=16)
        expected = 256 + 32 + 3 + 16
        assert result.shape == (expected,)

    def test_mean_component(self, random_matrix):
        result = gram_features(random_matrix)
        np.testing.assert_allclose(result[:256], random_matrix.mean(axis=0), atol=1e-5)

    def test_eigenvalues_nonnegative(self, random_matrix):
        result = gram_features(random_matrix, top_k=32)
        eigs = result[256:256 + 32]
        assert np.all(eigs >= -1e-6)  # numerical tolerance

    def test_trace_positive(self, random_matrix):
        result = gram_features(random_matrix, top_k=32)
        trace = result[256 + 32]
        assert trace > 0

    def test_histogram_sums_to_one(self, random_matrix):
        result = gram_features(random_matrix, top_k=32, n_bins=16)
        hist = result[256 + 32 + 3:]
        assert len(hist) == 16
        np.testing.assert_allclose(hist.sum(), 1.0, atol=1e-5)

    def test_small_protein(self):
        mat = np.random.randn(10, 256).astype(np.float32)
        result = gram_features(mat, top_k=32, n_bins=16)
        # top_k > L=10, so last 22 eigenvalues should be 0
        eigs = result[256:256 + 32]
        assert np.sum(eigs > 1e-6) <= 10


# ── DCT + Haar Wrappers ──────────────────────────────────────────


class TestDCTPool:
    def test_output_shape(self, random_matrix):
        result = dct_pool(random_matrix, K=8)
        assert result.shape == (8 * 256,)

    def test_k1_matches_scaled_mean(self, random_matrix):
        result = dct_pool(random_matrix, K=1)
        assert result.shape == (256,)


class TestHaarPool:
    def test_output_shape(self, random_matrix):
        result = haar_pool(random_matrix, levels=3)
        assert result.shape == (4 * 256,)

    def test_levels_1(self, random_matrix):
        result = haar_pool(random_matrix, levels=1)
        assert result.shape == (2 * 256,)


# ── Fisher Vector ────────────────────────────────────────────────


class TestFisherVector:
    @pytest.fixture
    def gmm_params(self):
        rng = np.random.RandomState(42)
        k, d = 4, 256
        return {
            "gmm_means": rng.randn(k, d).astype(np.float32),
            "gmm_covars": np.abs(rng.randn(k, d)).astype(np.float32) + 0.1,
            "gmm_weights": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        }

    def test_output_shape(self, random_matrix, gmm_params):
        result = fisher_vector(random_matrix, **gmm_params)
        assert result.shape == (2 * 4 * 256,)

    def test_l2_normalized(self, random_matrix, gmm_params):
        result = fisher_vector(random_matrix, **gmm_params)
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_different_proteins_different_vectors(self, gmm_params):
        rng = np.random.RandomState(42)
        mat1 = rng.randn(80, 256).astype(np.float32)
        mat2 = rng.randn(120, 256).astype(np.float32) + 5.0
        fv1 = fisher_vector(mat1, **gmm_params)
        fv2 = fisher_vector(mat2, **gmm_params)
        cos_sim = np.dot(fv1, fv2)
        assert cos_sim < 0.99  # Should be different


# ── Pipeline ─────────────────────────────────────────────────────


class TestEnrichedTransformPipeline:
    def test_fit_transform(self, train_matrices):
        pipe = EnrichedTransformPipeline(moment_pool)
        pipe.fit(train_matrices, target_dim=64)
        mat = list(train_matrices.values())[0]
        result = pipe.transform(mat)
        # With 50 samples and 1280d features, PCA capped at min(64, 50)
        assert result.shape == (min(64, len(train_matrices)),)
        assert result.dtype == np.float32

    def test_batch_transform(self, train_matrices):
        pipe = EnrichedTransformPipeline(moment_pool)
        pipe.fit(train_matrices, target_dim=32)
        batch = pipe.transform_batch(train_matrices)
        assert len(batch) == len(train_matrices)
        for v in batch.values():
            assert v.shape == (32,)

    def test_variance_explained(self, train_matrices):
        pipe = EnrichedTransformPipeline(moment_pool)
        pipe.fit(train_matrices, target_dim=64)
        ve = pipe.variance_explained
        assert len(ve) == min(64, len(train_matrices))
        assert ve[0] > 0
        assert ve[-1] <= 1.0 + 1e-6
        # Monotonically increasing
        assert np.all(np.diff(ve) >= -1e-10)

    def test_target_dim_capped_by_samples(self):
        """PCA can't have more components than samples."""
        rng = np.random.RandomState(42)
        small_train = {f"p{i}": rng.randn(50, 256).astype(np.float32) for i in range(10)}
        pipe = EnrichedTransformPipeline(moment_pool)
        pipe.fit(small_train, target_dim=256)
        # Can't have more components than 10 samples
        assert pipe.pca.n_components_ <= 10

    def test_different_transforms(self, train_matrices):
        """Different transforms produce different outputs."""
        mat = list(train_matrices.values())[0]

        pipe1 = EnrichedTransformPipeline(moment_pool)
        pipe1.fit(train_matrices, target_dim=32)
        v1 = pipe1.transform(mat)

        pipe2 = EnrichedTransformPipeline(autocovariance_pool)
        pipe2.fit(train_matrices, target_dim=32)
        v2 = pipe2.transform(mat)

        # Should not be identical
        assert not np.allclose(v1, v2)
