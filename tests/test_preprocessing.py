"""Tests for pre-compression preprocessing transforms."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.preprocessing import (
    all_but_the_top,
    center_embeddings,
    compute_corpus_stats,
    pca_rotate,
    zscore_embeddings,
)

RNG = np.random.RandomState(42)


@pytest.fixture
def corpus_1000x64():
    """Simulated corpus: 1000 residues, 64 channels."""
    return RNG.randn(1000, 64).astype(np.float32)


@pytest.fixture
def stats_64(corpus_1000x64):
    """Pre-computed corpus stats for 64-dim embeddings."""
    return compute_corpus_stats(corpus_1000x64, n_sample=1000, n_pcs=3, seed=42)


@pytest.fixture
def matrix_20x64():
    """Single protein: 20 residues, 64 channels."""
    return RNG.randn(20, 64).astype(np.float32)


class TestComputeCorpusStats:
    def test_output_keys(self, stats_64):
        expected_keys = {
            "mean_vec",
            "std_vec",
            "top_pcs",
            "explained_variance",
            "rotation_matrix",
        }
        assert set(stats_64.keys()) == expected_keys

    def test_mean_vec_shape(self, stats_64):
        assert stats_64["mean_vec"].shape == (64,)
        assert stats_64["mean_vec"].dtype == np.float32

    def test_std_vec_shape(self, stats_64):
        assert stats_64["std_vec"].shape == (64,)
        assert stats_64["std_vec"].dtype == np.float32

    def test_top_pcs_shape(self, stats_64):
        assert stats_64["top_pcs"].shape == (3, 64)
        assert stats_64["top_pcs"].dtype == np.float32

    def test_explained_variance_shape(self, stats_64):
        assert stats_64["explained_variance"].shape == (3,)
        assert stats_64["explained_variance"].dtype == np.float32

    def test_explained_variance_decreasing(self, stats_64):
        ev = stats_64["explained_variance"]
        assert ev[0] >= ev[1] >= ev[2]

    def test_rotation_matrix_shape(self, stats_64):
        assert stats_64["rotation_matrix"].shape == (64, 64)
        assert stats_64["rotation_matrix"].dtype == np.float32

    def test_top_pcs_are_unit_vectors(self, stats_64):
        norms = np.linalg.norm(stats_64["top_pcs"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_subsampling(self):
        """When n_sample < N, stats should still be computed."""
        big = RNG.randn(100_000, 16).astype(np.float32)
        stats = compute_corpus_stats(big, n_sample=500, n_pcs=2, seed=42)
        assert stats["mean_vec"].shape == (16,)
        assert stats["top_pcs"].shape == (2, 16)

    def test_deterministic(self, corpus_1000x64):
        s1 = compute_corpus_stats(corpus_1000x64, n_pcs=3, seed=42)
        s2 = compute_corpus_stats(corpus_1000x64, n_pcs=3, seed=42)
        np.testing.assert_array_equal(s1["mean_vec"], s2["mean_vec"])
        np.testing.assert_array_equal(s1["top_pcs"], s2["top_pcs"])


class TestCenterEmbeddings:
    def test_output_shape(self, matrix_20x64, stats_64):
        result = center_embeddings(matrix_20x64, stats_64["mean_vec"])
        assert result.shape == (20, 64)

    def test_output_dtype(self, matrix_20x64, stats_64):
        result = center_embeddings(matrix_20x64, stats_64["mean_vec"])
        assert result.dtype == np.float32

    def test_centering_shifts_mean_near_zero(self, corpus_1000x64, stats_64):
        """Centering the corpus itself should bring mean close to zero."""
        centered = center_embeddings(corpus_1000x64, stats_64["mean_vec"])
        mean_after = centered.mean(axis=0)
        np.testing.assert_allclose(mean_after, 0.0, atol=1e-5)

    def test_shape_preserved(self, stats_64):
        X = RNG.randn(50, 64).astype(np.float32)
        result = center_embeddings(X, stats_64["mean_vec"])
        assert result.shape == X.shape


class TestZscoreEmbeddings:
    def test_output_shape(self, matrix_20x64, stats_64):
        result = zscore_embeddings(
            matrix_20x64, stats_64["mean_vec"], stats_64["std_vec"]
        )
        assert result.shape == (20, 64)

    def test_output_dtype(self, matrix_20x64, stats_64):
        result = zscore_embeddings(
            matrix_20x64, stats_64["mean_vec"], stats_64["std_vec"]
        )
        assert result.dtype == np.float32

    def test_zero_std_no_nan(self):
        """Channels with zero std should NOT produce NaN."""
        X = np.ones((10, 4), dtype=np.float32)
        X[:, 2] = 5.0  # channel 2 is constant
        mean_vec = X.mean(axis=0)
        std_vec = X.std(axis=0)
        # std_vec[2] should be 0 for the constant channel
        assert std_vec[2] == 0.0

        result = zscore_embeddings(X, mean_vec, std_vec)
        assert not np.any(np.isnan(result)), "NaN found in z-scored output"
        assert not np.any(np.isinf(result)), "Inf found in z-scored output"

    def test_zero_std_channel_centered(self):
        """Zero-std channels should be centered but not scaled."""
        X = np.ones((10, 4), dtype=np.float32) * 3.0
        mean_vec = np.full(4, 3.0, dtype=np.float32)
        std_vec = np.zeros(4, dtype=np.float32)

        result = zscore_embeddings(X, mean_vec, std_vec)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_corpus_zscore_unit_variance(self, corpus_1000x64, stats_64):
        """Z-scoring the corpus should give near-unit variance per channel."""
        zscored = zscore_embeddings(
            corpus_1000x64, stats_64["mean_vec"], stats_64["std_vec"]
        )
        stds = zscored.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=0.05)

    def test_shape_preserved(self, stats_64):
        X = RNG.randn(50, 64).astype(np.float32)
        result = zscore_embeddings(X, stats_64["mean_vec"], stats_64["std_vec"])
        assert result.shape == X.shape


class TestAllButTheTop:
    def test_output_shape(self, matrix_20x64, stats_64):
        centered = center_embeddings(matrix_20x64, stats_64["mean_vec"])
        result = all_but_the_top(centered, stats_64["top_pcs"])
        assert result.shape == (20, 64)

    def test_output_dtype(self, matrix_20x64, stats_64):
        centered = center_embeddings(matrix_20x64, stats_64["mean_vec"])
        result = all_but_the_top(centered, stats_64["top_pcs"])
        assert result.dtype == np.float32

    def test_removes_dominant_pc(self, corpus_1000x64, stats_64):
        """After removal, projection onto top PC should be near zero."""
        centered = center_embeddings(corpus_1000x64, stats_64["mean_vec"])
        result = all_but_the_top(centered, stats_64["top_pcs"])

        # Projection of result onto PC1 should be near zero
        pc1 = stats_64["top_pcs"][0]
        projections = result @ pc1
        np.testing.assert_allclose(projections, 0.0, atol=1e-4)

    def test_removes_all_top_pcs(self, corpus_1000x64, stats_64):
        """Projections onto all top PCs should be near zero after removal."""
        centered = center_embeddings(corpus_1000x64, stats_64["mean_vec"])
        result = all_but_the_top(centered, stats_64["top_pcs"])

        for i in range(stats_64["top_pcs"].shape[0]):
            proj = result @ stats_64["top_pcs"][i]
            np.testing.assert_allclose(
                proj, 0.0, atol=1e-4, err_msg=f"PC {i} not removed"
            )

    def test_single_pc_removal(self, corpus_1000x64, stats_64):
        """Removing just PC1 should zero out only PC1 component."""
        centered = center_embeddings(corpus_1000x64, stats_64["mean_vec"])
        result = all_but_the_top(centered, stats_64["top_pcs"][:1])

        pc1_proj = result @ stats_64["top_pcs"][0]
        np.testing.assert_allclose(pc1_proj, 0.0, atol=1e-4)

    def test_shape_preserved(self, stats_64):
        X = RNG.randn(50, 64).astype(np.float32)
        result = all_but_the_top(X, stats_64["top_pcs"])
        assert result.shape == X.shape


class TestPcaRotate:
    def test_output_shape(self, matrix_20x64, stats_64):
        result = pca_rotate(matrix_20x64, stats_64["rotation_matrix"])
        assert result.shape == (20, 64)

    def test_output_dtype(self, matrix_20x64, stats_64):
        result = pca_rotate(matrix_20x64, stats_64["rotation_matrix"])
        assert result.dtype == np.float32

    def test_preserves_norms(self, matrix_20x64, stats_64):
        """Orthogonal rotation should preserve row norms."""
        result = pca_rotate(matrix_20x64, stats_64["rotation_matrix"])
        original_norms = np.linalg.norm(matrix_20x64, axis=1)
        rotated_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-3)

    def test_preserves_pairwise_distances(self, matrix_20x64, stats_64):
        """Orthogonal rotation should preserve pairwise distances."""
        result = pca_rotate(matrix_20x64, stats_64["rotation_matrix"])

        # Check a few pairwise distances
        for i, j in [(0, 1), (0, 5), (3, 7)]:
            d_orig = np.linalg.norm(matrix_20x64[i] - matrix_20x64[j])
            d_rot = np.linalg.norm(result[i] - result[j])
            np.testing.assert_allclose(d_rot, d_orig, atol=1e-3)

    def test_shape_preserved(self, stats_64):
        X = RNG.randn(50, 64).astype(np.float32)
        result = pca_rotate(X, stats_64["rotation_matrix"])
        assert result.shape == X.shape

    def test_rotation_is_invertible(self, matrix_20x64, stats_64):
        """Rotating and then inverse-rotating should recover original."""
        R = stats_64["rotation_matrix"]
        rotated = pca_rotate(matrix_20x64, R)
        # Inverse rotation: multiply by R (since R @ R.T = I for orthogonal)
        recovered = rotated @ R
        np.testing.assert_allclose(recovered, matrix_20x64, atol=1e-3)
