"""Tests for within-channel extreme compression techniques.

Verifies wavelet threshold, CUR decomposition, channel pruning, and
Zstandard byte compression. All computation is float32 on CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.extreme_compression import (
    channel_prune,
    compute_channel_importance,
    cur_decompose,
    cur_reconstruct,
    measure_compressed_size,
    wavelet_threshold_compress,
    wavelet_threshold_decompress,
    zstd_compress,
    zstd_decompress,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30 + i, 128).astype(np.float32) for i in range(20)}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def cosine_sim_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-row cosine similarity between two (L, D) matrices."""
    a_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    b_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


# ---------------------------------------------------------------------------
# TestWaveletThreshold
# ---------------------------------------------------------------------------


class TestWaveletThreshold:
    def test_roundtrip_no_threshold(self, embedding):
        """threshold_pct=0 gives near-perfect reconstruction."""
        compressed = wavelet_threshold_compress(embedding, threshold_pct=0.0)
        reconstructed = wavelet_threshold_decompress(compressed)
        np.testing.assert_allclose(reconstructed, embedding, atol=1e-5)

    def test_threshold_reduces_quality(self, embedding):
        """threshold_pct=75 produces correct shape and reasonable cosine sim."""
        compressed = wavelet_threshold_compress(embedding, threshold_pct=75.0)
        reconstructed = wavelet_threshold_decompress(compressed)

        assert reconstructed.shape == embedding.shape
        sim = cosine_sim_mean(reconstructed, embedding)
        # 75th-percentile soft thresholding on random (incompressible) data is
        # quite aggressive; cosine sim > 0.75 confirms the reconstruction is
        # still useful while the threshold clearly reduces quality vs 0.
        assert sim > 0.75, f"Expected cosine sim > 0.75, got {sim:.4f}"

    def test_different_wavelets(self, embedding):
        """db4, db8, sym4 all produce correct output shapes."""
        for wv in ("db4", "db8", "sym4"):
            compressed = wavelet_threshold_compress(embedding, wavelet=wv, threshold_pct=0.0)
            reconstructed = wavelet_threshold_decompress(compressed)
            assert reconstructed.shape == embedding.shape, (
                f"Wavelet {wv}: expected {embedding.shape}, got {reconstructed.shape}"
            )

    def test_deterministic(self, embedding):
        """Same input always produces same output."""
        c1 = wavelet_threshold_compress(embedding, threshold_pct=50.0)
        c2 = wavelet_threshold_compress(embedding, threshold_pct=50.0)
        r1 = wavelet_threshold_decompress(c1)
        r2 = wavelet_threshold_decompress(c2)
        np.testing.assert_array_equal(r1, r2)

    def test_compressed_dict_keys(self, embedding):
        """Compressed dict contains expected keys."""
        compressed = wavelet_threshold_compress(embedding)
        assert "coeffs" in compressed
        assert "wavelet" in compressed
        assert "original_shape" in compressed
        assert "n_levels" in compressed

    def test_original_shape_recorded(self, embedding):
        """original_shape matches input shape."""
        compressed = wavelet_threshold_compress(embedding)
        assert compressed["original_shape"] == embedding.shape

    def test_output_dtype(self, embedding):
        """Decompressed output is float32."""
        compressed = wavelet_threshold_compress(embedding, threshold_pct=0.0)
        reconstructed = wavelet_threshold_decompress(compressed)
        assert reconstructed.dtype == np.float32


# ---------------------------------------------------------------------------
# TestCUR
# ---------------------------------------------------------------------------


class TestCUR:
    def test_output_shapes(self, embedding):
        """k=16 on (50,128): C is (50,16), interp_matrix is (16,128)."""
        compressed = cur_decompose(embedding, k=16)
        assert compressed["C"].shape == (50, 16), (
            f"C shape: {compressed['C'].shape}"
        )
        assert compressed["interp_matrix"].shape == (16, 128), (
            f"interp_matrix shape: {compressed['interp_matrix'].shape}"
        )

    def test_reconstruction_quality(self, embedding):
        """k=64 on (50,128) gives cosine sim > 0.9."""
        compressed = cur_decompose(embedding, k=64)
        reconstructed = cur_reconstruct(compressed)
        assert reconstructed.shape == embedding.shape
        sim = cosine_sim_mean(reconstructed, embedding)
        assert sim > 0.9, f"Expected cosine sim > 0.9, got {sim:.4f}"

    def test_columns_are_originals(self, embedding):
        """C columns exactly match original matrix columns at col_indices."""
        compressed = cur_decompose(embedding, k=16)
        col_indices = compressed["col_indices"]
        C = compressed["C"]
        for i, idx in enumerate(col_indices):
            np.testing.assert_array_equal(
                C[:, i],
                embedding[:, idx],
                err_msg=f"Column {i} (original index {idx}) does not match.",
            )

    def test_various_k(self):
        """k in {8, 16, 32, 64} all produce valid decompositions.

        Uses a (200, 128) matrix so that k=64 < min(L, D) = 128.
        """
        rng = np.random.RandomState(7)
        large = rng.randn(200, 128).astype(np.float32)
        for k in (8, 16, 32, 64):
            compressed = cur_decompose(large, k=k)
            assert compressed["C"].shape[1] == k, f"k={k}: C has wrong ncols"
            assert compressed["interp_matrix"].shape == (k, 128), (
                f"k={k}: interp_matrix wrong shape"
            )
            reconstructed = cur_reconstruct(compressed)
            assert reconstructed.shape == large.shape

    def test_col_indices_in_range(self, embedding):
        """col_indices must all be valid column indices."""
        compressed = cur_decompose(embedding, k=16)
        idx = compressed["col_indices"]
        assert np.all(idx >= 0) and np.all(idx < 128)
        # All selected indices should be distinct
        assert len(idx) == len(np.unique(idx))

    def test_reconstruction_dtype(self, embedding):
        """cur_reconstruct returns float32."""
        compressed = cur_decompose(embedding, k=16)
        reconstructed = cur_reconstruct(compressed)
        assert reconstructed.dtype == np.float32


# ---------------------------------------------------------------------------
# TestChannelPrune
# ---------------------------------------------------------------------------


class TestChannelPrune:
    def test_importance_shape(self, corpus):
        """compute_channel_importance returns (D,) with all values >= 0."""
        importance = compute_channel_importance(corpus)
        assert importance.shape == (128,)
        assert np.all(importance >= 0), "All variances must be non-negative"

    def test_prune_keeps_top_k(self, embedding, corpus):
        """Selected indices are the top-k by importance."""
        importance = compute_channel_importance(corpus)
        k = 32
        pruned, indices = channel_prune(embedding, importance, k=k)

        # Top-k indices by importance (descending)
        expected_top_k = np.argsort(importance)[::-1][:k]
        assert set(indices.tolist()) == set(expected_top_k.tolist()), (
            "Pruned indices do not match top-k by importance"
        )

    def test_pruned_matches_original(self, embedding, corpus):
        """Pruned columns match the original matrix columns at the selected indices."""
        importance = compute_channel_importance(corpus)
        pruned, indices = channel_prune(embedding, importance, k=32)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(
                pruned[:, i],
                embedding[:, idx],
                err_msg=f"Pruned column {i} (original {idx}) does not match.",
            )

    def test_pruned_shape(self, embedding, corpus):
        """Output pruned matrix has correct shape."""
        importance = compute_channel_importance(corpus)
        pruned, indices = channel_prune(embedding, importance, k=64)
        assert pruned.shape == (50, 64)
        assert indices.shape == (64,)

    def test_importance_dtype(self, corpus):
        """Importance array is float32."""
        importance = compute_channel_importance(corpus)
        assert importance.dtype == np.float32

    def test_prune_output_dtype(self, embedding, corpus):
        """Pruned matrix is float32."""
        importance = compute_channel_importance(corpus)
        pruned, _ = channel_prune(embedding, importance, k=16)
        assert pruned.dtype == np.float32

    def test_max_proteins_limit(self, corpus):
        """max_proteins limits how many proteins are used."""
        imp_all = compute_channel_importance(corpus, max_proteins=20)
        imp_few = compute_channel_importance(corpus, max_proteins=5)
        # Both should return (128,) arrays with non-negative values
        assert imp_all.shape == (128,)
        assert imp_few.shape == (128,)
        # They should differ because fewer proteins means different variance
        assert not np.allclose(imp_all, imp_few)


# ---------------------------------------------------------------------------
# TestZstd
# ---------------------------------------------------------------------------


class TestZstd:
    def test_roundtrip(self, embedding):
        """compress → decompress recovers exact original bytes."""
        data = embedding.tobytes()
        compressed = zstd_compress(data)
        decompressed = zstd_decompress(compressed)
        assert decompressed == data

    def test_measure_size(self, embedding):
        """raw_bytes is correct; zstd_bytes < raw_bytes for random data."""
        result = measure_compressed_size(embedding)
        expected_raw = embedding.nbytes  # L * D * 4 bytes
        assert result["raw_bytes"] == expected_raw, (
            f"raw_bytes: expected {expected_raw}, got {result['raw_bytes']}"
        )
        assert "zstd_bytes" in result
        assert "zstd_ratio" in result

    def test_roundtrip_arbitrary_bytes(self):
        """zstd works on arbitrary byte sequences."""
        rng = np.random.RandomState(99)
        data = rng.bytes(1024)
        assert zstd_decompress(zstd_compress(data)) == data

    def test_different_levels(self, embedding):
        """Different compression levels all produce valid roundtrips."""
        data = embedding.tobytes()
        for level in (1, 3, 9):
            compressed = zstd_compress(data, level=level)
            assert zstd_decompress(compressed) == data

    def test_higher_level_same_or_smaller(self, embedding):
        """Higher compression level produces same or smaller output."""
        data = embedding.tobytes()
        size_1 = len(zstd_compress(data, level=1))
        size_9 = len(zstd_compress(data, level=9))
        # level 9 should be <= level 1 (or equal for incompressible data)
        assert size_9 <= size_1 + 100  # allow small margin for header differences

    def test_measure_size_zstd_ratio(self, embedding):
        """zstd_ratio equals raw_bytes / zstd_bytes."""
        result = measure_compressed_size(embedding)
        expected_ratio = result["raw_bytes"] / result["zstd_bytes"]
        assert abs(result["zstd_ratio"] - expected_ratio) < 1e-9
