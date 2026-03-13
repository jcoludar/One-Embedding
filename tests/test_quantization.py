"""Tests for quantization codecs: int8, int4, binary, PQ, RVQ."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.quantization import (
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
    quantize_binary,
    dequantize_binary,
    pq_fit,
    pq_encode,
    pq_decode,
    rvq_fit,
    rvq_encode,
    rvq_decode,
    compressed_size_bytes,
)


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30, 128).astype(np.float32) for i in range(50)}


# ---------------------------------------------------------------------------
# TestInt8
# ---------------------------------------------------------------------------

class TestInt8:
    def test_roundtrip_shape(self, embedding):
        compressed = quantize_int8(embedding)
        rec = dequantize_int8(compressed)
        assert rec.shape == embedding.shape

    def test_roundtrip_quality(self, embedding):
        compressed = quantize_int8(embedding)
        rec = dequantize_int8(compressed)
        np.testing.assert_allclose(rec, embedding, atol=0.05)

    def test_data_is_uint8(self, embedding):
        compressed = quantize_int8(embedding)
        assert compressed["data"].dtype == np.uint8

    def test_size_reduction(self, embedding):
        """Quantized data should use L*D bytes (vs 4*L*D for float32)."""
        L, D = embedding.shape
        compressed = quantize_int8(embedding)
        assert compressed["data"].nbytes == L * D

    def test_zero_range_channel(self):
        """Constant channels (zero range) should not cause division by zero."""
        matrix = np.random.RandomState(0).randn(20, 64).astype(np.float32)
        matrix[:, 5] = 3.14  # constant channel
        compressed = quantize_int8(matrix)
        rec = dequantize_int8(compressed)
        assert np.isfinite(rec).all()

    def test_dtype_field(self, embedding):
        compressed = quantize_int8(embedding)
        assert compressed["dtype"] == "int8"

    def test_output_dtype_float32(self, embedding):
        compressed = quantize_int8(embedding)
        rec = dequantize_int8(compressed)
        assert rec.dtype == np.float32


# ---------------------------------------------------------------------------
# TestInt4
# ---------------------------------------------------------------------------

class TestInt4:
    def test_roundtrip_shape(self, embedding):
        compressed = quantize_int4(embedding)
        rec = dequantize_int4(compressed)
        assert rec.shape == embedding.shape

    def test_packed_size(self, embedding):
        """Packed data should have shape (L, D//2)."""
        L, D = embedding.shape
        compressed = quantize_int4(embedding)
        assert compressed["data"].shape == (L, D // 2)

    def test_roundtrip_reasonable(self, embedding):
        """Cosine similarity between original and reconstructed should be > 0.9."""
        compressed = quantize_int4(embedding)
        rec = dequantize_int4(compressed)
        # Compute per-residue cosine similarity and take mean
        sim = cosine_similarity(embedding, rec)
        diag_sim = np.diag(sim)
        assert diag_sim.mean() > 0.9

    def test_data_is_uint8(self, embedding):
        compressed = quantize_int4(embedding)
        assert compressed["data"].dtype == np.uint8

    def test_dtype_field(self, embedding):
        compressed = quantize_int4(embedding)
        assert compressed["dtype"] == "int4"

    def test_odd_D(self):
        """Odd D should be handled by padding."""
        rng = np.random.RandomState(7)
        matrix = rng.randn(20, 65).astype(np.float32)
        compressed = quantize_int4(matrix)
        rec = dequantize_int4(compressed)
        assert rec.shape == (20, 65)

    def test_zero_range_channel(self):
        matrix = np.random.RandomState(1).randn(10, 32).astype(np.float32)
        matrix[:, 3] = -2.0
        compressed = quantize_int4(matrix)
        rec = dequantize_int4(compressed)
        assert np.isfinite(rec).all()


# ---------------------------------------------------------------------------
# TestBinary
# ---------------------------------------------------------------------------

class TestBinary:
    def test_roundtrip_shape(self, embedding):
        compressed = quantize_binary(embedding)
        rec = dequantize_binary(compressed)
        assert rec.shape == embedding.shape

    def test_packed_bits_shape(self, embedding):
        """Packed bits should have shape (L, D//8)."""
        L, D = embedding.shape
        compressed = quantize_binary(embedding)
        assert compressed["bits"].shape == (L, D // 8)

    def test_signs_preserved(self, embedding):
        """Sign agreement between original (centered) and reconstructed should be > 0.99."""
        compressed = quantize_binary(embedding)
        rec = dequantize_binary(compressed)
        means = compressed["means"]
        # Compare signs relative to per-channel means
        orig_signs = np.sign(embedding - means[np.newaxis, :])
        rec_signs = np.sign(rec - means[np.newaxis, :])
        agreement = (orig_signs == rec_signs).mean()
        assert agreement > 0.99

    def test_data_is_uint8(self, embedding):
        compressed = quantize_binary(embedding)
        assert compressed["bits"].dtype == np.uint8

    def test_dtype_field(self, embedding):
        compressed = quantize_binary(embedding)
        assert compressed["dtype"] == "binary"

    def test_output_dtype_float32(self, embedding):
        compressed = quantize_binary(embedding)
        rec = dequantize_binary(compressed)
        assert rec.dtype == np.float32

    def test_d_not_multiple_of_8(self):
        """D not a multiple of 8 should be handled by padding."""
        rng = np.random.RandomState(3)
        matrix = rng.randn(15, 100).astype(np.float32)  # 100 = 12*8 + 4
        compressed = quantize_binary(matrix)
        rec = dequantize_binary(compressed)
        assert rec.shape == (15, 100)

    def test_compressed_size_bytes(self, embedding):
        L, D = embedding.shape
        compressed = quantize_binary(embedding)
        size = compressed_size_bytes(compressed)
        # ceil(128/8) = 16 bytes per row
        import math
        expected = L * math.ceil(D / 8)
        assert size == expected


# ---------------------------------------------------------------------------
# TestPQ
# ---------------------------------------------------------------------------

class TestPQ:
    def test_fit_codebook_shape(self, corpus):
        M, n_centroids = 16, 32
        pq_model = pq_fit(corpus, M=M, n_centroids=n_centroids)
        sub_dim = 128 // M
        assert pq_model["codebook"].shape == (M, n_centroids, sub_dim)

    def test_encode_decode_shape(self, corpus, embedding):
        M, n_centroids = 16, 32
        pq_model = pq_fit(corpus, M=M, n_centroids=n_centroids)

        codes = pq_encode(embedding, pq_model)
        assert codes.shape == (50, M)
        assert codes.dtype == np.uint8

        decoded = pq_decode(codes, pq_model)
        assert decoded.shape == (50, 128)
        assert decoded.dtype == np.float32

    def test_encode_deterministic(self, corpus, embedding):
        M, n_centroids = 16, 32
        pq_model = pq_fit(corpus, M=M, n_centroids=n_centroids)

        codes1 = pq_encode(embedding, pq_model)
        codes2 = pq_encode(embedding, pq_model)
        np.testing.assert_array_equal(codes1, codes2)

    def test_fit_sub_dim_stored(self, corpus):
        pq_model = pq_fit(corpus, M=16, n_centroids=32)
        assert pq_model["sub_dim"] == 128 // 16
        assert pq_model["D"] == 128
        assert pq_model["M"] == 16
        assert pq_model["n_centroids"] == 32

    def test_centroids_in_range(self, corpus):
        pq_model = pq_fit(corpus, M=16, n_centroids=32)
        single_embed = next(iter(corpus.values()))
        codes = pq_encode(single_embed, pq_model)
        assert codes.max() < 32  # n_centroids
        assert codes.min() >= 0

    def test_decode_output_finite(self, corpus, embedding):
        pq_model = pq_fit(corpus, M=16, n_centroids=32)
        codes = pq_encode(embedding, pq_model)
        decoded = pq_decode(codes, pq_model)
        assert np.isfinite(decoded).all()


# ---------------------------------------------------------------------------
# TestRVQ
# ---------------------------------------------------------------------------

class TestRVQ:
    def test_fit_codebooks(self, corpus):
        n_levels, n_centroids = 3, 32
        rvq_model = rvq_fit(corpus, n_levels=n_levels, n_centroids=n_centroids)
        assert len(rvq_model["codebooks"]) == n_levels
        for cb in rvq_model["codebooks"]:
            assert cb.shape == (n_centroids, 128)

    def test_encode_decode_shape(self, corpus, embedding):
        n_levels, n_centroids = 3, 32
        rvq_model = rvq_fit(corpus, n_levels=n_levels, n_centroids=n_centroids)

        codes = rvq_encode(embedding, rvq_model)
        assert codes.shape == (50, n_levels)
        assert codes.dtype == np.uint8

        decoded = rvq_decode(codes, rvq_model)
        assert decoded.shape == (50, 128)
        assert decoded.dtype == np.float32

    def test_residuals_decrease(self, corpus, embedding):
        """MSE should decrease as we add more quantization levels."""
        n_levels, n_centroids = 3, 32
        rvq_model = rvq_fit(corpus, n_levels=n_levels, n_centroids=n_centroids)

        codes = rvq_encode(embedding, rvq_model)

        mse_values = []
        for k in range(1, n_levels + 1):
            # Decode using only first k levels
            partial_model = {
                "codebooks": rvq_model["codebooks"][:k],
                "n_levels": k,
                "n_centroids": n_centroids,
                "D": rvq_model["D"],
            }
            decoded_k = rvq_decode(codes[:, :k], partial_model)
            mse = np.mean((embedding - decoded_k) ** 2)
            mse_values.append(mse)

        # Each additional level should reduce MSE (strictly)
        for i in range(len(mse_values) - 1):
            assert mse_values[i + 1] < mse_values[i], (
                f"MSE did not decrease at level {i+1}: {mse_values}"
            )

    def test_fit_metadata(self, corpus):
        rvq_model = rvq_fit(corpus, n_levels=3, n_centroids=32)
        assert rvq_model["n_levels"] == 3
        assert rvq_model["n_centroids"] == 32
        assert rvq_model["D"] == 128

    def test_encode_deterministic(self, corpus, embedding):
        rvq_model = rvq_fit(corpus, n_levels=3, n_centroids=32)
        codes1 = rvq_encode(embedding, rvq_model)
        codes2 = rvq_encode(embedding, rvq_model)
        np.testing.assert_array_equal(codes1, codes2)

    def test_decode_output_finite(self, corpus, embedding):
        rvq_model = rvq_fit(corpus, n_levels=3, n_centroids=32)
        codes = rvq_encode(embedding, rvq_model)
        decoded = rvq_decode(codes, rvq_model)
        assert np.isfinite(decoded).all()


# ---------------------------------------------------------------------------
# compressed_size_bytes
# ---------------------------------------------------------------------------

class TestCompressedSizeBytes:
    def test_int8_size(self, embedding):
        L, D = embedding.shape
        compressed = quantize_int8(embedding)
        assert compressed_size_bytes(compressed) == L * D

    def test_int4_size(self, embedding):
        L, D = embedding.shape
        compressed = quantize_int4(embedding)
        # ceil(D/2) packed bytes per row
        import math
        assert compressed_size_bytes(compressed) == L * math.ceil(D / 2)

    def test_binary_size(self, embedding):
        L, D = embedding.shape
        import math
        compressed = quantize_binary(embedding)
        assert compressed_size_bytes(compressed) == L * math.ceil(D / 8)

    def test_int8_smaller_than_float32(self, embedding):
        L, D = embedding.shape
        compressed = quantize_int8(embedding)
        float32_bytes = L * D * 4
        assert compressed_size_bytes(compressed) < float32_bytes

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            compressed_size_bytes({"dtype": "bogus"})
