"""Tests for One Embedding transforms, dataclass, I/O, and similarity.

Verifies mathematical correctness using known properties of DCT, Haar wavelets,
and spectral analysis. No external dependencies beyond pytest, numpy, h5py.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.embedding import OneEmbedding
from src.one_embedding.io import load_one_embeddings, save_one_embeddings
from src.one_embedding.similarity import protein_cosine_similarity
from src.one_embedding.transforms import (
    dct_summary,
    haar_full_coefficients,
    haar_summary,
    inverse_dct,
    inverse_haar,
    spectral_fingerprint,
    spectral_moments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)


@pytest.fixture
def matrix_8x4():
    """Small deterministic (L=8, d=4) matrix for unit tests."""
    return RNG.randn(8, 4).astype(np.float32)


@pytest.fixture
def matrix_16x8():
    """Slightly larger matrix (L=16, d=8), power-of-2 length."""
    return RNG.randn(16, 8).astype(np.float32)


@pytest.fixture
def matrix_13x4():
    """Non-power-of-2 length (L=13, d=4) to test padding paths."""
    return RNG.randn(13, 4).astype(np.float32)


# ===========================================================================
# 1. DCT K=1 is proportional to mean pool
# ===========================================================================


class TestDCTMeanPool:
    """DCT coefficient 0 with ortho normalization equals mean * sqrt(L)."""

    def test_dct_k1_equals_mean_times_sqrt_L(self, matrix_8x4):
        L, d = matrix_8x4.shape
        coeffs_k1 = dct_summary(matrix_8x4, K=1)  # shape (d,)
        expected = matrix_8x4.mean(axis=0) * np.sqrt(L)
        np.testing.assert_allclose(coeffs_k1, expected, rtol=1e-5)

    def test_dct_k1_proportional_to_mean(self, matrix_16x8):
        """Verify proportionality for a different matrix size."""
        L, d = matrix_16x8.shape
        coeffs_k1 = dct_summary(matrix_16x8, K=1)
        mean_pool = matrix_16x8.mean(axis=0)
        # Ratio should be sqrt(L) for every dimension
        ratios = coeffs_k1 / mean_pool
        np.testing.assert_allclose(ratios, np.sqrt(L), rtol=1e-5)


# ===========================================================================
# 2. DCT round-trip (lossless at K=L)
# ===========================================================================


class TestDCTRoundTrip:
    """inverse_dct(dct_summary(M, K=L), d, L) should recover M exactly."""

    def test_round_trip_power_of_2(self, matrix_8x4):
        L, d = matrix_8x4.shape
        coeffs = dct_summary(matrix_8x4, K=L)
        recovered = inverse_dct(coeffs, d, L)
        np.testing.assert_allclose(recovered, matrix_8x4, atol=1e-5)

    def test_round_trip_larger(self, matrix_16x8):
        L, d = matrix_16x8.shape
        coeffs = dct_summary(matrix_16x8, K=L)
        recovered = inverse_dct(coeffs, d, L)
        np.testing.assert_allclose(recovered, matrix_16x8, atol=1e-5)

    def test_round_trip_non_power_of_2(self, matrix_13x4):
        L, d = matrix_13x4.shape
        coeffs = dct_summary(matrix_13x4, K=L)
        recovered = inverse_dct(coeffs, d, L)
        np.testing.assert_allclose(recovered, matrix_13x4, atol=1e-5)


# ===========================================================================
# 3. DCT truncation: K<L produces low-pass filtered version
# ===========================================================================


class TestDCTTruncation:
    """K=1 inverse should be constant along the sequence (= mean at every position)."""

    def test_k1_inverse_is_constant_per_position(self, matrix_8x4):
        L, d = matrix_8x4.shape
        coeffs_k1 = dct_summary(matrix_8x4, K=1)
        reconstructed = inverse_dct(coeffs_k1, d, L)

        # Every row should be identical (constant signal)
        for i in range(1, L):
            np.testing.assert_allclose(
                reconstructed[i], reconstructed[0], atol=1e-6,
                err_msg=f"Row {i} differs from row 0 in K=1 reconstruction",
            )

    def test_k1_inverse_value_equals_mean(self, matrix_8x4):
        """The constant value should be the column-wise mean of the original."""
        L, d = matrix_8x4.shape
        coeffs_k1 = dct_summary(matrix_8x4, K=1)
        reconstructed = inverse_dct(coeffs_k1, d, L)
        np.testing.assert_allclose(reconstructed[0], matrix_8x4.mean(axis=0), atol=1e-5)

    def test_increasing_K_reduces_error(self, matrix_16x8):
        """Reconstruction error should monotonically decrease as K increases."""
        L, d = matrix_16x8.shape
        prev_error = float("inf")
        for K in [1, 2, 4, 8, 16]:
            coeffs = dct_summary(matrix_16x8, K=K)
            recovered = inverse_dct(coeffs, d, L)
            error = np.mean((recovered - matrix_16x8) ** 2)
            assert error <= prev_error + 1e-7, (
                f"Error increased from {prev_error:.6f} to {error:.6f} at K={K}"
            )
            prev_error = error
        # At K=L the error should be essentially zero
        np.testing.assert_allclose(prev_error, 0.0, atol=1e-9)


# ===========================================================================
# 4. Haar round-trip
# ===========================================================================


class TestHaarRoundTrip:
    """inverse_haar(*haar_full_coefficients(M, levels), target_len=L) recovers M."""

    def test_round_trip_power_of_2(self, matrix_8x4):
        L, d = matrix_8x4.shape
        levels = 3  # log2(8) = 3 full levels
        approx, details = haar_full_coefficients(matrix_8x4, levels=levels)
        recovered = inverse_haar(approx, details, target_len=L)
        np.testing.assert_allclose(recovered, matrix_8x4, atol=1e-5)

    def test_round_trip_full_levels(self, matrix_16x8):
        L, d = matrix_16x8.shape
        levels = 4  # log2(16) = 4
        approx, details = haar_full_coefficients(matrix_16x8, levels=levels)
        recovered = inverse_haar(approx, details, target_len=L)
        np.testing.assert_allclose(recovered, matrix_16x8, atol=1e-5)

    def test_round_trip_non_power_of_2(self, matrix_13x4):
        """Non-power-of-2 should pad internally; round-trip truncates back."""
        L, d = matrix_13x4.shape
        levels = 4  # padded to 16
        approx, details = haar_full_coefficients(matrix_13x4, levels=levels)
        recovered = inverse_haar(approx, details, target_len=L)
        # Only the first L rows should match (padding zeros are discarded)
        np.testing.assert_allclose(recovered, matrix_13x4, atol=1e-5)

    def test_partial_levels_round_trip(self, matrix_16x8):
        """Fewer levels than full decomposition should still round-trip."""
        L, d = matrix_16x8.shape
        for levels in [1, 2, 3]:
            approx, details = haar_full_coefficients(matrix_16x8, levels=levels)
            recovered = inverse_haar(approx, details, target_len=L)
            np.testing.assert_allclose(
                recovered, matrix_16x8, atol=1e-5,
                err_msg=f"Haar round-trip failed at levels={levels}",
            )


# ===========================================================================
# 5. Haar level 0: summary should be the mean
# ===========================================================================


class TestHaarLevel0:
    """At 0 levels, Haar summary should just be the mean (no detail coefficients)."""

    def test_level0_is_mean(self, matrix_8x4):
        summary = haar_summary(matrix_8x4, levels=0)
        expected_mean = matrix_8x4.mean(axis=0)
        # With 0 levels no decomposition occurs; approx is padded input, pooled = mean
        # The summary has shape ((0+1)*d,) = (d,)
        assert summary.shape == (matrix_8x4.shape[1],)
        # The padding extends 8 -> 8 (already power of 2), so mean of padded = mean of original
        np.testing.assert_allclose(summary, expected_mean, atol=1e-5)

    def test_level0_full_coefficients(self, matrix_8x4):
        """haar_full_coefficients with 0 levels returns the padded input as approx."""
        approx, details = haar_full_coefficients(matrix_8x4, levels=0)
        assert len(details) == 0
        # approx should be the (possibly padded) input
        np.testing.assert_allclose(approx[:matrix_8x4.shape[0]], matrix_8x4, atol=1e-6)


# ===========================================================================
# 6. Spectral fingerprint: Parseval's theorem
# ===========================================================================


class TestParseval:
    """Total PSD energy should equal total signal energy: sum |DCT|^2 = sum x^2."""

    def test_parseval_single_band(self, matrix_8x4):
        """When n_bands=1, the single band sums all PSD energy."""
        fp = spectral_fingerprint(matrix_8x4, n_bands=1)
        signal_energy = np.sum(matrix_8x4 ** 2)
        psd_energy = np.sum(fp)
        np.testing.assert_allclose(psd_energy, signal_energy, rtol=1e-5)

    def test_parseval_multiple_bands(self, matrix_16x8):
        """Sum across all bands should equal signal energy regardless of n_bands."""
        for n_bands in [1, 2, 4, 8, 16]:
            fp = spectral_fingerprint(matrix_16x8, n_bands=n_bands)
            signal_energy = np.sum(matrix_16x8 ** 2)
            psd_energy = np.sum(fp)
            np.testing.assert_allclose(
                psd_energy, signal_energy, rtol=1e-5,
                err_msg=f"Parseval violated at n_bands={n_bands}",
            )

    def test_parseval_non_power_of_2(self, matrix_13x4):
        """Parseval should hold even for non-power-of-2 lengths."""
        fp = spectral_fingerprint(matrix_13x4, n_bands=4)
        signal_energy = np.sum(matrix_13x4 ** 2)
        psd_energy = np.sum(fp)
        np.testing.assert_allclose(psd_energy, signal_energy, rtol=1e-5)

    def test_band_energies_non_negative(self, matrix_8x4):
        """PSD is |DCT|^2, so all band energies must be non-negative."""
        fp = spectral_fingerprint(matrix_8x4, n_bands=4)
        assert np.all(fp >= 0), "Spectral fingerprint contains negative energy"


# ===========================================================================
# 7. Spectral moments: centroid of white-ish signal near L/2
# ===========================================================================


class TestSpectralMoments:
    """Spectral centroid of uniform random signal should be near center frequency."""

    def test_centroid_near_center(self):
        """For large uniform random data, centroid should approach L/2."""
        rng = np.random.RandomState(123)
        L, d = 256, 16
        matrix = rng.randn(L, d).astype(np.float32)
        moments = spectral_moments(matrix, n_moments=1)  # just centroid, shape (d,)
        centroids = moments  # shape (d,)
        # Expected centroid for flat PSD: (L-1)/2
        expected = (L - 1) / 2.0
        mean_centroid = centroids.mean()
        # Allow generous tolerance since finite random signal is not perfectly flat
        assert abs(mean_centroid - expected) < L * 0.1, (
            f"Mean centroid {mean_centroid:.1f} too far from expected {expected:.1f}"
        )

    def test_moments_output_shape(self, matrix_8x4):
        """Verify output shape for each n_moments value."""
        d = matrix_8x4.shape[1]
        for n_mom in [1, 2, 3, 4]:
            result = spectral_moments(matrix_8x4, n_moments=n_mom)
            assert result.shape == (n_mom * d,), (
                f"Expected shape ({n_mom * d},), got {result.shape}"
            )

    def test_dc_dominated_signal_centroid_near_zero(self):
        """A constant signal has all energy at DC (freq 0), centroid should be ~0."""
        L, d = 32, 4
        matrix = np.ones((L, d), dtype=np.float32) * 5.0
        moments = spectral_moments(matrix, n_moments=1)
        # Centroid should be very close to 0 (all energy in DC component)
        np.testing.assert_allclose(moments, 0.0, atol=1e-5)


# ===========================================================================
# 8. OneEmbedding dataclass
# ===========================================================================


class TestOneEmbedding:
    """Test .summary, .residues, .flat properties and from_compressed() factory."""

    def test_from_compressed_default_mean(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("P12345", "esm2", matrix_8x4)
        np.testing.assert_allclose(emb.summary, matrix_8x4.mean(axis=0), atol=1e-6)
        assert emb.transform == "mean"
        assert emb.protein_id == "P12345"
        assert emb.plm == "esm2"
        assert emb.latent_dim == 4
        assert emb.seq_len == 8

    def test_from_compressed_custom_summary(self, matrix_8x4):
        custom_summary = np.ones(4, dtype=np.float32)
        emb = OneEmbedding.from_compressed(
            "P99999", "prot_t5_xl", matrix_8x4,
            transform="dct_k8", summary=custom_summary,
        )
        np.testing.assert_allclose(emb.summary, custom_summary, atol=1e-7)
        assert emb.transform == "dct_k8"

    def test_residues_property(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("test", "esm2", matrix_8x4)
        np.testing.assert_array_equal(emb.residues, matrix_8x4.astype(np.float32))

    def test_flat_property(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("test", "esm2", matrix_8x4)
        flat = emb.flat
        d = matrix_8x4.shape[1]
        # flat = [summary(d) | residues.ravel(L*d)]
        expected_len = d + matrix_8x4.size
        assert flat.shape == (expected_len,)
        # First d elements should be the summary
        np.testing.assert_allclose(flat[:d], emb.summary, atol=1e-7)
        # Remaining should be the flattened residues
        np.testing.assert_allclose(flat[d:], matrix_8x4.astype(np.float32).ravel(), atol=1e-7)

    def test_summary_dim(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("test", "esm2", matrix_8x4)
        assert emb.summary_dim == matrix_8x4.shape[1]

    def test_float32_casting(self):
        """Input float64 should be stored as float32."""
        mat = np.random.randn(5, 3).astype(np.float64)
        emb = OneEmbedding.from_compressed("test", "esm2", mat)
        assert emb.residues.dtype == np.float32
        assert emb.summary.dtype == np.float32


# ===========================================================================
# 9. H5 round-trip
# ===========================================================================


class TestH5RoundTrip:
    """Save and load OneEmbedding, verify all fields match."""

    def test_single_protein_round_trip(self, matrix_8x4):
        emb = OneEmbedding.from_compressed(
            "protein_A", "prot_t5_xl", matrix_8x4, transform="dct_k4",
            summary=dct_summary(matrix_8x4, K=4),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            save_one_embeddings({"protein_A": emb}, h5_path)
            loaded = load_one_embeddings(h5_path)

        assert "protein_A" in loaded
        loaded_emb = loaded["protein_A"]
        assert loaded_emb.protein_id == "protein_A"
        assert loaded_emb.plm == "prot_t5_xl"
        assert loaded_emb.transform == "dct_k4"
        assert loaded_emb.latent_dim == emb.latent_dim
        assert loaded_emb.seq_len == emb.seq_len
        np.testing.assert_allclose(loaded_emb.summary, emb.summary, atol=1e-7)
        np.testing.assert_allclose(loaded_emb.residues, emb.residues, atol=1e-7)

    def test_multiple_proteins_round_trip(self, matrix_8x4, matrix_16x8):
        emb_a = OneEmbedding.from_compressed(
            "prot_A", "esm2", matrix_8x4[:, :4] if matrix_8x4.shape[1] == 4 else matrix_8x4,
        )
        # Use same latent_dim for both since they share one H5 file
        emb_b = OneEmbedding.from_compressed(
            "prot_B", "esm2", matrix_16x8[:, :4],
        )
        collection = {"prot_A": emb_a, "prot_B": emb_b}

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "multi.h5"
            save_one_embeddings(collection, h5_path)
            loaded = load_one_embeddings(h5_path)

        assert len(loaded) == 2
        for pid in ["prot_A", "prot_B"]:
            np.testing.assert_allclose(
                loaded[pid].summary, collection[pid].summary, atol=1e-7,
            )
            np.testing.assert_allclose(
                loaded[pid].residues, collection[pid].residues, atol=1e-7,
            )

    def test_h5_file_attributes(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("test", "prot_t5_xl", matrix_8x4)
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "attrs.h5"
            save_one_embeddings({"test": emb}, h5_path)
            with h5py.File(h5_path, "r") as f:
                assert f.attrs["format"] == "one_embedding"
                assert f.attrs["plm"] == "prot_t5_xl"
                assert f.attrs["latent_dim"] == 4
                assert f.attrs["n_proteins"] == 1


# ===========================================================================
# 10. Similarity functions
# ===========================================================================


class TestSimilarity:
    """protein_cosine_similarity for identical embeddings should be 1.0."""

    def test_identical_embeddings_similarity_one(self, matrix_8x4):
        emb = OneEmbedding.from_compressed("test", "esm2", matrix_8x4)
        sim = protein_cosine_similarity(emb, emb)
        np.testing.assert_allclose(sim, 1.0, atol=1e-6)

    def test_negated_embeddings_similarity_minus_one(self, matrix_8x4):
        emb_a = OneEmbedding.from_compressed("a", "esm2", matrix_8x4)
        emb_b = OneEmbedding.from_compressed("b", "esm2", -matrix_8x4)
        sim = protein_cosine_similarity(emb_a, emb_b)
        np.testing.assert_allclose(sim, -1.0, atol=1e-6)

    def test_orthogonal_summaries_similarity_zero(self):
        """Two proteins with orthogonal summary vectors should have similarity ~0."""
        d = 4
        mat_a = np.zeros((2, d), dtype=np.float32)
        mat_a[:, 0] = 1.0  # summary will point along dim 0
        mat_b = np.zeros((2, d), dtype=np.float32)
        mat_b[:, 1] = 1.0  # summary will point along dim 1
        emb_a = OneEmbedding.from_compressed("a", "esm2", mat_a)
        emb_b = OneEmbedding.from_compressed("b", "esm2", mat_b)
        sim = protein_cosine_similarity(emb_a, emb_b)
        np.testing.assert_allclose(sim, 0.0, atol=1e-6)

    def test_scaled_embeddings_same_similarity(self, matrix_8x4):
        """Cosine similarity is scale-invariant."""
        emb_a = OneEmbedding.from_compressed("a", "esm2", matrix_8x4)
        emb_b = OneEmbedding.from_compressed("b", "esm2", matrix_8x4 * 100.0)
        sim = protein_cosine_similarity(emb_a, emb_b)
        np.testing.assert_allclose(sim, 1.0, atol=1e-6)

    def test_similarity_bounded(self, matrix_8x4, matrix_16x8):
        """Cosine similarity should always be in [-1, 1]."""
        emb_a = OneEmbedding.from_compressed("a", "esm2", matrix_8x4)
        emb_b = OneEmbedding.from_compressed("b", "esm2", matrix_16x8[:, :4])
        sim = protein_cosine_similarity(emb_a, emb_b)
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
