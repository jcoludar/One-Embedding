"""Tests for topological and OT-based transforms.

Verifies: SWD self-distance, symmetry, variable-length support; SimHash
shape and determinism; AA-residual roundtrip and magnitude reduction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.topological import (
    AA_ORDER,
    aa_residual_decode,
    aa_residual_encode,
    compute_aa_centroids,
    simhash_decode_approx,
    simhash_encode,
    sliced_wasserstein_distance,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 64).astype(np.float32)


# ── TestSlicedWasserstein ────────────────────────────────────────────────────


class TestSlicedWasserstein:
    def test_self_distance_zero(self, embedding):
        """SWD(X, X) must be effectively zero."""
        d = sliced_wasserstein_distance(embedding, embedding, n_projections=50, seed=42)
        assert d < 1e-6, f"Self-distance should be 0, got {d}"

    def test_symmetric(self, embedding):
        """SWD(X, Y) == SWD(Y, X)."""
        rng = np.random.RandomState(7)
        Y = rng.randn(40, 64).astype(np.float32)
        d_xy = sliced_wasserstein_distance(embedding, Y, n_projections=50, seed=42)
        d_yx = sliced_wasserstein_distance(Y, embedding, n_projections=50, seed=42)
        assert abs(d_xy - d_yx) < 1e-5, f"SWD not symmetric: {d_xy} vs {d_yx}"

    def test_different_lengths(self):
        """SWD should work for point clouds of different sizes."""
        rng = np.random.RandomState(99)
        X = rng.randn(30, 32).astype(np.float32)
        Y = rng.randn(50, 32).astype(np.float32)
        d = sliced_wasserstein_distance(X, Y, n_projections=20, seed=0)
        assert np.isfinite(d), f"SWD returned non-finite value: {d}"
        assert d >= 0.0

    def test_non_negative(self, embedding):
        rng = np.random.RandomState(13)
        Y = rng.randn(50, 64).astype(np.float32)
        d = sliced_wasserstein_distance(embedding, Y, n_projections=20, seed=0)
        assert d >= 0.0

    def test_different_distributions_positive(self):
        """Two clearly separated distributions should have positive SWD."""
        rng = np.random.RandomState(0)
        X = rng.randn(50, 16).astype(np.float32)
        Y = rng.randn(50, 16).astype(np.float32) + 10.0  # far away
        d = sliced_wasserstein_distance(X, Y, n_projections=50, seed=42)
        assert d > 1.0, f"Expected large SWD for separated clouds, got {d}"


# ── TestSimHash ───────────────────────────────────────────────────────────────


class TestSimHash:
    def test_encode_shape(self, embedding):
        """bits array should be (L, n_bits // 8)."""
        n_bits = 256
        compressed = simhash_encode(embedding, n_bits=n_bits, seed=42)
        L = embedding.shape[0]
        assert compressed["bits"].shape == (L, n_bits // 8)

    def test_encode_dtype_uint8(self, embedding):
        compressed = simhash_encode(embedding, n_bits=256, seed=42)
        assert compressed["bits"].dtype == np.uint8

    def test_decode_shape(self, embedding):
        """Decoded approximation should have the original shape."""
        compressed = simhash_encode(embedding, n_bits=256, seed=42)
        rec = simhash_decode_approx(compressed)
        assert rec.shape == embedding.shape

    def test_decode_dtype_float32(self, embedding):
        compressed = simhash_encode(embedding, n_bits=256, seed=42)
        rec = simhash_decode_approx(compressed)
        assert rec.dtype == np.float32

    def test_deterministic(self, embedding):
        """Same input + same seed → identical packed bits."""
        c1 = simhash_encode(embedding, n_bits=128, seed=42)
        c2 = simhash_encode(embedding, n_bits=128, seed=42)
        np.testing.assert_array_equal(c1["bits"], c2["bits"])

    def test_different_seeds_different_bits(self, embedding):
        """Different seeds produce different bit patterns."""
        c1 = simhash_encode(embedding, n_bits=128, seed=42)
        c2 = simhash_encode(embedding, n_bits=128, seed=99)
        assert not np.array_equal(c1["bits"], c2["bits"])

    def test_metadata_stored(self, embedding):
        compressed = simhash_encode(embedding, n_bits=128, seed=7)
        assert compressed["n_bits"] == 128
        assert compressed["seed"] == 7
        assert compressed["original_shape"] == embedding.shape

    def test_invalid_n_bits_raises(self, embedding):
        with pytest.raises(ValueError):
            simhash_encode(embedding, n_bits=100, seed=42)  # not divisible by 8


# ── TestAAResidual ────────────────────────────────────────────────────────────


class TestAAResidual:
    @pytest.fixture
    def clustered_corpus(self):
        """Corpus where each AA has a distinctive embedding region."""
        rng = np.random.RandomState(42)
        n_aa = len(AA_ORDER)
        D = 32

        # Each AA gets a random centroid offset
        offsets = rng.randn(n_aa, D).astype(np.float32) * 5.0

        embeddings = {}
        sequences = {}
        for pid in range(50):
            L = 40
            seq = "".join(rng.choice(list(AA_ORDER), size=L))
            mat = np.zeros((L, D), dtype=np.float32)
            for pos, aa in enumerate(seq):
                aa_idx = AA_ORDER.index(aa)
                mat[pos] = offsets[aa_idx] + rng.randn(D).astype(np.float32) * 0.1
            embeddings[f"p{pid}"] = mat
            sequences[f"p{pid}"] = seq

        return embeddings, sequences, offsets

    def test_roundtrip(self, embedding):
        """encode then decode must exactly recover the original."""
        rng = np.random.RandomState(0)
        # Make a tiny corpus and centroid matrix of correct D
        D = embedding.shape[1]
        centroids = rng.randn(20, D).astype(np.float32)
        seq = "".join(rng.choice(list(AA_ORDER), size=embedding.shape[0]))

        residual = aa_residual_encode(embedding, seq, centroids)
        recovered = aa_residual_decode(residual, seq, centroids)
        np.testing.assert_allclose(recovered, embedding, atol=1e-5)

    def test_residuals_smaller(self, clustered_corpus):
        """For clustered data the residuals should have smaller mean |magnitude|."""
        embeddings, sequences, _ = clustered_corpus
        centroids = compute_aa_centroids(embeddings, sequences)

        pid = "p0"
        mat = embeddings[pid]
        seq = sequences[pid]

        residuals = aa_residual_encode(mat, seq, centroids)
        orig_norm = float(np.mean(np.abs(mat)))
        res_norm = float(np.mean(np.abs(residuals)))

        assert res_norm < orig_norm, (
            f"Residuals ({res_norm:.4f}) not smaller than originals ({orig_norm:.4f})"
        )

    def test_centroid_computation(self, clustered_corpus):
        """compute_aa_centroids returns (20, D) float32."""
        embeddings, sequences, _ = clustered_corpus
        centroids = compute_aa_centroids(embeddings, sequences)
        D = list(embeddings.values())[0].shape[1]
        assert centroids.shape == (20, D)
        assert centroids.dtype == np.float32

    def test_centroid_max_proteins(self, clustered_corpus):
        """compute_aa_centroids respects max_proteins limit."""
        embeddings, sequences, _ = clustered_corpus
        centroids = compute_aa_centroids(embeddings, sequences, max_proteins=10)
        D = list(embeddings.values())[0].shape[1]
        assert centroids.shape == (20, D)

    def test_roundtrip_non_standard_aa(self, embedding):
        """Non-standard AAs ('X') are passed through unchanged."""
        D = embedding.shape[1]
        rng = np.random.RandomState(1)
        centroids = rng.randn(20, D).astype(np.float32)
        # Sequence with some 'X' (unknown) AAs
        seq = "X" * embedding.shape[0]

        residual = aa_residual_encode(embedding, seq, centroids)
        recovered = aa_residual_decode(residual, seq, centroids)
        # No centroid subtraction for 'X', so residual == original
        np.testing.assert_array_equal(residual, embedding)
        np.testing.assert_allclose(recovered, embedding, atol=1e-6)
