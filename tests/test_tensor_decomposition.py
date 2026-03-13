"""Tests for tensor train decomposition and NMF encoding.

Verifies: roundtrip shape, compression quality monotonicity, storage
behaviour, and NMF fit / encode / decode pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.tensor_decomposition import (
    nmf_decode,
    nmf_encode,
    nmf_fit,
    tt_decompose,
    tt_reconstruct,
    tt_storage_bytes,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30, 128).astype(np.float32) for i in range(20)}


# ── TestTensorTrain ───────────────────────────────────────────────────────────


class TestTensorTrain:
    def test_roundtrip_shape(self, embedding):
        """Reconstructed matrix has the same shape as input."""
        compressed = tt_decompose(embedding, bond_dim=16)
        rec = tt_reconstruct(compressed)
        assert rec.shape == embedding.shape

    def test_reconstruct_dtype_float32(self, embedding):
        compressed = tt_decompose(embedding, bond_dim=16)
        rec = tt_reconstruct(compressed)
        assert rec.dtype == np.float32

    def test_higher_bond_dim_better(self, embedding):
        """MSE should decrease (or stay equal) as bond_dim increases."""
        mses = []
        for bd in [4, 8, 16, 32]:
            compressed = tt_decompose(embedding, bond_dim=bd)
            rec = tt_reconstruct(compressed)
            mse = float(np.mean((rec - embedding) ** 2))
            mses.append(mse)

        # Each step should be non-increasing
        for prev, curr in zip(mses, mses[1:]):
            assert curr <= prev + 1e-6, (
                f"MSE did not decrease: bond_dims gave MSEs {mses}"
            )

    def test_storage_decreases_with_bond_dim(self, embedding):
        """Smaller bond_dim → less storage bytes."""
        sizes = []
        for bd in [4, 8, 16, 32]:
            compressed = tt_decompose(embedding, bond_dim=bd)
            sizes.append(tt_storage_bytes(compressed))

        for prev, curr in zip(sizes, sizes[1:]):
            assert curr >= prev - 1, (
                f"Storage did not increase with bond_dim: {sizes}"
            )

    def test_full_bond_dim_near_lossless(self, embedding):
        """bond_dim = D (=128) should reproduce the matrix exactly."""
        L, D = embedding.shape
        compressed = tt_decompose(embedding, bond_dim=D)
        rec = tt_reconstruct(compressed)
        np.testing.assert_allclose(rec, embedding, atol=1e-4)

    def test_original_shape_stored(self, embedding):
        compressed = tt_decompose(embedding, bond_dim=8)
        assert compressed["original_shape"] == embedding.shape

    def test_bond_dim_stored(self, embedding):
        compressed = tt_decompose(embedding, bond_dim=8)
        assert compressed["bond_dim"] == 8

    def test_storage_bytes_positive(self, embedding):
        compressed = tt_decompose(embedding, bond_dim=8)
        assert tt_storage_bytes(compressed) > 0


# ── TestNMF ──────────────────────────────────────────────────────────────────


class TestNMF:
    def test_fit_shape(self, corpus):
        """NMF basis H should be (k, D)."""
        model = nmf_fit(corpus, k=16)
        assert model["H"].shape == (16, 128)

    def test_fit_stores_shift(self, corpus):
        model = nmf_fit(corpus, k=16)
        assert model["shift"].shape == (128,)

    def test_encode_decode_shape(self, embedding, corpus):
        """W is (L, k) and reconstruction is (L, D)."""
        model = nmf_fit(corpus, k=16)
        W = nmf_encode(embedding, model)
        rec = nmf_decode(W, model)
        assert W.shape == (50, 16)
        assert rec.shape == (50, 128)

    def test_encode_decode_dtype_float32(self, embedding, corpus):
        model = nmf_fit(corpus, k=16)
        W = nmf_encode(embedding, model)
        rec = nmf_decode(W, model)
        assert W.dtype == np.float32
        assert rec.dtype == np.float32

    def test_non_negative_weights(self, embedding, corpus):
        """NMF weights should be non-negative (up to solver tolerance)."""
        model = nmf_fit(corpus, k=16)
        W = nmf_encode(embedding, model)
        assert np.all(W >= -1e-6), f"Negative weights found: {W.min()}"

    def test_k_stored(self, corpus):
        model = nmf_fit(corpus, k=8)
        assert model["k"] == 8

    def test_d_stored(self, corpus):
        model = nmf_fit(corpus, k=8)
        assert model["D"] == 128

    def test_fit_subsample(self):
        """nmf_fit should run when corpus is larger than max_residues."""
        rng = np.random.RandomState(7)
        big_corpus = {f"p{i}": rng.randn(1000, 32).astype(np.float32) for i in range(5)}
        model = nmf_fit(big_corpus, k=4, max_residues=500)
        assert model["H"].shape == (4, 32)
