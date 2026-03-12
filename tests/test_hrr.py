"""Tests for Holographic Reduced Representations.

Verifies mathematical properties: bind/unbind algebra, encode/decode
recovery quality, K-slot improvement, determinism, edge cases.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.hrr import (
    hrr_bind,
    hrr_unbind,
    hrr_encode,
    hrr_decode,
    hrr_kslot_encode,
    hrr_kslot_decode,
    hrr_per_protein,
    hrr_per_residue,
    _position_vectors,
)

RNG = np.random.RandomState(42)


@pytest.fixture
def vec_1024():
    """Random 1024-d vector (ProtT5 dimension)."""
    return RNG.randn(1024).astype(np.float32)


@pytest.fixture
def matrix_50x256():
    """Small protein: L=50 residues, d=256."""
    return RNG.randn(50, 256).astype(np.float32)


@pytest.fixture
def matrix_100x1024():
    """Medium protein: L=100, d=1024 (raw ProtT5)."""
    return RNG.randn(100, 1024).astype(np.float32)


# --- Binding/Unbinding Algebra ---

class TestBindUnbind:
    def test_output_shape(self, vec_1024):
        b = hrr_bind(vec_1024, vec_1024)
        assert b.shape == (1024,)

    def test_output_dtype(self, vec_1024):
        b = hrr_bind(vec_1024, vec_1024)
        assert b.dtype == np.float32

    def test_commutative(self):
        a = RNG.randn(512).astype(np.float32)
        b = RNG.randn(512).astype(np.float32)
        np.testing.assert_allclose(hrr_bind(a, b), hrr_bind(b, a), atol=1e-5)

    def test_unbind_recovers_value(self):
        """unbind(key, bind(key, val)) should approximately recover val."""
        key = RNG.randn(1024).astype(np.float32)
        val = RNG.randn(1024).astype(np.float32)
        bound = hrr_bind(key, val)
        recovered = hrr_unbind(key, bound)
        cos_sim = np.dot(val, recovered) / (
            np.linalg.norm(val) * np.linalg.norm(recovered) + 1e-8
        )
        assert cos_sim > 0.5, f"Recovery cosine sim {cos_sim:.3f} too low"

    def test_identity_binding(self):
        """Binding with delta vector (1,0,0,...) returns input unchanged."""
        D = 256
        identity = np.zeros(D, dtype=np.float32)
        identity[0] = 1.0
        val = RNG.randn(D).astype(np.float32)
        result = hrr_bind(identity, val)
        np.testing.assert_allclose(result, val, atol=1e-5)


# --- Encode/Decode ---

class TestEncodeDecode:
    def test_encode_shape(self, matrix_100x1024):
        oe = hrr_encode(matrix_100x1024)
        assert oe.shape == (1024,)

    def test_encode_dtype(self, matrix_100x1024):
        oe = hrr_encode(matrix_100x1024)
        assert oe.dtype == np.float32

    def test_decode_shape(self, matrix_100x1024):
        oe = hrr_encode(matrix_100x1024)
        decoded = hrr_decode(oe, L=100)
        assert decoded.shape == (100, 1024)

    def test_single_residue_recovery(self):
        """L=1 protein: encode-decode should be near-perfect."""
        matrix = RNG.randn(1, 512).astype(np.float32)
        oe = hrr_encode(matrix)
        decoded = hrr_decode(oe, L=1)
        cos_sim = np.dot(matrix[0], decoded[0]) / (
            np.linalg.norm(matrix[0]) * np.linalg.norm(decoded[0]) + 1e-8
        )
        assert cos_sim > 0.95, f"Single-residue recovery {cos_sim:.3f}"

    def test_recovery_positive_cosine(self, matrix_50x256):
        """Average per-residue cosine sim should be positive for moderate L."""
        oe = hrr_encode(matrix_50x256)
        decoded = hrr_decode(oe, L=50)
        cos_sims = []
        for i in range(50):
            cs = np.dot(matrix_50x256[i], decoded[i]) / (
                np.linalg.norm(matrix_50x256[i])
                * np.linalg.norm(decoded[i])
                + 1e-8
            )
            cos_sims.append(cs)
        mean_cs = np.mean(cos_sims)
        assert mean_cs > 0.0, f"Mean cosine sim {mean_cs:.3f} should be positive"

    def test_permutation_sensitive(self, matrix_50x256):
        """Shuffling residue order should change the HRR vector."""
        oe1 = hrr_encode(matrix_50x256)
        shuffled = matrix_50x256[RNG.permutation(50)]
        oe2 = hrr_encode(shuffled)
        assert not np.allclose(oe1, oe2, atol=1e-3)

    def test_deterministic(self, matrix_50x256):
        """Same input + same seed → identical output."""
        oe1 = hrr_encode(matrix_50x256, seed=42)
        oe2 = hrr_encode(matrix_50x256, seed=42)
        np.testing.assert_array_equal(oe1, oe2)

    def test_different_seeds(self, matrix_50x256):
        """Different seeds → different outputs."""
        oe1 = hrr_encode(matrix_50x256, seed=42)
        oe2 = hrr_encode(matrix_50x256, seed=99)
        assert not np.allclose(oe1, oe2, atol=1e-3)


# --- K-Slot ---

class TestKSlot:
    def test_kslot_shape(self, matrix_50x256):
        slots = hrr_kslot_encode(matrix_50x256, K=8)
        assert slots.shape == (8, 256)

    def test_kslot_decode_shape(self, matrix_50x256):
        slots = hrr_kslot_encode(matrix_50x256, K=8)
        decoded = hrr_kslot_decode(slots, L=50)
        assert decoded.shape == (50, 256)

    def test_kslot_improves_recovery(self, matrix_50x256):
        """K=8 slots should give better recovery than K=1."""
        oe1 = hrr_encode(matrix_50x256)
        dec1 = hrr_decode(oe1, L=50)
        cs1 = np.mean([
            np.dot(matrix_50x256[i], dec1[i])
            / (np.linalg.norm(matrix_50x256[i]) * np.linalg.norm(dec1[i]) + 1e-8)
            for i in range(50)
        ])

        slots = hrr_kslot_encode(matrix_50x256, K=8)
        dec8 = hrr_kslot_decode(slots, L=50)
        cs8 = np.mean([
            np.dot(matrix_50x256[i], dec8[i])
            / (np.linalg.norm(matrix_50x256[i]) * np.linalg.norm(dec8[i]) + 1e-8)
            for i in range(50)
        ])
        assert cs8 > cs1, f"K=8 ({cs8:.3f}) should beat K=1 ({cs1:.3f})"

    def test_kslot_perfect_at_K_equals_L(self):
        """K=L gives one slot per residue → near-perfect recovery."""
        L, D = 10, 128
        matrix = RNG.randn(L, D).astype(np.float32)
        slots = hrr_kslot_encode(matrix, K=L)
        decoded = hrr_kslot_decode(slots, L=L)
        cos_sims = [
            np.dot(matrix[i], decoded[i])
            / (np.linalg.norm(matrix[i]) * np.linalg.norm(decoded[i]) + 1e-8)
            for i in range(L)
        ]
        assert np.mean(cos_sims) > 0.95

    def test_kslot_more_than_L(self):
        """K > L should still work (some slots empty)."""
        matrix = RNG.randn(5, 64).astype(np.float32)
        slots = hrr_kslot_encode(matrix, K=10)
        assert slots.shape == (10, 64)
        decoded = hrr_kslot_decode(slots, L=5)
        assert decoded.shape == (5, 64)


# --- Convenience Functions ---

class TestConvenience:
    def test_per_protein_k1(self, matrix_50x256):
        vec = hrr_per_protein(matrix_50x256, K=1)
        assert vec.shape == (256,)

    def test_per_protein_k8(self, matrix_50x256):
        vec = hrr_per_protein(matrix_50x256, K=8)
        assert vec.shape == (8 * 256,)

    def test_per_residue_k1(self, matrix_50x256):
        oe = hrr_per_protein(matrix_50x256, K=1)
        decoded = hrr_per_residue(oe, L=50, K=1)
        assert decoded.shape == (50, 256)

    def test_per_residue_k8(self, matrix_50x256):
        oe = hrr_per_protein(matrix_50x256, K=8)
        decoded = hrr_per_residue(oe, L=50, K=8)
        assert decoded.shape == (50, 256)


# --- Position Vectors ---

class TestPositionVectors:
    def test_shape(self):
        pos = _position_vectors(100, 512, seed=42)
        assert pos.shape == (100, 512)

    def test_dtype(self):
        pos = _position_vectors(10, 64, seed=42)
        assert pos.dtype == np.float32

    def test_deterministic(self):
        p1 = _position_vectors(10, 64, seed=42)
        p2 = _position_vectors(10, 64, seed=42)
        np.testing.assert_array_equal(p1, p2)
