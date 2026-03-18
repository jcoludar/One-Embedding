"""Tests for src/one_embedding/core/ — ABTT3 + RP512 + DCT K=4 core codec."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.core import Codec
from src.one_embedding.core.preprocessing import fit_abtt, apply_abtt
from src.one_embedding.core.projection import project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protein(L: int = 50, D: int = 1024, seed: int = 0) -> np.ndarray:
    """Return a (L, D) float32 array of synthetic raw embeddings."""
    rng = np.random.RandomState(seed)
    return rng.randn(L, D).astype(np.float32)


def _make_corpus(n: int = 5, D: int = 1024, seed: int = 0) -> dict:
    """Return a dict of n synthetic proteins."""
    rng = np.random.RandomState(seed)
    return {
        f"prot_{i}": rng.randn(30 + i * 10, D).astype(np.float32)
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# TestProjection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_output_shape(self):
        X = _make_protein(L=60, D=1024)
        out = project(X, d_out=512)
        assert out.shape == (60, 512), f"Expected (60, 512), got {out.shape}"

    def test_output_dtype_float32(self):
        X = _make_protein(L=40, D=1024)
        out = project(X, d_out=512)
        assert out.dtype == np.float32

    def test_deterministic_same_seed(self):
        X = _make_protein(L=50, D=1024)
        out1 = project(X, d_out=512, seed=42)
        out2 = project(X, d_out=512, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        X = _make_protein(L=50, D=1024)
        out1 = project(X, d_out=512, seed=42)
        out2 = project(X, d_out=512, seed=99)
        assert not np.allclose(out1, out2), "Different seeds should produce different projections"

    def test_norm_preservation_jl_scaling(self):
        """JL scaling sqrt(D/d_out) should make mean norm ratio close to 1.0."""
        rng = np.random.RandomState(7)
        X = rng.randn(200, 1024).astype(np.float32)
        projected = project(X, d_out=512, seed=42)

        norms_in = np.linalg.norm(X, axis=1)
        norms_out = np.linalg.norm(projected, axis=1)
        ratio = (norms_out / norms_in).mean()

        assert 0.9 <= ratio <= 1.1, (
            f"Norm ratio {ratio:.4f} outside [0.9, 1.1] — JL scaling may be wrong"
        )

    def test_small_d_out(self):
        """Works with d_out much smaller than D."""
        X = _make_protein(L=20, D=1024)
        out = project(X, d_out=64, seed=42)
        assert out.shape == (20, 64)

    def test_different_input_dims(self):
        """Works for ProtT5 (1024), ESM2 (1280), ESM-C (960)."""
        for D in [960, 1024, 1280]:
            X = _make_protein(L=30, D=D)
            out = project(X, d_out=512, seed=42)
            assert out.shape == (30, 512)


# ---------------------------------------------------------------------------
# TestPreprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_fit_abtt_returns_mean_and_top_pcs(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(1000, 1024).astype(np.float32)
        params = fit_abtt(residues, k=3)
        assert "mean" in params
        assert "top_pcs" in params
        assert params["mean"].shape == (1024,)
        assert params["top_pcs"].shape == (3, 1024)

    def test_fit_abtt_output_dtypes_float32(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(500, 1024).astype(np.float32)
        params = fit_abtt(residues, k=3)
        assert params["mean"].dtype == np.float32
        assert params["top_pcs"].dtype == np.float32

    def test_fit_abtt_top_pcs_are_unit_vectors(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(500, 1024).astype(np.float32)
        params = fit_abtt(residues, k=3)
        norms = np.linalg.norm(params["top_pcs"], axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_apply_abtt_shape_preserved(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(500, 1024).astype(np.float32)
        params = fit_abtt(residues, k=3)
        X = _make_protein(L=40, D=1024)
        out = apply_abtt(X, params)
        assert out.shape == X.shape

    def test_apply_abtt_dtype_float32(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(300, 1024).astype(np.float32)
        params = fit_abtt(residues, k=3)
        X = _make_protein(L=30, D=1024)
        out = apply_abtt(X, params)
        assert out.dtype == np.float32

    def test_apply_abtt_removes_pcs(self):
        """After ABTT, the top PCs should be nearly orthogonal to the residues."""
        rng = np.random.RandomState(0)
        residues = rng.randn(2000, 256).astype(np.float32)
        params = fit_abtt(residues, k=3)

        # Apply to a fresh batch
        X = rng.randn(100, 256).astype(np.float32)
        out = apply_abtt(X, params)

        # Projections of output onto top PCs should be near zero
        top_pcs = params["top_pcs"]  # (3, 256)
        projections = np.abs(out @ top_pcs.T)  # (100, 3)
        # Mean projection should be much smaller than before
        before_projections = np.abs((X - params["mean"]) @ top_pcs.T)
        assert projections.mean() < before_projections.mean() * 0.05, (
            "ABTT should strongly reduce PC projections"
        )

    def test_fit_abtt_subsamples_large_input(self):
        """fit_abtt should handle >50K rows without OOM."""
        rng = np.random.RandomState(0)
        residues = rng.randn(60_000, 64).astype(np.float32)
        params = fit_abtt(residues, k=2, seed=42)
        assert params["mean"].shape == (64,)
        assert params["top_pcs"].shape == (2, 64)

    def test_fit_abtt_deterministic(self):
        rng = np.random.RandomState(0)
        residues = rng.randn(500, 256).astype(np.float32)
        p1 = fit_abtt(residues, k=3, seed=42)
        p2 = fit_abtt(residues, k=3, seed=42)
        np.testing.assert_array_equal(p1["mean"], p2["mean"])
        # top_pcs may differ by sign flip — compare absolute values
        np.testing.assert_allclose(
            np.abs(p1["top_pcs"]), np.abs(p2["top_pcs"]), atol=1e-5
        )


# ---------------------------------------------------------------------------
# TestCodec
# ---------------------------------------------------------------------------

class TestCodec:
    def test_encode_output_shapes_default(self):
        codec = Codec(d_out=512, dct_k=4)
        raw = _make_protein(L=50, D=1024)
        result = codec.encode(raw)
        assert result["per_residue"].shape == (50, 512)
        assert result["protein_vec"].shape == (2048,)  # 4 * 512

    def test_encode_output_dtype_float16(self):
        codec = Codec()
        raw = _make_protein(L=30, D=1024)
        result = codec.encode(raw)
        assert result["per_residue"].dtype == np.float16
        assert result["protein_vec"].dtype == np.float16

    def test_encode_deterministic(self):
        codec = Codec(d_out=512, dct_k=4, seed=42)
        raw = _make_protein(L=40, D=1024)
        r1 = codec.encode(raw)
        r2 = codec.encode(raw)
        np.testing.assert_array_equal(r1["per_residue"], r2["per_residue"])
        np.testing.assert_array_equal(r1["protein_vec"], r2["protein_vec"])

    def test_encode_works_without_fit(self):
        """Codec encodes without ABTT params (no preprocessing)."""
        codec = Codec()
        raw = _make_protein(L=20, D=1024)
        result = codec.encode(raw)
        assert result["per_residue"].shape == (20, 512)
        assert result["protein_vec"].shape == (2048,)

    def test_encode_edge_case_L1(self):
        """Single-residue protein: L=1."""
        codec = Codec(d_out=512, dct_k=4)
        raw = _make_protein(L=1, D=1024)
        result = codec.encode(raw)
        assert result["per_residue"].shape == (1, 512)
        assert result["protein_vec"].shape == (2048,)  # zero-padded

    def test_encode_different_plm_dims(self):
        """Works for ProtT5 (1024d), ESM2 (1280d), ESM-C (960d)."""
        codec = Codec(d_out=512, dct_k=4)
        for D in [960, 1024, 1280]:
            raw = _make_protein(L=30, D=D)
            result = codec.encode(raw)
            assert result["per_residue"].shape == (30, 512), f"Failed for D={D}"
            assert result["protein_vec"].shape == (2048,), f"Failed for D={D}"

    def test_fit_from_corpus(self):
        """fit() + encode() changes output vs no-fit (ABTT is applied)."""
        corpus = _make_corpus(n=10, D=1024)
        codec_fit = Codec(d_out=512, dct_k=4, seed=42)
        codec_fit.fit(corpus)

        codec_raw = Codec(d_out=512, dct_k=4, seed=42)

        raw = _make_protein(L=50, D=1024, seed=999)
        r_fit = codec_fit.encode(raw)
        r_raw = codec_raw.encode(raw)

        # ABTT preprocessing changes the output
        assert not np.allclose(
            r_fit["per_residue"].astype(np.float32),
            r_raw["per_residue"].astype(np.float32),
        ), "fit() should change encode() output via ABTT"

    def test_retrieval_quality_similar_closer_than_random(self):
        """Similar proteins (same family) should have closer protein_vecs."""
        rng = np.random.RandomState(42)
        codec = Codec(d_out=512, dct_k=4, seed=42)

        # Create a "base" protein
        base = rng.randn(80, 1024).astype(np.float32)

        # "Similar": small perturbation of base
        similar = base + rng.randn(80, 1024).astype(np.float32) * 0.1

        # "Random": unrelated protein
        random_prot = rng.randn(80, 1024).astype(np.float32)

        vec_base = codec.encode(base)["protein_vec"].astype(np.float32)
        vec_similar = codec.encode(similar)["protein_vec"].astype(np.float32)
        vec_random = codec.encode(random_prot)["protein_vec"].astype(np.float32)

        # Cosine similarity
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        sim_close = cosine(vec_base, vec_similar)
        sim_random = cosine(vec_base, vec_random)

        assert sim_close > sim_random, (
            f"Similar protein (cos={sim_close:.3f}) should be closer than "
            f"random (cos={sim_random:.3f})"
        )

    def test_save_params_load_params_roundtrip(self, tmp_path):
        """save_params/load_params restores ABTT state exactly."""
        corpus = _make_corpus(n=8, D=1024)
        codec = Codec(d_out=512, dct_k=4, seed=42)
        codec.fit(corpus)

        params_path = tmp_path / "abtt_params.json"
        codec.save_params(str(params_path))

        # Load into a fresh codec
        codec2 = Codec(d_out=512, dct_k=4, seed=42)
        codec2.load_params(str(params_path))

        # Both codecs should produce identical output
        raw = _make_protein(L=50, D=1024, seed=7)
        r1 = codec.encode(raw)
        r2 = codec2.encode(raw)
        np.testing.assert_array_equal(r1["per_residue"], r2["per_residue"])
        np.testing.assert_array_equal(r1["protein_vec"], r2["protein_vec"])

    def test_save_params_raises_if_not_fitted(self, tmp_path):
        """save_params() raises RuntimeError if fit() not called."""
        codec = Codec()
        with pytest.raises(RuntimeError, match="fit"):
            codec.save_params(str(tmp_path / "params.json"))

    def test_save_params_creates_parent_dir(self, tmp_path):
        """save_params creates intermediate directories as needed."""
        corpus = _make_corpus(n=3, D=64)
        codec = Codec(d_out=64, dct_k=2, seed=42)
        codec.fit(corpus)

        deep_path = tmp_path / "a" / "b" / "params.json"
        codec.save_params(str(deep_path))
        assert deep_path.exists()

    def test_load_params_restores_hyperparams(self, tmp_path):
        """load_params restores d_out, dct_k, seed from file."""
        corpus = _make_corpus(n=3, D=64)
        codec = Codec(d_out=64, dct_k=2, seed=99)
        codec.fit(corpus)
        params_path = tmp_path / "params.json"
        codec.save_params(str(params_path))

        codec2 = Codec()  # default params
        codec2.load_params(str(params_path))
        assert codec2.d_out == 64
        assert codec2.dct_k == 2
        assert codec2.seed == 99
