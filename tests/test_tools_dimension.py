"""Tests for tool dimension auto-detection and .one.h5 support.

Tests that:
- load_per_residue works on .one.h5 files at 768d and 512d
- load_protein_vecs works on .one.h5 files at 3072d and 2048d
- The CNN classes can be instantiated with different input_dim values
- predict() with fallback methods works at any dimension (no weights needed)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _make_one_h5_single(d, L=80, D=512, V=2048):
    """Create a single-protein .one.h5 file."""
    from src.one_embedding.io import write_one_h5

    data = {
        "per_residue": np.random.randn(L, D).astype(np.float32),
        "protein_vec": np.random.randn(V).astype(np.float16),
        "sequence": "A" * L,
    }
    path = Path(d) / "single.one.h5"
    write_one_h5(str(path), data, protein_id="test_prot")
    return str(path)


def _make_one_h5_batch(d, n=3, L=80, D=512, V=2048):
    """Create a batch .one.h5 file."""
    from src.one_embedding.io import write_one_h5_batch

    proteins = {}
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": np.random.randn(L, D).astype(np.float32),
            "protein_vec": np.random.randn(V).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "batch.one.h5"
    write_one_h5_batch(str(path), proteins)
    return str(path)


def _make_oemb_batch(d, n=3, L=80, D=512, V=2048):
    """Create a legacy batch .oemb file."""
    from src.one_embedding.io import write_oemb_batch

    proteins = {}
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": np.random.randn(L, D).astype(np.float16),
            "protein_vec": np.random.randn(V).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.oemb"
    write_oemb_batch(str(path), proteins)
    return str(path)


# ── load_per_residue tests ───────────────────────────────────────────


class TestLoadPerResidue:
    def test_one_h5_single_512d(self):
        from src.one_embedding.tools._base import load_per_residue

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_single(d, D=512)
            result = load_per_residue(path)
            assert "test_prot" in result
            assert result["test_prot"].shape == (80, 512)
            assert result["test_prot"].dtype == np.float32

    def test_one_h5_single_768d(self):
        from src.one_embedding.tools._base import load_per_residue

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_single(d, D=768)
            result = load_per_residue(path)
            assert "test_prot" in result
            assert result["test_prot"].shape == (80, 768)
            assert result["test_prot"].dtype == np.float32

    def test_one_h5_batch_512d(self):
        from src.one_embedding.tools._base import load_per_residue

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=3, D=512)
            result = load_per_residue(path)
            assert len(result) == 3
            for pid, emb in result.items():
                assert emb.shape == (80, 512)
                assert emb.dtype == np.float32

    def test_one_h5_batch_768d(self):
        from src.one_embedding.tools._base import load_per_residue

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=3, D=768)
            result = load_per_residue(path)
            assert len(result) == 3
            for pid, emb in result.items():
                assert emb.shape == (80, 768)
                assert emb.dtype == np.float32

    def test_legacy_oemb_still_works(self):
        from src.one_embedding.tools._base import load_per_residue

        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb_batch(d, n=3, D=512)
            result = load_per_residue(path)
            assert len(result) == 3
            for pid, emb in result.items():
                assert emb.shape == (80, 512)
                assert emb.dtype == np.float32


# ── load_protein_vecs tests ──────────────────────────────────────────


class TestLoadProteinVecs:
    def test_one_h5_single_2048d(self):
        from src.one_embedding.tools._base import load_protein_vecs

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_single(d, V=2048)
            result = load_protein_vecs(path)
            assert "test_prot" in result
            assert result["test_prot"].shape == (2048,)
            assert result["test_prot"].dtype == np.float32

    def test_one_h5_single_3072d(self):
        from src.one_embedding.tools._base import load_protein_vecs

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_single(d, V=3072)
            result = load_protein_vecs(path)
            assert "test_prot" in result
            assert result["test_prot"].shape == (3072,)
            assert result["test_prot"].dtype == np.float32

    def test_one_h5_batch_2048d(self):
        from src.one_embedding.tools._base import load_protein_vecs

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=3, V=2048)
            result = load_protein_vecs(path)
            assert len(result) == 3
            for pid, vec in result.items():
                assert vec.shape == (2048,)

    def test_one_h5_batch_3072d(self):
        from src.one_embedding.tools._base import load_protein_vecs

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=3, V=3072)
            result = load_protein_vecs(path)
            assert len(result) == 3
            for pid, vec in result.items():
                assert vec.shape == (3072,)

    def test_legacy_oemb_still_works(self):
        from src.one_embedding.tools._base import load_protein_vecs

        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb_batch(d, n=3, V=2048)
            result = load_protein_vecs(path)
            assert len(result) == 3
            for pid, vec in result.items():
                assert vec.shape == (2048,)


# ── CNN instantiation tests ─────────────────────────────────────────


class TestCNNInstantiation:
    def test_disorder_cnn_512d(self):
        import torch
        import torch.nn as nn

        # Import and call _load_cnn for 512d (weights exist)
        from src.one_embedding.tools.disorder import _load_cnn, _MODEL_CACHE

        # Clear cache to force reload
        _MODEL_CACHE.pop("disorder_512", None)
        model = _load_cnn(input_dim=512)
        assert model is not None
        # Verify forward pass works
        x = torch.randn(1, 80, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 80, 1)

    def test_disorder_cnn_768d_loads(self):
        import torch

        from src.one_embedding.tools.disorder import _load_cnn, _MODEL_CACHE

        _MODEL_CACHE.pop("disorder_768", None)
        model = _load_cnn(input_dim=768)
        assert model is not None
        x = torch.randn(1, 80, 768)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 80, 1)

    def test_disorder_cnn_unsupported_dim(self):
        from src.one_embedding.tools.disorder import _load_cnn, _MODEL_CACHE

        _MODEL_CACHE.pop("disorder_1024", None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = _load_cnn(input_dim=1024)
            assert model is None
            assert len(w) == 1
            assert "1024" in str(w[0].message)

    def test_ss3_cnn_512d(self):
        import torch
        import torch.nn as nn

        from src.one_embedding.tools.ss3 import _load_cnn, _MODEL_CACHE

        _MODEL_CACHE.pop("ss3_512", None)
        model = _load_cnn(input_dim=512)
        assert model is not None
        x = torch.randn(1, 80, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 80, 3)

    def test_ss3_cnn_768d_loads(self):
        import torch

        from src.one_embedding.tools.ss3 import _load_cnn, _MODEL_CACHE

        _MODEL_CACHE.pop("ss3_768", None)
        model = _load_cnn(input_dim=768)
        assert model is not None
        x = torch.randn(1, 80, 768)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 80, 3)


# ── predict() fallback tests ────────────────────────────────────────


class TestPredictFallback:
    def test_disorder_norm_any_dimension(self):
        """Norm heuristic works at any dimension without weights."""
        from src.one_embedding.tools.disorder import predict

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=2, D=1024)
            result = predict(path, method="norm")
            assert len(result) == 2
            for pid, scores in result.items():
                assert scores.shape == (80,)
                assert np.isfinite(scores).all()

    def test_disorder_cnn_768d_works(self):
        """CNN method on 768d should use the trained CNN (weights exist)."""
        from src.one_embedding.tools.disorder import predict, _MODEL_CACHE

        _MODEL_CACHE.pop("disorder_768", None)
        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=2, D=768)
            result = predict(path, method="cnn")
            assert len(result) == 2
            for pid, scores in result.items():
                assert scores.shape == (80,)
                assert np.isfinite(scores).all()

    def test_disorder_cnn_512d_works(self):
        """CNN method on 512d should use the trained CNN (weights exist)."""
        from src.one_embedding.tools.disorder import predict

        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb_batch(d, n=2, D=512)
            result = predict(path, method="cnn")
            assert len(result) == 2
            for pid, scores in result.items():
                assert scores.shape == (80,)
                assert np.isfinite(scores).all()

    def test_ss3_heuristic_any_dimension(self):
        """Heuristic method works at any dimension without weights."""
        from src.one_embedding.tools.ss3 import predict

        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=2, D=1024)
            result = predict(path, method="heuristic")
            assert len(result) == 2
            for pid, preds in result.items():
                assert preds.shape == (80,)
                assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_ss3_cnn_768d_works(self):
        """CNN method on 768d should use the trained CNN (weights exist)."""
        from src.one_embedding.tools.ss3 import predict, _MODEL_CACHE

        _MODEL_CACHE.pop("ss3_768", None)
        with tempfile.TemporaryDirectory() as d:
            path = _make_one_h5_batch(d, n=2, D=768)
            result = predict(path, method="cnn")
            assert len(result) == 2
            for pid, preds in result.items():
                assert preds.shape == (80,)
                assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_ss3_cnn_512d_works(self):
        """CNN method on 512d should use the trained CNN (weights exist)."""
        from src.one_embedding.tools.ss3 import predict

        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb_batch(d, n=2, D=512)
            result = predict(path, method="cnn")
            assert len(result) == 2
            for pid, preds in result.items():
                assert preds.shape == (80,)
                assert set(np.unique(preds)).issubset({0, 1, 2})
