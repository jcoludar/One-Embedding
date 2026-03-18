"""Tests for conservation scoring from embeddings."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.conservation import (
    ConservationProbe,
    embedding_variance_conservation,
    embedding_norm_conservation,
)


class TestConservationProbe:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        D = 512
        N = 100
        emb = rng.randn(N, D).astype(np.float32)
        # Conservation correlates with first dimension
        scores = np.clip(0.5 + emb[:, 0] * 0.3, 0, 1)

        probe = ConservationProbe()
        probe.fit(emb, scores)
        predicted = probe.predict(emb)

        assert predicted.shape == (N,)
        assert predicted.min() >= 0.0
        assert predicted.max() <= 1.0
        # Should correlate with true scores
        corr = np.corrcoef(scores, predicted)[0, 1]
        assert corr > 0.5

    def test_save_load_roundtrip(self, tmp_path):
        probe = ConservationProbe(
            coef=np.random.randn(512),
            intercept=0.42,
        )
        path = str(tmp_path / "probe.npz")
        probe.save(path)
        loaded = ConservationProbe.load(path)
        np.testing.assert_allclose(probe.coef, loaded.coef)
        assert abs(probe.intercept - loaded.intercept) < 1e-10

    def test_predict_without_fit_raises(self):
        probe = ConservationProbe()
        with pytest.raises(ValueError):
            probe.predict(np.random.randn(10, 512))

    def test_output_clipped(self):
        probe = ConservationProbe(
            coef=np.ones(64) * 10,  # Very large coefficients
            intercept=5.0,
        )
        emb = np.random.randn(20, 64).astype(np.float32)
        scores = probe.predict(emb)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestEmbeddingVarianceConservation:
    def test_aligned_conserved_positions(self):
        """Positions that are identical across family should score high."""
        rng = np.random.RandomState(42)
        L, D = 10, 64
        # Position 3 is identical across all family members (conserved)
        base = rng.randn(L, D).astype(np.float32)
        family = {}
        for i in range(5):
            emb = base + rng.randn(L, D).astype(np.float32) * 0.5
            emb[3] = base[3]  # Position 3 is conserved
            family[f"prot_{i}"] = emb

        result = embedding_variance_conservation(family, method="aligned")
        scores = result["prot_0"]
        assert scores.shape == (L,)
        # Position 3 should have highest conservation
        assert scores[3] == scores.max()

    def test_aligned_requires_same_length(self):
        family = {
            "a": np.random.randn(10, 64),
            "b": np.random.randn(12, 64),  # Different length
        }
        with pytest.raises(ValueError):
            embedding_variance_conservation(family, method="aligned")

    def test_mean_pool_returns_all_proteins(self):
        rng = np.random.RandomState(42)
        family = {
            "a": rng.randn(10, 64).astype(np.float32),
            "b": rng.randn(15, 64).astype(np.float32),
            "c": rng.randn(8, 64).astype(np.float32),
        }
        result = embedding_variance_conservation(family, method="mean_pool")
        assert set(result.keys()) == {"a", "b", "c"}
        assert result["a"].shape == (10,)
        assert result["b"].shape == (15,)

    def test_conservation_range(self):
        rng = np.random.RandomState(42)
        family = {f"p{i}": rng.randn(10, 64) for i in range(5)}
        result = embedding_variance_conservation(family, method="aligned")
        for scores in result.values():
            assert scores.min() >= 0.0
            assert scores.max() <= 1.0


class TestEmbeddingNormConservation:
    def test_output_shape(self):
        emb = np.random.randn(20, 512).astype(np.float32)
        scores = embedding_norm_conservation(emb)
        assert scores.shape == (20,)

    def test_output_range(self):
        emb = np.random.randn(20, 512).astype(np.float32)
        scores = embedding_norm_conservation(emb)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_high_norm_high_conservation(self):
        """Residues with artificially high norms should score higher."""
        rng = np.random.RandomState(42)
        emb = rng.randn(10, 64).astype(np.float32) * 0.1
        emb[5] = rng.randn(64) * 10  # Much higher norm
        scores = embedding_norm_conservation(emb)
        assert scores[5] > 0.9
