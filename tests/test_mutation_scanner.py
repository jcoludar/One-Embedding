"""Tests for mutational landscape scanner."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.mutation_scanner import (
    embedding_displacement,
    position_sensitivity,
    MutationEffectProbe,
)


class TestEmbeddingDisplacement:
    def test_identical_zero_displacement(self):
        emb = np.random.randn(10, 64).astype(np.float32)
        disp = embedding_displacement(emb, emb)
        np.testing.assert_allclose(disp, 0.0, atol=1e-6)

    def test_different_positive_displacement(self):
        rng = np.random.RandomState(42)
        wt = rng.randn(10, 64).astype(np.float32)
        mut = rng.randn(10, 64).astype(np.float32)
        disp = embedding_displacement(wt, mut)
        assert disp.shape == (10,)
        assert np.all(disp >= 0)

    def test_single_position_mutation(self):
        rng = np.random.RandomState(42)
        wt = rng.randn(10, 64).astype(np.float32)
        mut = wt.copy()
        mut[5] = rng.randn(64)  # Mutate position 5
        disp = embedding_displacement(wt, mut)
        # Position 5 should have highest displacement
        assert disp[5] > disp.mean()


class TestPositionSensitivity:
    def test_output_shape(self):
        emb = np.random.randn(20, 64).astype(np.float32)
        sens = position_sensitivity(emb)
        assert sens.shape == (20,)

    def test_outlier_position_high_sensitivity(self):
        # Build embeddings where all positions point along dimension 0,
        # but position 7 points along dimension 1 — maximally unlike its neighbors.
        emb = np.zeros((15, 64), dtype=np.float32)
        emb[:, 0] = 1.0          # all positions along dim 0
        emb[7, 0] = 0.0
        emb[7, 1] = 1.0          # outlier along orthogonal dim 1
        sens = position_sensitivity(emb, window=3)
        assert sens[7] > sens.mean()


class TestMutationEffectProbe:
    def test_predict_shape(self):
        probe = MutationEffectProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(15, 64).astype(np.float32)
        scores = probe.predict(emb)
        assert scores.shape == (15, 20)

    def test_predict_landscape(self):
        probe = MutationEffectProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(10, 64).astype(np.float32)
        result = probe.predict_landscape(emb, "ACDEFGHIKL")
        assert result["scores"].shape == (10, 20)
        assert result["landscape"].shape == (10, 20)
        assert len(result["most_damaging"]) <= 20
        assert result["position_sensitivity"].shape == (10,)

    def test_wt_positions_zeroed_in_landscape(self):
        probe = MutationEffectProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(5, 64).astype(np.float32)
        result = probe.predict_landscape(emb, "ACDEF")
        # Position 0 is A (index 0 in AA_ORDER), should be 0
        assert result["landscape"][0, 0] == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        probe = MutationEffectProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(8, 64).astype(np.float32)
        pred1 = probe.predict(emb)

        path = str(tmp_path / "mut.npz")
        probe.save(path)
        loaded = MutationEffectProbe.load(path)
        pred2 = loaded.predict(emb)
        np.testing.assert_allclose(pred1, pred2, atol=1e-6)

    def test_predict_unfitted_raises(self):
        probe = MutationEffectProbe()
        with pytest.raises(ValueError):
            probe.predict(np.random.randn(10, 512).astype(np.float32))

    def test_aa_order_20(self):
        assert len(MutationEffectProbe.AA_ORDER) == 20
