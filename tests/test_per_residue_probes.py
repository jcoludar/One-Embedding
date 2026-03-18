"""Tests for per-residue prediction probes."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.per_residue_probes import (
    DisorderProbe,
    TopologyProbe,
    BindingProbe,
)


class TestDisorderProbe:
    def test_predict_shape(self):
        probe = DisorderProbe(input_dim=64, hidden=8, kernel=3)
        probe._init_weights()
        emb = np.random.randn(20, 64).astype(np.float32)
        out = probe.predict(emb)
        assert out.shape == (20,)

    def test_predict_binary(self):
        probe = DisorderProbe(input_dim=64, hidden=8, kernel=3)
        probe._init_weights()
        emb = np.random.randn(20, 64).astype(np.float32)
        binary = probe.predict_binary(emb, threshold=0.0)
        assert binary.shape == (20,)
        assert set(np.unique(binary)).issubset({0, 1})

    def test_save_load_roundtrip(self, tmp_path):
        probe = DisorderProbe(input_dim=64, hidden=8, kernel=3)
        probe._init_weights()
        emb = np.random.randn(10, 64).astype(np.float32)
        pred1 = probe.predict(emb)

        path = str(tmp_path / "disorder.npz")
        probe.save(path)
        loaded = DisorderProbe.load(path)
        pred2 = loaded.predict(emb)
        np.testing.assert_allclose(pred1, pred2, atol=1e-6)

    def test_predict_unfitted_raises(self):
        probe = DisorderProbe()
        with pytest.raises(ValueError):
            probe.predict(np.random.randn(10, 512).astype(np.float32))

    def test_fit_reduces_loss(self):
        """Training should reduce MSE loss."""
        rng = np.random.RandomState(42)
        D = 32
        # Simple linear relationship
        X = [rng.randn(20, D).astype(np.float32) for _ in range(5)]
        y = [x[:, 0] * 2 + 1 for x in X]  # y depends on first dim

        probe = DisorderProbe(input_dim=D, hidden=4, kernel=3)
        probe._init_weights()
        pred_before = np.concatenate([probe.predict(x) for x in X])
        loss_before = np.mean((pred_before - np.concatenate(y)) ** 2)

        probe.fit(X, y, lr=0.01, epochs=20, seed=42)
        pred_after = np.concatenate([probe.predict(x) for x in X])
        loss_after = np.mean((pred_after - np.concatenate(y)) ** 2)

        assert loss_after < loss_before


class TestTopologyProbe:
    def test_predict_shape(self):
        probe = TopologyProbe(input_dim=64, hidden=16, n_classes=5)
        probe._init_weights()
        emb = np.random.randn(30, 64).astype(np.float32)
        logits = probe.predict(emb)
        assert logits.shape == (30, 5)

    def test_predict_labels_string(self):
        probe = TopologyProbe(input_dim=64, hidden=16, n_classes=5)
        probe._init_weights()
        emb = np.random.randn(10, 64).astype(np.float32)
        labels = probe.predict_labels(emb)
        assert len(labels) == 10
        assert all(c in "ioHBS" for c in labels)

    def test_save_load_roundtrip(self, tmp_path):
        probe = TopologyProbe(input_dim=64, hidden=16)
        probe._init_weights()
        emb = np.random.randn(10, 64).astype(np.float32)
        pred1 = probe.predict(emb)

        path = str(tmp_path / "topo.npz")
        probe.save(path)
        loaded = TopologyProbe.load(path)
        pred2 = loaded.predict(emb)
        np.testing.assert_allclose(pred1, pred2, atol=1e-6)

    def test_predict_unfitted_raises(self):
        probe = TopologyProbe()
        with pytest.raises(ValueError):
            probe.predict(np.random.randn(10, 512).astype(np.float32))


class TestBindingProbe:
    def test_predict_shape(self):
        probe = BindingProbe(input_dim=64, hidden=32, n_types=3)
        probe._init_weights()
        emb = np.random.randn(25, 64).astype(np.float32)
        probs = probe.predict(emb)
        assert probs.shape == (25, 3)

    def test_predict_probabilities_range(self):
        probe = BindingProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(25, 64).astype(np.float32)
        probs = probe.predict(emb)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_binary(self):
        probe = BindingProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(15, 64).astype(np.float32)
        binary = probe.predict_binary(emb)
        assert binary.shape == (15, 3)
        assert set(np.unique(binary)).issubset({0, 1})

    def test_binding_types(self):
        assert BindingProbe.BINDING_TYPES == ["metal", "nucleic", "small_molecule"]

    def test_save_load_roundtrip(self, tmp_path):
        probe = BindingProbe(input_dim=64, hidden=32)
        probe._init_weights()
        emb = np.random.randn(10, 64).astype(np.float32)
        pred1 = probe.predict(emb)

        path = str(tmp_path / "binding.npz")
        probe.save(path)
        loaded = BindingProbe.load(path)
        pred2 = loaded.predict(emb)
        np.testing.assert_allclose(pred1, pred2, atol=1e-6)

    def test_predict_unfitted_raises(self):
        probe = BindingProbe()
        with pytest.raises(ValueError):
            probe.predict(np.random.randn(10, 512).astype(np.float32))
