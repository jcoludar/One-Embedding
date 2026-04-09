"""Tests for Stage 3 continuous regression helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.seq2oe_continuous import (
    prepare_continuous_targets,
    cosine_distance_loss,
    mse_loss,
    evaluate_continuous,
)


class TestPrepareContinuousTargets:
    def test_shape_and_dtype(self):
        rng = np.random.RandomState(0)
        train_embeddings = {
            f"p{i}": rng.randn(40 + i, 1024).astype(np.float32)
            for i in range(10)
        }
        all_embeddings = dict(train_embeddings)
        all_embeddings["q1"] = rng.randn(30, 1024).astype(np.float32)

        targets = prepare_continuous_targets(
            train_embeddings=train_embeddings,
            all_embeddings=all_embeddings,
            d_out=896,
            seed=42,
        )
        # Every protein in all_embeddings must have a target
        assert set(targets.keys()) == set(all_embeddings.keys())
        # Shapes match input lengths with d_out=896
        assert targets["p0"].shape == (40, 896)
        assert targets["p9"].shape == (49, 896)
        assert targets["q1"].shape == (30, 896)
        # Dtype is float32 (continuous, not uint8)
        assert targets["p0"].dtype == np.float32

    def test_train_only_codec_fit(self):
        """Codec centering must be computed from train embeddings only."""
        rng = np.random.RandomState(0)
        # Build two populations with VERY different means
        train_embeddings = {
            f"train{i}": rng.randn(30, 1024).astype(np.float32) + 10.0
            for i in range(5)
        }
        aux_embeddings = {
            f"aux{i}": rng.randn(30, 1024).astype(np.float32) - 10.0
            for i in range(5)
        }
        all_embeddings = {**train_embeddings, **aux_embeddings}

        targets = prepare_continuous_targets(
            train_embeddings=train_embeddings,
            all_embeddings=all_embeddings,
            d_out=896,
            seed=42,
        )

        # Train targets should be centered around 0 (after codec removes train mean)
        train_mean = np.concatenate([targets[pid].mean(axis=0, keepdims=True)
                                      for pid in train_embeddings]).mean()
        # Aux targets should NOT be centered around 0 (they were shifted by -10
        # relative to train), so their mean is noticeably below 0
        aux_mean = np.concatenate([targets[pid].mean(axis=0, keepdims=True)
                                    for pid in aux_embeddings]).mean()

        assert abs(train_mean) < 0.5, f"train mean {train_mean} should be ~0"
        assert aux_mean < -1.0, f"aux mean {aux_mean} should be << 0"

    def test_deterministic(self):
        rng = np.random.RandomState(0)
        train_embeddings = {"p1": rng.randn(40, 1024).astype(np.float32)}
        t1 = prepare_continuous_targets(
            train_embeddings, train_embeddings, d_out=896, seed=42
        )
        t2 = prepare_continuous_targets(
            train_embeddings, train_embeddings, d_out=896, seed=42
        )
        np.testing.assert_array_equal(t1["p1"], t2["p1"])


class TestCosineDistanceLoss:
    def test_perfect_match_is_zero(self):
        # 2 proteins, length 4, 896 dim
        torch.manual_seed(0)
        target = torch.randn(2, 4, 896)
        pred = target.clone()
        mask = torch.ones(2, 4)
        loss = cosine_distance_loss(pred, target, mask)
        assert abs(loss.item()) < 1e-5, f"expected 0, got {loss.item()}"

    def test_opposite_direction_is_two(self):
        torch.manual_seed(0)
        target = torch.randn(1, 3, 896)
        pred = -target
        mask = torch.ones(1, 3)
        loss = cosine_distance_loss(pred, target, mask)
        # 1 - cos(x, -x) = 1 - (-1) = 2.0
        assert abs(loss.item() - 2.0) < 1e-4

    def test_mask_excludes_padding(self):
        torch.manual_seed(0)
        target = torch.randn(1, 10, 896)
        # First 5 positions: perfect match
        # Last 5 positions: opposite direction, should be IGNORED by mask
        pred = target.clone()
        pred[0, 5:] = -pred[0, 5:]
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1.0  # only first 5 valid
        loss = cosine_distance_loss(pred, target, mask)
        assert abs(loss.item()) < 1e-5, \
            f"masked-out positions leaked into loss: {loss.item()}"

    def test_gradient_flows(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 4, 896, requires_grad=True)
        target = torch.randn(2, 4, 896)
        mask = torch.ones(2, 4)
        loss = cosine_distance_loss(pred, target, mask)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestMSELoss:
    def test_perfect_match_is_zero(self):
        target = torch.randn(2, 4, 896)
        pred = target.clone()
        mask = torch.ones(2, 4)
        loss = mse_loss(pred, target, mask)
        assert abs(loss.item()) < 1e-8

    def test_symmetry(self):
        torch.manual_seed(0)
        a = torch.randn(1, 3, 896)
        b = torch.randn(1, 3, 896)
        mask = torch.ones(1, 3)
        loss_ab = mse_loss(a, b, mask)
        loss_ba = mse_loss(b, a, mask)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-6

    def test_mask_excludes_padding(self):
        torch.manual_seed(0)
        target = torch.zeros(1, 10, 896)
        pred = torch.zeros(1, 10, 896)
        pred[0, 5:] = 100.0  # huge error on masked-out positions
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1.0
        loss = mse_loss(pred, target, mask)
        assert abs(loss.item()) < 1e-6, \
            f"masked-out positions leaked into loss: {loss.item()}"


class TestEvaluateContinuous:
    def test_untrained_model_returns_expected_keys_in_range(self):
        """Smoke test on a real untrained Seq2OE_CNN: metrics are in plausible
        ranges and the returned dict has all required keys."""
        from src.one_embedding.seq2oe import Seq2OE_CNN
        torch.manual_seed(0)
        model = Seq2OE_CNN(d_out=16, hidden=32, n_layers=2)

        rng = np.random.RandomState(0)
        sequences = {f"p{i}": "A" * 10 for i in range(3)}
        targets_np = {f"p{i}": rng.randn(10, 16).astype(np.float32)
                      for i in range(3)}

        metrics = evaluate_continuous(
            model=model,
            sequences=sequences,
            targets=targets_np,
            ids=set(sequences.keys()),
            device=torch.device("cpu"),
            batch_size=2,
            max_len=10,
        )
        required = [
            "cosine_sim", "cosine_distance", "mse", "bit_accuracy",
            "per_protein_bit_acc_mean", "per_protein_bit_acc_std",
            "per_protein_bit_acc_min", "per_protein_bit_acc_max",
            "dim_accuracies", "n_test",
        ]
        for k in required:
            assert k in metrics, f"missing key: {k}"
        assert -1.0 <= metrics["cosine_sim"] <= 1.0
        assert 0.0 <= metrics["mse"]
        assert 0.0 <= metrics["bit_accuracy"] <= 1.0
        assert len(metrics["dim_accuracies"]) == 16
        assert metrics["n_test"] == 3

    def test_fixed_model_matching_targets(self):
        """When the model output exactly equals the target,
        cosine_sim ≈ 1, mse ≈ 0, bit_accuracy ≈ 1."""
        torch.manual_seed(0)
        D = 16
        L = 10
        fixed_target = torch.randn(L, D)

        class FixedOutputModel(torch.nn.Module):
            def __init__(self, fixed):
                super().__init__()
                self.fixed = fixed  # (L, D)
            def eval(self):
                return self
            def forward(self, input_ids, mask):
                B = input_ids.shape[0]
                out = self.fixed.unsqueeze(0).expand(B, -1, -1).clone()
                return out * mask.unsqueeze(-1)

        model = FixedOutputModel(fixed_target)
        # All 3 proteins get the same target = fixed_target
        fixed_np = fixed_target.numpy()
        sequences = {f"p{i}": "A" * L for i in range(3)}
        targets_np = {f"p{i}": fixed_np.copy() for i in range(3)}

        metrics = evaluate_continuous(
            model=model,
            sequences=sequences,
            targets=targets_np,
            ids=set(sequences.keys()),
            device=torch.device("cpu"),
            batch_size=2,
            max_len=L,
        )
        assert metrics["cosine_sim"] > 0.999, \
            f"cosine_sim={metrics['cosine_sim']}, expected ~1.0"
        assert metrics["mse"] < 1e-6, f"mse={metrics['mse']}, expected ~0"
        assert metrics["bit_accuracy"] > 0.999, \
            f"bit_accuracy={metrics['bit_accuracy']}, expected ~1.0"
