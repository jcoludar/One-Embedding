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
