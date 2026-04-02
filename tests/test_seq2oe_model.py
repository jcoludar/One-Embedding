"""Tests for Seq2OE sequence-to-binary-embedding models."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.seq2oe import (
    Seq2OE_CNN, AA_VOCAB_SIZE, encode_sequence,
    Seq2OEDataset, prepare_binary_targets,
)


class TestSeq2OE_CNN:
    def test_output_shape(self):
        model = Seq2OE_CNN(d_out=896)
        x = torch.randint(0, AA_VOCAB_SIZE, (2, 50))
        mask = torch.ones(2, 50)
        logits = model(x, mask)
        assert logits.shape == (2, 50, 896)

    def test_output_shape_short(self):
        model = Seq2OE_CNN(d_out=896)
        x = torch.randint(0, AA_VOCAB_SIZE, (1, 10))
        mask = torch.ones(1, 10)
        logits = model(x, mask)
        assert logits.shape == (1, 10, 896)

    def test_masking(self):
        model = Seq2OE_CNN(d_out=896)
        x = torch.randint(0, AA_VOCAB_SIZE, (1, 20))
        mask = torch.zeros(1, 20)
        mask[0, :10] = 1.0
        logits = model(x, mask)
        assert logits[0, 15, :].abs().sum() == 0.0

    def test_param_count_stage1(self):
        model = Seq2OE_CNN(d_out=896, hidden=128, n_layers=5)
        n_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < n_params < 5_000_000, f"Got {n_params:,} params"

    def test_gradient_flow(self):
        model = Seq2OE_CNN(d_out=896)
        x = torch.randint(0, AA_VOCAB_SIZE, (2, 30))
        mask = torch.ones(2, 30)
        logits = model(x, mask)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


class TestEncodeSequence:
    def test_standard_aas(self):
        indices = encode_sequence("ACDEF")
        assert len(indices) == 5
        assert all(i > 0 for i in indices)

    def test_unknown_maps_to_x(self):
        indices = encode_sequence("AUB")
        x_idx = encode_sequence("X")[0]
        assert indices[1] == x_idx  # U -> X
        assert indices[2] == x_idx  # B -> X

    def test_lowercase(self):
        assert encode_sequence("acdef") == encode_sequence("ACDEF")


class TestPrepareBinaryTargets:
    def test_shape(self):
        rng = np.random.RandomState(0)
        embeddings = {
            "p1": rng.randn(50, 1024).astype(np.float32),
            "p2": rng.randn(30, 1024).astype(np.float32),
        }
        targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
        assert targets["p1"].shape == (50, 896)
        assert targets["p2"].shape == (30, 896)

    def test_binary_values(self):
        rng = np.random.RandomState(0)
        embeddings = {"p1": rng.randn(50, 1024).astype(np.float32)}
        targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
        unique = set(np.unique(targets["p1"]))
        assert unique.issubset({0, 1})

    def test_deterministic(self):
        rng = np.random.RandomState(0)
        embeddings = {"p1": rng.randn(50, 1024).astype(np.float32)}
        t1 = prepare_binary_targets(embeddings, d_out=896, seed=42)
        t2 = prepare_binary_targets(embeddings, d_out=896, seed=42)
        np.testing.assert_array_equal(t1["p1"], t2["p1"])

    def test_approximate_balance(self):
        rng = np.random.RandomState(0)
        embeddings = {"p1": rng.randn(200, 1024).astype(np.float32)}
        targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
        balance = targets["p1"].mean()
        assert 0.3 < balance < 0.7, f"Very imbalanced: {balance:.3f}"


class TestSeq2OEDataset:
    def test_basic(self):
        sequences = {"p1": "ACDEF", "p2": "GHIKLMN"}
        targets = {
            "p1": np.random.randint(0, 2, (5, 896)).astype(np.uint8),
            "p2": np.random.randint(0, 2, (7, 896)).astype(np.uint8),
        }
        ds = Seq2OEDataset(sequences, targets, max_len=20)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (20,)
        assert item["target"].shape == (20, 896)
        assert item["mask"].shape == (20,)
        assert item["length"] == 5

    def test_truncation(self):
        sequences = {"p1": "A" * 100}
        targets = {"p1": np.zeros((100, 896), dtype=np.uint8)}
        ds = Seq2OEDataset(sequences, targets, max_len=50)
        item = ds[0]
        assert item["length"] == 50

    def test_filters_to_common_keys(self):
        sequences = {"p1": "ACDEF", "p2": "GHIK", "p3": "LMNPQ"}
        targets = {"p1": np.zeros((5, 896), dtype=np.uint8),
                    "p3": np.zeros((5, 896), dtype=np.uint8)}
        ds = Seq2OEDataset(sequences, targets, max_len=20)
        assert len(ds) == 2  # p2 excluded (no target)
