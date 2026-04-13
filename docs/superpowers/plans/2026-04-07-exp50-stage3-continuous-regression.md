# Exp 50 Stage 3 Continuous Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a sequence-only dilated CNN to predict ProtT5's continuous 896d projected embedding (not just the binary sign bits), using CATH20 + DeepLoc as training pool (~25K proteins), with cosine + MSE loss and cosine similarity / MSE / bit accuracy as primary metrics. Three seeds on the same CATH20 H-split test set as Stage 2, for direct comparability.

**Architecture:** Unchanged from Stage 2 — same 10-layer dilated `Seq2OE_CNN` (4.2M params). Only the loss, targets, and training data change. Stage 3 code is fully additive: a new `seq2oe_continuous.py` module holds the target-preparation, loss, and evaluation helpers; new experiment + runner + leakage-filter scripts live alongside the Stage 1/2 scripts. No existing Stage 1/2 code is modified.

**Tech Stack:** PyTorch (MPS), NumPy, h5py, Biopython (for FASTA reading, already a project dependency), MMseqs2 (already installed at `/opt/homebrew/bin/mmseqs`).

**Spec reference:** `docs/superpowers/specs/2026-04-07-exp50-stage3-continuous-regression-design.md`

**Working directory:** `/Users/jcoludar/CascadeProjects/ProteEmbedExplorations/.worktrees/exp50-rigorous` (branch `exp50/rigorous-cath-split`).

---

### Task 1: `prepare_continuous_targets` helper

**Files:**
- Create: `src/one_embedding/seq2oe_continuous.py`
- Create: `tests/test_seq2oe_continuous.py`

- [ ] **Step 1: Write failing tests for prepare_continuous_targets**

```python
# tests/test_seq2oe_continuous.py
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
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestPrepareContinuousTargets -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.one_embedding.seq2oe_continuous'`

- [ ] **Step 3: Implement `prepare_continuous_targets` + module scaffolding**

```python
# src/one_embedding/seq2oe_continuous.py
"""Stage 3 continuous regression helpers for Seq2OE.

Unlike `seq2oe.py` (which uses binary BCE targets), Stage 3 predicts the
continuous 896d projected ProtT5 vector directly and trains with cosine +
MSE loss. This module provides the target-preparation, loss, and evaluation
utilities specific to that setup.

The model class itself (`Seq2OE_CNN`) is reused unchanged from `seq2oe.py` —
its 896-dim linear head outputs floats that we now interpret as continuous
regression values instead of pre-sigmoid logits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def prepare_continuous_targets(
    train_embeddings: dict[str, np.ndarray],
    all_embeddings: dict[str, np.ndarray],
    d_out: int = 896,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Fit `OneEmbeddingCodec` on train embeddings only, then apply the
    preprocessing pipeline (centering + random projection) to every protein
    in `all_embeddings`. Returns the continuous 896d projected vectors
    (pre-binarization) so Stage 3 can regress on them directly.

    Args:
        train_embeddings: {pid: (L, D) float32} used to fit codec centering
            stats. Typically the CATH20 H-split train fold.
        all_embeddings: {pid: (L, D) float32} the full set to encode. May be
            a strict superset of train_embeddings.
        d_out: Random projection target dimension. Matches Stage 2.
        seed: Deterministic RP matrix seed.

    Returns:
        {pid: (L, d_out) float32} continuous projected targets. Every key
        from `all_embeddings` appears in the result.
    """
    from src.one_embedding.codec_v2 import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(train_embeddings)

    targets: dict[str, np.ndarray] = {}
    for pid, raw in all_embeddings.items():
        projected = codec._preprocess(raw)  # (L, d_out), float32
        targets[pid] = projected.astype(np.float32, copy=False)
    return targets


# Loss and evaluation functions implemented in Tasks 2 and 3.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestPrepareContinuousTargets -v`
Expected: 3 passed.

Note: the `TestPrepareContinuousTargets` tests will pass but the file-level imports of `cosine_distance_loss`, `mse_loss`, `evaluate_continuous` will fail at collection time. To make Task 1 self-contained, add three NotImplementedError stubs at the bottom of the module:

```python
def cosine_distance_loss(*args, **kwargs):  # implemented in Task 2
    raise NotImplementedError("Implemented in Task 2")

def mse_loss(*args, **kwargs):  # implemented in Task 2
    raise NotImplementedError("Implemented in Task 2")

def evaluate_continuous(*args, **kwargs):  # implemented in Task 3
    raise NotImplementedError("Implemented in Task 3")
```

Then re-run: `uv run pytest tests/test_seq2oe_continuous.py::TestPrepareContinuousTargets -v`
Expected: 3 passed (test file imports succeed, stubs coexist).

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe_continuous.py tests/test_seq2oe_continuous.py
git commit -m "feat(exp50): prepare_continuous_targets helper for Stage 3"
```

---

### Task 2: Masked cosine + MSE loss functions

**Files:**
- Modify: `src/one_embedding/seq2oe_continuous.py`
- Modify: `tests/test_seq2oe_continuous.py`

- [ ] **Step 1: Append failing tests for loss functions**

Append to `tests/test_seq2oe_continuous.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestCosineDistanceLoss tests/test_seq2oe_continuous.py::TestMSELoss -v`
Expected: FAIL — `NotImplementedError: Implemented in Task 2` on every test.

- [ ] **Step 3: Replace the stubs with real implementations**

In `src/one_embedding/seq2oe_continuous.py`, REMOVE the `cosine_distance_loss` and `mse_loss` stubs and replace with:

```python
def cosine_distance_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Masked per-residue cosine distance, averaged over valid positions.

    For each (batch, residue) position, compute
        1 - cos(pred[b, r, :], target[b, r, :])
    then average over positions where mask == 1.

    Args:
        pred: (B, L, D) continuous predictions.
        target: (B, L, D) continuous targets.
        mask: (B, L) float, 1.0 for valid positions, 0.0 for padding.
        eps: Numerical stability floor on the norms.

    Returns:
        Scalar tensor in [0, 2]. 0 = perfect alignment, 2 = antiparallel.
    """
    # (B, L) cosine similarity per residue
    pred_norm = pred.norm(dim=-1).clamp_min(eps)
    target_norm = target.norm(dim=-1).clamp_min(eps)
    cos_sim = (pred * target).sum(dim=-1) / (pred_norm * target_norm)
    cos_dist = 1.0 - cos_sim  # (B, L)

    # Masked average
    n_valid = mask.sum().clamp_min(1.0)
    return (cos_dist * mask).sum() / n_valid


def mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Masked per-element MSE, averaged over valid positions AND all D dims.

    Args:
        pred: (B, L, D) continuous predictions.
        target: (B, L, D) continuous targets.
        mask: (B, L) float, 1.0 for valid positions, 0.0 for padding.

    Returns:
        Scalar tensor. Mean squared error over valid (b, r, d) cells.
    """
    # (B, L, D) squared error
    sq = (pred - target) ** 2
    # Broadcast mask to (B, L, 1)
    mask_3d = mask.unsqueeze(-1)
    # Total valid cells = valid_residues * D
    d = pred.shape[-1]
    n_valid = mask.sum().clamp_min(1.0) * d
    return (sq * mask_3d).sum() / n_valid
```

- [ ] **Step 4: Run all Task 2 tests**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestCosineDistanceLoss tests/test_seq2oe_continuous.py::TestMSELoss -v`
Expected: 7 passed.

Also re-run Task 1 tests to confirm no regression:
Run: `uv run pytest tests/test_seq2oe_continuous.py -v`
Expected: 10 passed (3 from Task 1 + 7 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe_continuous.py tests/test_seq2oe_continuous.py
git commit -m "feat(exp50): masked cosine distance + MSE loss for Stage 3"
```

---

### Task 3: `evaluate_continuous` metrics function

**Files:**
- Modify: `src/one_embedding/seq2oe_continuous.py`
- Modify: `tests/test_seq2oe_continuous.py`

- [ ] **Step 1: Append failing tests for evaluate_continuous**

Append to `tests/test_seq2oe_continuous.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestEvaluateContinuous -v`
Expected: FAIL — `NotImplementedError: Implemented in Task 3`.

- [ ] **Step 3: Implement `evaluate_continuous`**

In `src/one_embedding/seq2oe_continuous.py`, REMOVE the `evaluate_continuous` stub and replace with:

```python
def evaluate_continuous(
    model,
    sequences: dict[str, str],
    targets: dict[str, np.ndarray],
    ids: set,
    device: torch.device,
    batch_size: int = 8,
    max_len: int = 512,
) -> dict:
    """Evaluate a Seq2OE model on a held-out set with continuous metrics.

    Computes the four Stage 3 primary metrics at the per-residue level:
        - cosine_sim: mean cosine similarity per residue, averaged over all
          valid positions.
        - cosine_distance: 1 - cosine_sim (redundant but reported for
          symmetry with the loss name).
        - mse: mean squared error per (residue, dim) cell.
        - bit_accuracy: re-binarize both pred and target via
          `sign(x - per-protein-mean)` and compare bit-by-bit.

    Also returns `dim_accuracies` — a length-D array of per-dimension bit
    accuracies, matching Stage 2's reporting format for the Intersect@60
    aggregation.

    Args:
        model: any `nn.Module` with signature `forward(input_ids, mask)
            -> (B, L, D)`.
        sequences: {pid: str} amino-acid sequences.
        targets: {pid: (L, D) float32} continuous targets.
        ids: set of protein IDs to evaluate on (intersection with sequences
            and targets keys is used).
        device: torch device.
        batch_size: eval batch size.
        max_len: sequence length cap (padding/truncation matches training).

    Returns:
        dict with keys: cosine_sim, cosine_distance, mse, bit_accuracy,
        per_protein_bit_acc_mean, per_protein_bit_acc_std,
        per_protein_bit_acc_min, per_protein_bit_acc_max,
        dim_accuracies (list of length D), n_test.
    """
    from torch.utils.data import DataLoader
    from src.one_embedding.seq2oe import Seq2OEDataset

    eval_seqs = {k: sequences[k] for k in ids if k in sequences and k in targets}
    eval_tgts = {k: targets[k] for k in ids if k in sequences and k in targets}
    ds = Seq2OEDataset(eval_seqs, eval_tgts, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Accumulators
    cos_sum = 0.0
    cos_count = 0
    mse_sum = 0.0
    mse_count = 0
    bit_correct_total = 0
    bit_total = 0
    dim_bit_correct = None  # lazy-init once we see D
    dim_bit_total = 0
    per_protein_accs: list[float] = []

    model_mode = None
    if hasattr(model, "eval"):
        model_mode = model
        model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)  # (B, L, D), float32
            mask = batch["mask"].to(device)       # (B, L)
            lengths = batch["length"]

            pred = model(input_ids, mask)         # (B, L, D)

            # Cosine sim per residue
            pred_norm = pred.norm(dim=-1).clamp_min(1e-8)
            target_norm = target.norm(dim=-1).clamp_min(1e-8)
            cos = (pred * target).sum(dim=-1) / (pred_norm * target_norm)
            cos_sum += (cos * mask).sum().item()
            cos_count += mask.sum().item()

            # MSE
            d = pred.shape[-1]
            sq = (pred - target) ** 2
            mse_sum += (sq * mask.unsqueeze(-1)).sum().item()
            mse_count += mask.sum().item() * d

            # Bit accuracy after re-binarization (per-protein mean subtraction)
            if dim_bit_correct is None:
                dim_bit_correct = np.zeros(d, dtype=np.int64)

            for b in range(pred.shape[0]):
                L = int(lengths[b].item())
                if L == 0:
                    continue
                p = pred[b, :L].cpu().numpy()        # (L, D)
                t = target[b, :L].cpu().numpy()      # (L, D)
                p_bits = (p - p.mean(axis=0, keepdims=True)) > 0
                t_bits = (t - t.mean(axis=0, keepdims=True)) > 0
                eq = (p_bits == t_bits)              # (L, D) bool
                bit_correct_total += int(eq.sum())
                bit_total += eq.size
                dim_bit_correct += eq.sum(axis=0)
                dim_bit_total += L
                per_protein_accs.append(float(eq.mean()))

    cosine_sim = cos_sum / max(cos_count, 1.0)
    mse = mse_sum / max(mse_count, 1.0)
    bit_accuracy = bit_correct_total / max(bit_total, 1)
    dim_acc = (dim_bit_correct / max(dim_bit_total, 1)).tolist()
    per_prot = np.array(per_protein_accs) if per_protein_accs else np.zeros(1)

    return {
        "cosine_sim": float(cosine_sim),
        "cosine_distance": float(1.0 - cosine_sim),
        "mse": float(mse),
        "bit_accuracy": float(bit_accuracy),
        "per_protein_bit_acc_mean": float(per_prot.mean()),
        "per_protein_bit_acc_std": float(per_prot.std()),
        "per_protein_bit_acc_min": float(per_prot.min()),
        "per_protein_bit_acc_max": float(per_prot.max()),
        "dim_accuracies": dim_acc,
        "n_test": len(per_protein_accs),
    }
```

- [ ] **Step 4: Run all Task 3 tests**

Run: `uv run pytest tests/test_seq2oe_continuous.py::TestEvaluateContinuous -v`
Expected: 2 passed.

Also run the whole file:
Run: `uv run pytest tests/test_seq2oe_continuous.py -v`
Expected: 12 passed (3 + 7 + 2).

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe_continuous.py tests/test_seq2oe_continuous.py
git commit -m "feat(exp50): evaluate_continuous metrics for Stage 3"
```

---

### Task 4: MMseqs2 DeepLoc leakage filter script

**Files:**
- Create: `experiments/50_stage3_leakage_filter.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Exp 50 Stage 3 DeepLoc leakage filter.

Identifies DeepLoc proteins with > 30% sequence identity to any CATH20
H-split test protein at a given seed. The Stage 3 training script uses
this exclusion list to filter the DeepLoc auxiliary pool.

Usage:
    uv run python experiments/50_stage3_leakage_filter.py --seed 42
    uv run python experiments/50_stage3_leakage_filter.py --seed 43
    uv run python experiments/50_stage3_leakage_filter.py --seed 44
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe_splits import parse_cath_fasta, cath_cluster_split

DATA = ROOT / "data"
DEEPLOC_FASTA = (
    ROOT / "tools" / "reference" / "LightAttention" /
    "data_files" / "deeploc_complete_dataset.fasta"
)
CATH_FASTA = DATA / "external" / "cath20" / "cath20_labeled.fasta"
MMSEQS = shutil.which("mmseqs") or "/opt/homebrew/bin/mmseqs"
IDENTITY_THRESHOLD = 30.0  # percent


def write_fasta(sequences: dict, path: Path):
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")


def run_mmseqs_search(query_fa: Path, target_fa: Path, workdir: Path) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "results.tsv"
    cmd = [
        MMSEQS, "easy-search",
        str(query_fa), str(target_fa), str(out), str(workdir / "tmp"),
        "--min-seq-id", "0.0",
        "--threads", "4",
        "--format-output", "query,target,pident",
        "-v", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"MMseqs2 stdout: {res.stdout}")
        print(f"MMseqs2 stderr: {res.stderr}")
        raise RuntimeError("MMseqs2 search failed")
    hits: list[dict] = []
    if out.exists():
        with open(out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),
                    })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument(
        "--identity-threshold", type=float, default=IDENTITY_THRESHOLD,
        help="Exclude DeepLoc proteins with any hit at >= this %%id to "
             "CATH20 H-test (default: 30.0)",
    )
    args = parser.parse_args()

    print(f"Loading CATH20 FASTA...")
    cath_meta = parse_cath_fasta(CATH_FASTA)
    print(f"  {len(cath_meta)} CATH proteins parsed")

    print(f"Building CATH20 H-split (seed={args.seed})...")
    _, _, h_test_ids = cath_cluster_split(
        cath_meta, level="H", fractions=(0.8, 0.1, 0.1), seed=args.seed,
    )
    print(f"  {len(h_test_ids)} H-test proteins")

    print(f"Loading DeepLoc FASTA from {DEEPLOC_FASTA}...")
    deeploc_seqs = read_fasta(DEEPLOC_FASTA)
    print(f"  {len(deeploc_seqs)} DeepLoc sequences parsed")

    with tempfile.TemporaryDirectory(prefix="exp50_stage3_leakage_") as tmp:
        tmp_path = Path(tmp)
        cath_test_fa = tmp_path / "cath_h_test.fa"
        deeploc_fa = tmp_path / "deeploc.fa"

        # Write CATH H-test as query, DeepLoc as target
        cath_test_seqs = {pid: cath_meta[pid]["seq"] for pid in h_test_ids}
        write_fasta(cath_test_seqs, cath_test_fa)
        write_fasta(deeploc_seqs, deeploc_fa)

        print(f"Running MMseqs2 easy-search (cath_h_test -> deeploc)...")
        hits = run_mmseqs_search(cath_test_fa, deeploc_fa, tmp_path / "search")
        print(f"  {len(hits)} total hits")

    # Any DeepLoc protein (target) with any hit >= threshold is excluded
    excluded: set[str] = {
        h["target"] for h in hits if h["pident"] >= args.identity_threshold
    }

    report = {
        "seed": args.seed,
        "identity_threshold_pct": args.identity_threshold,
        "n_cath_h_test": len(h_test_ids),
        "n_deeploc_total": len(deeploc_seqs),
        "n_deeploc_excluded": len(excluded),
        "fraction_deeploc_excluded": len(excluded) / max(len(deeploc_seqs), 1),
        "excluded_deeploc_ids": sorted(excluded),
    }

    out_root = Path(args.output_root) / "leakage_filter"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"deeploc_leakage_excluded_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLeakage filter written to {out_path}")
    print(f"  DeepLoc proteins excluded: {len(excluded)} / {len(deeploc_seqs)} "
          f"({report['fraction_deeploc_excluded']*100:.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the filter for seed 42 (~3 minutes)**

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_stage3_leakage_filter.py --seed 42
```

Expected output:
```
Loading CATH20 FASTA...
  14433 CATH proteins parsed
Building CATH20 H-split (seed=42)...
  1444 H-test proteins
Loading DeepLoc FASTA from .../deeploc_complete_dataset.fasta...
  14004 DeepLoc sequences parsed
Running MMseqs2 easy-search (cath_h_test -> deeploc)...
  N total hits
Leakage filter written to results/exp50_stage3/leakage_filter/deeploc_leakage_excluded_seed42.json
  DeepLoc proteins excluded: X / 14004 (Y%)
```

Acceptance: the script ran to completion, produced valid JSON, and the exclusion fraction is < 20% (we expect a small handful of backdoor leaks, not most of DeepLoc).

- [ ] **Step 3: Inspect the filter output**

```bash
uv run python -c "
import json
r = json.load(open('results/exp50_stage3/leakage_filter/deeploc_leakage_excluded_seed42.json'))
print('seed:', r['seed'])
print('threshold:', r['identity_threshold_pct'])
print('cath_h_test:', r['n_cath_h_test'])
print('deeploc total:', r['n_deeploc_total'])
print('excluded:', r['n_deeploc_excluded'])
print('excluded fraction:', f\"{r['fraction_deeploc_excluded']*100:.1f}%\")
print('first 5 excluded ids:', r['excluded_deeploc_ids'][:5])
"
```

- [ ] **Step 4: Commit**

```bash
git add experiments/50_stage3_leakage_filter.py
git commit -m "feat(exp50): DeepLoc leakage filter against CATH H-test for Stage 3"
```

(The JSON output lives under `results/` which is gitignored, so no result file is committed.)

---

### Task 5: Main Stage 3 experiment script

**Files:**
- Create: `experiments/50c_stage3_continuous.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Exp 50 Stage 3: continuous regression with CATH20 + DeepLoc training pool.

Replaces Stage 2's per-bit BCE loss with cosine + MSE regression on the
896d projected ProtT5 vector. Adds DeepLoc (~13K proteins after leakage
filter) as auxiliary training signal. Keeps the same Stage 2 H-split test
set for direct bit-accuracy comparability.

Usage:
    PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py --seed 42
    PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py --seed 42 --smoke-test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe import Seq2OE_CNN, Seq2OEDataset
from src.one_embedding.seq2oe_continuous import (
    prepare_continuous_targets,
    cosine_distance_loss,
    mse_loss,
    evaluate_continuous,
)
from src.one_embedding.seq2oe_splits import (
    parse_cath_fasta,
    cath_cluster_split,
    save_split,
)
from src.utils.device import get_device

DATA = ROOT / "data"
DEEPLOC_FASTA = (
    ROOT / "tools" / "reference" / "LightAttention" /
    "data_files" / "deeploc_complete_dataset.fasta"
)
CATH_FASTA = DATA / "external" / "cath20" / "cath20_labeled.fasta"
CATH_H5 = DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5"
DEEPLOC_H5 = DATA / "residue_embeddings" / "prot_t5_xl_deeploc.h5"

# Training config (matches Stage 2, with cosine-sim early stop)
CONFIG = {
    "hidden": 256,
    "n_layers": 10,
    "epochs": 100,
    "lr": 5e-4,
    "batch_size": 8,
    "weight_decay": 1e-4,
    "patience": 15,
    "d_out": 896,
    "max_len": 512,
    "lambda_cos": 1.0,
    "lambda_mse": 0.1,
}


def ensure_leakage_filter(seed: int, output_root: Path) -> set[str]:
    """Ensure the DeepLoc leakage filter exists for `seed`; return the
    excluded-ID set."""
    filter_file = (
        output_root / "leakage_filter" /
        f"deeploc_leakage_excluded_seed{seed}.json"
    )
    if not filter_file.exists():
        print(f"Leakage filter not found, running now for seed {seed}...")
        cmd = [
            "uv", "run", "python",
            "experiments/50_stage3_leakage_filter.py",
            "--seed", str(seed),
            "--output-root", str(output_root),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        res = subprocess.run(cmd, env=env, cwd=ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"Leakage filter failed with exit {res.returncode}")
    with open(filter_file) as f:
        report = json.load(f)
    return set(report["excluded_deeploc_ids"])


def load_cath20_and_split(seed: int):
    """Load CATH20 sequences, embeddings, and build the H-split at `seed`."""
    print(f"Loading CATH20 FASTA from {CATH_FASTA}...")
    cath_meta = parse_cath_fasta(CATH_FASTA)
    sequences = {pid: info["seq"] for pid, info in cath_meta.items()}

    print(f"Loading CATH20 ProtT5 embeddings from {CATH_H5}...")
    embeddings = {}
    with h5py.File(CATH_H5, "r") as f:
        for pid in f.keys():
            embeddings[pid] = f[pid][:].astype(np.float32)

    common = sorted(set(sequences) & set(embeddings))
    sequences = {k: sequences[k] for k in common}
    embeddings = {k: embeddings[k] for k in common}
    print(f"  {len(common)} CATH proteins with both sequence + embedding")

    filt_meta = {k: cath_meta[k] for k in common}
    train_ids, val_ids, test_ids = cath_cluster_split(
        filt_meta, level="H", fractions=(0.8, 0.1, 0.1), seed=seed,
    )
    print(f"  H-split: {len(train_ids)} train, {len(val_ids)} val, "
          f"{len(test_ids)} test")
    return sequences, embeddings, train_ids, val_ids, test_ids


def load_deeploc(excluded_ids: set[str], smoke_test: bool = False):
    """Load DeepLoc sequences + embeddings, filtering out leakage exclusions."""
    print(f"Loading DeepLoc FASTA from {DEEPLOC_FASTA}...")
    all_seqs = read_fasta(DEEPLOC_FASTA)

    print(f"Loading DeepLoc embeddings from {DEEPLOC_H5}...")
    embeddings = {}
    with h5py.File(DEEPLOC_H5, "r") as f:
        keys = list(f.keys())
        if smoke_test:
            keys = keys[:50]
        for pid in keys:
            embeddings[pid] = f[pid][:].astype(np.float32)

    # Keep only proteins with sequence + embedding + not excluded
    usable = (
        (set(all_seqs) & set(embeddings)) - excluded_ids
    )
    sequences = {k: all_seqs[k] for k in usable}
    embeddings = {k: embeddings[k] for k in usable}
    print(f"  DeepLoc: {len(all_seqs)} total, "
          f"{len(sequences)} usable after leakage filter + seq/emb overlap")
    return sequences, embeddings


def train_continuous(
    sequences_all: dict,
    targets_all: dict,
    train_ids: set,
    val_ids: set,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
    smoke_test: bool = False,
) -> tuple:
    """Train a Seq2OE_CNN with cosine + MSE loss. Returns (model, history,
    elapsed_seconds)."""
    cfg = CONFIG.copy()
    if smoke_test:
        cfg["epochs"] = 2
        cfg["batch_size"] = 4

    # Seed all RNGs BEFORE model instantiation
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"STAGE 3 training (seed={seed})")
    print(f"  hidden={cfg['hidden']}, layers={cfg['n_layers']}")
    print(f"  lambda_cos={cfg['lambda_cos']}, lambda_mse={cfg['lambda_mse']}")
    print(f"{'='*60}")

    model = Seq2OE_CNN(d_out=cfg["d_out"], hidden=cfg["hidden"],
                       n_layers=cfg["n_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model = model.to(device)

    # Build train dataset (all train proteins: CATH20 H-train + DeepLoc filtered)
    train_seqs = {k: sequences_all[k] for k in train_ids if k in targets_all}
    train_tgts = {k: targets_all[k] for k in train_ids if k in targets_all}
    val_seqs = {k: sequences_all[k] for k in val_ids if k in targets_all}
    val_tgts = {k: targets_all[k] for k in val_ids if k in targets_all}

    train_ds = Seq2OEDataset(train_seqs, train_tgts, max_len=cfg["max_len"])
    val_ds = Seq2OEDataset(val_seqs, val_tgts, max_len=cfg["max_len"])

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=False, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    print(f"  Train: {len(train_ds)} proteins, Val: {len(val_ds)} proteins")
    print(f"  Batch size: {cfg['batch_size']}, Max len: {cfg['max_len']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_val_cos_dist = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_cosine_sim": [], "val_mse": [],
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)

            pred = model(ids, mask)
            l_cos = cosine_distance_loss(pred, tgt, mask)
            l_mse = mse_loss(pred, tgt, mask)
            loss = cfg["lambda_cos"] * l_cos + cfg["lambda_mse"] * l_mse

            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        val_cos_sum = 0.0
        val_cos_count = 0
        val_mse_sum = 0.0
        val_mse_count = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                tgt = batch["target"].to(device)
                mask = batch["mask"].to(device)
                pred = model(ids, mask)
                l_cos = cosine_distance_loss(pred, tgt, mask)
                l_mse = mse_loss(pred, tgt, mask)
                loss = cfg["lambda_cos"] * l_cos + cfg["lambda_mse"] * l_mse
                val_losses.append(loss.item())
                # Raw cosine sim / MSE for reporting
                pn = pred.norm(dim=-1).clamp_min(1e-8)
                tn = tgt.norm(dim=-1).clamp_min(1e-8)
                cos = (pred * tgt).sum(dim=-1) / (pn * tn)
                val_cos_sum += (cos * mask).sum().item()
                val_cos_count += mask.sum().item()
                sq = (pred - tgt) ** 2
                val_mse_sum += (sq * mask.unsqueeze(-1)).sum().item()
                val_mse_count += mask.sum().item() * cfg["d_out"]

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_cosine_sim = val_cos_sum / max(val_cos_count, 1.0)
        val_mse_ = val_mse_sum / max(val_mse_count, 1.0)
        val_cos_dist = 1.0 - val_cosine_sim

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cosine_sim"].append(val_cosine_sim)
        history["val_mse"].append(val_mse_)

        if val_cos_dist < best_val_cos_dist:
            best_val_cos_dist = val_cos_dist
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} "
                  f"val_cos_sim={val_cosine_sim:.4f} "
                  f"val_mse={val_mse_:.4f} | "
                  f"{elapsed:.0f}s")

        if epochs_no_improve >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch} "
                  f"(patience={cfg['patience']})")
            break

    # Reload best
    model.load_state_dict(torch.load(
        checkpoint_dir / "best_model.pt", map_location=device, weights_only=True,
    ))
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Stage 3 complete: best val cosine_dist={best_val_cos_dist:.4f} "
          f"@ epoch {best_epoch} ({elapsed:.0f}s)")
    return model, history, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: stage=3 split=h seed={args.seed}")

    load1, load5, _ = os.getloadavg()
    print(f"System load: {load1:.1f} (1m), {load5:.1f} (5m)")
    if load1 > 10:
        print("WARNING: System load >10, consider waiting before training")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Step 1: ensure leakage filter exists
    excluded = ensure_leakage_filter(args.seed, output_root)
    print(f"Leakage filter: {len(excluded)} DeepLoc proteins excluded")

    # Step 2: load CATH20 + build H-split
    cath_seqs, cath_embs, train_ids, val_ids, test_ids = load_cath20_and_split(
        args.seed
    )

    # Step 3: load DeepLoc (filtered)
    deeploc_seqs, deeploc_embs = load_deeploc(excluded, args.smoke_test)

    # Step 4: merge into combined sequences/embeddings dicts
    # (ID spaces are disjoint: CATH uses '12asA00', DeepLoc uses UniProt IDs)
    sequences_all = {**cath_seqs, **deeploc_seqs}
    embeddings_all = {**cath_embs, **deeploc_embs}
    print(f"Combined pool: {len(sequences_all)} total proteins "
          f"(CATH={len(cath_seqs)}, DeepLoc={len(deeploc_seqs)})")

    # Step 5: expand train pool to include DeepLoc filtered
    combined_train_ids = list(set(train_ids) | set(deeploc_seqs.keys()))
    print(f"Combined train pool: {len(combined_train_ids)} proteins "
          f"(CATH train {len(train_ids)} + DeepLoc {len(deeploc_seqs)})")

    # Step 6: fit codec on CATH20 train embeddings only, encode everything
    print(f"\nPreparing continuous targets (train-only codec fit)...")
    t0 = time.time()
    cath_train_embeddings = {pid: cath_embs[pid] for pid in train_ids}
    targets_all = prepare_continuous_targets(
        train_embeddings=cath_train_embeddings,
        all_embeddings=embeddings_all,
        d_out=CONFIG["d_out"],
        seed=args.seed,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    # Sanity on the target dtype
    sample_pid = next(iter(targets_all))
    print(f"  Sample target shape: {targets_all[sample_pid].shape}, "
          f"dtype: {targets_all[sample_pid].dtype}")

    # Step 7: persist the CATH H-split for reproducibility
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"h_seed{args.seed}.json"
    if not split_path.exists():
        save_split(
            split_path, train_ids, val_ids, test_ids,
            {"strategy": "cath_H", "dataset": "cath20", "seed": args.seed,
             "fractions": [0.8, 0.1, 0.1], "level": "H",
             "stage3_deeploc_augmented": True,
             "n_deeploc_added": len(deeploc_seqs)},
        )
        print(f"  Saved split to {split_path}")

    # Step 8: output dir for this run
    run_dir = output_root / "h_split" / "stage3" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 9: train
    model, history, train_seconds = train_continuous(
        sequences_all=sequences_all,
        targets_all=targets_all,
        train_ids=set(combined_train_ids),
        val_ids=set(val_ids),
        device=device,
        seed=args.seed,
        checkpoint_dir=run_dir,
        smoke_test=args.smoke_test,
    )

    # Step 10: evaluate on CATH H-test ONLY (not DeepLoc)
    print(f"\nEvaluating on CATH H-test ({len(test_ids)} proteins)...")
    metrics = evaluate_continuous(
        model=model,
        sequences=sequences_all,
        targets=targets_all,
        ids=set(test_ids),
        device=device,
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
    )
    print(f"  cosine_sim:  {metrics['cosine_sim']:.4f}")
    print(f"  mse:         {metrics['mse']:.4f}")
    print(f"  bit_accuracy: {metrics['bit_accuracy']:.4f}")
    print(f"  dims > 60%:  "
          f"{sum(1 for d in metrics['dim_accuracies'] if d > 0.60)}/"
          f"{len(metrics['dim_accuracies'])}")

    # Step 11: add config + write results.json
    metrics["config"] = {
        "stage": 3,
        "dataset": "cath20+deeploc",
        "split": "h",
        "seed": args.seed,
        "lambda_cos": CONFIG["lambda_cos"],
        "lambda_mse": CONFIG["lambda_mse"],
        "n_train_cath": len(train_ids),
        "n_train_deeploc": len(deeploc_seqs),
        "n_train_combined": len(combined_train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_deeploc_excluded": len(excluded),
    }
    metrics["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    metrics["best_val_loss"] = float(min(history["val_loss"]))
    metrics["best_val_cosine_sim"] = float(max(history["val_cosine_sim"]))
    metrics["train_seconds"] = float(train_seconds)

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test end to end**

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py \
    --seed 42 --smoke-test \
    --output-root /tmp/exp50_stage3_smoke
```

Expected: ~1-2 min. Loads CATH20 (~14K), loads 50 DeepLoc proteins (smoke), runs leakage filter if not cached, trains 2 epochs, evaluates on H-test, writes `/tmp/exp50_stage3_smoke/h_split/stage3/seed42/results.json`. Bit accuracy and cosine sim will be near random (model barely trained), that's fine — we're testing the pipeline runs.

- [ ] **Step 3: Verify smoke output schema**

```bash
uv run python -c "
import json
r = json.load(open('/tmp/exp50_stage3_smoke/h_split/stage3/seed42/results.json'))
print('keys:', sorted(r.keys()))
assert 'cosine_sim' in r
assert 'mse' in r
assert 'bit_accuracy' in r
assert 'dim_accuracies' in r
assert len(r['dim_accuracies']) == 896
assert 'config' in r
assert r['config']['stage'] == 3
assert r['config']['dataset'] == 'cath20+deeploc'
assert 'train_seconds' in r
assert 'best_epoch' in r
print('All required keys present.')
print(f\"cosine_sim: {r['cosine_sim']:.4f}\")
print(f\"bit_accuracy: {r['bit_accuracy']:.4f}\")
print(f\"train_seconds: {r['train_seconds']:.1f}\")
"
```

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf /tmp/exp50_stage3_smoke
git add experiments/50c_stage3_continuous.py
git commit -m "feat(exp50): Stage 3 continuous regression experiment script"
```

---

### Task 6: Runner script for 3-seed Stage 3 sweep

**Files:**
- Create: `experiments/50d_run_stage3.py`

- [ ] **Step 1: Write the runner**

```python
#!/usr/bin/env python3
"""Exp 50 Stage 3 runner: loop over 3 seeds, aggregate per-seed results.

Invokes `experiments/50c_stage3_continuous.py` once per seed (sequential,
no MPS parallelism). After all runs complete, aggregates the per-seed
results.json into a single summary.json and writes final_comparison.{json,md}
that joins Stage 3 with the Stage 1 / Stage 2 numbers from
`results/exp50_rigorous/`.

Usage:
    PYTHONUNBUFFERED=1 uv run python experiments/50d_run_stage3.py
    PYTHONUNBUFFERED=1 uv run python experiments/50d_run_stage3.py --seeds 42 43
    uv run python experiments/50d_run_stage3.py --aggregate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

LOAD_THRESHOLD = 10.0
LOAD_RETRY_INTERVAL_S = 60
LOAD_RETRY_MAX_S = 600


def wait_for_load(threshold: float = LOAD_THRESHOLD) -> bool:
    waited = 0
    while True:
        load1, _, _ = os.getloadavg()
        if load1 <= threshold:
            return True
        if waited >= LOAD_RETRY_MAX_S:
            print(f"[GIVE UP] Load {load1:.1f} > {threshold} after "
                  f"{waited}s of waiting — skipping this run")
            return False
        print(f"[WAIT] Load {load1:.1f} > {threshold}, "
              f"sleeping {LOAD_RETRY_INTERVAL_S}s...")
        time.sleep(LOAD_RETRY_INTERVAL_S)
        waited += LOAD_RETRY_INTERVAL_S


def run_one(seed: int, output_root: Path) -> str:
    """Returns 'ok', 'failed', or 'skipped'."""
    if not wait_for_load():
        return "skipped"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50c_stage3_continuous.py",
        "--seed", str(seed),
        "--output-root", str(output_root),
    ]
    print(f"\n{'='*72}")
    print(f"RUN: stage=3 seed={seed}")
    print(f"{'='*72}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"RUN COMPLETE in {elapsed:.0f}s (exit {res.returncode})")
    return "ok" if res.returncode == 0 else "failed"


def aggregate_seeds(output_root: Path) -> dict | None:
    base = output_root / "h_split" / "stage3"
    if not base.exists():
        return None
    per_seed = []
    for seed_dir in sorted(base.glob("seed*")):
        f = seed_dir / "results.json"
        if not f.exists():
            continue
        with open(f) as fh:
            per_seed.append(json.load(fh))
    if not per_seed:
        return None

    # Dim-length consistency check
    lens = {len(r["dim_accuracies"]) for r in per_seed}
    if len(lens) != 1:
        print(f"  WARNING: dim length mismatch across seeds: {lens}")
        return None

    cos = np.array([r["cosine_sim"] for r in per_seed])
    mse = np.array([r["mse"] for r in per_seed])
    bit = np.array([r["bit_accuracy"] for r in per_seed])
    dim_arrs = np.array([r["dim_accuracies"] for r in per_seed])  # (S, 896)

    intersect_60 = int((dim_arrs > 0.60).all(axis=0).sum())
    intersect_55 = int((dim_arrs > 0.55).all(axis=0).sum())
    mean_dims = dim_arrs.mean(axis=0)
    mean_60 = int((mean_dims > 0.60).sum())
    mean_55 = int((mean_dims > 0.55).sum())

    def pack(arr):
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": arr.tolist(),
        }

    summary = {
        "stage": 3,
        "split": "h",
        "n_seeds": len(per_seed),
        "seeds": [r["config"]["seed"] for r in per_seed],
        "cosine_sim": pack(cos),
        "mse": pack(mse),
        "bit_accuracy": pack(bit),
        "dims_above_60_intersect": intersect_60,
        "dims_above_55_intersect": intersect_55,
        "dims_above_60_mean": mean_60,
        "dims_above_55_mean": mean_55,
        "best_epochs": [r.get("best_epoch") for r in per_seed],
        "train_seconds": [r.get("train_seconds") for r in per_seed],
    }
    summary_path = base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")
    return summary


def load_rigorous_summaries() -> list[dict]:
    """Load the Stage 1/2 summaries from results/exp50_rigorous/ for
    inclusion in the final comparison table. Returns empty list if missing."""
    base = ROOT / "results" / "exp50_rigorous"
    if not base.exists():
        return []
    out = []
    for split in ["h", "t"]:
        for stage in [1, 2]:
            p = base / f"{split}_split" / f"stage{stage}" / "summary.json"
            if p.exists():
                with open(p) as f:
                    out.append(json.load(f))
    return out


def write_final_comparison(stage3_summary: dict, output_root: Path):
    rigorous = load_rigorous_summaries()
    all_summaries = rigorous + [stage3_summary]

    json_path = output_root / "final_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Markdown table
    lines = [
        "# Exp 50 final comparison (Stages 1-3)",
        "",
        "| Stage | Split | Bit acc (mean ± std) | Cosine sim | dims > 60% (intersect) | Seeds |",
        "|:-----:|:-----:|:--------------------:|:----------:|:----------------------:|:-----:|",
    ]
    for s in rigorous:
        stage = s["stage"]
        split = s["split"]
        mean = s["overall_bit_acc"]["mean"] * 100
        std = s["overall_bit_acc"]["std"] * 100
        dims60 = s["dims_above_60_intersect"]
        lines.append(
            f"| {stage} | {split} | {mean:.2f} ± {std:.2f} % "
            f"| — | {dims60} / 896 | {s['n_seeds']} |"
        )
    # Stage 3
    s3 = stage3_summary
    mean3 = s3["bit_accuracy"]["mean"] * 100
    std3 = s3["bit_accuracy"]["std"] * 100
    cos_mean = s3["cosine_sim"]["mean"]
    cos_std = s3["cosine_sim"]["std"]
    dims60 = s3["dims_above_60_intersect"]
    lines.append(
        f"| 3 | h | {mean3:.2f} ± {std3:.2f} % "
        f"| {cos_mean:.4f} ± {cos_std:.4f} "
        f"| {dims60} / 896 | {s3['n_seeds']} |"
    )

    md_path = output_root / "final_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nFinal comparison written to:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    total_wall_clock_start = time.time()
    n_ok, n_failed, n_skipped = 0, 0, 0

    if not args.aggregate_only:
        for seed in args.seeds:
            status = run_one(seed, output_root)
            if status == "ok":
                n_ok += 1
            elif status == "failed":
                n_failed += 1
            else:
                n_skipped += 1

    summary = aggregate_seeds(output_root)
    if summary is not None:
        write_final_comparison(summary, output_root)
    else:
        print("No summary to write — nothing completed successfully.")

    total_elapsed = time.time() - total_wall_clock_start
    print(f"\nStage 3 sweep complete: {n_ok} ok / {n_failed} failed / "
          f"{n_skipped} skipped (total wall-clock: {total_elapsed:.0f}s)")
    sys.exit(0 if n_failed == 0 and n_skipped == 0 else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run aggregate (no training)**

```bash
uv run python experiments/50d_run_stage3.py --aggregate-only
```

Expected:
```
No summary to write — nothing completed successfully.
Stage 3 sweep complete: 0 ok / 0 failed / 0 skipped (total wall-clock: ...s)
```
Exit 0 (nothing failed, nothing skipped).

- [ ] **Step 3: Verify --help shows all three flags**

```bash
uv run python experiments/50d_run_stage3.py --help
```

Expected: shows `--seeds`, `--output-root`, `--aggregate-only`.

- [ ] **Step 4: Commit**

```bash
git add experiments/50d_run_stage3.py
git commit -m "feat(exp50): runner for 3-seed Stage 3 sweep with aggregation"
```

---

### Task 7: Single-seed Stage 3 pilot run

**Files:** (none modified)

- [ ] **Step 1: Verify system load**

Run: `uptime`
Expected: load1 < 8. If higher, wait.

- [ ] **Step 2: Launch Stage 3 seed 42 pilot**

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py \
    --seed 42 \
    > results/exp50_stage3/pilot_stage3_seed42.log 2>&1
```

(Foreground — roughly 60–90 minutes expected. If running via a subagent that
times out, split into background + poll.)

- [ ] **Step 3: Inspect pilot result**

```bash
uv run python -c "
import json
r = json.load(open('results/exp50_stage3/h_split/stage3/seed42/results.json'))
print(f\"cosine_sim:     {r['cosine_sim']:.4f}\")
print(f\"mse:            {r['mse']:.4f}\")
print(f\"bit_accuracy:   {r['bit_accuracy']:.4f}\")
print(f\"best_epoch:     {r['best_epoch']}\")
print(f\"best_val_cos:   {r['best_val_cosine_sim']:.4f}\")
print(f\"train_seconds:  {r['train_seconds']:.0f}\")
import numpy as np
dims = np.array(r['dim_accuracies'])
print(f\"dims > 60%:     {(dims > 0.60).sum()} / 896\")
print(f\"config:         {r['config']}\")
"
```

**Acceptance:**
- Bit accuracy > 0.693 (beats Stage 2 seed 42 = 0.6927). A clean win is ≥ 0.703.
- Cosine similarity > 0.5 (much higher than random would be, since cosine on 896d random vectors is ~0 ± 1/sqrt(896)).
- No crashes during training.
- `best_epoch` is a real integer (not 1 — that would suggest early-stop on the first epoch, which means val is getting worse immediately).

If bit accuracy is LOWER than Stage 2 (< 0.693), do not launch the full sweep — stop and report back. Something is wrong and debugging is needed before burning compute on 2 more seeds.

If bit accuracy is between 0.693 and 0.703 (small improvement, not clean), the sweep is still worth running to get variance bounds, but flag the small gain in the report.

- [ ] **Step 4: No commit for this task**

The per-run artifacts under `results/` are gitignored. Task 7 has no code changes, only a verification run.

---

### Task 8: Full Stage 3 sweep + aggregation + memory update

**Files:**
- Modify: `~/.claude/projects/-Users-jcoludar-CascadeProjects-ProteEmbedExplorations/memory/project_exp50_seq2oe.md`
- Modify: `docs/superpowers/specs/2026-04-07-exp50-stage3-continuous-regression-design.md` (mark status Done)

- [ ] **Step 1: Launch the full sweep in the background**

The pilot (Task 7) already produced seed 42. Running the full sweep again
will overwrite it, which is fine for uniformity. Alternative: pass
`--seeds 43 44` to run only the remaining two.

Choose based on pilot outcome:
- Pilot bit acc ≥ 0.703 → use `--seeds 43 44` (save ~70 min)
- Pilot bit acc between 0.693 and 0.703 → use `--seeds 42 43 44` (re-run seed 42 for fresh logs since we may need to inspect all three uniformly)

Launch:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50d_run_stage3.py \
    --seeds 42 43 44 \
    > results/exp50_stage3/sweep.log 2>&1 &
echo "Launched PID $!"
```

- [ ] **Step 2: Poll sweep progress at reasonable intervals**

```bash
tail -30 results/exp50_stage3/sweep.log
ls results/exp50_stage3/h_split/stage3/seed*/results.json 2>/dev/null | wc -l
ps -ef | grep 50c_stage3 | grep -v grep
```

Expected final state: 3 results.json files. Do not poll more than once every 15 minutes during the sweep.

- [ ] **Step 3: Verify sweep success**

Once all 3 runs are done, the runner prints:
```
Stage 3 sweep complete: 3 ok / 0 failed / 0 skipped (total wall-clock: ...s)
```

And `results/exp50_stage3/final_comparison.md` has a 5-row table (Stages 1-2 H+T from the prior sweep, plus the Stage 3 row).

- [ ] **Step 4: Inspect the aggregated numbers**

```bash
cat results/exp50_stage3/final_comparison.md
echo "---summary---"
uv run python -c "
import json
s = json.load(open('results/exp50_stage3/h_split/stage3/summary.json'))
print(f\"cosine_sim:     {s['cosine_sim']['mean']:.4f} ± {s['cosine_sim']['std']:.4f}\")
print(f\"mse:            {s['mse']['mean']:.4f} ± {s['mse']['std']:.4f}\")
print(f\"bit_accuracy:   {s['bit_accuracy']['mean']*100:.2f} ± {s['bit_accuracy']['std']*100:.2f} %\")
print(f\"dims > 60 intersect: {s['dims_above_60_intersect']} / 896\")
print(f\"dims > 60 mean:      {s['dims_above_60_mean']} / 896\")
print(f\"seeds: {s['seeds']}\")
print(f\"train_seconds: {s['train_seconds']}\")
"
```

- [ ] **Step 5: Apply the decision tree from the spec**

Per spec §Success criteria:
- **If bit acc mean ≥ Stage 2's 69.28% + 1.0 pp = 70.28% AND std tight (< 0.1 pp):** Declare Stage 3 a clean win. Proceed to next phase (Stage 4 or downstream retention measurement).
- **If bit acc < 70.28% (flat or small improvement):** Schedule the CATH20-only ablation as a follow-up experiment to isolate the loss vs data levers.
- **If bit acc regression (< 69.28%):** Debug loss weighting first (drop `λ_mse` to 0.01 or 0), then codec-fit path, then learning rate.

Record the decision in the memory update.

- [ ] **Step 6: Update the Exp 50 memory file**

Append a new section to
`~/.claude/projects/-Users-jcoludar-CascadeProjects-ProteEmbedExplorations/memory/project_exp50_seq2oe.md`:

Use the actual numbers from Step 4. The section structure should be:

```markdown
## Stage 3 — continuous regression + DeepLoc augmentation (2026-04-07)

Switched BCE → cosine + MSE regression on 896d projected target, added
~13K DeepLoc proteins to training pool (leakage-filtered at 30% identity
vs CATH H-test), same Stage 2 architecture (10-layer dilated CNN, 4.2M
params), same CATH H-split test set, 3 seeds.

| Metric | Value (mean ± std across 3 seeds) |
|---|---|
| cosine_sim | <FILL_IN> |
| mse | <FILL_IN> |
| bit_accuracy | <FILL_IN> |
| dims > 60% (intersect) | <FILL_IN> / 896 |

Delta vs Stage 2 H (69.28 ± 0.02%): <FILL_IN> pp

**Decision:** <win / ablate / debug>. <one-sentence reasoning>.

Files:
- Spec: docs/superpowers/specs/2026-04-07-exp50-stage3-continuous-regression-design.md
- Plan: docs/superpowers/plans/2026-04-07-exp50-stage3-continuous-regression.md
- Module: src/one_embedding/seq2oe_continuous.py
- Main script: experiments/50c_stage3_continuous.py
- Runner: experiments/50d_run_stage3.py
- Leakage filter: experiments/50_stage3_leakage_filter.py
- Results: results/exp50_stage3/h_split/stage3/seed{42,43,44}/results.json + summary.json + final_comparison.md
```

Fill in the `<FILL_IN>` placeholders with actual numbers from Step 4.

- [ ] **Step 7: Mark the spec Done**

Edit `docs/superpowers/specs/2026-04-07-exp50-stage3-continuous-regression-design.md`. Change the `**Status:**` line at the top from:

```
**Status:** Approved — proceeding to implementation plan
```

to (using actual numbers):

```
**Status:** Done — see `results/exp50_stage3/final_comparison.md`. Stage 3: cosine_sim <X.XXXX ± X.XXXX>, bit_accuracy <XX.XX ± X.XX %> (3 seeds on CATH H-split test).
```

- [ ] **Step 8: Commit the status/memory changes**

```bash
git add docs/superpowers/specs/2026-04-07-exp50-stage3-continuous-regression-design.md
git commit -m "docs(exp50): mark Stage 3 spec as Done with headline numbers"
```

The memory file lives outside the repo, so it's already saved from Step 6 and does not need `git add`.

---

### Task 9: RNS (Random Neighbor Score) evaluation

**Reference:** Prabakaran & Bromberg, "Quantifying uncertainty in protein representations across models and tasks", Nature Methods 23, 796–804 (April 2026). doi:10.1038/s41592-026-03028-7.

**What RNS is:** For each protein, find its k nearest neighbors in embedding space (using a combined index of real + randomly-shuffled "junkyard" sequences). RNS = fraction of those k neighbors that are junkyard. High RNS = the embedding is indistinguishable from random noise (low confidence). Low RNS = the embedding sits in a biologically structured region (high confidence). Model-agnostic, task-agnostic, biologically grounded.

**Why it matters for us:** Bit accuracy and cosine similarity tell us how well the predicted embedding MATCHES ProtT5. RNS tells us whether the predicted embedding is BIOLOGICALLY MEANINGFUL. A model could achieve decent bit accuracy while placing proteins in the "junkyard" region of latent space — RNS catches that failure mode. It also directly probes the disorder retention gap (the paper shows IDPs have higher RNS across all PLMs — the signal is inherently uncertain for disordered proteins, which explains our persistent ~95% retention on disorder).

**Files:**
- Create: `src/one_embedding/rns.py`
- Create: `tests/test_rns.py`
- Create: `experiments/50_rns_evaluation.py`

- [ ] **Step 1: Generate shuffled junkyard sequences**

Write a helper `generate_junkyard_fasta(sequences, n_shuffles=5, seed=42)` in `src/one_embedding/rns.py` that takes a dict of `{pid: sequence}` and returns a dict of `{"{pid}_shuf{i}": shuffled_sequence}` for i in range(n_shuffles). Each shuffled sequence is a random permutation of the original's residues. Seed for reproducibility.

```python
# src/one_embedding/rns.py
"""Random Neighbor Score (RNS) for embedding quality evaluation.

Implements the RNS metric from Prabakaran & Bromberg (Nat Methods 2026):
for each protein, RNS_k = fraction of k nearest neighbors in the
embedding space that are randomly-shuffled (non-biological) sequences.

RNS ∈ [0, 1]. Lower = higher confidence (embedding is in a biologically
structured region). Higher = lower confidence (embedding is
indistinguishable from random noise).
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def generate_junkyard_sequences(
    sequences: dict[str, str],
    n_shuffles: int = 5,
    seed: int = 42,
) -> dict[str, str]:
    """Generate residue-shuffled copies of each sequence.

    For each protein, create n_shuffles random permutations of its amino
    acid sequence. The shuffled sequences have the same composition but
    no biological order — they serve as the 'junkyard' reference for RNS.

    Args:
        sequences: {pid: aa_sequence} real protein sequences.
        n_shuffles: Number of shuffled copies per protein.
        seed: RNG seed for reproducibility.

    Returns:
        {"{pid}_shuf{i}": shuffled_sequence} for all proteins and copies.
    """
    rng = random.Random(seed)
    junkyard: dict[str, str] = {}
    for pid in sorted(sequences):  # sorted for determinism
        residues = list(sequences[pid])
        for i in range(n_shuffles):
            shuffled = residues.copy()
            rng.shuffle(shuffled)
            junkyard[f"{pid}_shuf{i}"] = "".join(shuffled)
    return junkyard


def compute_rns(
    query_vectors: dict[str, np.ndarray],
    real_vectors: dict[str, np.ndarray],
    junkyard_vectors: dict[str, np.ndarray],
    k: int = 1000,
) -> dict[str, float]:
    """Compute RNS_k for each query protein.

    Builds a FAISS index over the union of real + junkyard protein vectors,
    then for each query finds its k nearest neighbors and reports the
    fraction that are junkyard.

    Args:
        query_vectors: {pid: (D,) float32} proteins to score. Typically
            the test set, embedded by the model being evaluated.
        real_vectors: {pid: (D,) float32} biologically real protein vectors
            (the 'anchor' set — typically the same test proteins embedded
            by ProtT5 or another trusted source, PLUS the training set).
        junkyard_vectors: {pid: (D,) float32} shuffled-sequence embeddings.
        k: Number of nearest neighbors. Paper recommends k > 100.

    Returns:
        {pid: rns_score} for each query. rns ∈ [0, 1].
    """
    import faiss

    # Build combined index: real + junkyard
    all_ids = []
    is_junkyard = []
    vecs = []
    for pid, vec in real_vectors.items():
        all_ids.append(pid)
        is_junkyard.append(False)
        vecs.append(vec.astype(np.float32))
    for pid, vec in junkyard_vectors.items():
        all_ids.append(pid)
        is_junkyard.append(True)
        vecs.append(vec.astype(np.float32))

    matrix = np.stack(vecs)  # (N, D)
    d = matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(matrix)

    is_junk_arr = np.array(is_junkyard)  # (N,) bool

    # Query
    query_ids = sorted(query_vectors.keys())
    query_mat = np.stack([query_vectors[pid].astype(np.float32)
                          for pid in query_ids])

    # k+1 because the query itself may be in the index (self-match)
    _, indices = index.search(query_mat, k + 1)

    scores: dict[str, float] = {}
    for i, pid in enumerate(query_ids):
        neighbors = indices[i]
        # Exclude self-match if present
        neighbor_mask = np.array([all_ids[j] != pid for j in neighbors])
        valid_neighbors = neighbors[neighbor_mask][:k]
        if len(valid_neighbors) == 0:
            scores[pid] = 1.0  # no valid neighbors → worst case
            continue
        n_junk = is_junk_arr[valid_neighbors].sum()
        scores[pid] = float(n_junk / len(valid_neighbors))

    return scores
```

- [ ] **Step 2: Write tests for junkyard generation + RNS computation**

```python
# tests/test_rns.py
"""Tests for RNS (Random Neighbor Score) evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.rns import generate_junkyard_sequences, compute_rns


class TestGenerateJunkyardSequences:
    def test_count_and_naming(self):
        seqs = {"p1": "ACDEF", "p2": "GHIKLMN"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=3, seed=42)
        assert len(junk) == 6  # 2 proteins × 3 shuffles
        assert "p1_shuf0" in junk
        assert "p1_shuf2" in junk
        assert "p2_shuf0" in junk

    def test_same_composition(self):
        seqs = {"p1": "AACDEF"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=5, seed=42)
        for pid, seq in junk.items():
            assert sorted(seq) == sorted("AACDEF")

    def test_deterministic(self):
        seqs = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
        j1 = generate_junkyard_sequences(seqs, seed=42)
        j2 = generate_junkyard_sequences(seqs, seed=42)
        assert j1 == j2

    def test_different_from_original(self):
        # With a 20-residue sequence, random shuffle ≠ original with high probability
        seqs = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=5, seed=42)
        n_different = sum(1 for s in junk.values() if s != seqs["p1"])
        assert n_different >= 4  # at least 4 of 5 should differ


class TestComputeRNS:
    def test_perfect_real_neighbors_gives_zero(self):
        """If a query's neighbors are all real proteins, RNS = 0."""
        d = 16
        rng = np.random.RandomState(0)
        # 10 real proteins in a tight cluster
        real = {f"r{i}": rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # 10 junkyard proteins FAR away
        junk = {f"j{i}": (rng.randn(d).astype(np.float32) + 100.0)
                for i in range(10)}
        # Query = one of the real proteins
        query = {"r0": real["r0"]}
        scores = compute_rns(query, real, junk, k=5)
        assert scores["r0"] < 0.01  # essentially 0

    def test_junkyard_neighbor_gives_high_rns(self):
        """If a query sits among junkyard vectors, RNS ≈ 1."""
        d = 16
        rng = np.random.RandomState(0)
        # 10 real proteins in one region
        real = {f"r{i}": rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # 10 junkyard in another region
        junk_center = rng.randn(d).astype(np.float32) * 0.1 + 50.0
        junk = {f"j{i}": junk_center + rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # Query sits right in the junkyard cluster
        query = {"q": junk_center.copy()}
        scores = compute_rns(query, real, junk, k=5)
        assert scores["q"] > 0.8

    def test_returns_score_for_every_query(self):
        d = 8
        rng = np.random.RandomState(0)
        real = {f"r{i}": rng.randn(d).astype(np.float32) for i in range(5)}
        junk = {f"j{i}": rng.randn(d).astype(np.float32) for i in range(5)}
        query = {f"r{i}": real[f"r{i}"] for i in range(3)}
        scores = compute_rns(query, real, junk, k=3)
        assert set(scores.keys()) == {"r0", "r1", "r2"}
        for v in scores.values():
            assert 0.0 <= v <= 1.0
```

- [ ] **Step 3: Run tests — expect pass**

Run: `uv run pytest tests/test_rns.py -v`
Expected: 7 passed.

- [ ] **Step 4: Commit the RNS module + tests**

```bash
git add src/one_embedding/rns.py tests/test_rns.py
git commit -m "feat(exp50): RNS (Random Neighbor Score) module for embedding quality evaluation

Implements the RNS metric from Prabakaran & Bromberg (Nat Methods 2026):
for each protein, RNS_k = fraction of k nearest neighbors that are
randomly-shuffled junkyard sequences. Model-agnostic embedding quality
metric that detects whether embeddings are biologically meaningful."
```

- [ ] **Step 5: Write the RNS evaluation script**

Create `experiments/50_rns_evaluation.py`:

This script computes RNS for multiple embedding sources on the CATH20 H-test
set, using shuffled CATH20 test sequences as the junkyard. Steps:

1. Load CATH20 H-test protein IDs + sequences (from the seed-42 H-split)
2. Generate 5 shuffled copies per test protein (7,220 junkyard sequences)
3. Check if junkyard ProtT5 embeddings exist at
   `data/residue_embeddings/prot_t5_xl_cath20_junkyard.h5`; if not, run
   the ProtT5 extraction script on the junkyard FASTA (**~1-3h MPS**)
4. Pool all embeddings to protein-level vectors (mean-pool across residues)
5. For each embedding source (raw ProtT5, Stage 2 binary OE decoded,
   Stage 3 continuous OE, seq2oe Stage 2 predicted, seq2oe Stage 3
   predicted), compute RNS at k = 100, 500, 1000
6. Report RNS distributions: mean, median, fraction with RNS=0, fraction
   with RNS > 0.5
7. Write `results/exp50_rns/rns_comparison.json` + `.md`

CLI: `--seed INT` (default 42), `--k INT` (default 1000),
     `--output-root PATH` (default `results/exp50_rns`),
     `--skip-extraction` (use cached junkyard embeddings only, don't run ProtT5)

**Implementation note:** the junkyard ProtT5 extraction is the only expensive
step (~1-3h on MPS for ~7K short sequences). Once cached, subsequent RNS
evaluations for new model checkpoints reuse the same junkyard and cost
seconds. The script should cache the junkyard FASTA and embeddings H5
aggressively so we never re-extract.

**What RNS evaluates (the comparison table the script produces):**

| Source | Description | Expected RNS |
|---|---|---|
| Raw ProtT5 | Baseline: ProtT5 per-residue embeddings, mean-pooled | Low (~0.05–0.10 per paper Fig 4a) |
| OE binary 896d decoded | Stage 2's binary codec, decoded back to float | Slightly higher than raw (compression adds noise) |
| OE Stage 3 continuous | Stage 3's continuous model output, mean-pooled | Similar to raw if the model works well |
| Seq2OE Stage 2 predicted | Stage 2 CNN's test-set predictions, mean-pooled | Unknown — the key new number |
| Seq2OE Stage 3 predicted | Stage 3 CNN's test-set predictions, mean-pooled | Unknown — should be better than Stage 2 if continuous helps |

This task does NOT write the full script (that depends on which model checkpoints
exist and what the ProtT5 extraction pipeline looks like). Instead, it:

1. Creates `src/one_embedding/rns.py` with the core logic (done in Steps 1-4)
2. Documents the evaluation script's design here so the implementer can build it
   once Stage 3 training results are available.

The actual evaluation script (`experiments/50_rns_evaluation.py`) should be written
AFTER Stage 3 training completes and we have checkpoints to evaluate.

- [ ] **Step 6: Run the evaluation (after Stage 3 training is done)**

Pre-requisites:
- Stage 2 checkpoints at `results/exp50_rigorous/h_split/stage2/seed42/best_model.pt` (already exist)
- Stage 3 checkpoints at `results/exp50_stage3/h_split/stage3/seed42/best_model.pt` (from Task 7)
- Junkyard ProtT5 embeddings cached (generated once, ~1-3h)

Run:
```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_rns_evaluation.py --seed 42
```

Expected output: a comparison table with RNS at k=1000 for each embedding source.
The headline number: **does Seq2OE Stage 3's RNS match raw ProtT5?**

- [ ] **Step 7: Save RNS results to the Stage 3 memory entry**

Append the RNS comparison table to the Stage 3 section of
`project_exp50_seq2oe.md`, alongside the cosine_sim / bit_accuracy / mse numbers.

---

## Self-review notes (checked against spec)

- ✅ Dataset loader uses `prot_t5_xl_cath20.h5` + `cath20_labeled.fasta` AND `prot_t5_xl_deeploc.h5` + `tools/reference/LightAttention/data_files/deeploc_complete_dataset.fasta` (spec §Training data)
- ✅ Continuous targets via `prepare_continuous_targets`, codec fit on CATH20 train only (spec §Targets, Task 1)
- ✅ DeepLoc re-projected through the same CATH-train-fit codec (Task 5 `prepare_continuous_targets` is called once on `embeddings_all` with `train_embeddings=cath_train_embeddings`)
- ✅ Leakage filter at 30% identity threshold, per-seed, saved to `leakage_filter/deeploc_leakage_excluded_seed{seed}.json` (spec §Training data, Task 4)
- ✅ Loss = λ_cos · cosine_distance + λ_mse · MSE with λ_cos=1.0, λ_mse=0.1 (Task 5 config, Task 2 implementations)
- ✅ Early stopping on val cosine distance, patience 15 (Task 5 training loop)
- ✅ Seeds [42, 43, 44], all RNGs seeded before model instantiation (Task 5 `train_continuous`)
- ✅ PYTHONUNBUFFERED=1 in all invocations (Tasks 4, 5, 6, 7, 8)
- ✅ Metrics: cosine_sim, cosine_distance, mse, bit_accuracy, dim_accuracies, per_protein_bit_acc stats (Task 3 `evaluate_continuous`)
- ✅ Outputs under `results/exp50_stage3/` with the tree from spec §Outputs
- ✅ Reuses `Seq2OE_CNN` and `Seq2OEDataset` unchanged (Seq2OEDataset already handles float32 targets via line 175 cast)
- ✅ No changes to `seq2oe_splits.py`, `seq2oe.py`, or `50_sequence_to_oe.py` — Stage 3 is fully additive
- ✅ `n_test`, `n_train_cath`, `n_train_deeploc`, `n_train_combined`, `n_deeploc_excluded` in results.json config (Task 5 main())
- ✅ Success-criteria decision tree applied in Task 8 Step 5

No unresolved spec items. No placeholder tasks in the implementation plan itself.
