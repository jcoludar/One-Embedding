# Exp 50: Sequence-to-One-Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a lightweight CNN to predict One Embedding binary representations (896 sign bits per residue) directly from amino acid sequence, bypassing ProtT5.

**Architecture:** One-hot amino acid input (L, 21) -> dilated 1D CNN -> (L, 896) binary logits. Progressive stages: Stage 1 baseline CNN (~2M params), Stage 2 deeper ResNet (~10M), Stage 3 hybrid CNN+attention (if needed). Binary cross-entropy loss on 896 bits per residue.

**Tech Stack:** PyTorch (MPS), NumPy, h5py, scikit-learn (probes). Existing codec (`OneEmbeddingCodec`) for target generation and benchmark infra from Exp 43/47.

---

### Task 1: Create the Seq2OE model architecture

**Files:**
- Create: `src/one_embedding/seq2oe.py`
- Test: `tests/test_seq2oe_model.py`

- [ ] **Step 1: Write failing test for Stage 1 CNN**

```python
# tests/test_seq2oe_model.py
"""Tests for Seq2OE sequence-to-binary-embedding models."""

import pytest
import torch

from src.one_embedding.seq2oe import Seq2OE_CNN, AA_VOCAB_SIZE


class TestSeq2OE_CNN:
    def test_output_shape(self):
        model = Seq2OE_CNN(d_out=896)
        # Batch of 2 proteins, length 50, one-hot encoded
        x = torch.randint(0, AA_VOCAB_SIZE, (2, 50))  # integer-encoded
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
        # Masked positions should be zeroed
        assert logits[0, 15, :].abs().sum() == 0.0

    def test_param_count_stage1(self):
        model = Seq2OE_CNN(d_out=896, hidden=128, n_layers=5)
        n_params = sum(p.numel() for p in model.parameters())
        # Stage 1: should be roughly 1-3M params
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seq2oe_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.one_embedding.seq2oe'`

- [ ] **Step 3: Write the Stage 1 CNN model**

```python
# src/one_embedding/seq2oe.py
"""Sequence-to-One-Embedding: predict binary OE from amino acid sequence.

Bypasses the PLM entirely. Trains on (sequence, binary_OE_target) pairs
where targets come from OneEmbeddingCodec applied to ProtT5 embeddings.

Stage 1: Dilated 1D CNN (~2M params, ~63 residue receptive field)
"""

import torch
import torch.nn as nn
from torch import Tensor

# Standard 20 amino acids + X (unknown) + padding token
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_VOCAB_SIZE = len(AA_ALPHABET) + 1  # +1 for padding (index 0)
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0 = padding


def encode_sequence(seq: str) -> list[int]:
    """Convert amino acid string to integer indices.

    Unknown residues (U, Z, O, B, etc.) map to X.
    """
    return [AA_TO_IDX.get(aa, AA_TO_IDX["X"]) for aa in seq.upper()]


class DilatedResBlock(nn.Module):
    """Single dilated conv residual block."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class Seq2OE_CNN(nn.Module):
    """Dilated CNN: amino acid sequence -> binary OE logits.

    Input: integer-encoded sequence (B, L) + mask (B, L)
    Output: (B, L, d_out) logits for binary bits
    """

    def __init__(
        self,
        d_out: int = 896,
        hidden: int = 128,
        n_layers: int = 5,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.d_out = d_out

        # Embedding: integer AA -> hidden dim
        self.embed = nn.Embedding(AA_VOCAB_SIZE, hidden, padding_idx=0)

        # Dilated residual blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            DilatedResBlock(hidden, dilation=2**i, kernel_size=kernel_size)
            for i in range(n_layers)
        ])

        # Output projection: hidden -> d_out logits
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: (B, L) integer-encoded amino acids
            mask: (B, L) float, 1.0 for real positions, 0.0 for padding

        Returns:
            (B, L, d_out) logits for each binary bit
        """
        # (B, L) -> (B, L, hidden)
        h = self.embed(x)

        # Conv expects (B, C, L)
        h = h.transpose(1, 2)  # (B, hidden, L)

        for block in self.blocks:
            h = h * mask.unsqueeze(1)  # zero out padding before each block
            h = block(h)

        h = h * mask.unsqueeze(1)  # final mask
        h = h.transpose(1, 2)  # (B, L, hidden)

        logits = self.head(h)  # (B, L, d_out)
        logits = logits * mask.unsqueeze(-1)  # zero padding positions
        return logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seq2oe_model.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe.py tests/test_seq2oe_model.py
git commit -m "feat(exp50): add Seq2OE_CNN model — sequence to binary OE"
```

---

### Task 2: Create the Seq2OE dataset and target preparation

**Files:**
- Modify: `src/one_embedding/seq2oe.py` (add dataset class + target prep)
- Test: `tests/test_seq2oe_model.py` (add dataset tests)

- [ ] **Step 1: Write failing tests for dataset and target prep**

Add to `tests/test_seq2oe_model.py`:

```python
import numpy as np
from src.one_embedding.seq2oe import (
    Seq2OE_CNN, AA_VOCAB_SIZE, encode_sequence,
    Seq2OEDataset, prepare_binary_targets,
)


class TestEncodeSequence:
    def test_standard_aas(self):
        indices = encode_sequence("ACDEF")
        assert len(indices) == 5
        assert all(i > 0 for i in indices)  # no padding

    def test_unknown_aa(self):
        indices = encode_sequence("ACUXZ")
        # U, Z should map to X's index
        x_idx = indices[2]  # X
        assert indices[3] == x_idx  # U -> X... wait
        # Actually U maps to X via get default
        # Let's just check they're all valid
        assert all(1 <= i <= AA_VOCAB_SIZE - 1 for i in indices)


class TestPrepareBinaryTargets:
    def test_shape(self):
        # Fake ProtT5 embeddings: 3 proteins
        embeddings = {
            "p1": np.random.randn(50, 1024).astype(np.float32),
            "p2": np.random.randn(30, 1024).astype(np.float32),
            "p3": np.random.randn(80, 1024).astype(np.float32),
        }
        targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
        assert targets["p1"].shape == (50, 896)
        assert targets["p2"].shape == (30, 896)
        assert targets["p3"].shape == (80, 896)

    def test_binary_values(self):
        embeddings = {
            "p1": np.random.randn(50, 1024).astype(np.float32),
        }
        targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
        unique = np.unique(targets["p1"])
        assert set(unique).issubset({0, 1}), f"Expected 0/1, got {unique}"

    def test_deterministic(self):
        embeddings = {
            "p1": np.random.randn(50, 1024).astype(np.float32),
        }
        t1 = prepare_binary_targets(embeddings, d_out=896, seed=42)
        t2 = prepare_binary_targets(embeddings, d_out=896, seed=42)
        np.testing.assert_array_equal(t1["p1"], t2["p1"])


class TestSeq2OEDataset:
    def test_basic(self):
        sequences = {"p1": "ACDEF", "p2": "GHIKLMN"}
        targets = {
            "p1": np.random.randint(0, 2, (5, 896)).astype(np.float32),
            "p2": np.random.randint(0, 2, (7, 896)).astype(np.float32),
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
        targets = {"p1": np.zeros((100, 896), dtype=np.float32)}
        ds = Seq2OEDataset(sequences, targets, max_len=50)
        item = ds[0]
        assert item["length"] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seq2oe_model.py::TestPrepareBinaryTargets -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement target preparation and dataset**

Add to `src/one_embedding/seq2oe.py`:

```python
import numpy as np
from torch.utils.data import Dataset

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.one_embedding.preprocessing import compute_corpus_stats, center_embeddings


def prepare_binary_targets(
    embeddings: dict[str, np.ndarray],
    d_out: int = 896,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Convert raw PLM embeddings to binary OE targets.

    Runs the codec pipeline (center + RP + sign) and extracts the
    pre-quantization binary bits as 0/1 arrays.

    Args:
        embeddings: {protein_id: (L, D) float32} raw PLM embeddings.
        d_out: Random projection target dimension.
        seed: Seed for deterministic RP matrix.

    Returns:
        {protein_id: (L, d_out) uint8} binary targets (0 or 1).
    """
    # Fit corpus stats for centering
    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(embeddings)

    targets = {}
    for pid, raw in embeddings.items():
        projected = codec._preprocess(raw)
        # Binary target: sign(projected - per-channel-mean) > 0
        means = projected.mean(axis=0)
        centered = projected - means[np.newaxis, :]
        bits = (centered > 0).astype(np.uint8)
        targets[pid] = bits

    return targets


class Seq2OEDataset(Dataset):
    """Dataset of (sequence, binary_target) pairs for Seq2OE training."""

    def __init__(
        self,
        sequences: dict[str, str],
        targets: dict[str, np.ndarray],
        max_len: int = 512,
    ):
        # Only keep proteins present in both
        common = sorted(set(sequences) & set(targets))
        self.ids = common
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        seq = self.sequences[pid]
        target = self.targets[pid]  # (L, d_out) uint8

        L = min(len(seq), target.shape[0], self.max_len)
        seq = seq[:L]
        target = target[:L]

        # Encode sequence to integer indices
        input_ids = torch.tensor(encode_sequence(seq), dtype=torch.long)

        # Pad
        padded_ids = torch.zeros(self.max_len, dtype=torch.long)
        padded_ids[:L] = input_ids

        padded_target = torch.zeros(self.max_len, target.shape[1], dtype=torch.float32)
        padded_target[:L] = torch.from_numpy(target.astype(np.float32))

        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[:L] = 1.0

        return {
            "id": pid,
            "input_ids": padded_ids,
            "target": padded_target,
            "mask": mask,
            "length": L,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seq2oe_model.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe.py tests/test_seq2oe_model.py
git commit -m "feat(exp50): add Seq2OE dataset + binary target preparation"
```

---

### Task 3: Write the Exp 50 experiment script — Stage 1 training

**Files:**
- Create: `experiments/50_sequence_to_oe.py`

- [ ] **Step 1: Write the experiment script**

```python
#!/usr/bin/env python3
"""Experiment 50: Sequence to One Embedding — predict binary OE from FASTA.

Can a lightweight CNN predict the 896 binary bits that OneEmbeddingCodec
produces from ProtT5, using only the amino acid sequence as input?

Progressive stages:
  Stage 1: Dilated CNN (~2M params, 5 layers, 128 hidden)
  Stage 2: Deeper ResNet (~10M params, 10 layers, 256 hidden)
  Stage 3: Hybrid CNN + Attention (if CNN plateaus)

Usage:
    # Smoke test (tiny data, 2 epochs)
    uv run python experiments/50_sequence_to_oe.py --smoke-test

    # Stage 1 baseline
    uv run python experiments/50_sequence_to_oe.py --stage 1

    # Stage 2 deeper
    uv run python experiments/50_sequence_to_oe.py --stage 2

    # Stage 1 + downstream evaluation
    uv run python experiments/50_sequence_to_oe.py --stage 1 --eval
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe import (
    Seq2OE_CNN, Seq2OEDataset, prepare_binary_targets, encode_sequence,
)
from src.utils.device import get_device

DATA = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "exp50"


def load_data(smoke_test: bool = False):
    """Load sequences + ProtT5 embeddings, return train/val/test splits."""
    fasta_path = DATA / "proteins" / "medium_diverse_5k.fasta"
    h5_path = DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5"

    print(f"Loading sequences from {fasta_path}")
    sequences = read_fasta(fasta_path)

    print(f"Loading ProtT5 embeddings from {h5_path}")
    embeddings = {}
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if smoke_test:
            keys = keys[:50]
        for pid in keys:
            embeddings[pid] = f[pid][:].astype(np.float32)

    # Filter to common keys
    common = sorted(set(sequences) & set(embeddings))
    sequences = {k: sequences[k] for k in common}
    embeddings = {k: embeddings[k] for k in common}
    print(f"  {len(common)} proteins with both sequence + embedding")

    # Split: 80/10/10
    rng = np.random.RandomState(42)
    ids = np.array(common)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    return sequences, embeddings, train_ids, val_ids, test_ids


def train_stage(
    stage: int,
    sequences: dict,
    targets: dict,
    train_ids: set,
    val_ids: set,
    device: torch.device,
    smoke_test: bool = False,
):
    """Train a Seq2OE model for a given stage."""
    # Stage configs
    configs = {
        1: {"hidden": 128, "n_layers": 5, "epochs": 50, "lr": 1e-3, "batch_size": 16},
        2: {"hidden": 256, "n_layers": 10, "epochs": 100, "lr": 5e-4, "batch_size": 8},
    }
    cfg = configs[stage]
    if smoke_test:
        cfg["epochs"] = 2
        cfg["batch_size"] = 4

    print(f"\n{'='*60}")
    print(f"STAGE {stage}: hidden={cfg['hidden']}, layers={cfg['n_layers']}")
    print(f"{'='*60}")

    model = Seq2OE_CNN(d_out=896, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model = model.to(device)

    # Datasets
    train_seqs = {k: sequences[k] for k in train_ids if k in targets}
    train_tgts = {k: targets[k] for k in train_ids if k in targets}
    val_seqs = {k: sequences[k] for k in val_ids if k in targets}
    val_tgts = {k: targets[k] for k in val_ids if k in targets}

    max_len = 512
    train_ds = Seq2OEDataset(train_seqs, train_tgts, max_len=max_len)
    val_ds = Seq2OEDataset(val_seqs, val_tgts, max_len=max_len)

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_bit_acc": [], "val_bit_acc": []}

    checkpoint_dir = RESULTS_DIR / f"stage{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_losses, train_correct, train_total = [], 0, 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)

            logits = model(ids, mask)  # (B, L, 896)

            # Masked BCE loss
            loss_per_bit = criterion(logits, tgt)  # (B, L, 896)
            mask_3d = mask.unsqueeze(-1)  # (B, L, 1)
            loss = (loss_per_bit * mask_3d).sum() / mask_3d.sum() / 896

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Bit accuracy
            with torch.no_grad():
                preds = (logits > 0).float()
                correct = ((preds == tgt) * mask_3d).sum().item()
                total = mask_3d.sum().item() * 896
                train_correct += correct
                train_total += total

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses, val_correct, val_total = [], 0, 0

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                tgt = batch["target"].to(device)
                mask = batch["mask"].to(device)

                logits = model(ids, mask)
                loss_per_bit = criterion(logits, tgt)
                mask_3d = mask.unsqueeze(-1)
                loss = (loss_per_bit * mask_3d).sum() / mask_3d.sum() / 896
                val_losses.append(loss.item())

                preds = (logits > 0).float()
                correct = ((preds == tgt) * mask_3d).sum().item()
                total = mask_3d.sum().item() * 896
                val_correct += correct
                val_total += total

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bit_acc"].append(train_acc)
        history["val_bit_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"{elapsed:.0f}s")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Reload best
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt",
                    map_location=device, weights_only=True)
    )

    # Save history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Stage {stage} complete: best val acc={history['val_bit_acc'][best_epoch-1]:.4f} "
          f"@ epoch {best_epoch} ({elapsed:.0f}s)")

    return model, history


def evaluate_bit_accuracy(model, sequences, targets, test_ids, device):
    """Evaluate bit accuracy on test set, report per-protein stats."""
    model.eval()
    test_seqs = {k: sequences[k] for k in test_ids if k in targets}
    test_tgts = {k: targets[k] for k in test_ids if k in targets}

    ds = Seq2OEDataset(test_seqs, test_tgts, max_len=512)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    all_correct, all_total = 0, 0
    per_protein_acc = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"]

            logits = model(ids, mask)
            preds = (logits > 0).float()
            mask_3d = mask.unsqueeze(-1)

            correct = ((preds == tgt) * mask_3d).sum().item()
            total = mask_3d.sum().item() * 896
            all_correct += correct
            all_total += total

            # Per-protein accuracy
            for i in range(len(lengths)):
                L = lengths[i].item()
                p_correct = (preds[i, :L] == tgt[i, :L]).float().mean().item()
                per_protein_acc.append(p_correct)

    overall = all_correct / max(all_total, 1)
    per_prot = np.array(per_protein_acc)

    print(f"\n{'='*60}")
    print(f"TEST SET BIT ACCURACY")
    print(f"{'='*60}")
    print(f"  Overall:    {overall:.4f} ({overall*100:.1f}%)")
    print(f"  Per-protein: {per_prot.mean():.4f} +/- {per_prot.std():.4f}")
    print(f"  Min/Max:    {per_prot.min():.4f} / {per_prot.max():.4f}")
    print(f"  Random baseline: 0.5000 (50.0%)")
    print(f"  Improvement over random: {(overall - 0.5) * 200:.1f}pp")

    return {"overall_bit_acc": overall, "per_protein_mean": per_prot.mean(),
            "per_protein_std": per_prot.std(), "n_test": len(per_protein_acc)}


def main():
    parser = argparse.ArgumentParser(description="Exp 50: Sequence to One Embedding")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Model stage (1=baseline CNN, 2=deeper)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with tiny data + 2 epochs")
    parser.add_argument("--eval", action="store_true",
                        help="Run downstream evaluation after training")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Check system load
    load1, load5, _ = __import__("os").getloadavg()
    print(f"System load: {load1:.1f} (1m), {load5:.1f} (5m)")
    if load1 > 10:
        print("WARNING: System load >10, consider waiting before training")

    # Load data
    sequences, embeddings, train_ids, val_ids, test_ids = load_data(args.smoke_test)

    # Prepare binary targets from ProtT5 embeddings
    print("\nPreparing binary targets (center + RP + sign)...")
    t0 = time.time()
    all_targets = prepare_binary_targets(embeddings, d_out=896, seed=42)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Quick sanity: what's the class balance?
    sample_pid = next(iter(all_targets))
    sample = all_targets[sample_pid]
    balance = sample.mean()
    print(f"  Class balance (sample): {balance:.3f} (ideal=0.5)")

    # Train
    model, history = train_stage(
        args.stage, sequences, all_targets,
        train_ids, val_ids, device, args.smoke_test,
    )

    # Test evaluation
    results = evaluate_bit_accuracy(model, sequences, all_targets, test_ids, device)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"stage{args.stage}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.eval:
        print("\n[TODO] Downstream evaluation (SS3, disorder, retrieval)")
        print("  This will be added once we see if bit accuracy is above random.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test to verify script works end-to-end**

Run: `uv run python experiments/50_sequence_to_oe.py --smoke-test`
Expected: Completes in <2 min, prints bit accuracy (even if near 50%)

- [ ] **Step 3: Commit**

```bash
git add experiments/50_sequence_to_oe.py
git commit -m "feat(exp50): add sequence-to-OE experiment script with Stage 1 CNN"
```

---

### Task 4: Run Stage 1 and analyze results

This task is manual — run after Tasks 1-3 are committed and system load is acceptable.

- [ ] **Step 1: Check system load**

Run: `pmset -g therm` and check `os.getloadavg()` is below 10.

- [ ] **Step 2: Run Stage 1 training**

Run: `uv run python experiments/50_sequence_to_oe.py --stage 1`

Expected runtime: ~30-60 min on M3 Max for 4000 proteins, 50 epochs.

- [ ] **Step 3: Analyze results**

Key questions:
- Is bit accuracy > 55%? If yes, the sequence carries predictive signal for OE bits.
- Is bit accuracy > 60%? Strong signal — proceed to Stage 2.
- Is bit accuracy ~50%? Failure — ProtT5 encodes non-trivial long-range/evolutionary info.

- [ ] **Step 4: Decision point**

If bit accuracy > 55%:
  - Proceed to Stage 2 (deeper model)
  - Add per-dimension accuracy analysis (which bits are easy/hard to predict?)

If bit accuracy ~50%:
  - Try adding BLOSUM62 features (evolutionary signal in a lookup table)
  - Or: declare negative result (also interesting and publishable)

---

### Task 5: Stage 2 — Deeper CNN (conditional on Stage 1 success)

**Files:**
- Modify: `src/one_embedding/seq2oe.py` (model already supports configurable hidden/n_layers)

- [ ] **Step 1: Run Stage 2**

Run: `uv run python experiments/50_sequence_to_oe.py --stage 2`

The Stage 2 config (hidden=256, n_layers=10, ~10M params) is already built into the experiment script.

- [ ] **Step 2: Compare Stage 1 vs Stage 2**

If Stage 2 improves significantly (>3pp over Stage 1), the model is capacity-limited and more params help.
If Stage 2 plateaus at same accuracy, the bottleneck is the input representation (one-hot), not model capacity.

- [ ] **Step 3: Commit results**

```bash
git add results/exp50/
git commit -m "results(exp50): Stage 1+2 bit accuracy results"
```

---

### Task 6: Downstream evaluation (conditional on >55% bit accuracy)

**Files:**
- Modify: `experiments/50_sequence_to_oe.py` (add `--eval` implementation)

- [ ] **Step 1: Add downstream benchmark to experiment script**

Extend the `--eval` path in `main()` to:
1. Generate predicted OE for CB513 proteins (requires CB513 sequences)
2. Feed predicted ±1 embeddings through existing SS3 probes
3. Compare retention vs real binary OE

This follows the same pattern as Exp 47's benchmark: train linear probe on train set, evaluate on test set with BCa CIs.

- [ ] **Step 2: Run downstream eval**

Run: `uv run python experiments/50_sequence_to_oe.py --stage 1 --eval`

- [ ] **Step 3: Report results and commit**

Compare:
- Real binary OE SS3 retention: 97.6%
- Predicted binary OE SS3 retention: ???%

```bash
git add experiments/50_sequence_to_oe.py results/exp50/
git commit -m "results(exp50): downstream SS3 retention from predicted OE"
```
