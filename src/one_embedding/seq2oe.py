"""Sequence-to-One-Embedding: predict binary OE from amino acid sequence.

Bypasses the PLM entirely. Trains on (sequence, binary_OE_target) pairs
where targets come from OneEmbeddingCodec applied to ProtT5 embeddings.

Stage 1: Dilated 1D CNN (~2M params, ~63 residue receptive field)
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

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

        self.embed = nn.Embedding(AA_VOCAB_SIZE, hidden, padding_idx=0)

        self.blocks = nn.ModuleList([
            DilatedResBlock(hidden, dilation=2**i, kernel_size=kernel_size)
            for i in range(n_layers)
        ])

        self.head = nn.Linear(hidden, d_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: (B, L) integer-encoded amino acids
            mask: (B, L) float, 1.0 for real positions, 0.0 for padding

        Returns:
            (B, L, d_out) logits for each binary bit
        """
        h = self.embed(x)  # (B, L, hidden)
        h = h.transpose(1, 2)  # (B, hidden, L)

        for block in self.blocks:
            h = h * mask.unsqueeze(1)
            h = block(h)

        h = h * mask.unsqueeze(1)
        h = h.transpose(1, 2)  # (B, L, hidden)

        logits = self.head(h)  # (B, L, d_out)
        logits = logits * mask.unsqueeze(-1)
        return logits


# ── Target preparation ────────────────────────────────────────────────


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
    from src.one_embedding.codec_v2 import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(embeddings)

    targets = {}
    for pid, raw in embeddings.items():
        projected = codec._preprocess(raw)
        # Binary: sign(x - per-channel-mean) > 0, matching quantize_binary()
        means = projected.mean(axis=0)
        centered = projected - means[np.newaxis, :]
        bits = (centered > 0).astype(np.uint8)
        targets[pid] = bits

    return targets


# ── Dataset ───────────────────────────────────────────────────────────


class Seq2OEDataset(Dataset):
    """Dataset of (sequence, binary_target) pairs for Seq2OE training."""

    def __init__(
        self,
        sequences: dict[str, str],
        targets: dict[str, np.ndarray],
        max_len: int = 512,
    ):
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

        input_ids = torch.tensor(encode_sequence(seq), dtype=torch.long)

        padded_ids = torch.zeros(self.max_len, dtype=torch.long)
        padded_ids[:L] = input_ids

        d_out = target.shape[1]
        padded_target = torch.zeros(self.max_len, d_out, dtype=torch.float32)
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
