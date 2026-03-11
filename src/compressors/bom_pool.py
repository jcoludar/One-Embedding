"""Wrapper around BoM-Pooling (Bag of Motifs) published baseline.

Paper: ISMB 2025
Code: https://github.com/Singh-Lab/bom-pooling

This is a pooling-only method (no decoder). We wrap it to conform to the
SequenceCompressor interface for fair comparison.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class BoMPoolCompressor(SequenceCompressor):
    """Simplified BoM-Pooling approximation.

    Uses windowed k-mer attention pooling. Since the original BoM code
    requires specific setup, this is a faithful re-implementation of the core idea:
    - Split sequence into overlapping windows
    - Attention-weighted pooling within each window
    - Concatenate/pool window representations

    Since it's pooling-only, decode() returns a broadcast of the pooled vector.
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int = 16,
        stride: int = 8,
        n_heads: int = 4,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._window_size = window_size
        self._stride = stride

        # Attention pooling within each window
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # Project windowed outputs to final representation
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._embed_dim

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        B, L, D = residue_states.shape

        # Extract overlapping windows
        window_reps = []
        for start in range(0, L, self._stride):
            end = min(start + self._window_size, L)
            if end - start < 2:
                continue

            window = residue_states[:, start:end, :]  # (B, W, D)
            w_mask = mask[:, start:end]  # (B, W)

            # Attention pooling with learned query
            query = self.query.expand(B, -1, -1)  # (B, 1, D)
            key_padding_mask = ~w_mask.bool()

            attn_out, _ = self.attn(query, window, window, key_padding_mask=key_padding_mask)
            window_reps.append(attn_out.squeeze(1))  # (B, D)

        if not window_reps:
            return residue_states.mean(dim=1, keepdim=True)

        # Mean-pool over window representations
        stacked = torch.stack(window_reps, dim=1)  # (B, n_windows, D)
        pooled = stacked.mean(dim=1)  # (B, D)
        pooled = self.output_proj(pooled)

        return pooled.unsqueeze(1)  # (B, 1, D)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        return latent.expand(-1, target_length, -1)
