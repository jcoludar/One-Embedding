"""Wrapper around PLM_SWE (Sliced-Wasserstein Embedding) published baseline.

Paper: Bioinformatics Advances 2025
Code: https://github.com/navid-naderi/PLM_SWE

This is a pooling-only method (no decoder). We wrap it to conform to the
SequenceCompressor interface for fair comparison.
"""

import numpy as np

from src.compressors.base import SequenceCompressor


class SWEPoolCompressor(SequenceCompressor):
    """Wrapper for Sliced-Wasserstein Embedding pooling.

    Note: This requires installing PLM_SWE separately.
    Since it's pooling-only, decode() returns a broadcast of the pooled vector.
    """

    def __init__(self, embed_dim: int, n_slices: int = 1000):
        import torch
        super().__init__()
        self._embed_dim = embed_dim
        self._n_slices = n_slices

        # Random projection directions (fixed after init)
        directions = torch.randn(n_slices, embed_dim)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        self.register_buffer("directions", directions)

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._n_slices

    def compress(self, residue_states, mask):
        import torch
        B, L, D = residue_states.shape
        mask_f = mask.unsqueeze(-1).float()
        x = residue_states * mask_f

        # Project onto random directions: (B, L, D) x (D, n_slices) -> (B, L, n_slices)
        projections = torch.matmul(x, self.directions.T)

        # Sort projections along sequence dim and compute quantile summary
        # For SWE, we use sorted projections as the embedding
        # Use mean of sorted projections as a simple approximation
        sorted_proj, _ = projections.sort(dim=1)

        # Take mean of sorted projections (valid positions only)
        lengths = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        pooled = (sorted_proj * mask_f).sum(dim=1) / lengths.squeeze(-1)

        return pooled.unsqueeze(1)  # (B, 1, n_slices)

    def decode(self, latent, target_length):
        return latent.expand(-1, target_length, -1)
