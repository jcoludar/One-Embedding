"""Mean pooling baseline (no learnable parameters, no decoder)."""

import torch
from torch import Tensor

from src.compressors.base import SequenceCompressor


class MeanPoolCompressor(SequenceCompressor):
    """Simple mean pooling baseline. K=1, no decoder."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._embed_dim

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        # mask: (B, L), residue_states: (B, L, D)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (residue_states * mask_expanded).sum(dim=1)  # (B, D)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = summed / lengths  # (B, D)
        return pooled.unsqueeze(1)  # (B, 1, D)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        # Simply broadcast the single token to all positions
        return latent.expand(-1, target_length, -1)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        latent = self.compress(residue_states, mask)
        reconstructed = self.decode(latent, residue_states.shape[1])
        return {"latent": latent, "reconstructed": reconstructed}
