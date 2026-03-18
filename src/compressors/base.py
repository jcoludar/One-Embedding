"""SequenceCompressor abstract base class."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class SequenceCompressor(nn.Module, ABC):
    """Base class for all sequence compressors.

    Input:  residue_states (B, L, D), mask (B, L)
    Output: latent (B, K, D') -- K tokens of dim D'

    Also: decode(latent, target_length) -> (B, L, D) approximate residue states
    """

    @abstractmethod
    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        """Compress residue states to K latent tokens."""
        ...

    @abstractmethod
    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        """Decode latent tokens back to approximate residue states."""
        ...

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """Number of latent tokens K."""
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of each latent token D'."""
        ...

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """Full encode-decode pass. Returns dict with 'latent' and 'reconstructed'."""
        latent = self.compress(residue_states, mask)
        target_length = residue_states.shape[1]
        reconstructed = self.decode(latent, target_length)
        return {"latent": latent, "reconstructed": reconstructed}

    def get_pooled(
        self, latent: Tensor, strategy: str = "mean", mask: Tensor | None = None,
    ) -> Tensor:
        """Pool latent tokens to a single vector for retrieval/classification.

        Args:
            latent: (B, K, D') latent tokens.
            strategy: "mean", "first", "mean_std", or "concat".
            mask: Optional (B, K) mask for variable-length sequences.

        Strategies:
            mean: Mean over K tokens -> (B, D')
            first: First token only -> (B, D')
            mean_std: Concatenate mean and std -> (B, 2*D')
            concat: Flatten all K tokens -> (B, K*D')
        """
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, K, 1)
            lengths = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
            if strategy == "mean":
                return (latent * mask_f).sum(dim=1) / lengths
            elif strategy == "mean_std":
                mean = (latent * mask_f).sum(dim=1) / lengths
                diff_sq = ((latent - mean.unsqueeze(1)) * mask_f).pow(2)
                std = (diff_sq.sum(dim=1) / lengths.clamp(min=2)).sqrt()
                return torch.cat([mean, std], dim=-1)

        if strategy == "mean":
            return latent.mean(dim=1)
        elif strategy == "first":
            return latent[:, 0, :]
        elif strategy == "mean_std":
            return torch.cat([latent.mean(dim=1), latent.std(dim=1)], dim=-1)
        elif strategy == "concat":
            return latent.flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
