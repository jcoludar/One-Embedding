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

    def get_pooled(self, latent: Tensor, strategy: str = "mean") -> Tensor:
        """Pool latent tokens to a single vector for retrieval/classification.

        Strategies:
            mean: Mean over K tokens -> (B, D')
            first: First token only -> (B, D')
            mean_std: Concatenate mean and std -> (B, 2*D')
            concat: Flatten all K tokens -> (B, K*D')
        """
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
