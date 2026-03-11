"""Learned Fourier basis coefficients with decoder (Strategy C)."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class FourierBasisCompressor(SequenceCompressor):
    """Treat each embedding dimension as a 1D signal along the sequence.

    Encoder: Project residue states, compute dot products with learned basis functions
             to get K coefficients per dimension.
    Decoder: Multiply coefficients by basis functions evaluated at target positions.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        n_tokens: int = 16,
        max_len: int = 1024,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._n_tokens = n_tokens
        self._max_len = max_len

        # Project to latent dim
        self.input_proj = nn.Linear(embed_dim, latent_dim)

        # Learned frequency parameters for basis functions
        self.frequencies = nn.Parameter(torch.randn(n_tokens) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_tokens))

        # Learned amplitude modulation per token per dim
        self.amplitude = nn.Parameter(torch.ones(n_tokens, latent_dim) * 0.1)

        # Encoder MLP to refine coefficients
        self.encoder_mlp = nn.Sequential(
            nn.Linear(n_tokens * latent_dim, n_tokens * latent_dim),
            nn.GELU(),
            nn.Linear(n_tokens * latent_dim, n_tokens * latent_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(latent_dim, embed_dim)

    @property
    def num_tokens(self) -> int:
        return self._n_tokens

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def _eval_basis(self, length: int, device: torch.device) -> Tensor:
        """Evaluate basis functions at positions 0..length-1.

        Returns: (length, n_tokens)
        """
        t = torch.linspace(0, 1, length, device=device)  # (L,)
        # basis_k(t) = sin(2*pi*freq_k*t + phase_k)
        basis = torch.sin(
            2 * math.pi * self.frequencies.unsqueeze(0) * t.unsqueeze(1)
            + self.phases.unsqueeze(0)
        )  # (L, K)
        return basis

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        B, L, D = residue_states.shape

        # Project input
        x = self.input_proj(residue_states)  # (B, L, latent_dim)
        x = x * mask.unsqueeze(-1).float()

        # Evaluate basis at input positions
        basis = self._eval_basis(L, x.device)  # (L, K)

        # Compute coefficients: dot product of signal with basis
        # (B, L, latent_dim) x (L, K) -> (B, K, latent_dim)
        mask_sum = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        coeffs = torch.einsum("bld,lk->bkd", x, basis) / mask_sum.unsqueeze(-1)

        # Apply amplitude modulation
        coeffs = coeffs * self.amplitude.unsqueeze(0)

        # Refine with MLP
        B_size = coeffs.shape[0]
        flat = coeffs.reshape(B_size, -1)
        flat = flat + self.encoder_mlp(flat)  # residual
        coeffs = flat.reshape(B_size, self._n_tokens, self._latent_dim)

        return coeffs  # (B, K, latent_dim)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        B, K, D = latent.shape

        # Evaluate basis at target positions
        basis = self._eval_basis(target_length, latent.device)  # (L, K)

        # Reconstruct: (B, K, latent_dim) x (L, K)^T -> (B, L, latent_dim)
        reconstructed = torch.einsum("bkd,lk->bld", latent, basis)

        # Project back to embed_dim
        return self.output_proj(reconstructed)  # (B, L, embed_dim)
