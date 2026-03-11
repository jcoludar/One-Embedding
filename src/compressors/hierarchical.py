"""Hierarchical conv windows -> regional -> global with decoder (Strategy B)."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class HierarchicalCompressor(SequenceCompressor):
    """Conv windows -> regional pooling -> global tokens, with transpose decoder.

    Encoder: 1D conv over residue states, then strided pooling, then self-attention.
    Decoder: Upsample + deconv to reconstruct residue states.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        n_tokens: int = 8,
        window_size: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._n_tokens = n_tokens
        self._window_size = window_size

        # Local: 1D conv to capture local patterns
        self.local_conv = nn.Sequential(
            nn.Conv1d(embed_dim, latent_dim, kernel_size=window_size, padding="same"),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=window_size, padding="same"),
            nn.GELU(),
        )

        # Regional: we'll use interpolation instead of adaptive pool (MPS compatible)
        self._regional_out = n_tokens

        # Global: self-attention over regional tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.global_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder: upsample + conv (avoids ConvTranspose1d length issues)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=window_size, padding="same"),
            nn.GELU(),
            nn.Conv1d(latent_dim, embed_dim, kernel_size=window_size, padding="same"),
        )

    @property
    def num_tokens(self) -> int:
        return self._n_tokens

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        B, L, D = residue_states.shape
        # Apply mask
        x = residue_states * mask.unsqueeze(-1).float()

        # Local conv: (B, L, D) -> (B, D, L) -> conv -> (B, latent_dim, L)
        x = x.transpose(1, 2)
        x = self.local_conv(x)

        # Regional pool via interpolation (MPS-compatible alternative to AdaptiveAvgPool1d)
        x = torch.nn.functional.interpolate(x, size=self._regional_out, mode="linear", align_corners=False)

        # Transpose for attention: (B, K, latent_dim)
        x = x.transpose(1, 2)

        # Global self-attention
        x = self.global_attn(x)

        return x  # (B, K, latent_dim)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        B, K, D = latent.shape

        # Interpolate latent tokens to target length
        x = latent.transpose(1, 2)  # (B, latent_dim, K)
        x = torch.nn.functional.interpolate(x, size=target_length, mode="linear", align_corners=False)

        # Refine with conv
        x = self.decoder_conv(x)  # (B, embed_dim, L)

        return x.transpose(1, 2)  # (B, L, embed_dim)
