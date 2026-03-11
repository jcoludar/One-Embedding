"""Learned attention pooling with cross-attention decoder (Strategy A)."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class AttentionPoolCompressor(SequenceCompressor):
    """K learned queries cross-attend over residue states, with cross-attention decoder.

    Encoder: Learned queries attend to residue states via multi-head cross-attention.
    Decoder: Positional queries attend to latent tokens to reconstruct residue states.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        n_tokens: int = 8,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
        n_proj_layers: int = 1,
        init_proj_weights: tuple | None = None,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._n_tokens = n_tokens

        # Project input to latent dim
        if n_proj_layers == 1:
            self.input_proj = nn.Linear(embed_dim, latent_dim)
        else:
            mid = (embed_dim + latent_dim) // 2
            self.input_proj = nn.Sequential(
                nn.Linear(embed_dim, mid), nn.GELU(), nn.Linear(mid, latent_dim)
            )

        # Initialize input projection with PCA weights if provided
        if init_proj_weights is not None and n_proj_layers == 1:
            weight, bias = init_proj_weights
            with torch.no_grad():
                self.input_proj.weight.copy_(torch.from_numpy(weight).float())
                self.input_proj.bias.copy_(torch.from_numpy(bias).float())

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, latent_dim) * 0.02)

        # Encoder: cross-attention layers (queries attend to residue states)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=n_encoder_layers)

        # Decoder: positional queries attend to latent tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection back to embed_dim
        if n_proj_layers == 1:
            self.output_proj = nn.Linear(latent_dim, embed_dim)
        else:
            mid = (embed_dim + latent_dim) // 2
            self.output_proj = nn.Sequential(
                nn.Linear(latent_dim, mid), nn.GELU(), nn.Linear(mid, embed_dim)
            )

        # Pool projection: maps pooled latent back to embed_dim (for pooled reconstruction loss)
        self.pool_proj = nn.Linear(latent_dim, embed_dim)

        # Positional encoding for decoder
        self.max_len = 1024
        self.register_buffer(
            "pos_enc",
            self._sinusoidal_pos_enc(self.max_len, latent_dim),
        )

    @staticmethod
    def _sinusoidal_pos_enc(max_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    @property
    def num_tokens(self) -> int:
        return self._n_tokens

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        B, L, D = residue_states.shape
        # Project to latent dim
        memory = self.input_proj(residue_states)  # (B, L, latent_dim)

        # Expand queries for batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, K, latent_dim)

        # Invert mask for PyTorch transformer (True = ignore)
        memory_key_padding_mask = ~mask.bool()

        # Cross-attend: queries attend to residue states
        latent = self.encoder(
            queries,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return latent  # (B, K, latent_dim)

    def pool_and_project(self, latent: Tensor) -> Tensor:
        """Mean-pool K tokens and project back to embed_dim.

        Args:
            latent: (B, K, D') latent tokens
        Returns:
            (B, D) projected pooled representation
        """
        pooled = latent.mean(dim=1)  # (B, D')
        return self.pool_proj(pooled)  # (B, D)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        B = latent.shape[0]

        # Positional queries for target positions
        pos_queries = self.pos_enc[:, :target_length, :].expand(B, -1, -1)

        # Cross-attend: positional queries attend to latent tokens
        decoded = self.decoder(pos_queries, latent)  # (B, L, latent_dim)

        # Project back to original dim
        return self.output_proj(decoded)  # (B, L, embed_dim)
