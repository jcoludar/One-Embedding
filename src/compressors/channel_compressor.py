"""Per-residue channel compressor: (L, D) → (L, D') pointwise MLP.

Model-agnostic compression that preserves per-residue information.
The PLM already baked cross-residue context through its attention layers,
so this operates independently on each residue position (pointwise).
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class ChannelCompressor(SequenceCompressor):
    """Pointwise MLP that compresses the channel dimension of per-residue embeddings.

    Input:  (B, L, D)  per-residue PLM embeddings
    Output: (B, L, D') compressed per-residue embeddings

    num_tokens = -1 (variable: one per residue, preserving sequence length).
    Use get_pooled() for retrieval/classification after compression.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_residual: bool = True,
        retrieval_head_dim: int | None = None,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._use_residual = use_residual
        self._retrieval_head_dim = retrieval_head_dim

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        # Encoder: D → H → D'
        self.input_norm = nn.LayerNorm(input_dim)
        self.enc_linear1 = nn.Linear(input_dim, hidden_dim)
        self.enc_norm1 = nn.LayerNorm(hidden_dim)
        self.enc_dropout = nn.Dropout(dropout)
        self.enc_proj = nn.Linear(hidden_dim, latent_dim)

        # Decoder (mirror): D' → H → D
        self.dec_linear1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_norm1 = nn.LayerNorm(hidden_dim)
        self.dec_dropout = nn.Dropout(dropout)
        self.dec_proj = nn.Linear(hidden_dim, input_dim)

        # Residual projections for dimension mismatches
        if use_residual:
            self.enc_res_proj = (
                nn.Linear(input_dim, hidden_dim, bias=False)
                if input_dim != hidden_dim
                else nn.Identity()
            )
            self.dec_res_proj = (
                nn.Linear(latent_dim, hidden_dim, bias=False)
                if latent_dim != hidden_dim
                else nn.Identity()
            )

        # Optional retrieval head: pool → MLP → projection
        if retrieval_head_dim is not None:
            self.retrieval_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, retrieval_head_dim),
            )

        # Cache for mask-aware pooling
        self._last_mask: Tensor | None = None

    @property
    def num_tokens(self) -> int:
        return -1  # Variable: one per residue

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def encode(self, x: Tensor) -> Tensor:
        """Encode per-residue: (B, L, D) → (B, L, D')."""
        h = self.input_norm(x)
        out = self.enc_dropout(nn.functional.gelu(self.enc_norm1(self.enc_linear1(h))))
        if self._use_residual:
            out = out + self.enc_res_proj(h)
        return self.enc_proj(out)

    def decode_latent(self, z: Tensor) -> Tensor:
        """Decode per-residue: (B, L, D') → (B, L, D)."""
        out = self.dec_dropout(nn.functional.gelu(self.dec_norm1(self.dec_linear1(z))))
        if self._use_residual:
            out = out + self.dec_res_proj(z)
        return self.dec_proj(out)

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        """Compress per-residue: (B, L, D) → (B, L, D').

        Stores mask for subsequent get_pooled() calls.
        """
        self._last_mask = mask
        latent = self.encode(residue_states)
        # Zero out padded positions
        return latent * mask.unsqueeze(-1).float()

    def decode(self, latent: Tensor, target_length: int = None) -> Tensor:
        """Decode per-residue: (B, L, D') → (B, L, D)."""
        return self.decode_latent(latent)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """Full encode-decode pass.

        Returns dict with 'latent', 'reconstructed', and optionally
        'retrieval_embedding' if retrieval_head_dim was set.
        """
        self._last_mask = mask
        latent = self.encode(residue_states)
        mask_f = mask.unsqueeze(-1).float()
        latent = latent * mask_f
        reconstructed = self.decode_latent(latent)
        reconstructed = reconstructed * mask_f
        result = {"latent": latent, "reconstructed": reconstructed}

        if self._retrieval_head_dim is not None:
            pooled = self.get_pooled(latent, strategy="mean", mask=mask)
            retrieval_emb = self.retrieval_head(pooled)
            retrieval_emb = nn.functional.normalize(retrieval_emb, dim=-1)
            result["retrieval_embedding"] = retrieval_emb

        return result

    def get_pooled(
        self, latent: Tensor, strategy: str = "mean", mask: Tensor | None = None
    ) -> Tensor:
        """Mask-aware pooling over sequence length for retrieval/classification.

        Args:
            latent: (B, L, D') compressed per-residue embeddings.
            strategy: "mean" (default), "max", or "mean_std".
            mask: (B, L) boolean/float mask. If None, uses mask from last compress().
        """
        if mask is None:
            mask = self._last_mask

        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            lengths = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)

            if strategy == "mean":
                return (latent * mask_f).sum(dim=1) / lengths
            elif strategy == "max":
                # Replace padded positions with -inf before max
                latent_masked = latent.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
                return latent_masked.max(dim=1).values
            elif strategy == "mean_std":
                mean = (latent * mask_f).sum(dim=1) / lengths
                diff_sq = ((latent - mean.unsqueeze(1)) * mask_f).pow(2)
                std = (diff_sq.sum(dim=1) / lengths.clamp(min=2)).sqrt()
                return torch.cat([mean, std], dim=-1)
            else:
                raise ValueError(f"Unknown pooling strategy: {strategy}")
        else:
            # No mask: simple pooling
            if strategy == "mean":
                return latent.mean(dim=1)
            elif strategy == "max":
                return latent.max(dim=1).values
            elif strategy == "mean_std":
                return torch.cat([latent.mean(dim=1), latent.std(dim=1)], dim=-1)
            else:
                raise ValueError(f"Unknown pooling strategy: {strategy}")
