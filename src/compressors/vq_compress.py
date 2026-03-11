"""VQ/FSQ discrete token compression with decoder (Strategy D)."""

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class VectorQuantizer(nn.Module):
    """Vector Quantization with EMA codebook updates and commitment loss."""

    def __init__(self, n_codes: int, code_dim: int, commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1 / n_codes, 1 / n_codes)

        # EMA tracking
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_weight", self.codebook.weight.clone())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize input tensor.

        Args:
            z: (B, L, D) continuous representations

        Returns:
            z_q: (B, L, D) quantized representations (with straight-through gradient)
            indices: (B, L) codebook indices
            vq_loss: scalar commitment + codebook loss
        """
        B, L, D = z.shape
        flat_z = z.reshape(-1, D)  # (B*L, D)

        # Find nearest codebook entry
        dists = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = dists.argmin(dim=-1)  # (B*L,)
        z_q_flat = self.codebook(indices)  # (B*L, D)

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(indices.shape[0], self.n_codes, device=z.device)
                one_hot.scatter_(1, indices.unsqueeze(1), 1)
                count = one_hot.sum(0)
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * count
                weight_sum = one_hot.t() @ flat_z
                self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * weight_sum
                n = self.ema_count.sum()
                count_stable = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n
                self.codebook.weight.data = self.ema_weight / count_stable.unsqueeze(1)

        # Losses
        commitment_loss = (flat_z - z_q_flat.detach()).pow(2).mean()
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_flat = flat_z + (z_q_flat - flat_z).detach()
        z_q = z_q_flat.reshape(B, L, D)
        indices = indices.reshape(B, L)

        return z_q, indices, vq_loss


class VQCompressor(SequenceCompressor):
    """VQ-based compression: encode residue groups to discrete codes, then decode.

    Encoder: Conv + pool to K positions, then vector quantize.
    Decoder: Lookup quantized codes + positional decoder to reconstruct.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        n_tokens: int = 8,
        n_codes: int = 512,
        n_heads: int = 4,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._n_tokens = n_tokens

        # Encoder: project + pool to K tokens
        self.input_proj = nn.Linear(embed_dim, latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self._pool_target = n_tokens  # Use interpolation instead of AdaptiveAvgPool1d (MPS compat)
        self.encoder_norm = nn.LayerNorm(latent_dim)

        # Vector quantizer
        self.vq = VectorQuantizer(n_codes, latent_dim)

        # Decoder: cross-attention from positional queries to VQ tokens
        import math
        max_len = 1024
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: latent_dim // 2 + latent_dim % 2])
        self.register_buffer("pos_enc", pe.unsqueeze(0))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.output_proj = nn.Linear(latent_dim, embed_dim)

        self._last_vq_loss = torch.tensor(0.0)

    @property
    def num_tokens(self) -> int:
        return self._n_tokens

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def vq_loss(self) -> Tensor:
        return self._last_vq_loss

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        B, L, D = residue_states.shape
        x = self.input_proj(residue_states)  # (B, L, latent_dim)
        x = x * mask.unsqueeze(-1).float()

        # Conv + pool
        x = x.transpose(1, 2)  # (B, latent_dim, L)
        x = self.encoder_conv(x)
        x = torch.nn.functional.interpolate(x, size=self._pool_target, mode="linear", align_corners=False)  # (B, latent_dim, K)
        x = x.transpose(1, 2)  # (B, K, latent_dim)
        x = self.encoder_norm(x)

        # Vector quantize
        z_q, indices, vq_loss = self.vq(x)
        self._last_vq_loss = vq_loss

        return z_q  # (B, K, latent_dim)

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        B = latent.shape[0]
        pos_queries = self.pos_enc[:, :target_length, :].expand(B, -1, -1)
        decoded = self.decoder(pos_queries, latent)
        return self.output_proj(decoded)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        latent = self.compress(residue_states, mask)
        reconstructed = self.decode(latent, residue_states.shape[1])
        return {
            "latent": latent,
            "reconstructed": reconstructed,
            "vq_loss": self._last_vq_loss,
        }
