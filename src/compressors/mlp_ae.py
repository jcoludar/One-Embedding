"""MLP autoencoder compressor operating on mean-pooled embeddings."""

import torch
import torch.nn as nn
from torch import Tensor

from src.compressors.base import SequenceCompressor


class MLPAutoencoder(SequenceCompressor):
    """MLP autoencoder that mean-pools per-residue input, then encodes/decodes.

    Designed for protein-level (mean-pooled) representations.
    K=1 always. Supports deeper architectures with residual connections.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512,),
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._use_residual = use_residual

        # Build encoder
        self.enc_layers = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        in_dim = embed_dim
        for h_dim in hidden_dims:
            self.enc_layers.append(nn.Linear(in_dim, h_dim))
            self.enc_norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.enc_proj = nn.Linear(in_dim, latent_dim)
        self.enc_dropout = nn.Dropout(dropout)

        # Build decoder (mirror)
        self.dec_layers = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            self.dec_layers.append(nn.Linear(in_dim, h_dim))
            self.dec_norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.dec_proj = nn.Linear(in_dim, embed_dim)
        self.dec_dropout = nn.Dropout(dropout)

        # Residual projections (for dimension mismatches)
        if use_residual:
            self.enc_res_projs = nn.ModuleList()
            in_dim = embed_dim
            for h_dim in hidden_dims:
                if in_dim != h_dim:
                    self.enc_res_projs.append(nn.Linear(in_dim, h_dim, bias=False))
                else:
                    self.enc_res_projs.append(nn.Identity())
                in_dim = h_dim

            self.dec_res_projs = nn.ModuleList()
            in_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                if in_dim != h_dim:
                    self.dec_res_projs.append(nn.Linear(in_dim, h_dim, bias=False))
                else:
                    self.dec_res_projs.append(nn.Identity())
                in_dim = h_dim

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def mean_pool(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool per-residue states to single vector."""
        mask_f = mask.unsqueeze(-1).float()
        return (residue_states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def encode(self, x: Tensor) -> Tensor:
        """Encode a (B, D) mean-pooled vector to (B, D') latent."""
        h = x
        for i, (layer, norm) in enumerate(zip(self.enc_layers, self.enc_norms)):
            out = self.enc_dropout(nn.functional.gelu(norm(layer(h))))
            if self._use_residual:
                out = out + self.enc_res_projs[i](h)
            h = out
        return self.enc_proj(h)

    def decode_latent(self, z: Tensor) -> Tensor:
        """Decode a (B, D') latent vector to (B, D) reconstructed."""
        h = z
        for i, (layer, norm) in enumerate(zip(self.dec_layers, self.dec_norms)):
            out = self.dec_dropout(nn.functional.gelu(norm(layer(h))))
            if self._use_residual:
                out = out + self.dec_res_projs[i](h)
            h = out
        return self.dec_proj(h)

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        pooled = self.mean_pool(residue_states, mask)
        z = self.encode(pooled)
        return z.unsqueeze(1)  # (B, 1, D')

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        z = latent.squeeze(1)
        recon = self.decode_latent(z)
        return recon.unsqueeze(1).expand(-1, target_length, -1)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        pooled = self.mean_pool(residue_states, mask)
        z = self.encode(pooled)
        recon = self.decode_latent(z)
        target_length = residue_states.shape[1]
        recon_broadcast = recon.unsqueeze(1).expand(-1, target_length, -1)
        return {
            "latent": z.unsqueeze(1),
            "reconstructed": recon_broadcast,
            "pooled_input": pooled,
            "pooled_recon": recon,
        }
