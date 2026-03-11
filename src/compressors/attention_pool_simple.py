"""DeepSets-style attention pooling compressor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.compressors.base import SequenceCompressor


class DeepSetsAttentionCompressor(SequenceCompressor):
    """Attention-weighted mean pooling followed by MLP compression.

    Uses DeepSets attention: alpha_i = softmax(w^T tanh(V x_i))
    pooled = sum(alpha_i * x_i)
    Then compresses via MLP encoder. K=1, single-vector output.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        attn_hidden: int = 256,
        hidden_dims: tuple[int, ...] = (512,),
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim

        # DeepSets attention
        self.attn_V = nn.Linear(embed_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1, bias=False)

        # MLP encoder
        enc_layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, embed_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def attention_pool(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        """Compute attention-weighted mean of residue states."""
        scores = self.attn_w(torch.tanh(self.attn_V(residue_states)))  # (B, L, 1)
        scores = scores.squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, L)
        pooled = (residue_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode_latent(self, z: Tensor) -> Tensor:
        return self.decoder_net(z)

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        pooled = self.attention_pool(residue_states, mask)
        z = self.encode(pooled)
        return z.unsqueeze(1)  # (B, 1, D')

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        z = latent.squeeze(1)
        recon = self.decode_latent(z)
        return recon.unsqueeze(1).expand(-1, target_length, -1)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        pooled = self.attention_pool(residue_states, mask)
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


class MultiScalePoolCompressor(SequenceCompressor):
    """Multi-scale statistics pooling followed by MLP compression.

    Computes mean, std, max over residues -> concatenate -> MLP compress.
    Captures per-residue variance without reconstruction overhead.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        stats: tuple[str, ...] = ("mean", "std", "max"),
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._stats = stats
        self._input_dim = embed_dim * len(stats)

        # MLP encoder
        enc_layers = []
        in_dim = self._input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (back to mean-pooled dim, not full stats)
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, embed_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    @property
    def num_tokens(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def multi_scale_pool(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        """Compute multi-scale statistics over residues."""
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        masked_states = residue_states * mask_f
        lengths = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)

        parts = []
        for stat in self._stats:
            if stat == "mean":
                parts.append(masked_states.sum(dim=1) / lengths)
            elif stat == "std":
                mean = masked_states.sum(dim=1) / lengths
                var = ((residue_states - mean.unsqueeze(1)).pow(2) * mask_f).sum(dim=1) / lengths
                parts.append(var.sqrt())
            elif stat == "max":
                # Replace masked positions with -inf for max (MPS-safe: use where instead of fancy indexing)
                mask_bool = mask.bool().unsqueeze(-1).expand_as(residue_states)
                masked_for_max = torch.where(mask_bool, residue_states, torch.tensor(float("-inf"), device=residue_states.device))
                parts.append(masked_for_max.max(dim=1).values)

        return torch.cat(parts, dim=-1)  # (B, D * n_stats)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode_latent(self, z: Tensor) -> Tensor:
        return self.decoder_net(z)

    def compress(self, residue_states: Tensor, mask: Tensor) -> Tensor:
        pooled = self.multi_scale_pool(residue_states, mask)
        z = self.encode(pooled)
        return z.unsqueeze(1)  # (B, 1, D')

    def decode(self, latent: Tensor, target_length: int) -> Tensor:
        z = latent.squeeze(1)
        recon = self.decode_latent(z)
        return recon.unsqueeze(1).expand(-1, target_length, -1)

    def forward(self, residue_states: Tensor, mask: Tensor) -> dict[str, Tensor]:
        stats_vec = self.multi_scale_pool(residue_states, mask)
        z = self.encode(stats_vec)
        recon = self.decode_latent(z)
        # Mean-pool for target comparison
        mask_f = mask.unsqueeze(-1).float()
        mean_pooled = (residue_states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        target_length = residue_states.shape[1]
        recon_broadcast = recon.unsqueeze(1).expand(-1, target_length, -1)
        return {
            "latent": z.unsqueeze(1),
            "reconstructed": recon_broadcast,
            "pooled_input": mean_pooled,
            "pooled_recon": recon,
        }
