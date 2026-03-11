"""Training objectives: reconstruction, masked prediction, contrastive losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReconstructionLoss(nn.Module):
    """MSE + cosine similarity loss for residue-level reconstruction."""

    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
            pred: (B, L, D) predicted residue states
            target: (B, L, D) original residue states
            mask: (B, L) boolean mask
        """
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        n_valid = mask_f.sum().clamp(min=1)

        # MSE over valid positions
        mse = ((pred - target).pow(2) * mask_f).sum() / (n_valid * target.shape[-1])

        # Cosine similarity over valid positions
        pred_flat = (pred * mask_f).reshape(-1, pred.shape[-1])
        target_flat = (target * mask_f).reshape(-1, target.shape[-1])
        # Only compute on non-zero (valid) positions
        valid_mask = mask.reshape(-1).bool()
        if valid_mask.any():
            cos_sim = F.cosine_similarity(pred_flat[valid_mask], target_flat[valid_mask], dim=-1)
            cos_loss = 1.0 - cos_sim.mean()
        else:
            cos_loss = torch.tensor(0.0, device=pred.device)

        total = self.mse_weight * mse + self.cosine_weight * cos_loss

        return {
            "loss": total,
            "mse": mse.detach(),
            "cosine_loss": cos_loss.detach() if isinstance(cos_loss, Tensor) else cos_loss,
        }


class MaskedPredictionLoss(nn.Module):
    """Predict masked amino acid identities from reconstructed residue states.

    Projects reconstructed states to 20 AA logits.
    """

    def __init__(self, embed_dim: int, mask_prob: float = 0.15):
        super().__init__()
        self.mask_prob = mask_prob
        self.head = nn.Linear(embed_dim, 20)  # 20 standard amino acids

    # Amino acid alphabet (canonical order)
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    def forward(
        self,
        reconstructed: Tensor,
        sequences: list[str],
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            reconstructed: (B, L, D) decoded residue states
            sequences: list of AA strings
            mask: (B, L) valid positions
        """
        B, L, D = reconstructed.shape
        device = reconstructed.device

        # Create random mask for prediction
        pred_mask = torch.rand(B, L, device=device) < self.mask_prob
        pred_mask = pred_mask & mask.bool()

        if not pred_mask.any():
            return {"loss": torch.tensor(0.0, device=device), "accuracy": torch.tensor(0.0, device=device)}

        # Get target labels
        targets = torch.full((B, L), -1, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:L]):
                if aa in self.AA_TO_IDX:
                    targets[i, j] = self.AA_TO_IDX[aa]

        # Predict at masked positions
        logits = self.head(reconstructed)  # (B, L, 20)
        masked_logits = logits[pred_mask]
        masked_targets = targets[pred_mask]

        # Filter out invalid targets
        valid = masked_targets >= 0
        if not valid.any():
            return {"loss": torch.tensor(0.0, device=device), "accuracy": torch.tensor(0.0, device=device)}

        loss = F.cross_entropy(masked_logits[valid], masked_targets[valid])
        accuracy = (masked_logits[valid].argmax(-1) == masked_targets[valid]).float().mean()

        return {"loss": loss, "accuracy": accuracy.detach()}


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss on latent representations."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Args:
            z1, z2: (B, D) pooled latent vectors from two augmented views
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        B = z1.shape[0]
        logits = z1 @ z2.T / self.temperature  # (B, B)
        labels = torch.arange(B, device=z1.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


class PooledReconstructionLoss(nn.Module):
    """Reconstruct mean-pooled input from pooled latent via a learned projection."""

    def forward(self, pooled_pred: Tensor, target_mean: Tensor) -> dict[str, Tensor]:
        """
        Args:
            pooled_pred: (B, D) predicted mean-pooled embedding (from pool_proj(mean(latent)))
            target_mean: (B, D) actual mean-pooled input embedding
        """
        mse = F.mse_loss(pooled_pred, target_mean)
        cos_loss = 1.0 - F.cosine_similarity(pooled_pred, target_mean, dim=-1).mean()
        loss = mse + 0.5 * cos_loss
        return {"loss": loss, "mse": mse.detach(), "cos_loss": cos_loss.detach()}


class VICRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance regularization.

    Prevents dimensional collapse without needing negative pairs.
    Operates on batch statistics of pooled latent vectors.
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
    ):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """
        Args:
            z: (B, D) pooled latent vectors for the batch
        """
        B, D = z.shape
        if B < 2:
            zero = torch.tensor(0.0, device=z.device)
            return {"loss": zero, "var_loss": zero, "cov_loss": zero}

        # Variance: hinge loss on per-dimension std (should be >= var_target)
        std = z.std(dim=0)  # (D,)
        var_loss = F.relu(self.var_target - std).mean()

        # Covariance: off-diagonal elements of cov matrix should be 0
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)
        # Zero out diagonal, penalize off-diagonal
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0)
        cov_loss = off_diag.sum() / D

        loss = self.var_weight * var_loss + self.cov_weight * cov_loss
        return {
            "loss": loss,
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }


class TokenOrthogonalityLoss(nn.Module):
    """Penalize redundancy between latent tokens.

    Encourages each of K tokens to carry different information by minimizing
    pairwise cosine similarity between tokens.
    """

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        """
        Args:
            latent: (B, K, D) latent tokens
        """
        # Normalize tokens
        z = F.normalize(latent, dim=-1)  # (B, K, D)
        # Pairwise cosine similarity: (B, K, K)
        sim = torch.bmm(z, z.transpose(1, 2))
        # Mask diagonal (self-similarity = 1, ignore)
        K = sim.shape[1]
        mask = ~torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        # Mean absolute off-diagonal similarity
        ortho_loss = sim.abs().masked_select(mask).mean()
        return {"loss": ortho_loss, "mean_cos": ortho_loss.detach()}


class InfoNCEFamilyLoss(nn.Module):
    """Supervised InfoNCE using family labels.

    Pulls together proteins from the same family, pushes apart different families.
    Uses label information from the training batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """
        Args:
            z: (B, D) normalized embeddings
            labels: (B,) integer family labels
        """
        z = F.normalize(z, dim=-1)
        B = z.shape[0]
        if B < 2:
            zero = torch.tensor(0.0, device=z.device)
            return {"loss": zero, "n_positives": 0}

        # Similarity matrix
        sim = z @ z.T / self.temperature  # (B, B)

        # Positive mask: same family, excluding self
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
        pos_mask = label_eq & self_mask

        n_positives = pos_mask.sum().item()
        if n_positives == 0:
            return {"loss": torch.tensor(0.0, device=z.device), "n_positives": 0}

        # For each anchor, compute log-softmax over all non-self entries
        # Then average over positive positions
        logits = sim.masked_fill(~self_mask, float("-inf"))
        log_prob = logits - logits.logsumexp(dim=1, keepdim=True)

        # Mean log probability of positives (index directly to avoid 0 * -inf = NaN)
        loss = -log_prob[pos_mask].mean()

        return {"loss": loss, "n_positives": int(n_positives)}


class MeanPoolReconLoss(nn.Module):
    """Reconstruction loss on mean-pooled vectors.

    For models that output pooled_input and pooled_recon in forward().
    """

    def forward(self, pooled_pred: Tensor, pooled_target: Tensor) -> dict[str, Tensor]:
        mse = F.mse_loss(pooled_pred, pooled_target)
        cos_loss = 1.0 - F.cosine_similarity(pooled_pred, pooled_target, dim=-1).mean()
        loss = mse + 0.5 * cos_loss
        return {"loss": loss, "mse": mse.detach(), "cos_loss": cos_loss.detach()}
