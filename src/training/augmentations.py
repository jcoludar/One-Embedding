"""Augmentations for contrastive learning on residue embeddings."""

import torch
from torch import Tensor


def random_crop(states: Tensor, mask: Tensor, min_frac: float = 0.5) -> tuple[Tensor, Tensor]:
    """Randomly crop a contiguous sub-sequence from each protein in the batch.

    Returns cropped states and mask with the same (B, L, D) shape (zero-padded).
    """
    B, L, D = states.shape
    new_states = torch.zeros_like(states)
    new_mask = torch.zeros_like(mask)

    for i in range(B):
        valid_len = mask[i].sum().int().item()
        if valid_len < 2:
            new_states[i] = states[i]
            new_mask[i] = mask[i]
            continue

        crop_len = max(2, int(valid_len * (min_frac + (1.0 - min_frac) * torch.rand(1).item())))
        start = torch.randint(0, valid_len - crop_len + 1, (1,)).item()
        new_states[i, :crop_len] = states[i, start : start + crop_len]
        new_mask[i, :crop_len] = 1.0

    return new_states, new_mask


def random_mask_residues(states: Tensor, mask: Tensor, mask_prob: float = 0.15) -> tuple[Tensor, Tensor]:
    """Zero out random residue positions (like BERT masking on embeddings)."""
    noise_mask = torch.rand(states.shape[:2], device=states.device) < mask_prob
    noise_mask = noise_mask & mask.bool()
    augmented = states.clone()
    augmented[noise_mask] = 0.0
    return augmented, mask


def gaussian_noise(states: Tensor, mask: Tensor, std: float = 0.1) -> tuple[Tensor, Tensor]:
    """Add Gaussian noise to residue embeddings."""
    noise = torch.randn_like(states) * std
    augmented = states + noise * mask.unsqueeze(-1).float()
    return augmented, mask
