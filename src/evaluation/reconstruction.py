"""Reconstruction quality metrics: MSE, cosine similarity of residue recovery."""

import numpy as np
import torch
from torch import Tensor

from src.compressors.base import SequenceCompressor
from src.utils.device import get_device


def evaluate_reconstruction(
    model: SequenceCompressor,
    embeddings: dict[str, np.ndarray],
    device: torch.device | None = None,
    max_len: int = 512,
) -> dict[str, float]:
    """Evaluate reconstruction quality across all proteins.

    Returns dict with: mse, cosine_sim, mse_std, cosine_sim_std, n_proteins.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_mse = []
    all_cosine = []

    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            emb_trunc = emb[:L]

            states = torch.from_numpy(emb_trunc).unsqueeze(0).to(device)  # (1, L, D)
            mask = torch.ones(1, L, device=device)

            output = model(states, mask)
            recon = output["reconstructed"]

            # MSE per residue
            diff = (recon[0, :L] - states[0, :L]).pow(2).mean(dim=-1)  # (L,)
            all_mse.append(diff.mean().item())

            # Cosine similarity per residue
            cos = torch.nn.functional.cosine_similarity(recon[0, :L], states[0, :L], dim=-1)
            all_cosine.append(cos.mean().item())

    return {
        "mse": float(np.mean(all_mse)),
        "cosine_sim": float(np.mean(all_cosine)),
        "mse_std": float(np.std(all_mse)),
        "cosine_sim_std": float(np.std(all_cosine)),
        "n_proteins": len(embeddings),
    }
