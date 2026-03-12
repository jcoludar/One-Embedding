"""End-to-end pipeline: sequences → OneEmbedding objects."""

from pathlib import Path

import numpy as np
import torch

from src.one_embedding.embedding import OneEmbedding
from src.one_embedding.registry import PLMRegistry
from src.one_embedding.transforms import (
    dct_summary,
    haar_summary,
    spectral_fingerprint,
    spectral_moments,
)
from src.utils.device import get_device


def compress_embeddings(
    model,
    embeddings: dict[str, np.ndarray],
    device: torch.device | None = None,
    max_len: int = 512,
) -> dict[str, np.ndarray]:
    """Compress per-residue embeddings using a trained ChannelCompressor.

    Args:
        model: Trained ChannelCompressor.
        embeddings: {protein_id: (L, D) ndarray} raw PLM embeddings.
        device: Device for inference.
        max_len: Maximum sequence length.

    Returns:
        {protein_id: (L, d) ndarray} compressed embeddings.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    compressed = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)  # (1, L, d)
            compressed[pid] = latent[0].cpu().numpy()

    return compressed


def apply_transform(
    matrix: np.ndarray,
    transform: str,
    K: int = 8,
    n_bands: int = 8,
    haar_levels: int = 3,
) -> np.ndarray:
    """Apply a transform to get a fixed-size summary from per-residue matrix.

    Args:
        matrix: (L, d) compressed per-residue embeddings.
        transform: One of "mean", "dct", "spectral", "spectral_moments", "haar".
        K: Number of DCT coefficients.
        n_bands: Number of PSD bands for spectral fingerprint.
        haar_levels: Number of Haar decomposition levels.

    Returns:
        Fixed-size summary vector.
    """
    if transform == "mean":
        return matrix.mean(axis=0).astype(np.float32)
    elif transform == "dct":
        return dct_summary(matrix, K=K)
    elif transform == "spectral":
        return spectral_fingerprint(matrix, n_bands=n_bands)
    elif transform == "spectral_moments":
        return spectral_moments(matrix)
    elif transform == "haar":
        return haar_summary(matrix, levels=haar_levels)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def encode_one_embedding(
    plm_name: str,
    pre_extracted: dict[str, np.ndarray] | None = None,
    fasta_dict: dict[str, str] | None = None,
    transform: str = "dct",
    K: int = 8,
    n_bands: int = 8,
    haar_levels: int = 3,
    latent_dim: int = 256,
    checkpoint_base: Path | str = Path("data/checkpoints/channel"),
    device: torch.device | None = None,
    max_len: int = 512,
    pre_compressed: dict[str, np.ndarray] | None = None,
) -> dict[str, OneEmbedding]:
    """Full pipeline: sequences → OneEmbedding objects.

    Args:
        plm_name: PLM name (e.g. "prot_t5_xl", "esm2_650m").
        pre_extracted: Pre-extracted raw PLM embeddings {id: (L, D)}.
        fasta_dict: Sequences to extract embeddings from (alternative to pre_extracted).
        transform: Transform for protein-level summary ("mean", "dct", "spectral", "haar").
        K: Number of DCT coefficients.
        n_bands: Number of PSD bands for spectral fingerprint.
        haar_levels: Haar wavelet decomposition levels.
        latent_dim: Compressed embedding dimension.
        checkpoint_base: Path to compressor checkpoints.
        device: Device for PLM and compressor inference.
        max_len: Maximum sequence length.
        pre_compressed: Pre-compressed embeddings {id: (L, d)} — skip compression.

    Returns:
        {protein_id: OneEmbedding}
    """
    registry = PLMRegistry(checkpoint_base=checkpoint_base)

    # Step 1: Get raw embeddings
    if pre_compressed is not None:
        compressed = pre_compressed
    else:
        if pre_extracted is None:
            if fasta_dict is None:
                raise ValueError("Provide pre_extracted, fasta_dict, or pre_compressed.")
            extractor = registry.get_extractor(plm_name)
            pre_extracted = extractor(fasta_dict, device=device)

        # Step 2: Compress
        compressor = registry.get_compressor(plm_name, latent_dim=latent_dim, device=device)
        compressed = compress_embeddings(
            compressor, pre_extracted, device=device, max_len=max_len
        )

    # Step 3: Apply transform and wrap
    result = {}
    for pid, matrix in compressed.items():
        summary = apply_transform(
            matrix, transform, K=K, n_bands=n_bands, haar_levels=haar_levels
        )
        result[pid] = OneEmbedding.from_compressed(
            protein_id=pid,
            plm=plm_name,
            matrix=matrix,
            transform=transform,
            summary=summary,
        )

    return result
