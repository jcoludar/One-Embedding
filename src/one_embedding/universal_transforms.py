"""Universal training-free transforms for per-residue → per-protein pooling.

All functions take (L, D) per-residue embeddings and return fixed-size vectors.
No learned parameters. PLM-agnostic.
"""

import numpy as np


def power_mean_pool(matrix: np.ndarray, p: float = 3.0) -> np.ndarray:
    """Generalized power mean pooling.

    p=1: arithmetic mean (standard). p=2: root mean square.
    p>1 emphasizes high-magnitude residues.

    Args:
        matrix: (L, D) per-residue embeddings.
        p: Power parameter.

    Returns:
        (D,) power mean vector.
    """
    if abs(p - 1.0) < 1e-8:
        return matrix.mean(axis=0).astype(np.float32)

    abs_matrix = np.abs(matrix).clip(1e-8)
    signs = np.sign(matrix.mean(axis=0))
    power_mean = np.mean(abs_matrix ** p, axis=0) ** (1.0 / p)
    return (signs * power_mean).astype(np.float32)


def norm_weighted_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean pool weighted by residue embedding L2 norms.

    Residues with higher norms contribute more — hypothesis: PLMs
    encode more information in higher-norm embeddings.

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (D,) norm-weighted mean vector.
    """
    norms = np.linalg.norm(matrix, axis=1)  # (L,)
    weights = norms / norms.sum().clip(1e-8)  # (L,) normalized
    return (weights[:, np.newaxis] * matrix).sum(axis=0).astype(np.float32)


def kernel_mean_embedding(
    matrix: np.ndarray,
    D_out: int = 2048,
    gamma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Kernel mean embedding via Random Fourier Features.

    Approximates the mean embedding in an RKHS with RBF kernel.
    Captures the DISTRIBUTION of residue embeddings, not just the mean.

    Reference: Rahimi & Recht (2007). Random Features for Large-Scale
    Kernel Machines.

    Args:
        matrix: (L, D) per-residue embeddings.
        D_out: Output dimensionality.
        gamma: RBF kernel bandwidth.
        seed: Fixed seed for random projection.

    Returns:
        (D_out,) kernel mean embedding.
    """
    L, D = matrix.shape
    rng = np.random.RandomState(seed)
    W = rng.randn(D, D_out).astype(np.float32) * np.sqrt(2 * gamma)
    b = rng.uniform(0, 2 * np.pi, D_out).astype(np.float32)

    projected = matrix @ W + b  # (L, D_out)
    features = np.sqrt(2.0 / D_out) * np.cos(projected)  # (L, D_out)
    return features.mean(axis=0).astype(np.float32)


def svd_spectrum(matrix: np.ndarray, k: int = 16) -> np.ndarray:
    """Top-k singular values as protein fingerprint.

    Captures the intrinsic dimensionality and energy distribution of
    the embedding cloud. Compact but loses directional info.

    Args:
        matrix: (L, D) per-residue embeddings.
        k: Number of singular values to keep.

    Returns:
        (k,) singular values in descending order, zero-padded if needed.
    """
    S = np.linalg.svd(matrix, compute_uv=False)
    if len(S) < k:
        S = np.pad(S, (0, k - len(S)))
    return S[:k].astype(np.float32)
