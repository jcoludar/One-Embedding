"""Universal training-free transforms for per-residue → per-protein pooling.

All functions take (L, D) per-residue embeddings and return fixed-size vectors.
No learned parameters. PLM-agnostic.
"""

from typing import Optional

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


def feature_hash(
    matrix: np.ndarray,
    d_out: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Feature hashing (hashing trick) — compress features D → d_out.

    Maps each input feature dimension to one of d_out output bins using
    deterministic hash functions. Sign-flips preserve inner products in
    expectation (unbiased estimator). Works for ANY input dimension D.

    This is the "universal adapter" — same d_out output from ProtT5(1024),
    ESM2(1280), ESM-C(960), or any future PLM. No matrix stored.

    Reference: Weinberger et al. (2009). Feature Hashing for Large Scale
    Multitask Learning.

    Args:
        matrix: (L, D) per-residue embeddings.
        d_out: Output dimensionality.
        seed: Fixed seed for hash functions (the "codec key").

    Returns:
        (L, d_out) feature-hashed per-residue embeddings.
    """
    L, D = matrix.shape
    rng = np.random.RandomState(seed)
    # Hash function: feature_dim → {0, ..., d_out-1}
    h = rng.randint(0, d_out, size=D).astype(np.int32)
    # Sign function: feature_dim → {-1, +1}
    s = rng.choice([-1, 1], size=D).astype(np.float32)

    # Apply: output[:, h[j]] += s[j] * input[:, j]
    output = np.zeros((L, d_out), dtype=np.float32)
    for j in range(D):
        output[:, h[j]] += s[j] * matrix[:, j]

    return output


def random_orthogonal_project(
    matrix: np.ndarray,
    d_out: int = 256,
    seed: int = 42,
    projection_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Random orthogonal projection D → d_out (Johnson-Lindenstrauss).

    Projects each residue embedding through a fixed random orthogonal matrix.
    JL theorem guarantees pairwise distances are preserved within (1 +/- eps)
    with high probability when d_out = O(log(n) / eps^2).

    The projection matrix R is deterministic (seeded) and serves as the
    "codec key" — same R for all proteins. Unlike feature hashing, R depends
    on D, so a different R is needed per PLM dimension.

    Args:
        matrix: (L, D) per-residue embeddings.
        d_out: Output dimensionality.
        seed: Fixed seed for projection matrix.
        projection_matrix: Pre-computed (D, d_out) matrix (for batch use).

    Returns:
        (L, d_out) projected per-residue embeddings.
    """
    L, D = matrix.shape

    if projection_matrix is not None:
        R = projection_matrix
    else:
        rng = np.random.RandomState(seed)
        # Generate random Gaussian matrix and orthogonalize
        R = rng.randn(D, d_out).astype(np.float32)
        # QR decomposition for orthogonal columns
        Q, _ = np.linalg.qr(R, mode="reduced")
        R = Q * np.sqrt(D / d_out)  # Scale to preserve norms

    return (matrix @ R).astype(np.float32)


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
