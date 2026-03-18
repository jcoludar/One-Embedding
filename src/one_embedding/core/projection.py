"""Random orthogonal projection for Johnson-Lindenstrauss dimensionality reduction.

Projects per-residue embeddings from dimension D down to d_out using a
seeded random orthogonal matrix. Pairwise distances are preserved within
(1 ± eps) with high probability (JL lemma).

The JL scaling factor sqrt(D / d_out) is applied so that embedding norms
are preserved in expectation, making the projected space metrically
compatible with the original.
"""

import numpy as np


def project(
    X: np.ndarray,
    d_out: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Project (L, D) embeddings to (L, d_out) via random orthogonal matrix.

    The projection matrix R is seeded and deterministic — same seed produces
    the same R for any input of the same D. This serves as the "codec key":
    a different PLM dimension (D) requires a different R, but the seed fixes
    which R is used.

    The JL scaling factor sqrt(D / d_out) is applied so that:
        E[||R x||²] = ||x||²
    preserving norms in expectation.

    Args:
        X: (L, D) per-residue embeddings, float32.
        d_out: Target dimensionality (default 512).
        seed: Random seed for the projection matrix (default 42).

    Returns:
        (L, d_out) projected embeddings, float32.
    """
    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape

    # Build deterministic orthogonal projection matrix
    rng = np.random.RandomState(seed)
    G = rng.randn(D, d_out).astype(np.float32)
    # QR decomposition: Q has orthonormal columns — (D, d_out) for D >= d_out
    Q, _ = np.linalg.qr(G, mode="reduced")  # Q: (D, min(D, d_out))

    # JL scaling: multiply by sqrt(D / d_out) to preserve norms
    scale = np.sqrt(float(D) / float(d_out))
    R = Q * scale  # (D, d_out)

    return (X @ R).astype(np.float32)
