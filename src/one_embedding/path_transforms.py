"""Path geometry transforms for protein embedding trajectories.

Treats protein per-residue embeddings (L, D) as a discrete path through R^D.
Applies concepts from rough path theory, differential geometry, and polymer
physics to extract training-free protein-level and residue-level features.

All functions are training-free and use only numpy/scipy.
"""

import numpy as np
from scipy.fft import dct, idct


# ── Displacement Encoding ────────────────────────────────────────


def displacement_encode(matrix: np.ndarray) -> np.ndarray:
    """First differences of the embedding trajectory.

    dx_i = x_{i+1} - x_i. Displacements are sparser than raw embeddings
    for smooth trajectories (neighboring residues have similar embeddings).

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (L-1, D) displacement vectors.
    """
    return np.diff(matrix, axis=0).astype(np.float32)


def displacement_decode(
    displacements: np.ndarray, x0: np.ndarray
) -> np.ndarray:
    """Reconstruct path from displacements via cumulative sum.

    x_i = x_0 + sum_{j=0}^{i-1} dx_j. Lossless reconstruction.

    Args:
        displacements: (L-1, D) displacement vectors.
        x0: (D,) starting position.

    Returns:
        (L, D) reconstructed per-residue embeddings.
    """
    path = np.vstack([x0[np.newaxis, :], displacements])
    return np.cumsum(path, axis=0).astype(np.float32)


def displacement_dct(matrix: np.ndarray, K: int = 4) -> np.ndarray:
    """DCT of displacement sequence — potentially more compressible than raw.

    Args:
        matrix: (L, D) per-residue embeddings.
        K: Number of DCT coefficients to keep.

    Returns:
        Flattened DCT coefficients of displacements, shape (K * D,).
    """
    dx = np.diff(matrix, axis=0)  # (L-1, D)
    K = min(K, dx.shape[0])
    coeffs = dct(dx, type=2, axis=0, norm="ortho")[:K]
    return coeffs.ravel().astype(np.float32)


def inverse_displacement_dct(
    coeffs_flat: np.ndarray, D: int, target_len: int, x0: np.ndarray
) -> np.ndarray:
    """Reconstruct per-residue from DCT of displacements.

    Args:
        coeffs_flat: Flattened DCT coefficients (K * D,).
        D: Embedding dimension.
        target_len: Original protein length L.
        x0: (D,) starting position.

    Returns:
        (L, D) reconstructed embeddings.
    """
    K = coeffs_flat.shape[0] // D
    coeffs = coeffs_flat.reshape(K, D)
    dx_reconstructed = idct(coeffs, type=2, axis=0, norm="ortho", n=target_len - 1)
    return displacement_decode(dx_reconstructed, x0)


# ── Path Signatures (Rough Path Theory) ─────────────────────────


def path_signature_depth2(matrix: np.ndarray) -> np.ndarray:
    """Truncated path signature at depth 2.

    The signature is a universal nonlinear feature of paths (Lyons' theorem).
    Depth 1 = total displacement. Depth 2 = iterated area integrals capturing
    sequential correlation structure.

    For a discrete path x_0,...,x_{L-1} in R^d:
        S^1_i = sum_k dx_k^i                        (d terms)
        S^2_{i,j} = sum_{s<t} dx_s^i * dx_t^j       (d^2 terms)

    Args:
        matrix: (L, d) path (use low d, e.g., random-projected to 16-32).

    Returns:
        (1 + d + d^2,) signature vector.
    """
    L, d = matrix.shape
    dx = np.diff(matrix, axis=0)  # (L-1, d)

    # Depth 0: scalar 1
    sig0 = np.array([1.0], dtype=np.float32)

    # Depth 1: total displacement
    sig1 = dx.sum(axis=0).astype(np.float32)  # (d,)

    # Depth 2: iterated integral via running sum
    # S^2_{i,j} = sum_{t} dx_t^j * (sum_{s<t} dx_s^i)
    running = np.zeros(d, dtype=np.float64)
    sig2 = np.zeros((d, d), dtype=np.float64)
    for t in range(len(dx)):
        sig2 += np.outer(running, dx[t])
        running += dx[t]

    return np.concatenate([sig0, sig1, sig2.ravel().astype(np.float32)])


def path_signature_depth3(matrix: np.ndarray) -> np.ndarray:
    """Truncated path signature at depth 3.

    Adds third-order iterated integrals capturing volume-like structure.
    Only practical for small d (e.g., d=16 gives 1+16+256+4096 = 4369 dims).

    Args:
        matrix: (L, d) path (use low d).

    Returns:
        (1 + d + d^2 + d^3,) signature vector.
    """
    L, d = matrix.shape
    dx = np.diff(matrix, axis=0)

    sig0 = np.array([1.0], dtype=np.float32)
    sig1 = dx.sum(axis=0).astype(np.float32)

    # Depth 2 and running sums for depth 3
    running1 = np.zeros(d, dtype=np.float64)  # sum_{s<t} dx_s
    sig2 = np.zeros((d, d), dtype=np.float64)
    running2 = np.zeros((d, d), dtype=np.float64)  # sum_{s<t} S2_partial
    sig3 = np.zeros((d, d, d), dtype=np.float64)

    for t in range(len(dx)):
        # Depth 3: S^3_{i,j,k} += running2_{i,j} * dx_t^k
        sig3 += running2[:, :, np.newaxis] * dx[t][np.newaxis, np.newaxis, :]
        # Depth 2: S^2_{i,j} += running1_i * dx_t^j
        increment2 = np.outer(running1, dx[t])
        sig2 += increment2
        running2 += increment2
        running1 += dx[t]

    return np.concatenate([
        sig0,
        sig1,
        sig2.ravel().astype(np.float32),
        sig3.ravel().astype(np.float32),
    ])


# ── Cross-Covariance ────────────────────────────────────────────


def lag_cross_covariance_eigenvalues(
    matrix: np.ndarray, k: int = 64
) -> np.ndarray:
    """Top-k singular values of lag-1 cross-covariance matrix.

    C = (1/(L-1)) * sum_i (x_i - mu) @ (x_{i+1} - mu)^T

    Captures temporal correlation structure — what mean pool misses.
    Two proteins with identical residue distributions but different
    orderings have the same mean pool but different cross-covariance.

    Args:
        matrix: (L, D) per-residue embeddings.
        k: Number of singular values to return.

    Returns:
        (k,) top singular values, zero-padded if needed.
    """
    L, D = matrix.shape
    mu = matrix.mean(axis=0)
    centered = matrix - mu

    # Cross-covariance: C = centered[:-1].T @ centered[1:] / (L-1)
    C = (centered[:-1].T @ centered[1:]) / max(L - 1, 1)

    # Singular values (C is not symmetric)
    # Move to CPU float64 for stability
    S = np.linalg.svd(C.astype(np.float64), compute_uv=False)

    if len(S) < k:
        S = np.pad(S, (0, k - len(S)))
    return S[:k].astype(np.float32)


# ── Discrete Curvature ──────────────────────────────────────────


def discrete_curvature(matrix: np.ndarray) -> np.ndarray:
    """Discrete curvature at each interior residue position.

    κ_i = ||Δ²x_i|| / ||Δx_i|| where Δx_i = x_{i+1} - x_i
    and Δ²x_i = Δx_{i+1} - Δx_i = x_{i+2} - 2x_{i+1} + x_i.

    High curvature = sharp turn in embedding trajectory.

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (L-2,) curvature values.
    """
    dx = np.diff(matrix, axis=0)  # (L-1, D)
    ddx = np.diff(dx, axis=0)  # (L-2, D)

    dx_norms = np.linalg.norm(dx[:-1], axis=1).clip(1e-8)  # (L-2,)
    ddx_norms = np.linalg.norm(ddx, axis=1)  # (L-2,)

    return (ddx_norms / dx_norms).astype(np.float32)


def displacement_magnitude(matrix: np.ndarray) -> np.ndarray:
    """Step size ||dx_i|| at each position — the "speed" along the path.

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (L-1,) displacement magnitudes.
    """
    dx = np.diff(matrix, axis=0)
    return np.linalg.norm(dx, axis=1).astype(np.float32)


def curvature_enriched(matrix: np.ndarray) -> np.ndarray:
    """Augment per-residue embeddings with geometric features.

    Appends curvature, displacement magnitude, and position fraction
    to each residue's embedding vector.

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (L, D+3) augmented embeddings.
    """
    L, D = matrix.shape

    # Displacement magnitude (pad to L with edge values)
    speed = displacement_magnitude(matrix)  # (L-1,)
    speed_padded = np.concatenate([speed, [speed[-1]]]) if L > 1 else np.zeros(L)

    # Curvature (pad to L with edge values)
    if L >= 3:
        curv = discrete_curvature(matrix)  # (L-2,)
        curv_padded = np.concatenate([[curv[0]], curv, [curv[-1]]])
    else:
        curv_padded = np.zeros(L)

    # Position fraction (0 to 1 along protein)
    pos_frac = np.linspace(0, 1, L)

    extra = np.column_stack([speed_padded, curv_padded, pos_frac])
    return np.hstack([matrix, extra]).astype(np.float32)


# ── Gyration Tensor / Shape Descriptors ─────────────────────────


def gyration_eigenspectrum(
    matrix: np.ndarray, k: int = 64
) -> np.ndarray:
    """Top-k eigenvalues of the gyration tensor (covariance matrix).

    The gyration tensor G = (1/L) Σ (x_i - μ)(x_i - μ)^T describes the
    shape of the residue embedding cloud. Eigenvalues capture spread
    along principal axes. This is NOT global PCA — it's computed
    per-protein with no cross-protein fitting.

    Args:
        matrix: (L, D) per-residue embeddings.
        k: Number of eigenvalues to return.

    Returns:
        (k,) eigenvalues in descending order, zero-padded if needed.
    """
    L, D = matrix.shape
    centered = matrix - matrix.mean(axis=0)

    # For L < D, compute L×L gram matrix instead (faster)
    if L < D:
        gram = (centered @ centered.T) / L
        eigvals = np.linalg.eigvalsh(gram.astype(np.float64))
        eigvals = np.sort(eigvals)[::-1]
    else:
        cov = (centered.T @ centered) / L
        eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
        eigvals = np.sort(eigvals)[::-1]

    eigvals = np.maximum(eigvals, 0)  # numerical stability
    if len(eigvals) < k:
        eigvals = np.pad(eigvals, (0, k - len(eigvals)))
    return eigvals[:k].astype(np.float32)


def shape_descriptors(matrix: np.ndarray) -> np.ndarray:
    """Polymer physics shape descriptors from the gyration tensor.

    Returns:
        (5,) array: [radius_of_gyration, asphericity, acylindricity,
                      relative_shape_anisotropy, effective_dimensionality]
    """
    eigvals = gyration_eigenspectrum(matrix, k=min(matrix.shape))
    eigvals = eigvals[eigvals > 1e-10]  # non-zero eigenvalues

    if len(eigvals) < 2:
        return np.zeros(5, dtype=np.float32)

    # Radius of gyration
    rg = np.sqrt(eigvals.sum())

    # Top 3 eigenvalues for shape descriptors
    lam = np.sort(eigvals)[::-1][:3]
    if len(lam) < 3:
        lam = np.pad(lam, (0, 3 - len(lam)))

    # Asphericity: deviation from spherical symmetry
    asphericity = lam[0] - 0.5 * (lam[1] + lam[2])

    # Acylindricity: deviation from cylindrical symmetry
    acylindricity = lam[1] - lam[2]

    # Relative shape anisotropy (0=sphere, 1=rod)
    trace = lam.sum()
    trace_sq = trace ** 2
    if trace_sq > 1e-10:
        rsa = 1.0 - 3.0 * (lam[0] * lam[1] + lam[1] * lam[2] + lam[0] * lam[2]) / trace_sq
    else:
        rsa = 0.0

    # Effective dimensionality (participation ratio of eigenvalues)
    all_eigvals = eigvals[eigvals > 1e-10]
    p = all_eigvals / all_eigvals.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_dim = np.exp(entropy)

    return np.array([rg, asphericity, acylindricity, rsa, eff_dim],
                    dtype=np.float32)


# ── Path Statistics (Composite) ─────────────────────────────────


def path_statistics(matrix: np.ndarray) -> np.ndarray:
    """Comprehensive path-aware scalar features.

    Combines displacement, curvature, and shape statistics into
    a compact fingerprint of the embedding trajectory.

    Args:
        matrix: (L, D) per-residue embeddings.

    Returns:
        (~30,) scalar path features.
    """
    L, D = matrix.shape
    features = []

    # Displacement stats
    speed = displacement_magnitude(matrix)
    features.extend([
        speed.mean(), speed.std(),
        np.median(speed),
        np.percentile(speed, 5), np.percentile(speed, 95),
    ])

    # End-to-end distance
    end_to_end = np.linalg.norm(matrix[-1] - matrix[0])
    features.append(end_to_end)

    # Contour length (total path length)
    contour_length = speed.sum()
    features.append(contour_length)

    # End-to-end / contour length ratio (1 = straight line, 0 = coiled)
    if contour_length > 1e-8:
        features.append(end_to_end / contour_length)
    else:
        features.append(0.0)

    # Curvature stats
    if L >= 3:
        curv = discrete_curvature(matrix)
        features.extend([
            curv.mean(), curv.std(),
            np.median(curv),
            np.percentile(curv, 5), np.percentile(curv, 95),
        ])
    else:
        features.extend([0.0] * 5)

    # Shape descriptors
    sd = shape_descriptors(matrix)
    features.extend(sd.tolist())

    # Mean squared displacement (MSD) at lag 1, 4, 16
    for lag in [1, 4, 16]:
        if L > lag:
            msd = np.mean(np.sum((matrix[lag:] - matrix[:-lag]) ** 2, axis=1))
            features.append(msd)
        else:
            features.append(0.0)

    # Autocorrelation of displacement directions
    if L >= 3:
        dx = np.diff(matrix, axis=0)
        dx_norms = np.linalg.norm(dx, axis=1, keepdims=True).clip(1e-8)
        unit_dx = dx / dx_norms
        # Lag-1 direction autocorrelation
        autocorr = np.sum(unit_dx[:-1] * unit_dx[1:], axis=1).mean()
        features.append(autocorr)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
