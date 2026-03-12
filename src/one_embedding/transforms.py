"""Mathematical transforms for One Embedding protein-level summaries.

All transforms operate on compressed per-residue matrices (L, d) and produce
fixed-size protein-level vectors. Inspired by signal processing and biophysics:

- DCT: Fourier-family spectral decomposition by frequency
- Spectral Fingerprint: Power spectral density (Brillouin-inspired, phase-free)
- Haar Wavelet: Fractal tree decomposition by spatial scale
"""

import numpy as np
from scipy.fft import dct, idct


# ---------------------------------------------------------------------------
# DCT (Discrete Cosine Transform)
# ---------------------------------------------------------------------------

def dct_summary(matrix: np.ndarray, K: int = 8) -> np.ndarray:
    """Extract first K DCT coefficients as a fixed-size protein summary.

    Treats each of the d embedding dimensions as a 1D signal along the protein
    sequence, decomposes by frequency. Coefficient 0 ≈ mean pool (DC component).
    Higher coefficients capture progressively finer spatial structure.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        K: Number of frequency coefficients to keep. K=1 ≈ mean pool.

    Returns:
        Flattened summary vector of shape (K * d,).
    """
    L = matrix.shape[0]
    K = min(K, L)
    coeffs = dct(matrix, type=2, axis=0, norm="ortho")  # (L, d)
    return coeffs[:K].ravel().astype(np.float32)


def inverse_dct(coeffs: np.ndarray, d: int, target_len: int) -> np.ndarray:
    """Reconstruct per-residue embeddings from DCT coefficients.

    Args:
        coeffs: Flattened DCT coefficients, shape (K * d,).
        d: Embedding dimension (to reshape).
        target_len: Original sequence length L.

    Returns:
        Reconstructed matrix of shape (target_len, d).
    """
    K = coeffs.shape[0] // d
    coeff_matrix = coeffs.reshape(K, d)
    reconstructed = idct(coeff_matrix, type=2, axis=0, norm="ortho", n=target_len)
    return reconstructed.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral Fingerprint (Brillouin-inspired PSD)
# ---------------------------------------------------------------------------

def spectral_fingerprint(
    matrix: np.ndarray,
    n_bands: int = 8,
) -> np.ndarray:
    """Brillouin-inspired power spectral density fingerprint.

    Computes the power spectrum |DCT|² of the embedding signal, then bins
    into frequency bands. Phase-free: two proteins with similar structure but
    different domain arrangements have similar PSD but different DCT phase.

    Analogous to how Brillouin/Raman spectroscopy captures vibrational mode
    energies without sensitivity to specific atomic positions.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        n_bands: Number of frequency bands to bin into.

    Returns:
        Band energy vector of shape (n_bands * d,).
    """
    coeffs = dct(matrix, type=2, axis=0, norm="ortho")  # (L, d)
    psd = coeffs ** 2  # power spectral density

    # Bin into n_bands frequency bands
    L = psd.shape[0]
    n_bands = min(n_bands, L)
    bands = np.array_split(psd, n_bands, axis=0)
    band_energies = np.array([b.sum(axis=0) for b in bands])  # (n_bands, d)

    return band_energies.ravel().astype(np.float32)


def spectral_moments(matrix: np.ndarray, n_moments: int = 4) -> np.ndarray:
    """Compute spectral moments of the embedding PSD.

    Returns compact features: centroid, bandwidth, skewness, kurtosis
    of the power spectrum for each embedding dimension.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        n_moments: Number of moments (1=centroid, 2=+bandwidth, etc.).

    Returns:
        Spectral moment features of shape (n_moments * d,).
    """
    coeffs = dct(matrix, type=2, axis=0, norm="ortho")
    psd = coeffs ** 2  # (L, d)
    L = psd.shape[0]

    # Normalized frequencies
    freqs = np.arange(L, dtype=np.float32)

    # Normalize PSD per dimension (avoid division by zero)
    total_power = psd.sum(axis=0, keepdims=True).clip(1e-12)
    psd_norm = psd / total_power  # (L, d) — probability distribution over frequencies

    moments = []

    # Moment 1: Spectral centroid (mean frequency)
    centroid = (freqs[:, None] * psd_norm).sum(axis=0)  # (d,)
    moments.append(centroid)

    if n_moments >= 2:
        # Moment 2: Spectral bandwidth (std of frequency)
        variance = ((freqs[:, None] - centroid[None, :]) ** 2 * psd_norm).sum(axis=0)
        bandwidth = np.sqrt(variance.clip(0))
        moments.append(bandwidth)

    if n_moments >= 3:
        # Moment 3: Spectral skewness
        std = bandwidth.clip(1e-12)
        skewness = (
            ((freqs[:, None] - centroid[None, :]) ** 3 * psd_norm).sum(axis=0)
            / std**3
        )
        moments.append(skewness)

    if n_moments >= 4:
        # Moment 4: Spectral kurtosis
        kurtosis = (
            ((freqs[:, None] - centroid[None, :]) ** 4 * psd_norm).sum(axis=0)
            / std**4
        )
        moments.append(kurtosis)

    return np.concatenate(moments).astype(np.float32)


# ---------------------------------------------------------------------------
# Haar Wavelet Decomposition
# ---------------------------------------------------------------------------

def haar_summary(matrix: np.ndarray, levels: int = 3) -> np.ndarray:
    """Haar wavelet decomposition — fractal tree of embedding differences.

    Recursively computes pairwise averages (approximation) and differences
    (detail) along the sequence axis. Each level captures structure at
    a different spatial scale. Detail coefficients at each level are
    mean-pooled to a single d-dimensional vector, giving a compact
    multi-scale summary.

    Output layout:
        [approx_pooled(d) | detail_coarsest_pooled(d) | ... | detail_finest_pooled(d)]
        = (levels + 1) × d dimensions

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        levels: Number of decomposition levels.

    Returns:
        Multi-scale summary vector of shape ((levels + 1) * d,).
    """
    L, d = matrix.shape

    # Pad to next power of 2 if necessary
    padded_len = 1
    while padded_len < L:
        padded_len *= 2

    if padded_len != L:
        padded = np.zeros((padded_len, d), dtype=np.float32)
        padded[:L] = matrix
    else:
        padded = matrix.astype(np.float32)

    approx = padded.copy()
    detail_pools = []

    for level in range(levels):
        n = approx.shape[0]
        if n < 2:
            break
        half = n // 2
        pairs = approx.reshape(half, 2, d)
        new_approx = (pairs[:, 0] + pairs[:, 1]) / np.sqrt(2)
        detail = (pairs[:, 0] - pairs[:, 1]) / np.sqrt(2)
        detail_pools.append(detail.mean(axis=0))  # pool to (d,)
        approx = new_approx

    # Final approximation: pool to single vector
    approx_pooled = approx.mean(axis=0)  # (d,)

    # Concatenate: approx + details from coarsest to finest
    result = [approx_pooled]
    for dp in reversed(detail_pools):
        result.append(dp)

    return np.concatenate(result).astype(np.float32)


def haar_full_coefficients(matrix: np.ndarray, levels: int = 3):
    """Full Haar wavelet decomposition (for inverse transform).

    Returns the raw wavelet coefficients at each level without pooling.

    Returns:
        (approx, details): approx is (n, d) array, details is list of (n_i, d) arrays.
    """
    L, d = matrix.shape

    padded_len = 1
    while padded_len < L:
        padded_len *= 2

    if padded_len != L:
        padded = np.zeros((padded_len, d), dtype=np.float32)
        padded[:L] = matrix
    else:
        padded = matrix.astype(np.float32)

    approx = padded.copy()
    details = []

    for level in range(levels):
        n = approx.shape[0]
        if n < 2:
            break
        half = n // 2
        pairs = approx.reshape(half, 2, d)
        new_approx = (pairs[:, 0] + pairs[:, 1]) / np.sqrt(2)
        detail = (pairs[:, 0] - pairs[:, 1]) / np.sqrt(2)
        details.append(detail)
        approx = new_approx

    return approx, details


def inverse_haar(
    approx: np.ndarray,
    details: list[np.ndarray],
    target_len: int,
) -> np.ndarray:
    """Reconstruct per-residue embeddings from full Haar wavelet coefficients.

    Use with haar_full_coefficients() output. Details are ordered from
    coarsest to finest (same order as haar_full_coefficients returns).

    Args:
        approx: Coarsest approximation coefficients (n, d).
        details: List of detail coefficient arrays, coarsest to finest.
        target_len: Original sequence length L.

    Returns:
        Reconstructed matrix of shape (target_len, d).
    """
    d = approx.shape[1]
    current = approx.copy()

    # Reverse details: reconstruct from coarsest to finest
    for det in reversed(details):
        n = current.shape[0]
        reconstructed = np.zeros((n * 2, d), dtype=np.float32)
        reconstructed[0::2] = (current + det) / np.sqrt(2)
        reconstructed[1::2] = (current - det) / np.sqrt(2)
        current = reconstructed

    return current[:target_len].astype(np.float32)
