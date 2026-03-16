"""Transposed matrix view transforms for protein embeddings.

These transforms operate on the (D, L) transposed view of per-residue embeddings,
treating each channel as a signal across residues. This perspective enables
signal-processing-style operations (resampling, statistics) across the sequence
dimension for each embedding channel independently.

All functions take (L, D) float32 input and return float32 output.
"""

import numpy as np
from scipy.signal import resample
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis


def channel_resample(
    X: np.ndarray,
    l_out: int = 64,
) -> np.ndarray:
    """Resample each channel from L to fixed l_out using scipy.signal.resample.

    Transposes the input to (D, L), then resamples each channel (row) from L
    points to l_out points using polyphase filtering. This creates a fixed-size
    representation regardless of protein length.

    Args:
        X: Per-residue embeddings, shape (L, D).
        l_out: Target number of residue samples per channel.

    Returns:
        Resampled matrix of shape (D, l_out).
    """
    L, D = X.shape
    # Transpose to (D, L) so each row is one channel's signal across residues
    Xt = X.T  # (D, L)
    # Resample along axis=1 (the L dimension)
    resampled = resample(Xt, l_out, axis=1)  # (D, l_out)
    return resampled.astype(np.float32)


def per_protein_svd(
    X: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """SVD of X.T, returning top-k left singular vectors weighted by singular values.

    Computes the SVD of the transposed matrix (D, L). The top-k left singular
    vectors (each of dimension D) are scaled by their corresponding singular
    values and flattened into a single vector. This captures the principal
    modes of variation across residues in channel space.

    If L < k, zero-pads the output to maintain consistent dimensionality.

    Args:
        X: Per-residue embeddings, shape (L, D).
        k: Number of top singular components to keep.

    Returns:
        Flattened vector of shape (D * k,).
    """
    L, D = X.shape
    Xt = X.T  # (D, L)

    # Handle case where L < k: we can get at most min(D, L) singular values
    n_components = min(D, L)

    if n_components == 0:
        return np.zeros(D * k, dtype=np.float32)

    U, S, _ = np.linalg.svd(Xt, full_matrices=False)
    # U: (D, min(D,L)), S: (min(D,L),)

    # Take top-k components, weighted by singular values
    actual_k = min(k, n_components)
    weighted = U[:, :actual_k] * S[:actual_k]  # (D, actual_k)

    # Flatten and zero-pad if we got fewer than k components
    result = np.zeros(D * k, dtype=np.float32)
    result[:D * actual_k] = weighted.flatten()

    return result


def channel_statistics(
    X: np.ndarray,
    stats: list[str] | None = None,
) -> np.ndarray:
    """Per-channel statistics across residues.

    Computes summary statistics for each of the D channels across L residues,
    producing a fixed-size descriptor regardless of protein length.

    Args:
        X: Per-residue embeddings, shape (L, D).
        stats: List of statistics to compute. Default:
            ["mean", "std", "min", "max", "skew", "kurtosis"].
            Supported: "mean", "std", "min", "max", "skew", "kurtosis".

    Returns:
        Feature vector of shape (D * n_stats,).
    """
    if stats is None:
        stats = ["mean", "std", "min", "max", "skew", "kurtosis"]

    L, D = X.shape
    parts = []

    for stat in stats:
        if stat == "mean":
            parts.append(X.mean(axis=0))
        elif stat == "std":
            parts.append(X.std(axis=0))
        elif stat == "min":
            parts.append(X.min(axis=0))
        elif stat == "max":
            parts.append(X.max(axis=0))
        elif stat == "skew":
            parts.append(scipy_skew(X, axis=0))
        elif stat == "kurtosis":
            parts.append(scipy_kurtosis(X, axis=0))
        else:
            raise ValueError(f"Unknown statistic: {stat!r}. "
                             f"Supported: mean, std, min, max, skew, kurtosis")

    return np.concatenate(parts).astype(np.float32)


def zero_pad_flatten(
    X: np.ndarray,
    l_max: int = 512,
) -> np.ndarray:
    """Pad or truncate to l_max residues, transpose to (D, l_max), and flatten.

    If L > l_max, truncates to the first l_max residues.
    If L < l_max, zero-pads along the residue axis.
    The result is transposed to (D, l_max) and flattened to (D * l_max,).

    Args:
        X: Per-residue embeddings, shape (L, D).
        l_max: Maximum sequence length (pad/truncate target).

    Returns:
        Flattened vector of shape (D * l_max,).
    """
    L, D = X.shape

    if L >= l_max:
        # Truncate to first l_max residues
        padded = X[:l_max]
    else:
        # Zero-pad along residue axis
        padded = np.zeros((l_max, D), dtype=X.dtype)
        padded[:L] = X

    # Transpose to (D, l_max) and flatten
    return padded.T.flatten().astype(np.float32)
