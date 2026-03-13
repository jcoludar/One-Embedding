"""Within-channel compression techniques for PLM per-residue embeddings.

All functions operate on (L, D) float32 matrices. No dimension mixing —
each channel (column) is compressed independently, preserving channel identity.

Functions:
- wavelet_threshold_compress / decompress: DWT per channel + soft thresholding
- cur_decompose / reconstruct: CUR decomposition selecting actual columns
- compute_channel_importance: per-channel variance across a corpus
- channel_prune: keep top-k highest-variance channels
- zstd_compress / decompress: Zstandard byte-level compression
- measure_compressed_size: raw vs zstd size statistics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt
import scipy.linalg
import zstandard


# ---------------------------------------------------------------------------
# Wavelet threshold compression
# ---------------------------------------------------------------------------


def wavelet_threshold_compress(
    matrix: np.ndarray,
    wavelet: str = "db4",
    threshold_pct: float = 75.0,
) -> dict[str, Any]:
    """Compress (L, D) matrix via per-channel DWT + soft thresholding.

    Each column (channel) is decomposed independently using a discrete
    wavelet transform. Coefficients below the given percentile are zeroed
    out via soft thresholding (sparsification step).

    Args:
        matrix: (L, D) float32 per-residue embeddings.
        wavelet: PyWavelets wavelet name (e.g. "db4", "db8", "sym4").
        threshold_pct: Percentile of absolute coefficient magnitudes used
            as the soft-threshold value. 0 = no thresholding (lossless
            in wavelet domain). 75 = 75th percentile threshold.

    Returns:
        dict with keys:
            "coeffs": list of (D,) arrays per wavelet level (and detail)
            "wavelet": wavelet name
            "original_shape": (L, D) tuple
            "n_levels": number of decomposition levels
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape

    # Determine number of levels automatically for axis=0 (length dimension)
    n_levels = pywt.dwt_max_level(L, wavelet)

    # Decompose each channel independently
    # pywt.wavedec returns [cA_n, cD_n, ..., cD_1]  (n+1 sub-bands)
    per_channel_coeffs = []
    for d in range(D):
        channel = matrix[:, d]
        coeffs = pywt.wavedec(channel, wavelet, level=n_levels)
        per_channel_coeffs.append(coeffs)

    # Compute global percentile threshold over all non-approximation coefficients
    # (detail sub-bands only), then apply soft threshold
    if threshold_pct > 0:
        all_detail_values = []
        for coeffs in per_channel_coeffs:
            for arr in coeffs[1:]:  # skip approximation (index 0)
                all_detail_values.append(np.abs(arr))
        if all_detail_values:
            all_abs = np.concatenate(all_detail_values)
            threshold = float(np.percentile(all_abs, threshold_pct))
        else:
            threshold = 0.0

        thresholded = []
        for coeffs in per_channel_coeffs:
            new_coeffs = [coeffs[0]]  # keep approximation intact
            for arr in coeffs[1:]:
                new_coeffs.append(pywt.threshold(arr, threshold, mode="soft"))
            thresholded.append(new_coeffs)
        per_channel_coeffs = thresholded

    # Store as list of level arrays, each level array has shape (D, level_len)
    # Transpose from per-channel to per-level for compact storage
    n_subbands = len(per_channel_coeffs[0])
    level_arrays = []
    for sb in range(n_subbands):
        # Collect this sub-band from all channels and stack → (D, len_sb)
        stacked = np.stack([per_channel_coeffs[d][sb] for d in range(D)], axis=0)
        level_arrays.append(stacked.astype(np.float32))

    return {
        "coeffs": level_arrays,
        "wavelet": wavelet,
        "original_shape": (L, D),
        "n_levels": n_levels,
    }


def wavelet_threshold_decompress(compressed: dict[str, Any]) -> np.ndarray:
    """Reconstruct (L, D) matrix from wavelet-compressed representation.

    Args:
        compressed: dict returned by wavelet_threshold_compress.

    Returns:
        (L, D) float32 reconstructed matrix.
    """
    level_arrays = compressed["coeffs"]
    wavelet = compressed["wavelet"]
    L, D = compressed["original_shape"]

    reconstructed = np.empty((L, D), dtype=np.float32)

    for d in range(D):
        # Reassemble per-channel coefficient list
        coeffs = [level_arrays[sb][d] for sb in range(len(level_arrays))]
        rec = pywt.waverec(coeffs, wavelet)
        # Trim or pad to original length L
        if len(rec) > L:
            rec = rec[:L]
        elif len(rec) < L:
            rec = np.pad(rec, (0, L - len(rec)))
        reconstructed[:, d] = rec.astype(np.float32)

    return reconstructed


# ---------------------------------------------------------------------------
# CUR decomposition
# ---------------------------------------------------------------------------


def cur_decompose(matrix: np.ndarray, k: int = 64) -> dict[str, Any]:
    """Column-selection decomposition (CUR) via pivoted QR column selection.

    Selects k actual columns of the matrix using LAPACK's column-pivoted QR
    factorization (scipy.linalg.qr with pivoting=True). The selected columns
    ARE original channels — no mixing. The least-squares interpolation matrix
    then expresses all D columns approximately in terms of the k selected ones.

    Args:
        matrix: (L, D) float32 per-residue embeddings.
        k: Number of columns (channels) to select.

    Returns:
        dict with keys:
            "C": (L, k) selected columns (actual original channels)
            "col_indices": (k,) indices of selected columns in [0, D)
            "interp_matrix": (k, D) matrix such that C @ interp_matrix ≈ matrix
            "original_shape": (L, D) tuple
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape
    # Pivoted QR can select at most min(L, D) columns (matrix rank upper bound)
    k = min(k, L, D)

    # Pivoted QR selects the k most linearly independent columns of matrix.
    # Apply QR to matrix directly: scipy.linalg.qr(A, pivoting=True) returns
    # permutation piv such that A[:, piv] = Q @ R. piv[:k] are the k
    # column indices selected by the pivoting strategy.
    _, _, piv = scipy.linalg.qr(
        matrix.astype(np.float64), pivoting=True, mode="economic"
    )
    col_indices = piv[:k].copy()  # shape (k,) — column indices in [0, D)

    C = matrix[:, col_indices].astype(np.float32)  # (L, k)

    # Build least-squares interpolation matrix: C @ interp_matrix ≈ matrix
    # Solve: min ||C @ X - matrix||_F  →  X = pinv(C) @ matrix
    C_double = C.astype(np.float64)
    interp_matrix = (
        np.linalg.lstsq(C_double, matrix.astype(np.float64), rcond=None)[0]
    ).astype(np.float32)  # (k, D)

    return {
        "C": C,
        "col_indices": col_indices,
        "interp_matrix": interp_matrix,
        "original_shape": (L, D),
    }


def cur_reconstruct(compressed: dict[str, Any]) -> np.ndarray:
    """Reconstruct (L, D) matrix from CUR decomposition.

    Args:
        compressed: dict returned by cur_decompose.

    Returns:
        (L, D) float32 reconstructed matrix.
    """
    C = compressed["C"]  # (L, k)
    interp_matrix = compressed["interp_matrix"]  # (k, D)
    return (C @ interp_matrix).astype(np.float32)


# ---------------------------------------------------------------------------
# Channel importance and pruning
# ---------------------------------------------------------------------------


def compute_channel_importance(
    embeddings: dict[str, np.ndarray],
    max_proteins: int = 1000,
) -> np.ndarray:
    """Compute per-channel variance across a corpus of protein embeddings.

    Each protein contributes its per-residue embedding matrix (L_i, D).
    We compute the variance of each channel (column) across ALL residues
    from ALL proteins, then return mean variance per channel.

    Args:
        embeddings: dict mapping protein ID → (L_i, D) float32 matrix.
        max_proteins: Maximum number of proteins to use (for speed).

    Returns:
        (D,) float32 array of per-channel mean variance (all values >= 0).
    """
    keys = list(embeddings.keys())[:max_proteins]

    # Collect all residue vectors per channel
    all_matrices = [embeddings[k] for k in keys]
    # Stack all residues together: (total_residues, D)
    stacked = np.vstack(all_matrices).astype(np.float32)

    # Per-channel variance across all residues
    variances = stacked.var(axis=0)  # (D,)
    return variances.astype(np.float32)


def channel_prune(
    matrix: np.ndarray,
    importance: np.ndarray,
    k: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-k highest-variance channels.

    Args:
        matrix: (L, D) float32 per-residue embeddings.
        importance: (D,) per-channel importance scores (e.g. variances).
        k: Number of channels to retain.

    Returns:
        Tuple of:
            - (L, k) pruned matrix with top-k channels
            - (k,) int array of selected channel indices (sorted descending
              by importance)
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    importance = np.asarray(importance)
    D = matrix.shape[1]
    k = min(k, D)

    # argsort descending — top-k highest importance
    sorted_indices = np.argsort(importance)[::-1]
    top_k_indices = sorted_indices[:k]

    pruned = matrix[:, top_k_indices]
    return pruned.astype(np.float32), top_k_indices.astype(np.intp)


# ---------------------------------------------------------------------------
# Zstandard byte-level compression
# ---------------------------------------------------------------------------


def zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compress bytes with Zstandard.

    Args:
        data: Raw bytes to compress.
        level: Compression level (1–22; higher = smaller but slower).

    Returns:
        Compressed bytes.
    """
    cctx = zstandard.ZstdCompressor(level=level)
    return cctx.compress(data)


def zstd_decompress(data: bytes) -> bytes:
    """Decompress Zstandard-compressed bytes.

    Args:
        data: Zstandard-compressed bytes.

    Returns:
        Decompressed bytes.
    """
    dctx = zstandard.ZstdDecompressor()
    return dctx.decompress(data)


def measure_compressed_size(
    matrix: np.ndarray,
    zstd_level: int = 3,
) -> dict[str, Any]:
    """Measure raw vs Zstandard-compressed byte sizes for a matrix.

    Args:
        matrix: (L, D) float32 matrix.
        zstd_level: Zstandard compression level.

    Returns:
        dict with keys:
            "raw_bytes": size of raw float32 bytes (L * D * 4)
            "zstd_bytes": size after Zstandard compression
            "zstd_ratio": raw_bytes / zstd_bytes (> 1 means compression)
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    raw = matrix.tobytes()
    compressed = zstd_compress(raw, level=zstd_level)
    raw_bytes = len(raw)
    zstd_bytes = len(compressed)
    zstd_ratio = raw_bytes / zstd_bytes if zstd_bytes > 0 else float("inf")
    return {
        "raw_bytes": raw_bytes,
        "zstd_bytes": zstd_bytes,
        "zstd_ratio": zstd_ratio,
    }
