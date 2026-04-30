"""Quantization codecs for PLM per-residue embeddings.

Provides training-free quantization methods that compress (L, D) float32
embeddings to lower bit-width representations, reducing storage while
preserving downstream task quality.

Methods:
- Int8: per-channel 8-bit uniform quantization (4x compression)
- Int4: per-channel 4-bit quantization, packed 2 values/byte (8x compression)
- Int2: per-channel 2-bit quantization, packed 4 values/byte (16x compression)
- Binary: per-channel 1-bit sign quantization (32x compression)
- PQ: Product Quantization — learned sub-vector codebooks
- RVQ: Residual Vector Quantization — hierarchical codebooks
"""

import math
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Int8 quantization
# ---------------------------------------------------------------------------

def quantize_int8(matrix: np.ndarray) -> dict:
    """Per-channel 8-bit uniform quantization.

    Maps each channel's float32 range to [0, 255] using affine transform.
    Zero-range channels (constant) use range=1.0 to avoid division by zero.

    Args:
        matrix: (L, D) float32 per-residue embeddings.

    Returns:
        dict with:
            data: (L, D) uint8 quantized values
            scales: (D,) float32 per-channel scale factors
            zero_points: (D,) int32 per-channel zero points
            original_shape: tuple (L, D)
            dtype: "int8"
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape

    ch_min = matrix.min(axis=0)  # (D,)
    ch_max = matrix.max(axis=0)  # (D,)
    ch_range = ch_max - ch_min   # (D,)

    # Handle zero-range channels
    ch_range = np.where(ch_range == 0.0, 1.0, ch_range)

    scales = ch_range / 255.0  # (D,)
    zero_points = np.round(-ch_min / scales).astype(np.int32)  # (D,)

    # Quantize: q = clamp(round(x / scale + zero_point), 0, 255)
    q = np.round(matrix / scales[np.newaxis, :] + zero_points[np.newaxis, :])
    data = np.clip(q, 0, 255).astype(np.uint8)

    return {
        "data": data,
        "scales": scales,
        "zero_points": zero_points,
        "original_shape": (L, D),
        "dtype": "int8",
    }


def dequantize_int8(compressed: dict) -> np.ndarray:
    """Reconstruct float32 embeddings from int8 quantized dict.

    Args:
        compressed: dict returned by quantize_int8.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    data = compressed["data"].astype(np.float32)
    scales = compressed["scales"]
    zero_points = compressed["zero_points"].astype(np.float32)

    return ((data - zero_points[np.newaxis, :]) * scales[np.newaxis, :]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Int4 quantization
# ---------------------------------------------------------------------------

def quantize_int4(matrix: np.ndarray) -> dict:
    """Per-channel 4-bit uniform quantization, packed 2 values per byte.

    Maps each channel's float32 range to [0, 15] (16 levels).
    Two channels packed per byte: even column index in high nibble (bits 7-4),
    odd column index in low nibble (bits 3-0).
    If D is odd, the matrix is padded to even D before packing.

    Args:
        matrix: (L, D) float32 per-residue embeddings.

    Returns:
        dict with:
            data: (L, ceil(D/2)) uint8 packed values
            scales: (D,) float32 per-channel scale factors (pre-padding)
            zero_points: (D,) int32 per-channel zero points (pre-padding)
            original_shape: tuple (L, D)  — original (unpadded) shape
            dtype: "int4"
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape

    ch_min = matrix.min(axis=0)  # (D,)
    ch_max = matrix.max(axis=0)  # (D,)
    ch_range = ch_max - ch_min

    ch_range = np.where(ch_range == 0.0, 1.0, ch_range)

    scales = ch_range / 15.0  # (D,) — 4 bits → 16 levels [0..15]
    zero_points = np.round(-ch_min / scales).astype(np.int32)

    q = np.round(matrix / scales[np.newaxis, :] + zero_points[np.newaxis, :])
    q = np.clip(q, 0, 15).astype(np.uint8)  # (L, D)

    # Pad D to even for packing
    D_pad = D + (D % 2)
    if D % 2 != 0:
        q = np.pad(q, ((0, 0), (0, 1)))  # pad last column with zeros

    # Pack: even columns in high nibble, odd columns in low nibble
    # Reshape to (L, D_pad//2, 2): axis-2 index 0=even col, 1=odd col
    q_reshaped = q.reshape(L, D_pad // 2, 2)
    packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]  # (L, D_pad//2)

    return {
        "data": packed.astype(np.uint8),
        "scales": scales,
        "zero_points": zero_points,
        "original_shape": (L, D),
        "dtype": "int4",
    }


def dequantize_int4(compressed: dict) -> np.ndarray:
    """Reconstruct float32 embeddings from int4 packed dict.

    Args:
        compressed: dict returned by quantize_int4.

    Returns:
        (L, D) float32 reconstructed embeddings (original unpadded shape).
    """
    data = compressed["data"]  # (L, D_pad//2) uint8
    scales = compressed["scales"]
    zero_points = compressed["zero_points"].astype(np.float32)
    L, D = compressed["original_shape"]

    # Unpack nibbles
    high = (data >> 4) & 0x0F  # even columns
    low = data & 0x0F           # odd columns

    # Interleave: shape (L, D_pad//2, 2) → (L, D_pad)
    D_pad = data.shape[1] * 2
    q = np.stack([high, low], axis=2).reshape(L, D_pad).astype(np.float32)

    # Trim padding
    q = q[:, :D]

    return ((q - zero_points[np.newaxis, :]) * scales[np.newaxis, :]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Int2 quantization
# ---------------------------------------------------------------------------

def quantize_int2(matrix: np.ndarray) -> dict:
    """Per-channel 2-bit uniform quantization, packed 4 values per byte.

    Maps each channel's float32 range to [0, 3] (4 levels).
    Four columns packed per byte: col 0 in bits 7-6, col 1 in bits 5-4,
    col 2 in bits 3-2, col 3 in bits 1-0.
    If D is not divisible by 4, the matrix is padded before packing.

    Args:
        matrix: (L, D) float32 per-residue embeddings.

    Returns:
        dict with:
            data: (L, ceil(D/4)) uint8 packed values
            scales: (D,) float32 per-channel scale factors
            zero_points: (D,) int32 per-channel zero points
            original_shape: tuple (L, D)
            dtype: "int2"
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape

    ch_min = matrix.min(axis=0)
    ch_max = matrix.max(axis=0)
    ch_range = ch_max - ch_min
    ch_range = np.where(ch_range == 0.0, 1.0, ch_range)

    scales = ch_range / 3.0  # 2 bits → 4 levels [0..3]
    zero_points = np.round(-ch_min / scales).astype(np.int32)

    q = np.round(matrix / scales[np.newaxis, :] + zero_points[np.newaxis, :])
    q = np.clip(q, 0, 3).astype(np.uint8)

    # Pad D to multiple of 4 for packing
    D_pad = D + (-D % 4)
    if D_pad > D:
        q = np.pad(q, ((0, 0), (0, D_pad - D)))

    # Pack 4 values per byte: positions 0,1,2,3 in bits 7-6,5-4,3-2,1-0
    q_reshaped = q.reshape(L, D_pad // 4, 4)
    packed = (
        (q_reshaped[:, :, 0] << 6)
        | (q_reshaped[:, :, 1] << 4)
        | (q_reshaped[:, :, 2] << 2)
        | q_reshaped[:, :, 3]
    )

    return {
        "data": packed.astype(np.uint8),
        "scales": scales,
        "zero_points": zero_points,
        "original_shape": (L, D),
        "dtype": "int2",
    }


def dequantize_int2(compressed: dict) -> np.ndarray:
    """Reconstruct float32 embeddings from int2 packed dict.

    Args:
        compressed: dict returned by quantize_int2.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    data = compressed["data"]  # (L, D_pad//4) uint8
    scales = compressed["scales"]
    zero_points = compressed["zero_points"].astype(np.float32)
    L, D = compressed["original_shape"]

    # Unpack 4 values per byte
    v0 = (data >> 6) & 0x03
    v1 = (data >> 4) & 0x03
    v2 = (data >> 2) & 0x03
    v3 = data & 0x03

    D_pad = data.shape[1] * 4
    q = np.stack([v0, v1, v2, v3], axis=2).reshape(L, D_pad).astype(np.float32)
    q = q[:, :D]

    return ((q - zero_points[np.newaxis, :]) * scales[np.newaxis, :]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Binary quantization
# ---------------------------------------------------------------------------

def quantize_binary(matrix: np.ndarray) -> dict:
    """Per-channel 1-bit sign quantization, packed 8 bits per byte.

    Computes sign(x - channel_mean) for each channel. The scale per channel
    is the mean of absolute deviations from the mean (mean absolute value of
    the centered distribution), enabling approximate reconstruction.

    Args:
        matrix: (L, D) float32 per-residue embeddings.

    Returns:
        dict with:
            bits: (L, ceil(D/8)) uint8 packed bits
            means: (D,) float32 per-channel means
            scales: (D,) float32 per-channel scales (mean absolute deviation)
            original_shape: tuple (L, D)
            dtype: "binary"
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape

    means = matrix.mean(axis=0)   # (D,)
    centered = matrix - means[np.newaxis, :]  # (L, D)
    scales = np.mean(np.abs(centered), axis=0)  # (D,)

    # Binary: 1 if positive, 0 if non-positive
    binary = (centered > 0).astype(np.uint8)  # (L, D)

    # Pad D to multiple of 8 for bit-packing
    D_pad = int(math.ceil(D / 8)) * 8
    if D_pad > D:
        binary = np.pad(binary, ((0, 0), (0, D_pad - D)))

    # Pack 8 bits per byte: bit 7 = column 0, bit 0 = column 7 within each group
    binary_reshaped = binary.reshape(L, D_pad // 8, 8)  # (L, num_bytes, 8)
    # Build packed bytes: shift each bit to its position
    shifts = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)
    packed = np.bitwise_or.reduce(
        (binary_reshaped << shifts[np.newaxis, np.newaxis, :]).astype(np.uint8),
        axis=2,
    )  # (L, D_pad//8)

    return {
        "bits": packed,
        "means": means,
        "scales": scales,
        "original_shape": (L, D),
        "dtype": "binary",
    }


def dequantize_binary(compressed: dict) -> np.ndarray:
    """Reconstruct float32 embeddings from binary quantized dict.

    Args:
        compressed: dict returned by quantize_binary.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    packed = compressed["bits"]  # (L, num_bytes) uint8
    means = compressed["means"]
    scales = compressed["scales"]
    L, D = compressed["original_shape"]

    num_bytes = packed.shape[1]
    D_pad = num_bytes * 8

    # Unpack bits: bit 7 → col 0, bit 0 → col 7
    shifts = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)
    # (L, num_bytes, 8)
    bits = ((packed[:, :, np.newaxis] >> shifts[np.newaxis, np.newaxis, :]) & 1)
    bits = bits.reshape(L, D_pad)[:, :D]  # (L, D)

    # Convert 0/1 to -1/+1
    signs = (bits.astype(np.float32) * 2.0 - 1.0)

    # Reconstruct: sign * scale + mean
    return (signs * scales[np.newaxis, :] + means[np.newaxis, :]).astype(np.float32)


# ─── Binary + per-residue magnitude (PolarQuant variant, Exp 51) ──────────────

def quantize_binary_magnitude(matrix: np.ndarray) -> dict:
    """Strict superset of binary: signs + per-channel scales/means + per-residue magnitude.

    Hybrid PolarQuant — extends the binary codec with one fp16 per-residue
    magnitude, stored alongside the existing per-channel scales and means.
    The decoder uses the binary path's full reconstruction (sign × scale +
    mean) for direction, then rescales each residue to match the recorded
    target magnitude. By construction the information content is a strict
    superset of pure binary; any deviation in retention vs binary tests
    whether the per-residue magnitude scalar actually carries useful signal.

    Storage per protein: ``ceil(D/8) * L`` bits + 2D fp32 (means/scales)
    + 2L bytes (fp16 magnitudes). For L=156, D=896 the magnitude overhead
    is ~1.3 % vs binary's footprint.

    Args:
        matrix: (L, D) float32 per-residue embeddings.

    Returns:
        dict with:
            bits: (L, ceil(D/8)) uint8 packed sign bits
            means: (D,) float32 per-channel means
            scales: (D,) float32 per-channel mean abs deviations
            magnitudes: (L,) float16 per-residue L2 norms (target reconstruction)
            original_shape: tuple (L, D)
            dtype: "binary_magnitude"
    """
    base = quantize_binary(matrix)  # bits, means, scales
    base["magnitudes"] = np.linalg.norm(np.asarray(matrix, dtype=np.float32),
                                        axis=1).astype(np.float16)
    base["dtype"] = "binary_magnitude"
    return base


def dequantize_binary_magnitude(compressed: dict) -> np.ndarray:
    """Reconstruct float32 embeddings from binary+magnitude dict.

    Two-stage decode:
      1. Run the binary path's reconstruction: ``y_base = signs * scales + means``.
         This gives the direction the codec already encoded.
      2. Rescale per residue so ‖ŷ_l‖₂ matches the recorded ``magnitudes[l]``.

    The rescaling is a per-residue multiplicative factor; for a linear probe
    it gets absorbed into weights for classification (scale-invariant under
    softmax) but materially changes regression predictions where target
    magnitude correlates with the per-residue label (e.g. CheZOD disorder).

    Args:
        compressed: dict returned by quantize_binary_magnitude.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    # Reuse the binary decoder for the direction
    base = dequantize_binary({
        "bits": compressed["bits"],
        "means": compressed["means"],
        "scales": compressed["scales"],
        "original_shape": compressed["original_shape"],
        "dtype": "binary",
    })  # (L, D)

    target_mag = compressed["magnitudes"].astype(np.float32)  # (L,)
    base_mag = np.linalg.norm(base, axis=1)  # (L,)
    # Avoid div-by-zero on degenerate residues
    factor = target_mag / np.clip(base_mag, 1e-8, None)
    return (base * factor[:, np.newaxis]).astype(np.float32)


# ---------------------------------------------------------------------------
# Product Quantization (PQ)
# ---------------------------------------------------------------------------

def pq_fit(
    embeddings: dict[str, np.ndarray],
    M: int = 32,
    n_centroids: int = 256,
    max_residues: int = 500_000,
    seed: int = 42,
) -> dict:
    """Fit a Product Quantization model on a corpus of per-residue embeddings.

    Splits D-dimensional embeddings into M sub-vectors of D/M dims each,
    then fits independent MiniBatchKMeans codebooks on each sub-space.

    Args:
        embeddings: Dict mapping protein_id → (L, D) float32 arrays.
        M: Number of sub-vector groups. Must divide D evenly.
        n_centroids: Number of centroids per sub-space (<=256 to fit in uint8).
        max_residues: Maximum total residues to use for fitting (subsampled).
        seed: Random seed for KMeans and subsampling.

    Returns:
        dict with:
            codebook: (M, n_centroids, sub_dim) float32
            M: int
            n_centroids: int
            sub_dim: int
            D: int
    """
    # Collect all residues
    all_residues = np.concatenate(
        [v.astype(np.float32) for v in embeddings.values()], axis=0
    )  # (N_total, D)

    N_total, D = all_residues.shape
    assert D % M == 0, f"D={D} must be divisible by M={M}"
    sub_dim = D // M

    # Subsample if needed
    if N_total > max_residues:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N_total, size=max_residues, replace=False)
        all_residues = all_residues[idx]

    # Fit one KMeans per sub-space
    codebook = np.zeros((M, n_centroids, sub_dim), dtype=np.float32)
    for m in range(M):
        sub = all_residues[:, m * sub_dim : (m + 1) * sub_dim]
        km = MiniBatchKMeans(
            n_clusters=n_centroids,
            random_state=seed,
            n_init=3,
            batch_size=min(10_000, len(sub)),
        )
        km.fit(sub)
        codebook[m] = km.cluster_centers_.astype(np.float32)

    return {
        "codebook": codebook,
        "M": M,
        "n_centroids": n_centroids,
        "sub_dim": sub_dim,
        "D": D,
    }


def pq_encode(matrix: np.ndarray, pq_model: dict) -> np.ndarray:
    """Encode per-residue embeddings using a fitted PQ model.

    Args:
        matrix: (L, D) float32 per-residue embeddings.
        pq_model: dict returned by pq_fit.

    Returns:
        (L, M) uint8 centroid indices.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape
    M = pq_model["M"]
    sub_dim = pq_model["sub_dim"]
    codebook = pq_model["codebook"]  # (M, n_centroids, sub_dim)

    codes = np.zeros((L, M), dtype=np.uint8)
    for m in range(M):
        sub = matrix[:, m * sub_dim : (m + 1) * sub_dim]  # (L, sub_dim)
        centers = codebook[m]  # (n_centroids, sub_dim)
        # Compute squared distances: ||sub - center||^2
        # = ||sub||^2 - 2*sub@centers.T + ||centers||^2
        dists = (
            np.sum(sub ** 2, axis=1, keepdims=True)
            - 2.0 * sub @ centers.T
            + np.sum(centers ** 2, axis=1)
        )  # (L, n_centroids)
        codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

    return codes


def pq_decode(codes: np.ndarray, pq_model: dict) -> np.ndarray:
    """Reconstruct per-residue embeddings from PQ codes.

    Args:
        codes: (L, M) uint8 centroid indices.
        pq_model: dict returned by pq_fit.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    L, M = codes.shape
    sub_dim = pq_model["sub_dim"]
    D = pq_model["D"]
    codebook = pq_model["codebook"]  # (M, n_centroids, sub_dim)

    reconstructed = np.zeros((L, D), dtype=np.float32)
    for m in range(M):
        idx = codes[:, m].astype(np.int32)
        reconstructed[:, m * sub_dim : (m + 1) * sub_dim] = codebook[m][idx]

    return reconstructed


# ---------------------------------------------------------------------------
# Residual Vector Quantization (RVQ)
# ---------------------------------------------------------------------------

def rvq_fit(
    embeddings: dict[str, np.ndarray],
    n_levels: int = 3,
    n_centroids: int = 256,
    max_residues: int = 500_000,
    seed: int = 42,
) -> dict:
    """Fit a Residual Vector Quantization model on a corpus of embeddings.

    Level 0 quantizes the raw embeddings. Each subsequent level quantizes
    the residuals (errors) from the previous level's reconstruction.

    Args:
        embeddings: Dict mapping protein_id → (L, D) float32 arrays.
        n_levels: Number of quantization levels.
        n_centroids: Number of centroids per level (<=256 to fit in uint8).
        max_residues: Maximum total residues for fitting (subsampled).
        seed: Random seed for KMeans and subsampling.

    Returns:
        dict with:
            codebooks: list of n_levels arrays, each (n_centroids, D) float32
            n_levels: int
            n_centroids: int
            D: int
    """
    all_residues = np.concatenate(
        [v.astype(np.float32) for v in embeddings.values()], axis=0
    )  # (N_total, D)

    N_total, D = all_residues.shape

    if N_total > max_residues:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N_total, size=max_residues, replace=False)
        all_residues = all_residues[idx]

    codebooks = []
    residuals = all_residues.copy()

    for level in range(n_levels):
        km = MiniBatchKMeans(
            n_clusters=n_centroids,
            random_state=seed + level,
            n_init=3,
            batch_size=min(10_000, len(residuals)),
        )
        km.fit(residuals)
        centers = km.cluster_centers_.astype(np.float32)  # (n_centroids, D)
        codebooks.append(centers)

        # Compute residuals for next level
        assignments = km.labels_  # (N,)
        reconstructed = centers[assignments]
        residuals = residuals - reconstructed

    return {
        "codebooks": codebooks,
        "n_levels": n_levels,
        "n_centroids": n_centroids,
        "D": D,
    }


def rvq_encode(matrix: np.ndarray, rvq_model: dict) -> np.ndarray:
    """Encode per-residue embeddings using a fitted RVQ model.

    Args:
        matrix: (L, D) float32 per-residue embeddings.
        rvq_model: dict returned by rvq_fit.

    Returns:
        (L, n_levels) uint8 centroid indices.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    L, D = matrix.shape
    codebooks = rvq_model["codebooks"]
    n_levels = rvq_model["n_levels"]

    codes = np.zeros((L, n_levels), dtype=np.uint8)
    residuals = matrix.copy()

    for level, centers in enumerate(codebooks):
        # Find nearest centroid
        dists = (
            np.sum(residuals ** 2, axis=1, keepdims=True)
            - 2.0 * residuals @ centers.T
            + np.sum(centers ** 2, axis=1)
        )  # (L, n_centroids)
        idx = np.argmin(dists, axis=1)  # (L,)
        codes[:, level] = idx.astype(np.uint8)

        # Subtract reconstruction to get residuals
        residuals = residuals - centers[idx]

    return codes


def rvq_decode(codes: np.ndarray, rvq_model: dict) -> np.ndarray:
    """Reconstruct per-residue embeddings from RVQ codes.

    Args:
        codes: (L, n_levels) uint8 centroid indices.
        rvq_model: dict returned by rvq_fit.

    Returns:
        (L, D) float32 reconstructed embeddings.
    """
    L, n_levels = codes.shape
    D = rvq_model["D"]
    codebooks = rvq_model["codebooks"]

    reconstructed = np.zeros((L, D), dtype=np.float32)
    for level, centers in enumerate(codebooks):
        idx = codes[:, level].astype(np.int32)
        reconstructed += centers[idx]

    return reconstructed


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compressed_size_bytes(compressed: dict) -> int:
    """Compute actual byte count of a quantized compressed dict.

    Counts only the data arrays (not metadata strings/tuples).

    Args:
        compressed: dict returned by any quantize_* function.

    Returns:
        Total bytes used by the compressed data arrays.
    """
    dtype = compressed.get("dtype", "")

    if dtype == "int8":
        return compressed["data"].nbytes

    elif dtype == "int4":
        return compressed["data"].nbytes

    elif dtype == "int2":
        return compressed["data"].nbytes

    elif dtype == "binary":
        return compressed["bits"].nbytes

    else:
        raise ValueError(f"Unknown compressed dtype: {dtype!r}")
