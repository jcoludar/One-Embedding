"""Topological and information-theoretic transforms for protein embeddings.

Provides:
  - Sliced Wasserstein Distance (OT-based set comparison)
  - Persistent homology image vectorisation (requires ripser + persim)
  - SimHash locality-sensitive hashing (bit packing, approximate decode)
  - Amino-acid-residual encoding (centroid subtraction per amino acid type)

All numpy/scipy; GPU not required. ripser/persim are optional.
"""

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

# Standard one-letter amino acid codes in alphabetical order
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}


# ── Sliced Wasserstein Distance ──────────────────────────────────────────────


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    """Sliced Wasserstein Distance between two point clouds.

    Projects both clouds onto random unit directions on the sphere,
    sorts the 1-D projections, interpolates to the same length, and
    averages the mean L1 distance across all directions.

    Complexity: O(n_projections * (L_X + L_Y) * D).

    Args:
        X: (L_X, D) point cloud (float32 or float64).
        Y: (L_Y, D) point cloud.
        n_projections: Number of random projections.
        seed: RNG seed for reproducibility.

    Returns:
        Scalar SWD (float).
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    _, D = X.shape

    rng = np.random.RandomState(seed)
    # Random Gaussian directions, then L2-normalise → uniform on S^{D-1}
    directions = rng.randn(n_projections, D)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms  # (n_projections, D)

    proj_X = X @ directions.T  # (L_X, n_projections)
    proj_Y = Y @ directions.T  # (L_Y, n_projections)

    total = 0.0
    L_X = X.shape[0]
    L_Y = Y.shape[0]

    for j in range(n_projections):
        sx = np.sort(proj_X[:, j])
        sy = np.sort(proj_Y[:, j])

        if L_X == L_Y:
            total += np.mean(np.abs(sx - sy))
        else:
            # Interpolate to common grid
            t_x = np.linspace(0.0, 1.0, L_X)
            t_y = np.linspace(0.0, 1.0, L_Y)
            t_common = np.linspace(0.0, 1.0, max(L_X, L_Y))
            ix = interp1d(t_x, sx, kind="linear")(t_common)
            iy = interp1d(t_y, sy, kind="linear")(t_common)
            total += np.mean(np.abs(ix - iy))

    return float(total / n_projections)


def sliced_wasserstein_matrix(
    embeddings: dict[str, np.ndarray],
    ids: list[str],
    n_projections: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Pairwise Sliced Wasserstein Distance matrix.

    Args:
        embeddings: Mapping from protein id → (L_i, D) array.
        ids: Ordered list of protein ids.
        n_projections: Number of random projections per pair.
        seed: RNG seed.

    Returns:
        (N, N) symmetric distance matrix (float64).
    """
    N = len(ids)
    mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = sliced_wasserstein_distance(
                embeddings[ids[i]], embeddings[ids[j]], n_projections, seed
            )
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ── Persistent Homology ──────────────────────────────────────────────────────


def persistence_image(
    matrix: np.ndarray,
    max_dim: int = 1,
    n_bins: int = 20,
    spread: float = 1.0,
) -> Optional[np.ndarray]:
    """Persistence image vectorisation of per-residue embedding cloud.

    Requires ripser and persim. Returns None when unavailable or on error.

    Args:
        matrix: (L, D) per-residue embeddings.
        max_dim: Maximum homological dimension (0 and 1 computed).
        n_bins: Number of bins per axis of the persistence image.
        spread: Gaussian spread parameter for PersistenceImager.

    Returns:
        (n_bins, n_bins, max_dim+1) float32 array, or None on failure.
    """
    try:
        from ripser import ripser as _ripser
        from persim import PersistenceImager
    except ImportError:
        return None

    try:
        L, D = matrix.shape
        # Subsample to control compute cost
        if L > 200:
            rng = np.random.RandomState(0)
            idx = rng.choice(L, size=200, replace=False)
            data = matrix[idx].astype(np.float64)
        else:
            data = matrix.astype(np.float64)

        diagrams = _ripser(data, maxdim=max_dim)["dgms"]

        pimgr = PersistenceImager(pixel_size=spread / n_bins, spread=spread)

        # Collect all finite points to fit the imager
        finite_dgms = []
        for dgm in diagrams:
            finite = dgm[np.isfinite(dgm[:, 1])]
            finite_dgms.append(finite)

        pimgr.fit(finite_dgms)

        images = []
        for dgm in finite_dgms:
            img = pimgr.transform([dgm])[0]
            # Resize to (n_bins, n_bins) by interpolation
            from scipy.ndimage import zoom as _zoom
            if img.shape[0] > 0 and img.shape[1] > 0:
                zy = n_bins / img.shape[0]
                zx = n_bins / img.shape[1]
                img = _zoom(img, (zy, zx), order=1)
            else:
                img = np.zeros((n_bins, n_bins), dtype=np.float64)
            # Ensure correct shape
            img = img[:n_bins, :n_bins]
            if img.shape != (n_bins, n_bins):
                img = np.pad(
                    img,
                    (
                        (0, n_bins - img.shape[0]),
                        (0, n_bins - img.shape[1]),
                    ),
                )
            images.append(img)

        result = np.stack(images, axis=-1).astype(np.float32)  # (n_bins, n_bins, max_dim+1)
        return result

    except Exception:
        return None


# ── SimHash ──────────────────────────────────────────────────────────────────


def simhash_encode(
    matrix: np.ndarray,
    n_bits: int = 1024,
    seed: int = 42,
) -> dict:
    """Locality-sensitive hashing via random hyperplanes (SimHash).

    For each residue computes the sign of its projection onto n_bits
    random unit Gaussian hyperplanes, then packs the bits into bytes.

    Args:
        matrix: (L, D) per-residue embeddings.
        n_bits: Number of hash bits (must be divisible by 8).
        seed: RNG seed for hyperplane generation.

    Returns:
        Dict with keys:
            bits          – (L, n_bits // 8) uint8 packed bit array
            n_bits        – n_bits used
            original_shape – (L, D)
            seed          – seed used
    """
    if n_bits % 8 != 0:
        raise ValueError(f"n_bits must be divisible by 8, got {n_bits}")

    matrix = matrix.astype(np.float32)
    L, D = matrix.shape

    rng = np.random.RandomState(seed)
    W = rng.randn(n_bits, D).astype(np.float32)
    # L2-normalise rows so each W[i] is a unit hyperplane normal
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W /= norms

    projections = matrix @ W.T  # (L, n_bits)
    signs = (projections >= 0).astype(np.uint8)  # (L, n_bits), 0 or 1

    # Pack 8 bits into 1 byte per group
    n_bytes = n_bits // 8
    packed = np.packbits(signs, axis=1)  # (L, n_bytes)

    return {
        "bits": packed,
        "n_bits": n_bits,
        "original_shape": (L, D),
        "seed": seed,
    }


def simhash_decode_approx(compressed: dict) -> np.ndarray:
    """Approximate decode of SimHash bits via pseudoinverse reconstruction.

    Unpacks bits to {-1, +1} signs and projects back using the Moore-Penrose
    pseudoinverse of the hyperplane matrix. Quality is poor (theoretical
    lower bound), but the shape is correct.

    Args:
        compressed: Dict returned by simhash_encode.

    Returns:
        (L, D) approximate reconstruction, float32.
    """
    bits = compressed["bits"]           # (L, n_bytes) uint8
    n_bits = compressed["n_bits"]
    L, D = compressed["original_shape"]
    seed = compressed["seed"]

    # Unpack bits → 0/1, shape (L, n_bits)
    signs_binary = np.unpackbits(bits, axis=1)[:, :n_bits].astype(np.float32)
    signs = signs_binary * 2.0 - 1.0  # {-1, +1}

    # Regenerate the same hyperplane matrix
    rng = np.random.RandomState(seed)
    W = rng.randn(n_bits, D).astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W /= norms  # (n_bits, D)

    # Solve signs ≈ matrix @ W.T for matrix via least-squares.
    # Rewrite as: W @ matrix.T = signs.T, solve for matrix.T.
    # Square random Gaussian matrices (n_bits == D) are ill-conditioned
    # (cond > 5000) due to Marchenko-Pastur eigenvalue spread. Use rcond
    # to truncate small singular values and stabilise reconstruction.
    # W (n_bits, D), signs.T (n_bits, L) → result (D, L)
    rcond = 0.1 if n_bits == D else None
    result, _, _, _ = np.linalg.lstsq(W, signs.T, rcond=rcond)
    result = result.T.astype(np.float32)  # (L, D)
    return result


# ── Amino-Acid Residual Encoding ─────────────────────────────────────────────


def compute_aa_centroids(
    embeddings: dict[str, np.ndarray],
    sequences: dict[str, str],
    max_proteins: int = 5000,
) -> np.ndarray:
    """Compute per-amino-acid centroid embeddings from a protein corpus.

    Accumulates mean per-residue embedding for each of the 20 standard
    amino acid types. Non-standard AAs are ignored.

    Args:
        embeddings: Mapping protein_id → (L_i, D) float32 array.
        sequences: Mapping protein_id → amino acid string (length L_i).
        max_proteins: Maximum number of proteins to use (uniform subsample).

    Returns:
        (20, D) float32 centroid matrix in AA_ORDER order.
    """
    ids = list(embeddings.keys())
    if len(ids) > max_proteins:
        rng = np.random.RandomState(42)
        ids = rng.choice(ids, size=max_proteins, replace=False).tolist()

    # Peek at D
    first = embeddings[ids[0]]
    D = first.shape[1]

    sums = np.zeros((20, D), dtype=np.float64)
    counts = np.zeros(20, dtype=np.int64)

    for pid in ids:
        mat = embeddings[pid].astype(np.float64)   # (L, D)
        seq = sequences[pid]
        L = min(len(seq), mat.shape[0])
        for pos in range(L):
            aa = seq[pos]
            idx = _AA_INDEX.get(aa, -1)
            if idx < 0:
                continue
            sums[idx] += mat[pos]
            counts[idx] += 1

    # Avoid division by zero for unseen AAs
    centroids = np.where(
        counts[:, np.newaxis] > 0,
        sums / np.maximum(counts[:, np.newaxis], 1),
        0.0,
    ).astype(np.float32)

    return centroids


def aa_residual_encode(
    matrix: np.ndarray,
    sequence: str,
    centroids: np.ndarray,
) -> np.ndarray:
    """Subtract per-AA centroid from each residue embedding.

    Args:
        matrix: (L, D) per-residue embeddings.
        sequence: Amino acid string of length >= L.
        centroids: (20, D) centroid array from compute_aa_centroids.

    Returns:
        (L, D) residual embeddings (smaller magnitude for typical proteins).
    """
    L = matrix.shape[0]
    residuals = matrix.copy().astype(np.float32)
    for pos in range(L):
        aa = sequence[pos] if pos < len(sequence) else "X"
        idx = _AA_INDEX.get(aa, -1)
        if idx >= 0:
            residuals[pos] -= centroids[idx]
    return residuals


def aa_residual_decode(
    residual: np.ndarray,
    sequence: str,
    centroids: np.ndarray,
) -> np.ndarray:
    """Add per-AA centroids back: perfect lossless roundtrip.

    Args:
        residual: (L, D) residual from aa_residual_encode.
        sequence: Amino acid string of length >= L.
        centroids: (20, D) centroid array.

    Returns:
        (L, D) reconstructed embeddings (identical to original input).
    """
    L = residual.shape[0]
    result = residual.copy().astype(np.float32)
    for pos in range(L):
        aa = sequence[pos] if pos < len(sequence) else "X"
        idx = _AA_INDEX.get(aa, -1)
        if idx >= 0:
            result[pos] += centroids[idx]
    return result
