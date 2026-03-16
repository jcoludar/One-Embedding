"""Pre-compression transforms for PLM per-residue embeddings.

Corpus-level statistical preprocessing to apply before codec compression.
All functions are training-free and operate on float32 numpy arrays.

Reference:
    Mu & Viswanath (2018) "All-but-the-Top: Simple and Effective Postprocessing
    for Word Representations" — removes dominant principal components to
    improve isotropy of embedding spaces.
"""

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA


def compute_corpus_stats(
    embeddings: np.ndarray,
    n_sample: int = 50_000,
    n_pcs: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Compute corpus-level statistics for preprocessing transforms.

    Fits PCA on a random subsample of residue embeddings to extract
    the mean, standard deviation, top principal components, and full
    rotation matrix.

    Args:
        embeddings: (N, D) matrix of residue embeddings (float32).
        n_sample: Maximum number of rows to subsample for PCA fitting.
            If N <= n_sample, all rows are used.
        n_pcs: Number of top principal components to extract for
            all-but-the-top removal.
        seed: Random seed for reproducible subsampling.

    Returns:
        Dictionary with keys:
            mean_vec: (D,) corpus mean vector.
            std_vec: (D,) per-channel standard deviation.
            top_pcs: (n_pcs, D) top principal component directions.
            explained_variance: (n_pcs,) variance explained by each PC.
            rotation_matrix: (D, D) full PCA rotation matrix (components).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    N, D = embeddings.shape

    # Subsample if corpus is large
    if N > n_sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=n_sample, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    # Fit full PCA to get rotation matrix (all D components)
    pca = PCA(n_components=min(D, sample.shape[0]), random_state=seed)
    pca.fit(sample)

    mean_vec = pca.mean_.astype(np.float32)
    std_vec = sample.std(axis=0).astype(np.float32)

    # Top PCs for all-but-the-top
    top_pcs = pca.components_[:n_pcs].astype(np.float32)
    explained_variance = pca.explained_variance_[:n_pcs].astype(np.float32)

    # Full rotation matrix
    rotation_matrix = pca.components_.astype(np.float32)

    return {
        "mean_vec": mean_vec,
        "std_vec": std_vec,
        "top_pcs": top_pcs,
        "explained_variance": explained_variance,
        "rotation_matrix": rotation_matrix,
    }


def center_embeddings(X: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """Center embeddings by subtracting corpus mean.

    Args:
        X: (L, D) per-residue embeddings.
        mean_vec: (D,) corpus mean vector from compute_corpus_stats.

    Returns:
        (L, D) centered embeddings (float32).
    """
    X = np.asarray(X, dtype=np.float32)
    mean_vec = np.asarray(mean_vec, dtype=np.float32)
    return (X - mean_vec).astype(np.float32)


def zscore_embeddings(
    X: np.ndarray, mean_vec: np.ndarray, std_vec: np.ndarray
) -> np.ndarray:
    """Standardize embeddings to zero mean and unit variance per channel.

    Channels with zero standard deviation are left as centered (not divided)
    to avoid NaN values.

    Args:
        X: (L, D) per-residue embeddings.
        mean_vec: (D,) corpus mean vector.
        std_vec: (D,) corpus standard deviation vector.

    Returns:
        (L, D) z-scored embeddings (float32).
    """
    X = np.asarray(X, dtype=np.float32)
    mean_vec = np.asarray(mean_vec, dtype=np.float32)
    std_vec = np.asarray(std_vec, dtype=np.float32)

    centered = X - mean_vec
    # Replace zero stds with 1.0 so division is a no-op (channel stays centered)
    safe_std = np.where(std_vec > 0, std_vec, 1.0)
    return (centered / safe_std).astype(np.float32)


def all_but_the_top(X: np.ndarray, top_pcs: np.ndarray) -> np.ndarray:
    """Remove top-k principal components from embeddings.

    Implements the postprocessing from Mu & Viswanath (2018): center,
    then project out the dominant PCs to improve isotropy.

    Note: this function assumes X has already been centered (mean-subtracted).
    If X is not centered, the caller should apply center_embeddings first.

    Args:
        X: (L, D) per-residue embeddings (should be centered).
        top_pcs: (K, D) top-k principal component directions (unit vectors).

    Returns:
        (L, D) embeddings with top-k PCs projected out (float32).
    """
    X = np.asarray(X, dtype=np.float32)
    top_pcs = np.asarray(top_pcs, dtype=np.float32)

    # Project out each top PC: X' = X - sum_k (X @ pc_k) * pc_k
    # Equivalent to: X' = X - X @ top_pcs.T @ top_pcs
    projection = X @ top_pcs.T @ top_pcs  # (L, D)
    return (X - projection).astype(np.float32)


def pca_rotate(X: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate embeddings to principal component axes.

    Applies the PCA rotation (without centering — caller should center
    first if desired). The rotation is orthogonal, so norms are preserved.

    Args:
        X: (L, D) per-residue embeddings.
        rotation_matrix: (D, D) PCA components matrix from
            compute_corpus_stats (rows are principal directions).

    Returns:
        (L, D) rotated embeddings (float32).
    """
    X = np.asarray(X, dtype=np.float32)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)

    # rotation_matrix rows are PC directions; X @ R.T rotates to PC space
    return (X @ rotation_matrix.T).astype(np.float32)
