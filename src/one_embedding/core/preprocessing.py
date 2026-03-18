"""ABTT (All-But-The-Top) preprocessing for PLM per-residue embeddings.

Removes the dominant principal components to improve isotropy of the
embedding space. Training-free — only requires a corpus of residue
embeddings to fit the mean and top PCs.

Reference:
    Mu & Viswanath (2018) "All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations."
"""

import numpy as np


def fit_abtt(
    residues: np.ndarray,
    k: int = 3,
    seed: int = 42,
) -> dict:
    """Fit ABTT parameters from stacked residue embeddings.

    Computes the corpus mean and the top-k principal component directions.
    Subsamples to 50K residues if the input is larger.

    Args:
        residues: (N, D) stacked per-residue embeddings from a corpus.
        k: Number of dominant PCs to remove (default 3).
        seed: Random seed for reproducible subsampling.

    Returns:
        dict with keys:
            mean: (D,) corpus mean vector, float32.
            top_pcs: (k, D) top-k principal directions (unit vectors), float32.
    """
    residues = np.asarray(residues, dtype=np.float32)
    N, D = residues.shape

    # Subsample to at most 50K rows for efficiency
    max_rows = 50_000
    if N > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=max_rows, replace=False)
        sample = residues[idx]
    else:
        sample = residues

    # Compute mean and center
    mean = sample.mean(axis=0)  # (D,)
    centered = sample - mean    # (N_sample, D)

    # SVD on the centered sample: top right singular vectors = top PCs
    # We only need the top k, so we use full_matrices=False and slice.
    # np.linalg.svd with full_matrices=False gives Vh of shape (min(N,D), D);
    # each row is a right singular vector (== principal component direction).
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    top_pcs = Vh[:k].astype(np.float32)  # (k, D) — already unit vectors

    return {
        "mean": mean.astype(np.float32),
        "top_pcs": top_pcs,
    }


def apply_abtt(X: np.ndarray, params: dict) -> np.ndarray:
    """Remove top-k PC projections from per-residue embeddings.

    Centers X by the corpus mean, then projects out each dominant PC.
    The result is a more isotropic embedding space.

    Args:
        X: (L, D) per-residue embeddings for a single protein.
        params: dict returned by fit_abtt (keys: "mean", "top_pcs").

    Returns:
        (L, D) preprocessed embeddings, float32.
    """
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(params["mean"], dtype=np.float32)
    top_pcs = np.asarray(params["top_pcs"], dtype=np.float32)

    # Center
    centered = X - mean  # (L, D)

    # Project out each top PC: X' = X - (X @ pc.T) * pc  for each pc
    # Vectorised: projection = centered @ top_pcs.T @ top_pcs  (L, D)
    projection = (centered @ top_pcs.T) @ top_pcs  # (L, k) @ (k, D) = (L, D)
    return (centered - projection).astype(np.float32)
