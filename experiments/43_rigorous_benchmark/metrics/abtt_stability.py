"""ABTT cross-corpus stability analysis.

Tests whether the top-k principal components removed by ABTT preprocessing
are properties of the PLM architecture (stable across different corpora) or
data-dependent (potential information leakage).

Uses principal angles between subspaces (Bjorck & Golub, 1973) to measure
the similarity of PC subspaces fitted on independent corpora.

Novel methodological contribution: no published protein embedding paper has
tested whether ABTT PCs are corpus-independent.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so `src.*` imports resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.one_embedding.core.preprocessing import fit_abtt


# ---------------------------------------------------------------------------
# Principal angles (Bjorck & Golub 1973)
# ---------------------------------------------------------------------------


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the principal angles between two subspaces.

    Uses the Bjorck & Golub (1973) algorithm: QR-orthonormalise each
    subspace's basis, then compute the SVD of Q_A.T @ Q_B.  The singular
    values are the cosines of the principal angles.

    Args:
        A: (k, D) matrix whose rows span the first subspace.
        B: (k, D) matrix whose rows span the second subspace.

    Returns:
        (k,) array of principal angles in radians, sorted ascending.
        Values lie in [0, pi/2].
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # QR on transposed matrices to get (D, k) orthonormal bases
    Q_A, _ = np.linalg.qr(A.T, mode="reduced")  # (D, k)
    Q_B, _ = np.linalg.qr(B.T, mode="reduced")  # (D, k)

    # SVD of the inner product matrix
    _, sigma, _ = np.linalg.svd(Q_A.T @ Q_B, full_matrices=False)

    # Clip to [0, 1] for numerical safety and convert to angles
    sigma = np.clip(sigma, 0.0, 1.0)
    angles = np.arccos(sigma)

    # Sort ascending (smallest angle first)
    angles = np.sort(angles)

    return angles


# ---------------------------------------------------------------------------
# Subspace similarity
# ---------------------------------------------------------------------------


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the subspace similarity between two subspaces.

    Defined as the mean of cos^2(theta_i) over all principal angles theta_i.
    Returns 1.0 for identical subspaces, 0.0 for orthogonal subspaces.

    Args:
        A: (k, D) matrix whose rows span the first subspace.
        B: (k, D) matrix whose rows span the second subspace.

    Returns:
        Scalar similarity in [0, 1].
    """
    angles = principal_angles(A, B)
    return float(np.mean(np.cos(angles) ** 2))


# ---------------------------------------------------------------------------
# Cross-corpus stability report
# ---------------------------------------------------------------------------


def cross_corpus_stability_report(
    corpora: dict[str, np.ndarray],
    k: int = 3,
    seed: int = 42,
) -> dict:
    """Analyse ABTT PC stability across multiple corpora.

    Fits ABTT on each corpus independently and measures the pairwise
    subspace similarity of the resulting top-k PC subspaces.

    Args:
        corpora: Mapping of corpus name to (N, D) ndarray of stacked
            residue embeddings (float32 or float64).
        k: Number of top PCs to compare (default 3).
        seed: Random seed passed to fit_abtt for reproducible subsampling.

    Returns:
        dict with keys:
            params: {k, seed, n_corpora, corpus_names}
            pairwise_similarity: {(name_a, name_b): float}
            pairwise_angles_deg: {(name_a, name_b): list[float]}
            min_similarity: float
            mean_similarity: float
            conclusion: "stable" if min_similarity > 0.95, else "unstable"
    """
    # Fit ABTT on each corpus
    fitted = {}
    for name, data in corpora.items():
        params = fit_abtt(np.asarray(data, dtype=np.float32), k=k, seed=seed)
        # Store top_pcs as float64 for numerical precision in angle computation
        fitted[name] = np.asarray(params["top_pcs"], dtype=np.float64)

    # Pairwise comparisons
    names = sorted(corpora.keys())
    pairwise_similarity: dict[tuple[str, str], float] = {}
    pairwise_angles_deg: dict[tuple[str, str], list[float]] = {}

    for name_a, name_b in combinations(names, 2):
        pcs_a = fitted[name_a]
        pcs_b = fitted[name_b]
        sim = subspace_similarity(pcs_a, pcs_b)
        angles = principal_angles(pcs_a, pcs_b)
        pairwise_similarity[(name_a, name_b)] = sim
        pairwise_angles_deg[(name_a, name_b)] = np.degrees(angles).tolist()

    similarities = list(pairwise_similarity.values())
    min_sim = float(min(similarities))
    mean_sim = float(np.mean(similarities))

    return {
        "params": {
            "k": k,
            "seed": seed,
            "n_corpora": len(corpora),
            "corpus_names": names,
        },
        "pairwise_similarity": pairwise_similarity,
        "pairwise_angles_deg": pairwise_angles_deg,
        "min_similarity": min_sim,
        "mean_similarity": mean_sim,
        "conclusion": "stable" if min_sim > 0.95 else "unstable",
    }
