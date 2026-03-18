"""Alignment-free conservation scoring from per-residue embeddings.

Two approaches:
1. Probe: linear model from single-sequence embedding → conservation (Kibby-style)
2. Family variance: stack homolog embeddings, compute per-position variance

Both work on decoded 512d per-residue embeddings from V2 codec.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ConservationProbe:
    """Linear probe predicting per-residue conservation from embeddings.

    conservation_score = intercept + dot(embedding, coefficients)

    Based on Kibby (Briefings in Bioinformatics 2023).
    """

    def __init__(self, coef: Optional[np.ndarray] = None, intercept: float = 0.0):
        self.coef = coef
        self.intercept = intercept

    def fit(self, embeddings: np.ndarray, conservation_scores: np.ndarray):
        """Fit linear regression from embeddings to conservation scores.

        Args:
            embeddings: (N, D) per-residue embeddings (concatenated across proteins)
            conservation_scores: (N,) conservation values in [0, 1]
        """
        # OLS: coef = (X^T X)^{-1} X^T y
        # Add regularization for numerical stability
        D = embeddings.shape[1]
        XtX = embeddings.T @ embeddings + 1e-6 * np.eye(D)
        Xty = embeddings.T @ conservation_scores
        self.coef = np.linalg.solve(XtX, Xty)
        self.intercept = float(np.mean(conservation_scores) - np.mean(embeddings @ self.coef))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict conservation scores for per-residue embeddings.

        Args:
            embeddings: (L, D) per-residue embeddings for one protein

        Returns:
            (L,) conservation scores clipped to [0, 1]
        """
        if self.coef is None:
            raise ValueError("Probe not fitted. Call fit() first.")
        scores = self.intercept + embeddings @ self.coef
        return np.clip(scores, 0.0, 1.0)

    def save(self, path: str):
        """Save probe to npz file."""
        np.savez(path, coef=self.coef, intercept=np.array([self.intercept]))

    @classmethod
    def load(cls, path: str) -> "ConservationProbe":
        """Load probe from npz file."""
        data = np.load(path)
        return cls(coef=data["coef"], intercept=float(data["intercept"][0]))


def embedding_variance_conservation(
    family_embeddings: Dict[str, np.ndarray],
    method: str = "mean_pool",
) -> Dict[str, np.ndarray]:
    """Compute per-position conservation from embedding variance across a family.

    Alignment-free: uses mean-pooled protein vectors to estimate position
    importance via embedding magnitude variance.

    For aligned embeddings (all same length L), computes column-wise variance
    directly — positions with low variance across family members are conserved.

    Args:
        family_embeddings: {protein_id: (L_i, D) embeddings}
        method: "mean_pool" for protein-level, "aligned" for position-level
            (requires all embeddings to have same L)

    Returns:
        {protein_id: (L_i,) conservation scores} where low values = conserved
        (inverted: 1 - normalized_variance, so high = conserved)
    """
    if method == "aligned":
        # All must have same length
        lengths = [emb.shape[0] for emb in family_embeddings.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"For 'aligned' method, all proteins must have same length. "
                f"Got lengths: {set(lengths)}"
            )

        # Stack: (n_proteins, L, D)
        names = sorted(family_embeddings.keys())
        stacked = np.stack([family_embeddings[n] for n in names])

        # Per-position variance across family members: (L, D)
        pos_var = np.var(stacked, axis=0)

        # Sum variance across embedding dimensions: (L,)
        total_var = pos_var.sum(axis=1)

        # Normalize and invert: high score = conserved
        if total_var.max() > 0:
            normalized = total_var / total_var.max()
        else:
            normalized = np.zeros_like(total_var)
        conservation = 1.0 - normalized

        return {name: conservation for name in names}

    elif method == "mean_pool":
        # For each protein, compute per-residue "distinctiveness"
        # relative to the family centroid
        names = sorted(family_embeddings.keys())
        centroids = []
        for name in names:
            centroids.append(family_embeddings[name].mean(axis=0))
        family_centroid = np.mean(centroids, axis=0)  # (D,)

        result = {}
        for name in names:
            emb = family_embeddings[name]  # (L, D)
            # Distance of each residue from family centroid
            dists = np.linalg.norm(emb - family_centroid, axis=1)  # (L,)
            # Normalize and invert
            if dists.max() > 0:
                normalized = dists / dists.max()
            else:
                normalized = np.zeros_like(dists)
            result[name] = 1.0 - normalized
        return result

    else:
        raise ValueError(f"Unknown method: {method}. Use 'aligned' or 'mean_pool'.")


def embedding_norm_conservation(embeddings: np.ndarray) -> np.ndarray:
    """Single-sequence conservation proxy from embedding norms.

    Residues with higher embedding norms tend to be more conserved
    (the PLM has stronger "opinions" about constrained positions).

    Args:
        embeddings: (L, D) per-residue embeddings

    Returns:
        (L,) conservation proxy scores in [0, 1]
    """
    norms = np.linalg.norm(embeddings, axis=1)  # (L,)
    if norms.max() - norms.min() > 1e-10:
        normalized = (norms - norms.min()) / (norms.max() - norms.min())
    else:
        normalized = np.ones_like(norms) * 0.5
    return normalized
