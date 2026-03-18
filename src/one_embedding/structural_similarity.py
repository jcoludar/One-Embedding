"""Structural similarity prediction and search from protein embeddings.

Predicts approximate TM-score from embedding cosine similarity.
Provides searchable index for fast structural neighbor lookup.

Based on TM-Vec (Nature Biotechnology 2023) and DCTdomain (Genome Research 2024).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        vectors: (N, D) protein embedding vectors

    Returns:
        (N, N) cosine similarity matrix
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    normalized = vectors / norms
    return normalized @ normalized.T


def cosine_to_tm_score(
    cosine_sim: np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
) -> np.ndarray:
    """Map cosine similarity to approximate TM-score.

    Default is identity mapping (uncalibrated). After calibration on
    proteins with known TM-scores, a and b can be fitted via linear
    regression: TM_score ≈ a * cosine_sim + b

    Args:
        cosine_sim: cosine similarity values
        a: slope (fitted from calibration data)
        b: intercept

    Returns:
        Approximate TM-scores clipped to [0, 1]
    """
    return np.clip(a * cosine_sim + b, 0.0, 1.0)


class StructuralSearchIndex:
    """Searchable index of protein embedding vectors for structural neighbors.

    Uses brute-force cosine similarity for small databases (<100K proteins)
    and approximate nearest neighbor for larger ones.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._names: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._metadata: Dict[str, dict] = {}
        # TM-score calibration parameters
        self.tm_a = 1.0
        self.tm_b = 0.0

    def build(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, dict]] = None,
    ):
        """Build the search index.

        Args:
            vectors: {protein_id: (D,) embedding vector}
            metadata: optional {protein_id: {key: value}} for annotations
        """
        self._names = sorted(vectors.keys())
        self._vectors = np.array(
            [vectors[n] for n in self._names], dtype=np.float32
        )
        if self.metric == "cosine":
            # Pre-normalize for fast cosine computation
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
            self._vectors_norm = self._vectors / norms
        self._metadata = metadata or {}

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_score: Optional[float] = None,
    ) -> List[dict]:
        """Find k nearest structural neighbors.

        Args:
            query: (D,) query embedding vector
            k: number of neighbors to return
            min_score: minimum similarity threshold

        Returns:
            list of dicts with keys: name, similarity, tm_score_approx, metadata
        """
        if self._vectors is None:
            raise ValueError("Index not built. Call build() first.")

        k = min(k, len(self._names))

        if self.metric == "cosine":
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            similarities = self._vectors_norm @ query_norm
        else:
            dists = np.linalg.norm(self._vectors - query[np.newaxis, :], axis=1)
            similarities = 1.0 / (1.0 + dists)

        top_k_idx = np.argpartition(similarities, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

        results = []
        for idx in top_k_idx:
            sim = float(similarities[idx])
            if min_score is not None and sim < min_score:
                continue
            tm_approx = float(cosine_to_tm_score(
                np.array([sim]), self.tm_a, self.tm_b
            )[0])
            results.append({
                "name": self._names[idx],
                "similarity": sim,
                "tm_score_approx": tm_approx,
                "metadata": self._metadata.get(self._names[idx], {}),
            })

        return results

    def search_batch(
        self,
        queries: Dict[str, np.ndarray],
        k: int = 10,
    ) -> Dict[str, List[dict]]:
        """Search for multiple queries."""
        return {name: self.search(vec, k) for name, vec in queries.items()}

    def calibrate_tm_score(
        self,
        pairs: List[Tuple[str, str]],
        true_tm_scores: List[float],
    ):
        """Calibrate cosine→TM-score mapping from known pairs.

        Fits a linear model: TM_score = a * cosine_sim + b

        Args:
            pairs: list of (protein_a, protein_b) name tuples
            true_tm_scores: corresponding true TM-scores
        """
        cos_sims = []
        for name_a, name_b in pairs:
            idx_a = self._names.index(name_a)
            idx_b = self._names.index(name_b)
            sim = float(self._vectors_norm[idx_a] @ self._vectors_norm[idx_b])
            cos_sims.append(sim)

        cos_sims = np.array(cos_sims)
        true_tm = np.array(true_tm_scores)

        # Simple linear regression
        X = np.column_stack([cos_sims, np.ones_like(cos_sims)])
        params = np.linalg.lstsq(X, true_tm, rcond=None)[0]
        self.tm_a = float(params[0])
        self.tm_b = float(params[1])

    def all_vs_all(self) -> np.ndarray:
        """Compute all-vs-all similarity matrix.

        Returns:
            (N, N) similarity matrix
        """
        if self._vectors is None:
            raise ValueError("Index not built.")
        return self._vectors_norm @ self._vectors_norm.T

    def save(self, path: str):
        np.savez(
            path,
            vectors=self._vectors,
            names=np.array(self._names),
            tm_a=np.array([self.tm_a]),
            tm_b=np.array([self.tm_b]),
            metric=np.array([self.metric]),
        )

    @classmethod
    def load(cls, path: str) -> "StructuralSearchIndex":
        data = np.load(path, allow_pickle=True)
        obj = cls(metric=str(data["metric"][0]))
        obj._names = list(data["names"])
        obj._vectors = data["vectors"]
        norms = np.linalg.norm(obj._vectors, axis=1, keepdims=True) + 1e-10
        obj._vectors_norm = obj._vectors / norms
        obj.tm_a = float(data["tm_a"][0])
        obj.tm_b = float(data["tm_b"][0])
        return obj
