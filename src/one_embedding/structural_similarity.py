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


class FAISSSearchIndex:
    """FAISS-backed search index for million-scale protein similarity search.

    Uses FAISS inner product search on L2-normalized vectors (= cosine similarity).
    Supports flat (exact) and IVF (approximate) index types.

    Usage:
        index = FAISSSearchIndex()
        index.build(vectors)                     # <100K: flat, ≥100K: IVF
        results = index.search(query_vec, k=10)  # [{name, similarity, ...}]
        index.save("proteins.faiss")
        index = FAISSSearchIndex.load("proteins.faiss")
    """

    def __init__(self, index_type: str = "auto", nprobe: int = 16):
        """
        Args:
            index_type: "flat" (exact), "ivf" (approximate), or "auto"
            nprobe: number of IVF cells to search (higher = more accurate, slower)
        """
        self.index_type = index_type
        self.nprobe = nprobe
        self._index = None
        self._names: List[str] = []
        self._metadata: Dict[str, dict] = {}
        self.tm_a = 1.0
        self.tm_b = 0.0

    def build(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, dict]] = None,
    ):
        """Build the FAISS index.

        Args:
            vectors: {protein_id: (D,) embedding vector}
            metadata: optional annotations per protein
        """
        import faiss

        self._names = sorted(vectors.keys())
        V = np.array([vectors[n] for n in self._names], dtype=np.float32)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(V)

        N, D = V.shape
        self._dim = D

        # Choose index type
        if self.index_type == "auto":
            use_ivf = N >= 100_000
        else:
            use_ivf = self.index_type == "ivf"

        if use_ivf:
            nlist = min(int(np.sqrt(N)), N // 10)
            nlist = max(nlist, 1)
            quantizer = faiss.IndexFlatIP(D)
            self._index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(V)
            self._index.nprobe = self.nprobe
        else:
            self._index = faiss.IndexFlatIP(D)

        self._index.add(V)
        self._metadata = metadata or {}

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_score: Optional[float] = None,
    ) -> List[dict]:
        """Find k nearest neighbors.

        Args:
            query: (D,) embedding vector
            k: number of neighbors
            min_score: minimum cosine similarity threshold

        Returns:
            list of dicts: name, similarity, tm_score_approx, metadata
        """
        import faiss

        if self._index is None:
            raise ValueError("Index not built. Call build() first.")

        q = query.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for sim, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(sim)
            if min_score is not None and sim < min_score:
                continue
            tm = float(np.clip(self.tm_a * sim + self.tm_b, 0.0, 1.0))
            results.append({
                "name": self._names[idx],
                "similarity": sim,
                "tm_score_approx": tm,
                "metadata": self._metadata.get(self._names[idx], {}),
            })

        return results

    def search_batch(
        self,
        queries: Dict[str, np.ndarray],
        k: int = 10,
    ) -> Dict[str, List[dict]]:
        """Batch search — uses FAISS batch search for efficiency."""
        import faiss

        if self._index is None:
            raise ValueError("Index not built.")

        names = sorted(queries.keys())
        Q = np.array([queries[n] for n in names], dtype=np.float32)
        faiss.normalize_L2(Q)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(Q, k)

        results = {}
        for i, name in enumerate(names):
            hits = []
            for sim, idx in zip(scores[i], indices[i]):
                if idx < 0:
                    continue
                tm = float(np.clip(self.tm_a * float(sim) + self.tm_b, 0.0, 1.0))
                hits.append({
                    "name": self._names[idx],
                    "similarity": float(sim),
                    "tm_score_approx": tm,
                    "metadata": self._metadata.get(self._names[idx], {}),
                })
            results[name] = hits

        return results

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index else 0

    def save(self, path: str):
        """Save index + metadata."""
        import faiss
        faiss.write_index(self._index, path)
        np.savez(
            path + ".meta",
            names=np.array(self._names),
            tm_a=np.array([self.tm_a]),
            tm_b=np.array([self.tm_b]),
        )

    @classmethod
    def load(cls, path: str) -> "FAISSSearchIndex":
        """Load saved index."""
        import faiss
        obj = cls()
        obj._index = faiss.read_index(path)
        meta = np.load(path + ".meta.npz", allow_pickle=True)
        obj._names = list(meta["names"])
        obj.tm_a = float(meta["tm_a"][0])
        obj.tm_b = float(meta["tm_b"][0])
        obj._dim = obj._index.d
        return obj
