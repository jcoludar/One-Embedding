"""Embedding-based protein family classifier.

k-NN classification using protein-level embedding vectors.
Works with protein_vec (2048d) from V2 codec or mean-pooled embeddings.

Based on ProtTucker/EAT (Rostlab, NAR Genomics 2022) and knnProtT5.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class EmbeddingClassifier:
    """k-NN protein family classifier on embedding vectors.

    Stores a reference database of labeled protein vectors. For a query,
    finds the k nearest neighbors and assigns the majority label.
    Uses cosine or euclidean distance.
    """

    def __init__(self, metric: str = "cosine", k: int = 1):
        self.metric = metric
        self.k = k
        self._names: List[str] = []
        self._labels: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # (N, D)
        self._norms: Optional[np.ndarray] = None  # (N,) for cosine

    def fit(self, vectors: Dict[str, np.ndarray], labels: Dict[str, str]):
        """Build reference database from labeled vectors.

        Args:
            vectors: {protein_id: (D,) embedding vector}
            labels: {protein_id: family_label}
        """
        common = sorted(set(vectors.keys()) & set(labels.keys()))
        self._names = common
        self._labels = [labels[n] for n in common]
        self._vectors = np.array([vectors[n] for n in common], dtype=np.float32)
        if self.metric == "cosine":
            self._norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10

    def predict(self, query: np.ndarray, k: Optional[int] = None) -> dict:
        """Classify a single query vector.

        Args:
            query: (D,) embedding vector
            k: override default k

        Returns:
            dict with keys: label, confidence, neighbors, distances
        """
        if self._vectors is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        k = k or self.k
        k = min(k, len(self._names))

        if self.metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            query_norm = np.linalg.norm(query) + 1e-10
            similarities = (self._vectors @ query) / (self._norms.squeeze() * query_norm)
            distances = 1.0 - similarities
        else:  # euclidean
            diff = self._vectors - query[np.newaxis, :]
            distances = np.linalg.norm(diff, axis=1)

        # Get k nearest
        top_k_idx = np.argpartition(distances, k)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        top_k_labels = [self._labels[i] for i in top_k_idx]
        top_k_dists = distances[top_k_idx]
        top_k_names = [self._names[i] for i in top_k_idx]

        # Majority vote
        label_counts = Counter(top_k_labels)
        best_label = label_counts.most_common(1)[0][0]
        confidence = label_counts[best_label] / k

        return {
            "label": best_label,
            "confidence": confidence,
            "neighbors": list(zip(top_k_names, top_k_labels, top_k_dists.tolist())),
            "distance": float(top_k_dists[0]),
        }

    def predict_batch(
        self, queries: Dict[str, np.ndarray], k: Optional[int] = None
    ) -> Dict[str, dict]:
        """Classify multiple query vectors.

        Args:
            queries: {protein_id: (D,) vector}

        Returns:
            {protein_id: prediction_dict}
        """
        return {name: self.predict(vec, k=k) for name, vec in queries.items()}

    def evaluate(
        self,
        test_vectors: Dict[str, np.ndarray],
        test_labels: Dict[str, str],
        k: Optional[int] = None,
    ) -> dict:
        """Evaluate classification accuracy on a test set.

        Uses leave-one-out if test proteins are in the reference database,
        or direct lookup if they're separate.

        Returns:
            dict with accuracy, per_family_accuracy, confusion summary
        """
        predictions = self.predict_batch(test_vectors, k=k)

        correct = 0
        total = 0
        family_correct: Dict[str, int] = {}
        family_total: Dict[str, int] = {}

        for name, pred in predictions.items():
            if name not in test_labels:
                continue
            true_label = test_labels[name]
            total += 1
            family_total[true_label] = family_total.get(true_label, 0) + 1

            if pred["label"] == true_label:
                correct += 1
                family_correct[true_label] = family_correct.get(true_label, 0) + 1

        accuracy = correct / total if total > 0 else 0.0

        per_family = {}
        for fam in sorted(family_total.keys()):
            per_family[fam] = {
                "correct": family_correct.get(fam, 0),
                "total": family_total[fam],
                "accuracy": family_correct.get(fam, 0) / family_total[fam],
            }

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_family": per_family,
        }

    def save(self, path: str):
        """Save classifier to npz file."""
        np.savez(
            path,
            vectors=self._vectors,
            names=np.array(self._names),
            labels=np.array(self._labels),
            metric=np.array([self.metric]),
            k=np.array([self.k]),
        )

    @classmethod
    def load(cls, path: str) -> "EmbeddingClassifier":
        """Load classifier from npz file."""
        data = np.load(path, allow_pickle=True)
        obj = cls(metric=str(data["metric"][0]), k=int(data["k"][0]))
        obj._vectors = data["vectors"]
        obj._names = list(data["names"])
        obj._labels = list(data["labels"])
        if obj.metric == "cosine":
            obj._norms = np.linalg.norm(obj._vectors, axis=1, keepdims=True) + 1e-10
        return obj
