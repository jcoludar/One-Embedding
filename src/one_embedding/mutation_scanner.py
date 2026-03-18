"""Mutational landscape scanning from per-residue embeddings.

Two approaches:
1. Zero-shot displacement: cosine distance between WT and mutant embeddings
2. Probe: VespaG-style FNN predicting per-position mutation effects

Works on decoded 512d per-residue embeddings from V2 codec.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def embedding_displacement(
    wt_embeddings: np.ndarray,
    mut_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute per-residue embedding displacement between WT and mutant.

    Large displacement at a position indicates the mutation has a significant
    effect on the local structural/functional context.

    Args:
        wt_embeddings: (L, D) wild-type per-residue embeddings
        mut_embeddings: (L, D) mutant per-residue embeddings

    Returns:
        (L,) displacement scores (cosine distance per position)
    """
    # Per-residue cosine distance
    wt_norm = np.linalg.norm(wt_embeddings, axis=1, keepdims=True) + 1e-10
    mut_norm = np.linalg.norm(mut_embeddings, axis=1, keepdims=True) + 1e-10
    cos_sim = np.sum(
        (wt_embeddings / wt_norm) * (mut_embeddings / mut_norm), axis=1
    )
    return 1.0 - cos_sim


def position_sensitivity(embeddings: np.ndarray, window: int = 5) -> np.ndarray:
    """Estimate per-position sensitivity from embedding local context.

    Positions where the embedding differs strongly from its neighbors are
    more likely to be functionally important and sensitive to mutations.

    Args:
        embeddings: (L, D) per-residue embeddings
        window: context window size (positions on each side)

    Returns:
        (L,) sensitivity scores (higher = more sensitive to mutation)
    """
    L, D = embeddings.shape
    scores = np.zeros(L, dtype=np.float32)

    for i in range(L):
        # Get neighbors
        start = max(0, i - window)
        end = min(L, i + window + 1)
        neighbors = np.concatenate([embeddings[start:i], embeddings[i+1:end]])

        if len(neighbors) == 0:
            continue

        # Mean neighbor embedding
        neighbor_mean = neighbors.mean(axis=0)

        # Cosine distance to local context
        cos_sim = np.dot(embeddings[i], neighbor_mean) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(neighbor_mean) + 1e-10
        )
        scores[i] = 1.0 - cos_sim

    return scores


class MutationEffectProbe:
    """FNN predicting per-position mutation effect scores.

    Architecture: Linear(D→hidden) + LeakyReLU + Linear(hidden→20)
    Output: 20 scores per position (one per amino acid substitution)

    Based on VespaG (Bioinformatics 2024).
    """

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, input_dim: int = 512, hidden: int = 256):
        self.input_dim = input_dim
        self.hidden = hidden
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def _init_weights(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / self.input_dim)
        self.w1 = rng.randn(self.hidden, self.input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        scale2 = np.sqrt(2.0 / self.hidden)
        self.w2 = rng.randn(20, self.hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(20, dtype=np.float32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict mutation effect scores.

        Args:
            embeddings: (L, D) per-residue embeddings

        Returns:
            (L, 20) mutation effect scores (one per AA substitution)
        """
        if self.w1 is None:
            raise ValueError("Probe not fitted. Call fit() or load().")

        # Layer 1: Linear + LeakyReLU
        h = embeddings @ self.w1.T + self.b1
        h = np.where(h > 0, h, 0.01 * h)  # LeakyReLU

        # Layer 2: Linear (no activation — regression output)
        scores = h @ self.w2.T + self.b2

        return scores

    def predict_landscape(
        self, embeddings: np.ndarray, sequence: str
    ) -> Dict[str, np.ndarray]:
        """Predict full mutational landscape.

        Args:
            embeddings: (L, D) per-residue embeddings
            sequence: amino acid sequence (length L)

        Returns:
            dict with:
                scores: (L, 20) raw scores
                landscape: (L, 20) scores with WT positions set to 0
                most_damaging: list of (position, mutation, score)
        """
        scores = self.predict(embeddings)

        # Zero out WT positions
        landscape = scores.copy()
        for i, aa in enumerate(sequence):
            if aa in self.AA_ORDER:
                wt_idx = self.AA_ORDER.index(aa)
                landscape[i, wt_idx] = 0.0

        # Find most damaging mutations
        most_damaging = []
        for i in range(len(sequence)):
            for j in range(20):
                if self.AA_ORDER[j] != sequence[i]:
                    most_damaging.append((i, f"{sequence[i]}{i+1}{self.AA_ORDER[j]}", float(landscape[i, j])))
        most_damaging.sort(key=lambda x: x[2])  # Most negative = most damaging

        return {
            "scores": scores,
            "landscape": landscape,
            "most_damaging": most_damaging[:20],  # Top 20
            "position_sensitivity": np.abs(landscape).mean(axis=1),
        }

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 input_dim=self.input_dim, hidden=self.hidden)

    @classmethod
    def load(cls, path: str) -> "MutationEffectProbe":
        data = np.load(path)
        obj = cls(int(data["input_dim"]), int(data["hidden"]))
        obj.w1, obj.b1, obj.w2, obj.b2 = data["w1"], data["b1"], data["w2"], data["b2"]
        return obj
