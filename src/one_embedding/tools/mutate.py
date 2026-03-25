"""Mutation effect scanning.

Note: Uses local embedding distance as an untrained heuristic proxy.
For validated predictions, use trained probes from src.evaluation.
"""
import numpy as np
from ._base import load_per_residue


def scan(oemb_path, **kwargs):
    """Estimate per-residue mutation sensitivity from embedding structure.
    Returns {pid: (L,) sensitivity scores}.
    """
    embeddings = load_per_residue(oemb_path)
    results = {}
    for pid, emb in embeddings.items():
        # Local context sensitivity: how different is each residue from its neighbors
        L = emb.shape[0]
        sensitivity = np.zeros(L)
        for i in range(L):
            window = emb[max(0, i-2):min(L, i+3)]
            center = emb[i]
            dists = np.linalg.norm(window - center, axis=1)
            sensitivity[i] = np.mean(dists)
        results[pid] = sensitivity
    return results


# Alias for uniform tool API
predict = scan
