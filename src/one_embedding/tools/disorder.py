"""Disorder prediction from compressed embeddings.

Note: Uses embedding norm as an untrained heuristic proxy.
For validated predictions, use trained probes from src.evaluation.
"""
import numpy as np
from ._base import load_per_residue


def predict(oemb_path, **kwargs):
    """Predict per-residue disorder scores.
    Returns {protein_id: (L,) scores} where higher = more disordered.
    """
    embeddings = load_per_residue(oemb_path)
    results = {}
    for pid, emb in embeddings.items():
        # Embedding norm as disorder proxy (disordered residues have lower norms)
        norms = np.linalg.norm(emb, axis=1)
        scores = 1.0 - (norms - norms.min()) / (norms.max() - norms.min() + 1e-10)
        results[pid] = scores
    return results
