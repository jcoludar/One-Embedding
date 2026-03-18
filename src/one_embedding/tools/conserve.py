"""Alignment-free conservation scoring."""
import numpy as np
from ._base import load_per_residue


def score(oemb_path, **kwargs):
    """Score per-residue conservation using embedding norm proxy.
    Returns {pid: (L,) conservation scores in [0,1]}.
    """
    embeddings = load_per_residue(oemb_path)
    results = {}
    for pid, emb in embeddings.items():
        norms = np.linalg.norm(emb, axis=1)
        scores = (norms - norms.min()) / (norms.max() - norms.min() + 1e-10)
        results[pid] = scores
    return results
