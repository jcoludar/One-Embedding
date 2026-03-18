"""Pairwise embedding alignment."""
import numpy as np
from ._base import load_per_residue


def align_pair(oemb_path, protein_a, protein_b, mode="global", **kwargs):
    """Align two proteins from an .oemb file.
    Returns {align_a, align_b, score, n_aligned, n_gaps_a, n_gaps_b}.
    """
    from src.one_embedding.aligner import align_embeddings
    embs = load_per_residue(oemb_path)
    if protein_a not in embs or protein_b not in embs:
        raise ValueError(f"Protein not found. Available: {list(embs.keys())[:5]}...")
    return align_embeddings(embs[protein_a], embs[protein_b], mode=mode)
