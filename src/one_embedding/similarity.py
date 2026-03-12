"""Comparison utilities for OneEmbedding objects."""

import numpy as np

from src.one_embedding.embedding import OneEmbedding


def protein_cosine_similarity(a: OneEmbedding, b: OneEmbedding) -> float:
    """Cosine similarity between fixed-size summary vectors."""
    a_norm = np.linalg.norm(a.summary).clip(1e-8)
    b_norm = np.linalg.norm(b.summary).clip(1e-8)
    return float(np.dot(a.summary, b.summary) / (a_norm * b_norm))


def late_interaction_score(a: OneEmbedding, b: OneEmbedding) -> float:
    """ColBERT-style late interaction on per-residue matrices.

    score = sum_i max_j cos(a_residue_i, b_residue_j)

    Handles variable-length proteins naturally.
    """
    a_res = a.residues
    b_res = b.residues

    a_norms = np.linalg.norm(a_res, axis=1, keepdims=True).clip(1e-8)
    b_norms = np.linalg.norm(b_res, axis=1, keepdims=True).clip(1e-8)
    a_normed = a_res / a_norms
    b_normed = b_res / b_norms

    sim_matrix = a_normed @ b_normed.T  # (L_a, L_b)
    return float(sim_matrix.max(axis=1).sum())


def pairwise_summary_matrix(
    embeddings: dict[str, OneEmbedding],
) -> tuple[list[str], np.ndarray]:
    """Compute all-pairs cosine similarity on summary vectors.

    Returns:
        (ordered_ids, similarity_matrix) where sim_matrix[i,j] = cos(summary_i, summary_j)
    """
    ids = list(embeddings.keys())
    n = len(ids)

    # Stack and normalize
    matrix = np.array([embeddings[pid].summary for pid in ids])  # (n, dim)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
    normed = matrix / norms

    sim = normed @ normed.T  # (n, n)
    return ids, sim
