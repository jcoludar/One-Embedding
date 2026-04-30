"""Random Neighbor Score (RNS) for embedding quality evaluation.

Implements the RNS metric from Prabakaran & Bromberg (Nat Methods 2026):
for each protein, RNS_k = fraction of k nearest neighbors in the
embedding space that are randomly-shuffled (non-biological) sequences.

RNS in [0, 1]. Lower = higher confidence (embedding is in a biologically
structured region). Higher = lower confidence (embedding is
indistinguishable from random noise).
"""

from __future__ import annotations

import random

import numpy as np


def generate_junkyard_sequences(
    sequences: dict[str, str],
    n_shuffles: int = 5,
    seed: int = 42,
) -> dict[str, str]:
    """Generate residue-shuffled copies of each sequence.

    For each protein, create n_shuffles random permutations of its amino
    acid sequence. The shuffled sequences have the same composition but
    no biological order — they serve as the 'junkyard' reference for RNS.

    Args:
        sequences: {pid: aa_sequence} real protein sequences.
        n_shuffles: Number of shuffled copies per protein.
        seed: RNG seed for reproducibility.

    Returns:
        {"{pid}_shuf{i}": shuffled_sequence} for all proteins and copies.
    """
    rng = random.Random(seed)
    junkyard: dict[str, str] = {}
    for pid in sorted(sequences):  # sorted for determinism
        residues = list(sequences[pid])
        for i in range(n_shuffles):
            shuffled = residues.copy()
            rng.shuffle(shuffled)
            junkyard[f"{pid}_shuf{i}"] = "".join(shuffled)
    return junkyard


def compute_rns(
    query_vectors: dict[str, np.ndarray],
    real_vectors: dict[str, np.ndarray],
    junkyard_vectors: dict[str, np.ndarray],
    k: int = 1000,
) -> dict[str, float]:
    """Compute RNS_k for each query protein.

    Builds a FAISS index over the union of real + junkyard protein vectors,
    then for each query finds its k nearest neighbors and reports the
    fraction that are junkyard.

    Args:
        query_vectors: {pid: (D,) float32} proteins to score. Typically
            the test set, embedded by the model being evaluated.
        real_vectors: {pid: (D,) float32} biologically real protein vectors
            (the 'anchor' set — typically the same test proteins embedded
            by ProtT5 or another trusted source, PLUS the training set).
        junkyard_vectors: {pid: (D,) float32} shuffled-sequence embeddings.
        k: Number of nearest neighbors. Paper recommends k > 100.

    Returns:
        {pid: rns_score} for each query. rns in [0, 1].
    """
    import faiss

    # Build combined index: real + junkyard
    all_ids = []
    is_junkyard = []
    vecs = []
    for pid, vec in real_vectors.items():
        all_ids.append(pid)
        is_junkyard.append(False)
        vecs.append(vec.astype(np.float32))
    for pid, vec in junkyard_vectors.items():
        all_ids.append(pid)
        is_junkyard.append(True)
        vecs.append(vec.astype(np.float32))

    matrix = np.stack(vecs)  # (N, D)
    d = matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(matrix)

    is_junk_arr = np.array(is_junkyard)  # (N,) bool

    # Query
    query_ids = sorted(query_vectors.keys())
    query_mat = np.stack([query_vectors[pid].astype(np.float32)
                          for pid in query_ids])

    # k+1 because the query itself may be in the index (self-match)
    _, indices = index.search(query_mat, k + 1)

    scores: dict[str, float] = {}
    for i, pid in enumerate(query_ids):
        neighbors = indices[i]
        # Exclude self-match if present
        neighbor_mask = np.array([all_ids[j] != pid for j in neighbors])
        valid_neighbors = neighbors[neighbor_mask][:k]
        if len(valid_neighbors) == 0:
            scores[pid] = 1.0  # no valid neighbors -> worst case
            continue
        n_junk = is_junk_arr[valid_neighbors].sum()
        scores[pid] = float(n_junk / len(valid_neighbors))

    return scores
