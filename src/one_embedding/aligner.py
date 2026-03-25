"""Embedding-space protein alignment.

Aligns proteins using cosine similarity between per-residue embeddings
as a substitution score, with standard Needleman-Wunsch (global) or
Smith-Waterman (local) dynamic programming.

Based on PEbA (Protein Embedding Based Alignment, BMC Bioinformatics 2024).
Enhancement: z-score filtering from EBA (Bioinformatics 2024) for twilight zone.
"""

import numpy as np
from typing import Tuple, Optional


def embedding_score_matrix(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    scale: float = 10.0,
) -> np.ndarray:
    """Compute cosine similarity matrix between per-residue embeddings.

    Args:
        emb_a: (L_A, D) per-residue embeddings for protein A
        emb_b: (L_B, D) per-residue embeddings for protein B
        scale: multiply cosine similarity by this factor (PEbA uses 10)

    Returns:
        (L_A, L_B) scoring matrix
    """
    # Normalize rows to unit length
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-10)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-10)
    # Cosine similarity matrix via dot product of normalized vectors
    sim = norm_a @ norm_b.T
    return sim * scale


def z_score_filter(score_matrix: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to suppress noise (EBA method).

    Row-wise and column-wise z-score normalization enhances true
    structural correspondence signals in the twilight zone.

    Args:
        score_matrix: (L_A, L_B) raw similarity scores

    Returns:
        (L_A, L_B) z-score filtered matrix
    """
    # Row-wise z-score
    row_mean = score_matrix.mean(axis=1, keepdims=True)
    row_std = score_matrix.std(axis=1, keepdims=True) + 1e-10
    z_row = (score_matrix - row_mean) / row_std

    # Column-wise z-score
    col_mean = z_row.mean(axis=0, keepdims=True)
    col_std = z_row.std(axis=0, keepdims=True) + 1e-10
    z_both = (z_row - col_mean) / col_std

    return z_both


def needleman_wunsch(
    score_matrix: np.ndarray,
    gap_open: float = -11.0,
    gap_extend: float = -1.0,
) -> Tuple[str, str, float]:
    """Global alignment using Needleman-Wunsch with affine gap penalties.

    Args:
        score_matrix: (L_A, L_B) substitution scores
        gap_open: penalty for opening a gap
        gap_extend: penalty for extending a gap

    Returns:
        (align_a, align_b, score) where align strings use indices
        and '-' for gaps. Score is the optimal alignment score.
    """
    L_A, L_B = score_matrix.shape

    # Three DP matrices: M (match), X (gap in B), Y (gap in A)
    NEG_INF = -1e9
    M = np.full((L_A + 1, L_B + 1), NEG_INF)
    X = np.full((L_A + 1, L_B + 1), NEG_INF)
    Y = np.full((L_A + 1, L_B + 1), NEG_INF)

    M[0, 0] = 0.0
    for i in range(1, L_A + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, L_B + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend

    # Traceback matrices: 0=M, 1=X, 2=Y
    trace_M = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)
    trace_X = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)
    trace_Y = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)

    for i in range(1, L_A + 1):
        for j in range(1, L_B + 1):
            # Match/mismatch
            scores_M = [
                M[i-1, j-1] + score_matrix[i-1, j-1],
                X[i-1, j-1] + score_matrix[i-1, j-1],
                Y[i-1, j-1] + score_matrix[i-1, j-1],
            ]
            M[i, j] = max(scores_M)
            trace_M[i, j] = int(np.argmax(scores_M))

            # Gap in B (advance in A)
            scores_X = [
                M[i-1, j] + gap_open,
                X[i-1, j] + gap_extend,
            ]
            X[i, j] = max(scores_X)
            trace_X[i, j] = [0, 1][int(np.argmax(scores_X))]

            # Gap in A (advance in B)
            scores_Y = [
                M[i, j-1] + gap_open,
                Y[i, j-1] + gap_extend,
            ]
            Y[i, j] = max(scores_Y)
            trace_Y[i, j] = [0, 2][int(np.argmax(scores_Y))]

    # Find best ending matrix
    end_scores = [M[L_A, L_B], X[L_A, L_B], Y[L_A, L_B]]
    best_matrix = int(np.argmax(end_scores))
    best_score = max(end_scores)

    # Traceback
    align_a = []
    align_b = []
    i, j = L_A, L_B
    current = best_matrix  # 0=M, 1=X, 2=Y

    while i > 0 or j > 0:
        if current == 0:  # M
            if i == 0 or j == 0:
                # Can't match — switch to gap state for remaining positions
                current = 1 if i > 0 else 2
                continue
            align_a.append(i - 1)
            align_b.append(j - 1)
            current = trace_M[i, j]
            i -= 1
            j -= 1
        elif current == 1:  # X (gap in B)
            align_a.append(i - 1)
            align_b.append(-1)  # gap
            current = trace_X[i, j]
            i -= 1
        else:  # Y (gap in A)
            align_a.append(-1)  # gap
            align_b.append(j - 1)
            current = trace_Y[i, j]
            j -= 1

    align_a.reverse()
    align_b.reverse()

    return align_a, align_b, float(best_score)


def smith_waterman(
    score_matrix: np.ndarray,
    gap_open: float = -11.0,
    gap_extend: float = -1.0,
) -> Tuple[str, str, float]:
    """Local alignment using Smith-Waterman with affine gap penalties.

    Args:
        score_matrix: (L_A, L_B) substitution scores
        gap_open: penalty for opening a gap
        gap_extend: penalty for extending a gap

    Returns:
        (align_a, align_b, score) where align lists contain position
        indices and -1 for gaps. Score is the optimal local alignment score.
    """
    L_A, L_B = score_matrix.shape

    NEG_INF = -1e9
    M = np.zeros((L_A + 1, L_B + 1))
    X = np.full((L_A + 1, L_B + 1), NEG_INF)
    Y = np.full((L_A + 1, L_B + 1), NEG_INF)

    trace_M = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)
    trace_X = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)
    trace_Y = np.zeros((L_A + 1, L_B + 1), dtype=np.int8)

    best_score = 0.0
    best_i, best_j, best_matrix = 0, 0, 0

    for i in range(1, L_A + 1):
        for j in range(1, L_B + 1):
            # Match
            scores_M = [
                M[i-1, j-1] + score_matrix[i-1, j-1],
                X[i-1, j-1] + score_matrix[i-1, j-1],
                Y[i-1, j-1] + score_matrix[i-1, j-1],
                0.0,  # local: can start fresh
            ]
            M[i, j] = max(scores_M)
            trace_M[i, j] = int(np.argmax(scores_M))

            # Gap in B
            scores_X = [
                M[i-1, j] + gap_open,
                X[i-1, j] + gap_extend,
            ]
            X[i, j] = max(scores_X)
            trace_X[i, j] = [0, 1][int(np.argmax(scores_X))]

            # Gap in A
            scores_Y = [
                M[i, j-1] + gap_open,
                Y[i, j-1] + gap_extend,
            ]
            Y[i, j] = max(scores_Y)
            trace_Y[i, j] = [0, 2][int(np.argmax(scores_Y))]

            for mat_idx, val in enumerate([M[i, j], X[i, j], Y[i, j]]):
                if val > best_score:
                    best_score = val
                    best_i, best_j, best_matrix = i, j, mat_idx

    # Traceback from best scoring cell
    align_a = []
    align_b = []
    i, j = best_i, best_j
    current = best_matrix

    while i > 0 or j > 0:
        if current == 0:  # M
            if M[i, j] <= 0 or trace_M[i, j] == 3:
                break
            align_a.append(i - 1)
            align_b.append(j - 1)
            current = trace_M[i, j]
            i -= 1
            j -= 1
        elif current == 1:  # X
            if X[i, j] <= 0:
                break
            align_a.append(i - 1)
            align_b.append(-1)
            current = trace_X[i, j]
            i -= 1
        else:  # Y
            if Y[i, j] <= 0:
                break
            align_a.append(-1)
            align_b.append(j - 1)
            current = trace_Y[i, j]
            j -= 1

    align_a.reverse()
    align_b.reverse()

    return align_a, align_b, float(best_score)


def align_embeddings(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    mode: str = "global",
    gap_open: float = -11.0,
    gap_extend: float = -1.0,
    scale: float = 10.0,
    use_z_score: bool = False,
) -> dict:
    """Align two proteins by their per-residue embeddings.

    Args:
        emb_a: (L_A, D) per-residue embeddings for protein A
        emb_b: (L_B, D) per-residue embeddings for protein B
        mode: "global" (Needleman-Wunsch) or "local" (Smith-Waterman)
        gap_open: gap opening penalty
        gap_extend: gap extension penalty
        scale: cosine similarity scaling factor
        use_z_score: apply z-score filtering (better for remote homologs)

    Returns:
        dict with keys: align_a, align_b, score, score_matrix
    """
    score_mat = embedding_score_matrix(emb_a, emb_b, scale=scale)
    if use_z_score:
        score_mat = z_score_filter(score_mat)

    if mode == "global":
        align_a, align_b, score = needleman_wunsch(score_mat, gap_open, gap_extend)
    elif mode == "local":
        align_a, align_b, score = smith_waterman(score_mat, gap_open, gap_extend)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'global' or 'local'.")

    return {
        "align_a": align_a,
        "align_b": align_b,
        "score": score,
        "score_matrix": score_mat,
        "n_aligned": sum(1 for a, b in zip(align_a, align_b) if a >= 0 and b >= 0),
        "n_gaps_a": sum(1 for a in align_a if a < 0),
        "n_gaps_b": sum(1 for b in align_b if b < 0),
    }
