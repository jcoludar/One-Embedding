"""Tensor Train decomposition and NMF for per-residue embedding compression.

Provides training-free (TT) and corpus-fitting (NMF) compression schemes
that operate on (L, D) per-residue embedding matrices.

Tensor Train: lossless in the limit (full bond_dim), lossy when truncated.
NMF: parts-based non-negative representation — fits a basis from a corpus
of proteins then encodes each protein as a weight matrix W (L, k).
"""

from typing import Any

import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization


# ── Tensor Train ────────────────────────────────────────────────────────────


def tt_decompose(matrix: np.ndarray, bond_dim: int = 16) -> dict:
    """Tensor Train decomposition of (L, D) via block-wise truncated SVD.

    Splits the D dimension into blocks of size bond_dim and compresses
    each block via truncated SVD. The final block is stored directly
    when it is smaller than bond_dim.

    Args:
        matrix: (L, D) per-residue embeddings, float32.
        bond_dim: Number of singular values to keep per block.

    Returns:
        Dict with keys:
            cores        – list of dicts, each with U (L, r), S (r,),
                           Vt (r, block_size), col_range (start, end)
            original_shape – (L, D)
            bond_dim     – bond_dim used
    """
    matrix = matrix.astype(np.float32)
    L, D = matrix.shape
    cores = []
    col = 0
    while col < D:
        end = min(col + bond_dim, D)
        block = matrix[:, col:end]  # (L, block_size)
        block_size = end - col

        if block_size < bond_dim:
            # Last (possibly smaller) block — store directly
            cores.append(
                {
                    "U": None,
                    "S": None,
                    "Vt": None,
                    "raw": block.copy(),
                    "col_range": (col, end),
                }
            )
        else:
            k = min(bond_dim, L, block_size)
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            U = U[:, :k].astype(np.float32)
            S = S[:k].astype(np.float32)
            Vt = Vt[:k, :].astype(np.float32)
            cores.append(
                {
                    "U": U,
                    "S": S,
                    "Vt": Vt,
                    "raw": None,
                    "col_range": (col, end),
                }
            )
        col = end

    return {
        "cores": cores,
        "original_shape": (L, D),
        "bond_dim": bond_dim,
    }


def tt_reconstruct(compressed: dict) -> np.ndarray:
    """Reconstruct (L, D) from TT cores.

    Args:
        compressed: Dict returned by tt_decompose.

    Returns:
        (L, D) reconstructed matrix, float32.
    """
    L, D = compressed["original_shape"]
    result = np.zeros((L, D), dtype=np.float32)

    for core in compressed["cores"]:
        col_start, col_end = core["col_range"]
        if core["raw"] is not None:
            result[:, col_start:col_end] = core["raw"]
        else:
            U, S, Vt = core["U"], core["S"], core["Vt"]
            result[:, col_start:col_end] = (U * S[np.newaxis, :]) @ Vt

    return result


def tt_storage_bytes(compressed: dict) -> int:
    """Compute total storage in bytes for all TT core arrays.

    Args:
        compressed: Dict returned by tt_decompose.

    Returns:
        Total number of bytes occupied by core arrays.
    """
    total = 0
    for core in compressed["cores"]:
        if core["raw"] is not None:
            total += core["raw"].nbytes
        else:
            total += core["U"].nbytes + core["S"].nbytes + core["Vt"].nbytes
    return total


# ── NMF ─────────────────────────────────────────────────────────────────────


def nmf_fit(
    embeddings: dict[str, np.ndarray],
    k: int = 32,
    max_residues: int = 200_000,
    seed: int = 42,
) -> dict:
    """Fit NMF basis H from a corpus of per-residue embeddings.

    Shifts all values to be non-negative by adding per-channel minimum.
    Subsamples the corpus uniformly if total residues exceed max_residues.

    Note: NMF is a LONG SHOT — the negative-shift distorts the data
    distribution significantly for embeddings that span negative values.

    Args:
        embeddings: Dict mapping protein id → (L_i, D) float32 array.
        k: Number of NMF components.
        max_residues: Maximum corpus size for fitting.
        seed: Random seed for NMF initialisation.

    Returns:
        Dict with keys:
            H     – (k, D) basis matrix
            shift – (D,) per-channel minimum subtracted before NMF
            k     – number of components
            D     – embedding dimensionality
    """
    # Collect and subsample corpus
    all_residues = np.concatenate(
        [v for v in embeddings.values()], axis=0
    ).astype(np.float32)  # (N, D)

    N, D = all_residues.shape
    if N > max_residues:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=max_residues, replace=False)
        all_residues = all_residues[idx]

    # Shift to non-negative
    shift = all_residues.min(axis=0)  # (D,)
    all_residues = all_residues - shift[np.newaxis, :]

    model = NMF(
        n_components=k,
        init="nndsvda",
        random_state=seed,
        max_iter=300,
    )
    model.fit(all_residues)
    H = model.components_.astype(np.float32)  # (k, D)

    return {"H": H, "shift": shift, "k": k, "D": D}


def nmf_encode(matrix: np.ndarray, nmf_model: dict) -> np.ndarray:
    """Encode (L, D) per-residue matrix as NMF weights (L, k).

    Args:
        matrix: (L, D) per-residue embeddings.
        nmf_model: Dict returned by nmf_fit.

    Returns:
        (L, k) non-negative weight matrix.
    """
    shift = nmf_model["shift"]
    H = nmf_model["H"]  # (k, D)

    X = (matrix.astype(np.float32) - shift[np.newaxis, :]).clip(0)

    W, _, _ = non_negative_factorization(
        X,
        H=H,
        n_components=nmf_model["k"],
        init="custom",
        update_H=False,
        random_state=42,
        max_iter=200,
    )
    return W.astype(np.float32)


def nmf_decode(W: np.ndarray, nmf_model: dict) -> np.ndarray:
    """Decode NMF weights back to (L, D) embedding approximation.

    Args:
        W: (L, k) weight matrix from nmf_encode.
        nmf_model: Dict returned by nmf_fit.

    Returns:
        (L, D) reconstructed embeddings (shifted back to original space).
    """
    H = nmf_model["H"]  # (k, D)
    shift = nmf_model["shift"]  # (D,)
    return (W @ H + shift[np.newaxis, :]).astype(np.float32)
