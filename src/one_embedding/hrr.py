"""Holographic Reduced Representations for protein embeddings.

Encodes variable-length per-residue embedding matrices (L, D) into
fixed-size vectors (D,) or (K, D) that preserve both per-protein and
per-residue information. Training-free, PLM-agnostic.

Per-protein:  use hrr_encode() output directly for retrieval/classification.
Per-residue:  use hrr_decode() to recover approximate (L, D) for probes.

Reference: Plate, T.A. (1995). Holographic Reduced Representations.
"""

import numpy as np

# Cache for position vectors (avoid recomputing for same args)
_POS_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _position_vectors(max_len: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic unitary position vectors for HRR.

    Uses real-valued vectors whose FFT has unit magnitude at every frequency
    (|FFT(p_i)[k]| = 1 for all k).  This ensures that unbind(p, bind(p, v)) = v
    exactly (up to floating-point precision), giving near-perfect recovery when
    only one residue is superposed in a slot.

    Construction: build frequency-domain vectors with random phases and unit
    magnitudes, then IFFT back to the time domain.  Conjugate symmetry of the
    frequency domain guarantees real-valued output.

    Args:
        max_len: Number of position vectors to generate.
        dim: Dimensionality of each vector.
        seed: Fixed random seed (deterministic, no learned params).

    Returns:
        (max_len, dim) float32 array of unitary position vectors.
    """
    cache_key = (max_len, dim, seed)
    if cache_key not in _POS_CACHE:
        rng = np.random.RandomState(seed)
        result = np.zeros((max_len, dim), dtype=np.float32)
        for i in range(max_len):
            freq = np.zeros(dim, dtype=complex)
            freq[0] = 1.0  # DC: real, unit magnitude
            if dim % 2 == 0:
                freq[dim // 2] = 1.0  # Nyquist: real, unit magnitude
                half = dim // 2 - 1
            else:
                half = dim // 2
            phases = rng.uniform(0, 2 * np.pi, half)
            freq[1 : 1 + half] = np.exp(1j * phases)
            freq[dim - half :] = np.conj(freq[1 : 1 + half][::-1])
            result[i] = np.fft.ifft(freq).real.astype(np.float32)
        _POS_CACHE[cache_key] = result
    return _POS_CACHE[cache_key]


def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution: bind(a, b) = IFFT(FFT(a) * FFT(b)).

    Binding associates two vectors into a new vector of the same dimension.
    The result is dissimilar to both inputs.

    Args:
        a, b: Vectors of shape (D,) or batch (N, D).

    Returns:
        Bound vector(s), same shape as input, dtype float32.
    """
    return np.real(
        np.fft.ifft(np.fft.fft(a, axis=-1) * np.fft.fft(b, axis=-1), axis=-1)
    ).astype(np.float32)


def hrr_unbind(key: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """Circular correlation: unbind(key, trace) = IFFT(conj(FFT(key)) * FFT(trace)).

    Approximate inverse of bind: unbind(k, bind(k, v)) ≈ v.

    Args:
        key: Position vector to unbind, shape (D,) or (N, D).
        trace: Holographic trace, shape (D,) or (N, D).

    Returns:
        Recovered vector(s), same shape, dtype float32.
    """
    return np.real(
        np.fft.ifft(
            np.conj(np.fft.fft(key, axis=-1)) * np.fft.fft(trace, axis=-1),
            axis=-1,
        )
    ).astype(np.float32)


def hrr_encode(
    matrix: np.ndarray,
    seed: int = 42,
    normalize: bool = True,
) -> np.ndarray:
    """Encode per-residue embeddings into a single HRR trace vector.

    For each position i:  trace += bind(pos[i], residue[i])
    The trace encodes positional information (unlike mean pool).

    Args:
        matrix: (L, D) per-residue embeddings.
        seed: Random seed for position vectors.
        normalize: If True, L2-normalize residue vectors before binding.

    Returns:
        (D,) holographic trace vector.
    """
    L, D = matrix.shape
    pos = _position_vectors(L, D, seed=seed)

    mat = matrix
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        mat = matrix / norms

    # Vectorized bind + sum
    bound = hrr_bind(pos[:L], mat)  # (L, D)
    return bound.sum(axis=0).astype(np.float32)


def hrr_decode(
    trace: np.ndarray,
    L: int,
    seed: int = 42,
) -> np.ndarray:
    """Decode all residue positions from an HRR trace vector.

    For each position i:  decoded[i] = unbind(pos[i], trace)
    Recovery is approximate due to cross-talk from other positions.
    Quality scales as ~sqrt(D/L).

    Args:
        trace: (D,) holographic trace vector.
        L: Number of residues to decode.
        seed: Must match the seed used for encoding.

    Returns:
        (L, D) approximate per-residue embeddings.
    """
    D = trace.shape[0]
    pos = _position_vectors(L, D, seed=seed)
    # Vectorized unbind via broadcasting
    return hrr_unbind(pos[:L], trace[np.newaxis, :])


def hrr_kslot_encode(
    matrix: np.ndarray,
    K: int = 8,
    seed: int = 42,
    normalize: bool = True,
) -> np.ndarray:
    """K-slot HRR: distribute residues across K independent traces.

    Assigns residues to slots by position blocks:
    slot j gets residues [j*L//K .. (j+1)*L//K).
    Each slot has ~L/K residues → lower cross-talk.

    Args:
        matrix: (L, D) per-residue embeddings.
        K: Number of slots.
        seed: Random seed for position vectors.
        normalize: L2-normalize residue vectors before binding.

    Returns:
        (K, D) multi-slot holographic representation.
    """
    L, D = matrix.shape
    pos = _position_vectors(L, D, seed=seed)

    mat = matrix
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        mat = matrix / norms

    slots = np.zeros((K, D), dtype=np.float32)
    for i in range(L):
        slot_idx = min(i * K // L, K - 1)
        slots[slot_idx] += hrr_bind(pos[i], mat[i])

    return slots


def hrr_kslot_decode(
    slots: np.ndarray,
    L: int,
    seed: int = 42,
) -> np.ndarray:
    """Decode all positions from K-slot HRR.

    Each position is decoded from its assigned slot.

    Args:
        slots: (K, D) multi-slot representation.
        L: Number of residues to decode.
        seed: Must match encoding seed.

    Returns:
        (L, D) approximate per-residue embeddings.
    """
    K, D = slots.shape
    pos = _position_vectors(L, D, seed=seed)

    decoded = np.zeros((L, D), dtype=np.float32)
    for i in range(L):
        slot_idx = min(i * K // L, K - 1)
        decoded[i] = hrr_unbind(pos[i], slots[slot_idx])

    return decoded


def hrr_per_protein(
    matrix: np.ndarray,
    K: int = 1,
    seed: int = 42,
    normalize: bool = True,
) -> np.ndarray:
    """Per-protein vector from per-residue matrix.

    K=1: returns (D,) single HRR trace (same dim as mean pool).
    K>1: returns (K*D,) concatenated K-slot traces.

    Args:
        matrix: (L, D) per-residue embeddings.
        K: Number of slots.
        seed: Random seed.
        normalize: L2-normalize residues before binding.

    Returns:
        Fixed-size protein-level vector.
    """
    if K == 1:
        return hrr_encode(matrix, seed=seed, normalize=normalize)
    slots = hrr_kslot_encode(matrix, K=K, seed=seed, normalize=normalize)
    return slots.ravel().astype(np.float32)


def hrr_per_residue(
    oe: np.ndarray,
    L: int,
    K: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Recover per-residue embeddings from per-protein vector.

    Args:
        oe: Per-protein vector — (D,) if K=1, (K*D,) if K>1.
        L: Number of residues to decode.
        K: Number of slots used during encoding.
        seed: Must match encoding seed.

    Returns:
        (L, D) approximate per-residue embeddings.
    """
    if K == 1:
        return hrr_decode(oe, L, seed=seed)
    D = oe.shape[0] // K
    slots = oe.reshape(K, D)
    return hrr_kslot_decode(slots, L, seed=seed)
