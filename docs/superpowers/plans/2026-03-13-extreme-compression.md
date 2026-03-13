# Experiment 28: Extreme Compression Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 18 novel compression techniques across 5 categories, benchmark all against the full evaluation suite, and find the Pareto-optimal codec at 10X/50X/100X compression.

**Architecture:** 4 new source modules (extreme_compression, quantization, tensor_decomposition, topological) with matching test files. One experiment script with 3 phases (single-stage, pipelines, cross-PLM). Follows Exp 25-27 patterns: step-based execution, JSON checkpointing, per-protein + per-residue evaluation.

**Tech Stack:** numpy, scipy, pywt (wavelets), zstandard (entropy coding), sklearn (NMF/k-means), optional: ripser+persim (TDA), pot (OT). All computation on CPU/numpy (no MPS).

**Spec:** `docs/superpowers/specs/2026-03-13-extreme-compression-benchmark-design.md`

---

## Chunk 1: Dependencies + Category A (Within-Channel Compression)

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add required and optional dependencies**

Add to `[project.dependencies]`:
```
"PyWavelets>=1.6.0",
"zstandard>=0.23.0",
```

Add new optional group:
```toml
[project.optional-dependencies]
extreme = ["pot>=0.9.0", "ripser>=0.6.0", "persim>=0.3.0"]
```

- [ ] **Step 2: Install and verify**

Run: `uv sync && uv run python -c "import pywt; import zstandard; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pywt, zstandard for Exp 28 extreme compression"
```

### Task 2: Create extreme_compression.py (Category A)

**Files:**
- Create: `src/one_embedding/extreme_compression.py`

Functions to implement:

- [ ] **Step 1: Write wavelet_threshold_compress and wavelet_threshold_decompress**

```python
"""Within-channel compression techniques for PLM embeddings.

All functions operate on (L, D) matrices without mixing channels.
"""

import numpy as np

def wavelet_threshold_compress(
    matrix: np.ndarray,
    wavelet: str = "db4",
    threshold_pct: float = 75.0,
) -> dict:
    """Per-channel DWT with coefficient thresholding.

    Args:
        matrix: (L, D) per-residue embeddings.
        wavelet: Wavelet family (db4, db8, sym4).
        threshold_pct: Percentage of coefficients to zero (0-100).

    Returns:
        dict with 'coeffs' (list of arrays per level), 'wavelet', 'original_shape',
        'slices' for reconstruction.
    """
    import pywt
    L, D = matrix.shape
    # Apply DWT per channel (column)
    all_coeffs = []
    for ch in range(D):
        coeffs = pywt.wavedec(matrix[:, ch], wavelet, mode="symmetric")
        # Flatten to compute threshold
        flat = np.concatenate([np.abs(c) for c in coeffs])
        if threshold_pct > 0:
            thresh = np.percentile(flat, threshold_pct)
            coeffs = [pywt.threshold(c, thresh, mode="soft") for c in coeffs]
        all_coeffs.append(coeffs)
    return {
        "coeffs": all_coeffs,
        "wavelet": wavelet,
        "original_shape": matrix.shape,
        "n_levels": len(all_coeffs[0]),
    }


def wavelet_threshold_decompress(compressed: dict) -> np.ndarray:
    """Reconstruct (L, D) from wavelet-thresholded coefficients."""
    import pywt
    L, D = compressed["original_shape"]
    result = np.zeros((L, D), dtype=np.float32)
    for ch, coeffs in enumerate(compressed["coeffs"]):
        rec = pywt.waverec(coeffs, compressed["wavelet"], mode="symmetric")
        result[:, ch] = rec[:L]  # trim padding
    return result
```

- [ ] **Step 2: Write cur_decompose and cur_reconstruct**

```python
def cur_decompose(
    matrix: np.ndarray,
    k: int = 64,
) -> dict:
    """CUR / Interpolative Decomposition — select k actual columns.

    No channel mixing: selected columns ARE original embedding dimensions.

    Args:
        matrix: (L, D) per-residue embeddings.
        k: Number of columns to select.

    Returns:
        dict with 'C' (L, k), 'col_indices' (k,), 'interp_matrix' (k, D),
        'original_shape'.
    """
    import scipy.linalg.interpolative as sli
    L, D = matrix.shape
    k = min(k, min(L, D))
    # Column ID: find k columns that approximate the matrix
    idx, proj = sli.interp_decomp(matrix, k)
    col_indices = idx[:k]
    C = matrix[:, col_indices]  # (L, k) — actual columns, no mixing
    # Reconstruction: matrix ≈ C @ interp_matrix
    interp_matrix = sli.reconstruct_interp_matrix(idx, proj)  # (k, D)
    return {
        "C": C.astype(np.float32),
        "col_indices": col_indices,
        "interp_matrix": interp_matrix.astype(np.float32),
        "original_shape": matrix.shape,
    }


def cur_reconstruct(compressed: dict) -> np.ndarray:
    """Reconstruct (L, D) from CUR decomposition."""
    return compressed["C"] @ compressed["interp_matrix"]
```

- [ ] **Step 3: Write channel_prune and channel_prune_reconstruct**

```python
def compute_channel_importance(
    embeddings: dict[str, np.ndarray],
    max_proteins: int = 1000,
) -> np.ndarray:
    """Compute per-channel variance across corpus.

    Args:
        embeddings: {protein_id: (L, D)} embeddings.
        max_proteins: Max proteins to sample for efficiency.

    Returns:
        (D,) array of per-channel variance (importance scores).
    """
    keys = list(embeddings.keys())[:max_proteins]
    all_vars = []
    for pid in keys:
        m = embeddings[pid]
        all_vars.append(np.var(m, axis=0))  # (D,) variance per channel
    return np.mean(all_vars, axis=0)  # (D,) mean variance across proteins


def channel_prune(
    matrix: np.ndarray,
    importance: np.ndarray,
    k: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep top-k highest-variance channels.

    Args:
        matrix: (L, D) per-residue embeddings.
        importance: (D,) channel importance scores.
        k: Number of channels to keep.

    Returns:
        (pruned (L, k), selected_indices (k,))
    """
    indices = np.argsort(importance)[-k:]  # top-k by importance
    indices = np.sort(indices)  # maintain original order
    return matrix[:, indices], indices
```

- [ ] **Step 4: Write zstd_compress and zstd_decompress helpers**

```python
def zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compress bytes with zstandard."""
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def zstd_decompress(data: bytes) -> bytes:
    """Decompress zstandard bytes."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def measure_compressed_size(matrix: np.ndarray, zstd_level: int = 3) -> dict:
    """Measure raw and zstd-compressed byte sizes.

    Returns:
        dict with 'raw_bytes', 'zstd_bytes', 'zstd_ratio'.
    """
    raw = matrix.tobytes()
    compressed = zstd_compress(raw, level=zstd_level)
    return {
        "raw_bytes": len(raw),
        "zstd_bytes": len(compressed),
        "zstd_ratio": len(raw) / len(compressed),
    }
```

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/extreme_compression.py
git commit -m "feat: add within-channel compression (wavelet, CUR, channel prune, zstd)"
```

### Task 3: Tests for Category A

**Files:**
- Create: `tests/test_extreme_compression.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for within-channel compression techniques."""

import numpy as np
import pytest
from src.one_embedding.extreme_compression import (
    wavelet_threshold_compress,
    wavelet_threshold_decompress,
    cur_decompose,
    cur_reconstruct,
    compute_channel_importance,
    channel_prune,
    zstd_compress,
    zstd_decompress,
    measure_compressed_size,
)


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30 + i, 128).astype(np.float32) for i in range(20)}


class TestWaveletThreshold:
    def test_roundtrip_no_threshold(self, embedding):
        comp = wavelet_threshold_compress(embedding, threshold_pct=0)
        rec = wavelet_threshold_decompress(comp)
        assert rec.shape == embedding.shape
        np.testing.assert_allclose(rec, embedding, atol=1e-5)

    def test_threshold_reduces_coefficients(self, embedding):
        comp = wavelet_threshold_compress(embedding, threshold_pct=75)
        rec = wavelet_threshold_decompress(comp)
        assert rec.shape == embedding.shape
        # Lossy — not exact but still close
        cos_sim = np.mean([
            np.dot(embedding[i], rec[i]) /
            (np.linalg.norm(embedding[i]) * np.linalg.norm(rec[i]) + 1e-8)
            for i in range(embedding.shape[0])
        ])
        assert cos_sim > 0.8

    def test_different_wavelets(self, embedding):
        for wav in ["db4", "db8", "sym4"]:
            comp = wavelet_threshold_compress(embedding, wavelet=wav, threshold_pct=50)
            rec = wavelet_threshold_decompress(comp)
            assert rec.shape == embedding.shape

    def test_deterministic(self, embedding):
        c1 = wavelet_threshold_decompress(wavelet_threshold_compress(embedding, threshold_pct=50))
        c2 = wavelet_threshold_decompress(wavelet_threshold_compress(embedding, threshold_pct=50))
        np.testing.assert_array_equal(c1, c2)


class TestCUR:
    def test_output_shapes(self, embedding):
        comp = cur_decompose(embedding, k=16)
        assert comp["C"].shape == (50, 16)
        assert comp["interp_matrix"].shape == (16, 128)
        assert len(comp["col_indices"]) == 16

    def test_reconstruction_quality(self, embedding):
        comp = cur_decompose(embedding, k=64)
        rec = cur_reconstruct(comp)
        assert rec.shape == embedding.shape
        # k=64 out of 128 should give good reconstruction
        cos_sim = np.mean([
            np.dot(embedding[i], rec[i]) /
            (np.linalg.norm(embedding[i]) * np.linalg.norm(rec[i]) + 1e-8)
            for i in range(embedding.shape[0])
        ])
        assert cos_sim > 0.9

    def test_columns_are_originals(self, embedding):
        """CUR selects ACTUAL columns — no mixing."""
        comp = cur_decompose(embedding, k=16)
        for i, col_idx in enumerate(comp["col_indices"]):
            np.testing.assert_array_equal(comp["C"][:, i], embedding[:, col_idx])

    def test_various_k(self, embedding):
        for k in [8, 16, 32, 64]:
            comp = cur_decompose(embedding, k=k)
            assert comp["C"].shape[1] == k


class TestChannelPrune:
    def test_importance_shape(self, corpus):
        imp = compute_channel_importance(corpus)
        assert imp.shape == (128,)
        assert np.all(imp >= 0)

    def test_prune_keeps_top_k(self, embedding, corpus):
        imp = compute_channel_importance(corpus)
        pruned, indices = channel_prune(embedding, imp, k=32)
        assert pruned.shape == (50, 32)
        assert len(indices) == 32
        # Verify these are the top-variance channels
        top_k = set(np.argsort(imp)[-32:])
        assert set(indices) == top_k

    def test_pruned_matches_original(self, embedding, corpus):
        imp = compute_channel_importance(corpus)
        pruned, indices = channel_prune(embedding, imp, k=32)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(pruned[:, i], embedding[:, idx])


class TestZstd:
    def test_roundtrip(self):
        data = b"hello world" * 1000
        compressed = zstd_compress(data)
        assert len(compressed) < len(data)
        decompressed = zstd_decompress(compressed)
        assert decompressed == data

    def test_measure_size(self, embedding):
        sizes = measure_compressed_size(embedding)
        assert sizes["raw_bytes"] == 50 * 128 * 4
        assert sizes["zstd_bytes"] < sizes["raw_bytes"]
        assert sizes["zstd_ratio"] > 1.0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_extreme_compression.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_extreme_compression.py
git commit -m "test: add tests for within-channel compression"
```

---

## Chunk 2: Category B (Quantization)

### Task 4: Create quantization.py

**Files:**
- Create: `src/one_embedding/quantization.py`

- [ ] **Step 1: Write int8/int4/binary quantization**

```python
"""Quantization codecs for PLM embeddings.

Per-channel affine quantization (int8, int4, binary) and
vector quantization (PQ, RVQ) for extreme compression.
"""

import numpy as np


# ─── Per-Channel Affine Quantization ───


def quantize_int8(
    matrix: np.ndarray,
) -> dict:
    """Per-channel int8 quantization (256 levels).

    Args:
        matrix: (L, D) float32 embeddings.

    Returns:
        dict with 'data' (L, D) uint8, 'scales' (D,), 'zero_points' (D,),
        'original_shape', 'dtype'.
    """
    L, D = matrix.shape
    mins = matrix.min(axis=0)   # (D,)
    maxs = matrix.max(axis=0)   # (D,)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0   # avoid division by zero
    scales = ranges / 255.0     # (D,)
    zero_points = np.round(-mins / scales).astype(np.int32)  # (D,)
    quantized = np.clip(np.round(matrix / scales + zero_points), 0, 255).astype(np.uint8)
    return {
        "data": quantized,
        "scales": scales.astype(np.float32),
        "zero_points": zero_points,
        "original_shape": matrix.shape,
        "dtype": "int8",
    }


def dequantize_int8(compressed: dict) -> np.ndarray:
    """Reconstruct float32 from int8 quantized data."""
    return (compressed["data"].astype(np.float32) - compressed["zero_points"]) * compressed["scales"]


def quantize_int4(
    matrix: np.ndarray,
) -> dict:
    """Per-channel int4 quantization (16 levels). Packed 2 per byte.

    Args:
        matrix: (L, D) float32 embeddings.

    Returns:
        dict with 'data' (L, D//2) uint8 packed, 'scales' (D,),
        'zero_points' (D,), 'original_shape', 'dtype'.
    """
    L, D = matrix.shape
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    scales = ranges / 15.0
    zero_points = np.clip(np.round(-mins / scales), 0, 15).astype(np.int32)
    quantized = np.clip(np.round(matrix / scales + zero_points), 0, 15).astype(np.uint8)
    # Pack two 4-bit values per byte: even columns in high nibble, odd in low
    if D % 2 != 0:
        quantized = np.pad(quantized, ((0, 0), (0, 1)), constant_values=0)
    packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]
    return {
        "data": packed,
        "scales": scales.astype(np.float32),
        "zero_points": zero_points,
        "original_shape": matrix.shape,
        "dtype": "int4",
    }


def dequantize_int4(compressed: dict) -> np.ndarray:
    """Reconstruct float32 from int4 packed data."""
    L, D = compressed["original_shape"]
    packed = compressed["data"]
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    # Interleave back
    D_padded = D + (D % 2)
    unpacked = np.zeros((L, D_padded), dtype=np.uint8)
    unpacked[:, 0::2] = high
    unpacked[:, 1::2] = low
    unpacked = unpacked[:, :D]
    return (unpacked.astype(np.float32) - compressed["zero_points"][:D]) * compressed["scales"][:D]


def quantize_binary(
    matrix: np.ndarray,
) -> dict:
    """Per-channel binary (1-bit) quantization.

    Store sign(x - mean) as packed bits + per-channel mean & scale.

    Args:
        matrix: (L, D) float32 embeddings.

    Returns:
        dict with 'bits' (L, ceil(D/8)) uint8, 'means' (D,),
        'scales' (D,), 'original_shape'.
    """
    L, D = matrix.shape
    means = matrix.mean(axis=0)  # (D,)
    centered = matrix - means
    # Scale = mean absolute deviation per channel
    scales = np.mean(np.abs(centered), axis=0)  # (D,)
    scales[scales == 0] = 1.0
    signs = (centered >= 0).astype(np.uint8)  # (L, D) binary
    # Pack 8 bits per byte
    n_bytes = (D + 7) // 8
    packed = np.zeros((L, n_bytes), dtype=np.uint8)
    for b in range(D):
        byte_idx = b // 8
        bit_idx = 7 - (b % 8)
        packed[:, byte_idx] |= signs[:, b] << bit_idx
    return {
        "bits": packed,
        "means": means.astype(np.float32),
        "scales": scales.astype(np.float32),
        "original_shape": matrix.shape,
        "dtype": "binary",
    }


def dequantize_binary(compressed: dict) -> np.ndarray:
    """Reconstruct float32 from binary quantized data."""
    L, D = compressed["original_shape"]
    packed = compressed["bits"]
    signs = np.zeros((L, D), dtype=np.float32)
    for b in range(D):
        byte_idx = b // 8
        bit_idx = 7 - (b % 8)
        signs[:, b] = ((packed[:, byte_idx] >> bit_idx) & 1).astype(np.float32)
    # Map 0→-1, 1→+1
    signs = signs * 2 - 1
    return signs * compressed["scales"] + compressed["means"]
```

- [ ] **Step 2: Write product quantization (PQ)**

```python
# ─── Product Quantization ───


def pq_fit(
    embeddings: dict[str, np.ndarray],
    M: int = 32,
    n_centroids: int = 256,
    max_residues: int = 500_000,
    seed: int = 42,
) -> dict:
    """Fit PQ codebook from corpus embeddings (unsupervised).

    Args:
        embeddings: {protein_id: (L, D)} corpus embeddings.
        M: Number of sub-vector groups.
        n_centroids: Centroids per sub-space (max 256 for uint8).
        max_residues: Max residues to sample for k-means.
        seed: Random seed.

    Returns:
        dict with 'codebook' (M, n_centroids, sub_dim), 'M', 'n_centroids',
        'sub_dim', 'D'.
    """
    from sklearn.cluster import MiniBatchKMeans

    # Collect all residues into a matrix
    rng = np.random.RandomState(seed)
    all_residues = []
    for pid, emb in embeddings.items():
        all_residues.append(emb.astype(np.float32))
    all_residues = np.vstack(all_residues)
    if len(all_residues) > max_residues:
        idx = rng.choice(len(all_residues), max_residues, replace=False)
        all_residues = all_residues[idx]

    D = all_residues.shape[1]
    sub_dim = D // M
    assert D % M == 0, f"D={D} must be divisible by M={M}"

    codebook = np.zeros((M, n_centroids, sub_dim), dtype=np.float32)
    for m in range(M):
        sub_data = all_residues[:, m * sub_dim : (m + 1) * sub_dim]
        kmeans = MiniBatchKMeans(
            n_clusters=n_centroids, random_state=seed, batch_size=1024, n_init=1,
        )
        kmeans.fit(sub_data)
        codebook[m] = kmeans.cluster_centers_

    return {
        "codebook": codebook,
        "M": M,
        "n_centroids": n_centroids,
        "sub_dim": sub_dim,
        "D": D,
    }


def pq_encode(
    matrix: np.ndarray,
    pq_model: dict,
) -> np.ndarray:
    """Encode (L, D) → (L, M) uint8 indices using fitted PQ codebook.

    Args:
        matrix: (L, D) per-residue embeddings.
        pq_model: Output of pq_fit().

    Returns:
        (L, M) uint8 centroid indices.
    """
    L = matrix.shape[0]
    M = pq_model["M"]
    sub_dim = pq_model["sub_dim"]
    codebook = pq_model["codebook"]  # (M, 256, sub_dim)
    codes = np.zeros((L, M), dtype=np.uint8)
    for m in range(M):
        sub_data = matrix[:, m * sub_dim : (m + 1) * sub_dim]  # (L, sub_dim)
        # Find nearest centroid for each residue
        dists = np.linalg.norm(
            sub_data[:, np.newaxis, :] - codebook[m][np.newaxis, :, :],
            axis=2,
        )  # (L, 256)
        codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
    return codes


def pq_decode(
    codes: np.ndarray,
    pq_model: dict,
) -> np.ndarray:
    """Decode (L, M) uint8 indices → (L, D) approximate embeddings."""
    L, M = codes.shape
    sub_dim = pq_model["sub_dim"]
    codebook = pq_model["codebook"]
    result = np.zeros((L, M * sub_dim), dtype=np.float32)
    for m in range(M):
        result[:, m * sub_dim : (m + 1) * sub_dim] = codebook[m][codes[:, m]]
    return result
```

- [ ] **Step 3: Write residual vector quantization (RVQ)**

```python
# ─── Residual Vector Quantization ───


def rvq_fit(
    embeddings: dict[str, np.ndarray],
    n_levels: int = 3,
    n_centroids: int = 256,
    max_residues: int = 500_000,
    seed: int = 42,
) -> dict:
    """Fit multi-level RVQ codebook from corpus.

    Level 0: coarse VQ on full D.
    Levels 1+: VQ on residuals from previous level.

    Args:
        embeddings: {protein_id: (L, D)} corpus.
        n_levels: Number of quantization levels.
        n_centroids: Centroids per level.
        max_residues: Max residues for k-means.
        seed: Random seed.

    Returns:
        dict with 'codebooks' list of (n_centroids, D), 'n_levels', etc.
    """
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.RandomState(seed)
    all_residues = np.vstack([e.astype(np.float32) for e in embeddings.values()])
    if len(all_residues) > max_residues:
        idx = rng.choice(len(all_residues), max_residues, replace=False)
        all_residues = all_residues[idx]

    D = all_residues.shape[1]
    codebooks = []
    current = all_residues.copy()

    for level in range(n_levels):
        kmeans = MiniBatchKMeans(
            n_clusters=n_centroids, random_state=seed + level, batch_size=1024, n_init=1,
        )
        kmeans.fit(current)
        codebooks.append(kmeans.cluster_centers_.astype(np.float32))
        # Compute residual for next level
        labels = kmeans.predict(current)
        current = current - kmeans.cluster_centers_[labels]

    return {
        "codebooks": codebooks,
        "n_levels": n_levels,
        "n_centroids": n_centroids,
        "D": D,
    }


def rvq_encode(
    matrix: np.ndarray,
    rvq_model: dict,
) -> np.ndarray:
    """Encode (L, D) → (L, n_levels) uint8 indices."""
    L = matrix.shape[0]
    n_levels = rvq_model["n_levels"]
    codes = np.zeros((L, n_levels), dtype=np.uint8)
    current = matrix.astype(np.float32).copy()
    for level in range(n_levels):
        cb = rvq_model["codebooks"][level]  # (256, D)
        dists = np.linalg.norm(current[:, np.newaxis, :] - cb[np.newaxis, :, :], axis=2)
        codes[:, level] = np.argmin(dists, axis=1).astype(np.uint8)
        current = current - cb[codes[:, level]]
    return codes


def rvq_decode(
    codes: np.ndarray,
    rvq_model: dict,
) -> np.ndarray:
    """Decode (L, n_levels) → (L, D) by summing codebook lookups."""
    L, n_levels = codes.shape
    D = rvq_model["D"]
    result = np.zeros((L, D), dtype=np.float32)
    for level in range(n_levels):
        result += rvq_model["codebooks"][level][codes[:, level]]
    return result
```

- [ ] **Step 4: Write byte size helpers**

```python
def compressed_size_bytes(compressed: dict) -> int:
    """Compute actual byte count of a compressed representation.

    Handles int8, int4, binary, PQ codes, etc.
    """
    dtype = compressed.get("dtype", "")
    if dtype == "int8":
        return (
            compressed["data"].nbytes +
            compressed["scales"].nbytes +
            compressed["zero_points"].nbytes
        )
    elif dtype == "int4":
        return (
            compressed["data"].nbytes +
            compressed["scales"].nbytes +
            compressed["zero_points"].nbytes
        )
    elif dtype == "binary":
        return (
            compressed["bits"].nbytes +
            compressed["means"].nbytes +
            compressed["scales"].nbytes
        )
    elif isinstance(compressed, np.ndarray) and compressed.dtype == np.uint8:
        # PQ/RVQ codes
        return compressed.nbytes
    else:
        raise ValueError(f"Unknown compressed format: {dtype}")
```

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/quantization.py
git commit -m "feat: add quantization codecs (int8, int4, binary, PQ, RVQ)"
```

### Task 5: Tests for Category B

**Files:**
- Create: `tests/test_quantization.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for quantization codecs."""

import numpy as np
import pytest
from src.one_embedding.quantization import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
    quantize_binary, dequantize_binary,
    pq_fit, pq_encode, pq_decode,
    rvq_fit, rvq_encode, rvq_decode,
)


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30, 128).astype(np.float32) for i in range(50)}


class TestInt8:
    def test_roundtrip_shape(self, embedding):
        comp = quantize_int8(embedding)
        rec = dequantize_int8(comp)
        assert rec.shape == embedding.shape
        assert comp["data"].dtype == np.uint8

    def test_roundtrip_quality(self, embedding):
        comp = quantize_int8(embedding)
        rec = dequantize_int8(comp)
        # Int8 should be very close
        np.testing.assert_allclose(rec, embedding, atol=0.05)

    def test_size_is_quarter(self, embedding):
        comp = quantize_int8(embedding)
        # Data is uint8: 1 byte vs 4 bytes per value
        assert comp["data"].nbytes == embedding.shape[0] * embedding.shape[1]


class TestInt4:
    def test_roundtrip_shape(self, embedding):
        comp = quantize_int4(embedding)
        rec = dequantize_int4(comp)
        assert rec.shape == embedding.shape

    def test_packed_size(self, embedding):
        comp = quantize_int4(embedding)
        assert comp["data"].shape == (50, 64)  # 128/2 packed

    def test_roundtrip_reasonable(self, embedding):
        comp = quantize_int4(embedding)
        rec = dequantize_int4(comp)
        cos_sim = np.mean([
            np.dot(embedding[i], rec[i]) /
            (np.linalg.norm(embedding[i]) * np.linalg.norm(rec[i]) + 1e-8)
            for i in range(embedding.shape[0])
        ])
        assert cos_sim > 0.9


class TestBinary:
    def test_roundtrip_shape(self, embedding):
        comp = quantize_binary(embedding)
        rec = dequantize_binary(comp)
        assert rec.shape == embedding.shape
        assert comp["bits"].shape == (50, 16)  # 128/8

    def test_signs_preserved(self, embedding):
        comp = quantize_binary(embedding)
        rec = dequantize_binary(comp)
        means = embedding.mean(axis=0)
        orig_signs = np.sign(embedding - means)
        rec_signs = np.sign(rec - comp["means"])
        # Signs should match perfectly
        agreement = np.mean(orig_signs == rec_signs)
        assert agreement > 0.99


class TestPQ:
    def test_fit_codebook_shape(self, corpus):
        model = pq_fit(corpus, M=16, n_centroids=32, max_residues=1000)
        assert model["codebook"].shape == (16, 32, 8)  # 128/16=8
        assert model["M"] == 16

    def test_encode_decode_shape(self, embedding, corpus):
        model = pq_fit(corpus, M=16, n_centroids=32, max_residues=1000)
        codes = pq_encode(embedding, model)
        assert codes.shape == (50, 16)
        assert codes.dtype == np.uint8
        rec = pq_decode(codes, model)
        assert rec.shape == (50, 128)

    def test_encode_deterministic(self, embedding, corpus):
        model = pq_fit(corpus, M=16, n_centroids=32, max_residues=1000)
        c1 = pq_encode(embedding, model)
        c2 = pq_encode(embedding, model)
        np.testing.assert_array_equal(c1, c2)


class TestRVQ:
    def test_fit_codebooks(self, corpus):
        model = rvq_fit(corpus, n_levels=3, n_centroids=32, max_residues=1000)
        assert len(model["codebooks"]) == 3
        assert model["codebooks"][0].shape == (32, 128)

    def test_encode_decode_shape(self, embedding, corpus):
        model = rvq_fit(corpus, n_levels=3, n_centroids=32, max_residues=1000)
        codes = rvq_encode(embedding, model)
        assert codes.shape == (50, 3)
        rec = rvq_decode(codes, model)
        assert rec.shape == (50, 128)

    def test_residuals_decrease(self, embedding, corpus):
        """Each RVQ level should reduce reconstruction error."""
        model = rvq_fit(corpus, n_levels=3, n_centroids=32, max_residues=1000)
        errors = []
        for n in range(1, 4):
            partial_model = {**model, "codebooks": model["codebooks"][:n], "n_levels": n}
            codes = rvq_encode(embedding, partial_model)
            rec = rvq_decode(codes, partial_model)
            errors.append(np.mean((embedding - rec) ** 2))
        # Each level should improve (lower MSE)
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_quantization.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_quantization.py
git commit -m "test: add quantization codec tests"
```

---

## Chunk 3: Category C (Novel Math) + Category D (Structure-Aware)

### Task 6: Create tensor_decomposition.py

**Files:**
- Create: `src/one_embedding/tensor_decomposition.py`

- [ ] **Step 1: Write tensor train decomposition**

```python
"""Tensor network and matrix factorization decompositions for PLM embeddings."""

import numpy as np


def tt_decompose(
    matrix: np.ndarray,
    bond_dim: int = 16,
) -> dict:
    """Tensor Train decomposition of (L, D) matrix.

    Treats the matrix as a sequence of L vectors of dimension D.
    Decomposes via sequential SVD with truncation.

    Args:
        matrix: (L, D) per-residue embeddings.
        bond_dim: Maximum bond dimension (controls compression).

    Returns:
        dict with 'cores' list of (r_{i-1}, n_i, r_i) arrays,
        'original_shape', 'bond_dim'.
    """
    L, D = matrix.shape
    # Reshape into a 2D unfolding and decompose left-to-right
    # Simple approach: treat as (L, D) and do truncated SVD
    # For a proper TT, we'd reshape into higher-order tensor
    # Here we use a practical block decomposition

    # Block approach: split D into chunks, each is a "mode"
    # Choose block size so we get a meaningful decomposition
    block_size = min(bond_dim, D)
    n_blocks = (D + block_size - 1) // block_size

    cores = []
    residual = matrix.copy()

    for i in range(n_blocks - 1):
        start = i * block_size
        end = min((i + 1) * block_size, D)
        block = residual[:, start:end]  # (L, block_size)

        U, S, Vt = np.linalg.svd(block, full_matrices=False)
        # Truncate to bond_dim
        r = min(bond_dim, len(S))
        U = U[:, :r]        # (L, r)
        S = S[:r]            # (r,)
        Vt = Vt[:r, :]       # (r, block_size)

        cores.append({
            "U": U,
            "S": S,
            "Vt": Vt,
            "col_range": (start, end),
        })

        # Project residual onto remaining space
        residual[:, start:end] = U @ np.diag(S) @ Vt

    # Last block stored directly
    if n_blocks > 0:
        start = (n_blocks - 1) * block_size
        cores.append({
            "direct": residual[:, start:],
            "col_range": (start, D),
        })

    return {
        "cores": cores,
        "original_shape": matrix.shape,
        "bond_dim": bond_dim,
    }


def tt_reconstruct(compressed: dict) -> np.ndarray:
    """Reconstruct (L, D) from TT cores."""
    L, D = compressed["original_shape"]
    result = np.zeros((L, D), dtype=np.float32)
    for core in compressed["cores"]:
        start, end = core["col_range"]
        if "direct" in core:
            result[:, start:end] = core["direct"]
        else:
            result[:, start:end] = core["U"] @ np.diag(core["S"]) @ core["Vt"]
    return result


def tt_storage_bytes(compressed: dict) -> int:
    """Compute storage cost of TT decomposition."""
    total = 0
    for core in compressed["cores"]:
        if "direct" in core:
            total += core["direct"].nbytes
        else:
            total += core["U"].nbytes + core["S"].nbytes + core["Vt"].nbytes
    return total


def nmf_fit(
    embeddings: dict[str, np.ndarray],
    k: int = 32,
    max_residues: int = 200_000,
    seed: int = 42,
) -> dict:
    """Fit NMF basis H from corpus (unsupervised).

    LONG SHOT: PLM embeddings contain negatives — we shift to non-negative.

    Args:
        embeddings: {protein_id: (L, D)} corpus.
        k: Number of NMF components.
        max_residues: Max residues for fitting.
        seed: Random seed.

    Returns:
        dict with 'H' (k, D), 'shift' (D,) per-channel minimum, 'k', 'D'.
    """
    from sklearn.decomposition import NMF

    rng = np.random.RandomState(seed)
    all_residues = np.vstack([e.astype(np.float32) for e in embeddings.values()])
    if len(all_residues) > max_residues:
        idx = rng.choice(len(all_residues), max_residues, replace=False)
        all_residues = all_residues[idx]

    # Shift to non-negative
    shift = all_residues.min(axis=0)  # (D,)
    all_residues = all_residues - shift  # now all >= 0

    model = NMF(n_components=k, random_state=seed, max_iter=300)
    model.fit(all_residues)

    return {
        "H": model.components_.astype(np.float32),  # (k, D)
        "shift": shift.astype(np.float32),
        "k": k,
        "D": all_residues.shape[1],
    }


def nmf_encode(
    matrix: np.ndarray,
    nmf_model: dict,
) -> np.ndarray:
    """Encode (L, D) → (L, k) weights via NNLS."""
    from sklearn.decomposition import non_negative_factorization
    shifted = matrix.astype(np.float32) - nmf_model["shift"]
    shifted = np.clip(shifted, 0, None)
    W, _, _ = non_negative_factorization(
        shifted, H=nmf_model["H"], n_components=nmf_model["k"],
        update_H=False, random_state=42,
    )
    return W.astype(np.float32)


def nmf_decode(
    W: np.ndarray,
    nmf_model: dict,
) -> np.ndarray:
    """Decode (L, k) weights → (L, D)."""
    return (W @ nmf_model["H"]) + nmf_model["shift"]
```

- [ ] **Step 2: Commit**

```bash
git add src/one_embedding/tensor_decomposition.py
git commit -m "feat: add tensor train decomposition and NMF codecs"
```

### Task 7: Create topological.py

**Files:**
- Create: `src/one_embedding/topological.py`

- [ ] **Step 1: Write optimal transport, persistent homology, SimHash, AA-residual**

```python
"""Topological, distributional, and structure-aware compression.

Includes: Sliced Wasserstein distance, persistent homology,
SimHash (LSH), and amino acid residual coding.
"""

import numpy as np


# ─── Sliced Wasserstein Distance ───


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    """Sliced Wasserstein distance between two point clouds.

    Projects both clouds onto random 1D slices, compares sorted projections.

    Args:
        X: (L1, D) first point cloud (residues of protein 1).
        Y: (L2, D) second point cloud (residues of protein 2).
        n_projections: Number of random 1D projections.
        seed: Random seed.

    Returns:
        float: Mean 1D Wasserstein distance across projections.
    """
    D = X.shape[1]
    rng = np.random.RandomState(seed)
    # Random unit vectors on the sphere
    directions = rng.randn(n_projections, D)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total_dist = 0.0
    for d in directions:
        proj_x = np.sort(X @ d)
        proj_y = np.sort(Y @ d)
        # Interpolate to same length for comparison
        n = max(len(proj_x), len(proj_y))
        px = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj_x)), proj_x)
        py = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj_y)), proj_y)
        total_dist += np.mean(np.abs(px - py))

    return total_dist / n_projections


def sliced_wasserstein_matrix(
    embeddings: dict[str, np.ndarray],
    ids: list[str],
    n_projections: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Compute pairwise SWD matrix for retrieval evaluation.

    Returns:
        (N, N) distance matrix.
    """
    N = len(ids)
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = sliced_wasserstein_distance(
                embeddings[ids[i]], embeddings[ids[j]],
                n_projections=n_projections, seed=seed,
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


# ─── Persistent Homology ───


def persistence_image(
    matrix: np.ndarray,
    max_dim: int = 1,
    n_bins: int = 20,
    spread: float = 1.0,
) -> np.ndarray | None:
    """Compute persistence image from residue point cloud.

    Args:
        matrix: (L, D) per-residue embeddings.
        max_dim: Maximum homological dimension (0=components, 1=loops).
        n_bins: Resolution of persistence image.
        spread: Gaussian spread for persistence image.

    Returns:
        (n_bins, n_bins, max_dim+1) persistence image, or None if libraries unavailable.
    """
    try:
        from ripser import ripser
        from persim import PersistenceImager
    except ImportError:
        return None

    # Subsample if too many residues (Rips is O(n^3))
    L = matrix.shape[0]
    if L > 200:
        rng = np.random.RandomState(42)
        idx = rng.choice(L, 200, replace=False)
        matrix = matrix[idx]

    result = ripser(matrix.astype(np.float64), maxdim=max_dim)
    diagrams = result["dgms"]

    images = np.zeros((n_bins, n_bins, max_dim + 1), dtype=np.float32)
    pimgr = PersistenceImager(
        pixel_size=spread, birth_range=(0, None), pers_range=(0, None),
    )

    for dim in range(min(max_dim + 1, len(diagrams))):
        dgm = diagrams[dim]
        # Remove infinite death points
        dgm = dgm[np.isfinite(dgm[:, 1])]
        if len(dgm) > 0:
            try:
                img = pimgr.transform(dgm.reshape(1, -1, 2))
                # Resize to target shape
                from scipy.ndimage import zoom
                img = img[0]
                if img.shape[0] != n_bins or img.shape[1] != n_bins:
                    img = zoom(img, (n_bins / img.shape[0], n_bins / img.shape[1]))
                images[:, :, dim] = img[:n_bins, :n_bins]
            except Exception:
                pass

    return images


# ─── SimHash (Locality-Sensitive Hashing) ───


def simhash_encode(
    matrix: np.ndarray,
    n_bits: int = 1024,
    seed: int = 42,
) -> dict:
    """Encode per-residue embeddings as binary SimHash codes.

    h(x) = sign(W @ x) where W is random Gaussian.

    Args:
        matrix: (L, D) per-residue embeddings.
        n_bits: Number of hash bits per residue.
        seed: Random seed.

    Returns:
        dict with 'bits' (L, n_bits//8) uint8, 'projection' (n_bits, D),
        'original_shape'.
    """
    L, D = matrix.shape
    rng = np.random.RandomState(seed)
    W = rng.randn(n_bits, D).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    projections = matrix @ W.T  # (L, n_bits)
    signs = (projections >= 0).astype(np.uint8)

    # Pack bits
    n_bytes = n_bits // 8
    packed = np.zeros((L, n_bytes), dtype=np.uint8)
    for b in range(n_bits):
        byte_idx = b // 8
        bit_idx = 7 - (b % 8)
        packed[:, byte_idx] |= signs[:, b] << bit_idx

    return {
        "bits": packed,
        "n_bits": n_bits,
        "original_shape": matrix.shape,
        "seed": seed,
    }


def simhash_decode_approx(
    compressed: dict,
) -> np.ndarray:
    """Approximate reconstruction from SimHash (for per-residue probes).

    Uses pseudoinverse of the random projection.
    Quality will be poor but this tests the theoretical limit.
    """
    L, D = compressed["original_shape"]
    n_bits = compressed["n_bits"]
    rng = np.random.RandomState(compressed["seed"])
    W = rng.randn(n_bits, D).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    # Unpack bits to signs
    signs = np.zeros((L, n_bits), dtype=np.float32)
    packed = compressed["bits"]
    for b in range(n_bits):
        byte_idx = b // 8
        bit_idx = 7 - (b % 8)
        signs[:, b] = ((packed[:, byte_idx] >> bit_idx) & 1).astype(np.float32)
    signs = signs * 2 - 1  # map 0→-1, 1→+1

    # Pseudoinverse reconstruction
    W_pinv = np.linalg.pinv(W)  # (D, n_bits)
    return (signs @ W_pinv.T).astype(np.float32)


# ─── Amino Acid Residual Coding ───


def compute_aa_centroids(
    embeddings: dict[str, np.ndarray],
    sequences: dict[str, str],
    max_proteins: int = 5000,
) -> np.ndarray:
    """Compute per-amino-acid centroid embeddings from corpus.

    Args:
        embeddings: {protein_id: (L, D)} embeddings.
        sequences: {protein_id: "ACDEFG..."} amino acid sequences.

    Returns:
        (20, D) centroids indexed by AA_INDEX.
    """
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

    # Collect per-AA sums and counts
    D = next(iter(embeddings.values())).shape[1]
    sums = np.zeros((20, D), dtype=np.float64)
    counts = np.zeros(20, dtype=np.int64)

    for pid in list(embeddings.keys())[:max_proteins]:
        if pid not in sequences:
            continue
        emb = embeddings[pid]
        seq = sequences[pid]
        L = min(len(seq), emb.shape[0])
        for i in range(L):
            aa = seq[i].upper()
            if aa in aa_to_idx:
                idx = aa_to_idx[aa]
                sums[idx] += emb[i]
                counts[idx] += 1

    # Average
    mask = counts > 0
    centroids = np.zeros((20, D), dtype=np.float32)
    centroids[mask] = (sums[mask] / counts[mask, np.newaxis]).astype(np.float32)
    return centroids


def aa_residual_encode(
    matrix: np.ndarray,
    sequence: str,
    centroids: np.ndarray,
) -> np.ndarray:
    """Encode as residuals from AA centroids.

    Args:
        matrix: (L, D) per-residue embeddings.
        sequence: Amino acid sequence string.
        centroids: (20, D) per-AA centroid embeddings.

    Returns:
        (L, D) residual matrix (smaller magnitude than original).
    """
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    L = min(len(sequence), matrix.shape[0])
    residual = matrix[:L].copy()
    for i in range(L):
        aa = sequence[i].upper()
        if aa in aa_to_idx:
            residual[i] -= centroids[aa_to_idx[aa]]
    return residual


def aa_residual_decode(
    residual: np.ndarray,
    sequence: str,
    centroids: np.ndarray,
) -> np.ndarray:
    """Decode residuals back to embeddings."""
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    L = min(len(sequence), residual.shape[0])
    result = residual[:L].copy()
    for i in range(L):
        aa = sequence[i].upper()
        if aa in aa_to_idx:
            result[i] += centroids[aa_to_idx[aa]]
    return result
```

- [ ] **Step 2: Commit**

```bash
git add src/one_embedding/topological.py
git commit -m "feat: add OT, TDA, SimHash, AA-residual codecs"
```

### Task 8: Tests for Category C + D

**Files:**
- Create: `tests/test_tensor_decomposition.py`
- Create: `tests/test_topological.py`

- [ ] **Step 1: Write tensor decomposition tests**

```python
"""Tests for tensor train and NMF decompositions."""

import numpy as np
import pytest
from src.one_embedding.tensor_decomposition import (
    tt_decompose, tt_reconstruct, tt_storage_bytes,
    nmf_fit, nmf_encode, nmf_decode,
)


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 128).astype(np.float32)


@pytest.fixture
def corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(30, 128).astype(np.float32) for i in range(20)}


class TestTensorTrain:
    def test_roundtrip_shape(self, embedding):
        comp = tt_decompose(embedding, bond_dim=16)
        rec = tt_reconstruct(comp)
        assert rec.shape == embedding.shape

    def test_higher_bond_dim_better(self, embedding):
        errors = []
        for bd in [4, 8, 16, 32]:
            comp = tt_decompose(embedding, bond_dim=bd)
            rec = tt_reconstruct(comp)
            errors.append(np.mean((embedding - rec) ** 2))
        # Higher bond dim → lower error (or equal for full rank)
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-6

    def test_storage_decreases_with_bond_dim(self, embedding):
        sizes = []
        for bd in [4, 8, 16]:
            comp = tt_decompose(embedding, bond_dim=bd)
            sizes.append(tt_storage_bytes(comp))
        # Lower bond dim = less storage
        assert sizes[0] <= sizes[1] <= sizes[2]


class TestNMF:
    def test_fit_shape(self, corpus):
        model = nmf_fit(corpus, k=16)
        assert model["H"].shape == (16, 128)

    def test_encode_decode_shape(self, embedding, corpus):
        model = nmf_fit(corpus, k=16)
        W = nmf_encode(embedding, model)
        assert W.shape == (50, 16)
        rec = nmf_decode(W, model)
        assert rec.shape == embedding.shape

    def test_non_negative_weights(self, embedding, corpus):
        model = nmf_fit(corpus, k=16)
        W = nmf_encode(embedding, model)
        assert np.all(W >= -1e-6)  # should be non-negative
```

- [ ] **Step 2: Write topological tests**

```python
"""Tests for topological, distributional, and structure-aware codecs."""

import numpy as np
import pytest
from src.one_embedding.topological import (
    sliced_wasserstein_distance,
    simhash_encode, simhash_decode_approx,
    compute_aa_centroids, aa_residual_encode, aa_residual_decode,
)


@pytest.fixture
def embedding():
    rng = np.random.RandomState(42)
    return rng.randn(50, 64).astype(np.float32)


class TestSlicedWasserstein:
    def test_self_distance_zero(self, embedding):
        d = sliced_wasserstein_distance(embedding, embedding)
        assert d < 1e-6

    def test_symmetric(self, embedding):
        rng = np.random.RandomState(7)
        other = rng.randn(40, 64).astype(np.float32)
        d1 = sliced_wasserstein_distance(embedding, other)
        d2 = sliced_wasserstein_distance(other, embedding)
        np.testing.assert_allclose(d1, d2, atol=1e-5)

    def test_different_lengths(self):
        rng = np.random.RandomState(42)
        a = rng.randn(30, 32).astype(np.float32)
        b = rng.randn(50, 32).astype(np.float32)
        d = sliced_wasserstein_distance(a, b)
        assert d > 0


class TestSimHash:
    def test_encode_shape(self, embedding):
        comp = simhash_encode(embedding, n_bits=512)
        assert comp["bits"].shape == (50, 64)  # 512/8=64

    def test_decode_shape(self, embedding):
        comp = simhash_encode(embedding, n_bits=512)
        rec = simhash_decode_approx(comp)
        assert rec.shape == embedding.shape

    def test_deterministic(self, embedding):
        c1 = simhash_encode(embedding, n_bits=512)
        c2 = simhash_encode(embedding, n_bits=512)
        np.testing.assert_array_equal(c1["bits"], c2["bits"])


class TestAAResiudal:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(5, 32).astype(np.float32)
        seq = "ACDEF"
        centroids = rng.randn(20, 32).astype(np.float32)
        residual = aa_residual_encode(emb, seq, centroids)
        rec = aa_residual_decode(residual, seq, centroids)
        np.testing.assert_allclose(rec, emb, atol=1e-5)

    def test_residuals_smaller(self):
        """Residuals should have smaller magnitude than originals."""
        rng = np.random.RandomState(42)
        # Make embeddings clustered near AA centroids
        centroids = rng.randn(20, 32).astype(np.float32)
        seq = "AAACCCDDD"
        emb = np.zeros((9, 32), dtype=np.float32)
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
        for i, aa in enumerate(seq):
            emb[i] = centroids[aa_to_idx[aa]] + rng.randn(32) * 0.1
        residual = aa_residual_encode(emb, seq, centroids)
        assert np.mean(np.abs(residual)) < np.mean(np.abs(emb))

    def test_centroid_computation(self):
        rng = np.random.RandomState(42)
        centroids_true = rng.randn(20, 32).astype(np.float32)
        # Create corpus where each protein is near its AA centroids
        embeddings = {}
        sequences = {}
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
        for i in range(100):
            seq = "".join(rng.choice(list(aa_order), 20))
            emb = np.array([centroids_true[aa_to_idx[aa]] for aa in seq]) + rng.randn(20, 32) * 0.01
            embeddings[f"p{i}"] = emb.astype(np.float32)
            sequences[f"p{i}"] = seq
        computed = compute_aa_centroids(embeddings, sequences)
        assert computed.shape == (20, 32)
```

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/test_tensor_decomposition.py tests/test_topological.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_tensor_decomposition.py tests/test_topological.py
git commit -m "test: add tests for tensor decomposition and topological codecs"
```

---

## Chunk 4: Experiment 28 Script

### Task 9: Create experiment script — helpers and Phase 1

**Files:**
- Create: `experiments/28_extreme_compression_benchmark.py`

- [ ] **Step 1: Write file header, imports, constants, and helpers**

```python
"""Experiment 28: Extreme Compression Benchmark.

18 novel compression techniques across 5 categories, benchmarked against
the full evaluation suite. Targets: 10X, 50X, 100X compression.

Steps:
  P1a: Category A — within-channel compression (wavelet, CUR, channel prune)
  P1b: Category B — quantization (int8, int4, binary, PQ, RVQ)
  P1c: Category C — novel math (tensor train, NMF, OT, TDA)
  P1d: Category D — structure-aware (AA-residual, SimHash)
  P2:  Multi-stage pipelines (E1-E6)
  P3:  Cross-PLM validation + multi-seed top-3

Usage:
  uv run python experiments/28_extreme_compression_benchmark.py --step P1a
  uv run python experiments/28_extreme_compression_benchmark.py  # all steps
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, evaluate_ss8_probe, load_cb513_csv,
)
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size
from src.one_embedding.transforms import dct_summary
from src.one_embedding.universal_transforms import (
    feature_hash, random_orthogonal_project,
)
from src.one_embedding.path_transforms import displacement_encode, displacement_decode
from src.one_embedding.extreme_compression import (
    wavelet_threshold_compress, wavelet_threshold_decompress,
    cur_decompose, cur_reconstruct,
    compute_channel_importance, channel_prune,
    zstd_compress, measure_compressed_size,
)
from src.one_embedding.quantization import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
    quantize_binary, dequantize_binary,
    pq_fit, pq_encode, pq_decode,
    rvq_fit, rvq_encode, rvq_decode,
)
from src.one_embedding.tensor_decomposition import (
    tt_decompose, tt_reconstruct, tt_storage_bytes,
    nmf_fit, nmf_encode, nmf_decode,
)
from src.one_embedding.topological import (
    sliced_wasserstein_distance, sliced_wasserstein_matrix,
    persistence_image,
    simhash_encode, simhash_decode_approx,
    compute_aa_centroids, aa_residual_encode, aa_residual_decode,
)
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "extreme_compression_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": [], "results": {}}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def mark_done(results, step):
    if step not in results.setdefault("steps_done", []):
        results["steps_done"].append(step)


def monitor():
    try:
        l1, l5, l15 = os.getloadavg()
        print(f"  System load: {l1:.1f} / {l5:.1f} / {l15:.1f}")
    except OSError:
        pass


def load_metadata():
    meta = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    meta, _ = filter_by_family_size(meta, min_members=3)
    return meta


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def load_plm_embeddings(plm_stem, dataset="medium5k"):
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
    embeddings = load_residue_embeddings(h5_path)
    if any("|" in k for k in list(embeddings.keys())[:5]):
        return {k.split("|")[0]: v for k, v in embeddings.items()}
    return embeddings


def cap_length(embeddings, max_len=MAX_LEN):
    return {k: v[:max_len] for k, v in embeddings.items()}


def time_codec(codec_fn, embeddings, ids, n_warmup=10, n_timed=100):
    """Benchmark encode speed (ms/protein)."""
    subset = [pid for pid in ids if pid in embeddings][:n_warmup + n_timed]
    # Warmup
    for pid in subset[:n_warmup]:
        codec_fn(embeddings[pid])
    # Timed
    times = []
    for pid in subset[n_warmup:n_warmup + n_timed]:
        t0 = time.perf_counter()
        codec_fn(embeddings[pid])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def eval_retrieval(vectors, metadata, test_ids):
    """Shorthand for retrieval evaluation."""
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids, metric="cosine",
    )


def eval_per_residue(coded_embs, plm_stem):
    """Evaluate SS3 and SS8 on coded per-residue embeddings."""
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if not cb513_path.exists():
        return {}
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)

    # Load CB513 embeddings
    cb_embs = load_plm_embeddings(plm_stem, dataset="cb513")
    cb_embs = cap_length(cb_embs)

    # Apply same codec to CB513 embeddings — caller must pass coded_embs
    # Here coded_embs should already have CB513 proteins
    avail = [pid for pid in ss3_labels if pid in coded_embs]
    if len(avail) < 50:
        return {}

    rng = random.Random(42)
    rng.shuffle(avail)
    n_train = int(len(avail) * 0.8)
    train_ids = avail[:n_train]
    test_ids_cb = avail[n_train:]

    results = {}
    ss3 = evaluate_ss3_probe(coded_embs, ss3_labels, train_ids, test_ids_cb)
    results["ss3_q3"] = ss3.get("q3", 0)
    ss8 = evaluate_ss8_probe(coded_embs, ss8_labels, train_ids, test_ids_cb)
    results["ss8_q8"] = ss8.get("q8", 0)
    return results


def eval_reconstruction(original, reconstructed, ids):
    """Mean CosSim between original and reconstructed per-residue."""
    cos_sims = []
    for pid in ids:
        if pid not in original or pid not in reconstructed:
            continue
        orig = original[pid]
        recon = reconstructed[pid]
        L = min(orig.shape[0], recon.shape[0])
        for i in range(L):
            norm_o = np.linalg.norm(orig[i])
            norm_r = np.linalg.norm(recon[i])
            if norm_o > 1e-8 and norm_r > 1e-8:
                cos_sims.append(np.dot(orig[i], recon[i]) / (norm_o * norm_r))
    return float(np.mean(cos_sims)) if cos_sims else 0.0
```

This continues with the step functions. The full experiment script follows the pattern from Exp 25-26 with:
- Each step function loads data, applies codecs, evaluates, saves results
- Codec registry pattern: list of (name, encode_fn, decode_fn, params)
- Results saved per-technique with all 9 metrics

- [ ] **Step 2: Write step P1a (Category A: within-channel)**

Write the `step_P1a(results)` function that:
1. Loads ProtT5 embeddings and metadata
2. Tests wavelet thresholding with params: wavelet={db4}, threshold_pct={50, 75, 90, 95}
3. Tests CUR decomposition with k={32, 64, 128, 256}
4. Tests channel pruning with k={64, 128, 256, 512}
5. Tests delta encoding (order=1)
6. For each: compute protein vector (mean pool of compressed), evaluate retrieval
7. For each: reconstruct per-residue, evaluate SS3/SS8 on CB513, compute CosSim
8. Measure compressed size (raw + zstd)
9. Time encode/decode speed
10. Save all results via `save_results()`

Follow the exact codec application and evaluation patterns from the helpers above.

- [ ] **Step 3: Write step P1b (Category B: quantization)**

Write `step_P1b(results)` that tests:
1. Int8 on raw ProtT5 (L, 1024)
2. Int8 on rp512 (L, 512)
3. Int4 on raw and rp512
4. Binary on raw and rp512
5. PQ with M={16, 32, 64}, fit on training split
6. RVQ with n_levels={2, 3, 4}, fit on training split
7. Same evaluation suite as P1a

- [ ] **Step 4: Write step P1c (Category C: novel math)**

Write `step_P1c(results)` that tests:
1. Tensor train with bond_dim={4, 8, 16, 32}
2. NMF with k={16, 32, 64}
3. Sliced Wasserstein distance (OT) as retrieval metric — pairwise matrix on test set
4. Persistent homology images (if ripser available) with max_dim=1, n_bins=20
5. Same evaluation suite where applicable

- [ ] **Step 5: Write step P1d (Category D: structure-aware)**

Write `step_P1d(results)` that tests:
1. AA-residual coding with quantize_residual={float16, int8, int4}
2. SimHash with n_bits={512, 1024, 2048}
3. Same evaluation suite

- [ ] **Step 6: Write step P2 (multi-stage pipelines)**

Write `step_P2(results)` that tests 6 predefined chains:
1. E1: wavelet(75%) → int8 → zstd
2. E2: CUR(k=64) → PQ(M=8)
3. E3: AA-residual → PQ(M=32)
4. E4: rp512 → int8 → zstd
5. E5: delta → int4 → zstd
6. E6: best_per_residue + OT/TDA for protein vector

- [ ] **Step 7: Write step P3 (cross-PLM + multi-seed)**

Write `step_P3(results)` that:
1. Runs top-5 techniques from P1+P2 on ESM2-650M and ESM-C-300M
2. Runs top-3 overall with seeds {42, 123, 7}
3. Reports mean +/- std

- [ ] **Step 8: Write main() with argparse**

```python
def main():
    parser = argparse.ArgumentParser(description="Experiment 28: Extreme Compression")
    parser.add_argument("--step", type=str, default=None)
    args = parser.parse_args()

    results = load_results()
    steps = {
        "P1A": step_P1a, "P1B": step_P1b,
        "P1C": step_P1c, "P1D": step_P1d,
        "P2": step_P2, "P3": step_P3,
    }

    if args.step:
        name = args.step.upper()
        if name in steps:
            steps[name](results)
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps)}")
    else:
        for name, fn in steps.items():
            fn(results)

    print("\nDone.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Commit**

```bash
git add experiments/28_extreme_compression_benchmark.py
git commit -m "feat: add Experiment 28 extreme compression benchmark"
```

### Task 10: Run Phase 1a and validate

- [ ] **Step 1: Run P1a**

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P1a`
Expected: Results printed and saved to `data/benchmarks/extreme_compression_results.json`

- [ ] **Step 2: Verify results JSON has Category A entries**

Run: `uv run python -c "import json; r=json.load(open('data/benchmarks/extreme_compression_results.json')); print(json.dumps({k: list(v.keys()) for k,v in r['results'].items()}, indent=2))"`

- [ ] **Step 3: Commit results**

```bash
git add data/benchmarks/extreme_compression_results.json
git commit -m "data: add Category A extreme compression results"
```

### Task 11: Run remaining phases

- [ ] **Step 1: Run P1b** (quantization — estimated 1-2 hours due to PQ/RVQ k-means)

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P1b`

- [ ] **Step 2: Run P1c** (novel math — estimated 1 hour, TDA may be slow)

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P1c`

- [ ] **Step 3: Run P1d** (structure-aware — estimated 30 min)

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P1d`

- [ ] **Step 4: Run P2** (pipelines — estimated 2 hours)

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P2`

- [ ] **Step 5: Run P3** (cross-PLM — estimated 3 hours)

Run: `uv run python experiments/28_extreme_compression_benchmark.py --step P3`

- [ ] **Step 6: Commit all results**

```bash
git add data/benchmarks/extreme_compression_results.json
git commit -m "data: complete Experiment 28 extreme compression benchmark"
```

### Task 12: Summary table and Pareto plot

- [ ] **Step 1: Add summary print function to experiment script**

Add a `step_summary(results)` function that:
1. Reads all results from JSON
2. Prints a formatted table: technique | fitting | Ret@1 | MRR | SS3 Q3 | CosSim | Size | Ratio | Encode ms | Decode ms
3. Identifies Pareto-optimal points on Ret@1 vs Size
4. Prints best codec recommendation at 10X, 50X, 100X

- [ ] **Step 2: Generate Pareto frontier plot**

Add matplotlib figure generation:
- X-axis: log(compression ratio), Y-axis: Ret@1
- Second plot: X-axis: log(compression ratio), Y-axis: SS3 Q3
- Color by category (A=blue, B=green, C=red, D=orange, E=purple)
- Pareto frontier line
- Save to `docs/figures/pub_extreme_compression_pareto.png`

- [ ] **Step 3: Commit**

```bash
git add experiments/28_extreme_compression_benchmark.py docs/figures/
git commit -m "feat: add summary table and Pareto plot for Exp 28"
```
