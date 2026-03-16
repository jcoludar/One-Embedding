# Experiment 29: Exhaustive Low-Hanging Fruit Sweep — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically test ~30 untried compression/evaluation techniques identified through comprehensive project audit, documenting all results.

**Architecture:** Three new source modules (`preprocessing.py`, `transposed_transforms.py`, `data_analysis.py`) plus extensions to existing modules. One main experiment script with step-based execution following existing patterns (argparse `--step`, JSON results, `benchmark_codec()` harness). Data characterization runs first to inform parameter choices.

**Tech Stack:** numpy, scipy (fft, signal, stats), sklearn (PCA, MLPClassifier), existing evaluation harness.

**Spec:** `docs/superpowers/specs/2026-03-16-exhaustive-fruit-sweep-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/one_embedding/preprocessing.py` | Pre-compression transforms: centering, z-score, All-but-the-Top, PCA rotation. Each takes (L,D) → (L,D). Stateless functions that accept pre-computed stats. |
| `src/one_embedding/transposed_transforms.py` | Transposed-matrix-view transforms: channel resampling, per-protein SVD, channel statistics, zero-pad+flatten. Take (L,D) → fixed-size output. |
| `src/one_embedding/data_analysis.py` | Corpus-level analysis: intrinsic dimensionality, channel distributions, inter-channel correlation. Returns analysis dicts for reporting. |
| `tests/test_preprocessing.py` | Tests for preprocessing module. |
| `tests/test_transposed_transforms.py` | Tests for transposed transforms module. |
| `experiments/29_exhaustive_fruit_sweep.py` | Main experiment: Parts A–I, step-based execution. |

### Modified Files
| File | Changes |
|------|---------|
| `src/one_embedding/universal_transforms.py` | Add: `sparse_random_project()`, `srht_project()`, `percentile_pool()`, `trimmed_mean_pool()` |
| `src/evaluation/retrieval.py` | Add: `compute_rns()` (Random Neighbor Score) |

---

## Chunk 1: Source Modules (preprocessing + transposed transforms)

### Task 1: Preprocessing Module

**Files:**
- Create: `src/one_embedding/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for all preprocessing transforms**

```python
# tests/test_preprocessing.py
"""Tests for embedding preprocessing transforms."""
import numpy as np
import pytest
from src.one_embedding.preprocessing import (
    compute_corpus_stats,
    center_embeddings,
    zscore_embeddings,
    all_but_the_top,
    pca_rotate,
)


class TestComputeCorpusStats:
    """Compute mean, std, top PCs from a corpus of residue embeddings."""

    def test_output_keys(self):
        rng = np.random.RandomState(0)
        corpus = {f"p{i}": rng.randn(50, 8).astype(np.float32) for i in range(20)}
        stats = compute_corpus_stats(corpus, n_sample=500, n_pcs=3)
        assert "mean_vec" in stats
        assert "std_vec" in stats
        assert "top_pcs" in stats
        assert "explained_variance" in stats

    def test_shapes(self):
        rng = np.random.RandomState(0)
        D = 16
        corpus = {f"p{i}": rng.randn(30, D).astype(np.float32) for i in range(10)}
        stats = compute_corpus_stats(corpus, n_sample=200, n_pcs=3)
        assert stats["mean_vec"].shape == (D,)
        assert stats["std_vec"].shape == (D,)
        assert stats["top_pcs"].shape == (3, D)

    def test_mean_is_close_to_actual(self):
        rng = np.random.RandomState(42)
        D = 8
        corpus = {f"p{i}": rng.randn(100, D).astype(np.float32) for i in range(50)}
        all_residues = np.vstack(list(corpus.values()))
        stats = compute_corpus_stats(corpus, n_sample=5000, n_pcs=1)
        np.testing.assert_allclose(stats["mean_vec"], all_residues.mean(axis=0), atol=0.05)


class TestCentering:
    def test_centered_mean_near_zero(self):
        rng = np.random.RandomState(0)
        D = 8
        mean_vec = rng.randn(D).astype(np.float32) * 5  # large bias
        X = rng.randn(50, D).astype(np.float32) + mean_vec
        centered = center_embeddings(X, mean_vec)
        np.testing.assert_allclose(centered.mean(axis=0), np.zeros(D), atol=0.5)

    def test_shape_preserved(self):
        X = np.random.randn(30, 16).astype(np.float32)
        mean_vec = np.zeros(16, dtype=np.float32)
        assert center_embeddings(X, mean_vec).shape == (30, 16)


class TestZScore:
    def test_standardized_channels(self):
        rng = np.random.RandomState(0)
        D = 8
        mean_vec = rng.randn(D).astype(np.float32) * 3
        std_vec = np.abs(rng.randn(D).astype(np.float32)) * 2 + 0.5
        X = rng.randn(200, D).astype(np.float32) * std_vec + mean_vec
        result = zscore_embeddings(X, mean_vec, std_vec)
        # After z-scoring with true stats, each channel ~N(0,1)
        assert result.shape == (200, D)

    def test_zero_std_safe(self):
        X = np.ones((10, 4), dtype=np.float32)
        mean_vec = np.ones(4, dtype=np.float32)
        std_vec = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = zscore_embeddings(X, mean_vec, std_vec)
        # Zero-std channels should be zeroed out, not NaN
        assert not np.any(np.isnan(result))


class TestAllButTheTop:
    def test_removes_top_pc_component(self):
        rng = np.random.RandomState(42)
        D = 16
        # Create data with a strong first PC
        base = rng.randn(100, D).astype(np.float32)
        dominant = rng.randn(D).astype(np.float32)
        dominant /= np.linalg.norm(dominant)
        X = base + 10 * np.outer(np.ones(100), dominant)  # strong bias
        top_pcs = dominant.reshape(1, D)
        result = all_but_the_top(X, top_pcs)
        # Projection onto dominant direction should be near zero
        projections = result @ dominant
        assert np.abs(projections.mean()) < 0.5

    def test_shape_preserved(self):
        X = np.random.randn(50, 8).astype(np.float32)
        pcs = np.random.randn(2, 8).astype(np.float32)
        assert all_but_the_top(X, pcs).shape == (50, 8)


class TestPCARotate:
    def test_shape_preserved(self):
        D = 16
        X = np.random.randn(50, D).astype(np.float32)
        rng = np.random.RandomState(0)
        R = np.linalg.qr(rng.randn(D, D))[0].astype(np.float32)
        result = pca_rotate(X, R)
        assert result.shape == (50, D)

    def test_norms_preserved(self):
        D = 8
        rng = np.random.RandomState(0)
        X = rng.randn(30, D).astype(np.float32)
        R = np.linalg.qr(rng.randn(D, D))[0].astype(np.float32)
        result = pca_rotate(X, R)
        orig_norms = np.linalg.norm(X, axis=1)
        rot_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(orig_norms, rot_norms, atol=1e-5)
```

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement preprocessing module**

```python
# src/one_embedding/preprocessing.py
"""Pre-compression transforms: centering, z-score, All-but-the-Top, PCA rotation.

Each function takes (L, D) → (L, D). Stateless — accepts pre-computed stats
from compute_corpus_stats(). Stats are computed once on a reference corpus
and reused for all proteins.

Reference:
  - Mu & Viswanath (2018). "All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations." ICLR 2018.
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_corpus_stats(
    embeddings: dict[str, np.ndarray],
    n_sample: int = 50_000,
    n_pcs: int = 3,
    seed: int = 42,
) -> dict:
    """Compute corpus-level statistics for preprocessing.

    Samples residues across all proteins to compute global mean, std,
    and top principal components.

    Args:
        embeddings: {protein_id: (L, D) array} dict.
        n_sample: Max residues to sample (for efficiency).
        n_pcs: Number of top PCs to compute.
        seed: Random seed for sampling.

    Returns:
        dict with keys: mean_vec (D,), std_vec (D,), top_pcs (n_pcs, D),
        explained_variance (n_pcs,), rotation_matrix (D, D).
    """
    rng = np.random.RandomState(seed)

    # Collect all residues
    all_residues = []
    for pid, emb in embeddings.items():
        all_residues.append(emb.astype(np.float32))
    all_residues = np.vstack(all_residues)

    # Subsample if too large
    if len(all_residues) > n_sample:
        idx = rng.choice(len(all_residues), n_sample, replace=False)
        sampled = all_residues[idx]
    else:
        sampled = all_residues

    D = sampled.shape[1]
    mean_vec = sampled.mean(axis=0).astype(np.float32)
    std_vec = sampled.std(axis=0).astype(np.float32)

    # PCA for top PCs and rotation matrix
    pca = PCA(n_components=min(n_pcs, D), random_state=seed)
    pca.fit(sampled)
    top_pcs = pca.components_.astype(np.float32)  # (n_pcs, D)
    explained_variance = pca.explained_variance_ratio_.astype(np.float32)

    # Full rotation matrix
    pca_full = PCA(n_components=D, random_state=seed)
    pca_full.fit(sampled)
    rotation_matrix = pca_full.components_.T.astype(np.float32)  # (D, D)

    return {
        "mean_vec": mean_vec,
        "std_vec": std_vec,
        "top_pcs": top_pcs,
        "explained_variance": explained_variance,
        "rotation_matrix": rotation_matrix,
    }


def center_embeddings(X: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """Subtract global mean vector from all residues.

    Args:
        X: (L, D) per-residue embeddings.
        mean_vec: (D,) global mean from compute_corpus_stats().

    Returns:
        (L, D) centered embeddings.
    """
    return (X - mean_vec).astype(np.float32)


def zscore_embeddings(
    X: np.ndarray, mean_vec: np.ndarray, std_vec: np.ndarray
) -> np.ndarray:
    """Per-channel z-score standardization.

    Args:
        X: (L, D) per-residue embeddings.
        mean_vec: (D,) per-channel means.
        std_vec: (D,) per-channel stds. Zero-std channels are zeroed out.

    Returns:
        (L, D) standardized embeddings.
    """
    safe_std = np.where(std_vec > 1e-8, std_vec, 1.0)
    result = (X - mean_vec) / safe_std
    # Zero out channels with zero std (constant channels)
    result[:, std_vec <= 1e-8] = 0.0
    return result.astype(np.float32)


def all_but_the_top(X: np.ndarray, top_pcs: np.ndarray) -> np.ndarray:
    """Remove top-k principal components from embeddings.

    Mu & Viswanath (2018): removing the dominant PCs exposes
    discriminative directions hidden by corpus-level bias.

    Args:
        X: (L, D) per-residue embeddings.
        top_pcs: (k, D) top principal components from compute_corpus_stats().

    Returns:
        (L, D) embeddings with top-k PCs projected out.
    """
    # Project out each PC: X' = X - X @ pc @ pc.T for each pc
    result = X.astype(np.float32).copy()
    for pc in top_pcs:
        pc = pc / np.linalg.norm(pc)  # ensure unit norm
        projections = result @ pc  # (L,)
        result -= np.outer(projections, pc)
    return result


def pca_rotate(X: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply PCA rotation (full-rank, no dimension reduction).

    Rotates to principal axes so variance is concentrated in first dims.
    Useful before RP: random projection on PCA-rotated data is more efficient.

    Args:
        X: (L, D) per-residue embeddings.
        rotation_matrix: (D, D) orthogonal rotation from compute_corpus_stats().

    Returns:
        (L, D) rotated embeddings (same dimensionality, different basis).
    """
    return (X @ rotation_matrix).astype(np.float32)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run pytest tests/test_preprocessing.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/one_embedding/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing transforms (center, zscore, All-but-the-Top, PCA rotate)"
```

### Task 2: Transposed Transforms Module

**Files:**
- Create: `src/one_embedding/transposed_transforms.py`
- Test: `tests/test_transposed_transforms.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transposed_transforms.py
"""Tests for transposed matrix view transforms."""
import numpy as np
import pytest
from src.one_embedding.transposed_transforms import (
    channel_resample,
    per_protein_svd,
    channel_statistics,
    zero_pad_flatten,
)


class TestChannelResample:
    def test_output_shape(self):
        X = np.random.randn(100, 16).astype(np.float32)
        result = channel_resample(X, l_out=32)
        assert result.shape == (16, 32)

    def test_fixed_size_regardless_of_L(self):
        rng = np.random.RandomState(0)
        X1 = rng.randn(50, 8).astype(np.float32)
        X2 = rng.randn(200, 8).astype(np.float32)
        r1 = channel_resample(X1, l_out=16)
        r2 = channel_resample(X2, l_out=16)
        assert r1.shape == r2.shape == (8, 16)

    def test_identity_when_L_equals_l_out(self):
        X = np.random.randn(32, 8).astype(np.float32)
        result = channel_resample(X, l_out=32)
        # Should be close to transposed input
        np.testing.assert_allclose(result, X.T, atol=1e-5)

    def test_protein_vector_from_flatten(self):
        X = np.random.randn(100, 16).astype(np.float32)
        result = channel_resample(X, l_out=8)
        vec = result.flatten()
        assert vec.shape == (16 * 8,)


class TestPerProteinSVD:
    def test_output_shape_k1(self):
        X = np.random.randn(50, 16).astype(np.float32)
        result = per_protein_svd(X, k=1)
        assert result.shape == (16,)

    def test_output_shape_k4(self):
        X = np.random.randn(50, 16).astype(np.float32)
        result = per_protein_svd(X, k=4)
        assert result.shape == (16 * 4,)

    def test_k1_correlates_with_mean(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8).astype(np.float32)
        svd_vec = per_protein_svd(X, k=1)
        mean_vec = X.mean(axis=0)
        # First SV direction should correlate with mean direction
        cos_sim = np.dot(svd_vec, mean_vec) / (
            np.linalg.norm(svd_vec) * np.linalg.norm(mean_vec) + 1e-8
        )
        assert abs(cos_sim) > 0.3  # correlated (sign may flip)

    def test_short_protein(self):
        X = np.random.randn(3, 16).astype(np.float32)
        result = per_protein_svd(X, k=4)
        # k > L: should zero-pad
        assert result.shape == (16 * 4,)


class TestChannelStatistics:
    def test_output_shape_default(self):
        X = np.random.randn(50, 8).astype(np.float32)
        result = channel_statistics(X)
        # Default: [mean, std, min, max, skew, kurtosis] = 6 stats
        assert result.shape == (8 * 6,)

    def test_subset_stats(self):
        X = np.random.randn(50, 8).astype(np.float32)
        result = channel_statistics(X, stats=["mean", "std"])
        assert result.shape == (8 * 2,)

    def test_mean_matches(self):
        X = np.random.randn(100, 4).astype(np.float32)
        result = channel_statistics(X, stats=["mean"])
        np.testing.assert_allclose(result, X.mean(axis=0), atol=1e-6)


class TestZeroPadFlatten:
    def test_output_shape(self):
        X = np.random.randn(50, 8).astype(np.float32)
        result = zero_pad_flatten(X, l_max=100)
        assert result.shape == (8 * 100,)

    def test_truncation(self):
        X = np.random.randn(200, 8).astype(np.float32)
        result = zero_pad_flatten(X, l_max=50)
        assert result.shape == (8 * 50,)

    def test_padding_is_zero(self):
        X = np.ones((10, 4), dtype=np.float32)
        result = zero_pad_flatten(X, l_max=20)
        # Last 10 * 4 = 40 values should be zero
        assert result.shape == (80,)
        # The padded region (positions 10-19 for each channel)
        reshaped = result.reshape(4, 20)  # (D, l_max)
        np.testing.assert_array_equal(reshaped[:, 10:], 0.0)
```

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run pytest tests/test_transposed_transforms.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement transposed transforms module**

```python
# src/one_embedding/transposed_transforms.py
"""Transposed matrix view transforms: treat (L, D) as D channels of length L.

The conventional view is L residues with D features each.
The transposed view is D channels, each a 1D signal of length L.
This opens channel-wise signal processing for variable-length → fixed-size.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis


def channel_resample(X: np.ndarray, l_out: int = 64) -> np.ndarray:
    """Resample each channel from L to fixed l_out positions.

    Instead of pooling (collapsing L), this downsamples the L-axis signal
    per channel. Preserves positional structure along the sequence.

    Args:
        X: (L, D) per-residue embeddings.
        l_out: Fixed output length per channel.

    Returns:
        (D, l_out) resampled channels. Flatten with .flatten() for (D*l_out,) vector.
    """
    L, D = X.shape
    # Transpose: (D, L) — each row is one channel
    Xt = X.T.astype(np.float64)  # (D, L) — scipy needs float64
    # Resample each channel from L to l_out using polyphase filtering
    resampled = scipy_signal.resample(Xt, l_out, axis=1)  # (D, l_out)
    return resampled.astype(np.float32)


def per_protein_svd(X: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract top-k left singular vectors weighted by singular values.

    SVD of (D, L): U (D, r) * S (r,) * Vt (r, L).
    Left singular vectors U[:, :k] are (D, 1) each — fixed size.
    Weighted by S to preserve magnitude information.

    Args:
        X: (L, D) per-residue embeddings.
        k: Number of singular vectors to keep.

    Returns:
        (D * k,) flattened weighted singular vectors. If L < k, zero-pads.
    """
    L, D = X.shape
    # SVD on transposed: (D, L)
    try:
        U, S, Vt = np.linalg.svd(X.T, full_matrices=False)  # U: (D, min(D,L))
    except np.linalg.LinAlgError:
        return np.zeros(D * k, dtype=np.float32)

    actual_k = min(k, len(S))
    # Weight by singular values: each column of U scaled by S
    weighted = U[:, :actual_k] * S[:actual_k]  # (D, actual_k)

    # Zero-pad if fewer than k components
    if actual_k < k:
        padded = np.zeros((D, k), dtype=np.float32)
        padded[:, :actual_k] = weighted
        weighted = padded

    return weighted.flatten().astype(np.float32)


def channel_statistics(
    X: np.ndarray,
    stats: list[str] | None = None,
) -> np.ndarray:
    """Per-channel statistics across residues.

    For each of D channels, compute summary statistics across L residues.
    Richer than mean pool — captures distribution shape per channel.

    Args:
        X: (L, D) per-residue embeddings.
        stats: Which statistics. Default: ["mean", "std", "min", "max", "skew", "kurtosis"].

    Returns:
        (D * n_stats,) flattened statistics vector.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max", "skew", "kurtosis"]

    D = X.shape[1]
    results = []
    for stat_name in stats:
        if stat_name == "mean":
            results.append(X.mean(axis=0))
        elif stat_name == "std":
            results.append(X.std(axis=0))
        elif stat_name == "min":
            results.append(X.min(axis=0))
        elif stat_name == "max":
            results.append(X.max(axis=0))
        elif stat_name == "skew":
            results.append(skew(X, axis=0).astype(np.float32))
        elif stat_name == "kurtosis":
            results.append(kurtosis(X, axis=0).astype(np.float32))
        else:
            raise ValueError(f"Unknown statistic: {stat_name}")

    return np.concatenate(results).astype(np.float32)


def zero_pad_flatten(X: np.ndarray, l_max: int = 512) -> np.ndarray:
    """Zero-pad (or truncate) to fixed length, transpose, and flatten.

    Preserves ALL positional information. The result is a (D * l_max,)
    vector where each D-sized block corresponds to one position.

    Args:
        X: (L, D) per-residue embeddings.
        l_max: Fixed output length. Proteins longer than l_max are truncated.

    Returns:
        (D * l_max,) flattened vector.
    """
    L, D = X.shape
    if L >= l_max:
        padded = X[:l_max]
    else:
        padded = np.zeros((l_max, D), dtype=np.float32)
        padded[:L] = X

    # Transpose to (D, l_max) so channels are contiguous, then flatten
    return padded.T.flatten().astype(np.float32)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run pytest tests/test_transposed_transforms.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/one_embedding/transposed_transforms.py tests/test_transposed_transforms.py
git commit -m "feat: add transposed matrix transforms (channel resample, per-protein SVD, channel stats)"
```

### Task 3: New Pooling and RP Variants in universal_transforms.py

**Files:**
- Modify: `src/one_embedding/universal_transforms.py`
- Test: `tests/test_universal_transforms.py` (existing, extend)

- [ ] **Step 1: Write failing tests for new functions**

Append to existing test file or create focused test. Four new functions: `sparse_random_project`, `srht_project`, `percentile_pool`, `trimmed_mean_pool`.

```python
# Add to tests/test_universal_transforms.py (or create new file)

def test_sparse_random_project_shape():
    from src.one_embedding.universal_transforms import sparse_random_project
    X = np.random.randn(50, 1024).astype(np.float32)
    result = sparse_random_project(X, d_out=512)
    assert result.shape == (50, 512)

def test_sparse_random_project_distance_preservation():
    from src.one_embedding.universal_transforms import sparse_random_project
    rng = np.random.RandomState(42)
    X = rng.randn(2, 128).astype(np.float32)
    orig_dist = np.linalg.norm(X[0] - X[1])
    proj = sparse_random_project(X, d_out=64)
    proj_dist = np.linalg.norm(proj[0] - proj[1])
    # JL: distances preserved within factor ~2 for reasonable d_out
    assert proj_dist > orig_dist * 0.3
    assert proj_dist < orig_dist * 3.0

def test_percentile_pool_shape():
    from src.one_embedding.universal_transforms import percentile_pool
    X = np.random.randn(50, 8).astype(np.float32)
    result = percentile_pool(X, percentiles=[25, 50, 75])
    assert result.shape == (8 * 3,)

def test_percentile_pool_p50_is_median():
    from src.one_embedding.universal_transforms import percentile_pool
    X = np.random.randn(100, 4).astype(np.float32)
    result = percentile_pool(X, percentiles=[50])
    expected = np.median(X, axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_trimmed_mean_pool_shape():
    from src.one_embedding.universal_transforms import trimmed_mean_pool
    X = np.random.randn(50, 8).astype(np.float32)
    result = trimmed_mean_pool(X, proportion=0.1)
    assert result.shape == (8,)

def test_trimmed_mean_pool_removes_outliers():
    from src.one_embedding.universal_transforms import trimmed_mean_pool
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4).astype(np.float32)
    X[0] = 100.0  # outlier
    trimmed = trimmed_mean_pool(X, proportion=0.05)
    plain_mean = X.mean(axis=0)
    # Trimmed mean should be closer to true mean (without outlier)
    true_mean = X[1:].mean(axis=0)
    assert np.linalg.norm(trimmed - true_mean) < np.linalg.norm(plain_mean - true_mean)
```

- [ ] **Step 2: Implement new functions**

Append to `src/one_embedding/universal_transforms.py`:

```python
def sparse_random_project(
    matrix: np.ndarray,
    d_out: int = 512,
    seed: int = 42,
    density: float = 1.0 / 3.0,
) -> np.ndarray:
    """Sparse random projection (Achlioptas 2003).

    Uses {-1, 0, +1} entries with probabilities {density/2, 1-density, density/2}.
    Same JL guarantees as dense Gaussian RP, but 3x sparser.

    Args:
        matrix: (L, D) per-residue embeddings.
        d_out: Output dimensionality.
        seed: Fixed seed.
        density: Fraction of non-zero entries (default 1/3).

    Returns:
        (L, d_out) projected embeddings.
    """
    L, D = matrix.shape
    rng = np.random.RandomState(seed)
    # Generate sparse matrix: each entry is {-1, 0, +1}
    R = np.zeros((D, d_out), dtype=np.float32)
    mask = rng.random((D, d_out)) < density
    signs = rng.choice([-1, 1], size=(D, d_out)).astype(np.float32)
    R[mask] = signs[mask]
    R *= np.sqrt(D / (d_out * density))  # Scale for norm preservation
    return (matrix @ R).astype(np.float32)


def percentile_pool(
    matrix: np.ndarray,
    percentiles: list[float] | None = None,
) -> np.ndarray:
    """Per-channel percentile pooling.

    Computes percentiles across residues for each channel.
    Captures distribution shape — a mini-histogram per channel.

    Args:
        matrix: (L, D) per-residue embeddings.
        percentiles: Which percentiles (default [10, 25, 50, 75, 90]).

    Returns:
        (D * len(percentiles),) concatenated percentile vectors.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    results = []
    for p in percentiles:
        results.append(np.percentile(matrix, p, axis=0).astype(np.float32))
    return np.concatenate(results).astype(np.float32)


def trimmed_mean_pool(
    matrix: np.ndarray,
    proportion: float = 0.1,
) -> np.ndarray:
    """Trimmed mean pool: remove top/bottom proportion before averaging.

    Robust to outlier residues (disordered termini, unusual insertions).

    Args:
        matrix: (L, D) per-residue embeddings.
        proportion: Fraction to trim from each end (default 0.1 = 10%).

    Returns:
        (D,) trimmed mean vector.
    """
    from scipy.stats import trim_mean
    return trim_mean(matrix, proportiontocut=proportion, axis=0).astype(np.float32)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run pytest tests/test_universal_transforms.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/one_embedding/universal_transforms.py tests/test_universal_transforms.py
git commit -m "feat: add sparse RP, percentile pool, trimmed mean pool"
```

### Task 4: Data Analysis Module

**Files:**
- Create: `src/one_embedding/data_analysis.py`

- [ ] **Step 1: Implement data analysis functions**

```python
# src/one_embedding/data_analysis.py
"""Corpus-level embedding analysis: intrinsic dimensionality, channel distributions.

Run once on a reference corpus to characterize the data and inform compression choices.
"""

import numpy as np
from scipy.stats import skew, kurtosis


def intrinsic_dimensionality(
    embeddings: dict[str, np.ndarray],
    n_sample: int = 50_000,
    seed: int = 42,
) -> dict:
    """Measure intrinsic dimensionality of the embedding space.

    Stacks residues into (N, D), computes SVD spectrum, reports
    participation ratio and variance-explained thresholds.

    Args:
        embeddings: {protein_id: (L, D)} dict.
        n_sample: Max residues to sample.
        seed: Random seed.

    Returns:
        dict with singular_values, participation_ratio,
        dims_90pct, dims_95pct, dims_99pct, cumulative_variance.
    """
    rng = np.random.RandomState(seed)
    all_residues = np.vstack([v.astype(np.float32) for v in embeddings.values()])

    if len(all_residues) > n_sample:
        idx = rng.choice(len(all_residues), n_sample, replace=False)
        sampled = all_residues[idx]
    else:
        sampled = all_residues

    # Center
    sampled = sampled - sampled.mean(axis=0)

    # SVD
    S = np.linalg.svd(sampled, compute_uv=False)
    S2 = S ** 2
    total = S2.sum()
    cumvar = np.cumsum(S2) / total

    # Participation ratio: (sum s_i^2)^2 / sum s_i^4
    pr = (S2.sum()) ** 2 / (S2 ** 2).sum()

    return {
        "singular_values": S.tolist(),
        "participation_ratio": float(pr),
        "dims_90pct": int(np.searchsorted(cumvar, 0.90) + 1),
        "dims_95pct": int(np.searchsorted(cumvar, 0.95) + 1),
        "dims_99pct": int(np.searchsorted(cumvar, 0.99) + 1),
        "total_dims": int(sampled.shape[1]),
        "n_samples": int(len(sampled)),
        "cumulative_variance": cumvar.tolist(),
    }


def channel_distributions(
    embeddings: dict[str, np.ndarray],
    n_sample: int = 50_000,
    seed: int = 42,
) -> dict:
    """Per-channel distribution analysis.

    For each channel: mean, std, skewness, kurtosis, inter-protein variance,
    intra-protein variance, discriminative ratio.

    Args:
        embeddings: {protein_id: (L, D)} dict.
        n_sample: Max residues to sample for global stats.
        seed: Random seed.

    Returns:
        dict with per_channel arrays and summary statistics.
    """
    rng = np.random.RandomState(seed)
    all_residues = np.vstack([v.astype(np.float32) for v in embeddings.values()])

    if len(all_residues) > n_sample:
        idx = rng.choice(len(all_residues), n_sample, replace=False)
        sampled = all_residues[idx]
    else:
        sampled = all_residues

    D = sampled.shape[1]

    # Global per-channel stats
    ch_mean = sampled.mean(axis=0)
    ch_std = sampled.std(axis=0)
    ch_skew = skew(sampled, axis=0)
    ch_kurt = kurtosis(sampled, axis=0)

    # Inter-protein vs intra-protein variance
    protein_means = np.array([v.mean(axis=0) for v in embeddings.values()])  # (N, D)
    inter_var = protein_means.var(axis=0)  # variance across protein means

    intra_vars = []
    for v in embeddings.values():
        intra_vars.append(v.var(axis=0))
    intra_var = np.mean(intra_vars, axis=0)  # mean within-protein variance

    # Discriminative ratio: inter / intra (high = channel discriminates between proteins)
    disc_ratio = inter_var / np.maximum(intra_var, 1e-10)

    # Channel correlation matrix
    corr_matrix = np.corrcoef(sampled.T)  # (D, D)

    return {
        "channel_mean": ch_mean.tolist(),
        "channel_std": ch_std.tolist(),
        "channel_skewness": ch_skew.tolist(),
        "channel_kurtosis": ch_kurt.tolist(),
        "inter_protein_variance": inter_var.tolist(),
        "intra_protein_variance": intra_var.tolist(),
        "discriminative_ratio": disc_ratio.tolist(),
        "mean_disc_ratio": float(disc_ratio.mean()),
        "top_10_discriminative_channels": np.argsort(disc_ratio)[-10:][::-1].tolist(),
        "bottom_10_channels": np.argsort(disc_ratio)[:10].tolist(),
        "mean_abs_correlation": float(np.abs(corr_matrix[np.triu_indices(D, k=1)]).mean()),
        "n_highly_correlated_pairs": int((np.abs(corr_matrix[np.triu_indices(D, k=1)]) > 0.8).sum()),
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/one_embedding/data_analysis.py
git commit -m "feat: add data analysis module (intrinsic dim, channel distributions)"
```

### Task 5: RNS Evaluation Function

**Files:**
- Modify: `src/evaluation/retrieval.py`

- [ ] **Step 1: Add compute_rns function**

Append to `src/evaluation/retrieval.py`:

```python
def compute_rns(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    k: int = 10,
    label_key: str = "family",
    metric: str = "cosine",
) -> dict:
    """Random Neighbor Score (Prabakaran & Bromberg 2025).

    For each protein, fraction of k-nearest neighbors that are from
    a different superfamily (biologically unrelated).

    Args:
        vectors: {protein_id: vector} dict.
        metadata: List of dicts with 'protein_id' and label_key.
        k: Number of neighbors.
        label_key: Label field for "related" definition.
        metric: Distance metric.

    Returns:
        dict with mean_rns, std_rns, per_protein_rns.
    """
    id_to_label = {m["protein_id"]: m.get(label_key, "") for m in metadata}
    pids = [pid for pid in vectors if pid in id_to_label]
    if len(pids) < k + 1:
        return {"mean_rns": 1.0, "std_rns": 0.0, "per_protein_rns": {}}

    vecs = np.array([vectors[pid] for pid in pids], dtype=np.float32)
    dist_matrix = cdist(vecs, vecs, metric=metric)

    rns_scores = {}
    for i, pid in enumerate(pids):
        # Find k nearest (excluding self)
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        # Count unrelated neighbors
        my_label = id_to_label[pid]
        n_unrelated = sum(1 for j in neighbors if id_to_label[pids[j]] != my_label)
        rns_scores[pid] = n_unrelated / k

    scores = list(rns_scores.values())
    return {
        "mean_rns": float(np.mean(scores)),
        "std_rns": float(np.std(scores)),
        "n_high_rns": int(sum(1 for s in scores if s > 0.6)),
        "per_protein_rns": rns_scores,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/retrieval.py
git commit -m "feat: add Random Neighbor Score (RNS) evaluation"
```

---

## Chunk 2: Main Experiment Script

### Task 6: Experiment 29 Script — Data Characterization (Part F)

**Files:**
- Create: `experiments/29_exhaustive_fruit_sweep.py`

- [ ] **Step 1: Create experiment script skeleton with Part F**

The script follows the exact pattern of experiments 25-28: argparse with --step,
JSON results file, load_results/save_results/mark_done helpers.

Part F (data characterization) runs first because it informs later parameter choices.

```python
#!/usr/bin/env python3
"""Experiment 29: Exhaustive Low-Hanging Fruit Sweep.

Tests ~30 untried techniques identified through comprehensive project audit.

Parts:
  F:  Data characterization (intrinsic dim, channel analysis) — run FIRST
  A:  Pre-processing (centering, z-score, All-but-the-Top, PCA rotation)
  B:  Transposed matrix view (channel resample, per-protein SVD, channel stats)
  C:  Improved pooling (percentile, trimmed mean)
  D:  RP variants (multi-seed, sparse RP)
  E:  Quantization combinations (int4+codec, JPEG pipeline, predictive coding)
  G:  Evaluation enhancements (RNS, remote homology, MLP probes, Matryoshka)
  H:  Reference corpus approaches (k-means residual, PCA as D-compression)
  I:  Multi-resolution framing (3-level retrieval)

Usage:
  uv run python experiments/29_exhaustive_fruit_sweep.py --step F
  uv run python experiments/29_exhaustive_fruit_sweep.py --step A
  uv run python experiments/29_exhaustive_fruit_sweep.py           # all steps
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
from scipy.spatial.distance import cdist

from src.evaluation.retrieval import evaluate_retrieval_from_vectors, compute_rns
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    load_cb513_csv,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.utils.h5_store import load_residue_embeddings
from src.one_embedding.universal_transforms import (
    random_orthogonal_project,
    feature_hash,
    sparse_random_project,
    percentile_pool,
    trimmed_mean_pool,
)
from src.one_embedding.transforms import dct_summary
from src.one_embedding.preprocessing import (
    compute_corpus_stats,
    center_embeddings,
    zscore_embeddings,
    all_but_the_top,
    pca_rotate,
)
from src.one_embedding.transposed_transforms import (
    channel_resample,
    per_protein_svd,
    channel_statistics,
    zero_pad_flatten,
)
from src.one_embedding.data_analysis import (
    intrinsic_dimensionality,
    channel_distributions,
)
from src.one_embedding.quantization import (
    quantize_int4,
    dequantize_int4,
    quantize_int8,
    dequantize_int8,
)
from src.one_embedding.extreme_compression import measure_compressed_size

# ── Constants ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "exhaustive_sweep_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512


# ── Helpers (same pattern as exp 25-28) ───────────────────────────
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
    if not h5_path.exists():
        print(f"  WARNING: {h5_path} not found")
        return {}
    embeddings = load_residue_embeddings(h5_path)
    if any("|" in k for k in list(embeddings.keys())[:5]):
        return {k.split("|")[0]: v for k, v in embeddings.items()}
    return embeddings


def cap_length(embs, max_len=MAX_LEN):
    return {k: v[:max_len] for k, v in embs.items()}


def eval_retrieval(vectors, metadata, test_ids, metric="cosine"):
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids, metric=metric,
    )


def load_cb513_data():
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if not cb513_path.exists():
        print("  CB513 not found, skipping per-residue probes")
        return None
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    cb_embs = load_plm_embeddings("prot_t5_xl", dataset="cb513")
    if not cb_embs:
        print("  CB513 embeddings not found")
        return None
    return {"ss3_labels": ss3_labels, "embeddings": cap_length(cb_embs)}


def eval_ss3(coded_cb_embs, ss3_labels):
    avail = [pid for pid in ss3_labels if pid in coded_cb_embs]
    if len(avail) < 10:
        return {"q3": 0.0}
    rng = random.Random(42)
    rng.shuffle(avail)
    n_train = int(len(avail) * 0.8)
    train_ids, test_ids_cb = avail[:n_train], avail[n_train:]
    return evaluate_ss3_probe(coded_cb_embs, ss3_labels, train_ids, test_ids_cb)


def time_encode(encode_fn, embeddings, sample_ids, n_timed=100):
    valid_ids = [pid for pid in sample_ids if pid in embeddings][:n_timed]
    if not valid_ids:
        return 0.0
    for pid in valid_ids[:10]:
        try:
            encode_fn(embeddings[pid])
        except Exception:
            pass
    t0 = time.perf_counter()
    for pid in valid_ids:
        encode_fn(embeddings[pid])
    elapsed = time.perf_counter() - t0
    return (elapsed / len(valid_ids)) * 1000.0


def benchmark_protein_vec(
    name, vec_fn, embeddings, test_ids, metadata, cb513_data=None,
    per_residue_fn=None, metric="cosine",
):
    """Benchmark a protein-level vector method.

    Args:
        name: Codec name.
        vec_fn: (L, D) → protein vector.
        embeddings: {pid: (L, D)} dict.
        test_ids: List of test protein IDs.
        metadata: Metadata list.
        cb513_data: Optional CB513 data for SS3 probe.
        per_residue_fn: Optional (L, D) → (L, d) for per-residue probe.
        metric: Distance metric for retrieval.
    """
    result = {"codec": name}

    # Protein vectors
    vectors = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        try:
            vectors[pid] = vec_fn(m)
        except Exception as e:
            continue

    if not vectors:
        return {"codec": name, "error": "no vectors"}

    # Retrieval
    ret = eval_retrieval(vectors, metadata, test_ids, metric=metric)
    result["family_ret1"] = ret["precision@1"]
    result["mrr"] = ret["mrr"]
    result["vec_dim"] = len(next(iter(vectors.values())))

    # SS3 probe (if per-residue function provided)
    if per_residue_fn is not None and cb513_data is not None:
        cb_coded = {}
        for pid in cb513_data["ss3_labels"]:
            if pid not in cb513_data["embeddings"]:
                continue
            m = cb513_data["embeddings"][pid].astype(np.float32)
            try:
                cb_coded[pid] = per_residue_fn(m).astype(np.float32)
            except Exception:
                continue
        if cb_coded:
            ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
            result["ss3_q3"] = ss3.get("q3", 0.0)

    # Size estimate
    sample = list(vectors.values())[:10]
    if sample:
        result["vec_bytes"] = int(np.mean([v.nbytes for v in sample]))

    # Speed
    result["encode_ms"] = time_encode(
        vec_fn, embeddings,
        [pid for pid in test_ids if pid in embeddings],
    )

    return result


# ══════════════════════════════════════════════════════════════════
# STEP F: Data Characterization
# ══════════════════════════════════════════════════════════════════

def step_F(results):
    """Part F: Intrinsic dimensionality + channel distributions."""
    print("\n" + "=" * 60)
    print("STEP F: Data Characterization")
    print("=" * 60)

    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    if not embeddings:
        print("  ERROR: no embeddings")
        return

    print(f"  Loaded {len(embeddings)} proteins")
    monitor()

    # F1: Intrinsic dimensionality
    print("\n--- F1: Intrinsic Dimensionality ---")
    idim = intrinsic_dimensionality(embeddings, n_sample=50_000)
    print(f"  Participation ratio: {idim['participation_ratio']:.1f}")
    print(f"  Dims for 90% var: {idim['dims_90pct']} / {idim['total_dims']}")
    print(f"  Dims for 95% var: {idim['dims_95pct']} / {idim['total_dims']}")
    print(f"  Dims for 99% var: {idim['dims_99pct']} / {idim['total_dims']}")
    results.setdefault("data_analysis", {})["intrinsic_dim"] = {
        k: v for k, v in idim.items() if k != "singular_values" and k != "cumulative_variance"
    }
    # Store singular values separately (large array)
    results["data_analysis"]["sv_top20"] = idim["singular_values"][:20]
    results["data_analysis"]["cumvar_milestones"] = {
        "90pct": idim["dims_90pct"],
        "95pct": idim["dims_95pct"],
        "99pct": idim["dims_99pct"],
    }

    # F2: Channel distributions
    print("\n--- F2: Channel Distributions ---")
    chdist = channel_distributions(embeddings, n_sample=50_000)
    print(f"  Mean discriminative ratio: {chdist['mean_disc_ratio']:.3f}")
    print(f"  Top-10 discriminative channels: {chdist['top_10_discriminative_channels']}")
    print(f"  Mean abs inter-channel correlation: {chdist['mean_abs_correlation']:.3f}")
    print(f"  Highly correlated pairs (|r|>0.8): {chdist['n_highly_correlated_pairs']}")
    results["data_analysis"]["channel_dist"] = {
        k: v for k, v in chdist.items()
        if k not in ("channel_mean", "channel_std", "channel_skewness",
                      "channel_kurtosis", "inter_protein_variance",
                      "intra_protein_variance", "discriminative_ratio")
    }

    # Also do ESM2
    print("\n--- F1b: ESM2 Intrinsic Dimensionality ---")
    esm_embs = cap_length(load_plm_embeddings("esm2_650m"))
    if esm_embs:
        idim_esm = intrinsic_dimensionality(esm_embs, n_sample=50_000)
        print(f"  ESM2 participation ratio: {idim_esm['participation_ratio']:.1f}")
        print(f"  ESM2 dims for 95% var: {idim_esm['dims_95pct']} / {idim_esm['total_dims']}")
        results["data_analysis"]["intrinsic_dim_esm2"] = {
            "participation_ratio": idim_esm["participation_ratio"],
            "dims_90pct": idim_esm["dims_90pct"],
            "dims_95pct": idim_esm["dims_95pct"],
            "dims_99pct": idim_esm["dims_99pct"],
        }

    mark_done(results, "F")
    save_results(results)
    print("\n  Step F DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP A: Pre-Processing Transforms
# ══════════════════════════════════════════════════════════════════

def step_A(results):
    """Part A: Centering, z-score, All-but-the-Top, PCA rotation before RP."""
    print("\n" + "=" * 60)
    print("STEP A: Pre-Processing Transforms")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    train_ids = split["train"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # Compute corpus stats from training set
    train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    print(f"  Computing corpus stats from {len(train_embs)} training proteins...")
    stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5)
    print(f"  Top-5 PC variance explained: {stats['explained_variance'].tolist()}")

    # Ground zero: raw rp512 + dct_K4 (reference)
    def _raw_rp_dct(m):
        compressed = random_orthogonal_project(m, d_out=512)
        return dct_summary(compressed, K=4)

    def _raw_rp(m):
        return random_orthogonal_project(m, d_out=512)

    print("\n  Ground zero: raw rp512+dct_K4...")
    r0 = benchmark_protein_vec("raw_rp512_dct_K4", _raw_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _raw_rp)
    results.setdefault("part_A", {})["raw_rp512_dct_K4"] = r0
    print(f"    Ret@1={r0.get('family_ret1', 'ERR')}, SS3={r0.get('ss3_q3', 'N/A')}")

    # A1: Centering + rp512 + dct_K4
    def _centered_rp_dct(m):
        mc = center_embeddings(m, stats["mean_vec"])
        compressed = random_orthogonal_project(mc, d_out=512)
        return dct_summary(compressed, K=4)

    def _centered_rp(m):
        mc = center_embeddings(m, stats["mean_vec"])
        return random_orthogonal_project(mc, d_out=512)

    print("\n  A1: centered + rp512 + dct_K4...")
    r1 = benchmark_protein_vec("centered_rp512_dct_K4", _centered_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _centered_rp)
    results["part_A"]["centered_rp512_dct_K4"] = r1
    print(f"    Ret@1={r1.get('family_ret1', 'ERR')}, SS3={r1.get('ss3_q3', 'N/A')}")

    # A2: Z-score + rp512 + dct_K4
    def _zscore_rp_dct(m):
        mz = zscore_embeddings(m, stats["mean_vec"], stats["std_vec"])
        compressed = random_orthogonal_project(mz, d_out=512)
        return dct_summary(compressed, K=4)

    def _zscore_rp(m):
        mz = zscore_embeddings(m, stats["mean_vec"], stats["std_vec"])
        return random_orthogonal_project(mz, d_out=512)

    print("\n  A2: z-score + rp512 + dct_K4...")
    r2 = benchmark_protein_vec("zscore_rp512_dct_K4", _zscore_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _zscore_rp)
    results["part_A"]["zscore_rp512_dct_K4"] = r2
    print(f"    Ret@1={r2.get('family_ret1', 'ERR')}, SS3={r2.get('ss3_q3', 'N/A')}")

    # A3: All-but-the-Top (k=1) + rp512 + dct_K4
    for k in [1, 3]:
        def _abtt_rp_dct(m, _k=k):
            ma = all_but_the_top(m, stats["top_pcs"][:_k])
            compressed = random_orthogonal_project(ma, d_out=512)
            return dct_summary(compressed, K=4)

        def _abtt_rp(m, _k=k):
            ma = all_but_the_top(m, stats["top_pcs"][:_k])
            return random_orthogonal_project(ma, d_out=512)

        name = f"abtt_k{k}_rp512_dct_K4"
        print(f"\n  A3: All-but-the-Top k={k} + rp512 + dct_K4...")
        r3 = benchmark_protein_vec(name, _abtt_rp_dct, embeddings,
                                    test_ids, metadata, cb513_data, _abtt_rp)
        results["part_A"][name] = r3
        print(f"    Ret@1={r3.get('family_ret1', 'ERR')}, SS3={r3.get('ss3_q3', 'N/A')}")

    # A4: PCA rotation + rp512 + dct_K4
    def _pca_rot_rp_dct(m):
        mr = pca_rotate(m, stats["rotation_matrix"])
        compressed = random_orthogonal_project(mr, d_out=512)
        return dct_summary(compressed, K=4)

    def _pca_rot_rp(m):
        mr = pca_rotate(m, stats["rotation_matrix"])
        return random_orthogonal_project(mr, d_out=512)

    print("\n  A4: PCA rotation + rp512 + dct_K4...")
    r4 = benchmark_protein_vec("pca_rot_rp512_dct_K4", _pca_rot_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _pca_rot_rp)
    results["part_A"]["pca_rot_rp512_dct_K4"] = r4
    print(f"    Ret@1={r4.get('family_ret1', 'ERR')}, SS3={r4.get('ss3_q3', 'N/A')}")

    # A5: Centering + All-but-the-Top k=1 (combined)
    def _center_abtt_rp_dct(m):
        mc = center_embeddings(m, stats["mean_vec"])
        ma = all_but_the_top(mc, stats["top_pcs"][:1])
        compressed = random_orthogonal_project(ma, d_out=512)
        return dct_summary(compressed, K=4)

    def _center_abtt_rp(m):
        mc = center_embeddings(m, stats["mean_vec"])
        ma = all_but_the_top(mc, stats["top_pcs"][:1])
        return random_orthogonal_project(ma, d_out=512)

    print("\n  A5: center + ABTT k=1 + rp512 + dct_K4...")
    r5 = benchmark_protein_vec("center_abtt1_rp512_dct_K4", _center_abtt_rp_dct,
                                embeddings, test_ids, metadata, cb513_data, _center_abtt_rp)
    results["part_A"]["center_abtt1_rp512_dct_K4"] = r5
    print(f"    Ret@1={r5.get('family_ret1', 'ERR')}, SS3={r5.get('ss3_q3', 'N/A')}")

    # Also test pre-processing on protein vectors directly (without RP, just mean pool)
    print("\n  A6-A8: Pre-processing on raw mean pool (no RP)...")
    for preproc_name, preproc_fn in [
        ("centered_mean", lambda m: center_embeddings(m, stats["mean_vec"]).mean(axis=0)),
        ("zscore_mean", lambda m: zscore_embeddings(m, stats["mean_vec"], stats["std_vec"]).mean(axis=0)),
        ("abtt1_mean", lambda m: all_but_the_top(m, stats["top_pcs"][:1]).mean(axis=0)),
    ]:
        r = benchmark_protein_vec(preproc_name, preproc_fn, embeddings,
                                   test_ids, metadata)
        results["part_A"][preproc_name] = r
        print(f"    {preproc_name}: Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "A")
    save_results(results)
    print("\n  Step A DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP B: Transposed Matrix View
# ══════════════════════════════════════════════════════════════════

def step_B(results):
    """Part B: Channel resampling, per-protein SVD, channel statistics."""
    print("\n" + "=" * 60)
    print("STEP B: Transposed Matrix View")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # B1: Channel resampling at various l_out
    for l_out in [32, 64, 128]:
        def _resample_vec(m, _l=l_out):
            return channel_resample(m, l_out=_l).flatten()

        name = f"channel_resample_l{l_out}"
        print(f"\n  B1: {name}...")
        r = benchmark_protein_vec(name, _resample_vec, embeddings, test_ids, metadata)
        results.setdefault("part_B", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B1b: Channel resampling with RP first (rp512, then resample)
    for l_out in [32, 64]:
        def _rp_resample_vec(m, _l=l_out):
            compressed = random_orthogonal_project(m, d_out=512)
            return channel_resample(compressed, l_out=_l).flatten()

        name = f"rp512_resample_l{l_out}"
        print(f"\n  B1b: {name}...")
        r = benchmark_protein_vec(name, _rp_resample_vec, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B2: Per-protein SVD
    for k in [1, 2, 4, 8]:
        def _svd_vec(m, _k=k):
            return per_protein_svd(m, k=_k)

        name = f"protein_svd_k{k}"
        print(f"\n  B2: {name}...")
        r = benchmark_protein_vec(name, _svd_vec, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B3: Channel statistics
    for stat_combo, stat_name in [
        (["mean", "std"], "mean_std"),
        (["mean", "std", "skew"], "mean_std_skew"),
        (["mean", "std", "min", "max"], "mean_std_min_max"),
    ]:
        def _chstat_vec(m, _stats=stat_combo):
            return channel_statistics(m, stats=_stats)

        name = f"channel_stats_{stat_name}"
        print(f"\n  B3: {name}...")
        r = benchmark_protein_vec(name, _chstat_vec, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B4: [mean|std] — the simplest transposed insight
    def _mean_std_vec(m):
        return np.concatenate([m.mean(axis=0), m.std(axis=0)]).astype(np.float32)

    print(f"\n  B4: [mean|std] concatenation...")
    r = benchmark_protein_vec("mean_std_concat", _mean_std_vec, embeddings,
                               test_ids, metadata, cb513_data)
    results["part_B"]["mean_std_concat"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "B")
    save_results(results)
    print("\n  Step B DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP C: Improved Pooling
# ══════════════════════════════════════════════════════════════════

def step_C(results):
    """Part C: Percentile pooling, trimmed mean."""
    print("\n" + "=" * 60)
    print("STEP C: Improved Pooling Strategies")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # C1: Percentile pooling variants
    for pcts, pname in [
        ([25, 50, 75], "p25_50_75"),
        ([10, 50, 90], "p10_50_90"),
        ([10, 25, 50, 75, 90], "p10_25_50_75_90"),
    ]:
        def _pct_vec(m, _p=pcts):
            return percentile_pool(m, percentiles=_p)

        name = f"percentile_{pname}"
        print(f"\n  C1: {name}...")
        r = benchmark_protein_vec(name, _pct_vec, embeddings, test_ids, metadata)
        results.setdefault("part_C", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # C1b: [mean | percentile_spread]
    def _mean_pct_spread(m):
        p25 = np.percentile(m, 25, axis=0)
        p75 = np.percentile(m, 75, axis=0)
        iqr = p75 - p25
        return np.concatenate([m.mean(axis=0), iqr]).astype(np.float32)

    print(f"\n  C1b: [mean | IQR]...")
    r = benchmark_protein_vec("mean_iqr", _mean_pct_spread, embeddings, test_ids, metadata)
    results["part_C"]["mean_iqr"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    # C2: Trimmed mean
    for prop in [0.05, 0.1, 0.2]:
        def _trim_vec(m, _p=prop):
            return trimmed_mean_pool(m, proportion=_p)

        name = f"trimmed_mean_{int(prop*100)}pct"
        print(f"\n  C2: {name}...")
        r = benchmark_protein_vec(name, _trim_vec, embeddings, test_ids, metadata)
        results["part_C"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    # C3: [mean | max | std] triple concat
    def _mean_max_std(m):
        return np.concatenate([m.mean(axis=0), m.max(axis=0), m.std(axis=0)]).astype(np.float32)

    print(f"\n  C3: [mean|max|std]...")
    r = benchmark_protein_vec("mean_max_std", _mean_max_std, embeddings, test_ids, metadata)
    results["part_C"]["mean_max_std"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    mark_done(results, "C")
    save_results(results)
    print("\n  Step C DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP D: RP Variants and Multi-Seed
# ══════════════════════════════════════════════════════════════════

def step_D(results):
    """Part D: Multi-seed RP variance, sparse RP."""
    print("\n" + "=" * 60)
    print("STEP D: RP Variants and Multi-Seed Characterization")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # D1: Multi-seed RP variance (10 seeds)
    seeds = [42, 123, 456, 789, 0, 7, 99, 2024, 31415, 271828]
    seed_results = []
    print("\n  D1: Multi-seed RP variance (10 seeds)...")
    for seed in seeds:
        def _rp_dct_seed(m, _s=seed):
            compressed = random_orthogonal_project(m, d_out=512, seed=_s)
            return dct_summary(compressed, K=4)

        def _rp_seed(m, _s=seed):
            return random_orthogonal_project(m, d_out=512, seed=_s)

        r = benchmark_protein_vec(f"rp512_dct_K4_s{seed}", _rp_dct_seed,
                                   embeddings, test_ids, metadata, cb513_data, _rp_seed)
        seed_results.append(r)
        print(f"    seed={seed}: Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

    ret1_values = [r["family_ret1"] for r in seed_results if "family_ret1" in r]
    ss3_values = [r.get("ss3_q3", 0) for r in seed_results if r.get("ss3_q3")]
    results.setdefault("part_D", {})["multi_seed"] = {
        "seeds": seeds,
        "ret1_mean": float(np.mean(ret1_values)),
        "ret1_std": float(np.std(ret1_values)),
        "ret1_min": float(np.min(ret1_values)),
        "ret1_max": float(np.max(ret1_values)),
        "ss3_mean": float(np.mean(ss3_values)) if ss3_values else None,
        "ss3_std": float(np.std(ss3_values)) if ss3_values else None,
        "per_seed": {f"s{s}": r for s, r in zip(seeds, seed_results)},
    }
    print(f"\n  Multi-seed Ret@1: {np.mean(ret1_values):.4f} +/- {np.std(ret1_values):.4f}")
    if ss3_values:
        print(f"  Multi-seed SS3:   {np.mean(ss3_values):.4f} +/- {np.std(ss3_values):.4f}")

    # D2: Sparse RP (Achlioptas)
    print("\n  D2: Sparse random projection...")
    def _sparse_rp_dct(m):
        compressed = sparse_random_project(m, d_out=512)
        return dct_summary(compressed, K=4)

    def _sparse_rp(m):
        return sparse_random_project(m, d_out=512)

    r_sparse = benchmark_protein_vec("sparse_rp512_dct_K4", _sparse_rp_dct,
                                      embeddings, test_ids, metadata, cb513_data, _sparse_rp)
    results["part_D"]["sparse_rp512_dct_K4"] = r_sparse
    print(f"    Ret@1={r_sparse.get('family_ret1', 'ERR')}, SS3={r_sparse.get('ss3_q3', 'N/A')}")

    mark_done(results, "D")
    save_results(results)
    print("\n  Step D DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP E: Quantization and Coding Combinations
# ══════════════════════════════════════════════════════════════════

def step_E(results):
    """Part E: int4 on codec output, JPEG-style DCT+quantize, predictive coding."""
    print("\n" + "=" * 60)
    print("STEP E: Quantization & Coding Combinations")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # E1: int4 quantization of codec output (rp512)
    print("\n  E1: int4 of rp512 output...")
    def _rp_int4_dct(m):
        compressed = random_orthogonal_project(m, d_out=512)
        # int4 round-trip on per-residue
        q, params = quantize_int4(compressed)
        deq = dequantize_int4(q, params)
        return dct_summary(deq, K=4)

    def _rp_int4(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q, params = quantize_int4(compressed)
        return dequantize_int4(q, params)

    r_e1 = benchmark_protein_vec("rp512_int4_dct_K4", _rp_int4_dct, embeddings,
                                  test_ids, metadata, cb513_data, _rp_int4)
    results.setdefault("part_E", {})["rp512_int4_dct_K4"] = r_e1
    print(f"    Ret@1={r_e1.get('family_ret1', 'ERR')}, SS3={r_e1.get('ss3_q3', 'N/A')}")

    # E1b: int8 of rp512 output
    print("\n  E1b: int8 of rp512 output...")
    def _rp_int8_dct(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q, params = quantize_int8(compressed)
        deq = dequantize_int8(q, params)
        return dct_summary(deq, K=4)

    def _rp_int8(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q, params = quantize_int8(compressed)
        return dequantize_int8(q, params)

    r_e1b = benchmark_protein_vec("rp512_int8_dct_K4", _rp_int8_dct, embeddings,
                                   test_ids, metadata, cb513_data, _rp_int8)
    results["part_E"]["rp512_int8_dct_K4"] = r_e1b
    print(f"    Ret@1={r_e1b.get('family_ret1', 'ERR')}, SS3={r_e1b.get('ss3_q3', 'N/A')}")

    # E2: JPEG-style pipeline (per-channel DCT → quantize DCT coefficients → measure entropy)
    print("\n  E2: JPEG-style DCT + coefficient quantization...")
    from scipy.fft import dct as scipy_dct, idct as scipy_idct

    for keep_frac in [0.25, 0.50, 0.75]:
        def _jpeg_encode(m, _frac=keep_frac):
            L, D = m.shape
            # DCT per channel along L
            Xt = m.T.astype(np.float64)  # (D, L)
            dct_coeffs = scipy_dct(Xt, type=2, norm='ortho', axis=1)  # (D, L)
            # Keep only first frac of coefficients per channel, zero the rest
            n_keep = max(1, int(L * _frac))
            dct_coeffs[:, n_keep:] = 0
            # Inverse DCT to get approximate per-residue
            reconstructed = scipy_idct(dct_coeffs, type=2, norm='ortho', axis=1)  # (D, L)
            return reconstructed.T.astype(np.float32)  # (L, D)

        def _jpeg_vec(m, _frac=keep_frac):
            recon = _jpeg_encode(m, _frac)
            return recon.mean(axis=0)

        def _jpeg_pr(m, _frac=keep_frac):
            return _jpeg_encode(m, _frac)

        name = f"jpeg_dct_keep{int(keep_frac*100)}pct"
        r = benchmark_protein_vec(name, _jpeg_vec, embeddings, test_ids, metadata,
                                   cb513_data, _jpeg_pr)
        results["part_E"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

    # E2b: JPEG with int4 quantization of DCT coefficients
    print("\n  E2b: JPEG DCT + int4 quantization of coefficients...")
    def _jpeg_int4_encode(m):
        L, D = m.shape
        Xt = m.T.astype(np.float64)
        dct_coeffs = scipy_dct(Xt, type=2, norm='ortho', axis=1)
        n_keep = max(1, int(L * 0.5))
        truncated = dct_coeffs[:, :n_keep].T.astype(np.float32)  # (n_keep, D)
        q, params = quantize_int4(truncated)
        deq = dequantize_int4(q, params)  # (n_keep, D)
        # Reconstruct via iDCT
        full_coeffs = np.zeros((D, L), dtype=np.float64)
        full_coeffs[:, :n_keep] = deq.T
        reconstructed = scipy_idct(full_coeffs, type=2, norm='ortho', axis=1)
        return reconstructed.T.astype(np.float32)

    def _jpeg_int4_vec(m):
        return _jpeg_int4_encode(m).mean(axis=0)

    r_j4 = benchmark_protein_vec("jpeg_dct50_int4", _jpeg_int4_vec, embeddings,
                                  test_ids, metadata, cb513_data, _jpeg_int4_encode)
    results["part_E"]["jpeg_dct50_int4"] = r_j4
    print(f"    Ret@1={r_j4.get('family_ret1', 'ERR')}, SS3={r_j4.get('ss3_q3', 'N/A')}")

    # E3: Predictive coding (DPCM)
    print("\n  E3: Predictive coding (order-1 DPCM)...")
    def _dpcm_encode(m):
        """Order-1 DPCM: store first residue + deltas."""
        L, D = m.shape
        deltas = np.diff(m, axis=0)  # (L-1, D)
        # Reconstruct from deltas
        recon = np.zeros_like(m)
        recon[0] = m[0]
        recon[1:] = m[0] + np.cumsum(deltas, axis=0)
        return recon

    def _dpcm_vec(m):
        return _dpcm_encode(m).mean(axis=0)

    r_dpcm = benchmark_protein_vec("dpcm_order1", _dpcm_vec, embeddings,
                                    test_ids, metadata, cb513_data, _dpcm_encode)
    results["part_E"]["dpcm_order1"] = r_dpcm
    print(f"    Ret@1={r_dpcm.get('family_ret1', 'ERR')}, SS3={r_dpcm.get('ss3_q3', 'N/A')}")

    # E3b: DPCM + int4 on deltas
    def _dpcm_int4_encode(m):
        L, D = m.shape
        deltas = np.diff(m, axis=0)  # (L-1, D)
        q, params = quantize_int4(deltas)
        deq = dequantize_int4(q, params)
        recon = np.zeros_like(m)
        recon[0] = m[0]
        recon[1:] = m[0] + np.cumsum(deq, axis=0)
        return recon

    def _dpcm_int4_vec(m):
        return _dpcm_int4_encode(m).mean(axis=0)

    r_dpcm4 = benchmark_protein_vec("dpcm_int4", _dpcm_int4_vec, embeddings,
                                     test_ids, metadata, cb513_data, _dpcm_int4_encode)
    results["part_E"]["dpcm_int4"] = r_dpcm4
    print(f"    Ret@1={r_dpcm4.get('family_ret1', 'ERR')}, SS3={r_dpcm4.get('ss3_q3', 'N/A')}")

    # E4: Delta variance measurement (how compressible are deltas?)
    print("\n  E4: Delta compressibility analysis...")
    raw_vars = []
    delta_vars = []
    for pid in list(test_ids)[:200]:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        deltas = np.diff(m, axis=0)
        raw_vars.append(m.var())
        delta_vars.append(deltas.var())
    results["part_E"]["delta_analysis"] = {
        "mean_raw_var": float(np.mean(raw_vars)),
        "mean_delta_var": float(np.mean(delta_vars)),
        "variance_reduction": float(1 - np.mean(delta_vars) / np.mean(raw_vars)),
    }
    print(f"    Raw variance: {np.mean(raw_vars):.4f}")
    print(f"    Delta variance: {np.mean(delta_vars):.4f}")
    print(f"    Variance reduction: {1 - np.mean(delta_vars)/np.mean(raw_vars):.1%}")

    mark_done(results, "E")
    save_results(results)
    print("\n  Step E DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP G: Evaluation Enhancements
# ══════════════════════════════════════════════════════════════════

def step_G(results):
    """Part G: RNS, remote homology analysis, MLP probes, Matryoshka."""
    print("\n" + "=" * 60)
    print("STEP G: Evaluation Enhancements")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # G1: RNS for raw vs codec
    print("\n  G1: Random Neighbor Score (RNS)...")
    # Raw mean pool
    raw_vecs = {pid: embeddings[pid].mean(axis=0) for pid in test_ids if pid in embeddings}
    rns_raw = compute_rns(raw_vecs, metadata, k=10, label_key="family")
    print(f"    Raw mean pool RNS: {rns_raw['mean_rns']:.4f} (high RNS={rns_raw['n_high_rns']})")

    # Codec (rp512 + dct_K4)
    codec_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        compressed = random_orthogonal_project(m, d_out=512)
        codec_vecs[pid] = dct_summary(compressed, K=4)
    rns_codec = compute_rns(codec_vecs, metadata, k=10, label_key="family")
    print(f"    Codec RNS: {rns_codec['mean_rns']:.4f} (high RNS={rns_codec['n_high_rns']})")

    results.setdefault("part_G", {})["rns"] = {
        "raw_mean_pool": {k: v for k, v in rns_raw.items() if k != "per_protein_rns"},
        "codec_rp512_dct_K4": {k: v for k, v in rns_codec.items() if k != "per_protein_rns"},
    }

    # G2: Remote homology analysis (superfamily and fold level retrieval)
    print("\n  G2: Remote homology (superfamily + fold level)...")
    for level in ["superfamily", "fold"]:
        id_to_label = {m["protein_id"]: m.get(level, "") for m in metadata}
        # Raw
        ret_raw = evaluate_retrieval_from_vectors(
            raw_vecs, metadata, label_key=level,
            query_ids=test_ids, database_ids=test_ids, metric="cosine",
        )
        # Codec
        ret_codec = evaluate_retrieval_from_vectors(
            codec_vecs, metadata, label_key=level,
            query_ids=test_ids, database_ids=test_ids, metric="cosine",
        )
        results["part_G"][f"remote_homology_{level}"] = {
            "raw_ret1": ret_raw["precision@1"],
            "codec_ret1": ret_codec["precision@1"],
            "raw_mrr": ret_raw["mrr"],
            "codec_mrr": ret_codec["mrr"],
        }
        print(f"    {level}: raw Ret@1={ret_raw['precision@1']:.4f}, codec Ret@1={ret_codec['precision@1']:.4f}")

    # G3: MLP probe on SS3 (2-layer, hidden=256)
    print("\n  G3: MLP probes on SS3 (raw vs codec)...")
    if cb513_data is not None:
        from sklearn.neural_network import MLPClassifier

        avail = [pid for pid in cb513_data["ss3_labels"] if pid in cb513_data["embeddings"]]
        rng = random.Random(42)
        rng.shuffle(avail)
        n_train = int(len(avail) * 0.8)
        train_ids_cb, test_ids_cb = avail[:n_train], avail[n_train:]

        # Collect raw per-residue features + labels
        X_train_raw, y_train = [], []
        X_test_raw, y_test = [], []
        X_train_codec, X_test_codec = [], []
        ss3_map = {"H": 0, "E": 1, "C": 2}

        for pid in train_ids_cb:
            emb = cb513_data["embeddings"][pid].astype(np.float32)
            labels = cb513_data["ss3_labels"][pid]
            min_len = min(len(emb), len(labels))
            X_train_raw.append(emb[:min_len])
            coded = random_orthogonal_project(emb, d_out=512)
            X_train_codec.append(coded[:min_len])
            y_train.extend([ss3_map.get(l, 2) for l in labels[:min_len]])

        for pid in test_ids_cb:
            emb = cb513_data["embeddings"][pid].astype(np.float32)
            labels = cb513_data["ss3_labels"][pid]
            min_len = min(len(emb), len(labels))
            X_test_raw.append(emb[:min_len])
            coded = random_orthogonal_project(emb, d_out=512)
            X_test_codec.append(coded[:min_len])
            y_test.extend([ss3_map.get(l, 2) for l in labels[:min_len]])

        X_train_raw = np.vstack(X_train_raw)
        X_test_raw = np.vstack(X_test_raw)
        X_train_codec = np.vstack(X_train_codec)
        X_test_codec = np.vstack(X_test_codec)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # LogisticRegression baseline (existing)
        from sklearn.linear_model import LogisticRegression
        lr_raw = LogisticRegression(max_iter=500, random_state=42).fit(X_train_raw, y_train)
        lr_codec = LogisticRegression(max_iter=500, random_state=42).fit(X_train_codec, y_train)
        lr_raw_q3 = float((lr_raw.predict(X_test_raw) == y_test).mean())
        lr_codec_q3 = float((lr_codec.predict(X_test_codec) == y_test).mean())

        # MLP probe
        mlp_raw = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                 random_state=42, early_stopping=True)
        mlp_raw.fit(X_train_raw, y_train)
        mlp_raw_q3 = float((mlp_raw.predict(X_test_raw) == y_test).mean())

        mlp_codec = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                   random_state=42, early_stopping=True)
        mlp_codec.fit(X_train_codec, y_train)
        mlp_codec_q3 = float((mlp_codec.predict(X_test_codec) == y_test).mean())

        results["part_G"]["mlp_probes"] = {
            "lr_raw_q3": lr_raw_q3,
            "lr_codec_q3": lr_codec_q3,
            "mlp_raw_q3": mlp_raw_q3,
            "mlp_codec_q3": mlp_codec_q3,
            "lr_gap": lr_raw_q3 - lr_codec_q3,
            "mlp_gap": mlp_raw_q3 - mlp_codec_q3,
        }
        print(f"    LR probe:  raw Q3={lr_raw_q3:.4f}, codec Q3={lr_codec_q3:.4f}, gap={lr_raw_q3 - lr_codec_q3:.4f}")
        print(f"    MLP probe: raw Q3={mlp_raw_q3:.4f}, codec Q3={mlp_codec_q3:.4f}, gap={mlp_raw_q3 - mlp_codec_q3:.4f}")

    # G4: Matryoshka dimension ordering
    print("\n  G4: Matryoshka dimension ordering...")
    # Compute per-dimension variance in projected space
    train_embs = {pid: embeddings[pid] for pid in split["train"] if pid in embeddings}
    all_projected = []
    for pid, emb in list(train_embs.items())[:500]:
        projected = random_orthogonal_project(emb[:MAX_LEN], d_out=512)
        all_projected.append(projected.mean(axis=0))  # protein-level mean
    all_projected = np.array(all_projected)
    dim_variance = all_projected.var(axis=0)  # (512,)
    sorted_dims = np.argsort(dim_variance)[::-1]  # high variance first

    # Test retrieval at truncated dimensions
    for n_dims in [128, 256, 384, 512]:
        keep = sorted_dims[:n_dims]

        def _matryoshka_vec(m, _keep=keep):
            compressed = random_orthogonal_project(m, d_out=512)
            protein_vec = dct_summary(compressed, K=4)
            # Select and reorder to keep high-variance dims
            # For retrieval: use sorted protein_vec
            return protein_vec[np.tile(_keep, 4)]  # repeat for K=4 DCT

        # Simpler: just use mean pool on sorted RP
        def _matryoshka_mean(m, _keep=keep):
            compressed = random_orthogonal_project(m, d_out=512)
            mean_vec = compressed.mean(axis=0)
            return mean_vec[_keep]

        name = f"matryoshka_mean_d{n_dims}"
        r = benchmark_protein_vec(name, _matryoshka_mean, embeddings, test_ids, metadata)
        results["part_G"].setdefault("matryoshka", {})[name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}")

    # Compare with random dim selection
    rng = np.random.RandomState(42)
    random_dims = rng.permutation(512)
    for n_dims in [128, 256]:
        keep = random_dims[:n_dims]
        def _random_subset_mean(m, _keep=keep):
            compressed = random_orthogonal_project(m, d_out=512)
            return compressed.mean(axis=0)[_keep]

        name = f"random_subset_mean_d{n_dims}"
        r = benchmark_protein_vec(name, _random_subset_mean, embeddings, test_ids, metadata)
        results["part_G"]["matryoshka"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "G")
    save_results(results)
    print("\n  Step G DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP H: Reference Corpus Approaches
# ══════════════════════════════════════════════════════════════════

def step_H(results):
    """Part H: k-means centroid residual coding, PCA as D-compression."""
    print("\n" + "=" * 60)
    print("STEP H: Reference Corpus Approaches")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    train_ids = split["train"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # H1: k-means centroid residual coding
    from sklearn.cluster import MiniBatchKMeans

    # Fit centroids on training residues
    train_residues = np.vstack([
        embeddings[pid].astype(np.float32)[:MAX_LEN]
        for pid in train_ids if pid in embeddings
    ])
    print(f"  Fitting k-means on {len(train_residues)} training residues...")

    for n_clusters in [64, 256]:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 batch_size=10000, n_init=3)
        kmeans.fit(train_residues)
        centroids = kmeans.cluster_centers_  # (k, D)

        def _kmeans_residual_vec(m, _km=kmeans, _c=centroids):
            labels = _km.predict(m)
            residuals = m - _c[labels]
            # Protein vector: mean of residuals + mean of centroid assignments
            centroid_mean = _c[labels].mean(axis=0)
            residual_mean = residuals.mean(axis=0)
            return np.concatenate([centroid_mean, residual_mean]).astype(np.float32)

        def _kmeans_residual_pr(m, _km=kmeans, _c=centroids):
            labels = _km.predict(m)
            residuals = m - _c[labels]
            return residuals

        name = f"kmeans_residual_k{n_clusters}"
        print(f"\n  H1: {name}...")
        r = benchmark_protein_vec(name, _kmeans_residual_vec, embeddings,
                                   test_ids, metadata, cb513_data, _kmeans_residual_pr)
        results.setdefault("part_H", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

    # H2: PCA as D-compression (512 components)
    from sklearn.decomposition import PCA

    print("\n  H2: PCA as standalone D-compression...")
    # Subsample training residues for PCA fit
    if len(train_residues) > 50_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_residues), 50_000, replace=False)
        pca_train = train_residues[idx]
    else:
        pca_train = train_residues

    for n_comp in [256, 512]:
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(pca_train)
        print(f"    PCA d={n_comp}: variance explained = {pca.explained_variance_ratio_.sum():.4f}")

        def _pca_dct(m, _pca=pca):
            compressed = _pca.transform(m).astype(np.float32)
            return dct_summary(compressed, K=4)

        def _pca_pr(m, _pca=pca):
            return _pca.transform(m).astype(np.float32)

        name = f"pca{n_comp}_dct_K4"
        r = benchmark_protein_vec(name, _pca_dct, embeddings, test_ids, metadata,
                                   cb513_data, _pca_pr)
        results["part_H"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

        # PCA mean pool (no DCT)
        def _pca_mean(m, _pca=pca):
            return _pca.transform(m).astype(np.float32).mean(axis=0)

        name2 = f"pca{n_comp}_mean"
        r2 = benchmark_protein_vec(name2, _pca_mean, embeddings, test_ids, metadata)
        results["part_H"][name2] = r2
        print(f"    {name2}: Ret@1={r2.get('family_ret1', 'ERR')}")

    # Clean up large array
    del train_residues, pca_train

    mark_done(results, "H")
    save_results(results)
    print("\n  Step H DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP I: Multi-Resolution Framing
# ══════════════════════════════════════════════════════════════════

def step_I(results):
    """Part I: Three-level retrieval system (SimHash → protein_vec → late interaction)."""
    print("\n" + "=" * 60)
    print("STEP I: Multi-Resolution Retrieval Framing")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    from src.one_embedding.topological import simhash_encode

    # Level 1: SimHash 1024-bit (Hamming distance)
    print("\n  Level 1: SimHash 1024-bit...")
    sh_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        sh_vecs[pid] = simhash_encode(m, n_bits=1024).astype(np.float32)

    ret_sh = eval_retrieval(sh_vecs, metadata, test_ids, metric="hamming")
    print(f"    SimHash Ret@1={ret_sh['precision@1']:.4f}")

    # Level 2: protein_vec 2048d (codec)
    print("\n  Level 2: Codec protein_vec 2048d...")
    codec_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        compressed = random_orthogonal_project(m, d_out=512)
        codec_vecs[pid] = dct_summary(compressed, K=4)

    ret_codec = eval_retrieval(codec_vecs, metadata, test_ids)
    print(f"    Codec Ret@1={ret_codec['precision@1']:.4f}")

    # Level 3: Late interaction on rp512 per-residue
    print("\n  Level 3: Late interaction on rp512...")
    from src.evaluation.late_interaction import evaluate_late_interaction
    rp_embs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        rp_embs[pid] = random_orthogonal_project(
            embeddings[pid].astype(np.float32)[:MAX_LEN], d_out=512
        )

    try:
        ret_late = evaluate_late_interaction(rp_embs, metadata, test_ids)
        print(f"    Late interaction Ret@1={ret_late.get('precision@1', 'ERR')}")
    except Exception as e:
        ret_late = {"precision@1": -1, "error": str(e)}
        print(f"    Late interaction failed: {e}")

    # Cascaded: SimHash top-100 → codec re-rank
    print("\n  Cascaded: SimHash top-100 → codec re-rank...")
    from scipy.spatial.distance import cdist as scipy_cdist
    pids_list = [pid for pid in test_ids if pid in sh_vecs and pid in codec_vecs]
    sh_matrix = np.array([sh_vecs[pid] for pid in pids_list])
    codec_matrix = np.array([codec_vecs[pid] for pid in pids_list])
    id_to_family = {m["protein_id"]: m.get("family", "") for m in metadata}

    # SimHash distance matrix
    sh_dists = scipy_cdist(sh_matrix, sh_matrix, metric="hamming")

    correct_cascade = 0
    total = 0
    for i, pid in enumerate(pids_list):
        if pid not in id_to_family or not id_to_family[pid]:
            continue
        # Top 100 by SimHash
        sh_order = np.argsort(sh_dists[i])
        top100 = [j for j in sh_order[1:101]]  # exclude self
        # Re-rank by codec cosine similarity
        query_codec = codec_vecs[pid]
        candidate_vecs = np.array([codec_matrix[j] for j in top100])
        cos_sims = candidate_vecs @ query_codec / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_codec) + 1e-10
        )
        best_j = top100[np.argmax(cos_sims)]
        best_pid = pids_list[best_j]
        if id_to_family.get(best_pid) == id_to_family[pid]:
            correct_cascade += 1
        total += 1

    cascade_ret1 = correct_cascade / total if total > 0 else 0
    print(f"    Cascade Ret@1={cascade_ret1:.4f}")

    results.setdefault("part_I", {})["multi_resolution"] = {
        "level1_simhash_ret1": ret_sh["precision@1"],
        "level2_codec_ret1": ret_codec["precision@1"],
        "level3_late_ret1": ret_late.get("precision@1", -1),
        "cascade_sh100_codec_ret1": cascade_ret1,
    }

    mark_done(results, "I")
    save_results(results)
    print("\n  Step I DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

ALL_STEPS = {
    "F": step_F,
    "A": step_A,
    "B": step_B,
    "C": step_C,
    "D": step_D,
    "E": step_E,
    "G": step_G,
    "H": step_H,
    "I": step_I,
}

STEP_ORDER = ["F", "A", "B", "C", "D", "E", "G", "H", "I"]


def main():
    parser = argparse.ArgumentParser(description="Experiment 29: Exhaustive Fruit Sweep")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step (F/A/B/C/D/E/G/H/I). Default: all.")
    args = parser.parse_args()

    results = load_results()

    if args.step:
        steps = [s.strip() for s in args.step.split(",")]
    else:
        steps = STEP_ORDER

    for step in steps:
        if step in results.get("steps_done", []):
            print(f"\n  Step {step} already done, skipping. (Delete from steps_done to re-run)")
            continue
        if step not in ALL_STEPS:
            print(f"\n  Unknown step: {step}")
            continue
        ALL_STEPS[step](results)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 29 SUMMARY")
    print("=" * 60)
    print(f"Steps completed: {results.get('steps_done', [])}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit experiment script**

```bash
git add experiments/29_exhaustive_fruit_sweep.py
git commit -m "feat: add Experiment 29 exhaustive fruit sweep script"
```

---

## Chunk 3: Execution

### Task 7: Run Part F (Data Characterization)

- [ ] **Step 1: Run**

```bash
cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations
uv run python experiments/29_exhaustive_fruit_sweep.py --step F
```

- [ ] **Step 2: Review results and record intrinsic dim finding**

Check `data/benchmarks/exhaustive_sweep_results.json` for dims_95pct.
This number informs whether rp512 is conservative or aggressive.

### Task 8: Run Part A (Pre-Processing)

- [ ] **Step 1: Run**

```bash
uv run python experiments/29_exhaustive_fruit_sweep.py --step A
```

- [ ] **Step 2: Review centering/ABTT results**

Compare centered_rp512_dct_K4 vs raw_rp512_dct_K4. If centering helps >0.005, it's a keeper.

### Task 9: Run Part B (Transposed Matrix View)

- [ ] **Step 1: Run**

```bash
uv run python experiments/29_exhaustive_fruit_sweep.py --step B
```

### Task 10: Run Parts C, D, E, G, H, I

Each step is run sequentially (one GPU-intensive job at a time per CLAUDE.md thermal rules):

- [ ] **Step 1: Run remaining parts**

```bash
uv run python experiments/29_exhaustive_fruit_sweep.py --step C
uv run python experiments/29_exhaustive_fruit_sweep.py --step D
uv run python experiments/29_exhaustive_fruit_sweep.py --step E
uv run python experiments/29_exhaustive_fruit_sweep.py --step G
uv run python experiments/29_exhaustive_fruit_sweep.py --step H
uv run python experiments/29_exhaustive_fruit_sweep.py --step I
```

### Task 11: Final Commit with Results

- [ ] **Step 1: Commit results**

```bash
git add data/benchmarks/exhaustive_sweep_results.json
git commit -m "feat: complete Experiment 29 exhaustive sweep results"
```

### Task 12: Update README with Key Findings

- [ ] **Step 1: Add Experiment 29 section to README**

Add results summary table and key findings to README.md.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Experiment 29 exhaustive sweep results to README"
```
