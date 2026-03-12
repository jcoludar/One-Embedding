"""Enriched pooling transforms for protein-level embeddings.

Instead of just mean pooling (L, d) → (d,), these compute richer statistics
that capture sequence order, variance, and multi-scale structure. Each returns
a raw feature vector that should be PCA-reduced to a target dimensionality.

The EnrichedTransformPipeline wraps a transform + fitted PCA for end-to-end use.
"""

import numpy as np
from sklearn.decomposition import PCA

from src.one_embedding.transforms import dct_summary, haar_summary


# ---------------------------------------------------------------------------
# Transform A: Moment Pooling
# ---------------------------------------------------------------------------

def moment_pool(matrix: np.ndarray) -> np.ndarray:
    """Five statistical moments of the residue embedding distribution.

    Concatenates [mean | std | skewness | lag-1 autocovariance | half_diff].
    The lag-1 autocovariance captures sequence order — it is NOT commutative.

    Args:
        matrix: Per-residue embeddings, shape (L, d).

    Returns:
        Feature vector of shape (5 * d,).
    """
    L, d = matrix.shape
    mu = matrix.mean(axis=0)              # (d,)
    std = matrix.std(axis=0).clip(1e-12)  # (d,)

    # Skewness per dimension
    centered = matrix - mu
    skew = (centered ** 3).mean(axis=0) / (std ** 3)

    # Lag-1 autocovariance: mean((x[t] - mu) * (x[t+1] - mu))
    if L > 1:
        autocov = (centered[:-1] * centered[1:]).mean(axis=0)
    else:
        autocov = np.zeros(d, dtype=np.float32)

    # N-to-C gradient: first half mean minus second half mean
    mid = L // 2
    if mid > 0:
        half_diff = matrix[:mid].mean(axis=0) - matrix[mid:].mean(axis=0)
    else:
        half_diff = np.zeros(d, dtype=np.float32)

    return np.concatenate([mu, std, skew, autocov, half_diff]).astype(np.float32)


# ---------------------------------------------------------------------------
# Transform B: Multi-Scale Autocovariance
# ---------------------------------------------------------------------------

def autocovariance_pool(
    matrix: np.ndarray,
    lags: list[int] | None = None,
) -> np.ndarray:
    """Mean + autocovariance at multiple spatial lags.

    Lag 1: local transitions. Lag 4: alpha helix periodicity (~3.6 residues).
    Lag 8: domain-scale structure.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        lags: Lag values to compute. Default [1, 2, 4, 8].

    Returns:
        Feature vector of shape ((1 + len(lags)) * d,).
    """
    if lags is None:
        lags = [1, 2, 4, 8]

    L, d = matrix.shape
    mu = matrix.mean(axis=0)
    centered = matrix - mu

    parts = [mu]
    for lag in lags:
        if L > lag:
            acov = (centered[:-lag] * centered[lag:]).mean(axis=0)
        else:
            acov = np.zeros(d, dtype=np.float32)
        parts.append(acov)

    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Transform C: Gram Matrix Features
# ---------------------------------------------------------------------------

def gram_features(
    matrix: np.ndarray,
    top_k: int = 32,
    n_bins: int = 16,
) -> np.ndarray:
    """Features from the residue self-similarity (Gram) matrix.

    Extracts eigenvalue spectrum, summary statistics, and similarity histogram
    from G = M @ M.T. Captures internal symmetry and repeat structure.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        top_k: Number of top eigenvalues to keep.
        n_bins: Number of bins for similarity histogram.

    Returns:
        Feature vector of shape (d + top_k + 3 + n_bins,).
    """
    L, d = matrix.shape
    mu = matrix.mean(axis=0)

    # Gram matrix: pairwise similarities
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
    normed = matrix / norms
    gram = normed @ normed.T  # (L, L) cosine similarity matrix

    # Top-k eigenvalues (sorted descending)
    eigvals = np.linalg.eigvalsh(gram)[::-1]  # sorted ascending, reverse
    k = min(top_k, len(eigvals))
    top_eigs = np.zeros(top_k, dtype=np.float32)
    top_eigs[:k] = eigvals[:k]

    # Summary stats
    trace = float(eigvals.sum())
    # Effective rank: exp(entropy of normalized eigenvalues)
    eigvals_pos = eigvals[eigvals > 1e-12]
    if len(eigvals_pos) > 0:
        p = eigvals_pos / eigvals_pos.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        eff_rank = 0.0
    # Log-determinant (sum of log positive eigenvalues)
    logdet = float(np.sum(np.log(eigvals_pos))) if len(eigvals_pos) > 0 else 0.0
    stats = np.array([trace, logdet, eff_rank], dtype=np.float32)

    # Histogram of off-diagonal similarities
    triu_idx = np.triu_indices(L, k=1)
    off_diag = gram[triu_idx]
    if len(off_diag) > 0:
        hist, _ = np.histogram(off_diag, bins=n_bins, range=(-1.0, 1.0))
        hist = hist.astype(np.float32) / len(off_diag)  # normalize
    else:
        hist = np.zeros(n_bins, dtype=np.float32)

    return np.concatenate([mu, top_eigs, stats, hist]).astype(np.float32)


# ---------------------------------------------------------------------------
# Transform D: DCT + PCA (reuse existing DCT)
# ---------------------------------------------------------------------------

def dct_pool(matrix: np.ndarray, K: int = 8) -> np.ndarray:
    """DCT coefficients — wrapper for PCA reduction pipeline.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        K: Number of DCT frequency coefficients.

    Returns:
        Feature vector of shape (K * d,).
    """
    return dct_summary(matrix, K=K)


# ---------------------------------------------------------------------------
# Transform E: Haar + PCA (reuse existing Haar)
# ---------------------------------------------------------------------------

def haar_pool(matrix: np.ndarray, levels: int = 3) -> np.ndarray:
    """Haar wavelet summary — wrapper for PCA reduction pipeline.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        levels: Number of wavelet decomposition levels.

    Returns:
        Feature vector of shape ((levels + 1) * d,).
    """
    return haar_summary(matrix, levels=levels)


# ---------------------------------------------------------------------------
# Transform F: Fisher Vector Encoding
# ---------------------------------------------------------------------------

def fisher_vector(
    matrix: np.ndarray,
    gmm_means: np.ndarray,
    gmm_covars: np.ndarray,
    gmm_weights: np.ndarray,
) -> np.ndarray:
    """Fisher Vector encoding of residue embeddings against a pre-fitted GMM.

    Each protein is encoded as the gradient of the log-likelihood of its
    residues under the GMM. Captures first-order (mean) and second-order
    (variance) deviations from each cluster center.

    Args:
        matrix: Per-residue embeddings, shape (L, d).
        gmm_means: GMM component means, shape (k, d).
        gmm_covars: GMM component diagonal variances, shape (k, d).
        gmm_weights: GMM component weights, shape (k,).

    Returns:
        Fisher vector of shape (2 * k * d,).
    """
    L, d = matrix.shape
    k = gmm_means.shape[0]

    # Compute responsibilities: gamma[t, c] = p(c | x_t)
    # Using log-space for numerical stability
    log_probs = np.zeros((L, k), dtype=np.float64)
    for c in range(k):
        diff = matrix - gmm_means[c]  # (L, d)
        var = gmm_covars[c].clip(1e-12)  # (d,)
        log_probs[:, c] = (
            np.log(gmm_weights[c] + 1e-30)
            - 0.5 * np.sum(np.log(var))
            - 0.5 * np.sum(diff ** 2 / var, axis=1)
        )

    # Softmax for responsibilities
    log_probs -= log_probs.max(axis=1, keepdims=True)
    gamma = np.exp(log_probs)
    gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-30)  # (L, k)

    # First-order (mean deviation) and second-order (variance deviation)
    fv_parts = []
    for c in range(k):
        wc = gmm_weights[c]
        gc = gamma[:, c]  # (L,)
        var = gmm_covars[c].clip(1e-12)  # (d,)
        sqrt_var = np.sqrt(var)

        diff = matrix - gmm_means[c]  # (L, d)

        # First-order: mean deviation weighted by responsibilities
        u_c = (gc[:, None] * diff / sqrt_var).sum(axis=0) / (L * np.sqrt(wc) + 1e-30)
        fv_parts.append(u_c)

        # Second-order: variance deviation
        v_c = (gc[:, None] * (diff ** 2 / var - 1.0)).sum(axis=0) / (
            L * np.sqrt(2 * wc) + 1e-30
        )
        fv_parts.append(v_c)

    fv = np.concatenate(fv_parts)

    # Power normalization: sign(x) * sqrt(|x|)
    fv = np.sign(fv) * np.sqrt(np.abs(fv))

    # L2 normalization
    norm = np.linalg.norm(fv).clip(1e-12)
    fv = fv / norm

    return fv.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline: Transform + PCA
# ---------------------------------------------------------------------------

class EnrichedTransformPipeline:
    """Wraps a transform function with a fitted PCA for dimensionality control.

    Usage:
        pipe = EnrichedTransformPipeline(moment_pool)
        pipe.fit(train_matrices, target_dim=256)
        vector = pipe.transform(matrix)  # (target_dim,)
    """

    def __init__(self, transform_fn, transform_kwargs=None):
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs or {}
        self.pca = None
        self.raw_dim = None

    def fit(self, matrices: dict[str, np.ndarray], target_dim: int = 256):
        """Fit PCA on raw features from training proteins.

        Args:
            matrices: {protein_id: (L, d) matrix} for training set.
            target_dim: Target output dimensionality.
        """
        raw = np.array([
            self.transform_fn(m, **self.transform_kwargs)
            for m in matrices.values()
        ])
        self.raw_dim = raw.shape[1]
        actual_dim = min(target_dim, self.raw_dim, raw.shape[0])
        self.pca = PCA(n_components=actual_dim, random_state=42)
        self.pca.fit(raw)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply transform + PCA to a single protein matrix.

        Args:
            matrix: Per-residue embeddings, shape (L, d).

        Returns:
            Reduced feature vector of shape (target_dim,).
        """
        raw = self.transform_fn(matrix, **self.transform_kwargs).reshape(1, -1)
        return self.pca.transform(raw)[0].astype(np.float32)

    def transform_batch(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply transform + PCA to a dict of protein matrices.

        Args:
            matrices: {protein_id: (L, d) matrix}.

        Returns:
            {protein_id: reduced feature vector}.
        """
        raw = np.array([
            self.transform_fn(m, **self.transform_kwargs)
            for m in matrices.values()
        ])
        reduced = self.pca.transform(raw).astype(np.float32)
        return dict(zip(matrices.keys(), reduced))

    @property
    def variance_explained(self) -> np.ndarray:
        """Cumulative variance explained by PCA components."""
        if self.pca is None:
            return np.array([])
        return np.cumsum(self.pca.explained_variance_ratio_)
