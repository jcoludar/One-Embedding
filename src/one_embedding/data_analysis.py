"""Corpus-level embedding analysis utilities.

Functions for characterizing embedding distributions across a corpus of
proteins: intrinsic dimensionality, per-channel statistics, and
inter/intra-protein variance decomposition.
"""

from typing import Optional

import numpy as np
from scipy.stats import kurtosis, skew


def intrinsic_dimensionality(
    embeddings: dict[str, np.ndarray],
    n_sample: int = 50000,
    seed: int = 42,
) -> dict:
    """Estimate intrinsic dimensionality of per-residue embeddings via SVD.

    Stacks all residues from all proteins, subsamples to n_sample rows,
    centers, and computes SVD to analyze the spectrum.

    Args:
        embeddings: Mapping from protein ID to (L_i, D) per-residue embeddings.
        n_sample: Maximum number of residues to subsample.
        seed: Random seed for reproducible subsampling.

    Returns:
        Dict with:
            participation_ratio: (sum s_i^2)^2 / sum s_i^4 — effective dimensionality
            dims_90pct: Number of dimensions capturing 90% variance
            dims_95pct: Number of dimensions capturing 95% variance
            dims_99pct: Number of dimensions capturing 99% variance
            total_dims: Total number of dimensions (min of n_sample, D)
            n_samples: Actual number of residues used
            singular_values: List of singular values
            cumulative_variance: List of cumulative variance fractions
    """
    # Stack all residues
    all_residues = np.vstack(list(embeddings.values()))  # (N_total, D)
    N_total, D = all_residues.shape

    # Subsample if needed
    rng = np.random.RandomState(seed)
    if N_total > n_sample:
        idx = rng.choice(N_total, size=n_sample, replace=False)
        all_residues = all_residues[idx]
    n_used = all_residues.shape[0]

    # Center
    all_residues = all_residues - all_residues.mean(axis=0)

    # SVD (on CPU, float32 is fine)
    all_residues = all_residues.astype(np.float32)
    _, S, _ = np.linalg.svd(all_residues, full_matrices=False)

    # Participation ratio = (sum s_i^2)^2 / sum s_i^4
    s2 = S ** 2
    participation_ratio = float((s2.sum()) ** 2 / (s2 ** 2).sum())

    # Cumulative variance
    variance = s2 / s2.sum()
    cumulative = np.cumsum(variance)

    # Threshold dimensions
    dims_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    dims_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    dims_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    return {
        "participation_ratio": participation_ratio,
        "dims_90pct": dims_90,
        "dims_95pct": dims_95,
        "dims_99pct": dims_99,
        "total_dims": int(len(S)),
        "n_samples": int(n_used),
        "singular_values": S.tolist(),
        "cumulative_variance": cumulative.tolist(),
    }


def channel_distributions(
    embeddings: dict[str, np.ndarray],
    n_sample: int = 50000,
    seed: int = 42,
) -> dict:
    """Analyze per-channel statistics and inter/intra-protein variance.

    Args:
        embeddings: Mapping from protein ID to (L_i, D) per-residue embeddings.
        n_sample: Maximum residues for correlation analysis.
        seed: Random seed for reproducible subsampling.

    Returns:
        Dict with:
            channel_mean: (D,) per-channel means across all residues
            channel_std: (D,) per-channel standard deviations
            channel_skewness: (D,) per-channel skewness
            channel_kurtosis: (D,) per-channel excess kurtosis
            inter_protein_variance: (D,) variance of per-protein channel means
            intra_protein_variance: (D,) mean of within-protein channel variances
            discriminative_ratio: (D,) inter/intra per channel
            mean_disc_ratio: scalar mean of discriminative_ratio
            top_10_discriminative_channels: indices of 10 highest ratio channels
            bottom_10_channels: indices of 10 lowest ratio channels
            mean_abs_correlation: mean |r| across all channel pairs
            n_highly_correlated_pairs: number of pairs with |r| > 0.8
    """
    # Stack all residues for global stats
    all_residues = np.vstack(list(embeddings.values()))  # (N_total, D)
    N_total, D = all_residues.shape

    # Per-channel global statistics
    ch_mean = all_residues.mean(axis=0).astype(np.float64)
    ch_std = all_residues.std(axis=0).astype(np.float64)
    ch_skew = skew(all_residues, axis=0).astype(np.float64)
    ch_kurt = kurtosis(all_residues, axis=0).astype(np.float64)

    # Inter-protein variance: variance of per-protein channel means
    protein_means = np.array(
        [emb.mean(axis=0) for emb in embeddings.values()], dtype=np.float64
    )  # (N_proteins, D)
    inter_var = protein_means.var(axis=0)  # (D,)

    # Intra-protein variance: mean of within-protein channel variances
    protein_vars = []
    for emb in embeddings.values():
        if emb.shape[0] > 1:
            protein_vars.append(emb.var(axis=0).astype(np.float64))
        else:
            protein_vars.append(np.zeros(D, dtype=np.float64))
    intra_var = np.mean(protein_vars, axis=0)  # (D,)

    # Discriminative ratio = inter / intra (avoid div by zero)
    disc_ratio = inter_var / np.clip(intra_var, 1e-10, None)

    # Top/bottom channels by discriminative ratio
    sorted_channels = np.argsort(disc_ratio)
    top_10 = sorted_channels[-10:][::-1]
    bottom_10 = sorted_channels[:10]

    # Correlation analysis (subsample for memory efficiency)
    rng = np.random.RandomState(seed)
    if N_total > n_sample:
        idx = rng.choice(N_total, size=n_sample, replace=False)
        sample = all_residues[idx]
    else:
        sample = all_residues

    corr = np.corrcoef(sample.T)  # (D, D)
    # Upper triangle only (exclude diagonal)
    upper_idx = np.triu_indices(D, k=1)
    abs_corr = np.abs(corr[upper_idx])
    mean_abs_corr = float(np.mean(abs_corr))
    n_high_corr = int(np.sum(abs_corr > 0.8))

    return {
        "channel_mean": ch_mean.tolist(),
        "channel_std": ch_std.tolist(),
        "channel_skewness": ch_skew.tolist(),
        "channel_kurtosis": ch_kurt.tolist(),
        "inter_protein_variance": inter_var.tolist(),
        "intra_protein_variance": intra_var.tolist(),
        "discriminative_ratio": disc_ratio.tolist(),
        "mean_disc_ratio": float(np.mean(disc_ratio)),
        "top_10_discriminative_channels": top_10.tolist(),
        "bottom_10_channels": bottom_10.tolist(),
        "mean_abs_correlation": mean_abs_corr,
        "n_highly_correlated_pairs": n_high_corr,
    }
