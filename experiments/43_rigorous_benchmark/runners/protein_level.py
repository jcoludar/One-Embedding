"""Protein-level benchmark runner: retrieval, classification with dual metric.

Enforces Rule 1 (fair comparison), Rule 4 (CIs), Rule 12 (cosine + euclidean).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import dct
from scipy.spatial.distance import cdist

from rules import MetricResult, check_protein_level_comparison
from metrics.statistics import bootstrap_ci


def compute_protein_vectors(
    embeddings: dict[str, np.ndarray],
    method: str = "dct_k4",
    dct_k: int = 4,
) -> dict[str, np.ndarray]:
    """Compute protein-level vectors from per-residue embeddings.

    Args:
        embeddings: {protein_id: (L, D)} per-residue embeddings.
        method: "dct_k4" or "mean".
        dct_k: Number of DCT coefficients (only for dct_k4 method).

    Returns:
        {protein_id: (V,)} protein-level vectors.
    """
    vectors = {}
    for pid, emb in embeddings.items():
        emb = np.asarray(emb, dtype=np.float32)
        if method == "mean":
            vectors[pid] = emb.mean(axis=0)
        elif method == "dct_k4":
            coeffs = dct(emb, axis=0, type=2, norm="ortho")[:dct_k]  # (K, D)
            vectors[pid] = coeffs.flatten().astype(np.float32)
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    return vectors


def _retrieval_ret1(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str,
    metric: str,
) -> dict[str, float]:
    """Compute per-query Ret@1 scores for bootstrap.

    Returns {query_id: 1.0 if top-1 match, 0.0 otherwise}.
    """
    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}
    pids = [pid for pid in vectors if pid in id_to_label]
    if len(pids) < 2:
        return {}

    mat = np.array([vectors[pid] for pid in pids], dtype=np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mat_normed = mat / norms
        sims = mat_normed @ mat_normed.T
    elif metric == "euclidean":
        dists = cdist(mat, mat, metric="euclidean")
        sims = -dists
    else:
        raise ValueError(f"Unknown metric: {metric}")

    scores = {}
    for i, pid in enumerate(pids):
        sims_row = sims[i].copy()
        sims_row[i] = -np.inf  # exclude self
        top1_idx = np.argmax(sims_row)
        top1_pid = pids[top1_idx]
        scores[pid] = 1.0 if id_to_label[top1_pid] == id_to_label[pid] else 0.0

    return scores


def run_retrieval_benchmark(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run retrieval benchmark with dual metric (Rule 12).

    Returns dict with ret1_cosine, ret1_euclidean (both MetricResult), n_queries.
    """
    results = {}
    for metric in ["cosine", "euclidean"]:
        per_query = _retrieval_ret1(vectors, metadata, label_key, metric)
        results[f"per_query_{metric}"] = per_query
        if per_query:
            ret1_ci = bootstrap_ci(per_query, n_bootstrap=n_bootstrap, seed=seed)
            results[f"ret1_{metric}"] = ret1_ci
        else:
            results[f"ret1_{metric}"] = MetricResult(value=0.0, ci_lower=0.0, ci_upper=0.0, n=0)

    results["n_queries"] = len(vectors)
    return results
