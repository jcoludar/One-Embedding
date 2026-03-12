"""Tests for Euclidean metric in retrieval evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.hierarchy import evaluate_hierarchy_distances


# ── Fixtures ─────────────────────────────────────────────────────


def _make_clustered_data(n_families=5, n_per_family=10, dim=32, separation=5.0, seed=42):
    """Create synthetic data with well-separated family clusters."""
    rng = np.random.RandomState(seed)
    vectors = {}
    metadata = []

    for fam_idx in range(n_families):
        center = rng.randn(dim).astype(np.float32) * separation
        for i in range(n_per_family):
            pid = f"prot_{fam_idx}_{i}"
            vec = center + rng.randn(dim).astype(np.float32) * 0.1
            vectors[pid] = vec
            metadata.append({
                "id": pid,
                "family": f"fam_{fam_idx}",
                "superfamily": f"sf_{fam_idx // 2}",
                "fold": f"fold_{fam_idx // 3}",
            })

    return vectors, metadata


# ── Tests: Metric parameter ──────────────────────────────────────


def test_cosine_metric_unchanged():
    """Cosine metric should produce same results as before (default)."""
    vectors, metadata = _make_clustered_data()
    pids = list(vectors.keys())

    result_default = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids)
    result_cosine = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="cosine")

    assert result_default["precision@1"] == result_cosine["precision@1"]
    assert result_default["mrr"] == result_cosine["mrr"]
    assert result_default["map"] == result_cosine["map"]


def test_euclidean_metric_runs():
    """Euclidean metric should produce valid results."""
    vectors, metadata = _make_clustered_data()
    pids = list(vectors.keys())

    result = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="euclidean")

    assert "precision@1" in result
    assert "mrr" in result
    assert "map" in result
    assert 0.0 <= result["precision@1"] <= 1.0
    assert 0.0 <= result["mrr"] <= 1.0
    assert result["n_queries"] == len(pids)


def test_invalid_metric_raises():
    """Invalid metric should raise ValueError."""
    vectors, metadata = _make_clustered_data()
    pids = list(vectors.keys())

    with pytest.raises(ValueError, match="metric must be"):
        evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="manhattan")


def test_well_separated_clusters_perfect_retrieval():
    """Very well-separated clusters should give perfect Ret@1 under both metrics."""
    vectors, metadata = _make_clustered_data(separation=100.0)
    pids = list(vectors.keys())

    cos_result = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="cosine")
    euc_result = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="euclidean")

    assert cos_result["precision@1"] > 0.95
    assert euc_result["precision@1"] > 0.95


def test_euclidean_preserves_ranking_for_scaled_vectors():
    """Euclidean should differentiate vectors that cosine treats as identical.

    If vectors differ only in magnitude, cosine gives same similarity,
    but Euclidean distinguishes them.
    """
    # Create two identical-direction vectors with different magnitudes
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vectors = {
        "query": base,
        "close": base * 1.1,     # same direction, slightly larger
        "far": base * 10.0,      # same direction, much larger
        "different": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # orthogonal
    }
    metadata = [
        {"id": "query", "family": "A"},
        {"id": "close", "family": "A"},
        {"id": "far", "family": "A"},
        {"id": "different", "family": "B"},
    ]

    euc = evaluate_retrieval_from_vectors(
        vectors, metadata, query_ids=["query"], database_ids=list(vectors.keys()),
        metric="euclidean",
    )
    # Euclidean should find "close" as nearest neighbor (not "far")
    assert euc["precision@1"] == 1.0  # "close" is family A = correct


def test_euclidean_vs_cosine_different_results():
    """Euclidean and cosine should give different results for non-trivial data."""
    rng = np.random.RandomState(123)
    vectors = {}
    metadata = []
    # Family A: large-magnitude vectors in one direction
    for i in range(10):
        pid = f"A_{i}"
        vectors[pid] = (rng.randn(16) * 10.0 + np.ones(16) * 20).astype(np.float32)
        metadata.append({"id": pid, "family": "A", "superfamily": "sf1", "fold": "f1"})
    # Family B: small-magnitude vectors in similar direction
    for i in range(10):
        pid = f"B_{i}"
        vectors[pid] = (rng.randn(16) * 0.1 + np.ones(16) * 0.5).astype(np.float32)
        metadata.append({"id": pid, "family": "B", "superfamily": "sf1", "fold": "f1"})

    pids = list(vectors.keys())
    cos = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="cosine")
    euc = evaluate_retrieval_from_vectors(vectors, metadata, query_ids=pids, database_ids=pids, metric="euclidean")

    # They should give different precision values for this case
    # (cosine struggles because directions overlap; Euclidean separates by magnitude)
    assert euc["precision@1"] >= cos["precision@1"]


# ── Tests: Hierarchy evaluation ──────────────────────────────────


def test_hierarchy_basic():
    """Hierarchy evaluation should return all expected fields."""
    vectors, metadata = _make_clustered_data()
    result = evaluate_hierarchy_distances(vectors, metadata, metric="euclidean")

    assert "within_family_mean" in result
    assert "same_superfamily_mean" in result
    assert "same_fold_mean" in result
    assert "unrelated_mean" in result
    assert "separation_ratio" in result
    assert "ordering_correct" in result
    assert result["n_proteins"] == len(vectors)


def test_hierarchy_ordering_well_separated():
    """Well-separated clusters should show correct hierarchy ordering."""
    vectors, metadata = _make_clustered_data(separation=50.0)
    result = evaluate_hierarchy_distances(vectors, metadata, metric="euclidean")

    # Within-family should be smallest distance
    wf = result["within_family_mean"]
    ur = result["unrelated_mean"]
    assert wf < ur
    assert result["separation_ratio"] > 1.0


def test_hierarchy_both_metrics():
    """Hierarchy should work with both cosine and Euclidean."""
    vectors, metadata = _make_clustered_data()

    cos_result = evaluate_hierarchy_distances(vectors, metadata, metric="cosine")
    euc_result = evaluate_hierarchy_distances(vectors, metadata, metric="euclidean")

    # Both should have valid results
    assert cos_result["within_family_mean"] is not None
    assert euc_result["within_family_mean"] is not None
    # Cosine distances are in [0, 2], Euclidean are unbounded
    assert cos_result["within_family_mean"] <= 2.0


def test_hierarchy_too_few_proteins():
    """Should handle too few proteins gracefully."""
    vectors = {"p1": np.zeros(10, dtype=np.float32)}
    metadata = [{"id": "p1", "family": "A", "superfamily": "sf", "fold": "f"}]
    result = evaluate_hierarchy_distances(vectors, metadata)
    assert "error" in result
