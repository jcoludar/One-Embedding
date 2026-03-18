"""Tests for ancestral embedding reconstruction."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.ancestral import (
    reconstruct_ancestral_embeddings,
    tree_to_node_list,
    ancestral_distance_matrix,
    embedding_to_nearest_aa,
)


class TestReconstructAncestralEmbeddings:
    def _simple_tree_nodes(self):
        """Binary tree: ((A:1, B:1):0.5, C:1.5) root"""
        return [
            {"id": 0, "name": "A", "is_leaf": True, "is_root": False,
             "children_ids": [], "branch_length": 1.0},
            {"id": 1, "name": "B", "is_leaf": True, "is_root": False,
             "children_ids": [], "branch_length": 1.0},
            {"id": 2, "name": "", "is_leaf": False, "is_root": False,
             "children_ids": [0, 1], "branch_length": 0.5},
            {"id": 3, "name": "C", "is_leaf": True, "is_root": False,
             "children_ids": [], "branch_length": 1.5},
            {"id": 4, "name": "", "is_leaf": False, "is_root": True,
             "children_ids": [2, 3], "branch_length": 0.0},
        ]

    def test_returns_all_nodes(self):
        nodes = self._simple_tree_nodes()
        leaf_embs = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        result = reconstruct_ancestral_embeddings(nodes, leaf_embs)
        assert len(result) == 5  # 3 leaves + 2 internal

    def test_leaf_values_preserved(self):
        nodes = self._simple_tree_nodes()
        leaf_embs = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([0.5, 0.5]),
        }
        result = reconstruct_ancestral_embeddings(nodes, leaf_embs)
        np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-6)

    def test_internal_is_weighted_mean(self):
        """Internal node joining A:1 and B:1 should be midpoint."""
        nodes = self._simple_tree_nodes()
        leaf_embs = {
            "A": np.array([2.0, 0.0]),
            "B": np.array([0.0, 2.0]),
            "C": np.array([1.0, 1.0]),
        }
        result = reconstruct_ancestral_embeddings(nodes, leaf_embs)
        # Node 2 joins A and B with equal branch lengths → midpoint
        np.testing.assert_allclose(result[2], [1.0, 1.0], atol=1e-6)

    def test_unequal_branch_lengths_bias(self):
        """Shorter branch → closer to that child."""
        nodes = [
            {"id": 0, "name": "A", "is_leaf": True, "is_root": False,
             "children_ids": [], "branch_length": 0.1},  # short
            {"id": 1, "name": "B", "is_leaf": True, "is_root": False,
             "children_ids": [], "branch_length": 10.0},  # long
            {"id": 2, "name": "", "is_leaf": False, "is_root": True,
             "children_ids": [0, 1], "branch_length": 0.0},
        ]
        leaf_embs = {
            "A": np.array([10.0]),
            "B": np.array([0.0]),
        }
        result = reconstruct_ancestral_embeddings(nodes, leaf_embs)
        # Should be much closer to A (short branch)
        assert result[2][0] > 5.0

    def test_multidimensional(self):
        """Works with high-dimensional embeddings."""
        nodes = self._simple_tree_nodes()
        D = 512
        rng = np.random.RandomState(42)
        leaf_embs = {
            "A": rng.randn(D).astype(np.float32),
            "B": rng.randn(D).astype(np.float32),
            "C": rng.randn(D).astype(np.float32),
        }
        result = reconstruct_ancestral_embeddings(nodes, leaf_embs)
        assert result[4].shape == (D,)

    def test_missing_leaf_raises(self):
        nodes = self._simple_tree_nodes()
        leaf_embs = {"A": np.array([1.0]), "B": np.array([2.0])}  # Missing C
        with pytest.raises(ValueError, match="No embedding for leaf"):
            reconstruct_ancestral_embeddings(nodes, leaf_embs)


class TestAncestralDistanceMatrix:
    def test_shape(self):
        embs = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([0.5, 0.5])}
        names = {0: "A", 1: "B", 2: ""}
        dist, name_list = ancestral_distance_matrix(embs, names)
        assert dist.shape == (3, 3)
        assert len(name_list) == 3
        assert "anc_2" in name_list

    def test_diagonal_zero(self):
        embs = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
        dist, _ = ancestral_distance_matrix(embs, {0: "A", 1: "B"})
        np.testing.assert_allclose(np.diag(dist), 0.0)


class TestEmbeddingToNearestAA:
    def test_identity_mapping(self):
        """When embedding matches an AA exactly, should recover it."""
        aa_embs = {
            "A": np.array([1.0, 0.0, 0.0]),
            "C": np.array([0.0, 1.0, 0.0]),
            "D": np.array([0.0, 0.0, 1.0]),
        }
        emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        seq = embedding_to_nearest_aa(emb, aa_embs)
        assert seq == "ACD"

    def test_nearest_match(self):
        aa_embs = {
            "A": np.array([1.0, 0.0]),
            "G": np.array([0.0, 1.0]),
        }
        emb = np.array([[0.9, 0.1], [0.1, 0.9]])
        seq = embedding_to_nearest_aa(emb, aa_embs)
        assert seq == "AG"
