"""Tests for embedding-space protein aligner."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.aligner import (
    embedding_score_matrix,
    z_score_filter,
    needleman_wunsch,
    smith_waterman,
    align_embeddings,
)


class TestEmbeddingScoreMatrix:
    def test_shape(self):
        emb_a = np.random.randn(10, 512).astype(np.float32)
        emb_b = np.random.randn(15, 512).astype(np.float32)
        mat = embedding_score_matrix(emb_a, emb_b)
        assert mat.shape == (10, 15)

    def test_identical_high_diagonal(self):
        emb = np.random.randn(8, 512).astype(np.float32)
        mat = embedding_score_matrix(emb, emb)
        # Diagonal should be highest in each row
        for i in range(8):
            assert mat[i, i] == pytest.approx(mat[i].max(), abs=1e-5)

    def test_scale_factor(self):
        emb_a = np.random.randn(5, 64).astype(np.float32)
        emb_b = np.random.randn(5, 64).astype(np.float32)
        mat1 = embedding_score_matrix(emb_a, emb_b, scale=1.0)
        mat10 = embedding_score_matrix(emb_a, emb_b, scale=10.0)
        np.testing.assert_allclose(mat10, mat1 * 10.0, atol=1e-5)

    def test_cosine_range(self):
        emb_a = np.random.randn(5, 64).astype(np.float32)
        emb_b = np.random.randn(5, 64).astype(np.float32)
        mat = embedding_score_matrix(emb_a, emb_b, scale=1.0)
        assert mat.min() >= -1.01
        assert mat.max() <= 1.01


class TestZScoreFilter:
    def test_shape_preserved(self):
        mat = np.random.randn(10, 15)
        filtered = z_score_filter(mat)
        assert filtered.shape == (10, 15)

    def test_zero_mean_rows(self):
        mat = np.random.randn(10, 15)
        filtered = z_score_filter(mat)
        # After double z-score, column means should be ~0
        col_means = filtered.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.1)


class TestNeedlemanWunsch:
    def test_identical_sequences_no_gaps(self):
        emb = np.random.randn(5, 64).astype(np.float32)
        mat = embedding_score_matrix(emb, emb)
        align_a, align_b, score = needleman_wunsch(mat)
        # Identical embeddings should align without gaps
        assert all(a >= 0 for a in align_a)
        assert all(b >= 0 for b in align_b)
        assert len(align_a) == 5
        assert align_a == list(range(5))
        assert align_b == list(range(5))

    def test_alignment_length_correct(self):
        mat = np.random.randn(8, 12)
        align_a, align_b, score = needleman_wunsch(mat)
        assert len(align_a) == len(align_b)
        # All positions of A and B must appear
        a_positions = [x for x in align_a if x >= 0]
        b_positions = [x for x in align_b if x >= 0]
        assert sorted(a_positions) == list(range(8))
        assert sorted(b_positions) == list(range(12))

    def test_score_positive_for_similar(self):
        emb = np.random.randn(10, 64).astype(np.float32)
        noise = np.random.randn(10, 64).astype(np.float32) * 0.1
        mat = embedding_score_matrix(emb, emb + noise)
        _, _, score = needleman_wunsch(mat)
        assert score > 0


class TestSmithWaterman:
    def test_local_alignment_subset(self):
        # Embed A has a matching region in the middle of B
        rng = np.random.RandomState(42)
        shared = rng.randn(5, 64).astype(np.float32)
        emb_a = shared  # 5 residues
        emb_b = np.vstack([rng.randn(3, 64), shared, rng.randn(3, 64)]).astype(np.float32)  # 11 residues

        mat = embedding_score_matrix(emb_a, emb_b)
        align_a, align_b, score = smith_waterman(mat)
        assert score > 0
        assert len(align_a) > 0

    def test_score_higher_than_zero(self):
        emb_a = np.random.randn(8, 64).astype(np.float32)
        emb_b = np.random.randn(8, 64).astype(np.float32)
        mat = embedding_score_matrix(emb_a, emb_b)
        _, _, score = smith_waterman(mat)
        assert score >= 0


class TestAlignEmbeddings:
    def test_global_mode(self):
        emb_a = np.random.randn(8, 512).astype(np.float32)
        emb_b = np.random.randn(10, 512).astype(np.float32)
        result = align_embeddings(emb_a, emb_b, mode="global")
        assert "align_a" in result
        assert "score" in result
        assert result["n_aligned"] > 0

    def test_local_mode(self):
        emb_a = np.random.randn(8, 512).astype(np.float32)
        emb_b = np.random.randn(10, 512).astype(np.float32)
        result = align_embeddings(emb_a, emb_b, mode="local")
        assert result["score"] >= 0

    def test_z_score_mode(self):
        emb_a = np.random.randn(8, 512).astype(np.float32)
        emb_b = np.random.randn(10, 512).astype(np.float32)
        result = align_embeddings(emb_a, emb_b, use_z_score=True)
        assert "score_matrix" in result

    def test_identical_perfect_alignment(self):
        emb = np.random.randn(6, 512).astype(np.float32)
        result = align_embeddings(emb, emb, mode="global")
        assert result["n_gaps_a"] == 0
        assert result["n_gaps_b"] == 0
        assert result["n_aligned"] == 6
