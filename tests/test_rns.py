"""Tests for RNS (Random Neighbor Score) evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.rns import generate_junkyard_sequences, compute_rns


class TestGenerateJunkyardSequences:
    def test_count_and_naming(self):
        seqs = {"p1": "ACDEF", "p2": "GHIKLMN"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=3, seed=42)
        assert len(junk) == 6  # 2 proteins x 3 shuffles
        assert "p1_shuf0" in junk
        assert "p1_shuf2" in junk
        assert "p2_shuf0" in junk

    def test_same_composition(self):
        seqs = {"p1": "AACDEF"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=5, seed=42)
        for pid, seq in junk.items():
            assert sorted(seq) == sorted("AACDEF")

    def test_deterministic(self):
        seqs = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
        j1 = generate_junkyard_sequences(seqs, seed=42)
        j2 = generate_junkyard_sequences(seqs, seed=42)
        assert j1 == j2

    def test_different_from_original(self):
        # With a 20-residue sequence, random shuffle != original with high prob
        seqs = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
        junk = generate_junkyard_sequences(seqs, n_shuffles=5, seed=42)
        n_different = sum(1 for s in junk.values() if s != seqs["p1"])
        assert n_different >= 4  # at least 4 of 5 should differ


class TestComputeRNS:
    def test_perfect_real_neighbors_gives_zero(self):
        """If a query's neighbors are all real proteins, RNS = 0."""
        d = 16
        rng = np.random.RandomState(0)
        # 10 real proteins in a tight cluster
        real = {f"r{i}": rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # 10 junkyard proteins FAR away
        junk = {f"j{i}": (rng.randn(d).astype(np.float32) + 100.0)
                for i in range(10)}
        # Query = one of the real proteins
        query = {"r0": real["r0"]}
        scores = compute_rns(query, real, junk, k=5)
        assert scores["r0"] < 0.01  # essentially 0

    def test_junkyard_neighbor_gives_high_rns(self):
        """If a query sits among junkyard vectors, RNS ~ 1."""
        d = 16
        rng = np.random.RandomState(0)
        # 10 real proteins in one region
        real = {f"r{i}": rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # 10 junkyard in another region
        junk_center = rng.randn(d).astype(np.float32) * 0.1 + 50.0
        junk = {f"j{i}": junk_center + rng.randn(d).astype(np.float32) * 0.1
                for i in range(10)}
        # Query sits right in the junkyard cluster
        query = {"q": junk_center.copy()}
        scores = compute_rns(query, real, junk, k=5)
        assert scores["q"] > 0.8

    def test_returns_score_for_every_query(self):
        d = 8
        rng = np.random.RandomState(0)
        real = {f"r{i}": rng.randn(d).astype(np.float32) for i in range(5)}
        junk = {f"j{i}": rng.randn(d).astype(np.float32) for i in range(5)}
        query = {f"r{i}": real[f"r{i}"] for i in range(3)}
        scores = compute_rns(query, real, junk, k=3)
        assert set(scores.keys()) == {"r0", "r1", "r2"}
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_exclude_shuffles_of_query_drops_same_source_junk(self):
        """exclude_shuffles_of_query=True must filter pid_shuf* from the
        query's neighbors, while keeping shuffles from other sources."""
        d = 8
        rng = np.random.RandomState(0)
        real_vec = rng.randn(d).astype(np.float32)
        other_vec = rng.randn(d).astype(np.float32) + 10.0
        real = {"r0": real_vec, "r1": other_vec}
        junk = {
            "r0_shuf0": real_vec.copy(),  # same-source — should be filtered
            "r0_shuf1": real_vec.copy(),
            "r0_shuf2": real_vec.copy(),
            "r1_shuf0": other_vec.copy(),  # other-source — kept
            "r1_shuf1": other_vec.copy(),
        }
        query = {"r0": real_vec}

        # Default: same-source shuffles are NOT filtered → r0's top-3 valid
        # neighbors are all r0_shuf* (junk), so RNS == 1.0
        scores_default = compute_rns(query, real, junk, k=3)
        assert scores_default["r0"] == 1.0

        # With the filter: r0_shuf* dropped → top-3 valid are r1 + 2 r1_shuf*
        # (1 real, 2 junk) → 2/3 junk
        scores_filtered = compute_rns(
            query, real, junk, k=3, exclude_shuffles_of_query=True
        )
        assert abs(scores_filtered["r0"] - 2.0 / 3.0) < 1e-6
