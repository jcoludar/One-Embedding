"""Unit tests for src/oe_autoeval/reference_set.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.oe_autoeval.reference_set import sample_reference_ids


def test_sample_is_deterministic_and_subset():
    train_ids = [f"tr{i}" for i in range(5000)]
    a = sample_reference_ids(train_ids, n=2000, seed=42)
    b = sample_reference_ids(train_ids, n=2000, seed=42)
    assert a == b
    assert len(a) == 2000
    assert set(a).issubset(set(train_ids))


def test_sample_caps_at_population():
    a = sample_reference_ids(["x", "y"], n=2000, seed=42)
    assert sorted(a) == ["x", "y"]


def test_sample_dedups_input():
    a = sample_reference_ids(["a", "a", "b", "b", "b"], n=10, seed=1)
    assert sorted(a) == ["a", "b"]


def test_different_seed_differs():
    train_ids = [f"tr{i}" for i in range(5000)]
    a = sample_reference_ids(train_ids, n=100, seed=1)
    b = sample_reference_ids(train_ids, n=100, seed=2)
    assert a != b
