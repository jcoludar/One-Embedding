"""Unit tests for src/one_embedding/vep.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.vep import select_diversity_subset, AssayInfo, prepare_reference_df


def _toy_reference_df() -> pd.DataFrame:
    """Synthetic ProteinGym reference rows covering the subset criteria."""
    rows = [
        # 5 small (<150 aa)
        ("A1_kinase", 100, "Kinase", "growth"),
        ("A2_kinase", 120, "Kinase", "binding"),
        ("A3_phos", 140, "Phosphatase", "stability"),
        ("A4_tf", 90, "TranscriptionFactor", "growth"),
        ("A5_struct", 130, "Structural", "stability"),
        # 8 medium (150-400 aa)
        ("B1_kinase", 250, "Kinase", "growth"),
        ("B2_kinase", 300, "Kinase", "binding"),
        ("B3_phos", 350, "Phosphatase", "growth"),
        ("B4_tf", 200, "TranscriptionFactor", "binding"),
        ("B5_struct", 280, "Structural", "stability"),
        ("B6_kinase", 220, "Kinase", "stability"),
        ("B7_tf", 180, "TranscriptionFactor", "growth"),
        ("B8_phos", 380, "Phosphatase", "binding"),
        # 4 large (>400 aa)
        ("C1_kinase", 500, "Kinase", "growth"),
        ("C2_phos", 600, "Phosphatase", "binding"),
        ("C3_tf", 700, "TranscriptionFactor", "stability"),
        ("C4_struct", 450, "Structural", "growth"),
    ]
    return pd.DataFrame(
        rows,
        columns=["DMS_id", "seq_len", "family", "fitness_type"],
    )


def test_select_diversity_subset_returns_15():
    df = _toy_reference_df()
    chosen = select_diversity_subset(df, n=15, seed=42)
    assert len(chosen) == 15
    assert all(isinstance(a, AssayInfo) for a in chosen)


def test_select_diversity_subset_size_buckets():
    df = _toy_reference_df()
    chosen = select_diversity_subset(df, n=15, seed=42)
    small = sum(1 for a in chosen if a.seq_len < 150)
    medium = sum(1 for a in chosen if 150 <= a.seq_len <= 400)
    large = sum(1 for a in chosen if a.seq_len > 400)
    assert small == 4, f"expected 4 small, got {small}"
    assert medium == 7, f"expected 7 medium, got {medium}"
    assert large == 4, f"expected 4 large, got {large}"


def test_select_diversity_subset_family_coverage():
    df = _toy_reference_df()
    chosen = select_diversity_subset(df, n=15, seed=42)
    families = {a.family for a in chosen}
    assert len(families) >= 4


def test_select_diversity_subset_fitness_coverage():
    df = _toy_reference_df()
    chosen = select_diversity_subset(df, n=15, seed=42)
    fitness = {a.fitness_type for a in chosen}
    assert len(fitness) >= 3


def test_select_diversity_subset_deterministic():
    df = _toy_reference_df()
    a = select_diversity_subset(df, n=15, seed=42)
    b = select_diversity_subset(df, n=15, seed=42)
    assert [x.dms_id for x in a] == [x.dms_id for x in b]


# ---------------------------------------------------------------------------
# prepare_reference_df tests
# ---------------------------------------------------------------------------

def test_prepare_reference_df_real_schema():
    """Maps the real ProteinGym column names to the canonical ones."""
    df = pd.DataFrame({
        "DMS_id": ["X1", "X2"],
        "taxon": ["Human", "Virus"],
        "coarse_selection_type": ["Activity", "Stability"],
        "seq_len": [100, 200],
    })
    out = prepare_reference_df(df)
    assert list(out["family"]) == ["Human", "Virus"]
    assert list(out["fitness_type"]) == ["Activity", "Stability"]
    assert list(out["seq_len"]) == [100, 200]


def test_prepare_reference_df_canonical_passthrough():
    """If canonical names are already present, leave them unchanged."""
    df = pd.DataFrame({
        "DMS_id": ["X1"],
        "family": ["Kinase"],
        "fitness_type": ["growth"],
        "seq_len": [100],
    })
    out = prepare_reference_df(df)
    assert list(out["family"]) == ["Kinase"]
    assert list(out["fitness_type"]) == ["growth"]


def test_prepare_reference_df_seq_len_from_target_seq():
    df = pd.DataFrame({
        "DMS_id": ["X1"],
        "taxon": ["Human"],
        "coarse_selection_type": ["Activity"],
        "target_seq": ["MAKVLR"],
    })
    out = prepare_reference_df(df)
    assert int(out["seq_len"].iloc[0]) == 6


def test_prepare_reference_df_missing_columns_raises():
    df = pd.DataFrame({"DMS_id": ["X1"], "seq_len": [100]})  # no family or taxon
    with pytest.raises(ValueError, match="family.*taxon"):
        prepare_reference_df(df)


def test_select_diversity_subset_on_real_proteingym_csv():
    """End-to-end smoke: real CSV -> prepare -> select. Catches column-mapping bugs."""
    real_csv = ROOT / "data" / "proteingym" / "DMS_substitutions.csv"
    if not real_csv.exists():
        pytest.skip("real ProteinGym reference CSV not downloaded")
    df = pd.read_csv(real_csv)
    prepared = prepare_reference_df(df)
    chosen = select_diversity_subset(prepared, n=15, seed=42)
    assert len(chosen) == 15
    families = {a.family for a in chosen}
    assert len(families) >= 2  # real data has 4 taxa, expect at least 2 to appear
    fitness = {a.fitness_type for a in chosen}
    assert len(fitness) >= 2  # real data has 5 selection types
