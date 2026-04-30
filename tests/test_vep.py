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


# ---------------------------------------------------------------------------
# Task 3: ProteinGym CSV loaders
# ---------------------------------------------------------------------------

from src.one_embedding.vep import (
    DMSAssay, load_dms_assay,
    load_clinvar_split,
    load_dms_assay_single_subs,
)


def _write_toy_dms(tmp_path):
    """Toy DMS CSV: WT=MAR..., 4 variants."""
    csv = tmp_path / "TOY_ASSAY.csv"
    csv.write_text(
        "mutant,mutated_sequence,DMS_score\n"
        "M1A,AAR,0.10\n"  # pos 0: M -> A
        "M1L,LAR,-0.50\n"  # pos 0: M -> L
        "A2T,MTR,0.30\n"  # pos 1: A -> T
        "R3K,MAK,0.05\n"  # pos 2: R -> K
    )
    return csv


def test_load_dms_assay_basic(tmp_path):
    csv = _write_toy_dms(tmp_path)
    assay = load_dms_assay(csv, dms_id="TOY_ASSAY")
    assert assay.dms_id == "TOY_ASSAY"
    assert assay.wt_sequence == "MAR"
    assert len(assay.variants) == 4
    v0 = assay.variants[0]
    assert (v0.mut_pos, v0.wt_aa, v0.mut_aa) == (0, "M", "A")
    assert v0.score == pytest.approx(0.10)


def test_load_dms_assay_recovers_wt_from_first_variant(tmp_path):
    """WT seq is recovered by un-mutating any variant — sanity check."""
    csv = _write_toy_dms(tmp_path)
    assay = load_dms_assay(csv, dms_id="TOY_ASSAY")
    assert assay.wt_sequence[0] == "M"
    assert assay.wt_sequence[1] == "A"
    assert assay.wt_sequence[2] == "R"


def _write_toy_clinvar(tmp_path):
    csv = tmp_path / "P12345_clinvar.csv"
    csv.write_text(
        "mutant,mutated_sequence,DMS_bin_score\n"
        "M1A,AAR,1\n"
        "A2T,MTR,0\n"
    )
    return csv


def test_load_clinvar_split(tmp_path):
    csv = _write_toy_clinvar(tmp_path)
    variants = load_clinvar_split(csv, pid="P12345")
    assert len(variants) == 2
    assert variants[0].pid == "P12345"
    assert variants[0].label == 1
    assert variants[0].mut_pos == 0
    assert variants[1].label == 0


# ---------------------------------------------------------------------------
# load_dms_assay_single_subs tests
# ---------------------------------------------------------------------------

def test_load_dms_assay_single_subs_filters_multimutants(tmp_path):
    """Rows with ':' in mutant string are dropped; single-subs kept."""
    csv = tmp_path / "MIXED.csv"
    csv.write_text(
        "mutant,mutated_sequence,DMS_score\n"
        "M1A,AAR,0.10\n"      # single — keep
        "M1A:R3K,AAK,0.20\n"  # double — drop
        "A2T,MTR,0.30\n"      # single — keep
        "M1A:A2T:R3K,ATK,-0.5\n"  # triple — drop
    )
    assay = load_dms_assay_single_subs(csv, dms_id="MIXED")
    assert assay.dms_id == "MIXED"
    assert len(assay.variants) == 2
    assert assay.wt_sequence == "MAR"
    assert {(v.mut_pos, v.mut_aa) for v in assay.variants} == {(0, "A"), (1, "T")}


def test_load_dms_assay_single_subs_recovers_wt_from_first_single(tmp_path):
    """If first row is multi-mutant, WT is still recovered from first SINGLE row."""
    csv = tmp_path / "MULTIFIRST.csv"
    csv.write_text(
        "mutant,mutated_sequence,DMS_score\n"
        "M1A:R3K,AAK,0.20\n"   # multi — skipped, doesn't define WT
        "A2T,MTR,0.30\n"       # first single — defines WT as MAR
        "M1A,AAR,0.10\n"
    )
    assay = load_dms_assay_single_subs(csv, dms_id="MULTIFIRST")
    assert assay.wt_sequence == "MAR"
    assert len(assay.variants) == 2


def test_load_dms_assay_single_subs_all_multi_raises(tmp_path):
    """All-multi-mutant CSV should raise ValueError."""
    csv = tmp_path / "ALLMULTI.csv"
    csv.write_text(
        "mutant,mutated_sequence,DMS_score\n"
        "M1A:R3K,AAK,0.20\n"
        "A2T:R3K,MTK,0.30\n"
    )
    with pytest.raises(ValueError, match="no single-substitution"):
        load_dms_assay_single_subs(csv, dms_id="ALLMULTI")


# ---------------------------------------------------------------------------
# C1 / I1 / I2 defensive guard tests
# ---------------------------------------------------------------------------

def test_load_dms_assay_empty_csv_raises(tmp_path):
    """Empty CSV (headers only) raises ValueError with file context."""
    csv = tmp_path / "EMPTY.csv"
    csv.write_text("mutant,mutated_sequence,DMS_score\n")  # header only
    with pytest.raises(ValueError, match="no data rows"):
        load_dms_assay(csv, dms_id="EMPTY")


def test_load_clinvar_split_empty_csv_raises(tmp_path):
    csv = tmp_path / "EMPTY.csv"
    csv.write_text("mutant,mutated_sequence,DMS_bin_score\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_clinvar_split(csv, pid="X")


def test_parse_mutant_rejects_position_zero():
    from src.one_embedding.vep import _parse_mutant
    with pytest.raises(ValueError, match="position must be >= 1"):
        _parse_mutant("M0A")


def test_parse_mutant_rejects_negative_position():
    from src.one_embedding.vep import _parse_mutant
    # Negative pos comes from input like "M-1A" — int("-1") parses fine
    with pytest.raises(ValueError, match="position must be >= 1"):
        _parse_mutant("M-1A")


# ---------------------------------------------------------------------------
# Task 4: build_variant_features
# ---------------------------------------------------------------------------

from src.one_embedding.vep import build_variant_features


def test_build_variant_features_shape():
    L, D = 50, 64
    rng = np.random.default_rng(0)
    wt = rng.standard_normal((L, D)).astype(np.float32)
    mut = rng.standard_normal((L, D)).astype(np.float32)
    feat = build_variant_features(wt_emb=wt, mut_emb=mut, mut_pos=10)
    assert feat.shape == (4 * D,), f"got {feat.shape}"


def test_build_variant_features_components():
    L, D = 5, 4
    wt = np.zeros((L, D), dtype=np.float32)
    mut = np.zeros((L, D), dtype=np.float32)
    wt[2] = np.array([1, 2, 3, 4], dtype=np.float32)
    mut[2] = np.array([5, 6, 7, 8], dtype=np.float32)
    wt[4] = np.array([1, 1, 1, 1], dtype=np.float32)  # affects mean(WT)
    feat = build_variant_features(wt_emb=wt, mut_emb=mut, mut_pos=2)
    np.testing.assert_array_equal(feat[:4], [1, 2, 3, 4])           # WT[mut_pos]
    np.testing.assert_array_equal(feat[4:8], [5, 6, 7, 8])           # mut[mut_pos]
    expected_mean_wt = wt.mean(axis=0)
    np.testing.assert_allclose(feat[8:12], expected_mean_wt, rtol=1e-6)
    expected_mean_mut = mut.mean(axis=0)
    np.testing.assert_allclose(feat[12:16], expected_mean_mut, rtol=1e-6)


def test_build_variant_features_pos_out_of_range():
    L, D = 10, 4
    wt = np.zeros((L, D), dtype=np.float32)
    mut = np.zeros((L, D), dtype=np.float32)
    with pytest.raises(IndexError):
        build_variant_features(wt_emb=wt, mut_emb=mut, mut_pos=L)


def test_build_variant_features_shape_mismatch():
    wt = np.zeros((5, 4), dtype=np.float32)
    mut = np.zeros((5, 8), dtype=np.float32)  # different D
    with pytest.raises(ValueError, match="shape mismatch"):
        build_variant_features(wt_emb=wt, mut_emb=mut, mut_pos=0)


def test_build_variant_features_negative_pos():
    wt = np.zeros((5, 4), dtype=np.float32)
    mut = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(IndexError):
        build_variant_features(wt_emb=wt, mut_emb=mut, mut_pos=-1)


# ---------------------------------------------------------------------------
# Task 5: fit_evaluate_ridge_probe
# ---------------------------------------------------------------------------

from src.one_embedding.vep import fit_evaluate_ridge_probe, ProbeResult


def test_fit_evaluate_ridge_probe_runs():
    """Smoke test: probe trains, returns the right shape, ρ in [-1, 1]."""
    rng = np.random.default_rng(123)
    n_variants, feat_dim = 100, 32
    X = rng.standard_normal((n_variants, feat_dim)).astype(np.float32)
    # Synthetic linear signal so ρ should be > 0
    true_w = rng.standard_normal(feat_dim).astype(np.float32)
    y = X @ true_w + 0.1 * rng.standard_normal(n_variants).astype(np.float32)

    result = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[42, 123, 456])
    assert isinstance(result, ProbeResult)
    assert -1.0 <= result.spearman_rho <= 1.0
    assert result.predictions.shape == y.shape
    assert result.spearman_rho > 0.5, f"ρ={result.spearman_rho} on linear-signal toy"


def test_fit_evaluate_ridge_probe_deterministic():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 16)).astype(np.float32)
    y = rng.standard_normal(80).astype(np.float32)
    a = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[42])
    b = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[42])
    np.testing.assert_allclose(a.predictions, b.predictions)
    assert a.spearman_rho == pytest.approx(b.spearman_rho)


def test_fit_evaluate_ridge_probe_handles_constant_y():
    """If y is constant, Spearman ρ is undefined (nan); should not crash."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    y = np.ones(50, dtype=np.float32)
    result = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[42])
    # ρ may be nan when y is constant; the call must not crash and must
    # return a ProbeResult with the right shapes.
    assert result.predictions.shape == y.shape
    assert result.n_variants == 50


def test_fit_evaluate_ridge_probe_seeds_change_predictions():
    """Different seeds should produce different OOF predictions."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.standard_normal(60).astype(np.float32)
    a = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[42])
    b = fit_evaluate_ridge_probe(X, y, n_folds=5, seeds=[999])
    # Some predictions should differ (different KFold splits)
    assert not np.allclose(a.predictions, b.predictions)


# ---------------------------------------------------------------------------
# Task 6: ClinVar zero-shot scorer
# ---------------------------------------------------------------------------

from src.one_embedding.vep import score_clinvar_zeroshot, clinvar_auc


def test_score_clinvar_zeroshot_identical_embeddings():
    """If WT == mut at the position, score is 0."""
    L, D = 10, 8
    wt = np.ones((L, D), dtype=np.float32)
    mut = wt.copy()
    s = score_clinvar_zeroshot(wt_emb=wt, mut_emb=mut, mut_pos=3)
    assert s == pytest.approx(0.0, abs=1e-6)


def test_score_clinvar_zeroshot_orthogonal():
    """Orthogonal -> cosine=0 -> score=1."""
    L, D = 5, 4
    wt = np.zeros((L, D), dtype=np.float32)
    mut = np.zeros((L, D), dtype=np.float32)
    wt[2] = np.array([1, 0, 0, 0], dtype=np.float32)
    mut[2] = np.array([0, 1, 0, 0], dtype=np.float32)
    s = score_clinvar_zeroshot(wt_emb=wt, mut_emb=mut, mut_pos=2)
    assert s == pytest.approx(1.0)


def test_clinvar_auc_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    auc = clinvar_auc(scores, labels)
    assert auc == pytest.approx(1.0)


def test_clinvar_auc_random():
    rng = np.random.default_rng(42)
    scores = rng.random(1000)
    labels = rng.integers(0, 2, 1000)
    auc = clinvar_auc(scores, labels)
    assert 0.4 < auc < 0.6


def test_score_clinvar_zeroshot_zero_norm():
    """Zero embedding -> score=1 (undefined cosine, worst case)."""
    L, D = 5, 4
    wt = np.zeros((L, D), dtype=np.float32)
    mut = np.zeros((L, D), dtype=np.float32)
    s = score_clinvar_zeroshot(wt_emb=wt, mut_emb=mut, mut_pos=2)
    assert s == 1.0


def test_score_clinvar_zeroshot_pos_out_of_range():
    L, D = 5, 4
    wt = np.zeros((L, D), dtype=np.float32)
    mut = np.zeros((L, D), dtype=np.float32)
    with pytest.raises(IndexError):
        score_clinvar_zeroshot(wt_emb=wt, mut_emb=mut, mut_pos=L)


# ---------------------------------------------------------------------------
# Task 7: BCa bootstrap CI helpers
# ---------------------------------------------------------------------------

from src.one_embedding.vep import bootstrap_ci_paired, bootstrap_ci_pearson


def test_bootstrap_ci_paired_basic():
    """Non-degenerate paired retention: codec is roughly 95% of raw with assay-level variance."""
    raw = np.array([0.50, 0.60, 0.70, 0.55, 0.65])
    codec = np.array([0.46, 0.58, 0.66, 0.51, 0.63])  # ~92-96% retention with spread
    result = bootstrap_ci_paired(raw, codec, n_boot=2000, seed=42)
    assert "retention_pct" in result
    assert "ci_low" in result
    assert "ci_high" in result
    # Point estimate around 95%
    assert 90 < result["retention_pct"] < 100
    # CI should bracket the point estimate (with some slack — paired bootstrap
    # CI may be biased)
    assert result["ci_low"] < result["retention_pct"] < result["ci_high"]
    # CI should be tight given small spread (codec ~ 0.92*raw to 0.97*raw)
    assert (result["ci_high"] - result["ci_low"]) < 10  # less than 10pp wide


def test_bootstrap_ci_pearson_basic():
    rng = np.random.default_rng(0)
    n = 30
    x = rng.standard_normal(n)
    y = x + 0.1 * rng.standard_normal(n)  # strong positive correlation
    result = bootstrap_ci_pearson(x, y, n_boot=2000, seed=42)
    assert "pearson_r" in result
    assert "ci_low" in result
    assert "ci_high" in result
    assert result["pearson_r"] > 0.9
    assert result["ci_low"] < result["pearson_r"] < result["ci_high"]


def test_bootstrap_ci_paired_shape_mismatch():
    raw = np.array([0.5, 0.6, 0.7])
    codec = np.array([0.5, 0.6])  # length mismatch
    with pytest.raises(ValueError, match="paired arrays must match"):
        bootstrap_ci_paired(raw, codec, n_boot=100, seed=42)


def test_bootstrap_ci_pearson_shape_mismatch():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError, match="must match"):
        bootstrap_ci_pearson(x, y, n_boot=100, seed=42)


def test_bootstrap_ci_paired_perfect_retention():
    """codec == raw exactly -> retention=100%, tight CI."""
    raw = np.array([0.5, 0.6, 0.7, 0.55, 0.65])
    codec = raw.copy()
    result = bootstrap_ci_paired(raw, codec, n_boot=1000, seed=42)
    assert result["retention_pct"] == pytest.approx(100.0)
    # CI should be a tight singleton at 100% (BCa fall-through to percentile)
    assert result["ci_low"] == pytest.approx(100.0, abs=1e-6)
    assert result["ci_high"] == pytest.approx(100.0, abs=1e-6)
