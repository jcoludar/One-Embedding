"""Unit tests for src/oe_autoeval/retention.py (multi-seed cluster-paired bootstrap)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.oe_autoeval.retention import accuracy, spearman, paired_retention


def test_metrics():
    assert np.isclose(accuracy([0, 1, 2, 1], [0, 1, 2, 0]), 0.75)
    assert spearman([1, 2, 3, 4], [1, 2, 3, 5]) > 0.9
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0  # constant -> 0, not NaN


def test_identical_preds_ratio_one_zero_sd():
    rng = np.random.RandomState(0)
    N = 200
    y = rng.randint(0, 3, N)
    preds = np.stack([y.copy() for _ in range(5)])
    ids = list(range(N))
    res = paired_retention(ids, ids, y, preds, preds, metric="accuracy", n_boot=300, seed=42)
    assert np.isclose(res["ratio"], 1.0)
    assert res["ci_low"] <= 1.0 <= res["ci_high"]
    assert res["between_seed_sd"] == 0.0
    assert res["n_groups"] == N            # group_ids=None -> each item its own group


def test_consistent_gap_excludes_one_small_sd():
    rng = np.random.RandomState(1)
    N = 400
    y = rng.randint(0, 2, N)
    raw = np.stack([y.copy() for _ in range(5)])
    oe_one = y.copy()
    oe_one[:60] ^= 1                                       # ~15% worse, identical per seed
    oe = np.stack([oe_one.copy() for _ in range(5)])
    ids = list(range(N))
    res = paired_retention(ids, ids, y, raw, oe, metric="accuracy", n_boot=500, seed=42)
    assert 0.8 < res["ratio"] < 0.9
    assert res["ci_high"] < 1.0                            # consistent deficit -> excludes 1
    assert res["between_seed_sd"] < 0.01
    assert res["delta"] < 0
    assert res["delta_ci"][1] < 0                          # BCa Δ CI also excludes 0


def test_outlier_seed_inflates_between_seed_sd():
    # With seed-averaging in the bootstrap, one flaky seed shows up as a LARGE between_seed_sd
    # (the probe-noise floor), not as a wide item CI. This is the intended decomposition.
    rng = np.random.RandomState(2)
    N = 400
    y = rng.randint(0, 2, N)
    raw = np.stack([y.copy() for _ in range(5)])
    good = y.copy()
    bad = y.copy()
    bad[:160] ^= 1                                         # one bad seed (40% wrong)
    oe = np.stack([good, good, good, good, bad])
    ids = list(range(N))
    res = paired_retention(ids, ids, y, raw, oe, metric="accuracy", n_boot=600, seed=42)
    assert res["between_seed_sd"] > 0.1                    # seed variance exposed separately
    assert 0.85 < res["ratio"] < 0.95                      # mean over the 5 seed ratios


def test_cluster_bootstrap_widens_ci_vs_iid():
    # 20 proteins x 20 residues; OE is all-right or all-wrong PER PROTEIN (block-correlated).
    # i.i.d. residue resampling underestimates variance; cluster (protein) resampling reflects
    # the true between-protein variability -> wider CI.
    n_prot, per = 20, 20
    N = n_prot * per
    rng = np.random.RandomState(3)
    y = rng.randint(0, 2, N)
    raw = np.stack([y.copy() for _ in range(3)])
    oe_one = y.copy()
    groups = np.repeat(np.arange(n_prot), per)
    for p in range(n_prot):
        if p >= n_prot // 2:                               # second half: OE wrong on whole protein
            oe_one[groups == p] ^= 1
    oe = np.stack([oe_one.copy() for _ in range(3)])
    ids = list(range(N))

    iid = paired_retention(ids, ids, y, raw, oe, group_ids=None, n_boot=600, seed=42)
    clustered = paired_retention(ids, ids, y, raw, oe, group_ids=groups, n_boot=600, seed=42)
    iid_w = iid["ci_high"] - iid["ci_low"]
    clu_w = clustered["ci_high"] - clustered["ci_low"]
    assert clustered["n_groups"] == n_prot
    assert clu_w > 2 * iid_w                               # cluster CI is much wider


def test_mismatched_item_ids_raises():
    N = 10
    y = np.zeros(N, int)
    preds = np.zeros((2, N), int)
    with pytest.raises(ValueError, match="identical test items"):
        paired_retention(list(range(N)), list(range(1, N + 1)), y, preds, preds)


def test_shape_mismatch_raises():
    y = np.zeros(2, int)
    with pytest.raises(ValueError, match="must both be"):
        paired_retention([0, 1], [0, 1], y, np.zeros((3, 2), int), np.zeros((2, 2), int))


def test_spearman_metric_runs():
    rng = np.random.RandomState(4)
    N = 150
    y = rng.randn(N)
    raw = np.stack([y + rng.randn(N) * 0.1 for _ in range(4)])
    oe = np.stack([y + rng.randn(N) * 0.5 for _ in range(4)])
    ids = list(range(N))
    res = paired_retention(ids, ids, y, raw, oe, metric="spearman", n_boot=200, seed=42)
    assert 0.0 < res["ratio"] < 1.0
    assert res["n_seeds"] == 4
