"""Tests for the DeepLoc localization benchmark runner (Exp 43, runners/localization.py).

Covers the bugs fixed while reviewing PR #2:
- config.py must keep CROSS_CHECK_* thresholds (phase B/C import them).
- _to_protein_vector DCT path must yield a constant-dim vector even for proteins
  shorter than the requested number of coefficients (no ragged arrays).
- feature_method name is authoritative for the number of DCT coefficients.
- a single, memory-safe codec feature extractor shared by both run paths.
Plus regression coverage for the CSV parser and the F1 cluster statistics.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest

from runners.localization import (
    _to_protein_vector,
    parse_deeploc_metadata_csv,
    macro_f1_from_clusters,
    _per_class_f1_from_clusters,
    extract_protein_features_from_codec,
    run_localization_probe_benchmark,
)


class TestConfigThresholdsPresent:
    """The PR comment-deleted these; run_phase_b / run_phase_c import them at module load."""

    def test_cross_check_constants_defined(self):
        import config
        assert config.CROSS_CHECK_WARN_PP == 3.0
        assert config.CROSS_CHECK_BLOCK_PP == 5.0


class TestToProteinVector:

    def test_mean_shape(self):
        x = np.random.randn(50, 64).astype(np.float32)
        v = _to_protein_vector(x, method="mean")
        assert v.shape == (64,)

    def test_dct_k4_shape(self):
        x = np.random.randn(50, 64).astype(np.float32)
        v = _to_protein_vector(x, method="dct_k4")
        assert v.shape == (64 * 4,)

    def test_dct_k8_shape_from_method_name(self):
        # method name alone must select k=8 — no dct_k arg threaded through.
        x = np.random.randn(50, 64).astype(np.float32)
        v = _to_protein_vector(x, method="dct_k8")
        assert v.shape == (64 * 8,)

    def test_dct_short_protein_keeps_constant_dim(self):
        # L=2 < k=4: must zero-pad to (4*D,), not produce a ragged (2*D,) vector.
        short = np.random.randn(2, 64).astype(np.float32)
        v = _to_protein_vector(short, method="dct_k4")
        assert v.shape == (64 * 4,)

    def test_dct_mixed_lengths_stack_into_matrix(self):
        # The real failure mode: np.array over proteins of different L must not go ragged.
        embs = [np.random.randn(L, 16).astype(np.float32) for L in (1, 3, 40, 200)]
        vecs = np.array([_to_protein_vector(e, method="dct_k4") for e in embs])
        assert vecs.shape == (4, 16 * 4)
        assert vecs.dtype != object

    def test_one_dim_passthrough(self):
        v = np.random.randn(128).astype(np.float32)
        assert np.allclose(_to_protein_vector(v, method="mean"), v)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="[Uu]nknown"):
            _to_protein_vector(np.random.randn(10, 8).astype(np.float32), method="bogus")


class TestParseDeeplocMetadataCsv:

    def _write_csv(self, path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ACC", "localization", "membrane_type", "split", "length"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def test_splits_route_train_and_test(self, tmp_path):
        p = tmp_path / "meta.csv"
        self._write_csv(p, [
            {"ACC": "A", "localization": "Nucleus", "membrane_type": "S", "split": "train", "length": 100},
            {"ACC": "B", "localization": "Cytoplasm", "membrane_type": "S", "split": "test", "length": 120},
        ])
        train, test = parse_deeploc_metadata_csv(p)
        assert train == {"A": "Nucleus"}
        assert test == {"B": "Cytoplasm"}

    def test_max_length_filters(self, tmp_path):
        p = tmp_path / "meta.csv"
        self._write_csv(p, [
            {"ACC": "A", "localization": "Nucleus", "membrane_type": "S", "split": "train", "length": 100},
            {"ACC": "B", "localization": "Nucleus", "membrane_type": "S", "split": "train", "length": 5000},
        ])
        train, _ = parse_deeploc_metadata_csv(p, max_length=2000)
        assert "A" in train and "B" not in train

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ACC", "split"])  # no localization
            w.writeheader()
            w.writerow({"ACC": "A", "split": "train"})
        with pytest.raises(ValueError, match="localization"):
            parse_deeploc_metadata_csv(p)


class TestClusterF1:

    def test_macro_f1_perfect(self):
        clusters = [{"y_true": c, "y_pred": c} for c in ("a", "a", "b", "b", "c")]
        assert macro_f1_from_clusters(clusters) == pytest.approx(1.0)

    def test_macro_f1_all_wrong(self):
        clusters = [{"y_true": "a", "y_pred": "b"}, {"y_true": "b", "y_pred": "a"}]
        assert macro_f1_from_clusters(clusters) == pytest.approx(0.0)

    def test_per_class_f1_keys_cover_seen_labels(self):
        clusters = [{"y_true": "a", "y_pred": "a"}, {"y_true": "b", "y_pred": "c"}]
        per = _per_class_f1_from_clusters(clusters)
        assert set(per) == {"a", "b", "c"}


class _IdentityCodec:
    """Stub codec: encode is a passthrough, decode returns the same residue matrix.

    Lets us test the extractor's memory-safe plumbing without the heavy real codec.
    """

    def encode(self, emb):
        return {"emb": emb}

    def decode_per_residue(self, enc):
        return enc["emb"]


class TestExtractProteinFeaturesFromCodec:

    @pytest.mark.parametrize("method", ["mean", "dct_k4", "dct_k8"])
    def test_features_match_direct_pool(self, method):
        # Fairness invariant: the compressed path (extractor) and the raw baseline
        # path (run_localization_probe_benchmark pools residue-level input via
        # _to_protein_vector) must build IDENTICAL features for the same input, so
        # retention is not an artifact of mismatched feature construction. With an
        # identity codec the only difference left is the compression itself.
        raw = {"p1": np.random.randn(30, 16).astype(np.float32),
               "p2": np.random.randn(12, 16).astype(np.float32)}
        feats = extract_protein_features_from_codec(raw, _IdentityCodec(), method)
        assert set(feats) == {"p1", "p2"}
        for pid in raw:
            assert np.allclose(feats[pid], _to_protein_vector(raw[pid], method=method))

    @pytest.mark.parametrize("method,k", [("dct_k4", 4), ("dct_k8", 8)])
    def test_dct_features_constant_dim_across_lengths(self, method, k):
        raw = {"short": np.random.randn(2, 16).astype(np.float32),
               "long": np.random.randn(80, 16).astype(np.float32)}
        feats = extract_protein_features_from_codec(raw, _IdentityCodec(), method)
        assert feats["short"].shape == feats["long"].shape == (16 * k,)


class TestRunLocalizationProbeBenchmark:

    def _synthetic(self, n_per_class_train=8, n_per_class_test=4, classes=("a", "b", "c"), D=24):
        rng = np.random.RandomState(0)
        embeddings, train_labels, test_labels = {}, {}, {}
        for ci, cls in enumerate(classes):
            center = np.zeros(D, dtype=np.float32)
            center[ci * 6:(ci + 1) * 6] = 8.0
            for j in range(n_per_class_train):
                pid = f"tr_{cls}_{j}"
                embeddings[pid] = (center + rng.randn(20, D).astype(np.float32) * 0.05)
                train_labels[pid] = cls
            for j in range(n_per_class_test):
                pid = f"te_{cls}_{j}"
                embeddings[pid] = (center + rng.randn(20, D).astype(np.float32) * 0.05)
                test_labels[pid] = cls
        return embeddings, train_labels, test_labels

    def test_separable_classes_high_q10(self):
        embeddings, train_labels, test_labels = self._synthetic()
        res = run_localization_probe_benchmark(
            embeddings=embeddings, train_labels=train_labels, test_labels=test_labels,
            test_name="synthetic", C_grid=[1.0], cv_folds=2, seeds=[42], n_bootstrap=50,
            feature_method="mean",
        )
        assert res["status"] == "done"
        assert res["q10"].value > 0.8
        assert 0.0 <= res["macro_f1"].value <= 1.0
        assert set(res["per_protein_scores"]) and set(res["per_protein_predictions"])

    @pytest.mark.parametrize("method", ["dct_k4", "dct_k8"])
    def test_dct_method_runs_end_to_end_with_mixed_lengths(self, method):
        # Exercises the originally-ragged path through the probe: proteins shorter
        # than k must still stack into a rectangular feature matrix.
        rng = np.random.RandomState(1)
        embeddings, train_labels, test_labels = {}, {}, {}
        for ci, cls in enumerate(("a", "b", "c")):
            for j in range(8):
                L = 2 if j % 4 == 0 else 25  # some proteins shorter than k=4/8
                pid = f"tr_{cls}_{j}"
                m = rng.randn(L, 24).astype(np.float32)
                m[:, ci * 6:(ci + 1) * 6] += 6.0
                embeddings[pid] = m
                train_labels[pid] = cls
            for j in range(4):
                pid = f"te_{cls}_{j}"
                m = rng.randn(25, 24).astype(np.float32)
                m[:, ci * 6:(ci + 1) * 6] += 6.0
                embeddings[pid] = m
                test_labels[pid] = cls
        res = run_localization_probe_benchmark(
            embeddings=embeddings, train_labels=train_labels, test_labels=test_labels,
            test_name="mixed", C_grid=[1.0], cv_folds=2, seeds=[42], n_bootstrap=30,
            feature_method=method,
        )
        assert res["status"] == "done"

    def test_insufficient_data_skips(self):
        res = run_localization_probe_benchmark(
            embeddings={"a": np.random.randn(10, 8).astype(np.float32)},
            train_labels={"a": "x"}, test_labels={"a": "x"},
            test_name="tiny", C_grid=[1.0], cv_folds=2, seeds=[42], n_bootstrap=10,
        )
        assert res["status"] == "skipped"
