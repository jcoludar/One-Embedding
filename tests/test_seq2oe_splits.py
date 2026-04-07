"""Tests for Seq2OE CATH-level cluster splits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.seq2oe_splits import (
    parse_cath_fasta,
    cath_cluster_split,
    save_split,
    load_split,
)


SAMPLE_FASTA = """>12asA00|3.30.930.10
MKTAYIAKQRQIS
>132lA00|1.10.530.10
XVFGRCELAAAM
>153lA00|1.10.530.10
RTDCYGNVNRIDT
>16pkA02|3.40.50.1260
YFAKVLGNPPRP
>16vpA00|1.10.1290.10
SRMPSPPMPVPP
>1914A00|3.30.720.10
MVLLESEQFL
"""


class TestParseCathFasta:
    def test_parses_ids_and_codes(self, tmp_path):
        f = tmp_path / "cath.fa"
        f.write_text(SAMPLE_FASTA)
        meta = parse_cath_fasta(f)
        assert set(meta.keys()) == {
            "12asA00", "132lA00", "153lA00",
            "16pkA02", "16vpA00", "1914A00",
        }
        assert meta["12asA00"]["seq"] == "MKTAYIAKQRQIS"
        assert meta["12asA00"]["C"] == 3
        assert meta["12asA00"]["A"] == 30
        assert meta["12asA00"]["T"] == "3.30.930"
        assert meta["12asA00"]["H"] == "3.30.930.10"

    def test_rejects_malformed_header(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">noCode\nAAAA\n")
        with pytest.raises(ValueError, match="no CATH code"):
            parse_cath_fasta(f)

    def test_rejects_malformed_code(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">pid|1.2.3\nAAAA\n")  # only 3 levels
        with pytest.raises(ValueError, match="expected 4 dot-separated"):
            parse_cath_fasta(f)


class TestCathClusterSplit:
    def _make_meta(self):
        """10 proteins in 2 classes, 6 H-codes, 4 T-codes."""
        return {
            # Class 1, T=1.10.530, H=1.10.530.10 (2 proteins)
            "p1": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.10"},
            "p2": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.10"},
            # Class 1, T=1.10.530, H=1.10.530.20 (2 proteins)
            "p3": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.20"},
            "p4": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.20"},
            # Class 1, T=1.10.290, H=1.10.290.10 (1 protein)
            "p5": {"seq": "A", "C": 1, "A": 10, "T": "1.10.290", "H": "1.10.290.10"},
            # Class 3, T=3.30.930, H=3.30.930.10 (3 proteins)
            "p6": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            "p7": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            "p8": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            # Class 3, T=3.30.720, H=3.30.720.10 (2 proteins)
            "p9": {"seq": "A", "C": 3, "A": 30, "T": "3.30.720", "H": "3.30.720.10"},
            "p10": {"seq": "A", "C": 3, "A": 30, "T": "3.30.720", "H": "3.30.720.10"},
        }

    def test_h_split_no_cluster_leakage(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        train_h = {meta[p]["H"] for p in train}
        val_h = {meta[p]["H"] for p in val}
        test_h = {meta[p]["H"] for p in test}
        assert train_h.isdisjoint(val_h)
        assert train_h.isdisjoint(test_h)
        assert val_h.isdisjoint(test_h)

    def test_t_split_no_cluster_leakage(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="T", fractions=(0.6, 0.2, 0.2), seed=42
        )
        train_t = {meta[p]["T"] for p in train}
        val_t = {meta[p]["T"] for p in val}
        test_t = {meta[p]["T"] for p in test}
        assert train_t.isdisjoint(val_t)
        assert train_t.isdisjoint(test_t)
        assert val_t.isdisjoint(test_t)

    def test_every_protein_assigned_exactly_once(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        all_assigned = set(train) | set(val) | set(test)
        assert all_assigned == set(meta.keys())
        assert len(train) + len(val) + len(test) == len(meta)

    def test_deterministic_same_seed(self):
        meta = self._make_meta()
        r1 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        r2 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        meta = self._make_meta()
        r1 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        r2 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=7)
        # With 6 H-codes and small counts they may occasionally collide;
        # assert at least one fold differs
        assert r1 != r2

    def test_invalid_level_raises(self):
        meta = self._make_meta()
        with pytest.raises(ValueError, match="level must be"):
            cath_cluster_split(meta, level="X", fractions=(0.6, 0.2, 0.2), seed=42)

    def test_fractions_must_sum_to_one(self):
        meta = self._make_meta()
        with pytest.raises(ValueError, match="must sum to 1"):
            cath_cluster_split(meta, level="H", fractions=(0.5, 0.2, 0.2), seed=42)

    def test_returns_sorted_lists(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)

    def test_independent_of_dict_insertion_order(self):
        """Same metadata in different insertion order must yield identical splits."""
        meta = self._make_meta()
        reversed_meta = dict(reversed(list(meta.items())))
        r1 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        r2 = cath_cluster_split(reversed_meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        assert r1 == r2

    def test_class_stratification_both_classes_represented(self):
        """With enough clusters, both classes should appear in every fold."""
        # Build a bigger synthetic set: 20 H-codes in class 1, 20 in class 3
        meta = {}
        for i in range(20):
            meta[f"c1p{i}"] = {
                "seq": "A", "C": 1, "A": 10,
                "T": f"1.10.{i}", "H": f"1.10.{i}.10",
            }
        for i in range(20):
            meta[f"c3p{i}"] = {
                "seq": "A", "C": 3, "A": 30,
                "T": f"3.30.{i}", "H": f"3.30.{i}.10",
            }
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        for fold, name in [(train, "train"), (val, "val"), (test, "test")]:
            classes = {meta[p]["C"] for p in fold}
            assert classes == {1, 3}, f"{name} missing a class: {classes}"


class TestSplitIO:
    def test_roundtrip(self, tmp_path):
        train = ["p1", "p2"]
        val = ["p3"]
        test = ["p4", "p5"]
        meta_info = {"level": "H", "seed": 42, "fractions": [0.6, 0.2, 0.2]}

        path = tmp_path / "split.json"
        save_split(path, train, val, test, meta_info)
        assert path.exists()

        loaded_train, loaded_val, loaded_test, loaded_meta = load_split(path)
        assert loaded_train == train
        assert loaded_val == val
        assert loaded_test == test
        assert loaded_meta == meta_info

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "split.json"
        save_split(path, ["p1"], ["p2"], ["p3"], {"level": "T", "seed": 0})
        assert path.exists()
