"""Tests for .one.h5 file format — write/read/inspect with freeform tags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.one_embedding.io import (
    ONE_H5_FORMAT,
    ONE_H5_VERSION,
    _decode_attr,
    inspect_one_h5,
    read_one_h5,
    read_one_h5_batch,
    read_oemb,
    write_oemb,
    write_one_h5,
    write_one_h5_batch,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_protein(seq_len=50, d_out=512, vec_dim=2048):
    """Create a random protein data dict."""
    rng = np.random.default_rng(42)
    return {
        "per_residue": rng.standard_normal((seq_len, d_out)).astype(np.float32),
        "protein_vec": rng.standard_normal(vec_dim).astype(np.float32),
    }


def _make_protein_with_extras(seq_len=50, d_out=512, vec_dim=2048):
    """Create protein data with sequence and per-protein tags."""
    data = _make_protein(seq_len, d_out, vec_dim)
    data["sequence"] = "M" * seq_len
    data["tags"] = {"organism": "E.coli", "quality": 0.95}
    return data


# ── _decode_attr ────────────────────────────────────────────────────


class TestDecodeAttr:
    def test_bytes_to_str(self):
        assert _decode_attr(b"hello") == "hello"

    def test_np_int(self):
        result = _decode_attr(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float(self):
        result = _decode_attr(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.001

    def test_str_passthrough(self):
        assert _decode_attr("already_str") == "already_str"

    def test_int_passthrough(self):
        assert _decode_attr(42) == 42

    def test_none_passthrough(self):
        assert _decode_attr(None) is None


# ── Single protein write/read roundtrip ─────────────────────────────


class TestWriteReadSingle:
    def test_basic_roundtrip(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = _make_protein()
        write_one_h5(path, data, protein_id="P12345")

        result = read_one_h5(path)

        assert result["protein_id"] == "P12345"
        assert result["format"] == ONE_H5_FORMAT
        assert result["version"] == ONE_H5_VERSION
        np.testing.assert_array_almost_equal(
            result["per_residue"], data["per_residue"], decimal=5
        )
        # protein_vec stored as fp16, so allow wider tolerance
        np.testing.assert_array_almost_equal(
            result["protein_vec"],
            data["protein_vec"].astype(np.float16),
            decimal=2,
        )

    def test_per_residue_dtype_is_float32(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = _make_protein()
        write_one_h5(path, data)
        result = read_one_h5(path)
        assert result["per_residue"].dtype == np.float32

    def test_protein_vec_dtype_is_float16(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = _make_protein()
        write_one_h5(path, data)
        result = read_one_h5(path)
        assert result["protein_vec"].dtype == np.float16

    def test_default_protein_id(self, tmp_path):
        path = tmp_path / "test.one.h5"
        write_one_h5(path, _make_protein())
        result = read_one_h5(path)
        assert result["protein_id"] == "protein"

    def test_with_sequence(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = _make_protein(seq_len=10)
        data["sequence"] = "MKTAYIAKQR"
        write_one_h5(path, data, protein_id="seqtest")

        result = read_one_h5(path)
        assert result["sequence"] == "MKTAYIAKQR"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "test.one.h5"
        write_one_h5(path, _make_protein())
        assert path.exists()


# ── Freeform tags ───────────────────────────────────────────────────


class TestFreeformTags:
    def test_root_tags(self, tmp_path):
        path = tmp_path / "tags.one.h5"
        data = _make_protein()
        tags = {"source_model": "prot_t5", "codec": "ABTT3+RP512+PQ128", "version_int": 2}
        write_one_h5(path, data, tags=tags)

        result = read_one_h5(path)
        assert result["tags"]["source_model"] == "prot_t5"
        assert result["tags"]["codec"] == "ABTT3+RP512+PQ128"
        assert result["tags"]["version_int"] == 2

    def test_per_protein_tags(self, tmp_path):
        path = tmp_path / "ptags.one.h5"
        data = _make_protein()
        data["tags"] = {"organism": "H.sapiens", "quality_score": 0.92}
        write_one_h5(path, data, protein_id="Q9Y6K1")

        result = read_one_h5(path)
        assert result["tags"]["organism"] == "H.sapiens"
        assert abs(result["tags"]["quality_score"] - 0.92) < 0.001

    def test_combined_root_and_protein_tags(self, tmp_path):
        path = tmp_path / "combined.one.h5"
        data = _make_protein()
        data["tags"] = {"domain": "kinase"}
        root_tags = {"pipeline": "v2"}
        write_one_h5(path, data, tags=root_tags)

        result = read_one_h5(path)
        assert result["tags"]["pipeline"] == "v2"
        assert result["tags"]["domain"] == "kinase"

    def test_no_tags_is_fine(self, tmp_path):
        path = tmp_path / "notags.one.h5"
        write_one_h5(path, _make_protein())
        result = read_one_h5(path)
        assert isinstance(result["tags"], dict)
        # tags dict should be empty when no tags provided
        assert len(result["tags"]) == 0


# ── Variable d_out ──────────────────────────────────────────────────


class TestVariableDimensions:
    @pytest.mark.parametrize("d_out", [256, 512, 768, 1024])
    def test_variable_d_out(self, tmp_path, d_out):
        path = tmp_path / f"d{d_out}.one.h5"
        data = _make_protein(seq_len=30, d_out=d_out)
        write_one_h5(path, data)

        result = read_one_h5(path)
        assert result["per_residue"].shape == (30, d_out)

    @pytest.mark.parametrize("vec_dim", [512, 1024, 2048, 4096])
    def test_variable_vec_dim(self, tmp_path, vec_dim):
        path = tmp_path / f"v{vec_dim}.one.h5"
        data = _make_protein(vec_dim=vec_dim)
        write_one_h5(path, data)

        result = read_one_h5(path)
        assert result["protein_vec"].shape == (vec_dim,)


# ── Batch write/read ────────────────────────────────────────────────


class TestBatchWriteRead:
    def _make_batch(self, n=5, d_out=512):
        rng = np.random.default_rng(123)
        proteins = {}
        for i in range(n):
            seq_len = 30 + i * 10
            proteins[f"prot_{i}"] = {
                "per_residue": rng.standard_normal((seq_len, d_out)).astype(np.float32),
                "protein_vec": rng.standard_normal(2048).astype(np.float32),
            }
        return proteins

    def test_batch_roundtrip(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = self._make_batch()
        write_one_h5_batch(path, proteins)

        result = read_one_h5_batch(path)
        assert set(result.keys()) == set(proteins.keys())

        for pid in proteins:
            np.testing.assert_array_almost_equal(
                result[pid]["per_residue"],
                proteins[pid]["per_residue"],
                decimal=5,
            )

    def test_batch_with_root_tags(self, tmp_path):
        path = tmp_path / "batch_tags.one.h5"
        proteins = self._make_batch(n=3)
        tags = {"dataset": "SCOPe", "split": "train"}
        write_one_h5_batch(path, proteins, tags=tags)

        info = inspect_one_h5(path)
        assert info["tags"]["dataset"] == "SCOPe"
        assert info["tags"]["split"] == "train"

    def test_batch_with_per_protein_tags(self, tmp_path):
        path = tmp_path / "batch_ptags.one.h5"
        proteins = self._make_batch(n=2)
        proteins["prot_0"]["tags"] = {"family": "globin"}
        proteins["prot_1"]["tags"] = {"family": "kinase", "confidence": 0.99}
        write_one_h5_batch(path, proteins)

        # Read and verify via raw h5py that per-protein tags are stored
        import h5py

        with h5py.File(path, "r") as f:
            assert str(f["prot_0"].attrs["family"]) == "globin"
            assert str(f["prot_1"].attrs["family"]) == "kinase"
            assert abs(float(f["prot_1"].attrs["confidence"]) - 0.99) < 0.001

    def test_batch_with_sequences(self, tmp_path):
        path = tmp_path / "batch_seq.one.h5"
        proteins = self._make_batch(n=2)
        proteins["prot_0"]["sequence"] = "MKTAYIAKQR"
        proteins["prot_1"]["sequence"] = "ACDEFGHIKLMNPQRSTVWY"
        write_one_h5_batch(path, proteins)

        import h5py

        with h5py.File(path, "r") as f:
            assert str(f["prot_0"].attrs["sequence"]) == "MKTAYIAKQR"
            assert str(f["prot_1"].attrs["sequence"]) == "ACDEFGHIKLMNPQRSTVWY"


# ── Batch subset loading ────────────────────────────────────────────


class TestBatchSubset:
    def _make_batch(self, n=5):
        rng = np.random.default_rng(99)
        proteins = {}
        for i in range(n):
            proteins[f"prot_{i}"] = {
                "per_residue": rng.standard_normal((20, 512)).astype(np.float32),
                "protein_vec": rng.standard_normal(2048).astype(np.float32),
            }
        return proteins

    def test_load_subset(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = self._make_batch()
        write_one_h5_batch(path, proteins)

        result = read_one_h5_batch(path, protein_ids=["prot_1", "prot_3"])
        assert set(result.keys()) == {"prot_1", "prot_3"}

    def test_load_single_from_batch(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = self._make_batch()
        write_one_h5_batch(path, proteins)

        result = read_one_h5_batch(path, protein_ids=["prot_2"])
        assert list(result.keys()) == ["prot_2"]

    def test_missing_protein_raises(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = self._make_batch(n=2)
        write_one_h5_batch(path, proteins)

        with pytest.raises(KeyError, match="not_here"):
            read_one_h5_batch(path, protein_ids=["not_here"])

    def test_load_none_means_all(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = self._make_batch(n=3)
        write_one_h5_batch(path, proteins)

        result = read_one_h5_batch(path, protein_ids=None)
        assert len(result) == 3


# ── Inspect ─────────────────────────────────────────────────────────


class TestInspect:
    def test_inspect_single(self, tmp_path):
        path = tmp_path / "single.one.h5"
        data = _make_protein(seq_len=100, d_out=512, vec_dim=2048)
        write_one_h5(path, data, protein_id="ABC123", tags={"model": "prot_t5"})

        info = inspect_one_h5(path)

        assert info["format"] == ONE_H5_FORMAT
        assert info["version"] == ONE_H5_VERSION
        assert info["n_proteins"] == 1
        assert info["d_out"] == 512
        assert info["protein_vec_dim"] == 2048
        assert info["protein_ids"] == ["ABC123"]
        assert info["tags"]["model"] == "prot_t5"
        assert info["size_bytes"] > 0

    def test_inspect_batch(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        rng = np.random.default_rng(7)
        proteins = {
            f"p{i}": {
                "per_residue": rng.standard_normal((20, 256)).astype(np.float32),
                "protein_vec": rng.standard_normal(1024).astype(np.float32),
            }
            for i in range(4)
        }
        write_one_h5_batch(path, proteins, tags={"dataset": "test"})

        info = inspect_one_h5(path)

        assert info["n_proteins"] == 4
        assert info["d_out"] == 256
        assert info["protein_vec_dim"] == 1024
        assert sorted(info["protein_ids"]) == ["p0", "p1", "p2", "p3"]
        assert info["tags"]["dataset"] == "test"

    def test_inspect_no_data_loaded(self, tmp_path):
        """Inspect should not load actual embedding arrays."""
        path = tmp_path / "big.one.h5"
        data = _make_protein(seq_len=500, d_out=1024, vec_dim=4096)
        write_one_h5(path, data)

        info = inspect_one_h5(path)
        # Just verify it returns shape info, not data
        assert info["d_out"] == 1024
        assert info["protein_vec_dim"] == 4096
        # No 'per_residue' key in result
        assert "per_residue" not in info


# ── Backward compatibility ──────────────────────────────────────────


class TestBackwardCompat:
    def test_old_oemb_still_readable(self, tmp_path):
        """Old .oemb files written with write_oemb can still be read via read_oemb."""
        path = tmp_path / "legacy.oemb"
        data = {
            "per_residue": np.random.randn(30, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float32),
            "sequence": "MKTAYIAKQR" * 3,
            "source_model": "prot_t5_xl",
            "codec": "ABTT3+RP512",
        }
        write_oemb(path, data, protein_id="legacy_prot")

        result = read_oemb(path)
        assert result["protein_id"] == "legacy_prot"
        assert result["oemb_version"] == "1.0"
        assert result["source_model"] == "prot_t5_xl"
        np.testing.assert_array_almost_equal(
            result["per_residue"], data["per_residue"], decimal=5
        )

    def test_read_one_h5_falls_back_for_oemb(self, tmp_path):
        """read_one_h5 should fall back to read_oemb for old format files."""
        path = tmp_path / "old.oemb"
        data = {
            "per_residue": np.random.randn(25, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float32),
        }
        write_oemb(path, data, protein_id="fallback_prot")

        # read_one_h5 should detect missing format attr and fall back
        result = read_one_h5(path)
        assert result["protein_id"] == "fallback_prot"
        assert result["oemb_version"] == "1.0"
        np.testing.assert_array_almost_equal(
            result["per_residue"], data["per_residue"], decimal=5
        )


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_residue_protein(self, tmp_path):
        path = tmp_path / "single_res.one.h5"
        data = _make_protein(seq_len=1, d_out=512)
        write_one_h5(path, data)
        result = read_one_h5(path)
        assert result["per_residue"].shape == (1, 512)

    def test_very_long_protein(self, tmp_path):
        path = tmp_path / "long.one.h5"
        data = _make_protein(seq_len=2000, d_out=512)
        write_one_h5(path, data)
        result = read_one_h5(path)
        assert result["per_residue"].shape == (2000, 512)

    def test_empty_batch(self, tmp_path):
        path = tmp_path / "empty_batch.one.h5"
        write_one_h5_batch(path, {})

        result = read_one_h5_batch(path)
        assert result == {}

        info = inspect_one_h5(path)
        assert info["n_proteins"] == 0
        assert info["protein_ids"] == []
        assert info["d_out"] is None
        assert info["protein_vec_dim"] is None

    def test_numeric_tags(self, tmp_path):
        """Tags with int and float values roundtrip correctly."""
        path = tmp_path / "num_tags.one.h5"
        data = _make_protein()
        tags = {"epoch": 42, "loss": 0.001, "name": "experiment_1"}
        write_one_h5(path, data, tags=tags)

        result = read_one_h5(path)
        assert result["tags"]["epoch"] == 42
        assert abs(result["tags"]["loss"] - 0.001) < 1e-6
        assert result["tags"]["name"] == "experiment_1"
