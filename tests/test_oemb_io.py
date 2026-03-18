"""Tests for the .oemb file format (H5-based single- and batch-protein I/O).

Covers write_oemb, read_oemb, write_oemb_batch, read_oemb_batch, inspect_oemb.
Receiver requires only h5py + numpy — no codec code.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.io import (
    OEMB_VERSION,
    inspect_oemb,
    read_oemb,
    read_oemb_batch,
    write_oemb,
    write_oemb_batch,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.RandomState(7)


def _make_data(L: int = 20, D: int = 512, V: int = 2048, **extra) -> dict:
    """Create a minimal protein data dict for testing."""
    return {
        "per_residue": RNG.randn(L, D).astype(np.float32),
        "protein_vec": RNG.randn(V).astype(np.float16),
        **extra,
    }


# ── Single-protein roundtrip ──────────────────────────────────────────────────


class TestWriteReadSingle:
    def test_per_residue_roundtrip(self, tmp_path):
        data = _make_data(L=20)
        p = tmp_path / "prot.oemb"
        write_oemb(p, data, protein_id="prot_A")
        loaded = read_oemb(p)
        np.testing.assert_array_almost_equal(
            loaded["per_residue"], data["per_residue"], decimal=5
        )

    def test_protein_vec_roundtrip(self, tmp_path):
        data = _make_data()
        p = tmp_path / "prot.oemb"
        write_oemb(p, data, protein_id="prot_B")
        loaded = read_oemb(p)
        # protein_vec is stored as fp16; compare with fp16 original
        np.testing.assert_array_equal(
            loaded["protein_vec"], data["protein_vec"].astype(np.float16)
        )

    def test_metadata_roundtrip(self, tmp_path):
        data = _make_data(
            sequence="ACDEFGH",
            source_model="prot_t5_xl",
            codec="ABTT3+RP512+PQ128",
        )
        p = tmp_path / "meta.oemb"
        write_oemb(p, data, protein_id="prot_C")
        loaded = read_oemb(p)

        assert loaded["protein_id"] == "prot_C"
        assert loaded["sequence"] == "ACDEFGH"
        assert loaded["source_model"] == "prot_t5_xl"
        assert loaded["codec"] == "ABTT3+RP512+PQ128"
        assert loaded["oemb_version"] == OEMB_VERSION

    def test_dtypes(self, tmp_path):
        data = _make_data()
        p = tmp_path / "dtypes.oemb"
        write_oemb(p, data, protein_id="x")
        loaded = read_oemb(p)
        assert loaded["per_residue"].dtype == np.float32
        assert loaded["protein_vec"].dtype == np.float16

    def test_shapes(self, tmp_path):
        data = _make_data(L=35, D=512, V=2048)
        p = tmp_path / "shapes.oemb"
        write_oemb(p, data, protein_id="s")
        loaded = read_oemb(p)
        assert loaded["per_residue"].shape == (35, 512)
        assert loaded["protein_vec"].shape == (2048,)

    def test_optional_fields_default_empty(self, tmp_path):
        """When optional fields are omitted they default to empty string."""
        data = _make_data()  # no sequence / source_model / codec
        p = tmp_path / "no_meta.oemb"
        write_oemb(p, data)
        loaded = read_oemb(p)
        assert loaded["sequence"] == ""
        assert loaded["source_model"] == ""
        assert loaded["codec"] == ""


# ── Batch roundtrip ───────────────────────────────────────────────────────────


class TestWriteReadBatch:
    def _batch_proteins(self) -> dict[str, dict]:
        return {
            "prot_1": _make_data(L=10, sequence="ACDE"),
            "prot_2": _make_data(L=25, sequence="FGHIK"),
            "prot_3": _make_data(L=5, sequence="LMNP"),
        }

    def test_all_keys_present(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        assert set(loaded.keys()) == set(proteins.keys())

    def test_per_residue_shapes(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        for pid, data in proteins.items():
            expected_shape = data["per_residue"].shape
            assert loaded[pid]["per_residue"].shape == expected_shape

    def test_per_residue_values(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        for pid, data in proteins.items():
            np.testing.assert_array_almost_equal(
                loaded[pid]["per_residue"], data["per_residue"], decimal=5
            )

    def test_protein_vec_values(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        for pid, data in proteins.items():
            np.testing.assert_array_equal(
                loaded[pid]["protein_vec"],
                data["protein_vec"].astype(np.float16),
            )

    def test_sequence_roundtrip(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        assert loaded["prot_1"]["sequence"] == "ACDE"
        assert loaded["prot_2"]["sequence"] == "FGHIK"

    def test_selective_load(self, tmp_path):
        proteins = self._batch_proteins()
        p = tmp_path / "batch.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p, protein_ids=["prot_1", "prot_3"])
        assert set(loaded.keys()) == {"prot_1", "prot_3"}

    def test_dtypes_batch(self, tmp_path):
        proteins = {"p": _make_data()}
        p = tmp_path / "dt.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        assert loaded["p"]["per_residue"].dtype == np.float32
        assert loaded["p"]["protein_vec"].dtype == np.float16


# ── Raw H5 validation ─────────────────────────────────────────────────────────


class TestIsValidH5:
    def test_single_has_required_datasets(self, tmp_path):
        data = _make_data()
        p = tmp_path / "raw.oemb"
        write_oemb(p, data, protein_id="raw_p")
        with h5py.File(p, "r") as f:
            assert "per_residue" in f
            assert "protein_vec" in f

    def test_single_has_version_attr(self, tmp_path):
        data = _make_data()
        p = tmp_path / "ver.oemb"
        write_oemb(p, data)
        with h5py.File(p, "r") as f:
            assert "oemb_version" in f.attrs
            assert f.attrs["oemb_version"] == OEMB_VERSION

    def test_batch_has_groups_with_datasets(self, tmp_path):
        proteins = {"a": _make_data(), "b": _make_data()}
        p = tmp_path / "batch_raw.oemb"
        write_oemb_batch(p, proteins)
        with h5py.File(p, "r") as f:
            for pid in ["a", "b"]:
                assert pid in f
                assert "per_residue" in f[pid]
                assert "protein_vec" in f[pid]

    def test_batch_root_n_proteins_attr(self, tmp_path):
        proteins = {"x": _make_data(), "y": _make_data()}
        p = tmp_path / "np.oemb"
        write_oemb_batch(p, proteins)
        with h5py.File(p, "r") as f:
            assert "n_proteins" in f.attrs
            assert int(f.attrs["n_proteins"]) == 2

    def test_per_residue_is_compressed(self, tmp_path):
        data = _make_data(L=50)
        p = tmp_path / "compressed.oemb"
        write_oemb(p, data)
        with h5py.File(p, "r") as f:
            ds = f["per_residue"]
            assert ds.compression == "gzip"


# ── inspect_oemb ──────────────────────────────────────────────────────────────


class TestInspectSingle:
    def test_file_type(self, tmp_path):
        data = _make_data(L=30, source_model="prot_t5_xl")
        p = tmp_path / "s.oemb"
        write_oemb(p, data, protein_id="pid_x")
        info = inspect_oemb(p)
        assert info["file_type"] == "single"

    def test_n_residues(self, tmp_path):
        data = _make_data(L=42)
        p = tmp_path / "nr.oemb"
        write_oemb(p, data, protein_id="q")
        info = inspect_oemb(p)
        assert info["n_residues"] == 42

    def test_n_proteins(self, tmp_path):
        data = _make_data()
        p = tmp_path / "np.oemb"
        write_oemb(p, data, protein_id="pq")
        info = inspect_oemb(p)
        assert info["n_proteins"] == 1

    def test_source_model(self, tmp_path):
        data = _make_data(source_model="esm2_650m")
        p = tmp_path / "sm.oemb"
        write_oemb(p, data)
        info = inspect_oemb(p)
        assert info["source_model"] == "esm2_650m"

    def test_protein_ids_list(self, tmp_path):
        data = _make_data()
        p = tmp_path / "pl.oemb"
        write_oemb(p, data, protein_id="my_prot")
        info = inspect_oemb(p)
        assert info["protein_ids"] == ["my_prot"]

    def test_size_bytes_positive(self, tmp_path):
        data = _make_data(L=10)
        p = tmp_path / "sz.oemb"
        write_oemb(p, data)
        info = inspect_oemb(p)
        assert info["size_bytes"] > 0

    def test_version(self, tmp_path):
        data = _make_data()
        p = tmp_path / "v.oemb"
        write_oemb(p, data)
        info = inspect_oemb(p)
        assert info["oemb_version"] == OEMB_VERSION


class TestInspectBatch:
    def test_file_type(self, tmp_path):
        proteins = {"a": _make_data(), "b": _make_data()}
        p = tmp_path / "b.oemb"
        write_oemb_batch(p, proteins)
        info = inspect_oemb(p)
        assert info["file_type"] == "batch"

    def test_n_proteins(self, tmp_path):
        proteins = {f"p{i}": _make_data() for i in range(5)}
        p = tmp_path / "5p.oemb"
        write_oemb_batch(p, proteins)
        info = inspect_oemb(p)
        assert info["n_proteins"] == 5

    def test_protein_ids_present(self, tmp_path):
        proteins = {"alpha": _make_data(), "beta": _make_data()}
        p = tmp_path / "ids.oemb"
        write_oemb_batch(p, proteins)
        info = inspect_oemb(p)
        assert set(info["protein_ids"]) == {"alpha", "beta"}

    def test_n_residues_none_for_batch(self, tmp_path):
        proteins = {"x": _make_data()}
        p = tmp_path / "nr.oemb"
        write_oemb_batch(p, proteins)
        info = inspect_oemb(p)
        assert info["n_residues"] is None


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_sequence_string(self, tmp_path):
        """sequence='' is valid and roundtrips correctly."""
        data = _make_data(L=5, sequence="")
        p = tmp_path / "empty_seq.oemb"
        write_oemb(p, data, protein_id="x")
        loaded = read_oemb(p)
        assert loaded["sequence"] == ""

    def test_single_residue(self, tmp_path):
        """L=1 is a valid degenerate case."""
        data = _make_data(L=1)
        p = tmp_path / "single_res.oemb"
        write_oemb(p, data, protein_id="mono")
        loaded = read_oemb(p)
        assert loaded["per_residue"].shape == (1, 512)
        np.testing.assert_array_almost_equal(
            loaded["per_residue"], data["per_residue"], decimal=5
        )

    def test_single_residue_inspect(self, tmp_path):
        data = _make_data(L=1)
        p = tmp_path / "single_inspect.oemb"
        write_oemb(p, data, protein_id="mono")
        info = inspect_oemb(p)
        assert info["n_residues"] == 1

    def test_batch_empty_sequence(self, tmp_path):
        """Batch proteins without sequence field default to empty string."""
        proteins = {"p1": _make_data(), "p2": _make_data()}
        # data dicts have no 'sequence' key
        p = tmp_path / "no_seq.oemb"
        write_oemb_batch(p, proteins)
        loaded = read_oemb_batch(p)
        assert loaded["p1"]["sequence"] == ""
        assert loaded["p2"]["sequence"] == ""

    def test_parent_dir_created(self, tmp_path):
        """write_oemb creates intermediate directories."""
        data = _make_data()
        p = tmp_path / "nested" / "subdir" / "prot.oemb"
        write_oemb(p, data)
        assert p.exists()

    def test_batch_parent_dir_created(self, tmp_path):
        proteins = {"x": _make_data()}
        p = tmp_path / "deep" / "dir" / "batch.oemb"
        write_oemb_batch(p, proteins)
        assert p.exists()

    def test_large_L_roundtrip(self, tmp_path):
        """Protein with L=500 residues roundtrips correctly."""
        data = _make_data(L=500)
        p = tmp_path / "large.oemb"
        write_oemb(p, data, protein_id="big")
        loaded = read_oemb(p)
        assert loaded["per_residue"].shape == (500, 512)
        np.testing.assert_array_almost_equal(
            loaded["per_residue"], data["per_residue"], decimal=5
        )
