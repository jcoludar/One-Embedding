"""Tests for OneEmbeddingCodec: encode, save, load, batch."""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec import OneEmbeddingCodec


@pytest.fixture
def codec():
    return OneEmbeddingCodec(d_out=512, dct_k=4, seed=42)


@pytest.fixture
def raw_embedding():
    """Fake ProtT5-like embedding: (L=50, D=1024)."""
    rng = np.random.RandomState(0)
    return rng.randn(50, 1024).astype(np.float32)


@pytest.fixture
def raw_h5(tmp_path):
    """H5 file with 3 fake proteins."""
    path = tmp_path / "raw.h5"
    rng = np.random.RandomState(0)
    with h5py.File(path, "w") as f:
        for i in range(3):
            L = 40 + i * 20  # 40, 60, 80
            f.create_dataset(f"prot_{i}", data=rng.randn(L, 1024).astype(np.float32))
    return path


class TestEncode:
    def test_output_shapes(self, codec, raw_embedding):
        result = codec.encode(raw_embedding)
        assert result["per_residue"].shape == (50, 512)
        assert result["protein_vec"].shape == (2048,)

    def test_output_dtypes_default_float16(self, codec, raw_embedding):
        result = codec.encode(raw_embedding)
        assert result["per_residue"].dtype == np.float16
        assert result["protein_vec"].dtype == np.float16

    def test_output_dtypes_explicit_float32(self, raw_embedding):
        codec32 = OneEmbeddingCodec(dtype="float32")
        result = codec32.encode(raw_embedding)
        assert result["per_residue"].dtype == np.float32
        assert result["protein_vec"].dtype == np.float32

    def test_metadata_fields(self, codec, raw_embedding):
        result = codec.encode(raw_embedding)
        meta = result["metadata"]
        assert meta["codec"] == "rp_dct"
        assert meta["d_in"] == 1024
        assert meta["d_out"] == 512
        assert meta["dct_k"] == 4
        assert meta["seed"] == 42
        assert meta["seq_len"] == 50
        assert meta["protein_vec_dim"] == 2048
        assert meta["dtype"] == "float16"

    def test_deterministic(self, codec, raw_embedding):
        r1 = codec.encode(raw_embedding)
        r2 = codec.encode(raw_embedding)
        np.testing.assert_array_equal(r1["per_residue"], r2["per_residue"])
        np.testing.assert_array_equal(r1["protein_vec"], r2["protein_vec"])

    def test_different_seeds_differ(self, raw_embedding):
        c1 = OneEmbeddingCodec(seed=42)
        c2 = OneEmbeddingCodec(seed=99)
        r1 = c1.encode(raw_embedding)
        r2 = c2.encode(raw_embedding)
        assert not np.allclose(r1["per_residue"], r2["per_residue"])

    def test_different_input_dims(self, codec):
        """Works with ESM2 (1280d) and ESM-C (960d)."""
        rng = np.random.RandomState(0)
        for D in [960, 1024, 1280]:
            raw = rng.randn(30, D).astype(np.float32)
            result = codec.encode(raw)
            assert result["per_residue"].shape == (30, 512)
            assert result["protein_vec"].shape == (2048,)
            assert result["metadata"]["d_in"] == D


class TestSaveLoad:
    def test_roundtrip(self, codec, raw_embedding, tmp_path):
        encoded = codec.encode(raw_embedding)
        path = codec.save(encoded, tmp_path / "test.h5", protein_id="my_protein")

        loaded = OneEmbeddingCodec.load(path)
        np.testing.assert_array_equal(loaded["per_residue"], encoded["per_residue"])
        np.testing.assert_array_equal(loaded["protein_vec"], encoded["protein_vec"])
        assert loaded["metadata"]["protein_id"] == "my_protein"
        assert loaded["metadata"]["d_out"] == 512

    def test_load_needs_no_codec(self, codec, raw_embedding, tmp_path):
        """Receiver can load without knowing codec parameters."""
        encoded = codec.encode(raw_embedding)
        path = codec.save(encoded, tmp_path / "test.h5")

        # Load with static method — no OneEmbeddingCodec instance needed
        loaded = OneEmbeddingCodec.load(str(path))
        assert loaded["per_residue"].shape == (50, 512)
        assert loaded["protein_vec"].shape == (2048,)
        assert loaded["per_residue"].dtype == np.float16

    def test_file_uses_gzip(self, codec, raw_embedding, tmp_path):
        encoded = codec.encode(raw_embedding)
        path = codec.save(encoded, tmp_path / "test.h5")
        with h5py.File(path, "r") as f:
            assert f["per_residue"].compression == "gzip"


class TestBatch:
    def test_encode_h5_individual(self, codec, raw_h5, tmp_path):
        out_dir = tmp_path / "encoded"
        paths = codec.encode_h5(str(raw_h5), str(out_dir))
        assert len(paths) == 3
        for p in paths:
            loaded = OneEmbeddingCodec.load(p)
            assert loaded["per_residue"].shape[1] == 512
            assert loaded["protein_vec"].shape == (2048,)

    def test_encode_h5_to_h5(self, codec, raw_h5, tmp_path):
        out_path = tmp_path / "encoded.h5"
        codec.encode_h5_to_h5(str(raw_h5), str(out_path))

        loaded = OneEmbeddingCodec.load_batch(str(out_path))
        assert len(loaded["protein_vecs"]) == 3
        assert len(loaded["per_residue"]) == 3
        assert loaded["metadata"]["codec"] == "rp_dct"

        for pid, vec in loaded["protein_vecs"].items():
            assert vec.shape == (2048,)

    def test_max_proteins(self, codec, raw_h5, tmp_path):
        out_path = tmp_path / "encoded.h5"
        codec.encode_h5_to_h5(str(raw_h5), str(out_path), max_proteins=2)
        loaded = OneEmbeddingCodec.load_batch(str(out_path))
        assert len(loaded["protein_vecs"]) == 2

    def test_load_subset(self, codec, raw_h5, tmp_path):
        out_path = tmp_path / "encoded.h5"
        codec.encode_h5_to_h5(str(raw_h5), str(out_path))
        loaded = OneEmbeddingCodec.load_batch(str(out_path),
                                               protein_ids=["prot_0"])
        assert len(loaded["protein_vecs"]) == 1
        assert "prot_0" in loaded["protein_vecs"]
