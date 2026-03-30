"""Tests for unified OneEmbeddingCodec."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec_v2 import auto_pq_m


class TestAutoPqM:
    def test_768d_gives_192(self):
        """768 // 4 = 192, divides evenly."""
        assert auto_pq_m(768) == 192

    def test_512d_gives_128(self):
        """512 // 4 = 128, divides evenly."""
        assert auto_pq_m(512) == 128

    def test_1024d_gives_256(self):
        """1024 // 4 = 256, divides evenly."""
        assert auto_pq_m(1024) == 256

    def test_1280d_gives_320(self):
        """1280 // 4 = 320, divides evenly."""
        assert auto_pq_m(1280) == 320

    def test_result_divides_d_out(self):
        for d in [256, 384, 512, 640, 768, 896, 1024, 1280]:
            m = auto_pq_m(d)
            assert d % m == 0, f"d_out={d}, pq_m={m}, remainder={d % m}"


from src.one_embedding.codec_v2 import OneEmbeddingCodec


class TestConstructor:
    def test_defaults(self):
        codec = OneEmbeddingCodec()
        assert codec.d_out == 768
        assert codec.quantization == "pq"
        assert codec.pq_m == 192

    def test_quantization_none(self):
        codec = OneEmbeddingCodec(quantization=None)
        assert codec.quantization is None
        assert codec.pq_m is None

    def test_quantization_int4(self):
        codec = OneEmbeddingCodec(quantization='int4')
        assert codec.quantization == 'int4'
        assert codec.pq_m is None

    def test_quantization_binary(self):
        codec = OneEmbeddingCodec(quantization='binary')
        assert codec.quantization == 'binary'

    def test_custom_pq_m(self):
        codec = OneEmbeddingCodec(quantization='pq', pq_m=192)
        assert codec.pq_m == 192

    def test_invalid_pq_m_raises(self):
        with pytest.raises(ValueError, match="must divide"):
            OneEmbeddingCodec(quantization='pq', pq_m=100)

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            OneEmbeddingCodec(quantization='jpeg')

    def test_d_out_override(self):
        codec = OneEmbeddingCodec(d_out=512)
        assert codec.d_out == 512
        assert codec.pq_m == 128


import h5py


@pytest.fixture
def raw_1024():
    rng = np.random.RandomState(42)
    return rng.randn(30, 1024).astype(np.float32)


@pytest.fixture
def small_corpus():
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(20 + i, 1024).astype(np.float32) for i in range(20)}


class TestEncodeDecode:
    def test_fp16_roundtrip(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        assert encoded["per_residue_fp16"].shape == (30, 768)
        assert encoded["per_residue_fp16"].dtype == np.float16
        decoded = codec.decode_per_residue(encoded)
        np.testing.assert_allclose(decoded, encoded["per_residue_fp16"].astype(np.float32), atol=1e-3)

    def test_int4_roundtrip(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=768, quantization='int4')
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)
        projected = codec._preprocess(raw_1024)
        cos_sim = np.mean([
            np.dot(projected[i], decoded[i]) / (np.linalg.norm(projected[i]) * np.linalg.norm(decoded[i]) + 1e-10)
            for i in range(30)
        ])
        assert cos_sim > 0.95

    def test_pq_roundtrip(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=128)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        assert encoded["pq_codes"].shape == (30, 128)
        assert encoded["pq_codes"].dtype == np.uint8
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)

    def test_binary_roundtrip(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=768, quantization='binary')
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)

    def test_protein_vec_always_fp16(self, raw_1024, small_corpus):
        for q in [None, 'int4', 'pq', 'binary']:
            codec = OneEmbeddingCodec(d_out=768, quantization=q)
            codec.fit(small_corpus)
            encoded = codec.encode(raw_1024)
            assert encoded["protein_vec"].dtype == np.float16
            assert encoded["protein_vec"].shape == (768 * 4,)

    def test_rp_skip_when_d_out_equals_d_in(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=1024, quantization=None)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        assert encoded["per_residue_fp16"].shape == (30, 1024)
        assert encoded["protein_vec"].shape == (1024 * 4,)

    def test_metadata_fields(self, raw_1024, small_corpus):
        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=128)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        meta = encoded["metadata"]
        assert meta["codec"] == "one_embedding"
        assert meta["version"] == 3
        assert meta["d_out"] == 768
        assert meta["quantization"] == "pq"
        assert meta["pq_m"] == 128
        assert meta["seq_len"] == 30


class TestSaveLoad:
    def test_fp16_h5_roundtrip(self, raw_1024, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        encoded = codec.encode(raw_1024)
        h5_path = tmp_path / "protein.h5"
        codec.save(encoded, str(h5_path))

        loaded = OneEmbeddingCodec.load(str(h5_path), codebook_path=str(cb_path))
        assert loaded["per_residue"].shape == (30, 768)
        np.testing.assert_allclose(
            loaded["per_residue"],
            encoded["per_residue_fp16"].astype(np.float32),
            atol=1e-3,
        )

    def test_pq_h5_roundtrip(self, raw_1024, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=128)
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        encoded = codec.encode(raw_1024)
        h5_path = tmp_path / "protein.h5"
        codec.save(encoded, str(h5_path))

        loaded = OneEmbeddingCodec.load(str(h5_path), codebook_path=str(cb_path))
        assert loaded["per_residue"].shape == (30, 768)
        assert loaded["metadata"]["quantization"] == "pq"

    def test_codebook_stores_quantization_params(self, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=192)
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        with h5py.File(cb_path, "r") as f:
            assert f.attrs["d_out"] == 768
            assert f.attrs["quantization"] == "pq"
            assert f.attrs["pq_M"] == 192

    def test_int4_h5_roundtrip(self, raw_1024, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=768, quantization='int4')
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        encoded = codec.encode(raw_1024)
        h5_path = tmp_path / "protein.h5"
        codec.save(encoded, str(h5_path))

        loaded = OneEmbeddingCodec.load(str(h5_path), codebook_path=str(cb_path))
        assert loaded["per_residue"].shape == (30, 768)
