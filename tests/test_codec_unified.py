"""Tests for unified OneEmbeddingCodec."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec_v2 import OneEmbeddingCodec, auto_pq_m


class TestAutoPqM:
    def test_768d_gives_192(self):
        assert auto_pq_m(768) == 192

    def test_512d_gives_128(self):
        assert auto_pq_m(512) == 128

    def test_1024d_gives_256(self):
        assert auto_pq_m(1024) == 256

    def test_1280d_gives_320(self):
        assert auto_pq_m(1280) == 320

    def test_result_divides_d_out(self):
        for d in [256, 384, 512, 640, 768, 896, 1024, 1280]:
            m = auto_pq_m(d)
            assert d % m == 0, f"d_out={d}, pq_m={m}, remainder={d % m}"


class TestConstructor:
    def test_defaults(self):
        codec = OneEmbeddingCodec()
        assert codec.d_out == 896
        assert codec.quantization == "binary"
        assert codec.pq_m is None
        assert codec.abtt_k == 0

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
        codec = OneEmbeddingCodec(quantization='pq', pq_m=224)
        assert codec.pq_m == 224

    def test_invalid_pq_m_raises(self):
        with pytest.raises(ValueError, match="must divide"):
            OneEmbeddingCodec(quantization='pq', pq_m=100)

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            OneEmbeddingCodec(quantization='jpeg')

    def test_d_out_override(self):
        codec = OneEmbeddingCodec(d_out=512)
        assert codec.d_out == 512
        # binary default: pq_m is None
        assert codec.pq_m is None

    def test_d_out_override_with_pq(self):
        codec = OneEmbeddingCodec(d_out=512, quantization='pq')
        assert codec.d_out == 512
        assert codec.pq_m == 128

    def test_abtt_k_default_zero(self):
        codec = OneEmbeddingCodec()
        assert codec.abtt_k == 0

    def test_abtt_k_override(self):
        codec = OneEmbeddingCodec(abtt_k=3)
        assert codec.abtt_k == 3

    def test_repr(self):
        codec = OneEmbeddingCodec()
        r = repr(codec)
        assert "d_out=896" in r
        assert "quantization='binary'" in r
        assert "abtt_k=0" in r

    def test_is_fitted_false_before_fit(self):
        codec = OneEmbeddingCodec(quantization=None)
        assert not codec.is_fitted

    def test_is_fitted_true_after_fit(self, small_corpus):
        codec = OneEmbeddingCodec(quantization=None)
        codec.fit(small_corpus)
        assert codec.is_fitted


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
        assert meta["version"] == 4
        assert meta["d_out"] == 768
        assert meta["quantization"] == "pq"
        assert meta["pq_m"] == 128
        assert meta["abtt_k"] == 0
        assert meta["seq_len"] == 30

    def test_pq_without_fit_raises(self, raw_1024):
        codec = OneEmbeddingCodec(d_out=768, quantization='pq')
        with pytest.raises(RuntimeError, match="fitted codebook"):
            codec.encode(raw_1024)

    def test_save_codebook_without_fit_raises(self):
        codec = OneEmbeddingCodec(quantization=None)
        with pytest.raises(RuntimeError, match="fit"):
            codec.save_codebook("/tmp/shouldnt_exist.h5")


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

    def test_binary_h5_roundtrip(self, raw_1024, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=768, quantization='binary')
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        encoded = codec.encode(raw_1024)
        h5_path = tmp_path / "protein.h5"
        codec.save(encoded, str(h5_path))

        loaded = OneEmbeddingCodec.load(str(h5_path), codebook_path=str(cb_path))
        assert loaded["per_residue"].shape == (30, 768)
        assert loaded["metadata"]["quantization"] == "binary"

    def test_codebook_stores_quantization_params(self, small_corpus, tmp_path):
        codec = OneEmbeddingCodec(d_out=896, quantization='pq', pq_m=224)
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        with h5py.File(cb_path, "r") as f:
            assert f.attrs["d_out"] == 896
            assert f.attrs["quantization"] == "pq"
            assert f.attrs["pq_M"] == 224
            assert f.attrs["abtt_k"] == 0


class TestBatchEncodeLoad:
    """Tests for encode_h5_to_h5 + load_batch roundtrip."""

    def _make_raw_h5(self, tmp_path, corpus):
        h5_path = tmp_path / "raw.h5"
        with h5py.File(h5_path, "w") as f:
            for pid, arr in corpus.items():
                f.create_dataset(pid, data=arr)
        return h5_path

    def test_fp16_batch_roundtrip(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        loaded = OneEmbeddingCodec.load_batch(str(out_h5))
        assert len(loaded) == len(small_corpus)
        for pid in small_corpus:
            assert loaded[pid]["per_residue"].shape[1] == 768
            assert loaded[pid]["protein_vec"].shape == (768 * 4,)

    def test_int4_batch_roundtrip(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization='int4')
        codec.fit(small_corpus)
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        loaded = OneEmbeddingCodec.load_batch(str(out_h5))
        assert len(loaded) == len(small_corpus)
        for pid in small_corpus:
            assert loaded[pid]["per_residue"].shape[1] == 768

    def test_pq_batch_roundtrip(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"
        cb_h5 = tmp_path / "codebook.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=128)
        codec.fit(small_corpus)
        codec.save_codebook(str(cb_h5))
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        loaded = OneEmbeddingCodec.load_batch(str(out_h5), codebook_path=str(cb_h5))
        assert len(loaded) == len(small_corpus)
        for pid in small_corpus:
            assert loaded[pid]["per_residue"].shape[1] == 768

    def test_binary_batch_roundtrip(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization='binary')
        codec.fit(small_corpus)
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        loaded = OneEmbeddingCodec.load_batch(str(out_h5))
        assert len(loaded) == len(small_corpus)
        for pid in small_corpus:
            assert loaded[pid]["per_residue"].shape[1] == 768

    def test_load_batch_subset(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        subset = list(small_corpus.keys())[:3]
        loaded = OneEmbeddingCodec.load_batch(str(out_h5), protein_ids=subset)
        assert len(loaded) == 3
        assert set(loaded.keys()) == set(subset)

    def test_load_batch_missing_id_raises(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        codec.encode_h5_to_h5(str(raw_h5), str(out_h5))

        with pytest.raises(KeyError, match="not found"):
            OneEmbeddingCodec.load_batch(str(out_h5), protein_ids=["nonexistent_id"])

    def test_pq_batch_without_fit_raises(self, small_corpus, tmp_path):
        raw_h5 = self._make_raw_h5(tmp_path, small_corpus)
        out_h5 = tmp_path / "compressed.h5"

        codec = OneEmbeddingCodec(d_out=768, quantization='pq')
        with pytest.raises(RuntimeError, match="fitted codebook"):
            codec.encode_h5_to_h5(str(raw_h5), str(out_h5))
