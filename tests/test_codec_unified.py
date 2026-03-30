"""Tests for unified OneEmbeddingCodec."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec_v2 import auto_pq_m


class TestAutoPqM:
    def test_768d_gives_128(self):
        assert auto_pq_m(768) == 128

    def test_512d_gives_64(self):
        assert auto_pq_m(512) == 64

    def test_1024d_gives_128(self):
        assert auto_pq_m(1024) == 128

    def test_1280d_gives_160(self):
        assert auto_pq_m(1280) == 160

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
        assert codec.pq_m == 128

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
        assert codec.pq_m == 64
