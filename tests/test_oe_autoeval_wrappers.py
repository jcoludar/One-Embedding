"""Unit tests for src/oe_autoeval/wrappers.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.oe_autoeval.wrappers import (
    oe_per_residue,
    oe_per_sequence,
    raw_per_residue,
    raw_per_sequence,
)


class _Rec:
    def __init__(self, seq):
        self.seq = seq


class _Tensor:
    def __init__(self, arr):
        self._a = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class FakeEmbedService:
    """Mimics biotrainer's embedding service: yields (record, torch-like tensor)."""

    def __init__(self, embs):
        self._embs = embs

    def generate_embeddings(self, input_data, reduce):
        for seq in input_data:
            yield _Rec(seq), _Tensor(self._embs[seq])


def _fit_codec(native_d, embs, d_out=None):
    codec = OneEmbeddingCodec() if d_out is None else OneEmbeddingCodec(d_out=d_out)
    codec.fit(embs)
    return codec


def _corpus(seq, native_d, seed=0):
    """A fit corpus with enough residues for PCA, keyed by sequence string."""
    rng = np.random.RandomState(seed)
    return {seq: rng.randn(len(seq), native_d).astype(np.float32)}


def test_oe_per_residue_rp_skipped_8m():
    seq = "MKLV" * 10  # L=40
    embs = _corpus(seq, 320)
    codec = _fit_codec(320, embs)              # 896 > 320 -> RP skipped, d_out_eff=320
    out = dict(oe_per_residue(codec, FakeEmbedService(embs), d_out_eff=320, sequences=[seq]))
    arr = out[seq]
    assert arr.dtype == np.float32
    assert arr.shape == (40, 320)


def test_oe_per_residue_rp_engaged_650m():
    seq = "MKLV" * 10  # L=40
    embs = _corpus(seq, 1280)
    codec = _fit_codec(1280, embs)             # 896 < 1280 -> RP -> d_out_eff=896
    out = dict(oe_per_residue(codec, FakeEmbedService(embs), d_out_eff=896, sequences=[seq]))
    arr = out[seq]
    assert arr.dtype == np.float32
    assert arr.shape == (40, 896)              # NOT packed (40, 112)


def test_oe_per_residue_wrong_d_out_eff_raises():
    seq = "MKLV" * 10
    embs = _corpus(seq, 320)
    codec = _fit_codec(320, embs)
    with pytest.raises(ValueError, match="per-residue shape"):
        list(oe_per_residue(codec, FakeEmbedService(embs), d_out_eff=896, sequences=[seq]))


def test_oe_per_sequence_shape_and_dtype():
    seq = "MKLV" * 10
    embs = _corpus(seq, 1280)
    codec = _fit_codec(1280, embs)             # d_out_eff=896
    out = dict(oe_per_sequence(codec, FakeEmbedService(embs), d_out_eff=896, sequences=[seq]))
    vec = out[seq]
    assert vec.shape == (4 * 896,)
    assert vec.dtype == np.float32


def test_oe_per_sequence_rejects_short_protein():
    good = "MKLV" * 10
    corpus = _corpus(good, 320)
    codec = _fit_codec(320, corpus)            # fit on a normal corpus
    short_embs = {"MK": np.ones((2, 320), np.float32)}
    with pytest.raises(ValueError, match="shorter than dct_k"):
        list(oe_per_sequence(codec, FakeEmbedService(short_embs), d_out_eff=320, sequences=["MK"]))


def test_raw_per_residue_passthrough():
    seq = "MKLV" * 10
    embs = {seq: np.arange(40 * 320, dtype=np.float32).reshape(40, 320)}
    out = dict(raw_per_residue(FakeEmbedService(embs), sequences=[seq]))
    assert out[seq].shape == (40, 320)
    assert np.allclose(out[seq], embs[seq])


def test_raw_per_sequence_is_mean_pool():
    seq = "MKLV"
    embs = {seq: np.ones((4, 320), np.float32) * 2.0}
    out = dict(raw_per_sequence(FakeEmbedService(embs), sequences=[seq]))
    assert out[seq].shape == (320,)
    assert np.allclose(out[seq], 2.0)
