"""Unit tests for src/oe_autoeval/driver.py (wiring + injected run)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.oe_autoeval.driver import (
    d_out_eff_for,
    build_codec,
    select_embedding_functions,
    run,
)


def test_d_out_eff_for_range():
    assert d_out_eff_for(320) == 320     # ESM2-8M  (RP skipped)
    assert d_out_eff_for(480) == 480     # 35M
    assert d_out_eff_for(640) == 640     # 150M
    assert d_out_eff_for(1280) == 896    # 650M (RP)
    assert d_out_eff_for(2560) == 896    # 3B
    assert d_out_eff_for(1024) == 896    # ProtT5


def test_build_codec_modes():
    assert build_codec("oe", native_d=1280).d_out == 896
    assert build_codec("oe_norp", native_d=1280).d_out == 1280     # forces RP off
    assert build_codec("oe_norp", native_d=320).d_out == 320       # == oe for small models
    assert build_codec("raw", native_d=1280) is None


def test_select_embedding_functions_names():
    fns_raw = select_embedding_functions("raw", codec=None, d_out_eff=320)
    fns_oe = select_embedding_functions("oe", codec=object(), d_out_eff=320)
    assert fns_raw["per_residue"].__name__ == "raw_pr"
    assert fns_raw["per_sequence"].__name__ == "raw_ps"
    assert fns_oe["per_residue"].__name__ == "oe_pr"
    assert fns_oe["per_sequence"].__name__ == "oe_ps"


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
    def __init__(self, embs):
        self._embs = embs

    def generate_embeddings(self, input_data, reduce):
        for seq in input_data:
            yield _Rec(seq), _Tensor(self._embs[seq])


def test_run_calls_pipeline_with_right_kwargs_and_drains():
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        captured["consumed"] = 0
        for i in range(3):
            captured["consumed"] += 1
            yield f"progress-{i}"

    rng = np.random.RandomState(0)
    ref = {f"p{i}": rng.randn(40, 320).astype(np.float32) for i in range(6)}
    svc = FakeEmbedService({})

    label = run(
        embed_service=svc,
        autoeval_pipeline=fake_pipeline,
        mode="oe",
        native_d=320,
        precision="fp32",
        label="ESM2-8M-ONE",
        output_dir="/tmp/oe_test",
        reference_embeddings=ref,
    )

    assert label == "ESM2-8M-ONE"
    assert captured["framework"] == "PBC"
    assert captured["embedder_name"] == "ESM2-8M-ONE"
    assert captured["use_half_precision"] is False
    assert "custom_embedding_function_per_residue" in captured
    assert "custom_embedding_function_per_sequence" in captured
    assert captured["consumed"] == 3          # generator was drained


def test_run_oe_norp_uses_native_width():
    """Regression for the fan-review #3 bug: oe_norp must feed d_eff=native_d (RP off),
    not min(896, native_d). Wrong d_eff would trip the wrapper shape guard. native_d=1024 ->
    codec d_out=1024, RP skipped -> per-residue width 1024; the guard must accept it."""
    seq = "MKLV" * 12  # L=48
    rng = np.random.RandomState(0)
    embs = {f"p{i}": rng.randn(48, 1024).astype(np.float32) for i in range(6)}
    embs[seq] = rng.randn(48, 1024).astype(np.float32)
    svc = FakeEmbedService(embs)
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        # invoke the per-residue closure on one protein — raises if d_eff is wrong
        list(kwargs["custom_embedding_function_per_residue"]([seq]))
        captured["ran"] = True
        yield "done"

    run(
        embed_service=svc,
        autoeval_pipeline=fake_pipeline,
        mode="oe_norp",
        native_d=1024,
        precision="fp16",
        label="ProtT5-NORP",
        output_dir="/tmp/oe_test",
        reference_embeddings={k: embs[k] for k in embs if k != seq},
    )
    assert captured.get("ran") is True   # width-1024 guard passed (no ValueError)


def test_run_fp16_flag():
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        yield "done"

    run(
        embed_service=FakeEmbedService({}),
        autoeval_pipeline=fake_pipeline,
        mode="raw",
        native_d=2560,
        precision="fp16",
        label="ESM2-3B",
        output_dir="/tmp/oe_test",
        reference_embeddings=None,
    )
    assert captured["use_half_precision"] is True
