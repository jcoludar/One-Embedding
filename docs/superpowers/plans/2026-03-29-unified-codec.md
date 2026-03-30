# Unified One Embedding Codec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge V1 (fp16) and V2 (quantized) codecs into a single `OneEmbeddingCodec` class with 768d default and ~30x PQ compression out of the box.

**Architecture:** Evolve `src/one_embedding/codec_v2.py` into the unified codec. Add fp16 and RP-skip paths alongside existing int4/PQ/binary. Replace `codec.py` with a thin import wrapper. New Exp 44 benchmarks all quantization types on 768d.

**Tech Stack:** numpy, h5py, scipy (DCT), sklearn (MiniBatchKMeans for PQ)

**Spec:** `docs/superpowers/specs/2026-03-29-unified-codec-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/one_embedding/codec_v2.py` | **Major modify** | Unified `OneEmbeddingCodec` — all quantization paths, configurable d_out/quantization/pq_m |
| `src/one_embedding/codec.py` | **Replace** | Thin wrapper: `from .codec_v2 import OneEmbeddingCodec` |
| `tests/test_codec_unified.py` | **Create** | Tests for unified codec — all quantization paths, RP skip, auto pq_m |
| `experiments/44_unified_codec_benchmark.py` | **Create** | Benchmark sweep: all quantization × 768d through Exp 43 framework |

Existing experiment scripts (`run_phase_a1.py`, `run_v2_validation.py`, etc.) are NOT modified — they import `Codec` from `core/codec.py` or `OneEmbeddingCodecV2` from `codec_v2.py`, both of which remain functional.

---

### Task 1: Auto pq_m helper and validation

**Files:**
- Modify: `src/one_embedding/codec_v2.py`
- Test: `tests/test_codec_unified.py`

- [ ] **Step 1: Write failing tests for auto_pq_m and validation**

Create `tests/test_codec_unified.py`:

```python
"""Tests for unified OneEmbeddingCodec."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec_v2 import auto_pq_m


class TestAutoPqM:
    def test_768d_gives_128(self):
        """768 // 6 = 128, which divides 768 evenly."""
        assert auto_pq_m(768) == 128

    def test_512d_gives_64(self):
        """512 // 6 = 85, not a factor. Largest factor <= 85 is 64."""
        assert auto_pq_m(512) == 64

    def test_1024d_gives_128(self):
        """1024 // 6 = 170, not a factor. Largest factor <= 170 is 128."""
        assert auto_pq_m(1024) == 128

    def test_1280d_gives_160(self):
        """1280 // 6 = 213, not a factor. 160 divides 1280 (8d subs)."""
        assert auto_pq_m(1280) == 160

    def test_result_divides_d_out(self):
        """Auto pq_m must always divide d_out evenly."""
        for d in [256, 384, 512, 640, 768, 896, 1024, 1280]:
            m = auto_pq_m(d)
            assert d % m == 0, f"d_out={d}, pq_m={m}, remainder={d % m}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_codec_unified.py -v`
Expected: FAIL — `auto_pq_m` not found

- [ ] **Step 3: Implement auto_pq_m**

Add to `src/one_embedding/codec_v2.py` after the imports, before the MODES dict:

```python
def auto_pq_m(d_out: int) -> int:
    """Compute default PQ M targeting ~6d sub-vectors (~30x compression).

    Finds the largest factor of d_out that is <= d_out // 6.
    For d_out=768: returns 128 (6d sub-vectors, 30x compression).
    For d_out=512: returns 64 (8d sub-vectors, 32x compression).
    """
    target = d_out // 6
    for m in range(target, 0, -1):
        if d_out % m == 0:
            return m
    return 1  # fallback (d_out=1 edge case)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_codec_unified.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/codec_v2.py tests/test_codec_unified.py
git commit -m "feat: auto_pq_m helper — targets ~30x compression for any d_out"
```

---

### Task 2: Unified OneEmbeddingCodec constructor with quantization parameter

**Files:**
- Modify: `src/one_embedding/codec_v2.py`
- Test: `tests/test_codec_unified.py`

- [ ] **Step 1: Write failing tests for the new constructor**

Append to `tests/test_codec_unified.py`:

```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec


class TestConstructor:
    def test_defaults(self):
        """Default: d_out=768, quantization='pq', auto pq_m=128."""
        codec = OneEmbeddingCodec()
        assert codec.d_out == 768
        assert codec.quantization == "pq"
        assert codec.pq_m == 128

    def test_quantization_none(self):
        """quantization=None means fp16 storage, no codebook needed for encode."""
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
        """pq_m must divide d_out evenly."""
        with pytest.raises(ValueError, match="must divide"):
            OneEmbeddingCodec(quantization='pq', pq_m=100)

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            OneEmbeddingCodec(quantization='jpeg')

    def test_d_out_override(self):
        codec = OneEmbeddingCodec(d_out=512)
        assert codec.d_out == 512
        assert codec.pq_m == 64  # auto for 512
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_codec_unified.py::TestConstructor -v`
Expected: FAIL — constructor signature doesn't match

- [ ] **Step 3: Rewrite the OneEmbeddingCodecV2 class as OneEmbeddingCodec**

In `src/one_embedding/codec_v2.py`, replace the `MODES` dict and the class definition. Keep the old `MODES` dict available as `_LEGACY_MODES` for backward compat of existing V2 codebooks. The full class header becomes:

```python
# Legacy mode lookup — only used for decoding old V2 files
_LEGACY_MODES = {
    "full":     {"type": "int4"},
    "balanced": {"type": "pq",     "M": 128, "K": 256},
    "compact":  {"type": "pq",     "M": 64,  "K": 256},
    "micro":    {"type": "pq",     "M": 32,  "K": 256},
    "binary":   {"type": "binary"},
}

# Keep MODES as alias for any code that imports it
MODES = _LEGACY_MODES

_VALID_QUANTIZATIONS = {None, "int4", "pq", "binary"}


class OneEmbeddingCodec:
    """Universal protein embedding codec.

    Compresses raw PLM per-residue embeddings (any PLM, any protein) into
    compact representations for storage and downstream tasks.

    Default: ABTT3 + RP to 768d + PQ M=128 → ~30x compression.

    Three knobs control the compression/fidelity trade-off:
        d_out: Dimensions after random projection (default 768).
            Higher = more fidelity, less compression.
            Set to input dim (e.g. 1024) to skip RP entirely.
        quantization: Per-residue storage method (default 'pq').
            None = fp16 (2.7x), 'int4' (10x), 'pq' (~30x), 'binary' (41x).
        pq_m: PQ subquantizers (default auto = d_out // 6).
            Only used when quantization='pq'. Must divide d_out evenly.

    Args:
        d_out: Output dimensionality for random projection (default 768).
        quantization: 'pq' (default), 'int4', 'binary', or None (fp16).
        pq_m: Number of PQ subquantizers. Auto-computed if None.
        dct_k: DCT coefficients for protein vector (default 4).
        seed: Fixed seed for RP matrix (default 42).
        codebook_path: Path to pre-fitted codebook H5.
    """

    def __init__(
        self,
        d_out: int = 768,
        quantization: str | None = "pq",
        pq_m: int | None = None,
        dct_k: int = 4,
        seed: int = 42,
        codebook_path: str | None = None,
    ):
        if quantization not in _VALID_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_VALID_QUANTIZATIONS}, got '{quantization}'"
            )

        self.d_out = d_out
        self.quantization = quantization
        self.dct_k = dct_k
        self.seed = seed
        self._proj_cache: dict[int, np.ndarray] = {}
        self._corpus_stats = None
        self._pq_model = None

        # Resolve pq_m
        if quantization == "pq":
            if pq_m is None:
                pq_m = auto_pq_m(d_out)
            if d_out % pq_m != 0:
                factors = [f for f in range(1, d_out + 1) if d_out % f == 0]
                raise ValueError(
                    f"pq_m={pq_m} must divide d_out={d_out} evenly. "
                    f"Valid factors: {factors}"
                )
            self.pq_m = pq_m
            self.pq_k = 256
        else:
            self.pq_m = None
            self.pq_k = None

        if codebook_path is not None:
            self._load_codebook(codebook_path)
```

Keep all existing methods (`_get_projection_matrix`, `_preprocess`, `fit`, `save_codebook`, `_load_codebook`, `encode`, `decode_per_residue`, `save`, `load`, `encode_h5_to_h5`) but update them to use `self.quantization` instead of `self.mode_cfg["type"]`. See Task 3 for the encode/decode changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_codec_unified.py::TestConstructor -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/codec_v2.py tests/test_codec_unified.py
git commit -m "feat: unified OneEmbeddingCodec constructor — d_out/quantization/pq_m API"
```

---

### Task 3: Encode/decode for all quantization types including fp16 and RP skip

**Files:**
- Modify: `src/one_embedding/codec_v2.py`
- Test: `tests/test_codec_unified.py`

- [ ] **Step 1: Write failing tests for encode/decode round-trips**

Append to `tests/test_codec_unified.py`:

```python
@pytest.fixture
def raw_1024():
    """Fake ProtT5 embedding (L=30, D=1024)."""
    rng = np.random.RandomState(42)
    return rng.randn(30, 1024).astype(np.float32)


@pytest.fixture
def small_corpus():
    """Small corpus for fit() — 20 proteins."""
    rng = np.random.RandomState(42)
    return {f"p{i}": rng.randn(20 + i, 1024).astype(np.float32) for i in range(20)}


class TestEncodeDecode:
    def test_fp16_roundtrip(self, raw_1024, small_corpus):
        """quantization=None → fp16 storage, lossless decode."""
        codec = OneEmbeddingCodec(d_out=768, quantization=None)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        assert encoded["per_residue_fp16"].shape == (30, 768)
        assert encoded["per_residue_fp16"].dtype == np.float16
        decoded = codec.decode_per_residue(encoded)
        # fp16 round-trip: should match to fp16 precision
        np.testing.assert_allclose(decoded, encoded["per_residue_fp16"].astype(np.float32), atol=1e-3)

    def test_int4_roundtrip(self, raw_1024, small_corpus):
        """quantization='int4' → encode then decode recovers approximate signal."""
        codec = OneEmbeddingCodec(d_out=768, quantization='int4')
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)
        # int4 is lossy but should preserve gross structure
        projected = codec._preprocess(raw_1024)
        cos_sim = np.mean([
            np.dot(projected[i], decoded[i]) / (np.linalg.norm(projected[i]) * np.linalg.norm(decoded[i]) + 1e-10)
            for i in range(30)
        ])
        assert cos_sim > 0.95

    def test_pq_roundtrip(self, raw_1024, small_corpus):
        """quantization='pq' → encode to codes, decode back."""
        codec = OneEmbeddingCodec(d_out=768, quantization='pq', pq_m=128)
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        assert encoded["pq_codes"].shape == (30, 128)
        assert encoded["pq_codes"].dtype == np.uint8
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)

    def test_binary_roundtrip(self, raw_1024, small_corpus):
        """quantization='binary' → 1-bit sign encode/decode."""
        codec = OneEmbeddingCodec(d_out=768, quantization='binary')
        codec.fit(small_corpus)
        encoded = codec.encode(raw_1024)
        decoded = codec.decode_per_residue(encoded)
        assert decoded.shape == (30, 768)

    def test_protein_vec_always_fp16(self, raw_1024, small_corpus):
        """protein_vec is fp16 regardless of quantization."""
        for q in [None, 'int4', 'pq', 'binary']:
            codec = OneEmbeddingCodec(d_out=768, quantization=q)
            codec.fit(small_corpus)
            encoded = codec.encode(raw_1024)
            assert encoded["protein_vec"].dtype == np.float16
            assert encoded["protein_vec"].shape == (768 * 4,)

    def test_rp_skip_when_d_out_equals_d_in(self, raw_1024, small_corpus):
        """d_out=1024 with 1024d input skips RP — output is 1024d."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_codec_unified.py::TestEncodeDecode -v`
Expected: FAIL

- [ ] **Step 3: Update _preprocess to support RP skip**

In `_preprocess`, add RP skip logic:

```python
def _preprocess(self, raw: np.ndarray) -> np.ndarray:
    """Apply ABTT3 + RP projection (skip RP if d_out >= d_in)."""
    raw = raw.astype(np.float32)
    if self._corpus_stats is not None:
        top3 = self._corpus_stats["top_pcs"][:3]
        raw = all_but_the_top(raw, top3)
    if self.d_out >= raw.shape[1]:
        return raw  # skip RP — lossless mode
    R = self._get_projection_matrix(raw.shape[1])
    return (raw @ R).astype(np.float32)
```

- [ ] **Step 4: Update encode() to handle all quantization types**

```python
def encode(self, raw: np.ndarray) -> dict:
    L, D = raw.shape
    projected = self._preprocess(raw)
    protein_vec = dct_summary(projected, K=self.dct_k).astype(np.float16)

    result = {
        "protein_vec": protein_vec,
        "metadata": {
            "codec": "one_embedding",
            "version": 3,
            "d_in": D,
            "d_out": self.d_out,
            "quantization": self.quantization,
            "pq_m": self.pq_m,
            "dct_k": self.dct_k,
            "seed": self.seed,
            "seq_len": L,
        },
    }

    if self.quantization is None:
        result["per_residue_fp16"] = projected.astype(np.float16)
    elif self.quantization == "int4":
        compressed = quantize_int4(projected)
        result["per_residue_data"] = compressed["data"]
        result["per_residue_scales"] = compressed["scales"]
        result["per_residue_zp"] = compressed["zero_points"]
    elif self.quantization == "binary":
        compressed = quantize_binary(projected)
        result["per_residue_bits"] = compressed["bits"]
        result["per_residue_means"] = compressed["means"]
        result["per_residue_scales"] = compressed["scales"]
    elif self.quantization == "pq":
        codes = pq_encode(projected, self._pq_model)
        result["pq_codes"] = codes

    return result
```

- [ ] **Step 5: Update decode_per_residue() for fp16 path**

```python
def decode_per_residue(self, encoded: dict) -> np.ndarray:
    meta = encoded["metadata"]
    L = meta["seq_len"]
    d_out = meta.get("d_out", self.d_out)
    quantization = meta.get("quantization", None)

    if quantization is None:
        return encoded["per_residue_fp16"].astype(np.float32)
    elif quantization == "int4":
        compressed = {
            "data": encoded["per_residue_data"],
            "scales": encoded["per_residue_scales"],
            "zero_points": encoded["per_residue_zp"],
            "original_shape": (L, d_out),
            "dtype": "int4",
        }
        return dequantize_int4(compressed)
    elif quantization == "binary":
        compressed = {
            "bits": encoded["per_residue_bits"],
            "means": encoded["per_residue_means"],
            "scales": encoded["per_residue_scales"],
            "original_shape": (L, d_out),
            "dtype": "binary",
        }
        return dequantize_binary(compressed)
    elif quantization == "pq":
        return pq_decode(encoded["pq_codes"], self._pq_model)
    raise ValueError(f"Unknown quantization: {quantization}")
```

- [ ] **Step 6: Update fit() to use self.quantization instead of self.mode_cfg**

```python
def fit(self, embeddings: dict[str, np.ndarray], max_residues: int = 500_000):
    self._corpus_stats = compute_corpus_stats(
        embeddings, n_sample=50_000, n_pcs=5, seed=self.seed
    )
    if self.quantization == "pq":
        preprocessed = {}
        for pid, m in embeddings.items():
            preprocessed[pid] = self._preprocess(m)
        self._pq_model = pq_fit(
            preprocessed, M=self.pq_m, n_centroids=self.pq_k,
            max_residues=max_residues, seed=self.seed,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_codec_unified.py -v`
Expected: All tests PASS (TestAutoPqM + TestConstructor + TestEncodeDecode)

- [ ] **Step 8: Commit**

```bash
git add src/one_embedding/codec_v2.py tests/test_codec_unified.py
git commit -m "feat: unified encode/decode — fp16, int4, pq, binary + RP skip for lossless"
```

---

### Task 4: H5 save/load and codebook persistence

**Files:**
- Modify: `src/one_embedding/codec_v2.py`
- Test: `tests/test_codec_unified.py`

- [ ] **Step 1: Write failing tests for save/load round-trips**

Append to `tests/test_codec_unified.py`:

```python
import tempfile
import h5py


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

    def test_int4_no_codebook_for_decode(self, raw_1024, small_corpus, tmp_path):
        """int4 decode doesn't need a codebook file (only ABTT stats)."""
        codec = OneEmbeddingCodec(d_out=768, quantization='int4')
        codec.fit(small_corpus)
        cb_path = tmp_path / "codebook.h5"
        codec.save_codebook(str(cb_path))

        encoded = codec.encode(raw_1024)
        h5_path = tmp_path / "protein.h5"
        codec.save(encoded, str(h5_path))

        # Load with codebook (for ABTT stats metadata, not PQ)
        loaded = OneEmbeddingCodec.load(str(h5_path), codebook_path=str(cb_path))
        assert loaded["per_residue"].shape == (30, 768)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_codec_unified.py::TestSaveLoad -v`
Expected: FAIL

- [ ] **Step 3: Update save(), save_codebook(), load() for unified API**

Update `save()` and `encode_h5_to_h5()` to handle the fp16 path (store `per_residue_fp16` as a dataset named `per_residue`). Update `save_codebook()` to store `quantization` attribute. Update `load()` and `load_batch()` (if present) to read `quantization` from metadata and dispatch accordingly — handle both V3 (new) and V2 (legacy `mode` field) formats.

Key change in `save()`:

```python
if self.quantization is None:
    grp.create_dataset("per_residue", data=encoded["per_residue_fp16"],
                        compression="gzip", compression_opts=4)
elif self.quantization == "int4":
    # ... existing int4 save code ...
```

Key change in `load()` — detect format version:

```python
# V3 format: has "quantization" in metadata
# V2 format: has "mode" in metadata (legacy)
quantization = meta.get("quantization")
if quantization is None and "mode" in meta:
    # Legacy V2 format — map mode to quantization type
    legacy_mode = meta["mode"]
    quantization = _LEGACY_MODES[legacy_mode]["type"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_codec_unified.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing V2 tests to verify backward compat**

Run: `uv run python -m pytest tests/test_codec.py -v`
Expected: These may need import updates (Task 5). Note failures but don't fix yet.

- [ ] **Step 6: Commit**

```bash
git add src/one_embedding/codec_v2.py tests/test_codec_unified.py
git commit -m "feat: unified H5 save/load — fp16/int4/pq/binary + legacy V2 compat"
```

---

### Task 5: Backward compat wrapper and existing test fixup

**Files:**
- Modify: `src/one_embedding/codec.py`
- Modify: `tests/test_codec.py`

- [ ] **Step 1: Replace codec.py with thin wrapper**

Replace the entire contents of `src/one_embedding/codec.py` with:

```python
"""Backward compatibility — imports unified codec from codec_v2.

All new code should use:
    from src.one_embedding.codec_v2 import OneEmbeddingCodec
"""

from src.one_embedding.codec_v2 import OneEmbeddingCodec

__all__ = ["OneEmbeddingCodec"]
```

- [ ] **Step 2: Update test_codec.py for the new API**

The existing tests use `OneEmbeddingCodec(d_out=512, dct_k=4, seed=42)` which is V1's fp16 codec. Update the fixture and assertions to work with the unified codec:

```python
@pytest.fixture
def codec():
    """V1-equivalent: fp16 at 512d."""
    return OneEmbeddingCodec(d_out=512, quantization=None, dct_k=4, seed=42)
```

Update `test_metadata_fields` — the codec name is now `"one_embedding"` and version is `3`. The metadata now includes `quantization` instead of `dtype`. Adjust assertions accordingly:

```python
def test_metadata_fields(self, codec, raw_embedding):
    result = codec.encode(raw_embedding)
    meta = result["metadata"]
    assert meta["codec"] == "one_embedding"
    assert meta["d_in"] == 1024
    assert meta["d_out"] == 512
    assert meta["quantization"] is None
    assert meta["seq_len"] == 50
```

Update `test_output_shapes` — fp16 encode stores data in `per_residue_fp16` key, not `per_residue`:

```python
def test_output_shapes(self, codec, raw_embedding):
    result = codec.encode(raw_embedding)
    assert result["per_residue_fp16"].shape == (50, 512)
    assert result["protein_vec"].shape == (2048,)
```

Update save/load tests — `load()` now returns per_residue from the H5 dataset, which works the same way but metadata fields differ.

- [ ] **Step 3: Run all codec tests**

Run: `uv run python -m pytest tests/test_codec.py tests/test_codec_unified.py -v`
Expected: All PASS

- [ ] **Step 4: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: 761+ tests PASS (existing + new unified tests)

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/codec.py tests/test_codec.py
git commit -m "refactor: codec.py becomes thin wrapper, update V1 tests for unified API"
```

---

### Task 6: Experiment 44 — 768d quantization benchmark sweep

**Files:**
- Create: `experiments/44_unified_codec_benchmark.py`

- [ ] **Step 1: Write the benchmark script**

This script runs all quantization types on 768d through the Exp 43 rigorous framework. Structure:

```python
#!/usr/bin/env python3
"""Exp 44: Unified codec benchmark — all quantization types on 768d.

Tests: fp16, int4, PQ M=128, PQ M=192, binary on 768d.
Also: fp16 on 1024d (lossless, no RP).
Framework: BCa CIs, CV-tuned probes, paired bootstrap retention.
Tasks: SS3, SS8, disorder (pooled rho), retrieval.
"""
```

The script should:

1. Load raw ProtT5 embeddings (SCOPe 5K, CB513, CheZOD) — same data as run_v2_validation.py
2. Compute raw 1024d baselines (SS3, SS8, disorder, retrieval) — run once, reuse
3. For each configuration:
   - Create `OneEmbeddingCodec(d_out=..., quantization=..., pq_m=...)`
   - Fit on SCOPe train split
   - Encode CB513/CheZOD/SCOPe
   - Run SS3, SS8, disorder, retrieval benchmarks
   - Compute paired bootstrap retention CIs
4. Save results to `data/benchmarks/rigorous_v1/exp44_unified_results.json`
5. Print summary table with ± values

Configurations list:

```python
CONFIGS = [
    {"name": "lossless",   "d_out": 1024, "quantization": None,     "pq_m": None},
    {"name": "fp16-768",   "d_out": 768,  "quantization": None,     "pq_m": None},
    {"name": "int4-768",   "d_out": 768,  "quantization": "int4",   "pq_m": None},
    {"name": "pq192-768",  "d_out": 768,  "quantization": "pq",     "pq_m": 192},
    {"name": "pq128-768",  "d_out": 768,  "quantization": "pq",     "pq_m": 128},
    {"name": "binary-768", "d_out": 768,  "quantization": "binary", "pq_m": None},
]
```

Reuse imports from Exp 43: `run_ss3_benchmark`, `run_ss8_benchmark`, `run_disorder_benchmark`, `run_retrieval_benchmark`, `paired_bootstrap_retention`, `paired_cluster_bootstrap_retention`.

Follow the same pattern as `run_v2_validation.py` but use the unified `OneEmbeddingCodec` instead of `OneEmbeddingCodecV2`.

- [ ] **Step 2: Verify it runs on a smoke test (1 protein, 1 config)**

Run: `uv run python experiments/44_unified_codec_benchmark.py --smoke-test`
(Add a `--smoke-test` flag that runs 1 config with 1 bootstrap iteration to verify wiring)

- [ ] **Step 3: Commit**

```bash
git add experiments/44_unified_codec_benchmark.py
git commit -m "feat(exp44): unified codec benchmark sweep — 6 configs on 768d+1024d"
```

- [ ] **Step 4: Run full benchmark (~3 hours)**

Run: `uv run python experiments/44_unified_codec_benchmark.py`

This produces the actual numbers that determine whether the ~30x default holds up.

- [ ] **Step 5: Update defaults if needed**

If pq128-768 (~30x) shows <93% SS3 or <90% disorder retention:
- Change `auto_pq_m` to target d_out // 4 instead of d_out // 6
- This gives M=192 (~20x) as default — still great compression
- Update docstring

- [ ] **Step 6: Update CLAUDE.md and README.md with new benchmark results**

Replace the V2 tier tables with the Exp 44 results. Update compression ratios and ± values.

- [ ] **Step 7: Commit results and doc updates**

```bash
git add data/benchmarks/rigorous_v1/exp44_unified_results.json CLAUDE.md README.md
git commit -m "data(exp44): unified codec benchmarks — 768d quantization sweep with paired CIs"
```
