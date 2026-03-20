# One Embedding 1.0 — Phase 1: Core Upgrade

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the one_embedding package from hardcoded RP512 to configurable d_out (default 768), rename file format to `.one.h5` with freeform tags, and update CLI/API.

**Architecture:** Modify existing `Codec` class default from 512→768, update `io.py` to write `.one.h5` with freeform tags while staying backward-compatible with `.oemb`, update `__init__.py` and `cli.py` for new defaults and `--d-out` flag. Tools auto-detect D from data shape.

**Tech Stack:** numpy, h5py, scipy (DCT), click (CLI), pytest

**Spec:** `docs/superpowers/specs/2026-03-20-one-embedding-1.0-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/one_embedding/core/codec.py` | Modify | Change default d_out 512→768, update for_plm() |
| `src/one_embedding/io.py` | Modify | Add `.one.h5` format, freeform tags, backward compat |
| `src/one_embedding/__init__.py` | Modify | Change encode() default d_out, update version |
| `src/one_embedding/cli.py` | Modify | Add --d-out flag, change default output extension |
| `src/one_embedding/tools/_base.py` | Modify | Support both .one.h5 and .oemb loading |
| `src/one_embedding/tools/disorder.py` | Modify | Auto-detect D, support 512d and 768d |
| `src/one_embedding/tools/ss3.py` | Modify | Same auto-detect pattern as disorder |
| `tests/test_core_codec.py` | Modify | Update expected shapes, add multi-d_out tests |
| `tests/test_io_one_h5.py` | Create | Tests for .one.h5 format, tags, backward compat |
| `tests/test_tools_dimension.py` | Create | Tests for tool auto-detection at any D |

---

### Task 1: Update Codec default d_out from 512 to 768

**Files:**
- Modify: `src/one_embedding/core/codec.py`
- Modify: `tests/test_core_codec.py`

- [ ] **Step 1: Update tests to expect 768d default**

In `tests/test_core_codec.py`, update `test_encode_output_shapes_default`:

```python
def test_encode_output_shapes_default(self):
    codec = Codec()
    raw = _make_protein(L=50, D=1024)
    result = codec.encode(raw)
    assert result["per_residue"].shape == (50, 768)      # was 512
    assert result["protein_vec"].shape == (3072,)          # was 2048 (4 * 768)
```

Also add a new test for explicit 512d:

```python
def test_encode_explicit_512d(self):
    codec = Codec(d_out=512)
    raw = _make_protein(L=50, D=1024)
    result = codec.encode(raw)
    assert result["per_residue"].shape == (50, 512)
    assert result["protein_vec"].shape == (2048,)
```

And a 768d explicit test:

```python
def test_encode_explicit_768d(self):
    codec = Codec(d_out=768)
    raw = _make_protein(L=50, D=1024)
    result = codec.encode(raw)
    assert result["per_residue"].shape == (50, 768)
    assert result["protein_vec"].shape == (3072,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core_codec.py::TestCodec::test_encode_output_shapes_default -v`
Expected: FAIL — shape (50, 512) != (50, 768)

- [ ] **Step 3: Update Codec default**

In `src/one_embedding/core/codec.py`, change line ~35:

```python
def __init__(self, d_out: int = 768, dct_k: int = 4, seed: int = 42) -> None:
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_core_codec.py -v`
Expected: All pass (update any other tests that hardcoded 512/2048 expectations)

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/core/codec.py tests/test_core_codec.py
git commit -m "feat: change Codec default d_out from 512 to 768"
```

---

### Task 2: `.one.h5` file format with freeform tags

**Files:**
- Modify: `src/one_embedding/io.py`
- Create: `tests/test_io_one_h5.py`

- [ ] **Step 1: Write tests for .one.h5 format**

Create `tests/test_io_one_h5.py`:

```python
"""Tests for .one.h5 file format."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.io import (
    write_one_h5, read_one_h5, write_one_h5_batch, read_one_h5_batch,
    inspect_one_h5, read_oemb, read_oemb_batch,
)


class TestOneH5Single:
    def test_write_read_roundtrip(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(50, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="test_prot")
        loaded = read_one_h5(path)
        np.testing.assert_array_equal(loaded["per_residue"], data["per_residue"])
        np.testing.assert_array_equal(loaded["protein_vec"], data["protein_vec"])
        assert loaded["protein_id"] == "test_prot"

    def test_freeform_tags(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(30, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float16),
        }
        tags = {"source_model": "prot_t5_xl", "d_out": 512, "lab": "rostlab"}
        write_one_h5(path, data, protein_id="p1", tags=tags)
        loaded = read_one_h5(path)
        assert loaded["tags"]["source_model"] == "prot_t5_xl"
        assert loaded["tags"]["d_out"] == 512
        assert loaded["tags"]["lab"] == "rostlab"

    def test_variable_d_out(self, tmp_path):
        """Files with different D values both work."""
        for d in [256, 512, 768, 1024]:
            path = tmp_path / f"test_{d}.one.h5"
            data = {
                "per_residue": np.random.randn(40, d).astype(np.float32),
                "protein_vec": np.random.randn(4 * d).astype(np.float16),
            }
            write_one_h5(path, data, protein_id="p1")
            loaded = read_one_h5(path)
            assert loaded["per_residue"].shape[1] == d

    def test_format_version(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(20, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="p1")
        loaded = read_one_h5(path)
        assert loaded["format"] == "one_embedding"
        assert loaded["version"] == "1.0"


class TestOneH5Batch:
    def test_batch_write_read(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = {
            "prot_a": {
                "per_residue": np.random.randn(30, 768).astype(np.float32),
                "protein_vec": np.random.randn(3072).astype(np.float16),
            },
            "prot_b": {
                "per_residue": np.random.randn(50, 768).astype(np.float32),
                "protein_vec": np.random.randn(3072).astype(np.float16),
            },
        }
        write_one_h5_batch(path, proteins)
        loaded = read_one_h5_batch(path)
        assert set(loaded.keys()) == {"prot_a", "prot_b"}
        assert loaded["prot_a"]["per_residue"].shape == (30, 768)

    def test_batch_freeform_tags(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = {
            "p1": {
                "per_residue": np.random.randn(20, 768).astype(np.float32),
                "protein_vec": np.random.randn(3072).astype(np.float16),
                "tags": {"organism": "human"},
            },
        }
        tags = {"source_model": "prot_t5_xl", "compression": "rp768"}
        write_one_h5_batch(path, proteins, tags=tags)
        loaded = read_one_h5_batch(path)
        # Root tags accessible
        info = inspect_one_h5(path)
        assert info["tags"]["source_model"] == "prot_t5_xl"

    def test_batch_subset_loading(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = {f"p{i}": {
            "per_residue": np.random.randn(20, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        } for i in range(10)}
        write_one_h5_batch(path, proteins)
        loaded = read_one_h5_batch(path, protein_ids=["p3", "p7"])
        assert set(loaded.keys()) == {"p3", "p7"}


class TestInspect:
    def test_inspect_single(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(40, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="p1", tags={"d_out": 768})
        info = inspect_one_h5(path)
        assert info["format"] == "one_embedding"
        assert info["n_proteins"] == 1
        assert info["d_out"] == 768
        assert info["protein_vec_dim"] == 3072

    def test_inspect_batch(self, tmp_path):
        path = tmp_path / "batch.one.h5"
        proteins = {f"p{i}": {
            "per_residue": np.random.randn(20 + i, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float16),
        } for i in range(5)}
        write_one_h5_batch(path, proteins)
        info = inspect_one_h5(path)
        assert info["n_proteins"] == 5
        assert info["d_out"] == 512


class TestBackwardCompat:
    def test_read_oemb_still_works(self, tmp_path):
        """Old .oemb files can still be read."""
        from src.one_embedding.io import write_oemb
        path = tmp_path / "old.oemb"
        data = {
            "per_residue": np.random.randn(30, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float16),
        }
        write_oemb(path, data, protein_id="old_prot")
        loaded = read_oemb(path)
        assert loaded["per_residue"].shape == (30, 512)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io_one_h5.py -v`
Expected: ImportError — `write_one_h5` not found

- [ ] **Step 3: Implement .one.h5 format functions in io.py**

Add to `src/one_embedding/io.py` (keep all existing oemb functions for backward compat):

```python
ONE_H5_FORMAT = "one_embedding"
ONE_H5_VERSION = "1.0"


def write_one_h5(
    path,
    data: dict,
    protein_id: str = "protein",
    tags: dict | None = None,
) -> None:
    """Write a single protein to .one.h5 format.

    Args:
        path: Output file path.
        data: Dict with 'per_residue' (L, D) and 'protein_vec' (V,).
        protein_id: Identifier for the protein.
        tags: Optional freeform metadata dict (stored as root HDF5 attributes).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = ONE_H5_FORMAT
        f.attrs["version"] = ONE_H5_VERSION
        f.attrs["n_proteins"] = 1

        # Freeform root tags
        if tags:
            for k, v in tags.items():
                f.attrs[k] = v

        grp = f.create_group(protein_id)
        grp.create_dataset(
            "per_residue", data=data["per_residue"],
            compression="gzip", compression_opts=4,
        )
        grp.create_dataset("protein_vec", data=data["protein_vec"])
        grp.attrs["seq_len"] = data["per_residue"].shape[0]

        # Per-protein tags
        if "sequence" in data:
            grp.attrs["sequence"] = data["sequence"]
        if "tags" in data:
            for k, v in data["tags"].items():
                grp.attrs[k] = v


def read_one_h5(path) -> dict:
    """Read a .one.h5 file (single or batch with one protein).

    Returns dict with: per_residue, protein_vec, protein_id, format, version, tags.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        version = str(f.attrs.get("version", ""))

        # Collect root-level tags (skip format/version/n_proteins)
        skip = {"format", "version", "n_proteins"}
        tags = {k: _decode_attr(v) for k, v in f.attrs.items() if k not in skip}

        # Find the protein group (first group key)
        protein_ids = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        if not protein_ids:
            # Might be old .oemb single format
            return read_oemb(path)

        pid = protein_ids[0]
        grp = f[pid]
        return {
            "per_residue": grp["per_residue"][:].astype(np.float32),
            "protein_vec": grp["protein_vec"][:],
            "protein_id": pid,
            "format": str(fmt),
            "version": version,
            "tags": tags,
        }


def write_one_h5_batch(
    path,
    proteins: dict[str, dict],
    tags: dict | None = None,
) -> None:
    """Write multiple proteins to a batch .one.h5 file.

    Args:
        path: Output file path.
        proteins: Dict of {protein_id: {per_residue, protein_vec, [tags]}}.
        tags: Optional root-level freeform metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = ONE_H5_FORMAT
        f.attrs["version"] = ONE_H5_VERSION
        f.attrs["n_proteins"] = len(proteins)

        if tags:
            for k, v in tags.items():
                f.attrs[k] = v

        for pid, data in proteins.items():
            grp = f.create_group(pid)
            grp.create_dataset(
                "per_residue", data=data["per_residue"],
                compression="gzip", compression_opts=4,
            )
            grp.create_dataset("protein_vec", data=data["protein_vec"])
            grp.attrs["seq_len"] = data["per_residue"].shape[0]

            if "sequence" in data:
                grp.attrs["sequence"] = data["sequence"]
            if "tags" in data:
                for k, v in data["tags"].items():
                    grp.attrs[k] = v


def read_one_h5_batch(
    path,
    protein_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Read a batch .one.h5 file.

    Args:
        path: Path to .one.h5 file.
        protein_ids: Optional subset to load. None = all.

    Returns:
        Dict of {protein_id: {per_residue, protein_vec}}.
    """
    path = Path(path)
    result = {}
    with h5py.File(path, "r") as f:
        ids = protein_ids or [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        for pid in ids:
            grp = f[pid]
            result[pid] = {
                "per_residue": grp["per_residue"][:].astype(np.float32),
                "protein_vec": grp["protein_vec"][:],
            }
            if "sequence" in grp.attrs:
                result[pid]["sequence"] = str(grp.attrs["sequence"])
    return result


def inspect_one_h5(path) -> dict:
    """Inspect a .one.h5 file without loading embeddings.

    Returns metadata dict with: format, version, n_proteins, d_out,
    protein_vec_dim, protein_ids, tags, size_bytes.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        fmt = str(f.attrs.get("format", "unknown"))
        version = str(f.attrs.get("version", ""))
        n_proteins = int(f.attrs.get("n_proteins", 0))

        skip = {"format", "version", "n_proteins"}
        tags = {k: _decode_attr(v) for k, v in f.attrs.items() if k not in skip}

        protein_ids = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        d_out = 0
        vec_dim = 0
        if protein_ids:
            grp = f[protein_ids[0]]
            if "per_residue" in grp:
                d_out = grp["per_residue"].shape[1]
            if "protein_vec" in grp:
                vec_dim = grp["protein_vec"].shape[0]

    return {
        "format": fmt,
        "version": version,
        "n_proteins": n_proteins or len(protein_ids),
        "d_out": d_out,
        "protein_vec_dim": vec_dim,
        "protein_ids": protein_ids,
        "tags": tags,
        "size_bytes": path.stat().st_size,
    }


def _decode_attr(v):
    """Decode HDF5 attribute to Python type."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_io_one_h5.py -v`
Expected: All pass

- [ ] **Step 5: Run existing tests to verify backward compat**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/one_embedding/io.py tests/test_io_one_h5.py
git commit -m "feat: .one.h5 file format with freeform tags and backward compat"
```

---

### Task 3: Update top-level API (encode/decode) for new defaults

**Files:**
- Modify: `src/one_embedding/__init__.py`

- [ ] **Step 1: Update encode() default d_out**

In `src/one_embedding/__init__.py`, change the `encode` function signature:

```python
def encode(input_path, output_path, d_out=768, dct_k=4, seed=42):
```

Also update the output format: write `.one.h5` instead of `.oemb`. Use `write_one_h5_batch` instead of the current raw H5 writing. Include tags: `source_model`, `d_out`, `compression` (= `f"abtt3_rp{d_out}_dct{dct_k}"`), `seed`.

- [ ] **Step 2: Update decode() to handle both formats**

`decode()` should auto-detect `.one.h5` vs `.oemb` by trying `read_one_h5` first, falling back to `read_oemb`.

- [ ] **Step 3: Update version**

```python
__version__ = "1.0.0"
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/__init__.py
git commit -m "feat: encode() defaults to d_out=768, writes .one.h5 format"
```

---

### Task 4: Update CLI with --d-out flag and .one.h5 default

**Files:**
- Modify: `src/one_embedding/cli.py`

- [ ] **Step 1: Add --d-out option to encode command**

```python
@main.command()
@click.argument("input")
@click.option("-o", "--output", default=None, help="Output .one.h5 path")
@click.option("--d-out", default=768, type=int, help="Projection dimensionality (default: 768)")
def encode(input, output, d_out):
    """Compress per-residue H5 embeddings to .one.h5 format."""
    if output is None:
        output = input.rsplit(".", 1)[0] + ".one.h5"
    _encode(input, output, d_out=d_out)
    click.echo(f"Encoded to {output} (d={d_out})")
```

- [ ] **Step 2: Update inspect command for .one.h5**

Use `inspect_one_h5` with fallback to `inspect_oemb` for old files.

- [ ] **Step 3: Run CLI smoke test**

Run: `uv run python -m src.one_embedding.cli --help`
Expected: Shows encode with --d-out option

- [ ] **Step 4: Commit**

```bash
git add src/one_embedding/cli.py
git commit -m "feat: CLI --d-out flag, default .one.h5 output"
```

---

### Task 5: Tool auto-detection of embedding dimension

**Files:**
- Modify: `src/one_embedding/tools/_base.py`
- Modify: `src/one_embedding/tools/disorder.py`
- Modify: `src/one_embedding/tools/ss3.py`
- Create: `tests/test_tools_dimension.py`

- [ ] **Step 1: Write tests for auto-detection**

Create `tests/test_tools_dimension.py`:

```python
"""Tests for tool dimension auto-detection."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.io import write_one_h5
from src.one_embedding.tools._base import load_per_residue, load_protein_vecs


class TestBaseLoading:
    def test_load_per_residue_768d(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(40, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="p1")
        loaded = load_per_residue(str(path))
        assert loaded["p1"].shape == (40, 768)

    def test_load_per_residue_512d(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(40, 512).astype(np.float32),
            "protein_vec": np.random.randn(2048).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="p1")
        loaded = load_per_residue(str(path))
        assert loaded["p1"].shape == (40, 512)

    def test_load_protein_vecs_3072d(self, tmp_path):
        path = tmp_path / "test.one.h5"
        data = {
            "per_residue": np.random.randn(40, 768).astype(np.float32),
            "protein_vec": np.random.randn(3072).astype(np.float16),
        }
        write_one_h5(path, data, protein_id="p1")
        loaded = load_protein_vecs(str(path))
        assert loaded["p1"].shape == (3072,)
```

- [ ] **Step 2: Update _base.py to support .one.h5**

Update `load_per_residue` and `load_protein_vecs` in `src/one_embedding/tools/_base.py` to try `.one.h5` format first (check for "format" attr = "one_embedding"), then fall back to `.oemb` format.

```python
def load_per_residue(path) -> dict:
    """Load per-residue embeddings from .one.h5 or .oemb file."""
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if fmt == "one_embedding":
            # .one.h5 format: protein groups with per_residue datasets
            result = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Group) and "per_residue" in f[key]:
                    result[key] = f[key]["per_residue"][:].astype(np.float32)
            return result
        elif "per_residue" in f:
            # Old .oemb single format
            pid = str(f.attrs.get("protein_id", "protein"))
            return {pid: f["per_residue"][:].astype(np.float32)}
        else:
            # Old .oemb batch format
            result = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Group) and "per_residue" in f[key]:
                    result[key] = f[key]["per_residue"][:].astype(np.float32)
            return result
```

Same pattern for `load_protein_vecs`.

- [ ] **Step 3: Update disorder.py CNN to auto-detect D**

In `src/one_embedding/tools/disorder.py`, make the CNN input dim dynamic:

```python
class CNN(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


_WEIGHT_FILES = {
    512: "disorder_cnn_512d.pt",
    768: "disorder_cnn_768d.pt",
}


def _load_cnn(input_dim: int = 512):
    """Load CNN model for given input dimension."""
    cache_key = f"disorder_{input_dim}"
    if cache_key not in _MODEL_CACHE:
        weight_name = _WEIGHT_FILES.get(input_dim)
        if weight_name is None:
            raise ValueError(
                f"No disorder CNN weights for d={input_dim}. "
                f"Available: {sorted(_WEIGHT_FILES.keys())}"
            )
        weight_path = WEIGHT_DIR / weight_name
        if not weight_path.exists():
            raise FileNotFoundError(f"Weights not found: {weight_path}")
        model = CNN(input_dim=input_dim)
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def predict(oemb_path, method="cnn", **kwargs) -> dict:
    per_residue = load_per_residue(oemb_path)
    if not per_residue:
        return {}

    # Auto-detect D from first protein
    first_emb = next(iter(per_residue.values()))
    input_dim = first_emb.shape[1]

    if method == "cnn":
        model = _load_cnn(input_dim=input_dim)
        # ... rest of predict logic unchanged
```

- [ ] **Step 4: Apply same pattern to ss3.py**

Same changes: parameterize CNN input_dim, add weight file lookup, auto-detect D in predict().

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tools_dimension.py tests/test_io_one_h5.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/one_embedding/tools/_base.py src/one_embedding/tools/disorder.py src/one_embedding/tools/ss3.py tests/test_tools_dimension.py
git commit -m "feat: tools auto-detect embedding dimension, support 512d and 768d"
```

---

### Task 6: Integration test — full pipeline end to end

**Files:**
- Create: `tests/test_integration_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration test: raw embeddings → encode → .one.h5 → tools → results."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.core import Codec
from src.one_embedding.io import write_one_h5_batch, read_one_h5_batch, inspect_one_h5
from src.one_embedding.tools._base import load_per_residue, load_protein_vecs


def _make_corpus(n=20, D=1024, seed=42):
    rng = np.random.RandomState(seed)
    return {f"prot_{i}": rng.randn(30 + i * 5, D).astype(np.float32) for i in range(n)}


class TestFullPipeline:
    @pytest.mark.parametrize("d_out", [512, 768])
    def test_encode_write_load_roundtrip(self, tmp_path, d_out):
        """Codec.encode → write_one_h5_batch → read_one_h5_batch."""
        corpus = _make_corpus(n=10, D=1024)

        # Fit and encode
        codec = Codec(d_out=d_out)
        codec.fit(corpus)

        encoded = {}
        for pid, raw in corpus.items():
            result = codec.encode(raw)
            encoded[pid] = {
                "per_residue": result["per_residue"].astype(np.float32),
                "protein_vec": result["protein_vec"],
            }

        # Write .one.h5
        path = tmp_path / "encoded.one.h5"
        write_one_h5_batch(path, encoded, tags={
            "source_model": "prot_t5_xl",
            "d_out": d_out,
            "compression": f"abtt3_rp{d_out}",
        })

        # Inspect
        info = inspect_one_h5(path)
        assert info["n_proteins"] == 10
        assert info["d_out"] == d_out

        # Load via tools
        per_res = load_per_residue(str(path))
        vecs = load_protein_vecs(str(path))
        assert len(per_res) == 10
        assert len(vecs) == 10
        assert next(iter(per_res.values())).shape[1] == d_out
        assert next(iter(vecs.values())).shape[0] == 4 * d_out

    def test_tags_survive_roundtrip(self, tmp_path):
        corpus = _make_corpus(n=3, D=1024)
        codec = Codec(d_out=768)
        codec.fit(corpus)

        encoded = {}
        for pid, raw in corpus.items():
            result = codec.encode(raw)
            encoded[pid] = {
                "per_residue": result["per_residue"].astype(np.float32),
                "protein_vec": result["protein_vec"],
                "tags": {"organism": "human", "length": raw.shape[0]},
            }

        path = tmp_path / "tagged.one.h5"
        write_one_h5_batch(path, encoded, tags={
            "lab": "rostlab",
            "experiment": "test_38",
        })

        info = inspect_one_h5(path)
        assert info["tags"]["lab"] == "rostlab"
        assert info["tags"]["experiment"] == "test_38"
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_integration_pipeline.py -v`
Expected: All pass

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All tests pass (new + old)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_pipeline.py
git commit -m "test: integration test for full encode → .one.h5 → tools pipeline"
```

---

### Task 7: Update pre-fitted weights for 768d default

**Files:**
- Modify: `src/one_embedding/core/codec.py` (for_plm default)

- [ ] **Step 1: Update for_plm() to return 768d by default**

```python
@classmethod
def for_plm(cls, model: str = "prot_t5", d_out: int = 768) -> "Codec":
    """Get a pre-fitted codec for a given PLM."""
    codec = cls(d_out=d_out)
    # ... load weights as before
    return codec
```

- [ ] **Step 2: Verify pre-fitted weights work at 768d**

The ABTT weights (mean + top_pcs) are dimension-independent — they work with any d_out since ABTT operates in the original D=1024 space before projection. The projection matrix is seeded and generated on-the-fly. So existing `.npz` files work unchanged.

Run: `uv run python -c "from src.one_embedding.core import Codec; c = Codec.for_plm('prot_t5'); print(c.d_out)"`
Expected: `768`

- [ ] **Step 3: Commit**

```bash
git add src/one_embedding/core/codec.py
git commit -m "feat: Codec.for_plm() defaults to d_out=768"
```

---

### Task 8: Final cleanup and full test pass

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -30`
Expected: All tests pass, no warnings about deprecated behavior

- [ ] **Step 2: Verify backward compat with old .oemb files**

Run quick manual check — create an old-style .oemb, read it with new code:

```python
uv run python -c "
from src.one_embedding.io import write_oemb, read_one_h5
import numpy as np
write_oemb('/tmp/test_old.oemb', {
    'per_residue': np.random.randn(30, 512).astype(np.float32),
    'protein_vec': np.random.randn(2048).astype(np.float16),
}, protein_id='old')
# Try reading with new format reader (should fall back)
d = read_one_h5('/tmp/test_old.oemb')
print(f'Shape: {d[\"per_residue\"].shape}, ID: {d[\"protein_id\"]}')
"
```

Expected: `Shape: (30, 512), ID: old`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: One Embedding 1.0 Phase 1 complete — d_out=768 default, .one.h5 format"
```

- [ ] **Step 4: Push**

```bash
git push
```
