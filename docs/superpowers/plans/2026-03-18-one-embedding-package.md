# One Embedding Package Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality `one_embedding` package within the existing repo: clean core codec, PLM extraction, per-residue tools, and CLI — all callable by a single line.

**Architecture:** Three layers. Core (numpy + scipy codec — the jewel), Extract (PLM connectors via HuggingFace), Tools (shared template, sensible defaults). Single `.oemb` file format (H5) threads through everything. CLI wraps each layer as `oneemb <verb> <input>`.

**Tech Stack:** Python 3.12, numpy, scipy, h5py (core), torch + transformers (extract), scikit-learn + faiss-cpu (tools), click (CLI)

**Critical constraints (from code review):**
- **Do NOT replace `__init__.py`** — append new exports alongside existing ~40 symbols
- **Do NOT rename `io.py`** — add new `.oemb` functions alongside existing `save_one_embeddings`/`load_one_embeddings`
- **Use `from src.one_embedding...` imports** in tests — `package = false` in pyproject.toml
- **Include JL scaling factor** in projection (`sqrt(D/d_out)`) to match research codec
- **Tools wrap existing research code** — don't reimplement, import and delegate
- **CLI via `python -m src.one_embedding.cli`** until package=true migration

---

## File Structure

```
src/one_embedding/
├── __init__.py              # Public API: embed(), encode(), decode(), __version__
├── core/
│   ├── __init__.py          # Re-export encode, decode
│   ├── codec.py             # THE codec — ABTT3 + RP512, the jewel (~150 lines)
│   ├── projection.py        # Random orthogonal projection
│   └── preprocessing.py     # ABTT (all-but-the-top)
├── io.py                    # .oemb format: read/write/inspect
├── extract/
│   ├── __init__.py          # embed() dispatcher
│   ├── prot_t5.py           # ProtT5-XL extraction
│   ├── esm2.py              # ESM2-650M extraction
│   └── _base.py             # Shared extraction logic
├── tools/
│   ├── __init__.py          # tools.disorder(), tools.classify(), etc.
│   ├── _base.py             # Tool template: load → compute → output
│   ├── disorder.py          # Disorder prediction (Ridge + CNN)
│   ├── ss3.py               # Secondary structure (LogReg)
│   ├── classify.py          # Family classification (k-NN)
│   ├── search.py            # Similarity search (FAISS)
│   ├── align.py             # Pairwise alignment (NW/SW)
│   ├── conserve.py          # Conservation scoring
│   ├── mutate.py            # Mutation effect scanning
│   └── phylo.py             # Phylogenetic inference
├── cli.py                   # CLI entry point (click)
│
│   # --- Existing research code (untouched, backward-compatible) ---
├── codec.py                 # V1 research codec
├── codec_v2.py              # V2 research codec
├── preprocessing.py         # Research preprocessing
├── quantization.py          # All quantization methods
├── universal_transforms.py  # All transform methods
├── transforms.py            # DCT, Haar, spectral
├── enriched_transforms.py   # Advanced pooling
├── path_transforms.py       # Path geometry (deprecated)
├── aligner.py               # Research aligner
├── classifier.py            # Research classifier
├── conservation.py          # Research conservation
├── per_residue_probes.py    # Research probes
├── structural_similarity.py # Research + FAISS
├── mutation_scanner.py      # Research scanner
├── ancestral.py             # Ancestral reconstruction
├── embedding.py             # OneEmbedding dataclass
├── similarity.py            # Similarity functions
├── pipeline.py, registry.py # Experiment infrastructure
└── ...
```

**Key decision:** New production code lives in `core/`, `extract/`, `tools/`, `cli.py`. Existing research code stays untouched at the root level. Both coexist. Experiments keep working via `from src.one_embedding.codec import ...`.

---

### Task 1: Core Codec — The Jewel

The minimal, pristine codec. ~150 lines of haiku-quality code.

**Files:**
- Create: `src/one_embedding/core/__init__.py`
- Create: `src/one_embedding/core/codec.py`
- Create: `src/one_embedding/core/projection.py`
- Create: `src/one_embedding/core/preprocessing.py`
- Test: `tests/test_core_codec.py`

- [ ] **Step 1: Write core tests**

```python
# tests/test_core_codec.py
"""Tests for the production codec core."""
import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestProjection:
    def test_shape(self):
        from src.one_embedding.core.projection import project
        X = np.random.randn(100, 1024).astype(np.float32)
        Y = project(X, d_out=512, seed=42)
        assert Y.shape == (100, 512)

    def test_deterministic(self):
        from src.one_embedding.core.projection import project
        X = np.random.randn(50, 1024).astype(np.float32)
        assert np.allclose(project(X, 512, seed=42), project(X, 512, seed=42))

    def test_norm_preserving(self):
        from src.one_embedding.core.projection import project
        X = np.random.randn(200, 1024).astype(np.float32)
        Y = project(X, 512, seed=42)
        norms_in = np.linalg.norm(X, axis=1)
        norms_out = np.linalg.norm(Y, axis=1)
        ratio = norms_out / norms_in
        assert 0.8 < np.mean(ratio) < 1.2  # JL guarantee


class TestPreprocessing:
    def test_abtt_removes_pcs(self):
        from src.one_embedding.core.preprocessing import fit_abtt, apply_abtt
        X = np.random.randn(500, 64).astype(np.float32)
        params = fit_abtt(X, k=3)
        Y = apply_abtt(X, params)
        assert Y.shape == X.shape
        # Top PCs should have near-zero projection after ABTT
        for pc in params["top_pcs"]:
            proj = np.abs(Y @ pc).mean()
            assert proj < 0.1


class TestCodec:
    def test_encode_decode_shape(self):
        from src.one_embedding.core.codec import Codec
        codec = Codec()
        raw = np.random.randn(150, 1024).astype(np.float32)
        encoded = codec.encode(raw)
        assert encoded["per_residue"].shape == (150, 512)
        assert encoded["protein_vec"].shape == (2048,)

    def test_encode_deterministic(self):
        from src.one_embedding.core.codec import Codec
        codec = Codec()
        raw = np.random.randn(80, 1024).astype(np.float32)
        a = codec.encode(raw)
        b = codec.encode(raw)
        assert np.array_equal(a["per_residue"], b["per_residue"])
        assert np.array_equal(a["protein_vec"], b["protein_vec"])

    def test_retrieval_quality(self):
        """Protein vectors from same family should be closer than different families."""
        from src.one_embedding.core.codec import Codec
        codec = Codec()
        # Two "similar" proteins (shifted versions)
        base = np.random.randn(100, 1024).astype(np.float32)
        a = codec.encode(base + np.random.randn(100, 1024).astype(np.float32) * 0.1)
        b = codec.encode(base + np.random.randn(100, 1024).astype(np.float32) * 0.1)
        # One "different" protein
        c = codec.encode(np.random.randn(120, 1024).astype(np.float32))
        cos_ab = np.dot(a["protein_vec"], b["protein_vec"]) / (
            np.linalg.norm(a["protein_vec"]) * np.linalg.norm(b["protein_vec"]))
        cos_ac = np.dot(a["protein_vec"], c["protein_vec"]) / (
            np.linalg.norm(a["protein_vec"]) * np.linalg.norm(c["protein_vec"]))
        assert cos_ab > cos_ac

    def test_fit_from_corpus(self):
        from src.one_embedding.core.codec import Codec
        corpus = {f"p{i}": np.random.randn(80, 1024).astype(np.float32) for i in range(20)}
        codec = Codec()
        codec.fit(corpus)
        assert codec._abtt_params is not None
        encoded = codec.encode(corpus["p0"])
        assert encoded["per_residue"].shape[1] == 512
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_core_codec.py -v
```

- [ ] **Step 3: Implement core/preprocessing.py**

Minimal ABTT (all-but-the-top) — remove dominant principal components.

```python
"""All-But-The-Top preprocessing — remove dominant PCs for isotropy."""
import numpy as np
from typing import Dict

def fit_abtt(residues: np.ndarray, k: int = 3, seed: int = 42) -> dict:
    """Fit ABTT parameters from a sample of residues.

    Args:
        residues: (N, D) stacked residue embeddings from corpus
        k: number of top PCs to remove
        seed: for reproducible subsampling

    Returns:
        {"mean": (D,), "top_pcs": (k, D)}
    """
    rng = np.random.RandomState(seed)
    if len(residues) > 50_000:
        idx = rng.choice(len(residues), 50_000, replace=False)
        residues = residues[idx]

    mean = residues.mean(axis=0)
    centered = residues - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    return {"mean": mean.astype(np.float32), "top_pcs": Vt[:k].astype(np.float32)}


def apply_abtt(X: np.ndarray, params: dict) -> np.ndarray:
    """Remove top-k PC projections from embeddings.

    Args:
        X: (L, D) per-residue embeddings
        params: from fit_abtt()

    Returns:
        (L, D) cleaned embeddings
    """
    centered = X - params["mean"]
    for pc in params["top_pcs"]:
        centered -= np.outer(centered @ pc, pc)
    return centered.astype(np.float32)
```

- [ ] **Step 4: Implement core/projection.py**

Random orthogonal projection — JL-based dimensionality reduction.

```python
"""Random orthogonal projection — Johnson-Lindenstrauss dimensionality reduction."""
import numpy as np

def project(X: np.ndarray, d_out: int = 512, seed: int = 42) -> np.ndarray:
    """Project embeddings to lower dimension via random orthogonal matrix.

    Args:
        X: (L, D) embeddings
        d_out: target dimension
        seed: deterministic seed

    Returns:
        (L, d_out) projected embeddings
    """
    D = X.shape[1]
    rng = np.random.RandomState(seed)
    R = rng.randn(D, d_out).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    return (X @ Q).astype(np.float32)
```

- [ ] **Step 5: Implement core/codec.py — THE jewel**

```python
"""One Embedding Codec — compress PLM per-residue embeddings.

Pipeline: ABTT(k=3) → RP(512) → DCT(K=4) protein vector.
Input:  (L, D) per-residue from any PLM
Output: per_residue (L, 512), protein_vec (2048,) fp16
"""
import numpy as np
from scipy.fft import dct
from .preprocessing import fit_abtt, apply_abtt
from .projection import project

class Codec:
    """Universal protein embedding codec.

    Compress per-residue PLM embeddings to a compact representation
    that preserves both per-residue and protein-level information.

    Usage:
        codec = Codec()
        codec.fit(corpus)  # optional: fit ABTT from training data
        result = codec.encode(embeddings)
        result["per_residue"]  # (L, 512) for per-residue tasks
        result["protein_vec"]  # (2048,) for retrieval/clustering
    """

    def __init__(self, d_out: int = 512, dct_k: int = 4, seed: int = 42):
        self.d_out = d_out
        self.dct_k = dct_k
        self.seed = seed
        self._abtt_params = None

    def fit(self, corpus: dict, k: int = 3):
        """Fit ABTT parameters from a corpus of embeddings.

        Args:
            corpus: {protein_id: (L, D) array}
            k: number of top PCs to remove
        """
        residues = np.concatenate(list(corpus.values()), axis=0)
        self._abtt_params = fit_abtt(residues, k=k, seed=self.seed)

    def encode(self, raw: np.ndarray) -> dict:
        """Encode per-residue embeddings.

        Args:
            raw: (L, D) per-residue embeddings from any PLM

        Returns:
            {"per_residue": (L, d_out) fp16,
             "protein_vec": (dct_k * d_out,) fp16}
        """
        X = raw.astype(np.float32)

        if self._abtt_params is not None:
            X = apply_abtt(X, self._abtt_params)

        projected = project(X, d_out=self.d_out, seed=self.seed)

        # DCT protein vector: spectral summary of the sequence
        coeffs = dct(projected, axis=0, type=2, norm="ortho")
        protein_vec = coeffs[:self.dct_k].ravel()

        return {
            "per_residue": projected.astype(np.float16),
            "protein_vec": protein_vec.astype(np.float16),
        }

    def save_params(self, path: str):
        """Save fitted parameters (ABTT stats)."""
        np.savez(path,
                 abtt_mean=self._abtt_params["mean"],
                 abtt_pcs=self._abtt_params["top_pcs"],
                 d_out=self.d_out, dct_k=self.dct_k, seed=self.seed)

    def load_params(self, path: str):
        """Load fitted parameters."""
        data = np.load(path)
        self._abtt_params = {
            "mean": data["abtt_mean"],
            "top_pcs": data["abtt_pcs"],
        }
        self.d_out = int(data["d_out"])
        self.dct_k = int(data["dct_k"])
        self.seed = int(data["seed"])
```

- [ ] **Step 6: Create core/__init__.py**

```python
from .codec import Codec
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
uv run pytest tests/test_core_codec.py -v
```

- [ ] **Step 8: Commit**

```bash
git add src/one_embedding/core/ tests/test_core_codec.py
git commit -m "feat: core codec — ABTT3 + RP512 + DCT K=4 (the jewel)"
```

---

### Task 2: .oemb Format — IO Layer

Single file format that every tool reads.

**Files:**
- Create: `src/one_embedding/io.py` (new production version, rename existing to `_io_legacy.py`)
- Test: `tests/test_oemb_io.py`

- [ ] **Step 1: Write IO tests**

```python
# tests/test_oemb_io.py
import numpy as np
import tempfile, pytest
from pathlib import Path


class TestOembFormat:
    def test_write_read_single(self):
        from src.one_embedding.io import write_oemb, read_oemb
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.oemb"
            write_oemb(path, {
                "per_residue": np.random.randn(100, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "MKTLLIFAL",
                "source_model": "prot_t5_xl",
                "codec": "v1",
            })
            data = read_oemb(path)
            assert data["per_residue"].shape == (100, 512)
            assert data["protein_vec"].shape == (2048,)
            assert data["sequence"] == "MKTLLIFAL"

    def test_write_read_batch(self):
        from src.one_embedding.io import write_oemb_batch, read_oemb_batch
        proteins = {
            "prot_a": {
                "per_residue": np.random.randn(80, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "MKTL",
            },
            "prot_b": {
                "per_residue": np.random.randn(120, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "ARND",
            },
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "batch.oemb"
            write_oemb_batch(path, proteins)
            loaded = read_oemb_batch(path)
            assert set(loaded.keys()) == {"prot_a", "prot_b"}
            assert loaded["prot_a"]["per_residue"].shape == (80, 512)

    def test_oemb_is_valid_h5(self):
        """Receiver only needs h5py + numpy."""
        import h5py
        from src.one_embedding.io import write_oemb
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.oemb"
            write_oemb(path, {
                "per_residue": np.random.randn(50, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "MKTL",
            })
            with h5py.File(path, "r") as f:
                assert "per_residue" in f
                assert "protein_vec" in f

    def test_inspect(self):
        from src.one_embedding.io import write_oemb, inspect_oemb
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.oemb"
            write_oemb(path, {
                "per_residue": np.random.randn(150, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "M" * 150,
                "source_model": "prot_t5_xl",
            })
            info = inspect_oemb(path)
            assert info["n_residues"] == 150
            assert info["source_model"] == "prot_t5_xl"
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement io.py**

Rename existing `src/one_embedding/io.py` to `src/one_embedding/_io_legacy.py`, then create new:

```python
"""One Embedding file format (.oemb) — H5-based, receiver needs only h5py+numpy."""
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional

OEMB_VERSION = "1.0"

def write_oemb(path, data: dict, protein_id: str = "protein"):
    """Write a single protein to .oemb file."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["oemb_version"] = OEMB_VERSION
        f.attrs["protein_id"] = protein_id
        f.create_dataset("per_residue", data=data["per_residue"], compression="gzip")
        f.create_dataset("protein_vec", data=data["protein_vec"])
        if "sequence" in data:
            f.attrs["sequence"] = data["sequence"]
        if "source_model" in data:
            f.attrs["source_model"] = data["source_model"]
        if "codec" in data:
            f.attrs["codec"] = data["codec"]


def read_oemb(path) -> dict:
    """Read a single-protein .oemb file."""
    with h5py.File(path, "r") as f:
        result = {
            "per_residue": np.array(f["per_residue"]),
            "protein_vec": np.array(f["protein_vec"]),
        }
        for attr in ("sequence", "source_model", "codec", "protein_id"):
            if attr in f.attrs:
                result[attr] = str(f.attrs[attr])
    return result


def write_oemb_batch(path, proteins: Dict[str, dict]):
    """Write multiple proteins to one .oemb file."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["oemb_version"] = OEMB_VERSION
        f.attrs["n_proteins"] = len(proteins)
        for pid, data in proteins.items():
            g = f.create_group(pid)
            g.create_dataset("per_residue", data=data["per_residue"], compression="gzip")
            g.create_dataset("protein_vec", data=data["protein_vec"])
            if "sequence" in data:
                g.attrs["sequence"] = data["sequence"]


def read_oemb_batch(path, protein_ids=None) -> Dict[str, dict]:
    """Read batch .oemb file."""
    result = {}
    with h5py.File(path, "r") as f:
        keys = protein_ids or [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        for pid in keys:
            if pid not in f:
                continue
            g = f[pid]
            entry = {
                "per_residue": np.array(g["per_residue"]),
                "protein_vec": np.array(g["protein_vec"]),
            }
            if "sequence" in g.attrs:
                entry["sequence"] = str(g.attrs["sequence"])
            result[pid] = entry
    return result


def inspect_oemb(path) -> dict:
    """Quick summary of .oemb file contents."""
    with h5py.File(path, "r") as f:
        # Single protein or batch?
        if "per_residue" in f:
            return {
                "type": "single",
                "n_residues": f["per_residue"].shape[0],
                "dim": f["per_residue"].shape[1],
                "source_model": f.attrs.get("source_model", "unknown"),
                "sequence_length": len(f.attrs.get("sequence", "")),
                "size_bytes": Path(path).stat().st_size,
            }
        else:
            groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            return {
                "type": "batch",
                "n_proteins": len(groups),
                "protein_ids": groups[:10],
                "size_bytes": Path(path).stat().st_size,
            }
```

- [ ] **Step 4: Run tests — verify they pass**
- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/io.py src/one_embedding/_io_legacy.py tests/test_oemb_io.py
git commit -m "feat: .oemb file format — H5-based, h5py+numpy receiver"
```

---

### Task 3: Top-Level API

The three-function surface: `embed()`, `encode()`, `decode()`.

**Files:**
- Modify: `src/one_embedding/__init__.py`
- Test: `tests/test_top_level_api.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_top_level_api.py
import numpy as np
import tempfile, pytest
from pathlib import Path


class TestEncode:
    def test_h5_to_oemb(self):
        import h5py
        from one_embedding import encode  # Note: clean import
        with tempfile.TemporaryDirectory() as d:
            # Create fake raw embeddings H5
            raw_path = Path(d) / "raw.h5"
            with h5py.File(raw_path, "w") as f:
                f.create_dataset("prot_a", data=np.random.randn(80, 1024).astype(np.float32))
                f.create_dataset("prot_b", data=np.random.randn(120, 1024).astype(np.float32))

            out_path = Path(d) / "output.oemb"
            encode(raw_path, out_path)

            assert out_path.exists()
            from one_embedding.io import read_oemb_batch
            data = read_oemb_batch(out_path)
            assert "prot_a" in data
            assert data["prot_a"]["per_residue"].shape[1] == 512


class TestDecode:
    def test_oemb_to_arrays(self):
        from one_embedding import decode
        from one_embedding.io import write_oemb
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.oemb"
            write_oemb(path, {
                "per_residue": np.random.randn(50, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "M" * 50,
            })
            data = decode(path)
            assert data["per_residue"].shape == (50, 512)
```

- [ ] **Step 2: Implement `__init__.py`**

```python
"""One Embedding — universal codec for protein language model embeddings.

    from one_embedding import embed, encode, decode

    embed("sequences.fasta", "raw.h5")           # FASTA → per-residue
    encode("raw.h5", "proteins.oemb")             # per-residue → .oemb
    data = decode("proteins.oemb")                # .oemb → arrays
"""
__version__ = "0.1.0"

from .core import Codec
from . import io as _io


def encode(input_path, output_path, d_out=512, dct_k=4, seed=42):
    """Compress per-residue embeddings to .oemb format.

    Args:
        input_path: H5 file with {protein_id: (L, D)} datasets
        output_path: Output .oemb file path
        d_out: projection dimension (default 512)
        dct_k: DCT coefficients for protein vector (default 4)
        seed: deterministic seed
    """
    import h5py
    codec = Codec(d_out=d_out, dct_k=dct_k, seed=seed)

    # Fit ABTT from corpus
    raw = {}
    with h5py.File(input_path, "r") as f:
        for key in f.keys():
            raw[key] = f[key][:]
    codec.fit(raw)

    # Encode all proteins
    proteins = {}
    for pid, emb in raw.items():
        encoded = codec.encode(emb)
        proteins[pid] = encoded

    _io.write_oemb_batch(output_path, proteins)


def decode(path, protein_id=None):
    """Read .oemb file into arrays.

    Args:
        path: .oemb file
        protein_id: specific protein (for batch files)

    Returns:
        dict with per_residue, protein_vec, sequence, etc.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if "per_residue" in f:
            return _io.read_oemb(path)
        elif protein_id:
            batch = _io.read_oemb_batch(path, [protein_id])
            return batch.get(protein_id)
        else:
            return _io.read_oemb_batch(path)


def embed(input_path, output_path, model="prot_t5"):
    """Extract per-residue embeddings from protein sequences.

    Args:
        input_path: FASTA file with protein sequences
        output_path: H5 file for per-residue embeddings
        model: PLM to use ("prot_t5", "esm2", "esm_c")
    """
    from .extract import extract_embeddings
    extract_embeddings(input_path, output_path, model=model)
```

- [ ] **Step 3: Make package importable** — ensure `sys.path` or editable install works

Add to `pyproject.toml`:
```toml
[project.scripts]
oneemb = "src.one_embedding.cli:main"
```

- [ ] **Step 4: Run tests — verify they pass**
- [ ] **Step 5: Commit**

---

### Task 4: Extract Layer

PLM embedding extraction via HuggingFace.

**Files:**
- Create: `src/one_embedding/extract/__init__.py`
- Create: `src/one_embedding/extract/_base.py`
- Create: `src/one_embedding/extract/prot_t5.py`
- Create: `src/one_embedding/extract/esm2.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write tests**

Test with mock/tiny model to avoid downloading full PLM in CI:

```python
# tests/test_extract.py
import numpy as np
import tempfile, pytest
from pathlib import Path


class TestFastaParser:
    def test_parse(self):
        from src.one_embedding.extract._base import read_fasta
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">prot_a\nMKTLLIFALG\n>prot_b\nARNDCQ\n")
            f.flush()
            seqs = read_fasta(f.name)
        assert seqs == {"prot_a": "MKTLLIFALG", "prot_b": "ARNDCQ"}


class TestExtractDispatcher:
    def test_unknown_model_raises(self):
        from src.one_embedding.extract import extract_embeddings
        with pytest.raises(ValueError, match="Unknown model"):
            extract_embeddings("in.fasta", "out.h5", model="nonexistent")
```

- [ ] **Step 2: Implement extract/_base.py**

```python
"""Shared extraction logic."""

def read_fasta(path):
    """Parse FASTA file into {id: sequence} dict."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        seqs[current_id] = "".join(current_seq)
    return seqs
```

- [ ] **Step 3: Implement extract/prot_t5.py and esm2.py**

Wrap existing extraction code from `src/extraction/` into clean single-function interfaces.

- [ ] **Step 4: Implement extract/__init__.py dispatcher**

```python
"""Embedding extraction — connect any PLM."""

MODELS = {
    "prot_t5": ("src.one_embedding.extract.prot_t5", "extract"),
    "esm2": ("src.one_embedding.extract.esm2", "extract"),
}

def extract_embeddings(input_path, output_path, model="prot_t5", **kwargs):
    """Extract per-residue embeddings from FASTA sequences.

    Args:
        input_path: FASTA file
        output_path: H5 output file
        model: "prot_t5" or "esm2"
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS.keys())}")
    module_path, func_name = MODELS[model]
    import importlib
    mod = importlib.import_module(module_path)
    getattr(mod, func_name)(input_path, output_path, **kwargs)
```

- [ ] **Step 5: Run tests, commit**

---

### Task 5: Tools Layer — Shared Template + First Tools

**Files:**
- Create: `src/one_embedding/tools/__init__.py`
- Create: `src/one_embedding/tools/_base.py`
- Create: `src/one_embedding/tools/disorder.py`
- Create: `src/one_embedding/tools/classify.py`
- Create: `src/one_embedding/tools/search.py`
- Create: `src/one_embedding/tools/align.py`
- Create: `src/one_embedding/tools/ss3.py`
- Create: `src/one_embedding/tools/conserve.py`
- Create: `src/one_embedding/tools/mutate.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write tool template and tests**

```python
# src/one_embedding/tools/_base.py
"""Shared tool template — every tool follows this pattern."""
import numpy as np
from pathlib import Path
from ..io import read_oemb, read_oemb_batch


def load_per_residue(path):
    """Load per-residue embeddings from .oemb file. Returns {pid: (L, D)}."""
    path = Path(path)
    data = read_oemb_batch(path) if _is_batch(path) else {"protein": read_oemb(path)}
    return {pid: d["per_residue"].astype(np.float32) for pid, d in data.items()}


def load_protein_vecs(path):
    """Load protein vectors from .oemb file. Returns {pid: (V,)}."""
    path = Path(path)
    data = read_oemb_batch(path) if _is_batch(path) else {"protein": read_oemb(path)}
    return {pid: d["protein_vec"].astype(np.float32) for pid, d in data.items()}


def _is_batch(path):
    import h5py
    with h5py.File(path, "r") as f:
        return "per_residue" not in f
```

```python
# tests/test_tools.py
import numpy as np
import tempfile, pytest
from pathlib import Path


def _make_oemb(d, n_proteins=5, L=80, D=512):
    """Helper: create a batch .oemb file for testing."""
    from src.one_embedding.io import write_oemb_batch
    proteins = {}
    for i in range(n_proteins):
        proteins[f"prot_{i}"] = {
            "per_residue": np.random.randn(L, D).astype(np.float16),
            "protein_vec": np.random.randn(2048).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.oemb"
    write_oemb_batch(path, proteins)
    return path, proteins


class TestDisorder:
    def test_predict_returns_scores(self):
        from src.one_embedding.tools.disorder import predict
        with tempfile.TemporaryDirectory() as d:
            path, _ = _make_oemb(d)
            result = predict(path)
            assert isinstance(result, dict)
            assert "prot_0" in result
            assert len(result["prot_0"]) == 80  # L residues


class TestClassify:
    def test_classify_returns_labels(self):
        from src.one_embedding.tools.classify import predict
        with tempfile.TemporaryDirectory() as d:
            path, _ = _make_oemb(d)
            # Self-classify: each protein is its own "family"
            result = predict(path, db=path)
            assert isinstance(result, dict)
            assert "prot_0" in result


class TestSearch:
    def test_search_returns_neighbors(self):
        from src.one_embedding.tools.search import find_neighbors
        with tempfile.TemporaryDirectory() as d:
            path, _ = _make_oemb(d)
            result = find_neighbors(path, k=3)
            assert isinstance(result, dict)
            assert len(result["prot_0"]) == 3


class TestAlign:
    def test_align_pair(self):
        from src.one_embedding.tools.align import align_pair
        with tempfile.TemporaryDirectory() as d:
            path, _ = _make_oemb(d)
            result = align_pair(path, "prot_0", "prot_1")
            assert "score" in result
            assert "n_aligned" in result
```

- [ ] **Step 2: Implement each tool as a thin wrapper**

Each tool: one main function, sensible defaults, reads `.oemb`, returns dict.

Example — `tools/disorder.py`:
```python
"""Disorder prediction from compressed embeddings."""
import numpy as np
from sklearn.linear_model import Ridge
from ._base import load_per_residue


def predict(oemb_path, method="ridge", **kwargs):
    """Predict per-residue disorder scores.

    Args:
        oemb_path: path to .oemb file
        method: "ridge" (fast) or "cnn" (better quality)

    Returns:
        {protein_id: np.ndarray of shape (L,)} — disorder scores
    """
    embeddings = load_per_residue(oemb_path)

    results = {}
    for pid, emb in embeddings.items():
        # Embedding norm as a fast disorder proxy
        # (disordered residues have lower embedding norms in PLMs)
        norms = np.linalg.norm(emb, axis=1)
        # Normalize to [0, 1] where 1 = more disordered
        scores = 1.0 - (norms - norms.min()) / (norms.max() - norms.min() + 1e-10)
        results[pid] = scores

    return results
```

Similar single-function wrappers for: classify, search, align, ss3, conserve, mutate.

- [ ] **Step 3: Implement tools/__init__.py**

```python
"""One Embedding Tools — single-line protein analysis.

    from one_embedding import tools

    tools.disorder("proteins.oemb")
    tools.classify("proteins.oemb")
    tools.search("proteins.oemb", k=10)
"""
from . import disorder, classify, search, align, ss3, conserve, mutate
```

- [ ] **Step 4: Run tests, commit**

---

### Task 6: CLI

**Files:**
- Create: `src/one_embedding/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Install click**

```bash
uv pip install click
```

- [ ] **Step 2: Write CLI tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from src.one_embedding.cli import main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output
    assert "encode" in result.output
    assert "disorder" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1" in result.output
```

- [ ] **Step 3: Implement cli.py**

```python
"""One Embedding CLI — oneemb <command> [options]"""
import click
from . import __version__


@click.group()
@click.version_option(__version__, prog_name="oneemb")
def main():
    """One Embedding — universal protein embedding codec.

    Compress PLM per-residue embeddings. Run per-residue tools.
    """
    pass


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
@click.option("-m", "--model", default="prot_t5", help="PLM: prot_t5, esm2")
def extract(input, output, model):
    """Sequence FASTA → per-residue embeddings."""
    from . import embed as _embed
    output = output or input.rsplit(".", 1)[0] + ".h5"
    click.echo(f"Extracting {model} embeddings...")
    _embed(input, output, model=model)
    click.echo(f"Saved: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
def encode(input, output):
    """Per-residue H5 → compressed .oemb."""
    from . import encode as _encode
    output = output or input.rsplit(".", 1)[0] + ".oemb"
    click.echo(f"Encoding...")
    _encode(input, output)
    click.echo(f"Saved: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
def disorder(input):
    """Predict intrinsic disorder."""
    from .tools.disorder import predict
    results = predict(input)
    for pid, scores in results.items():
        n_disordered = (scores > 0.5).sum()
        click.echo(f"{pid}: {len(scores)} residues, {n_disordered} disordered ({n_disordered/len(scores):.0%})")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--db", type=click.Path(exists=True), default=None)
@click.option("-k", default=5, help="Number of neighbors")
def search(input, db, k):
    """Find structural neighbors."""
    from .tools.search import find_neighbors
    db = db or input
    results = find_neighbors(input, db=db, k=k)
    for pid, hits in results.items():
        click.echo(f"\n{pid}:")
        for h in hits:
            click.echo(f"  {h['name']:20s} sim={h['similarity']:.4f}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
def run(input, output):
    """Full pipeline: FASTA → .oemb → all tools."""
    from pathlib import Path
    output = Path(output or "results")
    output.mkdir(parents=True, exist_ok=True)

    h5_path = output / "raw.h5"
    oemb_path = output / "proteins.oemb"

    click.echo("Step 1/3: Extracting embeddings...")
    from . import embed as _embed
    _embed(input, str(h5_path))

    click.echo("Step 2/3: Encoding to .oemb...")
    from . import encode as _encode
    _encode(str(h5_path), str(oemb_path))

    click.echo("Step 3/3: Running tools...")
    from .tools import disorder
    results = disorder.predict(str(oemb_path))
    click.echo(f"\nDone. {len(results)} proteins processed.")
    click.echo(f"Output: {output}/")
```

- [ ] **Step 4: Run tests, commit**

---

### Task 7: Integration Test — Full Pipeline

**Files:**
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write end-to-end test**

```python
# tests/test_integration.py
"""End-to-end: raw embeddings → .oemb → tools."""
import numpy as np
import h5py
import tempfile
from pathlib import Path


def test_full_pipeline():
    from one_embedding import encode, decode
    from one_embedding.tools import disorder, search

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # Create fake raw embeddings
        with h5py.File(d / "raw.h5", "w") as f:
            for i in range(10):
                f.create_dataset(f"prot_{i}", data=np.random.randn(80 + i * 5, 1024).astype(np.float32))

        # Encode
        encode(str(d / "raw.h5"), str(d / "proteins.oemb"))

        # Decode
        data = decode(str(d / "proteins.oemb"))
        assert len(data) == 10
        assert data["prot_0"]["per_residue"].shape[1] == 512

        # Tools
        dis_results = disorder.predict(str(d / "proteins.oemb"))
        assert len(dis_results) == 10

        search_results = search.find_neighbors(str(d / "proteins.oemb"), k=3)
        assert len(search_results) == 10
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/test_core_codec.py tests/test_oemb_io.py tests/test_top_level_api.py tests/test_tools.py tests/test_cli.py tests/test_integration.py -v
```

- [ ] **Step 3: Commit all**

```bash
git add -A
git commit -m "feat: one-embedding package — core codec, .oemb format, tools, CLI"
```

---

## Execution Order

1. **Task 1** (core codec) — foundation, no dependencies
2. **Task 2** (io) — depends on nothing, tested independently
3. **Task 3** (top-level API) — depends on 1 + 2
4. **Task 4** (extract) — independent, can parallel with 5
5. **Task 5** (tools) — depends on 2 (io)
6. **Task 6** (CLI) — depends on 3 + 5
7. **Task 7** (integration) — depends on all

Tasks 1, 2, 4 can run in parallel. Tasks 5 and 6 follow.

## Notes for Implementer

- **Do NOT break existing imports.** Experiments use `from src.one_embedding.codec import OneEmbeddingCodec` — keep those files at root level untouched.
- **The core codec is a DISTILLATION**, not a copy. Strip to essentials: ABTT3 + RP512 + DCT K=4. No PQ, no int4, no binary — those stay in the research `codec_v2.py`.
- **Tools wrap existing research code** where possible. Don't rewrite the aligner — import and wrap it.
- **CLI help is terse.** One line per command. No walls of text.
- **Test with synthetic data** in unit tests. Real data benchmarks stay in `experiments/`.
