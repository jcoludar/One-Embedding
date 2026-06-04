# OE × autoeval on LRZ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark the One Embedding (OE) codec as a compression layer on 6 PLMs
(ESM2 8M/35M/150M/650M/3B + ProtT5-half) through biotrainer `autoeval` PBC, on LRZ AI,
producing per-task OE-vs-raw retention with paired bootstrap CIs.

**Architecture:** A local, unit-tested Python package (`oe_autoeval/`) provides the OE
embedding-function wrappers, shape guards, reference-set builder, and the paired
retention-CI statistics. On LRZ, a fresh login-node venv (biotrainer==1.4.0 + OE codec) and
pyxis/enroot SLURM jobs run the embedding + probe training; per-item predictions are
harvested and fed to the local stats package. Compute-heavy embedding runs once per
(PLM×mode) and is cached; only cheap probe training repeats across 5 seeds.

**Tech Stack:** Python 3.11/3.12, numpy/scipy/h5py, `OneEmbeddingCodec` (PEE `src/one_embedding`),
biotrainer 1.4.0 (autoeval PBC), transformers (ESM2/ProtT5 encoders), SLURM + pyxis/enroot
(`pytorch-2.4.0-cuda12.1.sqsh`) on LRZ AI.

**Source design:** `docs/oe_autoeval_lrz/DESIGN.md` (esp. §14 resolutions). This plan
implements §14.

> **Fan-review #2 COMPLETE.** Phases A–D below are the first draft; **§R (bottom) records
> the review resolutions and SUPERSEDES the draft where noted** — notably the
> `paired_retention` signature (multi-seed + item-id alignment), the biotrainer-direct
> harvest (autoeval cannot vary seed in 1.4.0), the `get_embedding_service` call, and the
> LRZ venv/sbatch/HF_HOME scripts. Read §R before implementing any task.

---

## Conventions & decisions (locked from DESIGN §14)

- PLMs + dims: ESM2-8M(320), 35M(480), 150M(640), 650M(1280), 3B(2560), ProtT5-half(1024).
  `d_out_eff = min(896, native_D)`: 320/480/640/896/896/896. RP engages only for 650M/3B/ProtT5.
- OE config: `OneEmbeddingCodec()` default (896d, binary, abtt_k=0, dct_k=4, seed=42), fitted.
- Per-residue rep = `decode_per_residue(encode(raw))` → float32 `(L, d_out_eff)`.
- Per-sequence rep (OE) = `encode(raw)['protein_vec']` → float32 `(4*d_out_eff,)`; (raw) = mean-pool `(native_D,)`.
- Modes: `raw`, `oe` (default), `oe_norp` (650M + ProtT5 only — RP-vs-noRP control).
- Precision: 8M/35M/150M = fp32; 650M/ProtT5/3B = fp16 embedding pass (both arms identical).
- Probe seeds: 5 per (PLM×mode×task). PBC = 9 tasks (7 residue, 2 sequence: phages, scl).
- Centering reference = union of PBC *train* splits, N≈2000, seed=42 (leakage-safe).
- Retention = metric_OE/metric_raw via paired bootstrap over common resampled test items + BCa;
  report absolute Δ alongside. OE-vs-raw uses LRZ raw arm ONLY (ga38fak 8M = harness check).
- Interpreter (local): `/Users/jcoludar/CascadeProjects/ProteEmbedExplorations/.venv/bin/python`.
- LRZ workspace: `…/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz/` (keep setgid group). HOME is off-limits.

---

# Phase A — Local package + tests (no LRZ; fully unit-testable)

All Phase-A code lives in PEE under `src/oe_autoeval/` with tests in `tests/oe_autoeval/`.
biotrainer is NOT importable locally, so the autoeval call is isolated behind an injection
boundary and everything else is tested with synthetic data + a fake embedder.

### Task A1: Shape guards + OE embedding-function wrappers

**Files:**
- Create: `src/oe_autoeval/__init__.py`
- Create: `src/oe_autoeval/wrappers.py`
- Test: `tests/oe_autoeval/test_wrappers.py`

- [ ] **Step 1: Write failing test** (`tests/oe_autoeval/test_wrappers.py`)

```python
import numpy as np
import pytest
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.oe_autoeval.wrappers import oe_per_residue, oe_per_sequence, raw_per_residue, raw_per_sequence

class FakeEmbedService:
    """Mimics biotrainer embedding service: yields (record, torch-like tensor)."""
    def __init__(self, embs):  # embs: dict seq -> np.ndarray (L, D)
        self._embs = embs
    def generate_embeddings(self, input_data, reduce):
        class Rec:  # minimal stand-in for BiotrainerSequenceRecord
            def __init__(self, seq): self.seq = seq
        class T:   # minimal stand-in for torch.Tensor
            def __init__(self, a): self._a = a
            def cpu(self): return self
            def numpy(self): return self._a
        for seq in input_data:
            yield Rec(seq), T(self._embs[seq])

def _fit_codec(d_out, embs):
    c = OneEmbeddingCodec(d_out=d_out)
    c.fit(embs)
    return c

def test_oe_per_residue_shape_and_dtype():
    rng = np.random.RandomState(0)
    embs = {"MKLV"*3: rng.randn(12, 320).astype(np.float32)}
    codec = _fit_codec(896, embs)          # 896>320 -> RP skipped, d_out_eff=320
    svc = FakeEmbedService(embs)
    out = dict(oe_per_residue(codec, svc, d_out_eff=320, sequences=list(embs)))
    (seq, arr), = out.items()
    assert arr.dtype == np.float32
    assert arr.shape == (12, 320)

def test_oe_per_sequence_shape():
    rng = np.random.RandomState(0)
    embs = {"MKLVMKLV": rng.randn(8, 1280).astype(np.float32)}  # 650M-like
    codec = _fit_codec(896, embs)          # 896<1280 -> RP -> d_out_eff=896
    svc = FakeEmbedService(embs)
    out = dict(oe_per_sequence(codec, svc, d_out_eff=896, sequences=list(embs)))
    (seq, vec), = out.items()
    assert vec.shape == (4*896,)
    assert vec.dtype == np.float32

def test_oe_per_sequence_rejects_short_protein():
    rng = np.random.RandomState(0)
    embs = {"MK": rng.randn(2, 320).astype(np.float32)}   # L=2 < dct_k=4
    codec = _fit_codec(896, embs)
    svc = FakeEmbedService(embs)
    with pytest.raises(ValueError, match="shorter than dct_k"):
        list(oe_per_sequence(codec, svc, d_out_eff=320, sequences=list(embs)))

def test_raw_per_sequence_is_mean_pool():
    embs = {"MKLV": np.ones((4, 320), np.float32) * 2.0}
    svc = FakeEmbedService(embs)
    out = dict(raw_per_sequence(svc, sequences=list(embs)))
    (seq, vec), = out.items()
    assert vec.shape == (320,)
    assert np.allclose(vec, 2.0)
```

- [ ] **Step 2: Run, verify it fails** — `…/.venv/bin/python -m pytest tests/oe_autoeval/test_wrappers.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** (`src/oe_autoeval/wrappers.py`)

```python
"""OE / raw embedding-function wrappers for biotrainer autoeval custom embedders.

autoeval validates only the per-residue ROW count (L), never the feature width, so a
wrong-width array is stored as silent garbage. These wrappers hard-assert dtype+shape.
"""
import numpy as np


def oe_per_residue(codec, embed_service, d_out_eff, sequences):
    """Yield (sequence_str, (L, d_out_eff) float32) = decode_per_residue(encode(raw))."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        raw = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        encoded = codec.encode(raw)
        arr = np.asarray(codec.decode_per_residue(encoded), dtype=np.float32)
        if arr.shape != (len(rec.seq), d_out_eff):
            raise ValueError(
                f"per-residue shape {arr.shape} != {(len(rec.seq), d_out_eff)} for seq len {len(rec.seq)}"
            )
        yield rec.seq, arr


def oe_per_sequence(codec, embed_service, d_out_eff, sequences):
    """Yield (sequence_str, (4*d_out_eff,) float32) = encode(raw)['protein_vec']."""
    expected = 4 * d_out_eff
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        raw = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        if raw.shape[0] < 4:
            raise ValueError(f"protein len {raw.shape[0]} shorter than dct_k=4 ({rec.seq[:8]}…)")
        encoded = codec.encode(raw)
        vec = np.asarray(encoded["protein_vec"], dtype=np.float32)
        if vec.shape != (expected,):
            raise ValueError(f"per-sequence shape {vec.shape} != {(expected,)}")
        yield rec.seq, vec


def raw_per_residue(embed_service, sequences):
    """Yield (sequence_str, (L, native_D) float32) — uncompressed baseline."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        yield rec.seq, np.asarray(emb.cpu().numpy(), dtype=np.float32)


def raw_per_sequence(embed_service, sequences):
    """Yield (sequence_str, (native_D,) float32) — mean-pooled baseline."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        arr = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        yield rec.seq, arr.mean(axis=0).astype(np.float32)
```

- [ ] **Step 4: Run, verify pass** — same pytest command → PASS (4 tests).
- [ ] **Step 5: Commit** — `git add src/oe_autoeval tests/oe_autoeval && git commit -m "feat(oe_autoeval): OE/raw embedding-fn wrappers with shape guards"`

### Task A2: Reference-set builder (PBC train-split union → centering FASTA)

**Files:**
- Create: `src/oe_autoeval/reference_set.py`
- Test: `tests/oe_autoeval/test_reference_set.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.oe_autoeval.reference_set import sample_reference_ids

def test_sample_reference_ids_is_deterministic_and_disjoint_from_test():
    train_ids = [f"tr{i}" for i in range(5000)]
    a = sample_reference_ids(train_ids, n=2000, seed=42)
    b = sample_reference_ids(train_ids, n=2000, seed=42)
    assert a == b                      # deterministic
    assert len(a) == 2000
    assert set(a).issubset(set(train_ids))  # never from test (only train passed in)

def test_sample_reference_ids_caps_at_population():
    a = sample_reference_ids(["x", "y"], n=2000, seed=42)
    assert sorted(a) == ["x", "y"]
```

- [ ] **Step 2: Run, fail.**
- [ ] **Step 3: Implement**

```python
"""Build the centering reference set from the UNION of PBC train splits.

Disjoint from every PBC test split by construction (only train IDs are ever passed in),
so per-channel-mean centering carries no test leakage. Deterministic under seed.
"""
import numpy as np


def sample_reference_ids(train_ids, n=2000, seed=42):
    """Return a deterministic sample of up to n ids from the pooled train ids."""
    ids = sorted(set(train_ids))
    if len(ids) <= n:
        return ids
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(idx)]
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit** — `feat(oe_autoeval): deterministic PBC-train reference-set sampler`

> NOTE: the LRZ-side glue that reads PBC FASTAs, pools train splits, calls
> `sample_reference_ids`, and writes `reference_2000.fasta` is in Task B3 (needs the
> downloaded datasets). The sampler itself is pure and tested here.

### Task A3: Paired retention bootstrap (the rigor core)

**Files:**
- Create: `src/oe_autoeval/retention.py`
- Test: `tests/oe_autoeval/test_retention.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.oe_autoeval.retention import accuracy, spearman, paired_retention

def test_accuracy_and_spearman_metrics():
    y = np.array([0, 1, 2, 1]); p = np.array([0, 1, 2, 0])
    assert np.isclose(accuracy(y, p), 0.75)
    yv = np.array([1.0, 2, 3, 4]); pv = np.array([1.0, 2, 3, 5])
    assert spearman(yv, pv) > 0.9

def test_paired_retention_identical_preds_gives_ratio_one():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, 200)
    pred = (y == rng.randint(0, 3, 200)).astype(int) * y  # arbitrary but identical for raw & oe
    res = paired_retention(y, pred, pred, metric="accuracy", n_boot=300, seed=42)
    assert np.isclose(res["ratio"], 1.0)
    assert res["ci_low"] <= 1.0 <= res["ci_high"]

def test_paired_retention_uses_common_resample():
    # OE slightly worse -> ratio < 1, and CI excludes 1 when the gap is consistent
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 400)
    raw_pred = y.copy()                      # perfect
    oe_pred = y.copy(); oe_pred[:40] ^= 1    # 10% worse, deterministic
    res = paired_retention(y, raw_pred, oe_pred, metric="accuracy", n_boot=500, seed=42)
    assert 0.85 < res["ratio"] < 0.95
    assert res["ci_high"] < 1.0
```

- [ ] **Step 2: Run, fail.**
- [ ] **Step 3: Implement**

```python
"""Paired bootstrap for retention = metric_OE / metric_raw over a COMMON resample.

Two independent autoeval bootstraps cannot be combined into a paired CI; this resamples the
SAME test-item indices for both arms (per-item predictions required). BCa-lite: percentile
CI plus a bias-corrected variant. Multi-seed: pass per-seed prediction arrays and average
the metric across matched seeds before resampling (probe noise partially cancels).
"""
import numpy as np
from scipy.stats import spearmanr, norm


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def spearman(y_true, y_pred):
    rho = spearmanr(np.asarray(y_true), np.asarray(y_pred)).correlation
    return float(rho if rho == rho else 0.0)  # NaN -> 0


_METRICS = {"accuracy": accuracy, "spearman": spearman}


def paired_retention(y_true, raw_pred, oe_pred, metric="accuracy", n_boot=1000, seed=42):
    """Per-item paired bootstrap. raw_pred/oe_pred aligned to y_true (same item order).

    Returns dict: point ratio, percentile CI, BCa CI, absolute deltas.
    """
    m = _METRICS[metric]
    y_true = np.asarray(y_true)
    raw_pred = np.asarray(raw_pred); oe_pred = np.asarray(oe_pred)
    n = len(y_true)
    assert raw_pred.shape[0] == oe_pred.shape[0] == n, "arms must align to y_true"

    m_raw0, m_oe0 = m(y_true, raw_pred), m(y_true, oe_pred)
    ratio0 = m_oe0 / m_raw0 if m_raw0 != 0 else float("nan")

    rng = np.random.RandomState(seed)
    ratios = np.empty(n_boot)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)                 # COMMON resample for both arms
        mr = m(y_true[idx], raw_pred[idx])
        mo = m(y_true[idx], oe_pred[idx])
        ratios[b] = mo / mr if mr != 0 else np.nan
        deltas[b] = mo - mr
    valid = ratios[~np.isnan(ratios)]
    lo, hi = np.percentile(valid, [2.5, 97.5])

    # BCa bias-correction (acceleration omitted -> "BC"): adjust for median bias
    z0 = norm.ppf((valid < ratio0).mean()) if 0 < (valid < ratio0).mean() < 1 else 0.0
    a = 0.0
    def _bca(alpha):
        zl = norm.ppf(alpha)
        return norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    bca_lo, bca_hi = np.percentile(valid, [100 * _bca(0.025), 100 * _bca(0.975)])

    return {
        "ratio": ratio0, "metric_raw": m_raw0, "metric_oe": m_oe0,
        "delta": m_oe0 - m_raw0,
        "ci_low": float(lo), "ci_high": float(hi),
        "bca_low": float(bca_lo), "bca_high": float(bca_hi),
        "delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "n_items": int(n), "n_boot": int(n_boot),
    }
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit** — `feat(oe_autoeval): paired bootstrap retention CI (per-item, BC)`

### Task A4: The driver (autoeval call behind an injection boundary)

**Files:**
- Create: `src/oe_autoeval/driver.py`
- Test: `tests/oe_autoeval/test_driver.py`

- [ ] **Step 1: Failing test** — test the wiring (mode→wrapper selection, d_out_eff math, codec
  config), with autoeval injected as a fake so no biotrainer import is needed.

```python
import numpy as np
from src.oe_autoeval.driver import d_out_eff_for, build_codec, select_embedding_functions

def test_d_out_eff_for_range():
    assert d_out_eff_for(320) == 320     # 8M  (RP skipped)
    assert d_out_eff_for(480) == 480     # 35M
    assert d_out_eff_for(640) == 640     # 150M
    assert d_out_eff_for(1280) == 896    # 650M (RP)
    assert d_out_eff_for(2560) == 896    # 3B
    assert d_out_eff_for(1024) == 896    # ProtT5

def test_build_codec_modes():
    assert build_codec("oe", native_d=1280).d_out == 896
    assert build_codec("oe_norp", native_d=1280).d_out == 1280   # force no RP
    assert build_codec("raw", native_d=1280) is None

def test_select_embedding_functions_raw_vs_oe():
    fns_raw = select_embedding_functions("raw", codec=None, d_out_eff=320)
    fns_oe = select_embedding_functions("oe", codec=object(), d_out_eff=320)
    assert fns_raw["per_residue"].__name__ == "raw_pr"
    assert fns_oe["per_residue"].__name__ == "oe_pr"
```

- [ ] **Step 2: Run, fail.**
- [ ] **Step 3: Implement** (`src/oe_autoeval/driver.py`) — pure wiring + a `run()` that takes an
  injected `autoeval_pipeline` callable (default: import from biotrainer at call time).

```python
"""Parametrized OE×autoeval driver. The biotrainer import is deferred so the wiring is
unit-testable off-cluster."""
import functools
import numpy as np
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.oe_autoeval import wrappers

D_OUT = 896


def d_out_eff_for(native_d):
    return min(D_OUT, native_d)


def build_codec(mode, native_d):
    if mode == "raw":
        return None
    if mode == "oe":
        return OneEmbeddingCodec()                      # d_out=896 default
    if mode == "oe_norp":
        return OneEmbeddingCodec(d_out=native_d)        # force RP-off (control arm)
    raise ValueError(mode)


def select_embedding_functions(mode, codec, d_out_eff):
    if mode == "raw":
        def raw_pr(sequences): return wrappers.raw_per_residue(_SVC.get(), sequences)
        def raw_ps(sequences): return wrappers.raw_per_sequence(_SVC.get(), sequences)
        return {"per_residue": raw_pr, "per_sequence": raw_ps}
    def oe_pr(sequences): return wrappers.oe_per_residue(codec, _SVC.get(), d_out_eff, sequences)
    def oe_ps(sequences): return wrappers.oe_per_sequence(codec, _SVC.get(), d_out_eff, sequences)
    return {"per_residue": oe_pr, "per_sequence": oe_ps}


class _SVC:
    """Process-global embedding service handle (set by run(); keeps closures picklable-free)."""
    _svc = None
    @classmethod
    def set(cls, svc): cls._svc = svc
    @classmethod
    def get(cls):
        if cls._svc is None:
            raise RuntimeError("embedding service not initialized")
        return cls._svc


def fit_codec_from_reference(codec, reference_embeddings):
    """reference_embeddings: dict id -> (L, native_D). No-op for raw mode."""
    if codec is not None:
        codec.fit(reference_embeddings)
    return codec
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit** — `feat(oe_autoeval): parametrized driver wiring (mode/d_out_eff/codec)`

> The `run()` entrypoint that (a) builds the biotrainer embedding service, (b) embeds the
> reference set + fits the codec, (c) calls `autoeval_pipeline(...)` with the selected
> functions — is finalized in Task B5 once the 1.4.0 API is verified on LRZ (G1). Its body is
> ~15 lines and specified there. Keeping it out of Phase A avoids encoding an unverified API.

### Task A5: Full local test pass + lint

- [ ] **Step 1:** `…/.venv/bin/python -m pytest tests/oe_autoeval/ -q` → all green.
- [ ] **Step 2:** `…/.venv/bin/python -m pytest -q` (full suite still green; no regressions to the 878 existing tests).
- [ ] **Step 3: Commit** any fixups — `test(oe_autoeval): green local suite`.

---

# Phase B — LRZ environment (Phase-2 execution; scripted now, RUN only after plan approval)

> Every Phase-B/C/D step touches the cluster. Per the user's decision, these are authored now
> but executed only after this plan is approved. All scripts live in `scripts/lrz_oe_autoeval/`
> in PEE and are rsynced to `…/oe_autoeval_lrz/` on LRZ.

### Task B1: Fresh login-node venv with biotrainer 1.4.0 + OE codec — and G1 gate

**Files:** Create `scripts/lrz_oe_autoeval/00_build_venv.sh`

- [ ] **Step 1:** Author `00_build_venv.sh` (run on the LRZ **login node**):

```bash
#!/usr/bin/env bash
set -euo pipefail
WORK=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz
python3.11 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --upgrade pip
pip install "biotrainer==1.4.0"
pip install -e "$WORK/ProteEmbedExplorations"   # PEE checkout (OE codec), Task B2
python - <<'PY'
from biotrainer.autoeval import autoeval_pipeline   # G1: import must succeed
import inspect; sig = inspect.signature(autoeval_pipeline)
for k in ("embedder_name","framework","custom_embedding_function_per_residue","custom_embedding_function_per_sequence"):
    assert k in sig.parameters, f"MISSING kwarg {k}"
print("G1 OK:", list(sig.parameters))
PY
```

- [ ] **Step 2 (G1 gate):** Running it prints `G1 OK: [...]` with all four kwargs present.
  If biotrainer 1.4.0 pulls a conflicting transformers and breaks import → pin a compatible
  transformers in a constraints file and re-run. **Do not proceed past G1 failure.**
- [ ] **Step 3:** Record `pip freeze > "$WORK/venv.freeze.txt"` (dep closure for offline compute).
- [ ] **Step 4: Commit** the script (not the venv).

### Task B2: Deploy the OE codec (PEE) to LRZ

**Files:** Create `scripts/lrz_oe_autoeval/01_deploy_repo.sh`

- [ ] **Step 1:** Script: `git clone`/`rsync` the PEE repo to `$WORK/ProteEmbedExplorations`,
  `git rev-parse HEAD > $WORK/PEE_SHA.txt`. (Used by B1's `pip install -e`.)
- [ ] **Step 2:** Verify `python -c "from src.one_embedding.codec_v2 import OneEmbeddingCodec"`
  inside the venv. **Commit.**

### Task B3: Pre-stage PBC datasets + build reference FASTA — G3 gate

**Files:** Create `scripts/lrz_oe_autoeval/02_prestage_pbc.py`, `03_build_reference.py`

- [ ] **Step 1:** `02_prestage_pbc.py` (login node, has internet): trigger a CPU dry-run of
  `autoeval_pipeline(..., device="cpu")` just far enough to download PBC into
  `$BIOTRAINER_CACHE` (or call the dataset loader directly). Then hash the dataset dir
  (`sha256` of the sorted file list+sizes) → `$WORK/pbc_dataset.hash`. **G3 gate:** download
  completes, hash recorded.
- [ ] **Step 2:** `03_build_reference.py`: read PBC **train** FASTAs across all 9 tasks, pool ids,
  call `src.oe_autoeval.reference_set.sample_reference_ids(train_ids, n=2000, seed=42)`, write
  `$WORK/reference_2000.fasta`. Assert disjoint from every PBC **test** fasta (set-intersection
  == ∅) and log the count. **G4 gate** (L<4): log any sequence with len<4 across PBC; if the
  per-sequence tasks (phages/scl) contain any, the wrapper guard will raise — decide drop vs pad
  then (expected: none).
- [ ] **Step 3: Commit** scripts.

### Task B4: G2 verification + retention plumbing decision

**Files:** Create `scripts/lrz_oe_autoeval/04_probe_g2.py`

- [ ] **Step 1:** On the login node (CPU), run autoeval for ONE tiny task and inspect the
  per-task output dir for: (a) a biotrainer `out.yml`, (b) whether test-set per-item predictions
  are written, (c) whether `save_split_ids`/seed can be injected via autoeval's per-task config.
- [ ] **Step 2 (G2 decision):**
  - **Path 1 (autoeval-native):** if autoeval exposes per-task config overrides → set
    `save_split_ids: true` + vary `seed` across 5 runs into distinct `output_dir`s; harvest
    predictions from each `out.yml` via `biotrainer.inference.Inferencer.create_from_out_file`.
  - **Path 2 (biotrainer-direct fallback, guaranteed):** use autoeval only to embed + materialize
    the 9 PBC datasets, then drive `biotrainer` directly per (task×seed) on the **cached
    embeddings** with our own config (`seed: k`, `save_split_ids: true`), reading predictions
    from each `out.yml`. This always works (biotrainer is a standard config trainer) and fully
    controls the 5 seeds.
- [ ] **Step 3:** Record the chosen path in `$WORK/G2_DECISION.md`. **Commit** the probe script.

### Task B5: Finalize `driver.run()` against the verified 1.4.0 API

**Files:** Modify `src/oe_autoeval/driver.py`; create `scripts/lrz_oe_autoeval/run_arm.py`

- [ ] **Step 1:** Implement `run(embedder_hf_id, mode, native_d, precision, label, output_dir,
  reference_fasta, autoeval_pipeline=None)`:
  build `get_embedding_service(embedder_hf_id, device, use_half_precision=(precision=="fp16"))`;
  `_SVC.set(svc)`; embed `reference_fasta` (raw) → dict → `fit_codec_from_reference`; select
  functions; call the injected/imported `autoeval_pipeline(embedder_name=label, framework="PBC",
  custom_embedding_function_per_residue=fns["per_residue"], custom_embedding_function_per_sequence=fns["per_sequence"], output_dir=output_dir, use_half_precision=(precision=="fp16"))`.
- [ ] **Step 2:** `run_arm.py` = thin CLI (argparse: `--embedder --mode --native-d --precision
  --label --out --reference`) calling `driver.run(...)`. Used by every sbatch.
- [ ] **Step 3:** Add an integration test marked `@pytest.mark.lrz` (skipped locally) that asserts
  `run()` calls the injected pipeline with the right kwargs (fake pipeline records the call).
- [ ] **Step 4: Commit** — `feat(oe_autoeval): driver.run against verified biotrainer 1.4.0 API`.

### Task B6: enroot/pyxis sbatch template

**Files:** Create `scripts/lrz_oe_autoeval/arm.sbatch`

- [ ] **Step 1:** Template (parametrized by env: `EMBEDDER, MODE, NATIVE_D, PRECISION, LABEL,
  PARTITION, QOS`):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=oe_%x
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
WORK=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz
srun --container-image="$WORK/images/pytorch-2.4.0-cuda12.1.sqsh" \
     --container-mounts="$WORK:/work:rw" \
     bash -lc '
       set -euo pipefail
       export MKL_THREADING_LAYER=GNU
       export HF_HOME=/work/hf_cache
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
       export BIOTRAINER_CACHE=/work/biotrainer_cache
       source /work/venv/bin/activate
       python /work/scripts/run_arm.py \
         --embedder "'"$EMBEDDER"'" --mode "'"$MODE"'" --native-d "'"$NATIVE_D"'" \
         --precision "'"$PRECISION"'" --label "'"$LABEL"'" \
         --reference /work/reference_2000.fasta --out "/work/out/'"$LABEL"'"
     '
```

- [ ] **Step 2:** A `submit_all.sh` that exports the per-arm env and `sbatch`es each arm with the
  right `--partition/--qos` (Task D mapping). Respect MaxJobsPU=10 (it queues; fine).
- [ ] **Step 3: Commit.**

---

# Phase C — Calibration + the falsifiable validation gate

### Task C1: 8M raw calibration vs ga38fak baseline (harness gate)

- [ ] **Step 1:** Submit the `ESM2-8M / raw / fp32` arm on `lrz-v100x2 / qos=gpu`.
- [ ] **Step 2 (validation gate):** compare our 8M-raw per-task metrics to
  `autoeval_report_facebook-esm2_t6_8M_UR50D.json`. **Pass = per-task `|Δmetric| <
  max(2×probe-seed SD, 0.02)` on every PBC task AND our raw-8M within ga38fak's per-task CI on
  ≥7/9 tasks.** First confirm the invariants match: biotrainer==1.4.0, `pbc_dataset.hash`, CV
  split assignment, `max_seq_len=2000`, epochs=10, bootstrap=30, fp32. Log the comparison to
  `$WORK/out/CALIBRATION_8M.md`.
- [ ] **Step 3:** If the gate FAILS, stop and diagnose (which invariant diverged) before any
  further GPU. Do NOT compute retention against ga38fak regardless.

### Task C2: 8M OE end-to-end smoke

- [ ] **Step 1:** Submit `ESM2-8M / oe / fp32`. Confirm: codec fit ran, per-residue h5 has width
  320 (not packed 40), all 9 tasks produced reports, no shape asserts tripped.
- [ ] **Step 2:** Eyeball OE-vs-raw 8M retention (single seed) for sanity (expect ~0.95–1.0 on
  SS/retrieval-like tasks, lower on disorder). **Commit** calibration notes.

---

# Phase D — Full sweep + analysis

### Task D1: Fan out all arms × 5 seeds

- [ ] **Step 1:** Arm × partition/precision matrix:

| PLM | native_d | modes | precision | partition / qos |
|---|---:|---|---|---|
| ESM2-8M | 320 | raw, oe | fp32 | lrz-v100x2 / gpu |
| ESM2-35M | 480 | raw, oe | fp32 | lrz-v100x2 / gpu |
| ESM2-150M | 640 | raw, oe | fp32 | lrz-hgx-a100-80x4 / gpu |
| ESM2-650M | 1280 | raw, oe, oe_norp | fp16 | lrz-hgx-a100-80x4 / gpu |
| ESM2-3B | 2560 | raw, oe | fp16 | lrz-hgx-h100-94x4 / gpu |
| ProtT5-half | 1024 | raw, oe, oe_norp | fp16 | lrz-hgx-a100-80x4 / gpu |

- [ ] **Step 2:** For each arm, run the embedding+task pipeline once (cached), then the 5 probe
  seeds via the G2-chosen path (Task B4), writing per-item predictions per (arm×task×seed).
  14 embedding arms total; respect MaxJobsPU=10 (auto-queues).
- [ ] **Step 3:** Sidecar per run (args, PEE SHA, biotrainer ver, GPU model, CUDA/cuDNN,
  determinism flags) per CLAUDE reproducibility rule.

### Task D2: Compute retention + tables/figures

**Files:** Create `scripts/lrz_oe_autoeval/05_analyze.py`

- [ ] **Step 1:** For each (PLM×task): load raw + OE per-item predictions (aligned by item id),
  average metric across matched seeds, call `paired_retention(...)` → ratio + BCa CI + Δ.
- [ ] **Step 2:** Emit `results/retention_table.csv` (rows=PLM with native_d, d_out_eff,
  compression×, RP-engaged?; cols=9 PBC tasks; cells=retention±CI) + an absolute raw-vs-OE table.
  Separate the two regimes visually (RP vs no-RP) and include the oe_norp control deltas for
  650M/ProtT5. Frame as estimation; BH-FDR only if a "significant degradation" claim is made.
- [ ] **Step 3:** Barplots via the repo's `experiments/make_benchmark_barplots.py` conventions.
- [ ] **Step 4: Commit** analysis + results; write `docs/oe_autoeval_lrz/RESULTS.md`.

---

## Self-review (spec coverage vs DESIGN §14)

- §14.1-1/2 decode+fit → A1/A4/B5. §14.1-3 asserts/L<4 → A1. §14.1-4 9 tasks → B3/D2.
  §14.1-5 cache key mode → B4/B6 (label encodes mode). §14.1-6 ProtT5-half → D1 matrix.
  §14.1-7 fp16 big → D1. §14.1-8 reference set → A2/B3. §14.1-9 multi-seed → B4/D1.
  §14.1-10 paired CI → A3/D2. §14.1-11 RP control → A4(oe_norp)/D1/D2. §14.1-12 gate → C1.
  §14.1-13 estimation/FDR → D2. §14.2 infra → B1/B2/B6. §14.4 gates G1→B1, G2→B4, G3→B3,
  G4→B3, G5→A1(local)+C2(cluster). **No gaps found.**
- Open dependency: B5 + the integration test depend on the G1-verified API; flagged inline,
  not a placeholder.

## Execution handoff

This plan is authored for review first (fan-review #2). Phase A is local + safe to execute on
approval; Phases B–D touch LRZ and run only after the gates pass in order.

---

# §R. Fan-review #2 — findings & resolutions (2026-06-03)

Four reviewers (Phase-A code, autoeval integration, LRZ scripts, rigor fidelity). Net: the
wrapper/guard/codec layer is faithful; the **statistics core** and the **LRZ scripts** have
real defects that change the result or block execution. Resolutions below supersede the draft.

## R1 — Statistics core: `paired_retention` rewrite (Code-C1, Rigor-C1, Rigor-C2) — CRITICAL

The draft `paired_retention(y_true, raw_pred, oe_pred)` is single-seed and only asserts equal
length. Two blocking holes: (a) the 5-seed design has no executable home — "average across
seeds before resampling" discards the very probe-seed variance §14.1-9 exists to expose;
(b) a *paired* bootstrap requires the SAME test items in both arms, which equal-length does
not guarantee (autoeval's hold-out split is seed-derived; independent arms get different
splits). **Revised spec (replaces Task A3's signature + Task D2 Step-1):**

```python
def paired_retention(item_ids_raw, item_ids_oe, y_true, raw_preds, oe_preds,
                     metric="accuracy", n_boot=1000, seed=42):
    """Multi-seed paired bootstrap. raw_preds/oe_preds are (S, N) per-seed per-item
    predictions; y_true is (N,). item_ids_* are length-N id arrays for each arm.

    HARD INVARIANT: item_ids_raw == item_ids_oe (same ids, SAME order) — the arms must share
    the identical test split per (task, seed). Caller aligns/reorders by id first.

    Estimator: seeds are a RESAMPLE DIMENSION, not a pre-average. Each bootstrap iteration
    draws a common item index `idx` (size N) AND a matched seed s (same s for both arms, so
    RNG-paired probe noise partially cancels), then ratio_b = m(y[idx], oe[s][idx]) /
    m(y[idx], raw[s][idx]). Point ratio = mean over s of the full-sample matched-seed ratio.
    CI = BCa over the (item × seed) bootstrap with jackknife-over-items acceleration.
    Also returns between_seed_sd = SD of the S full-sample per-seed ratios (the probe-noise
    floor that 'OE deficit > probe noise' is judged against), and absolute delta + delta_ci.
    """
```

- Caller (D2) MUST pass per-seed prediction stacks + ids and align by id (reorder, never
  assume positional). Add tests: (i) 5 seeds, consistent gap → CI excludes 1, small
  `between_seed_sd`; (ii) one outlier seed → CI widens (seed variance propagates); (iii)
  mismatched `item_ids` → raises.
- Compute true **BCa** (jackknife acceleration over items), not BC — the repo convention is
  BCa. Rename all "BCa" keys accordingly (Rigor-M3).
- Small-denominator (disorder ρ): D2 **prefers the absolute-Δ panel whenever `metric_raw` is
  within ~2 SD of 0**, not just emits both (Rigor-M4).

## R2 — Shared split-ids across arms (Rigor-C2, Integration-C2/C3) — CRITICAL

For each (task, seed) the **split assignment is fixed once and shared by raw + oe + oe_norp**.
Mechanism (the only one that works in 1.4.0, see R3): pass an explicit pre-computed split-id
file into every arm's biotrainer config, OR assert post-hoc `split_ids(arm, task, seed)` are
identical across arms. Add **split-id-manifest identity** and **PEE git SHA** to the C1
invariant list (M1).

## R3 — autoeval cannot vary seed/save predictions in 1.4.0 → commit to biotrainer-direct (Integration-C2/C3) — CRITICAL

Verified in v1.4.0: the PBC config bank hardcodes `seed: 42`, `model_choice: LogReg`, and
never sets `save_split_ids`; the only injection hook (`add_custom_values_to_config`) sets just
`embedder_name/input_file/output_dir/embeddings_file/device`. So **Task B4 Path-1 is
impossible — drop it.** Path-2 (biotrainer-direct on cached embeddings) is the ONLY route, and
it's better: we fully control the config. Revised Task B4/D1 harvest:

- Use autoeval ONLY to materialize the 9 PBC datasets + (optionally) the embeddings cache.
- For each (task × seed), run `biotrainer` directly on the cached embeddings h5 with our config:
  `model_choice: CNN, num_epochs: 10, batch_size: 64, bootstrapping_iterations: 30, seed: k,
  save_split_ids: true` — i.e. **match the ga38fak baseline config (CNN/10/30), not the 1.4.0
  PBC default (LogReg)**, which also resolves Integration-I3 (the baseline used CNN, so stock
  PBC LogReg would neither match the gate nor the paper).
- Harvest per-item test predictions: `inferencer, iom = Inferencer.create_from_out_file(out.yml)`
  (it returns a **2-tuple**, and out.yml is read *without* split-ids), then re-run
  `inferencer.from_embeddings(test_embeddings, split_name="test")` →
  `{'metrics','mapped_predictions','mapped_probabilities'}`. Join raw↔OE on the **sequence
  hash** (autoeval stores `store_by_hash=True`, `seq_id=f"Seq{idx}"`), not the FASTA header.

## R4 — `get_embedding_service` call is positionally wrong (Integration-C1) — CRITICAL

1.4.0 signature: `get_embedding_service(embedder_name, custom_tokenizer_config,
use_half_precision=False, device=None, ...)`. The draft `get_embedding_service(embedder_hf_id,
device, …)` binds `device` to `custom_tokenizer_config`. **Use keywords (Task B5):**
`get_embedding_service(embedder_name=hf_id, custom_tokenizer_config=None,
use_half_precision=(precision=="fp16"), device=device)`. Also: `autoeval_pipeline` is a
**generator** — `run()` and the B3 pre-stage must drain it (`for _ in …: pass`) or nothing
happens (Integration-I1).

## R5 — LRZ scripts: venv, HF_HOME, sbatch form (LRZ-C1/C2/C3, I1/I2/I3) — CRITICAL/IMPORTANT

Recon-confirmed corrections (the working `plm_choice_lrz` / `taxembed_lrz` jobs are the
template):

- **Venv (LRZ-C1):** login node has only python3.10, no conda. Build the venv **inside the
  container**: `srun --container-image=<sqsh> … /opt/conda/bin/python3.11 -m venv
  --system-site-packages /work/venv` (so it inherits the container torch/CUDA). NOT on the
  login node, NOT python3.11 (absent), NOT without `--system-site-packages`.
- **HF_HOME (LRZ-C2):** point at the REAL cache `…/plm_choice_lrz/data/hf_cache` (all 6 models
  already there) — bind-mount it read-only and `export HF_HOME=<that>`. The draft's empty
  `/work/hf_cache` + offline flags = guaranteed failure.
- **sbatch (LRZ-C3):** use `#SBATCH --container-image=…/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh`,
  `#SBATCH --container-mounts=$WORK:/work:rw,<hf_cache>:<hf_cache>:ro`, `#SBATCH
  --container-workdir=/work` **directives** (not `srun --container-image`), body = `source
  /work/venv/bin/activate; python …`. Reference the image at its real `taxembed_lrz` path
  (I1). Add the missing `#SBATCH --partition/--qos/--cpus-per-task/--mem/--error` (I2/I3) —
  size `--mem`/`--cpus-per-task` to the 3B arm; partitions verified: `lrz-v100x2`,
  `lrz-hgx-a100-80x4`, `lrz-hgx-h100-94x4` all exist and accept `--qos=gpu` (MaxJobsPU=10).
- **Dataset cache (LRZ-I4):** the controlling env var for biotrainer's PBC cache is **unverified**
  — make "confirm the real cache env var + that the pre-staged dir is read offline by the
  container" an explicit **G3 sub-check**; ensure the login-node pre-stage dir and the
  container mount are the same physical path.

## R6 — RP-isolation contrast must be analyzed, not just collected (Rigor-I1) — IMPORTANT

Add to D2: for 650M and ProtT5, compute the RP contrast per task = `paired(oe vs oe_norp)`
over the same items (both share the model's raw embeddings) → ΔRP ± CI. Cross-PLM retention
trends must cite this contrast, not infer RP from model size. Otherwise oe_norp is wasted GPU.

## R7 — Soften the leakage claim (Rigor-I2) — IMPORTANT

The B3 assert covers **exact-id** disjointness only. Reword §14.1-8 / B3 to: "reference set is
exact-id-disjoint from all PBC test splits; centering is an unsupervised per-channel mean over
~2000 sequences, so residual homology influence is negligible" — drop "leakage-safe by
construction" (overclaims). (Optional stronger variant: mmseqs ≥30% dedup vs test.)

## R8 — Test/precision/repro minutiae (Code-I1/I3/M4, Rigor-I3/M2) — MINOR

- Add a per-residue wrapper test at the **RP-engaged** path (D=1280→896); the draft only tests
  RP-skipped (Code-I1).
- Document the silent **fp16→fp32 upcast** of `protein_vec` (codec emits fp16) — load-bearing;
  footnote in D2 that the 2 sequence-task (phages/scl) OE features carry fp16 precision while
  raw is fp32 (Code-I3, Rigor-M2).
- Add `sys.path.insert(0, repo_root)` to the test files for consistency with the other 49 test
  files / non-`python -m` invocation (Code-M4).
- **Do NOT force cuDNN-deterministic for probe training** — the 5-seed spread is the intended
  noise floor; log the per-seed metric vector in the sidecar (Rigor-I3).

## R9 — Revised gate ordering note

G1 (venv import) now depends on R5's in-container build. The ga38fak 8M gate (C1) is meaningful
ONLY after R3's CNN/10/30 config match; otherwise it is informational. OE-vs-raw still uses the
LRZ raw arm exclusively.

---

# §R3. Fan-review #3 (implementation) — APPLIED fixes (2026-06-03)

Four reviewers read the as-implemented `src/oe_autoeval/` package (25 tests green) + the LRZ
scripts. Fixes applied this session:

- **C (code bug): `oe_norp` width.** `driver.run` derived `d_eff = min(896, native_d)` = 896 for
  the no-RP control arms, but the no-RP codec emits `native_d` (1280/1024) → wrapper shape guard
  would crash every oe_norp arm. **Fixed:** `d_eff = min(codec.d_out, native_d)`. Added a
  regression test (`test_run_oe_norp_uses_native_width`).
- **Stat-C1: residue correlation.** Per-residue tasks flatten to per-residue items; resampling
  residues i.i.d. understates variance. **Fixed:** `paired_retention` now takes `group_ids` and
  does a **cluster bootstrap** (resample whole proteins) with **leave-one-group-out** jackknife;
  `05_analyze` passes protein id as the group for the 7 residue tasks, `None` for phages/scl.
  Test: `test_cluster_bootstrap_widens_ci_vs_iid`.
- **Stat-I1: estimator coherence.** The bootstrap now **averages over all S seeds per resample**
  (matches the point estimate), so `z0` is coherent; probe-seed variance is reported separately
  as `between_seed_sd`. Test updated (`test_outlier_seed_inflates_between_seed_sd`).
- **Stat-I2: decision rule.** `05_analyze` `deficit_exceeds_noise` is now units-consistent:
  `ci_high < 1 AND (1 - ratio) > 2·(between_seed_sd/√S)` (both on the ratio scale).
- **Stat-I3 / Code-I1: small-denominator + NaN.** Δ now has its own **BCa** CI; `05_analyze`
  flags `ratio_unreliable_prefer_delta` when raw spearman < 0.1; `between_seed_sd` guards on the
  non-NaN seed count.
- **E2E-C1/C2: the missing harvest.** Added `06_harvest_predictions.py` — the biotrainer-direct
  multi-seed (CNN/10/30, seeds 42–46) harvest that writes the `predictions.json`
  (`item_ids/y_true/y_pred/group_ids`) `05_analyze` consumes, with a **FIXED split shared across
  all arms and seeds** (only the probe seed varies → valid multi-seed *and* paired) and
  zero-padded residue ids (stable lexical sort). README run-order corrected (G2 probe after the
  8M-oe arm; harvest before analyze).

Remaining (correctly gated, cannot close until biotrainer is in-container): G1 venv import, G2
harvest mechanics (`04`/`06` Config/Inferencer specifics), G3 PBC cache env var + on-disk layout.
