# Hand-Off Doc — Run the Codec in 15 Minutes

**Date:** 2026-04-27
**Status:** Real hand-off (Task G.1). Verified end-to-end on M3 Max, fresh `uv sync`.

> Prior (2026-04-26 stub) preserved as Appendix for calibration.

This document gets a labmate (or you, six months from now) from "I have the
clone" to "I just reproduced a benchmark cell" in 15 minutes.

---

## 0. Prerequisites (2 min)

- Python 3.12 (`uv` will manage the venv)
- `git`, `uv` (`brew install uv`)
- Optional: `npm` (only if you want to rebuild the slide deck)
- ~70 GB free disk if you regenerate the PLM embeddings; ~5 GB if you only run the codec on pre-extracted data

---

## 1. Setup (3 min)

```bash
git clone <repo-url> ProteEmbedExplorations
cd ProteEmbedExplorations
uv sync --all-extras --all-groups   # installs torch/transformers/faiss/marp + dev tooling
uv run pytest tests/                 # 813 / 813 should pass; ~90s on M3 Max
```

If any test fails, **stop** — `git status` likely shows local edits that
shouldn't be there, or the venv is stale. Do not proceed until the baseline is
green.

---

## 2. Where the data lives (1 min)

| Path | Contents | Size | Tracked? |
|------|----------|:---:|:---:|
| `data/residue_embeddings/` | Per-PLM `(L, D)` H5 files (ProtT5, ESM2, ESM-C, ProstT5, ANKH) | 67 GB | gitignored |
| `data/benchmark_suite/per_residue/` | CB513, TS115, CASP12, CheZOD117, TriZOD348 splits | small | yes |
| `data/benchmark_suite/per_protein/` | SCOPe-5K, CATH20, DeepLoc splits | small | yes |
| `data/benchmark_suite/splits/` | Train/test partition JSONs (e.g. `cb513_80_20.json`) | tiny | yes |
| `data/benchmarks/rigorous_v1/` | Source-of-truth result JSONs (Exp 43/44/46/47) with BCa CIs | small | yes |
| `data/codebooks/` | PQ codebooks (one per PLM × config) | ~1 MB each | gitignored |
| `data/external/cath20/` | CATH-labeled FASTA (for Exp 50 splits) | small | gitignored |
| `results/` | Per-experiment outputs (model weights, run logs) | varies | gitignored |

**If `data/residue_embeddings/` is empty**, regenerate with:

```bash
uv run python experiments/01_extract_residue_embeddings.py --plm prot_t5_full --dataset cb513
# Repeat per PLM × dataset; ~1–2h per combination on M3 Max
```

---

## 3. Encode (2 min)

```python
import h5py, numpy as np
from src.one_embedding.codec_v2 import OneEmbeddingCodec

# Load some raw embeddings (any (L, D) ndarrays — here, the first 100 ProtT5 proteins)
with h5py.File("data/residue_embeddings/prot_t5_xl_cb513.h5") as f:
    raw = {pid: f[pid][:] for pid in list(f.keys())[:100]}

codec = OneEmbeddingCodec()  # default: 896d binary, ~37x compression, no codebook
codec.fit(raw)               # centering stats only (no codebook for binary)

# Single protein
enc = codec.encode(raw["1A0AA"])      # raw shape (L, 1024) → dict of compressed arrays
codec.save(enc, "/tmp/1A0AA.one.h5")  # ~17 KB at L=156

# Batch via H5
codec.encode_h5_to_h5("data/residue_embeddings/prot_t5_xl_cb513.h5", "/tmp/cb513.one.h5")
```

---

## 4. Decode (2 min) — `numpy + h5py` only, no codebook

This is the load-bearing "universal codec" claim — the receiver does NOT need
the `OneEmbeddingCodec` class. Default binary mode requires only the bit
layout below:

```python
import h5py, numpy as np

with h5py.File("/tmp/1A0AA.one.h5", "r") as f:
    bits   = f["per_residue_bits"][:]      # uint8, shape (L, ceil(D/8))
    means  = f.attrs["means"][:]            # (D,) centering vec
    scales = f.attrs["scales"][:]           # (D,) per-dim scale
    L      = int(f.attrs["seq_len"])
    D      = int(f.attrs["d_out"])

# Bit layout: column-major within each byte (bit 7 → col 0, bit 6 → col 1, ...)
unpacked = np.unpackbits(bits, axis=1, bitorder="big")[:, :D]
signs    = unpacked.astype(np.float32) * 2 - 1     # {0,1} → {-1,+1}
per_res  = signs * scales + means                   # (L, D) reconstructed

# Round-trip sanity check vs the raw above
print(per_res.shape)              # (L, 896)
print(per_res.dtype)              # float32
print(np.allclose(per_res, codec.decode(enc), atol=1e-3))   # True
```

For other quantization modes:
- **int4** (`d_out=896`, ~9×): unpack `per_residue_int4` (uint8, two values per byte) the same way; `signs * scales + means` works unchanged.
- **fp16**: read `per_residue_fp16` (float16); add `means` (no signs/scales).
- **PQ**: requires the codebook (`np.load("data/codebooks/prot_t5_full_pq224.npy")`). See `src/one_embedding/quantization.py:dequantize_pq`.

---

## 5. Reproduce one benchmark number (4 min)

Goal: re-derive the **ProtT5-XL SS3 retention 99.0 ± 0.5 %** cell from
`AUDIT_FINDINGS.md` / Exp 47. The split, codec, probe, and bootstrap are all
deterministic given the seeds.

```bash
# Run the Exp 47 sweep on ProtT5-XL (single-PLM single-config slice)
uv run python experiments/47_codec_sweep.py --plm prot_t5_full --config pq224-896 --task ss3
```

The script will:
1. Load CB513 train/test split (seeded 42, see `cb513_80_20.json`)
2. Fit `OneEmbeddingCodec(d_out=896, quantization='pq', pq_m=224, abtt_k=0)` on train
3. Encode train + test
4. Train a CV-tuned linear probe (`GridSearchCV` on C, 3-fold) on the compressed train
5. Evaluate on test, compute BCa B=10000 retention CI vs the raw-embedding baseline
6. Print SS3 retention ≈ 99.0 % with [98.something, 99.something] CI

If the script doesn't exist with `--config` / `--task` flags, run the full
`experiments/47_codec_sweep.py` script and read the corresponding cell from
the output JSON; the protocol is the same.

---

## 6. Common gotchas

These will save you 30 minutes each:

- **MPS is float32 only.** PyTorch on Apple Silicon does not support float64.
  Cast everything to `float32` explicitly (`x.float()` or `x.astype(np.float32)`).
- **`torch.linalg.svdvals` is not implemented on MPS.** Move tensors to CPU
  first (`x.cpu()`) before calling SVD-related ops. Watch for this in any
  preprocessing code.
- **`clip_grad_norm_` + inf gradients = NaN.** The fix is in our codebase
  already, but if you write new training code: `torch.nan_to_num_(grad)`
  before clipping.
- **`scikit-learn ≥ 1.8` removed `multi_class` from `LogisticRegression`.**
  All our probes use the default `'auto'` now. If you adapt the probe code
  for legacy sklearn, reintroduce `multi_class='multinomial'`.
- **`InfoNCEFamilyLoss` with `pos_mask`:** must index directly. `0 * -inf = NaN`
  if you compute it the obvious way (this is a fixed bug, mentioned in case
  it shows up elsewhere).
- **PLM embedding extraction is slow.** ProtT5-XL takes ~6 min/1000 proteins on
  M3 Max. Plan accordingly. ESM2-650M is ~3 min/1000.
- **`uv run pytest` runs ~90 s.** If it takes much longer, check `pmset -g
  therm` and `os.getloadavg()` — thermal throttling will mask real failures.
- **Never run more than 1 GPU-intensive job at a time** on the M3 Max. The
  unified-memory architecture means simultaneous training jobs OOM unpredictably.

---

## 7. Where to look next

| You want… | Read… |
|-----------|-------|
| The audit-grounded write-up | `docs/STATE_OF_THE_PROJECT.md` |
| The paper outline | `docs/MANUSCRIPT_SKELETON.md` |
| Audit findings (per-row triage) | `docs/AUDIT_FINDINGS.md` |
| Anticipated lab-talk questions | `docs/EXPECTED_QA.md` |
| Per-track audit evidence | `docs/_audit/{hygiene,splits,stats,params,claims,tooling,deps}.md` |
| Subagent reports from the audit | `docs/_audit/logs/01_*.md` … `18_*.md` |
| Full 200+ method enumeration | `docs/EXPERIMENTS.md` |
| Exp 50 design + plan | `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md` and the matching plan |
| Phylogenetics design | `docs/superpowers/specs/2026-03-17-embedding-phylogenetics-design.md` |
| Codec API surface | `src/one_embedding/codec_v2.py` (read top-to-bottom; ~600 lines) |
| Statistics protocol | `experiments/43_rigorous_benchmark/metrics/statistics.py` |

---

## 8. If something is broken

- Pre-existing pytest baseline: 813 / 813 passing as of audit (2026-04-27).
- Pre-existing `deptry`: clean.
- Pre-existing `git status --short`: empty.

If any of those is no longer true on the branch you're working on, that's
new drift. `git log --since=2026-04-27 -- <file>` to find the cause.

For dependency issues: `uv lock --check && uv sync --all-extras --all-groups`
should restore the canonical state.

---

## Appendix: Prior (written before audit, 2026-04-26)

> Preserved as the `stub(prior): HANDOFF.md` commit content.

A labmate clones the repo and within 15 minutes can:
1. Set up env (`uv sync` or equivalent — confirm during audit).
2. Locate or extract a PLM embedding for a small test set.
3. Encode with `OneEmbeddingCodec()`.
4. Decode and verify round-trip.
5. Reproduce one row of the 5-PLM table (predicted: ProtT5 SS3 retention).

### Predicted sections
- Setup (one block).
- Test data (where it lives in `data/`).
- Five-line encode demo.
- Five-line decode demo (must use only `h5py + numpy` for the binary default — no `OneEmbeddingCodec` import on the receiver side).
- "Reproduce one number" recipe.
- Common gotchas (predicted: MPS float32 only, `torch.linalg.svdvals` not on MPS, `clip_grad_norm_` + inf grads = NaN).
