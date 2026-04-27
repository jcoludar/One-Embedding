# OneEmbedding — universal codec for PLM per-residue embeddings

A single Python class that compresses any of five major protein language model (PLM) outputs by **~37× at 95–100 % task retention** across 4 task families and 9 datasets, with rigorous BCa confidence intervals throughout. Default mode (binary) needs **no codebook** — the receiver decodes with `numpy + h5py` in ~12 lines.

| | |
|---|---|
| **Compression** | ~37× (binary, default) — also int4 (9×), PQ M=224 (18×), fp16 (2.3×), lossless (2×) |
| **Retention** | 95–100 % across SS3, SS8, retrieval, disorder on 5 PLMs |
| **Storage** | ~17 KB / protein at L=156 mean (binary 896d) |
| **Receiver deps** | `numpy + h5py` only for binary / int4 / fp16; PQ requires codebook |
| **PLMs validated** | ProtT5-XL, ProstT5, ESM-C 600M, ANKH-large, ESM2-650M |
| **Tests** | 813 passing |
| **License** | MIT |

> **Talk-prep notes:** `docs/STATE_OF_THE_PROJECT.md` is the audit-grounded source of truth. `docs/AUDIT_FINDINGS.md` lists every cited number's traceability status. `docs/HANDOFF.md` is the 15-minute end-to-end run-through.

---

## Headline numbers

### Single-PLM sweep (Exp 47, ProtT5-XL, BCa CIs in source JSONs)

| Config | Compression | SS3 ret | SS8 ret | Ret@1 ret | Disorder ret |
|--------|:-----------:|:-------:|:-------:|:---------:|:------------:|
| lossless 1024d | 2× | 100.2 % | 100.0 % | 100.4 % | 100.0 % |
| fp16 896d | 2.3× | 100.0 % | 99.2 % | 100.6 % | 98.6 % |
| int4 896d | 9× | 99.8 % | 98.8 % | 100.4 % | 98.2 % |
| **PQ M=224 896d** *(max quality)* | **18×** | **99.0 %** | **98.5 %** | **100.6 %** | **95.4 %** |
| PQ M=128 896d | 32× | 97.5 % | 96.1 % | 100.1 % | 91.4 % |
| **binary 896d** *(default)* | **37×** | **97.6 %** | **95.0 %** | **100.4 %** | **94.9 %** |

### Multi-PLM validation (Exp 46, PQ M=224 896d, ~18×)

| PLM | dim | SS3 ret | SS8 ret | Ret@1 ret | Disorder ret |
|-----|:---:|:-------:|:-------:|:---------:|:------------:|
| ProstT5 | 1024 | 99.2 ± 0.3 % | 98.6 ± 0.5 % | 100.0 ± 0.5 % | 98.3 ± 1.1 % |
| ProtT5-XL | 1024 | 99.0 ± 0.5 % | 98.5 ± 0.6 % | 100.6 ± 0.6 % | 95.4 ± 1.9 % |
| ESM-C 600M | 1152 | 98.3 ± 0.5 % | 97.6 ± 0.7 % | 102.6 ± 2.9 % | 98.1 ± 1.0 % |
| ANKH-large | 1536 | 97.9 ± 0.5 % | 96.3 ± 0.8 % | 99.9 ± 0.6 % | 94.8 ± 2.3 % |
| ESM2-650M | 1280 | 97.6 ± 0.7 % | 96.5 ± 0.7 % | 97.8 ± 1.6 % | 98.8 ± 0.9 % |

Same train/test partition is applied per PLM (single split file, embeddings re-extracted per model). All cells bit-perfect against `data/benchmarks/rigorous_v1/exp46_multi_plm_results.json`.

---

## Quick start

### Install

```bash
git clone <repo-url> ProteEmbedExplorations
cd ProteEmbedExplorations
uv sync --all-extras --all-groups   # installs deps + dev tooling
uv run pytest tests/                  # 813 tests should pass
```

### Encode

```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec

codec = OneEmbeddingCodec()        # default: 896d binary, ~37× compression, no codebook
codec.fit(training_embeddings)     # dict of {pid: (L, D) np.ndarray} — for centering stats only

encoded = codec.encode(raw_embedding)             # raw shape (L, D) → compressed dict
codec.save(encoded, "protein.one.h5")             # ~17 KB at L=156

# Or batch via H5:
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.one.h5")
```

### Decode (binary default — `numpy + h5py` only, no codebook)

```python
import h5py, numpy as np

with h5py.File("compressed.one.h5", "r") as f:
    bits   = f["per_residue_bits"][:]         # uint8, packed bits
    means  = f.attrs["means"][:]              # (D,) centering vec
    scales = f.attrs["scales"][:]             # (D,) per-dim scale
    L      = int(f.attrs["seq_len"])
    D      = int(f.attrs["d_out"])

# Unpack bits (column-major within byte: bit 7 → col 0)
unpacked = np.unpackbits(bits, axis=1, bitorder="big")[:, :D]
signs    = unpacked.astype(np.float32) * 2 - 1            # {0,1} → {-1,+1}
per_res  = signs * scales + means                          # (L, D) reconstructed
```

That's the whole receiver path for the default binary mode — no `OneEmbeddingCodec` import, no codebook, just NumPy. (PQ mode requires the codebook; int4 and fp16 follow the same shape with different unpacking.)

### Pipeline

```
raw PLM (L, D)
    │
    ├── center  (subtract corpus mean)              ← from fit()
    │
    ├── ABTT    (remove top-k PCs)                  ← off by default (see Exp 45)
    │
    ├── RP       (random projection to d_out=896)   ← skipped if d_out ≥ D
    │
    ├── quantize (binary | int4 | PQ | fp16)
    │
    └── DCT K=4 (auxiliary protein vector for retrieval)
```

### Configuration knobs

| Knob | Default | Range | Notes |
|------|---------|-------|-------|
| `d_out` | 896 | 256–4096 | RP target dimension. Set to PLM `D` to skip RP entirely. |
| `quantization` | `'binary'` | `None`/`'int4'`/`'pq'`/`'binary'` | Per-residue storage. `None` = fp16. |
| `pq_m` | `auto` | divisor of `d_out` | PQ subquantizers; auto = largest factor ≤ `d_out//4`. |
| `abtt_k` | 0 | 0–`d_out` | Top-PC removal. Off by default — Exp 45 showed PC1 destroys disorder signal. |

---

## What's not (yet) solved

These open problems are explicit and surface in the talk:

1. **Disorder retention plateaus at ~95 %** — the geometric Exp 45 finding (PC1 ↔ disorder direction) explains why ABTT-style preprocessing doesn't help and why uniform quantization can't fully recover. **Exp 51 (PolarQuant)** is the designed fix: magnitude-augmented binary, expected +2–3 pp at 36×, no codebook.
2. **Sequence → binary OE prediction is capacity-bound at ~69 %** bit accuracy (Exp 50 Stages 1–3 all converge there). **Stage 4 (transformer)** is the architecture lever; designed but not yet executed.
3. **No co-distilled VESM baseline yet** — VESM (Bromberg lab, 2026) is the strongest plausible competitor; weights are public.
4. **VEP / ProteinGym evaluation missing** — the classical PLM-quality benchmark; earmarked.

See `docs/STATE_OF_THE_PROJECT.md` § "Open problems" and § "Next directions" for the full roadmap.

---

## Methodology (Rost-lab convention)

Every cited number ships its 95 % BCa CI in source JSONs under `data/benchmarks/rigorous_v1/`.

- **Bootstrap.** BCa (DiCiccio & Efron 1996), B=10,000, percentile fallback for n<25, jackknife acceleration for cluster bootstrap.
- **Disorder.** Pooled residue-level Spearman ρ (SETH/CAID convention), cluster bootstrap (resample proteins, recompute pooled ρ) per Davison & Hinkley 1997.
- **Retention.** Paired bootstrap — same protein-id resample drives raw and compressed in every iteration.
- **Probes.** CV-tuned (`GridSearchCV` on C/alpha grids, 3-fold), `random_state=42`.
- **Multi-seed.** Predictions averaged across seeds {42, 123, 456} *before* bootstrapping (Bouthillier et al. 2021).
- **Baselines.** Retrieval uses identical DCT K=4 pooling on raw and compressed.
- **Splits.** All cluster-controlled at the dataset level (see `docs/_audit/splits.md`); runtime asserted via `rules.check_no_leakage`.

The full audit (Phase C of `docs/superpowers/plans/2026-04-26-lab-talk-prep.md`) verified these protocols for every result table — see `docs/AUDIT_FINDINGS.md` (49 G / 34 Y / 9 R; **no headline number invalidated**).

---

## Repo layout

```
src/one_embedding/      Codec + research library
  codec_v2.py           OneEmbeddingCodec (the shipping class)
  quantization.py       binary / int4 / PQ implementations + decoders
  preprocessing.py      centering, ABTT, RP
  transforms.py         DCT, Haar, spectral
  io.py                 .one.h5 / .oemb format read/write
  cli.py                CLI (Click-based)
  ...

src/evaluation/         Benchmarks (retrieval, probes, structural)
src/extraction/         PLM embedding extraction (HuggingFace wrappers)
src/training/           Trainers (ChannelCompressor, MLP-AE, contrastive)
src/utils/              Device management (MPS / CPU / CUDA), H5 I/O

experiments/            47 numbered experiments, each self-contained
  43_rigorous_benchmark/  The Nature-level methodology (BCa, paired, CV-tuned)
  44_unified_codec_benchmark.py
  45_disorder_forensics.ipynb  ABTT-PC1 ↔ disorder geometry
  46_multi_plm_benchmark.py    5-PLM validation
  47_codec_sweep.py            Single-PLM sweep across configs
  50_sequence_to_oe.py         Sequence → binary OE prediction (in-progress)

tests/                  813 tests (pytest)

docs/
  STATE_OF_THE_PROJECT.md   Audit-grounded write-up (source of truth)
  MANUSCRIPT_SKELETON.md    Paper outline (target: Bioinformatics)
  HANDOFF.md                15-minute onboarding for a new collaborator
  EXPECTED_QA.md            Anticipated probes for the lab seminar
  CALIBRATION.md            Prior-vs-posterior summary across all docs
  AUDIT_FINDINGS.md         Phase-C audit roll-up (49 G / 34 Y / 9 R)
  EXPERIMENTS.md            Full 200+ method enumeration across 47 experiments
  _audit/                   Raw audit evidence (hygiene, splits, stats, claims)
  _priors/                  Pre-audit predictions (calibration trail)
  superpowers/              Specs and plans
```

---

## Reproducibility

```bash
uv sync --all-extras --all-groups   # installs torch, transformers, faiss-cpu, tmtools, marp, ruff, etc.
uv run pytest tests/                # 813 / 813 should pass on M3 Max
uv run deptry .                     # "No dependency issues found"
```

To regenerate the headline single-PLM table:

```bash
# Pre-extracted ProtT5 embeddings live in data/residue_embeddings/ (see Exp 01 to regenerate)
uv run python experiments/47_codec_sweep.py --plm prot_t5_full
```

The 5-PLM validation requires extracting all 5 PLMs first (~1–2 days on M3 Max for the full datasets); see `experiments/01_extract_residue_embeddings.py`.

### Hardware notes

- Validated on MacBook Pro M3 Max (96 GB), MPS backend.
- CUDA path tested in fixtures, not benchmarked at scale.
- All tensors `float32` (MPS does not support `float64`); `torch.linalg.svdvals` requires CPU on MPS.

---

## Citation

Manuscript in preparation. Until then:

```bibtex
@misc{oneembedding2026,
  title  = {OneEmbedding: a universal codec for protein-language-model embeddings},
  author = {<TBD>},
  year   = {2026},
  url    = {https://github.com/<TBD>/ProteEmbedExplorations}
}
```

---

## Acknowledgements

This codec stands on top of:
- **ProtT5 / ProstT5** (Heinzinger, Elnaggar et al., Rost lab)
- **ESM2 / ESM-C** (Lin, Hayes et al., Meta FAIR)
- **ANKH** (Elnaggar et al.)
- **NetSurfP-2.0** (Klausen et al.) for SS3/SS8 datasets
- **SETH / CheZOD / TriZOD** (Dass, Haak et al.) for disorder
- **CATH / SCOPe / DeepLoc** for retrieval and localisation
- The **Rost lab** for the rigorous-benchmarking conventions adopted throughout
- **Bromberg lab's RNS / VESM work** (companion direction; integration earmarked)

License: MIT (see `LICENSE`).
