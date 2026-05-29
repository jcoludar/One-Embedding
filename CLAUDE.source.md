---
masterbook:
  modules:
    - tier-1/*
    - tier-2/python-research
    - tier-2/gpu-thermal
    - tier-2/bioinformatics-pipeline
  substrates:
    - session-paperwork
    - paperwork-enforcement
  permissions_extra:
    - "Bash(pytest *)"
    - "Bash(squeue *)"
    - "Bash(sinfo *)"
    - "Bash(sacct *)"
    - "Bash(scontrol show *)"
    - "Bash(h5dump *)"
    - "Bash(h5ls *)"
    - "Bash(nvidia-smi *)"
    - "Bash(module avail *)"
    - "Bash(module list)"
    - "Bash(module spider *)"
---

# Protein Embedding Codec

## Project Overview
Universal codec for PLM per-residue embeddings. **200+ compression methods benchmarked** across 49 experiments, validated on **5 PLMs** (ProtT5, ESM2, ESM-C, ProstT5, ANKH). The unified codec uses center + RP 896d + binary by default, achieving **~17 KB/protein** (~37x compression) with **95–100% retention** across **5 task families** (SS3, SS8, retrieval, disorder, VEP) on **10 datasets** and 5 PLMs (Exp 46/47/55/56, BCa CIs). Binary skips the PQ codebook fit at encode time (~20× faster than PQ). Configurable knobs: `d_out` (896), `quantization` ('binary' / 'int2' / 'int4' / 'pq' / 'binary_magnitude'), `pq_m` (auto), `abtt_k` (0). Use `quantization='pq', pq_m=224` for maximum quality at 18x.

**All benchmark tables → [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Full experiment journey → [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).**

## Quick Start — One Embedding package (recommended)
```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec

codec = OneEmbeddingCodec()            # default: 896d binary, ~37x, no codebook
codec.fit(training_embeddings)          # dict of {pid: (L, D)} — for centering stats
encoded = codec.encode(raw_embeddings)  # binary: no codebook, ~1500 proteins/s on M3 Max
codec.save(encoded, "protein.one.h5")

data = OneEmbeddingCodec.load("protein.one.h5")
data['per_residue']   # (L, 896) for per-residue tasks
data['protein_vec']   # (3584,) for retrieval / clustering / UMAP

codec.encode_h5_to_h5("raw.h5", "compressed.h5")        # batch
batch = OneEmbeddingCodec.load_batch("compressed.h5")

# Max quality mode (PQ, needs codebook)
codec = OneEmbeddingCodec(quantization='pq', pq_m=224)
codec.fit(training_embeddings)
codec.save_codebook("codebook.h5")

# Max fidelity — no RP, ~100% retention on everything
codec = OneEmbeddingCodec(d_out=1024, quantization=None)
```

Pipeline: center → RP to d_out (896 default, skip if d_out≥D_in) → quantize → DCT K=4 protein vector. Binary is the default (37x, no codebook). ABTT **off** by default — Exp 45 showed it destroys disorder signal (but Exp 56 showed it's neutral for VEP).

### Low-level API (research / custom pipelines)
```bash
uv run python experiments/01_extract_residue_embeddings.py   # extract PLM embeddings (prerequisite)
uv run python experiments/25_plm_benchmark_suite.py          # 14 codecs × 3 PLMs
uv run python experiments/29_exhaustive_fruit_sweep.py       # 30+ techniques sweep
uv run python experiments/34_progressive_codec.py            # V2 codec tiers benchmark
uv run python experiments/36_toolkit_benchmark.py            # Disorder + SS3 retention
uv run python experiments/make_benchmark_barplots.py         # publication figures
```

## Headline numbers (Exp 47 standard tiers, ProtT5)
| Config | Compression | SS3 ret | Dis ret |
|--------|:-----------:|:-------:|:-------:|
| PQ M=224 896d | 18x | 99.0% | 95.4% |
| binary 896d (default) | 37x | 97.6% | 94.9% |

Disorder is the sole weak spot at high compression; retrieval is lossless (100.4%) across all tiers. Everything else (per-task, multi-PLM, VEP, ablations) is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Architecture
- `src/one_embedding/` — **Unified codec + research library**: `OneEmbeddingCodec` (`codec_v2.py`: fp16/int4/PQ/binary, configurable d_out/quantization/pq_m), transforms (DCT, Haar, spectral), universal codecs, preprocessing (ABTT, PCA rotation), quantization (int2/int4/int8/binary/PQ/RVQ), data analysis, I/O (`.one.h5`/`.oemb`)
- `src/compressors/` — ChannelCompressor (trained), AttentionPool, MLP-AE, VQ, baselines
- `src/extraction/` — ESM2 + ProtT5 + ESM-C embedding extraction
- `src/training/` — Unified trainer (reconstruction, contrastive, VICReg losses)
- `src/evaluation/` — Retrieval, per-residue probes (SS3/SS8/disorder/TM/SignalP), biological annotations (GO/EC/Pfam/taxonomy), statistical tests, FAISS search index
- `src/utils/` — Device management (MPS/CPU), H5 I/O
- `experiments/` — scripts 01–56 (gaps at 49, 52–54 = designed-not-run). Exp 43 = rigorous benchmark, 44 = codec sweep, 45 = disorder forensics, 46 = multi-PLM (5 PLMs), 47 = codec config sweep, 55 = VEP retention, 56 = VEP codec mega-sweep
- `tests/` — 878 tests; `.one.h5` format = H5 single/batch protein files (protein_vec + per_residue), legacy `.oemb` also supported

## Data location (paused-project note)
Embeddings (~100G of `.h5`) live in `~/Dropbox/Science/ProteEmbedExplorations_bigfiles/residue_embeddings/`; `data/residue_embeddings` is an absolute symlink there. All big files belong in that Dropbox folder — do not scatter heavy files in the repo.

## Hardware
- MacBook Pro (Mac15,10), Apple M3 Max, 14 cores (10P + 4E), 96 GB RAM. GPU: MPS (shared memory with system RAM).

## Key Conventions (project-specific)
- Python 3.12 (required for fair-esm compatibility); PyTorch with MPS (Apple Silicon) or CUDA
- All tensors float32 (MPS does not support float64); `torch.linalg.svdvals` not supported on MPS — move tensors to CPU first
- Experiments use `sys.path.insert(0, ...)` for imports from project root
- Per-residue embeddings stored as H5 with gzip; each protein's embedding shape `(L, D)`, L=sequence length, D=model dim
- `LogisticRegression` probes use `random_state=42` for reproducibility; scikit-learn 1.8 removed the `multi_class` param
