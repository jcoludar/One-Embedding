# Protein Embedding Codec

## Project Overview
Universal codec for PLM per-residue embeddings. **232 compression methods benchmarked** across 44 experiments. The unified codec uses ABTT3 + RP 768d + PQ M=192 by default, achieving **~34 KB/protein** (~20x compression) with **92.8–100% retention** across 12+ tasks on 8+ datasets (Exp 44, rigorous benchmarks with paired bootstrap CIs). Configurable: `d_out` (768), `quantization` ('pq'), `pq_m` (auto). Ranges from lossless (1024d fp16, 2x) to extreme (binary, 41x).

## Quick Start

### One Embedding Package (recommended)
```python
import one_embedding as oe

# Encode raw PLM embeddings to .one.h5 format
oe.encode("raw_embeddings.h5", "compressed.one.h5")

# Decode
data = oe.decode("compressed.one.h5")
data['per_residue']   # (L, D) for per-residue tasks (768d default)
data['protein_vec']   # (D*K,) for retrieval / clustering / UMAP

# Embed sequences directly (ProtT5 or ESM2)
vecs = oe.embed(["MKTAYIAKQRQISFVKSHFSRQ..."], model="prot_t5")
```

```bash
# CLI
one-embedding encode raw.h5 compressed.one.h5
one-embedding inspect compressed.one.h5
one-embedding disorder compressed.one.h5
one-embedding search query.one.h5 database/ --top-k 10

# 7 built-in tools: disorder, classify, search, align, ss3, conserve, mutate
```

### Low-level API (research / custom pipelines)
```bash
# Extract PLM embeddings (prerequisite)
uv run python experiments/01_extract_residue_embeddings.py

# Core codec benchmarks
uv run python experiments/25_plm_benchmark_suite.py          # 14 codecs × 3 PLMs
uv run python experiments/26_chained_codec_benchmark.py       # Chained codecs
uv run python experiments/29_exhaustive_fruit_sweep.py        # 30+ techniques sweep

# V2 extreme compression (Phases A–D)
uv run python experiments/31_bitwidth_sweep.py                # Binary/int2/int4 on ABTT3+RP512
uv run python experiments/32_pq_on_rp512.py                   # Product Quantization sweep
uv run python experiments/33_vq_codec.py                      # Vector Quantization + hybrids
uv run python experiments/34_progressive_codec.py              # V2 codec tiers benchmark

# Toolkit retention benchmarks (Exp 36/37)
uv run python experiments/36_toolkit_benchmark.py             # Disorder + SS3 retention
uv run python experiments/37_structural_retention.py           # lDDT + contact precision

# Generate all publication figures
uv run python experiments/make_benchmark_barplots.py          # Per-benchmark + V2 + Pareto
uv run python experiments/make_publication_figures.py          # Legacy figures

# Trained ChannelCompressor (optional, requires labels)
uv run python experiments/11_channel_compression.py
uv run python experiments/13_robust_validation.py --step R1
```

## One Embedding 1.0 (Recommended)

Pipeline: ABTT k=3 → RP to d_out (768 default, skip if d_out≥D_in) → quantize → DCT K=4 protein vector.
Three knobs: `d_out` (768), `quantization` ('pq'), `pq_m` (auto=192). PQ modes require codebook fitting.

| Config | Quantization | Size (L=175) | Compression | SS3 Ret | SS8 Ret | Dis Ret | Ret Ret |
|--------|-------------|:----------:|:-----------:|:-------:|:-------:|:-------:|:-------:|
| lossless (1024d) | fp16 | 366 KB | 2x | 100.0 ± 0.2% | 100.0 ± 0.3% | 99.9 ± 0.1% | 100.4 ± 0.5% |
| fp16 (768d) | fp16 | 275 KB | 2.7x | 99.1 ± 0.5% | 98.7 ± 0.6% | 95.0 ± 2.1% | 100.2 ± 0.6% |
| int4 (768d) | int4 | 67 KB | 10x | 99.2 ± 0.6% | 98.6 ± 0.6% | 94.8 ± 2.2% | 100.2 ± 0.6% |
| **PQ M=192 (768d, default)** | **PQ** | **34 KB** | **20x** | **98.8 ± 0.5%** | **97.6 ± 0.8%** | **92.8 ± 2.7%** | **100.2 ± 0.6%** |
| PQ M=128 (768d) | PQ | 23 KB | 30x | 97.1 ± 0.6% | 95.3 ± 0.8% | 90.6 ± 2.9% | 100.2 ± 0.6% |
| binary (768d) | 1-bit sign | 17 KB | 41x | 95.9 ± 0.7% | 93.6 ± 1.0% | 92.5 ± 2.7% | 100.2 ± 0.6% |

All Exp 44, rigorous (BCa CIs, CV-tuned probes, paired bootstrap retention, pooled disorder ρ). Retrieval **lossless across all configs** (100.2 ± 0.6%). int4 is indistinguishable from fp16 at 10x compression. Binary beats PQ M=128 on disorder (92.5% vs 90.6%) — RaBitQ effect.

```python
# Unified codec: default ~20x compression, just works
from src.one_embedding.codec_v2 import OneEmbeddingCodec
codec = OneEmbeddingCodec()
codec.fit(training_embeddings)
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.one.h5")

# Decode (receiver side — h5py + numpy + codebook)
data = OneEmbeddingCodec.load("compressed.one.h5", codebook_path="codebook.h5")
data['per_residue']   # (L, 768) for per-residue tasks
data['protein_vec']   # (3072,) for retrieval / clustering / UMAP

# Max fidelity — no RP, ~100% retention on everything
codec = OneEmbeddingCodec(d_out=1024, quantization=None)

# More compression — binary, 41x
codec = OneEmbeddingCodec(quantization='binary')
```

### Rigorous Retention Benchmarks (Exp 43, 768d vs raw ProtT5)

All numbers include 95% BCa bootstrap CIs (DiCiccio & Efron 1996, second-order accurate). Probes are CV-tuned (GridSearchCV on C/alpha, not hardcoded). Predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021). Retrieval uses fair baselines (same DCT K=4 pooling for raw and compressed). ABTT fitted on external SCOPe 5K corpus (cross-corpus stability verified: Ret@1 varies < 0.2pp across 4 fitting corpora).

#### Per-Residue Tasks (linear probe, BCa bootstrap CI, 3-seed averaged)

| Task | Level | Dataset (n) | Raw ProtT5 1024d | One Embedding 768d | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| SS3 (Q3) | per-residue | CB513 (103) | 0.840 [0.823, 0.852] | 0.833 [0.818, 0.845] | **99.1 ± 0.6%** |
| SS3 (Q3) | per-residue | TS115 (115) | 0.841 [0.829, 0.853] | 0.828 [0.816, 0.839] | **98.4 ± 0.5%** |
| SS3 (Q3) | per-residue | CASP12 (20) | 0.781 [0.748, 0.810] | 0.765 [0.730, 0.797] | **98.0 ± 1.2%** |
| SS8 (Q8) | per-residue | CB513 (103) | 0.716 [0.697, 0.734] | 0.707 [0.689, 0.725] | **98.8 ± 0.6%** |
| SS8 (Q8) | per-residue | TS115 (115) | 0.732 [0.715, 0.748] | 0.717 [0.701, 0.733] | **98.0 ± 0.7%** |
| SS8 (Q8) | per-residue | CASP12 (20) | 0.662 [0.629, 0.695] | 0.647 [0.611, 0.682] | **97.6 ± 1.7%** |
| Disorder (pooled ρ) | per-residue | CheZOD117 (117) | 0.663 [0.636, 0.688] | 0.629 [0.601, 0.656] | **94.9 ± 2.0%** |
| Disorder (pooled ρ) | per-residue | TriZOD348 (348) | 0.506 [0.476, 0.536] | 0.471 [0.439, 0.502] | **93.0 ± 2.6%** |
| Disorder (AUC-ROC) | per-residue | CheZOD117 (117) | 0.864 [0.848, 0.878] | 0.848 [0.831, 0.864] | **98.1%** |

Disorder uses **pooled residue-level** Spearman ρ (matching SETH/ODiNPred/ADOPT/UdonPred standard) with cluster bootstrap CIs (resample proteins, recompute pooled statistic — Davison & Hinkley 1997). AUC-ROC computed on binary Z<8 threshold (CAID standard).

CIs on raw and compressed **overlap** for all tasks — no statistically significant difference detected.
Cross-dataset consistency: SS3 max 1.1pp, SS8 max 1.2pp (both OK < 3pp threshold).

#### Protein-Level Tasks (cosine kNN / LogReg, paired bootstrap CI on retention)

| Task | Level | Dataset (n) | Raw ProtT5 1024d | One Embedding 768d | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| Family Ret@1 | per-protein | SCOPe 5K (2493) | 0.799 [0.783, 0.815] | 0.798 [0.782, 0.814] | **99.8 ± 0.4%** |
| Superfamily Ret@1 | per-protein | CATH20 (9518) | 0.841 [0.834, 0.849] | 0.841 [0.834, 0.849] | **100.0 ± 0.2%** |
| Localization (Q10) | per-protein | DeepLoc test (2768) | 0.810 [0.795, 0.824] | 0.806 [0.791, 0.820] | **99.5 ± 0.9%** |
| Localization (Q10) | per-protein | DeepLoc setHARD (490) | 0.608 [0.563, 0.651] | 0.606 [0.563, 0.651] | **99.7 ± 3.1%** |

#### ESM2 Multi-PLM Validation (1280d → 768d, 40% compression)

| Task | Raw ESM2 1280d | One Embedding 768d | Retention |
|------|:--------------:|:------------------:|:---------:|
| SS3 (Q3) | 0.836 [0.817, 0.851] | 0.801 [0.784, 0.816] | **95.8 ± 1.0%** |
| SS8 (Q8) | 0.715 [0.695, 0.734] | 0.684 [0.664, 0.703] | **95.7 ± 1.1%** |
| Ret@1 cosine | 0.675 | 0.675 | **100.0 ± 0.5%** |

#### Ablation: Component Contributions (Exp 43 Phase D)

| Condition | SS3 Q3 | Δ vs raw | Ret@1 cos | Δ vs raw |
|-----------|:------:|:--------:|:---------:|:--------:|
| Raw 1024d | 0.840 | baseline | 0.794 | baseline |
| + ABTT3 only | 0.841 | +0.1pp | 0.799 | +0.6pp |
| + RP768 only | 0.837 | −0.3pp | 0.793 | −0.0pp |
| + ABTT3 + RP768 | 0.833 | −0.7pp | 0.798 | +0.4pp |
| + ABTT3 + RP768 + fp16 | 0.833 | −0.7pp | 0.798 | +0.4pp |

fp16 quantization: **0.0pp** effect (completely lossless).
Length stress test: no degradation (short 99.8%, medium 100.7%, long 101.3%).

#### Legacy benchmarks (Exp 37, 512d codec)
| Structural lDDT | 100.7% | Exp 37 |
| Contact precision | 106.5% | Exp 37 |

## Experiment History & Methodology

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full journey (232 methods, 43 experiments, idea space).

**Benchmark methodology (Nature-level, Exp 43):**
- Bootstrap: BCa (DiCiccio & Efron 1996), B=10,000, percentile fallback for n<25
- Multi-seed: predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021)
- Disorder: pooled residue-level Spearman ρ (SETH/CAID standard) with cluster bootstrap (Davison & Hinkley 1997)
- Retrieval: 3 fair baselines (raw+mean, raw+DCT, raw+ABTT+DCT). Retention = compressed / baseline C
- Probes: CV-tuned (GridSearchCV on train set, 3-fold, C/alpha grids)
- ABTT leakage: formally tested — PCs differ across corpora but downstream Ret@1 varies <0.2pp (irrelevant)

758 tests, 12+ tasks, 8+ datasets, 2 PLMs. BCa CIs on everything.

## Architecture
- `one_embedding/` — **Published package** (one_embedding/core/ codec, one_embedding/extract/ PLM wrappers, one_embedding/tools/ 7 tools, one_embedding/cli.py, one_embedding/io.py .one.h5/.oemb format, one_embedding/__init__.py top-level API)
- `src/one_embedding/` — **Research library**: OneEmbeddingCodec (V1), OneEmbeddingCodecV2 (PQ), transforms (DCT, Haar, spectral), universal codecs, preprocessing (ABTT, PCA rotation), quantization (int2/int4/int8/binary/PQ/RVQ), path transforms, enriched transforms, data analysis
- `src/compressors/` — ChannelCompressor (trained), AttentionPool, MLP-AE, VQ, baselines
- `src/extraction/` — ESM2 + ProtT5 + ESM-C embedding extraction
- `src/training/` — Unified trainer with reconstruction, contrastive, VICReg losses
- `src/evaluation/` — Retrieval (cosine+euclidean), per-residue probes (SS3/SS8/disorder/TM/SignalP), biological annotations (GO/EC/Pfam/taxonomy), hierarchy, statistical tests, FAISS search index
- `src/utils/` — Device management (MPS/CPU), H5 I/O
- `experiments/` — 43 experiment scripts (01–43) + figure generators. Exp 43 = rigorous benchmark framework (14 golden rules, 758 tests)
- `tests/` — 758 tests across multiple modules
- `.one.h5 format` — H5-based single/batch protein embedding files (protein_vec + per_residue). Legacy `.oemb` also supported.

## Hardware
- MacBook Pro (Mac15,10) with Apple M3 Max, 14 cores (10P + 4E), 96 GB RAM
- GPU: MPS (Metal Performance Shaders) — shared memory with system RAM

## Thermal / Resource Management
- **Never run more than 1 GPU-intensive training job at a time**
- Monitor with `pmset -g therm` and `os.getloadavg()` during long runs
- If system load > 10 or thermal warnings appear, pause and wait

## Key Conventions
- Python 3.12 (required for fair-esm compatibility)
- PyTorch with MPS (Apple Silicon) or CUDA
- All tensors float32 (MPS does not support float64)
- MPS: `torch.linalg.svdvals` not supported — move tensors to CPU first
- Experiments use `sys.path.insert(0, ...)` for imports from project root
- Per-residue embeddings stored as H5 with gzip compression
- Each protein's embedding: shape (L, D) where L=sequence length, D=model dim
- LogisticRegression probes all use `random_state=42` for reproducibility
