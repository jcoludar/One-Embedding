# Protein Embedding Codec

## Project Overview
Universal codec for PLM per-residue embeddings. **200+ compression methods benchmarked** across 49 experiments, validated on **5 PLMs** (ProtT5, ESM2, ESM-C, ProstT5, ANKH). The unified codec uses center + RP 896d + binary by default, achieving **~17 KB/protein** (~37x compression) with **95–100% retention** across **5 task families** (SS3, SS8, retrieval, disorder, VEP) on **10 datasets** and 5 PLMs (Exp 46/47/55/56, BCa CIs). Binary skips the PQ codebook fit at encode time (~20× faster than PQ; precise per-PLM timing in Exp 47 logs). Configurable: `d_out` (896), `quantization` ('binary' / 'int2' / 'int4' / 'pq' / 'binary_magnitude'), `pq_m` (auto), `abtt_k` (0). Use `quantization='pq', pq_m=224` for maximum quality at 18x.

## Quick Start

### One Embedding Package (recommended)
```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec

# Fit codec on training data
codec = OneEmbeddingCodec()  # default: 896d binary, ~37x, no codebook
codec.fit(training_embeddings)  # dict of {pid: (L, D) arrays} — for centering stats

# Encode (binary: no codebook needed, ~1500 proteins/s on M3 Max — see Exp 47)
encoded = codec.encode(raw_embeddings)  # (L, 1024) -> compressed dict
codec.save(encoded, "protein.one.h5")

# Decode
data = OneEmbeddingCodec.load("protein.one.h5")
data['per_residue']   # (L, 896) for per-residue tasks
data['protein_vec']   # (3584,) for retrieval / clustering / UMAP

# Batch encode/decode
codec.encode_h5_to_h5("raw.h5", "compressed.h5")
batch = OneEmbeddingCodec.load_batch("compressed.h5")

# Max quality mode (PQ, needs codebook)
codec = OneEmbeddingCodec(quantization='pq', pq_m=224)
codec.fit(training_embeddings)
codec.save_codebook("codebook.h5")
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

Pipeline: center → RP to d_out (896 default, skip if d_out≥D_in) → quantize → DCT K=4 protein vector.
Four knobs: `d_out` (896), `quantization` ('binary'), `pq_m` (auto), `abtt_k` (0). Binary is the default: 37x compression, no codebook needed, ~1500 proteins/s encoding (M3 Max). Use `quantization='pq'` for max quality (18x). ABTT off by default — Exp 45 showed it destroys disorder signal.

#### Exp 44 sweep (legacy 768d codec)

Numbers below are from the **earlier d_out=768 codec sweep** (Exp 44). The current
default is d_out=896 — see the **Exp 47 codec sweep** table further down for the
shipping numbers. Exp 44 retained because its 6-config × 4-task grid is the
densest single-PLM measurement we have; it informs design choices but is not
the cited final.

`Size (L=175)` columns use a fixed reference protein length L=175 (not the
empirical mean — Exp 45 reports SCOPe-5K mean L=156).

| Config | Quantization | Size (L=175 ref) | Compression | SS3 Ret | SS8 Ret | Dis Ret | Ret Ret |
|--------|-------------|:----------:|:-----------:|:-------:|:-------:|:-------:|:-------:|
| lossless (1024d) | fp16 | 366 KB | 2x | 100.0 ± 0.2% | 100.0 ± 0.3% | 99.9 ± 0.1% | 100.4 ± 0.5% |
| fp16 (768d) | fp16 | 275 KB | 2.7x | 99.1 ± 0.5% | 98.7 ± 0.6% | 95.0 ± 2.1% | 100.2 ± 0.6% |
| int4 (768d) | int4 | 67 KB | 10x | 99.2 ± 0.6% | 98.6 ± 0.6% | 94.8 ± 2.2% | 100.2 ± 0.6% |
| PQ M=192 (768d) | PQ | 34 KB | 20x | 98.8 ± 0.5% | 97.6 ± 0.8% | 92.8 ± 2.7% | 100.2 ± 0.6% |
| PQ M=128 (768d) | PQ | 23 KB | 30x | 97.1 ± 0.6% | 95.3 ± 0.8% | 90.6 ± 2.9% | 100.2 ± 0.6% |
| binary (768d) | 1-bit sign | 17 KB | 41x | 95.9 ± 0.7% | 93.6 ± 1.0% | 92.5 ± 2.7% | 100.2 ± 0.6% |

All Exp 44, rigorous (BCa CIs, CV-tuned probes, paired bootstrap retention, pooled disorder ρ). Retrieval **lossless across all configs** (100.2 ± 0.6%). int4 is indistinguishable from fp16 at 10x compression. Binary beats PQ M=128 on disorder (92.5% vs 90.6%) — RaBitQ effect.

```python
# Default codec: binary, ~37x compression, no codebook needed
from src.one_embedding.codec_v2 import OneEmbeddingCodec
codec = OneEmbeddingCodec()  # binary 896d
codec.fit(training_embeddings)  # centering stats only
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.one.h5")

# Decode batch (receiver side — h5py + numpy only, no codebook)
proteins = OneEmbeddingCodec.load_batch("compressed.one.h5")
proteins['P12345']['per_residue']   # (L, 896) for per-residue tasks
proteins['P12345']['protein_vec']   # (3584,) for retrieval / clustering / UMAP

# Max fidelity — no RP, ~100% retention on everything
codec = OneEmbeddingCodec(d_out=1024, quantization=None)

# Max quality — PQ M=224, 18x (needs codebook)
codec = OneEmbeddingCodec(quantization='pq', pq_m=224)
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
| Disorder (pooled ρ) | per-residue | CheZOD117 (117) | 0.663 [0.585, 0.723] | 0.629 [0.548, 0.691] | **94.9 ± 2.0%** |
| Disorder (pooled ρ) | per-residue | TriZOD348 (348) | 0.506 [0.461, 0.566] | 0.471 [0.426, 0.533] | **93.0 ± 2.6%** |
| Disorder (AUC-ROC) | per-residue | CheZOD117 (117) | 0.890 [0.836, 0.922] | 0.877 [0.826, 0.909] | **98.5%** |

#### Variant Effect Prediction (Exp 55, supervised Ridge probe + zero-shot ClinVar AUC)

| Task | Level | Dataset (n_assays / n_variants) | Raw ProtT5 1024d | One Embedding 896d (binary, 37×) | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| DMS Spearman ρ (mean) | per-protein | ProteinGym diversity (15 / 37,919) | 0.645 | 0.640 | **99.2 ± 0.8%** |
| ClinVar AUC (zero-shot) | per-variant | ProteinGym clinical ≤500 aa (1,016 / 15,252) | 0.602 | **0.605** | **100.5%** |

3-seed averaged Ridge probe (5-fold outer CV, inner 3-fold GridSearch on α). BCa B=10,000 paired ratio-of-means bootstrap. Binary 896d retention is statistically indistinguishable from PQ M=224 (CIs overlap heavily) — binary is the recommended VEP tier. Per-residue mutational sensitivity survives 1-bit-per-dim quantization, in contrast to disorder (94.9% binary retention) which has a real ~5pp gap. See `docs/exp55_vep_retention.md` for the per-assay breakdown and methodological discussion.

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

#### Legacy benchmarks (Exp 37, 512d codec — pre-rigorous, no BCa CIs)
| Metric | Retention | Source | Note |
|---|:---:|---|---|
| Structural lDDT | 100.7% | Exp 37 | pre-rigorous; not re-validated through `metrics.statistics` |
| Contact precision | 106.5% | Exp 37 | same |
| TM-score Spearman | **57.4%** | Exp 37 (`structural_retention_results.json`) | same; **lower than lDDT/contact**, disclosed for completeness |

## Experiment History & Methodology

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full journey (200+ methods rolled up across 49 experiments — exact enumeration in `EXPERIMENTS.md`).

**Benchmark methodology (Nature-level, Exp 43):**
- Bootstrap: BCa (DiCiccio & Efron 1996), B=10,000, percentile fallback for n<25
- Multi-seed: predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021)
- Disorder: pooled residue-level Spearman ρ (SETH/CAID standard) with cluster bootstrap (Davison & Hinkley 1997)
- Retrieval: 3 fair baselines (raw+mean, raw+DCT, raw+ABTT+DCT). Retention = compressed / baseline C
- Probes: CV-tuned (GridSearchCV on train set, 3-fold, C/alpha grids)
- ABTT leakage: formally tested — PCs differ across corpora but downstream Ret@1 varies <0.2pp (irrelevant)

878 tests, 5 task families (SS3 / SS8 / retrieval / disorder / VEP), 10 datasets (CB513, TS115, CASP12, CheZOD117, TriZOD348, SCOPe 5K, CATH20, DeepLoc test, DeepLoc setHARD, ProteinGym DMS+ClinVar), 5 PLMs. BCa CIs on everything.

**Multi-PLM validation (Exp 46, center + RP896 + PQ224, ~18x):**

| PLM | dim | SS3 ret | SS8 ret | Ret@1 ret | Dis ret |
|-----|:---:|:-------:|:-------:|:---------:|:-------:|
| ProstT5 | 1024 | 99.2±0.3% | 98.6±0.5% | 100.0±0.5% | 98.3±1.1% |
| ProtT5-XL | 1024 | 99.0±0.5% | 98.5±0.6% | 100.6±0.6% | 95.4±1.9% |
| ESM-C 600M | 1152 | 98.3±0.5% | 97.6±0.7% | 102.6±2.9% | 98.1±1.0% |
| ANKH-large | 1536 | 97.9±0.5% | 96.3±0.8% | 99.9±0.6% | 94.8±2.3% |
| ESM2-650M | 1280 | 97.6±0.7% | 96.5±0.7% | 97.8±1.6% | 98.8±0.9% |

**VEP codec mega-sweep (Exp 56, ProtT5, paired BCa B=10,000):**

| Config | Compression | DMS retention | ClinVar AUC | Note |
|--------|:-----------:|:-------------:|:-----------:|------|
| Lossless 1024d | 2× | 100.0% | 0.602 | baseline |
| **Binary 1024 (no RP)** | **32×** | **100.7% [99.5, 103.2]** | 0.598 | best DMS retention in sweep |
| **int2 896d** | **18×** | **99.9% [99.2, 100.7]** | 0.530 | clean DMS, ClinVar collapses (probe-only tier) |
| binary 896 + ABTT8 | 37× | 99.2% [98.2, 100.2] | 0.598 | ABTT essentially neutral on VEP |
| binary 896 + ABTT3 | 37× | 99.0% [98.1, 99.7] | 0.588 | falsifies "ABTT-3 destroys signal" outside disorder |
| binary_magnitude 896 | ~30× | 99.6% [98.9, 100.5] | 0.605 | Exp 51 PolarQuant rehabilitated for VEP |
| pq128 896d | 32× | 98.9% [98.4, 99.4] | 0.588 | |
| **binary 512d** | **64×** | 98.1% [96.8, 100.0] | **0.609** | wins ClinVar AUC (RP isotropy effect) |
| pq64 896d | 64× | 97.6% [96.7, 98.6] | 0.581 | most aggressive DMS-OK setting |

Headlines: ABTT-k is essentially free for VEP at all tested k (≠ disorder); RP — not quantization — is binary's main loss source on DMS; int2 is a real DMS tier but supervised-only; PQ M=64 retains 97.6% at 64× compression. See `docs/exp56_vep_codec_megasweep.md` for the full per-axis breakdown and the per-arm ClinVar table.

**Codec sweep (Exp 47, ProtT5, standard tiers):**

| Config | Compression | SS3 ret | SS8 ret | Ret@1 ret | Dis ret |
|--------|:-----------:|:-------:|:-------:|:---------:|:-------:|
| lossless 1024d | 2x | 100.2% | 100.0% | 100.4% | 100.0% |
| fp16 896d | 2.3x | 100.0% | 99.2% | 100.6% | 98.6% |
| int4 896d | 9x | 99.8% | 98.8% | 100.4% | 98.2% |
| **PQ M=224 896d** | **18x** | **99.0%** | **98.5%** | **100.6%** | **95.4%** |
| PQ M=128 896d | 32x | 97.5% | 96.1% | 100.1% | 91.4% |
| binary 896d | 37x | 97.6% | 95.0% | 100.4% | 94.9% |

VQ/RVQ confirmed genuinely poor (not bug-caused): VQ K=16384 gets 79% SS3 ret, 58% Dis ret.

## Architecture
- `src/one_embedding/` — **Unified codec + research library**: OneEmbeddingCodec (codec_v2.py: fp16/int4/PQ/binary, configurable d_out/quantization/pq_m), transforms (DCT, Haar, spectral), universal codecs, preprocessing (ABTT, PCA rotation), quantization (int2/int4/int8/binary/PQ/RVQ), path transforms, enriched transforms, data analysis, I/O (.one.h5/.oemb format)
- `src/compressors/` — ChannelCompressor (trained), AttentionPool, MLP-AE, VQ, baselines
- `src/extraction/` — ESM2 + ProtT5 + ESM-C embedding extraction
- `src/training/` — Unified trainer with reconstruction, contrastive, VICReg losses
- `src/evaluation/` — Retrieval (cosine+euclidean), per-residue probes (SS3/SS8/disorder/TM/SignalP), biological annotations (GO/EC/Pfam/taxonomy), hierarchy, statistical tests, FAISS search index
- `src/utils/` — Device management (MPS/CPU), H5 I/O
- `experiments/` — experiment scripts numbered 01–56 (gaps at 49, 52–54 are designed-but-not-run earmarks) + figure generators. Exp 43 = rigorous benchmark, Exp 44 = unified codec sweep, Exp 45 = disorder forensics, Exp 46 = multi-PLM pipeline (5 PLMs), Exp 47 = codec config sweep, Exp 55 = VEP retention, Exp 56 = VEP codec mega-sweep
- `tests/` — 878 tests across multiple modules (50 in `tests/test_vep.py` for Exp 55)
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
