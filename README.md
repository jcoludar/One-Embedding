# One Embedding: Universal Compression for PLM Protein Embeddings

A universal codec that compresses any protein language model's per-residue output into a compact, fixed-schema representation -- the **One Embedding**. The 1.0 codec projects to 768d (configurable) and stores as float16 in `.one.h5` format, achieving **275 KB/protein** (2.5x compression) with **97-100% retention** across 12+ tasks (Exp 43, BCa bootstrap CIs). Optional extreme compression tiers (PQ/int4/binary on 512d) go down to 10--52 KB. Works with any PLM (ProtT5, ESM2, ESM-C). Receiver needs only `h5py` and `numpy`.

## TL;DR

Protein language models produce large variable-length per-residue embedding matrices `(L, D)`. The 1.0 codec compresses each using: All-but-the-Top (remove top-3 corpus PCs) -> random projection to 768d (configurable) -> store as float16 in `.one.h5` format. A protein-level vector `(D*dct_k,)` is computed via DCT K=4 for retrieval/clustering. The 768d default preserves 97-100% task retention across 12+ tasks on 8+ datasets (Exp 43, rigorous BCa bootstrap CIs). Optional extreme compression tiers (on 512d) use PQ/int4/binary quantization for 10--52 KB payloads.

- **Retrieval/clustering**: `protein_vec` -> (3072,) vector at 768d. Cosine similarity.
- **Per-residue (SS3, disorder)**: `per_residue` -> (L, 768) float16 embeddings.

232 compression methods benchmarked across 42 experiments to arrive at this design.

## Quick Start

### Python API

```python
from src.one_embedding.codec import OneEmbeddingCodec

# 1.0 codec: 768d float16 (default, ~275 KB/protein, 97-100% retention)
codec = OneEmbeddingCodec(d_out=768, dct_k=4)
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.one.h5")

# Decode (receiver side -- h5py + numpy only)
data = OneEmbeddingCodec.load("compressed.one.h5")
data['per_residue']   # (L, 768) float16 for per-residue tasks
data['protein_vec']   # (3072,) float16 for retrieval / clustering / UMAP

# With pre-fitted ABTT for ProtT5 or ESM2
codec = OneEmbeddingCodec.for_plm('prot_t5', d_out=768)
codec.encode_h5_to_h5("raw_prot_t5.h5", "compressed.one.h5")
```

### CLI

```bash
# Extract PLM embeddings from FASTA
one-embedding extract sequences.fasta embeddings.h5 --model prot_t5

# Compress to .one.h5 format (768d default)
one-embedding encode embeddings.h5 compressed.one.h5
one-embedding encode embeddings.h5 compressed.one.h5 --d-out 512  # or 512d

# Inspect contents
one-embedding inspect compressed.one.h5

# Built-in tools
one-embedding disorder compressed.one.h5
one-embedding search query.one.h5 database/ --top-k 10
one-embedding align protein_a.one.h5 protein_b.one.h5
```

### Running Experiments

```bash
# Setup (requires Python 3.12, uv package manager)
uv sync

# Extract embeddings (prerequisite for all experiments)
uv run python experiments/01_extract_residue_embeddings.py

# V2 codec benchmarks
uv run python experiments/32_pq_on_rp512.py              # PQ sweep on preprocessed space
uv run python experiments/34_progressive_codec.py          # V2 tiers benchmark

# Retention benchmarks
uv run python experiments/36_toolkit_benchmark.py          # Disorder + SS3 retention
uv run python experiments/37_structural_retention.py       # lDDT + contact precision

# Generate figures
uv run python experiments/make_benchmark_barplots.py       # Per-benchmark + V2 + Pareto
uv run python experiments/make_publication_figures.py       # Publication figures
```

## Compression Tiers (Optional Extreme Compression)

![V2 Pareto](docs/figures/pub_v2_pareto.png)

For storage-constrained applications, the V2 codec adds quantization on top of the 512d base: **ABTT k=3** -> **RP to 512d** -> **quantize** (PQ, int4, or binary) -> **DCT K=4** for protein-level vector. These tiers trade per-residue fidelity for smaller size, down to 10 KB/protein.

All tiers share the same preprocessing and protein vector. The only difference is per-residue quantization:

| Mode | Quantization | Size | Ret@1 | SS3 Q3 | SS8 Q8 | Disorder ρ | SS3 Ret | Dis Ret |
|------|-------------|:----:|:-----:|:------:|:------:|:----------:|:-------:|:-------:|
| **`full`** | **int4 scalar** | **48 KB** | **0.795** | **0.812** | **0.682** | **0.597** | **96.6%** | **90.0%** |
| **`balanced`** | **PQ M=128** | **26 KB** | **0.795** | **0.804** | **0.669** | **0.583** | **95.7%** | **88.0%** |
| `binary` | 1-bit sign | 15 KB | 0.795 | 0.771 | 0.638 | 0.596 | 91.7% | 90.0% |
| `compact` | PQ M=64 | 15 KB | 0.795 | 0.772 | 0.636 | 0.548 | 91.8% | 82.7% |
| `micro` | PQ M=32 | 10 KB | 0.795 | 0.731 | 0.591 | 0.495 | 87.0% | 74.6% |

All numbers rigorously benchmarked (Exp 43: BCa bootstrap CIs, CV-tuned probes, pooled disorder ρ). Retrieval is **lossless across all modes** (100.2%). Binary matches full for disorder (90.0%) — RaBitQ effect. Payload size: PQ modes store `L x M + 4096` bytes. Shared codebook: ~512 KB per mode.

### Per-Residue Quality Across Tiers

| | |
|:---:|:---:|
| ![V2 SS3](docs/figures/pub_v2_ss3.png) | ![V2 Disorder](docs/figures/pub_v2_disorder.png) |
| ![V2 TM](docs/figures/pub_v2_tm.png) | |

### When to Use Which Tier

| Use Case | Tier | Why |
|----------|------|-----|
| **General purpose** | `balanced` | Best quality/size trade-off (26 KB, 95.7% SS3) |
| **Maximum per-residue fidelity** | `full` | Highest SS3/disorder retention (48 KB, 96.6% SS3) |
| **Storage-constrained** | `compact` | Good quality at 15 KB (91.8% SS3) |
| **Retrieval-only** | `binary` | Lossless retrieval + 90% disorder at 15 KB |
| **Extreme compression** | `micro` | 10 KB, still 87.0% SS3 retention |

## Retention Benchmarks

How much task performance does the 1.0 codec (768d float16) preserve compared to raw ProtT5-XL 1024d embeddings?

### 768d Retention (Experiment 43 — rigorous, BCa CIs)

All numbers include 95% BCa bootstrap CIs. Probes CV-tuned. Predictions averaged across 3 seeds. Disorder uses pooled residue-level Spearman rho (SETH/CAID standard).

| Task | Dataset (n) | Raw ProtT5 1024d | Compressed 768d | Retention |
|------|-------------|:----------------:|:---------------:|:---------:|
| SS3 Q3 | CB513 (103) | 0.840 [0.823, 0.852] | 0.833 [0.818, 0.845] | **99.1%** |
| SS3 Q3 | TS115 (115) | 0.841 [0.829, 0.853] | 0.828 [0.816, 0.839] | **98.4%** |
| SS8 Q8 | CB513 (103) | 0.716 [0.697, 0.734] | 0.707 [0.689, 0.725] | **98.8%** |
| Disorder (pooled rho) | CheZOD117 (117) | 0.663 [0.636, 0.688] | 0.629 [0.601, 0.656] | **94.9%** |
| Family Ret@1 | SCOPe 5K (2493) | 0.799 [0.783, 0.815] | 0.798 [0.782, 0.814] | **99.8%** |
| Superfamily Ret@1 | CATH20 (9518) | 0.841 [0.834, 0.849] | 0.841 [0.834, 0.849] | **100.0%** |
| Localization Q10 | DeepLoc (2768) | 0.810 [0.795, 0.824] | 0.806 [0.791, 0.820] | **99.5%** |

### Structural Retention (Experiment 37, 512d)

| Metric | Retention | Dataset |
|--------|:---------:|---------|
| Local distance difference (lDDT) | **100.7%** | 50 SCOPe domains |
| Contact precision | **106.5%** | 50 SCOPe domains |

CIs on raw and compressed **overlap** for all tasks — no statistically significant difference detected. Cross-dataset consistency verified on 3 independent SS3/SS8 test sets (max 1.2pp divergence). ESM2 multi-PLM validation: 95.8% SS3, 100.0% retrieval.

## Embedding Phylogenetics (Experiment 35)

![Phylo Monophyly](docs/figures/pub_phylo_monophyly.png)

PLM embeddings encode enough evolutionary signal to reconstruct phylogenetic trees -- without sequence alignment. We implemented a full Bayesian MCMC framework with Brownian motion likelihood for 512-dimensional continuous character data (a capability no existing phylogenetic software supports).

| Tree Method | Data | Monophyletic Families |
|-------------|------|:---------------------:|
| FastTree (ML) | AA sequence | 4 / 12 |
| IQ-TREE WAG+I+G4 (ML) | AA sequence | 5 / 12 |
| Embedding NJ | per-protein 512d | 9 / 12 |
| Embedding BM MCMC (200K gen) | per-protein 512d | 10 / 12 |
| **BM MCMC warm-start from NJ** | **per-residue 320Kd** | **11 / 12** |
| BM MCMC (50K gen) | per-residue 320Kd | 10 / 12 |

ToxFam v2 benchmark: 84 proteins sampled from 12 diverse venom protein families (Snaclec, CRISP, Disintegrin, Actinoporin, Insulin, etc. — 7 proteins per family). Embedding trees recover 2x more monophyletic families than sequence-based maximum likelihood methods. The best result (11/12) comes from a per-residue BM MCMC warm-started from a neighbor-joining tree. The Brownian motion model treats each of the 512 compressed dimensions as an independent continuous trait evolving along the tree -- justified by the decorrelation from ABTT3 + random projection preprocessing.

**Implementation:** ExaBayes-style MCMC with vectorized Felsenstein pruning O(N*D), partial likelihood caching, extended SPR proposals, MC3 heated chains, and convergence diagnostics (ASDSF, ESS, PSRF). Cross-validated against RevBayes (sigma-squared CIs overlap). 71 tests.

## The Pipeline

```
Raw PLM output (L, 1024)            -- any PLM, any protein
  -> All-but-the-Top k=3            -- remove 3 corpus PCs (isotropy transform)
  -> Random project to 768d         -- fixed seed=42, norm-preserving (JL lemma)
  -> Store (L, 768) float16         -- per-residue: 1536 bytes/residue
  + DCT K=4 on projected embeddings -- protein vector: (3072,) fp16
  = .one.h5 file                    -- self-contained, h5py + numpy to decode
```

**Why each step matters:**

- **ABTT k=3**: Removes the dominant protein-identity PCs that dominate cosine similarity. Exposes discriminative family-level directions. +0.006 Ret@1 for free. From Mu & Viswanath (2018), validated for PLM protein embeddings.
- **RP 768d**: Johnson-Lindenstrauss dimensionality reduction. Preserves pairwise distances with high probability. Deterministic (fixed seed). ProtT5 has intrinsic dimensionality ~374, so 768d captures ~95% of variance with 97-100% task retention (Exp 43, BCa CIs). Configurable: use 512d for more compression.
- **DCT K=4**: Discrete Cosine Transform on the sequence dimension, keeping the first 4 coefficients per channel. Creates a fixed-size protein-level vector from variable-length per-residue embeddings. DCT K=1 === mean pooling (mathematically).

For extreme compression, PQ quantization can be applied on top of 512d projections (see Compression Tiers above).

## Storage Comparison

![Storage Comparison](docs/figures/pub_storage_comparison.png)

| Representation | Size/protein | Compression | Retrieval | Per-Residue |
|----------------|:------------:|:-----------:|:---------:|:-----------:|
| Raw ProtT5 (L, 1024) fp32 | 700 KB | 1x | Baseline | Baseline |
| **1.0 codec (RP768) fp16** | **275 KB** | **2.5x** | **0.798** | **(L, 768)** |
| V2 `balanced` PQ M=128 | 26 KB | 27x | 0.786 | (L, 512) |
| V2 `full` int4 | 52 KB | 14x | 0.786 | (L, 512) |
| V2 `compact` PQ M=64 | 15 KB | 47x | 0.786 | (L, 512) |
| V1 codec (RP512) fp16 | 179 KB | 4x | 0.780 | (L, 512) |
| protein_vec only (3072,) fp16 | 6 KB | 117x | 0.798 | No |
| mean pool only (1024,) fp32 | 4 KB | 175x | 0.734 | No |

Mean L=175 residues. V2 sizes are data payload; on-disk H5 files add ~7 KB.

## Built-in Tools

The package includes 7 tools that work directly on compressed `.one.h5` embeddings:

| Tool | Description | Method |
|------|-------------|--------|
| **disorder** | Intrinsic disorder prediction | Trained CNN probe (SETH-style), rho=0.707 |
| **ss3** | Secondary structure (3-class: H/E/C) | Trained CNN probe, Q3=0.855 |
| **search** | Similarity search / k-NN retrieval | Cosine similarity on protein_vec |
| **classify** | Family classification | k-NN against reference database |
| **align** | Pairwise residue alignment | Per-residue embedding alignment |
| **conserve** | Conservation scoring | Embedding norm heuristic |
| **mutate** | Mutation sensitivity scanning | Local context sensitivity |

CNN probes (disorder, ss3) are trained on compressed embeddings and ship as pre-trained weights (~460 KB each) supporting both 512d and 768d inputs. Conservation and mutation tools use untrained heuristics.

## Key Results: Training-Free Codec

![Codec Retrieval Benchmark](docs/figures/pub_codec_retrieval.png)

| Codec | Ret@1 | SS3 Q3 | Size | Per-Residue? |
|-------|:-----:|:------:|:----:|:------------:|
| **1.0 (RP768 fp16)** | **0.798** | **0.833** | **275 KB** | **Yes (L, 768)** |
| V2 full (int4) | 0.795 | 0.812 | 48 KB | Yes (L, 512) |
| V2 balanced (PQ M=128) | 0.795 | 0.804 | 26 KB | Yes (L, 512) |
| V2 binary (1-bit) | 0.795 | 0.771 | 15 KB | Yes (L, 512) |
| mean pool (ground zero) | 0.734 | 0.840 | 4 KB | No |
| *Trained CC d256* | *0.795* | *0.834* | *— (requires training)* | *Yes (256d)* |

ProtT5-XL on SCOPe 5K (n=2493). All numbers from Exp 43 (BCa CIs, CV-tuned probes).

## The Fundamental Trade-off

![Tradeoff Scatter](docs/figures/pub_tradeoff_scatter.png)

There is a fundamental tension between retrieval and per-residue quality:

- **L-compression** (collapsing the sequence dimension via pooling) boosts retrieval but destroys per-residue information
- **D-compression** (reducing embedding dimension via projection) preserves per-residue structure but barely helps retrieval

**Chained codecs solve this**: D-compress first (RP to 512d for per-residue), then smart-pool (DCT K=4 for retrieval). Both tasks are served from a single stored representation.

## Per-Residue Task Retention

![Per-Residue Retention](docs/figures/pub_per_residue_retention.png)

D-compression codecs (rp512, fh512) retain 93-97% of raw per-residue task performance across secondary structure, disorder, and membrane topology prediction. V2 `balanced` (red) shows the effect of PQ quantization on top: SS3 and SS8 remain close to the RP baseline, while disorder and TM see modest drops. The `full` (int4) tier matches the RP baseline almost exactly.

## Cross-PLM Results

![Cross-PLM](docs/figures/pub_cross_plm.png)

The choice of PLM matters far more than the choice of codec. ProtT5-XL outperforms ESM2-650M by ~0.12 Ret@1 across all codecs, and ESM2-650M outperforms ESM-C 300M by another ~0.23. The relative ranking of codecs is consistent across PLMs.

ABTT k=3 on ESM2 gives a massive +0.072 Ret@1 improvement (0.684 -> 0.755), because ESM2 has very concentrated PCs (intrinsic dimensionality = 41 vs ProtT5 = 374).

## Biology and Hierarchy Validation

![Biology & Hierarchy](docs/figures/pub_biology_hierarchy.png)

Codec performance validated on enzyme classification (EC numbers), Pfam domain retrieval, Gene Ontology semantic similarity, and SCOPe hierarchy separation.

## Evaluation Suite

Every codec is benchmarked against a comprehensive suite spanning retrieval, structure, biology, and per-residue probes. All evaluations use the same SCOPe 5K dataset (family-stratified train/test split, n=850 test queries) unless noted.

### Per-Protein Retrieval

| Metric | What it measures | Dataset |
|--------|-----------------|---------|
| Family Ret@1 | Nearest-neighbor same-family match (cosine) | SCOPe 5K |
| SF Ret@1 | Superfamily-level retrieval | SCOPe 5K |
| Fold Ret@1 | Fold-level retrieval | SCOPe 5K |
| MRR | Mean reciprocal rank | SCOPe 5K |

### Biological Annotation Correlation

| Metric | What it measures | Source |
|--------|-----------------|--------|
| GO Spearman rho | Embedding similarity vs Gene Ontology Jaccard | UniProt GO terms |
| EC Ret@1 (4 levels) | Enzyme classification retrieval | UniProt EC numbers |
| Pfam Ret@1 | Protein domain family retrieval | UniProt Pfam |

### Per-Residue Probes

| Task | Metric | Dataset |
|------|--------|---------|
| Secondary structure (3/8-class) | Q3 / Q8 accuracy | CB513 |
| Intrinsic disorder | Spearman rho | CheZOD |
| Transmembrane topology | Macro F1 | TMbed |

### Structural Validation

| Metric | What it measures | Source |
|--------|-----------------|--------|
| TM-score Spearman rho | Embedding similarity vs structural alignment | PDB via tmtools |
| lDDT | Local distance difference test | PDB structures |
| Contact precision | Top-L/5 contact prediction | PDB structures |

## Statistical Methodology

**Bootstrap CIs**: BCa (bias-corrected and accelerated, DiCiccio & Efron 1996), B=10,000, percentile fallback for n<25. Per-protein resampling (cluster bootstrap) for per-residue metrics.

**Multi-seed**: Predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021). Probes CV-tuned via GridSearchCV on training set.

**Disorder evaluation**: Pooled residue-level Spearman rho with cluster bootstrap CIs, matching SETH/ODiNPred/ADOPT/UdonPred/CAID standard. AUC-ROC on binary Z<8 threshold as secondary metric.

**Retrieval**: 3 fair baselines (raw+mean, raw+DCT K=4, raw+ABTT3+DCT K=4). Retention = compressed / baseline C.

**ABTT fitting**: Cross-corpus stability verified — PCs differ across 4 corpora (subspace similarity 0.18-0.71) but Ret@1 varies by only 0.20pp. Fitting corpus choice is irrelevant for downstream performance.

**Training-free codecs are deterministic** -- no training randomness. RP/FH use fixed seed=42.

## Base Codec (Training-Free, No Codebook)

The base codec requires no codebook fitting -- fully deterministic with zero dependencies beyond the RP seed:

```python
from src.one_embedding.codec import OneEmbeddingCodec

codec = OneEmbeddingCodec(d_out=768, dct_k=4)  # 768d default (1.0)
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.one.h5")

# Receiver needs only h5py + numpy -- no codec library, no codebook
data = OneEmbeddingCodec.load("compressed.one.h5")
data['per_residue']   # (L, 768) float16
data['protein_vec']   # (3072,) float16
```

The 1.0 codec stores `(L, 768)` float16 per-residue embeddings + `(3072,)` protein vector. Size: ~275 KB/protein (2.5x compression). No quantization beyond float16. Pre-fitted ABTT weights available for ProtT5 and ESM2 via `OneEmbeddingCodec.for_plm('prot_t5', d_out=768)`.

Use 512d (`d_out=512`, ~179 KB) for more compression, or add PQ tiers for extreme compression.

## The Journey: 232 Methods in 42 Experiments

### Phase 1-4: Trained Compression (Experiments 1-10)

Explored attention pooling, MLP autoencoders, ChannelCompressor. Attention pool failed at scale (lost to PCA-128 on larger data). ChannelCompressor with contrastive fine-tuning achieved Ret@1=0.795 (d256, 3-seed mean). Requires labels and training -- not universal.

### Phase 5: Universal Codec Quest (Experiments 18-24)

Pivoted to training-free codecs. Tested DCT, Haar wavelets, spectral fingerprints, path signatures, curvature, gyration tensors, Fisher vectors, kernel mean embeddings. Key negative: path geometry adds noise, not signal. Key positive: DCT K=1 === mean pool; [mean|max] concat is a free +4pp retrieval boost.

### Phase 6: The Chained Codec Breakthrough (Experiments 25-26)

Discovered that chaining D-compression (RP512) + L-compression (DCT K=4) solves the fundamental tension. 14 codecs x 3 PLMs benchmarked. Best: rp512+dct_K4 -> Ret@1=0.780, SS3=0.815.

### Phase 7: Preprocessing + Quantization (Experiment 29)

ABTT3 (remove top-3 PCs) discovered as a free retrieval boost (+0.006 Ret@1). int4 quantization verified near-lossless for retrieval. 30+ techniques swept across 9 categories. The V1 One Embedding: ABTT3+RP512+int4+DCT K4 -> Ret@1=0.784, SS3=0.809, ~48 KB.

### Phase 8: Extreme Compression (Experiment 28)

45 methods on raw 1024d: wavelets, CUR, channel pruning, PQ, RVQ, tensor train, NMF, SimHash. All on raw space. Best: PQ M=64 at 0.701. Key insight missed: should have tested on preprocessed space.

### Phase 9: V2 -- The Preprocessed Space Changes Everything (Experiments 31-34)

Re-tested all compression on ABTT3+RP512 (decorrelated, isotropic). Results dramatically better:

- **Binary (1-bit) beats int4 for retrieval** (0.787 vs 0.784) -- RaBitQ effect
- **PQ M=128 matches V1 quality at 50% less storage** (26 vs 52 KB payload)
- **Pure VQ fails in 512d** -- even K=16384 caps at 0.621 Ret@1
- **RVQ fails in 512d** -- residual norms barely decrease between levels
- **OPQ doesn't help** -- RP already decorrelates

### Phase 10: Retention Validation (Experiments 36-37)

Comprehensive toolkit and structural retention benchmarks on V2 balanced. SS3 retention 96.7% (LogReg) / 100.3% (CNN), family Ret@1 99.7%, structural lDDT 100.7%, contact precision 106.5%. Disorder retention 90.9% with Ridge, 99.0% with CNN probes.

### Phase 11: Embedding Phylogenetics (Experiment 35)

Applied PLM embeddings to phylogenetic tree inference via Brownian Motion MCMC. Embedding trees achieve 10-11/12 monophyletic families vs 4-5/12 for sequence-based ML/Bayesian methods. Cross-validated against RevBayes.

## What Works, What Doesn't

### Works

- ABTT preprocessing (removes dominant protein-identity PCs)
- Random projection (JL-based dimensionality reduction, norm-preserving)
- Product Quantization on the preprocessed space (sub-vector codebooks)
- DCT K=4 for protein-level vectors (spectral pooling)
- Binary quantization for retrieval-only use cases
- CNN probes on compressed embeddings (SETH-style architecture)

### Doesn't Work

- Path geometry features (signatures, curvature, gyration) -- add noise
- Fisher vectors, Gram features -- poor for family retrieval
- Delta/DPCM encoding -- residues are i.i.d., deltas have MORE variance
- Whole-vector VQ in 512d -- codebook can't cover the space
- RVQ in 512d -- residuals don't decrease meaningfully
- OPQ/learned rotation after RP -- RP already decorrelates
- Two-head joint training -- hurts retrieval vs sequential approach
- Entropy coding on PQ codes -- 7.81/8.00 bits entropy, already near-optimal

## Addendum: Trained ChannelCompressor

![Trained Addendum](docs/figures/pub_trained_addendum.png)

A pointwise MLP (1024 -> 512 -> 256) trained with unsupervised reconstruction then contrastive InfoNCE fine-tuning achieves Ret@1=0.795 +/- 0.012 (3-seed mean). Architecture: input (1024) -> LayerNorm -> Linear(512) -> GELU -> Residual -> Linear(256). Residual connections are critical (-0.169 without). Cross-dataset transfer validated on TS115, CheZOD, TMbed, ToxFam (F1=0.956, beats raw 1024d).

## Project Structure

```
src/
  one_embedding/           Research library
    core/                  Published codec (V1 Codec class, pre-fitted ABTT weights)
    codec_v2.py            V2 codec with PQ support (5 quality tiers)
    preprocessing.py       ABTT, PCA rotation
    quantization.py        int2/int4/int8/binary/PQ/RVQ
    transforms.py          DCT, Haar, spectral
    universal_transforms.py Random/feature-hashed projection
    extract/               ESM2 + ProtT5 embedding extraction
    tools/                 7 built-in tools (disorder, ss3, search, ...)
    io.py                  .one.h5 / .oemb file format (H5-based, single + batch)
    cli.py                 Click CLI: extract, encode, inspect, disorder, search, align
    __init__.py            Top-level API: encode(), decode(), embed()

  compressors/             ChannelCompressor (trained), AttentionPool, MLP-AE
  extraction/              ESM2 + ProtT5 + ESM-C embedding extraction
  training/                Unified trainer with reconstruction + contrastive losses
  evaluation/              Retrieval, per-residue probes (SS3/disorder/TM),
                           biological annotations (GO/EC/Pfam), FAISS search index

experiments/
  01-04                    Setup, baselines, strategy comparison
  archive/05-10            Scale-up, collapse diagnosis, Track A/B (archived)
  11-17                    ChannelCompressor training + validation
  18-23                    Universal codec candidates + path geometry
  25-26                    Universal codec benchmark + chained codecs
  28-29                    Extreme compression + exhaustive sweep
  31-34                    V2 codec: bitwidth, PQ, VQ, progressive tiers
  35                       Embedding phylogenetics (MCMC + MrBayes)
  36-37                    Toolkit + structural retention benchmarks
  make_benchmark_barplots.py    Per-benchmark + V2 figures
  make_publication_figures.py   Publication figures

data/
  proteins/                FASTA files + metadata
  residue_embeddings/      H5 per-residue embeddings
  codebooks/               Pre-fitted PQ codebooks (per mode)
  benchmarks/              JSON result files from all experiments
  checkpoints/             Trained model weights
```

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/) package manager
- PyTorch >= 2.0 with MPS (Apple Silicon) or CUDA
- ~10 GB disk for embeddings and checkpoints

```bash
uv sync  # Installs all dependencies from pyproject.toml
```

## License

MIT. See [LICENSE](LICENSE).
