# Experiment 28: Extreme Compression Benchmark

**Date:** 2026-03-13
**Status:** Approved

## Goal

Benchmark novel compression techniques that push far beyond the current 4X codec (26% of raw). Three objectives simultaneously:

1. **Extreme compression** — approach per-protein-only size (~4 KB) while retaining BOTH per-protein retrieval AND per-residue probes
2. **Close the quality gap** — improve Ret@1 beyond 0.780 (current codec) toward 0.808 (trained ChannelCompressor) without any label-supervised training
3. **Novel mathematical territory** — test techniques from topology, optimal transport, tensor networks, and information theory never applied to PLM embeddings

## Constraints

- **Model-agnostic**: Must work with any PLM output (L, D) for any D
- **Self-contained**: Codec ships everything needed to decode (codebooks, centroids, etc.)
- **Preferably training-free**: Deterministic transforms preferred; self-supervised corpus fitting (k-means, PCA) acceptable but clearly labeled
- **All SVD/linalg operations**: Use numpy on CPU (not torch on MPS) per CLAUDE.md
- **MAX_LEN = 512**: Cap sequence length as in Exp 25-27
- **Sequential execution**: One GPU-heavy task at a time per thermal constraints

## Baselines to Beat

| Baseline | Ret@1 | SS3 Q3 | Size/protein | Compression |
|----------|-------|--------|-------------|-------------|
| Raw ProtT5 per-residue | — | 0.845 | ~700 KB | 1X |
| Raw ProtT5 mean pool | 0.734 | — | 4 KB | 175X |
| Current codec (rp512+dct_K4 fp16) | 0.780 | 0.815 | 183 KB | 4X |
| Trained ChannelCompressor | 0.808 | 0.834 | ~350 KB | 2X |

**Target**: Size approaching 4 KB (the per-protein-only size), while retaining Ret@1 >= 0.734 AND SS3 Q3 close to 0.845.

## Evaluation Suite

Full suite matching Exp 25-27 coverage plus reconstruction quality and speed. Every technique measured on ALL applicable metrics.

### Per-Protein Retrieval (all techniques)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| Ret@1 (cosine) | Nearest-neighbor family match | SCOPe 5K test split |
| SF Ret@1 | Superfamily retrieval | SCOPe 5K test split |
| Fold Ret@1 | Fold-level retrieval | SCOPe 5K test split |
| MRR | Mean reciprocal rank | SCOPe 5K test split |
| MAP | Mean average precision | SCOPe 5K test split |

### Hierarchy (all techniques)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| Separation ratio | unrelated_dist / within_family_dist | SCOPe 5K |
| Ordering correct | within < SF < fold < unrelated | SCOPe 5K |

### Biology (all techniques)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| GO Spearman rho | Functional similarity correlation | UniProt GO terms |
| EC Ret@1 (4 levels) | Enzyme classification retrieval | UniProt EC numbers |
| Pfam Ret@1 | Domain family retrieval | UniProt Pfam |
| Taxonomy Spearman rho | Organism similarity correlation | NCBI taxonomy |

### Per-Residue Probes (techniques that preserve per-residue)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| SS3 Q3 | Secondary structure 3-class | CB513 |
| SS8 Q8 | Secondary structure 8-class | CB513 |
| Disorder rho | Intrinsic disorder prediction | CheZOD/SETH |
| TM macro F1 | Transmembrane topology | TMbed |
| SignalP F1 | Signal peptide detection | SignalP6 |

### Reconstruction Quality (techniques that preserve per-residue)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| CosSim | Mean cosine similarity orig↔reconstructed | SCOPe 5K |
| MSE | Mean squared reconstruction error | SCOPe 5K |

### Compression & Speed (all techniques)

| Metric | What it measures | Source |
|--------|-----------------|--------|
| Size (bytes/protein) | Actual serialized byte count (mean protein L=175) | `len(encoded_buffer)` |
| Size after zstd | Size after entropy coding | zstd level 3 |
| Compression ratio | Raw size / compressed size | Computed |
| Codec overhead | Shared codebook/centroid size | Computed (PQ, AA-residual) |
| Encode speed (ms/protein) | Time to compress one protein | Benchmarked |
| Decode speed (ms/protein) | Time to decompress one protein | Benchmarked |

**Size measurement**: Actual serialized byte count of the compressed representation (the bytes you would transmit). Measured by encoding to buffer and checking `len(buffer)`. Reported both raw and after zstd compression. Shared codebooks (PQ, AA centroids) reported separately as "codec overhead."

## Techniques (18 methods across 5 categories)

### Master Comparison Columns

Every technique is tracked with these attributes:

| Attribute | Values |
|-----------|--------|
| Fitting required? | None / Per-protein / Corpus-fitted |
| Per-residue preserved? | Yes / Approximate / No |
| Self-contained decode? | Yes (h5py only) / No (needs codec code) |

### Category A: Within-Channel Compression (no dimension mixing)

These compress each of the D channels independently, treating each as a 1D signal of length L.

#### A1. Per-Channel Wavelet Thresholding
- **Math**: Apply DWT (Daubechies db4) to each channel's L-length signal. Zero coefficients below threshold τ. Inverse DWT to reconstruct.
- **Params**: wavelet={db4, db8, sym4}, threshold_pct={50, 75, 90, 95} (% of coefficients zeroed)
- **Output**: (L, D) with many zeros → compressible by entropy coding
- **Per-residue**: Yes (reconstructed per-residue matrix)
- **Fitting**: None (deterministic transform)
- **Decode**: Needs pywt (inverse DWT) — NOT self-contained
- **Library**: PyWavelets (pywt)

#### A2. Per-Channel Delta Encoding
- **Math**: Store x[0] per channel + differences Δ[i] = x[i+1] - x[i]. Reuses existing `displacement_encode/decode` from `path_transforms.py`.
- **Params**: order={1, 2} (first or second differences)
- **Output**: (L, D) of deltas (smaller dynamic range → better quantization)
- **Per-residue**: Yes (cumulative sum to reconstruct)
- **Fitting**: None
- **Decode**: Self-contained (cumulative sum)
- **Implementation**: Reuse `src/one_embedding/path_transforms.py` displacement_encode/decode

#### A3. CUR / Interpolative Decomposition
- **Math**: Select k actual columns from (L, D) that best span the column space. (L, D) ≈ C(L,k) @ U(k,k) @ R(k,D). Selected columns ARE original embedding channels — literally zero mixing.
- **Params**: k={16, 32, 64, 128, 256}
- **Output**: C(L,k) + coefficient matrix
- **Per-residue**: Yes (reconstruct via C @ U @ R)
- **Fitting**: Per-protein (SVD of each matrix)
- **Decode**: Needs coefficient matrix + column indices
- **Library**: scipy.linalg.interpolative

#### A4. Channel Pruning by Variance
- **Math**: Compute per-channel variance across corpus. Keep top-k highest-variance channels.
- **Params**: k={64, 128, 256, 512}
- **Output**: (L, k) — strict subset of original channels
- **Per-residue**: Yes (but only k channels)
- **Fitting**: Corpus-fitted (needs variance stats from reference set)
- **Decode**: Self-contained (just column indices)

### Category B: Aggressive Quantization

#### B1. Per-Channel Int8 Quantization
- **Math**: Per channel: scale = (max - min) / 255, zero_point = round(-min/scale). Store uint8.
- **Params**: Applied to full D or after D-compression (rp512)
- **Output**: (L, D) uint8 + (D,) scales + (D,) zero_points
- **Size**: L×D + 2D×4 bytes. For L=175, D=1024: 179 KB + 8 KB = 187 KB (~27%)
- **Per-residue**: Yes (dequantize)
- **Fitting**: Per-protein (min/max)
- **Decode**: Self-contained (simple arithmetic)

#### B2. Per-Channel Int4 Quantization
- **Math**: Same as B1 but 4-bit (16 levels). Pack 2 values per byte.
- **Params**: Applied to full D or after D-compression
- **Output**: (L, D/2) uint8 packed + scales + zero_points
- **Size**: L×D/2 + overhead. For L=175, D=1024: ~90 KB + 8 KB = 98 KB (~14%)
- **Per-residue**: Yes (dequantize + unpack)
- **Fitting**: Per-protein
- **Decode**: Self-contained

#### B3. Binary Quantization (1-bit)
- **Math**: Per channel: store sign(x - mean) as 1 bit + mean + scale as float32.
- **Output**: (L, D/8) bytes packed + (D,) means + (D,) scales
- **Size**: L×D/8 + 2D×4. For L=175, D=1024: ~22 KB + 8 KB = 30 KB (~4%)
- **Per-residue**: Yes (approximate — sign × scale + mean)
- **Fitting**: Per-protein
- **Decode**: Self-contained

#### B4. Product Quantization (PQ)
- **Math**: Split D dims into M sub-vectors of d=D/M each. K-means each sub-space into 256 centroids (fit on corpus embeddings, no labels). Store 1-byte index per sub-vector per residue.
- **Params**: M={8, 16, 32, 64, 128}, n_centroids=256
- **Output**: (L, M) uint8 indices + codebook (M × 256 × d) float16
- **Size per protein**: L×M bytes. For L=175, M=32: 5.6 KB per protein
- **Codec overhead**: Codebook M×256×d×2 bytes (shared). For M=32, d=32: 512 KB
- **Per-residue**: Yes (lookup centroid per sub-vector)
- **Fitting**: Corpus-fitted (k-means, no labels)
- **Decode**: Needs codebook (ships with codec)
- **Library**: Manual k-means implementation (no faiss dependency)

#### B5. Residual Vector Quantization (RVQ)
- **Math**: Level 1: coarse VQ (k-means 256 centroids on full D). Level 2: PQ on residual. Level 3: PQ on residual-of-residual.
- **Params**: n_levels={2, 3, 4}, M_per_level=16
- **Output**: (L, n_levels) uint8 indices per coarse level + PQ indices
- **Size per protein**: L × (1 + M_per_level × (n_levels-1)) bytes. For L=175, 3 levels: ~5.4 KB
- **Codec overhead**: Multi-level codebooks (shared)
- **Per-residue**: Yes (sum of codebook lookups)
- **Fitting**: Corpus-fitted (k-means on embeddings and residuals)
- **Decode**: Needs codebooks

### Category C: Novel Mathematical Representations

#### C1. Tensor Train Decomposition
- **Math**: Reshape (L, D) into a higher-order tensor, decompose via TT-SVD. Each core G_i has shape (r_{i-1}, n_i, r_i) where r is bond dimension and n_i is the mode size. Sequential SVD left-to-right with truncation.
- **Params**: bond_dim={4, 8, 16, 32, 64}
- **Output**: List of TT-cores + metadata
- **Per-residue**: Yes (contract cores at position i)
- **Fitting**: Per-protein (SVD of each matrix — no corpus needed)
- **Decode**: Needs tensor contraction code
- **Library**: Manual implementation via numpy SVD on CPU
- **Novel**: First application to PLM embeddings. Same math as DMRG in quantum physics.

#### C2. Non-Negative Matrix Factorization (NMF)
- **Math**: (L, D) ≈ W(L, k) @ H(k, D), W≥0, H≥0. Parts-based decomposition.
- **Params**: k={8, 16, 32, 64}
- **Output**: W(L,k) per protein + H(k,D) fitted on corpus
- **Per-residue**: Yes (W @ H reconstructs per-residue)
- **Fitting**: Corpus-fitted (H matrix) — per-protein W via NNLS
- **Decode**: Needs H matrix + matmul
- **Library**: sklearn.decomposition.NMF
- **Caveat**: PLM embeddings contain negatives → shift to non-negative (add per-channel min). This is a known limitation — flagged as LONG SHOT. The shift distorts the distribution.

#### C3. Optimal Transport Pooling (Wasserstein)
- **Math**: Compare proteins via Sliced Wasserstein Distance (SWD) instead of cosine on mean pools. Project residue clouds onto random 1D slices, compare sorted projections.
- **Params**: n_projections={50, 100, 500}
- **Output**: No compressed representation — this is a DISTANCE METRIC, not a codec. Evaluated directly on raw or compressed embeddings.
- **Per-residue**: N/A (distance metric only — combine with any per-residue codec)
- **Fitting**: None (random projections are deterministic with seed)
- **Library**: Manual implementation (sort + L1 distance on projections)
- **Novel**: Wasserstein geometry for PLM embedding comparison. Tests whether distributional shape beats mean pool for retrieval.

#### C4. Persistent Homology (TDA)
- **Math**: Build Vietoris-Rips filtration on residue point cloud. Compute persistence diagrams (birth/death of topological features). Vectorize via persistence images.
- **Params**: max_dim={0, 1}, n_bins={20, 50}
- **Output**: Fixed-size persistence image (n_bins × n_bins × (max_dim+1))
- **Per-residue**: No (topological summary — supplementary protein vector)
- **Fitting**: None (deterministic from point cloud)
- **Decode**: N/A (summary descriptor, not a codec)
- **Library**: ripser + persim (optional — skip if not installable)
- **Novel**: TDA on PLM per-residue embedding clouds

### Category D: Structure-Aware Compression

#### D1. Amino Acid Residual Coding
- **Math**: Compute per-amino-acid centroid from corpus (20 centroids from sequence identity). Per residue: store residual = embedding - AA_centroid[aa_type]. Residuals have smaller magnitude → more compressible by subsequent quantization.
- **Params**: quantize_residual={float16, int8, int4}
- **Output**: (L, D) residuals + (20, D) AA centroids (shared across corpus)
- **Size**: Same shape as input, but residuals have smaller dynamic range → better quantization
- **Per-residue**: Yes (centroid[aa] + residual)
- **Fitting**: Corpus-fitted (AA centroids computed from embeddings + sequence)
- **Decode**: Needs AA centroid table + sequence
- **Novel**: Exploiting amino acid identity for embedding compression

#### D2. Locality-Sensitive Hashing (SimHash)
- **Math**: Random hyperplane hashing: h(x) = sign(Wx) where W is random Gaussian. Each hyperplane produces 1 bit. Hamming distance ≈ cosine distance.
- **Params**: n_bits={256, 512, 1024, 2048}
- **Output**: (L, n_bits/8) bytes packed binary codes
- **Size**: L × n_bits/8 bytes. For L=175, 1024 bits: ~22 KB
- **Per-residue**: Approximate (binary code per residue)
- **Fitting**: None (random hyperplanes with seed)
- **Decode**: Self-contained (binary → approximate float via pseudoinverse)
- **Novel**: LSH for PLM per-residue storage

### Category E: Multi-Stage Pipelines (Combinations)

#### E1. Wavelet + Int8 + zstd
- Chain: raw → per-channel DWT → threshold 75% → int8 quantize → zstd compress
- Expected: 20-50X compression

#### E2. CUR + PQ
- Chain: raw → CUR k=64 → PQ M=8 on selected columns
- Expected: 50-100X compression, no channel mixing in CUR stage

#### E3. AA-Residual + PQ
- Chain: raw → subtract AA centroids → PQ on residuals
- Expected: Smaller residuals → better PQ approximation → 50-100X

#### E4. RP512 + Int8 + zstd
- Chain: raw → rp512 → int8 quantize → zstd compress
- Expected: 20-40X (extends current codec with quantization)

#### E5. Delta + Int4 + zstd
- Chain: raw → delta encode → int4 quantize → zstd compress
- Expected: 30-60X (delta reduces dynamic range for better quantization)

#### E6. Best Single-Stage + OT/TDA
- Chain: best per-residue compression from above + optimal transport distance / persistence image for protein vector
- Tests whether novel protein descriptors improve retrieval when paired with aggressive per-residue compression

## Implementation Plan

### File Structure
```
src/one_embedding/
    extreme_compression.py    — Wavelet thresholding, CUR, channel pruning (Category A)
    quantization.py           — Int8/Int4/Binary/PQ/RVQ (Category B)
    tensor_decomposition.py   — Tensor train, NMF (Category C1-C2)
    topological.py            — Persistent homology, optimal transport, SimHash (C3-C4, D2)

tests/
    test_extreme_compression.py   — Shape, determinism, roundtrip tests for Category A
    test_quantization.py          — Quantize/dequantize roundtrip, bit-packing tests
    test_tensor_decomposition.py  — TT-SVD correctness, reconstruction error bounds
    test_topological.py           — Persistence diagram shape, OT distance properties

experiments/
    28_extreme_compression_benchmark.py  — Main benchmark script (--step P1..P3)

data/benchmarks/
    extreme_compression_results.json     — Full results
```

### Dependencies (new)
```
pywt          — PyWavelets (wavelet transforms) — required
zstandard     — zstd compression bindings — required
```

Optional (graceful skip if not installed):
```
pot           — Python Optimal Transport (Wasserstein)
ripser        — Persistent homology (C++ backend)
persim        — Persistence images/landscapes
```

All added to `[project.optional-dependencies] extreme = [...]` in pyproject.toml.

### Execution Order

**Phase 1 (--step P1): Single-stage techniques**
- Steps P1a through P1d (one per category A-D)
- Run each independently on ProtT5 SCOPe 5K
- Measure all 9 metrics
- ~18 techniques × ~3 param settings = ~54 runs
- Estimated time: 4-6 hours (sequential, with monitoring)
- Checkpoint after each technique via `mark_done()`

**Phase 2 (--step P2): Multi-stage pipelines**
- Combine winners from Phase 1
- Test 6 predefined chains + promising ad-hoc combinations
- Same 9 metrics
- Estimated time: 2-3 hours

**Phase 3 (--step P3): Cross-PLM validation & multi-seed**
- Run top-5 techniques on ESM2-650M and ESM-C-300M
- Top-3 overall with 3 seeds (s42, s123, s7) for variance
- Estimated time: 3-4 hours

### Speed Benchmarking Protocol
- Warm-up: 10 proteins discarded
- Timed: 100 proteins, report mean +/- std ms/protein
- Separate encode and decode timing
- CPU only (numpy) for reproducibility of timing
- System monitoring via `monitor()` after each technique

### Output
1. **Master comparison table** (all 18+ techniques × 9 metrics + fitting/decode columns)
2. **Pareto frontier plot** (Ret@1 vs log(size), SS3 vs log(size)) — log-scale x-axis
3. **Speed comparison** (encode/decode ms/protein bar chart)
4. **Best pipeline recommendation** at each compression target (10X, 50X, 100X)
5. **Results JSON** in `data/benchmarks/extreme_compression_results.json`
