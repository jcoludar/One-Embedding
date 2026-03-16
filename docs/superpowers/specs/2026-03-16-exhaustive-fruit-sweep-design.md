# Experiment 29: Exhaustive Low-Hanging Fruit Sweep

**Date**: 2026-03-16
**Goal**: Systematically try every untested low/medium-effort technique identified through comprehensive project audit, then document results.
**Constraint**: Training-free (no gradient descent, no labels) unless explicitly noted as "barely-trained" variant.

---

## Context

After 28 experiments and 52+ techniques, the project has:
- Best training-free codec: rp512+dct_K4 → Ret@1=0.780, SS3 Q3=0.815, 179 KB (26%)
- Best extreme compression: int4 → Ret@1=0.733, SS3=0.839, 85 KB (8x)
- Ground zero: raw mean pool → Ret@1=0.734, SS3=0.845

This experiment tests every remaining idea before declaring the search exhausted.

---

## Part A: Pre-Processing / Data Conditioning (Before Compression)

### A1. Global Mean Centering
- **What**: Subtract the global mean residue vector (computed over reference corpus) before RP.
- **Why**: RP preserves distances. The shared mean component doesn't discriminate between proteins — it's shared signal that wastes projection capacity.
- **How**: `mean_vec = mean over all residues in training set` (shape: (D,)). For each protein: `X_centered = X - mean_vec`. Then apply rp512 + dct_K4.
- **Store**: mean_vec (4 KB). Receiver needs it for decode.
- **Expected**: +0.01–0.03 Ret@1, neutral SS3.

### A2. Per-Channel Z-Score Standardization
- **What**: z-score each channel: `X_std = (X - channel_means) / channel_stds`.
- **Why**: PLM channels have different scales. RP treats all equally, so high-variance channels dominate. Standardizing equalizes contribution.
- **How**: Compute (mean, std) per channel on reference corpus. Apply before RP.
- **Store**: 2 * D floats (~8 KB). Receiver needs for decode.
- **Expected**: Helps ESM2 (1280d, more scale variation) more than ProtT5.

### A3. All-but-the-Top (Remove Top-k Principal Components)
- **What**: From NLP (Mu & Viswanath 2018). Remove the top-1 (or top-3) principal component from all embeddings. In word embeddings, this removes frequency bias and *improves* downstream tasks.
- **Why**: The dominant PC captures corpus-level bias (e.g., sequence length, average amino acid composition), not protein-specific signal. Removing it exposes the discriminative directions.
- **How**: PCA on reference corpus → compute top-k PCs → project out: `X_clean = X - X @ pc @ pc.T` for each top-k PC.
- **Store**: k principal component vectors (k * D floats, ~4-12 KB).
- **Expected**: +0.01–0.02 Ret@1 (well-documented in NLP). May help or be neutral for per-residue.

### A4. PCA Rotation Before RP (Rotate, Don't Reduce)
- **What**: Apply full-rank PCA rotation to align variance with axes, then RP on the rotated space.
- **Why**: PCA concentrates variance in first components. RP on PCA-rotated data projects from a space where "important stuff" is axis-aligned, making random projections more sample-efficient.
- **How**: Fit PCA(n_components=D) on reference corpus → get rotation matrix (D, D). Apply rotation before RP.
- **Store**: D*D rotation matrix (~4 MB for 1024). Receiver needs for decode.
- **Tradeoff**: 4 MB overhead vs potential quality gain. May not be worth it for the codec story.
- **Expected**: +0.005–0.015 Ret@1.

---

## Part B: Transposed Matrix View (The "Flip" Insight)

### B1. Channel Resampling (Interpolate L → fixed l)
- **What**: Transpose (L, D) → (D, L). Interpolate each channel signal from L to fixed l (e.g., l=64, 128, 256). Result: fixed (D, l) matrix.
- **Why**: Instead of collapsing L via pooling (loses positional info), downsample the L-axis signal. Preserves positional structure — residue 10 and residue 100 map to different positions in the output.
- **How**: `scipy.signal.resample(channel, l)` or `torch.nn.functional.interpolate(mode='linear')` per channel.
- **Variants**: l ∈ {32, 64, 128, 256}. Protein vector = flattened (D*l,) or mean-pooled across l.
- **Expected**: Novel approach. Retrieval comparable to mean pool for moderate l; per-residue partially recoverable via inverse interpolation.

### B2. Per-Protein SVD (Transposed)
- **What**: SVD of (D, L) matrix. Left singular vectors U[:, :k] are (D, k) — fixed size regardless of L. Flatten to (D*k,) protein vector. Singular values S[:k] encode magnitude.
- **Why**: Captures the principal modes of variation across residues. First singular vector = direction of maximum residue variance. Different from mean pool (which is the DC component).
- **How**: `U, S, Vt = torch.linalg.svd(X.T, full_matrices=False)`. Take top k.
- **Variants**: k ∈ {1, 2, 4, 8}. Protein vector: U[:,:k] * S[:k] flattened, or just U[:,:k] flattened.
- **Expected**: k=1 should correlate with but differ from mean pool. Higher k captures richer structure.

### B3. Zero-Pad + Flatten + RP
- **What**: Pad all proteins to L_max residues (zero-pad), flatten to (D * L_max,), then RP to d_out.
- **Why**: Preserves ALL positional information. JL guarantees hold on the flattened vector.
- **How**: L_max from dataset (or fixed, e.g., 512). Pad: `X_pad = zeros(L_max, D); X_pad[:L, :] = X`. Flatten → RP.
- **Concern**: L_max * D is very large (512 * 1024 = 524K dims). RP matrix is huge. May only be practical for protein-level vector, not storage.
- **Expected**: Good retrieval if L_max is reasonable. Impractical for large L_max.

### B4. Channel-wise Statistics (Explicit Transposed Pooling)
- **What**: For each of D channels, compute statistics across L residues: [mean, std, min, max, skew, range]. Result: (D, 6) → flatten to (6D,).
- **Why**: Richer than mean pool (D,) but fixed-size. Captures per-channel distribution shape.
- **How**: Standard numpy/torch operations on (L, D) → (D,) per statistic.
- **Variants**: [mean, std] (2D), [mean, std, skew] (3D), [mean, std, min, max] (4D), full 6D.
- **Expected**: 2D ([mean, std]) should improve over mean alone. Similar spirit to [mean|max] but per-channel std is more informative than max.

---

## Part C: Improved Pooling Strategies

### C1. PLM Attention-Weighted Pooling
- **What**: Extract attention matrices from the PLM (last layer). Use attention weights to create importance-weighted mean pool.
- **Why**: The PLM already "knows" which residues matter. Active sites, conserved cores get higher attention. Free information.
- **How**: During extraction, save `attentions=True`. Average last-layer attention → (L,) importance weights. Weighted mean: `protein_vec = sum(w_i * x_i) / sum(w_i)`.
- **Concern**: Requires re-extraction with attention saving. Storage overhead during extraction.
- **Expected**: +0.005–0.02 Ret@1. Most likely helps ProtT5 (better internal attention structure).

### C2. Percentile Pooling
- **What**: Per channel, compute percentiles across residues: [p10, p25, p50, p75, p90]. Concatenate.
- **Why**: Captures distribution shape without max's outlier sensitivity. A mini-histogram per channel.
- **How**: `np.percentile(X, [10, 25, 50, 75, 90], axis=0)` → (5, D) → flatten to (5D,).
- **Variants**: [p25, p50, p75] = (3D,), [p10, p50, p90] = (3D,).
- **Expected**: Should beat [mean|max] (2D) by capturing more distribution info.

### C3. Trimmed Mean Pool
- **What**: Remove top/bottom 10% of values per channel before averaging. Robust mean.
- **Why**: Removes outlier residues (disordered termini, unusual insertions) that skew the mean.
- **How**: `scipy.stats.trim_mean(X, proportiontocut=0.1, axis=0)`.
- **Expected**: Small improvement (+0.005). Most useful for proteins with long disordered regions.

---

## Part D: RP Variants and Characterization

### D1. Multi-Seed RP Variance
- **What**: Run the full codec (rp512 + dct_K4) with seeds {42, 123, 456, 789, 0, 7, 99, 2024, 31415, 271828}. Report mean ± std for all metrics.
- **Why**: The entire codec depends on one random matrix. Variance characterization is essential for the paper.
- **How**: Loop over seeds, full eval each.
- **Expected**: Low variance (Ret@1 std < 0.005) would strengthen the paper. High variance would be concerning.

### D2. Sparse Random Projection (Achlioptas)
- **What**: Replace dense Gaussian RP with sparse {-1, 0, +1} at probabilities {1/6, 2/3, 1/6}.
- **Why**: Same JL guarantees, 3x faster, projection matrix is 2 bits/entry (97% are zero). Practical for deployment.
- **How**: Generate sparse matrix, apply as projection.
- **Expected**: Quality within 0.005 of dense RP. Speed 3x better.

### D3. Subsampled Randomized Hadamard Transform (SRHT)
- **What**: Structured RP via Hadamard matrix. O(D log d) instead of O(Dd).
- **Why**: Fastest JL-compatible transform. Deterministic given a sign-flip vector.
- **How**: Sign-flip → Hadamard transform → subsample d rows.
- **Expected**: Quality within 0.005 of dense RP. Speed much better for large D.

---

## Part E: Quantization and Coding Combinations

### E1. int4 Quantization of Codec Output
- **What**: Apply int4 quantization to the (L, 512) output of OneEmbeddingCodec (rp512).
- **Why**: int4 on raw embeddings was near-lossless (Ret@1=0.733 vs 0.734). Codec output is lower-dimensional (512 vs 1024), so int4 should be even safer. Gets ~16x total compression with both tasks.
- **How**: `codec.encode()` → `quantize_int4(per_residue)`. Protein_vec stored separately as float16.
- **Expected**: Ret@1 ≈ 0.778 (vs 0.780 float16), SS3 ≈ 0.813. Size: ~45 KB.

### E2. JPEG Pipeline (DCT → Quantize DCT Coefficients → Entropy Code)
- **What**: Per channel along L: (1) DCT transform, (2) quantize DCT coefficients with frequency-dependent step sizes (coarser for high-freq), (3) Huffman or arithmetic coding.
- **Why**: This is the actual JPEG recipe. Quantizing in frequency domain is fundamentally better than quantizing in spatial domain because coefficient importance is ordered.
- **How**: Per channel: `dct_coeffs = scipy.fft.dct(channel, norm='ortho')`. Quantize: `q = round(dct_coeffs / step_sizes)` where step_sizes increase with frequency. Entropy code the quantized values.
- **Variants**: Aggressive (keep only first 25% of coefficients), moderate (50%), conservative (75%).
- **Expected**: Could achieve 8-16x compression with <2% quality loss. This is the compression technique that revolutionized images.

### E3. Predictive Coding (DPCM / FLAC-style)
- **What**: Per channel: predict residue i from residue i-1 (or linear predictor from i-1, i-2). Store prediction error (residual). Entropy-code residuals.
- **Why**: Neighboring residues have highly correlated embeddings. Prediction residuals have much lower variance → fewer bits needed. This is how FLAC achieves 50-60% lossless compression.
- **How**: `residual[i] = X[i] - X[i-1]` (order-1). Or `residual[i] = X[i] - (2*X[i-1] - X[i-2])` (order-2). Then quantize + entropy code.
- **Expected**: Lossless: ~50-60% of raw size. Lossy with quantization: ~25-35%.

### E4. Huffman Coding of int4 Values
- **What**: After int4 quantization, Huffman-code the quantized symbols.
- **Why**: int4 values aren't uniformly distributed — some bins are more common. Variable-length coding exploits this.
- **How**: Build Huffman tree from int4 value frequencies. Encode.
- **Expected**: ~20-30% further size reduction on top of int4.

### E5. Mixed-Precision Per Channel
- **What**: Allocate bit-width per channel based on importance (inter-protein variance). Top-k channels get 8 bits, next-k get 4 bits, rest get 2 bits or are dropped.
- **Why**: Not all channels need equal precision. Adaptive bit allocation (like MP3/AAC).
- **How**: Rank channels by importance → assign bit budgets → quantize each group.
- **Expected**: Better rate-distortion than uniform int4 at same average bit-width.

---

## Part F: Data Characterization (Informs Everything)

### F1. Global Intrinsic Dimensionality
- **What**: Stack all residue embeddings into (N_total, D) matrix. Compute SVD. Report singular value spectrum, participation ratio, and dimension at 90%/95%/99% variance explained.
- **Why**: If 95% of variance lives in 200 dims, RP to 512 is conservative. If it's 800 dims, RP to 512 is lossy. This number informs optimal d_out.
- **How**: Sample ~50K residues (random across proteins). SVD. Plot spectrum.
- **Expected**: Likely 200-400 intrinsic dimensions (typical for PLM embeddings).

### F2. Per-Channel Distribution Analysis
- **What**: For each of D channels, compute: mean, std, skewness, kurtosis, inter-protein variance, intra-protein variance. Plot histograms for representative channels.
- **Why**: Determines optimal quantization strategy. Gaussian channels → Lloyd-Max. Sparse channels → can be dropped. High inter/intra ratio channels → most discriminative.
- **How**: Sample residues, compute stats per channel.
- **Expected**: Will reveal which channels carry discriminative signal and guide mixed-precision allocation.

### F3. Inter-Channel Correlation Matrix
- **What**: Compute correlation matrix between all D channels. Identify clusters of redundant channels.
- **Why**: Highly correlated channels are redundant — compressing them together (via PCA or RP) is more efficient than treating independently.
- **How**: Correlation matrix of (N_samples, D). Hierarchical clustering.
- **Expected**: PLM embeddings likely have ~10-20 correlated channel clusters.

---

## Part G: Evaluation Enhancements

### G1. Random Neighbor Score (RNS)
- **What**: Prabakaran & Bromberg (2025). For each protein, compute fraction of k-nearest neighbors that are biologically unrelated (different superfamily).
- **Why**: Per-protein quality metric. Identifies which proteins are "damaged" by compression.
- **How**: For each protein in test set: find k=10 nearest neighbors in compressed space. Count those from different superfamilies. RNS = count / k.
- **Expected**: Codec RNS should be close to raw RNS. Proteins with high delta-RNS are compression casualties.

### G2. Remote Homology Analysis
- **What**: Analyze existing superfamily-level and fold-level Ret@1 results separately. Break down by SCOPe hierarchy level.
- **Why**: Family retrieval (Ret@1=0.780) is the easy problem. Superfamily and fold retrieval test remote homology detection — the hard and valuable problem.
- **How**: Already computed in retrieval metrics. Just need proper analysis and reporting.
- **Expected**: Codec likely loses more on remote homology than family retrieval.

### G3. Stronger Probes (2-Layer MLP)
- **What**: Replace LogisticRegression probes with 2-layer MLP (hidden=256, ReLU) for SS3, disorder, topology.
- **Why**: If MLP closes the gap between raw and compressed, information is present but nonlinearly encoded. Changes the story from "3% loss" to "0% loss with better decoder."
- **How**: `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)`.
- **Expected**: Gap shrinks by 30-50%. Important for paper narrative.

### G4. Matryoshka Dimension Ordering
- **What**: After RP to 512d, reorder projected dimensions by decreasing variance (measured on reference corpus). Test truncation to 256d, 128d.
- **Why**: Makes the codec adaptive — users needing less precision truncate to fewer dims. Zero cost at encode time (just reorder RP matrix columns).
- **How**: Measure per-dimension variance on corpus. Sort RP columns. Evaluate at 512, 256, 128 dims.
- **Expected**: Sorted-256d should outperform random-256d by 0.02-0.04 Ret@1.

---

## Part H: Reference Corpus Approaches

### H1. k-Means Centroid Residual Coding
- **What**: Learn k centroids (k=100-500) on reference residue embeddings via k-means. For each residue, store (centroid_id, residual). Residuals have lower variance → compress better.
- **Why**: Genomics analog (CRAM). Shared structure captured by centroids; protein-specific info in small residuals.
- **How**: k-means on sampled residues → codebook. Per residue: closest centroid + residual. Quantize residual with fewer bits.
- **Expected**: 2-4x additional compression on per-residue matrix with <1% quality loss.

### H2. Corpus-Based PCA as D-Compression
- **What**: Fit PCA(n_components=512) on reference corpus residues. Project all proteins.
- **Why**: The "obvious" approach that was only tested within enriched transforms, never as a standalone codec competing with RP.
- **How**: PCA fit on 50K sampled residues → (D, 512) projection matrix. Apply to new proteins.
- **Expected**: PCA should beat RP at same output dim because it's data-adapted. But requires storing/distributing the projection matrix (~2 MB).

---

## Part I: Multi-Resolution Framing

### I1. Three-Level Retrieval System
- **What**: Frame existing codec outputs as a multi-resolution retrieval system:
  - Level 1: SimHash 1024-bit (21 KB, Hamming distance for coarse filtering)
  - Level 2: protein_vec 2048d float16 (4 KB, cosine similarity for ranking)
  - Level 3: per_residue (L, 512) float16 (175 KB, late interaction for re-ranking)
- **Why**: This is how production search systems work (Google, Bing, FAISS). Coarse → fine.
- **How**: Compute all three representations. Benchmark retrieval quality at each level and cascaded.
- **Expected**: Cascaded retrieval matches Level 3 quality with Level 1 speed.

---

## Evaluation Protocol

All techniques evaluated on **ProtT5-XL, SCOPe 5K** (primary) with select winners re-evaluated on **ESM2-650M** and **ESM-C-300M**.

**Metrics** (per technique):
- Ret@1, MRR (family retrieval, cosine metric)
- SS3 Q3 (per-residue probe, where applicable)
- Storage size (bytes/protein)
- Encode time (ms/protein)

**Baselines for comparison**:
- Ground zero: raw mean pool → Ret@1=0.734, SS3=0.845
- Current best codec: rp512+dct_K4 float16 → Ret@1=0.780, SS3=0.815, 179 KB
- Current best trained: ChannelCompressor → Ret@1=0.808, SS3=0.834

**Significance**: Bootstrap 95% CI on Ret@1. Flag if improvement is within CI.

---

## Implementation Order

1. **Part F** (Data Characterization) — informs all other experiments
2. **Part A** (Pre-processing) — quick wins, modify existing codec
3. **Part B** (Transposed Matrix) — novel approach
4. **Part D** (RP Variants) — characterize and improve core codec
5. **Part C** (Pooling) — protein-level vector improvements
6. **Part E** (Quantization/Coding) — storage optimization
7. **Part G** (Evaluation) — strengthen paper
8. **Part H** (Reference Corpus) — semi-training-free approaches
9. **Part I** (Multi-Resolution) — framing and cascade evaluation

Then: **smart combinations** of winners from each part.

---

## Success Criteria

- At least 3 techniques that improve Ret@1 by ≥0.005 over current best (0.780)
- At least 1 technique that achieves ≥0.770 Ret@1 at ≤100 KB (currently no entry here)
- Multi-seed RP variance characterized (std reported)
- Global intrinsic dimensionality measured
- All results documented in `data/benchmarks/exhaustive_sweep_results.json`
