# Protein Embedding Codec

## Project Overview
Universal codec for PLM per-residue embeddings. **232 compression methods benchmarked** across 34 experiments. The recommended codec (V2 `balanced`) uses ABTT3 preprocessing + random projection to 512d + Product Quantization (M=128) to achieve **33 KB/protein** — 40% smaller than V1 — while preserving Ret@1=0.786 and SS3 Q3=0.807 (97% of raw ProtT5). Five selectable quality tiers from 17–55 KB. All tiers share identical retrieval quality; storage/size tradeoff is purely in per-residue fidelity.

## Quick Start
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

# Generate all publication figures
uv run python experiments/make_benchmark_barplots.py          # Per-benchmark + V2 + Pareto
uv run python experiments/make_publication_figures.py          # Legacy figures

# Trained ChannelCompressor (optional, requires labels)
uv run python experiments/11_channel_compression.py
uv run python experiments/13_robust_validation.py --step R1
```

## One Embedding V2 (Recommended)

Pipeline: ABTT k=3 (remove top-3 PCs) → RP to 512d → PQ/int4/binary quantize.
Protein vector (for retrieval): DCT K=4 of projected embeddings → (2048,) fp16.
Requires one-time codebook fitting on a training corpus (~512 KB shared file).

| Mode | Quantization | Size (L=175) | vs Mean Pool | Ret@1 | SS3 Q3 | SS8 Q8 | Disorder ρ | TM F1 |
|------|-------------|-------------|-------------|-------|--------|--------|-----------|-------|
| **`balanced`** | **PQ M=128** | **26 KB** | **13x** | **0.786** | **0.807** | **0.670** | **0.584** | **0.731** |
| `compact` | PQ M=64 | 15 KB | 7.5x | 0.786 | 0.778 | 0.637 | 0.549 | 0.701 |
| `binary` | 1-bit sign | 15 KB | 7.5x | 0.786 | 0.776 | 0.636 | 0.597 | 0.750 |
| `micro` | PQ M=32 | 10 KB | 4.7x | 0.786 | 0.739 | 0.594 | 0.495 | 0.579 |
| `full` | int4 scalar | 48 KB | 24x | 0.786 | 0.816 | 0.681 | 0.597 | 0.752 |

Size formula: L × M + 4096 bytes (PQ codes + protein_vec fp16). PQ codes are incompressible (7.81/8.00 bits entropy — balanced codebook utilization). Shared codebook: ~512 KB per mode (downloaded once).

```python
# One-time: fit codebook on training corpus
from src.one_embedding.codec_v2 import OneEmbeddingCodecV2
codec = OneEmbeddingCodecV2(mode='balanced')
codec.fit(training_embeddings)
codec.save_codebook('codebook.h5')

# Encode (sender side)
codec = OneEmbeddingCodecV2(mode='balanced', codebook_path='codebook.h5')
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.h5")

# Decode (receiver side — h5py + numpy + shared codebook)
data = OneEmbeddingCodecV2.load("compressed.h5", codebook_path="codebook.h5")
data['per_residue']   # (L, 512) for per-residue tasks
data['protein_vec']   # (2048,) for retrieval / clustering / UMAP
```

### V1 Codec (training-free, no codebook needed)
```python
from src.one_embedding.codec import OneEmbeddingCodec
codec = OneEmbeddingCodec(d_out=512, dct_k=4)
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.h5")
# Receiver needs only h5py — no codec code, no scipy
```

## The Journey: 232 Methods in 34 Experiments

### Phase 1–4: Trained Compression (Experiments 1–10)
Explored attention pooling, MLP autoencoders, ChannelCompressor. Trained ChannelCompressor with contrastive fine-tuning achieved Ret@1=0.795 (d256, 3-seed mean). Requires labels and training — not universal.

### Phase 5: Universal Codec Quest (Experiments 18–24)
Pivoted to training-free codecs. Tested DCT, Haar wavelets, spectral fingerprints, path signatures, curvature, gyration tensors, Fisher vectors, kernel mean embeddings. **Key negative**: path geometry adds noise, not signal. **Key positive**: DCT K=1 === mean pool (mathematically proven); [mean|max] concat is a free +4pp retrieval boost.

### Phase 6: The Chained Codec Breakthrough (Experiments 25–26)
Discovered that chaining D-compression (RP512) + L-compression (DCT K=4) solves the fundamental tension: D-compression preserves per-residue, L-compression boosts retrieval. 14 codecs × 3 PLMs benchmarked. Best: rp512+dct_K4 → Ret@1=0.780, SS3=0.815.

### Phase 7: Preprocessing + Quantization (Experiment 29)
ABTT3 (remove top-3 PCs) discovered as a free retrieval boost (+0.004 Ret@1). int4 quantization verified lossless for retrieval. 10-seed variance analysis: RP std=0.004. **The V1 One Embedding**: ABTT3+RP512+int4+DCT K4 → Ret@1=0.784, SS3=0.809, 48 KB.

### Phase 8: Extreme Compression (Experiment 28)
45 methods on ProtT5: wavelets, CUR decomposition, channel pruning, product quantization, residual VQ, tensor train, NMF, SimHash. **All on raw 1024d**. PQ M=64 best at 0.701. Key insight missed at the time: should have tested on preprocessed space.

### Phase 9: V2 Codec — The Preprocessed Space Changes Everything (Experiments 31–34)
Re-tested all compression on ABTT3+RP512 (decorrelated, isotropic). Results dramatically better:

- **Binary (1-bit) beats int4 for retrieval** (0.787 vs 0.784) — RaBitQ effect confirmed
- **PQ M=128 matches V1 quality at 40% less storage** (33 vs 55 KB)
- **PQ M=64 at 22 KB retains 93% SS3 quality**
- **Pure VQ fails in 512d** — even K=16384 caps at 0.621 Ret@1
- **RVQ fails in 512d** — residual norms barely decrease between levels
- **Hybrid VQ+PQ** works but doesn't beat pure PQ at same size
- **OPQ (learned rotation)** doesn't help — RP already decorrelates

## Idea Space: 232 Methods, What's Exhausted, What Remains

### Exhausted (confirmed dead ends)
- **More pooling variants** — 29 tested (mean, max, percentile, trimmed, power, norm-weighted, IQR, etc.). Mean pool is near-optimal for contrastive-trained PLMs. Diminishing returns past [mean|max].
- **Spectral/frequency transforms** — DCT, Haar, spectral fingerprint, spectral moments. DCT K=4 is the sweet spot. Higher K hurts. Haar is lossless but no retrieval gain.
- **Path geometry** — path signatures (depth 2, 3), discrete curvature, gyration tensor, displacement DCT, MSD, direction autocorrelation. ALL below ground zero. Mean pool already captures what matters.
- **Fisher vectors, Gram features** — 0.620 and 0.182 Ret@1. Poor for protein family retrieval.
- **Whole-vector VQ/RVQ in 512d** — codebook can't cover the space. Even K=16384: only 0.621 Ret@1. Residual norms barely decrease between RVQ levels.
- **Delta/temporal encoding** — residues are i.i.d. (lag-1 autocorrelation: negative). DPCM int4: 0.136 Ret@1 (catastrophic).
- **Learned rotation before PQ (OPQ)** — RP already decorrelates. OPQ makes things worse.
- **Entropy coding on PQ codes** — codes at 7.81/8.00 bits entropy. All 256 centroids used uniformly. Gzip/zstd: 0% compression. Balanced codebooks are a feature, not a bug.

### Partially explored (diminishing returns)
- **Extreme compression** — 50 methods: wavelets, CUR, pruning, tensor train, NMF, SimHash, PQ, optimal transport. All on raw 1024d. Best: PQ M=64 at 0.701. Re-testing on preprocessed space (Phase B) was the breakthrough.
- **Quantization bit-widths** — int8 lossless, int4 near-lossless, int2 decent (0.778/0.784), binary surprisingly good for retrieval (0.787 beats int4). RaBitQ double rotation helps int2 specifically.
- **Trained models** — ChannelCompressor (Ret@1=0.795), MLP-AE, AttentionPool, VQ-Compressor, HPO. Trained models achieve highest absolute quality but require labels and aren't universal.

### Genuinely unexplored (potential future work)
1. **Sequence-conditioned decoding** — use AA sequence + small correction code instead of storing per-residue. Could reach ~4 KB. Requires training a decoder (~5M params). Violates "h5py+numpy only" receiver constraint.
2. **Task-aware PQ codebook** — optimize codebook for SS3 quality directly, not MSE. Could improve per-residue at same bitrate.
3. **Matryoshka/Reverse Distillation** — restructure the PLM so first k dims ARE a smaller model's output. Changes the input, not the codec. Recent ICLR 2026 work shows this is feasible for ESM2.
4. **Asymmetric quantization** — different precision for different sub-spaces (variance-based). Our A2 non-uniform experiment showed modest gains (+0.003 Ret@1 vs uniform at same budget).
5. **Cross-protein shared dictionary** — learn "structural motif atoms" as a global dictionary. Different from PQ (global, not per-subspace). Untested.

### The fundamental limit
Theoretical floor: ~5–7 KB per protein (from intrinsic dimensionality ~80 × effective positions ~35). V2 `balanced` at 26 KB is ~4x above this. The gap is reconstruction precision: 128 sub-spaces × 256 centroids × 4d per sub-vector gives only 4 float32 values of freedom per sub-space, limiting per-residue fidelity. Going below 26 KB means accepting lower per-residue quality (which `compact` and `micro` do).

## What Works, What Doesn't

### Works
- ABTT preprocessing (removes dominant protein-identity PCs)
- Random projection (JL-based dimensionality reduction, norm-preserving)
- Product Quantization on the preprocessed space (sub-vector codebooks)
- DCT K=4 for protein-level vectors (spectral pooling)
- Binary quantization for retrieval-only use cases

### Doesn't Work
- Path geometry features (signatures, curvature, gyration) — add noise
- Fisher vectors, Gram features — poor for family retrieval
- Delta/DPCM encoding — residues are i.i.d., deltas have MORE variance
- Whole-vector VQ in 512d — codebook can't cover the space
- RVQ in 512d — residuals don't decrease meaningfully
- OPQ/learned rotation after RP — RP already decorrelates
- Two-head joint training — hurts retrieval vs sequential approach

## Architecture
- `src/one_embedding/` — **OneEmbeddingCodec** (V1), **OneEmbeddingCodecV2** (PQ), transforms (DCT, Haar, spectral), universal codecs, preprocessing (ABTT, PCA rotation), quantization (int2/int4/int8/binary/PQ/RVQ), path transforms, enriched transforms, data analysis
- `src/compressors/` — ChannelCompressor (trained), AttentionPool, MLP-AE, VQ, baselines
- `src/extraction/` — ESM2 + ProtT5 + ESM-C embedding extraction
- `src/training/` — Unified trainer with reconstruction, contrastive, VICReg losses
- `src/evaluation/` — Retrieval (cosine+euclidean), per-residue probes (SS3/SS8/disorder/TM/SignalP), biological annotations (GO/EC/Pfam/taxonomy), hierarchy, statistical tests
- `src/utils/` — Device management (MPS/CPU), H5 I/O
- `experiments/` — 34 experiment scripts (01–34) + figure generators
- `tests/` — 318 tests across 13 modules

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
