# Protein Embedding Codec

## Project Overview
Training-free universal codec for PLM per-residue embeddings. Compresses (L, D) to (L, 512) float16 — 25% of raw size — while preserving both per-residue tasks (97% SS3 retention) and protein-level retrieval (Ret@1=0.786 with ABTT k=3 pre-processing, 0.780 without). Float16 benchmarked with zero quality loss. Also includes a trained ChannelCompressor (1024d → 256d fp32, Ret@1=0.795) as a comparison point. Multi-seed RP variance: 0.779 +/- 0.004 (10 seeds).

## Quick Start
```bash
# Extract PLM embeddings (prerequisite)
uv run python experiments/01_extract_residue_embeddings.py

# Universal codec benchmark (training-free, main story)
uv run python experiments/25_plm_benchmark_suite.py          # 14 codecs × 3 PLMs
uv run python experiments/26_chained_codec_benchmark.py       # Chained codecs (best results)

# Generate publication figures
uv run python experiments/make_publication_figures.py

# Trained ChannelCompressor pipeline (optional, requires labels)
uv run python experiments/11_channel_compression.py           # Train ChannelCompressor
uv run python experiments/13_robust_validation.py --step R1   # Multi-seed validation
uv run python experiments/13_robust_validation.py --step R3   # Cross-dataset evaluation

# Exhaustive sweep of ~30 untried techniques
uv run python experiments/29_exhaustive_fruit_sweep.py        # All 9 steps (F, A-I)

# Earlier exploration
uv run python experiments/02_baseline_benchmarks.py           # PCA/mean-pool baselines
uv run python experiments/15_external_validation.py           # ToxFam external validation
uv run python experiments/16_hpo_contrastive.py               # Optuna HPO
uv run python experiments/17_scaling_and_ablations.py         # Scaling, ablations
```

## One Embedding Format
The output is a single fixed **(512, 512) matrix** per protein (~50 KB int4+gzip).

```python
# Encode: raw PLM → One Embedding
from src.one_embedding.codec import OneEmbeddingCodec
codec = OneEmbeddingCodec(d_out=512, dct_k=4)
codec.encode_h5_to_h5("raw_embeddings.h5", "compressed.h5")

# Use (receiver side — just h5py + numpy):
import h5py
f = h5py.File("compressed.h5", "r")
matrix = f["prot_123"]["one_embedding"][:]     # (512, 512) fixed
L = f["prot_123"].attrs["seq_len"]

# Retrieval: dct_summary(matrix[:, :L].T, K=4) → (2048,)
# Per-residue: matrix[:, :L].T → (L, 512)
```

Pipeline: ABTT k=3 (remove top-3 PCs) → RP to 512d → int4 quantize → transpose + zero-pad.
Best codec Ret@1=0.784 (Fam), 0.952 (SF), SS3 Q3=0.809 (96% of raw).

## Architecture
- `src/one_embedding/` - **OneEmbeddingCodec**, transforms (DCT, Haar, spectral), universal codecs, preprocessing (centering, ABTT, PCA rotation), transposed transforms, data analysis
- `src/compressors/` - ChannelCompressor (trained), AttentionPool (explored), baselines
- `src/extraction/` - ESM2 + ProtT5 + ESM-C embedding extraction
- `src/training/` - Unified trainer with reconstruction, contrastive losses, early stopping
- `src/evaluation/` - Retrieval, classification, reconstruction, per-residue probes, statistical tests
- `src/utils/` - Device management (MPS/CPU), H5 I/O
- `experiments/archive/` - Superseded exploration scripts (phases 5-10)

## Hardware
- MacBook Pro (Mac15,10) with Apple M3 Max, 14 cores (10P + 4E), 96 GB RAM
- GPU: MPS (Metal Performance Shaders) — shared memory with system RAM

## Thermal / Resource Management
- **Never run more than 1 GPU-intensive training job at a time** — concurrent MPS jobs cause excessive heat and slow down due to memory bandwidth contention
- Run training jobs sequentially, not in parallel (even with background tasks)
- Monitor with `pmset -g therm` and `os.getloadavg()` during long runs
- If system load > 10 or thermal warnings appear, pause and wait before continuing

## Key Conventions
- Python 3.12 (required for fair-esm compatibility)
- PyTorch with MPS (Apple Silicon) or CUDA
- All tensors float32 (MPS does not support float64)
- MPS: `torch.linalg.svdvals` not supported — move tensors to CPU first
- Experiments use `sys.path.insert(0, ...)` for imports from project root
- Per-residue embeddings stored as H5 with gzip compression
- Each protein's embedding: shape (L, D) where L=sequence length, D=model dim
- LogisticRegression probes all use `random_state=42` for reproducibility
