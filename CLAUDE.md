# Protein Embedding Compression Explorer

## Project Overview
Sequence-only learned compressor that compresses PLM per-residue embeddings (1024d → 256d) while retaining >98% downstream task utility. No structure model required.

## Quick Start
```bash
# Core pipeline (run in order)
uv run python experiments/11_channel_compression.py          # ChannelCompressor training
uv run python experiments/13_robust_validation.py --step R1  # Multi-seed ProtT5 training
uv run python experiments/13_robust_validation.py --step R3  # Cross-dataset evaluation

# Additional experiments
uv run python experiments/15_external_validation.py          # ToxFam external validation
uv run python experiments/16_hpo_contrastive.py              # Optuna HPO (all steps)
uv run python experiments/17_scaling_and_ablations.py        # Scaling, ablations, Pareto

# Earlier exploration (01-04 are prerequisite data/baseline steps)
uv run python experiments/01_extract_residue_embeddings.py   # Extract PLM embeddings
uv run python experiments/02_baseline_benchmarks.py          # PCA/mean-pool baselines
```

## Architecture
- `src/compressors/` - ChannelCompressor (best), AttentionPool (explored), baselines
- `src/extraction/` - ESM2 + ProtT5 embedding extraction
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
