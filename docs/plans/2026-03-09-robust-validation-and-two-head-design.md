# Design: Robust Validation + Two-Head Architecture

**Date**: 2026-03-09
**Status**: Approved
**Scope**: Experiments 13 (validation) and 14 (two-head unified model)

## Context

Phase 9A validated per-residue utility on CB513 (Q3 retention 0.985-0.990). But:
- ProtT5 contrastive 0.808 Ret@1 is single-seed (fragile)
- CB513 disorder was binary (uninformative) — need continuous CheZOD
- No TM topology or cross-dataset SS3 validation yet
- Two checkpoints still needed (unsup for per-residue, contrastive for retrieval)

## Experiment 13: Robust Validation

### 13a: Multi-Seed ProtT5 (R1)
Train ProtT5 ChannelCompressor d256 at seeds 123, 456:
- Unsupervised (200 epochs) then contrastive fine-tuning (100 epochs)
- Evaluate on Phase 8 benchmarks + CB513 probes
- Output: 3-seed mean+/-std for headline numbers

### 13b: CheZOD Continuous Disorder (R2+R3)
- Parse SETH/CheZOD: 1174 train + 117 test, continuous z-scores
- Extract ESM2-650M + ProtT5-XL embeddings
- Ridge regression probe, report Spearman rho
- Compare original vs PCA-256 vs unsup d256 vs contrastive d256

### 13c: TMbed Topology (R2+R3)
- Parse cv_00_annotated.fasta: 1307 proteins, 4-class (TM-helix, TM-beta, Signal, Other)
- Extract embeddings, 80/20 split, logistic regression probe
- Report accuracy + macro F1

### 13d: TS115 Cross-Dataset SS3 (R2+R3)
- Same CSV format as CB513, 115 proteins
- Validates CB513 Q3/Q8 results on independent set

### Script Structure
`experiments/13_robust_validation.py` with steps:
- R1: Multi-seed ProtT5 training (GPU-intensive, sequential)
- R2: Embedding extraction for CheZOD + TMbed + TS115 (GPU, sequential per PLM)
- R3: All probes (CPU-only)

Results: `data/benchmarks/robust_validation_results.json`

## Experiment 14: Two-Head Unified Architecture

### Architecture
```
Input (B, L, D) -> Shared Encoder -> Latent (B, L, D')
                                       |-> Reconstruction Head (decoder) -> (B, L, D)
                                       |-> Retrieval Head (pool+MLP+proj) -> (B, D_proj)
```

### Key Insight
Contrastive gradient flows through retrieval head projection, NOT directly through
per-residue latent space. The projection head absorbs family-discriminative restructuring.

### Training
Joint single-phase: loss = 1.0 * L_recon + 0.5 * L_infonce
200 epochs, ProtT5-XL, d256, 3 seeds.

### Implementation
Extend ChannelCompressor with optional `retrieval_head_dim` parameter.
Backwards-compatible — existing checkpoints still load.

### Success Criteria
Ret@1 >= 0.78 AND Q3/Q3_orig >= 0.98 from a single checkpoint.

## Files

| File | Action |
|------|--------|
| src/evaluation/per_residue_tasks.py | EDIT: add CheZOD/TMbed/TS115 parsers |
| src/compressors/channel_compressor.py | EDIT: add optional retrieval head |
| experiments/13_robust_validation.py | CREATE |
| experiments/14_two_head_training.py | CREATE |
