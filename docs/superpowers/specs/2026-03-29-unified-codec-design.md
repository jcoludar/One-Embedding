# Unified One Embedding Codec — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Merge V1 (fp16) and V2 (quantized) codecs into a single configurable class with 768d default and ~30x compression out of the box.

## Motivation

The current codebase has two separate codec classes:
- `OneEmbeddingCodec` (codec.py) — fp16 storage, 2.7x compression, no quantization
- `OneEmbeddingCodecV2` (codec_v2.py) — PQ/int4/binary on 512d, 14-47x compression

This splits the user experience and leaves V2 stuck on 512d. Disorder retention on 512d is 88-90%, which is the project's sole weak spot. Moving to 768d should recover ~4-6pp on disorder while maintaining strong compression.

## Design Principle: The Apple Approach

One codec. Sensible default. Configurable knobs for power users. No mode menus.

```python
# It just works — ~30x compression, PQ on 768d
codec = OneEmbeddingCodec()

# Max fidelity on disorder
codec = OneEmbeddingCodec(d_out=1024, quantization=None)

# Less compression, safer
codec = OneEmbeddingCodec(quantization='int4')  # 10x

# More compression
codec = OneEmbeddingCodec(pq_m=96)  # 41x

# Retrieval-focused
codec = OneEmbeddingCodec(quantization='binary')  # 41x
```

## API: Three Knobs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_out` | int | 768 | Dimensions after random projection. Higher = more fidelity, less compression. Set to input dim (e.g. 1024) to skip RP entirely. |
| `quantization` | str or None | `'pq'` | Per-residue storage method. `None` (fp16), `'int4'`, `'pq'`, `'binary'`. |
| `pq_m` | int or None | auto | PQ subquantizers. Only used when `quantization='pq'`. Auto = `d_out // 6` (targets ~30x compression with 6d sub-vectors). Must divide `d_out` evenly. |
| `dct_k` | int | 4 | DCT coefficients for protein vector. |
| `seed` | int | 42 | Fixed seed for RP matrix. |
| `codebook_path` | str or None | None | Path to pre-fitted codebook H5. Required for PQ decode. |

### Auto pq_m Logic

When `quantization='pq'` and `pq_m` is not specified:
- Target: 6d sub-vectors → `pq_m = d_out // 6`
- For d_out=768: pq_m = 128 (exact)
- For d_out=512: pq_m = 85 → not a factor, fall back to largest factor of d_out ≤ d_out//6. For 512: factors are ..., 64, 128, ... → pq_m = 64
- For d_out=1024: pq_m = 170 → not a factor, fall back → pq_m = 128 (1024/128 = 8d subs)
- Implementation: iterate down from d_out//6 to find first value where d_out % pq_m == 0
- Validation at init: `d_out % pq_m == 0` or raise ValueError with suggestion

### Compression Reference

At d_out=768 (default), for a protein with L=175 residues:

| quantization | Bytes/residue | Size (L=175) | Compression vs raw fp32 |
|-------------|:---:|:---:|:---:|
| None (fp16) | 1,536 | 275 KB | 2.7x |
| 'int4' | 384 | 67 KB | 10x |
| 'pq' (M=192) | 192 | 34 KB | 20x |
| **'pq' (M=128, default)** | **128** | **23 KB** | **~30x** |
| 'pq' (M=96) | 96 | 17 KB | 41x |
| 'binary' | 96 | 17 KB | 41x |

## Encode/Decode Pipeline

```
Raw PLM (L, D_in)
  -> ABTT k=3 (remove top-3 corpus PCs)
  -> Random Project to d_out (SKIP if d_out >= D_in)
  -> Compute protein_vec: DCT K=4 on projected -> (d_out * K,) fp16
  -> Quantize per-residue based on `quantization`:
       None   -> cast to fp16
       'int4' -> per-channel int4 (scales + zero_points, training-free)
       'pq'   -> PQ encode to (L, M) uint8 codes (requires fitted codebook)
       'binary' -> 1-bit sign + per-channel means/scales (training-free)
  -> Store in .one.h5 format
```

Key invariant: `protein_vec` is always computed from the projected (pre-quantization) embeddings and stored as fp16. Retrieval quality is independent of quantization choice.

### RP Skip Logic

When `d_out >= D_in` (e.g., d_out=1024 with ProtT5 1024d input), skip the random projection entirely. ABTT3 is still applied. This is the "lossless" configuration — no dimensionality reduction, just preprocessing + fp16 storage.

## Codebook Handling

| quantization | Needs fit()? | Needs codebook file? |
|-------------|:---:|:---:|
| None (fp16) | Yes (ABTT stats) | Yes (ABTT stats only, small) |
| 'int4' | Yes (ABTT stats) | Yes (ABTT stats only, small) |
| 'pq' | Yes (ABTT stats + PQ) | Yes (ABTT stats + PQ centroids) |
| 'binary' | Yes (ABTT stats) | Yes (ABTT stats only, small) |

All configurations need `fit()` for ABTT corpus stats. PQ additionally fits M codebooks of K=256 centroids each.

The codebook H5 stores:
- `top_pcs`: (n_pcs, D_in) float32 — ABTT principal components
- `mean_vec`: (D_in,) float32 — corpus mean
- `pq_codebook`: (M, K, sub_dim) float32 — PQ centroids (if PQ)
- Attributes: `d_out`, `quantization`, `pq_m`, `pq_K`, `seed`

## File Changes

| File | Action |
|------|--------|
| `src/one_embedding/codec_v2.py` | Evolve into unified `OneEmbeddingCodec`. Add fp16 path, d_out=768 default, `quantization` parameter with auto pq_m, RP skip logic. Rename class. |
| `src/one_embedding/codec.py` | Replace with thin wrapper: `from .codec_v2 import OneEmbeddingCodec` for backward compat |
| `experiments/44_v2_768d_benchmark.py` | New experiment — benchmark sweep of all quantization types on 768d |
| `tests/test_codec_unified.py` | New — tests for unified codec (all quantization paths, RP skip, auto pq_m, round-trip encode/decode) |
| Existing experiment scripts | No changes needed — they import `OneEmbeddingCodec` or `OneEmbeddingCodecV2`, both will resolve to the new class |

## Benchmark Plan (Experiment 44)

Run all configurations through the Exp 43 rigorous framework (BCa CIs, CV-tuned probes, paired bootstrap retention, pooled disorder rho):

### Configurations to benchmark

| Config | d_out | quantization | pq_m | Expected compression |
|--------|:---:|:---:|:---:|:---:|
| lossless | 1024 | None | — | 2x |
| fp16-768 | 768 | None | — | 2.7x |
| int4-768 | 768 | int4 | — | 10x |
| pq192-768 | 768 | pq | 192 | 20x |
| **pq128-768** | **768** | **pq** | **128** | **~30x (default)** |
| binary-768 | 768 | binary | — | 41x |

### Tasks to evaluate
- SS3 Q3 on CB513 (per-residue classification)
- SS8 Q8 on CB513 (per-residue classification)
- Disorder pooled rho on CheZOD117 (per-residue regression, cluster bootstrap)
- Family Ret@1 on SCOPe 5K (protein-level retrieval)

### Success criteria
- Default config (pq128-768, ~30x) achieves >93% retention on SS3 and >90% on disorder
- If not, bump default to pq192-768 (~20x) — a one-line change to auto pq_m logic

### Compute estimate
~30 min per configuration (3 seeds, CV-tuning, bootstrap). 6 configs = ~3 hours total.

## What We're NOT Doing

- No changes to the `one_embedding/` package-level API (it wraps the research codec)
- No changes to .one.h5 format structure (metadata already captures codec params)
- No named "mode" presets in the class API — just parameters
- No PQ M=96 or M=64 as named tiers — users set pq_m directly if they want extreme compression
- No OPQ (learned rotation) — proven unnecessary on RP'd embeddings
- No entropy coding — proven dead end (PQ codes at max entropy)
