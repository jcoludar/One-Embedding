# Exp 48 — Random Neighbor Score under OneEmbedding compression

**Date:** 2026-04-30
**Question:** does the codec change embedding quality as measured by RNS
(Prabakaran & Bromberg, *Nat Methods* 23, 796–804, 2026)?
**Source:** `experiments/48_rns_compression.py` →
`data/benchmarks/rigorous_v1/exp48_rns_compression.json`
**Wall time:** 30.8 s on M3 Max (after junkyard extraction).

## Setup

| | |
|---|---|
| **Real corpus** | CB513 — 511 proteins, ProtT5-XL per-residue embeddings (`prot_t5_xl_cb513.h5`). |
| **Junkyard corpus** | 5 residue-shuffled copies per CB513 sequence → 2 555 embeddings. ProtT5-XL extracted via `experiments/48a_extract_junkyard.py` (~9 min on MPS). |
| **k** | 100 nearest neighbors. |
| **Codec** | `OneEmbeddingCodec()` — defaults: `d_out=896`, `quantization='binary'`, `abtt_k=0`. Fitted on real CB513 (centering only; binary mode has no codebook). |
| **Decoder** | `codec.decode_per_residue` → `(L, 896)` `float32`. |
| **Pooling** | mean (DC component) and DCT-K=4 (the codec's shipping `protein_vec` pool). |
| **RNS** | fraction of k=100 NN that are junkyard. Range [0, 1]. Random-placement expectation under this corpus = 2 555 / 3 066 = **0.833**. Lower is better. |

## 2 × 2 grid

|  | mean pool | DCT-K=4 |
|---|---|---|
| **raw ProtT5 1024d** | C1 (1024,) | C2 (4096,) |
| **OE 896d binary**   | C3 (896,)  | C4 (3584,) — codec ships this |

## Aggregate RNS

| Condition | Dim | Mean RNS | 95 % CI | Median |
|-----------|:---:|:--------:|:--------|:------:|
| C1 raw + mean       | 1024 | **0.128** | [0.103, 0.154] | 0.000 |
| C2 raw + DCT-K=4    | 4096 | 0.370    | [0.340, 0.401] | 0.340 |
| C3 OE + mean        |  896 | **0.131** | [0.106, 0.158] | 0.000 |
| C4 OE + DCT-K=4     | 3584 | 0.511    | [0.486, 0.535] | 0.540 |

CIs are 10 000-sample percentile bootstraps over the 511 query proteins.

## Per-protein paired deltas

|  Comparison | Δ (mean) | 95 % CI | frac. proteins increased | sig. |
|---|:---:|:---|:---:|:---:|
| C2 − C1 (raw, DCT vs mean)        | +0.242 | [+0.219, +0.266] | 58.3 % | * |
| C4 − C3 (OE,  DCT vs mean)        | +0.379 | [+0.358, +0.401] | 88.1 % | * |
| C3 − C1 (compression at mean)     | **+0.003** | [+0.002, +0.005] | 14.5 % | * |
| C4 − C2 (compression at DCT-K=4)  | **+0.140** | [+0.127, +0.154] | 77.7 % | * |

`*` = 95 % CI excludes zero.

## Findings

1. **Compression is essentially free at mean pool.** C3 (OE-mean) sits +0.003
   above C1 (raw-mean). Statistically detectable but practically zero — the
   codec preserves the embedding-quality signal that RNS captures, when you
   pool by mean.
2. **Compression measurably costs RNS at DCT-K=4.** C4 (OE+DCT4, the codec's
   shipping `protein_vec`) is +0.140 above C2 (raw+DCT4). This is a 38 %
   relative shift at this metric. Most of the increase is in *which* proteins
   shift (77.7 % go up vs 14.5 % at mean pool), not the magnitude per protein.
3. **The bigger effect is the pooling choice itself, not the compression.**
   C2 − C1 = +0.242 (raw, DCT4 vs mean). Even on uncompressed ProtT5, DCT-K=4
   pooling lands the protein vector in a region that looks ~3 × more random
   to RNS than mean pooling does. Compression amplifies this (C4 − C3 = +0.379)
   but the gap exists pre-compression.
4. **All four conditions are well below the random-placement RNS of 0.833.**
   Even the worst (C4 = 0.51) preserves substantial biological structure
   relative to chance.
5. **The median RNS for C1 and C3 is 0.000.** Most CB513 proteins have *zero*
   junkyard nearest neighbors at mean pool, raw or compressed. The +0.003
   mean delta is driven by a small tail of harder cases. Concretely: only
   14.5 % of proteins see *any* RNS increase under compression at mean pool
   — for the other 85.5 %, the codec is RNS-identical.

## Caveats

- Single PLM (ProtT5-XL), single dataset (CB513). The qualitative conclusion
  (compression-at-mean ≈ free; compression-at-DCT4 ≈ +0.14) should replicate
  on other PLMs but the magnitudes will move.
- Junkyard generation uses 5 shuffles per protein. Higher `n_shuffles` would
  push the random baseline up but the *paired* deltas are insensitive to this.
- The percentile-bootstrap CIs above are not BCa-adjusted. We use BCa for the
  Exp 43 / 44 / 46 / 47 retention numbers; for RNS, percentile is fine because
  the per-protein RNS distribution is bounded in [0, 1] and the second-order
  correction matters less.
- RNS uses L2 distance via FAISS `IndexFlatL2`. Switching to cosine would
  re-rank some neighbors; we have not benchmarked that variant.

## What to do with this

- **For users who want RNS as a quality metric**, recommend pooling by mean
  over decoded OE bytes (= C3). Compression cost is +0.003. Don't use the
  shipping `protein_vec` (DCT-K=4) for RNS-style checks.
- **For retrieval / clustering / UMAP**, keep using `protein_vec` as designed
  — Exp 46/47 already validate Ret@1 lossless under compression. The RNS
  shift at DCT-K=4 is a *different* metric (neighborhood randomness, not
  task accuracy).
- **For the manuscript**, this is a single paragraph in §results: "compression
  is RNS-neutral at mean pool (Δ = +0.003 [+0.002, +0.005]); the DCT-K=4
  protein vector exhibits a +0.14 RNS shift, which is dominated by the
  pooling choice itself (raw DCT-K=4 already differs by +0.24 vs raw mean)."

## Reproduce

```bash
# 1. Junkyard ProtT5 extraction (~9 min on MPS, 2.6 GB output)
uv run python experiments/48a_extract_junkyard.py

# 2. The experiment itself (~30 s)
uv run python experiments/48_rns_compression.py
```
