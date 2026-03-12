# Experiment 25: Universal Codec Benchmark

**Date:** 2026-03-12
**Status:** Approved

## Goal

Benchmark every implemented training-free codec method across all 3 PLMs (ProtT5-XL, ESM2-650M, ESM-C 300M) on the full task suite: retrieval, hierarchy, biology, and per-residue probes. Produce a single unified comparison table.

## Codecs (14 methods)

| # | Name | Code | Output | Per-Residue |
|---|------|------|--------|-------------|
| 1 | mean_pool | `matrix.mean(axis=0)` | (D,) | No |
| 2 | max_pool | `matrix.max(axis=0)` | (D,) | No |
| 3 | mean_max | `[mean \| max]` concat | (2D,) | No |
| 4 | power_mean_p3 | `power_mean_pool(p=3)` | (D,) | No |
| 5 | norm_weighted | `norm_weighted_mean()` | (D,) | No |
| 6 | kernel_mean | `kernel_mean_embedding(D_out=2048)` | (2048,) | No |
| 7 | svd_spectrum | `svd_spectrum(k=16)` | (16,) | No |
| 8 | dct_K4 | `dct_summary(K=4)` mean of K=1 coeffs | (4D,) per-protein; invertible | Yes (inverse_dct) |
| 9 | haar_L3 | `haar_summary(levels=3)` | (4D,) | Yes (lossless via haar_full_coefficients) |
| 10 | feature_hash_512 | `feature_hash(d_out=512)` + mean | (512,) per-protein; (L,512) per-residue | Yes |
| 11 | random_proj_512 | `random_orthogonal_project(d_out=512)` + mean | (512,) per-protein; (L,512) per-residue | Yes |
| 12 | hrr_K1 | `hrr_encode()` | (D,) | Yes (hrr_decode) |
| 13 | hrr_K8 | `hrr_kslot_encode(K=8)` flat | (8D,) | Yes (hrr_kslot_decode) |
| 14 | mean_max_euclidean | Same as #3 but evaluated with Euclidean metric | (2D,) | No |

## PLMs

| PLM | Dim | H5 stem |
|-----|-----|---------|
| ProtT5-XL | 1024 | prot_t5_xl |
| ESM2-650M | 1280 | esm2_650m |
| ESM-C 300M | 960 | esmc_300m |

## Tasks

### Per-protein (all 14 codecs × 3 PLMs)
- **Retrieval**: family Ret@1, SF Ret@1, fold Ret@1 (cosine). #14 uses euclidean.
- **Hierarchy**: separation ratio (cosine).
- **Biology**: GO Spearman rho, EC full Ret@1, Pfam Ret@1, taxonomy Spearman rho.

### Per-residue (5 codecs × 3 PLMs)
Codecs: feature_hash_512, random_proj_512, hrr_K1, hrr_K8, dct_K4_inverse.

- **SS3** (CB513): Q3 accuracy
- **SS8** (CB513): Q8 accuracy
- **Disorder** (SETH): Spearman rho
- **TM topology** (TMbed): macro F1
- **SignalP** (SignalP6): macro F1 (ESM-C only has dedicated H5; ProtT5/ESM2 use validation H5)

## Architecture

Single script: `experiments/25_universal_codec_benchmark.py`

```
Steps:
  C1: Per-protein retrieval (14 codecs × 3 PLMs × cosine; #14 euclidean)
  C2: Hierarchy (14 codecs × 3 PLMs)
  C3: Biology (14 codecs × 3 PLMs × GO/EC/Pfam/taxonomy)
  C4: Per-residue probes (5 codecs × 3 PLMs × SS3/SS8/disorder/TM/SignalP)
  C5: Aggregator table + JSON
```

Each step:
1. Loads raw per-residue embeddings once per PLM
2. Applies codec transform to get per-protein vectors (or per-residue matrices)
3. Runs evaluation
4. Saves incrementally to `data/benchmarks/universal_codec_results.json`

## Codec Application Details

### Per-protein vector generation
For retrieval/hierarchy/biology, each codec produces a fixed-size vector per protein:
- mean_pool: `emb.mean(axis=0)`
- max_pool: `emb.max(axis=0)`
- mean_max: `np.concatenate([mean, max])`
- dct_K4: Use only K=1 coefficients (= scaled mean) for per-protein. Full K=4 for per-residue inverse.
- haar_L3: `haar_summary(levels=3)` — (4D,) vector
- feature_hash_512: `feature_hash(emb, 512).mean(axis=0)` — (512,)
- random_proj_512: `random_orthogonal_project(emb, 512).mean(axis=0)` — (512,)
- hrr_K1: `hrr_encode(emb)` — (D,)
- hrr_K8: `hrr_kslot_encode(emb, K=8).ravel()` — (8D,)

### Per-residue reconstruction
For SS3/SS8/disorder/TM/SignalP probes:
- feature_hash_512: `feature_hash(emb, 512)` → (L, 512)
- random_proj_512: `random_orthogonal_project(emb, 512)` → (L, 512)
- hrr_K1: `hrr_decode(trace, L)` → (L, D) approximate
- hrr_K8: `hrr_kslot_decode(slots, L)` → (L, D) approximate
- dct_K4: `inverse_dct(dct_summary(emb, K=4), D, L)` → (L, D) approximate

## Output

- Results JSON: `data/benchmarks/universal_codec_results.json`
- Summary table printed to stdout (markdown)
- Per-PLM radar plots: `data/plots/exp25/`
