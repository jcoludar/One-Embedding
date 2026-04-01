# The Journey: 232 Methods in 47 Experiments

Historical record of all compression methods tested, organized by experimental phase. For current benchmark numbers, see [CLAUDE.md](../CLAUDE.md).

## Phase 1-4: Trained Compression (Experiments 1-10)
Explored attention pooling, MLP autoencoders, ChannelCompressor. Trained ChannelCompressor with contrastive fine-tuning achieved Ret@1=0.795 (d256, 3-seed mean). Requires labels and training — not universal.

## Phase 5: Universal Codec Quest (Experiments 18-24)
Pivoted to training-free codecs. Tested DCT, Haar wavelets, spectral fingerprints, path signatures, curvature, gyration tensors, Fisher vectors, kernel mean embeddings. **Key negative**: path geometry adds noise, not signal. **Key positive**: DCT K=1 === mean pool (mathematically proven); [mean|max] concat is a free +4pp retrieval boost.

## Phase 6: The Chained Codec Breakthrough (Experiments 25-26)
Discovered that chaining D-compression (RP512) + L-compression (DCT K=4) solves the fundamental tension: D-compression preserves per-residue, L-compression boosts retrieval. 14 codecs x 3 PLMs benchmarked. Best: rp512+dct_K4 -> Ret@1=0.780, SS3=0.815.

## Phase 7: Preprocessing + Quantization (Experiment 29)
ABTT3 (remove top-3 PCs) discovered as a free retrieval boost (+0.006 Ret@1). int4 quantization verified lossless for retrieval. 10-seed variance analysis: RP std=0.004. **The V1 One Embedding**: ABTT3+RP512+int4+DCT K4 -> Ret@1=0.784, SS3=0.809, 48 KB.

## Phase 8: Extreme Compression (Experiment 28)
45 methods on ProtT5: wavelets, CUR decomposition, channel pruning, product quantization, residual VQ, tensor train, NMF, SimHash. **All on raw 1024d**. PQ M=64 best at 0.701. Key insight missed at the time: should have tested on preprocessed space.

## Phase 9: V2 Codec — The Preprocessed Space Changes Everything (Experiments 31-34)
Re-tested all compression on ABTT3+RP512 (decorrelated, isotropic). Results dramatically better:

- **Binary (1-bit) beats int4 for retrieval** (0.787 vs 0.784) — RaBitQ effect confirmed
- **PQ M=128 matches V1 quality at ~50% less storage** (23 vs 48 KB)
- **PQ M=64 at 22 KB retains 93% SS3 quality**
- **Pure VQ fails in 512d** — even K=16384 caps at 0.621 Ret@1
- **RVQ fails in 512d** — residual norms barely decrease between levels
- **Hybrid VQ+PQ** works but doesn't beat pure PQ at same size
- **OPQ (learned rotation)** doesn't help — RP already decorrelates

## Phase 10: Rigorous Validation (Experiment 43)
Built a rule-enforced benchmark framework (14 golden rules) to honestly validate the 1.0 codec. Fixed methodological issues from Exp 41 (unfair retrieval baseline, no CIs, hardcoded hyperparameters). Key results:

- **Phase A**: Fixed 5-task benchmark — corrected mean retention from "100.1%" to honest 98.1% with BCa bootstrap CIs
- **Phase B**: Cross-validated SS3/SS8 on 3 independent test sets (CB513, TS115, CASP12) — max 1.2pp divergence. ESM2 multi-PLM validation (95.8% SS3 at 1280->768, retrieval lossless)
- **Phase C**: CATH20 superfamily retrieval (9518 domains): **100.0% cosine retention** [99.8, 100.2]. DeepLoc localization: 99.5% retention. 27K+ protein embeddings extracted
- **Phase D**: Ablation — ABTT contributes +0.6pp retrieval, RP costs -0.3pp SS3, fp16 is 0.0pp (lossless). No length-dependent degradation
- **ABTT Stability**: Cross-corpus stability test — PCs differ across corpora (subspace similarity 0.18-0.71) but downstream Ret@1 varies by only 0.20pp. ABTT fitting corpus choice is irrelevant for performance.

797 tests, 12+ tasks, 8+ datasets, 5 PLMs. BCa CIs on everything.

### V2 Extreme Compression — Rigorous Re-validation (512d, Exp 43)
Replaced Exp 34 results (hardcoded C=1.0 probes) with Exp 43 framework (CV-tuned, BCa CIs, pooled disorder ρ). All 5 V2 modes on 512d benchmarked. Key finding: retrieval is **lossless across all modes** (100.2%). Binary matches full for disorder (90.0%) — RaBitQ effect.

### Phase 11: Unified Codec + 768d Quantization Sweep (Experiment 44)
Merged V1 (fp16) and V2 (quantized) into single `OneEmbeddingCodec` with three knobs: `d_out`, `quantization`, `pq_m`. First-ever benchmarks of PQ/int4/binary on 768d:

| Config | Size (L=175) | Compression | SS3 Ret | Dis Ret | Ret Ret |
|--------|:----------:|:-----------:|:-------:|:-------:|:-------:|
| lossless (1024d fp16) | 366 KB | 2x | 100.0 ± 0.2% | 99.9 ± 0.1% | 100.4 ± 0.5% |
| int4 (768d) | 67 KB | 10x | 99.2 ± 0.6% | 94.8 ± 2.2% | 100.2 ± 0.6% |
| **PQ M=192 (default)** | **34 KB** | **20x** | **98.8 ± 0.5%** | **92.8 ± 2.7%** | **100.2 ± 0.6%** |
| PQ M=128 | 23 KB | 30x | 97.1 ± 0.6% | 90.6 ± 2.9% | 100.2 ± 0.6% |
| binary (768d) | 17 KB | 41x | 95.9 ± 0.7% | 92.5 ± 2.7% | 100.2 ± 0.6% |

Key findings: int4 is indistinguishable from fp16 at 10x compression. Lossless (1024d, no RP) truly achieves ~100% on everything including disorder. Binary still beats PQ M=128 on disorder (RaBitQ on 768d). Default was PQ M=192 (~20x) — later changed to 896d PQ M=224 (~18x) after Exp 45 findings.

### Phase 12: Disorder Information Loss Forensics (Experiment 45)
Systematic investigation into why disorder retention (93-95%) lags behind all other tasks (97-100%).

**Key finding: ABTT PC1 is 73% aligned with the disorder prediction direction.** ABTT removes the primary channel through which a Ridge probe predicts disorder Z-scores. Per-stage decomposition: ABTT causes 50% of total disorder loss, RP 26%, PQ 24%.

Dropping ABTT recovers +3.3% disorder retention at 0.6pp retrieval cost. New default: center + RP 896d + PQ M=224 (~18x), `abtt_k=0`. ABTT still available via parameter for retrieval-focused use cases.

VQ/RVQ retested with correct centering — confirmed genuinely bad (SS3 79%, Dis 59% at best). The ABTT centering bug did not cause their failure.

### Phase 13: Multi-PLM Pipeline + 5-PLM Benchmark (Experiments 46-47)
Built registry-based pipeline (`46_multi_plm_benchmark.py`) that extracts and benchmarks any PLM with one command. 5 extractors: ProtT5, ESM2 (fair-esm), ESM-C (EvolutionaryScale API), ProstT5 (T5), ANKH (T5 raw tokenizer).

Extracted 30 embedding files across 5 PLMs × 6 datasets (~74 GB total).

Full rigorous benchmarks with BCa CIs on all 5 PLMs:

| PLM | SS3 raw | SS3 ret | Ret@1 raw | Dis ret |
|-----|:-------:|:-------:|:---------:|:-------:|
| ProstT5 | **0.869** | **99.2%** | 0.705 | 98.3% |
| ProtT5-XL | 0.840 | 99.0% | **0.794** | 95.4% |
| ESM-C 600M | 0.846 | 98.3% | 0.343 | 98.1% |
| ANKH-large | 0.861 | 97.9% | 0.708 | 94.8% |
| ESM2-650M | 0.836 | 97.6% | 0.589 | **98.8%** |

Codec sweep (Exp 47) on ProtT5: 6 standard tiers (lossless to binary) + VQ/RVQ retest + old default comparison. All verified.

---

## Idea Space: What's Exhausted, What Remains

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
Theoretical floor: ~5-7 KB per protein (from intrinsic dimensionality ~80 x effective positions ~35). V2 `balanced` at 26 KB is ~4x above this. The gap is reconstruction precision: 128 sub-spaces x 256 centroids x 4d per sub-vector gives only 4 float32 values of freedom per sub-space, limiting per-residue fidelity.

---

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
