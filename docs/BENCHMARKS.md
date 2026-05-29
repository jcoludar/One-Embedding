# Benchmarks

Full benchmark tables for the Protein Embedding Codec. Operational summary lives in `CLAUDE.md`; this file is the detailed record (moved out of `CLAUDE.md` to keep the assembled context lean). Methodology notes at the bottom.

## Exp 44 sweep (legacy 768d codec)

Numbers below are from the **earlier d_out=768 codec sweep** (Exp 44). The current default is d_out=896 — see the **Exp 47 codec sweep** table below for the shipping numbers. Exp 44 retained because its 6-config × 4-task grid is the densest single-PLM measurement we have; it informs design choices but is not the cited final.

`Size (L=175)` columns use a fixed reference protein length L=175 (not the empirical mean — Exp 45 reports SCOPe-5K mean L=156).

| Config | Quantization | Size (L=175 ref) | Compression | SS3 Ret | SS8 Ret | Dis Ret | Ret Ret |
|--------|-------------|:----------:|:-----------:|:-------:|:-------:|:-------:|:-------:|
| lossless (1024d) | fp16 | 366 KB | 2x | 100.0 ± 0.2% | 100.0 ± 0.3% | 99.9 ± 0.1% | 100.4 ± 0.5% |
| fp16 (768d) | fp16 | 275 KB | 2.7x | 99.1 ± 0.5% | 98.7 ± 0.6% | 95.0 ± 2.1% | 100.2 ± 0.6% |
| int4 (768d) | int4 | 67 KB | 10x | 99.2 ± 0.6% | 98.6 ± 0.6% | 94.8 ± 2.2% | 100.2 ± 0.6% |
| PQ M=192 (768d) | PQ | 34 KB | 20x | 98.8 ± 0.5% | 97.6 ± 0.8% | 92.8 ± 2.7% | 100.2 ± 0.6% |
| PQ M=128 (768d) | PQ | 23 KB | 30x | 97.1 ± 0.6% | 95.3 ± 0.8% | 90.6 ± 2.9% | 100.2 ± 0.6% |
| binary (768d) | 1-bit sign | 17 KB | 41x | 95.9 ± 0.7% | 93.6 ± 1.0% | 92.5 ± 2.7% | 100.2 ± 0.6% |

All Exp 44, rigorous (BCa CIs, CV-tuned probes, paired bootstrap retention, pooled disorder ρ). Retrieval **lossless across all configs** (100.2 ± 0.6%). int4 is indistinguishable from fp16 at 10x compression. Binary beats PQ M=128 on disorder (92.5% vs 90.6%) — RaBitQ effect.

## Rigorous Retention Benchmarks (Exp 43, 768d vs raw ProtT5)

All numbers include 95% BCa bootstrap CIs (DiCiccio & Efron 1996, second-order accurate). Probes are CV-tuned (GridSearchCV on C/alpha, not hardcoded). Predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021). Retrieval uses fair baselines (same DCT K=4 pooling for raw and compressed). ABTT fitted on external SCOPe 5K corpus (cross-corpus stability verified: Ret@1 varies < 0.2pp across 4 fitting corpora).

### Per-Residue Tasks (linear probe, BCa bootstrap CI, 3-seed averaged)

| Task | Level | Dataset (n) | Raw ProtT5 1024d | One Embedding 768d | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| SS3 (Q3) | per-residue | CB513 (103) | 0.840 [0.823, 0.852] | 0.833 [0.818, 0.845] | **99.1 ± 0.6%** |
| SS3 (Q3) | per-residue | TS115 (115) | 0.841 [0.829, 0.853] | 0.828 [0.816, 0.839] | **98.4 ± 0.5%** |
| SS3 (Q3) | per-residue | CASP12 (20) | 0.781 [0.748, 0.810] | 0.765 [0.730, 0.797] | **98.0 ± 1.2%** |
| SS8 (Q8) | per-residue | CB513 (103) | 0.716 [0.697, 0.734] | 0.707 [0.689, 0.725] | **98.8 ± 0.6%** |
| SS8 (Q8) | per-residue | TS115 (115) | 0.732 [0.715, 0.748] | 0.717 [0.701, 0.733] | **98.0 ± 0.7%** |
| SS8 (Q8) | per-residue | CASP12 (20) | 0.662 [0.629, 0.695] | 0.647 [0.611, 0.682] | **97.6 ± 1.7%** |
| Disorder (pooled ρ) | per-residue | CheZOD117 (117) | 0.663 [0.585, 0.723] | 0.629 [0.548, 0.691] | **94.9 ± 2.0%** |
| Disorder (pooled ρ) | per-residue | TriZOD348 (348) | 0.506 [0.461, 0.566] | 0.471 [0.426, 0.533] | **93.0 ± 2.6%** |
| Disorder (AUC-ROC) | per-residue | CheZOD117 (117) | 0.890 [0.836, 0.922] | 0.877 [0.826, 0.909] | **98.5%** |

### Variant Effect Prediction (Exp 55, supervised Ridge probe + zero-shot ClinVar AUC)

| Task | Level | Dataset (n_assays / n_variants) | Raw ProtT5 1024d | One Embedding 896d (binary, 37×) | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| DMS Spearman ρ (mean) | per-protein | ProteinGym diversity (15 / 37,919) | 0.645 | 0.640 | **99.2 ± 0.8%** |
| ClinVar AUC (zero-shot) | per-variant | ProteinGym clinical ≤500 aa (1,016 / 15,252) | 0.602 | **0.605** | **100.5%** |

3-seed averaged Ridge probe (5-fold outer CV, inner 3-fold GridSearch on α). BCa B=10,000 paired ratio-of-means bootstrap. Binary 896d retention is statistically indistinguishable from PQ M=224 (CIs overlap heavily) — binary is the recommended VEP tier. Per-residue mutational sensitivity survives 1-bit-per-dim quantization, in contrast to disorder (94.9% binary retention) which has a real ~5pp gap. See `docs/exp55_vep_retention.md` for the per-assay breakdown and methodological discussion.

Disorder uses **pooled residue-level** Spearman ρ (matching SETH/ODiNPred/ADOPT/UdonPred standard) with cluster bootstrap CIs (resample proteins, recompute pooled statistic — Davison & Hinkley 1997). AUC-ROC computed on binary Z<8 threshold (CAID standard).

CIs on raw and compressed **overlap** for all tasks — no statistically significant difference detected. Cross-dataset consistency: SS3 max 1.1pp, SS8 max 1.2pp (both OK < 3pp threshold).

### Protein-Level Tasks (cosine kNN / LogReg, paired bootstrap CI on retention)

| Task | Level | Dataset (n) | Raw ProtT5 1024d | One Embedding 768d | Retention |
|------|-------|-------------|:----------------:|:------------------:|:---------:|
| Family Ret@1 | per-protein | SCOPe 5K (2493) | 0.799 [0.783, 0.815] | 0.798 [0.782, 0.814] | **99.8 ± 0.4%** |
| Superfamily Ret@1 | per-protein | CATH20 (9518) | 0.841 [0.834, 0.849] | 0.841 [0.834, 0.849] | **100.0 ± 0.2%** |
| Localization (Q10) | per-protein | DeepLoc test (2768) | 0.810 [0.795, 0.824] | 0.806 [0.791, 0.820] | **99.5 ± 0.9%** |
| Localization (Q10) | per-protein | DeepLoc setHARD (490) | 0.608 [0.563, 0.651] | 0.606 [0.563, 0.651] | **99.7 ± 3.1%** |

### ESM2 Multi-PLM Validation (1280d → 768d, 40% compression)

| Task | Raw ESM2 1280d | One Embedding 768d | Retention |
|------|:--------------:|:------------------:|:---------:|
| SS3 (Q3) | 0.836 [0.817, 0.851] | 0.801 [0.784, 0.816] | **95.8 ± 1.0%** |
| SS8 (Q8) | 0.715 [0.695, 0.734] | 0.684 [0.664, 0.703] | **95.7 ± 1.1%** |
| Ret@1 cosine | 0.675 | 0.675 | **100.0 ± 0.5%** |

### Ablation: Component Contributions (Exp 43 Phase D)

| Condition | SS3 Q3 | Δ vs raw | Ret@1 cos | Δ vs raw |
|-----------|:------:|:--------:|:---------:|:--------:|
| Raw 1024d | 0.840 | baseline | 0.794 | baseline |
| + ABTT3 only | 0.841 | +0.1pp | 0.799 | +0.6pp |
| + RP768 only | 0.837 | −0.3pp | 0.793 | −0.0pp |
| + ABTT3 + RP768 | 0.833 | −0.7pp | 0.798 | +0.4pp |
| + ABTT3 + RP768 + fp16 | 0.833 | −0.7pp | 0.798 | +0.4pp |

fp16 quantization: **0.0pp** effect (completely lossless). Length stress test: no degradation (short 99.8%, medium 100.7%, long 101.3%).

### Legacy benchmarks (Exp 37, 512d codec — pre-rigorous, no BCa CIs)

| Metric | Retention | Source | Note |
|---|:---:|---|---|
| Structural lDDT | 100.7% | Exp 37 | pre-rigorous; not re-validated through `metrics.statistics` |
| Contact precision | 106.5% | Exp 37 | same |
| TM-score Spearman | **57.4%** | Exp 37 (`structural_retention_results.json`) | same; **lower than lDDT/contact**, disclosed for completeness |

## Multi-PLM validation (Exp 46, center + RP896 + PQ224, ~18x)

| PLM | dim | SS3 ret | SS8 ret | Ret@1 ret | Dis ret |
|-----|:---:|:-------:|:-------:|:---------:|:-------:|
| ProstT5 | 1024 | 99.2±0.3% | 98.6±0.5% | 100.0±0.5% | 98.3±1.1% |
| ProtT5-XL | 1024 | 99.0±0.5% | 98.5±0.6% | 100.6±0.6% | 95.4±1.9% |
| ESM-C 600M | 1152 | 98.3±0.5% | 97.6±0.7% | 102.6±2.9% | 98.1±1.0% |
| ANKH-large | 1536 | 97.9±0.5% | 96.3±0.8% | 99.9±0.6% | 94.8±2.3% |
| ESM2-650M | 1280 | 97.6±0.7% | 96.5±0.7% | 97.8±1.6% | 98.8±0.9% |

## VEP codec mega-sweep (Exp 56, ProtT5, paired BCa B=10,000)

| Config | Compression | DMS retention | ClinVar AUC | Note |
|--------|:-----------:|:-------------:|:-----------:|------|
| Lossless 1024d | 2× | 100.0% | 0.602 | baseline |
| **Binary 1024 (no RP)** | **32×** | **100.7% [99.5, 103.2]** | 0.598 | best DMS retention in sweep |
| **int2 896d** | **18×** | **99.9% [99.2, 100.7]** | 0.530 | clean DMS, ClinVar collapses (probe-only tier) |
| binary 896 + ABTT8 | 37× | 99.2% [98.2, 100.2] | 0.598 | ABTT essentially neutral on VEP |
| binary 896 + ABTT3 | 37× | 99.0% [98.1, 99.7] | 0.588 | falsifies "ABTT-3 destroys signal" outside disorder |
| binary_magnitude 896 | ~30× | 99.6% [98.9, 100.5] | 0.605 | Exp 51 PolarQuant rehabilitated for VEP |
| pq128 896d | 32× | 98.9% [98.4, 99.4] | 0.588 | |
| **binary 512d** | **64×** | 98.1% [96.8, 100.0] | **0.609** | wins ClinVar AUC (RP isotropy effect) |
| pq64 896d | 64× | 97.6% [96.7, 98.6] | 0.581 | most aggressive DMS-OK setting |

Headlines: ABTT-k is essentially free for VEP at all tested k (≠ disorder); RP — not quantization — is binary's main loss source on DMS; int2 is a real DMS tier but supervised-only; PQ M=64 retains 97.6% at 64× compression. See `docs/exp56_vep_codec_megasweep.md` for the full per-axis breakdown and the per-arm ClinVar table.

## Codec sweep (Exp 47, ProtT5, standard tiers)

| Config | Compression | SS3 ret | SS8 ret | Ret@1 ret | Dis ret |
|--------|:-----------:|:-------:|:-------:|:---------:|:-------:|
| lossless 1024d | 2x | 100.2% | 100.0% | 100.4% | 100.0% |
| fp16 896d | 2.3x | 100.0% | 99.2% | 100.6% | 98.6% |
| int4 896d | 9x | 99.8% | 98.8% | 100.4% | 98.2% |
| **PQ M=224 896d** | **18x** | **99.0%** | **98.5%** | **100.6%** | **95.4%** |
| PQ M=128 896d | 32x | 97.5% | 96.1% | 100.1% | 91.4% |
| binary 896d | 37x | 97.6% | 95.0% | 100.4% | 94.9% |

VQ/RVQ confirmed genuinely poor (not bug-caused): VQ K=16384 gets 79% SS3 ret, 58% Dis ret.

## Methodology (Nature-level, Exp 43)

- Bootstrap: BCa (DiCiccio & Efron 1996), B=10,000, percentile fallback for n<25
- Multi-seed: predictions averaged across 3 seeds before bootstrapping (Bouthillier et al. 2021)
- Disorder: pooled residue-level Spearman ρ (SETH/CAID standard) with cluster bootstrap (Davison & Hinkley 1997)
- Retrieval: 3 fair baselines (raw+mean, raw+DCT, raw+ABTT+DCT). Retention = compressed / baseline C
- Probes: CV-tuned (GridSearchCV on train set, 3-fold, C/alpha grids)
- ABTT leakage: formally tested — PCs differ across corpora but downstream Ret@1 varies <0.2pp (irrelevant)

878 tests, 5 task families (SS3 / SS8 / retrieval / disorder / VEP), 10 datasets (CB513, TS115, CASP12, CheZOD117, TriZOD348, SCOPe 5K, CATH20, DeepLoc test, DeepLoc setHARD, ProteinGym DMS+ClinVar), 5 PLMs. BCa CIs on everything.

See [docs/EXPERIMENTS.md](EXPERIMENTS.md) for the full 200+ method journey across 49 experiments.
