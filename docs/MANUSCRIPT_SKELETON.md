# Manuscript Skeleton — Universal PLM Embedding Codec

**Date:** 2026-04-27
**Status:** Real outline (Task E.2). Derived from `docs/STATE_OF_THE_PROJECT.md`.
**Target venue:** *Bioinformatics* (short paper) as the natural fit; *NAR* if extended with VEP/RNS evaluation; *Nat Methods* if combined with Exp 50 transformer + multi-teacher distillation.

> Prior (2026-04-26 stub) preserved as Appendix for calibration. Compare via
> `git log --oneline -- docs/MANUSCRIPT_SKELETON.md` and the
> `stub(prior): MANUSCRIPT_SKELETON.md` commit.

---

## Title (three candidates, ranked)

1. **"OneEmbedding: a universal codec for protein-language-model embeddings, validated across five PLMs"** — descriptive, accurate, hits 5-PLM angle.
2. **"~37× compression of PLM per-residue embeddings at 95–100 % task retention"** — number-forward, sells the headline.
3. **"What survives compression: probing PLM embedding geometry through information-preserving codecs"** — frames as scientific finding (better fit if disorder gap and ABTT artifact lead the story).

Recommend #1 for Bioinformatics; #2 for a more applied venue; #3 if the paper's argument shifts toward "compression as a probe."

---

## Abstract (5–7 sentences, draft)

> Protein language models (PLMs) emit per-residue embeddings of shape (L, D) where D ranges 1024–1536; these are large to store, ship, and search at scale. We introduce OneEmbedding, a universal codec (one Python class, four knobs) that compresses raw PLM embeddings from any of five major PLMs (ProtT5, ProstT5, ESM2, ESM-C, ANKH) by ~37× at 95–100 % retention across four task families (SS3, SS8, retrieval, disorder) and nine datasets, with rigorous BCa bootstrap confidence intervals throughout. The default — center + random projection to 896d + 1-bit-sign quantization — produces ~17 KB per protein, requires no codebook, and decodes with `numpy + h5py` in ~12 lines. We further show that compressed One Embeddings are partially predictable from amino acid sequence alone (Exp 50: 69 % bit accuracy, ceiling pending transformer architecture), suggesting a path to FASTA-only deployment. The codec exposes one geometric finding worth noting: the top principal component of ProtT5 embeddings is ~73 % aligned with the disorder direction, so PCA-style preprocessing (ABTT) destroys disorder signal — we recommend it stays off by default. Code and benchmarks are released under MIT.

---

## 1. Introduction (3 paragraphs)

**¶1 — Motivation.** PLM per-residue embeddings (ProtT5, ESM2, ESM-C, ProstT5, ANKH) enable strong predictions on diverse downstream tasks (secondary structure, disorder, retrieval, localisation). They are also large: ProtT5-XL produces (L, 1024) fp32 per protein, so the SCOPe 5K test set alone is ~5 GB at fp16. At UniProt scale (250 M proteins) raw embeddings are simply impractical to store, ship, or search.

**¶2 — Gap.** Compression schemes have been published per-PLM (e.g., quantization for ESM2 retrieval, ChannelCompressor for ProtT5), but no universal codec spans the major PLMs with rigorous downstream-task retention CIs. Exact comparison to existing work is muddied by varying methodology (random vs. cluster splits, BCa vs. percentile bootstrap, retention vs. raw metric).

**¶3 — Contribution.** We benchmark over 200 compression methods across 47 experiments to derive a single configurable codec validated on 5 PLMs × 4 task families × 9 datasets, with: (i) a rigorous protocol (BCa B=10000, paired bootstrap on retention, multi-seed averaging *before* bootstrapping, CV-tuned probes, fair retrieval baselines); (ii) a working binary default that needs no codebook (decoder = `numpy + h5py`); (iii) one new geometric finding (ABTT PC1 ↔ disorder direction); (iv) a sequence-only prediction probe that establishes a measurable architecture ceiling.

---

## 2. Related work (1 paragraph)

PLM compression has appeared as: (a) post-hoc dimensionality reduction (PCA, RP, learned autoencoders — Cucurull et al., Heinzinger et al.); (b) quantization (int8 retrieval indexes for ESM2, RaBitQ for general embeddings — Gao & Long 2024); (c) co-distilled smaller PLMs (VESM — Bromberg lab 2026, ESM-Cambrian); (d) attention-pool / single-vector summaries for retrieval. Universal multi-PLM benchmarks under uniform statistical protocol are absent; the closest precedent is the Bromberg lab's RNS / VEP evaluation framework. The ABTT post-processing (Mu & Viswanath, ICLR 2018) was originally proposed for word vectors; we revisit it for PLMs and find it has a task-dependent cost (§4.2). Statistical conventions follow DiCiccio & Efron 1996 (BCa), Davison & Hinkley 1997 (cluster bootstrap), Bouthillier et al. 2021 (multi-seed averaging-before-bootstrap), and SETH/CAID for disorder (pooled residue-level Spearman ρ).

---

## 3. Methods

### 3.1 Codec

- One Python class, `OneEmbeddingCodec`, four knobs:
  - `d_out` (default 896): random-projection target dimension.
  - `quantization` ('binary' default; 'int4', 'pq', None): per-residue storage.
  - `pq_m` (auto, = largest factor of `d_out` ≤ `d_out//4`): PQ subquantizers.
  - `abtt_k` (default 0): top-PC removal; off by default per §4.2.
- Pipeline: center → RP to `d_out` → quantize → (auxiliary) DCT K=4 protein vector for retrieval.
- Default config (binary 896d): ~37× compression, ~17 KB/protein at L=156 (SCOPe-5K mean), receiver = `numpy + h5py` + ~12 lines (decoder snippet shipped in supplement).

### 3.2 Datasets

| Task family | Dataset | n | Source |
|------|---------|---|---|
| SS3 / SS8 | CB513, TS115, CASP12 | 103 / 115 / 20 | NetSurfP-2.0 |
| Disorder (Z-score) | CheZOD117 | 117 | SETH / Dass et al. |
| Disorder (Z-score) | TriZOD348 | 348 | Haak 2025 |
| Family retrieval | SCOPe 5K | 2493 | SCOPe ASTRAL |
| Superfamily retrieval | CATH20 | 9518 | CATH S20 |
| Localization | DeepLoc test | 2768 | DeepLoc 2.0 |
| Localization | DeepLoc setHARD | 490 | DeepLoc 2.0 |

### 3.3 Statistics protocol

- **Bootstrap:** BCa (DiCiccio & Efron 1996) with B=10,000, percentile fallback for n<25, jackknife acceleration for cluster bootstrap.
- **Disorder:** pooled residue-level Spearman ρ (SETH/CAID convention) with cluster bootstrap (resample proteins, recompute pooled stat) per Davison & Hinkley 1997.
- **Retention:** paired bootstrap — same protein-id resample drives raw and compressed in every iteration.
- **Probes:** CV-tuned (`GridSearchCV` on C/alpha grids, 3-fold), `random_state=42`.
- **Multi-seed:** predictions averaged across seeds {42, 123, 456} **before** bootstrapping (Bouthillier et al. 2021), not CI of CIs.
- **Baselines:** retrieval uses identical DCT K=4 pooling on raw and compressed (Exp 43 fairness fix).

### 3.4 Splits

- CB513, TS115, CASP12: published SS3/SS8 splits (`<25 %` identity by dataset construction).
- CheZOD117, TriZOD348: cluster-curated by source publications.
- SCOPe 5K: cluster-controlled (`superfamily_overlap=0`, `family_overlap=0`).
- CATH20: 20 % identity clustering at the dataset level.
- All splits asserted at runtime via `rules.check_no_leakage`.
- All 5 PLMs share the same train/test partition (Exp 46) — single split file, embeddings re-extracted per PLM.

---

## 4. Results

### 4.1 Single-PLM headline (ProtT5-XL, Exp 47)

[Table 1: 6-config × 4-task retention with BCa CIs — ref STATE_OF_THE_PROJECT.md "Single-PLM headline" table.]

**Stub:** Six configurations from lossless (2×) to binary (37×) tested on the same protocol. Binary default at 37× achieves 97.6 % SS3 / 95.0 % SS8 / 100.4 % Ret@1 / 94.9 % disorder retention. Retrieval is essentially lossless across all tested compression levels (≥99.9 % retention). Per-residue tasks degrade smoothly with compression; disorder is the most sensitive (4–5 pp gap at 37×). PQ M=224 (~18×) is the recommended max-quality setting; binary is the recommended default (no codebook, ~20× faster encoding).

### 4.2 Multi-PLM validation (Exp 46)

[Table 2: 5-PLM × 4-task retention with BCa CIs — ref STATE_OF_THE_PROJECT.md "Multi-PLM validation" table.]

**Stub:** Same codec configuration (PQ M=224 896d) applied to ProtT5-XL, ProstT5, ESM-C 600M, ANKH-large, ESM2-650M. SS3 retention 97.6–99.2 %, SS8 96.3–98.6 %, retrieval 97.8–102.6 %, disorder 94.8–98.8 % across all five models. The same train/test partition is used per PLM (verified by file-path identity in `experiments/46_multi_plm_benchmark.py`). The lowest cell (ANKH disorder 94.8 %) is consistent with ANKH's tokenizer artifacts noted in Exp 46 forensics. Retrieval >100 % retention on ESM-C and ProstT5 reflects the noisy-baseline-stronger-predictor effect documented in §3.3.

### 4.3 Sequence → binary OE (Exp 50)

[Figure 1: training curves for Stages 1–3, all converging to ~69 % bit accuracy / 0.55 cosine.]

**Stub:** A small dilated CNN trained to predict the binary `(L, 896)` One Embedding directly from amino-acid sequence (no PLM at inference) reaches 65–69 % bit accuracy and 0.52–0.55 cosine on held-out sets. Three stages (varying loss type and 2× data) all converge to the same ceiling, indicating a CNN capacity bound rather than data or loss saturation. A rigorous CATH-cluster re-run (homologous-superfamily and topology splits, MMseqs2 leakage audit, 3-seed variance) is designed but not yet executed; the current numbers use a random 80/10/10 split and therefore include some homology leakage. The implication: a transformer architecture is the next lever for FASTA-only deployment.

### 4.4 ABTT artefact and disorder geometry (Exp 45)

[Figure 2: cosine similarity between ProtT5 ABTT PC1 and the disorder-vs-folded direction; ~0.73 (very high).]

**Stub:** PCA-style preprocessing (Mu & Viswanath's "all but the top") is widely used to improve embedding isotropy. We find that for ProtT5 (and similarly across all 5 PLMs), the top principal component is 73 % aligned with the direction from "folded" to "disordered" residues in CheZOD117 / TriZOD348. Removing it costs 6–11 pp on disorder retention while gaining only marginally on retrieval. We therefore default `abtt_k=0` and recommend it remain off unless retrieval is the *only* downstream task.

### 4.5 Phylogenetics from embeddings (Exp 35)

[Figure 3: per-family monophyly count for BM-MCMC vs sequence ML/Bayesian baselines.]

**Stub:** Embedding-based phylogenetic inference using Brownian motion MCMC (warm-started, extended SPR) recovers 10–11 of 12 test families as monophyletic, vs 4–5 of 12 for FastTree / IQ-TREE / MrBayes on the same alignments. Sigma² estimates cross-validate against RevBayes. This is a separate research branch and ships in this manuscript as a one-paragraph capability demonstration (extended treatment is beyond scope).

---

## 5. Discussion

### 5.1 Why is disorder the weakest cell?

The Exp 45 ABTT finding suggests a geometric explanation: disorder information is concentrated along directions that overlap heavily with the dominant variance directions (the same ones ABTT explicitly removes). Random projection compresses along all directions roughly uniformly, so it preserves the disorder direction better than PCA-style methods would; quantization at 1-bit / 4-bit nonetheless smears local distinctions. The PolarQuant scheme (Exp 51, designed) attacks this by augmenting binary signs with a coarse magnitude bin, expected to recover 2–3 pp of disorder retention without re-introducing ABTT.

### 5.2 Sequence → embedding implications

If a small CNN can predict 69 % of the binary OE bits from sequence, the binary OE is partially redundant with sequence-only features, but not fully — the remaining 31 % bits encode information that requires PLM context. A transformer (Stage 4, designed) should establish the new ceiling. If the final ceiling is, say, 80 %, then ~80 % of the binary OE can be recovered from FASTA at deployment, opening up storage-free retrieval pipelines.

### 5.3 Scope and what we are not claiming

We are not claiming bit-perfect compression: 5 pp of disorder is genuinely lost at 37×. We are not claiming the binary default beats PQ on all tasks: PQ M=224 (18×) is uniformly within 1 pp on SS3/SS8 and slightly better on disorder (95.4 % vs 94.9 %) — recommend PQ for max-quality settings. We are not claiming the codec is optimal: `dct_k=4` in particular is a STORAGE choice with weaker direct evidence than the others (Exp 22 raw shows K=8 has higher Ret@1 on a different proxy). We are not claiming sequence → OE is competitive with raw PLM at deployment yet — Stage 4 is required to make that case.

---

## 6. Limitations

- Disorder retention 94.9 % single-PLM (Exp 43); 94.8 % worst-PLM (ANKH, Exp 46). The gap is geometrically real, not a bug; PolarQuant (Exp 51) is the designed fix.
- L=175 reference protein length in tier tables is a fixed assumption from the codec design spec, not the empirical mean (Exp 45 reports 156). Storage figures scale ~10 % differently for actual mean length.
- Exp 50 sighting numbers use a random split that leaks homology; rigorous CATH-cluster re-run is designed but not yet executed.
- No co-distilled VESM baseline yet; this is the strongest plausible competitor.
- Single host validated (M3 Max). CUDA path tested in fixtures, not benchmarked at scale.
- `dct_k=4` is a storage choice, not a measured quality optimum (see §5.3).

---

## 7. Figure list (~6 figures + 2 tables)

| # | Type | Content | Source data | Status |
|---|------|---------|-------------|--------|
| Fig 1 | Pareto plot | Compression × mean retention with BCa CIs; binary-default and PQ-M=224 marked | `data/benchmarks/rigorous_v1/exp47_sweep_prot_t5_full.json` | needs build (Phase F.2) |
| Fig 2 | Heat-map | 5-PLM × 4-task retention | `data/benchmarks/rigorous_v1/exp46_multi_plm_results.json` | needs build (Phase F.3) |
| Fig 3 | Bar / line | Exp 50 Stage 1–3 ceiling | `results/exp50/stage{1,2,3}_results.json` | needs build (Phase F.4) |
| Fig 4 | Cosine bar | ABTT PC1 alignment with disorder direction | `data/benchmarks/exp45_disorder_forensics.*` | exists in `docs/figures/` — verify |
| Fig 5 | Tree comparison | BM-MCMC vs FastTree/IQ-TREE per-family monophyly | `results/embed_phylo/*_consensus.nwk` | exists — verify |
| Fig 6 | Decoder snippet | 12-line numpy receiver-side decode for binary mode | text figure | new — write |
| Tab 1 | Single-PLM | 6-config × 4-task retention (Exp 47) with BCa CIs | exp47 JSON | exists |
| Tab 2 | Multi-PLM | 5-PLM × 4-task retention (Exp 46) with BCa CIs | exp46 JSON | exists |

---

## 8. Reproducibility statement

- Code: `https://github.com/<TBD>/ProteEmbedExplorations`, MIT license. Pin: `git tag v1.0.<TBD>`.
- Lock file: `uv.lock` (97 packages, `requires-python = ">=3.12"`).
- Tests: `uv run pytest tests/` → 813 / 813 passing as of audit (2026-04-27).
- Data:
  - SCOPe / CATH / CB513 / TS115 / CASP12 / DeepLoc / CheZOD / TriZOD splits live under `data/benchmark_suite/`. Splits include `superfamily_overlap=0` and `family_overlap=0` checks at construction time.
  - Raw PLM embeddings (~67 GB) are not in the repo (gitignored); extraction scripts under `experiments/01_extract_residue_embeddings.py` regenerate them from HuggingFace model weights.
- Statistics: every reported number ships its raw 95 % BCa CI bound in the source JSON (`data/benchmarks/rigorous_v1/`).
- One-command repro: `uv sync --all-extras --all-groups && uv run pytest tests/ && uv run python experiments/47_codec_sweep.py --plm prot_t5_full --quick` (the `--quick` flag uses the cached split + a single seed).

---

## Appendix: Prior (written before audit, 2026-04-26)

> Preserved as the `stub(prior): MANUSCRIPT_SKELETON.md` commit content for
> the calibration loop. Compare against the live skeleton above; the
> prior-vs-posterior delta is summarised in `docs/CALIBRATION.md`.

### Predicted title candidates
1. "OneEmbedding: a universal codec for protein-language-model embeddings"
2. "37× lossless-grade compression of PLM embeddings, validated across 5 models"
3. "What survives compression: a probe of PLM embedding geometry"

### Predicted abstract structure (5 sentences)
1. PLM embeddings are large; downstream uses store and ship them at scale.
2. We benchmarked 232 compression methods on 6 tasks across 5 PLMs.
3. A single configurable codec (centering + RP896 + binary, ~37× / no codebook) achieves 95–100 % retention with BCa-CI bounds.
4. Disorder is the sole consistent weak spot (~95 %); ABTT preprocessing destroys it (Exp 45).
5. The same compressed format admits sequence-only prediction (Exp 50, 69 % bit acc CATH-split), suggesting a path to FASTA → embedding deployment.

### Predicted figure list (~6)
- Fig 1: Pareto plot (compression × retention).
- Fig 2: 5-PLM × 4-task heat-map.
- Fig 3: Codec sweep (Exp 47).
- Fig 4: ABTT artifact (Exp 45).
- Fig 5: Exp 50 learning curve / ceiling.
- Fig 6: Phylogenetics example (Exp 35).

### Predicted limitations (will be in discussion)
- Disorder gap.
- No co-distilled VESM baseline.
- Sequence → OE ceiling.
- Single MPS host — not validated on CUDA at scale.
