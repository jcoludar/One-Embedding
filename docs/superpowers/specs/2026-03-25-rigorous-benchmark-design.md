# Rigorous Benchmark Framework for One Embedding 1.0

**Date**: 2026-03-25
**Status**: Approved
**Goal**: Rigorously validate One Embedding codec performance against raw ProtT5 and ESM2 baselines using community-standard datasets, enforced methodological rules, and cross-validated results. Build package credibility first, paper-ready rigor follows.

## Motivation

The current "100.1% mean retention" claim has methodological issues:

1. **Unfair retrieval comparison**: Raw uses mean pooling, compressed uses DCT K=4 pooling (different methods, not codec quality)
2. **No error bars**: Point estimates only, no bootstrap CIs
3. **Unequal task weighting**: Simple mean across tasks with vastly different sample sizes
4. **Weak disorder probe**: Ridge alpha=1.0 (no tuning) gives 94.8%; CNN gives 99.0%
5. **Fixed hyperparameters**: No CV-tuning of C or alpha
6. **Narrow task coverage**: Only 5 tasks, no cross-dataset validation

Corrected retention estimate: **~97.8-98.8%**, not 100.1%. Still excellent, but the claim must be honest.

This spec designs a comprehensive, rule-enforced benchmark framework that:
- Fixes all known issues
- Expands to 18+ tasks across 15+ datasets
- Cross-validates every task on multiple independent datasets
- Tests on both ProtT5 and ESM2
- Compares head-to-head against DCTdomain (direct competitor)
- Stress-tests every shipped tool individually

---

## Golden Rules (Enforced in Code)

Every benchmark must pass these rules before any metric is computed. Implemented as assertions in `rules.py`.

### Rule 1 -- Fair Comparison
Raw and compressed must use identical preprocessing and pooling. If compressed uses DCT K=4, raw baseline must too. Enforced: a `ComparisonPair` dataclass that binds raw + compressed + method together and asserts matching configuration.

### Rule 2 -- No Train/Test Leakage
Zero protein overlap between train and test sets. Enforced: `assert len(set(train_ids) & set(test_ids)) == 0`.

### Rule 3 -- Redundancy Reduction
Train-to-test sequence identity must be below a declared threshold (default <=30%). Datasets with published thresholds (CB513 <25%, CheZOD SETH split, CATHe <20%) pass automatically with citation. New splits must run MMseqs2 easy-search verification.

### Rule 4 -- Statistical Significance
Every metric must report 95% bootstrap CI (10K iterations). No point estimates in final output. Implemented: every metric function returns `{"value": float, "ci_lower": float, "ci_upper": float, "n": int}`.

### Rule 5 -- Multiple Seeds
Probe training runs >= 3 random seeds (42, 123, 456). Report mean +/- std.

### Rule 6 -- Class Balance Reporting
For imbalanced tasks (class ratio > 3:1), report both macro and weighted metrics. Per-class breakdown mandatory. Flag classes with < 100 test samples.

### Rule 7 -- Probe Minimality
Default ("headline") probe is linear (LogReg/Ridge). CNN probe runs as secondary. Both reported. Linear measures embedding quality; CNN measures practical ceiling. If CNN >> linear by >5pp, flag it.

### Rule 8 -- Hyperparameter Selection
C (LogReg) and alpha (Ridge) selected via 3-fold CV on training set only. Search grid: C in [0.01, 0.1, 1.0, 10.0], alpha in [0.01, 0.1, 1.0, 10.0, 100.0]. No hardcoded values except the grid itself.

### Rule 9 -- Multi-PLM
Every benchmark runs on both ProtT5-XL (1024d) and ESM2-650M (1280d). Single-PLM results are flagged as incomplete.

### Rule 10 -- Ablation
Each codec component (ABTT, RP, quantization) tested in isolation to measure individual contribution. Minimum ablation matrix:
- Raw (no codec)
- ABTT only
- RP only (no ABTT)
- ABTT + RP (no quantization) = full codec
- ABTT + RP + float16 = shipped codec

### Rule 11 -- Cross-Validation Across Datasets
No single dataset trusted alone. Every task type validated on >= 2 independent datasets where available. If results diverge by > 5pp between datasets, investigate and document why.

### Rule 12 -- Dual Metric
All distance/similarity-based evaluations run both cosine and euclidean. Report both. Flag when they disagree by > 2pp.

---

## Benchmark Matrix

### Per-Residue Tasks

| Task | Dataset 1 (Primary) | Dataset 2 (Cross-check) | Dataset 3 (External) | Metric |
|------|---------------------|------------------------|---------------------|--------|
| **SS3** | CB513 (ours, 513 proteins) | TS115 (NetSurfP) | CASP12 (NetSurfP) | Q3, SOV, per-class acc |
| **SS8** | CB513 | TS115 | CASP12 | Q8, per-class acc |
| **Disorder** | CheZOD117 (SETH split) | TriZOD348 (predefined) | -- | Spearman rho, AUC-ROC (binary Z<8) |
| **RSA** | NetSurfP CB513 | NetSurfP TS115 | NetSurfP CASP12 | Pearson r |
| **TM Topology** | TMbed cv_00 | TMbed cv_01 | -- | Macro F1, per-class F1 |
| **Conservation** | ConSurf10k (VESPA) | -- | -- | MCC (9-class), Spearman rho |
| **Binding Residues** | BioLip (bindEmbed21) | -- | -- | AUC-ROC (metal, nucleic, small mol) |
| **Buried Residues** | ProteinGLUE | -- | -- | Accuracy, F1 |
| **PPI Interface** | ProteinGLUE | -- | -- | AUC-ROC, F1 |
| **Epitope** | ProteinGLUE | -- | -- | AUC-ROC, F1 |
| **Contact Prediction** | ProteinNet (TAPE) | -- | -- | Precision @ L/5 |

### Protein-Level Tasks

| Task | Dataset 1 | Dataset 2 | Metric |
|------|-----------|-----------|--------|
| **Subcellular Localization** | LA setDeepLoc (2768 prot, 30% PIDE) | LA setHARD (490 prot, 20% PIDE) | Q10 accuracy |
| **Family Retrieval** | SCOPe 5K (40% ASTRAL) | CATH20 (DCTdomain, 14.4K domains) | Ret@1, Ret@5, MRR (cosine + euclidean) |
| **Superfamily Retrieval** | SCOPe 5K | CATH20 | Ret@1, MRR |
| **CATH Superfamily Classification** | CATHe (<20% identity, 1773 classes) | -- | Accuracy, macro-F1 |
| **CATH Annotation Transfer** | ProtTucker/EAT (pretrained Tucker network) | -- | C/A/T/H accuracy |
| **GO Term Prediction** | goPredSim (CAFA3, kNN) | -- | Fmax (MFO/BPO/CCO) |
| **FunFam EC Purity** | FunFams clustering (DBSCAN) | -- | Cluster purity, sequence purity |
| **Fold Classification** | SCOPe 5K | TAPE Remote Homology | Accuracy |

### Structural Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| **lDDT correlation** | Exp 37 set | Spearman rho |
| **Contact precision** | Exp 37 set | Precision @ L/5 |
| **TM-score correlation** | SCOPe structural pairs | Spearman rho, Pearson r |

### Direct Competitor Comparison

| Method | Approach | Benchmark | Our equivalent |
|--------|----------|-----------|----------------|
| **DCTdomain** | ESM2 per-residue -> 2D-DCT -> 480d fingerprint | CATH20 (14,433 domains, 1,566 families) | ABTT3 + RP768 + DCT K=4 |

### Cross-Check Philosophy

Every task appears on >= 2 datasets where possible. Retention numbers that are consistent across datasets are trustworthy. Numbers that diverge flag a problem. Specific cross-checks:

- SS3 on CB513 vs TS115 vs CASP12: should be within 2pp
- Disorder on CheZOD117 vs TriZOD348: should be within 3pp (different score distributions)
- Retrieval on SCOPe 5K vs CATH20: different hierarchies, expect some divergence but same trend
- Localization on setDeepLoc vs setHARD: expect 15-20pp gap (by design), but RETENTION should be similar

---

## Tool-Level Deep Tests

Each of the 7 shipped tools gets independent rigorous testing.

### Tool 1: Disorder (`disorder.py`)

- **Cross-dataset**: Train CheZOD1174, test CheZOD117 AND TriZOD348
- **Compare to SETH**: Run actual SETH pretrained weights on compressed embeddings
- **Probe comparison**: Ridge vs CNN vs LogReg, 3 seeds, bootstrap CIs
- **Residue-level error analysis**: Spearman rho by amino acid type, by disorder level (ordered/twilight/disordered)
- **Length stress test**: Proteins >500, >1000, >2000 residues
- **Binary threshold**: AUC-ROC at Z-score < 8 (disordered)
- **Dual metric**: Cosine + euclidean distance-based disorder (if applicable)

### Tool 2: SS3 (`ss3.py`)

- **Cross-dataset**: Train CB513 80%, test CB513 20% + TS115 + CASP12
- **Per-class analysis**: H/E/C accuracy, confusion matrix
- **SOV score**: Segment overlap measure (standard in SS prediction)
- **Compare to ProtTrans pretrained probe**: Fixed weights, no retraining
- **Compression confusion**: Where does compression cause misclassification? E->C? H->C?

### Tool 3: Search (`search.py`)

- **SCOPe retrieval**: Ret@1, Ret@5, MRR, MAP with bootstrap CIs (cosine + euclidean)
- **CATH20 retrieval**: DCTdomain benchmark head-to-head
- **CATHe classification**: 1,773 superfamilies
- **ProtTucker/EAT**: Annotation transfer with our vectors
- **Scaling test**: Quality at 1K, 5K, 50K, 500K database sizes
- **Fair baseline**: Raw uses DCT K=4 (same as compressed)

### Tool 4: Classify (`classify.py`)

- **SCOPe 3-level**: Family, superfamily, fold classification
- **CATHe comparison**: Our kNN vs their ANN on same data (cosine + euclidean)
- **k sensitivity**: k=1, 3, 5, 10
- **Few-shot**: Performance at 1, 3, 5, 10 members per family

### Tool 5: Align (`align.py`)

- **vs TM-align**: Compare embedding alignment to structural alignment on SCOPe pairs (cosine + euclidean scoring matrices)
- **Twilight zone**: Pairs <25% sequence identity
- **Gap penalty sweep**: gap_open and gap_extend sensitivity
- **PEbA reproduction**: Match published numbers from the PEbA paper

### Tool 6: Conserve (`conserve.py`) -- Phase 1: Benchmark Heuristic

- **ConSurf10k ground truth**: Correlate norm-based scores with ConSurf 9-class labels
- **Per-family analysis**: Does heuristic work for some families?
- **Honest reporting**: If Spearman rho < 0.3, document as "experimental"

*(Phase E: Train real VESPA-style probe on ConSurf10k)*

### Tool 7: Mutate (`mutate.py`) -- Phase 1: Benchmark Heuristic

- **ProteinGym DMS data**: Correlate sensitivity scores with measured fitness effects
- **Per-protein Spearman**: Distribution of per-protein correlations (not aggregated)
- **Honest reporting**: Document limitations clearly

*(Phase E: Train real probe on DMS data)*

### Shared Across All Tools

- **Dimension check**: Every tool tested at 768d (default) and 512d (V2)
- **Multi-PLM**: ProtT5 and ESM2 for every tool
- **Edge cases**: Empty sequences, single-residue, non-standard amino acids (X, U, B, Z)
- **Determinism**: Same input -> same output, verified across 3 runs
- **Dual metric**: Cosine + euclidean wherever distance/similarity is used

### Unit Tests Per Tool

Each tool gets `tests/test_tool_{name}.py` with:

1. **Smoke test**: Runs without error on synthetic data (random embeddings)
2. **Shape test**: Output dimensions match input protein lengths
3. **Value range test**: Scores within expected bounds (e.g., disorder Z-scores, SS3 probabilities)
4. **Known-answer test**: Specific protein -> expected output within tolerance
5. **Edge case tests**: L=1, L=5000, non-standard AA, all-gap input
6. **Determinism test**: 3 runs produce identical output
7. **Dimension agnostic test**: Works on 512d and 768d input

---

## Code Architecture

```
experiments/
  43_rigorous_benchmark/
    __init__.py
    config.py              # Dataset paths, golden rule thresholds, seeds
    rules.py               # Golden rules as assertions (enforced before every benchmark)
    datasets/
      __init__.py
      cb513.py             # SS3/SS8/RSA loader + split + validate
      chezod.py            # CheZOD + TriZOD loaders
      netsurfp.py          # TS115, CASP12 (download + cache)
      proteinglue.py       # Buried, PPI, epitope tasks
      deeplocla.py         # Light Attention setDeepLoc + setHARD
      consurf.py           # ConSurf10k (VESPA)
      biolip.py            # bindEmbed21 binding residues
      cathe.py             # CATHe 1773-class superfamily
      cath20.py            # DCTdomain CATH20
      proteinnet.py        # Contact prediction (TAPE)
      scope.py             # SCOPe 5K (existing, wrapped)
      tmbed.py             # TMbed CV folds
    probes/
      __init__.py
      linear.py            # LogReg + Ridge with CV-tuned hyperparams
      cnn.py               # SETH-style 2-layer CNN
      fixed.py             # Pretrained probes (SETH, ProtTrans, VESPA)
    metrics/
      __init__.py
      per_residue.py       # Q3, Q8, SOV, Spearman, AUC-ROC, per-class
      retrieval.py         # Ret@k, MRR, MAP (cosine + euclidean)
      classification.py    # Accuracy, macro-F1, weighted-F1
      statistics.py        # Bootstrap CI, multi-seed permutation, Cohen's d
    runners/
      __init__.py
      per_residue.py       # Runs all per-residue tasks
      protein_level.py     # Runs retrieval, classification, localization
      tool_tests.py        # Runs tool-specific deep tests
      ablation.py          # ABTT / RP / quant individual contributions
      competitor.py        # DCTdomain head-to-head
    report.py              # Generate markdown report + figures

tests/
  test_tool_disorder.py
  test_tool_ss3.py
  test_tool_search.py
  test_tool_classify.py
  test_tool_align.py
  test_tool_conserve.py
  test_tool_mutate.py
  test_benchmark_rules.py  # Tests that the golden rules themselves work correctly
```

### Key Design Decisions

1. **`rules.py` runs before every benchmark**: If a rule fails, the benchmark aborts with a clear error message citing the violated rule. No silent violations.

2. **Uniform dataset interface**: Every dataset loader returns:
   ```python
   @dataclass
   class BenchmarkDataset:
       name: str
       task_type: str           # "per_residue" | "protein_level" | "structural"
       train_ids: list[str]
       test_ids: list[str]
       embeddings: dict[str, np.ndarray]  # {protein_id: (L, D)}
       labels: dict[str, Any]   # {protein_id: labels}
       metadata: dict           # source, identity_threshold, citation, etc.
   ```

3. **Probes are separate from datasets**: Any probe can run on any dataset. `linear.py` for headlines, `cnn.py` for ceiling, `fixed.py` for pretrained no-cheat baselines.

4. **Statistics are mandatory**: `metrics/statistics.py` wraps every metric computation. The return type is always `MetricResult(value, ci_lower, ci_upper, n, seeds_mean, seeds_std)`. You cannot get a bare float.

5. **ComparisonPair enforces fairness**:
   ```python
   @dataclass
   class ComparisonPair:
       raw_embeddings: dict[str, np.ndarray]
       compressed_embeddings: dict[str, np.ndarray]
       pooling_method: str      # must match
       distance_metric: str     # "cosine" AND "euclidean" (runs both)
       preprocessing: str       # must match
   ```
   Assertion: `raw.pooling_method == compressed.pooling_method`.

6. **Auto-generated report**: `report.py` produces a comprehensive markdown document with:
   - Retention table with CIs for all tasks
   - Per-class breakdowns for imbalanced tasks
   - Cross-dataset consistency checks
   - Ablation contribution table
   - Multi-PLM comparison
   - DCTdomain head-to-head
   - Tool-level summary (ready/experimental/heuristic)

---

## Phasing

### Phase A -- Fix & Validate (our existing benchmarks)

**Goal**: Fix all known issues in existing 5-task benchmark.

1. Fix unfair retrieval baseline: compute DCT K=4 on raw 1024d ProtT5
2. Add bootstrap CIs (10K iterations) to all existing metrics
3. CV-tune hyperparameters (C grid, alpha grid) on training sets
4. Add euclidean metric alongside cosine everywhere
5. Run 3 seeds (42, 123, 456) for all probes
6. Write unit tests for all 7 tools (`test_tool_*.py`)
7. Extract ESM2-650M embeddings for CB513, CheZOD, SCOPe 5K (if not already done)
8. Re-run all 5 tasks with fixes, report corrected retention

**Output**: Corrected retention table with CIs, tool unit tests passing.

### Phase B -- Expand Per-Residue

**Goal**: Add 6 new per-residue tasks from external datasets.

1. Download NetSurfP-3.0 data (TS115, CASP12 labels in NPZ format)
2. Download ProteinGLUE data (buried, PPI, epitope tasks)
3. Download ConSurf10k data (VESPA conservation labels)
4. Download BioLip data (bindEmbed21 binding residue labels)
5. Extract ProtT5 + ESM2 embeddings for all new protein sets
6. Compress all with 768d codec
7. Implement dataset loaders (`datasets/*.py`)
8. Run all per-residue benchmarks through rule-enforced framework
9. Cross-validate: SS3 on CB513 vs TS115 vs CASP12 (must agree within 2pp)
10. Cross-validate: Disorder on CheZOD117 vs TriZOD348

**Output**: 11 per-residue tasks, all with CIs, cross-dataset consistency verified.

### Phase C -- Expand Protein-Level

**Goal**: Add 7 protein-level benchmarks including competitor comparison.

1. Download Light Attention code + DeepLoc data (13K train + 3K test)
2. Download CATHe data from Zenodo (pre-computed embeddings or sequences)
3. Set up ProtTucker/EAT pipeline (download pretrained weights + lookup DBs)
4. Set up goPredSim (download GOA lookup, CAFA3 evaluation)
5. Set up DCTdomain CATH20 benchmark (already cloned in tools/reference/)
6. Set up FunFams clustering benchmark
7. Extract embeddings, compress, run all protein-level benchmarks
8. Head-to-head: our codec vs DCTdomain on CATH20 (both cosine + euclidean)

**Output**: 8 protein-level tasks, competitor comparison, all with CIs.

### Phase D -- Ablation & Stress Tests

**Goal**: Understand contribution of each codec component, test edge cases.

1. Ablation matrix: raw, ABTT-only, RP-only, ABTT+RP, ABTT+RP+fp16
2. Length stress tests: proteins >500, >1000, >2000 residues
3. Edge case tests: non-standard AA, very short proteins
4. PFMBench run (independent external validation, 38 tasks)
5. Per-class and per-protein error analysis
6. Generate final report with all results

**Output**: Ablation table, stress test results, PFMBench comparison, complete report.

### Phase E -- Conservation & Mutation Probes (Later)

**Goal**: Replace heuristic tools with trained probes.

1. Train conservation probe on ConSurf10k (VESPA architecture)
2. Train mutation sensitivity probe on ProteinGym DMS data
3. Re-benchmark conserve and mutate tools with real probes
4. Update tool weights and documentation

**Output**: Trained probes for conserve + mutate, updated retention numbers.

---

## Disk Space Estimate

| Dataset | Proteins | Raw embeddings (est.) | Compressed (768d) |
|---------|:--------:|:---------------------:|:------------------:|
| Existing (CB513, CheZOD, TriZOD, SCOPe, TMbed) | ~12K | 26 GB (already stored) | ~5 GB |
| ConSurf10k (VESPA) | 10.5K | ~8 GB | ~2 GB |
| BioLip (bindEmbed21) | ~3K | ~2 GB | ~0.5 GB |
| NetSurfP (TS115, CASP12) | ~700 | ~0.5 GB (labels only, reuse CB513 embeddings) | -- |
| ProteinGLUE | ~3K | ~2 GB | ~0.5 GB |
| DeepLoc (Light Attention) | ~14K | ~10 GB | ~3 GB |
| CATHe | 1M+ | ~4 GB (pooled vectors only) | ~1.5 GB |
| DCTdomain CATH20 | 14.4K | ~10 GB (ESM2) | ~3 GB |
| goPredSim / ProtTucker | -- | ~2.5 GB (pre-computed, download) | -- |
| **Total new** | | **~39 GB** | **~10.5 GB** |
| **Grand total** | | **~65 GB** | **~15.5 GB** |

Current disk: 221 GB free. Fits comfortably.

---

## Success Criteria

The benchmark framework is complete when:

1. All 12 golden rules are implemented and enforced in code
2. All 18+ tasks run on both ProtT5 and ESM2 with bootstrap CIs
3. Cross-dataset consistency verified (divergence < 5pp or documented)
4. Fair retrieval baseline (DCT K=4 for both raw and compressed)
5. Ablation table shows individual component contributions
6. DCTdomain head-to-head comparison completed
7. All 7 tool unit tests pass
8. Auto-generated report covers all results
9. Honest retention number reported (with CIs) replacing the "100.1%" claim
10. Conserve and mutate tools have honest assessments (heuristic performance documented)

---

## External Resources

### RostLab Tools
- SETH: github.com/Rostlab/SETH (disorder, CheZOD)
- EAT/ProtTucker: github.com/Rostlab/EAT (CATH retrieval)
- VESPA: github.com/Rostlab/VESPA (conservation, ConSurf10k)
- bindEmbed21: github.com/Rostlab/bindPredict (binding residues)
- goPredSim: github.com/Rostlab/goPredSim (GO terms)
- TMbed: github.com/BernhoferM/TMbed (TM topology)
- Light Attention: github.com/HannesStark/protein-localization (localization)
- ProtTrans: github.com/agemagician/ProtTrans (SS3/SS8 probes)

### Orengo Group Tools
- CATHe: github.com/vam-sin/CATHe (superfamily classification)
- CATHe2: github.com/Mouret-Orfeu/CATHe2 (data on Zenodo)
- DCTdomain: github.com/mgtools/DCTdomain (CATH20 benchmark)
- FunFams: github.com/Rostlab/FunFamsClustering (EC purity)
- eMMA: github.com/UCLOrengoGroup/eMMA (FunFam generation)

### Community Benchmarks
- ProteinGLUE: github.com/ibivu/protein-glue (7 per-residue tasks)
- PFMBench: github.com/biomap-research/PFMBench (38 tasks)
- TAPE: github.com/songlab-cal/tape (SS, contact, homology)
- NetSurfP-3.0: github.com/Eryk96/NetSurfP-3.0 (CB513/TS115/CASP12)

### Datasets (Download URLs)
- NetSurfP NPZ: services.healthtech.dtu.dk/services/NetSurfP-3.0/training_data/
- CATHe data: doi.org/10.5281/zenodo.14534966
- ConSurf10k: zenodo.org/records/5238537
- ProtTucker pre-computed: zenodo.org/records/14675997
- PFMBench: github.com/biomap-research/PFMBench
