# One Embedding Benchmark Suite — Design Specification

## Vision

A collection of purpose-specific benchmark datasets — published and handcrafted — with proper redundancy reduction, diverse taxa, and reproducible evaluation protocols. Serves codec validation, probe training, tool benchmarking, and the "Unknown Unknowns" paper.

## Principles

1. **Collection, not monolith** — each subset has its own purpose, splits, and protocol
2. **Published + handcrafted** — established benchmarks (CB513, CheZOD) alongside custom diversity sets from SpeciesEmbedding
3. **Rost-lab grade splits** — <30% sequence identity between train/test, MMseqs2 clustering, documented methodology
4. **PLM-agnostic** — same benchmark, multiple PLMs (ProtT5, ESM2 first, others later)
5. **Chunking is optional** — user selects; we benchmark to find what works first (Exp 40)

## Dataset Structure

```
data/benchmark_suite/
├── retrieval/
│   ├── scope_5k/                 # SCOPe 5K, family-labeled, RR 30%
│   └── toxfam/                   # ToxFam v2, 84 proteins, 12 venom families
│
├── per_residue/
│   ├── cb513/                    # SS3/SS8 (<25% identity by construction)
│   ├── chezod/                   # Disorder, SETH split 1174/117
│   ├── trizod/                   # Disorder, full 5786 proteins, TriZOD348 test
│   ├── udonpred_7/              # 7 disorder datasets from UdonPred paper
│   │   ├── trizod/  chezod/  softdis/  pdbflex/  atlas/  plddt/  disprot/
│   └── tmbed/                    # TM topology
│
├── diversity/
│   ├── venom_families/           # 3FTx, Kunitz, PLA2, conotoxin, ant/bee/snake venom
│   ├── non_toxin/                # Casein, MRJP/Yellow, serine proteases
│   └── cross_taxa/               # Sampled across tree of life (from ToxProt Feb2026)
│
├── stress/
│   ├── long_proteins/            # >2K residues (titin, etc.) — chunking experiments
│   └── short_proteins/           # <30 residues
│
├── plm_embeddings/               # Pre-computed per PLM
│   ├── prot_t5_xl/               # .h5 files per subset
│   └── esm2_650m/                # Same subsets, different PLM
│
├── splits/                       # All train/test splits
│   └── {dataset}_{method}_{identity}.json
│
└── metadata.json                 # Master index
```

## Redundancy Reduction Protocol

1. **Within each subset:** Cluster at 30% seq identity via MMseqs2 (`--min-seq-id 0.3 -c 0.8`), keep one representative per cluster
2. **Between train and test:** No train-test pair >30% identity. Split by cluster assignment (whole clusters go to train OR test, never both)
3. **Between ABTT corpus and test:** Verify zero ID overlap (SCOPe 5K uses `d*` IDs, CB513 uses `cb513_*`, CheZOD uses numeric — confirmed no overlap)
4. **Published splits preferred:** CheZOD 1174/117 (SETH), TriZOD348 (UdonPred), CB513 (non-redundant by construction)
5. **Documentation:** Every split recorded with: method, tool, version, identity threshold, coverage threshold, random seed

## PLM Configuration

```python
PLM_CONFIG = {
    "prot_t5":    {"max_len": 2048, "d_model": 1024, "chunk_overlap": 256},
    "esm2_650m":  {"max_len": 1022, "d_model": 1280, "chunk_overlap": 128},
    "esm2_3b":    {"max_len": 1022, "d_model": 2560, "chunk_overlap": 128},
    "prost_t5":   {"max_len": 2048, "d_model": 1024, "chunk_overlap": 256},
    "esmc_300m":  {"max_len": 2048, "d_model":  960, "chunk_overlap": 256},
    "ankh_large": {"max_len": 2048, "d_model": 1536, "chunk_overlap": 256},
}
```

## Chunking Strategy

User-selectable, not forced:

```python
oe.encode("raw.h5", "out.one.h5", chunking="none")       # truncate at max_len
oe.encode("raw.h5", "out.one.h5", chunking="auto")       # model-aware chunks
oe.encode("raw.h5", "out.one.h5", chunking=1024, overlap=128)  # custom
```

For chunked proteins:
1. Split sequence into windows of `max_len` with `chunk_overlap` residue overlap
2. Embed each chunk independently with the PLM
3. Average overlapping residue embeddings (weighted by distance from chunk boundary — center residues get full weight, edge residues blended)
4. Store full `(L, D)` — downstream tools never know it was chunked
5. `.one.h5` tag: `chunked: bool`, `chunk_size: int`, `chunk_overlap: int`

**Experiment 40 (prerequisite):** Before shipping a default, benchmark ~10 long proteins with: no chunking (truncated), chunk sizes 512/1024/2048, overlaps 0/128/256. Compare per-residue quality to find the sweet spot.

## Diversity Set — From SpeciesEmbedding

| Source | ~Proteins | Family | Taxa Coverage |
|--------|-----------|--------|--------------|
| 3FTx | 50 | Snake three-finger toxins | Reptilia |
| Kunitz | 80 | Protease inhibitors | Broad |
| PLA2 | 100 | Phospholipases | Reptilia, Insecta |
| Conotoxin | 159 | Cone snail toxins | Gastropoda |
| Snake venom | 300 | Multi-family | Elapidae, Colubridae |
| Ant venom | 100 | Hymenopteran toxins | Formicidae |
| Bee venom | 100 | Apoidea toxins | Apidae |
| Casein | 50 | Milk proteins | Mammalia |
| MRJP/Yellow | 200 | Royal jelly + yellow | Insecta |
| ToxProt sample | 500 | Cross-family toxins | Broad |
| **Total** | **~1,500** | **10+ families** | **Across tree of life** |

## Size Budget

| Subset | Proteins | ProtT5 Size | ESM2 Size | Status |
|--------|----------|-------------|-----------|--------|
| SCOPe 5K | 5,000 | 2.0 GB | 2.5 GB | Exists |
| CB513 | 513 | 0.2 GB | 0.2 GB | Exists |
| CheZOD | 1,291 | 0.5 GB | — | Exists |
| TriZOD | 5,786 | 2.5 GB | — | Partial (2,346 done) |
| UdonPred 7 | ~5,000 | 2.0 GB | — | To extract |
| TMbed | ~500 | 0.2 GB | — | Exists |
| ToxFam v2 | 84 | 0.03 GB | — | Exists |
| Diversity | ~1,500 | 0.6 GB | — | Partial |
| Stress | ~50 | 0.2 GB | — | To create |
| **Total** | **~20,000** | **~8 GB** | **+5 GB** | **~13 GB** |

Within the 15 GB budget. ESM2 only for key subsets (SCOPe, CB513).

## UdonPred Comparison Protocol

From their paper (biorxiv 2026.01.26.701679v2):
- 7 datasets: TriZOD, CheZOD, SoftDis, PDBflex, ATLAS, pLDDT, DisProt
- They use ProstT5; we use ProtT5 (apples-to-oranges, but informative)
- They train separate models per dataset, evaluate pairwise (7x7 matrix)
- Our approach: train one probe on compressed embeddings, evaluate on all 7 test sets
- Metric: Spearman correlation (continuous), AP+AUROC (binary)
- Their CheZOD-trained on CheZOD test: 0.684; ours: 0.713 (CNN on 512d compressed)

## Implementation Phases

### Phase A: Assemble benchmark data
1. Organize existing data into `data/benchmark_suite/` structure
2. Extract remaining TriZOD proteins (3,440 remaining)
3. Pull UdonPred 7 datasets (from their GitHub or recreate from sources)
4. Sample diversity proteins from SpeciesEmbedding
5. Curate stress test set (find titin + other long proteins in UniProt)
6. Run MMseqs2 redundancy reduction on custom subsets
7. Document all splits

### Phase B: Embed and validate
1. Embed all new proteins with ProtT5 (+ ESM2 for key subsets)
2. Compress with codec at 768d (new default)
3. Run all retention benchmarks on new data
4. Compare with UdonPred on their 7 datasets
5. Chunking experiment (Exp 40) on long proteins

### Phase C: Document and ship
1. Create `metadata.json` master index
2. Write benchmark README with evaluation protocols
3. Update main README with new benchmark numbers
4. Commit benchmark suite

## Success Criteria

- All 20K+ proteins embedded and organized in benchmark suite
- Every train/test split verified <30% identity (MMseqs2)
- Retention benchmarks match or exceed current numbers at 768d
- UdonPred comparison completed on all 7 datasets
- Chunking strategy validated on long proteins
- Diversity set covers 10+ protein families across tree of life
- Total size within 15 GB budget

## Non-Goals (for now)
- Embedding all 12 PLMs (just ProtT5 + ESM2)
- Web-accessible benchmark leaderboard
- Automated benchmark CI pipeline
- Solving the >2K length problem completely (Exp 40 informs, doesn't solve)
