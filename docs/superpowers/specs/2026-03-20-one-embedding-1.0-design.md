# One Embedding 1.0 — Design Specification

## Vision

Any PLM → chosen compression → `.one.h5` → set of tools. A production-quality universal protein embedding codec with configurable dimensionality, validated benchmarks, and 7+ built-in tools.

## What Exists Today

The `src/one_embedding/` package is already built with:
- **Core codec** (`core/codec.py`): ABTT3 + RP512 + DCT K=4 → per_residue (L, 512) + protein_vec (2048,)
- **File format** (`io.py`): `.oemb` HDF5 files (single + batch)
- **7 tools** (`tools/`): disorder, ss3, search, classify, align, conserve, mutate
- **CLI** (`cli.py`): encode, inspect, disorder, search, align
- **Extraction** (`extract/`): ProtT5 + ESM2 wrappers
- **560 tests** passing
- **Pre-fitted weights**: ABTT params for ProtT5 + ESM2, CNN probe weights for 512d

### Known Issues
- Phylo MCMC: zombie multiprocessing workers don't stop after runs end
- CNN probes trained only for 512d
- TriZOD benchmark incomplete (2,346/5,786 proteins extracted)
- No validation against UdonPred's 7-dataset disorder benchmark
- Codebook robustness at scale not proven
- Potential data leakage in benchmarks not audited

## What Changes for 1.0

### 1. Configurable Projection Dimension

**Current:** `d_out=512` hardcoded.
**New:** `d_out` configurable, default **768**.

Experiment 38 results justify this:
- RP 768d: SS3 0.837 (99.0% retention), TM F1 0.806 (98.9%), Disorder 0.629 (94.9%)
- RP 512d: SS3 0.819 (96.8%), TM F1 0.744 (91.2%), Disorder 0.598 (90.2%)
- Storage: ~37 KB (768d PQ) vs ~26 KB (512d PQ) — worth it for 7-8pp quality gain on TM
- Retrieval: identical within noise at all dimensions (0.785-0.787)

Changes to `Codec` class:
```python
codec = Codec(d_out=768, dct_k=4, seed=42)  # configurable
codec.fit(corpus)
result = codec.encode(raw)
# result['per_residue']: (L, 768) float16
# result['protein_vec']: (3072,) float16  [dct_k * d_out]
```

`Codec.for_plm('prot_t5')` returns d_out=768 by default.
`Codec.for_plm('prot_t5', d_out=512)` for backward compat.

### 2. `.one.h5` File Format

Rename from `.oemb` to `.one.h5`. Structure:

```
protein.one.h5
├── Attributes (root-level tags):
│   ├── format: "one_embedding"
│   ├── version: "1.0"
│   ├── n_proteins: int
│   └── [freeform: source_model, d_out, compression, corpus_hash, ...]
│
├── {protein_id}/
│   ├── per_residue: (L, D) float32, gzip
│   ├── protein_vec: (V,) float16
│   └── Attributes:
│       ├── seq_len: int
│       └── [freeform: sequence, organism, family, ...]
```

Key properties:
- `D` is variable — tools read shape from data, not config
- `V` = `dct_k * D` (e.g., 3072 for d=768, 2048 for d=512)
- Single-protein files = batch with n_proteins=1
- Tags are freeform HDF5 attributes — any metadata the user wants
- Backward compat: tools also read old `.oemb` files

### 3. Tool Adaptation

**CNN probes (disorder, ss3):**
- Retrain for 768d input (same SETH-style 2-layer Conv1d architecture)
- Ship weight files for both: `disorder_cnn_768d.pt`, `disorder_cnn_512d.pt`
- Auto-detect D from per_residue shape, load matching weights
- If no matching weights, fall back to dimension-agnostic method (norm heuristic)

**Other tools (search, classify, align, conserve, mutate):**
- Already dimension-agnostic (cosine similarity, norms, alignment on any D)
- protein_vec cosine works on any V
- Zero code changes needed

### 4. Quality Assurance Overhaul

#### 4a. Audit benchmarks for data leakage
- Check every train/test split: are test proteins truly unseen?
- Check CB513 split vs SCOPe 5K overlap
- Check CheZOD/TriZOD split vs training corpus overlap
- Verify <30% sequence identity between train/test (UdonPred standard)

#### 4b. Full TriZOD benchmark
- Extract all 5,786 TriZOD proteins (currently 2,346 done)
- Benchmark disorder on full TriZOD with both Ridge and CNN probes
- Compare head-to-head with UdonPred results on same test set (TriZOD348)

#### 4c. UdonPred comparison
- Paper: biorxiv 2026.01.26.701679v2 (docs/references/udonpred_2026.pdf)
- They benchmark on 7 datasets: TriZOD, CheZOD, SoftDis, PDBflex, ATLAS, pLDDT, DisProt
- Their CheZOD Spearman: 0.684 (CheZOD-trained) / 0.702 (TriZOD-trained on CheZOD test)
- Our CheZOD Spearman: 0.713 (CNN on compressed 512d) — competitive
- Run our probe on their exact evaluation protocol for apples-to-apples
- Check if UdonPred code already in our repo (github.com/davidwagemann/udonpred)

#### 4d. Codebook robustness
- Current codebook fitted on 5K SCOPe corpus (~50K residues)
- Test: fit on larger corpus (UniRef50 sample? SwissProt?) — does quality change?
- Test: fit on 1K vs 5K vs 20K proteins — where does it saturate?

#### 4e. Multi-PLM validation
- Validate full pipeline on: ProtT5-XL, ESM2-650M, ProstT5, ESM-C
- Pre-fit ABTT weights for each
- Confirm RP768 benefits generalize across PLMs (Exp 38 was ProtT5 only)

### 5. Code Quality

#### 5a. Fix known bugs
- Phylo MCMC zombie processes (multiprocessing workers don't terminate)
- MrBayes NEXUS consensus tree parsing (0/12 monophyletic is parsing artifact)

#### 5b. Clean up codebase
- Remove redundant research modules from `src/one_embedding/` (keep in `src/` for research)
- Production package should be lean: core/, extract/, tools/, io.py, cli.py, __init__.py
- Cut dead code, make it "poetry" — clean, readable, well-documented

#### 5c. Test the tests
- Run full test suite, verify 560 tests still pass
- Add tests for configurable d_out (768, 512, 256)
- Add tests for .one.h5 format (read/write, backward compat with .oemb)
- Add integration test: FASTA → encode → .one.h5 → disorder → results

### 6. CLI + API Updates

```bash
# New default (768d)
one-embedding encode raw.h5 compressed.one.h5

# Explicit dimension
one-embedding encode raw.h5 compressed.one.h5 --d-out 512

# All tools work on any .one.h5
one-embedding disorder compressed.one.h5
one-embedding inspect compressed.one.h5
```

```python
import one_embedding as oe

# Encode with new default
oe.encode("raw.h5", "compressed.one.h5")  # d_out=768

# Decode
data = oe.decode("compressed.one.h5")
data['per_residue']   # (L, 768) or (L, 512) — whatever was encoded
data['protein_vec']   # (3072,) or (2048,)

# Tools
oe.disorder("compressed.one.h5")  # auto-detects D
```

## Implementation Phases

### Phase 1: Core Upgrade (codec + format)
1. Make `Codec.d_out` configurable (default 768)
2. Update `io.py` for `.one.h5` format with freeform tags
3. Update `__init__.py` encode/decode for new defaults
4. Backward compat: read old `.oemb` files
5. Update CLI `--d-out` flag
6. Tests for all of the above

### Phase 2: Retrain Probes
1. Retrain disorder CNN for 768d on CheZOD
2. Retrain ss3 CNN for 768d on CB513
3. Auto-detection logic in tools (read D from data, load right weights)
4. Verify retention: 768d probes vs 512d probes vs raw

### Phase 3: Quality Assurance
1. Audit all train/test splits for leakage
2. Full TriZOD extraction + benchmark
3. UdonPred head-to-head comparison (7 datasets)
4. Codebook robustness at scale
5. Multi-PLM validation (ESM2, ProstT5, ESM-C)

### Phase 4: Polish
1. Fix phylo zombie processes
2. Fix MrBayes parsing
3. Clean codebase — remove dead code, sharpen everything
4. Update README and documentation
5. Pre-fit ABTT weights for all PLMs at 768d
6. Final test pass

## Success Criteria

- All 7 tools work on `.one.h5` files from any PLM at any d_out
- RP768 default: SS3 ≥0.835, TM F1 ≥0.800, Disorder ≥0.625, Ret@1 ≥0.780
- Full TriZOD benchmark with apples-to-apples UdonPred comparison
- No data leakage in any benchmark
- Clean, tested codebase ready for pip packaging
- Codebook robustness demonstrated (corpus size doesn't matter past N)

## Non-Goals (for 1.0)
- PQ quantization as a storage tier (future — store decoded float32 for now)
- Public adapters for third-party tools (future)
- pip packaging / PyPI release (next step after 1.0)
- Web API / cloud deployment
