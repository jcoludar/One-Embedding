# One Embedding Toolkit — Implementation Setup

## Architecture Summary from Code Study

All reference tools follow the same pattern:
**PLM embeddings → lightweight probe → per-residue or protein-level prediction**

The probes are remarkably small:

| Tool | Task | Input Dim | Architecture | Params | Our 512d Adaptation |
|------|------|-----------|-------------|--------|-------------------|
| SETH | Disorder | 1024 | 2×Conv2d(k=5), 28 hidden, Tanh | ~15K | Change input conv: 512→28 |
| TMbed | TM topology | 1024 | Pointwise→2×DepthwiseConv(k=9,21)→Pointwise, 64ch | ~30K | Change input conv: 512→64 |
| bindEmbed21 | Binding sites | 1024 | 2×Conv1d, ELU, Dropout=0.7 | ~30K | Change in_channels: 512 |
| VespaG | Mutation effects | 2560 | Linear(256)+LeakyReLU+Linear(20) | 660K | Change input: 512→256→20 |
| ProtTucker | Family classification | 1024 | Linear(256)+Tanh+Linear(128) | ~400K | Change input: 2048→256→128 |
| Kibby | Conservation | 1280 | Linear regression (dot product) | ~1.3K | Change coef shape: (512,) |
| ADOPT | Disorder | 650-1280 | Lasso regression | ~1K | No change (dim-agnostic) |
| TM-Vec | Structural similarity | 1024 | 2×TransformerEncoder+MeanPool+Linear(512) | ~4M | Change d_model: 512 |
| DCTdomain | Structural search | 480 | 2D-DCT[3,80]→int8→FAISS | 0 | Direct swap (512 for 480) |
| PEbA | Alignment | 1024 | cos_sim(emb_i, emb_j)×10 → NW/SW | 0 | Direct swap |
| Draupnir | Ancestral recon | 21 (BLOSUM) | BiGRU+OU-VAE, z_dim=30 | ~500K | Replace BLOSUM(21d) with 512d |

---

## Per-Tool Setup

### Tool 1: Phylogenetics [DONE]
- **File:** `experiments/35_embedding_phylogenetics.py`
- **Tests:** 71 passing
- **Benchmark:** ToxFam v2, 10/12 families monophyletic

### Tool 2: Embedding Aligner

**Reference:** PEbA (`tools/reference/PEbA/peba.py`)

**Core algorithm (from PEbA):**
```python
# Score: cosine similarity between per-residue embeddings, scaled by 10
cos_sim = np.dot(emb_A[i], emb_B[j]) / (norm_A[i] * norm_B[j]) * 10
# Then standard NW/SW with affine gap penalties (gopen=-11, gext=-1)
```

**Our implementation plan:**
1. `src/one_embedding/aligner.py`:
   - `embedding_score_matrix(emb_A, emb_B)` — cosine similarity matrix (L_A × L_B)
   - `z_score_filter(score_matrix)` — EBA-style row/column z-score normalization
   - `needleman_wunsch(score_matrix, gopen, gext)` — global alignment
   - `smith_waterman(score_matrix, gopen, gext)` — local alignment

2. **Benchmark:** BAliBASE 3 (RV11: <20% identity, RV12: 20-40%)
   - Metric: Sum-of-Pairs (SP) score vs reference alignments
   - Compare: PEbA on raw ProtT5 vs our 512d compressed vs BLOSUM62

3. **Tests:**
   - `test_score_matrix_shape` — (L_A, L_B) output
   - `test_identical_sequences_diagonal` — diagonal should be highest
   - `test_nw_produces_valid_alignment` — no overlapping positions
   - `test_alignment_better_than_random`
   - `test_z_score_improves_twilight_zone`

**Training data needed:** None (alignment is parameter-free after embedding extraction)

### Tool 3: Mutational Landscape

**Reference:** VespaG (`tools/reference/VespaG/vespag/models/fnn.py`)

**Core architecture (from VespaG):**
```python
class FNN(nn.Module):
    # Linear(input_dim → 256) + LeakyReLU + Dropout(0.2) + Linear(256 → 20)
    # Input: per-residue embedding (L, D)
    # Output: 20 mutation effect scores per position (L, 20)
    # Loss: MSE against GEMME pseudo-labels
```

**Our implementation plan:**
1. `src/one_embedding/mutation_scanner.py`:
   - `EmbeddingMutationScanner` — VespaG-style FNN on 512d
   - `compute_embedding_displacement(wt_emb, mut_emb)` — cosine distance between WT and mutant embeddings (zero-shot, no training)

2. **Benchmark:** ProteinGym (`tools/reference/ProteinGym/`)
   - 217 DMS assays, 2.7M variants
   - Metric: Spearman ρ between predicted and measured fitness
   - Compare: VespaG on raw ESM-2 vs our 512d probe vs zero-shot displacement

3. **Tests:**
   - `test_output_shape_L_by_20`
   - `test_self_mutation_near_zero` — WT→WT should have minimal effect
   - `test_conserved_position_higher_effect`

**Training data:** GEMME pseudo-labels (available in VespaG repo)

### Tool 4: Topology Predictor

**Reference:** TMbed (`tools/reference/TMbed/tmbed/model.py`)

**Core architecture (from TMbed):**
```python
class CNN(nn.Module):
    # Pointwise Conv(1024→64) + LayerNorm + ReLU
    # Parallel depthwise convs: k=9 (beta), k=21 (helix), groups=64
    # Concat: 64+64+64=192 → Dropout(0.5) → Pointwise Conv(192→5)
    # Post: Gaussian smooth(σ=1,k=7) + Viterbi decoder (27 states → 7 classes)
```

**Our implementation plan:**
1. `src/one_embedding/topology.py`:
   - `TopologyPredictor` — TMbed CNN adapted for 512d input
   - `ViterbiDecoder` — grammar-constrained decoding (min TM segment lengths)

2. **Benchmark:** PDBTM (TMbed training set, cross-validated)
   - Metric: F1 for TMH/TMB/SP per-residue prediction
   - Compare: TMbed on raw ProtT5 (1024d) vs our 512d

3. **Tests:**
   - `test_predicts_7_classes`
   - `test_viterbi_enforces_min_segment_length`
   - `test_known_tm_protein` — bacteriorhodopsin should have 7 TM helices

**Training data:** PDBTM dataset (experimentally determined TM topology)

### Tool 5: Family Classifier

**Reference:** ProtTucker/EAT (`tools/reference/EAT/train_prottucker.py`)

**Core architecture (from EAT):**
```python
class ProtTucker(nn.Module):
    # Linear(1024→256) + Tanh + Linear(256→128)
    # Trained with batch-hard triplet loss on CATH hierarchy
    # Inference: k-NN (L2 distance) on learned 128d space
```

**Our implementation plan:**
1. `src/one_embedding/classifier.py`:
   - `EmbeddingClassifier` — k-NN on protein_vec (2048d) for family assignment
   - `FinetuneHead` — optional ProtTucker-style FNN for contrastive refinement
   - `FAISSIndex` — million-scale search (DCTdomain pattern)

2. **Benchmark:** SCOPe 5K (already have Ret@1=0.786), CATH (ProtTucker benchmark)
   - Metric: Ret@1, Ret@5, MRR at family/superfamily/fold level
   - Compare: raw ProtT5 mean pool vs our protein_vec (2048d) vs ProtTucker (128d)

3. **Tests:**
   - `test_knn_correct_family`
   - `test_faiss_index_recall`
   - `test_protein_vec_retrieval`

**Training data:** CATH domain labels (available in EAT repo)

### Tool 6: Disorder Predictor

**Reference:** SETH (`tools/reference/SETH/SETH_1.py`)

**Core architecture (from SETH):**
```python
class CNN(nn.Module):
    # Conv2d(1024→28, k=5, pad=2) + Tanh
    # Conv2d(28→1, k=5, pad=2)
    # Output: continuous CheZOD Z-scores (< 8 = disordered)
```

**Our implementation plan:**
1. `src/one_embedding/disorder.py`:
   - `DisorderPredictor` — SETH-style 2-layer CNN on 512d input
   - Also: Lasso regression (ADOPT-style, even simpler)

2. **Benchmark:** CheZOD-117 test set
   - Metric: Spearman ρ between predicted and experimental Z-scores
   - Compare: SETH on raw ProtT5 (1024d) vs our 512d probe

3. **Tests:**
   - `test_output_continuous_zscore`
   - `test_known_disordered_protein` — titin PEVK region should be disordered
   - `test_threshold_8_separates_order_disorder`

**Training data:** CheZOD-1325 (cleared), available via SETH/ADOPT repos

### Tool 7: Conservation Scorer

**Reference:** Kibby (`tools/reference/kibby/my_library.py`)

**Core architecture (from Kibby):**
```python
class RegressionModel:
    # Linear: conservation = intercept + dot(embedding, coef)
    # coef shape: (embedding_dim,), intercept: scalar
    # Input: per-residue embedding → Output: conservation score [0,1]
```

**Our implementation plan:**
1. `src/one_embedding/conservation.py`:
   - `ConservationScorer` — linear probe on 512d (Kibby-style)
   - `FamilyConservation` — alignment-free: stack family embeddings, compute column variance

2. **Benchmark:** Kibby's validation set (conservation from MSAs)
   - Metric: MCC between predicted and MSA-derived conservation
   - Compare: Kibby on raw ESM-2 vs our 512d probe

3. **Tests:**
   - `test_output_0_to_1_range`
   - `test_conserved_cysteines_high_score` — disulfide cysteines should be conserved
   - `test_family_variance_correlates_with_conservation`

**Training data:** MSA-derived conservation scores (Kibby pre-trained models as reference)

### Tool 8: Functional Site Predictor

**Reference:** bindEmbed21DL (`tools/reference/bindPredict/architectures.py`)

**Core architecture (from bindPredict):**
```python
class CNN2Layers(nn.Module):
    # Conv1d(in_channels → feature_channels, k=5) + ELU + Dropout(0.7)
    # Conv1d(feature_channels → 3, k=5)
    # Output: 3 binding probabilities per residue (metal, nucleic, small molecule)
```

**Our implementation plan:**
1. `src/one_embedding/binding.py`:
   - `BindingSitePredictor` — bindEmbed21-style CNN on 512d
   - `LocalAnomalyScorer` — zero-shot: embedding deviation from local context

2. **Benchmark:** BioLip (1,014 proteins, 5-fold CV)
   - Metric: F1, MCC per binding type
   - Compare: bindEmbed21 on raw ProtT5 vs our 512d

3. **Tests:**
   - `test_predicts_3_binding_types`
   - `test_known_zinc_finger_metal_binding`

**Training data:** BioLip binding annotations

### Tool 9: Structural Similarity

**Reference:** TM-Vec (`tools/reference/tm-vec/tm_vec/embed_structure_model.py`)

**Core architecture (from TM-Vec):**
```python
class trans_basic_block(nn.Module):
    # TransformerEncoder(d=1024, nhead=4, ff=2048, layers=2)
    # Masked mean pooling
    # Dropout(0.1) + Linear(1024→512)
    # Loss: L1(cosine_sim(vec_A, vec_B), TM_score)
```

**Our implementation plan:**
1. `src/one_embedding/structural_similarity.py`:
   - `TMScorePredictor` — regression from protein_vec cosine → TM-score
   - `StructuralSearchIndex` — FAISS index on protein_vec (2048d)

2. **Benchmark:** CATH domains (TM-Vec benchmark, available on Zenodo)
   - Metric: Pearson r between predicted and true TM-score
   - Compare: TM-Vec on raw ProtT5 vs our protein_vec direct cosine

3. **Tests:**
   - `test_identical_proteins_score_1`
   - `test_unrelated_proteins_low_score`
   - `test_faiss_returns_structural_neighbors`

**Training data:** CATH domain pairs with TM-scores (Zenodo)

### Tool 10: Ancestral Reconstruction

**Reference:** Draupnir (`tools/reference/DRAUPNIR_ASR/draupnir/src/draupnir/models.py`)

**Core architecture (from Draupnir):**
```python
# OU-VAE: Ornstein-Uhlenbeck process as tree-structured prior
# Encoder: BiGRU(z_dim+21 → 256) → Linear(128) → μ,σ for latent z (30d)
# Decoder: BiGRU(z_dim+21 → 256) → Linear(21) → amino acid logits
# Ancestral: conditional multivariate normal from OU covariance
```

**Our implementation plan:**
1. `src/one_embedding/ancestral.py`:
   - `BMAncestralReconstruction` — Felsenstein pruning for ML ancestral state (we already have the BM likelihood)
   - `OUAncestralReconstruction` — OU process variant (mean-reverting BM)
   - `EmbeddingToSequence` — map ancestral embeddings back to nearest AA

2. **Benchmark:** Draupnir simulated datasets (β-Lactamase, SRC SH3)
   - Metric: Hamming distance between reconstructed and true ancestral sequences
   - Compare: Draupnir on BLOSUM(21d) vs our 512d embeddings as decoder input

3. **Tests:**
   - `test_ancestral_state_between_children` — ML estimate is weighted mean
   - `test_root_reconstruction_reasonable`
   - `test_known_ancestor_recovery` — simulated data with known ancestors

**Training data:** Simulated evolution datasets (Draupnir repo)

---

## Benchmark Data Inventory

| Benchmark | Source | Size | Used By | Status |
|-----------|--------|------|---------|--------|
| BAliBASE 3 | lbgi.fr/balibase | ~400 alignments | Tool 2 (Aligner) | Need to download |
| ProteinGym | OATML-Markslab/ProteinGym | 2.7M variants | Tool 3 (Mutations) | Cloned |
| PDBTM | pdbtm.enzim.hu | ~5K TM proteins | Tool 4 (Topology) | Need to download |
| CATH | cathdb.info | ~500K domains | Tools 5,9 | Need to download |
| SCOPe 5K | Already have | 5K proteins | Tool 5 | Done |
| CheZOD | Already have (Exp 13) | 1,442 proteins | Tool 6 (Disorder) | Done |
| BioLip | zhanggroup.org/BioLiP | ~1K proteins | Tool 8 (Binding) | Need to download |
| ToxFam v2 | Already have | 84 proteins | Tool 1 (Phylo) | Done |
| TM-Vec CATH pairs | Zenodo | ~100K pairs | Tool 9 | Need to download |
| Draupnir datasets | Draupnir repo | 6 datasets | Tool 10 | Cloned |

---

## Implementation Phases

### Phase A: Low-hanging fruit (1-2 days each)
1. **Tool 7: Conservation** — Kibby is literally `intercept + dot(emb, coef)`. Train linear probe on 512d.
2. **Tool 5: Classifier** — We already have Ret@1=0.786. Add FAISS index + formal k-NN evaluation.
3. **Tool 2: Aligner** — PEbA is `cos_sim * 10 + NW`. Pure numpy, no training.

### Phase B: Medium effort (2-3 days each)
4. **Tool 6: Disorder** — SETH is 2 conv layers. Train on CheZOD (we already have the data).
5. **Tool 4: Topology** — Port TMbed CNN (change input dim). Train on PDBTM.
6. **Tool 8: Binding** — Port bindEmbed21 CNN. Train on BioLip.

### Phase C: Higher effort (3-5 days each)
7. **Tool 9: Similarity** — Train TM-score regression. Build FAISS index.
8. **Tool 3: Mutations** — Train VespaG-style FNN or use zero-shot displacement.
9. **Tool 10: Ancestral** — Extend Exp 35 with BM ancestral state reconstruction.

### Phase D: Polish
10. Publication figures, comprehensive benchmark table, write-up.
