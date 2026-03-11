# Literature Review: Protein Embedding Compression & Evaluation

This document summarizes the literature research informing our experimental approach,
evaluation methodology, and architectural decisions.

---

## 1. pLM Evaluation Methodology

### Senoner, Koludarov, Gunther, Shehu, Rost & Bromberg (2025)
**"Which pLM to choose?"** bioRxiv 10.1101/2025.10.30.685515

Benchmarks 14 pLMs across 100M protein pairs. Introduces the distinction between:
- **Inherent information**: recoverable from raw embedding distances without any training
  (measured by kNN retrieval, silhouette score, etc.)
- **Extractable information**: revealed only through supervised training (linear probes,
  fine-tuning)

Key findings:
- Mid-scale models (ESM2-650M, ESM-C-600M) match larger ones on all biological properties
- Inherent info is flat with model size; extractable info increases slowly (~0.09-0.13 per
  billion parameters)
- Size-performance paradox: spending compute on bigger models yields diminishing returns

**Relevance**: Our ProtT5 classification advantage may reflect ProtT5's superior *inherent*
family-discriminative structure, not our compressor's learning. The compressor may preserve
(or destroy less of) pre-existing signal. We adopt the inherent/extractable framework for
evaluation.

### Prabakaran & Bromberg (2025)
**"Quantifying uncertainty in Protein Representations Across Models and Tasks."**
bioRxiv 10.1101/2025.04.30.651545

Proposes **Random Neighbor Score (RNS)**: fraction of a protein's K nearest neighbors in
embedding space that are random (unrelated) sequences.

- RNS > 0.6 indicates unreliable embeddings
- Tested on Astral40, IDPs, metagenomic sequences, hallucinated sequences
- Variant impact prediction degrades beyond RNS threshold

**Relevance**: We implement RNS on compressed embeddings to measure whether compression
destroys biological signal. If AttnPool's compressed embeddings have high RNS, the collapse
is explained — compression destroys meaningful information.

### Hermann et al. (2024)
**"Beware of Data Leakage from Protein LLM Pretraining."** MLCB 2024.

- Pretraining datasets overlap with downstream test sets, inflating performance
- Proposes pretraining-aware splits
- Case study: protein thermostability prediction shows consistent leakage

**Relevance**: Our SCOPe proteins are in ESM2/ProtT5's pretraining data (UniRef). This
inflates ALL pLM-based methods equally, so relative comparisons remain valid, but absolute
numbers should be interpreted cautiously.

---

## 2. Data Splitting Best Practices

### Joeres et al. (2025)
**"DataSAIL: Data Splitting Against Information Leakage."** Nature Communications 16, 3337.

- Formulates leakage-free splitting as combinatorial optimization (NP-hard)
- Clustering + integer linear programming for scalable solution
- Python package available (`datasail`)

**Relevance**: More principled than our superfamily-based splitting. Consider for publication
preparation.

### GraphPart (2023)
**Nargab 5(4).**

- More sophisticated homology partitioning than CD-HIT
- Retains more sequences while maintaining separation

**Relevance**: Our superfamily split is reasonable but not state-of-the-art. GraphPart or
DataSAIL would be more defensible for a publication.

### Our Approach
We use **superfamily-aware splitting**: entire superfamilies are assigned to either train or
test, never both. This prevents homology leakage at the superfamily level. Stratified by
SCOPe fold class (a-g) to maintain class balance.

---

## 3. Pooling & Aggregation Approaches

### Mean Pooling (Baseline)
The default approach: average all per-residue embeddings into one vector. Simple, fast, but:
- Dilutes signal for longer sequences
- Treats all residues equally regardless of biological importance
- Loses all positional/structural information

### Naderi et al. (2024/2025)
**"Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport."**
Bioinformatics Advances 5(1).

- Treats per-residue embeddings as samples from a distribution
- Uses **sliced-Wasserstein distances (SWE)** to map against learned reference anchors
- Creates fixed-size representation that respects residue importance
- Beats mean pooling on protein-drug and PPI tasks
- Enables smaller pLMs to match larger mean-pooled pLMs

**Relevance**: SWE is a strong alternative to our attention pooling. Both learn importance
weights, but SWE has better theoretical grounding in optimal transport.

### Naderi et al. (2026)
**"EvoPool: Evolution-Guided Pooling of PLM Embeddings."** bioRxiv 10.64898/2026.02.02.703349.

- Extends SWE with evolutionary information from homologs
- Constructs evolutionary anchors from homologous sequences

**Relevance**: Our SCOPe dataset has family structure that could serve a similar role.

### Lee et al. (2019)
**"Set Transformer."** ICML 2019.

- **Pooling by Multihead Attention (PMA)**: learnable seed vectors attend to set elements
- K query vectors -> K output vectors (permutation invariant)
- Generalizes mean/max/sum pooling

**Relevance**: Our AttentionPoolCompressor IS a PMA/Set Transformer. The architecture is
well-established and sound.

---

## 4. Compact Representation Learning (Cross-Domain)

### Khattab & Zaharia (2020)
**"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction
over BERT."** SIGIR 2020.

- Keeps per-token vectors for retrieval (late interaction)
- Score = sum of max token-query similarities
- Avoids compressing all information into one vector

**Relevance**: For retrieval, we can use ALL K latent tokens directly (late interaction)
instead of mean-pooling them. This uses multi-token representations as intended. We implement
this as `evaluate_late_interaction()`.

### Kusupati et al. (2022)
**"Matryoshka Representation Learning."** NeurIPS 2022.

- Single embedding encodes info at multiple granularities
- Loss computed on all prefix dimensions simultaneously
- Up to 14x smaller embeddings at same quality

**Relevance**: Future direction — train so first N dims of each latent token are sufficient
for retrieval, while all dims enable reconstruction.

### Jaegle et al. (2021)
**"Perceiver IO."** ICLR 2022.

- Universal latent bottleneck: cross-attention from inputs to fixed-size latent space
- Self-attention only on latents (scalable)
- Arbitrary output via decoder cross-attention

**Relevance**: Our architecture is essentially a Perceiver encoder + decoder. We lack
self-attention between latent tokens — adding this could help tokens communicate and
specialize.

### Bolya et al. (2023)
**"Token Merging (ToMe)."** ICLR 2023.

- Merges similar tokens within transformer blocks (vision)
- Bipartite matching on token similarities
- Merging > pruning for information preservation

**Relevance**: Hierarchical compression alternative — merging nearby residue embeddings
before attention pooling.

### Kharitonov et al. (2020)
**"Funnel-Transformer."** NeurIPS 2020.

- Progressive sequence length reduction through layers
- Decoder recovers full-length for token-level tasks

**Relevance**: Progressive bottleneck (1280->512->256->128) might work better than our
single-step compression.

---

## 5. Protein-Specific Compression

### CHEAP (Cell Patterns, 2025)
- Compresses ESM2 embeddings for cell biology tasks
- Dimension reduction with task-specific decoders

### ProT-VAE (PNAS, 2024)
- Variational autoencoder for protein sequences
- Learns latent space that supports generation

---

## 6. Standard Benchmarks

| Benchmark | Tasks | Splitting | Status |
|-----------|-------|-----------|--------|
| **TAPE** (2019) | 5: sec struct, contact, remote homology, fluorescence, stability | Homology-aware | Foundational |
| **PEER** (2022) | 17 across 5 categories | Implicit | Comprehensive |
| **ProteinGym** (2023) | ~2.7M mutations, 217 DMS assays | Limited | Gold standard for fitness |
| **FLIP/FLIP2** (2021/2024) | Fitness landscapes, engineering | Multiple split types | Current |
| **CAFA** | GO term prediction | Temporal holdout | Community standard |
| **DeepLoc** | Subcellular localization | Standard | Established |
| **EC-Bench** | Enzyme classification | Hierarchical | Recent |

---

## 7. How This Informs Our Approach

### Architecture Validation
Our AttentionPoolCompressor = Set Transformer PMA = Perceiver encoder. This is a
well-established architecture. The problem is not architectural.

### Evaluation Improvements (Implemented)
1. **Superfamily-aware splitting** — prevents homology leakage
2. **Validation-loss checkpointing** — prevents overfitting to training set
3. **Multiple seeds** — variance estimation
4. **Held-out evaluation** — no train-set metrics
5. **RNS** — detects when compression destroys biological signal
6. **Inherent vs extractable information** — separates what's in the embeddings
   from what training adds
7. **Late interaction retrieval** — uses all K tokens instead of collapsing to one vector

### Key Insight
The critical question is not "does our compressor work?" but rather "does our compressor
preserve more information than simpler alternatives (mean pool, PCA) at the same
compression ratio?" The literature shows that mean pooling is a strong baseline that is
hard to beat without careful training and evaluation.

### Future Directions
- Self-attention on latent tokens (Perceiver-style)
- Matryoshka-style multi-scale loss
- DataSAIL/GraphPart for more principled splitting
- TAPE/ProteinGym/FLIP2 external benchmarks
- SWE (optimal transport) as primary baseline
