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

### Prabakaran & Bromberg (2026)
**"Quantifying uncertainty in protein representations across models and tasks."**
Nature Methods 23, 796–804 (April 2026). doi:10.1038/s41592-026-03028-7.
(Published version of bioRxiv 10.1101/2025.04.30.651545.)

Proposes **Random Neighbor Score (RNS)**: fraction of a protein's K nearest neighbors in
embedding space that are random (shuffled) sequences. Model-agnostic embedding quality metric.

- RNS > 0.6 indicates unreliable embeddings
- Tested on Astral40 (14,711 domains), IDPs, metagenomic sequences, hallucinated sequences
- 19.1% of ProtT5 and 46.2% of ESM-2 human proteome embeddings are "un(der)learned" (RNS > 0)
- IDPs have systematically higher RNS across ALL PLMs — disordered regions have inherently uncertain embeddings
- Variant impact prediction (VEP) AUROC degrades sharply at RNS > 0.6 (Fig. 5)
- RNS inversely correlates with TM score (Pearson ρ = −0.70 at k=500 for ESM-2)

**Relevance (updated 2026-04-13)**: RNS is now Task 9 in our Stage 3 plan. Three applications:
1. **Codec retention metric**: does compression push proteins toward the junkyard?
2. **Seq2OE evaluation**: are CNN-predicted embeddings biologically plausible?
3. **Disorder gap explanation**: the paper's IDP finding directly explains our persistent
   94.9% disorder retention — disordered regions have inherently uncertain ProtT5 embeddings.
Implementation: `src/one_embedding/rns.py` (generate_junkyard_sequences + compute_rns),
needs ~7K shuffled-sequence ProtT5 embeddings as junkyard (~1-3h MPS).

### Dinh et al. (2026)
**"Compressing the collective knowledge of ESM into a single protein language model."**
Nature Methods 2026. doi:10.1038/s41592-026-03050-9.

Proposes **VESM**: co-distill multiple ESM variants into a single sequence-only PLM that
achieves SOTA variant effect prediction without requiring structure or MSA.

- Weights public: huggingface.co/ntranoslab/VESM (MIT license for ESM2 variants)
- Variants: VESM_35M, VESM_150M, VESM_650M, VESM_3B (all ESM2-based), VESM3 (ESM3, non-commercial)
- Code: github.com/ntranoslab/vesm
- Benchmarks: ProteinGym DMS, ProteinGym ClinVar, UniProt Balanced ClinVar
- Precomputed scores: huggingface.co/datasets/ntranoslab/vesm_scores

**Relevance**: Three angles for our project:
1. **Multi-teacher precedent**: validates that co-distilling multiple PLMs into one model works
   for proteins. Our multi-teacher earmark (Exp 54) would apply the same idea to
   embedding prediction rather than VEP scoring.
2. **VEP benchmark competitor**: if we add VEP to our eval suite, VESM scores are the bar.
3. **Potential teacher**: VESM_650M could serve as an additional teacher signal alongside
   ProtT5 — it already encodes the "consensus" of multiple ESM variants.

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
5. **RNS** — detects when compression destroys biological signal.
   **STATUS (2026-04-13)**: now Task 9 in Stage 3 plan. Module at `src/one_embedding/rns.py` (planned). Uses shuffled-sequence junkyard + FAISS kNN.
6. **Inherent vs extractable information** — separates what's in the embeddings
   from what training adds. *Earmarked, not pursued.*
7. **Late interaction retrieval** — uses all K tokens instead of collapsing to one vector. *Earmarked, not pursued.*
8. **VEP benchmark** — added 2026-04-13. ProteinGym/ClinVar variant effect prediction.
   RNS paper (Prabakaran 2026) shows RNS → VEP AUROC correlation. VESM (Dinh 2026) provides SOTA baseline + precomputed scores. *Earmarked, not pursued.*

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
