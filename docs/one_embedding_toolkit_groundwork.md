# One Embedding Toolkit — Groundwork Research

10 downstream tools built on the One Embedding V2 codec (ABTT3+RP512+PQ).
All operate on the same 26 KB/protein compressed H5 file.

**Architecture:** Store compressed, decode at tool runtime. PQ decode is
a codebook lookup (~0.1ms per protein). Receiver needs only h5py + numpy +
shared codebook.

```
Compressed H5 (26 KB/protein)
├── protein_vec (2048,) fp16  ──→  Tools 1, 5, 9 (protein-level, no decode needed)
└── per_residue PQ codes (L, 128) uint8  ──→  decode ──→ (L, 512) float32
                                                         ──→  Tools 2, 3, 4, 6, 7, 8, 10
```

---

## Tool 1: Embedding Phylogenetics [DONE]

**Status:** Implemented in Experiment 35.

**Analog:** ExaBayes (C++, https://github.com/aberer/exabayes)

**What we built:** Bayesian MCMC tree inference from protein embeddings using
Brownian motion likelihood. NJ starting tree, MC3 with heated chains,
consensus with median branch lengths. 71 tests.

**Results:** 10/12 ToxProt families monophyletic from mean-pooled 512d vectors.
Clade purity 0.720. Per-residue mode gives additional signal.

**Key files:**
- `experiments/35_embedding_phylogenetics.py`
- `tests/test_embedding_phylo.py`
- ExaBayes source: `SpeciesEmbedding/tools/exabayes-src/`

---

## Tool 2: Embedding-Space Aligner

**Goal:** Align proteins by per-residue embedding similarity instead of AA identity.
Better for remote homologs (<20% sequence identity, >0.5 TM-score).

### Closest analogs

| Tool | Paper | GitHub | Approach | Key Insight |
|------|-------|--------|----------|-------------|
| **PEbA** | BMC Bioinformatics 2024 | github.com/mgtools/PEbA | Cosine similarity as substitution score + NW/SW | Simplest: `S(i,j) = cos(emb_A[i], emb_B[j])` |
| **EBA** | Bioinformatics 2024 | git.scicore.unibas.ch/schwede/EBA | Same + z-score filtering | Z-score normalization dramatically improves twilight zone |
| **PLMAlign** | Nature Comms 2024 | github.com/maovshao/PLMAlign | Dot product + SW | Part of PLMSearch pipeline |
| **DEDAL** | Nature Methods 2023 | google-research/dedal | Differentiable SW, end-to-end learned | Joint optimization of embeddings + alignment |
| **vcMSA** | Genome Research 2023 | github.com/clairemcwhite/vcmsa | Clustering per-residue embeddings for MSA | No DP at all — graph-theoretic MSA |
| **ARIES** | bioRxiv Jan 2026 | (preprint) | DTW on PLM embeddings | No gap penalties needed |
| **CLAlign** | bioRxiv 2024 | (preprint) | LoRA contrastive fine-tuning + NW | 14-24% over PLMAlign with 10K training pairs |

### Our approach

Start with PEbA-style (simplest): cosine similarity between our 512d RP embeddings
as substitution score, standard NW with affine gap penalties. Add EBA's z-score
filtering. Our compressed (L, 512) embeddings preserve cosine similarity (verified
by 97% SS3 Q3 retention).

**Wild idea:** PQ codes (128 bytes/residue) as a structural alphabet (like
Foldseek's 3Di). String matching on PQ code sequences for ultra-fast pre-filtering,
then full embedding alignment on candidates.

### What to clone/study

```bash
git clone https://github.com/mgtools/PEbA.git      # Simplest, ProtT5 cosine + NW
git clone https://github.com/clairemcwhite/vcmsa.git  # Clustering MSA, no DP
```

---

## Tool 3: Mutational Landscape Scanner

**Goal:** Predict embedding-space effect of every possible single-point mutation.
Zero-shot deep mutational scanning from the compressed embedding.

### Closest analogs

| Tool | Paper | GitHub | Approach | Spearman (ProteinGym) |
|------|-------|--------|----------|----------------------|
| **ESM-1v** | PNAS 2021 | facebookresearch/esm | Masked marginals scoring | ~0.51 |
| **VespaG** | NAR 2024 | JSchlensok/VespaG | 660K-param NN on ESM-2 per-residue embeddings | 0.48 |
| **VESPA** | (Rostlab) | Rostlab/VESPA | ProtT5 embeddings → conservation → SAV effect | — |
| **EVE** | Nature 2021 | OATML-Markslab/EVE | Bayesian VAE on MSA, evolutionary index | ~0.50 |
| **GEMME** | MBE 2019 | lcqb.upmc.fr/GEMME | Evolutionary model from MSA | ~0.50 |
| **Evolocity** | Cell Systems 2022 | brianhie/evolocity | Evolutionary velocity in embedding space | — |
| **ProteinGym** | NeurIPS 2023 | OATML-Markslab/ProteinGym | Benchmark: 2.7M variants, 217 DMS assays | — |

### Our approach

Two levels:
1. **Fast (no PLM re-run):** For each position, compute the "embedding sensitivity" —
   how much does removing/masking this residue shift the neighboring embeddings?
   Approximate from the per-residue embedding gradients stored in the compressed file.
2. **Full (PLM re-run):** Mask each position, re-run ProtT5, get ΔEmbedding in 512d
   codec space. 19 × L forward passes. VespaG shows a shallow NN on per-residue
   embeddings matches SOTA at 0.48 Spearman.

**Key insight from VespaG:** A 660K-parameter shallow NN trained on per-residue
embeddings (not re-running the PLM) achieves competitive variant effect prediction.
We could train a similar probe on our 512d compressed embeddings.

### What to clone/study

```bash
git clone https://github.com/JSchlensok/VespaG.git   # Shallow NN on per-residue embeddings
git clone https://github.com/brianhie/evolocity.git    # Evolutionary velocity in embedding space
git clone https://github.com/OATML-Markslab/ProteinGym.git  # Benchmark suite
```

---

## Tool 4: Structure-Free Topology Predictor

**Goal:** Predict transmembrane topology (inside/TM-helix/outside) from per-residue
512d vectors. Port of TMbed concept to compressed embedding space.

### Closest analogs

| Tool | Paper | GitHub | Approach | Performance |
|------|-------|--------|----------|-------------|
| **TMbed** | BMC Bioinf 2022 | BernhoferM/TMbed | 1D-CNN + CRF on ProtT5 per-residue embeddings | SOTA for TM prediction |
| **DeepTMHMM** | bioRxiv 2022 | (DTU server) | Transformer + HMM on ESM-2 | Comparable to TMbed |
| **SETH** | Frontiers Bioinf 2022 | Rostlab/SETH | CNN on ProtT5 for SS3/SS8/disorder jointly | Q3=0.86 SS3 |

### Our approach

TMbed uses a lightweight 1D-CNN on 1024d ProtT5 embeddings. We already showed 88%
TMbed F1 retention in compressed 512d space (Experiment 15). A simple LogisticRegression
probe or 2-layer CNN on our (L, 512) decoded embeddings should give ~0.65+ F1.

The key advantage: runs from the 26 KB compressed file, no PLM needed at inference.

### What to clone/study

```bash
git clone https://github.com/BernhoferM/TMbed.git  # Architecture to port
```

---

## Tool 5: Family/Fold Classifier

**Goal:** Assign a new protein to its family/superfamily/fold using protein_vec
k-NN retrieval. Already essentially done — Ret@1=0.786 on SCOPe.

### Closest analogs

| Tool | Paper | GitHub | Approach |
|------|-------|--------|----------|
| **ProtTucker/EAT** | NAR Genomics 2022 | Rostlab/EAT | Contrastive ProtT5 FNN + k-NN on CATH |
| **knnProtT5** | Frontiers 2022 | — | Raw ProtT5 mean pool + k-NN (our baseline) |
| **CLEAN** | Science 2023 | tttianhao/CLEAN | Contrastive ESM-1b for EC number prediction |
| **DHR** | Nature Biotech 2024 | ml4bio/Dense-Homolog-Retrieval | Dual-encoder ESM + contrastive, 56% sensitivity gain |
| **DCTdomain** | Genome Research 2024 | mgtools/DCTdomain | DCT on ESM-2 + FAISS (remarkably similar to our codec!) |

### Our approach

Already have Ret@1=0.786 with DCT K=4 protein_vec (2048d). For production:
- FAISS index for million-scale search (DCTdomain pattern)
- Multi-label support for proteins in multiple families (CLEAN pattern)
- Domain-aware splitting for multi-domain proteins (DCTdomain/Merizo pattern)

**DCTdomain is strikingly similar to our codec** — they also use DCT on per-residue
PLM embeddings. Main difference: they do domain segmentation first.

### What to clone/study

```bash
git clone https://github.com/Rostlab/EAT.git          # ProtTucker annotation transfer
git clone https://github.com/mgtools/DCTdomain.git     # DCT fingerprints + FAISS (closest to us)
git clone https://github.com/tttianhao/CLEAN.git       # EC number contrastive learning
```

---

## Tool 6: Embedding Disorder Predictor

**Goal:** Predict intrinsically disordered regions from per-residue 512d vectors.
We showed CheZOD ρ=0.584 from PQ-compressed embeddings.

### Closest analogs

| Tool | Paper | GitHub | Approach | Performance |
|------|-------|--------|----------|-------------|
| **SETH** | Frontiers Bioinf 2022 | Rostlab/SETH | CNN on ProtT5, predicts SS + disorder jointly | CheZOD ρ~0.65 |
| **ODiNPred** | Nature Methods 2020 | protein-nmr/odinferno | Gradient boosting on PLM features | CheZOD ρ~0.64 |
| **flDPnn** | NAR 2021 | — | MLP on ESM-1b embeddings | AUC ~0.85 |
| **ESMDisPred** | 2024 | wasicse/ESMDisPred | CNN+Transformer on ESM2 | Top at CAID3 |
| **IUPred3** | NAR 2022 | (web) | Energy-based, no PLM | Classic baseline |

### Our approach

SETH uses a simple CNN on 1024d ProtT5. We'd port the architecture to 512d input.
Since our compressed embeddings retain 97% of per-residue information (Q3=0.807),
disorder prediction should transfer well. The advantage: runs from compressed file.

### What to clone/study

```bash
# SETH source is part of Rostlab tools
git clone https://github.com/Rostlab/SETH.git   # If available
```

---

## Tool 7: Conservation Scorer (Alignment-Free)

**Goal:** Predict per-position conservation from a single sequence's embeddings.
No MSA needed.

### Closest analogs

| Tool | Paper | GitHub | Approach |
|------|-------|--------|----------|
| **Kibby** | Brief Bioinf 2023 | esbgkannan/kibby | Linear regression on ESM2 per-residue → conservation |
| **VESPA conservation** | Human Genetics 2022 | Rostlab/VESPA | ProtT5 per-residue → conservation (MCC=0.596) |

### Our approach

Kibby shows a **linear probe** on per-residue embeddings predicts conservation nearly
as well as an actual MSA (MCC comparable to ConSeq). Our 512d compressed embeddings
should retain this signal. Two approaches:

1. **Probe-based:** Train linear regression from 512d → conservation score (Kibby-style)
2. **Embedding variance:** For a family of proteins, stack per-residue embeddings,
   compute column-wise variance. Low variance = conserved. Works without any training.

### What to clone/study

```bash
git clone https://github.com/esbgkannan/kibby.git  # Linear conservation from embeddings
```

---

## Tool 8: Functional Site Predictor

**Goal:** Identify active sites, binding sites, PTM sites from per-residue embedding
patterns.

### Closest analogs

| Tool | Paper | GitHub | Approach |
|------|-------|--------|----------|
| **bindEmbed21DL** | (Rostlab) | Rostlab/bindPredict | 2-layer CNN on ProtT5 → metal/nucleic/small molecule binding |
| **ScanNet** | Nature Methods 2022 | jertubiana/ScanNet | Geometric DL on 3D structure → binding sites |
| **IDBindT5** | Sci Reports 2024 | — | ProtT5 → binding in disordered regions |

### Our approach

bindEmbed21DL uses a simple 2-layer CNN on 1024d ProtT5 per-residue embeddings to
predict binding residues. Directly portable to our 512d. For a zero-shot approach
(no training): compute local embedding anomaly score — how much does each residue's
embedding differ from its ±5 neighbors? Spikes indicate functional importance.

### What to clone/study

```bash
git clone https://github.com/Rostlab/bindPredict.git  # Binding site prediction from ProtT5
```

---

## Tool 9: Fast Structural Similarity Search

**Goal:** Approximate TM-score from protein_vec distances without 3D structures.

### Closest analogs

| Tool | Paper | GitHub | Approach | TM-score correlation |
|------|-------|--------|----------|---------------------|
| **TM-Vec** | Nature Biotech 2023 | tymor22/tm-vec | Twin NN on ProtT5 → cosine predicts TM-score | r=0.936 |
| **TM-Vec 2** | bioRxiv Feb 2026 | — | BiLSTM distillation, lower cost | Improved |
| **Rprot-Vec** | BMC Bioinf 2025 | SuperZyccc/RProt-vec | GRU+CNN on ProtT5, 41% fewer params than TM-Vec | Improved |
| **PLMSearch** | Nature Comms 2024 | maovshao/PLMSearch | NN structural similarity from ESM-1b | 3x MMseqs2 sensitivity |
| **DCTdomain** | Genome Res 2024 | mgtools/DCTdomain | DCT fingerprints + FAISS | 0.1s/query on CPU |

### Our approach

TM-Vec trains a twin NN on top of ProtT5 per-residue embeddings. We could:
1. Train a regression head mapping our protein_vec cosine similarity → TM-score
2. Use DCTdomain's FAISS pattern for million-scale search
3. Our protein_vec (2048d) is already a DCT K=4 fingerprint — remarkably similar
   to DCTdomain's approach

### What to clone/study

```bash
git clone https://github.com/tymor22/tm-vec.git       # TM-score from embeddings
git clone https://github.com/SuperZyccc/RProt-vec.git  # Lighter alternative
git clone https://github.com/mgtools/DCTdomain.git     # DCT fingerprints (closest to us)
```

---

## Tool 10: Ancestral Embedding Reconstruction

**Goal:** Given a phylogenetic tree + per-residue embeddings at leaves, reconstruct
ancestral embeddings at internal nodes via Brownian motion.

### Closest analogs

| Tool | Paper | GitHub | Approach |
|------|-------|--------|----------|
| **Draupnir** | ICLR 2022 | LysSanzMoreta/DRAUPNIR_ASR | OU-VAE on phylogeny, continuous latent space |
| **LASE** | Nature MI 2024 | RSCJacksonLab/local-ancestral-sequence-embeddings | mASR + small PLM, smoother fitness landscapes |
| **GRASP** | (Boden Lab) | bodenlab/GRASP | ML ancestral reconstruction with POGs for indels |
| **ArDCA ASR** | MBE 2025 | (Julia) | Autoregressive DCA, epistasis-aware |
| **phytools/Rphylopars** | R packages | ericgoolsby/Rphylopars | Multivariate BM/OU ancestral reconstruction |
| **Evolocity** | Cell Systems 2022 | brianhie/evolocity | Evolutionary velocity in embedding space |

### Our approach

**Draupnir** is the closest — it does ancestral reconstruction in continuous latent
space using an Ornstein-Uhlenbeck process on a phylogenetic tree. Our Experiment 35
already has the BM likelihood and Felsenstein pruning. Adding ancestral state
reconstruction is a natural extension:

1. Build tree from embeddings (Tool 1, done)
2. At each internal node, the BM maximum-likelihood ancestral state is a
   weighted average of descendant values (weights = inverse branch lengths)
3. Reconstruct (L, 512) ancestral embeddings at each internal node
4. Optionally decode back to approximate ancestral sequence using nearest-neighbor
   in embedding→AA mapping

The `phytools::fastAnc()` algorithm gives O(n) ML ancestral reconstruction under
BM — simple weighted average propagation, already implicit in our Felsenstein pruning.

### What to clone/study

```bash
git clone https://github.com/LysSanzMoreta/DRAUPNIR_ASR.git  # OU-VAE ancestral reconstruction
git clone https://github.com/RSCJacksonLab/local-ancestral-sequence-embeddings.git  # LASE
git clone https://github.com/brianhie/evolocity.git  # Evolutionary velocity
git clone https://github.com/bodenlab/GRASP.git  # Classical ML ASR with POGs
```

---

## Priority Implementation Order

| Phase | Tools | Effort | Why First |
|-------|-------|--------|-----------|
| **Phase A** | 2 (Aligner), 7 (Conservation) | Low | Core capabilities, enable other tools |
| **Phase B** | 5 (Classifier), 9 (Similarity) | Low-Med | Already mostly done, add FAISS |
| **Phase C** | 4 (Topology), 6 (Disorder), 8 (Binding) | Med | Per-residue probes, port existing architectures |
| **Phase D** | 3 (Mutations), 10 (Ancestral) | Med-High | Novel applications, publishable |

## Repos to Clone (All)

```bash
cd /Users/jcoludar/CascadeProjects/SpeciesEmbedding/tools

# Aligner
git clone https://github.com/mgtools/PEbA.git
git clone https://github.com/clairemcwhite/vcmsa.git

# Mutation
git clone https://github.com/JSchlensok/VespaG.git
git clone https://github.com/OATML-Markslab/ProteinGym.git

# Topology/Disorder
git clone https://github.com/BernhoferM/TMbed.git

# Classifier/Similarity
git clone https://github.com/Rostlab/EAT.git
git clone https://github.com/mgtools/DCTdomain.git
git clone https://github.com/tymor22/tm-vec.git

# Ancestral
git clone https://github.com/LysSanzMoreta/DRAUPNIR_ASR.git
git clone https://github.com/brianhie/evolocity.git

# Conservation
git clone https://github.com/esbgkannan/kibby.git

# Binding sites
git clone https://github.com/Rostlab/bindPredict.git
```
