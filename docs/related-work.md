# Related work

A short, opinionated map of methods adjacent to OneEmbedding. Updated as we
encounter relevant prior work; not a survey.

---

## TEA — Rewriting protein alphabets with language models

Pantolini, Studer, Engist, Pudžiuvelytė, Pommerening, Waterhouse, Tauriello,
Steinegger, Schwede, Durairaj. *Rewriting protein alphabets with language models*.
bioRxiv 2025.11.27.690975 (v2 posted 2026-01-19), MLSB workshop.

**What it is.** A shallow contrastive head trained on top of ESM2 per-residue
embeddings to discretize them into a 20-letter alphabet ("TEA — The Embedded
Alphabet"). They train a custom substitution matrix and feed TEA-converted
sequences into vanilla **MMseqs2** for remote-homology search.

**Headline results (SCOPe40, the Foldseek paper benchmark).**

- TEA matches Foldseek (3Di) on family + superfamily sensitivity at MMseqs2 speed.
- TEA beats raw embedding-based alignment (EBA, ProtTucker, pLM-BLAST) on both
  speed and sensitivity.
- TEA + 3Di together outperforms either alone — orthogonal signal.
- 20 letters is the optimum in their alphabet-size ablation; below ~16 it falls
  off, above ~24 it plateaus.
- Steinegger (MMseqs2 / Foldseek author) is on the paper.

**What it means for OneEmbedding.**

A common question — "can we make MMseqs2 work on PLM embeddings?" — is **already
solved by TEA**, by the lab best-positioned to solve it. OneEmbedding does not
compete with TEA. The two systems sit at different layers:

| | OneEmbedding | TEA |
|---|---|---|
| Layer | storage / transport (lossy codec) | search-index alphabet (lossy discretization) |
| Output | per-residue (L, 896) bits + protein vector | per-residue 20-letter string |
| Training | none (centering + RP + binarization) | shallow contrastive head per PLM |
| Decoder | numpy + h5py, ~12 lines | MMseqs2 + custom substitution matrix |
| Use case | ship embeddings at scale, run probes downstream | run sequence-style remote homology search |

**For users who want sequence-style search:** use TEA (or wait until we add a
TEA-style head as a post-codec step). OneEmbedding is the bytes-on-disk and
embedding-quality layer underneath.

**Possible future experiments inspired by TEA.**

1. *Training-free alphabet from binary OE.* k-medoids or Lloyd-Max cluster the
   per-residue 896-bit codes into ~20 buckets, derive a substitution matrix
   from in-cluster Hamming distances, plug into MMseqs2. Cheap; quantifies how
   much of TEA's win is the contrastive training vs. the discretization step.
2. *TEA on top of OneEmbedding.* Apply the TEA head to compressed embeddings
   instead of raw ESM2. Measures whether 896d-binary preserves enough
   information to recover TEA-quality remote homology. This is a degradation
   test for the codec on a search-style task.

Neither is queued today. Both are clean follow-ups if the manuscript needs a
"OneEmbedding + TEA" story.

---

## RNS — Random Neighbor Score

Prabakaran & Bromberg, *Random neighbor score quantifies protein language
model embedding quality*, Nat Methods 23, 796–804 (2026).

**What it is.** A per-protein quality metric for PLM embeddings. For each
query, RNS_k = the fraction of its k nearest neighbors (in the embedding
space) that are *junkyard* — random-shuffled-residue copies of real proteins.
RNS in [0, 1]. Lower = the protein's embedding is in a biologically structured
region of space; higher = the embedding is indistinguishable from random noise.

**What it means for OneEmbedding.** RNS is a natural compression-stress test
for the codec's protein vector: if our DCT-K=4 representation is more
"random-looking" than raw mean-pool ProtT5, RNS will tell us. Implemented in
`src/one_embedding/rns.py`. Compared under compression in **Exp 48**.

**Companion direction (VEP).** The same paper shows RNS correlates with VEP
(variant effect prediction) performance. VESM (Bromberg lab, 2026) is the
designed VEP head — public weights, MIT. Earmarked but not yet benchmarked
against OneEmbedding.

---

## Foldseek + 3Di

van Kempen, Kim, Tumescheit, Mirdita, Lee, Gilchrist, Söding, Steinegger.
*Fast and accurate protein structure search with Foldseek*, Nat Biotechnol
42, 243–246 (2024).

The structural-alphabet baseline TEA targets. Uses 3Di (a 20-letter alphabet
derived from local 3D geometry) and feeds into MMseqs2. We use the SCOPe40
benchmark from this paper (via TEA's reproduction) as the comparison ground.

OneEmbedding does not compete with Foldseek directly — Foldseek requires
structures (PDB or AFDB), OneEmbedding compresses sequence-derived embeddings.
The two are complementary: Foldseek's 3Di and TEA combine well (per the TEA
paper); a OneEmbedding-derived alphabet would be a third, sequence-only
orthogonal signal.

---

## EBA / pLM-BLAST / ProtTucker

The "raw embedding-based alignment" line — score sequence pairs by aligning
their per-residue PLM embeddings directly, using a substitution-style score
derived from cosine similarity. Sensitive but slow (no k-mer indexing). TEA
beats them on the SCOPe40 benchmark on both axes.

OneEmbedding's per-residue (L, 896) reconstructed embeddings are
drop-in-compatible with this style of search at ~37× compression cost. We
have not benchmarked it; the per-residue retention is good (`pub_per_residue_retention.png`)
so any quality loss should be small.

---

## SETH / UdonPred / ADOPT / ODiNPred

The disorder-prediction line — CNN or transformer probes that take per-residue
PLM embeddings and produce per-residue Z-scores or disorder calls. We
benchmark against UdonPred (Dass et al., 2025) on CheZOD: our 0.686 pooled
Spearman ρ matches their 0.684 on the same CheZOD split (see `project_udonpred_comparison.md`
in memory; numbers reproduced in Exp 43 source JSONs).

OneEmbedding's `tools/disorder.py` is a SETH-style CNN probe, currently
trained at 768d and pending retraining at the 896d binary default.
