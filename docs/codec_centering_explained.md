# Centering in the OneEmbeddingCodec — what it is, what you provide

> User-facing reference. Explains the first transformation the codec applies
> (subtracting a corpus-mean vector), what data you need to provide, and what
> the codec stores. Written for users running their own PLM and considering
> the codec for their data, plus internal contributors who want the precise
> reference.

## TL;DR

The codec subtracts a **fixed (D,)-dimensional reference vector** from every
embedding before doing any other transformation. This vector is the
per-dimension mean of a representative corpus of embeddings from your PLM.

You provide that corpus **once**, by running your PLM on a few hundred to a
few thousand proteins of your choosing. The codec computes the mean during
`fit()`, stores it, and uses it forever after.

```python
codec = OneEmbeddingCodec()
codec.fit(corpus_embeddings)  # one-time. corpus_embeddings = {pid: (L, D) array}
                              # 200-3,000 proteins is plenty.
codec.encode(any_new_protein) # works immediately on any embedding from the same PLM
```

You do **not** need labels, fine-tuned models, or special datasets. Any
diverse-ish set of proteins from the target PLM is fine.

---

## Why we center at all

PLMs learn a *biased* distribution — they don't output zero-mean vectors. The
mean is per-channel (some hidden dimensions reliably skew positive, others
negative), and that bias eats into the signal-to-noise of every downstream
transformation in the codec:

- **Random projection** would waste capacity preserving the bias.
- **Quantization** loses precision near the bias-shifted region.
- **DCT-K** would put a huge constant into the DC coefficient.

Subtracting a representative mean isolates the *deviation* — the part of each
embedding that varies *between* proteins, residues, contexts. That deviation
is the signal. The mean itself is uninformative once you know which PLM you
have.

This is the same logic as standardising data in any ML pipeline: remove the
shared offset before learning structure.

## What you provide

A dict of embeddings:

```python
corpus = {
    "P12345": np.array(...),  # (L_1, D) per-residue embedding for protein 1
    "Q67890": np.array(...),  # (L_2, D) per-residue embedding for protein 2
    ...
}
```

- **Same PLM** as the embeddings you'll later encode. ProtT5 corpus → ProtT5 codec → ProtT5 inputs only. (Mixing PLMs would give you a wrong mean.)
- **Any reasonable diversity**. SCOPe representatives, UniRef50 random sample, your favourite proteome — all fine.
- **Lengths can vary**. The codec internally concatenates across the residue dimension and averages.
- **No labels needed**.
- **No fine-tuning needed**.

## How much data is enough

Mean estimation error scales as `1/√(n_residues)`. The means themselves are
small relative to typical embedding magnitudes (raw values are O(0.1–1)), so
you don't need much.

| Corpus size | Total residues (≈) | Mean stable to ± | Verdict |
|---|---:|---:|---|
| 50 proteins | 12K | 0.01 | works (noisy but fine) |
| 200 proteins | 50K | 0.005 | recommended floor |
| 500 proteins | 125K | 0.003 | typical choice |
| 3,000 proteins | 750K | 0.001 | very stable |
| 20,000 proteins | 5M | 0.0005 | overkill — sub-noise gains |

Practical guidance:
- **If you have a 3K-protein dataset**: use it as-is. You're well within the stable region.
- **If you have a 20K proteome**: also fine — diminishing returns, but no harm.
- **If you have 50 proteins**: still use them; the codec is not sensitive to mean precision.
- **If you have 5 proteins**: enough to run the math, but consider grabbing a public corpus for a slightly more representative mean.

## What the codec stores

After `codec.fit(corpus)`:

| What | Shape | Size |
|---|---|---|
| Centering mean | `(D,)` float32 | 4 KB for D=1024 |
| (PQ codebook, only if quantization='pq') | `(M, K, D/M)` | ~1–4 MB |
| (Random projection matrix, deterministic from seed) | not stored | recomputed from seed |

For the default `quantization='binary'` codec, the only persistent state is
the **(D,)** mean vector. Tiny. No codebook needed at decode time.

## Per-PLM, not per-input

A common point of confusion:

| Question | Answer |
|---|---|
| Is the mean re-computed for each input? | **No.** Frozen after `fit()`. |
| Is the mean re-computed for each batch? | **No.** Frozen after `fit()`. |
| Is the mean different for ProtT5 vs ESM2? | **Yes.** Each PLM has its own mean. |
| If I encode protein A and then protein B, do they share the mean? | **Yes.** That's the point. |

Self-centering each input would *destroy* the per-protein bias signal — which
contains real information about where each protein lives in embedding space.
The fixed corpus mean preserves that signal while removing the PLM-wide bias.

## End-to-end workflow

```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec

# Step 1: extract embeddings for your fit corpus
# (your PLM, your choice of proteins; one-time cost)
corpus_seqs = load_my_proteins()         # ~500-3000 proteins
corpus = {pid: prot_t5(seq) for pid, seq in corpus_seqs.items()}
# corpus is a dict {pid: (L, 1024) ndarray}

# Step 2: fit the codec
codec = OneEmbeddingCodec()              # binary 896d default
codec.fit(corpus)                        # learns the (1024,) corpus mean
# codec is now ready. You can save it if you like:
# codec.save_codebook("prot_t5_codec.h5")  # optional — only needed for PQ

# Step 3: encode anything (forever)
new_emb = prot_t5("MYSEQUENCE...")       # any protein, any time
encoded = codec.encode(new_emb)          # (L, 896) bits packed into bytes
```

## Pre-fitted codec story (planned)

A common pattern in the future will be: instead of running `fit()` on your
own corpus, you fetch a pre-computed mean for one of the standard PLMs:

```python
# planned API — not built yet
codec = OneEmbeddingCodec.load_pretrained("prot_t5_xl")  # fetches stored mean
codec.encode(my_protein)                                  # works immediately
```

The bundled means would each be ~4 KB and would save first-time users from
needing to provide a corpus. Tracked as the "pip-installable package" item
in `MEMORY.md`. For research code today, calling `.fit()` on a corpus you
have lying around is the path.

## What NOT to do

- ❌ **Don't fit on one PLM and encode with another.** ProtT5's mean is wrong
  for ESM2. The codec will silently produce garbage.
- ❌ **Don't center each input by its own mean.** That destroys the
  per-protein bias signal which carries real information.
- ❌ **Don't expect the mean to be "the average protein."** It's not — it's
  the average of all *residues* (concatenated), pooled across proteins.
- ❌ **Don't worry about which proteins you put in the corpus.** Diverse is
  preferred but uniformity-of-source matters more than identity. SCOPe
  representatives, random UniRef draws, a custom proteome — all fine.
- ❌ **Don't refit on every batch.** Fit once, save the codec, reuse forever.

## What COULD go wrong (rare)

- **PLM updated to a new version** with a different output distribution. Refit.
- **Tokenization conventions change** (e.g., a different special-token
  handling). Symptoms: the mean shifts dramatically. Refit.
- **Corpus is way too small** (<20 proteins). The mean estimate has high
  variance. Symptoms: encode→decode round-trip noise. Use more proteins.

## In summary

Centering is the simplest piece of the codec — a one-time precomputation, a
(D,)-vector subtraction at every encode. You provide a few hundred to a few
thousand proteins from your PLM, the codec stores their per-channel mean,
and forever after, that mean is subtracted before everything else.

If you're a downstream user with a 3K-protein dataset or a 20K proteome:
just hand it over. You have more than enough.

---

## Related documents

- `docs/exp55_vep_retention.md` — recent retention benchmark (5th task family) using this codec
- `docs/figures/exp55_transformations_demo.png` — visual walkthrough of all 4 transformations including centering
- `CLAUDE.md` — project overview, codec API summary, retention numbers
- `src/one_embedding/codec_v2.py` — implementation, `OneEmbeddingCodec.fit()`
- `src/one_embedding/preprocessing.py` — `compute_corpus_stats()`, `center_embeddings()`
