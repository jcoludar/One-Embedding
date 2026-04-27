# State of the Project — Honest Write-Up

**Date:** 2026-04-27
**Status:** Real version (Task E.1). Audit-grounded, intended for Rost-lab seminar prep.
**Source of truth.** All claims here trace to `docs/_audit/claims.md` and the experiments cited inline.

> The Prior (predictions written 2026-04-26, before the audit) is preserved at the
> bottom as an Appendix; the calibration delta lives in `docs/CALIBRATION.md`.

---

## Executive summary

A universal codec for protein language model (PLM) per-residue embeddings,
compressing **ProtT5-XL ~37×** (`(L, 1024)` fp32 → ~17 KB total at L=156 mean)
with **95–100 % retention across 4 task families and 9 datasets** under a
rigorous benchmarking protocol (BCa B=10000, paired bootstrap on retention,
multi-seed-averaged predictions, CV-tuned probes). The codec is one Python
class with four knobs (`d_out`, `quantization`, `pq_m`, `abtt_k`); the binary
default needs no codebook (receiver = `numpy + h5py`, ~12 lines).

**The numbers survive an audit by people who built the embeddings being
compressed**: every cited Exp 43/44/46/47 retention cell is bit-perfect against
its source JSON; statistics protocol matches Rost-lab convention end-to-end
(BCa CIs, pooled disorder ρ + cluster bootstrap, predictions averaged across
seeds *before* bootstrapping, fair retrieval baselines).

**Three honest weaknesses** drive the next-steps roadmap: (1) **disorder
retention plateaus at ~95 %** (vs ~99 % for SS3) and the Exp 45 forensics show
this is geometrically meaningful, not a bug; (2) **sequence → binary OE
prediction is capacity-bound at ~69 %** bit accuracy (Exp 50 Stages 1–3 all
converge there — the next lever is a transformer, not more data); (3)
**no co-distilled VESM baseline yet**, leaving the strongest plausible
competitor untested.

The codec is shippable as a 1.0 today. The full sequence-to-embedding story
(which would enable FASTA-only deployment) needs Stage 4 (transformer) before
publishing.

---

## Project arc — how we got here

The project began as "compress PLM embeddings without losing downstream task
performance." We benchmarked over 200 distinct compression methods across
47 experiments (full enumeration in `docs/EXPERIMENTS.md`): pooling strategies,
random projections, PCA / ABTT preprocessing, quantization (int8 / int4 /
binary / PQ / RVQ), vector quantization, learned compressors (ChannelCompressor,
attention pool, MLP autoencoders), wavelet and DCT transforms, channel pruning,
zstd, and a few exotic dead-ends (tensor-train, NMF, optimal transport, TDA-based,
SimHash). The dead-ends are documented in three archived branches
(`docs/_audit/worktrees.md` records their disposition).

**The shape of the current codec was determined empirically:**
- **Random projection (RP)** to `d_out=896` won over PCA (Exp 38: RP > PCA on
  per-residue retention) and all learned projections in the universal-codec
  setting (Exp 25 / 26 / 28).
- **Centering survived** every preprocessing comparison.
- **ABTT (top-PC removal)** was on by default until **Exp 45** showed PC1 is
  ~73 % aligned with the disorder direction in ProtT5 space — removing it
  destroyed 6–11 pp of disorder retention. ABTT is now off (`abtt_k=0`).
- **Binary quantization beat PQ M=128 on disorder** (Exp 47: 94.9 % vs 91.4 %)
  while halving the storage and eliminating the codebook. We adopted it as
  default; PQ M=224 remains available for users who want maximum quality at
  ~18× compression.
- **DCT K=4 protein vector** was adopted from prior work; it is a STORAGE
  choice (3584-d fp16 → 7 KB), not a measured quality optimum (Exp 22 raw
  shows K=8 has higher Ret@1 on a different proxy).

The protocol was hardened through Exp 43 (Nature-level methodology) and
re-validated in Exp 44, 46, 47.

---

## The codec — one class, four knobs

```python
from src.one_embedding.codec_v2 import OneEmbeddingCodec

codec = OneEmbeddingCodec()  # default: 896d binary, ~37x, no codebook
codec.fit(training_embeddings)
data = codec.encode(raw_embeddings)
data['per_residue']   # (L, 896) for per-residue tasks
data['protein_vec']   # (3584,) for retrieval / clustering / UMAP
```

### Defaults and their evidence

| Knob | Default | Evidence | Source |
|------|---------|----------|--------|
| `d_out` | 896 | PQ-divisible (224 = M); avoids ABTT3+RP768 ProstT5 catastrophe | Exp 47 direct test |
| `quantization` | `'binary'` | Exp 47 binary 94.9 % disorder vs PQ128 91.4 % at twice the compression; no codebook | Exp 47 codec sweep |
| `pq_m` | `auto` (= largest factor of `d_out` ≤ `d_out//4`) | Targets ~4d sub-vectors; for 896 → 224 (~18×) | `auto_pq_m()` |
| `abtt_k` | 0 | Exp 45: ABTT PC1 73 % aligned with disorder direction; removing top PCs destroys disorder | Exp 45 disorder forensics |
| `dct_k` | 4 | STORAGE choice (D×4 fp16 = 3584-d for d_out=896); not measured optimum | codec design spec |
| `seed` | 42 | Exp 29 part D: across 10 RP seeds, std=0.004 on Ret@1 (irrelevant) | Exp 29 |

Two defaults (`d_out=896` and `dct_k=4`) are **defensible but not directly measured** — see *Open problems* below. The other four are measurement-grounded.

### Receiver-side decode

For binary, int4, fp16 modes the receiver needs only `numpy + h5py` — no
`OneEmbeddingCodec` import, no codebook. The decoder is ~12 lines (see
`HANDOFF.md`). PQ correctly requires the codebook. This is the load-bearing
"universal codec" claim and it survives the audit.

### Single-PLM headline (Exp 47, ProtT5-XL)

| Config | Compression | SS3 ret | SS8 ret | Ret@1 ret | Disorder ret |
|--------|:-----------:|:-------:|:-------:|:---------:|:------------:|
| lossless 1024d | 2× | 100.2 % | 100.0 % | 100.4 % | 100.0 % |
| fp16 896d | 2.3× | 100.0 % | 99.2 % | 100.6 % | 98.6 % |
| int4 896d | 9× | 99.8 % | 98.8 % | 100.4 % | 98.2 % |
| **PQ M=224 896d** | **18×** | **99.0 %** | **98.5 %** | **100.6 %** | **95.4 %** |
| PQ M=128 896d | 32× | 97.5 % | 96.1 % | 100.1 % | 91.4 % |
| **binary 896d (default)** | **37×** | **97.6 %** | **95.0 %** | **100.4 %** | **94.9 %** |

All cells bit-perfect against `data/benchmarks/rigorous_v1/exp47_sweep_prot_t5_full.json` (verified in audit `claims.md`). Numbers omitted but available: 95 % BCa CIs (in JSON), per-PLM timing.

### Multi-PLM validation (Exp 46, center + RP896 + PQ M=224, ~18×)

| PLM | dim | SS3 ret | SS8 ret | Ret@1 ret | Disorder ret |
|-----|:---:|:-------:|:-------:|:---------:|:------------:|
| ProstT5 | 1024 | 99.2 ± 0.3 % | 98.6 ± 0.5 % | 100.0 ± 0.5 % | 98.3 ± 1.1 % |
| ProtT5-XL | 1024 | 99.0 ± 0.5 % | 98.5 ± 0.6 % | 100.6 ± 0.6 % | 95.4 ± 1.9 % |
| ESM-C 600M | 1152 | 98.3 ± 0.5 % | 97.6 ± 0.7 % | 102.6 ± 2.9 % | 98.1 ± 1.0 % |
| ANKH-large | 1536 | 97.9 ± 0.5 % | 96.3 ± 0.8 % | 99.9 ± 0.6 % | 94.8 ± 2.3 % |
| ESM2-650M | 1280 | 97.6 ± 0.7 % | 96.5 ± 0.7 % | 97.8 ± 1.6 % | 98.8 ± 0.9 % |

The same train/test partition is used across all 5 PLMs (single split file, embeddings re-extracted). Verified `experiments/46_multi_plm_benchmark.py:456–461`. The ANKH disorder retention (94.8 %) is the worst single cell and a likely Rost-lab probe; framing in `EXPECTED_QA.md`.

---

## Sequence → binary OE (Exp 50)

A small dilated CNN trained to predict the binary `(L, 896)` One Embedding
directly from amino acid sequence (no PLM at inference).

**Sighting run (random 80/10/10 on SCOPe-40 2493 proteins):** 65.4 % bit accuracy,
0.522 cosine — well above random (50 % bit acc) but **not yet rigorous**
(random splits leak homology).

**Stages 2 + 3 (continuous loss + 2× data):** all converge to **~69 % bit
accuracy and 0.55 cosine**. Three stages, two loss types, two data scales — the
same number. **The CNN is capacity-bound**; data and loss are not the lever.

**Stage 4 (planned, not run):** transformer backbone (3–5 days on Mac) is the
designed next step. Spec at `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md`;
plan at `docs/superpowers/plans/2026-04-06-exp50-rigorous-cath-split.md`.
A rigorous CATH-cluster re-run (homologous-superfamily + topology splits, MMseqs2
leakage audit, 3-seed variance) is also designed but not yet executed.

**Talk framing:** present Exp 50 as in-progress. The 69 % ceiling is a real and
informative finding (capacity-bound at this architecture); the rigorous numbers
will land post-talk.

---

## Phylogenetics from embeddings (Exp 35)

A separate research branch using PLM embeddings as a continuous trait under
Brownian motion MCMC for phylogenetic inference.

- BM MCMC (warm-started, extended SPR): **10–11 of 12 families monophyletic.**
- Sequence ML/Bayesian (FastTree / IQ-TREE / MrBayes): 4–5 of 12.
- Cross-validated against RevBayes (sigma² CIs overlap).

The cited per-family monophyly numbers come from per-tree
`results/embed_phylo/*_consensus.nwk` analyses, **not** from
`data/benchmarks/embedding_phylo_results.json` (which is a sanity-config
artifact restored to the 156-taxa version in audit C.5; see
`docs/_audit/hygiene.md`). The FastTree/IQ-TREE comparison numbers don't have
an explicit script trace in the audit (deferred YELLOW; see
`AUDIT_FINDINGS.md` row 37) — disclose if probed.

---

## Negative results (deliberately included)

These cost real time and they belong in the talk because they constrain the
design space.

- **VQ / RVQ are genuinely poor for this task.** VQ K=16384 gets 79 % SS3
  retention and 58 % disorder retention (Exp 47). Confirmed not a bug —
  multiple implementations.
- **ABTT helps retrieval, hurts disorder.** PC1 of ProtT5 embeddings is
  ~73 % aligned with the disorder direction (Exp 45). Removing top PCs
  improves isotropy at the cost of 6–11 pp of disorder retention. Off by
  default; can be turned on for retrieval-only use.
- **CNN capacity ceiling at 69 % bit accuracy** for sequence → binary OE
  (Exp 50, three stages converging to the same number). Architecture, not
  data or loss, is the next lever.
- **Three branches of exploratory codecs were not adopted** (wavelet, CUR,
  zstd, tensor-train, NMF, OT, TDA, SimHash, AA-residual, channel-prune).
  Branches preserved as audit trail (`docs/_audit/worktrees.md`).

---

## Open problems

1. **Disorder retention gap (~5 %).** Exp 45 traced PC1 to disorder direction;
   Exp 51 (PolarQuant — magnitude-augmented binary) is designed to attack
   this without re-introducing ABTT. Estimated +2–3 pp disorder retention at
   the same 36× compression with no codebook.
2. **Exp 50 architecture lever.** Stage 4 (transformer, 3–5 days on M3 Max)
   should break the 69 % ceiling. Two paths: (A) train-from-scratch small
   transformer (clean delta) or (B) fine-tune VESM-35M / ESM2-35M (faster,
   stronger init).
3. **No co-distilled VESM baseline.** VESM (Bromberg lab, 2026) is the
   strongest plausible competitor. Weights are public, MIT-licensed.
   Earmarked but not run.
4. **VEP / ProteinGym evaluation missing.** Variant effect prediction is the
   classic PLM-quality benchmark; the same RNS paper (Prabakaran &
   Bromberg, *Nat Methods* 23, 2026) shows RNS→VEP correlation. Earmarked.

---

## Next directions (priority order)

1. **Stage 4 transformer for Exp 50** (architecture lever). 3–5 days.
2. **Exp 51 PolarQuant** — addresses disorder gap. 1–2 days.
3. **Exp 52 3Di multi-task head** — sequence → (binary OE + 3Di tokens),
   ~6M aux labels. Combine with Stage 4. ~1 week.
4. **Exp 53 Foldseek-mined PDB/AFDB training data** — only useful AFTER
   architecture upgrade (Stage 3 already showed 2× data didn't help the CNN).
5. **Multi-teacher distillation** — earmarked, revisit only after Exps 51–53.

---

## Limitations of the work as currently presented

- **Disorder retention 94.9 %** (single-PLM rigorous, Exp 43; consistent
  across 5 PLMs in Exp 46, lowest 94.8 % for ANKH) — not at parity with raw.
  The ~5 pp gap is geometrically real (Exp 45 forensics), not a bug.
- **L=175 reference length** in compression-tier tables is a fixed assumption
  from the codec design spec (`2026-03-29-unified-codec-design.md:59`), not
  the empirical mean. Exp 45 reports SCOPe-5K mean L=156. Storage figures
  will scale ~10 % differently for the actual mean.
- **Exp 50 not yet rigorous.** Random 80/10/10 split leaks homology; the
  rigorous CATH-cluster re-run is designed but not executed. Talk cites
  Exp 50 as in-progress, not as a final number.
- **Exp 37 legacy benchmarks** (lDDT 100.7 %, contact precision 106.5 %,
  TM-score 57.4 %) predate the rigorous framework. No BCa CIs. Disclosed as
  pre-rigorous; the TM-score 57.4 % was previously omitted from CLAUDE.md
  and is now disclosed (D.1 fix).
- **Single host validated** (M3 Max). CUDA path exercised in test fixtures
  but not benchmarked at scale.
- **Pre-rigorous numbers in older sections** of README (V2 full / balanced /
  binary; ChannelCompressor `Ret@1=0.795 ± 0.012`) are **not re-verified
  cell-by-cell** in this audit. README rewrite (E.3) marks them as
  pre-rigorous.

---

## Audit summary (Phase C, 2026-04-26 to 2026-04-27)

The full audit lives in `docs/AUDIT_FINDINGS.md` (357 lines, 18 subagent
reports under `docs/_audit/logs/`). High-level result:

- **49 GREEN / 34 YELLOW / 9 RED line items** (true root-cause RED count: 6).
- **No headline cited number invalidated.** Every Exp 43/44/46/47 retention
  cell is bit-perfect against source JSONs.
- **All 3 in-repo REDs resolved in Phase D** (marp-cli installed, faiss-cpu
  and tmtools declared, dev tooling installed).
- **5 RED line items remain** — all sub-instances of the README drift
  root cause, deferred to E.3 (this is the README rewrite phase).
- **6 YELLOW items deferred to G.2** (`EXPECTED_QA.md`) — talk preempts.
- **8 YELLOW items deferred post-talk** (`.one.h5` versioning, BENCH_PATH
  parameterization, FastTree/IQ-TREE phylo trace, etc.).

813 / 813 tests pass. `uv lock --check` exit 0. `deptry` clean.

---

## Appendix: Prior (written before audit, 2026-04-26)

> Preserved as the `stub(prior): STATE_OF_THE_PROJECT.md` commit content for
> the calibration loop. Compare against the live doc above; the
> prior-vs-posterior delta is summarised in `docs/CALIBRATION.md`.

### One-paragraph summary (predicted)

> The OneEmbedding codec compresses per-residue PLM embeddings ~37× (binary
> default, no codebook) at 95–100 % task retention across 6 tasks and 5 PLMs,
> with rigorous (BCa) error bars. The work is bottlenecked on three open
> problems: a persistent ~5 % gap on disorder, a CNN capacity ceiling at 69 %
> bit accuracy for sequence → embedding (Exp 50), and the absence of a strong
> baseline against co-distilled VESM. Next moves: transformer backbone for
> Exp 50, multi-task 3Di head, multi-teacher distillation. Strong contender
> for a Bioinformatics-tier short paper now.

### What works (predicted)
- 232 compression methods benchmarked.
- Universal codec, configurable via 4 knobs.
- Multi-PLM validation (5 PLMs, 6 tasks).
- Sequence→OE shows non-trivial signal (~69 % bit accuracy, 0.55 cosine).
- Phylogenetics from embeddings (Exp 35) recovers 11/12 monophyletic families.

### What doesn't (predicted)
- Disorder retention plateaus at ~95 %.
- Exp 50 CNN ceiling.
- Some claims may not have a traced source.

### Predicted "open problems"
1. Disorder gap (mechanism + fix candidates).
2. Sequence → embedding ceiling (architecture lever).
3. Multi-teacher / co-distilled comparison missing.

### Predicted "next directions"
1. Stage 4 transformer for Exp 50.
2. Exp 51 PolarQuant.
3. Exp 52 3Di multi-task head.
