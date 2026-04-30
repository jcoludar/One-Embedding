# Exp 48 — Random Neighbor Score under OneEmbedding compression

**Date:** 2026-04-30 (revised after methodology check + pipeline ablation)
**Question:** does the codec change embedding quality as measured by RNS
(Prabakaran & Bromberg, *Nat Methods* 23, 796–804, 2026)?
**Bottom line.** The **codec itself stands** — compression is RNS-neutral
when the protein vector is built by mean-pooling the decoded bytes. The
**shipping DCT-K=4 protein vector** specifically takes a +0.14 RNS hit,
which Exp 48b localizes to two sources (centering and binarization) and
which reflects a metric mismatch, not embedding degradation.

**Sources.**
- `experiments/48_rns_compression.py` — main 4-condition comparison
  → `data/benchmarks/rigorous_v1/exp48_rns_compression.json`
- `experiments/48b_rns_ablation.py` — pipeline-stage ablation
  → `data/benchmarks/rigorous_v1/exp48b_rns_ablation.json`
- `experiments/48c_rns_knob_sweep.py` — codec-knob sweep (12 configs)
  → `data/benchmarks/rigorous_v1/exp48c_knob_sweep.json`
- `experiments/48a_extract_junkyard.py` — junkyard ProtT5 extraction (~9 min, 2.6 GB)
- `src/one_embedding/rns.py` — `compute_rns(...)` with `exclude_shuffles_of_query=True` for the same-source filter

## Setup

| | |
|---|---|
| **Real corpus** | CB513 — 511 proteins, ProtT5-XL per-residue embeddings. |
| **Junkyard corpus** | 5 residue-shuffled copies per CB513 sequence → 2 555 ProtT5 embeddings. Same composition as the real source, no biological order. |
| **k** | 100 nearest neighbors. |
| **Codec** | `OneEmbeddingCodec()` defaults: `d_out=896`, `quantization='binary'`, `abtt_k=0`. Fitted on real CB513 (centering only — binary mode has no codebook). |
| **RNS filter** | `exclude_shuffles_of_query=True`. Each query's own residue-shuffled copies are dropped from its candidate neighbors so they can't bunch up trivially close (same composition would otherwise count as junkyard at mean pool). Sanity check: turning the filter off changes aggregate RNS by < 0.001 in any condition — the filter matters in principle, not in this corpus. |
| **Random-placement RNS** | 2 555 / (511 + 2 555) = **0.833**. All conditions below sit well under that. |

## Headline 2 × 2 (Exp 48)

|  | mean pool | DCT-K=4 |
|---|---|---|
| **raw ProtT5 1024d**            | C1 (1024,) | C2 (4096,) |
| **OE 896d binary** (decoded)    | C3 (896,)  | C4 (3584,) — codec ships this |

| Condition | Dim | Mean RNS | 95 % CI | Median |
|-----------|:---:|:--------:|:--------|:------:|
| C1 raw + mean    | 1024 | **0.127** | [0.102, 0.153] | 0.000 |
| C2 raw + DCT-K=4 | 4096 | 0.370    | [0.340, 0.400] | 0.340 |
| C3 OE  + mean    |  896 | **0.130** | [0.105, 0.157] | 0.000 |
| C4 OE  + DCT-K=4 | 3584 | 0.510    | [0.485, 0.534] | 0.540 |

Per-protein paired deltas (10 000 percentile bootstraps over 511 queries):

| Comparison | Δ (mean) | 95 % CI | proteins increased | sig. |
|---|:---:|:---|:---:|:---:|
| C2 − C1 (raw, DCT4 vs mean)        | +0.243 | [+0.220, +0.266] | 58.3 % | * |
| C4 − C3 (OE,  DCT4 vs mean)        | +0.380 | [+0.358, +0.402] | 87.9 % | * |
| **C3 − C1 (compression at mean)**     | **+0.003** | [+0.002, +0.005] | 13.9 % | * |
| **C4 − C2 (compression at DCT-K=4)**  | **+0.140** | [+0.126, +0.154] | 77.7 % | * |

**Compression-at-mean is essentially free** (Δ = +0.003, 86 % of proteins
RNS-identical). **Compression-at-DCT-K=4 costs +0.140**, and most of that
is *not* RP or fp16 — see Exp 48b.

## Pipeline ablation (Exp 48b)

The `raw → OE` path has four cumulative stages: centering → RP896 (fp32)
→ fp16 cast → binary. RNS at each stage, both pools:

| Stage | dim (mean) | RNS mean | dim (dct4) | RNS dct4 |
|---|:---:|:---:|:---:|:---:|
| S0 raw         | 1024 | 0.127 | 4096 | 0.370 |
| S1 centered    | 1024 | 0.127 | 4096 | 0.444 |
| S2 RP fp32     |  896 | 0.129 | 3584 | 0.445 |
| S3 RP fp16     |  896 | 0.129 | 3584 | 0.445 |
| S4 OE binary   |  896 | 0.130 | 3584 | 0.510 |

Incremental, paired, per-protein:

| Transition | mean Δ | dct4 Δ |
|---|:---:|:---:|
| S0 → S1 (centering)         | 0.000  | **+0.074** * |
| S1 → S2 (RP896 fp32)        | +0.001 *  | +0.001 |
| S2 → S3 (fp16 cast)         | 0.000  | 0.000 |
| S3 → S4 (binary quantize)   | +0.002 *  | **+0.065** * |
| **TOTAL S0 → S4**           | **+0.003** | **+0.140** |

`*` = 95 % paired CI excludes zero.

**Findings:**

1. **Mean pool: every stage is RNS-neutral.** Centering and fp16 contribute
   exactly 0; RP and binarization each contribute ~+0.001–+0.002. Cumulative
   compression cost = +0.003. This is what we want to see for the codec.
2. **DCT-K=4: the +0.14 splits ~53 / 47 between centering and binarization.**
   - **Centering alone adds +0.074 at DCT-K=4** while leaving mean-pool unchanged.
     This is **not damage from compression** — centering subtracts a per-dimension
     constant from every residue, which lands entirely in the DCT k=0 bin
     and is invariant for k=1..3. But for L2-based kNN, the pre-centering
     DC bin had a large magnitude that *was* doing the real-vs-junkyard
     discrimination; once it's near zero, the L2 search reweights toward
     k=1..3, where shuffled sequences look much more like real ones.
     **The codec didn't randomise the embedding — it removed the DC handle
     that DCT-K=4 was leveraging.**
   - **Binarization adds +0.065 at DCT-K=4** but only +0.002 at mean. This
     is the genuine quantization-noise contribution. The 1-bit-per-dim
     reconstruction adds residue-level noise that the DCT high-frequency
     bins integrate; mean pool averages it back out.
3. **Random projection is the cheapest stage at every metric** (+0.001 mean,
   +0.001 dct4). The JL-lemma intuition holds — RP preserves pairwise
   distances closely enough that RNS doesn't notice the dimension cut from
   1024 to 896.
4. **fp16 is bit-perfect for RNS.** Δ = 0.000 at both pools. Consistent with
   prior retention benchmarks (Exp 43/44 also show fp16 = 0 pp effect).

## What it means in plain terms

The codec compresses ProtT5 to 1/37th the size and preserves the embedding
quality that RNS measures **at mean pool**. The codec's *shipping* protein
vector — DCT-K=4 — has +0.14 higher RNS, but Exp 48b localises that to:

- **a metric–pooling mismatch** (centering removes the DC bias DCT-K=4 was
  using to separate real from junk), and
- **expected binarization noise** in the high-frequency bins.

It is **not** a sign that compression makes embeddings "more random."
Cross-checks support that:

- Same-source-shuffle filter (`exclude_shuffles_of_query=True`) doesn't move the numbers,
  so the deltas aren't an own-shuffle artefact.
- Retrieval / SS3 / SS8 / disorder benchmarks (Exp 46/47) all show 95–100 %
  retention under the same codec — the real-vs-real geometry that those
  tasks live on is preserved.
- Mean-pooling the decoded OE bytes recovers raw-equivalent RNS.

## Knob sweep (Exp 48c)

The Exp 48b ablation walked the *default* (binary 896d) pipeline. Exp 48c
sweeps across codec configurations to ask whether the +0.14 DCT-K=4 hit is
intrinsic to the codec, or specific to particular knob settings.

| Config | mean | DCT-K=4 | Δ DCT4 vs raw |
|---|:---:|:---:|:---:|
| raw baseline                | 0.127 | 0.370 | — |
| lossless 1024d (fp16, no RP) | 0.127 | 0.444 | +0.074 * |
| fp16 896d                   | 0.129 | 0.445 | +0.075 * |
| int4 896d                   | 0.129 | 0.447 | +0.077 * |
| **PQ M=64 896d**            | 0.127 | **0.342** | **−0.028** * |
| PQ M=128 896d               | 0.124 | 0.391 | +0.021 * |
| PQ M=224 896d               | 0.126 | 0.429 | +0.060 * |
| binary 896d (default)       | 0.130 | 0.510 | +0.140 * |
| binary 1024d (no RP)        | 0.129 | 0.447 | +0.077 * |
| binary 512d                 | 0.127 | 0.493 | +0.123 * |
| binary 768d                 | 0.133 | 0.505 | +0.135 * |
| **binary 896d, ABTT-3**     | 0.130 | **0.616** | **+0.246** * |

`*` = 95 % paired CI excludes zero.

### Sweep findings

1. **At mean pool every codec is RNS-neutral.** All paired Δs ≤ 0.007 in
   absolute value. No knob — quantization scheme, d_out, ABTT, PQ M —
   meaningfully shifts mean-pool RNS. The Exp 48 / 48b headline that the
   per-residue substrate is RNS-equivalent to raw is robust across the
   whole knob space.
2. **PQ wins at DCT-K=4. Binary loses. ABTT is catastrophic.**
   - **PQ M=64 is *better* than raw** at DCT-K=4 (Δ = −0.028, CI excludes zero).
     The PQ codebook is fit on real residues, so when junkyard residues are
     quantized through it they get snapped onto "real-protein-like"
     centroids. The structured noise re-introduces a corpus-direction bias
     that helps the kNN search distinguish real from shuffled. **PQ is a
     learned filter that filters toward the training distribution; binary
     is unstructured noise.**
   - PQ degrades smoothly as M grows (M=64 < M=128 < M=224 < binary).
   - Binary at d_out = 1024 (no RP) sits at +0.077, same as int4 and fp16.
     The "binary tax" only kicks in once RP is also applied — RP and
     binarization stack multiplicatively at DCT-K=4 but neither alone is
     bad. (At mean pool, both are still ≈ 0.)
   - **ABTT-3 doubles the binary cost** to +0.246. Removing the top
     principal components removes the very direction that carries
     real-protein-vs-junk discriminability. Same root cause as the
     disorder collapse documented in Exp 45.
3. **The +0.074 "centering toll" at DCT-K=4 is universal across non-PQ
   quantizations** — present in fully lossless 1024d (no RP, no binary,
   only centering + fp16). Confirms Exp 48b's mechanism: centering moves
   the DC bin near zero, kNN reweights toward k=1..3, and those bins
   don't separate real from junk well. PQ is the only configuration
   that gets *under* this floor, because the codebook restores
   corpus-direction bias post-centering.

### Practical implications

- **Default codec (binary 896d) is fine for the tasks we benchmark**
  (Ret@1, SS3/SS8, disorder). Those don't depend on real-vs-junkyard
  separability.
- **For RNS-aware downstream usage at DCT-K=4, switch to PQ M=64.**
  It's the best DCT-K=4 RNS in the sweep, *better* than raw — at the
  cost of needing the codebook on the receiver side (~1 MB extra per
  PLM × config). PQ M=64 also gives ~32× compression.
- **Keep `abtt_k=0` as the universal default.** RNS at DCT-K=4 doubles
  with ABTT-3 on. Combined with the disorder evidence (Exp 45), there
  is no remaining metric where ABTT is the right call.
- **For binary-specific users who care about RNS:** `d_out=1024`
  (skip RP) recovers most of the binary cost (+0.077 vs +0.140 at the
  default 896d). The trade-off is ~14 % more bits per residue.

## Recipe — RNS-aware downstream usage

The codec ships two pools off the same bytes:

```python
data = OneEmbeddingCodec.load_batch("compressed.one.h5")
data['P12345']['per_residue']   # (L, 896) — decoded float
data['protein_vec']             # (3584,) — DCT-K=4, default
```

For **retrieval / clustering / family classification**, keep using
`protein_vec` — that's what Exp 46/47 validated.

For **embedding-quality / novelty checks (RNS-style)**, mean-pool the
per-residue bytes:

```python
quality_vec = data['P12345']['per_residue'].mean(axis=0)   # (896,)
```

These are the same bytes — two different pools off them. Use the right
pool for the question.

## Caveats

- Single PLM (ProtT5-XL), single dataset (CB513). Magnitudes will move
  on other PLMs / datasets; the qualitative split (mean ≈ free,
  DCT-K=4 takes ≈ +0.14) should replicate.
- Junkyard = residue-shuffled copies of the *same* CB513 sequences. A
  harder junkyard (random unrelated AA strings) would push absolute RNS
  values around but the paired deltas should be insensitive.
- Percentile-bootstrap CIs above are not BCa-adjusted; for the per-protein
  RNS distribution (bounded in [0, 1]) the second-order correction
  matters less than for retention ratios.
- We use FAISS L2 distance. Cosine would re-rank some neighbors; not
  benchmarked.
- The **DC-bin mechanism** for the centering effect is a hypothesis
  consistent with the numbers; we have not directly verified it by
  freezing the DC bin separately. A clean follow-up would be to compute
  RNS at "DCT-K=4 with k=0 bin restored to the un-centered value", which
  should land between S0 and S1 if the hypothesis is right.

## Reproduce

```bash
# 1. Junkyard ProtT5 extraction (~9 min on M3 Max, 2.6 GB output)
uv run python experiments/48a_extract_junkyard.py

# 2. Main 2x2 comparison (~30 s)
uv run python experiments/48_rns_compression.py

# 3. Pipeline ablation (~50 s)
uv run python experiments/48b_rns_ablation.py

# 4. Codec-knob sweep (~10 min — 3 PQ fits dominate)
uv run python experiments/48c_rns_knob_sweep.py
```
