# Exp 50 Stage 3 — Continuous Regression with Expanded Data

**Date:** 2026-04-07
**Status:** Approved — proceeding to implementation plan
**Builds on:** `2026-04-06-exp50-rigorous-design.md` (Stages 1 + 2 sweep)

## Motivation

Stages 1 and 2 of Exp 50 train a sequence-only CNN to predict the **binary**
One Embedding (896 sign-bits per residue) via per-bit BCE loss. They get
67.7% bit accuracy (Stage 1) and 69.3% bit accuracy (Stage 2) on the
rigorous CATH H-superfamily holdout, with all 896 dims individually above 60%.

Two structural limitations of that setup leave value on the table:

1. **Binarization throws away information.** A given 896d projected vector
   carries ~7 useful bits per dimension; the binary target keeps 1. The BCE
   loss therefore receives a much weaker signal than the underlying ProtT5
   embedding actually contains. A continuous regression target restores the
   full signal.

2. **CATH20 alone is data-starved.** ~14,433 proteins is a small fraction of
   what's available locally. DeepLoc has ~14,439 ProtT5 embeddings already
   extracted (`prot_t5_xl_deeploc.h5`, 20.5 GB), which roughly doubles the
   training pool with zero new PLM inference cost.

This spec describes Stage 3: a multi-seed run that switches to continuous
regression, adds DeepLoc to the training pool (after a leakage check), and
introduces cosine similarity / Euclidean distance as primary metrics
alongside bit accuracy. The architecture is unchanged from Stage 2, so the
delta is fully attributable to the loss + data changes.

## Goals

**Primary:** Establish a clean, multi-seed measurement of how much the
sequence-only model improves on the same CATH H-split test set when (a) the
loss switches from per-bit BCE to continuous regression on the 896d projected
target, and (b) the training pool expands from CATH20 (~11.5K train) to
CATH20+DeepLoc (~25K train, after leakage filtering).

**Secondary:** Establish cosine similarity and Euclidean distance as the
project's standard quality metrics for sequence-to-embedding models. These
align with how PLM embeddings are actually consumed downstream (cosine for
retrieval, distance for clustering) and avoid the lossiness of bit accuracy.

**Not a goal (this spec):** explicit per-task auxiliary heads (SS3, SS8,
disorder, localization) — deferred to Stage 4 if Stage 3 shows that pure
continuous distillation already saturates. Transformer architecture also
deferred. Multi-PLM teacher (ESM, ANKH) deferred.

## Architecture (unchanged from Stage 2)

- 10-layer dilated CNN, hidden dim 256, kernel size 3, total ~4.2 M params.
- Identical `Seq2OE_CNN` class from `src/one_embedding/seq2oe.py`, **except**
  the 896-bit linear head is reinterpreted as a 896d continuous regression
  head (no architectural change — same `nn.Linear(hidden, 896)` — but the
  output is treated as raw values, not logits).
- Mask-out at the same place: padding positions zeroed.

## Targets

- **Source:** raw ProtT5 per-residue embeddings (1024d).
- **Codec:** `OneEmbeddingCodec(d_out=896, quantization='binary', abtt_k=0)`,
  fit on **CATH20 H-split train embeddings only** (same train-only fit
  discipline as Stage 2 — the spec
  `2026-04-06-exp50-rigorous-design.md` requires this and we keep it).
- **Target tensor:** the **continuous 896d projected vector** that the codec
  produces internally before binarization. We extract it via
  `codec._preprocess(raw)` (the same call Stage 2 uses, but we keep the
  pre-binarization output instead of taking the sign).
- The continuous targets are computed once per protein and cached in memory
  for the duration of training (same as Stage 2's binary targets).
- For DeepLoc proteins, we use the **same** fitted codec (centering stats and
  RP matrix from CATH20 train) so train and aux data live in the same space.

## Training data

### Primary pool (always used)

- **CATH20 H-split train:** the same ~11,544 proteins from the Stage 2 H-split
  at the chosen seed. Loaded from `data/residue_embeddings/prot_t5_xl_cath20.h5`.

### Auxiliary pool (DeepLoc, leakage-filtered)

- **DeepLoc embeddings:** ~14,439 proteins from
  `data/residue_embeddings/prot_t5_xl_deeploc.h5`, keyed by UniProt ID.
- **DeepLoc sequences:** `tools/reference/LightAttention/data_files/deeploc_complete_dataset.fasta`
  (14,004 entries; 13,949 of the H5 keys overlap with FASTA IDs, i.e. 96.6%
  coverage). The 490 missing entries are dropped — there's no fallback fasta.
  Header format is `>{uniprot_id} {label}`; we parse only the first
  whitespace-delimited token as the ID.
- **Leakage filter:** before training, run MMseqs2 easy-search with the
  CATH20 H-split test set as queries against the (sequence-resolvable subset
  of) DeepLoc set. Any DeepLoc protein with a hit at **≥ 30 % sequence
  identity** to any CATH20 H-test protein is **excluded** from the
  auxiliary training pool. This protects the H-split test from backdoor
  leakage via UniProt-domain substring overlap.
- The leakage filter list is computed once per seed and saved to
  `results/exp50_stage3/leakage_filter/deeploc_leakage_excluded_seed{seed}.json`
  for reproducibility. It depends on the H-test set, which depends on seed,
  so each seed gets its own filter.
- After filtering, the expected DeepLoc auxiliary pool size is
  ~13,000–13,900 proteins (96.6% coverage minus a small leakage exclusion).

### Combined training pool

- `train_pool = cath20_h_train ∪ deeploc_filtered`
- All proteins shuffled together each epoch (single `DataLoader` over the
  union).
- **Merge strategy:** load the CATH20 H5 and the DeepLoc H5 into a single
  `embeddings: dict[str, np.ndarray]` keyed by protein ID. CATH20 IDs
  (`12asA00` style) and DeepLoc IDs (UniProt, `P12345` style) are disjoint,
  so there's no collision risk — a simple dict update is sufficient. The
  same pattern applies to the sequences dict.
- Targets for both pools are computed using the **same** codec (fit on CATH20
  train only).

### Val and test (unchanged from Stage 2)

- Val: CATH20 H-split val (1,445 proteins). Used for early stopping.
- Test: CATH20 H-split test (1,444 proteins). Used for the headline metrics.
- DeepLoc proteins are **never** used for val or test in this spec — they
  serve only as additional training signal. This keeps the Stage 3 vs Stage 2
  comparison clean (same eval set, same H-split semantics).

## Loss

Per-residue, masked over valid positions:

```
L = λ_cos · (1 - cos_per_residue(pred, target))   # cosine distance, [0, 2]
  + λ_mse · MSE_per_residue(pred, target)         # squared error
```

- `cos_per_residue`: for each (B, L) position, compute the cosine similarity
  between the 896d predicted and target vectors at that position. Average
  over masked positions.
- `MSE_per_residue`: standard masked MSE over the 896 dimensions, averaged
  over masked positions.
- **Initial weights:** `λ_cos = 1.0`, `λ_mse = 0.1`. The cosine term is
  bounded [0, 2] and dominant; the MSE term is a magnitude regularizer that
  prevents the model from collapsing to a trivial scale. These weights are
  fixed for the exploratory run; tuning is deferred to a follow-up if needed.
- **No BCE term.** This is the whole point — we're abandoning the per-bit
  binary loss in favor of the continuous target.

## Training config

- **Optimizer:** AdamW, lr 5e-4, weight decay 1e-4 (same as Stage 2).
- **Schedule:** cosine annealing over 100 epochs (same as Stage 2).
- **Batch size:** 8 (same as Stage 2).
- **Max length:** 512 residues (same as Stage 2). Longer DeepLoc sequences
  are truncated to 512, matching how the existing CATH20 H5 was extracted.
- **Early stopping:** patience 15 on **val cosine distance** (not val loss),
  measured on CATH20 H-val. Cosine is the primary signal; loss may include
  the MSE auxiliary term.
- **Grad clip:** 1.0 with `nan_to_num_` sanitization (project convention).
- **Seeds:** [42, 43, 44]. Multi-seed wiring already in `train_stage` from
  Task 5 fixup — pass `seed=args.seed` through.
- **PYTHONUNBUFFERED=1** for live logs.

## Primary metrics (per run, on CATH20 H-test)

All metrics computed at the per-residue level then averaged across all valid
residues across all test proteins. Confidence intervals are seed-level
(across 3 seeds).

| Metric | Definition | Interpretation |
|---|---|---|
| `cosine_sim` | mean cosine similarity per residue | how aligned predicted and target embeddings are. 1.0 = perfect, 0.0 = orthogonal. Closest to "is this useful for retrieval". |
| `cosine_distance` | `1 - cosine_sim` | for early stopping. |
| `mse` | mean squared error per element, averaged over 896 dims and all residues | magnitude fidelity. Equal to `euclidean_per_residue² / 896`, so they carry the same information — we report MSE because it's also in the loss. |
| `bit_accuracy` | re-binarize predicted vector via per-protein `sign(pred - mean(pred))` and compare to the binarized target | apples-to-apples comparison with Stages 1 and 2. |
| `dims_above_60_intersect` | dims where every seed had per-dim bit acc > 0.60 individually | matches Stage 2 reporting. |

**Direct Stage 2 comparison** is on **bit accuracy** only. Stage 2 outputs
pre-sigmoid logits, not continuous projected vectors, so their cosine
similarity with the Stage 3 target is not a fair comparison (different
output space). Cosine similarity is a **new primary metric** established in
Stage 3 as the baseline for Stage 4+.

The headline result for the memory entry is the Stage 3 cosine similarity
mean ± std across 3 seeds, plus the bit-accuracy delta against Stage 2's
69.28 ± 0.02% on the same H-split test set.

## Aggregation

For each (split, stage) — only `(h, 3)` in this spec since we're not
re-running T-split or Stages 1 / 2:

- `cosine_sim_mean`, `cosine_sim_std` across seeds (ddof=1)
- Same for `mse`, `bit_accuracy`
- `dims_above_60_intersect`, `dims_above_60_mean` (Stage 2 reporting style)

Written to `results/exp50_stage3/h_split/stage3/summary.json` and
`results/exp50_stage3/final_comparison.md` (which compares Stage 3 to the
Stage 1 / Stage 2 numbers from `results/exp50_rigorous/`).

## Outputs

```
results/exp50_stage3/
  leakage_filter/
    deeploc_leakage_excluded_seed42.json
    deeploc_leakage_excluded_seed43.json
    deeploc_leakage_excluded_seed44.json
  h_split/
    stage3/
      seed42/
        best_model.pt
        history.json
        results.json
      seed43/ (same)
      seed44/ (same)
      summary.json
  final_comparison.md       # Stage 1 vs Stage 2 vs Stage 3
  final_comparison.json
```

`results/exp50_rigorous/` from the previous spec is **not modified** —
Stage 3 lives in its own output tree.

## Code changes

### New module: `src/one_embedding/seq2oe_continuous.py`

The continuous training loop is different enough from `seq2oe.py` (different
loss, different targets, multi-pool data loader) that it deserves its own
module rather than complicating the existing one. Functions:

- `prepare_continuous_targets(embeddings, codec) -> dict[str, np.ndarray]` —
  apply `codec._preprocess` to each protein, return the continuous (L, 896)
  array. Public, stable interface.
- `Seq2OEContinuousDataset` — same shape as `Seq2OEDataset` but yields
  float32 targets instead of uint8.
- `cosine_distance_loss(pred, target, mask) -> Tensor` — masked per-residue
  cosine distance, averaged.
- `mse_loss(pred, target, mask) -> Tensor` — masked per-residue MSE.
- `evaluate_continuous(model, sequences, targets, ids, device) -> dict` —
  computes the six primary metrics on a held-out set.

The model class itself (`Seq2OE_CNN`) is reused unchanged — its output is
already 896d, we just interpret it as continuous instead of binary logits.

### New experiment script: `experiments/50c_stage3_continuous.py`

Modeled on `experiments/50_sequence_to_oe.py` but specialized for continuous
training:

- Loads CATH20 (always) and DeepLoc (always for stage 3).
- Builds the H-split via existing `cath_cluster_split` (for direct
  comparability with Stage 2's H-test set at the same seed).
- Runs the leakage filter once per seed, caches the exclusion list.
- Builds the combined train pool: CATH20 H-train + DeepLoc filtered.
- Fits the codec on CATH20 train only.
- Computes continuous targets for everything (CATH20 train+val+test + DeepLoc
  filtered) using the same codec.
- Trains, with early stopping on val cosine distance.
- Evaluates on CATH20 H-test only, reporting all six metrics.
- Saves results.json with the same schema as Stage 2 plus the new metrics.
- CLI flags: `--seed INT`, `--output-root PATH` (default
  `results/exp50_stage3`), `--smoke-test`. No `--dataset` or `--split` —
  this script is hard-coded to CATH20 + DeepLoc with H-split, since that's
  what Stage 3 means. Simpler than reusing the dispatcher.

### New runner script: `experiments/50d_run_stage3.py`

Loops over the 3 seeds, calls `50c_stage3_continuous.py`, aggregates per-seed
results into `summary.json`, then writes `final_comparison.{json,md}` that
joins Stage 3 with the existing Stage 1 / Stage 2 numbers from
`results/exp50_rigorous/`. Same structure as `50b_run_rigorous.py` from Task 7,
but simpler (only one stage, one split, three seeds).

### New leakage filter helper: `experiments/50_stage3_leakage_filter.py`

Standalone script that runs MMseqs2 easy-search to identify DeepLoc proteins
with > 30 % identity to any CATH20 H-split test protein at a given seed.
Writes the exclusion list. The continuous training script calls this once
per seed if the cache file is missing. Mirrors the structure of
`experiments/50_leakage_audit.py` from Task 6.

### Tests

- `tests/test_seq2oe_continuous.py`:
  - `prepare_continuous_targets` produces (L, 896) float32 arrays
  - cosine_distance_loss returns 0 on perfect match, > 0 on mismatch, masked
    correctly
  - mse_loss is symmetric, masked correctly
  - `evaluate_continuous` returns all six metrics with sane ranges
  - smoke test on a synthetic 5-protein set

No changes to `seq2oe_splits.py`, `seq2oe.py`, `50_sequence_to_oe.py`, or
any of the Stage 1/2 code. Stage 3 is fully additive.

## Compute budget (rough)

- Stage 2 took ~38 min per seed on 11,544 CATH20 train proteins (with
  early-stop at ~epoch 18 of 100).
- Stage 3 trains on ~25,000 proteins (~2.17× more), and DeepLoc proteins
  are longer on average (median 354 vs CATH20's 144), so the per-epoch
  forward/backward cost increases proportionally. Net per-epoch scale
  factor ≈ 2×.
- The loss scalar cost is comparable to Stage 2's per-bit BCE
  (cosine + MSE on 896 dims is similar FLOP count to 896-way BCE).
- Per-seed estimate: **~60–90 min** depending on early-stop behavior under
  the new loss.
- 3 seeds × ~75 min ≈ **~3.75 hours training wall-clock**.
- Plus a one-time MMseqs2 leakage filter run: ~3 minutes per seed × 3 =
  ~10 minutes.
- **Total: ~3–5 hours overnight.**

## Risks

- **DeepLoc embeddings have different sequence-length distribution.** DeepLoc
  median length ≈ 354, max 512 (truncated). CATH20 median ≈ 144. The model
  may need to handle longer sequences more carefully. Mitigation: existing
  DataLoader pads to max_len=512 and masks; the architecture is fully
  convolutional and length-agnostic. Should just work, but watch the val
  cosine on the first run.
- **Codec centering stats are CATH20-only.** DeepLoc embeddings get
  re-centered with CATH20-train means, which is mildly off-distribution.
  This is the right call (we want a single shared embedding space tied to
  the eval set), but may slightly handicap the DeepLoc samples as training
  signal. Acceptable.
- **MSE term may collapse the model to predicting a constant mean.** This is
  why we keep λ_mse small (0.1) relative to cosine (1.0). Watch the cosine
  similarity on epoch 1; if it's near 1.0 immediately the model is gaming
  the magnitude term, and we should drop λ_mse to 0.01 or zero.
- **Leakage filter may exclude > 5 % of DeepLoc.** If so, that's interesting
  (real overlap exists) but doesn't break the experiment. The filter list is
  saved for inspection.
- **Cosine similarity ceiling is set by capacity + data, not by the
  projection.** The random projection is a deterministic linear operation,
  so a student that perfectly approximates ProtT5 would reach
  `cosine_sim = 1.0` on the projected target. Realistically we'll plateau
  well below 1.0 because the 4.2M-param CNN has limited capacity and CATH20
  + DeepLoc is still a small distillation corpus. The Stage 3 number
  establishes what this architecture can reach; the headroom to 1.0 is the
  budget for Stage 4+ (transformer, more data, distillation at scale).

## Success criteria

A clean three-seed measurement on CATH20 H-split test that lets us answer:

1. **Primary:** does re-binarized Stage 3 bit accuracy beat Stage 2's
   **69.28 ± 0.02%** on the same H-split? Target: at least +1.0 pp (i.e.
   ≥ 70.3%). A smaller improvement or a regression triggers the ablation.
2. **Primary:** what cosine similarity does this architecture reach on the
   H-test set? Establishes the baseline for Stage 4+.
3. **Secondary:** is the per-seed std on bit accuracy comparable to Stage 2
   (< 0.1 pp)? And is cosine sim std < 0.01? If variance balloons, something
   is unstable and we need to debug before trusting the numbers.
4. **Secondary:** per-dim `dims_above_60_intersect` stays at 896/896 (we
   don't regress on any single dim).

**Decision tree after the sweep:**
- **≥ +1.0 pp bit accuracy AND tight std:** declare Stage 3 a clean win.
  Move on to Stage 4 (transformer architecture) or the downstream-retention
  measurement experiment we discussed.
- **< +1.0 pp (flat or small improvement):** run the ablation — a Stage 3
  variant that uses continuous regression on CATH20 **only** (no DeepLoc).
  This isolates whether the loss change or the data expansion was the lever.
- **Regression (worse than Stage 2):** debug loss weighting first (drop
  `λ_mse` to 0.01 or 0), then codec-fit path (is DeepLoc's off-distribution
  centering hurting?), then learning rate.
