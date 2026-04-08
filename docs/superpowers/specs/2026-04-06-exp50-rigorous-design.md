# Exp 50 Rigorous Re-run — Sequence → Binary OE with CATH-level Splits

**Date:** 2026-04-06
**Status:** Done — see `results/exp50_rigorous/final_comparison.md`. Stage 1 H 67.68 ± 0.04%, Stage 1 T 67.73 ± 0.02%, Stage 2 H 69.28 ± 0.02%, Stage 2 T 69.65 ± 0.09% (3 seeds each, all 896/896 dims > 60% on every run).
**Supersedes:** Naive random-split run in `results/exp50/stage1_results.json` (2493 proteins, 80/10/10 random)

## Motivation

The sighting run of Exp 50 (Stage 1, random 80/10/10 on 2493 SCOPe-40 proteins)
produced 65.4% test bit accuracy — +30.8 pp above random. That result is
suggestive but scientifically weak for three reasons:

1. **Random splits leak homology.** Proteins at 40% sequence identity can still
   share folds and local motifs. A random split does not test generalization to
   unseen protein families — it tests interpolation within a diverse set.
2. **Only 2493 proteins.** Small enough that Stage 2 (4.2M params) risks
   overfitting before the split integrity question is even addressed.
3. **Single seed.** No variance estimate — the 65.4% number has no error bar.

This spec describes a rigorous re-run that fixes all three issues.

## Goals

**Primary:** Report an honest, CI-bearing estimate of how much of the 896 binary
OE bits a sequence-only CNN can predict when evaluated on held-out
CATH homologous superfamilies (no homology leakage).

**Secondary:** Report the fold-level generalization stress test (held-out CATH
topologies) to bound how much of the signal is fold-local vs fold-transferable.

**Not a goal (this spec):** Downstream eval (SS3/disorder/retrieval via
reconstructed OE); attention-hybrid Stage 3; hyperparameter tuning.

## Dataset

- **Source:** `data/residue_embeddings/prot_t5_xl_cath20.h5` — ProtT5-XL
  embeddings for **14,433** CATH S20 domains (already extracted).
- **FASTA with labels:** `data/external/cath20/cath20_labeled.fasta` — headers
  are `>{pid}|{C}.{A}.{T}.{H}` where the four numbers are CATH
  Class / Architecture / Topology / Homologous-Superfamily codes.
- **Sequence identity:** dataset is already pre-clustered at 20% — any two
  proteins in the set share < 20% pairwise identity. Within-train redundancy is
  minimal by construction.
- **Lengths:** median ≈ 144, max 512 (embeddings are already truncated in the H5).
- **Binary OE targets:** produced by `OneEmbeddingCodec(d_out=896,
  quantization='binary', abtt_k=0)`, with centering stats **fit on the training
  fold of the current split only** (not on the full dataset) to avoid
  fitting-stat leakage into val/test.

## Splits

Two split strategies are run in parallel (same seeds, same data, different
clustering level):

### H-split (main result)

- **Unit:** CATH homologous superfamily (the `H` code in `C.A.T.H`).
- **Rule:** every protein whose H-code appears in train stays in train. No
  H-code appears in more than one fold. Test proteins have zero homologs in
  train.
- **Ratios:** 80% / 10% / 10% of H-superfamilies → train / val / test.
- **Stratification:** per-Class round-robin — for each Class value C ∈ {1..5}
  independently, list all H-codes in that class, shuffle with the seed, then
  walk the shuffled list assigning each H-code to the fold whose current
  population is furthest below its target fraction (greedy balance). This
  guarantees every Class has proportional representation in every fold.
- **Semantics:** this is the community-standard "no homology leakage" split.

### T-split (stress test)

- **Unit:** CATH topology (the `T` code, i.e. fold).
- **Rule:** every protein whose T-code appears in train stays in train. Test
  proteins have entirely novel folds.
- **Ratios:** same 80/10/10 of T-topologies.
- **Stratification:** same Class-level stratification.
- **Semantics:** tests whether a local sequence-to-bit mapping generalizes
  across folds, or whether the model memorized fold-specific motifs.

### Common

- **Seeds:** 3 seeds (42, 43, 44). Splits are deterministic per (level, seed).
- **Saved split files:** `results/exp50_rigorous/splits/{h,t}_seed{42,43,44}.json`
  containing `{"train_ids": [...], "val_ids": [...], "test_ids": [...]}`.
- **Fallback:** if a given H-code or T-code has only one member, it goes into
  train (cannot form a cluster that straddles).

## Leakage audit

Run **once**, on H-split seed 42, before any training:

- Write train and test sequences to FASTA files.
- Run `mmseqs2 easy-search query=test target=train --min-seq-id 0.0`.
- Report:
  - Number of test proteins with any hit above 20% identity.
  - Max per-query identity distribution (mean, median, max, percentiles).
  - Counts at thresholds: ≥ 20, 25, 30, 40, 50, 60 % identity.
- Save to `results/exp50_rigorous/leakage_audit.json`.
- Acceptance: if > 5% of test proteins have a train hit above 40% identity, the
  split is questionable; log a warning and flag for review. (Unlikely given the
  dataset is already 20%-clustered and we're also cluster-splitting by H, but
  the audit is cheap and mandatory.)

## Models (unchanged from sighting run)

- **Stage 1:** 5-layer dilated CNN, hidden=128, 613,760 params.
- **Stage 2:** 10-layer dilated CNN, hidden=256, 4,183,424 params.
- Both use `Seq2OE_CNN` from `src/one_embedding/seq2oe.py` unchanged.

## Training config (unchanged)

- Stage 1: 50 epochs, batch size 16, lr 1e-3, cosine schedule, early-stop
  patience 15 on val loss.
- Stage 2: 100 epochs, batch size 8, lr 5e-4, cosine schedule, early-stop
  patience 15 on val loss.
- Loss: BCEWithLogitsLoss, masked over valid residues and all 896 bits.
- AdamW, weight decay 1e-4, grad clip 1.0, NaN sanitization before clipping.
- `PYTHONUNBUFFERED=1` for all runs so stdout flushes in real time.

## Metrics per run

- `overall_bit_acc` — total correct bits / total valid bits across test set.
- `per_protein_mean`, `per_protein_std`, `per_protein_min`, `per_protein_max`
  — from per-protein accuracy array.
- `dim_accuracies` — length-896 array of per-bit test accuracy.
- Summary stats on dim array: mean, std, `n_above_55`, `n_above_60`.
- `best_epoch`, `best_val_loss`, `best_val_bit_acc`, `train_seconds`.
- Training history (per-epoch train/val loss and acc) to `history.json`.

## Aggregation

For each (stage, split) pair, aggregate across 3 seeds:

- `overall_bit_acc_mean`, `overall_bit_acc_std`
- `per_protein_mean_across_seeds` (mean of per-seed per-protein means, plus std)
- Dim-level aggregation, two flavours reported side by side:
  - **Intersect@60**: number of the 896 dims where *every one of the 3 seeds*
    achieved > 60% test acc on that dim individually. Stricter, robust count.
  - **Mean@60**: number of dims whose mean-across-seeds per-dim acc > 60%.
    Standard, noisier count.
  - Same two flavours at the 55% threshold.

Final comparison table: H-split vs T-split vs (for reference) the old random
split, per stage. Mean ± std where multi-seed, single number where not.

## Outputs

```
results/exp50_rigorous/
  leakage_audit.json
  splits/
    h_seed42.json
    h_seed43.json
    h_seed44.json
    t_seed42.json
    t_seed43.json
    t_seed44.json
  h_split/
    stage1/
      seed42/
        best_model.pt
        history.json
        results.json
      seed43/ (same)
      seed44/ (same)
      summary.json
    stage2/
      (same structure)
  t_split/
    (same structure)
  final_comparison.json
  final_comparison.md          # human-readable table
```

The existing `results/exp50/stage1*` files are **not modified** — they remain as
the sighting-run baseline for reference.

## Code changes

### New module `src/one_embedding/seq2oe_splits.py`

Functions:

- `parse_cath_fasta(path) -> dict[str, dict]` — loads the labeled FASTA and
  returns `{pid: {"seq": str, "C": int, "A": int, "T": str, "H": str}}`.
  The T and H codes are kept as strings (joined dotted form) because they are
  hierarchical identifiers, not integer features.
- `cath_cluster_split(metadata, level: str, fractions: tuple[float, float, float], seed: int) -> tuple[list[str], list[str], list[str]]` —
  `level` is `"H"` or `"T"`. Groups proteins by the chosen level code, shuffles
  the groups (stratified by Class), and assigns whole groups to train/val/test
  to match the target fractions as closely as possible. Returns sorted lists of
  pids.
- `save_split(path, train_ids, val_ids, test_ids, meta: dict)` / `load_split(path)` —
  JSON I/O for reproducibility, with `meta` carrying `level`, `seed`, `fractions`.

Tests in `tests/test_seq2oe_splits.py`:
- Parser round-trip on a small synthetic fasta.
- H-split and T-split both produce disjoint groups (assert no overlap at the
  cluster-code level).
- Deterministic under same seed, different under different seeds.
- Class-stratification maintained within a tolerance.
- Fallback for singleton groups (they go to train).

### Refactor `experiments/50_sequence_to_oe.py`

Add CLI flags:
- `--dataset {cath20, medium5k}` (default `cath20`)
- `--split {h, t, random}` (default `h`)
- `--seed INT` (default `42`)
- `--output-root PATH` (default `results/exp50_rigorous`)

Valid (dataset, split) combinations:
- `(cath20, h)` — main rigorous run. Requires `cath20_labeled.fasta`.
- `(cath20, t)` — stress test. Same.
- `(cath20, random)` — new naive baseline on CATH data (useful for measuring
  the "rigour penalty" against the same data size).
- `(medium5k, random)` — exact back-compat with the sighting run. The only
  configuration that writes under `results/exp50/` by default instead of
  `results/exp50_rigorous/`. No metadata file needed.
- `(medium5k, h)` / `(medium5k, t)` — **not supported**, error out with a
  clear message (no CATH labels for the medium5k set).

Refactor `load_data()` to dispatch on `--dataset` and `--split`:
- For `cath20`, load embeddings from the CATH H5 and sequences + labels from
  `cath20_labeled.fasta`.
- For `random`, shuffle the intersection of sequences and embeddings with the
  given seed and take 80/10/10.
- For `h` / `t`, call `cath_cluster_split` with the given seed.

**Crucial:** codec centering stats are fit on train only. The `fit`/`encode`
split is moved from a single global call to:

1. Split first.
2. Fit `OneEmbeddingCodec` on `{pid: embeddings[pid] for pid in train_ids}`.
3. Build binary targets for all three folds from that fit.

Stage checkpointing / outputs go under
`{output_root}/{split}_split/stage{N}/seed{seed}/`.

### New runner `experiments/50b_run_rigorous.py`

Orchestration script — no model code, just shells out to
`50_sequence_to_oe.py` for each (stage, split, seed) triple:

1. Run leakage audit (H-split seed 42) → `leakage_audit.json`.
2. For each `split` in `['h', 't']`:
   - For each `seed` in `[42, 43, 44]`:
     - For each `stage` in `[1, 2]`:
       - Invoke the main script with `PYTHONUNBUFFERED=1`.
3. Aggregate per-seed JSONs into `summary.json` per (split, stage).
4. Write `final_comparison.json` and `final_comparison.md`.

The runner does one training run at a time (no parallelism across runs — MPS
contention on shared memory makes parallel GPU training slower than
sequential). System load is checked before each run and the script aborts if
`load1 > 10`.

### Unchanged

- `src/one_embedding/seq2oe.py` — model, dataset, and `prepare_binary_targets`
  are reused as-is. Only `prepare_binary_targets` will be called with a
  train-only embeddings dict instead of the full set.

## Compute budget (rough)

All numbers MPS on the M3 Max.

- **Stage 1:** sighting run took 292 s on 1994 train proteins (5.8 s/epoch ×
  50 epochs). Scaling to ~11,500 train proteins (14,433 × 80%) ≈ 4.6× →
  ~27 s/epoch × 50 epochs ≈ **~23 min/seed**.
- **Stage 2:** sighting run reached 11 s on epoch 1 at batch size 8 before it
  was killed; extrapolating gives ~11 s/epoch on 1994 proteins. Scaled to
  ~11,500 train: ~50 s/epoch × 100 epochs ≈ **~83 min/seed worst case**,
  probably **~50 min** with patience-15 early stop kicking in around
  epoch 55–65.
- Total wall: `(23 + 83) × 3 seeds × 2 splits` ≈ **~636 min ≈ 10.6 h worst
  case**, **~6–7 h** realistic with early stop.
- Recommendation: launch overnight via the runner.

## Risk / things that could go wrong

- **Cluster sizes uneven.** H-superfamilies are not uniform in size — a single
  large cluster could push the train fraction off target. Accept a tolerance
  of ± 5 pp on the actual train fraction; warn if exceeded.
- **Tiny val/test.** If only ~10% of 14,433 proteins lands in test that's ~1400
  — still plenty for the per-protein bootstrap. If the actual fraction is
  noticeably off because of cluster-size variance, log it but proceed.
- **Singleton H-codes.** CATH has many H-codes with a single representative
  (especially in S20). These go to train — documented as a fallback.
- **T-split too hard.** If Stage 1 val acc stays near 50% on T-split, that's a
  result, not a bug — report it and move on.
- **Overfitting on Stage 2 with H-split.** With 11,500 training proteins and
  4.2M params the model could still overfit. Early stopping handles this; just
  report val-selected test numbers.
- **MMseqs2 speed.** 14K × 14K easy-search on a laptop should finish in
  minutes. Not a concern.

## Success criteria

- All 12 training runs complete without crashes.
- Leakage audit shows > 95% of test proteins with no > 40% identity hit in
  train on H-split (expected given the pre-clustering).
- Stage 1 H-split overall bit accuracy has a clear CI (std across 3 seeds < 1
  pp).
- The paper-ready result is a table of the form:

  | Stage | Split     | Bit acc (mean ± std, 3 seeds) | dims > 60% (intersect) |
  |-------|-----------|-------------------------------|------------------------|
  | 1     | H         | X.X ± 0.X %                   | N / 896                |
  | 1     | T         | Y.Y ± 0.Y %                   | M / 896                |
  | 2     | H         | ...                           | ...                    |
  | 2     | T         | ...                           | ...                    |

- At minimum: we can report whether sequence-only bit prediction survives a
  rigorous no-homology-leakage split, with honest error bars.
