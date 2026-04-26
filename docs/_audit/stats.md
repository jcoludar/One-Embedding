# Stats Audit (Task C.4)

**Date:** 2026-04-26
**HEAD at audit:** `a88c589` (after C.3 commit)

Goal: confirm that every cited result in CLAUDE.md was produced with the
documented bootstrap method (BCa, B=10,000, percentile fallback for n<25),
the documented probe protocol (CV-tuned LogReg/Ridge, multi-seed averaged
*before* bootstrap), and the documented baseline pooling (same DCT K=4 for
raw and compressed retrieval).

## Evidence index

- Bootstrap impl: `experiments/43_rigorous_benchmark/metrics/statistics.py`
- Probe impl: `experiments/43_rigorous_benchmark/probes/linear.py`
- Per-residue runners: `experiments/43_rigorous_benchmark/runners/per_residue.py`
- Protein-level runner: `experiments/43_rigorous_benchmark/runners/protein_level.py`
- Rules / leakage assertions: `experiments/43_rigorous_benchmark/rules.py`
- Configs: `experiments/43_rigorous_benchmark/config.py` (constants:
  `SEEDS=[42,123,456]`, `BOOTSTRAP_N=10_000`, `C_GRID=[0.01,0.1,1.0,10.0]`,
  `ALPHA_GRID=[0.01,0.1,1.0,10.0,100.0]`, `CV_FOLDS=3`)

The same module is shared by Exp 43, 44, 46, 47 — every cited table is
produced by the same code path.

## Bootstrap implementation

`metrics/statistics.py` exposes five entry points; every cited result uses
exactly one of them.

### `bootstrap_ci` (single-system point estimate, used for raw and compressed standalone)

- Method choice (line 63): `use_bca = n >= 25; method = "BCa" if use_bca else "percentile"`.
- Hands the array to `scipy.stats.bootstrap` with `n_resamples=n_bootstrap`
  (default 10,000) and `random_state=np.random.RandomState(seed)`.
- NaN / exception fallback (lines 82–97): if BCa returns NaN or raises, the
  function re-runs with `method="percentile"` and stamps `ci_method="percentile"`.

[GREEN] Matches CLAUDE.md description.

### `paired_bootstrap_retention` (retention ratio: comp/raw, used for SS3 / SS8 / Ret@1)

- Aligns IDs across the two systems (line 135), passes both arrays to
  `scipy.stats.bootstrap` with **`paired=True`** (line 162) — same bootstrap
  resample applied to both raw and compressed in each iteration. This is the
  Rost-lab-required paired bootstrap.
- Same BCa/percentile choice as `bootstrap_ci`. Same fallback path.

[GREEN] Matches CLAUDE.md description; correct protocol.

### `cluster_bootstrap_ci` (pooled metrics, e.g. pooled Spearman ρ for disorder)

- Resamples at the **cluster (protein)** level — `idx = rng.randint(0, n, size=n)` over
  `cluster_data_list` (lines 311–317). Then computes the pooled statistic on
  the union of residues from the selected proteins.
- Implements BCa **manually** (not via scipy): bias correction `z0` via
  `norm.ppf(prop)`, acceleration `a_hat` via leave-one-cluster-out jackknife
  (lines 322–337). Falls back to percentile on NaN / ZeroDivision /
  FloatingPointError (lines 351–354).
- Uses the SETH/CAID standard pooled-residue Spearman; respects the
  hierarchical-data structure (Davison & Hinkley 1997 Ch. 2.4 — cited in
  docstring line 301).

[GREEN] Matches CLAUDE.md description; this is the textbook-correct
cluster bootstrap.

### `paired_cluster_bootstrap_retention` (paired cluster bootstrap on disorder retention ratio)

- Joint resampling of protein IDs for both raw and compressed (lines 410–419);
  computes pooled ρ for both on each draw; stores the retention ratio.
- Same manual BCa with jackknife acceleration. Same percentile fallback.

[GREEN]

### `averaged_multi_seed` (Bouthillier et al. 2021 — average BEFORE bootstrap)

- Takes `seed_scores: list[dict[str, float]]` — one dict per seed.
- Per-item averaging (lines 507–511): for each common ID, average the seed-scores,
  then bootstrap the **averaged** dict via `bootstrap_ci`.
- Side-channel `seeds_mean` / `seeds_std` are stored on the result for transparency
  about probe variance, but the CI is on the averaged predictions — NOT a CI of CIs.

[GREEN] Correctly implements the Bouthillier prescription.

### Per-experiment binding (where the table numbers actually come from)

| Cited table | Producing script | Bootstrap call | n=tested in code |
|---|---|---|---|
| CLAUDE.md "Per-Residue Tasks (BCa CI)" — SS3/SS8 | `experiments/43_rigorous_benchmark/run_phase_a1.py` | `averaged_multi_seed` (inside `run_ss[38]_benchmark`) for raw + compressed; `paired_bootstrap_retention` for retention | yes — pytest `test_benchmark_*` |
| CLAUDE.md "Per-Residue Tasks" — Disorder pooled ρ | `run_phase_a1.py` | `cluster_bootstrap_ci` for raw + compressed; `paired_cluster_bootstrap_retention` for retention | yes |
| CLAUDE.md "Protein-Level Tasks" — SCOPe / CATH20 / DeepLoc | `run_phase_a1.py` (SCOPe) and `run_phase_c.py` (CATH/DeepLoc) | `bootstrap_ci` (per-query 0/1 Ret@1 → mean → BCa); `paired_bootstrap_retention` for retention | yes |
| CLAUDE.md "ESM2 Multi-PLM Validation" | `run_phase_b.py` | same as Phase A1 | yes |
| CLAUDE.md "Codec sweep (Exp 47, ProtT5, standard tiers)" | `experiments/47_codec_sweep.py` | imports the same `metrics.statistics` module — identical call paths | yes |
| CLAUDE.md "Multi-PLM validation (Exp 46, …)" | `experiments/46_multi_plm_benchmark.py` | imports the same `metrics.statistics` module | yes |
| CLAUDE.md "Unified Codec Benchmark Sweep (Exp 44)" | `experiments/44_unified_codec_benchmark.py` | imports the same `metrics.statistics` module | yes |

All 232-method historical numbers (Exp 25–34) predate the rigorous framework
and are NOT cited in the current CLAUDE.md headline tables; they have NO
bootstrap CIs. The CLAUDE.md tables explicitly cited (Exp 43, 44, 46, 47)
all flow through the same audited code path.

## Probe implementation

`probes/linear.py`:

- **Classification (`train_classification_probe`, lines 15–71):**
  `LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)`,
  wrapped in `GridSearchCV(param_grid={"C": C_grid}, cv=cv_folds, scoring="accuracy", n_jobs=-1)`.
  Default `C_grid = [0.01, 0.1, 1.0, 10.0]`. **CV-tuned, not hardcoded.**
- **Regression (`train_regression_probe`, lines 74–117):**
  `RidgeCV(alphas=alpha_grid, cv=cv_folds)`. Default `alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]`.
  Note: `RidgeCV` is internally CV-tuned, equivalent to `GridSearchCV(Ridge, …)` but
  optimized. Standard sklearn pattern.
- **Random state:** `random_state=42` (or per-seed) is plumbed through both probes.

[GREEN]

## Multi-seed protocol (averaging BEFORE bootstrap)

The Rost-lab-critical question: are predictions averaged across seeds, or
are CIs averaged?

**Answer (verified):** predictions averaged.

`runners/per_residue.py:run_ss3_benchmark` (similar for SS8 and disorder):

```python
seed_per_protein_scores = []
for seed in seeds:                                       # [42, 123, 456]
    probe_result = train_classification_probe(..., seed=seed)
    per_protein_scores = _per_protein_accuracy(...)      # {pid: float}
    seed_per_protein_scores.append(per_protein_scores)

# Average across seeds, then bootstrap (Bouthillier et al. 2021)
q3 = averaged_multi_seed(seed_per_protein_scores, n_bootstrap=n_bootstrap, seed=seeds[0])
```

`averaged_multi_seed` averages per-item scores across seeds (lines 507–511),
then computes a single BCa bootstrap CI on the averaged dict.

For disorder (`run_disorder_benchmark`), the averaging happens at the
**residue prediction** level (lines 489–503): `y_pred_avg = np.mean(y_pred_stacked, axis=0)`
across seeds, per protein and per residue, BEFORE the cluster bootstrap.

This is exactly the Bouthillier prescription. [GREEN]

Note: Ridge regression is closed-form and deterministic — multi-seed averaging
is a no-op for the pointwise predictions but still tracked for protocol
consistency (docstring acknowledges this at `run_disorder_benchmark` lines
427–433).

## Baseline pooling (retrieval fairness)

The CLAUDE.md "Retrieval uses fair baselines (same DCT K=4 pooling for raw and
compressed)" claim. Verified at three levels:

### Phase A1 (SCOPe 5K, run_phase_a1.py lines 478–550)

The runner explicitly computes **three** raw baselines:
- A: `compute_protein_vectors(raw_scope, method="mean")` — context only.
- B: `compute_protein_vectors(raw_scope, method="dct_k4", dct_k=4)` — fair pooling vs codec.
- C: `compute_protein_vectors(abtt_scope, method="dct_k4", dct_k=4)` — full pipeline minus RP.
- Compressed: `codec.encode(raw_scope)["protein_vec"]` which is `dct_summary(projected, K=4)`
  internally (codec_v2.py:249).

Retention is reported against **Baseline C** (line 538), the most stringent
fair comparison (matches preprocessing). The codebase computes paired
bootstrap retention `paired_bootstrap_retention(ret_C["per_query_cosine"], ret_comp["per_query_cosine"], …)`.

[GREEN]

### Exp 46 (run_benchmark_suite, lines 527–536)

Both raw and compressed use `compute_protein_vectors(..., method="dct_k4")`
with `dct_k=4` default. Raw pooling is on the 1024d (or PLM-native dim) raw
embeddings; compressed pooling is on the dequantized per-residue (which lives
in `d_out=896` after RP). Pooling METHOD is identical; the source DIMENSION
differs only because RP is part of the codec definition.

[GREEN] (CLAUDE.md narrative about "fair pooling" refers to method, not
pre-pooling dim; matches the implementation.)

### Exp 47 (run_sweep, lines 296–298, 333–334)

Same pattern as Exp 46. Raw `compute_protein_vectors(sc_embs, method="dct_k4")`
once per PLM; compressed `compute_protein_vectors(comp_sc, method="dct_k4")`
per config. Same pooling, different sources.

[GREEN]

### Exp 44 (44_unified_codec_benchmark.py lines 210, 245)

Raw retrieval baseline: `compute_protein_vectors(raw_scope, method="dct_k4")`.
Compressed: uses `codec.encode(raw_scope)["protein_vec"]` directly (i.e. the
DCT-K=4 vector internal to the codec). Both DCT-K=4. [GREEN]

## Per-table conformance

| Cited table | Bootstrap | Probe CV | Multi-seed | Baseline fair | Status |
|---|---|---|---|---|---|
| CLAUDE.md "Rigorous Retention Benchmarks (Exp 43)" — SS3 / SS8 / Ret / Localization | BCa B=10,000 (n≥25); percentile fallback | `GridSearchCV(C)`, `cv=3`, `random_state` per seed | 3-seed predictions averaged before bootstrap | Baseline C (raw + ABTT3 + DCT K=4); paired retention bootstrap | GREEN |
| CLAUDE.md "Rigorous Retention Benchmarks (Exp 43)" — Disorder pooled ρ | manual BCa via cluster bootstrap, B=10,000; jackknife acceleration | `RidgeCV(alpha)`, `cv=3` | residue-level predictions averaged across seeds before pooling | Disorder retention via `paired_cluster_bootstrap_retention` (joint resample) | GREEN |
| CLAUDE.md "ESM2 Multi-PLM Validation (Exp 43 Phase B)" | same as Phase A1 | same | same | same | GREEN |
| CLAUDE.md "Ablation: Component Contributions (Exp 43 Phase D)" | same | same | same | same | GREEN |
| CLAUDE.md "Unified Codec (Exp 44)" | same code path; `paired_bootstrap_retention` for SS3/SS8/Ret, `paired_cluster_bootstrap_retention` for disorder | same | same | DCT K=4 both | GREEN |
| CLAUDE.md "Multi-PLM validation (Exp 46)" — 5 PLMs × 4 tasks | same | same | same | same | GREEN |
| CLAUDE.md "Codec sweep (Exp 47, ProtT5, standard tiers)" | same | same | same | same | GREEN |
| CLAUDE.md "Legacy benchmarks (Exp 37)" — lDDT 100.7 % / contact 106.5 % | not via this framework — predates Exp 43 | — | — | — | YELLOW (cited as "legacy"; no BCa CI shown) |

## Severity tally

- [GREEN] 13 — All five bootstrap entry points correct and matching CLAUDE.md;
  CV-tuned probes with `random_state` plumbing; multi-seed averaged BEFORE
  bootstrap (Bouthillier-correct); paired retention bootstrap on the right
  ratio; cluster bootstrap on the right level for disorder; jackknife
  acceleration for the manual BCa; percentile fallback for n<25; baseline
  pooling fair across Exp 44/46/47 (DCT K=4 both sides); rules.py asserts
  no train/test leakage at runtime; Ridge non-stochasticity correctly
  documented; Phase A1 explicitly reports 3 baselines (mean, DCT, ABTT+DCT)
  for transparency
- [YELLOW] 1 — Exp 37 lDDT / contact precision numbers in CLAUDE.md are tagged
  "Legacy benchmarks" without BCa CIs (predate Exp 43 framework). Suggest
  either re-running through `metrics.statistics` for parity, or explicitly
  marking as "Exp 37 sigma not reported" in the talk
- [RED] 0

**Distribution:** 13 GREEN / 1 YELLOW / 0 RED.

## Overall verdict

The bootstrap / probe / baseline protocol is rigorous and matches every claim
in CLAUDE.md. The same `metrics.statistics` module backs Exp 43, 44, 46, 47;
no shadow implementation exists. The Rost-lab-required protocol elements
(BCa, paired retention, cluster bootstrap for disorder, multi-seed averaging
before bootstrap, CV-tuned probes) are all present and correctly applied.

The single yellow flag (Exp 37 legacy without BCa) is honest and pre-Exp-43;
not a regression in the current framework.
