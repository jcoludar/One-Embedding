# Audit Findings — Lab Talk Prep

**Status:** Prior recorded; audit pending.

## Prior (written before audit, 2026-04-26)

What I currently expect the audit to find:

### Greens (expected to hold up)
- Bootstrap CIs (BCa, B=10000) — Exp 43 onwards used the rigorous protocol consistently.
- Multi-seed averaging-before-bootstrap — implemented post-Bouthillier.
- CV-tuned probes (`GridSearchCV` on C/alpha) — Exp 43 onwards.
- Pooled disorder ρ + cluster bootstrap — Exp 43 fix.
- 5-PLM Exp 46 splits — same partition across PLMs by construction (single split, embeddings re-extracted).

### Yellows (expected to need clarification, not fixes)
- Disorder retention 94.9 % — real, but the floor (raw 1024d) and the noise of probes need to be co-reported.
- ANKH disorder retention 94.8 % — same caveat.
- DCT K=4 — chosen empirically; need a one-line "why not K=8" answer.
- The receiver-side "no codebook" claim for binary — needs a clean demo (decoder uses `h5py + numpy` only, nothing from `OneEmbeddingCodec`).

### Reds (expected to require fixes or honest demotion)
- `one_embedding/` package referenced in CLAUDE.md / MEMORY.md but not at repo root → confirm location, fix references everywhere.
- Modified phylo file (n_taxa 156→24) → recover or document the canonical run.
- Possibly: a few claims in CLAUDE.md / README without a traced source experiment.
- Possibly: dead code, TODO/FIXME markers in `src/`.
- Possibly: pytest failures in the current state (we have not run the full suite recently).

### Predicted distribution
~70 % green, ~20 % yellow, ~10 % red.

## Posterior (filled by audit, Tasks C.1–C.10)

### Repo hygiene (Task C.1, evidence: `docs/_audit/hygiene.md`)

- [GREEN] `git status --short` is clean at audit start (HEAD `7cc2e72`).
- [GREEN] `.gitignore` comprehensive: 67 GB of data correctly excluded; `.claude/`, `.venv/`, `.worktrees/`, secrets all covered.
- [GREEN] All 7 tracked binary/weight files (`.npz`, `.pt`) are ≤ 700 KB. No accidental large-file pollution.
- [GREEN] All code-level imports use `from src.one_embedding.codec_v2 ...` — consistent with reality. CLAUDE.md is internally consistent on this.
- [GREEN] Recent commits are well-scoped and well-named (Phase B prior stubs land cleanly).
- [YELLOW] CLAUDE.md presents both 768d (Exp 44) and 896d (Exp 47) tables without an explicit "legacy" tag on the 768d one. Two adjacent benchmark generations in one doc is confusing. Fix during D.1.
- [YELLOW] Test count drift: CLAUDE.md says 798, MEMORY.md still mentions 795 and 632 in different sections. Reconcile after pytest baseline (C.2). Fix during D.1.
- [YELLOW] `data/benchmarks/embedding_phylo_results.json` had n_taxa reduced from 156 → 24 by an earlier ad-hoc run; the file is committed but the canonical 156-taxon result needs to be confirmed/restored. Phase D fix.
- [RED] README.md describes the OLD default everywhere except the TL;DR headline:
  - line 12: "**44** experiments" (should be 47);
  - lines 21–34 Quick Start: `OneEmbeddingCodec()` annotated as "default PQ M=192 on 768d, ~20x";
  - lines 27–30 receiver comment: "h5py + numpy + codebook" (current binary default needs no codebook);
  - lines 76–101 tables and "When to Use What" all 768d-anchored, PQ-default;
  - lines 152–166 pipeline diagram and prose still recommend ABTT k=3 (off by default since Exp 45).
  This is the single biggest drift item. Major Phase D rewrite required.
- [RED] User auto-memory `~/.claude/projects/.../memory/MEMORY.md` (lines 91–96, 113, 175) treats `one_embedding/` as a top-level package. NOT a checked-in file (cannot fix in repo), but worth noting for future sessions.

**Distribution:** 5 GREEN / 3 YELLOW / 2 RED.

### Code correctness (Task C.2, evidence: `docs/_audit/pytest_baseline.txt`, `code_markers.txt`, `codec_review.md`)

- [GREEN] **813/813 tests pass** in 90 s. (Docs claim 798 — outdated, undercount; the *actual* count is higher. Re-state in CLAUDE.md/MEMORY.md as Phase D.1 fix.)
- [GREEN] **Zero TODO/FIXME/HACK/XXX markers in `src/`.** Only one true `[TODO]` exists outside `src/`: `experiments/50_sequence_to_oe.py:439` ("Downstream evaluation (SS3, disorder, retrieval)") — known follow-up for the active Stage 3 work, not a defect.
- [GREEN] **Receiver-side decode claim VERIFIED** for binary default: H5 file contains `per_residue_bits` + `means` + `scales`, decoded by ~12 lines of pure `numpy` with `h5py` for I/O. No codebook needed. CLAUDE.md headline claim holds.
- [GREEN] **Receiver claim also holds** for int4 and lossless/fp16. PQ correctly requires a codebook (matches the CLAUDE.md narrative).
- [GREEN] `OneEmbeddingCodec` defaults (lines 84–93) match CLAUDE.md prose: `d_out=896`, `quantization='binary'`, `pq_m=None` (auto), `abtt_k=0`, `dct_k=4`, `seed=42`. Constructor docstring (lines 64–82) is correctly aligned with current state.
- [YELLOW] **Hidden defaults / silent skips in codec_v2.py** — none are bugs, but each could surprise a careful Rost-lab user:
  - `auto_pq_m` docstring example only shows 768/512 cases (line 50–51), not the new 896 default.
  - `_preprocess` (line 147) silently SKIPS centering when `is_fitted == False` and quantization is binary/int4 (no codebook needed). User encoding without `fit()` gets uncentered binary — works but inconsistent with class docstring's "default: center + RP + binary".
  - `compute_corpus_stats(..., n_pcs=5)` is hardcoded (line 167), but `abtt_k` is user-configurable. Setting `abtt_k=10` would silently use only 5 PCs (Python slice `top_pcs[:10]` truncates without warning).
  - `encode_h5_to_h5` writes file-level metadata (`quantization`, `d_out`) but per-protein groups only get `seq_len` and `d_in` attrs. External users inspecting `f[pid].attrs` would miss key info; only `load_batch` correctly merges them.
  - `version: 4` is hardcoded; no upgrade path documented for older `.one.h5` files.
- [YELLOW] **Self-contained binary decoder snippet missing from docs.** While the receiver CAN decode with `numpy + h5py` only, they need to know the bit-unpacking layout (bit 7 → col 0 in column-major within byte). For the talk + paper, ship a 15-line standalone snippet. Fix during D.1.
- [YELLOW] Two informational `RuntimeWarning`s during pytest (BCa CI degenerate-data warning in `test_benchmark_*` and "catastrophic cancellation" in `test_transposed_transforms` on a deliberately-constant matrix). These are tests verifying degenerate-input handling, not failures. Document in pytest_baseline notes; no fix needed.
- [GREEN] No RED findings in this section. Code is in good shape.

**Distribution:** 6 GREEN / 3 YELLOW / 0 RED.

### Combined posterior so far (C.1 + C.2)

11 GREEN / 6 YELLOW / 2 RED.

Distribution mostly matches the prior prediction (~70 % green / 20 % yellow / 10 % red). The two REDs are exactly what I anticipated — README drift and `one_embedding/` package-location confusion in MEMORY.md — both Phase D.1 fixes.

### Splits (Task C.3, evidence: `docs/_audit/splits.md`)

- [GREEN] **Exp 46 multi-PLM split fairness VERIFIED.** Both `cb513_80_20.json` and `esm2_650m_5k_split.json` are loaded once per benchmark call (lines 456–461 of `46_multi_plm_benchmark.py`) and applied to whichever PLM's H5 file is being tested — the train/test ID list never depends on PLM identity. Single source of truth. The headline "5-PLM, same split" claim holds.
- [GREEN] **SCOPe split is strictly cluster-controlled.** `data/benchmark_suite/splits/esm2_650m_5k_split.json` records `superfamily_overlap=0` and `family_overlap=0` in its `statistics` block — no train/test family or superfamily collision. The codec is fitted on the train half only; test families are unseen during ABTT/PQ fitting.
- [GREEN] **Train/test uniqueness is asserted at runtime.** `rules.check_no_leakage` raises on any ID appearing in both; it is called at the top of `run_ss3_benchmark`, `run_ss8_benchmark`, `run_disorder_benchmark` (`runners/per_residue.py:267, 351, 467`).
- [GREEN] **Disorder splits use predefined non-redundant partitions.** CheZOD = SETH (Dass et al. 2020) split; TriZOD = TriZOD348 (Haak 2025) split — both cluster-curated by their original publications.
- [GREEN] **Codec fit corpus is external to all test sets.** All Exp 43 / 44 / 46 / 47 results fit ABTT/PQ on the SCOPe 5K train subset and evaluate on CB513 / CheZOD / TriZOD / SCOPe-test — distinct datasets or held-out IDs. Exp 43's `run_abtt_stability.py` separately verifies cross-corpus ABTT stability (variance < 0.2 pp Ret@1 across 4 fitting corpora).
- [GREEN] Exp 44 and Exp 47 inherit Exp 43's split files via shared `config.py`; verified by file-path identity (no per-experiment shadow splits).
- [YELLOW] **CB513 train/test is a within-CB513 random split** (408/103, seed=42). CB513 is already `<25 % seq id` by dataset design, so additional structural leakage from the random split is bounded — but a lab-Q probe will land here first. **Pre-empt with one sentence in the talk:** "CB513 is `<25 % id` by construction; we use the published 408/103 random split with seed=42." Not a real defect, but a presentation choice.
- [YELLOW] **Exp 46 SCOPe split filename is misleading.** It is named `esm2_650m_5k_split.json` (historical artifact from Exp ~17), but the split is PLM-agnostic — used by ProtT5, ESM2, ESM-C, ProstT5, ANKH alike. Phase D rename + loader-update.
- [YELLOW] **Exp 50 sighting results use a random 80/10/10 split** (`50_sequence_to_oe.py:82–94`). The design spec at `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md` explicitly identifies this and the rigorous CATH-cluster re-run plan (`docs/superpowers/plans/2026-04-06-exp50-rigorous-cath-split.md`) is the planned fix; Task 6 (MMseqs2 leakage audit) of that plan has not been executed (`results/exp50_rigorous/` does not exist). **Talk should cite Exp 50 as in-progress, not as a final number.**
- [RED] None.

**Distribution:** 6 GREEN / 3 YELLOW / 0 RED.

### Statistics (Task C.4, evidence: `docs/_audit/stats.md`)

- [GREEN] **Bootstrap is BCa B=10,000 with percentile fallback** for n<25 and on NaN / exception. Implemented via `scipy.stats.bootstrap(method="BCa")` for plain & paired cases (`metrics/statistics.py:69–104, 158–196`), and via manual BCa with jackknife acceleration for the cluster-bootstrap cases (`metrics/statistics.py:319–366, 423–469`). Constant `BOOTSTRAP_N=10_000` in `config.py`.
- [GREEN] **Paired bootstrap is correct for retention.** `paired_bootstrap_retention` calls `scipy.stats.bootstrap(..., paired=True)` so the same protein-id resample drives raw and compressed in every iteration — correlated noise cancels. Used by Exp 43 / 44 / 46 / 47 for SS3 / SS8 / Ret@1 retention.
- [GREEN] **Disorder uses cluster bootstrap, not residue bootstrap.** `cluster_bootstrap_ci` resamples at the protein (cluster) level then recomputes the pooled Spearman ρ over all residues from the selected proteins (Davison & Hinkley 1997). For retention, `paired_cluster_bootstrap_retention` jointly resamples the same cluster IDs for raw and compressed.
- [GREEN] **Multi-seed predictions averaged BEFORE bootstrap (Bouthillier 2021).** `averaged_multi_seed` averages per-item scores across seeds, then bootstraps the averaged dict — NOT a CI of CIs. For disorder, residue-level predictions are averaged across seeds in `run_disorder_benchmark` (lines 489–503) before the cluster bootstrap.
- [GREEN] **Probes are CV-tuned, not hardcoded.** Classification: `LogisticRegression` wrapped in `GridSearchCV(param_grid={"C": [0.01,0.1,1.0,10.0]}, cv=3, scoring="accuracy")`. Regression: `RidgeCV(alphas=[0.01,0.1,1.0,10.0,100.0], cv=3)`. `random_state=seed` is plumbed through.
- [GREEN] **Same `metrics.statistics` module backs Exp 43, 44, 46, 47.** No shadow implementation. Verified by import statements: each experiment script imports from `experiments/43_rigorous_benchmark/metrics/statistics.py` (via `sys.path` insert).
- [GREEN] **Retrieval baselines are fair (DCT K=4 on both raw and compressed).** Exp 43 Phase A1 reports three explicit raw baselines (mean / DCT-K=4 / ABTT+DCT-K=4) and computes retention against the most stringent (Baseline C). Exp 44/46/47 use DCT K=4 on both raw and compressed throughout.
- [GREEN] **Train/test leakage assertion `rules.check_no_leakage` runs at the top of every per-residue benchmark** (`runners/per_residue.py:267, 351, 467`) — raises if any ID appears in both lists.
- [GREEN] **Per-table conformance verified for all 7 cited CLAUDE.md tables** (Exp 43 SS/Ret/Localization, Exp 43 Disorder, Exp 43 Phase B ESM2, Exp 43 Phase D Ablation, Exp 44, Exp 46, Exp 47). Every cell is produced by the same audited code path.
- [YELLOW] **Exp 37 legacy lDDT / contact precision numbers** (CLAUDE.md "Legacy benchmarks (Exp 37)" — `lDDT 100.7 %`, `contact precision 106.5 %`) predate the Exp 43 rigorous framework and are reported WITHOUT BCa CIs. Either re-run through `metrics.statistics` or explicitly say "Exp 37 sigma not reported" in the talk.
- [RED] None.

**Distribution:** 9 GREEN / 1 YELLOW / 0 RED.

### Phylo (Task C.5, evidence: `docs/_audit/hygiene.md` "Task C.5" section)

- [GREEN] **Provenance traced.** `data/benchmarks/embedding_phylo_results.json` is written exclusively by `experiments/35_embedding_phylogenetics.py:2387`. The path is hardcoded at line 2086 — NOT parameterized by `--dataset` — so each invocation overwrites the previous content with whatever dataset was run last. Per-dataset detailed outputs DO use parameterized filenames (`results/embed_phylo/{ds}_*.nwk`, 31 datasets present).
- [GREEN] **Diff is exactly the documented deltas.** `n_taxa: 156→24` and `mcmc_time_s: 89.81→7.88`; all other fields (n_dims=512, mode=per_protein, n_generations=20000, n_runs=1, n_chains=2, asdsf=0.0, converged=true) are bit-identical between versions.
- [GREEN] **Zero downstream dependency.** `embedding_phylo_results.json` is not read by any other Python script or doc; it is a sanity artifact only. The cited "10–11 of 12 families monophyletic" claim in CLAUDE.md / MEMORY.md comes from analyzing `results/embed_phylo/*_consensus.nwk`, not from this file.
- [GREEN] **Decision applied: option (a) — restore 156-taxa version.** Restored via `git show 8b1fbf1:data/benchmarks/embedding_phylo_results.json > data/benchmarks/embedding_phylo_results.json`. Rationale: the 156-taxa run is the more representative artifact (longer convergence time on a real-sized dataset), the 24-taxa was an in-flight downsized rerun, and the file has no downstream consumers so the choice is purely about leaving the most representative sanity number in the repo.
- [YELLOW] **Both versions used trivial-test config** (20K generations × 1 run × 2 chains; ASDSF=0.0 indicates insufficient sampling rather than true convergence). Neither matches the rigorous full config (200K generations × 4 chains × 2 runs, the script's argparse defaults). For the talk, do not claim this file represents the rigorous run; cite the per-dataset `_consensus.nwk` outputs instead.
- [YELLOW] **Script bug to fix in Phase D:** parameterize `BENCH_PATH` in `experiments/35_embedding_phylogenetics.py:2086` by `--dataset` (e.g. `embedding_phylo_{ds}_results.json`) to prevent future overwrites. Also add a one-line note to `EXPECTED_QA.md` so a Rost-lab member spotting the n=24 run knows it's a sanity artifact.
- [RED] None — the resolved status is acceptable for talk purposes.

**Distribution:** 4 GREEN / 2 YELLOW / 0 RED.

### Parameters (Task C.6, evidence: `docs/_audit/params.md`)

- [GREEN] **`quantization='binary'`** is the best-evidenced default. Exp 47 directly compares binary-896 (37x) vs PQ128-896 (32x): binary wins on disorder (94.9% vs 91.4%), is decoder-self-contained (no codebook), and matches PQ on SS3/Ret. Exp 44 (`exp44_unified_results.json`) confirms RaBitQ effect on the older 768d sweep too.
- [GREEN] **`abtt_k=0`** justified by Exp 45 disorder forensics: ABTT PC1 73 % aligned with the Ridge disorder direction; per-stage decomposition shows ABTT alone causes 50 % of total disorder loss; dropping it recovers +3.3 pp disorder for −0.6 pp retrieval. Headline-defensible.
- [GREEN] **`pq_m='auto'` (= d_out//4)** rule documented in code (codec_v2.py:46–57); selection target is "~4d sub-vectors per subquantizer". Both d_out=768→M=192 (Exp 44) and d_out=896→M=224 (Exp 47) used the rule and are directly validated.
- [GREEN] **`seed=42`** — Exp 29 part_D formally tested 10 RP seeds: Ret@1 std=0.004, max-min=0.015. Seed=42 sits within 1 SD of the mean. Choice is irrelevant beyond reproducibility.
- [YELLOW] **`d_out=896`** is **defensible but interpolated.** No clean d_out ∈ {512, 768, 896, 1024} sweep at fixed quantization exists. The 896 choice is justified by (a) divisibility for PQ subquantizers, (b) avoiding ABTT3+768d's ProstT5 catastrophe (SS3 → 85.6 %), and (c) Exp 47 directly testing PQ M=224 at 896d. The "why 896 not 1024 or 768?" probe has only an inferred answer, not a measured one. Pre-empt with one sentence in the talk.
- [YELLOW] **`dct_k=4`** has weaker evidence than implied. EXPERIMENTS.md states "DCT K=4 is the sweet spot. Higher K hurts" — but Exp 22 raw `path_geometry_results.json` shows K=8 Ret@1 > K=4 Ret@1 (0.712 vs 0.666) on a displacement-DCT proxy. No formal K-sweep on the current preprocessed pipeline (center → RP896 → binary → DCT K). The right framing is **storage**: K=4 → protein_vec = D×4 fp16 (7 KB); K=8 doubles that. Frame as a compression choice, not a measured quality optimum.
- [YELLOW] **Hidden defaults** flagged in C.2 (`codec_review.md`) confirmed here:
  - `compute_corpus_stats(n_pcs=5)` is hardcoded — setting `abtt_k=10` would silently use only 5 PCs (slice truncation, no warning).
  - `version=4` is hardcoded with no documented upgrade path.
  - `_preprocess` silently skips centering when `is_fitted=False` and quantization is binary/int4 (caught in C.2).
- [YELLOW] **Commit-message drift** — current defaults landed in commit `34e159a` (titled `chore: gitignore results/, .claude/, large data files`) but the rationale lives in the **sibling** commit `8b1fbf1` (`feat: Exp 45-47 — disorder forensics, 5-PLM pipeline, binary default`) which doesn't touch `codec_v2.py` itself. A Rost-lab reader using `git blame` on the default lines would land on a misleadingly-named hygiene commit. Documentation, not correctness; no fix in audit.
- [RED] None — every default has at least an inferred-but-defensible evidence chain. No "someone just picked it" cases.

**Distribution:** 4 GREEN / 4 YELLOW / 0 RED.

### Claims register (Task C.7, evidence: `docs/_audit/claims.md`)

- [GREEN] **All Exp 43 result-table cells (Phase A1, B, C, D) match source JSONs bit-perfectly.** SS3/SS8/Disorder per-residue numbers, Ret@1 SCOPe 5K, CATH20, DeepLoc test+setHARD, ESM2 Phase B (SS3/SS8/Ret@1), ablation table, length stress — every cited number traces to `phase_a1/b/c/d_results.json` exactly (rounding ≤ 0.05).
- [GREEN] **Exp 46 5-PLM table is bit-perfect** (5 PLMs × 4 tasks × {value, ±} = 40 cells, all match `exp46_multi_plm_results.json`).
- [GREEN] **Exp 47 codec sweep table** (6 standard tiers on ProtT5) bit-perfect against `exp47_codec_sweep.json`.
- [GREEN] **Exp 44 unified codec sweep table** (6 configs × 4 metrics = 24 cells) bit-perfect against `exp44_unified_results.json`.
- [GREEN] **Exp 37 legacy structural numbers** (lDDT 100.7%, contact 106.5%, n=50 SCOPe domains) match `structural_retention_results.json` exactly.
- [GREEN] **All sample-size claims** (n=2493, 9518, 2768, 490, 117, 348, 103, 115, 20) match the `n=` fields in the source JSONs.
- [GREEN] **Methodology claims** (BCa B=10,000; multi-seed [42,123,456]; GridSearchCV; cluster bootstrap; 3 fair retrieval baselines; ABTT cross-corpus stability < 0.2 pp) all trace to source code/JSONs (re-verified from C.4).
- [GREEN] **Bibliographic citations** (DiCiccio & Efron 1996; Bouthillier 2021; Davison & Hinkley 1997) are real, accurate references.
- [YELLOW] **"232 compression methods benchmarked"** — roll-up estimate, not enumerated. Distinct counts in EXPERIMENTS.md sum approximately to this figure (29 pooling + 50 extreme + 30+ Exp 29 + 14 Exp 26 + 10 Exp 47 × 5 PLMs + ...) but no script outputs "232". Pre-empt with one slide: either soften to "200+" or list explicitly.
- [YELLOW] **"6 tasks"** — Exp 46 actually reports 4 tasks (SS3, SS8, Ret@1, Disorder). The "6" framing rolls in CB513/TS115/CASP12 SS variants as separate tasks. Loose definition inflates count. **A Rost-lab member will count tables and ask.**
- [YELLOW] **"8 datasets"** — actually 9 in cited tables (CB513, TS115, CASP12, CheZOD117, TriZOD348, SCOPe 5K, CATH20, DeepLoc test, DeepLoc setHARD). Off by one.
- [YELLOW] **"L=175" reference protein length** in compression-tier table titles. NOT empirical; a fixed assumption from the codec design spec (`docs/superpowers/specs/2026-03-29-unified-codec-design.md:59`). Exp 45 separately reports `mean_protein_length=156` for the actual SCOPe 5K subset. The L=175 number is an arithmetic convenience, not a measured value. Not an error per se but the "Size (L=175)" label invites confusion.
- [YELLOW] **"1500 proteins/s encoding"** and **"20x faster than PQ"** speed claims live only in commit message `8b1fbf1`, not in any result JSON. PQ side has timing in `exp45_new_default_results.json` (PQ encode 70.6 µs/residue → ~80 prot/s); 1558/80 ≈ 19.5x → "20x" reasonable. But binary timing has no JSON record.
- [YELLOW] **fp16-896 compression "2.3x" vs JSON label "2x"** — both correct (4096/(896×2) = 2.286 → 2.3 doc, 2 JSON-rounded). Cosmetic precision inconsistency.
- [YELLOW] **AUC retention "98.6%"** vs computed `0.877/0.890 = 98.54%` — off by 0.06pp due to rounding intermediate values. Minor.
- [YELLOW] **Pre-rigorous-era numbers** in README L208–L213 (V2 full / V2 balanced / V2 binary), L330–L352 (Exp 26/28/36/37 narrative), and Addendum (ChannelCompressor `Ret@1=0.795 ± 0.012`). Reference older result files (`chained_codec_results.json`, `extreme_compression_results.json`, etc.) that were NOT re-verified cell-by-cell in this audit. Disclaim in talk as "pre-rigorous, no BCa CIs".
- [YELLOW] **TM-score Spearman 0.5742** (Exp 37 `structural_retention_results.json:spearman_retention`) is **NOT cited anywhere in CLAUDE.md or README.md** despite living in the same result file as the headline lDDT 100.7% / contact 106.5%. **Omission, not error** — but a Rost-lab probe ("what about TM-score?") will land here. Either disclose (57% retention) or explicitly note "TM-score retention low; lDDT/contact preserved" in the talk.
- [YELLOW] **Phylo "FastTree 4/12, IQ-TREE 5/12, BM MCMC 11/12"** numbers in README L138–L143. The MCMC results are in per-tree `_consensus.nwk` files (per C.5), but the FastTree/IQ-TREE comparison numbers don't have an obvious source script in the audit-traced JSONs. Likely correct (consistent with the rest of Exp 35 narrative) but the trace is implicit, not explicit.
- [RED] **README.md "44 experiments"** (line 12, 318) — should be 47. (Same item as C.1 RED.)
- [RED] **README.md "default PQ M=192 on 768d, ~20x"** (line 21) — old default; should be binary 896d ~37x. (Same as C.1 RED.)
- [RED] **README.md "ABTT3 + RP 768d + PQ M=192 → ~20x compression, ~34 KB/protein"** (line 80, 316) — old default; same drift item.
- [RED] **README.md tier table "PQ M=192 (default)" cell** (line 87, 178) — old default; same drift item.

**Distribution:** 8 GREEN / 11 YELLOW / 4 RED (across the 28 most material claims; full register has 78 cells with 50 GREEN / 24 YELLOW / 4 RED).

The 4 REDs are all the same "README is out of date" item (already a single RED in C.1 — the 4 line items here are distinct **numeric** claims that all share that root cause).

### Combined Posterior so far (C.1 + C.2 + C.3 + C.4 + C.5 + C.6 + C.7)

| Subsection | GREEN | YELLOW | RED |
|---|---:|---:|---:|
| Repo hygiene (C.1) | 5 | 3 | 2 |
| Code correctness (C.2) | 6 | 3 | 0 |
| Splits (C.3) | 6 | 3 | 0 |
| Statistics (C.4) | 9 | 1 | 0 |
| Phylo (C.5) | 4 | 2 | 0 |
| Parameters (C.6) | 4 | 4 | 0 |
| Claims register (C.7) | 8 | 11 | 4 |
| **Total** | **42** | **27** | **6** |

(C.7 RED items 1–4 are sub-instances of the same C.1 RED #1 — the README is one drift hotspot
generating multiple distinct numeric mismatches. The "true" RED *count* is 2 (README drift +
MEMORY.md `one_embedding/` confusion); the table totals show 6 line items because we count each
distinct numeric claim separately.)

The cumulative ratio (~56 % green / 36 % yellow / 8 % red after C.7) is more
yellow than the prior predicted (~70/20/10). Reason: C.6/C.7 surface a class
of "loosely-defined or commit-message-only" claims that were not anticipated.
None invalidate a headline result — but they show that several of the "round
numbers" in the docs (232 methods, 6 tasks, 8 datasets, 1500 proteins/s, L=175)
are presentation conveniences rather than measured quantities.

**No Exp 43/44/46/47 retention or CI is invalidated.** The methodologically-rigorous
half of the docs is bit-perfect against its source JSONs. The drift is all
in (a) older README narrative, (b) marketing-layer count claims, and (c)
the L=175 reference-protein assumption.

### Tooling (Task C.8, evidence: `docs/_audit/tooling.md`)

- [GREEN] **`pytest`** — declared (`>=9.0.2` in `[dependency-groups].dev`), installed (9.0.2), runnable. 813/813 tests pass (C.2).
- [YELLOW] **`ruff`** missing — no linter declared, no `[tool.ruff]` section, no `.ruff.toml`. Not a blocker, but a clone-and-look inspector will notice the absence.
- [YELLOW] **`mypy`** missing — no type checker, no `mypy.ini`, no `[tool.mypy]`. No `py.typed` marker either. Same severity as `ruff`: optics, not correctness.
- [YELLOW] **`pytest-cov`** missing — coverage is not measured. The "813 tests" claim is undercut by having no coverage figure.
- [YELLOW] **`pre-commit`** missing — no `.pre-commit-config.yaml`, no hooks installed. Visible to anyone who clones; not blocking.
- [YELLOW] **`jupyter` / `jupytext`** missing — `experiments/45_disorder_forensics.ipynb` exists in the repo and would fail to re-execute on a fresh clone. Not on talk critical path.
- [RED] **`marp-cli`** missing AND blocks Task H.1 (slide deck production). `marp` is not on `PATH`; `npx` IS available (`/opt/homebrew/bin/npx` v11.6.2), so install path is `npm install -g @marp-team/marp-cli` in D.4. This is the only tool gap with a concrete downstream block.

**Distribution:** 1 GREEN / 6 YELLOW / 1 RED.

The single RED gates slide production, not any benchmark claim. All YELLOWs are repo-hygiene gaps that D.4 will close (declare in `[dependency-groups].dev`, add minimal `[tool.ruff]` and `[tool.mypy]` blocks, add a basic `.pre-commit-config.yaml`). No code change required.

### Dependencies (Task C.9, evidence: `docs/_audit/deps.md`)

- [GREEN] **`uv lock --check` exits 0** — `uv.lock` is in sync with `pyproject.toml` (97 packages resolved without re-locking).
- [GREEN] **No phantom deps.** Every dep in `[project].dependencies` and the `biocentral` / `extreme` extras maps to a real top-level import in `src/` or `experiments/`.
- [GREEN] **No yanked / EOL packages** in the lock (`grep -i yank uv.lock` empty).
- [GREEN] **`requires-python = ">=3.12"` consistent** across `pyproject.toml`, `uv.lock`, and `.venv` (Python 3.12.12).
- [GREEN] **False-positive `datasets` is a local subpackage**, not the HuggingFace library — at `experiments/43_rigorous_benchmark/datasets/{netsurfp,trizod}.py`.
- [GREEN] **`deptry` deferred to D.5.** Manual import-grep used as fallback (limitation noted; CI-enforced check is a Phase D recommendation).
- [YELLOW] **`uv sync --dry-run` would uninstall 16 venv-only packages** — `numba`, `umap-learn`, `dgl`, `e3nn`, `faiss-cpu`, `opt-einsum*`, `pyarrow`, `psutil`, `protobuf`, `pynndescent`, `requests`, `torchdata`, `llvmlite`, `charset-normalizer`, `urllib3`. Drift from earlier ad-hoc `uv pip install`. Cleanup = run `uv sync` (no `--dry-run`) in D.5.
- [YELLOW] **`scipy` is imported in 16 files but not declared in pyproject.** Available as transitive of `scikit-learn`/`pot`/`ripser`. Fragile if any parent drops it. Promote to direct dep in D.5.
- [YELLOW] **`click` is imported in 2 files but not declared in pyproject.** Available as transitive of `typer` ← `huggingface-hub` ← `transformers`. Same fragility class as `scipy`. Promote to direct dep in D.5.
- [RED] **`faiss-cpu` is imported in `src/one_embedding/structural_similarity.py` (5 sites) but not declared in `pyproject.toml` or `uv.lock`.** A fresh clone + `uv sync` would NOT install it; the FAISS retrieval index path is broken on a clean install. Declare as `[project.optional-dependencies].structural`.
- [RED] **`tmtools` is imported in `src/evaluation/structural_validation.py` (TM-score path) but not declared.** Fresh clone + `uv sync` would not install it. Currently lazy-imported with a graceful error message; should still be declared.

**Distribution:** 6 GREEN / 3 YELLOW / 2 RED.

Both REDs are missing `[project.optional-dependencies]` entries for OPTIONAL paths (FAISS index / TM-score). **Neither breaks the headline codec / Exp 43–47 retention numbers**, which only need the already-declared `numpy`, `h5py`, `scipy`, `scikit-learn`, `torch`, `transformers`. A Rost-lab clone-and-encode test for the binary default (`numpy + h5py` only on receive side) succeeds without `faiss` / `tmtools`.
