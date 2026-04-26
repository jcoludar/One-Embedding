# Audit Group 2 (C.3 + C.4 + C.5) — Spec Compliance Review

**Subagent type:** general-purpose
**Reviewing:** commits `a88c589` (C.3), `088bf37` (C.4), `1f10f41` (C.5)

## Verification Report

**Result: SPEC COMPLIANT** with zero substantive deviations.

### Commits and file modifications

| Commit | Message | Files | Match plan? |
|---|---|---|---|
| `a88c589` | `audit(splits): leakage check across all cited benchmarks` | `docs/_audit/splits.md` (new, 153 lines), `docs/AUDIT_FINDINGS.md` (+15) | YES — verbatim message, only expected files |
| `088bf37` | `audit(stats): bootstrap + probe + baseline conformance` | `docs/_audit/stats.md` (new, 238 lines), `docs/AUDIT_FINDINGS.md` (+16) | YES — verbatim message, only expected files |
| `1f10f41` | `audit(phylo): document the 24-taxa rerun decision` | `docs/_audit/hygiene.md` (+60), `docs/AUDIT_FINDINGS.md` (+32), `data/benchmarks/embedding_phylo_results.json` (4-line restore) | YES — verbatim message; phylo restore is the documented option (a) |

### No-fixes rule respected

- `git diff a88c589^ 1f10f41 -- src/` → 0 lines (no source modifications)
- `git diff a88c589^ 1f10f41 -- experiments/` → 0 lines (no experiment modifications)
- Only data-area file modified is `data/benchmarks/embedding_phylo_results.json` — the documented C.5 restore.

### Plan-step completeness

- **C.3 (5 steps):** all addressed in `splits.md` per-experiment sections (Exp 43/44/46/47/50). The optional `scripts/audit/check_split_leakage.py` was NOT created — but the plan said "Possibly create" and the implementer instead leveraged the existing `experiments/validate_split_leakage.py` (cited at `splits.md:16-18`). Acceptable.
- **C.4 (5 steps):** all addressed; per-table conformance table at `stats.md:197-208` covers all 7 cited CLAUDE.md result tables plus the legacy Exp 37 row marked YELLOW.
- **C.5 (4 steps):** all addressed; option (a) chosen and documented in detail.

### Spot-check verification of citations (anti-fabrication)

All cited file:line locations verified to exist and contain the claimed code:
- `experiments/46_multi_plm_benchmark.py:456-461` — split loading, **confirmed**
- `experiments/43_rigorous_benchmark/metrics/statistics.py:60-100` — BCa with percentile fallback, **confirmed**
- `experiments/43_rigorous_benchmark/metrics/statistics.py:155-170` — paired bootstrap with `paired=True`, **confirmed**
- `experiments/43_rigorous_benchmark/rules.py:73` — `check_no_leakage`, **confirmed**
- `experiments/43_rigorous_benchmark/runners/per_residue.py:267, 351, 467` — leakage assertions, **all 3 confirmed**
- `experiments/43_rigorous_benchmark/config.py:47-51` — `SEEDS=[42,123,456]`, `BOOTSTRAP_N=10_000`, `CV_FOLDS=3`, `C_GRID`, `ALPHA_GRID`, **all confirmed verbatim**
- `experiments/50_sequence_to_oe.py:82-94` — random 80/10/10 split with seed=42, **confirmed**
- SCOPe split file `data/benchmark_suite/splits/esm2_650m_5k_split.json` actually contains `"superfamily_overlap": 0`, `"family_overlap": 0` (`statistics` block) — **confirmed by direct json read**
- `experiments/35_embedding_phylogenetics.py:2086` — hardcoded `BENCH_PATH`, **confirmed**

### `AUDIT_FINDINGS.md` structure

- Prior section preserved verbatim (matches `git show 532146c:docs/AUDIT_FINDINGS.md`).
- Posterior section now contains five sub-sections plus a cumulative table — matches the C.1/C.2 convention.
- Severity tags consistent with Group 1.
- Combined posterior table at lines 123-132 with totals 30 GREEN / 12 YELLOW / 2 RED.

### Phylo restore (C.5)

The diff in `embedding_phylo_results.json` is exactly the two documented changes:
- `n_taxa: 24 → 156`
- `mcmc_time_s: 7.877 → 89.811`

All other fields bit-identical. Sourced from the `8b1fbf1` commit. The "zero downstream consumers" claim verified: `grep -r "embedding_phylo_results.json"` shows the file is referenced only by docs and the writer script `experiments/35_embedding_phylogenetics.py:2086`. **No script reads it.**

### Was option (a) the right call?

**Yes.**
- Both versions are sanity-test config; the choice is between two equally-trivial artifacts.
- 156-taxa is the version committed in the official `8b1fbf1` commit — i.e., the version on `main` before Task A.1 captured the in-flight overwrite. Restoring it is the conservative choice.
- The 24-taxa downsize was an in-flight ad-hoc rerun and has no inherent claim to canonicity.
- Option (b) "keep 24 as intentional" would require defending why a downsized rerun is the canonical reference.
- Option (c) "leave ambiguous, flag in EXPECTED_QA.md" defers the decision the audit was supposed to make.
- The implementer also recorded two Phase D follow-ups (parameterize `BENCH_PATH` by `--dataset`; note the file's sanity-artifact nature in `EXPECTED_QA.md`).

### Summary

- **C.3:** 6 GREEN / 3 YELLOW / 0 RED — accurate.
- **C.4:** 9 GREEN / 1 YELLOW / 0 RED — accurate.
- **C.5:** 4 GREEN / 2 YELLOW / 0 RED — accurate.

**Combined posterior 30 GREEN / 12 YELLOW / 2 RED.**

No issues found.
