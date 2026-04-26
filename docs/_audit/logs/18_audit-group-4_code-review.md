# Audit Group 4 (C.8 + C.9 + C.10) — Code Quality Review

**Subagent type:** superpowers:code-reviewer
**Reviewing:** BASE `83d21ef` → HEAD `db3a0fc`

## Overall Assessment

**APPROVED with minor bookkeeping noise.** All three commits substantively correct, evidence-grounded, ready for Phase D inline execution. Independent verification confirms every load-bearing claim: `uv lock --check` exit 0, all 5 faiss import sites at cited line numbers, tmtools at line 98, missing tools genuinely missing, only `pytest` declared in `pyproject.toml`. The triage table is the consequential artifact and it is implementable as-is.

The discrepancies found are bookkeeping noise (off-by-one counts, slight mis-interpretation of one Group 3 nit) that don't compromise the audit's core claims or the Phase D fix queue.

## What was done well

1. **`tooling.md` exemplary structure for Phase D-actionable inventory.** Per-tool table answers Declared / Installed / Runnable / Action; install commands concretely specified; "Notes for the talk" section gives defender's framing.
2. **`deps.md` rigorous on transitive-direct-import distinction.** 4-tier severity classification maps to Phase D fix priority. `datasets`-as-local-subpackage false-positive explicitly disclosed.
3. **45-row triage table genuinely scannable.** Sorted RED-first then YELLOW; one-clause Finding + concrete Fix Plan + Owner. Phase D engineer can sequentially work through it.
4. **"Top 3 Phase D priorities" collapses 9 REDs into 3 actionable buckets.** Bundling row 43 (venv drift) with rows 44–45 (scipy/click promotion) into the same `pyproject.toml` edit is expert-level optimization.
5. **C.10 cleanup of orphan "Combined Posterior so far" snapshots was thorough.** Diff is clean (+164/−36).
6. **Independent re-verification holds.** All 5 faiss imports at exact lines (243, 289, 323, 358, 370). tmtools at line 98. `which marp` returns "not found"; `which npx` returns `/opt/homebrew/bin/npx`. Bit-perfect against repo state.

## Issues

### Important

**(I-1) `deps.md` says "scipy imported in 16 files" — actual count is 24.** Verified: `grep -rEl '^import scipy|^from scipy' src experiments tests` returns 24 matches (13 in src/, 10 in experiments/, 1 in tests/). Triage table row 44 propagates the wrong count. Fix: edit `deps.md:128` and `AUDIT_FINDINGS.md` triage row 44.

**(I-2) C.7 "11 YELLOW" doesn't match 10 YELLOW bullets in C.7 prose.** Cumulative table cell, C.7 distribution line, and disambiguation note all say 11; prose has 10 bullets. Either one C.7 YELLOW lost during C.10 reorg, or "11" is wrong in 3 places. Cumulative 36 YELLOW also off-by-one as a result.

**(I-3) C.8 "6 YELLOW" doesn't match 5 YELLOW bullets in C.8 prose subsection.** `tooling.md` separates jupyter/jupytext as 6 line items; AUDIT_FINDINGS prose collapses to 5 (jupyter+jupytext combined). Triage table also has 5 rows. Internal inconsistency — split or collapse.

**(I-4) Group 3 i1 was REINTERPRETED rather than fixed.** Group 3 i1 asked about 78-vs-101 inside `claims.md`. The C.10 fix added a "78-vs-28" disambiguation in AUDIT_FINDINGS.md instead — both valid, but not the same issue. `claims.md:211` still says "78 (excluding trivial repeats)" without showing which 23 rows were de-duplicated.

### Suggestions

- **(S-1)** Add a 5-line TOC to AUDIT_FINDINGS.md (now 357 lines).
- **(S-2)** Triage table "Finding" column is wide; rows wrap on 80-col terminals.
- **(S-3)** "Counts re-check" section at line 347 is redundant with the cumulative table at line 36.
- **(S-4)** Add "Last verified" date column to triage rows.

## Direct answers to review-prompt concerns

- `tooling.md` Phase D-actionable? **YES.**
- `deps.md` rigorous on transitive distinction? **YES** conceptually; I-1 flags wrong count.
- 45-row triage navigable? **YES.** RED-first, concrete Owner, one-clause Findings.
- Cumulative totals prominent? **YES.** First thing in Posterior; dagger marker on the "9" cell.
- Scales at 357 lines? **Borderline.** TOC (S-1) would help.
- C.10 diff clean? **YES.** No orphan refs in active docs.
- Group 3 i1 / i2 fixes discoverable? **PARTIALLY for i1** (see I-4); **YES for i2** (dagger + footnote prose + reinforcement at line 354).

## Risk assessment for Phase D

The triage table is implementable. The only risk is the YELLOW-count off-by-ones (I-2, I-3) propagating into Phase D's "are we done?" check. **Recommend addressing I-2 and I-3 before D.1 begins.** I-1 and I-4 are post-talk polish.

**Confidence in the audit's core claim ("No headline cited number invalidated"): HIGH.** Bit-perfect verification of Exp 43/44/46/47 cells well-documented in `claims.md`; methodology audit (C.4) GREEN across the board; 9 RED line items all about repo-hygiene / declaration drift, not numeric correctness.
