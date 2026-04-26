# Audit Group 2 (C.3 + C.4 + C.5) — Code Quality Review

**Subagent type:** superpowers:code-reviewer
**Reviewing:** BASE `1ff3a71` → HEAD `1f10f41`

## Summary verdict: APPROVED — strong artifact quality

Three audit commits land cleanly. The artifacts are coherent, internally consistent, and read well as a set. Arithmetic checks out, structure is consistent across files, and the no-fixes rule is honored except for the one explicitly-permitted phylo data restore.

**Distribution:** 19 GREEN findings on artifact quality / 3 minor suggestions / 0 critical issues.

## Per-concern verdict

### 1. `splits.md` and `stats.md` organization — GREEN

Both files follow the same skeleton: Goal → Evidence index → Per-experiment evidence → Cross-cutting checks → Severity tally → Open follow-ups / verdict. A Rost-lab reader who jumps in cold gets: question → evidence → roll-up.

### 2. Per-table conformance schema in stats.md — GREEN with one nit

All 8 rows follow consistent column schema. The disorder row's "Multi-seed" cell says "residue-level predictions averaged across seeds before pooling" while peer rows say "3-seed predictions averaged before bootstrap". Both correct, but the level differs (residue vs per-protein). Could unify by adding "(per-residue)" tag.

### 3. C.5 hygiene.md append — GREEN

The H2 header `## Task C.5 — Phylo file decision (2026-04-26)` is a clear section break. The "Context / Provenance / What this file is / What this file is NOT / Decision / Action taken / Open follow-ups" sub-structure is more rigorous than C.1's looser organization, which is appropriate because C.5 makes a definitive decision.

### 4. C.5 commit message — GREEN

Exemplary: decision stated upfront, rationale enumerated as 5 concrete bullets, Phase D follow-ups recorded, cumulative posterior tally included for context. Diff itself confirms the message.

### 5. AUDIT_FINDINGS.md cumulative table arithmetic — GREEN

Independently verified: C.1 (5/3/2), C.2 (6/3/0), C.3 (6/3/0), C.4 (9/1/0), C.5 (4/2/0). Column sums: 30/12/2 — matches table totals exactly. 68.2%/27.3%/4.5% — text says "~68/27/5" which is fair rounding.

Per-file severity counts (`splits.md` line 137: "[GREEN] 8") differ from rollup (AUDIT_FINDINGS rollup "[GREEN] 6"). Not a bug — rollup consolidates conceptually-distinct findings while file enumerates discrete pieces of evidence — but a reader cross-checking might notice.

### 6. C.5 data-restore diffs minimal — GREEN

Exactly 4 lines change (`n_taxa: 24→156`, `mcmc_time_s: 7.88→89.81`); all other fields byte-identical. Restored content matches `git show 8b1fbf1:data/benchmarks/embedding_phylo_results.json` exactly. Diff is exactly what the commit message describes.

### 7. Skipping `scripts/audit/check_split_leakage.py` — GREEN, fully justified

Plan said "Possibly create" (line 627), making it optional. The implementer's reasoning is documented in `splits.md:16-18`: existing `experiments/validate_split_leakage.py` (MMseqs2-based) was used during Exp 17/24 split validation; not re-run for this audit since reused splits inherit its conclusions. Verification (split-summary `superfamily_overlap=0`, `family_overlap=0`) is conclusive without re-running MMseqs2. Decision aligns with the no-fixes audit philosophy.

## Minor improvement suggestions (none blocking)

1. **stats.md disorder row** — add "(per-residue)" tag to flag the level difference, or use a table footnote.
2. **AUDIT_FINDINGS.md cumulative table** — add one sentence explaining the rollup-vs-detail count difference.
3. **splits.md** could end with a 2-3 sentence overall verdict paragraph parallel to `stats.md:229-238`.

## What was done well

- Headlines lead with the question being answered, then evidence, then verdict.
- File-path-and-line citations everywhere — every claim is independently verifiable.
- Honest YELLOW/RED tagging — Exp 50's random split flagged YELLOW even though it's a known limitation; CB513 internal random split flagged YELLOW as a "lab-Q magnet" pre-emptive mention.
- Cross-references within the audit set — temporal traceability via HEAD-at-audit notes.
- Phylo restore done right — provenance, decision rationale, action, prevention follow-ups all traceable.
- Bootstrap audit cites scipy parameter names directly (`paired=True`, `n_resamples`, `random_state=...`) rather than paraphrasing.
- Both audit reports converge to the same `metrics.statistics` module and state "no shadow implementation" — eliminates a major Rost-lab worry-vector.

## Recommendation

**Ship as-is.** The three minor suggestions above are polish, not blockers. Group 2 audit work is publication-quality.
