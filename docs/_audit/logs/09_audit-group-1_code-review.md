# Audit Group 1 (C.1 + C.2) — Code Quality Review

**Subagent type:** superpowers:code-reviewer
**Reviewing:** BASE `7cc2e72` → HEAD `1ff3a71`

## Overall Verdict

**Approve.** Audit artifacts are well-built, internally consistent, atomically committed, and clearly honor the no-fixes discipline. Three minor "Important" nits, four "Suggestions". Nothing blocks moving on to Audit Group 2.

## What was done well

1. **Atomicity is genuine.** Each commit touches exactly the files its task scope mandated. C.1 = `hygiene.md` + AUDIT_FINDINGS.md delta; C.2 = `pytest_baseline.txt` + `code_markers.txt` + `codec_review.md` + AUDIT_FINDINGS.md delta. No drift across boundaries; no source code touched. Commit messages verbatim from plan.
2. **Self-contained artifact files.** Each file carries front-matter (date, HEAD SHA at audit, scope) so a future reader doesn't need the plan to orient.
3. **Severity tagging is consistent.** `[GREEN]/[YELLOW]/[RED]` used uniformly with explicit per-section distribution tallies.
4. **Findings are well-evidenced with cited line numbers.** Spot-checked codec_v2.py line citations (113, 141, 147, 152, 167, 273, 479–483) all resolve to the claimed code.
5. **No-fixes discipline plainly visible from the diffs.** Every diff line is in `docs/`. Source files were read but never edited.
6. **Receiver-side decode verification is the strongest piece.** "Binary needs no codebook" headline is treated with appropriate skepticism; conclusion is precise ("h5py + numpy alone are sufficient for binary, but the receiver also needs to know the bit layout — ship a 15-line snippet in D.1").
7. **Pytest baseline captured raw with full warning context.** Warnings preserved verbatim and correctly classified as informational.

## Issues

### Important (should fix before final submission, but don't block this group)

**I-1. Defaults inventory table uses "TBD" inconsistently.** The implemented table uses literal `TBD (Exp 47 sweep)` for `d_out`, `TBD (Exp 47 + Exp 45 disorder finding)` for `quantization`, etc. Reading these as a future maintainer is confusing — `TBD` typically means "to be determined", but here it appears to mean "justification not in code, but external evidence exists". Suggest: drop "TBD" and just cite the experiment, OR replace with `External (Exp NN)`.

**I-2. Combined posterior tally is mechanically wrong.** The C.2 section actually contains 6 explicit GREEN bullets but the last one ("No RED findings") is a meta-finding, not a finding. Counting it as one of the six is double-counting. Real C.2 is 5G/3Y/0R, combined would be 10/6/2 (not 11/6/2).

**I-3. One stale codec_v2 line citation.** `codec_review.md:34` says line 269; actual line is 249. Observation correct, line number off by 20. Fix during D.1.

### Suggestions (nice to have)

- **S-1.** `codec_review.md`'s receiver-side check could end with the actual ~12-line numpy snippet (currently referenced but not pasted).
- **S-2.** `hygiene.md` "Summary table" undercounts (4 GREEN vs 5 in section).
- **S-3.** `hygiene.md`'s top-20 table shows raw `du -k` output unsorted/unlabeled.
- **S-4.** AUDIT_FINDINGS.md "outdated, undercount" claim could cite `pytest_baseline.txt:6` directly.

### Markdown / lint
No broken tables. All have consistent column counts. All code fences closed. List indentation consistent.

## Recommendation

**Proceed to Audit Group 2.** The three "Important" issues are quality-of-life nits that won't compromise downstream tracks. They can be folded into a single docs touch-up at the end of Audit Phase C, or rolled into D.1 when CLAUDE.md/README are rewritten.
