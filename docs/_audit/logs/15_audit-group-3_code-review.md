# Audit Group 3 (C.6 + C.7) — Code Quality Review

**Subagent type:** superpowers:code-reviewer
**Reviewing:** BASE `1f10f41` → HEAD `179c2a4`

## Overall Assessment

**APPROVED.** Both artifacts are well-built, evidence-grounded, and ready for a Rost-lab inspector. The audit work is rigorous: bit-perfectness of cited Exp 47 numbers verified (`exp47_sweep_prot_t5_full.json` matches the table verbatim) and the misleading-commit-blame finding confirmed (commit `34e159a` titled "chore: gitignore..." indeed contains the 324-line codec_v2.py rewrite with new defaults).

## What was done well

1. **`params.md` (128 lines, 10.5 KB) — exemplary structure.** One-row-per-default table at the top, consistent column schema (Param | Default | File:line | Evidence | Source experiment | Commit | Status), then an explicit "Hidden / derived defaults" sub-table, then per-default narratives for the YELLOW cases. Easy to scan in <60 seconds. The narrative subsections give a defender's script for each YELLOW that's lab-talk-ready.

2. **`claims.md` (260 lines, 26 KB) — well-grouped, not a wall.** The 13 sub-tables are partitioned by source document and table-of-origin (Headline / Compression tier / Per-Residue / Protein-Level / ESM2 / Ablation / Legacy / Multi-PLM / Codec sweep / Methodology / README-only / Phylogeny / Counts / Hardware). Each sub-table has its own one-line legend pointing at the source JSON. This scales much better than one monolithic table.

3. **Bit-perfect verification holds up to spot-check.** Verified `exp47_sweep_prot_t5_full.json` against CLAUDE.md L184–L189: every cell matches (lossless 100.2/100.0/100.4/100.0, fp16 100.0/99.2/100.6/98.6, int4 99.8/98.8/100.4/98.2, pq224 99.0/98.5/100.6/95.4, pq128 97.5/96.1/100.1/91.4, binary 97.6/95.0/100.4/94.9). The "bit-perfect" framing is accurate, not bluster.

4. **YELLOWs are actionable.** Each YELLOW has a specific surface for the talk: `n_pcs=5` silent truncation captured well enough for Phase D fix; "232 compression methods" → "soften to '200+' or list explicitly"; "L=175" → "fixed assumption from codec design spec; Exp 45 reports actual mean=156"; DCT K=4 → "frame as compression choice, not quality-optimal".

5. **Misleading-commit-blame finding is well-documented.** Verified independently: `34e159a` contains the 324-line codec_v2.py change with new defaults; `8b1fbf1` does not. Framed as "a hygiene issue (commit discipline), not a correctness issue."

6. **AUDIT_FINDINGS.md scales well at 25 KB / 197 lines.** Per-task subsections are bullet-point summaries (not full re-tables); they delegate detail to per-task evidence files.

## Issues

### Important

**(i1) The "78 claims" headline doesn't match the visible row count of ~101.** I counted 101 non-header data rows in `claims.md`. The summary says "Total claims traced: 78 (excluding trivial repeats)". The parenthetical doesn't explain *which* 23 rows were de-duplicated. A Rost-lab reader doing the same row count will be confused. Recommend: either show the de-dup math in a one-line table at top of Summary, or reframe as "Total rows traced: 101; distinct claim categories: 78". AUDIT_FINDINGS.md L165 also says "full register has 78 cells" — wrong word ("cells" implies the 101 rows). Use "distinct claims" consistently.

**(i2) RED-count framing in AUDIT_FINDINGS.md L182–L185 is correct but a Rost-lab inspector will skim and miss it.** Currently the footnote-as-paragraph is correct but discoverable only on close reading. The cumulative table reads "Total RED = 6" without an asterisk. Recommend: add asterisk/footnote indicator on the "6" cell.

### Suggestions

- **(s2)** Process-improvement record: add a one-line note about commit-message hygiene ("when changing user-facing defaults in a sweeping commit, the subject line should reflect the most consequential change").
- **(s3)** `claims.md` could have a "RED-only at-a-glance" sub-table at the top.
- **(s4)** `params.md` could include a Phase D fix path for `dct_k`'s YELLOW (run a clean K∈{2,4,8} sweep).
- **(s5)** `claims.md` Verification column inconsistency (full CIs in some rows, "match" in others) — harmonize or add legend.

## Direct answers to review-prompt concerns

- `params.md` well-organized? **YES.** Hidden-defaults sub-table is a great touch.
- `claims.md` digestible at 26 KB? **YES.** Grouped into 13 navigable sub-tables.
- "78 claims" narrative aligned with table size? **NO** — see (i1).
- YELLOWs actionable? **YES.** Each has pre-empt plan + Phase D fix where applicable.
- `n_pcs=5` finding precise enough? **YES.** File:line, slice operation, behavior all captured.
- Misleading commit blame — process improvement worth recording? **YES** — see (s2).
- AUDIT_FINDINGS.md still readable at 25 KB / 197 lines? **YES.** Will scale to C.10.

## Final disposition

**Group 3 is approved.** The two Important fixes (i1 row-count, i2 RED footnote) are low-effort cosmetic touch-ups that would strengthen the artifacts before the lab talk but don't block proceeding to Group 4.

The work materially advances the audit:
- 11 declared+hidden defaults inventoried with evidence chains
- 101 numeric claim cells / 78 distinct claims traced
- bit-perfect verification of all cited Exp 43/44/46/47 retention numbers
- explicit pre-empt scripts for the most-probable Rost-lab probes
- a process lesson (commit-message hygiene) worth carrying forward
