# Audit Group 3 (C.6 + C.7) — Spec Compliance Review

**Subagent type:** general-purpose
**Reviewing:** commits `67a0fe9` (C.6) and `179c2a4` (C.7)

## Verification Report

### Spec compliance: PASS

**Files & commits — both correct:**
- `docs/_audit/params.md` created in `67a0fe9` (10 KB).
- `docs/_audit/claims.md` created in `179c2a4` (26 KB).
- `docs/AUDIT_FINDINGS.md` modified in both commits.
- Commit messages match plan template verbatim.
- `git diff 67a0fe9^ 179c2a4 -- src/ README.md CLAUDE.md` is empty — no unauthorized fixes (no-fixes rule respected).
- Only `docs/_audit/` and `docs/AUDIT_FINDINGS.md` touched. Posterior section properly extends prior subsections (C.1–C.5 preserved bit-identical).

**Severity tags consistent with prior groups:** `[GREEN]/[YELLOW]/[RED]` used throughout.

**Plan template followed:**
- params.md: table has Param | Default | File:line (added) | Evidence | Source experiment | Commit | Status — extends plan template with File:line for traceability.
- claims.md: table has Claim | Doc:line | Source | Source result | Commit | Status — matches plan template.

### Spot-check results

**Codec defaults (verified against `src/one_embedding/codec_v2.py:84–93`):**
- `d_out=896` ✓ line 86
- `quantization='binary'` ✓ line 87
- `pq_m=None` ✓ line 88 (auto via `auto_pq_m(d_out)` at line 46)
- `abtt_k=0` ✓ line 89
- `dct_k=4` ✓ line 90
- `seed=42` ✓ line 91
- 5 hidden defaults verified: `pq_k=256` (119), `n_pcs=5` (167), `n_sample=50_000` (167), `max_residues=500_000` (159), `version=4` (255).

**Bit-perfect verification of cited numbers:**
- Exp 47 binary-896 prot_t5_full Disorder retention: JSON = 94.93%, CLAUDE.md L189 cites 94.9% ✓
- Exp 47 binary-896 SS3 = 97.64%, cites 97.6% ✓ ; SS8 = 94.99%, cites 95.0% ✓ ; Ret@1 = 100.35%, cites 100.4% ✓
- Exp 46 prot_t5_full Disorder retention = 95.39% [93.28, 96.99] (CI half-width ≈ 1.86), CLAUDE.md L175 cites 95.4±1.9% ✓
- Exp 22 path_geometry K=4 prec@1=0.666, K=8 prec@1=0.712 — exactly matches the dct_k YELLOW evidence in params.md.

**RED #1 ("44 experiments"):** README.md L12 reads `232 compression methods benchmarked across 44 experiments`. CLAUDE.md L4 reads `47 experiments`. RED is justified.

**Claim count:** claims.md has ~99 raw table rows across 11 tables; the "78 claims" reported sums GREEN(50)+YELLOW(24)+RED(4) consistently. Internal arithmetic correct.

### Scope expansion (C.6): Reasonable

The 5 hidden defaults beyond the named 6 are all real, hardcoded, and behaviour-affecting. The `n_pcs=5` finding is genuinely useful — it identifies a silent truncation bug if a user sets `abtt_k>5`. The expansion adds rigor without padding.

### RED framing (C.7): Correctly framed

The 4 C.7 REDs are correctly disclosed as sub-instances of 1 root cause (README drift), already counted as 1 RED in C.1. The combined posterior table explicitly notes: "The 4 REDs are all the same 'README is out of date' item (already a single RED in C.1 — the 4 line items here are distinct **numeric** claims that all share that root cause)." The "true RED count is 2" framing for the cumulative posterior is honest. No double-counting hidden in the totals.

### Minor observation (non-blocking)

The summary at claims.md says "8 GREEN / 11 YELLOW / 4 RED (across 28 most material claims; full register has 78 cells with 50 GREEN / 24 YELLOW / 4 RED)" — the parenthetical clarifies the two-level summary. The framing is internally consistent if one reads each row as "1 claim with multiple supporting cells".

### Final verdict

**SPEC COMPLIANT.** Both tasks completed all required steps, files at correct paths, commits scoped and named per template, AUDIT_FINDINGS.md properly extended. Bit-perfect verification holds for all 4 spot-checked numbers. Scope expansion (5 hidden defaults in C.6) and RED framing (4 sub-instances of 1 root cause in C.7) are both well-justified and explicitly disclosed in the deliverable.
