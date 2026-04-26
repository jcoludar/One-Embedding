# Audit Group 4 (C.8 + C.9 + C.10) — Spec Compliance Review

**Subagent type:** general-purpose
**Reviewing:** commits `4760a6f` (C.8), `060191f` (C.9), `db3a0fc` (C.10)

## Verification Report — Spec compliant

All commit messages match plan templates exactly:
- C.8: `audit(tooling): dev-tool inventory + gap list` ✓
- C.9: `audit(deps): pyproject + uv.lock + import/declare parity` ✓
- C.10: `audit(roll-up): final triage table` ✓

### C.8 — Tooling inventory (commit `4760a6f`)
- `docs/_audit/tooling.md` exists (109 lines) using plan's required template (Tool / Declared / Installed / Runnable / Action — plan line 846-855) ✓
- `docs/AUDIT_FINDINGS.md` updated with `### Tooling (Task C.8, ...)` section (1G/6Y/1R distribution) ✓
- Commit modifies only the two expected files ✓
- **Independently verified missing tools:** `uv run ruff --version` and `uv run mypy --version` both return "No such file or directory"; `which marp` returns "marp not found"; `uv run pytest --version` returns 9.0.2 ✓
- Independently verified `pyproject.toml` declares only `pytest>=9.0.2` in `[dependency-groups].dev` ✓
- All 4 plan steps complete ✓

### C.9 — Dependency audit (commit `060191f`)
- `docs/_audit/deps.md` exists (213 lines) using plan's required template sections — uv.lock sync status, Phantom deps, Undeclared deps, Yanked / EOL packages, Python version constraint (plan lines 906-921) ✓
- `docs/AUDIT_FINDINGS.md` updated with `### Dependencies (Task C.9, ...)` section (6G/3Y/2R distribution) ✓
- Commit modifies only `docs/AUDIT_FINDINGS.md` +18 and `docs/_audit/deps.md` +213 ✓
- **Independently verified `uv lock --check` returns "Resolved 97 packages in 7ms" with exit 0 ✓**
- **Independently verified `faiss` claim:** 5 import sites in `src/one_embedding/structural_similarity.py` at lines 243, 289, 323, 358, 370 — exact match to claim ✓
- **Independently verified `tmtools` import** at `src/evaluation/structural_validation.py:98` ✓
- Confirmed neither `faiss-cpu` nor `tmtools` declared in `pyproject.toml` ✓
- All 6 plan steps complete ✓

### C.10 — Final roll-up (commit `db3a0fc`)
- 45-row triage table present at `docs/AUDIT_FINDINGS.md:253-299` with required columns `# | Track | Finding | Color | Fix plan | Owner` (plan line 953-958) ✓
- Counts: 9 R + 36 Y = 45 explicit rows in the triage table; GREENs consolidated as footnote per spec ✓
- **Per-track tally sums correctly: G=49, Y=36, R=9 = 94 line items, exactly matching the cumulative table at lines 36-47** ✓
- Final cumulative totals table placed at top of Posterior section per Step 2 ✓
- Commit modifies only `docs/AUDIT_FINDINGS.md` (+164/-36) ✓
- Per-track Posterior subsections preserved alongside the new triage table — matches controller-side guidance ✓
- All 3 plan steps complete ✓

### Group 3 review nits (i1, i2) — actually addressed
- **i1 (78-claims vs 28-material discrepancy):** Section "Claims-register vs cumulative-table — note on counts" at `docs/AUDIT_FINDINGS.md:69-77` explicitly disambiguates ✓
- **i2 (true root-cause RED framing):** Footnote at lines 49-56 (`† The "9 RED" cell counts distinct line-item findings... true root-cause RED count is 6`) plus reinforcement at lines 349 and 354 ✓

### No-fixes rule — verified
- `git diff 4760a6f^ db3a0fc -- pyproject.toml` returns empty ✓
- `git diff 4760a6f^ db3a0fc -- src/ experiments/` returns empty ✓
- `git diff --stat` across all 3 commits shows ONLY `docs/AUDIT_FINDINGS.md`, `docs/_audit/deps.md`, `docs/_audit/tooling.md` modified ✓

### No issues found

The implementer's claims are accurate and the implementation matches the spec across all three tasks. The 3 RED findings (1R from C.8, 2R from C.9) are genuine audit findings about repo state, not implementation defects in the audit itself.
