# Audit Group 4 (C.8 + C.9 + C.10) — Implementer Report

**Subagent type:** general-purpose
**Tasks:** C.8 (tooling inventory), C.9 (dependency + uv.lock audit), C.10 (final triage roll-up)

## Report

**Status:** DONE_WITH_CONCERNS

Three commits landed; concerns are all substantive findings (not procedural failures): two new REDs surfaced in C.9 (faiss-cpu + tmtools undeclared) and one in C.8 (marp-cli missing). All are documented and pushed to Phase D fix queue.

### C.8 — Tooling inventory
- **Commit:** `4760a6f` (`audit(tooling): dev-tool inventory + gap list`)
- Only `pytest` is declared in `pyproject.toml` (`[dependency-groups].dev`).
- `ruff`, `mypy`, `pytest-cov`, `pre-commit`, `jupyter`, `jupytext` all missing — pure repo-hygiene gaps; D.4 install.
- `marp-cli` (Node) absent AND blocks Phase H.1 slide deck — `npx` IS available, so install path is `npm install -g @marp-team/marp-cli`.
- Evidence file: `docs/_audit/tooling.md`.
- **Distribution: 1 GREEN / 6 YELLOW / 1 RED.**

### C.9 — Dependency + uv.lock audit
- **Commit:** `060191f` (`audit(deps): pyproject + uv.lock + import/declare parity`)
- `uv lock --check` exits 0 (97 packages in sync); no yanked packages; `requires-python = ">=3.12"` consistent across pyproject/lock/venv.
- `uv sync --dry-run` would uninstall 16 venv-only packages (drift from earlier ad-hoc `uv pip install`).
- **Two undeclared imports in `src/`** (REDs): `faiss-cpu` (5 sites in `structural_similarity.py`) and `tmtools` (TM-score in `structural_validation.py`). Both would break a fresh clone + `uv sync` install of those code paths. Headline codec / Exp 43–47 numbers unaffected.
- `scipy` (16 files) and `click` (2 files) imported directly but only available as transitives — fragile, promote in D.5.
- False positive: `from datasets.netsurfp import ...` resolves to a LOCAL subpackage, not the HuggingFace `datasets` library.
- `deptry` not installed; manual import-grep used as the plan-specified fallback (deptry install deferred to D.5).
- Evidence file: `docs/_audit/deps.md`.
- **Distribution: 6 GREEN / 3 YELLOW / 2 RED.**

### C.10 — Final roll-up triage table
- **Commit:** `db3a0fc` (`audit(roll-up): final triage table`)
- Built single-row-per-distinct-finding executive table after the per-track Posterior subsections; 45 rows total (all 9 REDs + all 36 YELLOWs from C.1–C.9). Per-track GREENs consolidated in a footnote.
- Final cumulative totals placed at TOP of Posterior section.
- Removed the now-superseded "Combined Posterior so far (C.1+C.2)" and "(C.1–C.7)" intermediate snapshots — replaced by the single C.1–C.9 totals table at top.
- Group 3 review nits addressed: (i1) added a "Claims-register vs cumulative-table" subsection explaining the 78-vs-28 count discrepancy is materiality classification, not double-counting; (i2) added a footnote on the "9 RED" cell making the "true root-cause is 6" framing visible.
- **Final cumulative posterior (C.1–C.9): 49 GREEN / 36 YELLOW / 9 RED** (line items). True root-cause RED count = **6**.

### Final repo state

```
db3a0fc audit(roll-up): final triage table
060191f audit(deps): pyproject + uv.lock + import/declare parity
4760a6f audit(tooling): dev-tool inventory + gap list
83d21ef audit(logs): subagent reports for audit groups 2-3 (C.3-C.7)
179c2a4 audit(claims): traceability register for CLAUDE/README numbers
```
`git status --short`: empty.

### Top 3 Phase D fix priorities (from triage table)

1. **D.1: README.md rewrite** — covers triage rows 1, 3, 4, 5, 6 (one root cause, 5 line items). Highest-visibility correctness gap; old PQ M=192 / 768d / "44 experiments" / ABTT k=3 narrative all need to swap to current binary 896d / 47 experiments / ABTT off. Estimated <4 h.
2. **D.4: install `marp-cli`** — covers row 7. Only RED gating a downstream task on the talk-prep critical path (Phase H.1 slide deck). `npm install -g @marp-team/marp-cli`. <30 min.
3. **D.5: declare `faiss-cpu` and `tmtools`** — covers rows 8 and 9. Add a `[project.optional-dependencies].structural` block. Bundle the venv-drift cleanup (row 43) and the scipy / click promotion (rows 44–45) into the same pyproject edit. <30 min.

### Total finding count in triage table

**45 rows** (9 RED + 36 YELLOW). Plus 49 GREEN findings consolidated in the "GREENs (consolidated)" footnote section.

### Headline takeaway

No Exp 43/44/46/47 retention or CI claim is invalidated by the C.8/C.9 work either. Both new REDs (faiss-cpu, tmtools) are missing manifest entries for OPTIONAL code paths (FAISS retrieval index, TM-score) — neither is on the headline codec encode/decode path, which only needs `numpy + h5py` for the binary default receiver. The audit's narrative is intact: methodology rigorous, headline numbers bit-perfect against source JSONs, drift concentrated in (a) README narrative, (b) marketing-layer count claims, (c) tooling/dep manifests that were never set up to professional-repo standard.
