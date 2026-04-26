# Audit Group 1 (C.1 + C.2) — Spec Compliance Review

**Subagent type:** general-purpose
**Reviewing:** commits `dd2409d` (C.1) and `1ff3a71` (C.2)

## Audit Group 1 (C.1 + C.2) Review — Spec Compliant

### File presence and locations — VERIFIED
- C.1 created `docs/_audit/hygiene.md` (164 lines) and modified `docs/AUDIT_FINDINGS.md`.
- C.2 created `pytest_baseline.txt` (82 lines), `code_markers.txt` (20 lines), `codec_review.md` (85 lines), and modified `AUDIT_FINDINGS.md` again.

### Commits — VERIFIED
- `dd2409d` — message exactly `audit(hygiene): repo state + package-location drift` (matches plan line 562). Touches only `docs/AUDIT_FINDINGS.md` + `docs/_audit/hygiene.md`.
- `1ff3a71` — message exactly `audit(code): pytest baseline + markers + codec_v2 review` (matches plan line 618). Touches only the four expected files.

### No-fixes rule — VERIFIED
`git diff dd2409d^ 1ff3a71 -- README.md CLAUDE.md` returns empty. `git diff dd2409d^ 1ff3a71 -- src/` returns empty. The implementer correctly inventoried without fixing.

### AUDIT_FINDINGS.md structure — VERIFIED
Prior section (B.1) preserved unchanged across both commits. Posterior section was `(empty)` at the start of C.1 and now has two well-organized subsections (`### Repo hygiene` and `### Code correctness`) keyed to evidence files, with severity tags `[GREEN]`/`[YELLOW]`/`[RED]` and per-section distribution tallies plus a combined tally.

### Findings quality — WELL-EVIDENCED, NOT HAND-WAVY

- `hygiene.md` cites README line numbers precisely (line 12 for "44 experiments", lines 21–34 for stale Quick Start, lines 76–101 for old PQ tables, lines 152–166 for ABTT pipeline). Spot-checked all five — correct.
- `codec_review.md` does not hand-wave the receiver-side check. It explicitly states "VERIFIED: yes, h5py + numpy alone are sufficient for the binary default" with the exact H5 dataset list (`per_residue_bits` + `means` + `scales`), describes the ~12-line numpy decode (`bits*2-1`, then `signs * scales + means`), points to the implementation in `quantization.py:319-346`, and breaks out conclusions per quantization mode (binary/int4/lossless = no codebook; PQ = needs codebook).
- `pytest_baseline.txt` last line is `Exit code: 0`. Final line shows `813 passed, 10 warnings in 89.98s`.
- `code_markers.txt` exists (929 bytes) with substantive content — both src/ scan (zero) and broader scan (1 true TODO in experiments/50_sequence_to_oe.py:439).

### Step completion — VERIFIED
C.1 (6 steps): all completed. C.2 (5 steps): all completed.

### `-x` deviation — REASONABLE
The implementer dropped `-x` from `uv run pytest tests/ -x --tb=short`. The plan's intent for Step 1 is a "pytest baseline" — capturing actual pass/fail state. With `-x`, a single early failure would have stopped the run. The implementer's choice to omit `-x` is what allowed the strong `[GREEN] 813/813 tests pass` finding. Defensible and the right call.

### One minor observation (not a blocker)
The implementer included two RED items in Repo hygiene but one (MEMORY.md treats `one_embedding/` as top-level) explicitly notes "NOT a checked-in file". Could arguably be YELLOW since the repo itself is fine — but the implementer's reasoning (worth flagging for future sessions) is sound and severity is documented transparently.

### Verdict
Spec compliant. Both tasks executed faithfully, no scope creep, no premature fixing, well-evidenced findings.
