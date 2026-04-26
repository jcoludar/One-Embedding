# Audit Group 1 (C.1 + C.2) — Implementer Report

**Subagent type:** general-purpose
**Tasks:** C.1 (repo hygiene + drift inventory), C.2 (code correctness — pytest + markers + codec read)

## Report

**Status:** DONE_WITH_CONCERNS

(The "concerns" are the audit findings themselves — the audit succeeded but found the README drift to be larger than the prior estimated.)

### C.1 — Repo hygiene + drift inventory

**Commit:** `dd2409d`

Biggest findings:
- **README.md is severely out-of-date.** Only the TL;DR headline (line 3) reflects current binary/896d default. Quick Start (lines 21–34), Compression Configurations table (76–101), pipeline diagram (152–166), and ABTT prose all still describe the old PQ/768d/ABTT-on default. RED — major Phase D rewrite.
- **README.md says "44 experiments"** (line 12); CLAUDE.md and MEMORY.md say 47. Off by 3.
- **CLAUDE.md has two coexisting tables:** Exp 44 (768d) at lines 69–76 and Exp 47 (896d) at lines 182–189, with no explicit "(legacy)" tag. YELLOW.
- **`one_embedding/` package** is at `src/one_embedding/`, NOT top-level. CLAUDE.md and README code imports are correct; only the user's `~/.claude/.../MEMORY.md` (lines 91–96, 113, 175) treats it as top-level — that's outside the repo.
- **`.gitignore` is comprehensive,** 67 GB of data correctly excluded, all 7 tracked binary/weight files ≤ 700 KB.

**Distribution:** 5 GREEN / 3 YELLOW / 2 RED.

### C.2 — Code correctness

**Commit:** `1ff3a71`

Biggest findings:
- **813/813 tests pass** in 90 s. Docs claim "798 tests" — actual is higher; reconcile in D.1.
- **Zero TODO/FIXME/HACK/XXX markers in `src/`.** Only one true `[TODO]` exists, in `experiments/50_sequence_to_oe.py:439` (active Stage 3 follow-up, expected).
- **Receiver-side decode claim VERIFIED for binary default.** Decoder is ~12 lines of pure numpy; no codebook required. CLAUDE.md headline holds. PQ correctly needs codebook (matches docs). YELLOW: docs don't ship a self-contained decoder snippet — would help paper readers.
- **Hidden defaults / silent skips in `codec_v2.py`** (none are bugs, but each could surprise a Rost-lab probe):
  - Stale `auto_pq_m` docstring example (still 768/512, not 896).
  - `_preprocess` silently SKIPS centering when `is_fitted == False` and quantization is binary/int4 (no codebook needed). Inconsistent with class docstring's "default: center + RP + binary".
  - `n_pcs=5` hardcoded in `compute_corpus_stats` while `abtt_k` is user-configurable; setting `abtt_k > 5` silently truncates.
  - `encode_h5_to_h5` writes per-protein groups without `quantization`/`d_out` attrs — only `load_batch` correctly merges them.

**Pytest result:** 813 passed, 0 failed, 0 errors, 10 RuntimeWarnings (all expected — testing degenerate-input handling).
**Code marker count:** 0 in `src/`, 1 true TODO in `experiments/`.

**Distribution:** 6 GREEN / 3 YELLOW / 0 RED.

## Final state

```
1ff3a71 audit(code): pytest baseline + markers + codec_v2 review
dd2409d audit(hygiene): repo state + package-location drift
7cc2e72 stub(prior): CALIBRATION.md
52f7fde stub(prior): EXPECTED_QA.md
93825bc stub(prior): HANDOFF.md
```

`git status --short`: empty.

## Combined Posterior so far

**11 GREEN / 6 YELLOW / 2 RED.** Closely matches the prior prediction (~70/20/10). The 2 REDs (README drift, MEMORY.md package-location confusion) are exactly the items the prior anticipated — both queued for Phase D.1 fix.

## Genuinely surprising

- **Code is cleaner than expected.** Zero true markers in `src/`. 813 tests vs documented 798 means coverage has *grown* since the last doc update.
- **codec_v2.py is well-factored.** The static `load`/`load_batch`/`_read_per_residue_from_h5` design genuinely makes the receiver-side decode work without instantiating the codec.
- **README is more out-of-date than expected.** The headline was rewritten for the binary/896d era but the body wasn't updated to match. This is the single biggest fix item heading into Phase D.
