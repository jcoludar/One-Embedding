---
date: 2026-06-04
started_at: 2026-06-04T13:17:50+02:00
slug: plm-choice-lrz-jobcheck
status: paused
followups:
  - "pLM-choice LRZ jobs (states checked 2026-06-04 via sacct): 5665232 rinit_prot_t5 COMPLETED (2026-06-03 14:41, 1:19); 5665236 orph_bench COMPLETED (14:32, 1:29); 5665231 rinit_esm3 still PENDING(Priority) — never ran (A100 congestion, same as the OE sweep). Only job STATES verified, not OUTPUTS."
  - "TODO (pLM-choice, unverified): confirm rinit_prot_t5 produced 319/319 (was smoke-only before — 1:19 elapsed is suspiciously short); re-pull the full 15-pLM orphan_benchmark.json from 5665236; get rinit_esm3 to actually run (resubmit or wait for A100 backfill) for the 319 esm3 embeddings."
  - "Resume from the referenced paused logs: 2026-06-03-plm-choice-lrz-checkin.md (the 3 jobs) and 2026-06-03-oe-autoeval-lrz-design-plan-impl.md (OE sweep). OE side was handled today in 2026-06-04-oe-autoeval-grab-reports.md."
ended_at: 2026-06-04T13:22:57+02:00
---

# pLM-Choice — LRZ job status check (resume)

Cross-project session. Work lands in `SpeciesEmbedding/projects/plm_choice/` and
on LRZ (`ge94xik2@login-02`); ProteEmbedExplorations repo not expected to change
(logged here because the session-paperwork harness lives here).

## Focus

Ivan asked: "How are we doing for pLM choices? We were running a few jobs on LRZ,
how are they faring?" Resume from two paused sessions:

1. `2026-06-03-plm-choice-lrz-checkin.md` (paused) — 3 jobs to verify:
   - `5665231 rinit_esm3` (A100-80) → expect 319/319, no CUDA OOM
   - `5665232 rinit_prot_t5` (V100) → expect 319/319 (was smoke-only before)
   - `5665236 orph_bench` (CPU) → re-pull full 15-pLM `orphan_benchmark.json`
2. `2026-06-03-oe-autoeval-lrz-design-plan-impl.md` (paused) — OE×autoeval sweep
   jobs `5665512-5665525` (6 PLMs × {raw,oe} + 650M/ProtT5 oe_norp).

## What we did

Checked the 3 pLM-choice job **states** (sacct, 2026-06-04) — this session's focus overlapped
with the OE-sweep work, which took the foreground; pLM-choice outputs were not verified.

| job | name | state | note |
|---|---|---|---|
| 5665232 | rinit_prot_t5 | COMPLETED | 2026-06-03 14:41, 1:19 elapsed (verify 319/319 — short runtime suggests possible smoke-only) |
| 5665236 | orph_bench | COMPLETED | 2026-06-03 14:32, 1:29 elapsed (re-pull full 15-pLM orphan_benchmark.json) |
| 5665231 | rinit_esm3 | PENDING | never ran — A100 queue congestion (same as the OE A100 arms) |

**Closed as paused** (not done): rinit_esm3 hasn't run, and the two COMPLETED jobs' outputs are
unverified. See `followups:` for the concrete next steps. The OE×autoeval side of "how are the
LRZ jobs faring" is fully handled in `2026-06-04-oe-autoeval-grab-reports.md`.
