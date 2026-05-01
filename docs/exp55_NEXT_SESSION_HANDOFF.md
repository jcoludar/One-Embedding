# Exp 55 → next session handoff

The full handoff document lives at `results/HANDOFF_exp55_to_next.md` (gitignored, kept local).
The matching memory entry is `project_session_20260501_exp55.md`.

## One-line summary

Exp 55 shipped: **binary 896d at 37× retains 99.2% of VEP** [98.4, 100.1] and *beats* lossless on ClinVar (0.605 vs 0.602). VEP is the codec's 5th task family.

## What to read in order

1. `docs/exp55_vep_retention.md` — full writeup (TL;DR / rationale / design+data / results / outcomes / links)
2. `results/HANDOFF_exp55_to_next.md` — operational handoff: merge checklist, what's queued, what's earmarked
3. `results/exp55_session_log.md` — bug audit + timing post-mortem (9 fixes during execution + OOM streaming refactor)
4. Memory: `project_session_20260501_exp55.md`, `project_exp56_codec_megasweep_idea.md`, `project_disorder_methodology_check.md`, `project_earmarks_20260501.md`

## What's queued (designed-not-built)

- **Exp 56** — codec mega-sweep on VEP (incl. ABTT, pq64, int2, binary_magnitude, no-RP). Cheapest next experiment; reuses cached embeddings. (`project_exp56_codec_megasweep_idea.md`)
- **Exp 57** — auto-fit centering: empirically validate "200-500 proteins is enough" + ship `OneEmbeddingCodec.auto_fit(plm, n=300)` API. Reuses cached SCOPe-5K embeddings. ~1 day. (`project_exp57_auto_fit_centering.md`)
- **Disorder methodology audit** — re-run with CNN probe + per-residue Pearson + AUC; existing memory note hints disorder gap is probe-induced. (`project_disorder_methodology_check.md`)
- **pLDDT, perplexity, CPP** — three new task families; pLDDT has the strongest cross-check value for the disorder question. (`project_earmarks_20260501.md`)

## What's deferred

- VESM head-to-head VEP comparison
- Multi-PLM VEP
- Reusable plot library + writeup template + publishing skill (started designing in this session, not built)
