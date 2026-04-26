# Lab Talk Prep — Audit, Write-Up, Slides, Calibration

**Date:** 2026-04-26
**Status:** Design approved verbally; pending user review of this written spec before plan-writing.
**Budget:** ~3 days of work (no per-day allocation; steps end on output, not the clock).
**Owner:** Ivan
**Audience for the talk:** Full Rost lab (incl. Burkhard Rost, Michael Heinzinger). Field-aware, will probe deeply on benchmarking rigor, error bars, architecture, parameter intentionality.
**Slot:** 15 min talk + 10 min Q&A.

## Motivation

The repo (`ProteEmbedExplorations`) holds 47 experiments, 232 compression methods, 5-PLM validation, and a shipped `OneEmbeddingCodec`. The code, results, and write-ups have grown faster than they have been audited. A lab talk for an audience that includes the people who built ProtT5/ProstT5 is a forcing function to:

1. verify every cited number against its raw output (no claims survive uninspected),
2. defend every default parameter with evidence,
3. clean up doc/code drift (`one_embedding/` package, modified phylo file, in-flight worktree),
4. produce three layered written artefacts (write-up, manuscript skeleton, handoff) that share a single source of truth,
5. anticipate the sharpest probes the audience will plausibly ask, and
6. measure how well our prior beliefs about the project match what the audit actually finds — explicit calibration.

This spec describes the prep workflow.

## Goals

**Primary.** Walk into the talk able to defend every cited number with raw output, every default parameter with evidence, every architectural choice with the alternative we considered.

**Secondary.** Produce three derived written artefacts (state-of-project write-up, manuscript skeleton, handoff doc) and a public README that all reflect the same vetted reality. Plus an FAQ doc for anticipated probes and a calibration retrospective.

**Not a goal (this spec).** New experiments, new benchmarks, new code beyond what audit fixes require. The only new artefacts are the docs themselves and the slide deck.

## Audience constraints (driving every decision)

- **Heinzinger built the embeddings we compress.** Disorder retention 94.9 % and the ABTT-removal default (Exp 45) will be probed. Multi-PLM table will be probed (ANKH worst at 94.8 % disorder retention).
- **Rost famously asks "what's the baseline?" and "what's the significance?"** Every figure must show its baseline; every claim must show its CI.
- **Q&A is 10 min.** The lab will prod orthogonally to the slides — anticipated-Q&A doc is non-negotiable.
- **15 min talk.** No slack for filler. Allocation is fixed: 10 min on the codec (humble framing), 2.5 min on Exp 50 (sequence → OE), 2.5 min on negative results + roadmap.

## Deliverables

All committed under `docs/` (or repo root for README). Slides under `slides/`.

| # | Path | Type | Purpose |
|---|------|------|---------|
| 1 | `docs/AUDIT_FINDINGS.md` | internal | Green / yellow / red findings + fix plan |
| 2 | `docs/STATE_OF_THE_PROJECT.md` | future-self | Brutally honest write-up; the source of truth |
| 3 | `docs/MANUSCRIPT_SKELETON.md` | manuscript | Section-by-section paper outline derived from #2 |
| 4 | `README.md` | public | Rewrite to match current code reality |
| 5 | `docs/HANDOFF.md` | onboarding | A labmate can run the codec and reproduce a benchmark in 15 min |
| 6 | `docs/EXPECTED_QA.md` | talk prep | 10–15 sharpest anticipated probes + crisp answers |
| 7 | `docs/CALIBRATION.md` | meta | Per-doc: prior vs. posterior — where we were right, where wrong, why |
| 8 | `slides/lab-talk/` (date pinned at Step 10) | talk | Marp markdown → PDF |
| 9 | (done) `~/CascadeProjects/students/embedding_information_loss/README.md` | student stub | Written; outside this repo, not under VCS yet (decide at hand-off) |

## Calibration loop (the meta-deliverable)

The point of the calibration loop is to make our beliefs falsifiable *before* we measure them, and to record where we were wrong.

1. **Stubs as priors.** Before the audit, write all 7 doc stubs in one pass. Each opens with what we currently believe the finished doc will say (key claims, expected red findings, predicted reds-per-area, expected hardest questions, etc.). Commit each individually with `stub(prior): <doc>`.
2. **Real work overwrites the stubs.** As the audit and writing happen, each doc grows past its stub. Commit deltas with normal messages.
3. **Final calibration.** Before the second dry-run, write `CALIBRATION.md`: for each doc, one paragraph on what the prior got right, what it got wrong, what changed materially, why. The git history (`stub(prior): X` → final) is the audit trail.

This is intellectual honesty as a *deliverable* — the kind of artefact a Rost-style audience respects.

## Workflow (steps in dependency order, no day boundaries)

### Step 0 — Commit current state (don't curate yet)

Capture present reality on disk in git. **Do not** decide what is main-quality vs. future-direction at this step — that is a triage decision after the audit (Step 3). Includes:

- The modified `data/benchmarks/embedding_phylo_results.json` (currently shows a 24-taxa re-run replacing the 156-taxa canonical result — commit as-is on `main`; flag in audit).
- Any uncommitted work in `.worktrees/exp50-rigorous` (branch `exp50/rigorous-cath-split`, head `b970338`) — committed on its branch, branch left alone.
- Any other uncommitted/untracked work surfaced by `git status`.

### Step 1 — Stubs as priors

Write 7 doc stubs in one pass (deliverables #1–#7), each opening with a "Prior (written before audit)" section. Commit individually with messages of the form `stub(prior): <doc>`.

### Step 2 — Audit

Working sweep, output landing in `AUDIT_FINDINGS.md` with green / yellow / red items. Five tracks:

1. **Repo hygiene.** `git status` clean, `.gitignore` correct, no large binaries tracked, doc/code drift inventoried (e.g. `one_embedding/` package referenced in CLAUDE.md but not at repo root — confirm where it actually lives, fix references).
2. **Code correctness.** `src/one_embedding/codec_v2.py` re-read end-to-end. Tests run (`pytest tests/`); record pass/fail. Search for `TODO|FIXME|HACK|XXX`. Confirm receiver-side decode path needs nothing beyond `h5py + numpy`.
3. **Benchmark correctness.** For every cited result:
   - **Splits.** CATH/SCOPe — confirm no homology leakage between train and test (especially Exp 46 multi-PLM table).
   - **Probes.** CV-tuned (`GridSearchCV`), `random_state=42`, multi-seed predictions averaged *before* bootstrapping (Bouthillier et al. 2021).
   - **Bootstrap.** BCa with B=10000 (DiCiccio & Efron 1996); percentile fallback for n<25.
   - **Disorder.** Pooled residue-level Spearman ρ (SETH/CAID convention), cluster bootstrap (resample proteins, recompute pooled stat).
   - **Retention.** Paired bootstrap CI (Exp 43 Phase B's correction).
   - **Baselines.** Same DCT pooling for raw and compressed (Exp 43 fairness fix).
4. **Parameter intentionality.** Every default in `OneEmbeddingCodec` defended with evidence + a one-line citation (experiment ID + result). Defaults to defend: `d_out=896`, `quantization='binary'`, `pq_m='auto'` (rule), `abtt_k=0`, DCT K=4, RP seed.
5. **Claims register.** Every numeric claim in `CLAUDE.md` and `README.md` traced to (a) the experiment script that produced it, (b) the commit hash of the run, (c) the raw output file. Untraced claims = red.

### Step 3 — Triage findings + curate branches

Apply risk policy per red finding:

- **(i) fix in <4 h** → schedule fix, continue.
- **(ii) fix in 4–12 h** → demote the affected number on slides + flag in talk: "currently being re-validated, here's what we know."
- **(iii) unfixable in time** → cut the result, name the omission honestly.

Same step: decide what stays on `main` vs. moves to `future-directions/` branch or folder. Anything in flight, not validated, or not part of the talk story moves out of the main narrative without being deleted.

### Step 4 — `STATE_OF_THE_PROJECT.md`

Real version. Audit findings folded in. Section per workstream: codec, multi-PLM validation, sequence → OE (Exp 50), negative results, open problems, next directions. Honest about every limit.

### Step 5 — `MANUSCRIPT_SKELETON.md`

Section-by-section paper outline derived from Step 4: title candidates, abstract draft, intro arc, methods (codec + benchmarks + statistics), results (the four big tables — single-PLM rigorous, codec sweep, multi-PLM, Exp 50 ceiling), discussion (disorder gap, ABTT artefact, sequence → OE implications), limitations, figure list with one-line captions.

### Step 6 — `README.md` rewrite

Public-facing. Match current code reality (no claims about a top-level `one_embedding/` package if it doesn't exist there). Quick start, headline numbers (with CIs), pointers to the manuscript skeleton and the state-of-project doc.

### Step 7 — Slide figures

Build only what is missing from existing `docs/figures/`. Likely additions:
- Pareto plot (compression × retention, marking the binary-default and PQ224 points).
- 5-PLM × 4-task heat-map.
- Exp 50 learning-curve / ceiling plot.

### Step 8 — `HANDOFF.md`

A labmate runs the codec end-to-end in 15 min: setup, where the data live, how to extract a new PLM, how to encode / decode / load batch, how to reproduce a single benchmark.

### Step 9 — `EXPECTED_QA.md`

10–15 sharpest probes + crisp answers. Predicted candidates (will refine during audit):
- Why ABTT removed by default? (Exp 45 evidence.)
- Disorder retention 94.9 % — is this real? Compare to baseline noise floor.
- Why binary as default rather than PQ? (No codebook, 1500 prot/s, on par with PQ M=128 on disorder.)
- Why DCT K=4? Why not learned pool?
- ANKH disorder retention worst at 94.8 % — what's special about ANKH?
- Exp 50 plateau at 69 % bit accuracy — capacity, data, or architecture?
- Cross-PLM split fairness — same train/test partitions across all 5 PLMs?
- What's the comparison vs. just storing FASTA + a small predictor?
- Why not co-distilled VESM as a baseline?
- How does retrieval stay at 100 % when per-residue tasks lose 5–8 %?

### Step 10 — Slide deck

Marp markdown → PDF. Allocation:
- Slides 1–2 (1 min) — title, motivation, problem framing.
- Slides 3–10 (8 min) — codec story, A-thesis, with humility. Methods, headline tables, multi-PLM grid, Pareto.
- Slides 11–12 (2.5 min) — Exp 50 (sequence → binary OE), CATH-split rigorous numbers, the ceiling.
- Slides 13–14 (2.5 min) — negative results (VQ failures, CNN ceiling, disorder gap), roadmap (transformer, multi-task, multi-teacher).
- Slide 15 (1 min) — summary + asks.

### Step 11 — Two timed dry runs

First solo (record audio if useful). Second with a watcher and the clock. Capture issues, apply fixes. If a third run needs to happen, schedule it.

### Step 12 — `CALIBRATION.md`

Written between the first and second dry runs. Per doc (1–7): one paragraph. Where the prior was right; where wrong; what changed materially; why our prior was wrong (overconfidence, blind spot, missing data). Closes the loop.

## Risk policy

No silent demotions. Any cited number that fails audit gets:
- fixed and re-validated, or
- demoted on the slides with an explicit verbal flag, or
- cut from the deck and named honestly as omitted.

If Step 2 surfaces something that invalidates the talk's spine (e.g. a leakage bug in the multi-PLM split), pause, surface it to the user, replan.

## Tooling decisions

- **Slide format.** Marp markdown → PDF. Source-controlled, fast to iterate, no proprietary lock-in. (Override to Keynote/Beamer if the lab has a template requirement.)
- **Stat conventions.** Already established (Exp 43 + Exp 46): BCa B=10000, paired bootstrap on retention, pooled disorder ρ + cluster bootstrap, CV-tuned probes, multi-seed averaged before bootstrap, `random_state=42`. Audit confirms compliance; no changes.
- **Repo conventions.** `docs/` for all written artefacts; `slides/` for the deck; `future-directions/` (branch or folder, decided in Step 3) for anything not in the talk's spine.

## Cadence

Steps end when their committed output reflects their actual work, not when a clock ticks. Some steps (5/6/8) are derivative and can parallelise; default sequential — branching is a deliberate choice, not a default.

## Success criteria (concrete)

- All seven docs (#1–#7) committed and reflect reality.
- Slide deck exports to PDF without error and runs in ≤15 min (verified by dry runs).
- Every number on a slide has a slide-private "source: <experiment + commit + file>" comment in the Marp source.
- `git status` clean on `main` at the time of the talk.
- `CALIBRATION.md` written; every prior commit has a posterior counterpart in the diff.
- `pytest tests/` baseline recorded in `AUDIT_FINDINGS.md` (Step 2). Final state: every test that passed at audit time still passes at talk time. Pre-existing failures are documented, not silently inherited.
