# Lab Talk Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the OneEmbedding repo, defend every cited number, and ship 7 docs + a Marp slide deck for a 25-min Rost-lab seminar — with a calibration loop (priors written before the audit, posteriors after) that makes our intellectual honesty a deliverable.

**Architecture:** Each doc is committed first as a "prior stub" (`stub(prior): X`) capturing what we currently believe it will say, then overwritten with the real version after audit + writing. Git history *is* the calibration trail. `CALIBRATION.md` summarises the deltas. Audit outputs live under `docs/_audit/` (raw evidence) and roll up into `docs/AUDIT_FINDINGS.md` (green/yellow/red).

**Tech Stack:** Markdown for all docs, Marp for slides (markdown → PDF), pytest for the test baseline, bash + Python for audit scripts, git for the prior/posterior history.

**Spec reference:** `docs/superpowers/specs/2026-04-26-presentation-prep-design.md`

---

## File Structure

### New files (deliverables)
| Path | Type | Created at | Final at |
|------|------|------------|----------|
| `docs/AUDIT_FINDINGS.md` | audit report | Task B.1 (stub) | Task C.10 (filled) |
| `docs/STATE_OF_THE_PROJECT.md` | write-up | Task B.2 (stub) | Task E.1 |
| `docs/MANUSCRIPT_SKELETON.md` | paper outline | Task B.3 (stub) | Task E.2 |
| `docs/_priors/README_REWRITE_PRIOR.md` | README rewrite prior | Task B.4 | (immutable after) |
| `docs/HANDOFF.md` | onboarding | Task B.5 (stub) | Task G.1 |
| `docs/EXPECTED_QA.md` | anticipated probes | Task B.6 (stub) | Task G.2 |
| `docs/CALIBRATION.md` | prior↔posterior summary | Task B.7 (stub) | Task I.3 |
| `docs/_audit/` | raw audit evidence | Tasks C.1–C.5 | (append-only) |
| `slides/lab-talk/talk.md` | Marp source | Task H.1 | Task H.1 |
| `slides/lab-talk/figures/` | talk-specific figures | Tasks F.1–F.4 | (append-only) |

### Modified files
| Path | Modified at | Reason |
|------|------------|--------|
| `data/benchmarks/embedding_phylo_results.json` | Task A.1 | Commit current modified state |
| `README.md` | Task E.3 | Final rewrite (real, not stub) |
| `CLAUDE.md`, `MEMORY.md` | Tasks D.* | Only if claims register flags untraceable numbers |

---

## Phase A — Step 0: Commit current state

### Task A.1: Commit the modified phylogeny result file

**Files:**
- Modify: `data/benchmarks/embedding_phylo_results.json` (already modified on disk)

**Context:** `git diff` shows `n_taxa: 156 → 24`, `mcmc_time_s: 89.81 → 7.88`. Looks like a downscaled rerun overwrote the canonical 156-taxa result. We commit it as-is and flag it in audit (Task C.5). No fix attempted at this step.

- [ ] **Step 1: Re-confirm the diff**

Run:
```bash
git diff data/benchmarks/embedding_phylo_results.json
```
Expected: shows `n_taxa` 156→24 and `mcmc_time_s` 89.81→7.88 only.

- [ ] **Step 2: Commit with audit-flag note**

```bash
git add data/benchmarks/embedding_phylo_results.json
git commit -m "$(cat <<'EOF'
chore: commit in-flight phylo result downsize (24 taxa) — audit flag

The canonical 156-taxa MCMC result was overwritten by a 24-taxa
downscaled rerun. Committing as-is to capture present reality.
Triage in docs/AUDIT_FINDINGS.md (Task C.5):
  - is the original 156-taxa result recoverable from git history?
  - was the downsize intentional?
  - what should the canonical file be at talk time?

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
git status --short
```
Expected: empty (or only untracked). The previously-modified file is now committed.

---

### Task A.2: Inspect and commit any work in the exp50 worktree

**Files:**
- Inspect: `.worktrees/exp50-rigorous/` (separate working tree, branch `exp50/rigorous-cath-split`)

**Context:** `git worktree list` shows `b970338 [exp50/rigorous-cath-split]` in `.worktrees/exp50-rigorous/`. We commit pending work on its own branch but do **not** merge to `main`.

- [ ] **Step 1: Check uncommitted state in the worktree**

```bash
cd .worktrees/exp50-rigorous
git status --short
git log --oneline -5
cd -
```

- [ ] **Step 2: If uncommitted, commit on the worktree's branch**

If `git status --short` from Step 1 was empty, skip to Step 3. Otherwise, review each file:
```bash
cd .worktrees/exp50-rigorous
git diff
git diff --cached
# Decide per file. Stage only files you've reviewed:
git add <specific-file-1> <specific-file-2> ...
git commit -m "wip(exp50): capture in-flight rigorous CATH-split work"
cd -
```

Do not use `git add -p` (interactive — would hang in unattended runs) or `git add .` (sweeps in unintended files).

- [ ] **Step 3: Note the head**

Append to `docs/_audit/worktrees.md` (create if missing):
```
# Worktree state at audit start

`.worktrees/exp50-rigorous/`
- branch: exp50/rigorous-cath-split
- head: <hash from `git -C .worktrees/exp50-rigorous rev-parse HEAD`>
- status: not merged to main; revisit at Step D.3 (curate branches).
```

- [ ] **Step 4: Commit the audit note**

```bash
git add docs/_audit/worktrees.md
git commit -m "audit(worktrees): record exp50 worktree head"
```

---

### Task A.3: Verify nothing else is uncommitted

**Files:**
- Read: working tree

- [ ] **Step 1: Final clean check**

```bash
git status --short
```
Expected: empty.

If non-empty: pause, surface to user, decide per-file (commit on appropriate branch, gitignore, or `git restore`). Do **not** mass-commit untracked files — review each.

- [ ] **Step 2: Snapshot HEAD for the prior commits to reference**

```bash
echo "Pre-stub HEAD: $(git rev-parse HEAD)" > docs/_audit/baseline.md
git add docs/_audit/baseline.md
git commit -m "audit(baseline): record HEAD before stub commits"
```

---

## Phase B — Step 1: Write 7 stubs as priors

**Convention.** Each stub starts with a `## Prior (written before audit, YYYY-MM-DD)` section listing what we currently believe the finished doc will say. The rest of the doc is empty placeholder sections to be filled later.

Each task = one commit with message `stub(prior): <doc>`. Commit them sequentially so the git log shows the prior order cleanly.

### Task B.1: Stub `AUDIT_FINDINGS.md`

**Files:**
- Create: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Write the stub**

```markdown
# Audit Findings — Lab Talk Prep

**Status:** Prior recorded; audit pending.

## Prior (written before audit, 2026-04-26)

What I currently expect the audit to find:

### Greens (expected to hold up)
- Bootstrap CIs (BCa, B=10000) — Exp 43 onwards used the rigorous protocol consistently.
- Multi-seed averaging-before-bootstrap — implemented post-Bouthillier.
- CV-tuned probes (`GridSearchCV` on C/alpha) — Exp 43 onwards.
- Pooled disorder ρ + cluster bootstrap — Exp 43 fix.
- 5-PLM Exp 46 splits — same partition across PLMs by construction (single split, embeddings re-extracted).

### Yellows (expected to need clarification, not fixes)
- Disorder retention 94.9 % — real, but the floor (raw 1024d) and the noise of probes need to be co-reported.
- ANKH disorder retention 94.8 % — same caveat.
- DCT K=4 — chosen empirically; need a one-line "why not K=8" answer.
- The receiver-side "no codebook" claim for binary — needs a clean demo (decoder uses `h5py + numpy` only, nothing from `OneEmbeddingCodec`).

### Reds (expected to require fixes or honest demotion)
- `one_embedding/` package referenced in CLAUDE.md / MEMORY.md but not at repo root → confirm location, fix references everywhere.
- Modified phylo file (n_taxa 156→24) → recover or document the canonical run.
- Possibly: a few claims in CLAUDE.md / README without a traced source experiment.
- Possibly: dead code, TODO/FIXME markers in `src/`.
- Possibly: pytest failures in the current state (we have not run the full suite recently).

### Predicted distribution
~70 % green, ~20 % yellow, ~10 % red.

## Posterior (filled by audit, Tasks C.1–C.10)

(empty)
```

- [ ] **Step 2: Commit**

```bash
git add docs/AUDIT_FINDINGS.md
git commit -m "stub(prior): AUDIT_FINDINGS.md"
```

---

### Task B.2: Stub `STATE_OF_THE_PROJECT.md`

**Files:**
- Create: `docs/STATE_OF_THE_PROJECT.md`

- [ ] **Step 1: Write the stub**

```markdown
# State of the Project — Honest Write-Up

**Status:** Prior recorded; real version pending.

## Prior (written before audit, 2026-04-26)

What I currently believe the finished write-up will say.

### One-paragraph summary (predicted)
> The OneEmbedding codec compresses per-residue PLM embeddings ~37× (binary default, no codebook) at 95–100 % task retention across 6 tasks and 5 PLMs, with rigorous (BCa) error bars. The work is bottlenecked on three open problems: a persistent ~5 % gap on disorder, a CNN capacity ceiling at 69 % bit accuracy for sequence → embedding (Exp 50), and the absence of a strong baseline against co-distilled VESM. Next moves: transformer backbone for Exp 50, multi-task 3Di head, multi-teacher distillation. Strong contender for a Bioinformatics-tier short paper now.

### What works (predicted to hold up)
- 232 compression methods benchmarked.
- Universal codec, configurable via 4 knobs.
- Multi-PLM validation (5 PLMs, 6 tasks).
- Sequence→OE shows non-trivial signal (~69 % bit accuracy, 0.55 cosine).
- Phylogenetics from embeddings (Exp 35) recovers 11/12 monophyletic families.

### What doesn't (predicted weak spots)
- Disorder retention plateaus at ~95 %.
- Exp 50 CNN ceiling.
- Some claims may not have a traced source.

### Predicted "open problems" sections
1. Disorder gap (mechanism + fix candidates).
2. Sequence → embedding ceiling (architecture lever).
3. Multi-teacher / co-distilled comparison missing.

### Predicted "next directions"
1. Stage 4 transformer for Exp 50.
2. Exp 51 PolarQuant (magnitude-augmented binary).
3. Exp 52 3Di multi-task head.

## Posterior (filled at Task E.1)

(empty)
```

- [ ] **Step 2: Commit**

```bash
git add docs/STATE_OF_THE_PROJECT.md
git commit -m "stub(prior): STATE_OF_THE_PROJECT.md"
```

---

### Task B.3: Stub `MANUSCRIPT_SKELETON.md`

**Files:**
- Create: `docs/MANUSCRIPT_SKELETON.md`

- [ ] **Step 1: Write the stub**

```markdown
# Manuscript Skeleton — Predicted Outline

**Status:** Prior recorded; real outline at Task E.2.

## Prior (written before audit, 2026-04-26)

### Predicted title candidates
1. "OneEmbedding: a universal codec for protein-language-model embeddings"
2. "37× lossless-grade compression of PLM embeddings, validated across 5 models"
3. "What survives compression: a probe of PLM embedding geometry"

### Predicted abstract structure (5 sentences)
1. PLM embeddings are large; downstream uses store and ship them at scale.
2. We benchmarked 232 compression methods on 6 tasks across 5 PLMs.
3. A single configurable codec (centering + RP896 + binary, ~37× / no codebook) achieves 95–100 % retention with BCa-CI bounds.
4. Disorder is the sole consistent weak spot (~95 %); ABTT preprocessing destroys it (Exp 45).
5. The same compressed format admits sequence-only prediction (Exp 50, 69 % bit acc CATH-split), suggesting a path to FASTA → embedding deployment.

### Predicted figure list (~6)
- Fig 1: Pareto plot (compression × retention).
- Fig 2: 5-PLM × 4-task heat-map.
- Fig 3: Codec sweep (Exp 47).
- Fig 4: ABTT artifact (Exp 45).
- Fig 5: Exp 50 learning curve / ceiling.
- Fig 6: Phylogenetics example (Exp 35).

### Predicted limitations (will be in discussion)
- Disorder gap.
- No co-distilled VESM baseline.
- Sequence → OE ceiling.
- Single MPS host — not validated on CUDA at scale.

## Posterior (filled at Task E.2)

(empty)
```

- [ ] **Step 2: Commit**

```bash
git add docs/MANUSCRIPT_SKELETON.md
git commit -m "stub(prior): MANUSCRIPT_SKELETON.md"
```

---

### Task B.4: Prior of the README rewrite

**Files:**
- Create: `docs/_priors/README_REWRITE_PRIOR.md`

**Why a separate file.** The live `README.md` should not be a stub at any point. So the README's prior lives in `docs/_priors/` as an immutable record; the real rewrite happens to `README.md` at Task E.3.

- [ ] **Step 1: Write the prior**

```markdown
# README Rewrite — Prior

**Status:** Predicted shape of the rewritten README. Real rewrite at Task E.3 lands in `/README.md`.

## Prior (2026-04-26): predicted README sections

1. **Title + one-line pitch.** "Universal codec for PLM per-residue embeddings — 37× compression, 95–100 % retention, 5 PLMs."
2. **Headline numbers** (with CIs): the lossless / int4 / PQ M=224 / binary tier table from CLAUDE.md.
3. **Quick start** — `OneEmbeddingCodec()` four-line example. Will need to verify the import path matches whatever the audit decides about `one_embedding/` package vs `src/one_embedding/`.
4. **Pipeline diagram** (text or ASCII): center → RP → quantize → DCT K=4.
5. **5-PLM validation table** (Exp 46).
6. **Codec sweep** (Exp 47).
7. **What's not solved** — disorder gap, Exp 50 ceiling.
8. **Pointers** to `docs/STATE_OF_THE_PROJECT.md`, `docs/MANUSCRIPT_SKELETON.md`, `docs/HANDOFF.md`.
9. **Citation block** — placeholder until manuscript exists.

### What I expect to remove from the current README
- Any reference to a top-level `one_embedding/` package if it doesn't exist.
- Stale tier tables that pre-date the unified codec / Exp 47 numbers.
- Anything not traceable to a current experiment script + result file.

### What I expect to keep
- The 5-PLM table (Exp 46 numbers are recent).
- The Pareto / codec story.
- The methodology block (BCa, paired bootstrap, etc.).
```

- [ ] **Step 2: Commit**

```bash
mkdir -p docs/_priors
git add docs/_priors/README_REWRITE_PRIOR.md
git commit -m "stub(prior): README rewrite plan"
```

---

### Task B.5: Stub `HANDOFF.md`

**Files:**
- Create: `docs/HANDOFF.md`

- [ ] **Step 1: Write the stub**

```markdown
# Hand-Off Doc — Run the Codec in 15 Minutes

**Status:** Prior recorded; real hand-off at Task G.1.

## Prior (2026-04-26): predicted shape

A labmate clones the repo and within 15 minutes can:
1. Set up env (`uv sync` or equivalent — confirm during audit).
2. Locate or extract a PLM embedding for a small test set.
3. Encode with `OneEmbeddingCodec()`.
4. Decode and verify round-trip.
5. Reproduce one row of the 5-PLM table (predicted: ProtT5 SS3 retention).

### Predicted sections
- Setup (one block).
- Test data (where it lives in `data/`).
- Five-line encode demo.
- Five-line decode demo (must use only `h5py + numpy` for the binary default — no `OneEmbeddingCodec` import on the receiver side).
- "Reproduce one number" recipe.
- Common gotchas (predicted: MPS float32 only, `torch.linalg.svdvals` not on MPS, `clip_grad_norm_` + inf grads = NaN).

## Posterior (filled at Task G.1)

(empty)
```

- [ ] **Step 2: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "stub(prior): HANDOFF.md"
```

---

### Task B.6: Stub `EXPECTED_QA.md`

**Files:**
- Create: `docs/EXPECTED_QA.md`

- [ ] **Step 1: Write the stub**

```markdown
# Expected Q&A — Anticipated Probes for the Lab Seminar

**Status:** Prior recorded; refined at Task G.2.

## Prior (2026-04-26): predicted hardest questions

Numbered by severity (1 = most likely / hardest).

1. **Why was ABTT removed by default?** Predicted answer: Exp 45 showed PC1 is 73 % aligned with the disorder direction; removing it costs 6–11 pp of disorder retention.
2. **Disorder retention 94.9 % — what's the baseline noise floor?** Predicted answer: probe-level retest variability ~0.5 pp; the gap is real and exceeds the floor by ~10×. Need to verify in audit.
3. **Why binary as the default rather than PQ?** Predicted answer: 1500 prot/s encode vs 75 prot/s for PQ; on par with PQ M=128 on disorder; no codebook to ship.
4. **ANKH disorder 94.8 % — what's special about ANKH?** Predicted answer: ANKH's tokenizer has known subword artifacts (we hit it in Exp 46); the result is consistent with that artifact, not the codec.
5. **Cross-PLM split fairness?** Predicted answer: same train/test partition across all 5 PLMs (single split, embeddings re-extracted per model). Need to verify in audit.
6. **Why DCT K=4?** Predicted answer: empirical sweep at codec design time. Need a citation. May convert to "we haven't tested K=8" if no evidence.
7. **Co-distilled VESM as baseline?** Predicted answer: not done; honest gap. On the next-steps list.
8. **Exp 50 plateau at 69 %?** Predicted answer: capacity-bound; 3 stages × 2 loss types × 2 data scales all converge to the same number. Architecture is the next lever.
9. **How does retrieval stay at 100 % when per-residue tasks lose 5–8 %?** Predicted answer: retrieval lives in cosine geometry, which RP/quantization preserve well; per-residue tasks need fine-grained directions that quantization smears.
10. **What's the comparison vs FASTA + a small predictor?** Predicted answer: not done explicitly. Honest gap. On the next-steps list (this is essentially what Exp 50 is sneaking up on).
11. **CATH split — do you cluster at H, T, or A?** Predicted answer: H-split is main, T-split is stress test (Exp 50 design).
12. **Why not also report median ± IQR?** Predicted answer: BCa CIs already cover the asymmetry; happy to add medians on request.

## Posterior (refined at Task G.2)

(empty)
```

- [ ] **Step 2: Commit**

```bash
git add docs/EXPECTED_QA.md
git commit -m "stub(prior): EXPECTED_QA.md"
```

---

### Task B.7: Stub `CALIBRATION.md`

**Files:**
- Create: `docs/CALIBRATION.md`

- [ ] **Step 1: Write the stub**

```markdown
# Calibration — Prior vs Posterior Across All 7 Docs

**Status:** Stub. Filled at Task I.3 (between dry runs).

## Prior (2026-04-26): expected calibration shape

Predicted dimensions of mis-calibration:
- **Audit reds**: I predict 70/20/10 green/yellow/red. Real distribution likely closer to 60/25/15 — I usually under-estimate edge-case rot in older code.
- **README**: predicted 70 % overlap between prior outline and final. The biggest unknown is the actual location of the package (`one_embedding/` vs `src/one_embedding/`).
- **EXPECTED_QA**: predicted I will have 8 of 12 questions roughly right. The 4 wrong ones will be questions I didn't anticipate at all (epistemic blind spots).
- **STATE_OF_THE_PROJECT**: predicted the open-problems section will gain 1–2 items I hadn't thought of when writing the stub.
- **Manuscript skeleton**: predicted figure list survives ~80 %; abstract gets reworded substantially.

## Final calibration (filled at Task I.3)

For each of the 7 docs, one paragraph:
- Where the prior was right.
- Where it was wrong.
- What changed materially.
- Why our prior was wrong (overconfidence / blind spot / missing data).

(empty — six more paragraphs to write at I.3)
```

- [ ] **Step 2: Commit**

```bash
git add docs/CALIBRATION.md
git commit -m "stub(prior): CALIBRATION.md"
```

---

## Phase C — Step 2: Audit

Output lives in `docs/_audit/` (raw evidence) and rolls up into `docs/AUDIT_FINDINGS.md`. Each task ends with a small commit; `AUDIT_FINDINGS.md` is updated cumulatively.

### Task C.1: Repo hygiene + drift inventory

**Files:**
- Create: `docs/_audit/hygiene.md`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Run hygiene checks**

```bash
git status --short
git ls-files | wc -l
git ls-files | xargs -I{} du -k {} 2>/dev/null | sort -rn | head -20
```

- [ ] **Step 2: Confirm one_embedding/ package location**

```bash
ls one_embedding/ 2>/dev/null || echo "NOT at top level"
ls src/one_embedding/ 2>/dev/null
```

- [ ] **Step 3: Find every reference to the package**

Use Grep to find references in CLAUDE.md, README.md, MEMORY.md, docs/:
- Pattern: `from one_embedding|import one_embedding|^\s*one_embedding/`
- Record each hit + file:line in `docs/_audit/hygiene.md`.

- [ ] **Step 4: Write hygiene report**

```markdown
# Hygiene Audit

## Repo size & largest tracked files
<paste output>

## one_embedding/ package
- Top-level: <yes|no>
- src/one_embedding/: <yes|no>
- References in docs that need fixing: <list of file:line>

## Other drift items
<list>
```

- [ ] **Step 5: Roll up to AUDIT_FINDINGS.md**

Append a `## Repo hygiene` section under "Posterior" with green/yellow/red items.

- [ ] **Step 6: Commit**

```bash
git add docs/_audit/hygiene.md docs/AUDIT_FINDINGS.md
git commit -m "audit(hygiene): repo state + package-location drift"
```

---

### Task C.2: Code correctness — pytest baseline + markers + codec read

**Files:**
- Create: `docs/_audit/pytest_baseline.txt`
- Create: `docs/_audit/code_markers.txt`
- Create: `docs/_audit/codec_review.md`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Run pytest baseline**

```bash
uv run pytest tests/ -x --tb=short 2>&1 | tee docs/_audit/pytest_baseline.txt
echo "Exit code: $?" >> docs/_audit/pytest_baseline.txt
```

- [ ] **Step 2: Search for code markers**

Use Grep with pattern `TODO|FIXME|HACK|XXX` over `src/`, output to `docs/_audit/code_markers.txt`.

- [ ] **Step 3: Re-read `src/one_embedding/codec_v2.py` end-to-end**

Open the file, read it top to bottom. Note in `docs/_audit/codec_review.md`:
- Lines that are unclear, suspicious, or carry implicit assumptions.
- Any default values not explained by a comment.
- Any error path that would surprise the user.
- Any place where receiver-side decode would need more than `h5py + numpy`.

Template for `codec_review.md`:
```markdown
# Codec V2 Review

## Defaults inventory
| Param | Default | Justification source | Comment |
|-------|---------|---------------------|---------|

## Suspicious / unclear lines
- `codec_v2.py:NNN` — <observation>

## Receiver-side decode check
- Does decode require codec object? <yes|no, with reason>
- Does decode require numpy + h5py only? <yes|no>
```

- [ ] **Step 4: Roll up to AUDIT_FINDINGS.md**

Append `## Code correctness` section.

- [ ] **Step 5: Commit**

```bash
git add docs/_audit/pytest_baseline.txt docs/_audit/code_markers.txt docs/_audit/codec_review.md docs/AUDIT_FINDINGS.md
git commit -m "audit(code): pytest baseline + markers + codec_v2 review"
```

---

### Task C.3: Benchmark correctness — splits leakage

**Files:**
- Create: `docs/_audit/splits.md`
- Possibly create: `scripts/audit/check_split_leakage.py`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Locate every benchmark's split files**

For each cited result in CLAUDE.md (Exp 43, 44, 46, 47, 50), find:
- The script that produced the split.
- The CATH/SCOPe annotation file used.
- The train/val/test partition (saved or recomputed at run time).

Record in `docs/_audit/splits.md`.

- [ ] **Step 2: Verify Exp 46 multi-PLM uses identical splits**

Each PLM's run should partition on the same protein IDs. Cross-check by reading the split logic in `experiments/46_multi_plm_benchmark.py`.

- [ ] **Step 3: For Exp 50, run MMseqs2 leakage audit if not already done**

Check `results/exp50/` for an existing leakage report. If absent, defer to Exp 50's own plan (it has a leakage audit task).

- [ ] **Step 4: Write splits report**

```markdown
# Splits Audit

| Experiment | Split type | Partition file | Leakage check | Status |
|------------|-----------|----------------|---------------|--------|
| Exp 43 | random | <path or "computed at runtime"> | <result> | green/yellow/red |
| Exp 46 | <type> | <path> | <result> | <status> |
| Exp 47 | <type> | <path> | <result> | <status> |
| Exp 50 | CATH-H + CATH-T | data/external/cath20/... | MMseqs2 separate | green |
```

- [ ] **Step 5: Roll up + commit**

```bash
git add docs/_audit/splits.md docs/AUDIT_FINDINGS.md
git commit -m "audit(splits): leakage check across all cited benchmarks"
```

---

### Task C.4: Benchmark correctness — bootstrap, probes, baselines

**Files:**
- Create: `docs/_audit/stats.md`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Inspect bootstrap implementation**

For each result table cited in CLAUDE.md, find the script and confirm:
- Bootstrap is BCa with B=10000.
- Percentile fallback applies for n<25.
- For disorder: cluster bootstrap (resample proteins, recompute pooled stat).
- For retention: paired bootstrap.

Record per-table evidence in `docs/_audit/stats.md`.

- [ ] **Step 2: Inspect probe implementation**

Confirm each probe is `GridSearchCV` over C/alpha grids (not hardcoded), `random_state=42`, and that predictions are averaged over seeds {42, 43, 44} *before* bootstrapping.

- [ ] **Step 3: Inspect baseline pooling**

For retrieval / per-protein retention, confirm raw and compressed both use the same DCT K=4 pooling (Exp 43 fairness fix).

- [ ] **Step 4: Write stats report**

```markdown
# Stats Audit

| Table source | Bootstrap | Probe CV | Multi-seed | Baseline fair | Status |
|--------------|-----------|----------|------------|---------------|--------|
| Exp 43 § 4.1 | BCa B=10000 | yes | 3-seed avg pre-boot | DCT K=4 both | green |
| Exp 44 § ... | <result> | <result> | <result> | <result> | <status> |
| Exp 46 § ... | ... | ... | ... | ... | ... |
| Exp 47 § ... | ... | ... | ... | ... | ... |
```

- [ ] **Step 5: Roll up + commit**

```bash
git add docs/_audit/stats.md docs/AUDIT_FINDINGS.md
git commit -m "audit(stats): bootstrap + probe + baseline conformance"
```

---

### Task C.5: Phylo file investigation

**Files:**
- Modify: `docs/_audit/hygiene.md`
- Modify: `docs/AUDIT_FINDINGS.md`

**Context:** `data/benchmarks/embedding_phylo_results.json` was overwritten in Task A.1 with a 24-taxa rerun. We now decide what to do.

- [ ] **Step 1: Recover the previous version**

Find the commit that introduced the 24-taxa change (Task A.1 commit), then read its parent:
```bash
PRE_A1_COMMIT=$(git log --diff-filter=M --pretty=format:'%H %s' -- data/benchmarks/embedding_phylo_results.json | grep -i "in-flight phylo result downsize" | head -1 | awk '{print $1}')
git show ${PRE_A1_COMMIT}^:data/benchmarks/embedding_phylo_results.json > /tmp/phylo_prev.json
diff /tmp/phylo_prev.json data/benchmarks/embedding_phylo_results.json
```
Expected: shows `n_taxa: 156→24` and `mcmc_time_s: 89.81→7.88` only.

- [ ] **Step 2: Find the script that wrote the file**

Use Grep for `embedding_phylo_results.json` over `experiments/` and `src/`. Determine which run-config produces 156 taxa vs 24.

- [ ] **Step 3: Decision**

Three options:
- (a) The 156-taxa run is the canonical reference → restore from `/tmp/phylo_prev.json`, commit, document the 24-taxa rerun in `docs/_audit/`.
- (b) The 24-taxa rerun is intentional and the 156 was a stale artifact → keep current, document.
- (c) Unclear → keep current, flag in `EXPECTED_QA.md`, mention in talk under "honest open problem."

Record decision in `docs/AUDIT_FINDINGS.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/_audit/hygiene.md docs/AUDIT_FINDINGS.md
[ -e data/benchmarks/embedding_phylo_results.json.restored ] && git add data/benchmarks/embedding_phylo_results.json
git commit -m "audit(phylo): document the 24-taxa rerun decision"
```

---

### Task C.6: Parameter intentionality

**Files:**
- Create: `docs/_audit/params.md`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Inventory every default in `OneEmbeddingCodec`**

Read `src/one_embedding/codec_v2.py` and list each constructor default. For each, fill:

```markdown
| Param | Default | Evidence | Source experiment | Commit | Status |
|-------|---------|----------|-------------------|--------|--------|
| `d_out` | 896 | "PQ M=224 ≈ lossless on retrieval, 1pp drop on SS" | Exp 47 | <hash> | green |
| `quantization` | 'binary' | "37×, no codebook, on par with PQ128 on disorder" | Exp 47 | <hash> | green |
| `pq_m` | 'auto' | "2 dims/subvector heuristic" | <exp> | <hash> | <status> |
| `abtt_k` | 0 | "ABTT PC1 73 % aligned with disorder; +6–11 pp recovery" | Exp 45 | <hash> | green |
| DCT K | 4 | <find evidence> | <exp> | <hash> | <status> |
| RP seed | <value> | <find> | <exp> | <hash> | <status> |
```

- [ ] **Step 2: For any default with status ≠ green, mark red in AUDIT_FINDINGS.md**

- [ ] **Step 3: Commit**

```bash
git add docs/_audit/params.md docs/AUDIT_FINDINGS.md
git commit -m "audit(params): defaults inventory + evidence trace"
```

---

### Task C.7: Claims register

**Files:**
- Create: `docs/_audit/claims.md`
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: List every numeric claim in CLAUDE.md and README.md**

Extract every percentage, every retention number, every CI bound. ~50–80 claims expected.

- [ ] **Step 2: For each, fill the trace**

```markdown
| Claim | Doc:line | Source script | Source result file | Commit | Status |
|-------|----------|---------------|-------------------|--------|--------|
| "SS3 ret 99.1 % BCa [98.5,99.6]" | CLAUDE.md:115 | experiments/43_rigorous_benchmark/... | results/exp43/... | <hash> | green |
| "37× compression" | CLAUDE.md:8 | experiments/47_codec_sweep.py | results/exp47/... | <hash> | green |
| ... | ... | ... | ... | ... | ... |
```

- [ ] **Step 3: Mark red any claim that fails to trace**

- [ ] **Step 4: Commit**

```bash
git add docs/_audit/claims.md docs/AUDIT_FINDINGS.md
git commit -m "audit(claims): traceability register for CLAUDE/README numbers"
```

---

### Task C.8: Final AUDIT_FINDINGS.md roll-up

**Files:**
- Modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Reorganise the Posterior section into a single triage table**

```markdown
## Posterior — final audit roll-up

| # | Track | Finding | Color | Fix plan | Owner |
|---|-------|---------|-------|----------|-------|
| 1 | hygiene | one_embedding package not at top level | red | fix CLAUDE/README/MEMORY refs | E.3 |
| 2 | code | pytest baseline: <N> pass, <M> fail | yellow | document failures, no fixes | done |
| 3 | splits | Exp 46 splits identical across PLMs | green | n/a | done |
| ... | ... | ... | ... | ... | ... |
```

- [ ] **Step 2: Compute the final green/yellow/red totals**

Add to the top of the Posterior section.

- [ ] **Step 3: Commit**

```bash
git add docs/AUDIT_FINDINGS.md
git commit -m "audit(roll-up): final triage table"
```

---

## Phase D — Step 3: Triage + curate

### Task D.1: Apply quick fixes (<4h items)

**Files:**
- Modify: variable (depends on findings)

- [ ] **Step 1: From AUDIT_FINDINGS.md, list every red marked "fix in <4h"**

- [ ] **Step 2: For each, make the fix, verify, commit**

For each fix, the commit message should reference the audit finding:
```bash
git commit -m "fix(audit-N): <one-line description>"
```

- [ ] **Step 3: Update AUDIT_FINDINGS.md to mark fixes as done**

- [ ] **Step 4: Commit the audit update**

```bash
git add docs/AUDIT_FINDINGS.md
git commit -m "audit(triage): mark applied fixes"
```

---

### Task D.2: Decide demote / cut for medium and unfixable reds

**Files:**
- Modify: `docs/AUDIT_FINDINGS.md`
- Modify: `docs/EXPECTED_QA.md`

- [ ] **Step 1: For each red in (ii) "fix in 4–12h" or (iii) "unfixable" buckets**

Decide one of:
- (a) Demote on slides + flag verbally → add to `EXPECTED_QA.md` with the verbal flag.
- (b) Cut from the deck → add to `EXPECTED_QA.md` as a question we'll answer if asked ("we omitted X because Y").

- [ ] **Step 2: Update both docs**

- [ ] **Step 3: Commit**

```bash
git add docs/AUDIT_FINDINGS.md docs/EXPECTED_QA.md
git commit -m "audit(triage): demote/cut decisions for medium reds"
```

---

### Task D.3: Curate branches — main vs future-directions

**Files:**
- Possibly modify: `docs/AUDIT_FINDINGS.md`

- [ ] **Step 1: Inventory branches**

```bash
git branch -a
git worktree list
```

- [ ] **Step 2: For each non-main branch, decide**

- Stays as feature branch (in flight).
- Folds into `future-directions/<topic>` branch.
- Gets archived (no further work; mention in handoff).

- [ ] **Step 3: Apply renames / merges as decided (no destructive ops without user confirmation)**

If a branch is to be renamed:
```bash
git branch -m <old> future-directions/<new>
```

If a worktree should remain in place: leave it. Just document in `docs/_audit/worktrees.md`.

- [ ] **Step 4: Commit if any docs changed**

```bash
git add docs/AUDIT_FINDINGS.md docs/_audit/worktrees.md
git commit -m "audit(curate): branch + worktree disposition"
```

---

## Phase E — Steps 4-6: Real versions of derived docs

### Task E.1: `STATE_OF_THE_PROJECT.md` (real)

**Files:**
- Modify: `docs/STATE_OF_THE_PROJECT.md` (overwrite the stub, keeping the prior section at the top for the calibration trail)

- [ ] **Step 1: Build the section list**

Required sections:
- One-paragraph executive summary.
- Project arc: 2–3 paragraphs on how we got from "compress PLM embeddings" to the current codec.
- The codec: what it does, four knobs, default configuration, table of headline numbers (lossless / int4 / PQ M=224 / binary).
- Multi-PLM validation (Exp 46): the 5×4 table.
- Codec sweep (Exp 47): which configs Pareto-dominate.
- Sequence → OE (Exp 50): rigorous CATH-split results, where the 69 % ceiling came from.
- Phylogenetics (Exp 35): one paragraph + headline number.
- Negative results: ABTT artifact, VQ failures, CNN ceiling.
- Open problems: disorder gap (with proposed mechanism), Exp 50 architecture lever, missing baselines.
- Next directions: Stage 4 transformer, Exp 51 PolarQuant, Exp 52 3Di multi-task, Exp 53 Foldseek data, multi-teacher distillation.
- Limitations of the work as currently presented.

- [ ] **Step 2: Write the doc**

Keep the `## Prior` section at the top (move it down so the live read order is: title → executive summary → project arc → ... → prior at end as historical record). Or split into a separate file. Decision: keep prior in same file, demoted to an appendix `## Appendix: prior (2026-04-26)` at the end. Commit-history is the audit trail; readability of the live doc takes priority.

- [ ] **Step 3: Verify every cited number against the claims register (Task C.7)**

Cross-reference each table value with its trace.

- [ ] **Step 4: Commit**

```bash
git add docs/STATE_OF_THE_PROJECT.md
git commit -m "feat(docs): STATE_OF_THE_PROJECT.md real version"
```

---

### Task E.2: `MANUSCRIPT_SKELETON.md` (real)

**Files:**
- Modify: `docs/MANUSCRIPT_SKELETON.md`

- [ ] **Step 1: Derive sections from `STATE_OF_THE_PROJECT.md`**

Required sections:
- Title (3 candidates).
- Abstract (5–7 sentences).
- Introduction (3 paragraphs: motivation, gap, contribution).
- Related work (1 paragraph: prior compression work, prior PLM benchmarking).
- Methods (codec, statistics, datasets — bullet form for the skeleton).
- Results (per main table, one paragraph stub).
- Discussion (disorder gap, ABTT artifact, sequence→OE implications, scope).
- Limitations.
- Figure list (~6) with one-line captions.
- Reproducibility statement.

- [ ] **Step 2: Write**

Same approach as E.1: prior demoted to appendix.

- [ ] **Step 3: Commit**

```bash
git add docs/MANUSCRIPT_SKELETON.md
git commit -m "feat(docs): MANUSCRIPT_SKELETON.md real version"
```

---

### Task E.3: `README.md` rewrite (real)

**Files:**
- Modify: `README.md`
- Reference: `docs/_priors/README_REWRITE_PRIOR.md`

- [ ] **Step 1: Read the prior**

Read `docs/_priors/README_REWRITE_PRIOR.md` to remember the predicted shape.

- [ ] **Step 2: Rewrite `README.md`**

Required sections:
- Title + one-line pitch.
- Headline number tier table (with CIs).
- Quick start (three lines: install, encode, decode). Verify the import path against the audit finding (top-level package vs `src/one_embedding/`).
- Pipeline diagram (text or ASCII, no emojis).
- 5-PLM validation table (Exp 46).
- Codec sweep highlights (Exp 47).
- "What's not solved" — disorder, Exp 50.
- Pointers to `docs/STATE_OF_THE_PROJECT.md`, `docs/MANUSCRIPT_SKELETON.md`, `docs/HANDOFF.md`.
- License + citation.

- [ ] **Step 3: Verify every number against the claims register**

- [ ] **Step 4: Verify the quick-start example actually runs**

```bash
uv run python -c "
<paste the README quick-start verbatim>
"
```
Expected: no error; output matches what the README claims.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite to match current code reality"
```

---

## Phase F — Step 7: Slide figures

### Task F.1: Inventory existing figures

**Files:**
- Create: `docs/_audit/figures.md`

- [ ] **Step 1: List `docs/figures/` contents**

```bash
ls -la docs/figures/
```

- [ ] **Step 2: For each existing figure, decide reuse / regenerate / drop**

Record decisions in `docs/_audit/figures.md`.

- [ ] **Step 3: Identify missing figures for the talk**

Likely additions:
- Pareto plot (compression × retention).
- 5-PLM × 4-task heat-map.
- Exp 50 learning curve / ceiling.

- [ ] **Step 4: Commit**

```bash
git add docs/_audit/figures.md
git commit -m "audit(figures): existing inventory + missing list"
```

---

### Task F.2: Pareto plot

**Files:**
- Create: `slides/lab-talk/figures/pareto.py`
- Create: `slides/lab-talk/figures/pareto.png`

- [ ] **Step 1: Write the plot script**

Use the codec sweep numbers from Exp 47 (configs in CLAUDE.md table). Plot compression on x, mean retention (across SS3/SS8/Ret/Disorder) on y, with error bars from the BCa CIs. Mark the binary-default and PQ M=224 points.

```python
"""Pareto plot for the talk: compression × retention."""

import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load Exp 47 results
results = json.loads(Path("results/exp47/codec_sweep.json").read_text())  # confirm path during audit

fig, ax = plt.subplots(figsize=(6, 4))

for config in results['configs']:
    x = config['compression']
    y = config['mean_retention']
    yerr = config['mean_ci_halfwidth']
    ax.errorbar(x, y, yerr=yerr, fmt='o', label=config['name'])
    ax.annotate(config['name'], (x, y), fontsize=8)

ax.set_xlabel('Compression (×)')
ax.set_ylabel('Mean retention')
ax.set_xscale('log')
ax.set_title('Pareto: compression × retention (Exp 47)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('slides/lab-talk/figures/pareto.png', dpi=150)
```

- [ ] **Step 2: Run the script**

```bash
uv run python slides/lab-talk/figures/pareto.py
```

- [ ] **Step 3: Sanity-check the figure**

Open `slides/lab-talk/figures/pareto.png`. Verify:
- Axes labeled.
- Both binary-default and PQ M=224 points visible.
- Error bars present.
- No truncated labels.

- [ ] **Step 4: Commit**

```bash
git add slides/lab-talk/figures/pareto.py slides/lab-talk/figures/pareto.png
git commit -m "feat(slides): Pareto plot for codec sweep"
```

---

### Task F.3: 5-PLM × 4-task heat-map

**Files:**
- Create: `slides/lab-talk/figures/multi_plm_heatmap.py`
- Create: `slides/lab-talk/figures/multi_plm_heatmap.png`

- [ ] **Step 1: Write the script**

Use Exp 46 numbers (5 PLMs × 4 tasks: SS3 ret, SS8 ret, Ret@1 ret, Disorder ret). Heat-map cells = retention; diverging colormap centered at 100 %.

```python
"""5-PLM × 4-task retention heatmap."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

results = json.loads(Path("results/exp46/multi_plm_results.json").read_text())

plms = ['ProstT5', 'ProtT5-XL', 'ESM-C-600M', 'ANKH-large', 'ESM2-650M']
tasks = ['SS3', 'SS8', 'Ret@1', 'Disorder']
matrix = np.array([[results[p][t] for t in tasks] for p in plms])

fig, ax = plt.subplots(figsize=(5, 4))
norm = mcolors.TwoSlopeNorm(vmin=90, vcenter=100, vmax=103)
im = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')

ax.set_xticks(range(len(tasks)), tasks)
ax.set_yticks(range(len(plms)), plms)
for i in range(len(plms)):
    for j in range(len(tasks)):
        ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=9)

fig.colorbar(im, ax=ax, label='Retention (%)')
ax.set_title('5-PLM × 4-task retention (Exp 46)')
fig.tight_layout()
fig.savefig('slides/lab-talk/figures/multi_plm_heatmap.png', dpi=150)
```

- [ ] **Step 2: Run + sanity-check + commit**

```bash
uv run python slides/lab-talk/figures/multi_plm_heatmap.py
git add slides/lab-talk/figures/multi_plm_heatmap.py slides/lab-talk/figures/multi_plm_heatmap.png
git commit -m "feat(slides): 5-PLM × 4-task heatmap"
```

---

### Task F.4: Exp 50 learning curve / ceiling

**Files:**
- Create: `slides/lab-talk/figures/exp50_ceiling.py`
- Create: `slides/lab-talk/figures/exp50_ceiling.png`

- [ ] **Step 1: Write the script**

Use Exp 50 stages 1–3 results (bit accuracy + cosine sim per stage). Show the plateau visually.

```python
"""Exp 50 architecture ceiling at ~69%."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Confirm result path during audit
stage1 = json.loads(Path("results/exp50/stage1_results.json").read_text())
stage2 = json.loads(Path("results/exp50/stage2_results.json").read_text())
stage3 = json.loads(Path("results/exp50/stage3_results.json").read_text())

stages = ['Stage 1\n(2.5K, BCE)', 'Stage 2\n(2.5K, BCE)', 'Stage 3\n(5K, MSE)']
bit_acc = [stage1['bit_acc'], stage2['bit_acc'], stage3['bit_acc']]
cos_sim = [stage1['cos_sim'], stage2['cos_sim'], stage3['cos_sim']]

fig, ax1 = plt.subplots(figsize=(5, 4))

ax1.bar([i - 0.2 for i in range(3)], bit_acc, width=0.4, label='Bit accuracy', color='C0')
ax1.set_ylabel('Bit accuracy', color='C0')
ax1.set_ylim(0.6, 0.75)
ax1.axhline(0.69, ls='--', color='C0', alpha=0.5, label='Ceiling ~69%')

ax2 = ax1.twinx()
ax2.bar([i + 0.2 for i in range(3)], cos_sim, width=0.4, label='Cosine sim', color='C1')
ax2.set_ylabel('Cosine similarity', color='C1')
ax2.set_ylim(0.5, 0.6)

ax1.set_xticks(range(3), stages)
ax1.set_title('Exp 50: CNN capacity ceiling')
fig.tight_layout()
fig.savefig('slides/lab-talk/figures/exp50_ceiling.png', dpi=150)
```

- [ ] **Step 2: Run + sanity-check + commit**

```bash
uv run python slides/lab-talk/figures/exp50_ceiling.py
git add slides/lab-talk/figures/exp50_ceiling.py slides/lab-talk/figures/exp50_ceiling.png
git commit -m "feat(slides): Exp 50 ceiling figure"
```

---

## Phase G — Steps 8-9: HANDOFF + EXPECTED_QA

### Task G.1: `HANDOFF.md` (real)

**Files:**
- Modify: `docs/HANDOFF.md`

- [ ] **Step 1: Verify each step actually works**

Walk through the handoff yourself: clone → install → extract one embedding → encode → decode → reproduce one number. Capture exact commands and outputs.

- [ ] **Step 2: Write the doc**

Required sections:
- Setup (one block).
- Test data location.
- Encode demo (≤5 lines).
- Decode demo (≤5 lines, must use only h5py + numpy for the binary default — confirm with audit).
- Reproduce one number recipe (predicted: ProtT5 SS3 retention).
- Common gotchas.
- Where to look next: STATE_OF_THE_PROJECT.md, MANUSCRIPT_SKELETON.md.
- Appendix: prior (preserved from stub).

- [ ] **Step 3: Validate by running**

Run every command in the doc, top to bottom. If any fails, fix the doc.

- [ ] **Step 4: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "feat(docs): HANDOFF.md real version, end-to-end verified"
```

---

### Task G.2: `EXPECTED_QA.md` (real)

**Files:**
- Modify: `docs/EXPECTED_QA.md`

- [ ] **Step 1: Refine prior list against the audit findings**

Walk through the prior questions (Task B.6). For each:
- If audit confirmed the predicted answer: keep it, add a one-line evidence pointer.
- If audit changed the answer: rewrite it.
- If audit revealed a new question (e.g. a red finding we'll demote): add it.

- [ ] **Step 2: Add 3–5 questions discovered during audit**

These are the ones the prior didn't anticipate. Examples that may emerge:
- "What does your gitignore actually exclude?" if hygiene flagged something unusual.
- "Why is the modified phylo file 24-taxa?" if Task C.5 chose option (c).
- "How are your test failures distributed?" if pytest baseline shows non-zero failures.

- [ ] **Step 3: Add evidence pointer per answer**

Format: `<answer>. **Evidence:** <experiment + commit + result file>.`

- [ ] **Step 4: Add a "questions we cannot fully answer" section**

For honest acknowledgment of gaps.

- [ ] **Step 5: Commit**

```bash
git add docs/EXPECTED_QA.md
git commit -m "feat(docs): EXPECTED_QA.md refined against audit"
```

---

## Phase H — Step 10: Slide deck

### Task H.1: Build the Marp slide deck

**Files:**
- Create: `slides/lab-talk/talk.md`
- Possibly create: `slides/lab-talk/theme.css` (only if default Marp theme is insufficient)
- Create: `slides/lab-talk/talk.pdf` (built artifact)

- [ ] **Step 1: Initialize Marp file**

```markdown
---
marp: true
paginate: true
theme: default
size: 16:9
title: OneEmbedding — universal codec for protein-language-model embeddings
---

<!-- _class: lead -->
# OneEmbedding
### Universal codec for PLM per-residue embeddings

Ivan · Rost lab seminar · 2026

---
```

- [ ] **Step 2: Build slides 1–2 (1 min, motivation)**

Title + the problem (PLM embeddings are large, downstream uses are constrained, what if we could compress them losslessly-grade).

- [ ] **Step 3: Build slides 3–10 (8 min, codec story with humility)**

Per slide, include in a Marp HTML comment the source trace: `<!-- source: experiments/47_codec_sweep.py @ commit b6e1698 -->`.

Slide content order:
- 3: Codec architecture (4-knob diagram).
- 4: Method statistics block (BCa, paired bootstrap, multi-seed, CV-tuned, pooled disorder, fair baselines).
- 5: Headline tier table (lossless / int4 / PQ M=224 / binary).
- 6: Pareto plot.
- 7: 5-PLM heat-map.
- 8: Exp 47 codec sweep table.
- 9: Receiver-side decoder (the "no codebook" demo).
- 10: One real protein — show encode → file → decode → SS3 prediction matches.

- [ ] **Step 4: Build slides 11–12 (2.5 min, Exp 50)**

- 11: Sequence → OE setup + CATH H/T splits.
- 12: Stage 1–3 results + ceiling figure.

- [ ] **Step 5: Build slides 13–14 (2.5 min, negative results + roadmap)**

- 13: Negative results (VQ failed, ABTT destroys disorder, CNN ceiling, missing baselines).
- 14: Roadmap (Stage 4 transformer, Exp 51 PolarQuant, Exp 52 3Di multi-task, Exp 53 Foldseek, multi-teacher).

- [ ] **Step 6: Build slide 15 (1 min, summary + asks)**

What we have, what's open, what feedback would help.

- [ ] **Step 7: Build PDF**

First-time check that Marp is available:
```bash
which npx || echo "NEED NODE — install via brew install node"
npx --yes @marp-team/marp-cli@latest --version
```

Build:
```bash
npx --yes @marp-team/marp-cli@latest slides/lab-talk/talk.md -o slides/lab-talk/talk.pdf
```
If a Chromium-related error appears, set `--allow-local-files` and ensure `CHROME_PATH` points to a Chromium binary (e.g. `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`).

- [ ] **Step 8: Sanity-check the PDF**

Open `slides/lab-talk/talk.pdf`. Verify:
- 15 slides.
- All figures rendered.
- No broken paths.

- [ ] **Step 9: Commit**

```bash
git add slides/lab-talk/talk.md slides/lab-talk/talk.pdf
git commit -m "feat(slides): lab-talk deck — first build"
```

---

## Phase I — Steps 11-12: Dry runs + calibration

### Task I.1: Dry run #1 (solo, timed)

**Files:**
- Create: `slides/lab-talk/dry_run_1.md`

- [ ] **Step 1: Open the PDF, start a timer, deliver the talk to yourself**

Speak out loud (or whisper). Note slides where you stumble, slides that run long, slides where the words don't match what's on the slide.

- [ ] **Step 2: Write the post-mortem**

```markdown
# Dry Run #1 — solo

**Date:** 2026-04-XX
**Total elapsed:** XX:XX (target 15:00)

## Per-slide notes
| Slide | Time spent | Issues | Fix |
|-------|-----------|--------|-----|
| 1 | 0:30 | none | none |
| 2 | 1:00 | "motivation" felt thin | add concrete numbers |
| ... |

## Top 3 issues to fix before run #2
1. ...
2. ...
3. ...
```

- [ ] **Step 3: Commit**

```bash
git add slides/lab-talk/dry_run_1.md
git commit -m "talk(dry-run): solo run #1 post-mortem"
```

---

### Task I.2: Apply dry-run-1 fixes

**Files:**
- Modify: `slides/lab-talk/talk.md`
- Possibly modify: figures, EXPECTED_QA.md

- [ ] **Step 1: Apply each top-3 fix from `dry_run_1.md`**

- [ ] **Step 2: Rebuild the PDF**

```bash
npx @marp-team/marp-cli@latest slides/lab-talk/talk.md -o slides/lab-talk/talk.pdf
```

- [ ] **Step 3: Commit**

```bash
git add slides/lab-talk/talk.md slides/lab-talk/talk.pdf
git commit -m "talk(slides): apply dry-run-1 fixes"
```

---

### Task I.3: `CALIBRATION.md` (real, between dry runs)

**Files:**
- Modify: `docs/CALIBRATION.md`

- [ ] **Step 1: For each of 7 docs, diff prior vs final**

```bash
# Example for STATE_OF_THE_PROJECT.md
git log --oneline docs/STATE_OF_THE_PROJECT.md
git diff <prior_commit> HEAD -- docs/STATE_OF_THE_PROJECT.md
```

- [ ] **Step 2: Write one paragraph per doc**

Format per doc:
```markdown
### `docs/STATE_OF_THE_PROJECT.md`

**Right:** <what the prior nailed>.
**Wrong:** <what changed>.
**Why wrong:** <overconfidence / blind spot / missing data>.
**Material change:** <one specific delta>.
```

- [ ] **Step 3: Add a global summary**

```markdown
## Global calibration

- Prior green/yellow/red prediction: 70/20/10.
- Posterior actual: <X>/<Y>/<Z>.
- Calibration error: <delta>.
- Top 3 systemic biases I had: <list>.
```

- [ ] **Step 4: Commit**

```bash
git add docs/CALIBRATION.md
git commit -m "feat(docs): CALIBRATION.md prior vs posterior"
```

---

### Task I.4: Dry run #2 (with watcher, timed)

**Files:**
- Create: `slides/lab-talk/dry_run_2.md`

- [ ] **Step 1: Deliver the talk to a human watcher with the clock visible**

Note: this requires user availability. If user unavailable, do solo + record audio.

- [ ] **Step 2: Capture watcher's questions**

These are real probes from someone who hasn't seen the work — invaluable signal.

- [ ] **Step 3: Write the post-mortem**

Same format as dry_run_1.md.

- [ ] **Step 4: Commit**

```bash
git add slides/lab-talk/dry_run_2.md
git commit -m "talk(dry-run): watched run #2 post-mortem"
```

---

### Task I.5: Apply dry-run-2 fixes; final PDF

**Files:**
- Modify: `slides/lab-talk/talk.md`
- Possibly modify: figures, `docs/EXPECTED_QA.md` (new questions discovered)
- Modify: `slides/lab-talk/talk.pdf` (final)

- [ ] **Step 1: Apply fixes**

- [ ] **Step 2: Rebuild PDF**

```bash
npx @marp-team/marp-cli@latest slides/lab-talk/talk.md -o slides/lab-talk/talk.pdf
```

- [ ] **Step 3: Final repo clean check**

```bash
git status --short
```
Expected: empty.

- [ ] **Step 4: Final commit**

```bash
git add slides/lab-talk/talk.md slides/lab-talk/talk.pdf docs/EXPECTED_QA.md
git commit -m "talk(slides): final deck after dry run #2"
```

- [ ] **Step 5: Tag the talk-day state**

```bash
git tag -a lab-talk-2026-04-XX -m "State at the time of the lab seminar"
```

---

## Done criteria

- [ ] All 7 doc deliverables (B.1–B.7 stubs → real versions) are committed.
- [ ] `slides/lab-talk/talk.pdf` exists and renders the 15 slides correctly.
- [ ] `git status --short` is empty on `main`.
- [ ] Pytest baseline (Task C.2) is documented; no test that passed at audit time fails at talk time.
- [ ] `docs/CALIBRATION.md` has one paragraph per doc + global summary.
- [ ] Talk dry-run timing is within 14:00–15:30 inclusive.

---

## Notes for the executor

- **No silent demotions.** Any cited number that doesn't survive audit gets fixed, demoted-with-disclosure, or cut-and-named. Risk policy in the spec governs.
- **Commit on every step that produces a file.** The git history is the audit trail and the calibration trail.
- **Stop and ask** if Phase C surfaces something that invalidates the talk's spine (e.g. a leakage bug in Exp 46). Replan with the user.
- **Worktree:** this plan runs in `main`, not in an isolated worktree. The exp50-rigorous worktree is left alone (Task A.2).
- **Tooling:** Marp via `npx @marp-team/marp-cli@latest` (no global install required if npx is available).
