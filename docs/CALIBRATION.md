# Calibration — Prior vs Posterior Across All 7 Docs

**Date:** 2026-04-27
**Status:** Real version (Task I.3). Written between dry-run-1 and dry-run-2.

> The Prior section (2026-04-26 stub, recorded *before* the audit) is preserved
> at the bottom as Appendix. Compare against the live calibration below.

This is the meta-deliverable: did our predictions about each doc match what we
actually produced? Where they didn't, why?

The git history is the audit trail — `git log --oneline -- docs/<X>.md` shows
each doc going from `stub(prior): X` (commit on 2026-04-26) to its final
real version (this week). The diff between those commits IS the calibration
delta this document narrates.

---

## Global calibration

Three patterns from comparing the seven priors to their posteriors:

### 1. **I correctly predicted the SHAPE of the audit findings, but underestimated MARKETING-layer drift.**

Prior: 70 % G / 20 % Y / 10 % R. Posterior: 53 % G / 37 % Y / 10 % R (line items).
- Greens: under-counted (predicted 70, actual 49 / 92 → ~53 %).
- Yellows: under-counted (predicted 20, actual 34 / 92 → ~37 %).
- Reds: nearly exact (predicted 10, actual 9 / 92 → ~10 %).

The yellow under-count was the systemic miss: I didn't anticipate "marketing-layer count claims" (232 methods, 6 tasks, 8 datasets, 1500 prot/s, L=175) as a category. These are claims that are *roughly* right, never untraceable, but won't survive a count-the-tables pass. They're 7 of the 34 yellow line items.

### 2. **I correctly predicted the headline scientific findings; I missed structural-process issues.**

Right:
- Disorder gap (predicted ~5 %, actual 4–5 %) — bit-perfect.
- Exp 50 capacity ceiling (predicted 69 %, actual 69 %) — bit-perfect.
- ABTT removal motivation (predicted Exp 45 PC1 73 % aligned, actual 73 %) — bit-perfect.
- Multi-PLM retention bands (predicted 95–100 %, actual 94.8–102.6 %) — accurate.
- VESM baseline gap, VEP/ProteinGym gap — correctly predicted as honest gaps.

Missed:
- README drift severity (prior implied 1 RED; audit found drift generates 5 distinct numeric reds plus the narrative).
- Commit-message hygiene issue (the binary default landed in commit `34e159a` titled "chore: gitignore...").
- 5 hidden defaults in `OneEmbeddingCodec` beyond the 6 named in the constructor (n_pcs, max_residues, version, etc.).
- Heinzinger-specific probe vector (ProstT5+Foldseek comparison) — added Q22 in EXPECTED_QA.
- TM-score 57.4 % retention silently omitted from CLAUDE.md (now disclosed).

### 3. **The deliverables themselves grew larger than predicted.**

Prior implicitly assumed each doc would be 1–2 KB. Actual sizes:
- AUDIT_FINDINGS.md: 357 lines (~14 KB).
- STATE_OF_THE_PROJECT.md: 288 insertions in the real version.
- MANUSCRIPT_SKELETON.md: 180 insertions.
- README.md: 181 lines (DOWN from legacy 444 — predicted "rewrite" was right).
- HANDOFF.md: 199 insertions.
- EXPECTED_QA.md: 240 insertions, 22 questions vs predicted 12.
- CALIBRATION.md: this doc (~200 lines).

This isn't a problem per se — the audit surfaced more than predicted, and the docs scaled. But it does suggest the **next iteration of the calibration loop should predict word counts**, not just topic coverage.

---

## Per-doc calibration

### 1. `docs/AUDIT_FINDINGS.md`

- **Prior commit:** `532146c` (2026-04-26).
- **Final at:** `db3a0fc` (Phase C close, 2026-04-27) + later D.X progress section at `8484870`.
- **Right:** Distribution prediction (70/20/10) was approximately correct on REDs (10 % actual), under-shot on yellows.
- **Wrong:** I imagined the audit would produce a small number of bullet findings; it produced a 45-row triage table covering every distinct line item across 9 audit tracks.
- **Material change:** The Posterior section grew from "(empty)" to a 5-section per-track posterior + cumulative table + 45-row triage table + Phase D progress section + GREENs consolidated footnote. The audit logs (`docs/_audit/logs/01–18`) are entirely new — they didn't exist as a concept in the prior.
- **Why the prior was wrong:** I predicted a single doc; the audit naturally produced an evidence-doc-per-track structure (`docs/_audit/{hygiene,splits,stats,params,claims,tooling,deps}.md`) that I hadn't anticipated would emerge. The *shape* of audit findings (3-tier severity, traceability columns) emerged from doing the work, not from the prior plan.

### 2. `docs/STATE_OF_THE_PROJECT.md`

- **Prior commit:** `1658eed`.
- **Final at:** `b83e2de`.
- **Right:** The one-paragraph executive summary in the prior was 90 % accurate — the disorder gap, the Exp 50 ceiling, the VESM baseline gap, the codec configuration. The "what works" and "what doesn't" lists transferred almost verbatim into the Executive Summary section.
- **Wrong:** I underestimated how much project-arc context belonged in the doc. The prior had no "how we got here" section; the final has 4 paragraphs walking through the design decisions chronologically. This was a Rost-lab-context realization (an audience that's seen the field needs to know *why* we ended up at center+RP+binary, not just that we did).
- **Material change:** Added: project arc, defaults-with-evidence table (the 6-knob inventory), receiver-side decode subsection, multi-PLM and Exp 50 detail sections, negative results enumeration.
- **Why the prior was wrong:** I treated "write-up" as a summary doc; the audit-driven version naturally became a justification doc. Rost-lab framing forced more "evidence per choice" structure.

### 3. `docs/MANUSCRIPT_SKELETON.md`

- **Prior commit:** `89f3908`.
- **Final at:** `2b20bfd`.
- **Right:** The figure list survived ~85 % (Pareto, multi-PLM heatmap, Exp 50 ceiling, ABTT artefact, phylogenetics — kept; lost: a "codec sweep" duplicate of Pareto). The three title candidates survived 100 %. The honest weak spots in §6 came directly from the prior.
- **Wrong:** The abstract was reworded substantially (predicted), but the prior abstract was actually *more* accurate than I expected — only sentence 2 needed updating ("232" → "200+"). The real change was adding §3.2 Datasets table, §3.4 Splits, §5.3 "what we are not claiming" — sections the prior didn't anticipate.
- **Material change:** Added: explicit dataset table, explicit splits section, explicit "what we are not claiming" subsection, reproducibility statement with one-command repro line.
- **Why the prior was wrong:** I predicted "abstract gets reworded substantially" — actually the abstract is one of the most-stable parts. The instability was in *Methods* (specifically: explicit dataset table, splits, fairness clauses). I undercounted the Rost-lab demand for explicit dataset enumeration.

### 4. `docs/_priors/README_REWRITE_PRIOR.md` → `README.md` (rewrite)

- **Prior commit:** `9a815e4`.
- **Final at:** `6ee41b3`.
- **Right:** The predicted section list (title + 1-line pitch, headline numbers, quick start, pipeline, multi-PLM table, codec sweep, what's not solved, pointers, citation) is what the new README has, in exactly that order.
- **Wrong:** I predicted the rewrite would *grow* the README (more sections, more depth). Actual: README *shrank* from 444 to 181 lines. The reason: by deleting the legacy V2/extreme-codec/trained-codec sections (which were pre-rigorous), the README became more focused.
- **Material change:** Decoder snippet inline (not in prior). At-a-glance fact box at top (not in prior). Acknowledgements section explicitly thanking the upstream PLM groups including Rost lab (not in prior).
- **Why the prior was wrong:** I assumed "rewrite" meant "rewrite + extend." Actually "rewrite" meant "rewrite to current reality + drop legacy." The new README is shorter because the audit revealed which sections were stale enough to delete entirely.

### 5. `docs/HANDOFF.md`

- **Prior commit:** `93825bc`.
- **Final at:** `0fc7e1e`.
- **Right:** The 5-step shape (setup → data → encode → decode → reproduce) survived. The "common gotchas" prediction (MPS float32, svdvals, clip_grad_norm + NaN) was 100 % accurate.
- **Wrong:** I underestimated how much "where does the data live" detail was needed. The prior had one sentence; the final has a per-path table with size + tracked-status columns.
- **Material change:** Added: § 2 data-locations table, § 4 per-quantization-mode decode notes, § 8 "if something is broken" section anchoring the canonical baseline (813/813 + deptry clean + git status empty).
- **Why the prior was wrong:** "15-minute onboarding" was a budget, not a content size. The audit revealed how many concrete file paths a newcomer needs upfront — far more than I'd guessed.

### 6. `docs/EXPECTED_QA.md`

- **Prior commit:** `52f7fde`.
- **Final at:** `bac545a`.
- **Right:** **8 of 12 prior questions transferred verbatim** to the final 22-question list (Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8 = ABTT, disorder floor, binary-vs-PQ, ANKH, splits, DCT K, VESM, Exp 50). My prior prediction "I will have 8 of 12 right" landed within rounding distance.
- **Wrong:** The 4 added by the audit (Q12 "232 methods", Q13 "6 vs 4 tasks", Q14 "L=175 reference", Q15 "1500 prot/s provenance") are exactly the *marketing-layer* questions I underestimated globally. Plus 6 more from audit specifics (Q16 TM-score, Q17 phylo trivial-config, Q18 README drift, Q19 commit-message hygiene, Q20 d_out=896 interpolated, Q21 bootstrap details, Q22 Heinzinger-Foldseek).
- **Material change:** Added: a "Questions we cannot fully answer" section (5 honest gaps) — entirely new.
- **Why the prior was wrong:** I predicted my blind spots correctly in shape ("4 questions I didn't anticipate at all") but missed the specific category — they were all process / marketing-layer, not scientific. Scientific predictions were strong; structural predictions were weak.

### 7. `docs/CALIBRATION.md` (this doc)

- **Prior commit:** `7cc2e72`.
- **Final at:** (this commit).
- **Right:** Predicted "predicted dimensions of mis-calibration" — the 70/20/10 audit prior, the README package-location uncertainty, the EXPECTED_QA "4 wrong of 12" prediction, the STATE_OF_THE_PROJECT "1–2 items I hadn't thought of." All four predicted dimensions of mis-calibration *did* turn out to matter; I was approximately right about the kinds of error.
- **Wrong:** I predicted "8 of 12 EXPECTED_QA questions roughly right" — actual is 8 of 12 (right), but I also added 10 more questions from the audit, taking the count to 22. The prior framed this as "4 wrong"; the reality is "8 right + 10 more I should have anticipated." Two different framings, same underlying reality.
- **Material change:** The "Global calibration" section at the top is new — the prior treated calibration per-doc; the posterior reveals systemic patterns (marketing-layer blindness, scientific accuracy).
- **Why the prior was wrong:** I treated calibration as a per-item exercise; doing it revealed it's a *categorical* exercise. The marketing-layer-blindness pattern shows up in 4 of the 7 docs (AUDIT, README, EXPECTED_QA, this doc). That's a calibration insight the prior couldn't have predicted because it required doing all 6 docs first.

---

## Calibration scorecard (qualitative)

| Doc | Prior accuracy | Notes |
|------|:------:|------|
| AUDIT_FINDINGS | **strong on shape, weak on scale** | Distribution prediction ~right; size of evidence trail underestimated 5×. |
| STATE_OF_THE_PROJECT | **strong on content, weak on structure** | Executive summary accurate; missed need for project-arc and per-knob-evidence sections. |
| MANUSCRIPT_SKELETON | **strong on title/figures, weak on Methods** | Figure list 85 % accurate; underestimated dataset-table demand. |
| README rewrite | **strong on sections, wrong direction on size** | Predicted growth; actual was shrinkage (delete legacy). |
| HANDOFF | **strong on shape, weak on data-detail** | Common gotchas perfectly predicted; data-locations table needed. |
| EXPECTED_QA | **strong scientific, weak process** | 8/12 prior questions transferred; missed marketing-layer + Heinzinger-specific. |
| CALIBRATION | **right kinds of error, wrong granularity** | Predicted *kinds* of mis-calibration accurately; missed that the systemic patterns matter more than per-doc deltas. |

**Overall systemic finding:** Scientific predictions were strong (disorder gap %, Exp 50 ceiling %, ABTT alignment %, retention bands). Structural and process predictions were weak (audit doc structure, README direction, marketing-layer count claims, commit-message hygiene). For future calibration loops in this project: **predict concretely, not just qualitatively** — and especially **predict the structural shape** of the deliverable, not just its content.

---

## Appendix: Prior (written 2026-04-26)

> Preserved as the `stub(prior): CALIBRATION.md` commit content.

### Predicted dimensions of mis-calibration
- **Audit reds**: predict 70/20/10 G/Y/R. Real distribution likely closer to 60/25/15 — I usually under-estimate edge-case rot in older code.
- **README**: predicted 70 % overlap between prior outline and final. The biggest unknown is the actual location of the package (`one_embedding/` vs `src/one_embedding/`).
- **EXPECTED_QA**: predicted I will have 8 of 12 questions roughly right. The 4 wrong ones will be questions I didn't anticipate at all (epistemic blind spots).
- **STATE_OF_THE_PROJECT**: predicted the open-problems section will gain 1–2 items I hadn't thought of when writing the stub.
- **Manuscript skeleton**: predicted figure list survives ~80 %; abstract gets reworded substantially.

### Final calibration (filled at Task I.3)

> Was: "(empty — six more paragraphs to write at I.3)". Now filled — see live doc above.
