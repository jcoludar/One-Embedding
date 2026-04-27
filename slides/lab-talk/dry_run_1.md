# Dry Run #1 — solo self-review of the deck (controller-side)

**Date:** 2026-04-27
**Reviewer:** Claude (controller, fresh-eyes pass over the just-built deck)
**Substitutes for:** the actual solo timed dry run, which the human user will perform with a stopwatch. This pass catches issues a fresh reader would notice; the human run measures actual timing and verbal-flow issues.

---

## Per-slide notes (14 slides, target 15 min)

| Slide | Topic | Estimated time | Issues / fixes |
|------|-------|:--:|----|
| 1 | Title | 0:30 | none |
| 2 | The problem (storage scale) | 0:45 | Numbers (5 GB / 6 GB / etc.) are good anchors. Add: "compressed: 17 KB/protein → SCOPe-5K = 100 MB" so the upside is concrete? — defer to dry run 2 if Q&A asks. |
| 3 | Method (4-knob codec) | 1:00 | Strong slide. Pipeline arrow reads cleanly. Knobs table is dense — speaker should narrate by row, not all at once. |
| 4 | Statistics protocol | 1:00 | Six bullets is dense. **Suggest:** speaker says "I'll go through the headline stats first, then come back to the protocol if anyone wants details" — gives Q&A a hook. |
| 5 | Headline tier table | 1:15 | Six rows × four columns may be hard to read at the back. Bold rows already help; speaker should explicitly say "binary at 37× and PQ M=224 at 18×" while pointing. |
| 6 | Pareto figure | 1:00 | Figure stands on its own. Good. |
| 7 | Multi-PLM heatmap | 1:00 | Heatmap reads well. ANKH disorder 94.8 % cell is the obvious probe; speaker should pre-empt it. |
| 8 | Receiver-side decode snippet | 1:15 | Code may be small at the back. Mitigation: speaker says "this is the entire receiver path — twelve lines of NumPy. PQ would need the codebook; binary doesn't." Don't read the code line-by-line. |
| 9 | What didn't work | 1:00 | Honest negative-results slide. Rost lab will respect this. |
| 10 | Exp 50 setup | 1:00 | The "honest caveat" line about random splits is critical — read it explicitly. Don't gloss it. |
| 11 | Exp 50 ceiling figure | 1:30 | The 69 % red-dashed ceiling line is the punch. Speaker says "three different stages, three different conditions, all converge to one number — that's a capacity ceiling." |
| 12 | What's not solved | 1:30 | Four problems × ~22 sec each. Tight. May need to drop #4 (VEP/ProteinGym) on a "we'll skip the obvious one" basis to recover budget. |
| 13 | Roadmap | 2:00 | Three time-buckets + earmarks. Likely runs over. **Compression option:** drop "Multi-teacher distillation" + "Open earmarks" line if running tight. |
| 14 | Summary + asks | 1:00 | Strong close. The "Asks" line is the lab-talk virtue — invites collaboration. |

**Total estimated:** ~14:45 (within the 15-min slot).

---

## Top 3 fixes to consider before run #2 (with stopwatch)

1. **Slide 12 (what's not solved) is tight at four bullets.** If the timed run shows it overruns, drop the VEP / ProteinGym bullet — it's the least directly defended and can be picked up in Q&A if Rost asks "what about VEP?".

2. **Slide 8 (decoder snippet) needs a verbal anchor.** The code block is small. Before showing it, the speaker should say one sentence: "this is the load-bearing claim — the receiver decodes with NumPy and h5py, twelve lines." Then let the audience read it for ~10 seconds before moving on. Otherwise it'll feel like a wall.

3. **Slide 4 (statistics) is six bullets.** Suggest collapsing to 4 with parentheticals:
   - BCa B=10 000 (paired for retention; cluster for disorder, pooled ρ)
   - Probes CV-tuned with `random_state=42`
   - Multi-seed averaged before bootstrap (Bouthillier 2021)
   - Same `metrics.statistics` module across Exp 43–47 — no shadow implementations
   This compresses to ~45 sec without losing rigor signaling.

---

## Things I cannot evaluate without running it

- Actual timing per slide (I'm estimating)
- Whether the Marp PDF renders the heatmap colors correctly on the projection setup
- Whether the speaker's verbal flow matches the slide order naturally
- Whether the font sizes are readable at the back of the room

The user's actual timed dry run (slides/lab-talk/dry_run_2.md) will catch these.

---

## Sanity-check items the speaker should verify before the talk

- [ ] Open `slides/lab-talk/talk.pdf` in the projector setup — confirm 14 slides, all figures render, no text cropping.
- [ ] The 3 figures (`pareto.png`, `multi_plm_heatmap.png`, `exp50_ceiling.png`) are embedded into the PDF (not linked) — check by opening the PDF on a machine without `slides/lab-talk/figures/`.
- [ ] Test the laser pointer / clicker workflow.
- [ ] Have `docs/EXPECTED_QA.md` open on a second screen (or printed) for the Q&A.
- [ ] Pre-glance at `docs/AUDIT_FINDINGS.md` Phase D progress — know the resolved/deferred statuses for any audit-related question.
