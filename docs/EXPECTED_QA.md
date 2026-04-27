# Expected Q&A — Anticipated Probes for the Lab Seminar

**Date:** 2026-04-27
**Status:** Real version (Task G.2). Refined against audit findings.
**Audience:** Full Rost lab incl. Burkhard Rost and Michael Heinzinger. Format: 15 min talk + 10 min Q&A.

> Prior (2026-04-26 stub) preserved as Appendix for calibration. Compare via
> `git log --oneline -- docs/EXPECTED_QA.md`. Calibration delta in `docs/CALIBRATION.md`.

---

## Reading order

Q1–Q5: most likely / load-bearing answers — drill these in dry runs.
Q6–Q15: probable probes; have the one-sentence answer ready.
Q16–Q22: low-probability but high-stakes (Heinzinger-specific, methods-specific).

---

## Q1. Why is ABTT off by default?

**Answer.** Exp 45 (`docs/_audit/params.md:abtt_k=0`): the top principal component of ProtT5 embeddings is **~73 % aligned** with the direction from "folded" to "disordered" residues in CheZOD117. Removing it costs **6–11 pp of disorder retention** while gaining only marginally on retrieval. We default `abtt_k=0` and document the trade-off; users who need ABTT for retrieval-only pipelines can flip the knob.

**Evidence.** `experiments/45_disorder_helpers.py` ABTT-PC1 cosine analysis. CLAUDE.md L67 + STATE_OF_THE_PROJECT.md § "Project arc". This finding generalises across all 5 PLMs (Exp 46 forensics).

---

## Q2. Disorder retention is ~95 %. What's the baseline noise floor — is the 5 pp gap real?

**Answer.** Yes, the gap is geometrically real. Three lines of evidence:
1. The same gap reproduces across all 5 PLMs (94.8–98.8 %, Exp 46 — `data/benchmarks/rigorous_v1/exp46_multi_plm_results.json`), arguing it's not noise specific to one corpus.
2. The Exp 45 ABTT-PC1 finding (Q1) suggests a geometric mechanism: disorder lives in directions that quantization smears.
3. Multi-seed std is small (~0.4 pp on bit accuracy from Exp 29 part D), so 5 pp is ≈10× the floor.

We are also designing **Exp 51 (PolarQuant)** to attack this — magnitude-augmented binary expected to recover 2–3 pp at the same 36× compression with no codebook.

---

## Q3. Why binary as the default rather than PQ M=224?

**Answer.** Three reasons:
1. **No codebook.** Receiver decodes with `numpy + h5py` only (~12 lines). PQ requires shipping the codebook (~1 MB per PLM × per-config), which complicates the universal-codec story.
2. **~20× faster encoding** at no quality cost on retrieval (100.4 % both) and SS3/SS8 (binary 97.6 / 95.0 vs PQ M=224 99.0 / 98.5 — ~1.5 pp gap, well-bounded).
3. **Binary slightly beats PQ M=128 on disorder** (94.9 % vs 91.4 %) — RaBitQ-style theoretical effect (random projection + sign quantization preserves cosine).

For users who want max quality, **PQ M=224 (~18×)** is the recommended setting and is documented as such in CLAUDE.md and the README.

---

## Q4. ANKH disorder retention 94.8 % is the worst cell. What's special about ANKH?

**Answer.** ANKH's tokenizer has known subword-level artifacts that we observed earlier in Exp 46 (when extracting embeddings). The 94.8 % retention is consistent with that — i.e., the codec is preserving whatever signal is there, but ANKH itself encodes disorder slightly less cleanly than ProtT5/ProstT5. The cell is within 2–3 pp of the others, so it doesn't represent a codec failure.

**Honest follow-up:** we have not done a careful ANKH-vs-ProtT5 disorder benchmark on the *raw* embeddings (without our codec) to confirm. That's the right experiment if anyone asks.

---

## Q5. Cross-PLM split fairness — same train/test partition across all 5 PLMs?

**Answer.** Yes, verified file-path identity in `experiments/46_multi_plm_benchmark.py:456–461`. The split files (`cb513_80_20.json`, `esm2_650m_5k_split.json`) are loaded once per benchmark call and applied to whichever PLM's H5 file is being tested. The train/test ID list never depends on PLM identity.

Aside on the file naming — `esm2_650m_5k_split.json` is misleadingly named (historical artifact from Exp ~17) but is actually PLM-agnostic. Phase D rename was deferred (cosmetic, not on talk path).

---

## Q6. Why DCT K=4? Why not K=8 or attention pool?

**Answer.** This is an honest weak spot. `dct_k=4` is a **STORAGE choice**, not a measured quality optimum. Storage: K=4 → protein_vec = `D×4` fp16 = 7 KB at d_out=896; K=8 doubles that.

**The relevant raw evidence we have** is from Exp 22 (`path_geometry_results.json`), which on a *different* proxy (displacement-DCT) shows K=8 has higher Ret@1 than K=4 (0.712 vs 0.666). We have NOT done a formal K-sweep on the current preprocessed pipeline (center → RP896 → binary → DCT K).

Reframe: "K=4 is the compression-aware default; for retrieval-critical applications K=8 might be worth ~2× storage."

---

## Q7. No co-distilled VESM (Bromberg lab) baseline — why not?

**Answer.** Honest gap. VESM (2026 *Nat Methods*, MIT-licensed weights) is the strongest plausible competitor. Earmarked in `MEMORY.md` next-steps. The argument we'd make at a venue:
- The codec is **agnostic to the upstream PLM**. We validated on 5 different ones, including very different architectures (ProtT5 encoder-only T5 vs ANKH vs ESM-C). The same compression ratios and retention bands should hold for VESM.
- We have not run it because VESM came out late in this project's timeline.

If pressed: "We will run VESM as part of the manuscript revision; expect retention bands consistent with the other 5 PLMs."

---

## Q8. Exp 50 plateau at ~69 % — is that really a CNN ceiling, or just under-trained?

**Answer.** Capacity-bound, not under-trained. Three lines of evidence converge to the same number:
- **Stage 1**: 65–66 % bit accuracy (val), 65.4 % test. Plot in slide 11.
- **Stage 2**: hidden=256, layers=10, 4.2M params — converges to ~69 %.
- **Stage 3**: continuous loss + 2× data — same ~69 %.

Three different architectures × loss × data conditions all converging to one number is the signature of capacity exhaustion. **Stage 4 (transformer)** is the designed next lever — 3–5 days on M3 Max. (Spec: `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md`.)

**Caveat:** the current Exp 50 numbers use a random 80/10/10 split that leaks homology. The rigorous CATH-cluster re-run (H-split + T-split + MMseqs2 leakage audit + 3-seed variance) is designed but not yet executed. Talk frames Exp 50 as in-progress.

---

## Q9. How does retrieval stay near 100 % retention when per-residue tasks lose 5–8 %?

**Answer.** Two reasons:
1. **Different geometric demands.** Retrieval lives in cosine space at the protein level (after DCT pooling); cosines are preserved well by RP + sign quantization (RaBitQ result, Gao & Long 2024). Per-residue tasks need fine-grained per-residue directions that quantization smears.
2. **Baseline-stronger-predictor effect.** Some retrieval cells exceed 100 % (e.g., ESM-C at 102.6 %): this is the noisy-baseline-stronger-predictor effect — the compressed protein vector slightly outperforms the noisy raw cosine baseline because the random projection acts as a mild denoiser. Documented in §3.3 of the methods.

---

## Q10. What's the comparison vs storing FASTA + a small predictor at deployment?

**Answer.** Not directly benchmarked yet — but Exp 50 is the lower bound on this. If the CNN can predict 65–69 % of binary OE bits from sequence alone, then the binary OE is at most that-much redundant with sequence-only features. The remaining ~31 % bits encode information that requires PLM context.

If Stage 4 (transformer) lifts the ceiling to ~80 %, the FASTA-only deployment story becomes more compelling. Until then: storing the binary OE is the correct deployment choice for any task that needs >65 % of the embedding's signal.

---

## Q11. Splits — what cluster level for CATH? Are SCOPe and CATH leakage-controlled?

**Answer.**
- **CATH20**: clustered at 20 % sequence identity at the dataset level. We use CATH H (homologous superfamily) clusters as units; main split is H-split, stress test is T-split (topology). See Exp 50 design spec.
- **SCOPe 5K**: split is cluster-controlled; `superfamily_overlap=0`, `family_overlap=0` recorded in `data/benchmark_suite/splits/esm2_650m_5k_split.json` (`statistics` block).
- **CB513 / TS115 / CASP12**: published splits, `<25 %` identity by dataset construction.
- **Disorder (CheZOD117 / TriZOD348)**: cluster-curated by source publications.
- All splits asserted at runtime via `rules.check_no_leakage`.

---

## Q12. The talk says "200+ compression methods" — can you enumerate?

**Answer.** Roll-up estimate. Distinct contributions per experiment family:
- 14 codecs in Exp 25 / 26 (universal codec quest, chained codec)
- 50+ in Exp 28 / 29 (extreme compression, exhaustive fruit sweep)
- 6 in Exp 31 (bitwidth)
- 30+ in Exp 32 / 33 (PQ on RP, VQ codecs)
- 10+ in Exp 34 (V2 progressive)
- 6 in Exp 47 (final codec sweep)
- ~70 in earlier exploratory branches (now archived; see `docs/_audit/worktrees.md`)

If pressed: full enumeration in `docs/EXPERIMENTS.md`. We rounded "232" → "200+" in the audit (D.1 fix) because no script outputs "232" as a clean count.

---

## Q13. The talk says "6 tasks on 5 PLMs" — but Exp 46 looks like 4 tasks?

**Answer.** **You're right. The "6" rolled in CB513 / TS115 / CASP12 SS3 variants as separate tasks** (which is how they're often reported). The honest count is **4 task families × 9 datasets × 5 PLMs**. The CLAUDE.md and README were updated in audit D.1 to reflect this.

---

## Q14. The "L=175 reference" in the storage tier table — what's that about?

**Answer.** L=175 is a fixed reference length used to convert d_out × bytes-per-residue → KB-per-protein for the tier table. It is NOT the empirical mean — Exp 45 reports SCOPe-5K mean L=156. Storage figures scale ~10 % differently for the actual mean. Tagged as "(L=175 ref)" in CLAUDE.md after audit D.1.

---

## Q15. The "1500 proteins/s encoding" speed claim — what hardware, what config?

**Answer.** M3 Max, binary mode, single-process. Originally claimed in commit message `8b1fbf1`; not in any committed result JSON. Caveated as "~1500 prot/s on M3 Max — see Exp 47 logs" after audit D.1. PQ encode is ~75 prot/s same hardware (codebook fit dominates), giving the "~20× faster" claim.

---

## Q16. The Exp 37 lDDT 100.7 % / contact precision 106.5 % — those don't have BCa CIs?

**Answer.** Correct. Exp 37 predates the rigorous framework (Exp 43). Tagged as "pre-rigorous, no BCa CIs" in CLAUDE.md after audit D.1. The companion **TM-score Spearman 57.4 %** retention from the same JSON was previously omitted from CLAUDE.md and is now disclosed (also D.1).

If pressed on TM-score: "lDDT and contact precision are local-structure metrics; TM-score is a global topology metric. The codec preserves local geometry better than global topology — consistent with the per-residue vs per-protein gap pattern we see elsewhere."

---

## Q17. The phylo file `embedding_phylo_results.json` n_taxa=156 — is that the rigorous run?

**Answer.** No, it's a sanity-config artefact (20K generations × 1 run × 2 chains; ASDSF=0.0 indicates insufficient sampling). The rigorous-config defaults are 200K × 4 chains × 2 runs. The cited "10–11 of 12 families monophyletic" claim comes from per-tree `_consensus.nwk` files in `results/embed_phylo/`, NOT from this JSON. Audit-traced in `docs/_audit/hygiene.md` (C.5 phylo decision).

The `BENCH_PATH` in `experiments/35_embedding_phylogenetics.py:2086` is hardcoded — every invocation overwrites the file. Phase D fix (parameterize by `--dataset`) was deferred post-talk.

---

## Q18. README "44 experiments" but CLAUDE.md says 47 — which is right?

**Answer.** 47. README was severely out-of-date; full rewrite landed in audit E.3 (commit `6ee41b3`). The README is now consistent with CLAUDE.md.

---

## Q19. `git blame` on `OneEmbeddingCodec` defaults shows commit `34e159a` titled "chore: gitignore..." — that doesn't sound like a defaults change?

**Answer.** Commit-message hygiene issue. The binary/896d/abtt_k=0/auto-pq defaults DID land in `34e159a`, even though the subject line says "gitignore". The rationale is in the sibling commit `8b1fbf1` ("Exp 45-47 — disorder forensics, 5-PLM pipeline, binary default"), which doesn't touch `codec_v2.py` itself. Honest answer: a commit-discipline lapse during the v1 ship, not a correctness issue. Documented in `docs/_audit/params.md`.

---

## Q20. Why d_out=896 specifically? Why not 512, 768, or 1024?

**Answer.** Three reasons:
1. **PQ-divisibility.** 896 = 7 × 128 = 4 × 224, divides cleanly for PQ subquantizers.
2. **Avoids the Exp 47 "ABTT3 + RP768" ProstT5 catastrophe** (SS3 dropped to 85.6 % at that combo).
3. **Direct test in Exp 47.** PQ M=224 at d_out=896 is the recommended max-quality setting.

**Honest caveat:** there is no clean d_out ∈ {512, 768, 896, 1024} factorial sweep at fixed quantization in our results. The 896 choice is justified above but not measured against exhaustive alternatives. Tagged YELLOW in `docs/_audit/params.md`.

---

## Q21. Bootstrap is BCa? Multi-seed is averaged before or after bootstrap?

**Answer.** Yes BCa (DiCiccio & Efron 1996), B=10000, percentile fallback for n<25. Implemented via `scipy.stats.bootstrap(method="BCa")` for plain & paired; manual BCa with jackknife acceleration for cluster bootstrap (`experiments/43_rigorous_benchmark/metrics/statistics.py`).

**Multi-seed** averaging happens BEFORE bootstrap (Bouthillier et al. 2021 — averaging predictions across seeds, then bootstrapping the averaged predictions, NOT averaging CIs across seeds). Audit-verified for Exp 43, 44, 46, 47.

**Disorder** uses cluster bootstrap (resample proteins, recompute pooled Spearman ρ on residue union — Davison & Hinkley 1997). **Retention** uses paired bootstrap (same protein-id resample drives raw and compressed in every iteration).

---

## Q22. (Heinzinger-specific) Why didn't you compare against ProstT5 + Foldseek directly?

**Answer.** ProstT5 is one of the 5 PLMs we benchmark — Table 2 shows ProstT5 retention bands (99.2 / 98.6 / 100.0 / 98.3 — best disorder retention of the 5). For Foldseek + 3Di tokens specifically: Exp 52 (designed, not run) is to add a 3Di multi-task head on top of the codec, which would integrate exactly the ProstT5 + Foldseek philosophy with sequence-only inference.

---

## Questions we cannot fully answer

These are honestly disclosed gaps:

- **VEP / ProteinGym performance** (variant effect prediction) — not benchmarked. Earmarked.
- **CUDA wall-time at scale** — host validated on M3 Max; CUDA path tested in fixtures but not benchmarked at UniProt scale.
- **Direct comparison vs ESM2-quantized retrieval indices** (Meta's published quant scheme) — not done; would require their exact protocol replication.
- **`.one.h5` schema versioning** — currently `version=4` is hardcoded; no documented upgrade path for older files. Phase D YELLOW deferred post-talk.
- **FastTree / IQ-TREE per-family monophyly numbers** in README L138 — not explicitly traced to a script in the audit (likely correct, but the trace is implicit). Disclose if probed.

---

## Appendix: Prior (written before audit, 2026-04-26)

> Preserved as the `stub(prior): EXPECTED_QA.md` commit content for
> calibration. Compare against the live doc above; the
> prior-vs-posterior delta is summarised in `docs/CALIBRATION.md`.

Numbered by predicted severity (1 = most likely / hardest).

1. **Why was ABTT removed by default?** → Q1 above (kept).
2. **Disorder retention 94.9 % — what's the baseline noise floor?** → Q2 (kept, sharper answer).
3. **Why binary as the default rather than PQ?** → Q3 (kept).
4. **ANKH disorder 94.8 % — what's special about ANKH?** → Q4 (kept).
5. **Cross-PLM split fairness?** → Q5 (kept; verified in audit).
6. **Why DCT K=4?** → Q6 (kept; honest weak spot acknowledged).
7. **Co-distilled VESM as baseline?** → Q7 (kept).
8. **Exp 50 plateau at 69 %?** → Q8 (kept).
9. **How does retrieval stay at 100 % when per-residue tasks lose 5–8 %?** → Q9 (kept).
10. **What's the comparison vs FASTA + a small predictor?** → Q10 (kept).
11. **CATH split — H, T, or A?** → Q11 (expanded to splits-overall).
12. **Why not also report median ± IQR?** → answer in audit (BCa already covers asymmetry); demoted from priority list.

**Newly added by audit (not in prior):** Q12 (232 enumeration), Q13 (6 vs 4 tasks), Q14 (L=175 reference), Q15 (1500 prot/s provenance), Q16 (TM-score 57.4 %), Q17 (phylo trivial config), Q18 (README 44 vs 47), Q19 (commit-message hygiene), Q20 (d_out=896 interpolated), Q21 (bootstrap details), Q22 (Heinzinger-specific Foldseek), plus the "we cannot fully answer" section.
