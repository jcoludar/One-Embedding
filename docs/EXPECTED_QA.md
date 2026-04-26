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
