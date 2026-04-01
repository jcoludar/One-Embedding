# Disorder Information Loss Investigation

**Date:** 2026-03-30
**Goal:** Identify exactly where and why the codec loses disorder prediction signal (93-95% retention vs 97-100% for all other tasks), and discover actionable improvements.

## Context

The unified codec (ABTT3 + RP 768d + PQ M=192) retains 97-100% of signal for SS3, SS8, retrieval, and localization — but disorder pooled Spearman rho drops to 93-95%. AUC-ROC (binary disorder/order) retains 98.6%, suggesting the codec preserves the *direction* of disorder signal but loses *magnitude resolution*. Prior analysis (Exp 38) found RP causes ~80% of the loss. Ridge probe is deterministic (multi-seed is a no-op), while CNN probes recover to 99%.

## Investigation Structure

Five layers, each building on findings from the previous. The notebook can stop at any layer once actionable insights emerge.

### Layer 1: Per-Protein, Per-Stage Rho Decomposition

Run the full pipeline on CheZOD 117 test proteins AND TriZOD 348 test proteins. For each protein, compute Spearman rho at every codec stage:

| Stage | Embeddings | How |
|-------|-----------|-----|
| Raw 1024d | original | Train Ridge on raw, predict on raw |
| + ABTT3 | centered + top-3 PCs removed | Train Ridge on ABTT'd, predict on ABTT'd |
| + RP 768d | projected to 768d | Train Ridge on projected, predict on projected |
| + PQ M=192 | quantized then decoded | Train Ridge on decoded, predict on decoded |

Output: `{protein_id: {stage: rho}}` for all proteins in both datasets. Compute per-stage delta: `rho_drop_abtt`, `rho_drop_rp`, `rho_drop_pq` for each protein.

Aggregates: pooled rho at each stage (headline metric), plus per-protein rho distributions.

### Layer 2: Stratified Protein Selection

From Layer 1 results, select proteins for deep analysis:

**From CheZOD 117:**
- 2-3 **average** proteins (rho drop near median)
- 2-3 **worst losers** (largest total rho drop, raw → decoded)
- 2-3 **surprising winners** (rho improves or stays flat after compression)

**From TriZOD 348:**
- Same stratification (2-3 average, 2-3 worst, 2-3 winners)

For each selected protein, characterize:
- Sequence length
- Fraction disordered (Z < 8) and fraction ordered
- Z-score distribution shape (bimodal ordered/disordered? uniform? all-ordered?)
- Which pipeline stage causes the most damage for THIS protein
- Cross-dataset check: if a protein appears in both datasets, do results agree?

Look for patterns: do losers share properties (short? mostly ordered? boundary-heavy?)?

### Layer 3: Subspace-Level Forensics

For the selected proteins, analyze which *directions in embedding space* carry disorder signal:

1. **Disorder-informative subspace:** Across all training residues, compute correlation between each of the 1024 raw dimensions and Z-scores. Identify the top-K dimensions (by |correlation|) that are most predictive of disorder. This defines a "disorder subspace."

2. **ABTT overlap:** Do the top-3 PCs removed by ABTT overlap with the disorder-informative subspace? Compute cosine similarity between ABTT PCs and the disorder-correlated directions. If ABTT removes disorder signal, this is actionable.

3. **RP preservation:** After random projection to 768d, how much of the disorder subspace variance is preserved? Compare the projection of disorder-informative dimensions before and after RP. Use explained variance ratio or reconstruction error on the disorder subspace specifically (not overall).

4. **Per-protein subspace analysis:** For the worst-loss proteins, check if their disorder-informative directions differ from the corpus average. If some proteins use "unusual" channels for disorder, RP (which is corpus-agnostic) might destroy protein-specific signal.

### Layer 4: Residue-Level Error Maps

For each selected protein (from Layer 2), generate diagnostic plots:

1. **Residue trace:** x-axis = residue position, overlaid lines for:
   - True Z-score
   - Raw-probe predicted Z-score
   - Compressed-probe predicted Z-score
   - Highlight regions where raw gets it right but compressed doesn't (error > threshold)

2. **Error heatmap:** Residue position vs pipeline stage, colored by |prediction error|. Shows if errors concentrate at IDR boundaries, loops, or specific secondary structure elements.

3. **Embedding distance:** For the worst residues, compute L2 distance between raw and compressed embedding vectors. Correlate embedding distortion with prediction error. If high-distortion residues are also high-error residues, the codec is directly responsible.

### Layer 5: Targeted Recovery Experiments

Based on findings from Layers 1-4, test fixes. Which experiments to run depends on what we find, but candidates include:

- **If ABTT kills disorder signal:** Test ABTT k=1, k=2 instead of k=3. Compare disorder rho at each k. If k=2 recovers disorder without hurting retrieval, that's a free win.
- **If RP dimensionality is the bottleneck:** Test d_out = 896, 832 (between 768 and 1024). Plot disorder rho vs d_out to find the knee.
- **If specific subspace matters:** Test weighted RP that gives higher weight to disorder-informative dimensions. Or try PCA-based projection (ordered by variance) instead of random projection, keeping the top-768 PCA components.
- **If magnitude resolution matters (AUC-ROC >> rho):** The problem is quantization granularity in the disorder-relevant range. Test fp16 at 768d (no PQ) — if rho recovers, PQ is the culprit despite the "RP causes 80%" finding being about an older pipeline.

## Output Format

**Phase 1 (notebook):** `experiments/45_disorder_forensics.ipynb`
- Interactive, visual, exploratory
- All plots inline, markdown annotations
- Can stop at any layer

**Phase 2 (script):** `experiments/45_disorder_forensics.py`
- Reproducible, parameterized by PLM name
- Runs on ProtT5, ESM2-650M, ESM-C 650M
- Outputs JSON results + diagnostic plots to `results/disorder_forensics/`

## Success Criteria

1. We can name the specific pipeline stage(s) responsible for disorder loss, with per-protein evidence
2. We understand whether the loss is uniform or concentrated in specific protein types
3. We have at least one actionable hypothesis for improving disorder retention
4. The analysis is reproducible across PLMs (once ESM2/ESM-C embeddings are extracted)

## Non-Goals

- Attention head analysis (premature — dimensions don't map to heads after projection layers)
- Training new disorder-specific probes (that's a separate investigation)
- Changing the default codec parameters (any improvements would be validated in Exp 44 framework first)
