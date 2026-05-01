# Exp 56 — VEP codec mega-sweep (ABTT, dimensionality, alt quantization)

<!-- Figures inserted after the run completes -->
![Retention overview](figures/exp56_retention_overview.png)

## TL;DR

<!-- Filled in after the full sweep lands. Smoke-test sneak preview:
     binary_896_abtt3 = 99.0% ± 0.8%, int2_896 = 99.9% ± 0.8%.
     If those numbers hold across all 15 assays in the full run we have
     the headline: ABTT is essentially free for VEP (≠ disorder), and
     int2 is a viable mid-tier between binary and int4. -->

## Why we ran this

Exp 55 measured 5 standard codec tiers — lossless, fp16, int4, PQ M=224, binary — and found ≥99.2% retention across the board on ProteinGym DMS + ClinVar. Two threads of the surrounding evidence, however, were left hanging:

1. **ABTT** (top-PC removal) is the project default's odd one out — it's set to `abtt_k=0` because Exp 45 showed `abtt_k=3` destroys disorder. We never tested whether the same is true for VEP. If ABTT is mostly noise removal at the protein-level (where the disorder signal happens to live in PC1), it might be neutral or even helpful for variant-level signals; if VEP shares disorder's PC1 sensitivity, it would lose 1–5pp like disorder did.

2. **Compression aggressiveness** has been mapped at four points (binary 37×, PQ M=224 18×, int4 9×, fp16 2.3×) but with gaps that matter for the codec's "Pareto" claim. Specifically: how does VEP behave at PQ M=64 (64×), at int2 (18× without a codebook), at binary 1024d (no RP), at binary 512d (64× via RP)? The Exp 47 disorder breakdown started fraying near 32×, but that was disorder; VEP's 99.2% binary retention left us suspecting the front-line task tolerates aggressive codecs better than disorder does.

3. **`binary_magnitude`** (PolarQuant, Exp 51) was rejected for disorder. Inclusion here is a single-arm sanity check — if the rejection turns out to be task-specific, VEP retention will tell us.

The combinatorial space is large; this experiment runs a focused 12-arm sweep along three axes (ABTT-k, dimensionality, quantization variant) plus a re-run of `lossless` so each new arm has paired predictions for its retention CI.

## Design & data

**Datasets.** Identical to Exp 55. 15 ProteinGym DMS substitution assays (37,919 single-substitution variants, diversity-selected ≤2,000 aa) + ProteinGym ClinVar split (1,016 proteins ≤500 aa, 15,252 missense variants). Embeddings are reused from Exp 55's H5 caches via symlink — no PLM forward passes.

**PLM.** ProtT5-XL, 1024d per-residue.

**Codec arms (13 total).** Lossless 1024 (paired baseline) plus 12 new arms across three axes:

| Axis | Arms |
|---|---|
| **ABTT-k on binary 896** | binary_896_abtt1, binary_896_abtt3, binary_896_abtt8 |
| **Dimensionality on binary** | binary_1024 (no RP), binary_1024_abtt3, binary_512 (64×) |
| **Alt quantization (896d)** | binary_magnitude_896, pq128_896 (32×), pq64_896 (64×), int2_896 (18×) |
| **ABTT × quantization (896d)** | fp16_896_abtt3, int4_896_abtt3 |

The Exp 55 arms (binary_896, fp16_896, int4_896, pq224_896) are not re-run — their retention numbers are already published. They serve as reference points in the cross-experiment table at the end.

**Probe.** Identical to Exp 55: per-variant 4·d_out feature (concat WT_emb[mut_pos], mut_emb[mut_pos], mean(WT), mean(mut)). Ridge regression per assay, 5-fold outer CV, inner 3-fold GridSearch over α, predictions averaged across seeds {42, 123, 456}.

**Statistical rigor.** BCa B=10,000 paired ratio-of-means bootstrap. Retention = compressed mean per-assay ρ / lossless mean per-assay ρ.

**Compute.** ~2h MPS-light wall (Ridge + bootstrap, no PLM forward). Memory bounded by Exp 55's streaming variant-loader.

## Results

<!-- Tables go here once the run completes:
       a) overview retention table (13 codecs, paired BCa CIs)
       b) ABTT-k axis with delta from Exp 55 binary_896 (99.2%)
       c) dimensionality breakdown
       d) quantization breakdown
       e) ClinVar AUC (zero-shot) per arm
-->

## Conclusions & outcomes

<!-- Filled in once results land. Predictions to falsify (from the design memo):

1. ABTT will hurt VEP by 1–5pp (mirrors disorder).
2. binary_1024 (no RP) beats binary_896 by ~0.5pp.
3. PQ M=64 retains ≥99% on VEP.
4. binary_magnitude does not help VEP.

Smoke (3 codecs, all 15 assays) already partially falsified #1 — binary+ABTT3
retained 99.0% (vs binary alone at 99.2%, well within CI overlap). Need full
run to call it.
-->

## Out of scope (and why)

- **Multi-PLM.** ProtT5 only here; multi-PLM VEP is a Phase-2 follow-up.
- **Indels.** Substitutions only.
- **Re-extraction.** Cached Exp 55 embeddings reused exactly.
- **RNS ride-along.** Skipped — Exp 55 documented mean-pool RNS as degenerate (per-residue shuffle preserves sums). Reviving requires a non-mean protein vector, out of scope here.
- **VESM head-to-head.** Earmarked separately.

## Links

- **Spec / design memo:** `memory/project_exp56_codec_megasweep_idea.md`
- **Plan:** this document
- **Results JSON:** `data/benchmarks/rigorous_v1/exp56_vep_codec_megasweep.json`
- **Run log:** `results/exp56_main_run.log` (gitignored)
- **Code:**
  - `experiments/56_vep_codec_megasweep.py` — runner
  - `experiments/56_make_figures.py` — figures
  - `src/one_embedding/codec_v2.py` — int2 wiring added in commit `221592e`
- **Figures:** `docs/figures/exp56_retention_overview.png`, `exp56_abtt_effect.png`, `exp56_axes_breakdown.png`
- **Related experiments:**
  - Exp 55 (same probe + bootstrap on 5 standard tiers) — direct extension
  - Exp 45 (disorder forensics — ABTT3 destroys disorder)
  - Exp 47 (codec sweep on SS3/SS8/disorder/retrieval — disorder gap mapped here)
  - Exp 48c (RNS knob sweep — PQ M=64 beats raw on RNS)
  - Exp 51 (PolarQuant rejected for disorder — `binary_magnitude_896` retests)
