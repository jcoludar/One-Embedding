# Exp 51 — PolarQuant (binary + per-residue magnitude)

**Date:** 2026-04-30
**Status:** Negative result — hypothesis rejected.
**Question:** does adding one fp16 magnitude scalar per residue on top of
the binary OE codec recover the disorder-retention gap (94.9 % on binary
vs 98.2 % on int4)?
**Answer:** no. Hybrid PolarQuant gives 94.4 ± 1.9 % disorder retention vs
binary's 94.9 ± 1.8 % — CIs overlap, effectively a wash (-0.5 pp, well
within seed noise).

**Sources.**
- `experiments/51_polarquant_eval.py` — runner (re-uses Exp 47 sweep machinery)
- `data/benchmarks/rigorous_v1/exp51_polarquant.json` — full results (BCa CIs)
- `src/one_embedding/quantization.py:quantize_binary_magnitude` /
  `dequantize_binary_magnitude` — the implementation
- `tests/test_codec_unified.py` — 3 unit tests (round-trip, per-residue norm
  recovery, H5 round-trip), all green

## Setup

Four configs evaluated on ProtT5-XL with the Exp 47 protocol (BCa CIs,
paired bootstrap retention, CV-tuned probes, 3-seed averaging):

| Config | Compression | Description |
|---|:---:|---|
| binary 896d              | 37× | baseline (current default) |
| **binary_magnitude 896d** | 36× | **EXP 51: signs + per-channel scales/means + per-residue fp16 magnitude** |
| int4 896d                | 9×  | upper reference |
| PQ M=224 896d            | 18× | quality leader |

## Results

| Config | SS3 ret | SS8 ret | Ret@1 ret | **Dis ret** |
|---|:---:|:---:|:---:|:---:|
| binary 896d              | 97.6 ± 0.6 % | 95.0 ± 0.7 % | 100.4 ± 1.0 % | **94.9 ± 1.8 %** |
| **binary_magnitude 896d** | 97.6 ± 0.6 % | 95.4 ± 0.7 % | 100.4 ± 1.0 % | **94.4 ± 1.9 %** |
| int4 896d                | 99.8 ± 0.4 % | 98.8 ± 0.4 % | 100.4 ± 0.6 % | 98.2 ± 1.2 % |
| PQ M=224 896d            | 99.0 ± 0.5 % | 98.5 ± 0.6 % | 100.6 ± 0.6 % | 95.4 ± 1.9 % |

Per-task vs binary baseline:

- **Disorder**: −0.5 pp, CIs heavily overlap. Not significant.
- SS3: identical to 0.1 pp.
- SS8: +0.4 pp, within seed noise.
- Retrieval: identical (cosine is magnitude-invariant — expected).

## Implementation

Two implementations were attempted:

1. **Pure PolarQuant** (matches the original design memo `b · m · k`): drop
   per-channel scales and means, replace with one global LS-optimal
   reconstruction constant `k = sqrt(2 / (π·D))`. **Result: regressed
   disorder by 5.7 pp** (94.9 → 89.2). Reason: dropped 1 792 floats of
   per-channel calibration to gain only L floats of per-residue magnitude;
   for typical L=156 we lost much more than we gained.
2. **Hybrid PolarQuant** (the version in `quantize_binary_magnitude` now):
   strict superset of binary's information — keeps signs, per-channel
   scales, and per-channel means, then adds per-residue magnitudes as a
   multiplicative rescaling at decode. By construction can only equal or
   exceed binary. **Result: equal to binary within noise.**

The hybrid result is the load-bearing one: even when the codec keeps
*everything* binary already had AND adds a per-residue magnitude scalar,
the magnitude doesn't carry useful disorder signal that the linear probe
couldn't already extract from the signs.

## Why the hypothesis was wrong

The design memo's prediction — "disorder is a regression target, magnitude
carries genuine signal — high-norm regions tend to be ordered, low-norm
regions disordered" — assumed per-residue norm in projected space
correlates strongly with order/disorder. The empirical result implies
either (a) that correlation isn't actually strong in projected ProtT5
space, or (b) the linear probe already extracts the same signal from the
sign pattern itself (signs are 1-bit projections, not random — they
already encode some magnitude information indirectly).

The cleaner story for the disorder gap (94.9 % binary vs 98.2 % int4) is
that **it's from sign-quantization noise on direction, not magnitude
loss.** Binary discards the magnitude on each *channel*, not just the
overall residue norm. The disorder signal lives in fine per-channel
distinctions that 1 bit per dim can't capture — three or four extra bits
per dim (int4) recover most of it. There is no "polar shortcut" that gets
this back at 1 bit per dim.

## What this means for the project

- **`abtt_k=0` and `quantization='binary'` remain the default codec.**
  No change to the README.
- **The `binary_magnitude` codec mode stays in the codebase** as a tested,
  documented variant (3 unit tests, H5 round-trip, hybrid implementation).
  Future research that finds a use for per-residue magnitudes can reuse
  it, and the code documents the negative finding.
- **README "What's not (yet) solved" #1 (disorder plateau)** stays as is.
  Item #5 (PQ-as-learned-filter) is still the most plausible remaining
  angle — it operates on the *direction* the binary codec quantizes,
  which is where the disorder signal actually lives.
- **Multi-teacher distillation** (the earmark) becomes a stronger
  candidate for the next disorder-retention attempt: if the gap is in
  the *direction* and not in the *magnitude*, an alternative teacher
  could put the signal into directions binary preserves better.

## Reproduce

```bash
uv run python experiments/51_polarquant_eval.py
# ~22 min on M3 Max (4 codec configs × ~3.5 min each).
# Output: results/exp51_polarquant.json
```
