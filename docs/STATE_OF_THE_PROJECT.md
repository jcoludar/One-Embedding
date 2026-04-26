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
