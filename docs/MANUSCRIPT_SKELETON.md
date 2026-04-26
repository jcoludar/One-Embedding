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
