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
