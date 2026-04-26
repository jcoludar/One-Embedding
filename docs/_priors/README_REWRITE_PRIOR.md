# README Rewrite — Prior

**Status:** Predicted shape of the rewritten README. Real rewrite at Task E.3 lands in `/README.md`.

## Prior (2026-04-26): predicted README sections

1. **Title + one-line pitch.** "Universal codec for PLM per-residue embeddings — 37× compression, 95–100 % retention, 5 PLMs."
2. **Headline numbers** (with CIs): the lossless / int4 / PQ M=224 / binary tier table from CLAUDE.md.
3. **Quick start** — `OneEmbeddingCodec()` four-line example. Will need to verify the import path matches whatever the audit decides about `one_embedding/` package vs `src/one_embedding/`.
4. **Pipeline diagram** (text or ASCII): center → RP → quantize → DCT K=4.
5. **5-PLM validation table** (Exp 46).
6. **Codec sweep** (Exp 47).
7. **What's not solved** — disorder gap, Exp 50 ceiling.
8. **Pointers** to `docs/STATE_OF_THE_PROJECT.md`, `docs/MANUSCRIPT_SKELETON.md`, `docs/HANDOFF.md`.
9. **Citation block** — placeholder until manuscript exists.

### What I expect to remove from the current README
- Any reference to a top-level `one_embedding/` package if it doesn't exist.
- Stale tier tables that pre-date the unified codec / Exp 47 numbers.
- Anything not traceable to a current experiment script + result file.

### What I expect to keep
- The 5-PLM table (Exp 46 numbers are recent).
- The Pareto / codec story.
- The methodology block (BCa, paired bootstrap, etc.).
