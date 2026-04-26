# Audit Findings — Lab Talk Prep

**Status:** Prior recorded; audit pending.

## Prior (written before audit, 2026-04-26)

What I currently expect the audit to find:

### Greens (expected to hold up)
- Bootstrap CIs (BCa, B=10000) — Exp 43 onwards used the rigorous protocol consistently.
- Multi-seed averaging-before-bootstrap — implemented post-Bouthillier.
- CV-tuned probes (`GridSearchCV` on C/alpha) — Exp 43 onwards.
- Pooled disorder ρ + cluster bootstrap — Exp 43 fix.
- 5-PLM Exp 46 splits — same partition across PLMs by construction (single split, embeddings re-extracted).

### Yellows (expected to need clarification, not fixes)
- Disorder retention 94.9 % — real, but the floor (raw 1024d) and the noise of probes need to be co-reported.
- ANKH disorder retention 94.8 % — same caveat.
- DCT K=4 — chosen empirically; need a one-line "why not K=8" answer.
- The receiver-side "no codebook" claim for binary — needs a clean demo (decoder uses `h5py + numpy` only, nothing from `OneEmbeddingCodec`).

### Reds (expected to require fixes or honest demotion)
- `one_embedding/` package referenced in CLAUDE.md / MEMORY.md but not at repo root → confirm location, fix references everywhere.
- Modified phylo file (n_taxa 156→24) → recover or document the canonical run.
- Possibly: a few claims in CLAUDE.md / README without a traced source experiment.
- Possibly: dead code, TODO/FIXME markers in `src/`.
- Possibly: pytest failures in the current state (we have not run the full suite recently).

### Predicted distribution
~70 % green, ~20 % yellow, ~10 % red.

## Posterior (filled by audit, Tasks C.1–C.10)

### Repo hygiene (Task C.1, evidence: `docs/_audit/hygiene.md`)

- [GREEN] `git status --short` is clean at audit start (HEAD `7cc2e72`).
- [GREEN] `.gitignore` comprehensive: 67 GB of data correctly excluded; `.claude/`, `.venv/`, `.worktrees/`, secrets all covered.
- [GREEN] All 7 tracked binary/weight files (`.npz`, `.pt`) are ≤ 700 KB. No accidental large-file pollution.
- [GREEN] All code-level imports use `from src.one_embedding.codec_v2 ...` — consistent with reality. CLAUDE.md is internally consistent on this.
- [GREEN] Recent commits are well-scoped and well-named (Phase B prior stubs land cleanly).
- [YELLOW] CLAUDE.md presents both 768d (Exp 44) and 896d (Exp 47) tables without an explicit "legacy" tag on the 768d one. Two adjacent benchmark generations in one doc is confusing. Fix during D.1.
- [YELLOW] Test count drift: CLAUDE.md says 798, MEMORY.md still mentions 795 and 632 in different sections. Reconcile after pytest baseline (C.2). Fix during D.1.
- [YELLOW] `data/benchmarks/embedding_phylo_results.json` had n_taxa reduced from 156 → 24 by an earlier ad-hoc run; the file is committed but the canonical 156-taxon result needs to be confirmed/restored. Phase D fix.
- [RED] README.md describes the OLD default everywhere except the TL;DR headline:
  - line 12: "**44** experiments" (should be 47);
  - lines 21–34 Quick Start: `OneEmbeddingCodec()` annotated as "default PQ M=192 on 768d, ~20x";
  - lines 27–30 receiver comment: "h5py + numpy + codebook" (current binary default needs no codebook);
  - lines 76–101 tables and "When to Use What" all 768d-anchored, PQ-default;
  - lines 152–166 pipeline diagram and prose still recommend ABTT k=3 (off by default since Exp 45).
  This is the single biggest drift item. Major Phase D rewrite required.
- [RED] User auto-memory `~/.claude/projects/.../memory/MEMORY.md` (lines 91–96, 113, 175) treats `one_embedding/` as a top-level package. NOT a checked-in file (cannot fix in repo), but worth noting for future sessions.

**Distribution:** 5 GREEN / 3 YELLOW / 2 RED.

### Code correctness (Task C.2, evidence: `docs/_audit/pytest_baseline.txt`, `code_markers.txt`, `codec_review.md`)

- [GREEN] **813/813 tests pass** in 90 s. (Docs claim 798 — outdated, undercount; the *actual* count is higher. Re-state in CLAUDE.md/MEMORY.md as Phase D.1 fix.)
- [GREEN] **Zero TODO/FIXME/HACK/XXX markers in `src/`.** Only one true `[TODO]` exists outside `src/`: `experiments/50_sequence_to_oe.py:439` ("Downstream evaluation (SS3, disorder, retrieval)") — known follow-up for the active Stage 3 work, not a defect.
- [GREEN] **Receiver-side decode claim VERIFIED** for binary default: H5 file contains `per_residue_bits` + `means` + `scales`, decoded by ~12 lines of pure `numpy` with `h5py` for I/O. No codebook needed. CLAUDE.md headline claim holds.
- [GREEN] **Receiver claim also holds** for int4 and lossless/fp16. PQ correctly requires a codebook (matches the CLAUDE.md narrative).
- [GREEN] `OneEmbeddingCodec` defaults (lines 84–93) match CLAUDE.md prose: `d_out=896`, `quantization='binary'`, `pq_m=None` (auto), `abtt_k=0`, `dct_k=4`, `seed=42`. Constructor docstring (lines 64–82) is correctly aligned with current state.
- [YELLOW] **Hidden defaults / silent skips in codec_v2.py** — none are bugs, but each could surprise a careful Rost-lab user:
  - `auto_pq_m` docstring example only shows 768/512 cases (line 50–51), not the new 896 default.
  - `_preprocess` (line 147) silently SKIPS centering when `is_fitted == False` and quantization is binary/int4 (no codebook needed). User encoding without `fit()` gets uncentered binary — works but inconsistent with class docstring's "default: center + RP + binary".
  - `compute_corpus_stats(..., n_pcs=5)` is hardcoded (line 167), but `abtt_k` is user-configurable. Setting `abtt_k=10` would silently use only 5 PCs (Python slice `top_pcs[:10]` truncates without warning).
  - `encode_h5_to_h5` writes file-level metadata (`quantization`, `d_out`) but per-protein groups only get `seq_len` and `d_in` attrs. External users inspecting `f[pid].attrs` would miss key info; only `load_batch` correctly merges them.
  - `version: 4` is hardcoded; no upgrade path documented for older `.one.h5` files.
- [YELLOW] **Self-contained binary decoder snippet missing from docs.** While the receiver CAN decode with `numpy + h5py` only, they need to know the bit-unpacking layout (bit 7 → col 0 in column-major within byte). For the talk + paper, ship a 15-line standalone snippet. Fix during D.1.
- [YELLOW] Two informational `RuntimeWarning`s during pytest (BCa CI degenerate-data warning in `test_benchmark_*` and "catastrophic cancellation" in `test_transposed_transforms` on a deliberately-constant matrix). These are tests verifying degenerate-input handling, not failures. Document in pytest_baseline notes; no fix needed.
- [GREEN] No RED findings in this section. Code is in good shape.

**Distribution:** 6 GREEN / 3 YELLOW / 0 RED.

### Combined posterior so far (C.1 + C.2)

11 GREEN / 6 YELLOW / 2 RED.

Distribution mostly matches the prior prediction (~70 % green / 20 % yellow / 10 % red). The two REDs are exactly what I anticipated — README drift and `one_embedding/` package-location confusion in MEMORY.md — both Phase D.1 fixes.
