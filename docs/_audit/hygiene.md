# Hygiene Audit (Task C.1)

**Date:** 2026-04-26
**HEAD at audit:** `7cc2e72`
**Working tree at start:** clean (`git status --short` empty).

## Repo size & largest tracked files

- Total disk usage: **70 GB** (`.` including all gitignored data).
- `data/` alone: **67 GB** (gitignored — embeddings, codebooks, structures, etc.).
- `.git/`: **58 MB**.
- Tracked files: **551**.
- Top-20 largest tracked files (sizes in KB):

```
776  docs/figures/pub_all_ss3_q3.png
684  data/phylo_benchmark/toxfam_v2.nex.trprobs
680  src/one_embedding/tools/weights/ss3_cnn_768d.pt
676  src/one_embedding/tools/weights/disorder_cnn_768d.pt
632  data/benchmark_suite/per_residue/trizod/moderate_rest_set.fasta
520  data/benchmark_suite/diversity/all_diversity.fasta
480  data/annotations/uniprot_annotations.json
456  src/one_embedding/tools/weights/ss3_cnn_512d.pt
452  src/one_embedding/tools/weights/disorder_cnn_512d.pt
416  uv.lock
368  data/protspace_bundles/ToxProt_ABTT3_rp512_int4.parquetbundle
336  data/phylo_benchmark/output/revbayes_trees_run_2.trees
336  data/phylo_benchmark/output/revbayes_trees_run_1.trees
316  docs/figures/pub_biology_hierarchy.png
268  data/benchmark_suite/diversity/venom_families/sequences.fasta
256  data/benchmark_suite/diversity/non_toxin/sequences.fasta
248  docs/figures/pub_codec_retrieval.png
232  docs/figures/pub_float16_benchmark.png
208  docs/figures/pub_v2_pareto.png
196  docs/figures/pub_storage_comparison.png
```

All tracked binaries are ≤ 776 KB. The 7 tracked `.npz`/`.pt` files (CNN probe weights and ABTT NPZs)
are intentional for reproducibility — none exceed 700 KB.

```
data/structures/tm_scores_200.npz                           126K
src/one_embedding/tools/weights/abtt_esm2_650m.npz           21K
src/one_embedding/tools/weights/abtt_prot_t5_xl.npz          17K
src/one_embedding/tools/weights/disorder_cnn_512d.pt        451K
src/one_embedding/tools/weights/disorder_cnn_768d.pt        675K
src/one_embedding/tools/weights/ss3_cnn_512d.pt             453K
src/one_embedding/tools/weights/ss3_cnn_768d.pt             677K
```

## one_embedding/ package

- **Top-level `one_embedding/`:** **NO** (`ls one_embedding/` → "NOT at top level").
- **`src/one_embedding/`:** **YES** (rich package: `__init__.py`, `cli.py`, `codec.py`,
  `codec_v2.py`, `core/`, `extract/`, `tools/`, ~30 modules total).

### References that imply a top-level package (RED — fix in Phase D)

These references treat `one_embedding/` as if it lived at the repo root. They need to be
reworded (either prefix with `src/` or rephrase as e.g. "the `one_embedding` package, which
lives at `src/one_embedding/`"):

- `README.md:390` — directory tree shows `one_embedding/` indented under `src/`. **OK
  in context** (the line above is `src/`), but a casual reader may misread it.
- `~/.claude/projects/.../memory/MEMORY.md:113` — `Package: \`one_embedding/\` (core/, extract/, tools/, cli.py, io.py)` — implies top-level. **RED**.
- `~/.claude/projects/.../memory/MEMORY.md:175` — `Update \`one_embedding/\` package to use unified codec (codec_v2)` — implies top-level. **RED**.
- MEMORY.md lines 91–96 — bullet list `one_embedding/core/`, `one_embedding/extract/`, etc. (without `src/` prefix). Strong implication of top-level. **RED** (× 6 lines).

NOTE: MEMORY.md is the user's `~/.claude/projects/...` memory file, not a checked-in repo file. Mentioning here for completeness; cannot fix via commit.

### References that are correct as-is (GREEN)

- `CLAUDE.md:10`, `CLAUDE.md:82` — both use `from src.one_embedding.codec_v2 import ...`. Correct.
- `CLAUDE.md:194` — `\`src/one_embedding/\`` description. Correct.
- `README.md:19`, `README.md:304` — both use `from src.one_embedding.codec_v2 import ...`. Correct.
- All `docs/superpowers/plans/2026-03-18-one-embedding-package.md` references (lines 582–1123): historical plan from before the package was internalized under `src/`. Not active drift.

### Summary
- Code-level imports (`from src.one_embedding...`) are consistent with reality.
- Doc-level prose treats the package as living at repo root in MEMORY.md (out of repo). The repo itself is OK.

## Other drift items

### CLAUDE.md headline vs content drift (YELLOW — fix in Phase D)

- **Line 4** — headline says default is "center + RP 896d + binary, ~17 KB/protein, ~37x compression".
- **Lines 69–76** — table titled "One Embedding 1.0" but rows are **all 768d** (Exp 44).
  These are the *old* default-config numbers; the new 896d Exp 47 numbers are at lines
  182–189. The table doubles up confusingly — one section per generation — without an
  explicit "(legacy 768d Exp 44)" tag at the top of lines 69–76.
- Line 99 header: `Rigorous Retention Benchmarks (Exp 43, 768d vs raw ProtT5)` — entire
  Exp 43 table is 768d, never re-run at 896d. Acceptable as labeled, but the headline
  number "98.1% mean retention" floating around does not come from this table.
- Implication for talk: a Rost-lab listener could ask "wait, your table is 768d but you
  said 896d default — which is which?" Be ready to explain this is intentional (Exp 43
  predates Exp 47, both still relevant) or unify to one default.

### README.md is stale relative to current default (YELLOW — major rewrite in Phase D)

- **Line 3** (TL;DR headline): correctly says "default: center + RP 896d + binary, ~17 KB,
  37x compression".
- **Line 12**: says "232 compression methods benchmarked across **44 experiments**" — but
  CLAUDE.md and MEMORY.md say **47 experiments**. Off by 3. **YELLOW**.
- **Lines 21–34** ("Quick Start / Python API"): code example shows
  `OneEmbeddingCodec()` with prose comment "~20x compression (default PQ M=192 on 768d)"
  and decoded shapes `(L, 768)` / `(3072,)`. **CONTRADICTS** the headline (line 3) and
  the actual current default (`(L, 896)` / `(3584,)`). **RED — must fix before talk.**
- **Lines 27–30**: receiver-side comment says "h5py + numpy + codebook". For the *current
  binary default* this is wrong — binary needs no codebook. The plan task notes this is
  a key claim to verify; for the old PQ default it was correct. **RED**.
- **Lines 76–91** ("Compression Configurations" table): all rows still 768d. Labelled as
  Exp 44, which is correct historically — but the surrounding paragraph (line 80) still
  states `Default: ABTT3 + RP 768d + PQ M=192`. **RED**.
- **Lines 93–101** ("When to Use What" table): same issue — all rows reference 768d
  defaults and the old PQ default.
- **Line 152** ("The Pipeline" diagram): pipeline still depicts `Raw PLM (L, 1024) → ABTT
  k=3 → RP 768d → fp16 → DCT K=4 → (3072,)`. ABTT k=3 is **off by default** now
  (Exp 45 finding); pipeline should show 896d / no ABTT / binary. **RED**.
- **Line 162** ("Why each step matters → ABTT k=3"): paragraph still recommends ABTT.
  Contradicts current default and Exp 45 disorder finding. **RED**.

This is the single biggest drift item in the repo. README is in PQ/768d/ABTT-on era;
codec ships in binary/896d/ABTT-off era.

### Tests count drift (YELLOW)

- CLAUDE.md line 168: "**798 tests**, 6 tasks, 8 datasets, 5 PLMs."
- MEMORY.md still references "795 tests" and "632 tests" in different places.
- CLAUDE.md line 200 (Architecture): "**798 tests across multiple modules**".
- C.2 will run pytest; the actual count will go in the pytest baseline.

### Modified phylo file in baseline (YELLOW — earlier known item)

- `data/benchmarks/embedding_phylo_results.json` was the only modified file at the start
  of this work session per the system status block ("M data/benchmarks/embedding_phylo_results.json").
- Now clean (we committed Phase A). The file's `n_taxa` was reduced from 156 → 24 in some
  earlier ad-hoc run. Mentioned in the prior. Real fix in Phase D.

### `.gitignore` is comprehensive (GREEN)

- 67 GB of data correctly ignored (`*.h5`, `data/residue_embeddings/`, `data/checkpoints/`,
  `data/external/`, etc.).
- `.claude/`, `.venv/`, `.worktrees/`, `tools/reference/` ignored.
- No secrets pattern (e.g. `.env*`) is missing — `.env` and `.env.local` covered.

### Recent commits look clean (GREEN)

```
7cc2e72 stub(prior): CALIBRATION.md
52f7fde stub(prior): EXPECTED_QA.md
93825bc stub(prior): HANDOFF.md
9a815e4 stub(prior): README rewrite plan
89f3908 stub(prior): MANUSCRIPT_SKELETON.md
```

Phase B prior-stubs land in order; Phase A baseline commit is just behind those.

### Summary table

| Severity | Count | Items |
|---|---|---|
| GREEN | 4 | Code imports correct, gitignore comprehensive, recent commits clean, tracked binaries small |
| YELLOW | 3 | CLAUDE.md 768d/896d table coexistence, tests-count drift, phylo file recovered |
| RED | 2 | README.md describes old PQ/768d/ABTT default (multiple sections); MEMORY.md treats `one_embedding/` as top-level (not a repo file) |
