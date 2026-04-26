# Audit Group 2 (C.3 + C.4 + C.5) — Implementer Report

**Subagent type:** general-purpose
**Tasks:** C.3 (splits leakage), C.4 (bootstrap/probes/baselines), C.5 (phylo file decision)

## Report

**Status:** DONE_WITH_CONCERNS

The audit completed cleanly and surfaced no PRIORITY-1 issues — every Rost-lab-critical headline claim survives.

### C.3 — Splits leakage
- **Commit:** `a88c589`
- **Distribution:** 6 GREEN / 3 YELLOW / 0 RED
- **Top findings:**
  - **Exp 46 multi-PLM split-fairness claim VERIFIED:** `cb513_80_20.json` and `esm2_650m_5k_split.json` are loaded once per benchmark call and applied PLM-agnostically. Single source of truth (file-path identity).
  - SCOPe split is genuinely strict: `superfamily_overlap=0`, `family_overlap=0` recorded in the split's `statistics` block.
  - `rules.check_no_leakage` runs at the top of every per-residue benchmark and raises on any train/test ID collision.
  - **YELLOW:** CB513 train/test is a within-CB513 random 80/20 (already `<25 % id` by dataset design — bounded but a Rost-lab Q magnet; pre-empt with one sentence in the talk).
  - **YELLOW:** SCOPe split file is named `esm2_650m_5k_split.json` but is PLM-agnostic — misleading filename, Phase D rename.
  - **YELLOW:** Exp 50 sighting results use a random 80/10/10 split; rigorous CATH-cluster re-run already designed (`docs/superpowers/plans/2026-04-06-exp50-rigorous-cath-split.md`) — talk cites Exp 50 as in-progress.

### C.4 — Bootstrap, probes, baselines
- **Commit:** `088bf37`
- **Distribution:** 9 GREEN / 1 YELLOW / 0 RED
- **Top findings:**
  - **BCa B=10,000 with percentile fallback for n<25 — verified.** Implemented via `scipy.stats.bootstrap(method="BCa")` for plain & paired; manual BCa with jackknife acceleration for cluster bootstrap (`metrics/statistics.py`).
  - **Paired bootstrap is correct for retention:** `paired=True` ensures the same protein-id resample drives raw and compressed in every iteration.
  - **Cluster bootstrap is correct for disorder:** resamples at the protein (cluster) level, recomputes pooled Spearman ρ on the union of residues; matches Davison & Hinkley 1997.
  - **Multi-seed averaging happens BEFORE bootstrap (Bouthillier 2021 — verified):** `averaged_multi_seed` averages per-item scores across seeds 42/123/456, then bootstraps the averaged dict (NOT a CI of CIs). For disorder, residue-level predictions are averaged across seeds before the cluster bootstrap.
  - **Probes are CV-tuned:** `LogisticRegression` in `GridSearchCV(C, cv=3)`; `RidgeCV(alphas, cv=3)`. `random_state=seed` plumbed through.
  - **Same `metrics.statistics` module backs Exp 43, 44, 46, 47** — no shadow implementations. Per-table conformance verified for all 7 cited CLAUDE.md tables.
  - **Retrieval baselines are fair:** Exp 43 Phase A1 reports three explicit raw baselines (mean / DCT-K=4 / ABTT+DCT-K=4) and computes retention against the most stringent (Baseline C). Exp 44/46/47 use DCT K=4 on both raw and compressed.
  - **YELLOW:** Exp 37 legacy lDDT / contact precision numbers in CLAUDE.md predate the rigorous framework and lack BCa CIs (cited as "legacy benchmarks").

### C.5 — Phylo file decision
- **Commit:** `1f10f41`
- **Distribution:** 4 GREEN / 2 YELLOW / 0 RED
- **Decision: option (a) — restored the 156-taxa version.**
- **Rationale:**
  - Provenance: `experiments/35_embedding_phylogenetics.py:2086` hardcodes the output path; not parameterized by `--dataset`, so every invocation overwrites the file with whatever dataset was run last.
  - Both 156 and 24 versions used identical trivial-sanity config (20K gen × 1 run × 2 chains; ASDSF=0.0). Neither matches the rigorous full config (200K × 4 chains × 2 runs).
  - File has **zero downstream consumers** — no script or doc reads it. The cited "10–11 of 12 families monophyletic" claim comes from `results/embed_phylo/*_consensus.nwk` analysis, not this file.
  - The 156-taxa is the more representative artifact (longer convergence on a real-sized dataset).
- **Phase D follow-ups recorded:** parameterize `BENCH_PATH` by `--dataset`; add a one-line note to `EXPECTED_QA.md` about this being a sanity artifact.

### Final repo state

```
1f10f41 audit(phylo): document the 24-taxa rerun decision
088bf37 audit(stats): bootstrap + probe + baseline conformance
a88c589 audit(splits): leakage check across all cited benchmarks
1ff3a71 audit(code): pytest baseline + markers + codec_v2 review
dd2409d audit(hygiene): repo state + package-location drift
7cc2e72 stub(prior): CALIBRATION.md
52f7fde stub(prior): EXPECTED_QA.md
93825bc stub(prior): HANDOFF.md
```
`git status --short`: empty.

### Combined Posterior so far (C.1–C.5)

| Subsection | GREEN | YELLOW | RED |
|---|---:|---:|---:|
| Repo hygiene (C.1) | 5 | 3 | 2 |
| Code correctness (C.2) | 6 | 3 | 0 |
| Splits (C.3) | 6 | 3 | 0 |
| Statistics (C.4) | 9 | 1 | 0 |
| Phylo (C.5) | 4 | 2 | 0 |
| **Total** | **30** | **12** | **2** |

Cumulative ~68/27/5 — tracks the prior (~70/20/10). Both REDs remain in the doc-drift hotspot (README.md old defaults; MEMORY.md `one_embedding/` package-location confusion). **No new REDs from Group 2.** No headline claim was invalidated; all Rost-lab-critical protocol elements survive scrutiny.
