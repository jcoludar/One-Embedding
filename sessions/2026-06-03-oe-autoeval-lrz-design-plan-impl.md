---
date: 2026-06-03
started_at: 2026-06-03T15:30:00+02:00
slug: oe-autoeval-lrz-design-plan-impl
status: paused
followups:
  - "RESUME (sweep runs server-side on LRZ regardless of session/VPN). Jobs 5665512-5665525 = 6 PLMs x {raw,oe} + 650M/ProtT5 oe_norp. 1) status: `ssh ai 'squeue -u ge94xik2 -o \"%i %j %T %R\"'` (any oe_ left?) + `ssh ai 'sacct -u ge94xik2 --starttime now-1day --format=JobID,JobName%18,State,Elapsed -n'`. 2) collect: `ssh ai 'python3 .../oe_autoeval_lrz/ProteEmbedExplorations/scripts/lrz_oe_autoeval/collect_reports.py .../oe_autoeval_lrz/out .../oe_autoeval_lrz/reports'`. 3) pull: `rsync -a ai:.../oe_autoeval_lrz/reports/ /Users/jcoludar/CascadeProjects/ProteEmbedExplorations/results/oe_autoeval_reports/`. 4) per-PLM table: `.venv/bin/python scripts/lrz_oe_autoeval/compare_reports.py --raw <raw.json> --oe <oe.json> --label <PLM>`. Any FAILED arm: re-submit via `submit_all.sh --only LABEL:MODE` (it resumes from cached embeddings/tasks). Local shareable JSONs already: results/oe_autoeval_reports/ (8M raw+OE so far)."
  - "LRZ EXECUTION IN PROGRESS. G1 PASS: /work/venv built in-container = biotrainer 1.4.0 + torch 2.12.0+cu126 (CUDA works on the V100; the default cu130/CUDA-13 was too new for the node's 12.2 driver) + codec/oe_autoeval import OK. Compute nodes HAVE internet (in-container pip works). Build recipe: scripts/lrz_oe_autoeval/00_build_venv.sh (isolated venv, NO --system-site-packages [torchvision clash], NO `pip install -e` [PEE pins py>=3.12; use sys.path], torch cu126 installed before biotrainer)."
  - "FIRST RESULT (ESM2-8M, CNN/FNN probes, autoeval 30x bootstrap, single seed): OE-vs-raw retention = secondary_structure 99.3%, disorder 98.1%, conservation 96.6%, scl 105.9%. OE=center+binary at 320d (no RP, 896>320), ~32x. Pipeline validated end-to-end. Compare via scripts/lrz_oe_autoeval/compare_reports.py --raw <raw.json> --oe <oe.json>."
  - "PBC = 4 tasks (conservation, disorder[regression/spearman], scl[sequence_to_class], secondary_structure) — NOT the 9 the fan-review inferred. Probe per task (matches ga38fak baseline): CNN for the 3 residue tasks, FNN for scl (CNN invalid for sequence_to_class -> caused the first 8M-CNN failure). Patched in venv config bank + baked into 00_build_venv.sh."
  - "FULL RANGE SUBMITTED: jobs 5665512-5665525 (6 PLMs x {raw,oe} + 650M/ProtT5 oe_norp). 8M skips (done). Watcher = background task b9vsvu3ja (fires when queue drains; reports which autoeval_report_*.json landed). Then: rsync each report, run compare_reports.py per PLM -> full retention table; oe_norp isolates the RP->896 effect for 650M/ProtT5."
  - "Reference set: scripts/lrz_oe_autoeval/build_reference_from_cache.py (bare python3, login node) -> reference_2000.fasta from PBC TRAIN splits, leakage-clean (drops 246 test-in-other-task; 2000 of 24876 clean train, disjoint from 1117 test). PBC cached at ~/.cache/biotrainer/autoeval/PBC/supervised/<task>/preprocessed_0_2000/<task>.fasta (SET=train|val|test headers)."
  - "API (explore_biotrainer.py): autoeval self-downloads PBC; get_unique_framework_sequences('PBC',0,2000); `from biotrainer.config import Config` WRONG. Single-seed autoeval-native path used (LogReg/CNN deterministic-ish + 30x bootstrap) instead of the 06_harvest multi-seed path — 06/05 still carry the 9-task assumption + Config import bug if that rigor pass is wanted later."
  - "STALL DIAGNOSED: full local suite does not hang — it SEGFAULTS (SIGSEGV) at test_rns.py::test_returns_score_for_every_query via faiss index.search (src/one_embedding/rns.py:116). Pre-existing Apple-Silicon FAISS crash, NOT from this work. New oe_autoeval suite = 25/25 (run at ~43%, before the 69% crash). No regression."
  - "Nothing committed yet. New files: src/oe_autoeval/*, tests/test_oe_autoeval_*, docs/oe_autoeval_lrz/{DESIGN,PLAN}.md, scripts/lrz_oe_autoeval/*."
ended_at: 2026-06-03T16:58:49+02:00
---

# OE × autoeval on LRZ — design → plan → Phase-A impl (3 fan-review rounds)

## Goal
Benchmark the OneEmbedding codec as a compression layer on 6 PLMs (ESM2 8M/35M/150M/650M/3B +
ProtT5-half) through biotrainer `autoeval` PBC, on LRZ AI (`ge94xik2`), with OE-vs-raw retention
+ paired bootstrap CIs. Triggered by a colleague's `~/Downloads/run-autoeval.py` + an ESM2-8M
baseline report.

## What we did (this session)
1. **Diagnosed the colleague's script:** per-residue yielded packed `per_residue_bits` (silent
   garbage to the probe) and never called `codec.fit()` (uncentered). Both confirmed against the
   codec + biotrainer v1.4.0 source.
2. **Decisions (via the user):** run on **LRZ AI** (`ge94xik2`), full **ESM2 range + ProtT5**,
   biocentral can't host the custom-OE autoeval (remote custom-embedder source is disabled), so
   LRZ. Methodology: **fan reviews at every phase boundary**.
3. **DESIGN** (`docs/oe_autoeval_lrz/DESIGN.md`) + **fan review #1** (4 agents) → §14 resolutions:
   PBC is 9 tasks (2 sequence-level), OE behaves in two regimes (RP only engages >896d), need
   multi-seed probes + paired CIs, fp16 for big models, ProtT5-half, biotrainer not on LRZ.
4. **PLAN** (`docs/oe_autoeval_lrz/PLAN.md`) + **fan review #2** → §R resolutions: rewrote the
   retention estimator (multi-seed + shared split), committed to the biotrainer-direct harvest
   (autoeval 1.4.0 can't vary the probe seed), fixed `get_embedding_service` call, fixed the LRZ
   venv/HF_HOME/sbatch (in-container venv, real HF cache, `#SBATCH` pyxis directives).
5. **Phase-A implementation** (`src/oe_autoeval/`: wrappers, reference_set, retention, driver) +
   25 unit tests, all green. **Authored all LRZ scripts** (`scripts/lrz_oe_autoeval/`) — NOT run.
6. **Fan review #3** (implementation) → applied fixes: the `oe_norp` width bug, cluster bootstrap
   over proteins (residue correlation), seed-averaging for estimator coherence, units-consistent
   decision rule, Δ BCa, and the missing `06_harvest_predictions.py` seam. (See PLAN §R3.)

## Key decisions worth remembering
- OE per-residue rep for the benchmark = `decode_per_residue(encode(raw))` (matches `vep.py`),
  NOT the packed bits. Centering via `fit()` on a PBC-train-union reference (N≈2000).
- Retention = cluster-paired multi-seed bootstrap; residue tasks resample whole proteins.
- LRZ infra is in good shape: pyxis image `pytorch-2.4.0-cuda12.1.sqsh`, all 6 models HF-cached
  in `plm_choice_lrz/data/hf_cache`, 707 GB DSS free; biotrainer must be pip-installed in-container.

## Next
LRZ execution on greenlight (see followups). Nothing on the cluster has been touched (read-only
recon only).
