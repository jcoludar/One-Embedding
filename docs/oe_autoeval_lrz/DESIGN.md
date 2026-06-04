# OE × autoeval on LRZ — Design (Phase 0)

**Status:** Fan-review #1 COMPLETE. §§2–13 are the original draft; **§14 records the
review resolutions and SUPERSEDES the original where they conflict** (ProtT5 variant,
partitions, precision, retention methodology, run budget). Read §14 for the current plan.
**Author:** Claude (driven by Ivan).
**Date:** 2026-06-03.

## 1. Goal

Benchmark the **One Embedding (OE) codec** as a drop-in compression layer on top of
six protein language models, using the **Rost-lab biotrainer `autoeval` PBC framework**,
and compare OE-compressed performance against the raw (uncompressed) PLM on the same
tasks. Output is one `autoeval_report_*.json` per (PLM × {raw, OE}) that is directly
comparable to the existing baseline `autoeval_report_facebook-esm2_t6_8M_UR50D.json`.

The deliverable answers: **how much task performance does OE retain, per task, across
the PLM size range and across the RP/no-RP regimes?**

## 2. Scope

**PLMs (6):**

| PLM | HF id | native D | RP engages (d_out=896)? | OE per-residue dim | protein_vec dim |
|---|---|---:|:--:|---:|---:|
| ESM2-8M | `facebook/esm2_t6_8M_UR50D` | 320 | no | 320 | 1280 |
| ESM2-35M | `facebook/esm2_t12_35M_UR50D` | 480 | no | 480 | 1920 |
| ESM2-150M | `facebook/esm2_t30_150M_UR50D` | 640 | no | 640 | 2560 |
| ESM2-650M | `facebook/esm2_t33_650M_UR50D` | 1280 | yes →896 | 896 | 3584 |
| ESM2-3B | `facebook/esm2_t36_3B_UR50D` | 2560 | yes →896 | 896 | 3584 |
| ProtT5 | `Rostlab/prot_t5_xl_uniref50` | 1024 | yes →896 | 896 | 3584 |

**Framework:** PBC (matches the provided baseline). Other autoeval frameworks
(FLIP etc.) are out of scope unless explicitly added later.

**OE config:** default `OneEmbeddingCodec()` — 896d, binary, no codebook, abtt_k=0.
This is the shipped default and the headline config. (PQ / int tiers are a possible
follow-up sweep, not this pass.)

**Cluster:** LRZ AI (`login.ai.lrz.de`), account `ge94xik2`, project `pr63ci`,
enroot/pyxis containers. NOT the Rostlab in-house cluster where the `ga38fak`
baselines were produced.

## 3. Background

- **OE codec** (`src/one_embedding/codec_v2.py`, `OneEmbeddingCodec`): pipeline is
  center → (RP to d_out, **skipped when d_out ≥ native D**) → quantize (binary default)
  → DCT-K4 protein vector. `encode(raw (L,D))` returns a dict; `decode_per_residue(encoded)`
  returns the dequantized `(L, d_out_eff)` reconstruction; `protein_vec` is the per-protein
  DCT summary. `fit()` computes the per-channel centering mean from a corpus (binary needs
  no codebook, but **does need centering stats for the documented pipeline**).
- **autoeval / PBC**: `biotrainer.autoeval.autoeval_pipeline(...)` trains task-specific
  probes (CNN for residue→class tasks, etc.) on the supplied embeddings and reports
  bootstrapped metrics. It accepts `custom_embedding_function_per_residue` and
  `custom_embedding_function_per_sequence` generators, OR a precomputed embeddings file,
  OR a built-in/HF embedder. The provided baseline used biotrainer **1.4.0**, `device: cuda`,
  `seed: 42`, bootstrapping_iterations 30, CNN probe, 10 epochs, hold-out CV.
- **Colleague's script** (`/Users/jcoludar/Downloads/run-autoeval.py`): wraps a biotrainer
  embedding service with `OneEmbeddingCodec()` and feeds per-residue / per-sequence OE
  outputs into `autoeval_pipeline(framework="PBC", ...)`.

## 4. Defects in the colleague's script (must fix before any run)

**D1 — per-residue yields packed bits (BENCHMARKS GARBAGE).**
`embed_per_residue` yields `encoded['per_residue_bits']`, which is bit-packed bytes of
shape `(L, ceil(D_eff/8))` uint8 (`quantization.py:280`). The per-residue CNN probe would
train on packed bytes, not the embedding — destroying all geometry. **Fix:** yield
`codec.decode_per_residue(encoded)` → `(L, d_out_eff)` float32 dequantized reconstruction
(what an actual OE consumer uses). This is the faithful OE per-residue representation.

**D2 — no `fit()` → uncentered binary.**
The script never calls `codec.fit()`, so centering is silently skipped and the benchmark
measures *uncentered* binary, not OE's documented "center + binary" pipeline
(`codec_v2.py:148-159`). **Fix:** fit centering stats once per PLM (see §6) before running
autoeval.

**D3 — stale comment / shape assumption (cosmetic).**
`embed_per_sequence` comment says `protein_vec` is "(1024)"; actual dim is
`4 × d_out_eff` (1280–3584 depending on PLM, per §2 table). No functional impact, but the
comment is wrong; the benchmark must not assume a fixed per-sequence dim across PLMs.

## 5. OE-across-the-range behavior (design-level finding, not a bug)

Because OE's default `d_out=896` skips RP when `896 ≥ native D`:

- ESM2-8M/35M/150M → **binary-only** OE (no projection); compression ≈ 32× (fp32→1bit),
  per-residue dim = native.
- ESM2-650M/3B and ProtT5 → **RP→896 + binary**; compression higher (dim reduction on top
  of binary), per-residue dim = 896.

The benchmark report will therefore show two regimes. This is expected and worth a row in
the results table (per-PLM compression ratio + OE dim). We do **not** force RP on small
models (that would benchmark a non-default config); we benchmark OE **as shipped**.

## 6. Centering-fit strategy

Centering is a per-channel mean in the **native** PLM space, fit once per PLM, unsupervised.

- **Reference set:** a fixed FASTA of N≈500 sequences, shared across all 6 PLMs, seed=42.
  Provenance candidate: random UniRef50 sample, or pooled PBC *training*-split sequences
  (never test). Centering is unsupervised so leakage risk is negligible, but a dedicated
  reference set keeps it clean and identical across PLMs.
- **Procedure:** for each PLM, embed the reference set with the raw model → `codec.fit(dict)`
  → persist the centering vector (so the autoeval run is deterministic and re-runnable).
- The fitted codec is then used inside both custom embedding functions.

Open question for review: reference-set provenance + size (500 enough? matches the
repo's "200–500 proteins enough for centering" finding, `project_exp57_auto_fit_centering`).

## 7. OE config decisions

- Per-residue function → `codec.decode_per_residue(encode(raw))` (D1 fix).
- Per-sequence function → `encode(raw)['protein_vec']` (already correct).
- One `OneEmbeddingCodec()` per PLM, fitted (D2 fix), reused across all PBC tasks for that PLM.
- Defaults: d_out=896, quantization='binary', abtt_k=0, dct_k=4, seed=42.

## 8. LRZ AI execution architecture

**Container (enroot/pyxis).** Build/import an image with: CUDA PyTorch, `transformers`
(ESM2 + T5 encoders), `biotrainer==1.4.0` (pinned to match the baseline; provides
`biotrainer.autoeval`), and the OE codec (`src/one_embedding/` mounted or pip-installed
from the PEE repo). Verify `from biotrainer.autoeval import autoeval_pipeline` and the
`custom_embedding_function_*` kwargs exist in 1.4.0 before committing GPU.

**Pre-staging (login node has internet; compute nodes do NOT).**
1. PBC datasets → trigger autoeval's dataset download on the **login node** into a
   persistent cache (`$BIOTRAINER_CACHE` / `~/.cache/biotrainer/autoeval/PBC/...`).
2. HF model weights → download all 6 models on the login node into `$HF_HOME`
   (3B ≈ 11 GB; ProtT5-xl ≈ 5.4 GB). Install HF token if any model is gated (ESM2/ProtT5
   are open).
3. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` in the sbatch so compute nodes never
   reach for the network.

**SLURM layout — one job per (PLM × {raw, OE}).** Resource sizing:

| PLM | partition | GPU | est. VRAM (fp32) |
|---|---|---|---|
| 8M / 35M / 150M | V100 / A100 | 1× | < 8 GB |
| 650M | A100 | 1× | ~12 GB |
| ProtT5-xl | A100 / H100 | 1× | ~20 GB |
| 3B | H100-94GB | 1× | ~45 GB |

Walltime sized from baseline (8M PBC tracked 2.7M residues; embedding dominates wall time,
heavier for 3B). Start small (8M) to calibrate, then fan out. Each job: source container,
export offline env, run the parametrized driver with `--embedder <hf_id> --mode {raw,oe}`.

**Storage.** Outputs to a persistent project dir (`$WORK`/DSS), NOT home. One
`autoeval_report_<embedder>_<mode>.json` per job + logs.

## 9. Baseline strategy

We hold only the **ESM2-8M raw** baseline (`ga38fak`, Rostlab cluster). For apples-to-apples
across the full set and to control for cluster/GPU differences, **regenerate raw baselines
for all 6 PLMs on LRZ AI** (autoeval standard mode, no OE wrap). Validation gate: our fresh
ESM2-8M *raw* report must reproduce the provided `ga38fak` 8M JSON within bootstrap CIs —
if it does, the comparison framework is trustworthy and we proceed; if not, investigate
(dataset version, biotrainer version, seed) before burning GPU on the rest.

Total runs: 6 PLMs × {raw, OE} = **12 autoeval jobs** (+ 6 centering-fit pre-passes).

## 10. Outputs & comparison

- Per task t, per PLM p: `retention(p,t) = metric_OE(p,t) / metric_raw(p,t)`, with
  bootstrap CIs (autoeval already bootstraps; propagate or recompute paired CIs per the
  repo's `feedback_error_margins_everywhere` rule).
- Headline table: rows = PLM (with native D, OE dim, compression×), cols = PBC tasks,
  cells = retention ± CI. Plus a raw-vs-OE absolute table.
- Figures via the repo's existing barplot helpers where applicable.

## 11. Reproducibility

- Pin `biotrainer==1.4.0`; record git SHA of PEE (OE codec) used.
- seed=42 everywhere (autoeval config + codec + reference-set sampling).
- Persist: centering vectors per PLM, the reference FASTA, container image id, sbatch
  scripts, autoeval configs. Sidecar log per run (args, SHA, timestamp) per CLAUDE.md.

## 12. Risks & open questions (for fan-review)

- R1: biotrainer 1.4.0 autoeval API surface — does it accept the colleague's
  `custom_embedding_function_*` kwargs and `framework="PBC"` exactly? (Verify before build.)
- R2: PBC dataset download mechanism on a login node — is there a `--download-only` path,
  or must we trigger via a CPU dry-run? Are datasets gated?
- R3: enroot image — build locally and import, or pull a published biotrainer image?
  Does LRZ AI allow the needed base image?
- R4: 3B fp32 on H100-94GB with long sequences (max_seq_len 2000) — VRAM headroom?
  half-precision is off in the baseline; do we match (fp32) or allow fp16 for 3B only
  (and document the deviation)?
- R5: centering reference set provenance + leakage stance — acceptable?
- R6: per-sequence PBC tasks — does PBC include sequence-level tasks that consume
  `protein_vec`, or is it residue-level only? (Affects whether D-varying protein_vec matters.)
- R7: comparability of LRZ-AI raw runs vs the Rostlab `ga38fak` baseline (R9 validation gate)
  — is reproduction-within-CI a realistic bar across clusters?
- R8: does autoeval re-embed per task (expensive) or cache embeddings across tasks for a
  PLM? If per-task, OE encode runs many times — fine (cheap) but embedding re-runs are not.

## 13. Phase plan (with fan-review gates)

- **Phase 0 Design** (this doc) → **fan review #1** (4 critical lenses) → revise.
- **Phase 1 Plan/Spec** → implementation plan (writing-plans) → **fan review #2** → revise.
- **Phase 2 Implement** → driver script + sbatch + enroot build + pre-stage scripts →
  **fan review #3** → revise.
- **Phase 3 Execute** → 8M calibration + baseline-validation gate → fan out to all 12 jobs.

---

# 14. Fan-review #1 — findings & resolutions (2026-06-03)

Four parallel critical reviewers (codec correctness, autoeval API, LRZ infra, experimental
rigor). Net verdicts: codec side **sound**; autoeval side **sound** (biggest risk = silent
shape mismatch); infra **feasible with two required edits**; rigor **not yet shippable
without multi-seed probes + paired retention CIs**. Resolutions below; these supersede the
draft where noted.

## 14.1 Revised key decisions (the current plan)

1. **Per-residue OE rep = `decode_per_residue(encode(raw))`** (float32, `(L, d_out_eff)`),
   matching the repo's canonical VEP consumer path (`vep.py`). Confirmed correct, not raw bits.
2. **Per-PLM `fit()` for centering**, but note: `quantize_binary` *always* per-protein-centers,
   so D2's real effect is "corpus- vs per-protein-centered sign thresholds" (larger when RP
   engages on 650M/3B/ProtT5), not "uncentered vs centered." Still required for the documented
   pipeline.
3. **Driver asserts (load-bearing — autoeval only checks row-count L, not feature width, so a
   wrong-width array is stored as silent garbage):** every per-residue yield is `float32` with
   `shape == (len(seq), d_out_eff)`; every per-sequence yield is 1-D `(4*d_out_eff,)`. Also
   assert/guard `L ≥ 4` (else `protein_vec` width collapses to `min(4,L)*d_eff` — ragged).
4. **PBC = 9 tasks, not 4** (registry expands by split). Enumerated:
   residue→class: `conservation`, `secondary_structure`, `membrane`, `frustration-classification`;
   residue→value (spearman): `disorder_chezod`, `disorder_trizod`, `frustration-regression`;
   **sequence→class: `phages`, `scl`** (these two consume `protein_vec`). So the per-sequence
   path is load-bearing for 2/9 tasks — validate it, don't treat as cosmetic.
5. **Embeddings cache once per embedder** (ESM2/ProtT5 forward + OE encode run once per PLM,
   reused by all 9 task probes). Cache key MUST include mode `{raw,oe}` so OE/raw never collide.
6. **ProtT5 variant → `Rostlab/prot_t5_xl_half_uniref50-enc`** (half-precision encoder; the
   standard ProtT5 embedding model AND the only ProtT5 already HF-cached on LRZ). Native D=1024
   (RP→896 engages). Supersedes §2's `prot_t5_xl_uniref50`/fp32 framing.
7. **Precision: big models run fp16/bf16 for the embedding forward** (3B fp32 at L=2000 OOMs
   even on H100-94GB). Keep both arms of a PLM at identical precision; footnote that 3B/ProtT5
   absolutes aren't precision-comparable to the small-ESM2 fp32 rows. Probe training stays fp32.
8. **Centering reference set = union of PBC *training* splits across all 9 tasks**, N≈2000
   (disjoint from every PBC test split *by construction* → leakage-safe; over-provisioned since
   the pre-pass is cheap). Replaces the "UniRef50 or 500 seqs" candidate and the circular Exp-57
   justification. Persist the exact reference FASTA + provenance hash.
9. **RIGOR — multi-seed probes (the biggest change):** train each (PLM × mode × task) probe under
   **≥5 probe seeds** (autoeval's training seed, distinct from the codec seed). Embeddings are
   cached, so this multiplies only the cheap probe-training, not the GPU embedding. Report
   retention as a distribution; quantify between-seed SD so we can assert "OE deficit > probe
   noise." Without this, a 1–3% retention number is indistinguishable from probe jitter (the
   repo's own disorder audit shows a 4-pt swing from probe choice alone).
10. **RIGOR — paired retention CIs from per-item predictions:** persist per-test-item predictions
    for raw and OE; compute `retention = metric_OE/metric_raw` via **paired bootstrap over a
    common resampled item set** (+ matched probe seeds), BCa per repo convention. Two independent
    autoeval bootstraps CANNOT be combined into a valid paired CI. Report absolute Δ alongside the
    ratio (small-denominator hazard for disorder ρ etc.).
11. **RIGOR — RP-vs-noRP disambiguation arm:** for ESM2-650M and ProtT5, also run OE in
    **binary-only (no RP)** config, so the RP→896 effect is isolated at fixed model. Otherwise
    cross-PLM retention trends conflate model size with RP-vs-noRP. Report regimes visually
    separated with the "RP engaged?" column carried into the results table.
12. **8M reproduction gate (vs ga38fak) is a HARNESS sanity check only — never a baseline.**
    OE-vs-raw retention always uses the **LRZ raw** arm. Make the gate falsifiable: match
    biotrainer==1.4.0 exactly, the PBC dataset content hash, CV split assignment, max_seq_len,
    epochs, bootstrap iters, precision; pass = per-task `|Δmetric| < max(2×probe-seed SD, ε)` on
    every task. Cross-cluster GPU nondeterminism means "within CI," not bit-exact.
13. **Inference framing = estimation** (retention ± CI), not NHST; if any "OE significantly
    degrades task X" claim is made, apply BH-FDR across the task×PLM grid.

## 14.2 LRZ infra corrections (supersede §8)

- **Real partitions/QOS:** `lrz-v100x2` (V100-16GB), `lrz-hgx-a100-80x4`/`lrz-dgx-a100-80x8`
  (A100-80GB), `lrz-hgx-h100-94x4` (H100-94GB), MIG slices. QOS `gpu`: MaxWall 2 days,
  **MaxJobsPU=10** → the sweep queues 10-at-a-time (fine unattended; not 12 concurrent).
  Calibrate 8M on `lrz-v100x2/qos=gpu` (proven). 650M→A100-80 (V100-16 too tight). 3B→H100-94 fp16.
- **biotrainer is NOT installed on LRZ anywhere** — the #1 blocker. Build a **fresh** login-node
  venv (do NOT perturb the `plm_choice_lrz` venv, which pins transformers 4.44.2 and may conflict
  with biotrainer 1.4.0). Resolve the full dep closure + verify `from biotrainer.autoeval import
  autoeval_pipeline` on the login node before any GPU (also satisfies R1).
- **Containers:** reuse the existing `pytorch-2.4.0-cuda12.1.sqsh` enroot image via pyxis
  (`--container-image=<sqsh> --container-mounts=<proj>:/work:rw`, then `source venv/bin/activate`).
  No image build/import needed. Set `MKL_THREADING_LAYER=GNU` (load-bearing on this image).
- **Models already HF-cached offline** at `plm_choice_lrz/data/hf_cache/hub` (all 5 ESM2 +
  ProtT5-half). Set `HF_HOME` to that DSS path, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
  Only PBC datasets still need staging.
- **PBC dataset pre-stage:** no `--download-only` flag exists; trigger via a short **CPU dry-run
  of `autoeval_pipeline` on the login node** (downloads the single unauthenticated TUM-Nextcloud
  archive into the biotrainer cache). Compute nodes are offline.
- **Storage:** everything (OE repo checkout, venv, PBC cache, outputs/logs) under DSS
  `…/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz/` (707 GB free). Keep setgid group
  `pr63ci-dss-0004`. NOT home (tiny quota).
- **Deploy OE codec:** `git clone` PEE on login node + `pip install -e .` into the venv (record
  the PEE git SHA). Fold the 6 centering pre-passes into the start of each GPU job (embed
  reference set → fit → run autoeval) to stay under MaxJobsPU rather than as separate GPU jobs.

## 14.3 Revised run budget

- **GPU embedding arms (expensive, once each, cached):** 6 raw + 6 OE-default + 2 OE-noRP
  (650M, ProtT5) = **14 embedding passes**.
- **Probe training:** 14 arms × 9 tasks × **5 probe seeds**. Only the first seed-invocation per
  arm embeds; the other 4 reuse the cache (cheap, probe-only). Heaviest GPU cost = 3B (×2 arms)
  and ProtT5 (×3 arms) forward passes.

## 14.4 Phase-1 verification gates (confirm BEFORE building/running)

- **G1 (blocking):** stand up the login-node venv; verify biotrainer==1.4.0 imports and exposes
  `autoeval_pipeline(custom_embedding_function_per_residue=…, _per_sequence=…, framework="PBC")`.
- **G2 (blocking for rigor):** can autoeval (a) export **per-test-item predictions** and (b)
  vary the **probe training seed** across invocations? If not, §14.1-10/§14.1-9 need a fallback
  (e.g., patch task configs, or accept summary-metric retention with a documented weaker CI).
- **G3:** PBC dataset download succeeds on the login node; record its content hash.
- **G4:** confirm PBC has no `L<4` sequences (else activate the `protein_vec` guard).
- **G5:** ProtT5-half encode path through OE works end-to-end on a few sequences locally.
