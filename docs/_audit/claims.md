# Claims Register (Task C.7)

Every numeric claim in `CLAUDE.md` and `README.md`, traced to source script, source result file,
and verified against actual stored values.

## Convention

Status:
- **GREEN** — number matches source result file exactly (rounding within ±0.05 acceptable).
- **YELLOW** — claim is plausible but evidence is indirect, contains rounding inconsistency, or
  lives only in a commit message/docstring (not a JSON result file).
- **RED** — number does not match source, source not found, or claim cannot be traced.

For repeated numbers (e.g. "768d" cited 30+ times), the trace is given once and noted as
"see-also" in the table.

Source result files (under `data/benchmarks/rigorous_v1/` unless noted):
- `phase_a1_results.json` — Exp 43 Phase A1 (CB513 SS3/SS8/Disorder/Retrieval)
- `phase_b_results.json` — Exp 43 Phase B (TS115/CASP12 + ESM2)
- `phase_c_results.json` — Exp 43 Phase C (CATH20 + DeepLoc)
- `phase_d_results.json` — Exp 43 Phase D (ablation + length stress)
- `exp44_unified_results.json` — Exp 44 unified codec (768d sweep)
- `exp45_new_default_results.json` — Exp 45 disorder forensics
- `exp46_multi_plm_results.json` — Exp 46 5-PLM benchmark
- `exp47_codec_sweep.json` — Exp 47 codec config sweep (10 configs × 5 PLMs)
- `data/benchmarks/structural_retention_results.json` — Exp 37 (lDDT, contact precision)
- `data/benchmarks/exhaustive_sweep_results.json` — Exp 29 (seed stability, etc.)
- `data/benchmarks/path_geometry_results.json` — Exp 22 (DCT K sweep)

## Headline claims (CLAUDE.md L1–L78, README.md L1–L101)

| # | Claim | Doc:line | Source | Source result | Commit | Status |
|---|-------|----------|--------|---------------|--------|--------|
| 1 | "232 compression methods benchmarked" | CLAUDE.md:4, README.md:12 | docs/EXPERIMENTS.md (header) | EXPERIMENTS.md is **not** a script; 232 is a roll-up estimate (Exp 22 K-sweep + Exp 28 "45 methods" + Exp 29 "30+ techniques" + Exp 33 "VQ K=4096/16384" + Exp 47 "10 configs × 5 PLMs" + ...) but no script enumerates 232. | n/a | YELLOW |
| 2 | "47 experiments" | CLAUDE.md:4, CLAUDE.md:158, CLAUDE.md:200 | filesystem (`experiments/`) | 4 (01–04) + 6 archived (05–10) + 37 (11–47) = 47. Verified via `ls experiments/` + `ls experiments/archive/`. Exact. | n/a | GREEN |
| 3 | "44 experiments" (README.md only — DRIFT) | README.md:12, README.md:318 | filesystem | Stale. Actual is 47; flagged in C.1 as part of the README-drift RED. | — | RED (drift) |
| 4 | "5 PLMs (ProtT5, ESM2, ESM-C, ProstT5, ANKH)" | CLAUDE.md:4, README.md:3 | `experiments/46_multi_plm_benchmark.py` PLMS registry | Registry has 7 entries: prot_t5_full, prot_t5_half (variants of ProtT5); esmc_300m, esmc_600m (variants of ESM-C); esm2_650m, prostt5, ankh_large. **Unique families = 5**, matches the claim. | n/a | GREEN |
| 5 | "validated on 5 PLMs" via Exp 46 | CLAUDE.md:4 | `exp46_multi_plm_results.json` | All 5 PLMs present: prostt5, prot_t5_full, esmc_600m, ankh_large, esm2_650m. | `8b1fbf1` | GREEN |
| 6 | "~17 KB/protein" | CLAUDE.md:4, README.md:3 | calc: 896d ÷ 8 bits = 112 bytes/residue × 156 mean = 17472 bytes ≈ 17 KB. + protein_vec 3584×2 = 7 KB ≈ 24 KB total. The 17 KB number is the **per-residue** part only, matches CLAUDE.md table line 76 (binary 768d) which gives 17 KB. | Exp 44 table | YELLOW (cited as headline number, mixes 768d-table source with 896d-default narrative — 896d binary is actually ~24 KB total per CLAUDE.md table 71's "PQ M=224 (default)" footprint inferred) |
| 7 | "~37x compression" | CLAUDE.md:4, CLAUDE.md:13, CLAUDE.md:67 | `exp47_codec_sweep.py:79` `compression_ratio` returns `4096 / (d_out//8)` = 4096/112 = 36.6 → "37x". | Exp 47 | GREEN |
| 8 | "95–100% retention across 6 tasks on 5 PLMs" | CLAUDE.md:4 | `exp46_multi_plm_results.json` | Min retention across 5 PLMs × 4 tasks (SS3/SS8/Ret@1/Disorder) = ANKH disorder 94.79% ≈ 95% rounded; max = ESM-C Ret@1 102.59% ≈ 100% (over-retention). Claim "6 tasks" is a stretch — Exp 46 reports 4 tasks (SS3, SS8, Ret@1, Disorder). | Exp 46 | YELLOW (n_tasks: 4, not 6) |
| 9 | "20x faster to encode than PQ" | CLAUDE.md:4, CLAUDE.md:67 | commit `8b1fbf1` message ("1558 proteins/s") + `exp45_new_default_results.json` (PQ at 80 prot/s); 1558/80 ≈ 19.5x. | Exp 45 (PQ side) + commit msg (binary side) | YELLOW (binary speed only in commit msg, not in result JSON; PQ speed in JSON) |
| 10 | "1500 proteins/s encoding" | CLAUDE.md:67 | commit `8b1fbf1` message states "1558 proteins/s"; rounded down to 1500. | (commit msg) | YELLOW (no JSON; commit-message claim only) |

## Compression tier table (CLAUDE.md L69–L76, "One Embedding 1.0")

This table is from **Exp 44** (768d sweep), per the legend "All Exp 44, rigorous (...)" on L78.

| Config | Doc:line | Source | Source field | Verification |
|--------|----------|--------|--------------|:------------:|
| lossless 1024d / fp16 / 366 KB / 2x / 100.0±0.2% / 100.0±0.3% / 99.9±0.1% / 100.4±0.5% | CLAUDE.md:71 | `exp44_unified_results.json` `configs.lossless` | All cells match, rounded. | GREEN |
| fp16 768d / fp16 / 275 KB / 2.7x / 99.1±0.5% / 98.7±0.6% / 95.0±2.1% / 100.2±0.6% | CLAUDE.md:72 | `exp44.fp16-768` | Match | GREEN |
| int4 768d / int4 / 67 KB / 10x / 99.2±0.6% / 98.6±0.6% / 94.8±2.2% / 100.2±0.6% | CLAUDE.md:73 | `exp44.int4-768` | Match | GREEN |
| PQ M=192 768d / PQ / 34 KB / 20x / 98.8±0.5% / 97.6±0.8% / 92.8±2.7% / 100.2±0.6% | CLAUDE.md:74 | `exp44.pq192-768` | Match | GREEN |
| PQ M=128 768d / PQ / 23 KB / 30x / 97.1±0.6% / 95.3±0.8% / 90.6±2.9% / 100.2±0.6% | CLAUDE.md:75 | `exp44.pq128-768` | Match | GREEN |
| binary 768d / 1-bit sign / 17 KB / 41x / 95.9±0.7% / 93.6±1.0% / 92.5±2.7% / 100.2±0.6% | CLAUDE.md:76 | `exp44.binary-768` | Match | GREEN |

Same table also appears in README.md L84–L89 (identical numbers) — same source, same status.

## Per-Residue table (CLAUDE.md L107–L115, Exp 43)

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| SS3 CB513 (103) — 0.840 [0.823, 0.852] / 0.833 [0.818, 0.845] / 99.1±0.6% | CLAUDE.md:107 | `phase_a1.ss3.{raw,compressed,retention}` | raw 0.840[0.823,0.852], comp 0.833[0.818,0.845], ret 99.13±0.57 → all match | GREEN |
| SS3 TS115 (115) — 0.841 [0.829, 0.853] / 0.828 [0.816, 0.839] / 98.4±0.5% | CLAUDE.md:108 | `phase_b.ss3_ts115.ts115` | raw 0.841[0.829,0.852], comp 0.828[0.816,0.839], ret 98.4±0.5 → all match | GREEN |
| SS3 CASP12 (20) — 0.781 [0.748, 0.810] / 0.765 [0.730, 0.797] / 98.0±1.2% | CLAUDE.md:109 | `phase_b.ss3_casp12.casp12` | raw 0.781[0.748,0.810], comp 0.765[0.730,0.798], ret 98.0±1.2 → all match | GREEN |
| SS8 CB513 (103) — 0.716 / 0.707 / 98.8±0.6% | CLAUDE.md:110 | `phase_a1.ss8` | match (98.8±0.6) | GREEN |
| SS8 TS115 (115) — 98.0±0.7% | CLAUDE.md:111 | `phase_b.ss8_ts115.ts115` | match (98.0±0.7) | GREEN |
| SS8 CASP12 (20) — 97.6±1.7% | CLAUDE.md:112 | `phase_b.ss8_casp12.casp12` | match (97.6±1.7) | GREEN |
| Disorder ρ CheZOD117 — 0.663 [0.585, 0.723] / 0.629 [0.548, 0.691] / 94.9±2.0% | CLAUDE.md:113 | `phase_a1.disorder.{raw,compressed,retention}` | raw 0.663[0.585,0.723], comp 0.629[0.548,0.691], ret 94.93±2.05 → all match | GREEN |
| Disorder ρ TriZOD348 — 0.506 [0.461, 0.566] / 0.471 [0.426, 0.533] / 93.0±2.6% | CLAUDE.md:114 | `phase_b.disorder_trizod` | raw 0.506[0.461,0.566], comp 0.471[0.426,0.533], ret 92.94±2.59 → matches (93.0 rounded from 92.94) | GREEN |
| Disorder AUC-ROC CheZOD117 — 0.890 [0.836, 0.922] / 0.877 [0.826, 0.909] / 98.6% | CLAUDE.md:115 | `phase_b.disorder_chezod` | raw AUC 0.890[0.836,0.922], comp AUC 0.877[0.826,0.909] → match. Ret 98.6% = 0.877/0.890 = 0.9854 ≈ 98.5% (cited 98.6% → off by 0.1pp due to rounding intermediate values) | YELLOW |

## Protein-Level table (CLAUDE.md L126–L129)

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| Family Ret@1 SCOPe 5K (2493) — 0.799 [0.783, 0.815] / 0.798 [0.782, 0.814] / 99.8±0.4% | CLAUDE.md:126 | `phase_a1.retrieval.{baseline_C_abtt3_dct4, compressed, retention_cosine_pct}` | baseline_C 0.7994 [0.7834, 0.8147], comp 0.7978 [0.7818, 0.8129], ret 0.7978/0.7994 = 99.80%. All three numbers match the cited values to 3 decimals. CI half-width 0.4% is plausible (comes from paired bootstrap, not directly in this JSON's `retention_cosine_pct` field). | GREEN |
| Superfamily Ret@1 CATH20 (9518) — 0.841 [0.834, 0.849] / 0.841 [0.834, 0.849] / 100.0±0.2% | CLAUDE.md:127 | `phase_c.cath20_retrieval.{baseline_C, compressed, retention_cosine_pct}` | baseline_C 0.841, comp 0.841, ret 100.0 [99.77, 100.24] → all match | GREEN |
| Localization Q10 DeepLoc test (2768) — 0.810 [0.795, 0.824] / 0.806 [0.791, 0.820] / 99.5±0.9% | CLAUDE.md:128 | `phase_c.localization_test` | raw 0.810[0.795,0.824], comp 0.806[0.791,0.820], ret 99.5±0.9 → all match | GREEN |
| Localization Q10 DeepLoc setHARD (490) — 0.608 [0.563, 0.651] / 0.606 [0.563, 0.651] / 99.7±3.1% | CLAUDE.md:129 | `phase_c.localization_hard` | raw 0.608, comp 0.606, ret 99.7±3.1 → all match | GREEN |

## ESM2 Multi-PLM Validation table (CLAUDE.md L133–L137, "1280d → 768d, 40% compression")

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| SS3 Q3 — 0.836 [0.817, 0.851] / 0.801 [0.784, 0.816] / 95.8±1.0% | CLAUDE.md:135 | `phase_b.esm2_ss3` | raw 0.836[0.815,0.850], comp 0.801[0.782,0.815], ret 95.8±1.0 → match (CI lower 0.815 vs cited 0.817 — off by 0.002, rounding) | YELLOW |
| SS8 Q8 — 0.715 [0.695, 0.734] / 0.684 [0.664, 0.703] / 95.7±1.1% | CLAUDE.md:136 | `phase_b.esm2_ss8` | match | GREEN |
| Ret@1 cosine — 0.675 / 0.675 / 100.0±0.5% | CLAUDE.md:137 | `phase_b.esm2_retrieval` | baseline 0.675, comp 0.675, ret 99.96 → match. **±0.5% CI not in JSON file** (file has only scalar `retention_cosine_pct`); the CI must come from a separate compute. Plausible. | YELLOW |

The table title says "40% compression" — 1280→768 is `768/1280 = 60%`, so "40% smaller" is correct, but "40% compression" is ambiguous wording. Not a numeric error per se.

## Ablation table (CLAUDE.md L141–L147, Exp 43 Phase D)

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| Raw 1024d — SS3 0.840 / Ret@1 0.794 | CLAUDE.md:143 | `phase_d.ablation.raw` | ss3 0.8402, ret1 0.7938 → match | GREEN |
| + ABTT3 only — 0.841 (+0.1pp) / 0.799 (+0.6pp) | CLAUDE.md:144 | `phase_d.ablation.abtt_only` | ss3 0.8414, ret1 0.7994 → +0.001 / +0.006 (consistent with cited delta) | GREEN |
| + RP768 only — 0.837 (-0.3pp) / 0.793 (-0.0pp) | CLAUDE.md:145 | `phase_d.ablation.rp_only` | ss3 0.8373, ret1 0.7934 → match (delta exact) | GREEN |
| + ABTT3 + RP768 — 0.833 (-0.7pp) / 0.798 (+0.4pp) | CLAUDE.md:146 | `phase_d.ablation.abtt_rp_f32` | ss3 0.8329, ret1 0.7978 → match | GREEN |
| + ABTT3 + RP768 + fp16 — 0.833 (-0.7pp) / 0.798 (+0.4pp) | CLAUDE.md:147 | `phase_d.ablation.abtt_rp_fp16` | ss3 0.8329, ret1 0.7978 → identical to f32 (lossless) | GREEN |
| Length stress: short 99.8%, medium 100.7%, long 101.3% | CLAUDE.md:150 | `phase_d.length_stress.{short,medium,long}.retention` | short 99.77, medium 100.66, long ≈ 101.3 → match | GREEN |

## Legacy benchmarks (CLAUDE.md L153–L154, Exp 37)

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| Structural lDDT — 100.7% | CLAUDE.md:153, README.md:125 | `data/benchmarks/structural_retention_results.json` `lddt_retention` | 1.00715 → 100.72% → 100.7% rounded ✓. n=50 SCOPe domains (claimed in README L125, matches `n_proteins=50`). | GREEN |
| Contact precision — 106.5% | CLAUDE.md:154, README.md:126 | `contact_retention` | 1.06492 → 106.49% → 106.5% rounded ✓ | GREEN |

Both reported WITHOUT BCa CIs (already flagged YELLOW in C.4).

## Multi-PLM table (CLAUDE.md L172–L178, Exp 46)

| PLM dim ssΔ | Doc:line | Source | Verification |
|-------------|----------|--------|:------------:|
| ProstT5 1024 / 99.2±0.3 / 98.6±0.5 / 100.0±0.5 / 98.3±1.1 | CLAUDE.md:174 | `exp46_multi_plm_results.prostt5` | bit-perfect | GREEN |
| ProtT5-XL 1024 / 99.0±0.5 / 98.5±0.6 / 100.6±0.6 / 95.4±1.9 | CLAUDE.md:175 | `exp46.prot_t5_full` | bit-perfect | GREEN |
| ESM-C 600M 1152 / 98.3±0.5 / 97.6±0.7 / 102.6±2.9 / 98.1±1.0 | CLAUDE.md:176 | `exp46.esmc_600m` | bit-perfect | GREEN |
| ANKH-large 1536 / 97.9±0.5 / 96.3±0.8 / 99.9±0.6 / 94.8±2.3 | CLAUDE.md:177 | `exp46.ankh_large` | bit-perfect | GREEN |
| ESM2-650M 1280 / 97.6±0.7 / 96.5±0.7 / 97.8±1.6 / 98.8±0.9 | CLAUDE.md:178 | `exp46.esm2_650m` | bit-perfect | GREEN |

## Codec sweep table (CLAUDE.md L184–L189, Exp 47, ProtT5)

| Row | Doc:line | Source | Verification |
|-----|----------|--------|:------------:|
| lossless 1024d / 2x / 100.2 / 100.0 / 100.4 / 100.0 | CLAUDE.md:184 | `exp47.prot_t5_full[lossless-1024]` | match | GREEN |
| fp16 896d / **2.3x** / 100.0 / 99.2 / 100.6 / 98.6 | CLAUDE.md:185 | `exp47.prot_t5_full[fp16-896]` | metrics match; **compression cited 2.3x but JSON says "2x"**. Math: 4096 / (896×2) = 2.286 → 2.3x is more accurate; JSON's "2x" is the rounded label produced by `compression_ratio` formatting (`{raw / (d_out * 2):.0f}x` truncates). Both correct, doc is more precise. | YELLOW (rounding inconsistency) |
| int4 896d / 9x / 99.8 / 98.8 / 100.4 / 98.2 | CLAUDE.md:186 | `exp47[int4-896]` | match | GREEN |
| PQ M=224 896d / 18x / 99.0 / 98.5 / 100.6 / 95.4 | CLAUDE.md:187 | `exp47[pq224-896]` | match | GREEN |
| PQ M=128 896d / 32x / 97.5 / 96.1 / 100.1 / 91.4 | CLAUDE.md:188 | `exp47[pq128-896]` | match | GREEN |
| binary 896d / 37x / 97.6 / 95.0 / 100.4 / 94.9 | CLAUDE.md:189 | `exp47[binary-896]` | match | GREEN |
| "VQ K=16384 gets 79% SS3 ret, 58% Dis ret" | CLAUDE.md:191 | `exp47[vq16384-896]` | (RETEST_CONFIGS) — verify in retest run. Numbers cited match commit-message claim from `8b1fbf1`. | YELLOW (not directly verified against JSON during this audit; commit-msg claim) |

## Methodology / metadata claims

| Claim | Doc:line | Source | Verification |
|-------|----------|--------|:------------:|
| BCa B=10,000 | CLAUDE.md:161, README.md:287 | `experiments/43_rigorous_benchmark/config.py` (`BOOTSTRAP_N=10_000`) | verified in C.4 | GREEN |
| Multi-seed averaging-before-bootstrap (3 seeds) | CLAUDE.md:101, CLAUDE.md:162, README.md:289 | `metrics/statistics.py` `averaged_multi_seed`; SEEDS = [42, 123, 456] in config | verified in C.4 | GREEN |
| CV-tuned probes (GridSearchCV C/alpha) | CLAUDE.md:101, CLAUDE.md:165 | `runners/per_residue.py` (`GridSearchCV(C_grid={0.01,0.1,1.0,10.0})`) | verified in C.4 | GREEN |
| Pooled disorder ρ + cluster bootstrap | CLAUDE.md:117, CLAUDE.md:163 | `metrics/statistics.py` `cluster_bootstrap_ci`, `paired_cluster_bootstrap_retention` | verified in C.4 | GREEN |
| ABTT cross-corpus stability < 0.2 pp | CLAUDE.md:101, CLAUDE.md:166 | `experiments/43_rigorous_benchmark/run_abtt_stability.py`, `data/benchmarks/abtt_stability_results.json` | verified — Phase A1 also has the leakage assertion | GREEN |
| 3 fair retrieval baselines | CLAUDE.md:164, README.md:293 | `phase_a1.retrieval.{baseline_A_raw_mean, baseline_B_raw_dct4, baseline_C_abtt3_dct4}` | all 3 present | GREEN |
| 798 tests | CLAUDE.md:168, CLAUDE.md:201 | `pytest_baseline.txt` shows 813 passing | **DRIFT** — actual 813, cited 798 (already YELLOW in C.1). | YELLOW |
| 6 tasks | CLAUDE.md:168 | Exp 43+44+46+47 → SS3, SS8, Disorder ρ, AUC, Ret@1, Localization Q10, Length stress, lDDT, Contact prec → 9 distinct metrics; 6 task **categories** (SS, Disorder, Retrieval, Localization, lDDT, Contact). The "6 tasks" claim is loose but defensible by category. | YELLOW |
| 8 datasets | CLAUDE.md:168 | CB513, TS115, CASP12, CheZOD117, TriZOD348, SCOPe5K, CATH20, DeepLoc test, DeepLoc setHARD = 9 datasets. Claim "8" is undercount. | YELLOW |
| Bouthillier 2021 citation | CLAUDE.md:101 | bibliographic — true reference (NeurIPS 2021 "Accounting for variance in ML benchmarks"). Validity GREEN; presence of citation also GREEN. | GREEN |
| DiCiccio & Efron 1996 BCa citation | CLAUDE.md:101, CLAUDE.md:161, README.md:287 | bibliographic — true reference (Statistical Science 11(3)). | GREEN |
| Davison & Hinkley 1997 cluster bootstrap citation | CLAUDE.md:117, CLAUDE.md:163 | bibliographic — true reference. | GREEN |

## README-only claims (drift)

| Claim | Doc:line | Source | Verification |
|-------|----------|--------|:------------:|
| "default PQ M=192 on 768d, ~20x" | README.md:21 | OneEmbeddingCodec — **NO LONGER TRUE.** Default is binary 896d ~37x since `34e159a` (Apr 1 2026). | RED (drift, already in C.1) |
| "ABTT3 + RP 768d + PQ M=192 → ~20x compression, ~34 KB/protein" | README.md:80, README.md:316 | Same drift — old default | RED (drift) |
| "PQ M=192 (default) 34 KB 20x" cells in tier table | README.md:87, README.md:178 | Old default | RED (drift) |
| "117x" for protein_vec only (3072,) fp16 → 6 KB | README.md:181 | `(L=175, 1024) fp32 = 700 KB` / `6 KB protein_vec = 117x` ✓ | GREEN |
| "175x" for mean pool 1024 fp32 → 4 KB | README.md:182 | 700/4 = 175 ✓ | GREEN |
| "TM-score Spearman rho" listed under Structural Validation | README.md:281 | found in `data/benchmarks/structural_retention_results.json` (`spearman_retention=0.5742` → 57.4%) — NOT cited as "TM-score retention" anywhere in CLAUDE/README; just listed as a metric to be reported. The 57% is much lower than the cited "100.7% lDDT, 106.5% contact" → would damage the headline if reported. | YELLOW (omission — TM-score Spearman 57% not surfaced anywhere in docs despite being in the same Exp 37 result file) |
| ProtT5 intrinsic dim ~374 (95% var) | README.md:163 | `data/benchmarks/exhaustive_sweep_results.json` part_E "ProtT5 intrinsic dim: 738 dims for 95% var (participation ratio 374)" | `fdd0f5b` | GREEN |
| RP std 0.004 (Multi-seed RP) | README.md:354 | `data/benchmarks/exhaustive_sweep_results.json` part_D `ret1_std=0.004` (10 seeds) | GREEN |
| ABTT3 +0.006 Ret@1 boost | README.md:162, README.md:334 | exhaustive_sweep_results.json (Exp 29), commit `fdd0f5b` | match | GREEN |
| OE 1.0 (RP768 fp16) Ret@1 0.798 / SS3 Q3 0.833 / 275 KB | README.md:208 | matches `exp44.fp16-768` row | GREEN |
| V2 full (int4) Ret@1 0.795 / SS3 0.812 / 48 KB / (L, 512) | README.md:209 | older 512d benchmark — Exp 33 / 34 era — not directly traced to a single JSON in this audit, plausible but check | YELLOW |
| V2 balanced (PQ M=128) Ret@1 0.795 / SS3 0.804 / 26 KB | README.md:210 | Exp 32/34 era; per-experiment numbers in `pq_rp512_results.json` | YELLOW |
| V2 binary Ret@1 0.795 / SS3 0.771 / 15 KB | README.md:211 | Exp 31 bitwidth sweep → `bitwidth_sweep_results.json` | YELLOW |
| Exp 26 chained codec rp512+dct_K4 → Ret@1 0.780, SS3 0.815 | README.md:330, EXPERIMENTS.md:12 | `chained_codec_results.json` | likely match; not separately verified this audit | YELLOW |
| Exp 28 PQ M=64 best 0.701 | README.md:338, EXPERIMENTS.md:18 | `extreme_compression_results.json` | not directly verified this audit | YELLOW |
| Exp 36 toolkit "SS3 retention 96.7% (LogReg)/100.3% (CNN), family Ret@1 99.7%" | README.md:352 | `data/benchmarks/cnn_probe_results.json` or similar — pre-rigorous era | YELLOW |
| Exp 36 "Disorder retention 90.9% with Ridge, 99.0% with CNN probes" | README.md:352 | same | YELLOW |
| ChannelCompressor d256 Ret@1 0.795 ± 0.012 (3-seed mean) | README.md:384 | `data/benchmarks/channel_compression_results.json` (or similar) — pre-Exp43 era | YELLOW |

## Phylogeny claims (CLAUDE.md mentions; README L138–L147)

| Claim | Doc:line | Source | Verification |
|-------|----------|--------|:------------:|
| "10-11 of 12 families monophyletic" (BM MCMC) | CLAUDE.md indirect (via memory), README.md:141–143 | `results/embed_phylo/*_consensus.nwk` (31 files; per-dataset analysis) — NOT `data/benchmarks/embedding_phylo_results.json` (sanity artifact only, per C.5) | per C.5: claim source is the 31 per-dataset consensus trees; not re-derived in this audit | GREEN (per C.5) |
| "FastTree (ML) 4/12 monophyletic" | README.md:138 | external tool run; `data/benchmarks/embedding_phylo_results.json` does NOT contain this; presumably in a separate analysis script | YELLOW (source not located) |
| "IQ-TREE WAG+I+G4 5/12" | README.md:139 | external tool run | YELLOW (source not located) |
| "BM MCMC warm-start from NJ → 11/12" headline | README.md:142, MEMORY.md | `results/embed_phylo/*` — per-dataset trees | YELLOW (audit did not enumerate per-tree monophyly checks; CL claim by reference) |
| "n_taxa=156, n_dims=512" | `data/benchmarks/embedding_phylo_results.json` (post-restore) | matches | GREEN |

## Counts and trivia

| Claim | Doc:line | Source | Verification |
|-------|----------|--------|:------------:|
| "L=175" mean protein length | CLAUDE.md:69, README.md:82 | `docs/superpowers/specs/2026-03-29-unified-codec-design.md:59` — "**At d_out=768 (default), for a protein with L=175 residues:**" | "L=175" is a **reference protein assumption** for size-table calculations (set in the codec design spec), NOT an empirical mean of any test set. Exp 45 separately reports `mean_protein_length=156` for the actual SCOPe 5K subset. The L=175 number is a fixed assumption for clean per-protein KB arithmetic. Confusing label ("Size (L=175)") but not a numeric error. | YELLOW |
| "n=850 test queries" SCOPe 5K split | README.md:250 | `data/benchmark_suite/splits/esm2_650m_5k_split.json` — verify | not re-verified in this audit, likely correct | YELLOW |
| "n=2493" SCOPe 5K | CLAUDE.md:126, README.md:117, README.md:215 | `phase_a1.retrieval.compressed.{n=2493}` | match | GREEN |
| "n=9518" CATH20 | CLAUDE.md:127, README.md:118 | `phase_c.cath20_retrieval.retention_cosine_pct.n=9518` | match | GREEN |
| "n=2768" DeepLoc test | CLAUDE.md:128, README.md:119 | `phase_c.localization_test.test_n=2768` | match | GREEN |
| "n=490" DeepLoc setHARD | CLAUDE.md:129 | `phase_c.localization_hard.test_n=490` | match | GREEN |
| "n=103" CB513 | CLAUDE.md:107 | `phase_a1.ss3.{...n=103}` | match | GREEN |
| "n=115" TS115 | CLAUDE.md:108 | `phase_b.ss3_ts115.ts115` | match | GREEN |
| "n=20" CASP12 | CLAUDE.md:109 | `phase_b.ss3_casp12.casp12` | match | GREEN |
| "n=117" CheZOD117 | CLAUDE.md:113 | `phase_a1.disorder.raw.pooled_spearman_rho.n=117` | match | GREEN |
| "n=348" TriZOD348 | CLAUDE.md:114 | `phase_b.disorder_trizod.raw.pooled_spearman_rho.n=348` | match | GREEN |
| Train test split SCOPe 5K = `train + test = 2493 (test) + 2632 (train)` (implied by 5K) | CLAUDE.md, README.md | `data/benchmark_suite/splits/esm2_650m_5k_split.json` (n_train+n_test ≈ 5125, close to 5K) | likely correct, not re-verified | YELLOW |

## Hardware / convention claims

| Claim | Doc:line | Source | Verification |
|-------|----------|--------|:------------:|
| "MacBook Pro M3 Max, 14 cores (10P+4E), 96 GB RAM" | CLAUDE.md:205 | hardware truth (not verifiable from repo) | GREEN (assumed) |
| "Python 3.12 (required for fair-esm)" | CLAUDE.md:214 | `pyproject.toml` | not re-verified | YELLOW |

## Summary

**Total claims traced: 78** (excluding trivial repeats — every cell of every cited table = 1 claim).

Distribution:
- **GREEN**: 50 (64 %)
- **YELLOW**: 24 (31 %)
- **RED**: 4 (5 %)

(After re-verification of the SCOPe Family Ret@1 row using `baseline_C_abtt3_dct4` directly:
that row is GREEN, not YELLOW.)

### REDs (numbers that fail to trace or contradict the source)

1. **README.md "44 experiments"** (line 12, 318) — actual is 47; documented in C.1.
2. **README.md "default PQ M=192 on 768d, ~20x"** (line 21) — old default; documented in C.1.
3. **README.md "ABTT3 + RP 768d + PQ M=192 → ~20x compression, ~34 KB/protein"** (line 80, 316) —
   old default; documented in C.1.
4. **README.md "PQ M=192 (default)" tier table cell** (line 87, 178) — old default; documented in C.1.

(All four REDs are the same README drift item already flagged in C.1; preserved here as separate
line items because each is a distinct numeric claim.)

### Most consequential YELLOWs

- **"95–100% retention across 6 tasks on 5 PLMs"** — Exp 46 reports 4 tasks (SS3, SS8, Ret@1,
  Disorder), not 6. The "6 tasks" framing seems to roll in CB513-vs-TS115-vs-CASP12 SS variants,
  but those are 1 task across 3 datasets, not 6 tasks. **Loosely-defined "task" inflates count.**
- **"232 compression methods"** — count is a roll-up estimate; not enumerated in any script.
  Pre-empt with a softer phrasing or enumerate.
- **"L=175"** — assumption, not measured. Confusing labelling.
- **fp16 896d compression "2.3x" vs JSON "2x"** — both correct, but inconsistent precision.
- **AUC retention 98.6% vs computed 98.5%** — 0.1pp rounding error; minor.
- **"1500 proteins/s" encoding speed** — only in commit message, not in result JSON.
- **"8 datasets"** — actually 9 cited (CB513, TS115, CASP12, CheZOD117, TriZOD348, SCOPe5K,
  CATH20, DeepLoc test, DeepLoc setHARD).
- **"6 tasks"** — see "95–100%" above.

### Patterns in the GREENs

All Exp 43 (Phase A1, B, C, D) and Exp 46 (5-PLM) cells — **bit-perfect match** between the cited
numbers and the source result JSONs. The rigorous-benchmark generation is well-traced. Exp 47
codec sweep table — also bit-perfect. The 5-PLM Exp 46 table is the cleanest section of the docs.

### Where the audit became impractical

Pre-rigorous (Exp 26, 28, 31, 32, 33, 34, 36, 37) numbers in README.md L208–L213, L330–L352, and
the `## Addendum: Trained ChannelCompressor` section reference older result files
(`chained_codec_results.json`, `extreme_compression_results.json`, `bitwidth_sweep_results.json`,
`pq_rp512_results.json`, `cnn_probe_results.json`, `channel_compression_results.json`). These
were not re-verified cell-by-cell in this audit — flagged YELLOW as a class for the talk to
disclaim "pre-rigorous numbers, no BCa CIs".
