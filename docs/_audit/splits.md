# Splits Audit (Task C.3)

**Date:** 2026-04-26
**HEAD at audit:** `1ff3a71`

Goal: verify every cited benchmark uses a leakage-controlled train/test partition,
that Exp 46 (multi-PLM) uses identical splits across PLMs, and that the
random-split corner cases are honestly understood.

## Evidence index

- Split files: `data/benchmark_suite/splits/*.json`
- Split summary: `data/benchmark_suite/splits/split_summary.json`
- SCOPe metadata: `data/proteins/metadata_5k.csv` (id, family, superfamily, fold, class)
- CATH labels: `data/external/cath20/cath20_labeled.fasta` (`>id|C.A.T.H` headers)
- Existing leakage probe: `experiments/validate_split_leakage.py` (MMseqs2-based,
  used during Exp 17 / 24 split validation; not re-run for this audit since the
  reused splits inherit its conclusions)

## Per-experiment split provenance

### Exp 43 (rigorous benchmark — `experiments/43_rigorous_benchmark/`)

| Dataset | Split file | Type | Train | Test | Leakage status |
|---|---|---|---:|---:|---|
| CB513 (SS3/SS8) | `cb513_80_20.json` | random 80/20 (seed=42) on a CB513-internal partition | 408 | 103 | dataset is `<25 % seq id` by construction (CB513 design); within-CB513 random split adds ~no extra structural leakage |
| CheZOD (disorder) | `chezod_seth.json` | predefined SETH split (Dass et al. 2020) | 1174 | 117 | non-redundant by design |
| TriZOD (disorder) | `trizod_predefined.json` | predefined TriZOD348 split (Haak 2025) | 5438 | 348 | cluster-based, predefined |
| SCOPe 5K (retrieval / codec fit) | `esm2_650m_5k_split.json` | superfamily-aware (split summary records `superfamily_overlap=0`, `family_overlap=0`) | 1643 | 850 | strict — no superfamily overlap; means BOTH retrieval queries AND fitted ABTT/PQ codebook never see the test families |
| TS115 / CASP12 (SS3/SS8) | none — full datasets used as out-of-sample tests with the CB513 train fold as training | n=115 / n=20 | full | full | dataset-design leakage controls are the original publication's |
| CATH20 (Phase C retrieval) | none — full set used (filtered to superfamilies with ≥3 members) | full | full | within-CATH20 (CATH 4.2 S20 nonredundant; pre-clustered) |
| DeepLoc test / setHARD (Phase C localization) | published DeepLoc splits (LightAttention `data_files/`) | predefined | 9503 | 2768 / 490 | cluster-based by DeepLoc authors |

**Verdict:** [GREEN] for all but CB513-internal split, which is [YELLOW] in the
strict cluster-leakage sense (random within an already-curated `<25 % id` set —
no statistically harmful leakage, but a Rost-lab Q probably points here first).

### Exp 44 (unified codec sweep — `experiments/44_unified_codec_benchmark.py`)

Reuses Exp 43's split files via `experiments/43_rigorous_benchmark/config.SPLITS`
(lines 36–39 in `44_unified_codec_benchmark.py`):

```python
from config import (RAW_EMBEDDINGS, SPLITS, LABELS, METADATA, RESULTS_DIR, …)
…
cb513_split = load_split(SPLITS["cb513"])
scope_split = load_split(SPLITS["scope_5k"])
```

The codec is fitted on `scope_train_ids` only (line 231). Test proteins never
touch the codec fitting set. **Same partition as Exp 43 — verified by file
identity.**

**Verdict:** [GREEN] (inherits Exp 43's split protocol).

### Exp 46 (5-PLM benchmark — `experiments/46_multi_plm_benchmark.py`)

The headline claim is "same split across all 5 PLMs." Verified:

`run_benchmark_suite` (lines 456–461) loads the splits once per PLM call,
reading the **same JSON files** from disk:

```python
with open(DATA / "benchmark_suite" / "splits" / "cb513_80_20.json") as f:
    cb_split = json.load(f)
with open(DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json") as f:
    sc_split = json.load(f)
```

The PLM identity only enters via `emb_path(plm_name, dataset)` (line 326),
which selects the H5 of pre-extracted embeddings. The train/test ID lists are
PLM-agnostic. The codec fit (line 480) uses `sc_train` filtered for that PLM's
embedding availability — so a missing protein in one PLM only shrinks that
PLM's training set; it does NOT introduce a different test set.

**Verdict:** [GREEN] — single-source-of-truth split, applied across all 5 PLMs.

One [YELLOW] caveat worth noting: the SCOPe split file is named
`esm2_650m_5k_split.json`, a historical artifact (it was originally generated
during ESM2 work at Exp ~17). The split itself is PLM-agnostic; the filename
is misleading. Worth renaming as a Phase D polish item.

### Exp 47 (codec config sweep — `experiments/47_codec_sweep.py`)

Same pattern as Exp 46 — loads the same split files at lines 270–272, fits the
codec once per config on `sc_train_embs`. The SAME raw benchmark is computed
once per PLM (lines 292–303) and reused for retention against every config —
this is intentional efficiency, doesn't affect fairness.

**Verdict:** [GREEN] (inherits Exp 43's split protocol).

### Exp 50 (sequence → One Embedding CNN — `experiments/50_sequence_to_oe.py`)

`load_data` (lines 82–94) does a **random 80/10/10 split** with `seed=42` on
the SCOPe 5K protein IDs:

```python
rng = np.random.RandomState(42)
ids = np.array(common); rng.shuffle(ids)
n_train = int(0.8 * n); n_val = int(0.1 * n)
train_ids = set(ids[:n_train]); …
```

There is no leakage filter applied. The current Stage 1/2/3 results in
`results/exp50/stage1_results.json` were measured under this random split.

**Verdict:** [YELLOW] for the *Sighting* (sketched) results currently in the
repo; the design spec at `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md`
explicitly identifies this as the weakness and the rigorous CATH-cluster re-run
(`docs/superpowers/plans/2026-04-06-exp50-rigorous-cath-split.md`) plans to
replace it. **Per-task instruction we defer to that separate plan; the
talk should cite Exp 50 as in-progress rather than as a final number.**

The Exp 50 plan's Task 6 ("MMseqs2 leakage audit script") — see lines 906–1083
of that plan — has NOT been executed yet (no `results/exp50_rigorous/` dir
exists). The audit there will run once the rigorous CATH-split re-run is done.

## Cross-cutting checks

- **Train/test ID uniqueness assertion.** `experiments/43_rigorous_benchmark/rules.py:73`
  defines `check_no_leakage(train_ids, test_ids)` which raises if any ID
  appears in both. It is called at the top of every per-residue benchmark
  (`runners/per_residue.py:267, 351, 467`). [GREEN]
- **Codec fit corpus is external to test sets.** All cited tables fit
  ABTT/PQ on the SCOPe 5K **train** subset, then evaluate on CB513 / CheZOD /
  TriZOD / SCOPe-test — distinct datasets or held-out IDs. Cross-corpus
  stability separately verified by Exp 43's `run_abtt_stability.py`. [GREEN]
- **Family-keyed retrieval ranking respects the SCOPe hierarchy.**
  `compute_protein_vectors → run_retrieval_benchmark` ranks by cosine
  similarity over **all** SCOPe-5K vectors and checks the top-1 hit's
  `family` field. Train/test split governs the codec fitting only — at
  retrieval time every protein is a candidate (correct: this is closed-pool
  retrieval, not "test query → train pool"). [GREEN]

## Severity tally

- [GREEN] 8 — Exp 43 (most splits), Exp 44, Exp 46 (split-shared-across-PLMs claim
  verified), Exp 47, train/test uniqueness assertion, external codec fit
  corpus, retrieval ranking semantics, the `superfamily_overlap=0` SCOPe split
- [YELLOW] 3 — CB513 random within-`<25%`-curated split (lab-Q magnet), Exp 46
  split filename `esm2_650m_5k_split.json` is misleading, Exp 50 random
  80/10/10 (acknowledged, designed-around)
- [RED] 0

## Open follow-ups for the talk (not for this audit)

- Phase D: rename `esm2_650m_5k_split.json` → `scope_5k_superfamily_split.json`
  and update all loaders.
- Phase E (Exp 50 rigorous re-run): execute the MMseqs2 leakage audit script
  from `2026-04-06-exp50-rigorous-cath-split.md` Task 6.
- Talk slide: when introducing CB513 numbers, briefly add
  "CB513 is `<25 % id` by design; we use the published 408/103 random split
  with seed=42 for the train/test partition." Pre-empts the obvious Q.
