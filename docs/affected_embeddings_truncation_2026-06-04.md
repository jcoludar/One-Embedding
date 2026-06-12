# Truncation-affected embeddings — needs re-embedding (2026-06-04)

A silent ProtT5/ANKH tokenizer truncation (`truncation=True, max_length=512`/`514`)
was found and fixed on 2026-06-04 (branch `fix/plm-tokenizer-truncation`). Every
embedding produced by a truncating extractor is suspect: proteins longer than the
cap were silently chopped (the embedding has fewer rows than the protein, with no
error). These files must be **re-embedded before reuse**.

**Storage:** `data/residue_embeddings/` is a symlink into
`~/Dropbox/Science/ProteEmbedExplorations_bigfiles/residue_embeddings/` (files are
Dropbox online-only). **Recovery if deleted:** Dropbox "Deleted files" / version
history (30 days), or re-extract with the now-fixed code.

## AFFECTED — delete + re-embed (19 files)

ProtT5-XL — extractor capped at 512 (`src/extraction/prot_t5_extractor.py`, now fixed):
- `prot_t5_xl_casp12.h5`
- `prot_t5_xl_cath20.h5`
- `prot_t5_xl_cb513.h5`
- `prot_t5_xl_chezod.h5`
- `prot_t5_xl_deeploc.h5`   ← highest exposure (full-length proteins, many >512)
- `prot_t5_xl_trizod.h5`
- `prot_t5_xl_ts115.h5`

ProstT5 — routed through the same 512-capped ProtT5 extractor (Exp 46 `extractor="prot_t5"`):
- `prostt5_casp12.h5`
- `prostt5_cb513.h5`
- `prostt5_chezod.h5`
- `prostt5_scope_5k.h5`
- `prostt5_trizod.h5`
- `prostt5_ts115.h5`

ANKH-large — Exp 46 inline tokenizer capped at 514 (now fixed):
- `ankh_large_casp12.h5`
- `ankh_large_cb513.h5`
- `ankh_large_chezod.h5`
- `ankh_large_scope_5k.h5`
- `ankh_large_trizod.h5`
- `ankh_large_ts115.h5`

## CLEAN — keep (verified not truncated)

- `prot_t5_xl_proteingym_clinvar.h5`, `prot_t5_xl_proteingym_diversity.h5` — VEP
  (Exp 55b). Both callers pre-filter WT length (<900 DMS, <500 ClinVar) *below* the
  latent 1024 cap, so truncation never fired.
- `esm2_650m_*.h5` (casp12, cb513, chezod, small500, trizod, ts115) — ESM2 via
  fair-esm `batch_converter`; no `max_length`, full sequence.
- `esmc_300m_cb513.h5`, `esmc_600m_*.h5` (casp12, cb513, chezod, scope_5k, trizod,
  ts115) — ESM-C silently truncated only >2000 aa; no protein in these datasets
  exceeds 2000, so they were never actually truncated.

## Re-embed how

Re-run extraction with the fixed code (now skip-not-truncate, full length):
`experiments/46_multi_plm_benchmark.py` (`extract_all`) regenerates the benchmark
matrix; or `src/one_embedding/extract` for the OE path. Downstream benchmark JSONs
under `data/benchmarks/` derived from these embeddings should be regenerated after
re-embedding.

## Caveat

Retention *ratios* (the headline 95–100% numbers) are unaffected — both codec and
raw baseline were computed from the same truncated embeddings, and the per-residue
probes truncate labels to the same 512 window, so truncation cancels. Re-embedding
matters for **absolute** long-protein numbers and any cross-PLM absolute comparison.
