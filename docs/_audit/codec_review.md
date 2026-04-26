# Codec V2 Review (Task C.2)

**Date:** 2026-04-26
**File:** `src/one_embedding/codec_v2.py` (529 lines)
**HEAD at audit:** `dd2409d` (audit C.1 commit) over `7cc2e72`.

## Defaults inventory

Constructor signature (lines 84–93). All defaults derived from `OneEmbeddingCodec.__init__`.

| Param | Default | Justification source | Comment |
|---|---|---|---|
| `d_out` | `896` | TBD (Exp 47 sweep) | Headline: "RP to 896d, ~37x compression". Sets the output dim of random projection; if `>= d_in`, RP is skipped (line 152). |
| `quantization` | `"binary"` | TBD (Exp 47 + Exp 45 disorder finding) | Switched from `"pq"` (M=192) to `"binary"` per session_20260401. Binary needs no codebook → "h5py + numpy only" claim. |
| `pq_m` | `None` (auto via `auto_pq_m(d_out)`) | Inline doc lines 47–52 (largest factor of `d_out` ≤ `d_out//4`) | Only used if `quantization=='pq'`. For 896 → 224 (4d sub-vectors); 768 → 192; 512 → 128. |
| `abtt_k` | `0` | TBD (Exp 45 — disorder forensics, ABTT PC1 is 73 % aligned with disorder direction) | Off by default. Setting to 3 enables ABTT3 preprocessing (legacy 1.0 default). |
| `dct_k` | `4` | TBD | DCT coefficients per channel for `protein_vec`. K=4 = 4d-rolling summary; protein_vec has shape `(d_out * dct_k,)`. |
| `seed` | `42` | Repo convention | Determines RP matrix and PQ k-means init. |
| `codebook_path` | `None` | n/a | Optional fast-path to skip `fit()` when a pre-fitted codebook exists. |

The `auto_pq_m(d_out)` helper (lines 46–57) returns the largest factor of `d_out` that is `<= d_out // 4`. For `d_out=896`, that's 224. **Comment on lines 50–51 is stale**: "For d_out=768: returns 192 (4d sub-vectors, 20x compression). For d_out=512: returns 128 (4d sub-vectors, 32x compression)." — does not mention the current default 896. **YELLOW** — fix during D.1.

The class docstring (lines 64–82) is correctly aligned with current defaults (896d / binary / abtt_k=0).

## Suspicious / unclear lines

- `codec_v2.py:50–51` — `auto_pq_m` docstring examples still anchor on 768 / 512. Add 896 example. **YELLOW**.
- `codec_v2.py:113` — when `pq_m` does not divide `d_out`, error message uses `range(1, d_out + 1)` and filters factors `<= d_out // 2`. For `d_out=896` this lists 16 factors; the slice `factors[-10:]` shows the largest 10. Marginally helpful — a reader has to guess the intent. Comment would help. **YELLOW**.
- `codec_v2.py:141` — `Q * np.sqrt(d_in / self.d_out)` — variance-preserving scaling for projection. No comment. Implicit assumption: `d_in > d_out`. If `d_in < d_out` we never reach this line because `_preprocess` skips RP (line 152), so the scaling is mathematically only ever applied with reduction. Worth a one-line comment. **YELLOW**.
- `codec_v2.py:152` — `if self.d_out >= raw.shape[1]: return raw` — silent skip. A user passing `d_out=1024` to a 1024d PLM will get unprojected output, which is correct behavior (lossless mode), but the surface API does not log/warn. The class docstring at lines 73–74 mentions this; the inline `_preprocess` doc on line 145 is too terse ("(skip RP if d_out >= d_in)"). Acceptable.
- `codec_v2.py:241–245` — `encode()` requires fit when PQ is selected. Good early failure. Same check is duplicated at lines 479–483 in `encode_h5_to_h5`. Acceptable but could share a helper.
- `codec_v2.py:147–148` — `if self._corpus_stats is not None: raw = center_embeddings(raw, self._corpus_stats["mean_vec"])`. **Implicit:** if `fit()` was never called and quantization is binary or int4 (no codebook required), centering is silently SKIPPED. This means the user is encoding raw embeddings without centering — which is inconsistent with the class docstring claim "default: center + RP 896d + binary". **YELLOW** — could surprise a careful user. Possibly worth raising or warning when `is_fitted == False` and a non-PQ quantization is in use.
- `codec_v2.py:166–168` — `compute_corpus_stats(embeddings, n_sample=50_000, n_pcs=5, seed=self.seed)`. The `n_pcs=5` is hardcoded but `abtt_k` is configurable up to whatever the user sets (default 0). If a user instantiates with `abtt_k=10`, the slice on line 150 (`top_pcs[:self.abtt_k]`) will run off the end (Python slice gracefully truncates). Won't crash but will silently use only 5 PCs while the user asked for 10. **YELLOW** — should validate or pass `abtt_k` into the corpus-stats computation.
- `codec_v2.py:249` — `protein_vec = dct_summary(projected, K=self.dct_k).astype(np.float16)` — protein vector is computed *after* projection but *before* quantization. This means `protein_vec` is fp16 of the projected (not quantized) embedding — same regardless of which quantization is selected. Good for retrieval consistency (Exp 47 100 % retrieval retention) but worth noting that the protein_vec used for retrieval is essentially "lossless" relative to per-residue. Worth a comment.
- `codec_v2.py:273` — `result["per_residue_fp16"] = projected.astype(np.float16)` — when `quantization=None`, per-residue is fp16 not fp32. So "lossless" mode is actually fp16, not bit-perfect float32. Consistent with class docstring (line 76: "None = fp16"); not a bug, just confirming.
- `codec_v2.py:255` — `"version": 4`. Hardcoded. No upgrade story for older `.one.h5` files in this code (a v3 file would parse but probably break elsewhere). Out of scope for this audit. **YELLOW** — note for paper.
- `codec_v2.py:496–505` — `encode_h5_to_h5` writes file-level metadata containing `quantization`, `d_out`, etc. *but* per-protein groups only get `seq_len` and `d_in` attrs (lines 524–525). `load_batch` correctly merges them (lines 466–471). This means the per-protein view of metadata is correct only via `load_batch`. Direct inspection of `f[pid].attrs` from external code would miss `quantization`, `d_out`, etc. **YELLOW** — could trip up a Rost-lab user inspecting our files manually. Worth a `load_batch_iter` documentation note.

## Receiver-side decode check

**Question:** can a third party decode our `.one.h5` files using only `numpy + h5py`?

### For binary default (Quantization = "binary", abtt_k=0)

- The H5 file contains: `protein_vec` (float16), `per_residue_bits` (uint8 packed), `per_residue_means` (float32, D), `per_residue_scales` (float32, D). Per `_write_per_residue_to_h5` lines 336–343.
- Decoding requires: bit-unpacking (shifts 7..0, mask, reshape), then `signs = bits*2-1`, then `signs * scales + means`. **All numpy operations.**
- The receiver can do this with **only** `h5py + numpy` IF they know the bit layout (bit 7 → col 0, bit 0 → col 7, packed little-endian-within-byte but big-endian-within-bit-position). The codebase implements this in `dequantize_binary` (`quantization.py:319-346`) — about 12 lines of numpy.
- **No codebook file is read.** No `OneEmbeddingCodec` import is required at all (the static method `OneEmbeddingCodec.load(...)` calls `dequantize_binary` internally, but the algorithm could be re-implemented externally in ~12 lines).
- **VERIFIED: yes, h5py + numpy alone are sufficient for the binary default.** The CLAUDE.md/README.md claim "Receiver needs `h5py` and `numpy` only" is technically correct — but the receiver also needs to know (or copy) the ~12 lines of decode logic. The docs do not provide a self-contained decode snippet. **YELLOW** — for the paper, ship a 15-line standalone "binary decoder" example.

### For PQ mode (max-quality, ~18x)

- The H5 file contains: `pq_codes` (uint8, L×M).
- Decoding requires: a separate codebook H5 file with `pq_codebook` (M, K, sub_dim) + `pq_M`, `pq_K`, `pq_sub_dim`, `pq_D` attrs. See `_read_per_residue_from_h5` lines 388–400 and `pq_decode`.
- `pq_decode` in `quantization.py` is also pure numpy (a gather + concatenate over sub-vectors).
- **NOT pure h5py + numpy with one file** — needs the codebook. The CLAUDE.md doc clearly says "PQ needs codebook"; this matches reality. GREEN.

### For int4 mode

- Requires `per_residue_data`, `per_residue_scales`, `per_residue_zp`. `dequantize_int4` is pure numpy. **No codebook.** GREEN.

### For lossless / fp16 mode (`quantization=None`)

- Requires `per_residue` only. `astype(np.float32)`. **No codebook.** GREEN.

### Summary of receiver-side claim

- The "h5py + numpy only" headline claim **holds** for binary (default), int4, and lossless/fp16.
- It does **not** hold for PQ — needs codebook H5. Documented correctly.
- The README.md still has a stale "h5py + numpy + codebook" comment in its Quick Start (line 27–30) tied to the old PQ default. This is C.1 finding — fix in Phase D.

## Other observations

- The encode path is well-tested (40 tests in `test_codec_unified.py` per pytest baseline; 11 in `test_codec.py`; 15 in `test_codec_unified` related; also `test_io_one_h5.py` covers H5 round-trip).
- Pure-static methods (`load`, `load_batch`, `_read_per_residue_from_h5`, `_write_per_residue_to_h5`) are correctly factored — `load_batch` does not need an instantiated codec, which supports the "receiver needs no codec" story for binary.
- `auto_pq_m` correctness for large `d_out`: for 896 returns 224 (`896 // 4 = 224`, `896 % 224 == 0`). For 1024 returns 256. For odd or prime values would degrade gracefully (returns 1 if no factor exists, but odd primes never reach `quantization='pq'` in practice). OK.
- `_proj_cache` (line 104) caches the projection matrix per `d_in`. Allows multiple input dims (e.g. ProtT5 1024, ESM2 1280, ESM-C 1152) to share the same codec instance. Good design.
- `__repr__` (lines 127–132) is informative.

## Net summary for talk

- **Code is clean.** Zero TODO/FIXME/HACK markers in `src/`. 813/813 tests pass.
- **Receiver claim verified** for binary default (no codebook required, pure numpy + h5py).
- **Doc nits exist** but are minor and localised: stale comment in `auto_pq_m`, no warning when `is_fitted == False`, hardcoded `n_pcs=5` vs configurable `abtt_k`. None affect headline claims.
- **No surprising error paths.** The duplicated PQ-fit check is mildly redundant but defensive.
