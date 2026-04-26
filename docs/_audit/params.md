# Parameter Intentionality Audit (Task C.6)

Inventory of every default in `OneEmbeddingCodec.__init__` (`src/one_embedding/codec_v2.py:84–93`).
For each: file:line of default, evidence chain, source experiment, commit hash, status.

## Defaults table

| Param | Default | File:line | Evidence | Source experiment | Commit | Status |
|-------|---------|-----------|----------|-------------------|--------|--------|
| `d_out` | `896` | codec_v2.py:86 | Exp 47 standard sweep selected 896d after Exp 45 found ABTT3+768d "catastrophe" on ProstT5 (SS3 drops to 85.6%). 896d preserves more variance (~88% of 1024) while still allowing ~37x binary compression. PQ M=224 chosen because 896 is divisible by many small factors (and 224 = 896//4). | Exp 47 | `34e159a` (default change in code), `8b1fbf1` (commit msg explanation) | YELLOW |
| `quantization` | `'binary'` | codec_v2.py:87 | Exp 47 `prot_t5_full` sweep: binary SS3 ret 97.6%, Dis 94.9% (better than PQ M=128's 97.5/91.4 at 32x). Exp 44 confirmed binary beats PQ M=128 on disorder (RaBitQ effect). Binary is decoder-self-contained (no codebook). | Exp 47, Exp 44 | `34e159a`, `8b1fbf1` | GREEN |
| `pq_m` | `None` (auto = `d_out // 4`) | codec_v2.py:88, auto rule codec_v2.py:46–57 | Heuristic `auto_pq_m(d_out)` returns "largest factor of d_out that is ≤ d_out//4". For d_out=896 → 224 → 4d sub-vectors → 18x. Explicit rationale: ~4d sub-vectors target. Documented in docstring (line 47). | Exp 44 (M=192 chosen at d=768//4); Exp 47 (M=224 at d=896//4) | `4d2750a` (set d//4 default), `34e159a` (api docstring) | GREEN |
| `abtt_k` | `0` | codec_v2.py:89 | Exp 45 disorder forensics: ABTT PC1 is 73% aligned with disorder direction. Per-stage decomposition: ABTT contributes 50% of total disorder loss (vs 26% RP, 24% PQ). Dropping ABTT recovers +3.3% disorder retention at 0.6pp retrieval cost. Documented in CLAUDE.md and EXPERIMENTS.md Phase 12. | Exp 45 | `8b1fbf1` (commit msg explicitly states "abtt_k=0 default"), `34e159a` (code change) | GREEN |
| `dct_k` | `4` | codec_v2.py:90 | Exp 22 path geometry tested K∈{1, 2, 4, 8} for retrieval and K∈{4, 8, 16} for SS3. Higher K monotonically improves retrieval but doubles dim every step. K=4 chosen as compression-vs-quality sweet spot — gives (D × 4) protein vector. Note: Exp 22 K=8 actually scored higher Ret@1 (0.712 vs 0.666) but at 2× dim cost. EXPERIMENTS.md line 90: "DCT K=4 is the sweet spot. Higher K hurts" — this contradicts the Exp 22 raw numbers (K=8 better than K=4 on prot_t5_xl), but K=4 was the chained-codec sweet spot per Exp 26 ("rp512+dct_K4 → Ret@1 0.780, SS3 0.815"). No formal K=4-vs-K=8 retention comparison on the modern preprocessed pipeline. | Exp 22, Exp 26, Exp 29 | `f30c72f` (initial K=4 in toxprot bundle), `7902735` ("core codec — ABTT3 + RP512 + DCT K=4 (the jewel)") | YELLOW |
| `seed` | `42` | codec_v2.py:91 | Exp 29 `part_D` formally tested 10 RP seeds (42, 123, 456, 789, 0, 7, 99, 2024, 31415, 271828): Ret@1 mean 0.7787, std 0.004; min 0.7718, max 0.7871. Seed=42 (s42) Ret@1 = 0.780, SS3 = 0.815 — both within 1 SD of mean. Seed choice is irrelevant beyond reproducibility. | Exp 29 part_D | `fdd0f5b` ("Multi-seed RP: 0.779 ± 0.004 — very stable") | GREEN |

## Hidden / derived defaults (not in __init__ signature but worth flagging)

| Param | Default | File:line | Evidence | Status |
|-------|---------|-----------|----------|--------|
| `pq_k` (PQ centroids per subquantizer) | `256` | codec_v2.py:119 | Hardcoded — standard PQ uses 256 (so each code fits in 1 byte / uint8). No experiment varies this. Industry standard from Jégou et al. 2011. | GREEN |
| `compute_corpus_stats(n_pcs=5)` | `5` | codec_v2.py:167 | Hardcoded to fit top 5 PCs even when `abtt_k > 0`. C.2 audit (codec_review.md) flagged: setting `abtt_k=10` would silently use only 5 PCs (slice truncation, no warning). | YELLOW |
| `compute_corpus_stats(n_sample=50_000)` | `50000` | codec_v2.py:167 | Sampling cap for PCA fit — chosen as a compute/quality tradeoff, no experiment formally varies it. ABTT cross-corpus stability (Exp 43, `run_abtt_stability.py`) confirmed PC subspace varies but downstream Ret@1 is stable < 0.2pp across 4 different fitting corpora — so sample size effect is bounded by that. | GREEN |
| `fit(max_residues=500_000)` | `500000` | codec_v2.py:159 | Sampling cap for PQ codebook fitting (passed to k-means). Standard practice; no formal sweep. | GREEN |
| `version` | `4` | codec_v2.py:255 | Hardcoded format version. C.2 noted no upgrade path documented for older versions. | YELLOW |

## Commit history of the current defaults

The current "Four knobs (d_out=896, binary, pq_m=auto, abtt_k=0)" defaults all landed in the same compound commit:

- `34e159a` (Apr 1 2026, `chore: gitignore results/, .claude/, large data files`)
  — **Despite the misleading "chore" subject line, this commit also changed the codec defaults from
  `(d_out=768, quant='pq', no abtt_k param)` to `(d_out=896, quant='binary', abtt_k=0)`** plus
  324-line update to `codec_v2.py` (the bulk of the API rewrite).
- `8b1fbf1` (Apr 1 2026, `feat: Exp 45-47 — disorder forensics, 5-PLM pipeline, binary default`)
  — sibling commit that **explains** the rationale in its message but does NOT touch `codec_v2.py`
  itself. The two commits travel as a pair (same author timestamp, same date).

The mismatch between subject line and content of `34e159a` is a hygiene issue (commit
discipline), not a correctness issue. **Worth flagging in handoff** so a future reader doesn't
assume the binary default arrived in `8b1fbf1` (the message-bearing commit) rather than `34e159a`
(the code-bearing commit).

## Per-default narrative

### `d_out=896` — YELLOW

**Why YELLOW, not GREEN:** the choice of 896 specifically (rather than 768 or 1024) is justified by
"divisible by many small factors" and "preserves more variance than 768d after Exp 45 ABTT removal"
— but no experiment **isolates** d_out as a variable holding everything else fixed. Exp 47 sweeps
6 configs at 896d but only 2 at 1024d (lossless + PQ M=256) and 2 at 768d (old defaults with ABTT3).
A clean d_out ∈ {512, 768, 896, 1024} sweep at the same quantization (e.g. all binary, no ABTT)
does not exist in the result JSONs. The current default is **defensible** (Exp 47 PQ M=224 at 896d
hits 99.0/98.5/100.6/95.4) but the d_out choice itself is **interpolated**, not directly measured.

A Rost-lab probe — "why 896 rather than 1024 (slightly better fidelity, marginal storage cost) or
768 (the old default, well-validated)?" — has only the **inferred** answer "896 is the sweet spot
for PQ-divisibility AND avoids ABTT3+768d's ProstT5 catastrophe." Cite this verbatim in the talk.

### `quantization='binary'` — GREEN

Best-evidenced default. Exp 47 sweep on prot_t5_full directly compares binary-896 (37x) vs
PQ128-896 (32x) vs PQ224-896 (18x) vs int4-896 (9x) vs fp16-896 (2x) vs lossless-1024 (2x). Binary
holds 97.6% SS3 / 94.9% Dis / 100.4% Ret@1 with no codebook — strictly Pareto-better than PQ128 on
Dis. Five-PLM Exp 46 (separate result file) further confirms binary at 17 KB/protein. The
"~20x faster than PQ" claim has no direct timing comparison in result JSONs (only `binary` total
229.7s vs `pq224` 529.0s in Exp 47 — a 2.3x ratio that includes all benchmark overhead, not pure
encode kernel). The 20x figure is a commit-message claim only.

### `pq_m='auto'` (= d_out // 4) — GREEN

Heuristic is documented in code (line 47), tested in both Exp 44 (`d_out=768 → M=192`) and Exp 47
(`d_out=896 → M=224`). Both selections correspond to ~4d sub-vectors and ~18-20x compression. The
"~4d sub-vectors" target is informally tied to PQ literature (Jégou 2011 cites 8-bit codes per
subquantizer; sub-dim=4 keeps each subspace small enough for 256 centroids to cover well). No
explicit M-sweep with the d=896 default in the result JSONs (M=128, 224 are tested; M=192, 256
are also nearby points), but the heuristic outputs the M=224 value Exp 47 directly validated.

### `abtt_k=0` — GREEN

Best-evidenced behavioural change. Exp 45 disorder forensics (`exp45_new_default_results.json`,
commit `8b1fbf1` message section "Exp 45") establishes:
- ABTT PC1 cosine-aligned 73% with the disorder Ridge probe direction.
- Per-stage decomposition: ABTT 50%, RP 26%, PQ 24% of total disorder loss.
- Dropping ABTT: +3.3pp disorder retention, -0.6pp retrieval (net win for disorder).

ABTT remains accessible as `abtt_k=3` for retrieval-only use cases. Both the decision and the
trade-off cost are documented. Headline-defensible.

### `dct_k=4` — YELLOW

Status downgraded from GREEN because EXPERIMENTS.md states "DCT K=4 is the sweet spot. Higher K
hurts" — but Exp 22 raw `path_geometry_results.json` shows K=8 retrieval > K=4 retrieval (0.712 vs
0.666 on the displacement-DCT proxy). Modern chained-codec results (Exp 26) used K=4 directly
without a K-sweep. **No formal K∈{2, 4, 8} retention sweep on the current preprocessed
pipeline (center → RP896 → binary → DCT K).** A Rost-lab "why not K=8?" probe will land here. The
**only** clean defense is "K=4 keeps protein_vec at D × 4 (= 3584 floats × fp16 = 7 KB), doubling
to K=8 doubles that to 14 KB protein_vec for marginal retrieval gain." Frame as a compression-side
choice, not a quality-optimal one.

### `seed=42` — GREEN

Cleanest defaults of all. Exp 29 part_D explicitly tested 10 seeds; std=0.004 on Ret@1 means seed
choice is irrelevant. Documented as "deterministic — fixed seed for reproducibility" in README.

## Summary

**6 declared defaults inventoried + 5 hidden defaults flagged.**

Distribution:
- 4 GREEN (`quantization`, `pq_m`, `abtt_k`, `seed`) + 3 hidden GREEN (`pq_k`, `n_sample`, `max_residues`)
- 2 YELLOW (`d_out`, `dct_k`) + 2 hidden YELLOW (`n_pcs=5`, `version=4`)
- 0 RED

Both YELLOWs share the same root cause: **the current defaults emerged from a sequence of
narrowing experiments (Exp 22 → 26 → 29 → 44 → 45 → 47), not from a single clean factorial sweep**.
For the lab talk, both are defensible with one sentence each — but neither is GREEN-by-evidence.

The YELLOW for `d_out=896` is the highest-priority one to pre-empt: it will be the natural follow-up
question to "why binary, not PQ?". The YELLOW for `dct_k=4` is presentation-soft (auditor question:
"why 4, not 8?" — answer: storage size of protein_vec).

## Cross-references

- C.2 (`codec_review.md`) already flagged hidden defaults `n_pcs=5`, `version=4`, silent skip
  in `_preprocess` when `is_fitted=False`. Those are repeated here for completeness.
- C.4 (`stats.md`) confirmed all 7 cited Exp 43/44/46/47 tables match the result JSONs — the
  evidence chain for each default's cited number is verified via that audit.
