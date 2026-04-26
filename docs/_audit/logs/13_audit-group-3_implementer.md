# Audit Group 3 (C.6 + C.7) — Implementer Report

**Subagent type:** general-purpose
**Tasks:** C.6 (parameter intentionality), C.7 (claims register)

## Report

**Status:** DONE_WITH_CONCERNS

### C.6 — Parameter intentionality
- **Commit:** `67a0fe9` — `audit(params): defaults inventory + evidence trace`
- **Defaults inventoried:** 6 declared (`d_out`, `quantization`, `pq_m`, `abtt_k`, `dct_k`, `seed`) + 5 hidden (`pq_k=256`, `n_pcs=5`, `n_sample=50000`, `max_residues=500000`, `version=4`)
- **Distribution:** 7 GREEN / 4 YELLOW / 0 RED
- **Most surprising finding:** the binary/896d/abtt_k=0/auto-pq defaults all landed in commit `34e159a` titled `chore: gitignore results/, .claude/, large data files` — the **rationale-bearing** commit `8b1fbf1` is a sibling that doesn't touch `codec_v2.py` itself. A `git blame` on the default lines lands on a misleadingly-named hygiene commit. Hygiene issue, not correctness.
- **`d_out=896` and `dct_k=4` are YELLOWs** — both defensible but **interpolated, not directly measured**. No clean d_out ∈ {512, 768, 896, 1024} factorial sweep at fixed quantization exists; no formal K∈{2,4,8} retention sweep on the current preprocessed pipeline. Both are the natural Rost-lab follow-up questions; pre-empt with one sentence each.

### C.7 — Claims register
- **Commit:** `179c2a4` — `audit(claims): traceability register for CLAUDE/README numbers`
- **Total claims traced:** 78
- **Distribution:** 50 GREEN (64%) / 24 YELLOW (31%) / 4 RED (5%)
- **Headline result:** Exp 43/44/46/47 retention tables and CIs are **bit-perfect** against their source JSONs. Every cell of the rigorous-benchmark generation matches.
- **RED claims (all README drift, all sub-instances of the C.1 README-drift hotspot):**
  1. README "44 experiments" (line 12, 318) — actual is 47
  2. README "default PQ M=192 on 768d, ~20x" (line 21) — old default
  3. README "ABTT3 + RP 768d + PQ M=192 → ~20x compression, ~34 KB/protein" (line 80, 316)
  4. README tier-table "PQ M=192 (default)" cell (line 87, 178)
- **Most consequential YELLOWs:**
  - "232 compression methods" — roll-up estimate, not enumerated in any script
  - "6 tasks on 5 PLMs" — Exp 46 actually has 4 tasks (loose definition inflates count)
  - "8 datasets" — actually 9 cited
  - "1500 proteins/s" — only in commit message, not in result JSON
  - "L=175" — assumption from design spec, not measured (Exp 45 reports L_mean=156)
  - **TM-score Spearman 0.5742** retention from Exp 37 is **NOT cited anywhere** despite being in the same result file as the lDDT 100.7%/contact 106.5% headlines. Omission, not error — but a Rost-lab probe will land here.
- **No PRIORITY-1 invalidation.** The 37x compression, BCa CIs, multi-PLM 95–100% retention, all 5-PLM split fairness — all hold against the source data.

### Final repo state
- `git status --short`: only the 2 pre-existing untracked subagent log files (logs 11, 12 from Group 2 — to be committed by controller)
- `git log --oneline -5`:
  ```
  179c2a4 audit(claims): traceability register for CLAUDE/README numbers
  67a0fe9 audit(params): defaults inventory + evidence trace
  fde0fb1 audit(logs): save subagent reports for Phases A, B, C.1–C.5
  1f10f41 audit(phylo): document the 24-taxa rerun decision
  088bf37 audit(stats): bootstrap + probe + baseline conformance
  ```

### Cumulative posterior tally (after C.7)

| Subsection | GREEN | YELLOW | RED |
|---|---:|---:|---:|
| Repo hygiene (C.1) | 5 | 3 | 2 |
| Code correctness (C.2) | 6 | 3 | 0 |
| Splits (C.3) | 6 | 3 | 0 |
| Statistics (C.4) | 9 | 1 | 0 |
| Phylo (C.5) | 4 | 2 | 0 |
| Parameters (C.6) | 4 | 4 | 0 |
| Claims register (C.7) | 8 | 11 | 4 |
| **Total** | **42** | **27** | **6** |

(C.7's 4 RED line items are sub-instances of the same README drift item from C.1 — distinct numeric claims sharing one root cause. The "true" RED *root-cause* count remains 2.)

The cumulative ratio (~56% green / 36% yellow / 8% red) is more yellow than the prior predicted (~70/20/10) because C.6/C.7 surface "marketing-layer" count claims (232 methods, 6 tasks, 8 datasets, 1500 prot/s, L=175) and commit-message-only evidence chains. None invalidate a Rost-lab-critical headline; all are presentation polish for D.1.
