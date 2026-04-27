# Worktree state at audit start

`.worktrees/exp50-rigorous/`
- branch: exp50/rigorous-cath-split
- head: b9703382ca0eef88729f023c2e35182ad0ecf9ff
- status: not merged to main; revisit at Step D.3 (curate branches).
- untracked at audit time: three machine-local symlinks to main-repo data
  directories (`data/external`, `data/proteins`, `data/residue_embeddings`).
  Not staged — committing absolute-path symlinks would be wrong for any
  other checkout. Most recent commit on the branch is the RNS-module work
  (`feat(exp50): RNS module for embedding quality evaluation (Task 9)`);
  no uncommitted code work in the worktree.

## Task D.3 — Branch curation decision (2026-04-27)

### Branches in the repo

| Branch | Status | Decision |
|--------|--------|----------|
| `main` | active | stays — the talk's narrative lives here |
| `exp50/rigorous-cath-split` | in-flight (worktree at `.worktrees/exp50-rigorous`) | **stays** — real ongoing work; talk cites Exp 50 as in-progress |
| `worktree-agent-a072eca5` | leftover from earlier subagent run | **archive in place (keep, no merge, no delete)** — see below |
| `worktree-agent-a74f01b2` | leftover from earlier subagent run | same |
| `worktree-agent-aa88abaa` | leftover from earlier subagent run | same |

### Why the three `worktree-agent-*` branches are archived in place

Each holds a substantive commit from the early codec-exploration era
(Exp 26–29 period; all share base `a0c4310 feat: add float16 default codec
with Exp 27 benchmark`):

- `worktree-agent-a072eca5` → wavelet, CUR, channel-prune, zstd codecs
- `worktree-agent-a74f01b2` → tensor train, NMF, OT, TDA, SimHash, AA-residual codecs
- `worktree-agent-aa88abaa` → original int8 / int4 / binary / PQ / RVQ implementations
  (later consolidated into the unified `OneEmbeddingCodec`)

The shipping codec consolidates int4 / binary / PQ / fp16 into a unified
constructor (`OneEmbeddingCodec(d_out, quantization, pq_m, abtt_k)`). The
remaining techniques (wavelet, CUR, NMF, OT, TDA, SimHash, AA-residual,
tensor-train, channel-prune, zstd) were either confirmed inferior in later
experiments (Exp 27 / 29 / 44 / 47) or never adopted because they didn't add
Pareto-relevant value.

**No deletion** because:
- The branches contain real experimentation history; deleting destroys the
  audit trail of "we tried these things and they didn't beat the unified codec."
- A future researcher (or a Rost-lab visitor asking "did you try X?") can
  `git log <branch>` to see exactly what was attempted.

**No merge** because the work was either consolidated under a different name
or intentionally not adopted.

**Rename deferred** post-talk. A clean naming convention would be
`archive/2026-03-codec-exploration/<topic>` but renaming branches has
remote-tracking implications — out of scope for this audit. Phase D.3 simply
records the disposition.
