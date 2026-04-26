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
