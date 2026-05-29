---
date: 2026-05-29
started_at: 2026-05-29T11:45:00+02:00
slug: dropbox-offload-and-harness
status: done
followups:
  - "Dropbox: once upload finishes, set ~/Dropbox/Science/ProteEmbedExplorations_bigfiles to online-only to reclaim ~100G local disk."
  - "Decide whether docs/ringverse.md (untracked, pre-existing) should be tracked or gitignored."
  - "Optionally delete merged branches exp55/vep-retention and exp56/codec-megasweep (worktrees already removed)."
  - "Tighten .claude/paperwork.yaml later (add must-be-modified-this-session) once the begin-session/debrief loop is routine."
ended_at: 2026-05-29T12:24:56+02:00
---

# Dropbox big-file offload + masterbook harnessing

## What we did
1. **Offloaded embeddings to Dropbox.** Moved all 34 `.h5` files (~100G: 5-PLM set + ProteinGym VEP) to `~/Dropbox/Science/ProteEmbedExplorations_bigfiles/residue_embeddings/`. `data/residue_embeddings` is now an absolute symlink there; the kept exp50 worktree chains through it. Local project footprint 109G → 5.8G.
2. **Collapsed worktrees.** Removed `exp55-vep-retention` and `exp56-codec-megasweep` (both merged into main, verified clean). Kept `exp50-rigorous` (10 unmerged commits, pushed to origin).
3. **Committed + pushed.** Added a no-trailing-slash `data/residue_embeddings` gitignore entry so the symlink stays ignored (commit `e6b06dc`); pushed `main` to `github.com/jcoludar/One-Embedding`.
4. **Student-activity check.** No third-party PRs/branches/forks on `One-Embedding` or `SpeciesEmbedding` — all activity is Ivan's.
5. **Harnessed the repo via the Plans_tasks_manager masterbook.** Wrote `CLAUDE.source.md` (tier-1/* + tier-2 python-research/gpu-thermal/bioinformatics-pipeline + substrates session-paperwork + paperwork-enforcement), ran `assemble.py` to generate `CLAUDE.md` + `AGENTS.md` + `.claude/{commands,hooks}` + merged `settings.json`. Moved the dense benchmark tables to `docs/BENCHMARKS.md`. Added `sessions/` + this log, and a conservative `.claude/paperwork.yaml` (blocking enforcement, but no fragile must-be-modified rule).

## Why
Ivan is pausing the project and wanted heavy files offloaded to Dropbox (to be set online-only) under one dedicated folder — not scattered piecemeal. He also wanted the repo wired into the same session-paperwork harness the other Cascade projects (CardGame1) use, routed through the masterbook.

## Decisions worth remembering
- Big files live ONLY in the Dropbox dedicated folder; relocate gitignored heavies there with an absolute symlink + a no-slash gitignore entry. Don't move git-tracked content (checkpoints JSONs, vendored tools/reference) — it dirties the repo.
- `CLAUDE.md` is now GENERATED — edit `CLAUDE.source.md` and re-run the assembler; never hand-edit `CLAUDE.md`.
- Deployed hooks need `chmod +x` (the directly-invoked ones); a non-executable Stop hook would block the session.

## Files / artefacts touched
- `CLAUDE.source.md` (new, tracked), `CLAUDE.md` (now generated), `AGENTS.md` (symlink), `docs/BENCHMARKS.md` (new)
- `.claude/{settings.json, paperwork.yaml, commands/, hooks/}` (deployed)
- `sessions/README.md`, this session log
- `.gitignore` (symlink ignore + generated-artifact ignores)
- `Plans_tasks_manager/science/one-embedding.md` (refreshed)

## Open threads / next steps
See `followups:` above.
