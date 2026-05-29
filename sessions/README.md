# Sessions

One file per Claude Code session — a running record of what we did, why, and what's pending. Created at `/begin-session`, closed at `/debrief`. The `paperwork-enforcement` Stop hook (`.claude/paperwork.yaml`) checks today's log before letting the session end.

## Naming

`sessions/YYYY-MM-DD-<slug>.md` — date first, kebab-case slug after. Multiple sessions per day are fine; differentiate by slug.

## Template

```markdown
---
date: 2026-05-29
started_at: 2026-05-29T14:30:00+02:00   # filled by /begin-session, don't hand-edit
ended_at: 2026-05-29T16:15:00+02:00     # filled by Stop hook, don't hand-edit
slug: short-name-here
status: in_progress                      # in_progress (live) → done | paused (at /debrief)
followups: []                            # short bullets the next session needs
---

# <Session title>

## What we did
## Why
## Decisions worth remembering
## Files / artefacts touched
## Open threads / next steps
```

`started_at` is written by `/begin-session`; `ended_at` is written automatically by the `session_stop_log_timing.py` Stop hook — don't hand-edit either. Set `status` honestly at `/debrief`: `done` (all planned work landed) or `paused` (safe stopping point; `followups` describe the next concrete step).

## Reading order at session start

1. The most recent file in this folder — what just happened.
2. Auto-memory at `~/.claude/projects/-Users-jcoludar-CascadeProjects-ProteEmbedExplorations/memory/MEMORY.md`.
3. `CLAUDE.md` (generated from `CLAUDE.source.md`) for project context.
