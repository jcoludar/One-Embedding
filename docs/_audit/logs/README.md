# Audit Logs — Subagent Dispatch Records

Each file in this directory captures the report from one subagent dispatch made during the lab-talk prep audit (Phase C of the plan at `docs/superpowers/plans/2026-04-26-lab-talk-prep.md`).

## Naming convention

`NN_<phase>_<role>.md`

Where:
- `NN` is a two-digit sequence number across all dispatches (so files sort chronologically).
- `<phase>` identifies which plan phase or audit group the dispatch belongs to (e.g. `phase-a`, `phase-b`, `audit-group-1` for Tasks C.1+C.2, `audit-group-2` for C.3+C.4+C.5).
- `<role>` is one of: `implementer`, `spec-review`, `code-review`.

## Why these exist

The user requested explicit logs so the audit trail is reproducible: every finding, every reviewer judgment, every implementer's self-review can be re-read later. The audit findings themselves live in `docs/AUDIT_FINDINGS.md` and the structured evidence in `docs/_audit/<topic>.md`; these logs add the *process* layer (what was asked, what was reported, what reviewers said) so the audit can be re-litigated post-talk if needed.

## Index

(filled as logs are written; see file timestamps for chronology)
