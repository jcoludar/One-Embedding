---
date: 2026-06-04
started_at: 2026-06-04T09:00:00+02:00
slug: oe-autoeval-grab-reports
status: paused
followups:
  - "SWEEP INCOMPLETE — runs server-side on LRZ. DONE (reports collected + local at results/oe_autoeval_reports/): ESM2-8M raw+OE, ESM2-35M raw+OE, ESM2-3B OE. PENDING on A100 (Priority, queued since 2026-06-03): ESM2-150M raw+OE (5665516/17), ESM2-650M raw+OE+norp (5665518/19/20), ProtT5 raw+OE+norp (5665523/24/25). RESUBMITTED: ESM2-3B RAW = job 5666300 (was OUT_OF_MEMORY at 120G; arm.sbatch --mem now 320G)."
  - "RESUME = re-run the grab when more arms finish: ssh ai 'python3 .../collect_reports.py .../out .../reports'; rsync -a ai:.../reports/ results/oe_autoeval_reports/; then .venv/bin/python scripts/lrz_oe_autoeval/compare_reports.py --raw <PLM>_raw.json --oe <PLM>_OE.json --label <PLM>. Status check: ssh ai 'squeue -u ge94xik2' + 'sacct -u ge94xik2 --starttime now-2day -n'."
  - "BLOCKER (soft): 150M/650M/ProtT5 stuck PENDING(Priority) on lrz-hgx-a100-80x4 ~overnight — queue congestion, not a bug. 2-day walltime (submitted ~2026-06-03 18:00 -> expires ~2026-06-05 18:00). If they expire unrun: resubmit (submit_all.sh --only LABEL:MODE). Option to speed up: 150M (640d fp32) could run on lrz-v100x2 instead of A100 (edit submit_all matrix)."
  - "RESULTS so far (no-RP regime, OE=center+binary, small ESM2): 8M retention SS 99.3 / disorder 98.1 / conservation 96.6 / scl 105.9 %; 35M SS 100.3 / conservation 97.5 / disorder 96.1 / scl 92.2 %. The RP-engaged arms (650M/3B/ProtT5, RP->896) are the still-missing interesting half; oe_norp (650M/ProtT5) isolates the RP effect."
  - "Probe config (matches ga38fak baseline, NOT the LogReg default): CNN for the 3 residue tasks + FNN for scl. Patched in venv config bank + baked into 00_build_venv.sh. PBC = 4 tasks. Single-seed autoeval-native path (30x bootstrap CIs) — the 06_harvest multi-seed rigor pass is optional/not done."
  - "Nothing committed. New since 2026-06-03 log: scripts/lrz_oe_autoeval/{build_reference_from_cache,collect_reports,compare_reports,explore_biotrainer}.py, results/oe_autoeval_reports/*.json (5 reports), arm.sbatch mem 320G, config-bank CNN+FNN patch. tools/reference accidentally rsynced to LRZ (exclude list since fixed; can rm remote copy)."
ended_at: 2026-06-04T13:20:48+02:00
---

# OE × autoeval on LRZ — grab reports + handoff (continuation of 2026-06-03)

Continuation of `sessions/2026-06-03-oe-autoeval-lrz-design-plan-impl.md` (design → plan →
package → LRZ bring-up → first 8M result → full range submitted). User disconnected overnight;
reconnected today to collect results.

## What we did today
1. **Reconnected to LRZ** (watcher had died on the VPN drop — false "done"; jobs unaffected,
   they run server-side). Checked real status via squeue/sacct.
2. **Collected 5 finished reports** → `results/oe_autoeval_reports/` (8M raw+OE, 35M raw+OE,
   3B OE). Same JSON format as the colleague's baseline — shareable as-is.
3. **Computed 35M retention** (97.5 / 100.3 / 92.2 / 96.1 % across conservation / SS / scl /
   disorder) — consistent with 8M; both are the no-RP regime.
4. **Fixed the 3B RAW OOM** (OUT_OF_MEMORY at 120G — 2560-d raw per-residue is huge; OE's 896-d
   binary fit fine, a nice illustration of OE's value). Bumped arm.sbatch `--mem` 120→320G,
   resubmitted as job 5666300.

## Why
User wanted the shareable autoeval JSONs (auto-written per arm) gathered, and a clean handoff so
the rest of the sweep can finish unattended.

## Decisions worth remembering
- Reports are auto-produced by autoeval; "making the JSONs" = just collecting them.
- 3B raw needs ≥320G RAM; OE 3B is cheap (binary). Memory is itself a data point for OE.
- Probe = CNN(residue)+FNN(scl) to match the baseline (see followups).

## Files touched
- New: `scripts/lrz_oe_autoeval/collect_reports.py`, `results/oe_autoeval_reports/*.json` (5).
- Edited: `scripts/lrz_oe_autoeval/arm.sbatch` (mem 320G), this log.
- (Yesterday: the package, LRZ scripts, DESIGN/PLAN, config-bank patch, reference builder.)

## What's next
Re-collect when the A100 arms (150M/650M/ProtT5) + 3B raw finish (see followups for the exact
commands), then the full retention table incl. the RP regime + the oe_norp RP-isolation contrast.
Optional: the 06_harvest multi-seed paired-bootstrap rigor pass; commit the work to a branch.

## Recordkeeper
No `masterbook/RECORDKEEPER_MIRROR.md`-tracked files modified this session (only project code +
LRZ scripts + session logs).
