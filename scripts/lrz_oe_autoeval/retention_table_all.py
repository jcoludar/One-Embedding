#!/usr/bin/env python
"""Consolidated OE-vs-raw retention table across all collected autoeval arms.

Reuses the single-arm parsing in compare_reports.py (task_metric / PRIMARY) and emits one
table over every PLM present in a reports dir, plus the RP-isolation (OE-noRP) contrast for the
arms that have it (650M, ProtT5). retention = metric_OE / metric_raw per task; CI is the same
conservative ratio of the two independent autoeval bootstrap CIs that compare_reports.py reports
(NOT paired; single probe seed). For the rigorous paired multi-seed CI use 06_harvest + 05_analyze.

  .venv/bin/python scripts/lrz_oe_autoeval/retention_table_all.py [reports_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reports import PRIMARY, task_metric  # noqa: E402
import json  # noqa: E402

DEFAULT_REPORTS = Path(__file__).resolve().parents[2] / "results" / "oe_autoeval_reports"
# PLMs in ascending capacity; ProtT5 last. RP engages (raw dim > 896) for 650M / 3B / ProtT5.
PLMS = ["ESM2-8M", "ESM2-35M", "ESM2-150M", "ESM2-650M", "ESM2-3B", "ProtT5"]
TASK_SHORT = {
    "PBC-secondary_structure": "SS (acc)",
    "PBC-disorder": "disorder (ρ)",
    "PBC-conservation": "conserv (acc)",
    "PBC-scl": "scl (acc)",
}


def load(reports_dir, label):
    p = reports_dir / f"autoeval_report_{label}.json"
    return json.load(open(p)) if p.exists() else None


def ret_cell(raw, oe, task, metric):
    r = task_metric(raw, task, metric)
    o = task_metric(oe, task, metric)
    if r is None or o is None:
        return None
    rmean, omean = r["mean"], o["mean"]
    ret = omean / rmean if rmean else float("nan")
    return {"raw": rmean, "oe": omean, "ret": ret}


def emit(reports_dir, oe_suffix, heading):
    print(f"\n{heading}")
    hdr = f"{'PLM':<12}" + "".join(f"{TASK_SHORT[t]:>16}" for t in PRIMARY)
    print(hdr)
    print("-" * len(hdr))
    rets_by_task = {t: [] for t in PRIMARY}
    for plm in PLMS:
        raw = load(reports_dir, f"{plm}_raw")
        oe = load(reports_dir, f"{plm}_{oe_suffix}")
        if raw is None or oe is None:
            continue
        cells = []
        for task, metric in PRIMARY.items():
            c = ret_cell(raw, oe, task, metric)
            if c is None:
                cells.append(f"{'--':>16}")
            else:
                cells.append(f"{c['ret']*100:>14.1f}%")
                rets_by_task[task].append(c["ret"])
        print(f"{plm:<12}" + "".join(cells))
    print("-" * len(hdr))
    means = []
    for task in PRIMARY:
        vals = rets_by_task[task]
        means.append(f"{sum(vals)/len(vals)*100:>14.1f}%" if vals else f"{'--':>16}")
    print(f"{'mean':<12}" + "".join(means))


def main():
    reports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORTS
    print(f"reports: {reports_dir}")
    print("retention = OE / raw  (PBC, CNN+FNN probe, autoeval 30x bootstrap, single seed)")
    emit(reports_dir, "OE", "=== OE (center + RP→896 + binary) vs raw ===")
    emit(reports_dir, "OE-noRP", "=== OE-noRP (center + binary, NO random projection) vs raw "
         "[650M / ProtT5 only — isolates the RP effect] ===")
    print("\nNote: disorder is Spearman ρ (small denominator — read absolute values, not just the "
          "ratio).\n      SS/conserv/scl are accuracy. Independent (unpaired) CIs; single probe seed.")


if __name__ == "__main__":
    main()
