#!/usr/bin/env python
"""Gather all autoeval_report_*.json from the per-arm output dirs into one shareable folder
with self-describing names (..._raw / ..._OE / ..._OE-noRP). Bare python3 (login node).

  python collect_reports.py <out_dir> <reports_dir>
"""
import shutil
import sys
from pathlib import Path


def clean(label):
    if label.endswith("-NORP"):
        return label[:-5] + "_OE-noRP"
    if label.endswith("-ONE"):
        return label[:-4] + "_OE"
    return label + "_raw"


def main():
    out = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        label = d.name
        rep = d / label / f"autoeval_report_{label}.json"
        if rep.exists():
            target = dest / f"autoeval_report_{clean(label)}.json"
            shutil.copy(rep, target)
            print("collected", target.name)
            n += 1
        else:
            print("(pending)", label)
    print(f"\n{n} reports -> {dest}")


if __name__ == "__main__":
    main()
