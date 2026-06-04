#!/usr/bin/env python
"""Compute OE-vs-raw retention from two biotrainer autoeval report JSONs (single-seed path).

Uses autoeval's own per-task bootstrapping CIs (30 iterations). retention = metric_OE/metric_raw
per task. NOTE: the two arms are bootstrapped independently (not paired) and this is a single
probe seed — for the fully rigorous paired multi-seed CI, use the 06_harvest + 05_analyze path.
This is the fast, autoeval-native comparison.

  python compare_reports.py --raw raw_report.json --oe oe_report.json [--label ESM2-8M]
"""
import argparse
import json

# PBC primary metric per task (biotrainer 1.4.0).
PRIMARY = {
    "PBC-conservation": "accuracy",
    "PBC-secondary_structure": "accuracy",
    "PBC-scl": "accuracy",
    "PBC-disorder": "spearmans-corr-coeff",
}


def find_bootstrapping(node):
    """First 'bootstrapping.results' list found anywhere under node."""
    if isinstance(node, dict):
        bs = node.get("bootstrapping")
        if isinstance(bs, dict) and isinstance(bs.get("results"), list):
            return bs["results"]
        for v in node.values():
            r = find_bootstrapping(v)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find_bootstrapping(v)
            if r is not None:
                return r
    return None


def task_metric(report, task, metric_name):
    results = report["supervised_results"]["PBC"]["results"]
    if task not in results:
        return None
    boot = find_bootstrapping(results[task])
    if not boot:
        return None
    for entry in boot:
        if entry.get("name") == metric_name:
            return {"mean": entry["mean"], "lower": entry["lower"], "upper": entry["upper"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--oe", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    raw = json.load(open(args.raw))
    oe = json.load(open(args.oe))

    print(f"\nOE-vs-raw retention {args.label}  (PBC, CNN probe, autoeval 30x bootstrap, single seed)")
    print(f"{'task':<26}{'metric':<22}{'raw':>16}{'OE':>16}{'retention':>22}")
    print("-" * 102)
    for task, metric in PRIMARY.items():
        r = task_metric(raw, task, metric)
        o = task_metric(oe, task, metric)
        if r is None or o is None:
            print(f"{task:<26}{metric:<22}{'(missing)':>16}")
            continue
        rmean, omean = r["mean"], o["mean"]
        ret = omean / rmean if rmean else float("nan")
        # conservative ratio bound from the two independent CIs
        lo = o["lower"] / r["upper"] if r["upper"] else float("nan")
        hi = o["upper"] / r["lower"] if r["lower"] else float("nan")
        raw_s = f"{rmean:.4f}[{r['lower']:.3f},{r['upper']:.3f}]"
        oe_s = f"{omean:.4f}[{o['lower']:.3f},{o['upper']:.3f}]"
        ret_s = f"{ret*100:6.1f}% [{lo*100:.1f},{hi*100:.1f}]"
        print(f"{task:<26}{metric:<22}{raw_s:>16}{oe_s:>16}{ret_s:>22}")
    print("-" * 102)
    print("retention = OE/raw; CI is a conservative ratio of the two independent bootstrap CIs "
          "(not paired). disorder uses spearman (small-denominator -> read the absolute values).")


if __name__ == "__main__":
    main()
