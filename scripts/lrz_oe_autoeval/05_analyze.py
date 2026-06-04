#!/usr/bin/env python
"""Compute OE-vs-raw retention from harvested per-item predictions (PLAN §R1/R2/R6).

For each (PLM × task): load raw and OE per-item TEST predictions across the 5 probe seeds,
align by sequence/residue id (HARD: same items, same order across arms), stack to (S, N), and
call src.oe_autoeval.retention.paired_retention → ratio + BCa CI + between_seed_sd + Δ. For
650M and ProtT5 also compute the RP-isolation contrast paired(oe vs oe_norp). Emits a CSV +
RESULTS.md. Run on the login node / CPU after the sweep + harvest.

  source config.sh
  /work/venv/bin/python 05_analyze.py --preds-dir /work/out --out /work/results

Input layout (produced by the biotrainer-direct harvest, PLAN §R3) — CONFIRM/adjust to the
harvest's actual file names:
  <preds-dir>/<LABEL>/<task>/seed<k>/predictions.json
     = {"item_ids": [...], "y_true": [...], "y_pred": [...]}  (test split only)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = "/work/ProteEmbedExplorations"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from src.oe_autoeval.retention import paired_retention  # noqa: E402

# task -> metric (from the PBC enumeration, PLAN §R / DESIGN §14.1-4)
TASK_METRIC = {
    "conservation": "accuracy", "secondary_structure": "accuracy", "membrane": "accuracy",
    "frustration-classification": "accuracy", "phages": "accuracy", "scl": "accuracy",
    "disorder_chezod": "spearman", "disorder_trizod": "spearman",
    "frustration-regression": "spearman",
}
# raw arm label, OE arm label, optional no-RP label, native_d, d_out_eff
PLMS = [
    ("ESM2-8M", "ESM2-8M", "ESM2-8M-ONE", None, 320, 320),
    ("ESM2-35M", "ESM2-35M", "ESM2-35M-ONE", None, 480, 480),
    ("ESM2-150M", "ESM2-150M", "ESM2-150M-ONE", None, 640, 640),
    ("ESM2-650M", "ESM2-650M", "ESM2-650M-ONE", "ESM2-650M-NORP", 1280, 896),
    ("ESM2-3B", "ESM2-3B", "ESM2-3B-ONE", None, 2560, 896),
    ("ProtT5", "ProtT5", "ProtT5-ONE", "ProtT5-NORP", 1024, 896),
]
SEEDS = (42, 43, 44, 45, 46)


def load_arm(preds_dir, label, task):
    """Return (item_ids, y_true, group_ids, preds (S, N)) aligned across seeds by item id."""
    ids_ref = y_ref = grp_ref = None
    stack = []
    for k in SEEDS:
        f = Path(preds_dir) / label / task / f"seed{k}" / "predictions.json"
        d = json.loads(f.read_text())
        ids, y, p = d["item_ids"], np.asarray(d["y_true"]), np.asarray(d["y_pred"])
        grp = np.asarray(d.get("group_ids", ids))    # protein id per residue (cluster bootstrap)
        order = np.argsort(ids)                       # ids are zero-padded -> lexical == residue order
        ids_sorted = [ids[i] for i in order]
        if ids_ref is None:
            ids_ref, y_ref, grp_ref = ids_sorted, y[order], grp[order]
        elif ids_sorted != ids_ref:
            raise ValueError(f"seed {k} item ids differ for {label}/{task} — split not shared")
        stack.append(p[order])
    return ids_ref, y_ref, grp_ref, np.stack(stack)


def retention_row(preds_dir, raw_label, oe_label, task):
    metric = TASK_METRIC[task]
    ids_r, y_r, grp_r, raw = load_arm(preds_dir, raw_label, task)
    ids_o, y_o, grp_o, oe = load_arm(preds_dir, oe_label, task)
    # same test items, same labels, same clustering across arms (PLAN §R2).
    if not np.array_equal(y_r, y_o):
        raise ValueError(f"y_true mismatch across arms for {oe_label}/{task}")
    # per-residue tasks cluster-bootstrap by protein; per-sequence tasks have group==item.
    group_ids = grp_r if metric and task not in ("phages", "scl") else None
    return paired_retention(ids_r, ids_o, y_r, raw, oe, group_ids=group_ids,
                            metric=metric, n_boot=2000, seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for plm, raw_label, oe_label, norp_label, native_d, d_eff in PLMS:
        for task in TASK_METRIC:
            try:
                res = retention_row(args.preds_dir, raw_label, oe_label, task)
            except FileNotFoundError:
                continue
            rp_engaged = native_d > 896
            # Small-denominator: a ratio is unreliable when the raw metric is near 0 (disorder
            # spearman) -> prefer the absolute Δ (which has its own BCa CI). fan-review #3 stat-I3.
            ratio_unreliable = (TASK_METRIC[task] == "spearman" and res["metric_raw"] < 0.1)
            # Units-consistent decision rule (fan-review #3 stat-I2): the deficit must (a) have an
            # item-variance CI excluding 1, AND (b) exceed 2 probe-seed standard errors. Both on
            # the ratio scale. (Not multiplicity-corrected — a diagnostic, not a per-cell test.)
            seed_se = res["between_seed_sd"] / max(res["n_seeds"], 1) ** 0.5
            deficit_real = (res["ci_high"] < 1.0) and ((1.0 - res["ratio"]) > 2.0 * seed_se)
            row = {
                "plm": plm, "native_d": native_d, "d_out_eff": d_eff,
                "rp_engaged": rp_engaged, "task": task, "metric": TASK_METRIC[task],
                "metric_raw": res["metric_raw"], "metric_oe": res["metric_oe"],
                "retention": res["ratio"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "between_seed_sd": res["between_seed_sd"],
                "delta": res["delta"], "delta_ci": res["delta_ci"],
                "ratio_unreliable_prefer_delta": ratio_unreliable,
                "deficit_exceeds_noise": deficit_real,
            }
            # RP-isolation contrast (PLAN §R6): oe vs oe_norp at fixed PLM.
            if norp_label is not None:
                try:
                    rp = retention_row(args.preds_dir, norp_label, oe_label, task)
                    row["rp_contrast_ratio"] = rp["ratio"]
                    row["rp_contrast_ci"] = [rp["ci_low"], rp["ci_high"]]
                except FileNotFoundError:
                    pass
            rows.append(row)

    csv_path = out / "retention_table.csv"
    if rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} retention rows -> {csv_path}")
    print("Reminder (PLAN §R1/R8): prefer the absolute-Δ panel when metric_raw is near 0 "
          "(disorder ρ); per-sequence (phages/scl) OE features are fp16. Frame as estimation; "
          "BH-FDR only if claiming a significant degradation.")


if __name__ == "__main__":
    main()
