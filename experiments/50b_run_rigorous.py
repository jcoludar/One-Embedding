#!/usr/bin/env python3
"""Exp 50 rigorous runner: iterate stages × splits × seeds, aggregate.

Invokes `experiments/50_sequence_to_oe.py` for every (stage, split, seed)
triple, one at a time (no MPS parallelism). Also runs the leakage audit once
at the start of a fresh sweep. After all runs complete, aggregates per-seed
results into summary.json per (split, stage) and writes a final comparison
table.

Usage:
    uv run python experiments/50b_run_rigorous.py                # full sweep
    uv run python experiments/50b_run_rigorous.py --stages 1     # only stage 1
    uv run python experiments/50b_run_rigorous.py --splits h     # only H
    uv run python experiments/50b_run_rigorous.py --seeds 42 43  # subset of seeds
    uv run python experiments/50b_run_rigorous.py --aggregate-only
    uv run python experiments/50b_run_rigorous.py --output-root /tmp/test_run
    uv run python experiments/50b_run_rigorous.py --aggregate-only --output-root /tmp/test_run

The leakage audit (50_leakage_audit.py) is run automatically before the
training loop if the expected output file does not already exist. Pass
--aggregate-only to skip both the audit and training.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

LOAD_THRESHOLD = 10.0
LOAD_RETRY_INTERVAL_S = 60
LOAD_RETRY_MAX_S = 600  # 10 minutes


def wait_for_load(threshold: float = LOAD_THRESHOLD) -> bool:
    """Wait up to LOAD_RETRY_MAX_S for system load to drop below threshold.

    Returns True if load is OK to proceed, False if we gave up.
    """
    waited = 0
    while True:
        load1, _, _ = os.getloadavg()
        if load1 <= threshold:
            return True
        if waited >= LOAD_RETRY_MAX_S:
            print(
                f"[GIVE UP] Load {load1:.1f} > {threshold} after "
                f"{waited}s of waiting — skipping this run"
            )
            return False
        print(f"[WAIT] Load {load1:.1f} > {threshold}, sleeping {LOAD_RETRY_INTERVAL_S}s...")
        time.sleep(LOAD_RETRY_INTERVAL_S)
        waited += LOAD_RETRY_INTERVAL_S


def run_leakage_audit(output_root: Path) -> bool:
    """Run the H-split seed-42 leakage audit if not already done."""
    audit_file = output_root / "leakage_audit_h_seed42.json"
    if audit_file.exists():
        print(f"[AUDIT] Leakage audit already exists at {audit_file}, skipping")
        return True

    print(f"\n{'='*72}")
    print(f"AUDIT: H-split seed 42 leakage check")
    print(f"{'='*72}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50_leakage_audit.py",
        "--split", "h", "--seed", "42",
        "--output-root", str(output_root),
    ]
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    print(f"AUDIT exit {res.returncode}")
    return res.returncode == 0


def run_one(stage: int, split: str, seed: int, output_root: Path) -> str:
    """Invoke the main experiment script for one (stage, split, seed).

    Returns 'ok', 'failed', or 'skipped'.
    """
    if not wait_for_load():
        return "skipped"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50_sequence_to_oe.py",
        "--dataset", "cath20",
        "--split", split,
        "--stage", str(stage),
        "--seed", str(seed),
        "--output-root", str(output_root),
    ]
    print(f"\n{'='*72}")
    print(f"RUN: stage={stage} split={split} seed={seed}")
    print(f"{'='*72}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"RUN COMPLETE in {elapsed:.0f}s (exit {res.returncode})")
    return "ok" if res.returncode == 0 else "failed"


def aggregate_seeds(split: str, stage: int, output_root: Path) -> dict | None:
    """Aggregate per-seed results.json files into one summary dict."""
    base = output_root / f"{split}_split" / f"stage{stage}"
    if not base.exists():
        return None

    per_seed = []
    for seed_dir in sorted(base.glob("seed*")):
        results_file = seed_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            per_seed.append(json.load(f))

    if not per_seed:
        return None

    # Guard against accidentally mixing runs with different d_out values
    lens = {len(r["dim_accuracies"]) for r in per_seed}
    if len(lens) != 1:
        print(
            f"  WARNING: dim length mismatch across seeds for "
            f"{split}/stage{stage}: {lens}"
        )
        return None

    overall = np.array([r["overall_bit_acc"] for r in per_seed])
    per_prot_means = np.array([r["per_protein_mean"] for r in per_seed])
    dim_arrs = np.array([r["dim_accuracies"] for r in per_seed])  # (S, 896)

    # Intersect@60: every seed had that dim > 0.60
    intersect_60 = int((dim_arrs > 0.60).all(axis=0).sum())
    intersect_55 = int((dim_arrs > 0.55).all(axis=0).sum())
    mean_dims = dim_arrs.mean(axis=0)
    mean_60 = int((mean_dims > 0.60).sum())
    mean_55 = int((mean_dims > 0.55).sum())

    summary = {
        "split": split,
        "stage": stage,
        "n_seeds": len(per_seed),
        "seeds": [r["config"]["seed"] for r in per_seed],
        "overall_bit_acc": {
            "mean": float(overall.mean()),
            "std": float(overall.std(ddof=1)) if len(overall) > 1 else 0.0,
            "values": overall.tolist(),
        },
        "per_protein_mean": {
            "mean": float(per_prot_means.mean()),
            "std": float(per_prot_means.std(ddof=1)) if len(per_prot_means) > 1 else 0.0,
        },
        "dims_above_60_intersect": intersect_60,
        "dims_above_55_intersect": intersect_55,
        "dims_above_60_mean": mean_60,
        "dims_above_55_mean": mean_55,
        "best_epochs": [r.get("best_epoch") for r in per_seed],
        "train_seconds": [r.get("train_seconds") for r in per_seed],
    }
    summary_path = base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")
    return summary


def write_final_comparison(summaries: list[dict], output_root: Path):
    """Write final_comparison.json and .md."""
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    lines = [
        "# Exp 50 rigorous comparison",
        "",
        "| Stage | Split | Bit acc (mean ± std) | Per-protein mean | dims > 60% (intersect) | dims > 60% (mean) | Seeds |",
        "|:-----:|:-----:|:--------------------:|:----------------:|:----------------------:|:-----------------:|:-----:|",
    ]
    for s in sorted(summaries, key=lambda x: (x["stage"], x["split"])):
        mean = s["overall_bit_acc"]["mean"] * 100
        std = s["overall_bit_acc"]["std"] * 100
        pp_mean = s["per_protein_mean"]["mean"] * 100
        pp_std = s["per_protein_mean"]["std"] * 100
        lines.append(
            f"| {s['stage']} | {s['split']} "
            f"| {mean:.2f} ± {std:.2f} % "
            f"| {pp_mean:.2f} ± {pp_std:.2f} % "
            f"| {s['dims_above_60_intersect']} / 896 "
            f"| {s['dims_above_60_mean']} / 896 "
            f"| {s['n_seeds']} |"
        )
    md_path = output_root / "final_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nFinal comparison written to:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--splits", nargs="+", default=["h", "t"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training and audit, just re-aggregate existing results")
    parser.add_argument("--output-root", default=None,
                        help="Root directory for results "
                             "(default: <repo>/results/exp50_rigorous)")
    args = parser.parse_args()

    if args.output_root is None:
        output_root = ROOT / "results" / "exp50_rigorous"
    else:
        output_root = Path(args.output_root)

    total_wall_clock_start = time.time()
    n_ok = 0
    n_failed = 0
    n_skipped = 0

    if not args.aggregate_only:
        if not run_leakage_audit(output_root):
            print("WARNING: leakage audit failed — proceeding with training anyway")

        for stage in args.stages:
            for split in args.splits:
                for seed in args.seeds:
                    status = run_one(stage, split, seed, output_root)
                    if status == "ok":
                        n_ok += 1
                    elif status == "failed":
                        n_failed += 1
                        print("Run failed — continuing with next configuration")
                    else:
                        n_skipped += 1

    # Aggregate
    summaries = []
    for split in args.splits:
        for stage in args.stages:
            s = aggregate_seeds(split, stage, output_root)
            if s is not None:
                summaries.append(s)

    if summaries:
        write_final_comparison(summaries, output_root)
    else:
        print("No summaries to write — nothing completed successfully.")

    total_elapsed = time.time() - total_wall_clock_start
    print(
        f"\nSweep complete: {n_ok} ok / {n_failed} failed / {n_skipped} skipped "
        f"(total wall-clock: {total_elapsed:.0f}s)"
    )
    sys.exit(0 if n_failed == 0 and n_skipped == 0 else 1)


if __name__ == "__main__":
    main()
