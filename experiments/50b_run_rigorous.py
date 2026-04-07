#!/usr/bin/env python3
"""Exp 50 rigorous runner: iterate stages × splits × seeds, aggregate.

Invokes `experiments/50_sequence_to_oe.py` for every (stage, split, seed)
triple, one at a time (no MPS parallelism). After all runs complete,
aggregates per-seed results into summary.json per (split, stage) and writes
a final comparison table.

Usage:
    uv run python experiments/50b_run_rigorous.py                # full sweep
    uv run python experiments/50b_run_rigorous.py --stages 1     # only stage 1
    uv run python experiments/50b_run_rigorous.py --splits h     # only H
    uv run python experiments/50b_run_rigorous.py --seeds 42 43  # subset of seeds
    uv run python experiments/50b_run_rigorous.py --aggregate-only
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
OUTPUT_ROOT = ROOT / "results" / "exp50_rigorous"


def run_one(stage: int, split: str, seed: int) -> bool:
    """Invoke the main experiment script for one (stage, split, seed)."""
    load1, _, _ = os.getloadavg()
    if load1 > 10:
        print(f"[SKIP] System load {load1:.1f} > 10 — aborting this run")
        return False

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50_sequence_to_oe.py",
        "--dataset", "cath20",
        "--split", split,
        "--stage", str(stage),
        "--seed", str(seed),
        "--output-root", str(OUTPUT_ROOT),
    ]
    print(f"\n{'='*72}")
    print(f"RUN: stage={stage} split={split} seed={seed}")
    print(f"{'='*72}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"RUN COMPLETE in {elapsed:.0f}s (exit {res.returncode})")
    return res.returncode == 0


def aggregate_seeds(split: str, stage: int) -> dict | None:
    """Aggregate per-seed results.json files into one summary dict."""
    base = OUTPUT_ROOT / f"{split}_split" / f"stage{stage}"
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


def write_final_comparison(summaries: list[dict]):
    """Write final_comparison.json and .md."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "final_comparison.json"
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
    md_path = OUTPUT_ROOT / "final_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nFinal comparison written to:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--splits", nargs="+", default=["h", "t"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training, just re-aggregate existing results")
    args = parser.parse_args()

    if not args.aggregate_only:
        for split in args.splits:
            for seed in args.seeds:
                for stage in args.stages:
                    ok = run_one(stage, split, seed)
                    if not ok:
                        print("Run failed — continuing with next configuration")

    # Aggregate
    summaries = []
    for split in args.splits:
        for stage in args.stages:
            s = aggregate_seeds(split, stage)
            if s is not None:
                summaries.append(s)

    if summaries:
        write_final_comparison(summaries)
    else:
        print("No summaries to write — nothing completed successfully.")


if __name__ == "__main__":
    main()
