#!/usr/bin/env python3
"""Exp 50 Stage 3 runner: loop over 3 seeds, aggregate per-seed results.

Invokes `experiments/50c_stage3_continuous.py` once per seed (sequential,
no MPS parallelism). After all runs complete, aggregates the per-seed
results.json into a single summary.json and writes final_comparison.{json,md}
that joins Stage 3 with the Stage 1 / Stage 2 numbers from
`results/exp50_rigorous/`.

Usage:
    PYTHONUNBUFFERED=1 uv run python experiments/50d_run_stage3.py
    PYTHONUNBUFFERED=1 uv run python experiments/50d_run_stage3.py --seeds 42 43
    uv run python experiments/50d_run_stage3.py --aggregate-only
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
LOAD_RETRY_MAX_S = 600


def wait_for_load(threshold: float = LOAD_THRESHOLD) -> bool:
    waited = 0
    while True:
        load1, _, _ = os.getloadavg()
        if load1 <= threshold:
            return True
        if waited >= LOAD_RETRY_MAX_S:
            print(f"[GIVE UP] Load {load1:.1f} > {threshold} after "
                  f"{waited}s of waiting — skipping this run")
            return False
        print(f"[WAIT] Load {load1:.1f} > {threshold}, "
              f"sleeping {LOAD_RETRY_INTERVAL_S}s...")
        time.sleep(LOAD_RETRY_INTERVAL_S)
        waited += LOAD_RETRY_INTERVAL_S


def run_one(seed: int, output_root: Path) -> str:
    """Returns 'ok', 'failed', or 'skipped'."""
    if not wait_for_load():
        return "skipped"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50c_stage3_continuous.py",
        "--seed", str(seed),
        "--output-root", str(output_root),
    ]
    print(f"\n{'='*72}")
    print(f"RUN: stage=3 seed={seed}")
    print(f"{'='*72}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"RUN COMPLETE in {elapsed:.0f}s (exit {res.returncode})")
    return "ok" if res.returncode == 0 else "failed"


def aggregate_seeds(output_root: Path) -> dict | None:
    base = output_root / "h_split" / "stage3"
    if not base.exists():
        return None
    per_seed = []
    for seed_dir in sorted(base.glob("seed*")):
        f = seed_dir / "results.json"
        if not f.exists():
            continue
        with open(f) as fh:
            per_seed.append(json.load(fh))
    if not per_seed:
        return None

    # Dim-length consistency check
    lens = {len(r["dim_accuracies"]) for r in per_seed}
    if len(lens) != 1:
        print(f"  WARNING: dim length mismatch across seeds: {lens}")
        return None

    cos = np.array([r["cosine_sim"] for r in per_seed])
    mse = np.array([r["mse"] for r in per_seed])
    bit = np.array([r["bit_accuracy"] for r in per_seed])
    dim_arrs = np.array([r["dim_accuracies"] for r in per_seed])  # (S, 896)

    intersect_60 = int((dim_arrs > 0.60).all(axis=0).sum())
    intersect_55 = int((dim_arrs > 0.55).all(axis=0).sum())
    mean_dims = dim_arrs.mean(axis=0)
    mean_60 = int((mean_dims > 0.60).sum())
    mean_55 = int((mean_dims > 0.55).sum())

    def pack(arr):
        # ddof=1 sample std with n=3 is a rough seed-variance bar, NOT a
        # confidence interval. The real test-set CIs live in each per-seed
        # results.json (BCa bootstrap from the eval pipeline). This std
        # only tells you whether 3 different RNG seeds agree.
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": arr.tolist(),
        }

    summary = {
        "stage": 3,
        "split": "h",
        "n_seeds": len(per_seed),
        "seeds": [r["config"]["seed"] for r in per_seed],
        "cosine_sim": pack(cos),
        "mse": pack(mse),
        "bit_accuracy": pack(bit),
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


def load_rigorous_summaries() -> list[dict]:
    """Load the Stage 1/2 summaries from results/exp50_rigorous/ for
    inclusion in the final comparison table. Returns empty list if missing."""
    base = ROOT / "results" / "exp50_rigorous"
    if not base.exists():
        return []
    out = []
    for split in ["h", "t"]:
        for stage in [1, 2]:
            p = base / f"{split}_split" / f"stage{stage}" / "summary.json"
            if p.exists():
                with open(p) as f:
                    out.append(json.load(f))
    return out


def write_final_comparison(stage3_summary: dict, output_root: Path):
    rigorous = load_rigorous_summaries()
    all_summaries = rigorous + [stage3_summary]

    json_path = output_root / "final_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Markdown table
    lines = [
        "# Exp 50 final comparison (Stages 1-3)",
        "",
        "| Stage | Split | Bit acc (mean ± std) | Cosine sim | dims > 60% (intersect) | Seeds |",
        "|:-----:|:-----:|:--------------------:|:----------:|:----------------------:|:-----:|",
    ]
    for s in rigorous:
        stage = s["stage"]
        split = s["split"]
        mean = s["overall_bit_acc"]["mean"] * 100
        std = s["overall_bit_acc"]["std"] * 100
        dims60 = s["dims_above_60_intersect"]
        lines.append(
            f"| {stage} | {split} | {mean:.2f} ± {std:.2f} % "
            f"| — | {dims60} / 896 | {s['n_seeds']} |"
        )
    # Stage 3
    s3 = stage3_summary
    mean3 = s3["bit_accuracy"]["mean"] * 100
    std3 = s3["bit_accuracy"]["std"] * 100
    cos_mean = s3["cosine_sim"]["mean"]
    cos_std = s3["cosine_sim"]["std"]
    dims60 = s3["dims_above_60_intersect"]
    lines.append(
        f"| 3 | h | {mean3:.2f} ± {std3:.2f} % "
        f"| {cos_mean:.4f} ± {cos_std:.4f} "
        f"| {dims60} / 896 | {s3['n_seeds']} |"
    )

    md_path = output_root / "final_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nFinal comparison written to:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    # Resolve to absolute so the per-seed subprocess invocations all see
    # the same target dir regardless of cwd.
    output_root = Path(args.output_root).resolve()
    total_wall_clock_start = time.time()
    n_ok, n_failed, n_skipped = 0, 0, 0

    if not args.aggregate_only:
        for seed in args.seeds:
            status = run_one(seed, output_root)
            if status == "ok":
                n_ok += 1
            elif status == "failed":
                n_failed += 1
            else:
                n_skipped += 1

    summary = aggregate_seeds(output_root)
    if summary is not None:
        write_final_comparison(summary, output_root)
    else:
        print("No summary to write — nothing completed successfully.")

    total_elapsed = time.time() - total_wall_clock_start
    print(f"\nStage 3 sweep complete: {n_ok} ok / {n_failed} failed / "
          f"{n_skipped} skipped (total wall-clock: {total_elapsed:.0f}s)")
    sys.exit(0 if n_failed == 0 and n_skipped == 0 else 1)


if __name__ == "__main__":
    main()
