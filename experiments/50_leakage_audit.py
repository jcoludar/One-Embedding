#!/usr/bin/env python3
"""Exp 50 leakage audit: MMseqs2 search test -> train on the H-split.

Runs once on seed 42 to verify the H-level cluster split produces test
sequences with minimal homology in train. Writes
`results/exp50_rigorous/leakage_audit_h_seed42.json`.

Usage:
    uv run python experiments/50_leakage_audit.py
    uv run python experiments/50_leakage_audit.py --split t --seed 43
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.seq2oe_splits import parse_cath_fasta, cath_cluster_split

DATA = ROOT / "data"
# Resolve mmseqs2 binary at import time. Prefer $PATH; fall back to the common
# Homebrew location on macOS so the script also works on machines without mmseqs
# in the user's interactive shell PATH.
MMSEQS = shutil.which("mmseqs") or "/opt/homebrew/bin/mmseqs"


def write_fasta(sequences: dict, path: Path):
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")


def run_search(query_fa: Path, target_fa: Path, workdir: Path) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "results.tsv"
    cmd = [
        MMSEQS, "easy-search",
        str(query_fa), str(target_fa), str(out), str(workdir / "tmp"),
        "--min-seq-id", "0.0",
        "--threads", "4",
        "--format-output", "query,target,pident,alnlen,evalue,bits",
        "-v", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"MMseqs2 stderr: {res.stderr}")
        raise RuntimeError("MMseqs2 search failed")
    hits: list[dict] = []
    if out.exists():
        with open(out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),  # already 0-100
                    })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["h", "t"], default="h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str,
                        default="results/exp50_rigorous")
    args = parser.parse_args()

    print(f"Loading CATH20 FASTA...")
    cath_fa = DATA / "external" / "cath20" / "cath20_labeled.fasta"
    meta = parse_cath_fasta(cath_fa)
    print(f"  {len(meta)} proteins parsed")

    level = args.split.upper()
    print(f"Building {level}-split (seed={args.seed})...")
    train_ids, val_ids, test_ids = cath_cluster_split(
        meta, level=level, fractions=(0.8, 0.1, 0.1), seed=args.seed,
    )
    print(f"  train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    with tempfile.TemporaryDirectory(prefix="exp50_audit_") as tmp:
        tmp_path = Path(tmp)
        train_fa = tmp_path / "train.fa"
        test_fa = tmp_path / "test.fa"
        write_fasta({k: meta[k]["seq"] for k in train_ids}, train_fa)
        write_fasta({k: meta[k]["seq"] for k in test_ids}, test_fa)

        print(f"Running MMseqs2 easy-search (test -> train)...")
        hits = run_search(test_fa, train_fa, tmp_path / "search")
        print(f"  {len(hits)} total hits")

    # Aggregate max per-query identity
    query_max: dict[str, float] = defaultdict(float)
    for h in hits:
        if h["pident"] > query_max[h["query"]]:
            query_max[h["query"]] = h["pident"]

    # Test queries with no MMseqs hits get 0% max identity (treated as no leakage).
    max_per_query = [query_max.get(pid, 0.0) for pid in test_ids]
    arr = np.array(max_per_query)

    # JSON serialization coerces int dict keys to strings, so build the dicts
    # with str keys from the start — keeps the in-memory and on-disk schemas
    # identical so downstream consumers can index uniformly.
    thresholds = [20, 25, 30, 40, 50, 60]
    counts = {str(t): int((arr >= t).sum()) for t in thresholds}
    pcts = {str(t): float((arr >= t).mean() * 100) for t in thresholds}

    report = {
        "split": level,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "max_identity_per_test_query": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        },
        "test_queries_with_train_hit_above_pct": counts,
        "test_queries_with_train_hit_above_pct_fraction": pcts,
    }

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"leakage_audit_{level.lower()}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLeakage audit written to {out_path}")
    print(f"  mean max-identity: {arr.mean():.1f}%")
    print(f"  fraction >= 40% identity: {pcts['40']:.1f}%")

    if pcts["40"] > 5.0:
        print(f"WARNING: {pcts['40']:.1f}% of test proteins have a train hit "
              f">= 40% identity — split may be leakier than expected")


if __name__ == "__main__":
    main()
