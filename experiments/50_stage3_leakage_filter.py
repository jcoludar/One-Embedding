#!/usr/bin/env python3
"""Exp 50 Stage 3 DeepLoc leakage filter.

Identifies DeepLoc proteins with > 30% sequence identity to any CATH20
H-split test protein at a given seed. The Stage 3 training script uses
this exclusion list to filter the DeepLoc auxiliary pool.

Usage:
    uv run python experiments/50_stage3_leakage_filter.py --seed 42
    uv run python experiments/50_stage3_leakage_filter.py --seed 43
    uv run python experiments/50_stage3_leakage_filter.py --seed 44
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe_splits import parse_cath_fasta, cath_cluster_split

DATA = ROOT / "data"
DEEPLOC_FASTA = (
    ROOT / "tools" / "reference" / "LightAttention" /
    "data_files" / "deeploc_complete_dataset.fasta"
)
CATH_FASTA = DATA / "external" / "cath20" / "cath20_labeled.fasta"
MMSEQS = shutil.which("mmseqs") or "/opt/homebrew/bin/mmseqs"
IDENTITY_THRESHOLD = 30.0  # percent


def write_fasta(sequences: dict, path: Path):
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")


def run_mmseqs_search(query_fa: Path, target_fa: Path, workdir: Path) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "results.tsv"
    cmd = [
        MMSEQS, "easy-search",
        str(query_fa), str(target_fa), str(out), str(workdir / "tmp"),
        "--min-seq-id", "0.0",
        "--threads", "4",
        "--format-output", "query,target,pident",
        "-v", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"MMseqs2 stdout: {res.stdout}")
        print(f"MMseqs2 stderr: {res.stderr}")
        raise RuntimeError("MMseqs2 search failed")
    hits: list[dict] = []
    if out.exists():
        with open(out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),
                    })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument(
        "--identity-threshold", type=float, default=IDENTITY_THRESHOLD,
        help="Exclude DeepLoc proteins with any hit at >= this %%id to "
             "CATH20 H-test (default: 30.0)",
    )
    args = parser.parse_args()

    print("Loading CATH20 FASTA...")
    cath_meta = parse_cath_fasta(CATH_FASTA)
    print(f"  {len(cath_meta)} CATH proteins parsed")

    print(f"Building CATH20 H-split (seed={args.seed})...")
    _, _, h_test_ids = cath_cluster_split(
        cath_meta, level="H", fractions=(0.8, 0.1, 0.1), seed=args.seed,
    )
    print(f"  {len(h_test_ids)} H-test proteins")

    print(f"Loading DeepLoc FASTA from {DEEPLOC_FASTA}...")
    deeploc_seqs = read_fasta(DEEPLOC_FASTA)
    print(f"  {len(deeploc_seqs)} DeepLoc sequences parsed")

    with tempfile.TemporaryDirectory(prefix="exp50_stage3_leakage_") as tmp:
        tmp_path = Path(tmp)
        cath_test_fa = tmp_path / "cath_h_test.fa"
        deeploc_fa = tmp_path / "deeploc.fa"

        # Write CATH H-test as query, DeepLoc as target
        cath_test_seqs = {pid: cath_meta[pid]["seq"] for pid in h_test_ids}
        write_fasta(cath_test_seqs, cath_test_fa)
        write_fasta(deeploc_seqs, deeploc_fa)

        print("Running MMseqs2 easy-search (cath_h_test -> deeploc)...")
        hits = run_mmseqs_search(cath_test_fa, deeploc_fa, tmp_path / "search")
        print(f"  {len(hits)} total hits")

    # Any DeepLoc protein (target) with any hit >= threshold is excluded
    excluded: set[str] = {
        h["target"] for h in hits if h["pident"] >= args.identity_threshold
    }

    report = {
        "seed": args.seed,
        "identity_threshold_pct": args.identity_threshold,
        "n_cath_h_test": len(h_test_ids),
        "n_deeploc_total": len(deeploc_seqs),
        "n_deeploc_excluded": len(excluded),
        "fraction_deeploc_excluded": len(excluded) / max(len(deeploc_seqs), 1),
        "excluded_deeploc_ids": sorted(excluded),
    }

    out_root = Path(args.output_root) / "leakage_filter"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"deeploc_leakage_excluded_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLeakage filter written to {out_path}")
    print(f"  DeepLoc proteins excluded: {len(excluded)} / {len(deeploc_seqs)} "
          f"({report['fraction_deeploc_excluded']*100:.1f}%)")


if __name__ == "__main__":
    main()
