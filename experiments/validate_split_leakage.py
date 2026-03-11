#!/usr/bin/env python3
"""Validate that classification splits don't leak via sequence similarity.

Uses MMseqs2 to compute all-vs-all sequence identity between train and test sets.
Reports statistics on cross-set similarity to detect memorization risk.
"""

import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.extraction.data_loader import filter_by_family_size, load_metadata_csv, read_fasta
from src.evaluation.splitting import (
    family_stratified_split,
    load_split,
    superfamily_aware_split,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MMSEQS_BIN = "mmseqs"


def write_fasta(sequences: dict[str, str], path: Path):
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")


def run_mmseqs_search(
    query_fasta: Path,
    target_fasta: Path,
    workdir: Path,
    min_seq_id: float = 0.0,
    threads: int = 4,
) -> list[dict]:
    """Run MMseqs2 easy-search: query vs target, return all hits."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_file = workdir / "results.tsv"

    cmd = [
        MMSEQS_BIN, "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(out_file),
        str(workdir / "tmp"),
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads),
        "--format-output", "query,target,pident,alnlen,evalue,bits",
        "-v", "1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MMseqs2 error: {result.stderr}")
        raise RuntimeError("MMseqs2 search failed")

    hits = []
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),
                        "alnlen": int(parts[3]),
                        "evalue": float(parts[4]),
                        "bits": float(parts[5]),
                    })
    return hits


def analyze_leakage(hits: list[dict], split_name: str):
    """Analyze and report sequence similarity between train and test."""
    if not hits:
        print(f"\n  [{split_name}] No hits found between train and test (good!)")
        return

    identities = [h["pident"] for h in hits]

    # Per-query max identity
    query_max = defaultdict(float)
    for h in hits:
        query_max[h["query"]] = max(query_max[h["query"]], h["pident"])
    max_per_query = list(query_max.values())

    print(f"\n  [{split_name}] Train-vs-test sequence similarity:")
    print(f"    Total hits: {len(hits)}")
    print(f"    Queries with hits: {len(query_max)}")
    print(f"    Identity (all hits): mean={np.mean(identities):.1f}%, "
          f"median={np.median(identities):.1f}%, max={np.max(identities):.1f}%")
    print(f"    Max identity per query: mean={np.mean(max_per_query):.1f}%, "
          f"median={np.median(max_per_query):.1f}%, max={np.max(max_per_query):.1f}%")

    # Distribution of max identities
    thresholds = [30, 40, 50, 60, 70, 80, 90, 95]
    print(f"    Queries with max identity above threshold:")
    for t in thresholds:
        n = sum(1 for v in max_per_query if v >= t)
        pct = n / len(max_per_query) * 100
        print(f"      >= {t}%: {n}/{len(max_per_query)} ({pct:.1f}%)")


def main():
    # Load data
    fasta_path = DATA_DIR / "proteins" / "medium_diverse_5k.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"

    sequences = read_fasta(fasta_path)
    metadata = load_metadata_csv(meta_path)
    filt_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    filt_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    print(f"Dataset: {len(filt_seq)} proteins, "
          f"{len(set(m['family'] for m in filt_meta))} families")

    # 1. Superfamily-aware split (for retrieval — should have LOW cross-set similarity)
    print("\n" + "=" * 60)
    print("SPLIT 1: Superfamily-aware (retrieval split)")
    print("=" * 60)

    sf_train, sf_test, _ = superfamily_aware_split(filt_meta, test_fraction=0.3, seed=42)
    sf_train_set, sf_test_set = set(sf_train), set(sf_test)

    train_fams = set(m["family"] for m in filt_meta if m["id"] in sf_train_set)
    test_fams = set(m["family"] for m in filt_meta if m["id"] in sf_test_set)
    print(f"  Train: {len(sf_train)} proteins, {len(train_fams)} families")
    print(f"  Test:  {len(sf_test)} proteins, {len(test_fams)} families")
    print(f"  Family overlap: {len(train_fams & test_fams)}")

    # 2. Family-stratified split (for classification — will have HIGHER cross-set similarity)
    print("\n" + "=" * 60)
    print("SPLIT 2: Family-stratified (classification split)")
    print("=" * 60)

    cls_train, cls_test = family_stratified_split(filt_meta, test_fraction=0.3, seed=42)
    cls_train_set, cls_test_set = set(cls_train), set(cls_test)

    train_fams_cls = set(m["family"] for m in filt_meta if m["id"] in cls_train_set)
    test_fams_cls = set(m["family"] for m in filt_meta if m["id"] in cls_test_set)
    print(f"  Train: {len(cls_train)} proteins, {len(train_fams_cls)} families")
    print(f"  Test:  {len(cls_test)} proteins, {len(test_fams_cls)} families")
    print(f"  Family overlap: {len(train_fams_cls & test_fams_cls)}")

    # 3. Run MMseqs2 on both splits
    with tempfile.TemporaryDirectory(prefix="split_leakage_") as tmpdir:
        tmpdir = Path(tmpdir)

        for split_name, train_ids_set, test_ids_set in [
            ("superfamily_aware", sf_train_set, sf_test_set),
            ("family_stratified", cls_train_set, cls_test_set),
        ]:
            print(f"\n{'=' * 60}")
            print(f"MMseqs2 search: {split_name} (test queries vs train database)")
            print(f"{'=' * 60}")

            work = tmpdir / split_name
            work.mkdir()

            # Write FASTAs
            train_fa = work / "train.fasta"
            test_fa = work / "test.fasta"
            write_fasta({k: v for k, v in filt_seq.items() if k in train_ids_set}, train_fa)
            write_fasta({k: v for k, v in filt_seq.items() if k in test_ids_set}, test_fa)

            # Search test (query) against train (database)
            hits = run_mmseqs_search(
                query_fasta=test_fa,
                target_fasta=train_fa,
                workdir=work / "search",
                min_seq_id=0.2,  # Report hits >= 20% identity
                threads=4,
            )
            analyze_leakage(hits, split_name)


if __name__ == "__main__":
    main()
