#!/usr/bin/env python3
"""Phase 1.3: Establish baselines with mean pool, SWE, and BoM-Pooling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.h5_store import load_residue_embeddings
from src.extraction.data_loader import load_metadata_csv
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results
from src.compressors.mean_pool import MeanPoolCompressor
from src.compressors.swe_pool import SWEPoolCompressor
from src.compressors.bom_pool import BoMPoolCompressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"


def main():
    # Load data
    print("Loading embeddings and metadata...")
    embeddings = load_residue_embeddings(DATA_DIR / "residue_embeddings" / "esm2_8m_tiny100.h5")
    metadata = load_metadata_csv(DATA_DIR / "proteins" / "metadata.csv")
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    all_results = []

    # 1. Raw mean pooling (no model)
    print("\n=== Mean Pool (raw) ===")
    results = run_benchmark_suite(None, embeddings, metadata, name="mean_pool_raw")
    all_results.append(results)

    # 2. Mean pool compressor (through ABC)
    print("\n=== Mean Pool Compressor ===")
    model = MeanPoolCompressor(embed_dim)
    results = run_benchmark_suite(model, embeddings, metadata, name="mean_pool")
    all_results.append(results)

    # 3. SWE Pooling
    print("\n=== SWE Pool ===")
    model = SWEPoolCompressor(embed_dim, n_slices=500)
    results = run_benchmark_suite(model, embeddings, metadata, name="swe_pool")
    all_results.append(results)

    # 4. BoM Pooling
    print("\n=== BoM Pool ===")
    model = BoMPoolCompressor(embed_dim, window_size=16, stride=8)
    results = run_benchmark_suite(model, embeddings, metadata, name="bom_pool")
    all_results.append(results)

    # Save results
    save_benchmark_results(all_results, BENCHMARKS_DIR / "baseline_benchmarks.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("BASELINE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Retr-Fam@1':<12} {'Retr-Fold@1':<12} {'Class-Fam':<12} {'Ratio':<10}")
    print("-" * 80)
    for r in all_results:
        name = r["name"]
        ret_fam = r.get("retrieval_family", {}).get("precision@1", "N/A")
        ret_fold = r.get("retrieval_fold", {}).get("precision@1", "N/A")
        cls_fam = r.get("classification_family", {}).get("accuracy_mean", "N/A")
        ratio = r.get("compression", {}).get("compression_ratio", "N/A")
        ret_fam_str = f"{ret_fam:.3f}" if isinstance(ret_fam, float) else ret_fam
        ret_fold_str = f"{ret_fold:.3f}" if isinstance(ret_fold, float) else ret_fold
        cls_fam_str = f"{cls_fam:.3f}" if isinstance(cls_fam, float) else cls_fam
        ratio_str = f"{ratio:.4f}" if isinstance(ratio, float) else ratio
        print(f"{name:<20} {ret_fam_str:<12} {ret_fold_str:<12} {cls_fam_str:<12} {ratio_str:<10}")

    print(f"\nFull results: {BENCHMARKS_DIR / 'baseline_benchmarks.json'}")


if __name__ == "__main__":
    main()
