#!/usr/bin/env python3
"""Phase 3: Narrowing - scale up top 2-3 strategies.

After running 03_quick_comparison.py, this script:
1. Reads the comparison results to identify top strategies
2. Scales up to 500 proteins + ESM2-35M
3. Trains for 500 epochs with contrastive loss
4. Hyperparameter sweep on K
5. Ablation of loss objectives
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.h5_store import load_residue_embeddings, save_residue_embeddings
from src.utils.device import get_device
from src.extraction.data_loader import (
    load_metadata_csv, read_fasta, curate_scope_set,
    write_fasta, save_metadata_csv,
)
from src.extraction.esm_extractor import extract_residue_embeddings
from src.compressors.attention_pool import AttentionPoolCompressor
from src.compressors.hierarchical import HierarchicalCompressor
from src.compressors.fourier_basis import FourierBasisCompressor
from src.compressors.vq_compress import VQCompressor
from src.training.trainer import train_compressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"

STRATEGY_CLASSES = {
    "attention_pool": AttentionPoolCompressor,
    "hierarchical": HierarchicalCompressor,
    "fourier": FourierBasisCompressor,
    "vq": VQCompressor,
}


def select_top_strategies(comparison_path: Path, top_n: int = 3) -> list[str]:
    """Read Phase 2 results and pick top strategies by combined score."""
    with open(comparison_path) as f:
        results = json.load(f)

    scores = []
    for r in results:
        name = r["name"]
        recon_cos = r.get("reconstruction", {}).get("cosine_sim", 0)
        ret_fam = r.get("retrieval_family", {}).get("precision@1", 0)
        cls_fam = r.get("classification_family", {}).get("accuracy_mean", 0)
        # Combined score: weighted average
        combined = 0.3 * recon_cos + 0.4 * ret_fam + 0.3 * cls_fam
        scores.append((name, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    print("Strategy ranking:")
    for name, score in scores:
        print(f"  {name}: {score:.3f}")

    top = [name for name, _ in scores[:top_n]]
    print(f"\nSelected top {top_n}: {top}")
    return top


def make_model(strategy_key: str, embed_dim: int, K: int = 8, latent_dim: int = 128):
    """Create a compressor model by strategy name prefix."""
    if "attention" in strategy_key:
        return AttentionPoolCompressor(embed_dim, latent_dim, K, n_heads=4)
    elif "hierarchical" in strategy_key:
        return HierarchicalCompressor(embed_dim, latent_dim, K, n_heads=4)
    elif "fourier" in strategy_key:
        return FourierBasisCompressor(embed_dim, latent_dim, K)
    elif "vq" in strategy_key:
        return VQCompressor(embed_dim, latent_dim, K, n_codes=512, n_heads=4)
    else:
        raise ValueError(f"Unknown strategy: {strategy_key}")


def main():
    device = get_device()
    print(f"Device: {device}")

    # Step 1: Select top strategies from Phase 2
    comparison_path = BENCHMARKS_DIR / "novel_comparison.json"
    if not comparison_path.exists():
        print("ERROR: Run 03_quick_comparison.py first!")
        sys.exit(1)

    top_strategies = select_top_strategies(comparison_path, top_n=2)

    # Step 2: Curate 500-protein dataset (if not already done)
    proteins_dir = DATA_DIR / "proteins"
    fasta_500_path = proteins_dir / "small_diverse_500.fasta"
    meta_500_path = proteins_dir / "metadata_500.csv"

    if not fasta_500_path.exists():
        print("\nCurating 500-protein set...")
        scope_fasta = proteins_dir / "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
        if not scope_fasta.exists():
            print("ERROR: SCOPe FASTA not found. Run 01_extract_residue_embeddings.py first!")
            sys.exit(1)

        fasta_dict, metadata = curate_scope_set(scope_fasta, n_proteins=500, seed=123)
        write_fasta(fasta_dict, fasta_500_path)
        save_metadata_csv(metadata, meta_500_path)
        print(f"  {len(fasta_dict)} proteins curated")
    else:
        print("500-protein set already exists")

    # Step 3: Extract ESM2-35M embeddings (if not done)
    h5_path_35m = DATA_DIR / "residue_embeddings" / "esm2_35m_small500.h5"
    if not h5_path_35m.exists():
        print("\nExtracting ESM2-35M embeddings for 500 proteins...")
        fasta_dict = read_fasta(fasta_500_path)
        embeddings = extract_residue_embeddings(
            fasta_dict,
            model_name="esm2_t12_35M_UR50D",
            batch_size=4,
            device=device,
        )
        save_residue_embeddings(embeddings, h5_path_35m)
    else:
        print("ESM2-35M embeddings already exist")

    # Load data
    embeddings = load_residue_embeddings(h5_path_35m)
    metadata = load_metadata_csv(meta_500_path)
    sequences = read_fasta(fasta_500_path)
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"\n{len(embeddings)} proteins, embed_dim={embed_dim}")

    # Step 4: Hyperparameter sweep for top strategies
    all_results = []
    K_values = [4, 8, 16]

    import os

    # Load any previously completed results (for resuming interrupted runs)
    prev_results_path = BENCHMARKS_DIR / "narrowing_results.json"
    completed_names = set()
    if prev_results_path.exists():
        prev = json.load(open(prev_results_path))
        all_results.extend(prev)
        completed_names = {r["name"] for r in prev}
        print(f"Resuming: {len(completed_names)} runs already done: {completed_names}")

    for strategy_name in top_strategies:
        for K in K_values:
            run_name = f"{strategy_name}_K{K}_500"

            if run_name in completed_names:
                print(f"\nSkipping {run_name} (already completed)")
                continue

            print(f"\n{'='*60}")
            print(f"Training: {run_name}")

            # Monitor system load
            try:
                load1, load5, load15 = os.getloadavg()
                print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
            except OSError:
                pass

            model = make_model(strategy_name, embed_dim, K=K, latent_dim=128)
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            start = time.time()
            history = train_compressor(
                model=model,
                embeddings=embeddings,
                sequences=sequences,
                epochs=200,
                batch_size=8,
                lr=1e-3,
                recon_weight=1.0,
                masked_weight=0.1,
                contrastive_weight=0.1,
                device=device,
                checkpoint_dir=CHECKPOINTS_DIR / run_name,
                log_every=50,
            )
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.0f}s")

            # Load best
            best_path = CHECKPOINTS_DIR / run_name / "best_model.pt"
            if best_path.exists():
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

            results = run_benchmark_suite(model, embeddings, metadata, name=run_name, device=device)
            results["training_time_s"] = elapsed
            results["K"] = K
            results["strategy"] = strategy_name
            all_results.append(results)

            # Save incrementally so we don't lose results if interrupted
            save_benchmark_results(all_results, BENCHMARKS_DIR / "narrowing_results.json")

    # Save
    save_benchmark_results(all_results, BENCHMARKS_DIR / "narrowing_results.json")

    # Print summary
    print("\n" + "=" * 110)
    print("NARROWING RESULTS")
    print("=" * 110)
    print(f"{'Run':<30} {'Recon MSE':<12} {'Recon Cos':<12} {'Retr-Fam@1':<12} {'Class-Fam':<12} {'Ratio':<10}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: x.get("retrieval_family", {}).get("precision@1", 0), reverse=True):
        name = r["name"]
        recon_mse = r.get("reconstruction", {}).get("mse", "N/A")
        recon_cos = r.get("reconstruction", {}).get("cosine_sim", "N/A")
        ret_fam = r.get("retrieval_family", {}).get("precision@1", "N/A")
        cls_fam = r.get("classification_family", {}).get("accuracy_mean", "N/A")
        ratio = r.get("compression", {}).get("compression_ratio", "N/A")

        def fmt(v, p=3):
            return f"{v:.{p}f}" if isinstance(v, float) else str(v)

        print(f"{name:<30} {fmt(recon_mse, 4):<12} {fmt(recon_cos):<12} {fmt(ret_fam):<12} {fmt(cls_fam):<12} {fmt(ratio, 4):<10}")


if __name__ == "__main__":
    main()
