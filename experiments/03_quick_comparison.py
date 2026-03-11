#!/usr/bin/env python3
"""Phase 2: Quick comparison of all 4 novel compression strategies.

Trains each strategy for 100 epochs on ESM2-8M embeddings of 100 proteins,
then benchmarks all on the same data.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.h5_store import load_residue_embeddings
from src.utils.device import get_device
from src.extraction.data_loader import load_metadata_csv, read_fasta
from src.compressors.attention_pool import AttentionPoolCompressor
from src.compressors.hierarchical import HierarchicalCompressor
from src.compressors.fourier_basis import FourierBasisCompressor
from src.compressors.vq_compress import VQCompressor
from src.training.trainer import train_compressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"


def check_system_load():
    """Print current system load to monitor thermal state."""
    try:
        load = os.getloadavg()
        print(f"  System load: {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f} (1/5/15 min)")
    except OSError:
        pass


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    embeddings = load_residue_embeddings(DATA_DIR / "residue_embeddings" / "esm2_8m_tiny100.h5")
    metadata = load_metadata_csv(DATA_DIR / "proteins" / "metadata.csv")
    fasta_path = DATA_DIR / "proteins" / "tiny_diverse_100.fasta"
    sequences = read_fasta(fasta_path) if fasta_path.exists() else None

    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    # Define strategies
    strategies = {
        "attention_pool_K8": AttentionPoolCompressor(
            embed_dim=embed_dim, latent_dim=128, n_tokens=8,
            n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        ),
        "hierarchical_K8": HierarchicalCompressor(
            embed_dim=embed_dim, latent_dim=128, n_tokens=8,
            window_size=8, n_heads=4, n_layers=2,
        ),
        "fourier_K16": FourierBasisCompressor(
            embed_dim=embed_dim, latent_dim=128, n_tokens=16,
        ),
        "vq_K8": VQCompressor(
            embed_dim=embed_dim, latent_dim=128, n_tokens=8,
            n_codes=512, n_heads=4, n_decoder_layers=2,
        ),
    }

    all_results = []

    for name, model in strategies.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  K={model.num_tokens}, D'={model.latent_dim}")
        check_system_load()

        start = time.time()
        history = train_compressor(
            model=model,
            embeddings=embeddings,
            sequences=sequences,
            epochs=100,
            batch_size=8,
            lr=1e-3,
            recon_weight=1.0,
            masked_weight=0.1 if sequences else 0.0,
            device=device,
            checkpoint_dir=CHECKPOINTS_DIR / name,
            log_every=20,
        )
        elapsed = time.time() - start
        print(f"  Training done in {elapsed:.0f}s (best epoch={history['best_epoch']}, loss={history['best_loss']:.4f})")

        # Load best checkpoint
        best_path = CHECKPOINTS_DIR / name / "best_model.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

        # Benchmark
        print(f"\n  Benchmarking {name}...")
        check_system_load()
        results = run_benchmark_suite(model, embeddings, metadata, name=name, device=device)
        results["training_time_s"] = elapsed
        results["n_parameters"] = sum(p.numel() for p in model.parameters())
        all_results.append(results)

    # Save all results
    save_benchmark_results(all_results, BENCHMARKS_DIR / "novel_comparison.json")

    # Print comparison table
    print("\n" + "=" * 100)
    print("NOVEL STRATEGY COMPARISON")
    print("=" * 100)
    header = f"{'Strategy':<22} {'Recon MSE':<12} {'Recon Cos':<12} {'Retr-Fam@1':<12} {'Class-Fam':<12} {'Ratio':<10} {'Time(s)':<8}"
    print(header)
    print("-" * 100)
    for r in all_results:
        name = r["name"]
        recon_mse = r.get("reconstruction", {}).get("mse", "N/A")
        recon_cos = r.get("reconstruction", {}).get("cosine_sim", "N/A")
        ret_fam = r.get("retrieval_family", {}).get("precision@1", "N/A")
        cls_fam = r.get("classification_family", {}).get("accuracy_mean", "N/A")
        ratio = r.get("compression", {}).get("compression_ratio", "N/A")
        train_t = r.get("training_time_s", "N/A")

        def fmt(v, p=3):
            return f"{v:.{p}f}" if isinstance(v, float) else str(v)

        print(f"{name:<22} {fmt(recon_mse, 4):<12} {fmt(recon_cos):<12} {fmt(ret_fam):<12} {fmt(cls_fam):<12} {fmt(ratio, 4):<10} {fmt(train_t, 0):<8}")

    print(f"\nFull results: {BENCHMARKS_DIR / 'novel_comparison.json'}")


if __name__ == "__main__":
    main()
