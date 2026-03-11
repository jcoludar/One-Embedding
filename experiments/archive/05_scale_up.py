#!/usr/bin/env python3
"""Phase 4: Scale up — local experiments before LRZ.

Steps:
  1. Fair baseline: mean-pool on ESM2-35M 497 proteins (apples-to-apples)
  2. ESM2-650M: extract + train attention_pool K=8 on 497 proteins
  3. ProtT5-XL: extract + train attention_pool K=8 on 497 proteins
  4. 5K proteins: curate + ESM2-650M extract + train attention_pool K=8
  5. LRZ handoff: generate SLURM scripts

Usage:
  uv run python experiments/05_scale_up.py              # run all steps
  uv run python experiments/05_scale_up.py --step 1     # run single step
  uv run python experiments/05_scale_up.py --step 2 3   # run steps 2 and 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.compressors.attention_pool import AttentionPoolCompressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results
from src.extraction.data_loader import (
    curate_scope_set,
    load_metadata_csv,
    read_fasta,
    save_metadata_csv,
    write_fasta,
)
from src.extraction.esm_extractor import extract_residue_embeddings
from src.training.trainer import train_compressor
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings, save_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
RESULTS_PATH = BENCHMARKS_DIR / "scale_up_results.json"


# ── Helpers ──────────────────────────────────────────────────────────


def monitor():
    """Print system load."""
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except OSError:
        pass


def load_results() -> list[dict]:
    """Load existing results for resume support."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    save_benchmark_results(results, RESULTS_PATH)


def is_done(results: list[dict], name: str) -> bool:
    return any(r["name"] == name for r in results)


def fmt(v, p=3):
    if v is None:
        return "N/A"
    if isinstance(v, float) and v != v:  # NaN check
        return "N/A"
    if isinstance(v, (float, int)):
        return f"{v:.{p}f}"
    return str(v)


def train_and_benchmark(
    name: str,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    sequences: dict[str, str],
    device: torch.device,
    K: int = 8,
    latent_dim: int = 128,
    epochs: int = 200,
) -> dict:
    """Train attention_pool and run full benchmark suite."""
    embed_dim = next(iter(embeddings.values())).shape[-1]

    model = AttentionPoolCompressor(embed_dim, latent_dim, K, n_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: AttentionPoolCompressor(D={embed_dim}, D'={latent_dim}, K={K})")
    print(f"  Parameters: {n_params:,}")
    monitor()

    start = time.time()
    history = train_compressor(
        model=model,
        embeddings=embeddings,
        sequences=sequences,
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        recon_weight=1.0,
        masked_weight=0.1,
        contrastive_weight=0.1,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR / name,
        log_every=50,
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.0f}s")

    # Load best checkpoint
    best_path = CHECKPOINTS_DIR / name / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )

    results = run_benchmark_suite(
        model, embeddings, metadata, name=name, device=device
    )
    results["training_time_s"] = elapsed
    results["embed_dim"] = embed_dim
    results["K"] = K
    results["latent_dim"] = latent_dim
    results["n_proteins"] = len(embeddings)
    results["best_epoch"] = history["best_epoch"]
    return results


# ── Step 1: Fair baseline ────────────────────────────────────────────


def step1_fair_baseline(all_results: list[dict], device: torch.device) -> list[dict]:
    """Mean-pool baseline on ESM2-35M 497 proteins — apples-to-apples with Phase 3."""
    name = "meanpool_esm2_35m_500"
    if is_done(all_results, name):
        print(f"\n[Step 1] {name} already done, skipping.")
        return all_results

    print(f"\n{'='*60}")
    print(f"STEP 1: Fair baseline -- {name}")
    print(f"{'='*60}")

    h5_path = DATA_DIR / "residue_embeddings" / "esm2_35m_small500.h5"
    meta_path = DATA_DIR / "proteins" / "metadata_500.csv"

    if not h5_path.exists():
        print("ERROR: ESM2-35M embeddings not found. Run 04_narrowing.py first!")
        return all_results

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    embed_dim = next(iter(embeddings.values())).shape[-1]

    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    # model=None triggers raw mean-pool in benchmark suite
    results = run_benchmark_suite(None, embeddings, metadata, name=name, device=device)
    results["embed_dim"] = embed_dim
    results["n_proteins"] = len(embeddings)
    all_results.append(results)
    save_results(all_results)

    ret = results["retrieval_family"]["precision@1"]
    cls = results["classification_family"]["accuracy_mean"]
    print(f"\n  >> Mean-pool ESM2-35M: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")
    return all_results


# ── Step 2: ESM2-650M ────────────────────────────────────────────────


def step2_esm2_650m(all_results: list[dict], device: torch.device) -> list[dict]:
    """Extract ESM2-650M embeddings for 497 proteins, train attention_pool K=8."""
    fasta_path = DATA_DIR / "proteins" / "small_diverse_500.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_500.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_small500.h5"

    print(f"\n{'='*60}")
    print("STEP 2: ESM2-650M on 497 proteins")
    print(f"{'='*60}")

    # --- Extract embeddings ---
    if not h5_path.exists():
        print("Extracting ESM2-650M embeddings...")
        fasta_dict = read_fasta(fasta_path)
        monitor()
        embs = extract_residue_embeddings(
            fasta_dict,
            model_name="esm2_t33_650M_UR50D",
            batch_size=4,
            device=device,
        )
        save_residue_embeddings(embs, h5_path)
    else:
        print("ESM2-650M embeddings already exist")

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    # --- Mean-pool baseline on 650M ---
    mp_name = "meanpool_esm2_650m_500"
    if not is_done(all_results, mp_name):
        print(f"\n  Running mean-pool baseline on ESM2-650M...")
        mp_results = run_benchmark_suite(
            None, embeddings, metadata, name=mp_name, device=device
        )
        mp_results["embed_dim"] = embed_dim
        mp_results["n_proteins"] = len(embeddings)
        all_results.append(mp_results)
        save_results(all_results)
        ret = mp_results["retrieval_family"]["precision@1"]
        cls = mp_results["classification_family"]["accuracy_mean"]
        print(f"  >> Mean-pool ESM2-650M: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")

    # --- Attention pool ---
    ap_name = "attnpool_esm2_650m_500"
    if not is_done(all_results, ap_name):
        print(f"\n  Training attention_pool on ESM2-650M...")
        results = train_and_benchmark(
            ap_name, embeddings, metadata, sequences, device
        )
        all_results.append(results)
        save_results(all_results)
        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> AttnPool ESM2-650M: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")
    else:
        print(f"  {ap_name} already done, skipping.")

    return all_results


# ── Step 3: ProtT5 showdown ──────────────────────────────────────────


def step3_prot_t5(all_results: list[dict], device: torch.device) -> list[dict]:
    """Extract ProtT5-XL embeddings for 497 proteins, train attention_pool K=8."""
    fasta_path = DATA_DIR / "proteins" / "small_diverse_500.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_500.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "prott5_xl_small500.h5"

    print(f"\n{'='*60}")
    print("STEP 3: ProtT5-XL showdown on 497 proteins")
    print(f"{'='*60}")

    # --- Extract embeddings ---
    if not h5_path.exists():
        print("Extracting ProtT5-XL embeddings...")
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings

        fasta_dict = read_fasta(fasta_path)
        monitor()
        embs = extract_prot_t5_embeddings(
            fasta_dict,
            model_name="Rostlab/prot_t5_xl_uniref50",
            batch_size=4,
            device=device,
        )
        save_residue_embeddings(embs, h5_path)
    else:
        print("ProtT5-XL embeddings already exist")

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    # --- Mean-pool baseline on ProtT5 ---
    mp_name = "meanpool_prott5_500"
    if not is_done(all_results, mp_name):
        print(f"\n  Running mean-pool baseline on ProtT5...")
        mp_results = run_benchmark_suite(
            None, embeddings, metadata, name=mp_name, device=device
        )
        mp_results["embed_dim"] = embed_dim
        mp_results["n_proteins"] = len(embeddings)
        all_results.append(mp_results)
        save_results(all_results)
        ret = mp_results["retrieval_family"]["precision@1"]
        cls = mp_results["classification_family"]["accuracy_mean"]
        print(f"  >> Mean-pool ProtT5: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")

    # --- Attention pool ---
    ap_name = "attnpool_prott5_500"
    if not is_done(all_results, ap_name):
        print(f"\n  Training attention_pool on ProtT5...")
        results = train_and_benchmark(
            ap_name, embeddings, metadata, sequences, device
        )
        all_results.append(results)
        save_results(all_results)
        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> AttnPool ProtT5: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")
    else:
        print(f"  {ap_name} already done, skipping.")

    return all_results


# ── Step 4: Scale to 5K ──────────────────────────────────────────────


def step4_scale_5k(all_results: list[dict], device: torch.device) -> list[dict]:
    """Curate 5K proteins from SCOPe, extract ESM2-650M, train attention_pool K=8."""
    proteins_dir = DATA_DIR / "proteins"
    fasta_path = proteins_dir / "medium_diverse_5k.fasta"
    meta_path = proteins_dir / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_medium5k.h5"

    print(f"\n{'='*60}")
    print("STEP 4: Scale to 5K proteins + ESM2-650M")
    print(f"{'='*60}")

    # --- Curate 5K proteins ---
    if not fasta_path.exists():
        print("Curating 5K-protein set from SCOPe...")
        scope_fasta = proteins_dir / "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
        if not scope_fasta.exists():
            print("ERROR: SCOPe FASTA not found. Run 01_extract_residue_embeddings.py first!")
            return all_results

        fasta_dict, metadata = curate_scope_set(
            scope_fasta, n_proteins=5000, seed=456
        )
        write_fasta(fasta_dict, fasta_path)
        save_metadata_csv(metadata, meta_path)
        print(f"  {len(fasta_dict)} proteins curated")
    else:
        print("5K protein set already exists")

    # --- Extract ESM2-650M embeddings ---
    if not h5_path.exists():
        print("Extracting ESM2-650M embeddings for 5K proteins...")
        fasta_dict = read_fasta(fasta_path)
        monitor()
        embs = extract_residue_embeddings(
            fasta_dict,
            model_name="esm2_t33_650M_UR50D",
            batch_size=2,
            device=device,
        )
        save_residue_embeddings(embs, h5_path)
    else:
        print("ESM2-650M 5K embeddings already exist")

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  {len(embeddings)} proteins, embed_dim={embed_dim}")

    # --- Mean-pool baseline ---
    mp_name = "meanpool_esm2_650m_5k"
    if not is_done(all_results, mp_name):
        print(f"\n  Running mean-pool baseline on 5K proteins...")
        mp_results = run_benchmark_suite(
            None, embeddings, metadata, name=mp_name, device=device
        )
        mp_results["embed_dim"] = embed_dim
        mp_results["n_proteins"] = len(embeddings)
        all_results.append(mp_results)
        save_results(all_results)
        ret = mp_results["retrieval_family"]["precision@1"]
        cls = mp_results["classification_family"]["accuracy_mean"]
        print(f"  >> Mean-pool ESM2-650M 5K: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")

    # --- Attention pool ---
    ap_name = "attnpool_esm2_650m_5k"
    if not is_done(all_results, ap_name):
        print(f"\n  Training attention_pool on 5K proteins...")
        results = train_and_benchmark(
            ap_name, embeddings, metadata, sequences, device, epochs=200
        )
        all_results.append(results)
        save_results(all_results)
        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> AttnPool ESM2-650M 5K: Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}")
    else:
        print(f"  {ap_name} already done, skipping.")

    return all_results


# ── Step 5: LRZ handoff ──────────────────────────────────────────────


def step5_lrz_handoff():
    """Print LRZ instructions and point to SLURM scripts."""
    print(f"\n{'='*60}")
    print("STEP 5: LRZ handoff")
    print(f"{'='*60}")

    slurm_dir = Path(__file__).resolve().parent.parent / "slurm"
    if slurm_dir.exists():
        scripts = list(slurm_dir.glob("*.sh"))
        print(f"\n  SLURM scripts in {slurm_dir}/:")
        for s in sorted(scripts):
            print(f"    {s.name}")
    else:
        print("  No slurm/ directory found.")

    print("\n  LRZ experiments to run:")
    print("    1. ESM2-3B extraction on 5K+ proteins (11GB model, needs A100)")
    print("    2. 50K protein dataset (H5 files ~50GB+)")
    print("    3. Multi-run hyperparameter sweeps")
    print("    4. Standard benchmark evaluation (TAPE, FLIP, ProteinGym)")
    print("\n  Transfer best checkpoint + code:")
    print(f"    rsync -avz --exclude='data/' . lrz:~/proteembed/")
    print(f"    rsync -avz data/checkpoints/ lrz:~/proteembed/data/checkpoints/")


# ── Summary ──────────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    """Print comparison table including Phase 3 reference."""
    # Load Phase 3 best for reference
    phase3_path = BENCHMARKS_DIR / "narrowing_results.json"
    phase3_ref = None
    if phase3_path.exists():
        with open(phase3_path) as f:
            phase3 = json.load(f)
        # Best by retrieval: attention_pool_K8_K8_500
        for r in phase3:
            if r["name"] == "attention_pool_K8_K8_500":
                phase3_ref = r
                break

    print(f"\n{'='*130}")
    print("SCALE-UP RESULTS SUMMARY")
    print(f"{'='*130}")
    header = f"{'Name':<32} {'PLM':<10} {'D':<6} {'N':<6} {'Recon Cos':<12} {'Ret-Fam@1':<12} {'Cls-Fam':<12} {'Ratio':<10}"
    print(header)
    print("-" * 130)

    # Include Phase 3 reference
    if phase3_ref:
        r = phase3_ref
        print(
            f"{'[Phase3] attnpool_esM2-35m_K8':<32} {'35M':<10} {480:<6} {497:<6} "
            f"{fmt(r['reconstruction']['cosine_sim']):<12} "
            f"{fmt(r['retrieval_family']['precision@1']):<12} "
            f"{fmt(r['classification_family']['accuracy_mean']):<12} "
            f"{fmt(r['compression']['compression_ratio'], 4):<10}"
        )
        print("-" * 130)

    # Print new results sorted by retrieval
    for r in sorted(
        all_results,
        key=lambda x: x.get("retrieval_family", {}).get("precision@1", 0),
        reverse=True,
    ):
        name = r["name"]
        embed = r.get("embed_dim", "?")
        n = r.get("n_proteins", "?")
        recon_cos = r.get("reconstruction", {}).get("cosine_sim")
        ret_fam = r.get("retrieval_family", {}).get("precision@1")
        cls_fam = r.get("classification_family", {}).get("accuracy_mean")
        ratio = r.get("compression", {}).get("compression_ratio")

        # Determine PLM name from run name
        plm = "?"
        if "35m" in name:
            plm = "35M"
        elif "650m" in name:
            plm = "650M"
        elif "prott5" in name:
            plm = "ProtT5"

        print(
            f"{name:<32} {plm:<10} {str(embed):<6} {str(n):<6} "
            f"{fmt(recon_cos):<12} {fmt(ret_fam):<12} {fmt(cls_fam):<12} {fmt(ratio, 4):<10}"
        )

    # Key comparisons
    print(f"\n{'='*60}")
    print("KEY COMPARISONS")
    print(f"{'='*60}")

    def get_result(name):
        return next((r for r in all_results if r["name"] == name), None)

    pairs = [
        ("Does richer PLM help?", "attnpool_esm2_650m_500", "meanpool_esm2_650m_500"),
        ("Does attnpool beat mean on 35M?", "meanpool_esm2_35m_500", None),
        ("AttnPool ESM2-35M vs ProtT5 mean?", "meanpool_prott5_500", None),
        ("Does scale help? (5K vs 500)", "attnpool_esm2_650m_5k", "attnpool_esm2_650m_500"),
    ]
    for question, name_a, name_b in pairs:
        ra = get_result(name_a)
        rb = get_result(name_b) if name_b else None
        if ra:
            ret_a = ra.get("retrieval_family", {}).get("precision@1", 0)
            print(f"\n  {question}")
            print(f"    {name_a}: Ret@1={ret_a:.3f}")
            if rb:
                ret_b = rb.get("retrieval_family", {}).get("precision@1", 0)
                print(f"    {name_b}: Ret@1={ret_b:.3f}")
                delta = ret_a - ret_b
                print(f"    Delta: {delta:+.3f}")
            if phase3_ref:
                ret_p3 = phase3_ref["retrieval_family"]["precision@1"]
                print(f"    [Phase3 ref] attnpool_esm2_35m_K8: Ret@1={ret_p3:.3f}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Scale up experiments")
    parser.add_argument(
        "--step",
        type=int,
        nargs="*",
        default=None,
        help="Run specific step(s), e.g. --step 1 2. Default: all.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS_PATH}")

    all_results = load_results()
    print(f"Loaded {len(all_results)} existing results")

    steps = args.step or [1, 2, 3, 4, 5]

    if 1 in steps:
        all_results = step1_fair_baseline(all_results, device)
    if 2 in steps:
        all_results = step2_esm2_650m(all_results, device)
    if 3 in steps:
        all_results = step3_prot_t5(all_results, device)
    if 4 in steps:
        all_results = step4_scale_5k(all_results, device)
    if 5 in steps:
        step5_lrz_handoff()

    print_summary(all_results)


if __name__ == "__main__":
    main()
