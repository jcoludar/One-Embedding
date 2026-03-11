#!/usr/bin/env python3
"""Phase 5: Fix AttentionPool collapse at scale.

Steps:
  1. Re-evaluate existing models on filtered data (no singletons)
  2. Ablation: latent_dim=256
  3. Ablation: latent_dim=512
  4. Ablation: K=16 tokens
  5. Ablation: 4 encoder layers, 8 heads
  6. Ablation: contrastive_weight=0.5
  7. Ablation: batch_size=32
  8. Combined best from steps 2-7
  9. ProtT5-XL at 5K scale

Usage:
  uv run python experiments/06_fix_collapse.py              # run all steps
  uv run python experiments/06_fix_collapse.py --step 1     # run single step
  uv run python experiments/06_fix_collapse.py --step 2 3   # run steps 2 and 3
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
    filter_by_family_size,
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
RESULTS_PATH = BENCHMARKS_DIR / "fix_collapse_results.json"


# ── Helpers ──────────────────────────────────────────────────────────


def monitor():
    """Print system load."""
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except OSError:
        pass


def load_results() -> list[dict]:
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
    if isinstance(v, float) and v != v:
        return "N/A"
    if isinstance(v, (float, int)):
        return f"{v:.{p}f}"
    return str(v)


def load_5k_filtered():
    """Load 5K dataset and filter to families with >= 3 members.

    Returns (embeddings, metadata, sequences, filtered_metadata, kept_ids).
    """
    fasta_path = DATA_DIR / "proteins" / "medium_diverse_5k.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_medium5k.h5"

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)

    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)

    # Filter embeddings and sequences to kept_ids
    filtered_emb = {k: v for k, v in embeddings.items() if k in kept_ids}
    filtered_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filtered_meta))
    print(f"  Filtered: {len(filtered_emb)}/{len(embeddings)} proteins, "
          f"{n_fam} families (min 3 members)")
    return embeddings, metadata, sequences, filtered_meta, filtered_emb, filtered_seq


def train_and_benchmark(
    name: str,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    sequences: dict[str, str],
    device: torch.device,
    K: int = 8,
    latent_dim: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    n_proj_layers: int = 1,
    epochs: int = 100,
    batch_size: int = 8,
    contrastive_weight: float = 0.1,
) -> dict:
    """Train attention_pool and run full benchmark suite."""
    embed_dim = next(iter(embeddings.values())).shape[-1]

    model = AttentionPoolCompressor(
        embed_dim,
        latent_dim,
        K,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_proj_layers=n_proj_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: AttentionPoolCompressor(D={embed_dim}, D'={latent_dim}, K={K}, "
          f"heads={n_heads}, enc_layers={n_encoder_layers}, proj_layers={n_proj_layers})")
    print(f"  Parameters: {n_params:,}")
    monitor()

    start = time.time()
    history = train_compressor(
        model=model,
        embeddings=embeddings,
        sequences=sequences,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        recon_weight=1.0,
        masked_weight=0.1,
        contrastive_weight=contrastive_weight,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR / name,
        log_every=25,
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.0f}s")
    monitor()

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
    results["n_heads"] = n_heads
    results["n_encoder_layers"] = n_encoder_layers
    results["n_proj_layers"] = n_proj_layers
    results["n_proteins"] = len(embeddings)
    results["n_families"] = len(set(m["family"] for m in metadata))
    results["epochs"] = epochs
    results["batch_size"] = batch_size
    results["contrastive_weight"] = contrastive_weight
    results["best_epoch"] = history["best_epoch"]
    return results


# ── Step 1: Re-evaluate on filtered data ──────────────────────────


def step1_reeval_filtered(all_results: list[dict], device: torch.device) -> list[dict]:
    """Re-evaluate existing models on cleaned (no-singleton) dataset."""
    print(f"\n{'='*60}")
    print("STEP 1: Re-evaluate on filtered data (no singletons)")
    print(f"{'='*60}")

    (full_emb, full_meta, full_seq,
     filt_meta, filt_emb, filt_seq) = load_5k_filtered()

    # --- 1a. Mean-pool on filtered 5K ---
    name = "meanpool_esm2_650m_5k_filtered"
    if not is_done(all_results, name):
        print(f"\n  Evaluating {name}...")
        results = run_benchmark_suite(
            None, filt_emb, filt_meta, name=name, device=device
        )
        embed_dim = next(iter(filt_emb.values())).shape[-1]
        results["embed_dim"] = embed_dim
        results["n_proteins"] = len(filt_emb)
        results["n_families"] = len(set(m["family"] for m in filt_meta))
        all_results.append(results)
        save_results(all_results)

        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {name} already done")

    # --- 1b. Existing attnpool checkpoint on filtered 5K ---
    name = "attnpool_esm2_650m_5k_filtered"
    if not is_done(all_results, name):
        ckpt = CHECKPOINTS_DIR / "attnpool_esm2_650m_5k" / "best_model.pt"
        if ckpt.exists():
            print(f"\n  Loading existing checkpoint: {ckpt}")
            embed_dim = next(iter(filt_emb.values())).shape[-1]
            model = AttentionPoolCompressor(embed_dim, 128, 8, n_heads=4)
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
            model = model.to(device)

            results = run_benchmark_suite(
                model, filt_emb, filt_meta, name=name, device=device
            )
            results["embed_dim"] = embed_dim
            results["n_proteins"] = len(filt_emb)
            results["n_families"] = len(set(m["family"] for m in filt_meta))
            results["note"] = "existing checkpoint, filtered eval only"
            all_results.append(results)
            save_results(all_results)

            ret = results["retrieval_family"]["precision@1"]
            cls = results["classification_family"]["accuracy_mean"]
            print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
        else:
            print(f"  WARNING: Checkpoint not found: {ckpt}")
    else:
        print(f"  {name} already done")

    # --- 1c. Mean-pool on unfiltered (reference) ---
    name = "meanpool_esm2_650m_5k_unfiltered_ref"
    if not is_done(all_results, name):
        print(f"\n  Evaluating {name} (unfiltered reference)...")
        results = run_benchmark_suite(
            None, full_emb, full_meta, name=name, device=device
        )
        embed_dim = next(iter(full_emb.values())).shape[-1]
        results["embed_dim"] = embed_dim
        results["n_proteins"] = len(full_emb)
        results["n_families"] = len(set(m["family"] for m in full_meta))
        results["note"] = "unfiltered reference"
        all_results.append(results)
        save_results(all_results)

        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {name} already done")

    return all_results


# ── Steps 2-7: Ablation study ────────────────────────────────────


def run_ablation(
    all_results: list[dict],
    device: torch.device,
    step_num: int,
    name: str,
    description: str,
    **kwargs,
) -> list[dict]:
    """Run a single ablation: train on filtered 5K, benchmark."""
    if is_done(all_results, name):
        print(f"\n[Step {step_num}] {name} already done, skipping.")
        return all_results

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

    _, _, _, filt_meta, filt_emb, filt_seq = load_5k_filtered()

    results = train_and_benchmark(
        name=name,
        embeddings=filt_emb,
        metadata=filt_meta,
        sequences=filt_seq,
        device=device,
        **kwargs,
    )
    all_results.append(results)
    save_results(all_results)

    ret = results["retrieval_family"]["precision@1"]
    cls = results["classification_family"]["accuracy_mean"]
    print(f"\n  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    return all_results


def step2_latent256(all_results: list[dict], device: torch.device) -> list[dict]:
    return run_ablation(
        all_results, device, 2,
        name="ablation_latent256",
        description="latent_dim=256 (5x compression vs 10x)",
        latent_dim=256,
    )


def step3_latent512(all_results: list[dict], device: torch.device) -> list[dict]:
    # Only run if step 2 showed improvement
    step2_result = next((r for r in all_results if r["name"] == "ablation_latent256"), None)
    baseline = next((r for r in all_results if r["name"] == "attnpool_esm2_650m_5k_filtered"), None)

    if step2_result and baseline:
        step2_ret = step2_result.get("retrieval_family", {}).get("precision@1", 0)
        base_ret = baseline.get("retrieval_family", {}).get("precision@1", 0)
        if step2_ret <= base_ret:
            print(f"\n[Step 3] Skipping latent_dim=512: latent_dim=256 didn't improve "
                  f"(256={step2_ret:.3f} vs base={base_ret:.3f})")
            return all_results

    return run_ablation(
        all_results, device, 3,
        name="ablation_latent512",
        description="latent_dim=512 (2.5x compression)",
        latent_dim=512,
    )


def step4_k16(all_results: list[dict], device: torch.device) -> list[dict]:
    return run_ablation(
        all_results, device, 4,
        name="ablation_k16",
        description="K=16 tokens (more structural slots)",
        K=16,
    )


def step5_deep_encoder(all_results: list[dict], device: torch.device) -> list[dict]:
    return run_ablation(
        all_results, device, 5,
        name="ablation_deep_encoder",
        description="4 encoder layers, 8 heads (more capacity)",
        n_encoder_layers=4,
        n_heads=8,
    )


def step6_contrastive(all_results: list[dict], device: torch.device) -> list[dict]:
    return run_ablation(
        all_results, device, 6,
        name="ablation_contrastive05",
        description="contrastive_weight=0.5 (force discrimination)",
        contrastive_weight=0.5,
    )


def step7_batch32(all_results: list[dict], device: torch.device) -> list[dict]:
    return run_ablation(
        all_results, device, 7,
        name="ablation_batch32",
        description="batch_size=32 (more negatives for InfoNCE)",
        batch_size=32,
    )


# ── Step 8: Combined best ────────────────────────────────────────


def step8_combined(all_results: list[dict], device: torch.device) -> list[dict]:
    """Combine the best ablation findings."""
    name = "ablation_combined"
    if is_done(all_results, name):
        print(f"\n[Step 8] {name} already done, skipping.")
        return all_results

    print(f"\n{'='*60}")
    print("STEP 8: Combined best ablation settings")
    print(f"{'='*60}")

    # Find baseline filtered result
    baseline = next(
        (r for r in all_results if r["name"] == "attnpool_esm2_650m_5k_filtered"), None
    )
    base_ret = baseline["retrieval_family"]["precision@1"] if baseline else 0

    # Check each ablation for improvement
    ablation_names = [
        ("ablation_latent256", "latent_dim", 256),
        ("ablation_latent512", "latent_dim", 512),
        ("ablation_k16", "K", 16),
        ("ablation_deep_encoder", "n_encoder_layers+n_heads", "4+8"),
        ("ablation_contrastive05", "contrastive_weight", 0.5),
        ("ablation_batch32", "batch_size", 32),
    ]

    best_kwargs = {}
    print(f"\n  Baseline filtered Ret@1: {base_ret:.3f}")
    print(f"  Ablation results:")

    for abl_name, param, value in ablation_names:
        r = next((r for r in all_results if r["name"] == abl_name), None)
        if r:
            ret = r["retrieval_family"]["precision@1"]
            delta = ret - base_ret
            marker = " <<< IMPROVED" if delta > 0.01 else ""
            print(f"    {abl_name}: Ret@1={ret:.3f} (delta={delta:+.3f}){marker}")

            # Include if it improved by > 0.01
            if delta > 0.01:
                if param == "n_encoder_layers+n_heads":
                    best_kwargs["n_encoder_layers"] = 4
                    best_kwargs["n_heads"] = 8
                elif param == "latent_dim":
                    # Take the better latent_dim if both improved
                    existing = best_kwargs.get("latent_dim")
                    if existing is None:
                        best_kwargs["latent_dim"] = value
                    else:
                        # Keep whichever had higher ret
                        existing_r = next(
                            (r2 for r2 in all_results
                             if r2.get("latent_dim") == existing and "ablation_latent" in r2["name"]),
                            None,
                        )
                        if existing_r and ret > existing_r["retrieval_family"]["precision@1"]:
                            best_kwargs["latent_dim"] = value
                else:
                    best_kwargs[param] = value
        else:
            print(f"    {abl_name}: not run")

    if not best_kwargs:
        print("\n  No ablations improved significantly. Using latent_dim=256 + batch_size=32 as default combo.")
        best_kwargs = {"latent_dim": 256, "batch_size": 32}

    print(f"\n  Combined config: {best_kwargs}")

    _, _, _, filt_meta, filt_emb, filt_seq = load_5k_filtered()
    results = train_and_benchmark(
        name=name,
        embeddings=filt_emb,
        metadata=filt_meta,
        sequences=filt_seq,
        device=device,
        **best_kwargs,
    )
    all_results.append(results)
    save_results(all_results)

    ret = results["retrieval_family"]["precision@1"]
    cls = results["classification_family"]["accuracy_mean"]
    print(f"\n  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    return all_results


# ── Step 9: ProtT5-XL at 5K scale ────────────────────────────────


def step9_prott5_5k(all_results: list[dict], device: torch.device) -> list[dict]:
    """ProtT5-XL at 5K protein scale."""
    print(f"\n{'='*60}")
    print("STEP 9: ProtT5-XL at 5K scale")
    print(f"{'='*60}")

    fasta_path = DATA_DIR / "proteins" / "medium_diverse_5k.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "prott5_xl_medium5k.h5"

    # --- Extract ProtT5 embeddings for 5K ---
    if not h5_path.exists():
        print("Extracting ProtT5-XL embeddings for 5K proteins...")
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings

        fasta_dict = read_fasta(fasta_path)
        monitor()
        embs = extract_prot_t5_embeddings(
            fasta_dict,
            model_name="Rostlab/prot_t5_xl_uniref50",
            batch_size=2,
            device=device,
        )
        save_residue_embeddings(embs, h5_path)
    else:
        print("ProtT5-XL 5K embeddings already exist")

    # Load and filter
    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)

    filt_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    filt_emb = {k: v for k, v in embeddings.items() if k in kept_ids}
    filt_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filt_meta))
    embed_dim = next(iter(filt_emb.values())).shape[-1]
    print(f"  {len(filt_emb)} proteins (filtered), {n_fam} families, embed_dim={embed_dim}")

    # --- Mean-pool baseline ---
    mp_name = "meanpool_prott5_5k_filtered"
    if not is_done(all_results, mp_name):
        print(f"\n  Evaluating mean-pool ProtT5 5K (filtered)...")
        mp_results = run_benchmark_suite(
            None, filt_emb, filt_meta, name=mp_name, device=device
        )
        mp_results["embed_dim"] = embed_dim
        mp_results["n_proteins"] = len(filt_emb)
        mp_results["n_families"] = n_fam
        all_results.append(mp_results)
        save_results(all_results)

        ret = mp_results["retrieval_family"]["precision@1"]
        cls = mp_results["classification_family"]["accuracy_mean"]
        print(f"  >> {mp_name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {mp_name} already done")

    # --- AttnPool default config ---
    ap_name = "attnpool_prott5_5k_filtered"
    if not is_done(all_results, ap_name):
        print(f"\n  Training attnpool ProtT5 5K (default config)...")
        results = train_and_benchmark(
            name=ap_name,
            embeddings=filt_emb,
            metadata=filt_meta,
            sequences=filt_seq,
            device=device,
            epochs=100,
        )
        all_results.append(results)
        save_results(all_results)

        ret = results["retrieval_family"]["precision@1"]
        cls = results["classification_family"]["accuracy_mean"]
        print(f"  >> {ap_name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {ap_name} already done")

    # --- AttnPool with best config from Phase 2 ---
    combined = next((r for r in all_results if r["name"] == "ablation_combined"), None)
    if combined:
        ap_best_name = "attnpool_prott5_5k_bestconfig"
        if not is_done(all_results, ap_best_name):
            # Extract config from combined result
            best_kwargs = {}
            for key in ["latent_dim", "K", "n_heads", "n_encoder_layers",
                        "n_proj_layers", "batch_size", "contrastive_weight"]:
                if key in combined and combined[key] != {
                    "latent_dim": 128, "K": 8, "n_heads": 4,
                    "n_encoder_layers": 2, "n_proj_layers": 1,
                    "batch_size": 8, "contrastive_weight": 0.1,
                }.get(key):
                    best_kwargs[key] = combined[key]

            if best_kwargs:
                print(f"\n  Training attnpool ProtT5 5K (best config: {best_kwargs})...")
                results = train_and_benchmark(
                    name=ap_best_name,
                    embeddings=filt_emb,
                    metadata=filt_meta,
                    sequences=filt_seq,
                    device=device,
                    epochs=100,
                    **best_kwargs,
                )
                all_results.append(results)
                save_results(all_results)

                ret = results["retrieval_family"]["precision@1"]
                cls = results["classification_family"]["accuracy_mean"]
                print(f"  >> {ap_best_name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
            else:
                print("  Best config matches defaults, skipping duplicate run.")
        else:
            print(f"  {ap_best_name} already done")
    else:
        print("  Step 8 (combined) not run yet; skipping best-config ProtT5 run.")

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    """Print comparison table."""
    print(f"\n{'='*130}")
    print("FIX-COLLAPSE RESULTS SUMMARY")
    print(f"{'='*130}")
    header = (
        f"{'Name':<40} {'N':<6} {'Fam':<6} {'D->D\'':<10} {'K':<4} "
        f"{'Recon Cos':<12} {'Ret@1':<10} {'Cls':<10} {'Time':<8}"
    )
    print(header)
    print("-" * 130)

    # Load Phase 4 references
    p4_path = BENCHMARKS_DIR / "scale_up_results.json"
    if p4_path.exists():
        with open(p4_path) as f:
            p4_results = json.load(f)
        for r in p4_results:
            if r["name"] in ("meanpool_esm2_650m_5k", "attnpool_esm2_650m_5k"):
                _print_row(r, prefix="[P4] ")
        print("-" * 130)

    for r in all_results:
        _print_row(r)

    # Key comparisons
    print(f"\n{'='*60}")
    print("KEY COMPARISONS")
    print(f"{'='*60}")

    def get(name):
        return next((r for r in all_results if r["name"] == name), None)

    # Filtering impact
    filt = get("meanpool_esm2_650m_5k_filtered")
    unfilt = get("meanpool_esm2_650m_5k_unfiltered_ref")
    if filt and unfilt:
        f_ret = filt["retrieval_family"]["precision@1"]
        u_ret = unfilt["retrieval_family"]["precision@1"]
        print(f"\n  Filtering impact (mean-pool): {u_ret:.3f} -> {f_ret:.3f} ({f_ret - u_ret:+.3f})")

    # Best ablation vs baseline
    baseline = get("attnpool_esm2_650m_5k_filtered")
    combined = get("ablation_combined")
    if baseline and combined:
        b_ret = baseline["retrieval_family"]["precision@1"]
        c_ret = combined["retrieval_family"]["precision@1"]
        print(f"\n  Old checkpoint vs combined: {b_ret:.3f} -> {c_ret:.3f} ({c_ret - b_ret:+.3f})")

    # AttnPool vs mean-pool gap
    if filt and combined:
        mp_ret = filt["retrieval_family"]["precision@1"]
        ap_ret = combined["retrieval_family"]["precision@1"]
        gap = ap_ret - mp_ret
        print(f"\n  AttnPool vs mean-pool gap: {gap:+.3f} (target: within 0.05)")


def _print_row(r: dict, prefix: str = ""):
    name = prefix + r["name"]
    n = r.get("n_proteins", "?")
    n_fam = r.get("n_families", "?")
    embed = r.get("embed_dim", "?")
    latent = r.get("latent_dim", r.get("compression", {}).get("D_prime", "?"))
    K = r.get("K", r.get("compression", {}).get("K", "?"))
    recon_cos = r.get("reconstruction", {}).get("cosine_sim")
    ret = r.get("retrieval_family", {}).get("precision@1")
    cls = r.get("classification_family", {}).get("accuracy_mean")
    t = r.get("training_time_s")
    t_str = f"{t:.0f}s" if t else "-"

    print(
        f"{name:<40} {str(n):<6} {str(n_fam):<6} "
        f"{str(embed)+'->'+str(latent):<10} {str(K):<4} "
        f"{fmt(recon_cos):<12} {fmt(ret):<10} {fmt(cls):<10} {t_str:<8}"
    )


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Fix AttentionPool collapse")
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

    steps = args.step or list(range(1, 10))

    if 1 in steps:
        all_results = step1_reeval_filtered(all_results, device)
    if 2 in steps:
        all_results = step2_latent256(all_results, device)
    if 3 in steps:
        all_results = step3_latent512(all_results, device)
    if 4 in steps:
        all_results = step4_k16(all_results, device)
    if 5 in steps:
        all_results = step5_deep_encoder(all_results, device)
    if 6 in steps:
        all_results = step6_contrastive(all_results, device)
    if 7 in steps:
        all_results = step7_batch32(all_results, device)
    if 8 in steps:
        all_results = step8_combined(all_results, device)
    if 9 in steps:
        all_results = step9_prott5_5k(all_results, device)

    print_summary(all_results)


if __name__ == "__main__":
    main()
