#!/usr/bin/env python3
"""Phase 5 (corrected): Fix evaluation methodology, then ablate.

Fixes applied vs 06_fix_collapse.py:
  - Superfamily-aware train/test split (no homology leakage)
  - Validation-loss checkpointing (not training loss)
  - Multiple seeds per config for variance estimation
  - All ablations run unconditionally
  - Held-out evaluation (not train-set metrics)

Steps:
  0. Data preparation: load, filter, split, save
  1. Baselines: mean-pool, PCA (no training needed) + RNS & inherent info
  2. Default AttnPool (3 seeds) — generalization test
  3. Controlled ablations (2 seeds each, 4 configs)
  4. Pooling strategy evaluation + late interaction + RNS + inherent info
  5. ProtT5 re-validation (2 seeds)

Usage:
  uv run python experiments/07_corrected_eval.py              # run all steps
  uv run python experiments/07_corrected_eval.py --step 0 1   # run specific steps
  uv run python experiments/07_corrected_eval.py --step 2 --seed 42  # single seed
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
from src.evaluation.splitting import (
    load_split,
    save_split,
    split_embeddings,
    split_statistics,
    superfamily_aware_split,
)
from src.extraction.data_loader import (
    filter_by_family_size,
    load_metadata_csv,
    read_fasta,
)
from src.evaluation.embedding_quality import compute_rns, compute_inherent_information
from src.evaluation.late_interaction import evaluate_late_interaction
from src.training.trainer import train_compressor
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "corrected"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
RESULTS_PATH = BENCHMARKS_DIR / "corrected_eval_results.json"
SPLIT_DIR = DATA_DIR / "splits"

DEFAULT_SEEDS = [42, 123, 456]
ABLATION_SEEDS = [42, 123]


# ── Helpers ──────────────────────────────────────────────────────


def monitor():
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
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    if isinstance(v, (float, int)):
        return f"{v:.{p}f}"
    return str(v)


def load_5k_data():
    """Load 5K dataset, filter to families >= 3 members."""
    fasta_path = DATA_DIR / "proteins" / "medium_diverse_5k.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_medium5k.h5"

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)

    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    filt_emb = {k: v for k, v in embeddings.items() if k in kept_ids}
    filt_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filtered_meta))
    print(f"  Loaded: {len(filt_emb)}/{len(embeddings)} proteins, {n_fam} families")
    return filt_emb, filtered_meta, filt_seq


def get_or_create_split(metadata, split_name="esm2_650m_5k"):
    """Get or create the superfamily-aware split."""
    split_path = SPLIT_DIR / f"{split_name}_split.json"
    if split_path.exists():
        print(f"  Loading existing split: {split_path}")
        return load_split(split_path)

    print("  Creating superfamily-aware split (70/30 train/test)...")
    train_ids, test_ids, eval_ids = superfamily_aware_split(
        metadata, test_fraction=0.3, min_test_family_size=2, seed=42
    )
    stats = split_statistics(metadata, train_ids, test_ids, eval_ids)
    save_split(train_ids, test_ids, eval_ids, split_path, stats=stats)

    print(f"  Split saved: {split_path}")
    print(f"  Train: {stats['n_train']} proteins, {stats['n_train_superfamilies']} superfamilies")
    print(f"  Test:  {stats['n_test']} proteins, {stats['n_test_superfamilies']} superfamilies")
    print(f"  Eval:  {stats['n_eval']} proteins (test subset with family >= 2)")
    print(f"  Superfamily overlap: {stats['superfamily_overlap']} (should be 0)")
    assert stats["superfamily_overlap"] == 0, "FATAL: Superfamily leakage in split!"
    return train_ids, test_ids, eval_ids


def train_and_benchmark(
    name: str,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    sequences: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    eval_ids: list[str],
    device: torch.device,
    seed: int = 42,
    K: int = 8,
    latent_dim: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    n_proj_layers: int = 1,
    epochs: int = 100,
    batch_size: int = 8,
    contrastive_weight: float = 0.0,
    recon_weight: float = 1.0,
    masked_weight: float = 0.1,
) -> dict:
    """Train on train_ids, checkpoint on val loss, benchmark on test_ids."""
    embed_dim = next(iter(embeddings.values())).shape[-1]

    model = AttentionPoolCompressor(
        embed_dim, latent_dim, K,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_proj_layers=n_proj_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: D={embed_dim}, D'={latent_dim}, K={K}, heads={n_heads}, "
          f"enc={n_encoder_layers}, seed={seed}")
    print(f"  Parameters: {n_params:,}")
    monitor()

    # Split embeddings for validation
    train_set = set(train_ids)
    test_set = set(test_ids)
    val_emb = {k: v for k, v in embeddings.items() if k in test_set}
    val_seq = {k: v for k, v in sequences.items() if k in test_set}

    ckpt_dir = CHECKPOINTS_DIR / f"{name}_s{seed}"
    start = time.time()
    history = train_compressor(
        model=model,
        embeddings=embeddings,
        sequences=sequences,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        recon_weight=recon_weight,
        masked_weight=masked_weight,
        contrastive_weight=contrastive_weight,
        device=device,
        checkpoint_dir=ckpt_dir,
        log_every=25,
        seed=seed,
        protein_ids=train_set,
        validation_embeddings=val_emb,
        validation_sequences=val_seq,
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.0f}s (best epoch: {history['best_epoch']})")
    monitor()

    # Load best checkpoint
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Benchmark on held-out test set
    results = run_benchmark_suite(
        model, embeddings, metadata, name=name, device=device,
        train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
    )
    results["training_time_s"] = elapsed
    results["seed"] = seed
    results["embed_dim"] = embed_dim
    results["K"] = K
    results["latent_dim"] = latent_dim
    results["n_heads"] = n_heads
    results["n_encoder_layers"] = n_encoder_layers
    results["n_proj_layers"] = n_proj_layers
    results["n_train"] = len([i for i in train_ids if i in embeddings])
    results["n_test"] = len([i for i in test_ids if i in embeddings])
    results["n_eval"] = len([i for i in eval_ids if i in embeddings])
    results["epochs"] = epochs
    results["batch_size"] = batch_size
    results["contrastive_weight"] = contrastive_weight
    results["best_epoch"] = history["best_epoch"]
    return results


# ── Step 0: Data Preparation ────────────────────────────────────


def step0_prepare(device: torch.device):
    print(f"\n{'='*60}")
    print("STEP 0: Data preparation")
    print(f"{'='*60}")

    filt_emb, filt_meta, filt_seq = load_5k_data()
    train_ids, test_ids, eval_ids = get_or_create_split(filt_meta)

    # Verify split integrity
    stats = split_statistics(filt_meta, train_ids, test_ids, eval_ids)
    print(f"\n  Split verification:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    return filt_emb, filt_meta, filt_seq, train_ids, test_ids, eval_ids


# ── Step 1: Baselines ───────────────────────────────────────────


def step1_baselines(all_results, device, embeddings, metadata, sequences,
                    train_ids, test_ids, eval_ids):
    print(f"\n{'='*60}")
    print("STEP 1: Baselines (no training)")
    print(f"{'='*60}")

    # Mean-pool baseline
    name = "baseline_meanpool_esm2_650m"
    if not is_done(all_results, name):
        print(f"\n  Evaluating {name}...")
        results = run_benchmark_suite(
            None, embeddings, metadata, name=name, device=device,
            train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
        )
        results["embed_dim"] = next(iter(embeddings.values())).shape[-1]
        results["n_train"] = len([i for i in train_ids if i in embeddings])
        results["n_test"] = len([i for i in test_ids if i in embeddings])
        results["n_eval"] = len([i for i in eval_ids if i in embeddings])
        all_results.append(results)
        save_results(all_results)
        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {name} already done")

    # Embedding quality metrics on mean-pooled baseline (test set only)
    test_set = set(test_ids)
    mp_vectors = {pid: emb.mean(axis=0) for pid, emb in embeddings.items()
                  if pid in test_set}
    test_meta = [m for m in metadata if m["id"] in test_set]

    name_rns = "quality_meanpool_esm2_650m"
    if not is_done(all_results, name_rns):
        print(f"\n  Computing RNS on mean-pooled test set...")
        rns = compute_rns(mp_vectors, n_random=1000, k=10, seed=42)
        print(f"  >> RNS: mean={rns['rns_mean']:.3f}, unreliable={rns['frac_unreliable']:.1%}")

        print(f"  Computing inherent information (family)...")
        inherent_fam = compute_inherent_information(mp_vectors, test_meta, label_key="family")
        print(f"  >> kNN purity@1={inherent_fam['knn_purity_k1']:.3f}, "
              f"silhouette={inherent_fam['silhouette']:.3f}")

        print(f"  Computing inherent information (superfamily)...")
        inherent_sf = compute_inherent_information(mp_vectors, test_meta, label_key="superfamily")
        print(f"  >> kNN purity@1={inherent_sf['knn_purity_k1']:.3f}, "
              f"silhouette={inherent_sf['silhouette']:.3f}")

        quality_results = {
            "name": name_rns,
            "rns": rns,
            "inherent_family": inherent_fam,
            "inherent_superfamily": inherent_sf,
        }
        all_results.append(quality_results)
        save_results(all_results)
    else:
        print(f"  {name_rns} already done")

    # PCA baselines
    from sklearn.decomposition import PCA

    train_set = set(train_ids)

    for n_components in [128, 256, 512]:
        name = f"baseline_pca{n_components}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Evaluating {name}...")
        # Fit PCA on train mean-pooled
        train_vecs = []
        train_pids = []
        for pid, emb in embeddings.items():
            if pid in train_set:
                train_vecs.append(emb.mean(axis=0))
                train_pids.append(pid)

        X_train = np.array(train_vecs)
        n_comp = min(n_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_train)

        # Transform all proteins
        all_vecs = {}
        for pid, emb in embeddings.items():
            mp = emb.mean(axis=0)
            all_vecs[pid] = pca.transform(mp.reshape(1, -1))[0]

        # Evaluate retrieval and classification using the PCA vectors
        from src.evaluation.retrieval import evaluate_retrieval
        from src.evaluation.classification import evaluate_linear_probe

        id_to_label = {m["id"]: m["family"] for m in metadata}
        eval_emb = {pid: all_vecs[pid] for pid in all_vecs}

        # For retrieval, create mock "embeddings" that are already 1D (no per-residue)
        # We need to call evaluate_retrieval with model=None and pre-pooled vectors
        # Hack: wrap vectors as (1, D) arrays so mean-pool returns them as-is
        mock_emb = {pid: vec.reshape(1, -1) for pid, vec in all_vecs.items()}
        ret_results = evaluate_retrieval(
            None, mock_emb, metadata, label_key="family",
            query_ids=eval_ids, database_ids=test_ids,
        )
        cls_results = evaluate_linear_probe(
            None, mock_emb, metadata, label_key="family",
            train_ids=train_ids, test_ids=test_ids,
        )

        results = {
            "name": name,
            "split_mode": "held_out",
            "n_components": n_comp,
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "retrieval_family": ret_results,
            "classification_family": cls_results,
            "n_train": len(train_pids),
            "n_test": len([i for i in test_ids if i in embeddings]),
            "n_eval": len([i for i in eval_ids if i in embeddings]),
        }
        all_results.append(results)
        save_results(all_results)
        ret = ret_results.get("precision@1", 0)
        cls_acc = cls_results.get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}, "
              f"ExplVar={results['explained_variance']:.3f}")

    # L2-weighted mean baseline
    name = "baseline_l2_weighted_mean"
    if not is_done(all_results, name):
        print(f"\n  Evaluating {name}...")
        mock_emb = {}
        for pid, emb in embeddings.items():
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            weights = norms / (norms.sum() + 1e-8)
            weighted = (emb * weights).sum(axis=0)
            mock_emb[pid] = weighted.reshape(1, -1)

        from src.evaluation.retrieval import evaluate_retrieval
        from src.evaluation.classification import evaluate_linear_probe

        ret_results = evaluate_retrieval(
            None, mock_emb, metadata, label_key="family",
            query_ids=eval_ids, database_ids=test_ids,
        )
        cls_results = evaluate_linear_probe(
            None, mock_emb, metadata, label_key="family",
            train_ids=train_ids, test_ids=test_ids,
        )
        results = {
            "name": name,
            "split_mode": "held_out",
            "retrieval_family": ret_results,
            "classification_family": cls_results,
            "n_train": len([i for i in train_ids if i in embeddings]),
            "n_test": len([i for i in test_ids if i in embeddings]),
        }
        all_results.append(results)
        save_results(all_results)
        ret = ret_results.get("precision@1", 0)
        cls_acc = cls_results.get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")
    else:
        print(f"  {name} already done")

    return all_results


# ── Step 2: Default AttnPool (3 seeds) ──────────────────────────


def step2_default(all_results, device, embeddings, metadata, sequences,
                  train_ids, test_ids, eval_ids, seeds=None):
    print(f"\n{'='*60}")
    print("STEP 2: Default AttnPool (generalization test)")
    print(f"{'='*60}")

    if seeds is None:
        seeds = DEFAULT_SEEDS

    for seed in seeds:
        name = f"attnpool_default_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results = train_and_benchmark(
            name=name,
            embeddings=embeddings, metadata=metadata, sequences=sequences,
            train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
            device=device, seed=seed,
        )
        all_results.append(results)
        save_results(all_results)

        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")

    # Print seed summary
    seed_results = [r for r in all_results if r["name"].startswith("attnpool_default_s")]
    if len(seed_results) >= 2:
        rets = [r["retrieval_family"]["precision@1"] for r in seed_results]
        clss = [r["classification_family"]["accuracy_mean"] for r in seed_results]
        print(f"\n  Default AttnPool summary ({len(seed_results)} seeds):")
        print(f"    Ret@1: {np.mean(rets):.3f} +/- {np.std(rets):.3f}")
        print(f"    Cls:   {np.mean(clss):.3f} +/- {np.std(clss):.3f}")

    return all_results


# ── Step 3: Controlled Ablations ─────────────────────────────────


ABLATIONS = {
    "ablation_latent256": {
        "description": "latent_dim=256 (5x compression vs 10x)",
        "latent_dim": 256,
    },
    "ablation_latent512": {
        "description": "latent_dim=512 (2.5x compression)",
        "latent_dim": 512,
    },
    "ablation_contrastive_batch32": {
        "description": "batch_size=32, contrastive_weight=0.3",
        "batch_size": 32,
        "contrastive_weight": 0.3,
    },
    "ablation_deep_wide": {
        "description": "n_heads=8, n_encoder_layers=4",
        "n_heads": 8,
        "n_encoder_layers": 4,
    },
}


def step3_ablations(all_results, device, embeddings, metadata, sequences,
                    train_ids, test_ids, eval_ids, seeds=None):
    print(f"\n{'='*60}")
    print("STEP 3: Controlled ablations (all unconditional)")
    print(f"{'='*60}")

    if seeds is None:
        seeds = ABLATION_SEEDS

    for abl_name, abl_config in ABLATIONS.items():
        desc = abl_config.pop("description", abl_name)
        print(f"\n  --- {desc} ---")

        for seed in seeds:
            name = f"{abl_name}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            results = train_and_benchmark(
                name=name,
                embeddings=embeddings, metadata=metadata, sequences=sequences,
                train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
                device=device, seed=seed,
                **abl_config,
            )
            all_results.append(results)
            save_results(all_results)

            ret = results.get("retrieval_family", {}).get("precision@1", 0)
            cls = results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")

        # Re-add description for future iterations
        abl_config["description"] = desc

    return all_results


# ── Step 4: Pooling Strategy Evaluation ──────────────────────────


def step4_pooling_strategies(all_results, device, embeddings, metadata,
                             train_ids, test_ids, eval_ids):
    print(f"\n{'='*60}")
    print("STEP 4: Pooling strategy evaluation (no retraining)")
    print(f"{'='*60}")

    strategies = ["first", "mean_std", "concat"]

    # Find all trained checkpoints
    ckpt_configs = []

    # Default models
    for seed in DEFAULT_SEEDS:
        ckpt_dir = CHECKPOINTS_DIR / f"attnpool_default_s{seed}_s{seed}"
        if not ckpt_dir.exists():
            ckpt_dir = CHECKPOINTS_DIR / f"attnpool_default_s{seed}"
        if (ckpt_dir / "best_model.pt").exists():
            ckpt_configs.append(("attnpool_default", seed, 128, 8, 4, 2))

    # Ablation models
    for abl_name, abl_config in ABLATIONS.items():
        latent_dim = abl_config.get("latent_dim", 128)
        K = abl_config.get("K", 8)
        n_heads = abl_config.get("n_heads", 4)
        n_enc = abl_config.get("n_encoder_layers", 2)

        for seed in ABLATION_SEEDS:
            ckpt_dir = CHECKPOINTS_DIR / f"{abl_name}_s{seed}_s{seed}"
            if not ckpt_dir.exists():
                ckpt_dir = CHECKPOINTS_DIR / f"{abl_name}_s{seed}"
            if (ckpt_dir / "best_model.pt").exists():
                ckpt_configs.append((abl_name, seed, latent_dim, K, n_heads, n_enc))

    embed_dim = next(iter(embeddings.values())).shape[-1]

    for base_name, seed, latent_dim, K, n_heads, n_enc in ckpt_configs:
        for strategy in strategies:
            name = f"{base_name}_s{seed}_pool_{strategy}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            ckpt_path = CHECKPOINTS_DIR / f"{base_name}_s{seed}" / "best_model.pt"
            if not ckpt_path.exists():
                ckpt_path = CHECKPOINTS_DIR / f"{base_name}_s{seed}_s{seed}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  WARNING: Checkpoint not found for {name}, skipping")
                continue

            model = AttentionPoolCompressor(
                embed_dim, latent_dim, K,
                n_heads=n_heads, n_encoder_layers=n_enc,
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            model = model.to(device)

            print(f"  Evaluating {name}...")
            results = run_benchmark_suite(
                model, embeddings, metadata, name=name, device=device,
                train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
                pooling_strategy=strategy,
            )
            results["pooling_strategy"] = strategy
            results["base_model"] = base_name
            results["seed"] = seed
            all_results.append(results)
            save_results(all_results)

            ret = results.get("retrieval_family", {}).get("precision@1", 0)
            cls = results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")

    # ── Late-interaction retrieval + RNS + inherent info per checkpoint ──

    print(f"\n  --- Late interaction retrieval & quality metrics ---")
    test_set = set(test_ids)
    test_meta = [m for m in metadata if m["id"] in test_set]

    for base_name, seed, latent_dim, K, n_heads, n_enc in ckpt_configs:
        name = f"{base_name}_s{seed}_late_interaction"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        ckpt_path = CHECKPOINTS_DIR / f"{base_name}_s{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINTS_DIR / f"{base_name}_s{seed}_s{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found for {name}, skipping")
            continue

        model = AttentionPoolCompressor(
            embed_dim, latent_dim, K,
            n_heads=n_heads, n_encoder_layers=n_enc,
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        # Late interaction retrieval
        print(f"  Late interaction: {name}...")
        late_ret = evaluate_late_interaction(
            model, embeddings, metadata, label_key="family",
            query_ids=eval_ids, database_ids=test_ids, device=device,
        )
        print(f"  >> Late interaction Ret@1={late_ret.get('precision@1', 0):.3f}")

        # RNS on mean-pooled compressed embeddings
        from src.evaluation.retrieval import _get_latent_vectors
        comp_vectors = _get_latent_vectors(model, embeddings, device)
        test_vectors = {pid: v for pid, v in comp_vectors.items() if pid in test_set}

        print(f"  RNS on compressed embeddings...")
        rns = compute_rns(test_vectors, n_random=1000, k=10, seed=42)
        print(f"  >> RNS: mean={rns['rns_mean']:.3f}, unreliable={rns['frac_unreliable']:.1%}")

        # Inherent information on compressed embeddings
        print(f"  Inherent information (family)...")
        inherent = compute_inherent_information(test_vectors, test_meta, label_key="family")
        print(f"  >> kNN purity@1={inherent['knn_purity_k1']:.3f}, "
              f"silhouette={inherent['silhouette']:.3f}")

        monitor()

        results = {
            "name": name,
            "base_model": base_name,
            "seed": seed,
            "late_interaction_retrieval": late_ret,
            "rns": rns,
            "inherent_family": inherent,
        }
        all_results.append(results)
        save_results(all_results)

    return all_results


# ── Step 5: ProtT5 Re-validation ─────────────────────────────────


def step5_prott5(all_results, device):
    print(f"\n{'='*60}")
    print("STEP 5: ProtT5-XL re-validation (497 proteins)")
    print(f"{'='*60}")

    # Use the 500-protein dataset with ProtT5 embeddings
    fasta_path = DATA_DIR / "proteins" / "small_diverse_500.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_500.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "prott5_xl_small500.h5"

    if not h5_path.exists():
        print(f"  ProtT5 embeddings not found: {h5_path}")
        print("  Skipping ProtT5 re-validation.")
        return all_results

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)

    # Filter to families >= 3
    filt_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    filt_emb = {k: v for k, v in embeddings.items() if k in kept_ids}
    filt_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filt_meta))
    print(f"  ProtT5 dataset: {len(filt_emb)} proteins, {n_fam} families")

    # Create split for this dataset
    train_ids, test_ids, eval_ids = get_or_create_split(filt_meta, split_name="prott5_500")
    stats = split_statistics(filt_meta, train_ids, test_ids, eval_ids)
    print(f"  Train: {stats['n_train']}, Test: {stats['n_test']}, Eval: {stats['n_eval']}")
    print(f"  Superfamily overlap: {stats['superfamily_overlap']}")

    # Mean-pool baseline
    name = "baseline_meanpool_prott5_500"
    if not is_done(all_results, name):
        print(f"\n  Evaluating {name}...")
        results = run_benchmark_suite(
            None, filt_emb, filt_meta, name=name, device=device,
            train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
        )
        results["embed_dim"] = next(iter(filt_emb.values())).shape[-1]
        all_results.append(results)
        save_results(all_results)
        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")
    else:
        print(f"  {name} already done")

    # AttnPool on ProtT5 (2 seeds)
    for seed in [42, 123]:
        name = f"attnpool_prott5_500_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results = train_and_benchmark(
            name=name,
            embeddings=filt_emb, metadata=filt_meta, sequences=filt_seq,
            train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
            device=device, seed=seed,
            epochs=200,
        )
        all_results.append(results)
        save_results(all_results)

        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls:.3f}")

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    print(f"\n{'='*120}")
    print("CORRECTED EVALUATION RESULTS")
    print(f"{'='*120}")

    header = (
        f"{'Name':<45} {'Split':<10} {'Ret@1':<10} {'Cls':<10} "
        f"{'Recon Cos':<12} {'Time':<8} {'Seed':<6}"
    )
    print(header)
    print("-" * 120)

    # Group by base name for averaging
    from collections import defaultdict
    groups = defaultdict(list)

    for r in all_results:
        name = r["name"]
        ret = r.get("retrieval_family", {}).get("precision@1")
        cls = r.get("classification_family", {}).get("accuracy_mean")
        recon = r.get("reconstruction", {}).get("cosine_sim")
        t = r.get("training_time_s")
        t_str = f"{t:.0f}s" if t else "-"
        seed = r.get("seed", "-")
        split = r.get("split_mode", "?")

        print(
            f"{name:<45} {split:<10} {fmt(ret):<10} {fmt(cls):<10} "
            f"{fmt(recon):<12} {t_str:<8} {str(seed):<6}"
        )

        # Group seeds
        base = name.rsplit("_s", 1)[0] if "_s" in name and name[-1].isdigit() else name
        if ret is not None:
            groups[base].append({"ret": ret, "cls": cls})

    # Print seed averages
    print(f"\n{'='*80}")
    print("SEED AVERAGES")
    print(f"{'='*80}")
    print(f"{'Config':<45} {'N':<4} {'Ret@1 mean':<12} {'Ret@1 std':<12} {'Cls mean':<12}")
    print("-" * 80)

    for base, runs in sorted(groups.items()):
        if len(runs) >= 2:
            rets = [r["ret"] for r in runs]
            clss = [r["cls"] for r in runs if r["cls"] is not None]
            if clss:
                print(
                    f"{base:<45} {len(runs):<4} "
                    f"{np.mean(rets):.3f}        {np.std(rets):.3f}        "
                    f"{np.mean(clss):.3f}"
                )

    # Print quality metrics summary
    quality_results = [r for r in all_results if "rns" in r]
    if quality_results:
        print(f"\n{'='*100}")
        print("EMBEDDING QUALITY METRICS")
        print(f"{'='*100}")
        print(f"{'Name':<45} {'RNS mean':<10} {'Unreliable':<12} "
              f"{'kNN@1':<10} {'Silhouette':<12}")
        print("-" * 100)
        for r in quality_results:
            rns = r.get("rns", {})
            inh = r.get("inherent_family", {})
            print(
                f"{r['name']:<45} "
                f"{fmt(rns.get('rns_mean')):<10} "
                f"{fmt(rns.get('frac_unreliable')):<12} "
                f"{fmt(inh.get('knn_purity_k1')):<10} "
                f"{fmt(inh.get('silhouette')):<12}"
            )

    # Print late interaction results
    late_results = [r for r in all_results if "late_interaction_retrieval" in r]
    if late_results:
        print(f"\n{'='*80}")
        print("LATE INTERACTION RETRIEVAL (ColBERT-style)")
        print(f"{'='*80}")
        print(f"{'Name':<45} {'Ret@1':<10} {'Ret@5':<10} {'Mean':<10}")
        print("-" * 80)
        for r in late_results:
            lr = r["late_interaction_retrieval"]
            print(
                f"{r['name']:<45} "
                f"{fmt(lr.get('precision@1')):<10} "
                f"{fmt(lr.get('precision@5')):<10} "
                f"{fmt(lr.get('mean_precision')):<10}"
            )


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 5 (corrected): Fix evaluation")
    parser.add_argument(
        "--step", type=int, nargs="*", default=None,
        help="Run specific step(s). Default: all.",
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=None,
        help="Override seeds for steps 2-3 (e.g. --seed 42).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS_PATH}")

    all_results = load_results()
    print(f"Loaded {len(all_results)} existing results")

    steps = args.step or list(range(0, 6))

    # Step 0 always runs to prepare data
    filt_emb, filt_meta, filt_seq, train_ids, test_ids, eval_ids = step0_prepare(device)

    if 1 in steps:
        all_results = step1_baselines(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids,
        )

    if 2 in steps:
        seeds = args.seed or DEFAULT_SEEDS
        all_results = step2_default(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, seeds=seeds,
        )

    if 3 in steps:
        seeds = args.seed or ABLATION_SEEDS
        all_results = step3_ablations(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, seeds=seeds,
        )

    if 4 in steps:
        all_results = step4_pooling_strategies(
            all_results, device, filt_emb, filt_meta,
            train_ids, test_ids, eval_ids,
        )

    if 5 in steps:
        all_results = step5_prott5(all_results, device)

    print_summary(all_results)


if __name__ == "__main__":
    main()
