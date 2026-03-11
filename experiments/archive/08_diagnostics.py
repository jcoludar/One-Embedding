#!/usr/bin/env python3
"""Phase 6: Diagnose why AttnPool loses to PCA, then fix it.

Phase A — Diagnostics (isolate root cause):
  A1. Token diversity measurement (no training)
  A2. Distance alignment: mean-pool vs compressed space (no training)
  A3. Nonlinear MLP autoencoder on mean-pooled vectors
  A4. PCA-initialized AttnPool

Phase B — Fixes (based on diagnostic results):
  B1. Pooled reconstruction loss
  B2. VICReg regularization
  B3. Token orthogonality loss
  B4. Combined: PCA-init + pooled recon + VICReg + token ortho
  B5. Family-stratified classification evaluation

Usage:
  uv run python experiments/08_diagnostics.py                  # run all
  uv run python experiments/08_diagnostics.py --step A1 A2     # diagnostics only
  uv run python experiments/08_diagnostics.py --step B4        # combined fix
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from src.compressors.attention_pool import AttentionPoolCompressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results
from src.evaluation.embedding_quality import compute_token_diversity
from src.evaluation.splitting import (
    family_stratified_split,
    load_split,
    save_split,
    split_statistics,
    superfamily_aware_split,
)
from src.extraction.data_loader import (
    filter_by_family_size,
    load_metadata_csv,
    read_fasta,
)
from src.training.trainer import train_compressor
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "diagnostics"
RESULTS_PATH = DATA_DIR / "benchmarks" / "diagnostics_results.json"
SPLIT_DIR = DATA_DIR / "splits"

SEEDS = [42, 123]


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


def get_splits(metadata):
    """Get both superfamily-aware (retrieval) and family-stratified (classification) splits."""
    # Superfamily-aware split (for retrieval)
    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    if sf_split_path.exists():
        train_ids, test_ids, eval_ids = load_split(sf_split_path)
    else:
        train_ids, test_ids, eval_ids = superfamily_aware_split(
            metadata, test_fraction=0.3, seed=42
        )
        stats = split_statistics(metadata, train_ids, test_ids, eval_ids)
        save_split(train_ids, test_ids, eval_ids, sf_split_path, stats=stats)

    # Family-stratified split (for classification)
    cls_split_path = SPLIT_DIR / "esm2_650m_5k_cls_split.json"
    if cls_split_path.exists():
        with open(cls_split_path) as f:
            cls_data = json.load(f)
        cls_train_ids = cls_data["cls_train_ids"]
        cls_test_ids = cls_data["cls_test_ids"]
    else:
        cls_train_ids, cls_test_ids = family_stratified_split(
            metadata, test_fraction=0.3, min_family_size=2, seed=42
        )
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(cls_split_path, "w") as f:
            json.dump({"cls_train_ids": cls_train_ids, "cls_test_ids": cls_test_ids}, f, indent=2)
        print(f"  Saved classification split: {cls_split_path}")

    return train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids


def train_and_benchmark(
    name, embeddings, metadata, sequences,
    train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
    device, seed=42, model=None, **train_kwargs,
):
    """Train (if model provided or created from kwargs), then benchmark."""
    embed_dim = next(iter(embeddings.values())).shape[-1]
    K = train_kwargs.pop("K", 8)
    latent_dim = train_kwargs.pop("latent_dim", 128)
    n_heads = train_kwargs.pop("n_heads", 4)
    n_encoder_layers = train_kwargs.pop("n_encoder_layers", 2)
    n_decoder_layers = train_kwargs.pop("n_decoder_layers", 2)
    init_proj_weights = train_kwargs.pop("init_proj_weights", None)
    epochs = train_kwargs.pop("epochs", 100)
    batch_size = train_kwargs.pop("batch_size", 8)

    if model is None:
        model = AttentionPoolCompressor(
            embed_dim, latent_dim, K,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            init_proj_weights=init_proj_weights,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: D={embed_dim}, D'={latent_dim}, K={K}, params={n_params:,}, seed={seed}")
    monitor()

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
        device=device,
        checkpoint_dir=ckpt_dir,
        log_every=25,
        seed=seed,
        protein_ids=train_set,
        validation_embeddings=val_emb,
        validation_sequences=val_seq,
        **train_kwargs,
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.0f}s (best epoch: {history['best_epoch']})")
    monitor()

    # Load best checkpoint
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Benchmark with both retrieval and classification splits
    results = run_benchmark_suite(
        model, embeddings, metadata, name=name, device=device,
        train_ids=train_ids, test_ids=test_ids, eval_ids=eval_ids,
        cls_train_ids=cls_train_ids, cls_test_ids=cls_test_ids,
    )
    results["training_time_s"] = elapsed
    results["seed"] = seed
    results["embed_dim"] = embed_dim
    results["K"] = K
    results["latent_dim"] = latent_dim
    results["n_params"] = n_params
    results["best_epoch"] = history["best_epoch"]
    return results, model


# ── A1: Token Diversity ──────────────────────────────────────────


def step_a1(all_results, device, embeddings):
    print(f"\n{'='*60}")
    print("A1: Token diversity measurement (no training)")
    print(f"{'='*60}")

    # Load existing Phase 5 checkpoints
    phase5_ckpt_dir = DATA_DIR / "checkpoints" / "corrected"
    configs = [
        ("attnpool_default", 42, 128, 8, 4, 2),
        ("attnpool_default", 123, 128, 8, 4, 2),
        ("attnpool_default", 456, 128, 8, 4, 2),
        ("ablation_latent256", 42, 256, 8, 4, 2),
        ("ablation_latent512", 42, 512, 8, 4, 2),
        ("ablation_deep_wide", 42, 128, 8, 8, 4),
    ]

    embed_dim = next(iter(embeddings.values())).shape[-1]

    for base_name, seed, latent_dim, K, n_heads, n_enc in configs:
        name = f"token_diversity_{base_name}_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        ckpt_path = phase5_ckpt_dir / f"{base_name}_s{seed}_s{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue

        model = AttentionPoolCompressor(
            embed_dim, latent_dim, K,
            n_heads=n_heads, n_encoder_layers=n_enc,
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )

        print(f"  Computing token diversity for {base_name}_s{seed}...")
        diversity = compute_token_diversity(model, embeddings, device=device)
        print(f"  >> Pairwise cos: {diversity['mean_pairwise_cos']:.4f} +/- {diversity['std_pairwise_cos']:.4f}")
        print(f"  >> Effective rank: {diversity['mean_effective_rank']:.2f} +/- {diversity['std_effective_rank']:.2f}")

        result = {"name": name, "base_model": base_name, "seed": seed, **diversity}
        all_results.append(result)
        save_results(all_results)

    return all_results


# ── A2: Distance Alignment ───────────────────────────────────────


def step_a2(all_results, device, embeddings, metadata, test_ids):
    print(f"\n{'='*60}")
    print("A2: Distance alignment diagnostic (no training)")
    print(f"{'='*60}")

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    name = "distance_alignment"
    if is_done(all_results, name):
        print(f"  {name} already done")
        return all_results

    test_set = set(test_ids)
    test_pids = [pid for pid in test_ids if pid in embeddings]
    id_to_fam = {m["id"]: m["family"] for m in metadata}

    # Mean-pool space
    mp_vecs = {pid: embeddings[pid].mean(axis=0) for pid in test_pids}

    # Compressed space (load default checkpoint)
    embed_dim = next(iter(embeddings.values())).shape[-1]
    phase5_ckpt = DATA_DIR / "checkpoints" / "corrected" / "attnpool_default_s42_s42" / "best_model.pt"
    if not phase5_ckpt.exists():
        print(f"  WARNING: Default checkpoint not found, skipping A2")
        return all_results

    model = AttentionPoolCompressor(embed_dim, 128, 8, n_heads=4, n_encoder_layers=2)
    model.load_state_dict(torch.load(phase5_ckpt, map_location=device, weights_only=True), strict=False)
    model = model.to(device).eval()

    comp_vecs = {}
    with torch.no_grad():
        for pid in test_pids:
            emb = embeddings[pid]
            L = min(emb.shape[0], 512)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).float().to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)
            pooled = model.get_pooled(latent)
            comp_vecs[pid] = pooled[0].cpu().numpy()

    # Pairwise analysis (subsample for speed)
    n = min(len(test_pids), 500)
    pids = test_pids[:n]

    mp_sims, comp_sims, same_fam = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pids[i], pids[j]
            mp_sim = np.dot(mp_vecs[pi], mp_vecs[pj]) / (
                np.linalg.norm(mp_vecs[pi]) * np.linalg.norm(mp_vecs[pj]) + 1e-8
            )
            comp_sim = np.dot(comp_vecs[pi], comp_vecs[pj]) / (
                np.linalg.norm(comp_vecs[pi]) * np.linalg.norm(comp_vecs[pj]) + 1e-8
            )
            mp_sims.append(mp_sim)
            comp_sims.append(comp_sim)
            same_fam.append(1 if id_to_fam.get(pi) == id_to_fam.get(pj) else 0)

    mp_sims = np.array(mp_sims)
    comp_sims = np.array(comp_sims)
    same_fam = np.array(same_fam)

    # Rank correlation between spaces
    rho, pval = spearmanr(mp_sims, comp_sims)

    # ROC-AUC for same-family
    mp_auc = roc_auc_score(same_fam, mp_sims)
    comp_auc = roc_auc_score(same_fam, comp_sims)

    print(f"  Spearman rank correlation (mp vs comp): rho={rho:.4f}, p={pval:.2e}")
    print(f"  ROC-AUC (same-family): mean-pool={mp_auc:.4f}, compressed={comp_auc:.4f}")
    print(f"  AUC gap: {mp_auc - comp_auc:.4f}")

    result = {
        "name": name,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "auc_meanpool": float(mp_auc),
        "auc_compressed": float(comp_auc),
        "auc_gap": float(mp_auc - comp_auc),
        "n_pairs": len(mp_sims),
        "n_same_family": int(same_fam.sum()),
    }
    all_results.append(result)
    save_results(all_results)
    return all_results


# ── A3: MLP Autoencoder on Mean-Pooled ───────────────────────────


class MeanPoolAutoencoder(nn.Module):
    """Simple MLP autoencoder on mean-pooled protein embeddings."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def step_a3(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("A3: Nonlinear MLP autoencoder on mean-pooled")
    print(f"{'='*60}")

    from src.evaluation.retrieval import evaluate_retrieval
    from src.evaluation.classification import evaluate_linear_probe

    embed_dim = next(iter(embeddings.values())).shape[-1]
    train_set = set(train_ids)

    for latent_dim in [128, 256]:
        name = f"mlp_ae_meanpool_d{latent_dim}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")

        # Prepare mean-pooled vectors
        train_vecs = []
        train_pids = []
        all_vecs = {}
        for pid, emb in embeddings.items():
            mp = emb.mean(axis=0)
            all_vecs[pid] = mp
            if pid in train_set:
                train_vecs.append(mp)
                train_pids.append(pid)

        X_train = np.array(train_vecs, dtype=np.float32)
        X_train_t = torch.from_numpy(X_train).to(device)

        model = MeanPoolAutoencoder(embed_dim, latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        # Train
        start = time.time()
        best_loss = float("inf")
        for epoch in range(1, 201):
            model.train()
            # Mini-batch
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0
            n_batches = 0
            for i in range(0, len(perm), 64):
                batch = X_train_t[perm[i:i+64]]
                optimizer.zero_grad()
                recon, z = model(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % 50 == 0:
                print(f"    Epoch {epoch}/200 | Loss={avg_loss:.6f}")

        elapsed = time.time() - start
        print(f"  Training done in {elapsed:.0f}s")

        # Encode all proteins
        model.eval()
        encoded_vecs = {}
        with torch.no_grad():
            for pid, mp in all_vecs.items():
                x = torch.from_numpy(mp).unsqueeze(0).float().to(device)
                z = model.encode(x)
                encoded_vecs[pid] = z[0].cpu().numpy()

        # Evaluate retrieval
        mock_emb = {pid: vec.reshape(1, -1) for pid, vec in encoded_vecs.items()}
        ret_results = evaluate_retrieval(
            None, mock_emb, metadata, label_key="family",
            query_ids=eval_ids, database_ids=test_ids,
        )
        cls_results = evaluate_linear_probe(
            None, mock_emb, metadata, label_key="family",
            train_ids=cls_train_ids, test_ids=cls_test_ids,
        )

        result = {
            "name": name,
            "split_mode": "held_out",
            "latent_dim": latent_dim,
            "retrieval_family": ret_results,
            "classification_family": cls_results,
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        ret = ret_results.get("precision@1", 0)
        cls_acc = cls_results.get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── A4: PCA-Initialized AttnPool ─────────────────────────────────


def step_a4(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("A4: PCA-initialized AttnPool")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    train_set = set(train_ids)

    # Fit PCA on train mean-pooled
    train_vecs = np.array([
        embeddings[pid].mean(axis=0) for pid in train_ids if pid in embeddings
    ])
    pca = PCA(n_components=128, random_state=42)
    pca.fit(train_vecs)

    # PCA components as projection weights
    pca_weight = pca.components_.astype(np.float32)  # (128, 1280)
    pca_bias = (-pca.mean_ @ pca.components_.T).astype(np.float32)  # (128,)

    for seed in SEEDS:
        # Trainable PCA-init
        name = f"pca_init_attnpool_s{seed}"
        if not is_done(all_results, name):
            print(f"\n  Training {name}...")
            results, _ = train_and_benchmark(
                name, embeddings, metadata, sequences,
                train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
                device, seed=seed,
                init_proj_weights=(pca_weight, pca_bias),
                recon_weight=1.0, masked_weight=0.1,
            )
            all_results.append(results)
            save_results(all_results)
            ret = results.get("retrieval_family", {}).get("precision@1", 0)
            cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")
        else:
            print(f"  {name} already done")

        # Frozen PCA projection
        name_frozen = f"pca_frozen_attnpool_s{seed}"
        if not is_done(all_results, name_frozen):
            print(f"\n  Training {name_frozen}...")
            model = AttentionPoolCompressor(
                embed_dim, 128, 8, n_heads=4, n_encoder_layers=2,
                init_proj_weights=(pca_weight, pca_bias),
            )
            # Freeze input projection
            for p in model.input_proj.parameters():
                p.requires_grad = False

            results, _ = train_and_benchmark(
                name_frozen, embeddings, metadata, sequences,
                train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
                device, seed=seed, model=model,
                recon_weight=1.0, masked_weight=0.1,
            )
            all_results.append(results)
            save_results(all_results)
            ret = results.get("retrieval_family", {}).get("precision@1", 0)
            cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {name_frozen}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")
        else:
            print(f"  {name_frozen} already done")

    return all_results


# ── B1: Pooled Reconstruction Loss ───────────────────────────────


def step_b1(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B1: Pooled reconstruction loss")
    print(f"{'='*60}")

    for seed in SEEDS:
        name = f"pool_recon_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results, _ = train_and_benchmark(
            name, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
            device, seed=seed,
            recon_weight=0.1, masked_weight=0.1, pool_recon_weight=1.0,
        )
        all_results.append(results)
        save_results(all_results)
        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── B2: VICReg ───────────────────────────────────────────────────


def step_b2(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B2: VICReg regularization")
    print(f"{'='*60}")

    for seed in SEEDS:
        name = f"vicreg_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results, _ = train_and_benchmark(
            name, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
            device, seed=seed,
            recon_weight=1.0, masked_weight=0.1, vicreg_weight=0.1,
        )
        all_results.append(results)
        save_results(all_results)
        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── B3: Token Orthogonality ──────────────────────────────────────


def step_b3(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B3: Token orthogonality loss")
    print(f"{'='*60}")

    for seed in SEEDS:
        name = f"token_ortho_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results, _ = train_and_benchmark(
            name, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
            device, seed=seed,
            recon_weight=1.0, masked_weight=0.1, token_ortho_weight=0.1,
        )
        all_results.append(results)
        save_results(all_results)
        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── B4: Combined Fix ─────────────────────────────────────────────


def step_b4(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B4: Combined: PCA-init + pooled recon + VICReg + token ortho")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Fit PCA
    train_vecs = np.array([
        embeddings[pid].mean(axis=0) for pid in train_ids if pid in embeddings
    ])
    pca = PCA(n_components=128, random_state=42)
    pca.fit(train_vecs)
    pca_weight = pca.components_.astype(np.float32)
    pca_bias = (-pca.mean_ @ pca.components_.T).astype(np.float32)

    for seed in SEEDS:
        name = f"combined_fix_s{seed}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  Training {name}...")
        results, model = train_and_benchmark(
            name, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
            device, seed=seed,
            init_proj_weights=(pca_weight, pca_bias),
            recon_weight=0.1, masked_weight=0.1,
            pool_recon_weight=1.0, vicreg_weight=0.1, token_ortho_weight=0.1,
        )
        all_results.append(results)
        save_results(all_results)

        ret = results.get("retrieval_family", {}).get("precision@1", 0)
        cls_acc = results.get("classification_family", {}).get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={ret:.3f}, Cls={cls_acc:.3f}")

        # Also measure token diversity
        print(f"  Computing token diversity for {name}...")
        diversity = compute_token_diversity(model, embeddings, device=device)
        print(f"  >> Pairwise cos: {diversity['mean_pairwise_cos']:.4f}")
        print(f"  >> Effective rank: {diversity['mean_effective_rank']:.2f}")

        div_result = {"name": f"token_diversity_{name}", **diversity}
        all_results.append(div_result)
        save_results(all_results)

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    print(f"\n{'='*100}")
    print("DIAGNOSTIC RESULTS SUMMARY")
    print(f"{'='*100}")

    # Token diversity
    div_results = [r for r in all_results if r["name"].startswith("token_diversity_")]
    if div_results:
        print(f"\n--- Token Diversity ---")
        print(f"{'Name':<45} {'Pairwise Cos':<15} {'Eff Rank':<12}")
        print("-" * 72)
        for r in div_results:
            print(f"{r['name']:<45} {r['mean_pairwise_cos']:.4f}         {r['mean_effective_rank']:.2f}")

    # Distance alignment
    dist_results = [r for r in all_results if r["name"] == "distance_alignment"]
    if dist_results:
        r = dist_results[0]
        print(f"\n--- Distance Alignment ---")
        print(f"  Spearman rho: {r['spearman_rho']:.4f}")
        print(f"  AUC (mean-pool): {r['auc_meanpool']:.4f}")
        print(f"  AUC (compressed): {r['auc_compressed']:.4f}")

    # Retrieval comparison
    ret_results = [r for r in all_results if "retrieval_family" in r]
    if ret_results:
        print(f"\n--- Retrieval Comparison ---")
        print(f"{'Name':<45} {'Ret@1':<10} {'Cls':<10}")
        print("-" * 65)

        # Reference baselines from Phase 5
        print(f"{'[ref] mean-pool (1280d)':<45} {'0.618':<10} {'-':<10}")
        print(f"{'[ref] PCA-128':<45} {'0.454':<10} {'-':<10}")
        print(f"{'[ref] AttnPool default':<45} {'0.384':<10} {'-':<10}")
        print("-" * 65)

        for r in ret_results:
            ret = r["retrieval_family"].get("precision@1", 0)
            cls = r.get("classification_family", {}).get("accuracy_mean", "-")
            if isinstance(cls, float):
                cls = f"{cls:.3f}"
            print(f"{r['name']:<45} {ret:<10.3f} {cls:<10}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Diagnostics & Fixes")
    parser.add_argument(
        "--step", nargs="*", default=None,
        help="Run specific step(s): A1 A2 A3 A4 B1 B2 B3 B4. Default: all.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS_PATH}")

    all_results = load_results()
    print(f"Loaded {len(all_results)} existing results")

    # Load data
    filt_emb, filt_meta, filt_seq = load_5k_data()
    train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids = get_splits(filt_meta)

    stats = split_statistics(filt_meta, train_ids, test_ids, eval_ids)
    print(f"  Retrieval split: {stats['n_train']} train / {stats['n_test']} test")
    print(f"  Classification split: {len(cls_train_ids)} train / {len(cls_test_ids)} test")

    steps = args.step or ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    steps = [s.upper() for s in steps]

    if "A1" in steps:
        all_results = step_a1(all_results, device, filt_emb)

    if "A2" in steps:
        all_results = step_a2(all_results, device, filt_emb, filt_meta, test_ids)

    if "A3" in steps:
        all_results = step_a3(
            all_results, device, filt_emb, filt_meta,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "A4" in steps:
        all_results = step_a4(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B1" in steps:
        all_results = step_b1(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B2" in steps:
        all_results = step_b2(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B3" in steps:
        all_results = step_b3(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B4" in steps:
        all_results = step_b4(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
