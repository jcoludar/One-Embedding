#!/usr/bin/env python3
"""Phase 8: Per-Residue Channel Compression.

Compresses PLM per-residue embeddings (L, D) → (L, D') via pointwise MLP,
preserving per-residue information while achieving channel compression.

Steps:
  C1: Baselines (raw mean-pool, per-residue PCA)
  C2: ChannelCompressor unsupervised (D' = 64, 128, 256)
  C3: ChannelCompressor + contrastive fine-tuning
  C4: Per-residue benchmarks (SS3 linear probe)
  C5: ProtT5 replication + ToxProt external validation

Usage:
  uv run python experiments/11_channel_compression.py                # run all
  uv run python experiments/11_channel_compression.py --step C1      # baselines only
  uv run python experiments/11_channel_compression.py --step C2      # unsupervised
  uv run python experiments/11_channel_compression.py --step C3      # contrastive
  uv run python experiments/11_channel_compression.py --step C4      # per-residue eval
  uv run python experiments/11_channel_compression.py --step C5      # ProtT5
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

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.reconstruction import evaluate_reconstruction
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
from src.training.objectives import InfoNCEFamilyLoss, ReconstructionLoss
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings, save_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "channel_compression_results.json"
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
    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    if sf_split_path.exists():
        train_ids, test_ids, eval_ids = load_split(sf_split_path)
    else:
        train_ids, test_ids, eval_ids = superfamily_aware_split(
            metadata, test_fraction=0.3, seed=42
        )
        stats = split_statistics(metadata, train_ids, test_ids, eval_ids)
        save_split(train_ids, test_ids, eval_ids, sf_split_path, stats=stats)

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

    return train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids


def train_channel_compressor(
    model, embeddings, train_ids, device,
    epochs=200, batch_size=16, lr=1e-3, max_len=512, seed=42,
    validation_embeddings=None,
):
    """Train ChannelCompressor with per-residue reconstruction loss."""
    from src.training.trainer import train_compressor

    train_emb = {k: v for k, v in embeddings.items() if k in set(train_ids)}

    val_emb = None
    if validation_embeddings is not None:
        val_emb = validation_embeddings

    history = train_compressor(
        model, train_emb,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        recon_weight=1.0,
        masked_weight=0.0,
        contrastive_weight=0.0,
        device=device,
        max_len=max_len,
        seed=seed,
        validation_embeddings=val_emb,
    )
    return history


def evaluate_channel_model(model, embeddings, metadata, eval_ids, test_ids,
                           cls_train_ids, cls_test_ids, device):
    """Evaluate a ChannelCompressor using the standard benchmark suite."""
    results = run_benchmark_suite(
        model, embeddings, metadata,
        name="temp",
        device=device,
        train_ids=cls_train_ids,  # not used for retrieval db
        test_ids=test_ids,
        eval_ids=eval_ids,
        cls_train_ids=cls_train_ids,
        cls_test_ids=cls_test_ids,
    )
    return results


# ── C1: Baselines ────────────────────────────────────────────────


def step_c1(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("C1: Baselines (raw mean-pool, per-residue PCA)")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # C1a: Raw mean-pool (1280d reference)
    name = "channel_baseline_meanpool"
    if not is_done(all_results, name):
        print(f"\n  {name}: Raw mean-pool (D={embed_dim})...")
        results = run_benchmark_suite(
            None, embeddings, metadata, name=name,
            device=device, test_ids=test_ids, eval_ids=eval_ids,
            cls_train_ids=cls_train_ids, cls_test_ids=cls_test_ids,
        )
        all_results.append(results)
        save_results(all_results)
        p1 = results.get("retrieval_family", {}).get("precision@1", 0)
        print(f"  >> {name}: Ret@1={p1:.3f}")
    else:
        print(f"  {name} already done")

    # C1b: Per-residue PCA baselines
    from sklearn.decomposition import PCA

    # Collect all residue vectors from training set for PCA fitting
    print("\n  Fitting PCA on training residue vectors...")
    train_set = set(train_ids)
    all_residues = []
    for pid in train_ids:
        if pid in embeddings:
            emb = embeddings[pid][:512]
            all_residues.append(emb)
    all_residues = np.concatenate(all_residues, axis=0)
    print(f"  PCA fitting on {len(all_residues)} residue vectors")

    for d_prime in [64, 128, 256]:
        name = f"channel_baseline_pca_d{d_prime}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  {name}: PCA → {d_prime}d...")
        pca = PCA(n_components=d_prime, random_state=42)
        pca.fit(all_residues)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained_var:.3f}")

        # Project all embeddings
        pca_embeddings = {}
        for pid, emb in embeddings.items():
            pca_embeddings[pid] = pca.transform(emb).astype(np.float32)

        # Mean-pool and evaluate (no model needed)
        mock_emb = {}
        for pid, emb in pca_embeddings.items():
            mock_emb[pid] = emb.mean(axis=0).reshape(1, -1)

        ret = evaluate_retrieval(
            None, mock_emb, metadata, label_key="family",
            query_ids=eval_ids, database_ids=test_ids,
        )
        cls = evaluate_linear_probe(
            None, mock_emb, metadata, label_key="family",
            train_ids=cls_train_ids, test_ids=cls_test_ids,
        )

        result = {
            "name": name,
            "split_mode": "held_out",
            "latent_dim": d_prime,
            "method": "pca",
            "explained_variance": float(explained_var),
            "compression_ratio": float(d_prime / embed_dim),
            "retrieval_family": ret,
            "classification_family": cls,
        }
        all_results.append(result)
        save_results(all_results)

        p1 = ret.get("precision@1", 0)
        cls_acc = cls.get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={p1:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── C2: Unsupervised ChannelCompressor ───────────────────────────


def step_c2(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("C2: ChannelCompressor unsupervised training")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Prepare validation embeddings (test set)
    test_set = set(test_ids)
    val_emb = {k: v for k, v in embeddings.items() if k in test_set}

    for d_prime in [64, 128, 256]:
        for seed in SEEDS:
            name = f"channel_unsup_d{d_prime}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = ChannelCompressor(
                input_dim=embed_dim,
                latent_dim=d_prime,
                dropout=0.1,
                use_residual=True,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  D={embed_dim} → {embed_dim//2} → {d_prime}, params={n_params:,}")

            start = time.time()
            history = train_channel_compressor(
                model, embeddings, train_ids, device,
                epochs=200, batch_size=16, lr=1e-3, seed=seed,
                validation_embeddings=val_emb,
            )
            elapsed = time.time() - start
            print(f"  Training done in {elapsed:.0f}s (best epoch: {history['best_epoch']})")

            # Load best checkpoint if available
            # (train_compressor doesn't return checkpoint path, but saves internally)
            # Re-evaluate with trained model
            model = model.to(device)

            # Evaluate
            eval_results = evaluate_channel_model(
                model, embeddings, metadata,
                eval_ids, test_ids, cls_train_ids, cls_test_ids, device,
            )

            # Also evaluate per-residue reconstruction on test set
            recon = evaluate_reconstruction(model, val_emb, device)

            # Save checkpoint
            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": d_prime,
                "method": "channel_unsup",
                "use_residual": True,
                "n_params": n_params,
                "seed": seed,
                "best_epoch": history["best_epoch"],
                "reconstruction": recon,
                "retrieval_family": eval_results.get("retrieval_family", {}),
                "retrieval_superfamily": eval_results.get("retrieval_superfamily", {}),
                "classification_family": eval_results.get("classification_family", {}),
                "compression": eval_results.get("compression", {}),
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
            mrr = eval_results.get("retrieval_family", {}).get("mrr", 0)
            cls_acc = eval_results.get("classification_family", {}).get("accuracy_mean", 0)
            cos = recon.get("cosine_sim", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}, CosSim={cos:.3f}")

    return all_results


# ── C3: Contrastive Fine-Tuning ──────────────────────────────────


def step_c3(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("C3: ChannelCompressor + contrastive fine-tuning")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Build family label mapping for training proteins
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    train_set = set(train_ids)
    train_pids_with_fam = [pid for pid in train_ids if pid in id_to_fam and pid in embeddings]
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids_with_fam))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

    for d_prime in [128, 256]:
        for seed in SEEDS:
            name = f"channel_contrastive_d{d_prime}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            # Load pre-trained unsupervised model
            pretrained_name = f"channel_unsup_d{d_prime}_s{seed}"
            pretrained_ckpt = CHECKPOINTS_DIR / pretrained_name / "best_model.pt"
            if not pretrained_ckpt.exists():
                print(f"  {name}: pre-trained checkpoint {pretrained_name} not found, skipping")
                continue

            print(f"\n  Fine-tuning {name} from {pretrained_name}...")
            monitor()

            model = ChannelCompressor(
                input_dim=embed_dim,
                latent_dim=d_prime,
                dropout=0.1,
                use_residual=True,
            )
            model.load_state_dict(
                torch.load(pretrained_ckpt, map_location=device, weights_only=True)
            )
            model = model.to(device)

            # Record pre-finetune reconstruction quality for monitoring
            test_set = set(test_ids)
            val_emb = {k: v for k, v in embeddings.items() if k in test_set}
            pre_recon = evaluate_reconstruction(model, val_emb, device)
            pre_recon_mse = pre_recon["mse"]
            print(f"  Pre-finetune recon: MSE={pre_recon_mse:.6f}, CosSim={pre_recon['cosine_sim']:.3f}")

            # Freeze decoder, fine-tune encoder only
            for p in model.dec_linear1.parameters():
                p.requires_grad = False
            for p in model.dec_norm1.parameters():
                p.requires_grad = False
            model.dec_proj.requires_grad_(False)
            model.dec_dropout.requires_grad_(False)
            if model._use_residual:
                for p in model.dec_res_proj.parameters():
                    p.requires_grad = False

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            infonce_fn = InfoNCEFamilyLoss(temperature=0.07)
            recon_loss_fn = ReconstructionLoss().to(device)

            # Prepare batched data
            torch.manual_seed(seed)
            np.random.seed(seed)

            max_len = 512
            batch_size = 128
            recon_reg_weight = 0.1

            start = time.time()

            for epoch in range(1, 101):
                model.train()

                # Build random batches of proteins
                perm = np.random.permutation(len(train_pids_with_fam))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i + batch_size]
                    batch_pids = [train_pids_with_fam[j] for j in idx]

                    # Prepare padded batch
                    batch_embs = []
                    batch_masks = []
                    batch_labels = []
                    for pid in batch_pids:
                        emb = embeddings[pid]
                        L = min(emb.shape[0], max_len)
                        padded = np.zeros((max_len, embed_dim), dtype=np.float32)
                        padded[:L] = emb[:L]
                        mask = np.zeros(max_len, dtype=np.float32)
                        mask[:L] = 1.0
                        batch_embs.append(padded)
                        batch_masks.append(mask)
                        batch_labels.append(fam_to_idx[id_to_fam[pid]])

                    states = torch.from_numpy(np.array(batch_embs)).to(device)
                    masks = torch.from_numpy(np.array(batch_masks)).to(device)
                    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

                    optimizer.zero_grad()

                    # Forward: compress and pool
                    output = model(states, masks)
                    latent = output["latent"]
                    pooled = model.get_pooled(latent, strategy="mean", mask=masks)

                    # InfoNCE on pooled vectors
                    cl = infonce_fn(pooled, labels)

                    # Reconstruction regularization (monitor for drift)
                    recon_result = recon_loss_fn(output["reconstructed"], states, masks)

                    loss = cl["loss"] + recon_reg_weight * recon_result["loss"]
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                if epoch % 25 == 0 or epoch == 1:
                    # Check reconstruction drift
                    model.eval()
                    with torch.no_grad():
                        curr_recon = evaluate_reconstruction(model, val_emb, device)
                    curr_mse = curr_recon["mse"]
                    drift = (curr_mse - pre_recon_mse) / pre_recon_mse if pre_recon_mse > 0 else 0
                    elapsed = time.time() - start

                    drift_warning = ""
                    if drift > 0.10:
                        drift_warning = " ⚠️ DRIFT>10%"
                        recon_reg_weight = min(recon_reg_weight * 1.5, 1.0)
                        print(f"    Increasing recon_reg_weight to {recon_reg_weight:.2f}")

                    print(f"    Epoch {epoch:3d}/100 | Loss={epoch_loss/max(n_batches,1):.4f} | "
                          f"ReconDrift={drift:+.1%} | CosSim={curr_recon['cosine_sim']:.3f}{drift_warning} | {elapsed:.0f}s")
                    model.train()

            elapsed = time.time() - start
            print(f"  Contrastive fine-tuning done in {elapsed:.0f}s")

            # Unfreeze for evaluation
            for p in model.parameters():
                p.requires_grad = True

            # Evaluate
            eval_results = evaluate_channel_model(
                model, embeddings, metadata,
                eval_ids, test_ids, cls_train_ids, cls_test_ids, device,
            )

            post_recon = evaluate_reconstruction(model, val_emb, device)

            # Save checkpoint
            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": d_prime,
                "method": "channel_contrastive",
                "use_residual": True,
                "seed": seed,
                "pre_finetune_recon": pre_recon,
                "post_finetune_recon": post_recon,
                "retrieval_family": eval_results.get("retrieval_family", {}),
                "retrieval_superfamily": eval_results.get("retrieval_superfamily", {}),
                "classification_family": eval_results.get("classification_family", {}),
                "compression": eval_results.get("compression", {}),
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
            mrr = eval_results.get("retrieval_family", {}).get("mrr", 0)
            cls_acc = eval_results.get("classification_family", {}).get("accuracy_mean", 0)
            cos_post = post_recon.get("cosine_sim", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}, CosSim_post={cos_post:.3f}")

    return all_results


# ── C4: Per-Residue Benchmarks ───────────────────────────────────


def step_c4(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids):
    print(f"\n{'='*60}")
    print("C4: Per-residue benchmarks (SS3 linear probe)")
    print(f"{'='*60}")

    from src.evaluation.per_residue_tasks import run_per_residue_benchmarks

    embed_dim = next(iter(embeddings.values())).shape[-1]
    benchmark_data_dir = DATA_DIR / "per_residue_benchmarks"

    # Check if benchmark data exists
    if not benchmark_data_dir.exists():
        print(f"  Per-residue benchmark data not found at {benchmark_data_dir}")
        print("  To run C4, prepare the following directories:")
        print(f"    {benchmark_data_dir}/cb513/  (cb513_sequences.fasta + cb513_ss.txt)")
        print(f"    {benchmark_data_dir}/chezod/ (chezod_sequences.fasta + chezod_scores.txt)")
        print(f"    {benchmark_data_dir}/tmbed/  (tmbed_sequences.fasta + tmbed_topology.txt)")
        print("  Skipping C4.")
        return all_results

    # C4a: Evaluate on original (full-dim) embeddings as reference
    name = "channel_perresidue_original"
    if not is_done(all_results, name):
        print(f"\n  {name}: Original {embed_dim}d embeddings...")
        pr_results = run_per_residue_benchmarks(
            embeddings, benchmark_data_dir, name=name,
        )
        result = {"name": name, "method": "original", "embed_dim": embed_dim, **pr_results}
        all_results.append(result)
        save_results(all_results)

    # C4b: Evaluate compressed embeddings from C2 and C3
    for d_prime in [64, 128, 256]:
        for method in ["channel_unsup", "channel_contrastive"]:
            for seed in SEEDS:
                src_name = f"{method}_d{d_prime}_s{seed}"
                name = f"channel_perresidue_{method}_d{d_prime}_s{seed}"

                if is_done(all_results, name):
                    print(f"  {name} already done")
                    continue

                ckpt_path = CHECKPOINTS_DIR / src_name / "best_model.pt"
                if not ckpt_path.exists():
                    continue

                print(f"\n  {name}: Compressed D'={d_prime}...")

                model = ChannelCompressor(
                    input_dim=embed_dim,
                    latent_dim=d_prime,
                    dropout=0.1,
                    use_residual=True,
                )
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True)
                )
                model = model.to(device)
                model.eval()

                # Compress all embeddings
                compressed_emb = {}
                with torch.no_grad():
                    for pid, emb in embeddings.items():
                        L = min(emb.shape[0], 512)
                        states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
                        mask = torch.ones(1, L, device=device)
                        latent = model.compress(states, mask)
                        compressed_emb[pid] = latent[0, :L].cpu().numpy()

                pr_results = run_per_residue_benchmarks(
                    compressed_emb, benchmark_data_dir, name=name,
                )

                result = {
                    "name": name,
                    "method": method,
                    "latent_dim": d_prime,
                    "seed": seed,
                    **pr_results,
                }
                all_results.append(result)
                save_results(all_results)

    return all_results


# ── C5: ProtT5 Replication ───────────────────────────────────────


def step_c5(all_results, device, embeddings_esm, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("C5: ProtT5 replication (model-agnostic test)")
    print(f"{'='*60}")

    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"

    # Step 1: Extract ProtT5 embeddings if not cached
    if h5_path.exists():
        print("  Loading cached ProtT5 embeddings...")
        prot_t5_emb = load_residue_embeddings(h5_path)
    else:
        print("  Extracting ProtT5-XL embeddings (this will take 1-2 hours)...")
        monitor()
        try:
            from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
            prot_t5_emb = extract_prot_t5_embeddings(sequences, batch_size=4, device=device)
            save_residue_embeddings(prot_t5_emb, h5_path)
        except Exception as e:
            print(f"  ProtT5 extraction failed: {e}")
            print("  Skipping C5.")
            return all_results

    t5_dim = next(iter(prot_t5_emb.values())).shape[-1]
    print(f"  ProtT5 dim={t5_dim}, {len(prot_t5_emb)} proteins")

    # Filter to matching proteins
    common_ids = set(prot_t5_emb.keys()) & set(m["id"] for m in metadata)
    prot_t5_emb = {k: v for k, v in prot_t5_emb.items() if k in common_ids}

    # Step 2: Train ChannelCompressor on ProtT5
    d_prime = 256
    seed = 42
    name = f"channel_prot_t5_unsup_d{d_prime}_s{seed}"

    if not is_done(all_results, name):
        print(f"\n  Training {name} (ProtT5 {t5_dim} → {d_prime})...")
        monitor()

        model = ChannelCompressor(
            input_dim=t5_dim,
            latent_dim=d_prime,
            dropout=0.1,
            use_residual=True,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        test_set = set(test_ids)
        val_emb = {k: v for k, v in prot_t5_emb.items() if k in test_set}

        start = time.time()
        history = train_channel_compressor(
            model, prot_t5_emb, train_ids, device,
            epochs=200, batch_size=16, lr=1e-3, seed=seed,
            validation_embeddings=val_emb,
        )
        elapsed = time.time() - start
        print(f"  Training done in {elapsed:.0f}s")

        model = model.to(device)
        eval_results = evaluate_channel_model(
            model, prot_t5_emb, metadata,
            eval_ids, test_ids, cls_train_ids, cls_test_ids, device,
        )
        recon = evaluate_reconstruction(model, val_emb, device)

        ckpt_dir = CHECKPOINTS_DIR / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        result = {
            "name": name,
            "split_mode": "held_out",
            "plm": "ProtT5-XL",
            "input_dim": t5_dim,
            "latent_dim": d_prime,
            "method": "channel_unsup",
            "n_params": n_params,
            "seed": seed,
            "reconstruction": recon,
            "retrieval_family": eval_results.get("retrieval_family", {}),
            "classification_family": eval_results.get("classification_family", {}),
            "compression": eval_results.get("compression", {}),
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
        cos = recon.get("cosine_sim", 0)
        print(f"  >> {name}: Ret@1={p1:.3f}, CosSim={cos:.3f}")

    # Step 3: Contrastive fine-tuning on ProtT5
    contrastive_name = f"channel_prot_t5_contrastive_d{d_prime}_s{seed}"

    if not is_done(all_results, contrastive_name):
        pretrained_ckpt = CHECKPOINTS_DIR / name / "best_model.pt"
        if pretrained_ckpt.exists():
            print(f"\n  Fine-tuning {contrastive_name}...")

            model = ChannelCompressor(
                input_dim=t5_dim,
                latent_dim=d_prime,
                dropout=0.1,
                use_residual=True,
            )
            model.load_state_dict(
                torch.load(pretrained_ckpt, map_location=device, weights_only=True)
            )
            model = model.to(device)

            # Build labels
            id_to_fam = {m["id"]: m["family"] for m in metadata}
            train_pids = [pid for pid in train_ids if pid in id_to_fam and pid in prot_t5_emb]
            unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids))
            fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

            # Freeze decoder
            for p in model.dec_linear1.parameters():
                p.requires_grad = False
            for p in model.dec_norm1.parameters():
                p.requires_grad = False
            model.dec_proj.requires_grad_(False)
            if model._use_residual:
                for p in model.dec_res_proj.parameters():
                    p.requires_grad = False

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            infonce_fn = InfoNCEFamilyLoss(temperature=0.07)
            recon_loss_fn = ReconstructionLoss().to(device)

            torch.manual_seed(seed)
            np.random.seed(seed)
            max_len = 512
            batch_size = 128

            start = time.time()
            for epoch in range(1, 101):
                model.train()
                perm = np.random.permutation(len(train_pids))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i + batch_size]
                    batch_pids = [train_pids[j] for j in idx]

                    batch_embs = []
                    batch_masks = []
                    batch_labels = []
                    for pid in batch_pids:
                        emb = prot_t5_emb[pid]
                        L = min(emb.shape[0], max_len)
                        padded = np.zeros((max_len, t5_dim), dtype=np.float32)
                        padded[:L] = emb[:L]
                        mask = np.zeros(max_len, dtype=np.float32)
                        mask[:L] = 1.0
                        batch_embs.append(padded)
                        batch_masks.append(mask)
                        batch_labels.append(fam_to_idx[id_to_fam[pid]])

                    states = torch.from_numpy(np.array(batch_embs)).to(device)
                    masks = torch.from_numpy(np.array(batch_masks)).to(device)
                    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

                    optimizer.zero_grad()
                    output = model(states, masks)
                    latent = output["latent"]
                    pooled = model.get_pooled(latent, strategy="mean", mask=masks)
                    cl = infonce_fn(pooled, labels)
                    recon_result = recon_loss_fn(output["reconstructed"], states, masks)
                    loss = cl["loss"] + 0.1 * recon_result["loss"]
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                if epoch % 25 == 0 or epoch == 1:
                    elapsed = time.time() - start
                    print(f"    Epoch {epoch:3d}/100 | Loss={epoch_loss/max(n_batches,1):.4f} | {elapsed:.0f}s")

            elapsed = time.time() - start

            for p in model.parameters():
                p.requires_grad = True

            eval_results = evaluate_channel_model(
                model, prot_t5_emb, metadata,
                eval_ids, test_ids, cls_train_ids, cls_test_ids, device,
            )
            post_recon = evaluate_reconstruction(model, val_emb if 'val_emb' in dir() else {}, device)

            ckpt_dir = CHECKPOINTS_DIR / contrastive_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": contrastive_name,
                "split_mode": "held_out",
                "plm": "ProtT5-XL",
                "input_dim": t5_dim,
                "latent_dim": d_prime,
                "method": "channel_contrastive",
                "seed": seed,
                "retrieval_family": eval_results.get("retrieval_family", {}),
                "classification_family": eval_results.get("classification_family", {}),
                "compression": eval_results.get("compression", {}),
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
            cls_acc = eval_results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {contrastive_name}: Ret@1={p1:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("PHASE 8 SUMMARY: Per-Residue Channel Compression")
    print(f"{'='*80}")
    print(f"{'Name':<45} {'Ret@1':>6} {'MRR':>6} {'Cls':>6} {'CosSim':>6} {'Ratio':>6}")
    print(f"{'-'*80}")

    for r in all_results:
        name = r.get("name", "?")
        ret = r.get("retrieval_family", {})
        cls = r.get("classification_family", {})
        recon = r.get("reconstruction", r.get("post_finetune_recon", {}))
        comp = r.get("compression", {})

        p1 = ret.get("precision@1", float("nan"))
        mrr = ret.get("mrr", float("nan"))
        cls_acc = cls.get("accuracy_mean", float("nan"))
        cos = recon.get("cosine_sim", float("nan"))
        ratio = comp.get("compression_ratio", r.get("compression_ratio", float("nan")))

        print(f"{name:<45} {p1:>6.3f} {mrr:>6.3f} {cls_acc:>6.3f} {cos:>6.3f} {ratio:>6.3f}")

    # Per-residue results
    pr_results = [r for r in all_results if "ss3" in r or "disorder" in r or "tm_topology" in r]
    if pr_results:
        print(f"\n{'='*80}")
        print("PER-RESIDUE BENCHMARKS")
        print(f"{'='*80}")
        print(f"{'Name':<45} {'SS3 Q3':>7} {'Disorder':>8} {'TM Acc':>7}")
        print(f"{'-'*80}")
        for r in pr_results:
            name = r.get("name", "?")
            ss3_q3 = r.get("ss3", {}).get("q3", float("nan"))
            disorder = r.get("disorder", {}).get("spearman_rho", float("nan"))
            tm_acc = r.get("tm_topology", {}).get("accuracy", float("nan"))
            print(f"{name:<45} {ss3_q3:>7.3f} {disorder:>8.3f} {tm_acc:>7.3f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Per-Residue Channel Compression")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step: C1, C2, C3, C4, C5")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("\nLoading data...")
    embeddings, metadata, sequences = load_5k_data()
    train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids = get_splits(metadata)
    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}, Eval: {len(eval_ids)}")
    print(f"  Cls train: {len(cls_train_ids)}, Cls test: {len(cls_test_ids)}")

    all_results = load_results()

    steps = {
        "C1": lambda: step_c1(all_results, device, embeddings, metadata,
                              train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids),
        "C2": lambda: step_c2(all_results, device, embeddings, metadata, sequences,
                              train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids),
        "C3": lambda: step_c3(all_results, device, embeddings, metadata,
                              train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids),
        "C4": lambda: step_c4(all_results, device, embeddings, metadata,
                              train_ids, test_ids, eval_ids),
        "C5": lambda: step_c5(all_results, device, embeddings, metadata, sequences,
                              train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids),
    }

    if args.step:
        step_key = args.step.upper()
        if step_key in steps:
            all_results = steps[step_key]()
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
            return
    else:
        for step_key in ["C1", "C2", "C3", "C4", "C5"]:
            all_results = steps[step_key]()

    print_summary(all_results)


if __name__ == "__main__":
    main()
