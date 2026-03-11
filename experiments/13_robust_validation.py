#!/usr/bin/env python3
"""Phase 9B-Validation: Robust multi-seed + multi-dataset validation.

Steps:
  R1: Multi-seed ProtT5 ChannelCompressor training (seeds 123, 456)
  R2: Embedding extraction for CheZOD, TMbed, TS115 (ESM2-650M + ProtT5-XL)
  R3: All probes — CheZOD disorder, TMbed topology, TS115 SS3/SS8,
      plus multi-seed ProtT5 CB513 probes

Usage:
  uv run python experiments/13_robust_validation.py --step R1
  uv run python experiments/13_robust_validation.py --step R2
  uv run python experiments/13_robust_validation.py --step R3
  uv run python experiments/13_robust_validation.py          # run all
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    evaluate_disorder_probe,
    evaluate_tm_probe,
    load_cb513_csv,
    load_chezod_seth,
    load_tmbed_annotated,
)
from src.evaluation.benchmark_suite import run_benchmark_suite
from src.evaluation.reconstruction import evaluate_reconstruction
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.splitting import load_split
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv, read_fasta
from src.training.objectives import InfoNCEFamilyLoss, ReconstructionLoss
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings, save_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "robust_validation_results.json"
SPLIT_DIR = DATA_DIR / "splits"

CB513_CSV = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
TS115_CSV = DATA_DIR / "per_residue_benchmarks" / "TS115.csv"
TMBED_FASTA = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
CHEZOD_DIR = DATA_DIR / "per_residue_benchmarks"

PROBE_SEEDS = [42, 123, 456]


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
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {len(results)} results to {RESULTS_PATH}")


def is_done(results: list[dict], name: str) -> bool:
    return any(r["name"] == name for r in results)


def compress_embeddings(model, embeddings, device, max_len=512):
    model.eval()
    compressed = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)
            compressed[pid] = latent[0, :L].cpu().numpy()
    return compressed


def load_checkpoint(ckpt_name, input_dim, device):
    ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best_model.pt"
    if not ckpt_path.exists():
        return None
    # Infer latent_dim from checkpoint name
    for token in ckpt_name.split("_"):
        if token.startswith("d") and token[1:].isdigit():
            latent_dim = int(token[1:])
            break
    model = ChannelCompressor(
        input_dim=input_dim, latent_dim=latent_dim, dropout=0.1, use_residual=True,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model.to(device)


# ── R1: Multi-Seed ProtT5 Training ──────────────────────────────


def step_r1(device):
    print(f"\n{'='*60}")
    print("R1: Multi-Seed ProtT5 ChannelCompressor Training")
    print(f"{'='*60}")

    all_results = load_results()

    # Load ProtT5 5K embeddings and metadata
    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if not h5_path.exists():
        print("  ProtT5 5K embeddings not found. Run experiment 11 C5 first.")
        return

    prot_t5_emb = load_residue_embeddings(h5_path)
    t5_dim = next(iter(prot_t5_emb.values())).shape[-1]

    # Load metadata and splits
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    prot_t5_emb = {k: v for k, v in prot_t5_emb.items() if k in kept_ids}

    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    train_ids, test_ids, eval_ids = load_split(sf_split_path)

    cls_split_path = SPLIT_DIR / "esm2_650m_5k_cls_split.json"
    with open(cls_split_path) as f:
        cls_data = json.load(f)
    cls_train_ids = cls_data["cls_train_ids"]
    cls_test_ids = cls_data["cls_test_ids"]

    id_to_fam = {m["id"]: m["family"] for m in filtered_meta}
    train_pids = [pid for pid in train_ids if pid in id_to_fam and pid in prot_t5_emb]
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

    test_set = set(test_ids)
    val_emb = {k: v for k, v in prot_t5_emb.items() if k in test_set}

    d_prime = 256
    new_seeds = [123, 456]

    for seed in new_seeds:
        # Phase 1: Unsupervised
        unsup_name = f"channel_prot_t5_unsup_d{d_prime}_s{seed}"
        ckpt_dir = CHECKPOINTS_DIR / unsup_name

        if not (ckpt_dir / "best_model.pt").exists():
            print(f"\n  Training {unsup_name}...")
            monitor()

            model = ChannelCompressor(
                input_dim=t5_dim, latent_dim=d_prime, dropout=0.1, use_residual=True,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {n_params:,}")

            from src.training.trainer import train_compressor
            train_emb = {k: v for k, v in prot_t5_emb.items() if k in set(train_ids)}

            start = time.time()
            history = train_compressor(
                model, train_emb, epochs=200, batch_size=16, lr=1e-3,
                recon_weight=1.0, masked_weight=0.0, contrastive_weight=0.0,
                device=device, max_len=512, seed=seed,
                validation_embeddings=val_emb,
            )
            elapsed = time.time() - start
            print(f"  Unsupervised done in {elapsed:.0f}s (best epoch: {history['best_epoch']})")

            model = model.to(device)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            # Evaluate unsupervised
            recon = evaluate_reconstruction(model, val_emb, device)
            eval_results = run_benchmark_suite(
                model, prot_t5_emb, filtered_meta, name=unsup_name,
                device=device, test_ids=test_ids, eval_ids=eval_ids,
                cls_train_ids=cls_train_ids, cls_test_ids=cls_test_ids,
            )

            result = {
                "name": unsup_name, "plm": "ProtT5-XL", "method": "channel_unsup",
                "latent_dim": d_prime, "seed": seed,
                "reconstruction": recon,
                "retrieval_family": eval_results.get("retrieval_family", {}),
                "classification_family": eval_results.get("classification_family", {}),
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
            cos = recon.get("cosine_sim", 0)
            print(f"  >> {unsup_name}: Ret@1={p1:.3f}, CosSim={cos:.3f}")
        else:
            print(f"  {unsup_name} checkpoint exists, skipping training")

        # Phase 2: Contrastive fine-tuning
        contrastive_name = f"channel_prot_t5_contrastive_d{d_prime}_s{seed}"
        contrastive_ckpt_dir = CHECKPOINTS_DIR / contrastive_name

        if not (contrastive_ckpt_dir / "best_model.pt").exists():
            pretrained_ckpt = ckpt_dir / "best_model.pt"
            if not pretrained_ckpt.exists():
                print(f"  {contrastive_name}: pre-trained not found, skipping")
                continue

            print(f"\n  Fine-tuning {contrastive_name}...")
            monitor()

            model = ChannelCompressor(
                input_dim=t5_dim, latent_dim=d_prime, dropout=0.1, use_residual=True,
            )
            model.load_state_dict(torch.load(pretrained_ckpt, map_location=device, weights_only=True))
            model = model.to(device)

            pre_recon = evaluate_reconstruction(model, val_emb, device)
            print(f"  Pre-finetune CosSim={pre_recon['cosine_sim']:.3f}")

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
                [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4,
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

                    batch_embs, batch_masks, batch_labels = [], [], []
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

            # Evaluate
            eval_results = run_benchmark_suite(
                model, prot_t5_emb, filtered_meta, name=contrastive_name,
                device=device, test_ids=test_ids, eval_ids=eval_ids,
                cls_train_ids=cls_train_ids, cls_test_ids=cls_test_ids,
            )
            post_recon = evaluate_reconstruction(model, val_emb, device)

            contrastive_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), contrastive_ckpt_dir / "best_model.pt")

            result = {
                "name": contrastive_name, "plm": "ProtT5-XL", "method": "channel_contrastive",
                "latent_dim": d_prime, "seed": seed,
                "pre_finetune_recon": pre_recon,
                "post_finetune_recon": post_recon,
                "retrieval_family": eval_results.get("retrieval_family", {}),
                "classification_family": eval_results.get("classification_family", {}),
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = eval_results.get("retrieval_family", {}).get("precision@1", 0)
            cls_acc = eval_results.get("classification_family", {}).get("accuracy_mean", 0)
            print(f"  >> {contrastive_name}: Ret@1={p1:.3f}, Cls={cls_acc:.3f}")
        else:
            print(f"  {contrastive_name} checkpoint exists, skipping training")

    print("\n  R1 complete.")


# ── R2: Embedding Extraction for New Datasets ────────────────────


def step_r2(device):
    print(f"\n{'='*60}")
    print("R2: Embedding Extraction for CheZOD, TMbed, TS115")
    print(f"{'='*60}")

    # Collect all sequences that need embedding
    datasets = {}

    # CheZOD
    chezod_seqs, _, chezod_train, chezod_test = load_chezod_seth(CHEZOD_DIR)
    if chezod_seqs:
        datasets["chezod"] = chezod_seqs
        print(f"  CheZOD: {len(chezod_seqs)} proteins")

    # TMbed
    tmbed_seqs, _ = load_tmbed_annotated(TMBED_FASTA)
    if tmbed_seqs:
        datasets["tmbed"] = tmbed_seqs
        print(f"  TMbed: {len(tmbed_seqs)} proteins")

    # TS115
    ts115_seqs, _, _, _ = load_cb513_csv(TS115_CSV)
    if ts115_seqs:
        datasets["ts115"] = ts115_seqs
        print(f"  TS115: {len(ts115_seqs)} proteins")

    # Merge all sequences for a single extraction pass per PLM
    # Cap at 2000 residues to avoid OOM/slowdown on very long proteins
    MAX_SEQ_LEN = 2000
    all_seqs = {}
    n_skipped = 0
    for ds_name, seqs in datasets.items():
        for pid, seq in seqs.items():
            if len(seq) > MAX_SEQ_LEN:
                n_skipped += 1
                continue
            # Prefix with dataset to avoid ID collisions
            key = f"{ds_name}_{pid}"
            all_seqs[key] = seq

    print(f"  Total unique sequences to extract: {len(all_seqs)} (skipped {n_skipped} > {MAX_SEQ_LEN} residues)")

    # ESM2-650M extraction
    esm2_h5 = DATA_DIR / "residue_embeddings" / "esm2_650m_validation.h5"
    if esm2_h5.exists():
        print(f"\n  ESM2-650M validation embeddings already exist at {esm2_h5}")
    else:
        print(f"\n  Extracting ESM2-650M embeddings for {len(all_seqs)} proteins...")
        monitor()
        from src.extraction.esm_extractor import extract_residue_embeddings
        esm2_emb = extract_residue_embeddings(
            all_seqs, model_name="esm2_t33_650M_UR50D", batch_size=4, device=device
        )
        save_residue_embeddings(esm2_emb, esm2_h5)
        del esm2_emb
        if device.type == "mps":
            torch.mps.empty_cache()

    # ProtT5-XL extraction
    t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"
    if t5_h5.exists():
        print(f"\n  ProtT5-XL validation embeddings already exist at {t5_h5}")
    else:
        print(f"\n  Extracting ProtT5-XL embeddings for {len(all_seqs)} proteins...")
        monitor()
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
        t5_emb = extract_prot_t5_embeddings(all_seqs, batch_size=4, device=device)
        save_residue_embeddings(t5_emb, t5_h5)
        del t5_emb
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n  R2 complete.")


# ── R3: All Probes ───────────────────────────────────────────────


def _get_dataset_embeddings(all_emb, prefix):
    """Extract embeddings for a specific dataset by prefix."""
    p = f"{prefix}_"
    return {k[len(p):]: v for k, v in all_emb.items() if k.startswith(p)}


def _run_ss_probes(embeddings, ss3_labels, ss8_labels, splits):
    """Run SS3/SS8 probes across splits, return averaged metrics."""
    q3_all, q8_all = [], []
    for seed, (train_ids, test_ids) in splits.items():
        r3 = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids)
        q3_all.append(r3["q3"])
        r8 = evaluate_ss8_probe(embeddings, ss8_labels, train_ids, test_ids)
        q8_all.append(r8["q8"])
    return {
        "q3_mean": float(np.mean(q3_all)), "q3_std": float(np.std(q3_all)),
        "q8_mean": float(np.mean(q8_all)), "q8_std": float(np.std(q8_all)),
    }


def _make_splits(protein_ids, seeds=PROBE_SEEDS, train_frac=0.8):
    splits = {}
    for seed in seeds:
        rng = random.Random(seed)
        ids = list(protein_ids)
        rng.shuffle(ids)
        n = int(len(ids) * train_frac)
        splits[seed] = (ids[:n], ids[n:])
    return splits


def step_r3(device):
    print(f"\n{'='*60}")
    print("R3: All Per-Residue Probes")
    print(f"{'='*60}")

    all_results = load_results()

    # Load validation embeddings
    esm2_h5 = DATA_DIR / "residue_embeddings" / "esm2_650m_validation.h5"
    t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"

    if not esm2_h5.exists() or not t5_h5.exists():
        print("  Validation embeddings not found. Run R2 first.")
        return

    esm2_all = load_residue_embeddings(esm2_h5)
    t5_all = load_residue_embeddings(t5_h5)

    # Load dataset labels
    chezod_seqs, chezod_scores, chezod_train_ids, chezod_test_ids = load_chezod_seth(CHEZOD_DIR)
    tmbed_seqs, tmbed_topo = load_tmbed_annotated(TMBED_FASTA)
    ts115_seqs, ts115_ss3, ts115_ss8, ts115_dis = load_cb513_csv(TS115_CSV)

    # Also load CB513 data for multi-seed ProtT5 probes
    cb513_seqs, cb513_ss3, cb513_ss8, cb513_dis = load_cb513_csv(CB513_CSV)
    cb513_splits_path = SPLIT_DIR / "cb513_probe_splits.json"
    with open(cb513_splits_path) as f:
        cb513_splits_raw = json.load(f)
    cb513_splits = {int(k): (tr, te) for k, (tr, te) in cb513_splits_raw.items()}

    # Load CB513 embeddings for multi-seed ProtT5 eval
    cb513_t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    cb513_t5_emb = load_residue_embeddings(cb513_t5_h5) if cb513_t5_h5.exists() else {}

    # ── 3a: Multi-seed ProtT5 CB513 probes ──────────────────────
    print(f"\n  --- Multi-Seed ProtT5 CB513 Probes ---")

    for seed in [42, 123, 456]:
        for method in ["unsup", "contrastive"]:
            ckpt_name = f"channel_prot_t5_{method}_d256_s{seed}"
            result_name = f"r3_prott5_cb513_{method}_d256_s{seed}"

            if is_done(all_results, result_name):
                print(f"  {result_name} already done")
                continue

            ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  {result_name}: checkpoint not found, skipping")
                continue

            print(f"  {result_name}...")
            model = load_checkpoint(ckpt_name, 1024, device)
            compressed = compress_embeddings(model, cb513_t5_emb, device)

            q3_all, q8_all = [], []
            for s, (tr, te) in cb513_splits.items():
                r3 = evaluate_ss3_probe(compressed, cb513_ss3, tr, te)
                q3_all.append(r3["q3"])
                r8 = evaluate_ss8_probe(compressed, cb513_ss8, tr, te)
                q8_all.append(r8["q8"])

            result = {
                "name": result_name, "plm": "ProtT5-XL", "dataset": "CB513",
                "method": method, "dim": 256, "model_seed": seed,
                "q3_mean": float(np.mean(q3_all)), "q3_std": float(np.std(q3_all)),
                "q8_mean": float(np.mean(q8_all)), "q8_std": float(np.std(q8_all)),
            }
            all_results.append(result)
            save_results(all_results)
            print(f"    Q3={result['q3_mean']:.3f}+-{result['q3_std']:.3f}, "
                  f"Q8={result['q8_mean']:.3f}+-{result['q8_std']:.3f}")
            del model, compressed

    # ── 3b: CheZOD Disorder ─────────────────────────────────────
    print(f"\n  --- CheZOD Disorder Probes ---")

    for plm_name, all_emb, embed_dim in [
        ("ESM2-650M", esm2_all, 1280),
        ("ProtT5-XL", t5_all, 1024),
    ]:
        chezod_emb = _get_dataset_embeddings(all_emb, "chezod")
        if not chezod_emb:
            print(f"  No CheZOD embeddings for {plm_name}")
            continue

        # Original full-dim
        name = f"r3_chezod_{plm_name}_original"
        if not is_done(all_results, name):
            print(f"  {name}...")
            dis_result = evaluate_disorder_probe(
                chezod_emb, chezod_scores, chezod_train_ids, chezod_test_ids
            )
            result = {"name": name, "plm": plm_name, "dataset": "CheZOD",
                       "method": "original", "dim": embed_dim, **dis_result}
            all_results.append(result)
            save_results(all_results)
            print(f"    Spearman rho={dis_result['spearman_rho']:.3f}")

        # PCA-256
        from sklearn.decomposition import PCA
        name = f"r3_chezod_{plm_name}_pca_d256"
        if not is_done(all_results, name):
            print(f"  {name}...")
            train_residues = []
            for pid in chezod_train_ids:
                if pid in chezod_emb:
                    train_residues.append(chezod_emb[pid][:512])
            train_residues = np.concatenate(train_residues, axis=0)
            pca = PCA(n_components=256, random_state=42)
            pca.fit(train_residues)
            pca_emb = {pid: pca.transform(emb).astype(np.float32)
                       for pid, emb in chezod_emb.items()}
            dis_result = evaluate_disorder_probe(
                pca_emb, chezod_scores, chezod_train_ids, chezod_test_ids
            )
            result = {"name": name, "plm": plm_name, "dataset": "CheZOD",
                       "method": "pca", "dim": 256, **dis_result}
            all_results.append(result)
            save_results(all_results)
            print(f"    Spearman rho={dis_result['spearman_rho']:.3f}")

        # ChannelCompressor checkpoints
        ckpt_prefix = "channel" if plm_name == "ESM2-650M" else "channel_prot_t5"
        for method in ["unsup", "contrastive"]:
            ckpt_name = f"{ckpt_prefix}_{method}_d256_s42"
            if plm_name == "ESM2-650M":
                # Use s123 which had best retrieval
                ckpt_name = f"channel_{method}_d256_s123"
            result_name = f"r3_chezod_{plm_name}_{method}_d256"
            if is_done(all_results, result_name):
                print(f"  {result_name} already done")
                continue

            model = load_checkpoint(ckpt_name, embed_dim, device)
            if model is None:
                print(f"  {result_name}: checkpoint {ckpt_name} not found")
                continue

            print(f"  {result_name} (from {ckpt_name})...")
            compressed = compress_embeddings(model, chezod_emb, device)
            dis_result = evaluate_disorder_probe(
                compressed, chezod_scores, chezod_train_ids, chezod_test_ids
            )
            result = {"name": result_name, "plm": plm_name, "dataset": "CheZOD",
                       "method": method, "dim": 256, "checkpoint": ckpt_name, **dis_result}
            all_results.append(result)
            save_results(all_results)
            print(f"    Spearman rho={dis_result['spearman_rho']:.3f}")
            del model, compressed

    # ── 3c: TMbed Topology ──────────────────────────────────────
    print(f"\n  --- TMbed Topology Probes ---")

    tmbed_splits = _make_splits(list(tmbed_seqs.keys()))

    for plm_name, all_emb, embed_dim in [
        ("ESM2-650M", esm2_all, 1280),
        ("ProtT5-XL", t5_all, 1024),
    ]:
        tmbed_emb = _get_dataset_embeddings(all_emb, "tmbed")
        if not tmbed_emb:
            print(f"  No TMbed embeddings for {plm_name}")
            continue

        # Original
        name = f"r3_tmbed_{plm_name}_original"
        if not is_done(all_results, name):
            print(f"  {name}...")
            tm_accs, tm_f1s = [], []
            for seed, (tr, te) in tmbed_splits.items():
                tm_result = evaluate_tm_probe(tmbed_emb, tmbed_topo, tr, te)
                tm_accs.append(tm_result["accuracy"])
                tm_f1s.append(tm_result["macro_f1"])
            result = {"name": name, "plm": plm_name, "dataset": "TMbed",
                       "method": "original", "dim": embed_dim,
                       "accuracy_mean": float(np.mean(tm_accs)),
                       "accuracy_std": float(np.std(tm_accs)),
                       "macro_f1_mean": float(np.mean(tm_f1s)),
                       "macro_f1_std": float(np.std(tm_f1s))}
            all_results.append(result)
            save_results(all_results)
            print(f"    Acc={result['accuracy_mean']:.3f}, F1={result['macro_f1_mean']:.3f}")

        # ChannelCompressor checkpoints
        ckpt_prefix = "channel" if plm_name == "ESM2-650M" else "channel_prot_t5"
        for method in ["unsup", "contrastive"]:
            ckpt_name = f"{ckpt_prefix}_{method}_d256_s42"
            if plm_name == "ESM2-650M":
                ckpt_name = f"channel_{method}_d256_s123"
            result_name = f"r3_tmbed_{plm_name}_{method}_d256"
            if is_done(all_results, result_name):
                print(f"  {result_name} already done")
                continue

            model = load_checkpoint(ckpt_name, embed_dim, device)
            if model is None:
                print(f"  {result_name}: checkpoint {ckpt_name} not found")
                continue

            print(f"  {result_name} (from {ckpt_name})...")
            compressed = compress_embeddings(model, tmbed_emb, device)
            tm_accs, tm_f1s = [], []
            for seed, (tr, te) in tmbed_splits.items():
                tm_result = evaluate_tm_probe(compressed, tmbed_topo, tr, te)
                tm_accs.append(tm_result["accuracy"])
                tm_f1s.append(tm_result["macro_f1"])
            result = {"name": result_name, "plm": plm_name, "dataset": "TMbed",
                       "method": method, "dim": 256, "checkpoint": ckpt_name,
                       "accuracy_mean": float(np.mean(tm_accs)),
                       "accuracy_std": float(np.std(tm_accs)),
                       "macro_f1_mean": float(np.mean(tm_f1s)),
                       "macro_f1_std": float(np.std(tm_f1s))}
            all_results.append(result)
            save_results(all_results)
            print(f"    Acc={result['accuracy_mean']:.3f}, F1={result['macro_f1_mean']:.3f}")
            del model, compressed

    # ── 3d: TS115 Cross-Dataset SS3/SS8 ─────────────────────────
    print(f"\n  --- TS115 Cross-Dataset SS3/SS8 ---")

    ts115_splits = _make_splits(list(ts115_seqs.keys()))

    for plm_name, all_emb, embed_dim in [
        ("ESM2-650M", esm2_all, 1280),
        ("ProtT5-XL", t5_all, 1024),
    ]:
        ts115_emb = _get_dataset_embeddings(all_emb, "ts115")
        if not ts115_emb:
            print(f"  No TS115 embeddings for {plm_name}")
            continue

        # Original
        name = f"r3_ts115_{plm_name}_original"
        if not is_done(all_results, name):
            print(f"  {name}...")
            metrics = _run_ss_probes(ts115_emb, ts115_ss3, ts115_ss8, ts115_splits)
            result = {"name": name, "plm": plm_name, "dataset": "TS115",
                       "method": "original", "dim": embed_dim, **metrics}
            all_results.append(result)
            save_results(all_results)
            print(f"    Q3={metrics['q3_mean']:.3f}, Q8={metrics['q8_mean']:.3f}")

        # ChannelCompressor
        ckpt_prefix = "channel" if plm_name == "ESM2-650M" else "channel_prot_t5"
        for method in ["unsup", "contrastive"]:
            ckpt_name = f"{ckpt_prefix}_{method}_d256_s42"
            if plm_name == "ESM2-650M":
                ckpt_name = f"channel_{method}_d256_s123"
            result_name = f"r3_ts115_{plm_name}_{method}_d256"
            if is_done(all_results, result_name):
                print(f"  {result_name} already done")
                continue

            model = load_checkpoint(ckpt_name, embed_dim, device)
            if model is None:
                print(f"  {result_name}: checkpoint not found")
                continue

            print(f"  {result_name} (from {ckpt_name})...")
            compressed = compress_embeddings(model, ts115_emb, device)
            metrics = _run_ss_probes(compressed, ts115_ss3, ts115_ss8, ts115_splits)
            result = {"name": result_name, "plm": plm_name, "dataset": "TS115",
                       "method": method, "dim": 256, "checkpoint": ckpt_name, **metrics}
            all_results.append(result)
            save_results(all_results)
            print(f"    Q3={metrics['q3_mean']:.3f}, Q8={metrics['q8_mean']:.3f}")
            del model, compressed

    # ── Summary ─────────────────────────────────────────────────
    print_summary(all_results)
    print("\n  R3 complete.")


def print_summary(all_results):
    print(f"\n{'='*80}")
    print("ROBUST VALIDATION SUMMARY")
    print(f"{'='*80}")

    # Multi-seed ProtT5 Ret@1
    prott5_ret = []
    for r in all_results:
        if r.get("plm") == "ProtT5-XL" and r.get("method") == "channel_contrastive":
            p1 = r.get("retrieval_family", {}).get("precision@1")
            if p1 is not None:
                prott5_ret.append((r.get("seed", "?"), p1))
    if prott5_ret:
        print(f"\n  ProtT5 Contrastive Ret@1 by seed:")
        for seed, p1 in sorted(prott5_ret):
            print(f"    Seed {seed}: {p1:.3f}")
        vals = [p for _, p in prott5_ret]
        print(f"    Mean: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # Multi-seed CB513 Q3
    cb513_q3 = {}
    for r in all_results:
        if r.get("dataset") == "CB513" and "q3_mean" in r:
            key = f"{r.get('method', '?')}_s{r.get('model_seed', '?')}"
            cb513_q3[key] = r["q3_mean"]
    if cb513_q3:
        print(f"\n  ProtT5 CB513 Q3 by checkpoint:")
        for k, v in sorted(cb513_q3.items()):
            print(f"    {k}: Q3={v:.3f}")

    # CheZOD
    chezod = [(r["name"], r.get("spearman_rho", 0)) for r in all_results if r.get("dataset") == "CheZOD"]
    if chezod:
        print(f"\n  CheZOD Disorder (Spearman rho):")
        for name, rho in chezod:
            print(f"    {name}: rho={rho:.3f}")

    # TMbed
    tmbed = [(r["name"], r.get("accuracy_mean", 0), r.get("macro_f1_mean", 0))
             for r in all_results if r.get("dataset") == "TMbed"]
    if tmbed:
        print(f"\n  TMbed Topology:")
        for name, acc, f1 in tmbed:
            print(f"    {name}: Acc={acc:.3f}, F1={f1:.3f}")

    # TS115
    ts115 = [(r["name"], r.get("q3_mean", 0), r.get("q8_mean", 0))
             for r in all_results if r.get("dataset") == "TS115"]
    if ts115:
        print(f"\n  TS115 Cross-Dataset SS3/SS8:")
        for name, q3, q8 in ts115:
            print(f"    {name}: Q3={q3:.3f}, Q8={q8:.3f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Robust Validation")
    parser.add_argument("--step", type=str, default=None, help="Run specific step: R1, R2, R3")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    steps = {
        "R1": lambda: step_r1(device),
        "R2": lambda: step_r2(device),
        "R3": lambda: step_r3(device),
    }

    if args.step:
        step_key = args.step.upper()
        if step_key in steps:
            steps[step_key]()
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
            return
    else:
        for step_key in ["R1", "R2", "R3"]:
            steps[step_key]()


if __name__ == "__main__":
    main()
