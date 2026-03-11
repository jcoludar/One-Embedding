#!/usr/bin/env python3
"""Phase 11: Scaling Analysis, Failure Analysis, Architecture Ablations.

Answers key questions for publication:
  S1: Does more data help? (scaling curves)
  S2: What fails? (per-family Ret@1 breakdown)
  S3: What matters? (architecture ablations)
  S4: Can we go to 8x compression? (d128 contrastive on ProtT5)
  S5: Pareto visualization (compression vs performance)

Usage:
  uv run python experiments/17_scaling_and_ablations.py --step S1
  uv run python experiments/17_scaling_and_ablations.py --step S2
  uv run python experiments/17_scaling_and_ablations.py --step S3
  uv run python experiments/17_scaling_and_ablations.py --step S4
  uv run python experiments/17_scaling_and_ablations.py --step S5
  uv run python experiments/17_scaling_and_ablations.py              # run all
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.reconstruction import evaluate_reconstruction
from src.evaluation.splitting import load_split
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.training.objectives import InfoNCEFamilyLoss, ReconstructionLoss
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "scaling_ablation_results.json"
SPLIT_DIR = DATA_DIR / "splits"
PLOTS_DIR = DATA_DIR / "plots"

UNSUP_CHECKPOINT = "channel_prot_t5_unsup_d256_s42"
MAX_LEN = 512
SEEDS = [42, 123, 456]


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


def load_prot_t5_data():
    """Load ProtT5 5K embeddings and metadata (families >= 3 members)."""
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"

    print("  Loading ProtT5-XL embeddings...")
    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)

    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    embeddings = {k: v for k, v in embeddings.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filtered_meta))
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  Loaded: {len(embeddings)} proteins, {n_fam} families, dim={embed_dim}")
    return embeddings, filtered_meta


def get_splits():
    """Load superfamily-aware and classification splits."""
    sf_split = load_split(SPLIT_DIR / "esm2_650m_5k_split.json")
    train_ids, test_ids, eval_ids = sf_split

    with open(SPLIT_DIR / "esm2_650m_5k_cls_split.json") as f:
        cls_data = json.load(f)
    return train_ids, test_ids, eval_ids, cls_data["cls_train_ids"], cls_data["cls_test_ids"]


def run_contrastive_finetuning(
    model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
    lr=1e-4, temperature=0.07, batch_size=128, epochs=100,
    recon_reg_weight=0.1, seed=42, freeze_decoder=True,
):
    """Contrastive fine-tuning loop. Returns trained model."""
    embed_dim = next(iter(embeddings.values())).shape[-1]
    model = model.to(device)

    if freeze_decoder:
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
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    infonce_fn = InfoNCEFamilyLoss(temperature=temperature)
    recon_loss_fn = ReconstructionLoss().to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_pids))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            batch_pids = [train_pids[j] for j in idx]

            batch_embs, batch_masks, batch_labels = [], [], []
            for pid in batch_pids:
                emb = embeddings[pid]
                L = min(emb.shape[0], MAX_LEN)
                padded = np.zeros((MAX_LEN, embed_dim), dtype=np.float32)
                padded[:L] = emb[:L]
                mask = np.zeros(MAX_LEN, dtype=np.float32)
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
            loss = cl["loss"] + recon_reg_weight * recon_result["loss"]
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | Loss={epoch_loss / max(n_batches, 1):.4f}")

    for p in model.parameters():
        p.requires_grad = True

    return model


def prepare_contrastive_labels(metadata, embeddings, train_pids):
    """Build family label mappings for contrastive training."""
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    pids = [pid for pid in train_pids if pid in id_to_fam and pid in embeddings]
    unique_fams = sorted(set(id_to_fam[pid] for pid in pids))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}
    return pids, id_to_fam, fam_to_idx


# ── S1: Scaling Analysis ────────────────────────────────────────


def step_s1(embeddings, metadata, train_ids, test_ids, eval_ids,
            cls_train_ids, cls_test_ids, device):
    print(f"\n{'='*60}")
    print("S1: Scaling analysis — how much contrastive data is needed?")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    all_results = load_results()
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    id_to_fam = {m["id"]: m["family"] for m in metadata}

    for frac in fractions:
        name = f"scaling_contrastive_f{int(frac * 100):03d}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        # Subsample training IDs
        rng = np.random.RandomState(42)
        ids = [pid for pid in train_ids if pid in embeddings and pid in id_to_fam]
        rng.shuffle(ids)
        n = max(10, int(len(ids) * frac))
        sampled = ids[:n]

        # Filter to families with >= 2 members in sample
        fam_counts = Counter(id_to_fam[pid] for pid in sampled)
        valid_fams = {f for f, c in fam_counts.items() if c >= 2}
        train_pids = [pid for pid in sampled if id_to_fam[pid] in valid_fams]

        n_fams = len(valid_fams)
        print(f"\n  {name}: {len(train_pids)} proteins, {n_fams} families "
              f"(from {n} sampled at {frac:.0%})")
        monitor()

        # Load unsupervised checkpoint
        model = ChannelCompressor(
            input_dim=embed_dim, latent_dim=256, dropout=0.1, use_residual=True,
        )
        ckpt_path = CHECKPOINTS_DIR / UNSUP_CHECKPOINT / "best_model.pt"
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )

        unique_fams_sorted = sorted(valid_fams)
        fam_to_idx = {f: i for i, f in enumerate(unique_fams_sorted)}

        start = time.time()
        model = run_contrastive_finetuning(
            model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
            seed=42, epochs=100,
        )
        elapsed = time.time() - start

        # Evaluate
        model.eval()
        ret = evaluate_retrieval(
            model, embeddings, metadata, label_key="family", k_values=[1, 5],
            device=device, query_ids=eval_ids, database_ids=test_ids,
            pooling_strategy="mean",
        )
        cls = evaluate_linear_probe(
            model, embeddings, metadata, label_key="family",
            device=device, train_ids=cls_train_ids, test_ids=cls_test_ids,
        )

        result = {
            "name": name,
            "step": "S1",
            "fraction": frac,
            "n_train_proteins": len(train_pids),
            "n_train_families": n_fams,
            "retrieval": {
                "precision@1": ret.get("precision@1", 0),
                "precision@5": ret.get("precision@5", 0),
                "mrr": ret.get("mrr", 0),
            },
            "classification": cls,
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        ret1 = ret.get("precision@1", 0)
        print(f"  >> {name}: Ret@1={ret1:.3f}, MRR={ret.get('mrr', 0):.3f}, "
              f"Cls={cls.get('accuracy_mean', 0):.3f} ({elapsed:.0f}s)")

        print("  30s thermal cooldown...")
        time.sleep(30)
        monitor()


# ── S2: Failure Analysis ────────────────────────────────────────


def step_s2(embeddings, metadata, test_ids, eval_ids, device):
    print(f"\n{'='*60}")
    print("S2: Failure analysis — per-family Ret@1 breakdown")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Load best contrastive model
    ckpt_name = "channel_prot_t5_contrastive_d256_s42"
    model = ChannelCompressor(
        input_dim=embed_dim, latent_dim=256, dropout=0.1, use_residual=True,
    )
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / ckpt_name / "best_model.pt",
                    map_location=device, weights_only=True)
    )

    # Retrieval with per-query scores
    ret = evaluate_retrieval(
        model, embeddings, metadata, label_key="family", k_values=[1, 5],
        device=device, query_ids=eval_ids, database_ids=test_ids,
        pooling_strategy="mean", return_per_query=True,
    )

    per_query = ret.get("per_query", {})
    pq_p1 = per_query.get("precision@1", {})
    pq_mrr = per_query.get("mrr", {})

    if not pq_p1:
        print("  ERROR: No per-query results returned")
        return

    # Build per-family analysis
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    id_to_sf = {m["id"]: m["superfamily"] for m in metadata}
    id_to_class = {m["id"]: m.get("class_name", "?") for m in metadata}

    family_scores = defaultdict(list)
    family_mrr = defaultdict(list)
    for qid, score in pq_p1.items():
        fam = id_to_fam.get(qid, "unknown")
        family_scores[fam].append(score)
        if qid in pq_mrr:
            family_mrr[fam].append(pq_mrr[qid])

    # Per-family stats
    family_stats = []
    test_set = set(test_ids)
    fam_sizes_in_test = Counter(id_to_fam[m["id"]] for m in metadata if m["id"] in test_set)

    for fam, scores in family_scores.items():
        # Find a representative protein for class/superfamily info
        rep_pid = next((m["id"] for m in metadata if m["family"] == fam), None)
        avg_seqlen = np.mean([
            embeddings[pid].shape[0]
            for pid in pq_p1 if id_to_fam.get(pid) == fam and pid in embeddings
        ]) if any(pid in embeddings for pid in pq_p1 if id_to_fam.get(pid) == fam) else 0

        family_stats.append({
            "family": fam,
            "superfamily": id_to_sf.get(rep_pid, "?") if rep_pid else "?",
            "class": id_to_class.get(rep_pid, "?") if rep_pid else "?",
            "mean_ret1": float(np.mean(scores)),
            "mean_mrr": float(np.mean(family_mrr.get(fam, [0]))),
            "n_queries": len(scores),
            "n_in_test_db": fam_sizes_in_test.get(fam, 0),
            "avg_seq_length": float(avg_seqlen),
        })

    # Sort by mean_ret1
    family_stats.sort(key=lambda x: x["mean_ret1"])

    print(f"\n  Total families evaluated: {len(family_stats)}")
    print(f"  Overall Ret@1: {ret.get('precision@1', 0):.3f}")

    # Bottom-10 (hardest)
    print(f"\n  Bottom 10 families (hardest):")
    print(f"  {'Family':<35} {'Ret@1':>6} {'MRR':>6} {'#Q':>3} {'#DB':>3} {'AvgLen':>6} {'Class':<8}")
    for fs in family_stats[:10]:
        print(f"  {fs['family'][:35]:<35} {fs['mean_ret1']:>6.3f} {fs['mean_mrr']:>6.3f} "
              f"{fs['n_queries']:>3d} {fs['n_in_test_db']:>3d} {fs['avg_seq_length']:>6.0f} {fs['class']:<8}")

    # Top-10 (easiest)
    print(f"\n  Top 10 families (easiest):")
    print(f"  {'Family':<35} {'Ret@1':>6} {'MRR':>6} {'#Q':>3} {'#DB':>3} {'AvgLen':>6} {'Class':<8}")
    for fs in family_stats[-10:]:
        print(f"  {fs['family'][:35]:<35} {fs['mean_ret1']:>6.3f} {fs['mean_mrr']:>6.3f} "
              f"{fs['n_queries']:>3d} {fs['n_in_test_db']:>3d} {fs['avg_seq_length']:>6.0f} {fs['class']:<8}")

    # Correlations
    sizes = [fs["n_in_test_db"] for fs in family_stats]
    scores_arr = [fs["mean_ret1"] for fs in family_stats]
    lengths = [fs["avg_seq_length"] for fs in family_stats if fs["avg_seq_length"] > 0]
    scores_len = [fs["mean_ret1"] for fs in family_stats if fs["avg_seq_length"] > 0]

    if len(sizes) > 2:
        corr_size = float(np.corrcoef(sizes, scores_arr)[0, 1])
        print(f"\n  Correlation(family_size_in_db, Ret@1): r={corr_size:.3f}")
    if len(lengths) > 2:
        corr_len = float(np.corrcoef(lengths, scores_len)[0, 1])
        print(f"  Correlation(avg_seq_length, Ret@1): r={corr_len:.3f}")

    # Class-level aggregation
    class_scores = defaultdict(list)
    for fs in family_stats:
        class_scores[fs["class"]].append(fs["mean_ret1"])

    print(f"\n  Per-class mean Ret@1:")
    for cls in sorted(class_scores.keys()):
        mean = np.mean(class_scores[cls])
        n = len(class_scores[cls])
        print(f"    {cls}: {mean:.3f} ({n} families)")

    # Save analysis
    analysis = {
        "overall_ret1": ret.get("precision@1", 0),
        "overall_mrr": ret.get("mrr", 0),
        "n_families": len(family_stats),
        "n_queries": len(pq_p1),
        "bottom_10": family_stats[:10],
        "top_10": family_stats[-10:],
        "per_class": {cls: {"mean_ret1": float(np.mean(s)), "n_families": len(s)}
                      for cls, s in class_scores.items()},
        "all_families": family_stats,
    }
    if len(sizes) > 2:
        analysis["corr_size_ret1"] = corr_size
    if len(lengths) > 2:
        analysis["corr_seqlen_ret1"] = corr_len

    out_path = DATA_DIR / "benchmarks" / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved failure analysis to {out_path}")


# ── S3: Architecture Ablations ──────────────────────────────────


def step_s3(embeddings, metadata, train_ids, test_ids, eval_ids,
            cls_train_ids, cls_test_ids, device):
    print(f"\n{'='*60}")
    print("S3: Architecture ablations")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    d_prime = 256
    all_results = load_results()

    train_pids, id_to_fam, fam_to_idx = prepare_contrastive_labels(
        metadata, embeddings, train_ids,
    )
    test_set = set(test_ids)
    val_emb = {k: v for k, v in embeddings.items() if k in test_set}

    ablations = [
        ("ablation_no_layernorm", "no_layernorm"),
        ("ablation_no_residual", "no_residual"),
        ("ablation_no_decoder_freeze", "no_decoder_freeze"),
    ]

    for name, ablation_type in ablations:
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        print(f"\n  {name}...")
        monitor()

        if ablation_type == "no_decoder_freeze":
            # Reuse existing unsupervised checkpoint, just skip freeze
            model = ChannelCompressor(
                input_dim=embed_dim, latent_dim=d_prime, dropout=0.1, use_residual=True,
            )
            model.load_state_dict(
                torch.load(CHECKPOINTS_DIR / UNSUP_CHECKPOINT / "best_model.pt",
                           map_location=device, weights_only=True)
            )
            print(f"  Loaded unsup checkpoint, contrastive WITHOUT decoder freeze")

            start = time.time()
            model = run_contrastive_finetuning(
                model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
                seed=42, freeze_decoder=False,
            )
            elapsed = time.time() - start

        else:
            # Need to train unsupervised from scratch with modified architecture
            if ablation_type == "no_residual":
                model = ChannelCompressor(
                    input_dim=embed_dim, latent_dim=d_prime,
                    dropout=0.1, use_residual=False,
                )
                print(f"  Architecture: no residual connections")
            elif ablation_type == "no_layernorm":
                model = ChannelCompressor(
                    input_dim=embed_dim, latent_dim=d_prime,
                    dropout=0.1, use_residual=True,
                )
                # Replace LayerNorm with Identity
                model.input_norm = nn.Identity()
                model.enc_norm1 = nn.Identity()
                model.dec_norm1 = nn.Identity()
                print(f"  Architecture: no LayerNorm (replaced with Identity)")

            # Train unsupervised
            from src.training.trainer import train_compressor

            unsup_ckpt_dir = CHECKPOINTS_DIR / f"{name}_unsup"
            print(f"  Training unsupervised ({ablation_type})...")
            start_unsup = time.time()
            train_compressor(
                model, {k: v for k, v in embeddings.items() if k in set(train_ids)},
                epochs=200, batch_size=16, lr=1e-3, device=device,
                max_len=MAX_LEN, seed=42,
                validation_embeddings=val_emb,
                checkpoint_dir=unsup_ckpt_dir,
            )
            unsup_elapsed = time.time() - start_unsup
            print(f"  Unsupervised done in {unsup_elapsed:.0f}s")

            # Load best unsup checkpoint
            best_path = unsup_ckpt_dir / "best_model.pt"
            if best_path.exists():
                model.load_state_dict(
                    torch.load(best_path, map_location=device, weights_only=True)
                )

            # Contrastive fine-tuning
            print(f"  Contrastive fine-tuning ({ablation_type})...")
            start = time.time()
            model = run_contrastive_finetuning(
                model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
                seed=42,
            )
            elapsed = unsup_elapsed + (time.time() - start)

        # Evaluate
        model.eval()
        ret = evaluate_retrieval(
            model, embeddings, metadata, label_key="family", k_values=[1, 5],
            device=device, query_ids=eval_ids, database_ids=test_ids,
            pooling_strategy="mean",
        )
        cls = evaluate_linear_probe(
            model, embeddings, metadata, label_key="family",
            device=device, train_ids=cls_train_ids, test_ids=cls_test_ids,
        )
        recon = evaluate_reconstruction(model, val_emb, device)

        # Save checkpoint
        ckpt_dir = CHECKPOINTS_DIR / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        result = {
            "name": name,
            "step": "S3",
            "ablation": ablation_type,
            "latent_dim": d_prime,
            "retrieval": {
                "precision@1": ret.get("precision@1", 0),
                "precision@5": ret.get("precision@5", 0),
                "mrr": ret.get("mrr", 0),
            },
            "classification": cls,
            "reconstruction": recon,
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        ret1 = ret.get("precision@1", 0)
        cos = recon.get("cosine_sim", 0)
        print(f"  >> {name}: Ret@1={ret1:.3f}, Cls={cls.get('accuracy_mean', 0):.3f}, "
              f"CosSim={cos:.3f} ({elapsed:.0f}s)")

        print("  30s thermal cooldown...")
        time.sleep(30)
        monitor()


# ── S4: d128 Contrastive on ProtT5 ──────────────────────────────


def step_s4(embeddings, metadata, train_ids, test_ids, eval_ids,
            cls_train_ids, cls_test_ids, device):
    print(f"\n{'='*60}")
    print("S4: d128 contrastive on ProtT5 — can we go to 8x compression?")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    d_prime = 128
    all_results = load_results()

    train_pids, id_to_fam, fam_to_idx = prepare_contrastive_labels(
        metadata, embeddings, train_ids,
    )
    test_set = set(test_ids)
    val_emb = {k: v for k, v in embeddings.items() if k in test_set}

    from src.training.trainer import train_compressor

    for seed in SEEDS:
        unsup_name = f"prot_t5_unsup_d128_s{seed}"
        contrastive_name = f"prot_t5_contrastive_d128_s{seed}"

        # Step 1: Unsupervised training
        unsup_ckpt_dir = CHECKPOINTS_DIR / f"channel_{unsup_name}"
        if not (unsup_ckpt_dir / "best_model.pt").exists():
            print(f"\n  Training unsupervised {unsup_name}...")
            monitor()

            model = ChannelCompressor(
                input_dim=embed_dim, latent_dim=d_prime,
                dropout=0.1, use_residual=True,
            )

            start = time.time()
            train_compressor(
                model, {k: v for k, v in embeddings.items() if k in set(train_ids)},
                epochs=200, batch_size=16, lr=1e-3, device=device,
                max_len=MAX_LEN, seed=seed,
                validation_embeddings=val_emb,
                checkpoint_dir=unsup_ckpt_dir,
            )
            elapsed = time.time() - start
            print(f"  Unsupervised done in {elapsed:.0f}s")

            print("  30s thermal cooldown...")
            time.sleep(30)
            monitor()
        else:
            print(f"  {unsup_name} checkpoint exists")

        # Step 2: Contrastive fine-tuning
        if is_done(all_results, contrastive_name):
            print(f"  {contrastive_name} already done")
            continue

        print(f"\n  Contrastive fine-tuning {contrastive_name}...")
        monitor()

        model = ChannelCompressor(
            input_dim=embed_dim, latent_dim=d_prime,
            dropout=0.1, use_residual=True,
        )
        model.load_state_dict(
            torch.load(unsup_ckpt_dir / "best_model.pt",
                        map_location=device, weights_only=True)
        )

        start = time.time()
        model = run_contrastive_finetuning(
            model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
            seed=seed,
        )
        elapsed = time.time() - start

        # Evaluate
        model.eval()
        ret = evaluate_retrieval(
            model, embeddings, metadata, label_key="family", k_values=[1, 5],
            device=device, query_ids=eval_ids, database_ids=test_ids,
            pooling_strategy="mean",
        )
        cls = evaluate_linear_probe(
            model, embeddings, metadata, label_key="family",
            device=device, train_ids=cls_train_ids, test_ids=cls_test_ids,
        )
        recon = evaluate_reconstruction(model, val_emb, device)

        # Save checkpoint
        ckpt_dir = CHECKPOINTS_DIR / f"channel_{contrastive_name}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        result = {
            "name": contrastive_name,
            "step": "S4",
            "plm": "ProtT5-XL",
            "latent_dim": d_prime,
            "seed": seed,
            "retrieval": {
                "precision@1": ret.get("precision@1", 0),
                "precision@5": ret.get("precision@5", 0),
                "mrr": ret.get("mrr", 0),
            },
            "classification": cls,
            "reconstruction": recon,
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        ret1 = ret.get("precision@1", 0)
        cos = recon.get("cosine_sim", 0)
        print(f"  >> {contrastive_name}: Ret@1={ret1:.3f}, "
              f"Cls={cls.get('accuracy_mean', 0):.3f}, CosSim={cos:.3f} ({elapsed:.0f}s)")

        print("  30s thermal cooldown...")
        time.sleep(30)
        monitor()

    # Summary
    d128_results = [r for r in load_results() if r.get("step") == "S4"]
    if d128_results:
        ret1_vals = [r["retrieval"]["precision@1"] for r in d128_results]
        print(f"\n  d128 ProtT5 contrastive: Ret@1 = {np.mean(ret1_vals):.3f} "
              f"+/- {np.std(ret1_vals):.3f} (n={len(ret1_vals)})")


# ── S5: Pareto Visualization ────────────────────────────────────


def step_s5():
    print(f"\n{'='*60}")
    print("S5: Pareto visualization — compression vs performance")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect data from all result files
    points = []  # (label, d_prime, ret1, marker, color)

    # ESM2 results from channel_compression_results.json
    esm2_path = DATA_DIR / "benchmarks" / "channel_compression_results.json"
    if esm2_path.exists():
        with open(esm2_path) as f:
            esm2_data = json.load(f)

        for r in esm2_data:
            name = r.get("name", "")
            ret1 = r.get("retrieval_family", {}).get("precision@1")
            if ret1 is None:
                continue

            if name == "channel_baseline_meanpool":
                points.append(("ESM2 Mean-pool", 1280, ret1, "s", "#999999"))
            elif name.startswith("channel_baseline_pca_d"):
                d = r.get("latent_dim", 0)
                points.append((f"ESM2 PCA", d, ret1, "^", "#2196F3"))
            elif name.startswith("channel_unsup_d") and not name.startswith("channel_unsup_d64"):
                d = r.get("latent_dim", 0)
                points.append((f"ESM2 Unsup", d, ret1, "o", "#4CAF50"))
            elif name.startswith("channel_contrastive_d"):
                d = r.get("latent_dim", 0)
                points.append((f"ESM2 Contrastive", d, ret1, "D", "#FF9800"))

    # ProtT5 results from channel_compression_results.json
    for r in (esm2_data if esm2_path.exists() else []):
        name = r.get("name", "")
        ret1 = r.get("retrieval_family", {}).get("precision@1")
        if ret1 is None:
            continue
        if name.startswith("channel_prot_t5_unsup"):
            points.append(("ProtT5 Unsup", 256, ret1, "o", "#8BC34A"))
        elif name.startswith("channel_prot_t5_contrastive"):
            points.append(("ProtT5 Contrastive", 256, ret1, "D", "#E91E63"))

    # ProtT5 3-seed results from robust_validation_results.json
    robust_path = DATA_DIR / "benchmarks" / "robust_validation_results.json"
    if robust_path.exists():
        with open(robust_path) as f:
            robust_data = json.load(f)
        for r in robust_data:
            name = r.get("name", "")
            if name.startswith("prot_t5_contrastive_d256_s"):
                ret1 = r.get("retrieval", {}).get("precision@1")
                if ret1:
                    points.append(("ProtT5 Contrastive", 256, ret1, "D", "#E91E63"))

    # New d128 ProtT5 results from this experiment
    our_results = load_results()
    for r in our_results:
        if r.get("step") == "S4":
            ret1 = r["retrieval"]["precision@1"]
            points.append(("ProtT5 Contrastive", 128, ret1, "D", "#E91E63"))

    # Ablation results
    for r in our_results:
        if r.get("step") == "S3":
            ret1 = r["retrieval"]["precision@1"]
            abl = r.get("ablation", "")
            label = f"Ablation: {abl.replace('_', ' ')}"
            points.append((label, 256, ret1, "x", "#9C27B0"))

    if not points:
        print("  No data points found for Pareto plot")
        return

    # Aggregate by (label, d_prime) → mean ret1
    agg = defaultdict(list)
    for label, d, ret1, marker, color in points:
        agg[(label, d, marker, color)].append(ret1)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Track labels for legend (avoid duplicates)
    legend_entries = {}

    for (label, d, marker, color), vals in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        mean_ret1 = np.mean(vals)
        std_ret1 = np.std(vals) if len(vals) > 1 else 0

        if label not in legend_entries:
            legend_entries[label] = ax.errorbar(
                d, mean_ret1, yerr=std_ret1 if std_ret1 > 0 else None,
                fmt=marker, color=color, markersize=10, capsize=3,
                label=label, markeredgecolor="black", markeredgewidth=0.5,
            )
        else:
            ax.errorbar(
                d, mean_ret1, yerr=std_ret1 if std_ret1 > 0 else None,
                fmt=marker, color=color, markersize=10, capsize=3,
                markeredgecolor="black", markeredgewidth=0.5,
            )

    # Connect methods across dimensions
    for method_label in ["ESM2 PCA", "ESM2 Unsup", "ESM2 Contrastive", "ProtT5 Contrastive"]:
        method_points = [(d, np.mean(vals)) for (label, d, _, _), vals in agg.items()
                         if label == method_label]
        if len(method_points) > 1:
            method_points.sort()
            ds, rs = zip(*method_points)
            color = next(c for (l, _, _, c), _ in agg.items() if l == method_label)
            ax.plot(ds, rs, "--", color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel("Latent Dimension (d')", fontsize=12)
    ax.set_ylabel("Retrieval Precision@1", fontsize=12)
    ax.set_title("Compression vs. Retrieval Performance (Pareto)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024, 1280])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 0.9)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "pareto_compression_vs_retrieval.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved Pareto plot to {out_path}")

    # Also create scaling curve if S1 results exist
    s1_results = [r for r in our_results if r.get("step") == "S1"]
    if s1_results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        fracs = [r["fraction"] for r in s1_results]
        ret1s = [r["retrieval"]["precision@1"] for r in s1_results]
        n_prots = [r["n_train_proteins"] for r in s1_results]

        ax2.plot(fracs, ret1s, "o-", color="#E91E63", markersize=8, linewidth=2)
        for f, r, n in zip(fracs, ret1s, n_prots):
            ax2.annotate(f"n={n}", (f, r), textcoords="offset points",
                        xytext=(5, 10), fontsize=8, color="#666666")

        # Baseline reference line
        ax2.axhline(y=0.795, color="#999999", linestyle="--", linewidth=1, label="Baseline (100%)")
        ax2.set_xlabel("Fraction of Training Data", fontsize=12)
        ax2.set_ylabel("Retrieval Precision@1", fontsize=12)
        ax2.set_title("Scaling Analysis: How Much Contrastive Data Is Needed?", fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)

        out_path2 = PLOTS_DIR / "scaling_curve.png"
        fig2.tight_layout()
        fig2.savefig(out_path2, dpi=150)
        plt.close(fig2)
        print(f"  Saved scaling curve to {out_path2}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Scaling, failure analysis, ablations")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step: S1, S2, S3, S4, S5")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    steps = [args.step.upper()] if args.step else ["S1", "S2", "S3", "S4", "S5"]

    # Only load data if needed (S5 can run standalone)
    need_data = any(s in steps for s in ["S1", "S2", "S3", "S4"])

    embeddings, metadata = None, None
    train_ids = test_ids = eval_ids = cls_train_ids = cls_test_ids = None

    if need_data:
        embeddings, metadata = load_prot_t5_data()
        train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids = get_splits()

    for step in steps:
        if step == "S1":
            step_s1(embeddings, metadata, train_ids, test_ids, eval_ids,
                    cls_train_ids, cls_test_ids, device)
        elif step == "S2":
            step_s2(embeddings, metadata, test_ids, eval_ids, device)
        elif step == "S3":
            step_s3(embeddings, metadata, train_ids, test_ids, eval_ids,
                    cls_train_ids, cls_test_ids, device)
        elif step == "S4":
            step_s4(embeddings, metadata, train_ids, test_ids, eval_ids,
                    cls_train_ids, cls_test_ids, device)
        elif step == "S5":
            step_s5()
        else:
            print(f"Unknown step: {step}")
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
