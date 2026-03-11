#!/usr/bin/env python3
"""Phase 10: Hyperparameter Optimization for Contrastive Fine-Tuning.

Uses Optuna to search for optimal contrastive fine-tuning hyperparameters
for the ProtT5-XL ChannelCompressor d256 pipeline.

Steps:
  H0: Create HPO validation split from training data
  H1: Run Optuna HPO (30 trials, TPE sampler)
  H2: Retrain top-3 configs on full training set (3 seeds each)
  H3: Final evaluation on held-out test set + statistical comparison

Usage:
  uv run python experiments/16_hpo_contrastive.py --step H0    # create split
  uv run python experiments/16_hpo_contrastive.py --step H1    # HPO search
  uv run python experiments/16_hpo_contrastive.py --step H2    # retrain top-3
  uv run python experiments/16_hpo_contrastive.py --step H3    # final eval
  uv run python experiments/16_hpo_contrastive.py              # run all
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

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.reconstruction import evaluate_reconstruction
from src.evaluation.splitting import (
    load_split,
    save_split,
    split_statistics,
    superfamily_aware_split,
)
from src.evaluation.statistical_tests import paired_bootstrap_test, multi_seed_permutation_test, cohens_d
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv, read_fasta
from src.training.objectives import InfoNCEFamilyLoss, ReconstructionLoss
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "hpo_contrastive_results.json"
SPLIT_DIR = DATA_DIR / "splits"
HPO_SPLIT_PATH = SPLIT_DIR / "hpo_val_split.json"
HPO_STUDY_PATH = DATA_DIR / "benchmarks" / "hpo_study.json"

D_PRIME = 256
UNSUP_CHECKPOINT = "channel_prot_t5_unsup_d256_s42"
BASELINE_CHECKPOINT = "channel_prot_t5_contrastive_d256_s42"

SEEDS = [42, 123, 456]
MAX_LEN = 512


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


def load_data():
    """Load ProtT5 5K embeddings and metadata."""
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"

    print("  Loading ProtT5-XL embeddings...")
    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)

    # Filter to families >= 3 members
    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    embeddings = {k: v for k, v in embeddings.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filtered_meta))
    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"  Loaded: {len(embeddings)} proteins, {n_fam} families, dim={embed_dim}")
    return embeddings, filtered_meta


def get_original_split():
    """Load the original superfamily-aware split (train/test)."""
    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    train_ids, test_ids, eval_ids = load_split(sf_split_path)
    return train_ids, test_ids, eval_ids


def load_checkpoint(ckpt_name, input_dim, device):
    """Load a ChannelCompressor checkpoint."""
    ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = ChannelCompressor(
        input_dim=input_dim, latent_dim=D_PRIME, dropout=0.1, use_residual=True,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model.to(device)


def run_contrastive_finetuning(
    model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
    lr=1e-4, temperature=0.07, batch_size=128, epochs=100,
    recon_reg_weight=0.1, seed=42, report_fn=None,
):
    """Run contrastive fine-tuning loop. Returns trained model and final metrics.

    Args:
        report_fn: Optional callback(epoch, ret1) for Optuna pruning.
    """
    embed_dim = next(iter(embeddings.values())).shape[-1]
    model = model.to(device)

    # Freeze decoder
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

            batch_embs = []
            batch_masks = []
            batch_labels = []
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

            # Guard against NaN gradients (inf * 0 = NaN with clip_grad_norm_)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Report for pruning at checkpoint epochs
        if report_fn is not None and epoch in (25, 50, 75, 100):
            report_fn(epoch, epoch_loss / max(n_batches, 1))

    # Unfreeze for evaluation
    for p in model.parameters():
        p.requires_grad = True

    return model


def evaluate_on_split(model, embeddings, metadata, eval_ids, device):
    """Quick evaluation: Ret@1, MRR on a set of eval IDs."""
    results = evaluate_retrieval(
        model, embeddings, metadata,
        label_key="family",
        k_values=[1],
        device=device,
        query_ids=eval_ids,
        database_ids=eval_ids,
        pooling_strategy="mean",
    )
    return results


# ── H0: Create HPO Validation Split ─────────────────────────────


def step_h0(metadata, train_ids):
    print(f"\n{'='*60}")
    print("H0: Create HPO validation split from training data")
    print(f"{'='*60}")

    if HPO_SPLIT_PATH.exists():
        print(f"  HPO split already exists at {HPO_SPLIT_PATH}")
        with open(HPO_SPLIT_PATH) as f:
            data = json.load(f)
        print(f"  hpo_train: {len(data['hpo_train_ids'])}, hpo_val: {len(data['hpo_val_ids'])}")
        return data["hpo_train_ids"], data["hpo_val_ids"]

    # Filter metadata to only training proteins
    train_set = set(train_ids)
    train_meta = [m for m in metadata if m["id"] in train_set]

    print(f"  Training proteins: {len(train_meta)}")

    # Use superfamily_aware_split on training proteins only
    hpo_train_ids, hpo_val_ids, hpo_eval_ids = superfamily_aware_split(
        train_meta, test_fraction=0.2, seed=99,
    )

    stats = split_statistics(train_meta, hpo_train_ids, hpo_val_ids, hpo_eval_ids)

    print(f"  HPO split: {len(hpo_train_ids)} train, {len(hpo_val_ids)} val")
    print(f"  Superfamily overlap: {stats['superfamily_overlap']} (must be 0)")
    print(f"  Family overlap: {stats['family_overlap']}")
    print(f"  HPO val families: {stats['n_test_families']}")
    print(f"  HPO eval (fam>=2 in val): {len(hpo_eval_ids)}")

    assert stats["superfamily_overlap"] == 0, "HPO split has superfamily leakage!"

    # Save
    HPO_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hpo_train_ids": hpo_train_ids,
        "hpo_val_ids": hpo_val_ids,
        "hpo_eval_ids": hpo_eval_ids,
        "statistics": stats,
    }
    with open(HPO_SPLIT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved HPO split to {HPO_SPLIT_PATH}")

    return hpo_train_ids, hpo_val_ids


# ── H1: Optuna HPO ──────────────────────────────────────────────


def step_h1(embeddings, metadata, hpo_train_ids, hpo_val_ids, device):
    print(f"\n{'='*60}")
    print("H1: Optuna HPO for contrastive fine-tuning")
    print(f"{'='*60}")

    import optuna

    if HPO_STUDY_PATH.exists():
        print(f"  HPO study results already exist at {HPO_STUDY_PATH}")
        with open(HPO_STUDY_PATH) as f:
            study_data = json.load(f)
        print(f"  Best trial: {study_data['best_trial']}")
        return study_data

    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Build family label mapping for training proteins
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    train_pids = [pid for pid in hpo_train_ids if pid in id_to_fam and pid in embeddings]
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

    # HPO eval IDs (val proteins with family >= 2 members in val)
    with open(HPO_SPLIT_PATH) as f:
        hpo_data = json.load(f)
    hpo_eval_ids = hpo_data["hpo_eval_ids"]

    print(f"  HPO train: {len(train_pids)} proteins")
    print(f"  HPO eval: {len(hpo_eval_ids)} proteins (for Ret@1)")

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
        temperature = trial.suggest_float("temperature", 0.03, 0.15, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        recon_reg_weight = trial.suggest_float("recon_reg_weight", 0.01, 0.5, log=True)
        epochs = trial.suggest_categorical("epochs", [50, 100, 150])
        dropout = trial.suggest_float("dropout", 0.05, 0.25)

        print(f"\n  Trial {trial.number}: lr={lr:.2e}, temp={temperature:.3f}, "
              f"bs={batch_size}, recon={recon_reg_weight:.3f}, ep={epochs}, do={dropout:.2f}")

        # Load fresh unsupervised checkpoint
        model = ChannelCompressor(
            input_dim=embed_dim, latent_dim=D_PRIME, dropout=dropout, use_residual=True,
        )
        ckpt_path = CHECKPOINTS_DIR / UNSUP_CHECKPOINT / "best_model.pt"
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

        # Handle dropout mismatch: checkpoint was trained with dropout=0.1,
        # but we may use a different value. Architecture weights are compatible.
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

        # Define pruning callback
        def report_fn(epoch, loss):
            # Evaluate on hpo_val for pruning
            model.eval()
            with torch.no_grad():
                quick_eval = evaluate_on_split(
                    model, embeddings, metadata, hpo_eval_ids, device,
                )
            ret1 = quick_eval.get("precision@1", 0.0)
            mrr = quick_eval.get("mrr", 0.0)
            composite = 0.7 * ret1 + 0.3 * mrr
            trial.report(composite, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            model.train()

        try:
            model = run_contrastive_finetuning(
                model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
                lr=lr, temperature=temperature, batch_size=batch_size,
                epochs=epochs, recon_reg_weight=recon_reg_weight, seed=42,
                report_fn=report_fn,
            )
        except optuna.TrialPruned:
            raise

        # Final evaluation on hpo_val
        model.eval()
        eval_results = evaluate_on_split(
            model, embeddings, metadata, hpo_eval_ids, device,
        )
        ret1 = eval_results.get("precision@1", 0.0)
        mrr = eval_results.get("mrr", 0.0)
        composite = 0.7 * ret1 + 0.3 * mrr

        print(f"  Trial {trial.number} result: Ret@1={ret1:.3f}, MRR={mrr:.3f}, "
              f"composite={composite:.3f}")

        # Cooldown between trials
        print("  30s thermal cooldown...")
        time.sleep(30)
        monitor()

        return composite

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25),
    )

    print(f"\n  Starting HPO: 30 trials, TPE sampler, MedianPruner")
    monitor()
    study.optimize(objective, n_trials=30)

    # Save study results
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })

    study_data = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "all_trials": trials_data,
        "n_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
    }

    # Get top-3 completed trials
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    top3 = completed[:3]
    study_data["top3_trials"] = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in top3
    ]

    HPO_STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HPO_STUDY_PATH, "w") as f:
        json.dump(study_data, f, indent=2)
    print(f"\n  HPO complete. Best composite={study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")
    print(f"  Completed: {study_data['n_completed']}, Pruned: {study_data['n_pruned']}")
    print(f"  Saved to {HPO_STUDY_PATH}")

    return study_data


# ── H2: Retrain Top-3 Configs ───────────────────────────────────


def step_h2(embeddings, metadata, train_ids, device):
    print(f"\n{'='*60}")
    print("H2: Retrain top-3 HPO configs on full training set (3 seeds)")
    print(f"{'='*60}")

    if not HPO_STUDY_PATH.exists():
        print("  HPO study not found. Run H1 first.")
        return

    with open(HPO_STUDY_PATH) as f:
        study_data = json.load(f)

    top3 = study_data.get("top3_trials", [])
    if not top3:
        print("  No completed trials found in HPO study.")
        return

    embed_dim = next(iter(embeddings.values())).shape[-1]
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    train_pids = [pid for pid in train_ids if pid in id_to_fam and pid in embeddings]
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

    all_results = load_results()

    for rank, trial_info in enumerate(top3):
        params = trial_info["params"]
        trial_num = trial_info["number"]

        for seed in SEEDS:
            name = f"hpo_top{rank+1}_trial{trial_num}_s{seed}"

            if is_done(all_results, name):
                print(f"  {name} already done, skipping")
                continue

            print(f"\n  Training {name}...")
            print(f"  Params: {params}")
            monitor()

            # Load fresh unsupervised checkpoint
            model = ChannelCompressor(
                input_dim=embed_dim, latent_dim=D_PRIME,
                dropout=params.get("dropout", 0.1), use_residual=True,
            )
            ckpt_path = CHECKPOINTS_DIR / UNSUP_CHECKPOINT / "best_model.pt"
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True),
                strict=True,
            )

            start = time.time()
            model = run_contrastive_finetuning(
                model, embeddings, train_pids, id_to_fam, fam_to_idx, device,
                lr=params["lr"],
                temperature=params["temperature"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                recon_reg_weight=params["recon_reg_weight"],
                seed=seed,
            )
            elapsed = time.time() - start
            print(f"  Training done in {elapsed:.0f}s")

            # Save checkpoint
            ckpt_dir = CHECKPOINTS_DIR / f"hpo_{name}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "hpo_trial": trial_num,
                "hpo_rank": rank + 1,
                "params": params,
                "seed": seed,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            # Thermal cooldown
            print("  30s thermal cooldown...")
            time.sleep(30)
            monitor()


# ── H3: Final Evaluation ────────────────────────────────────────


def step_h3(embeddings, metadata, train_ids, test_ids, eval_ids, device):
    print(f"\n{'='*60}")
    print("H3: Final evaluation + statistical comparison")
    print(f"{'='*60}")

    if not HPO_STUDY_PATH.exists():
        print("  HPO study not found. Run H1 first.")
        return

    with open(HPO_STUDY_PATH) as f:
        study_data = json.load(f)

    all_results = load_results()
    embed_dim = next(iter(embeddings.values())).shape[-1]

    # Load original split for classification
    cls_split_path = SPLIT_DIR / "esm2_650m_5k_cls_split.json"
    with open(cls_split_path) as f:
        cls_data = json.load(f)
    cls_train_ids = cls_data["cls_train_ids"]
    cls_test_ids = cls_data["cls_test_ids"]

    top3 = study_data.get("top3_trials", [])

    # Evaluate all retrained models
    for rank, trial_info in enumerate(top3):
        trial_num = trial_info["number"]

        for seed in SEEDS:
            name = f"hpo_top{rank+1}_trial{trial_num}_s{seed}"
            eval_name = f"eval_{name}"

            # Check if already evaluated
            if is_done(all_results, eval_name):
                print(f"  {eval_name} already done, skipping")
                continue

            ckpt_dir = CHECKPOINTS_DIR / f"hpo_{name}"
            if not (ckpt_dir / "best_model.pt").exists():
                print(f"  Checkpoint not found for {name}, skipping")
                continue

            print(f"\n  Evaluating {name}...")
            model = ChannelCompressor(
                input_dim=embed_dim, latent_dim=D_PRIME,
                dropout=trial_info["params"].get("dropout", 0.1),
                use_residual=True,
            )
            model.load_state_dict(
                torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True)
            )
            model = model.to(device)

            # Retrieval (with per-query scores for bootstrap)
            ret_results = evaluate_retrieval(
                model, embeddings, metadata,
                label_key="family", k_values=[1, 3, 5],
                device=device, query_ids=eval_ids, database_ids=test_ids,
                pooling_strategy="mean", return_per_query=True,
            )

            # Classification
            cls_results = evaluate_linear_probe(
                model, embeddings, metadata,
                label_key="family", device=device,
                train_ids=cls_train_ids, test_ids=cls_test_ids,
            )

            # Reconstruction
            test_set = set(test_ids)
            val_emb = {k: v for k, v in embeddings.items() if k in test_set}
            recon = evaluate_reconstruction(model, val_emb, device)

            result = {
                "name": eval_name,
                "hpo_trial": trial_num,
                "hpo_rank": rank + 1,
                "seed": seed,
                "params": trial_info["params"],
                "retrieval": {
                    "precision@1": ret_results.get("precision@1", 0),
                    "precision@3": ret_results.get("precision@3", 0),
                    "precision@5": ret_results.get("precision@5", 0),
                    "mrr": ret_results.get("mrr", 0),
                    "map": ret_results.get("map", 0),
                },
                "classification": cls_results,
                "reconstruction": recon,
            }

            # Save per-query scores for bootstrap test
            per_query = ret_results.get("per_query", {})
            if per_query:
                pq_path = ckpt_dir / "per_query_scores.json"
                with open(pq_path, "w") as f:
                    json.dump(per_query, f)

            all_results.append(result)
            save_results(all_results)

            ret1 = ret_results.get("precision@1", 0)
            mrr = ret_results.get("mrr", 0)
            cls_acc = cls_results.get("accuracy_mean", 0)
            print(f"  >> {eval_name}: Ret@1={ret1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    # ── Summary: Compare HPO best vs baseline ────────────────────

    print(f"\n{'='*60}")
    print("Summary: HPO vs Baseline")
    print(f"{'='*60}")

    # Collect per-config aggregates
    config_results = {}
    for r in all_results:
        if not r["name"].startswith("eval_hpo_"):
            continue
        config_key = f"top{r['hpo_rank']}_trial{r['hpo_trial']}"
        if config_key not in config_results:
            config_results[config_key] = {"ret1": [], "mrr": [], "params": r["params"]}
        config_results[config_key]["ret1"].append(r["retrieval"]["precision@1"])
        config_results[config_key]["mrr"].append(r["retrieval"]["mrr"])

    # Baseline: ProtT5 contrastive d256 (3-seed results from experiment 13)
    robust_results_path = DATA_DIR / "benchmarks" / "robust_validation_results.json"
    baseline_ret1 = []
    if robust_results_path.exists():
        with open(robust_results_path) as f:
            robust_data = json.load(f)
        for r in robust_data:
            name_r = r.get("name", "")
            if name_r.startswith("prot_t5_contrastive_d256_s") and "retrieval" in r:
                ret1_val = r["retrieval"].get("precision@1", r["retrieval"].get("ret@1"))
                if ret1_val is not None:
                    baseline_ret1.append(ret1_val)
    if not baseline_ret1:
        # Fallback to known values if results file unavailable or has different format
        baseline_ret1 = [0.808, 0.785, 0.793]  # seeds 42, 123, 456
        print("  (Using hardcoded baseline values — results JSON not found or format mismatch)")
    baseline_mean = np.mean(baseline_ret1)
    baseline_std = np.std(baseline_ret1)

    print(f"\n  Baseline: Ret@1 = {baseline_mean:.3f} +/- {baseline_std:.3f}")

    best_config = None
    best_mean = 0

    for config, data in config_results.items():
        mean_ret1 = np.mean(data["ret1"])
        std_ret1 = np.std(data["ret1"])
        mean_mrr = np.mean(data["mrr"])
        print(f"  {config}: Ret@1 = {mean_ret1:.3f} +/- {std_ret1:.3f}, "
              f"MRR = {mean_mrr:.3f}, params = {data['params']}")

        if mean_ret1 > best_mean:
            best_mean = mean_ret1
            best_config = config

    if best_config and config_results[best_config]["ret1"]:
        hpo_ret1 = config_results[best_config]["ret1"]
        d = cohens_d(hpo_ret1, baseline_ret1)

        # Multi-seed permutation test
        perm_result = multi_seed_permutation_test(hpo_ret1, baseline_ret1)

        print(f"\n  Best HPO config: {best_config}")
        print(f"  HPO Ret@1: {np.mean(hpo_ret1):.3f} +/- {np.std(hpo_ret1):.3f}")
        print(f"  Baseline:  {baseline_mean:.3f} +/- {baseline_std:.3f}")
        print(f"  Cohen's d: {d:.3f}")
        print(f"  Permutation test p-value: {perm_result['p_value']:.4f} "
              f"({'exhaustive' if perm_result['exhaustive'] else 'random'})")

        # Interpretation
        diff = np.mean(hpo_ret1) - baseline_mean
        if diff > 0.02:
            print(f"  Interpretation: HPO found improvement (+{diff:.3f} Ret@1)")
        elif diff > -0.01:
            print(f"  Interpretation: Near-baseline, current hyperparameters are near-optimal")
        else:
            print(f"  Interpretation: Null result, current config at Pareto frontier")

    # Paired bootstrap (if per-query scores available for baseline)
    baseline_ckpt = CHECKPOINTS_DIR / BASELINE_CHECKPOINT
    baseline_pq_path = baseline_ckpt / "per_query_scores.json"

    if best_config:
        best_trial_info = config_results[best_config]
        # Find corresponding eval with seed=42 for bootstrap
        best_rank = best_config.split("_")[0].replace("top", "")
        best_trial_num = best_config.split("trial")[1]
        hpo_pq_name = f"hpo_top{best_rank}_trial{best_trial_num}_s42"
        hpo_pq_path = CHECKPOINTS_DIR / f"hpo_{hpo_pq_name}" / "per_query_scores.json"

        if baseline_pq_path.exists() and hpo_pq_path.exists():
            with open(baseline_pq_path) as f:
                baseline_pq = json.load(f)
            with open(hpo_pq_path) as f:
                hpo_pq = json.load(f)

            if "precision@1" in baseline_pq and "precision@1" in hpo_pq:
                boot = paired_bootstrap_test(
                    hpo_pq["precision@1"], baseline_pq["precision@1"],
                )
                print(f"\n  Paired bootstrap (n={boot['n_queries']} queries):")
                print(f"    HPO mean: {boot['mean_a']:.3f}, Baseline mean: {boot['mean_b']:.3f}")
                print(f"    Diff: {boot['mean_diff']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
                print(f"    p-value: {boot['p_value']:.4f}")
        elif not baseline_pq_path.exists():
            # Generate baseline per-query scores
            print("\n  Generating baseline per-query scores for bootstrap test...")
            baseline_model = load_checkpoint(BASELINE_CHECKPOINT, embed_dim, device)
            baseline_ret = evaluate_retrieval(
                baseline_model, embeddings, metadata,
                label_key="family", k_values=[1],
                device=device, query_ids=eval_ids, database_ids=test_ids,
                pooling_strategy="mean", return_per_query=True,
            )
            per_query = baseline_ret.get("per_query", {})
            if per_query:
                baseline_pq_path.parent.mkdir(parents=True, exist_ok=True)
                with open(baseline_pq_path, "w") as f:
                    json.dump(per_query, f)
                print(f"  Saved baseline per-query scores to {baseline_pq_path}")

                if hpo_pq_path.exists():
                    with open(hpo_pq_path) as f:
                        hpo_pq = json.load(f)
                    if "precision@1" in per_query and "precision@1" in hpo_pq:
                        boot = paired_bootstrap_test(
                            hpo_pq["precision@1"], per_query["precision@1"],
                        )
                        print(f"\n  Paired bootstrap (n={boot['n_queries']} queries):")
                        print(f"    HPO mean: {boot['mean_a']:.3f}, Baseline mean: {boot['mean_b']:.3f}")
                        print(f"    Diff: {boot['mean_diff']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
                        print(f"    p-value: {boot['p_value']:.4f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="HPO for contrastive fine-tuning")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step: H0, H1, H2, H3")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load data
    embeddings, metadata = load_data()
    train_ids, test_ids, eval_ids = get_original_split()

    steps = [args.step] if args.step else ["H0", "H1", "H2", "H3"]

    for step in steps:
        step = step.upper()

        if step == "H0":
            hpo_train_ids, hpo_val_ids = step_h0(metadata, train_ids)

        elif step == "H1":
            # Load HPO split
            if not HPO_SPLIT_PATH.exists():
                print("  HPO split not found. Running H0 first...")
                hpo_train_ids, hpo_val_ids = step_h0(metadata, train_ids)
            else:
                with open(HPO_SPLIT_PATH) as f:
                    hpo_data = json.load(f)
                hpo_train_ids = hpo_data["hpo_train_ids"]
                hpo_val_ids = hpo_data["hpo_val_ids"]

            step_h1(embeddings, metadata, hpo_train_ids, hpo_val_ids, device)

        elif step == "H2":
            step_h2(embeddings, metadata, train_ids, device)

        elif step == "H3":
            step_h3(embeddings, metadata, train_ids, test_ids, eval_ids, device)

        else:
            print(f"Unknown step: {step}")
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
