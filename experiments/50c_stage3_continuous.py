#!/usr/bin/env python3
"""Exp 50 Stage 3: continuous regression with CATH20 + DeepLoc training pool.

Replaces Stage 2's per-bit BCE loss with cosine + MSE regression on the
896d projected ProtT5 vector. Adds DeepLoc (~13K proteins after leakage
filter) as auxiliary training signal. Keeps the same Stage 2 H-split test
set for direct bit-accuracy comparability.

Usage:
    PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py --seed 42
    PYTHONUNBUFFERED=1 uv run python experiments/50c_stage3_continuous.py --seed 42 --smoke-test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe import Seq2OE_CNN, Seq2OEDataset
from src.one_embedding.seq2oe_continuous import (
    prepare_continuous_targets,
    cosine_distance_loss,
    mse_loss,
    evaluate_continuous,
)
from src.one_embedding.seq2oe_splits import (
    parse_cath_fasta,
    cath_cluster_split,
    save_split,
)
from src.utils.device import get_device

DATA = ROOT / "data"
DEEPLOC_FASTA = (
    ROOT / "tools" / "reference" / "LightAttention" /
    "data_files" / "deeploc_complete_dataset.fasta"
)
CATH_FASTA = DATA / "external" / "cath20" / "cath20_labeled.fasta"
CATH_H5 = DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5"
DEEPLOC_H5 = DATA / "residue_embeddings" / "prot_t5_xl_deeploc.h5"

# Training config (matches Stage 2, with cosine-sim early stop)
CONFIG = {
    "hidden": 256,
    "n_layers": 10,
    "epochs": 100,
    "lr": 5e-4,
    "batch_size": 8,
    "weight_decay": 1e-4,
    "patience": 15,
    "d_out": 896,
    "max_len": 512,
    "lambda_cos": 1.0,
    "lambda_mse": 0.1,
}


def ensure_leakage_filter(seed: int, output_root: Path) -> set[str]:
    """Ensure the DeepLoc leakage filter exists for `seed`; return the
    excluded-ID set."""
    filter_file = (
        output_root / "leakage_filter" /
        f"deeploc_leakage_excluded_seed{seed}.json"
    )
    if not filter_file.exists():
        print(f"Leakage filter not found, running now for seed {seed}...")
        cmd = [
            "uv", "run", "python",
            "experiments/50_stage3_leakage_filter.py",
            "--seed", str(seed),
            "--output-root", str(output_root),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        res = subprocess.run(cmd, env=env, cwd=ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"Leakage filter failed with exit {res.returncode}")
    with open(filter_file) as f:
        report = json.load(f)
    return set(report["excluded_deeploc_ids"])


def load_cath20_and_split(seed: int):
    """Load CATH20 sequences, embeddings, and build the H-split at `seed`."""
    print(f"Loading CATH20 FASTA from {CATH_FASTA}...")
    cath_meta = parse_cath_fasta(CATH_FASTA)
    sequences = {pid: info["seq"] for pid, info in cath_meta.items()}

    print(f"Loading CATH20 ProtT5 embeddings from {CATH_H5}...")
    embeddings = {}
    with h5py.File(CATH_H5, "r") as f:
        for pid in f.keys():
            embeddings[pid] = f[pid][:].astype(np.float32)

    common = sorted(set(sequences) & set(embeddings))
    sequences = {k: sequences[k] for k in common}
    embeddings = {k: embeddings[k] for k in common}
    print(f"  {len(common)} CATH proteins with both sequence + embedding")

    filt_meta = {k: cath_meta[k] for k in common}
    train_ids, val_ids, test_ids = cath_cluster_split(
        filt_meta, level="H", fractions=(0.8, 0.1, 0.1), seed=seed,
    )
    print(f"  H-split: {len(train_ids)} train, {len(val_ids)} val, "
          f"{len(test_ids)} test")
    return sequences, embeddings, train_ids, val_ids, test_ids


def load_deeploc(excluded_ids: set[str], smoke_test: bool = False):
    """Load DeepLoc sequences + embeddings, filtering out leakage exclusions."""
    print(f"Loading DeepLoc FASTA from {DEEPLOC_FASTA}...")
    all_seqs = read_fasta(DEEPLOC_FASTA)

    print(f"Loading DeepLoc embeddings from {DEEPLOC_H5}...")
    embeddings = {}
    with h5py.File(DEEPLOC_H5, "r") as f:
        keys = list(f.keys())
        if smoke_test:
            keys = keys[:50]
        for pid in keys:
            embeddings[pid] = f[pid][:].astype(np.float32)

    # Keep only proteins with sequence + embedding + not excluded
    usable = (
        (set(all_seqs) & set(embeddings)) - excluded_ids
    )
    sequences = {k: all_seqs[k] for k in usable}
    embeddings = {k: embeddings[k] for k in usable}
    print(f"  DeepLoc: {len(all_seqs)} total, "
          f"{len(sequences)} usable after leakage filter + seq/emb overlap")
    return sequences, embeddings


def train_continuous(
    sequences_all: dict,
    targets_all: dict,
    train_ids: set,
    val_ids: set,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
    smoke_test: bool = False,
) -> tuple:
    """Train a Seq2OE_CNN with cosine + MSE loss. Returns (model, history,
    elapsed_seconds)."""
    cfg = CONFIG.copy()
    if smoke_test:
        cfg["epochs"] = 2
        cfg["batch_size"] = 4

    # Seed all RNGs BEFORE model instantiation
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"STAGE 3 training (seed={seed})")
    print(f"  hidden={cfg['hidden']}, layers={cfg['n_layers']}")
    print(f"  lambda_cos={cfg['lambda_cos']}, lambda_mse={cfg['lambda_mse']}")
    print(f"{'='*60}")

    model = Seq2OE_CNN(d_out=cfg["d_out"], hidden=cfg["hidden"],
                       n_layers=cfg["n_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model = model.to(device)

    # Build train dataset (all train proteins: CATH20 H-train + DeepLoc filtered)
    train_seqs = {k: sequences_all[k] for k in train_ids if k in targets_all}
    train_tgts = {k: targets_all[k] for k in train_ids if k in targets_all}
    val_seqs = {k: sequences_all[k] for k in val_ids if k in targets_all}
    val_tgts = {k: targets_all[k] for k in val_ids if k in targets_all}

    train_ds = Seq2OEDataset(train_seqs, train_tgts, max_len=cfg["max_len"])
    val_ds = Seq2OEDataset(val_seqs, val_tgts, max_len=cfg["max_len"])

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=False, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    print(f"  Train: {len(train_ds)} proteins, Val: {len(val_ds)} proteins")
    print(f"  Batch size: {cfg['batch_size']}, Max len: {cfg['max_len']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_val_cos_dist = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_cosine_sim": [], "val_mse": [],
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)

            pred = model(ids, mask)
            l_cos = cosine_distance_loss(pred, tgt, mask)
            l_mse = mse_loss(pred, tgt, mask)
            loss = cfg["lambda_cos"] * l_cos + cfg["lambda_mse"] * l_mse

            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        val_cos_sum = 0.0
        val_cos_count = 0
        val_mse_sum = 0.0
        val_mse_count = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                tgt = batch["target"].to(device)
                mask = batch["mask"].to(device)
                pred = model(ids, mask)
                l_cos = cosine_distance_loss(pred, tgt, mask)
                l_mse = mse_loss(pred, tgt, mask)
                loss = cfg["lambda_cos"] * l_cos + cfg["lambda_mse"] * l_mse
                val_losses.append(loss.item())
                # Raw cosine sim / MSE for reporting
                pn = pred.norm(dim=-1).clamp_min(1e-8)
                tn = tgt.norm(dim=-1).clamp_min(1e-8)
                cos = (pred * tgt).sum(dim=-1) / (pn * tn)
                val_cos_sum += (cos * mask).sum().item()
                val_cos_count += mask.sum().item()
                sq = (pred - tgt) ** 2
                val_mse_sum += (sq * mask.unsqueeze(-1)).sum().item()
                val_mse_count += mask.sum().item() * cfg["d_out"]

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_cosine_sim = val_cos_sum / max(val_cos_count, 1.0)
        val_mse_ = val_mse_sum / max(val_mse_count, 1.0)
        val_cos_dist = 1.0 - val_cosine_sim

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cosine_sim"].append(val_cosine_sim)
        history["val_mse"].append(val_mse_)

        if val_cos_dist < best_val_cos_dist:
            best_val_cos_dist = val_cos_dist
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} "
                  f"val_cos_sim={val_cosine_sim:.4f} "
                  f"val_mse={val_mse_:.4f} | "
                  f"{elapsed:.0f}s")

        if epochs_no_improve >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch} "
                  f"(patience={cfg['patience']})")
            break

    # Reload best
    model.load_state_dict(torch.load(
        checkpoint_dir / "best_model.pt", map_location=device, weights_only=True,
    ))
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Stage 3 complete: best val cosine_dist={best_val_cos_dist:.4f} "
          f"@ epoch {best_epoch} ({elapsed:.0f}s)")
    return model, history, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=str, default="results/exp50_stage3"
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: stage=3 split=h seed={args.seed}")

    load1, load5, _ = os.getloadavg()
    print(f"System load: {load1:.1f} (1m), {load5:.1f} (5m)")
    if load1 > 10:
        print("WARNING: System load >10, consider waiting before training")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Step 1: ensure leakage filter exists
    excluded = ensure_leakage_filter(args.seed, output_root)
    print(f"Leakage filter: {len(excluded)} DeepLoc proteins excluded")

    # Step 2: load CATH20 + build H-split
    cath_seqs, cath_embs, train_ids, val_ids, test_ids = load_cath20_and_split(
        args.seed
    )

    # Step 3: load DeepLoc (filtered)
    deeploc_seqs, deeploc_embs = load_deeploc(excluded, args.smoke_test)

    # Step 4: merge into combined sequences/embeddings dicts
    # (ID spaces are disjoint: CATH uses '12asA00', DeepLoc uses UniProt IDs)
    sequences_all = {**cath_seqs, **deeploc_seqs}
    embeddings_all = {**cath_embs, **deeploc_embs}
    print(f"Combined pool: {len(sequences_all)} total proteins "
          f"(CATH={len(cath_seqs)}, DeepLoc={len(deeploc_seqs)})")

    # Step 5: expand train pool to include DeepLoc filtered
    combined_train_ids = list(set(train_ids) | set(deeploc_seqs.keys()))
    print(f"Combined train pool: {len(combined_train_ids)} proteins "
          f"(CATH train {len(train_ids)} + DeepLoc {len(deeploc_seqs)})")

    # Step 6: fit codec on CATH20 train embeddings only, encode everything
    print(f"\nPreparing continuous targets (train-only codec fit)...")
    t0 = time.time()
    cath_train_embeddings = {pid: cath_embs[pid] for pid in train_ids}
    targets_all = prepare_continuous_targets(
        train_embeddings=cath_train_embeddings,
        all_embeddings=embeddings_all,
        d_out=CONFIG["d_out"],
        seed=args.seed,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    # Sanity on the target dtype
    sample_pid = next(iter(targets_all))
    print(f"  Sample target shape: {targets_all[sample_pid].shape}, "
          f"dtype: {targets_all[sample_pid].dtype}")

    # Step 7: persist the CATH H-split for reproducibility
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"h_seed{args.seed}.json"
    if not split_path.exists():
        save_split(
            split_path, train_ids, val_ids, test_ids,
            {"strategy": "cath_H", "dataset": "cath20", "seed": args.seed,
             "fractions": [0.8, 0.1, 0.1], "level": "H",
             "stage3_deeploc_augmented": True,
             "n_deeploc_added": len(deeploc_seqs)},
        )
        print(f"  Saved split to {split_path}")

    # Step 8: output dir for this run
    run_dir = output_root / "h_split" / "stage3" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 9: train
    model, history, train_seconds = train_continuous(
        sequences_all=sequences_all,
        targets_all=targets_all,
        train_ids=set(combined_train_ids),
        val_ids=set(val_ids),
        device=device,
        seed=args.seed,
        checkpoint_dir=run_dir,
        smoke_test=args.smoke_test,
    )

    # Step 10: evaluate on CATH H-test ONLY (not DeepLoc)
    print(f"\nEvaluating on CATH H-test ({len(test_ids)} proteins)...")
    metrics = evaluate_continuous(
        model=model,
        sequences=sequences_all,
        targets=targets_all,
        ids=set(test_ids),
        device=device,
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
    )
    print(f"  cosine_sim:  {metrics['cosine_sim']:.4f}")
    print(f"  mse:         {metrics['mse']:.4f}")
    print(f"  bit_accuracy: {metrics['bit_accuracy']:.4f}")
    print(f"  dims > 60%:  "
          f"{sum(1 for d in metrics['dim_accuracies'] if d > 0.60)}/"
          f"{len(metrics['dim_accuracies'])}")

    # Step 11: add config + write results.json
    metrics["config"] = {
        "stage": 3,
        "dataset": "cath20+deeploc",
        "split": "h",
        "seed": args.seed,
        "lambda_cos": CONFIG["lambda_cos"],
        "lambda_mse": CONFIG["lambda_mse"],
        "n_train_cath": len(train_ids),
        "n_train_deeploc": len(deeploc_seqs),
        "n_train_combined": len(combined_train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_deeploc_excluded": len(excluded),
    }
    metrics["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    metrics["best_val_loss"] = float(min(history["val_loss"]))
    metrics["best_val_cosine_sim"] = float(max(history["val_cosine_sim"]))
    metrics["train_seconds"] = float(train_seconds)

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
