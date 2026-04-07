#!/usr/bin/env python3
"""Experiment 50: Sequence to One Embedding — predict binary OE from FASTA.

Can a lightweight CNN predict the 896 binary bits that OneEmbeddingCodec
produces from ProtT5, using only the amino acid sequence as input?

Progressive stages:
  Stage 1: Dilated CNN (~2M params, 5 layers, 128 hidden)
  Stage 2: Deeper ResNet (~10M params, 10 layers, 256 hidden)
  Stage 3: Hybrid CNN + Attention (if CNN plateaus)

Usage:
    # Smoke test (tiny data, 2 epochs)
    uv run python experiments/50_sequence_to_oe.py --smoke-test

    # Stage 1 baseline
    uv run python experiments/50_sequence_to_oe.py --stage 1

    # Stage 2 deeper
    uv run python experiments/50_sequence_to_oe.py --stage 2

    # Stage 1 + downstream evaluation
    uv run python experiments/50_sequence_to_oe.py --stage 1 --eval
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

from src.extraction.data_loader import read_fasta
from src.one_embedding.seq2oe import (
    Seq2OE_CNN, Seq2OEDataset, prepare_binary_targets,
)
from src.utils.device import get_device

DATA = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "exp50"


# ═══════════════════════════════════════════════════════════════════════
# Target preparation (train-only fit to avoid centering-stat leakage)
# ═══════════════════════════════════════════════════════════════════════

def prepare_binary_targets_train_fit(
    train_embeddings: dict,
    all_embeddings: dict,
    d_out: int = 896,
    seed: int = 42,
) -> dict:
    """Fit OneEmbeddingCodec on train embeddings only, then encode all folds.

    This prevents centering-stat leakage from val/test into the binary targets.
    """
    from src.one_embedding.codec_v2 import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(train_embeddings)

    targets = {}
    for pid, raw in all_embeddings.items():
        projected = codec._preprocess(raw)
        means = projected.mean(axis=0)
        centered = projected - means[np.newaxis, :]
        bits = (centered > 0).astype(np.uint8)
        targets[pid] = bits

    return targets


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(
    dataset: str,
    split: str,
    seed: int,
    smoke_test: bool = False,
):
    """Load sequences + embeddings and build the requested split.

    Returns:
        (sequences, embeddings, train_ids, val_ids, test_ids, split_meta)
        where `split_meta` is a dict recording the split strategy for JSON I/O.
    """
    if dataset == "cath20":
        fasta_path = DATA / "external" / "cath20" / "cath20_labeled.fasta"
        h5_path = DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5"
    elif dataset == "medium5k":
        fasta_path = DATA / "proteins" / "medium_diverse_5k.fasta"
        h5_path = DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Loading sequences from {fasta_path}")
    if dataset == "cath20":
        from src.one_embedding.seq2oe_splits import parse_cath_fasta
        cath_meta = parse_cath_fasta(fasta_path)
        sequences = {pid: info["seq"] for pid, info in cath_meta.items()}
    else:
        sequences = read_fasta(fasta_path)
        cath_meta = None

    print(f"Loading ProtT5 embeddings from {h5_path}")
    embeddings = {}
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if smoke_test:
            keys = keys[:50]
        for pid in keys:
            embeddings[pid] = f[pid][:].astype(np.float32)

    # Keep only proteins with both seq and embedding
    common = sorted(set(sequences) & set(embeddings))
    sequences = {k: sequences[k] for k in common}
    embeddings = {k: embeddings[k] for k in common}
    print(f"  {len(common)} proteins with both sequence + embedding")

    # Build the split
    if split == "random":
        rng = np.random.RandomState(seed)
        ids = np.array(common)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids = list(ids[:n_train])
        val_ids = list(ids[n_train:n_train + n_val])
        test_ids = list(ids[n_train + n_val:])
        split_meta = {
            "strategy": "random",
            "dataset": dataset,
            "seed": seed,
            "fractions": [0.8, 0.1, 0.1],
        }
    elif split in ("h", "t"):
        from src.one_embedding.seq2oe_splits import cath_cluster_split
        assert cath_meta is not None, "CATH metadata required for h/t split"
        # Restrict cath_meta to proteins that also have embeddings
        filt_meta = {k: cath_meta[k] for k in common if k in cath_meta}
        level = split.upper()
        fractions = (0.8, 0.1, 0.1) if not smoke_test else (0.6, 0.2, 0.2)
        train_ids, val_ids, test_ids = cath_cluster_split(
            filt_meta, level=level, fractions=fractions, seed=seed,
        )
        split_meta = {
            "strategy": f"cath_{level}",
            "dataset": dataset,
            "seed": seed,
            "fractions": list(fractions),
            "level": level,
        }
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"  Split ({split}): "
          f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    return sequences, embeddings, train_ids, val_ids, test_ids, split_meta


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {"hidden": 128, "n_layers": 5,  "epochs": 50,  "lr": 1e-3, "batch_size": 16},
    2: {"hidden": 256, "n_layers": 10, "epochs": 100, "lr": 5e-4, "batch_size": 8},
}


def train_stage(
    stage: int,
    sequences: dict,
    targets: dict,
    train_ids: set,
    val_ids: set,
    device: torch.device,
    smoke_test: bool = False,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
):
    """Train a Seq2OE model for a given stage."""
    cfg = STAGE_CONFIGS[stage].copy()
    if smoke_test:
        cfg["epochs"] = 2
        cfg["batch_size"] = 4

    # Seed all RNGs that affect training reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"STAGE {stage}: hidden={cfg['hidden']}, layers={cfg['n_layers']}")
    print(f"{'='*60}")

    model = Seq2OE_CNN(d_out=896, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model = model.to(device)

    # Datasets
    train_seqs = {k: sequences[k] for k in train_ids if k in targets}
    train_tgts = {k: targets[k] for k in train_ids if k in targets}
    val_seqs = {k: sequences[k] for k in val_ids if k in targets}
    val_tgts = {k: targets[k] for k in val_ids if k in targets}

    max_len = 512
    train_ds = Seq2OEDataset(train_seqs, train_tgts, max_len=max_len)
    val_ds = Seq2OEDataset(val_seqs, val_tgts, max_len=max_len)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    print(f"  Train: {len(train_ds)} proteins, Val: {len(val_ds)} proteins")
    print(f"  Batch size: {cfg['batch_size']}, Max len: {max_len}")

    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_bit_acc": [], "val_bit_acc": []}

    if checkpoint_dir is None:
        checkpoint_dir = RESULTS_DIR / f"stage{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_losses, train_correct, train_total = [], 0, 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)

            logits = model(ids, mask)  # (B, L, 896)

            # Masked BCE loss: average over valid positions and bits
            loss_per_bit = criterion(logits, tgt)  # (B, L, 896)
            mask_3d = mask.unsqueeze(-1)  # (B, L, 1)
            n_valid = mask_3d.sum() * 896
            loss = (loss_per_bit * mask_3d).sum() / n_valid.clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            # Sanitize gradients before clipping (project convention)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Bit accuracy
            with torch.no_grad():
                preds = (logits > 0).float()
                correct = ((preds == tgt) * mask_3d).sum().item()
                total = n_valid.item()
                train_correct += correct
                train_total += total

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses, val_correct, val_total = [], 0, 0

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                tgt = batch["target"].to(device)
                mask = batch["mask"].to(device)

                logits = model(ids, mask)
                loss_per_bit = criterion(logits, tgt)
                mask_3d = mask.unsqueeze(-1)
                n_valid = mask_3d.sum() * 896
                loss = (loss_per_bit * mask_3d).sum() / n_valid.clamp(min=1)
                val_losses.append(loss.item())

                preds = (logits > 0).float()
                correct = ((preds == tgt) * mask_3d).sum().item()
                val_correct += correct
                val_total += n_valid.item()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_bit_acc"].append(float(train_acc))
        history["val_bit_acc"].append(float(val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"{elapsed:.0f}s")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Reload best
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt",
                    map_location=device, weights_only=True)
    )

    # Save history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_val_acc = history["val_bit_acc"][best_epoch - 1]
    elapsed = time.time() - start
    print(f"\n  Stage {stage} complete: best val acc={best_val_acc:.4f} "
          f"@ epoch {best_epoch} ({elapsed:.0f}s)")

    return model, history, elapsed


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_bit_accuracy(model, sequences, targets, test_ids, device):
    """Evaluate bit accuracy on test set with per-protein breakdown."""
    model.eval()
    test_seqs = {k: sequences[k] for k in test_ids if k in targets}
    test_tgts = {k: targets[k] for k in test_ids if k in targets}

    ds = Seq2OEDataset(test_seqs, test_tgts, max_len=512)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    all_correct, all_total = 0, 0
    per_protein_acc = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"]

            logits = model(ids, mask)
            preds = (logits > 0).float()
            mask_3d = mask.unsqueeze(-1)

            correct = ((preds == tgt) * mask_3d).sum().item()
            total = (mask_3d.sum() * 896).item()
            all_correct += correct
            all_total += total

            # Per-protein accuracy
            for i in range(len(lengths)):
                L = lengths[i].item()
                p_correct = (preds[i, :L] == tgt[i, :L]).float().mean().item()
                per_protein_acc.append(p_correct)

    overall = all_correct / max(all_total, 1)
    per_prot = np.array(per_protein_acc)

    print(f"\n{'='*60}")
    print(f"TEST SET BIT ACCURACY ({len(per_protein_acc)} proteins)")
    print(f"{'='*60}")
    print(f"  Overall:    {overall:.4f} ({overall*100:.1f}%)")
    print(f"  Per-protein: {per_prot.mean():.4f} +/- {per_prot.std():.4f}")
    print(f"  Min/Max:    {per_prot.min():.4f} / {per_prot.max():.4f}")
    print(f"  Random baseline: 0.5000 (50.0%)")
    print(f"  Improvement over random: {(overall - 0.5) * 200:.1f}pp")

    return {
        "overall_bit_acc": float(overall),
        "per_protein_mean": float(per_prot.mean()),
        "per_protein_std": float(per_prot.std()),
        "per_protein_min": float(per_prot.min()),
        "per_protein_max": float(per_prot.max()),
        "n_test": len(per_protein_acc),
    }


def analyze_per_dimension(model, sequences, targets, test_ids, device):
    """Analyze which of the 896 binary dimensions are easiest/hardest to predict."""
    model.eval()
    test_seqs = {k: sequences[k] for k in test_ids if k in targets}
    test_tgts = {k: targets[k] for k in test_ids if k in targets}

    ds = Seq2OEDataset(test_seqs, test_tgts, max_len=512)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    dim_correct = np.zeros(896)
    dim_total = np.zeros(896)

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"]

            logits = model(ids, mask)
            preds = (logits > 0).float()

            for i in range(len(lengths)):
                L = lengths[i].item()
                correct = (preds[i, :L] == tgt[i, :L]).float().cpu().numpy()  # (L, 896)
                dim_correct += correct.sum(axis=0)
                dim_total += L

    dim_acc = dim_correct / np.maximum(dim_total, 1)

    print(f"\n  Per-dimension accuracy:")
    print(f"    Mean: {dim_acc.mean():.4f}")
    print(f"    Std:  {dim_acc.std():.4f}")
    print(f"    Best 10 dims: {np.sort(dim_acc)[-10:][::-1]}")
    print(f"    Worst 10 dims: {np.sort(dim_acc)[:10]}")
    n_above_55 = (dim_acc > 0.55).sum()
    n_above_60 = (dim_acc > 0.60).sum()
    print(f"    Dims >55% acc: {n_above_55}/896")
    print(f"    Dims >60% acc: {n_above_60}/896")

    return {"dim_accuracies": dim_acc.tolist()}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 50: Sequence to One Embedding")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Model stage (1=baseline CNN, 2=deeper)")
    parser.add_argument("--dataset", choices=["cath20", "medium5k"],
                        default="cath20",
                        help="Which embeddings + sequences to use")
    parser.add_argument("--split", choices=["h", "t", "random"],
                        default="h",
                        help="Split strategy: h/t = CATH cluster level, "
                             "random = shuffled 80/10/10")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for split and training RNGs")
    parser.add_argument("--output-root", type=str,
                        default="results/exp50_rigorous",
                        help="Root dir for all outputs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with tiny data + 2 epochs")
    parser.add_argument("--eval", action="store_true",
                        help="Run downstream evaluation after training")
    args = parser.parse_args()

    # Validate dataset/split combinations
    if args.dataset == "medium5k" and args.split in ("h", "t"):
        parser.error(
            "--dataset medium5k does not have CATH labels; "
            "use --split random or --dataset cath20"
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: dataset={args.dataset} split={args.split} "
          f"seed={args.seed} stage={args.stage}")

    # Check system load
    load1, load5, _ = os.getloadavg()
    print(f"System load: {load1:.1f} (1m), {load5:.1f} (5m)")
    if load1 > 10:
        print("WARNING: System load >10, consider waiting before training")

    # Load data + build split
    sequences, embeddings, train_ids, val_ids, test_ids, split_meta = load_data(
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    # Prepare binary targets — codec centering fit on TRAIN EMBEDDINGS ONLY
    print("\nPreparing binary targets (center + RP896 + sign, train-only fit)...")
    t0 = time.time()
    train_embeddings = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    all_targets = prepare_binary_targets_train_fit(
        train_embeddings=train_embeddings,
        all_embeddings=embeddings,
        d_out=896,
        seed=args.seed,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Sanity: class balance on a sample
    sample_pid = next(iter(all_targets))
    balance = all_targets[sample_pid].mean()
    print(f"  Class balance (sample): {balance:.3f} (ideal=0.5)")

    # Output dir per (split, stage, seed)
    output_root = Path(args.output_root)
    run_dir = output_root / f"{args.split}_split" / f"stage{args.stage}" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist split for reproducibility
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{args.split}_seed{args.seed}.json"
    if not split_path.exists():
        from src.one_embedding.seq2oe_splits import save_split
        save_split(split_path, train_ids, val_ids, test_ids, split_meta)
        print(f"  Saved split to {split_path}")

    # Train
    model, history, train_seconds = train_stage(
        args.stage, sequences, all_targets,
        set(train_ids), set(val_ids), device, args.smoke_test,
        seed=args.seed,
        checkpoint_dir=run_dir,
    )

    # Test evaluation
    results = evaluate_bit_accuracy(
        model, sequences, all_targets, set(test_ids), device
    )

    # Per-dimension analysis
    dim_results = analyze_per_dimension(
        model, sequences, all_targets, set(test_ids), device
    )
    results["dim_accuracies"] = dim_results["dim_accuracies"]
    results["config"] = {
        "stage": args.stage,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
    }
    results["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    results["best_val_loss"] = float(min(history["val_loss"]))
    results["best_val_bit_acc"] = float(
        history["val_bit_acc"][results["best_epoch"] - 1]
    )
    results["train_seconds"] = float(train_seconds)

    # Save results
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.eval:
        print("\n[TODO] Downstream evaluation (SS3, disorder, retrieval)")


if __name__ == "__main__":
    main()
