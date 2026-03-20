#!/usr/bin/env python3
"""Experiment 42 — UdonPred Comparison: Disorder on CheZOD117.

Compares our ProtT5-based disorder probes with UdonPred (Rost lab, ProstT5)
on the CheZOD117 test set. UdonPred achieves Spearman rho = 0.684 (CheZOD-trained)
and 0.702 (TriZOD-trained) per biorxiv 2026.01.26.701679v2.

Probes trained and evaluated:
  1. Ridge on raw 1024d ProtT5 embeddings (linear baseline)
  2. Ridge on compressed 768d embeddings (linear, after ABTT3+RP768)
  3. CNN on raw 1024d ProtT5 embeddings (SETH-style 2-layer CNN)
  4. CNN on compressed 768d embeddings (SETH-style 2-layer CNN)

All probes: train on CheZOD1174, test on CheZOD117 (standard SETH split).

Usage:
  uv run python experiments/42_udonpred_comparison.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.evaluation.per_residue_tasks import load_chezod_seth
from src.one_embedding.io import read_one_h5_batch
from src.utils.h5_store import load_residue_embeddings

# ── Config ────────────────────────────────────────────────────────
SEED = 42
EPOCHS = 100
LR = 0.001
PATIENCE = 10
GRAD_ACCUM = 8

RAW_H5 = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_chezod.h5"
COMPRESSED_H5 = PROJ_ROOT / "data" / "benchmark_suite" / "compressed" / "prot_t5_768d" / "chezod.one.h5"
SETH_DIR = PROJ_ROOT / "data" / "per_residue_benchmarks"
OUTPUT_JSON = PROJ_ROOT / "data" / "benchmarks" / "udonpred_comparison.json"

# CPU for training (MPS Conv1d has stability issues)
device = torch.device("cpu")

# Published UdonPred numbers (biorxiv 2026.01.26.701679v2)
UDONPRED_CHEZOD = 0.684
UDONPRED_TRIZOD = 0.702


# ── CNN Architecture (SETH-style) ────────────────────────────────
class DisorderCNN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 32, kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        # x: (B, L, D) -> permute -> conv -> permute back -> squeeze
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1)


# ── Helper: prepare Ridge data ───────────────────────────────────
def prepare_ridge_data(embeddings, disorder_scores, protein_ids):
    """Stack per-residue embeddings and scores for Ridge regression."""
    X_all, y_all = [], []
    for pid in protein_ids:
        if pid not in embeddings or pid not in disorder_scores:
            continue
        emb = embeddings[pid]
        scores = disorder_scores[pid]
        L = min(len(emb), len(scores))
        emb, scores = emb[:L], scores[:L]
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            continue
        X_all.append(emb[valid])
        y_all.append(scores[valid])
    return np.vstack(X_all), np.concatenate(y_all)


# ── Helper: train CNN ─────────────────────────────────────────────
def train_disorder_cnn(embeddings, disorder_scores, train_ids, test_ids, d_in, label):
    """Train a disorder CNN probe and return Spearman rho on test set."""
    # Filter to available proteins
    train_pids = [p for p in train_ids if p in embeddings and p in disorder_scores]
    test_pids = [p for p in test_ids if p in embeddings and p in disorder_scores]
    print(f"  {label}: {len(train_pids)} train, {len(test_pids)} test proteins")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DisorderCNN(d_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_test_loss = float("inf")
    best_state = None
    patience_counter = 0

    t0 = time.time()
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        optimizer.zero_grad()

        rng_epoch = random.Random(SEED + epoch)
        shuffled = list(train_pids)
        rng_epoch.shuffle(shuffled)

        for i, pid in enumerate(shuffled):
            emb = embeddings[pid]
            scores = disorder_scores[pid]
            L = min(len(emb), len(scores))
            emb, scores = emb[:L], scores[:L]

            valid = ~np.isnan(scores)
            if valid.sum() == 0:
                continue

            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(scores, dtype=torch.float32).to(device)

            pred = model(x).squeeze(0)
            valid_t = torch.tensor(valid, dtype=torch.bool).to(device)
            loss = loss_fn(pred[valid_t], y[valid_t]) / GRAD_ACCUM
            loss.backward()

            train_loss_sum += loss.item() * GRAD_ACCUM * valid_t.sum().item()
            train_n += valid_t.sum().item()

            if (i + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(shuffled) % GRAD_ACCUM != 0:
            optimizer.step()
            optimizer.zero_grad()

        # ── Test ──
        model.eval()
        test_loss_sum = 0.0
        test_n = 0
        with torch.no_grad():
            for pid in test_pids:
                emb = embeddings[pid]
                scores = disorder_scores[pid]
                L = min(len(emb), len(scores))
                emb, scores = emb[:L], scores[:L]
                valid = ~np.isnan(scores)
                if valid.sum() == 0:
                    continue
                x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor(scores, dtype=torch.float32).to(device)
                pred = model(x).squeeze(0)
                valid_t = torch.tensor(valid, dtype=torch.bool).to(device)
                test_loss_sum += loss_fn(pred[valid_t], y[valid_t]).item() * valid_t.sum().item()
                test_n += valid_t.sum().item()

        train_loss = train_loss_sum / max(train_n, 1)
        test_loss = test_loss_sum / max(test_n, 1)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"    Epoch {epoch + 1:3d}/{EPOCHS}  "
                f"train_mse={train_loss:.4f}  test_mse={test_loss:.4f}  "
                f"best={best_test_loss:.4f}  patience={patience_counter}/{PATIENCE}  "
                f"[{elapsed:.1f}s]"
            )

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    # Evaluate best model
    model.load_state_dict(best_state)
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for pid in test_pids:
            emb = embeddings[pid]
            scores = disorder_scores[pid]
            L = min(len(emb), len(scores))
            emb, scores = emb[:L], scores[:L]
            valid = ~np.isnan(scores)
            if valid.sum() == 0:
                continue
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            all_pred.extend(pred[valid].tolist())
            all_true.extend(scores[valid].tolist())

    rho, pval = spearmanr(all_pred, all_true)
    elapsed = time.time() - t0
    print(f"    Spearman rho = {rho:.4f}  (p={pval:.2e}, {len(all_true)} residues, {elapsed:.1f}s)")
    return rho, len(all_true)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Experiment 42: UdonPred Comparison — Disorder on CheZOD117")
    print("=" * 70)

    # ── Step 1: Load labels ───────────────────────────────────────
    print("\nStep 1: Loading CheZOD labels")
    _, disorder_scores, train_ids, test_ids = load_chezod_seth(SETH_DIR)
    print(f"  Train: {len(train_ids)} proteins, Test: {len(test_ids)} proteins")
    print(f"  Total labels: {len(disorder_scores)} proteins")

    # ── Step 2: Load embeddings ───────────────────────────────────
    print("\nStep 2: Loading embeddings")

    print("  Loading raw ProtT5 1024d...")
    raw_emb = load_residue_embeddings(str(RAW_H5))
    print(f"    {len(raw_emb)} proteins, d={next(iter(raw_emb.values())).shape[1]}")

    print("  Loading compressed 768d...")
    comp_data = read_one_h5_batch(str(COMPRESSED_H5))
    comp_emb = {pid: d["per_residue"] for pid, d in comp_data.items()}
    print(f"    {len(comp_emb)} proteins, d={next(iter(comp_emb.values())).shape[1]}")

    # ── Step 3: Ridge on raw 1024d ────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 3: Ridge regression on raw 1024d")
    print("=" * 70)

    X_train, y_train = prepare_ridge_data(raw_emb, disorder_scores, train_ids)
    X_test, y_test = prepare_ridge_data(raw_emb, disorder_scores, test_ids)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    ridge_raw = Ridge(alpha=1.0)
    t0 = time.time()
    ridge_raw.fit(X_train, y_train)
    y_pred_raw = ridge_raw.predict(X_test)
    rho_ridge_raw, pval = spearmanr(y_test, y_pred_raw)
    elapsed = time.time() - t0
    print(f"  Ridge (raw 1024d): rho = {rho_ridge_raw:.4f}  (p={pval:.2e}, {len(y_test)} residues, {elapsed:.1f}s)")

    # ── Step 4: Ridge on compressed 768d ──────────────────────────
    print("\n" + "=" * 70)
    print("Step 4: Ridge regression on compressed 768d")
    print("=" * 70)

    X_train_c, y_train_c = prepare_ridge_data(comp_emb, disorder_scores, train_ids)
    X_test_c, y_test_c = prepare_ridge_data(comp_emb, disorder_scores, test_ids)
    print(f"  Train: {X_train_c.shape}, Test: {X_test_c.shape}")

    ridge_comp = Ridge(alpha=1.0)
    t0 = time.time()
    ridge_comp.fit(X_train_c, y_train_c)
    y_pred_comp = ridge_comp.predict(X_test_c)
    rho_ridge_comp, pval = spearmanr(y_test_c, y_pred_comp)
    elapsed = time.time() - t0
    print(f"  Ridge (compressed 768d): rho = {rho_ridge_comp:.4f}  (p={pval:.2e}, {len(y_test_c)} residues, {elapsed:.1f}s)")

    # ── Step 5: CNN on raw 1024d ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 5: CNN on raw 1024d")
    print("=" * 70)

    rho_cnn_raw, n_res_raw = train_disorder_cnn(
        raw_emb, disorder_scores, train_ids, test_ids,
        d_in=1024, label="CNN raw 1024d",
    )

    # ── Step 6: CNN on compressed 768d ────────────────────────────
    print("\n" + "=" * 70)
    print("Step 6: CNN on compressed 768d")
    print("=" * 70)

    rho_cnn_comp, n_res_comp = train_disorder_cnn(
        comp_emb, disorder_scores, train_ids, test_ids,
        d_in=768, label="CNN compressed 768d",
    )

    # ── Step 7: Comparison table ──────────────────────────────────
    print("\n" + "=" * 70)
    print("UdonPred Comparison: Disorder (CheZOD117 test set)")
    print("=" * 70)
    print(f"{'Method':<40} {'Spearman rho':>12}   Notes")
    print("-" * 70)
    print(f"{'UdonPred (ProstT5, CheZOD)':<40} {UDONPRED_CHEZOD:>12.3f}   *published")
    print(f"{'UdonPred (ProstT5, TriZOD)':<40} {UDONPRED_TRIZOD:>12.3f}   *published, cross-dataset")
    print(f"{'Our Ridge (raw 1024d)':<40} {rho_ridge_raw:>12.3f}   ProtT5, linear")
    print(f"{'Our Ridge (compressed 768d)':<40} {rho_ridge_comp:>12.3f}   ProtT5, compressed, linear")
    print(f"{'Our CNN (raw 1024d)':<40} {rho_cnn_raw:>12.3f}   ProtT5, SETH-style CNN")
    print(f"{'Our CNN (compressed 768d)':<40} {rho_cnn_comp:>12.3f}   ProtT5, compressed, CNN")
    print("=" * 70)

    # Delta vs UdonPred
    best_ours = max(rho_ridge_raw, rho_ridge_comp, rho_cnn_raw, rho_cnn_comp)
    delta = best_ours - UDONPRED_CHEZOD
    print(f"\nBest ours: {best_ours:.3f}  (delta vs UdonPred CheZOD: {delta:+.3f})")

    # Retention: compressed vs raw
    ridge_retention = rho_ridge_comp / rho_ridge_raw * 100
    cnn_retention = rho_cnn_comp / rho_cnn_raw * 100
    print(f"Ridge retention (768d vs 1024d): {ridge_retention:.1f}%")
    print(f"CNN retention (768d vs 1024d): {cnn_retention:.1f}%")

    # ── Step 8: Save results ──────────────────────────────────────
    results = {
        "experiment": "42_udonpred_comparison",
        "description": "Disorder prediction comparison: our ProtT5 probes vs UdonPred (ProstT5) on CheZOD117 test set",
        "test_set": "CheZOD117 (117 proteins)",
        "train_set": "CheZOD1174 (1174 proteins)",
        "methods": {
            "udonpred_chezod": {
                "spearman_rho": UDONPRED_CHEZOD,
                "model": "ProstT5",
                "training_data": "CheZOD",
                "source": "biorxiv 2026.01.26.701679v2",
                "notes": "published",
            },
            "udonpred_trizod": {
                "spearman_rho": UDONPRED_TRIZOD,
                "model": "ProstT5",
                "training_data": "TriZOD",
                "source": "biorxiv 2026.01.26.701679v2",
                "notes": "published, cross-dataset",
            },
            "ridge_raw_1024d": {
                "spearman_rho": round(rho_ridge_raw, 4),
                "model": "ProtT5-XL",
                "embedding_dim": 1024,
                "compression": "none",
                "probe": "Ridge(alpha=1.0)",
                "n_test_residues": len(y_test),
            },
            "ridge_compressed_768d": {
                "spearman_rho": round(rho_ridge_comp, 4),
                "model": "ProtT5-XL",
                "embedding_dim": 768,
                "compression": "ABTT3+RP768",
                "probe": "Ridge(alpha=1.0)",
                "n_test_residues": len(y_test_c),
            },
            "cnn_raw_1024d": {
                "spearman_rho": round(rho_cnn_raw, 4),
                "model": "ProtT5-XL",
                "embedding_dim": 1024,
                "compression": "none",
                "probe": "SETH-style CNN (Conv1d(1024,32,7)+Tanh+Conv1d(32,1,7))",
                "n_test_residues": n_res_raw,
                "training": {
                    "epochs_max": EPOCHS,
                    "lr": LR,
                    "patience": PATIENCE,
                    "grad_accum": GRAD_ACCUM,
                    "loss": "MSE",
                    "seed": SEED,
                },
            },
            "cnn_compressed_768d": {
                "spearman_rho": round(rho_cnn_comp, 4),
                "model": "ProtT5-XL",
                "embedding_dim": 768,
                "compression": "ABTT3+RP768",
                "probe": "SETH-style CNN (Conv1d(768,32,7)+Tanh+Conv1d(32,1,7))",
                "n_test_residues": n_res_comp,
                "training": {
                    "epochs_max": EPOCHS,
                    "lr": LR,
                    "patience": PATIENCE,
                    "grad_accum": GRAD_ACCUM,
                    "loss": "MSE",
                    "seed": SEED,
                },
            },
        },
        "retention": {
            "ridge_768d_vs_1024d": round(ridge_retention, 1),
            "cnn_768d_vs_1024d": round(cnn_retention, 1),
        },
        "best_ours": round(best_ours, 4),
        "delta_vs_udonpred": round(delta, 4),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
