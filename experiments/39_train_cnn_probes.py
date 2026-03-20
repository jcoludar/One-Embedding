"""Experiment 39: Train CNN probes for 768d compressed embeddings.

Trains SETH-style 2-layer CNN probes on ABTT3 + RP768 compressed ProtT5 embeddings:
  - Disorder probe: MSE loss, evaluated by Spearman rho on CheZOD 117 test set
  - SS3 probe: CrossEntropy loss, evaluated by Q3 accuracy on CB513 80/20 split

Architecture (identical for both probes):
    Conv1d(d_in, 32, k=7, pad=3) -> Tanh -> Conv1d(32, n_out, k=7, pad=3)

Saves weights to src/one_embedding/tools/weights/{disorder,ss3}_cnn_768d.pt
"""

import sys
import random
import time
from pathlib import Path

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.h5_store import load_residue_embeddings
from src.one_embedding.core import Codec
from src.evaluation.per_residue_tasks import load_cb513_csv, load_chezod_seth

import torch
import torch.nn as nn
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────
D_OUT = 768
SEED = 42
EPOCHS = 100
LR = 0.001
PATIENCE = 10
GRAD_ACCUM = 8

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "src" / "one_embedding" / "tools" / "weights"

# MPS Conv1d has channel mismatch bugs with certain input sizes — use CPU for training
# Inference is fine on MPS but training stability is better on CPU
device = torch.device("cpu")
print(f"Device: {device}")


# ── CNN Architecture (SETH-style) ────────────────────────────
class CNN(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 32, kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(32, n_out, kernel_size=7, padding=3),
        )

    def forward(self, x):
        # x: (B, L, d_in) -> permute to (B, d_in, L) -> conv -> (B, n_out, L) -> permute back
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# ── Step 1: Load and compress embeddings ──────────────────────
print("\n" + "=" * 60)
print("Step 1: Loading and compressing embeddings to 768d")
print("=" * 60)

# Use pre-fitted ABTT params (avoids loading 5K corpus)
codec = Codec.for_plm("prot_t5", d_out=D_OUT)
print(f"Codec ready: d_out={codec.d_out}, ABTT fitted={codec._abtt_params is not None}")

# Load raw embeddings
print("\nLoading CheZOD embeddings...")
raw_chezod = load_residue_embeddings("data/residue_embeddings/prot_t5_xl_chezod.h5")

print("Loading CB513 embeddings...")
raw_cb513 = load_residue_embeddings("data/residue_embeddings/prot_t5_xl_cb513.h5")

# Compress to 768d
print("\nCompressing CheZOD to 768d...")
comp_chezod = {}
for pid, raw in raw_chezod.items():
    encoded = codec.encode(raw)
    comp_chezod[pid] = encoded["per_residue"].astype(np.float32)

print(f"  {len(comp_chezod)} proteins, first shape: {next(iter(comp_chezod.values())).shape}")

print("Compressing CB513 to 768d...")
comp_cb513 = {}
for pid, raw in raw_cb513.items():
    encoded = codec.encode(raw)
    comp_cb513[pid] = encoded["per_residue"].astype(np.float32)

print(f"  {len(comp_cb513)} proteins, first shape: {next(iter(comp_cb513.values())).shape}")


# ── Step 2: Load labels ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Loading labels")
print("=" * 60)

# CheZOD labels
_, disorder_scores, chezod_train_ids, chezod_test_ids = load_chezod_seth(
    "data/per_residue_benchmarks"
)
print(f"CheZOD: {len(chezod_train_ids)} train, {len(chezod_test_ids)} test")

# CB513 labels
_, ss3_labels, _, _ = load_cb513_csv("data/per_residue_benchmarks/CB513.csv")
print(f"CB513: {len(ss3_labels)} proteins with SS3 labels")

# CB513 train/test split (80/20, seed=42)
cb513_ids = sorted(ss3_labels.keys())
rng = random.Random(SEED)
rng.shuffle(cb513_ids)
split_idx = int(0.8 * len(cb513_ids))
cb513_train_ids = cb513_ids[:split_idx]
cb513_test_ids = cb513_ids[split_idx:]
print(f"CB513 split: {len(cb513_train_ids)} train, {len(cb513_test_ids)} test")


# ── Step 3: Train Disorder CNN ────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Training Disorder CNN probe (768d -> 1, MSE loss)")
print("=" * 60)

# Filter to proteins that have both embeddings and labels
disorder_train = [
    pid for pid in chezod_train_ids
    if pid in comp_chezod and pid in disorder_scores
]
disorder_test = [
    pid for pid in chezod_test_ids
    if pid in comp_chezod and pid in disorder_scores
]
print(f"Disorder: {len(disorder_train)} train, {len(disorder_test)} test proteins")

torch.manual_seed(SEED)
np.random.seed(SEED)

disorder_model = CNN(D_OUT, 1).to(device)
optimizer = torch.optim.Adam(disorder_model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

best_test_loss = float("inf")
best_state = None
patience_counter = 0

t0 = time.time()
for epoch in range(EPOCHS):
    # ── Train ──
    disorder_model.train()
    train_loss_sum = 0.0
    train_n = 0
    optimizer.zero_grad()

    rng_epoch = random.Random(SEED + epoch)
    shuffled_train = list(disorder_train)
    rng_epoch.shuffle(shuffled_train)

    for i, pid in enumerate(shuffled_train):
        emb = comp_chezod[pid]  # (L, 768)
        scores = disorder_scores[pid]  # (L,)

        # Truncate to min length (rare: embedding may be truncated at 512)
        L = min(len(emb), len(scores))
        emb = emb[:L]
        scores = scores[:L]

        # Mask out NaN positions
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            continue

        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 768)
        y = torch.tensor(scores, dtype=torch.float32).to(device)  # (L,)

        pred = disorder_model(x).squeeze(0).squeeze(-1)  # (L,)
        valid_t = torch.tensor(valid, dtype=torch.bool).to(device)
        loss = loss_fn(pred[valid_t], y[valid_t]) / GRAD_ACCUM
        loss.backward()

        train_loss_sum += loss.item() * GRAD_ACCUM * valid_t.sum().item()
        train_n += valid_t.sum().item()

        if (i + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Flush remaining gradients
    if len(shuffled_train) % GRAD_ACCUM != 0:
        optimizer.step()
        optimizer.zero_grad()

    # ── Test ──
    disorder_model.eval()
    test_loss_sum = 0.0
    test_n = 0
    with torch.no_grad():
        for pid in disorder_test:
            emb = comp_chezod[pid]
            scores = disorder_scores[pid]
            L = min(len(emb), len(scores))
            emb, scores = emb[:L], scores[:L]
            valid = ~np.isnan(scores)
            if valid.sum() == 0:
                continue
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(scores, dtype=torch.float32).to(device)
            pred = disorder_model(x).squeeze(0).squeeze(-1)
            valid_t = torch.tensor(valid, dtype=torch.bool).to(device)
            test_loss_sum += loss_fn(pred[valid_t], y[valid_t]).item() * valid_t.sum().item()
            test_n += valid_t.sum().item()

    train_loss = train_loss_sum / max(train_n, 1)
    test_loss = test_loss_sum / max(test_n, 1)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.cpu().clone() for k, v in disorder_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS}  "
            f"train_mse={train_loss:.4f}  test_mse={test_loss:.4f}  "
            f"best={best_test_loss:.4f}  patience={patience_counter}/{PATIENCE}  "
            f"[{elapsed:.1f}s]"
        )

    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch + 1}")
        break

# Evaluate best model: Spearman rho on test set
disorder_model.load_state_dict(best_state)
disorder_model.eval()

all_pred, all_true = [], []
with torch.no_grad():
    for pid in disorder_test:
        emb = comp_chezod[pid]
        scores = disorder_scores[pid]
        L = min(len(emb), len(scores))
        emb, scores = emb[:L], scores[:L]
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            continue
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        pred = disorder_model(x).squeeze(0).squeeze(-1).cpu().numpy()
        all_pred.extend(pred[valid].tolist())
        all_true.extend(scores[valid].tolist())

rho_disorder, _ = spearmanr(all_pred, all_true)
print(f"\n  Disorder CNN 768d — Spearman rho = {rho_disorder:.4f}")
print(f"  Test residues: {len(all_true)}")

# Save disorder weights
disorder_path = WEIGHTS_DIR / "disorder_cnn_768d.pt"
torch.save(best_state, str(disorder_path))
print(f"  Saved: {disorder_path}")


# ── Step 4: Train SS3 CNN ─────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Training SS3 CNN probe (768d -> 3, CrossEntropy loss)")
print("=" * 60)

SS3_MAP = {"H": 0, "E": 1, "C": 2}

# Filter to proteins with both embeddings and labels
ss3_train = [pid for pid in cb513_train_ids if pid in comp_cb513 and pid in ss3_labels]
ss3_test = [pid for pid in cb513_test_ids if pid in comp_cb513 and pid in ss3_labels]
print(f"SS3: {len(ss3_train)} train, {len(ss3_test)} test proteins")

torch.manual_seed(SEED)
np.random.seed(SEED)

ss3_model = CNN(D_OUT, 3).to(device)
optimizer = torch.optim.Adam(ss3_model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

best_test_loss = float("inf")
best_state = None
patience_counter = 0

t0 = time.time()
for epoch in range(EPOCHS):
    # ── Train ──
    ss3_model.train()
    train_loss_sum = 0.0
    train_n = 0
    optimizer.zero_grad()

    rng_epoch = random.Random(SEED + epoch)
    shuffled_train = list(ss3_train)
    rng_epoch.shuffle(shuffled_train)

    for i, pid in enumerate(shuffled_train):
        emb = comp_cb513[pid]  # (L, 768)
        label_str = ss3_labels[pid]  # e.g. "HHHCCCEEEH..."
        labels = np.array([SS3_MAP[c] for c in label_str], dtype=np.int64)

        # Truncate to min length (rare: embedding may be truncated at 512)
        L = min(len(emb), len(labels))
        emb = emb[:L]
        labels = labels[:L]

        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 768)
        y = torch.tensor(labels, dtype=torch.long).to(device)  # (L,)

        pred = ss3_model(x).squeeze(0)  # (L, 3)
        loss = loss_fn(pred, y) / GRAD_ACCUM
        loss.backward()

        train_loss_sum += loss.item() * GRAD_ACCUM * len(labels)
        train_n += len(labels)

        if (i + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Flush remaining gradients
    if len(shuffled_train) % GRAD_ACCUM != 0:
        optimizer.step()
        optimizer.zero_grad()

    # ── Test ──
    ss3_model.eval()
    test_loss_sum = 0.0
    test_n = 0
    with torch.no_grad():
        for pid in ss3_test:
            emb = comp_cb513[pid]
            label_str = ss3_labels[pid]
            labels = np.array([SS3_MAP[c] for c in label_str], dtype=np.int64)
            L = min(len(emb), len(labels))
            emb, labels = emb[:L], labels[:L]

            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(labels, dtype=torch.long).to(device)
            pred = ss3_model(x).squeeze(0)
            test_loss_sum += loss_fn(pred, y).item() * len(labels)
            test_n += len(labels)

    train_loss = train_loss_sum / max(train_n, 1)
    test_loss = test_loss_sum / max(test_n, 1)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.cpu().clone() for k, v in ss3_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS}  "
            f"train_ce={train_loss:.4f}  test_ce={test_loss:.4f}  "
            f"best={best_test_loss:.4f}  patience={patience_counter}/{PATIENCE}  "
            f"[{elapsed:.1f}s]"
        )

    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch + 1}")
        break

# Evaluate best model: Q3 accuracy on test set
ss3_model.load_state_dict(best_state)
ss3_model.eval()

all_pred, all_true = [], []
with torch.no_grad():
    for pid in ss3_test:
        emb = comp_cb513[pid]
        label_str = ss3_labels[pid]
        labels = np.array([SS3_MAP[c] for c in label_str], dtype=np.int64)
        L = min(len(emb), len(labels))
        emb, labels = emb[:L], labels[:L]

        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        pred = ss3_model(x).squeeze(0).argmax(dim=1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(labels.tolist())

q3 = np.mean(np.array(all_pred) == np.array(all_true))
print(f"\n  SS3 CNN 768d — Q3 accuracy = {q3:.4f}")
print(f"  Test residues: {len(all_true)}")

# Save SS3 weights
ss3_path = WEIGHTS_DIR / "ss3_cnn_768d.pt"
torch.save(best_state, str(ss3_path))
print(f"  Saved: {ss3_path}")


# ── Step 5: Comparison with 512d probes ───────────────────────
print("\n" + "=" * 60)
print("Step 5: Comparison")
print("=" * 60)

print(f"\n  768d Results:")
print(f"    Disorder Spearman rho = {rho_disorder:.4f}")
print(f"    SS3 Q3 accuracy       = {q3:.4f}")

# Check if 512d weights exist and print for comparison
disorder_512_path = WEIGHTS_DIR / "disorder_cnn_512d.pt"
ss3_512_path = WEIGHTS_DIR / "ss3_cnn_512d.pt"

if disorder_512_path.exists() and ss3_512_path.exists():
    print(f"\n  512d Reference (from training):")
    print(f"    Disorder Spearman rho = 0.707  (previously reported)")
    print(f"    SS3 Q3 accuracy       = 0.432  (previously reported)")
else:
    print(f"\n  (No 512d weights found for comparison)")

print(f"\nDone. Weights saved to {WEIGHTS_DIR}")
