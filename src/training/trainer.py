"""Unified training loop for all compressor strategies."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.compressors.base import SequenceCompressor
from src.training.objectives import (
    ReconstructionLoss,
    MaskedPredictionLoss,
    ContrastiveLoss,
    PooledReconstructionLoss,
    VICRegLoss,
    TokenOrthogonalityLoss,
)
from src.training.augmentations import random_crop, gaussian_noise
from src.utils.device import get_device


class ResidueEmbeddingDataset(Dataset):
    """Dataset of per-residue embeddings with variable lengths."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        sequences: dict[str, str] | None = None,
        max_len: int = 512,
    ):
        self.ids = list(embeddings.keys())
        self.embeddings = embeddings
        self.sequences = sequences or {}
        self.max_len = max_len
        self.embed_dim = next(iter(embeddings.values())).shape[-1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = self.embeddings[pid]  # (L, D)
        L = min(emb.shape[0], self.max_len)
        emb = emb[:L]

        # Pad to max_len
        padded = np.zeros((self.max_len, self.embed_dim), dtype=np.float32)
        padded[:L] = emb
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:L] = 1.0

        seq = self.sequences.get(pid, "")[:L]

        return {
            "id": pid,
            "states": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "sequence": seq,
            "length": L,
        }


def train_compressor(
    model: SequenceCompressor,
    embeddings: dict[str, np.ndarray],
    sequences: dict[str, str] | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    recon_weight: float = 1.0,
    masked_weight: float = 0.1,
    contrastive_weight: float = 0.0,
    pool_recon_weight: float = 0.0,
    vicreg_weight: float = 0.0,
    token_ortho_weight: float = 0.0,
    device: torch.device | None = None,
    checkpoint_dir: Path | str | None = None,
    log_every: int = 10,
    max_len: int = 512,
    seed: int = 42,
    protein_ids: set[str] | None = None,
    validation_embeddings: dict[str, np.ndarray] | None = None,
    validation_sequences: dict[str, str] | None = None,
    patience: int | None = None,
) -> dict:
    """Train a compressor model and return training history.

    Args:
        seed: Random seed for reproducibility.
        protein_ids: If provided, only train on these protein IDs.
        validation_embeddings: If provided, compute val loss and checkpoint on it.
        validation_sequences: Sequences for validation proteins (for masked loss).

    Returns dict with keys: 'losses', 'best_epoch', 'best_loss'.
    """
    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    if device is None:
        device = get_device()

    # Filter to requested protein IDs
    if protein_ids is not None:
        embeddings = {k: v for k, v in embeddings.items() if k in protein_ids}
        if sequences:
            sequences = {k: v for k, v in sequences.items() if k in protein_ids}
        assert len(embeddings) > 0, "No proteins matched protein_ids filter"

    model = model.to(device)
    g = torch.Generator().manual_seed(seed)
    dataset = ResidueEmbeddingDataset(embeddings, sequences, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    # Validation loader
    val_loader = None
    if validation_embeddings is not None:
        val_dataset = ResidueEmbeddingDataset(
            validation_embeddings, validation_sequences, max_len=max_len
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Objectives
    recon_loss_fn = ReconstructionLoss().to(device)

    masked_loss_fn = None
    if masked_weight > 0 and sequences:
        embed_dim = next(iter(embeddings.values())).shape[-1]
        masked_loss_fn = MaskedPredictionLoss(embed_dim).to(device)

    contrastive_loss_fn = None
    if contrastive_weight > 0:
        contrastive_loss_fn = ContrastiveLoss().to(device)

    pool_recon_loss_fn = None
    if pool_recon_weight > 0:
        pool_recon_loss_fn = PooledReconstructionLoss().to(device)

    vicreg_loss_fn = None
    if vicreg_weight > 0:
        vicreg_loss_fn = VICRegLoss().to(device)

    token_ortho_loss_fn = None
    if token_ortho_weight > 0:
        token_ortho_loss_fn = TokenOrthogonalityLoss().to(device)

    # Optimizer
    params = list(model.parameters())
    if masked_loss_fn:
        params += list(masked_loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "total_loss": [], "recon_mse": [], "recon_cos": []}
    if masked_loss_fn:
        history["masked_loss"] = []
        history["masked_acc"] = []
    if contrastive_loss_fn:
        history["contrastive_loss"] = []

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if masked_loss_fn:
            masked_loss_fn.train()

        epoch_losses = []
        epoch_metrics = {k: [] for k in history if k not in ("epoch",)}

        for batch in loader:
            states = batch["states"].to(device)
            mask = batch["mask"].to(device)
            seqs = batch["sequence"]

            optimizer.zero_grad()

            # Forward
            output = model(states, mask)
            reconstructed = output["reconstructed"]
            latent = output["latent"]

            # Reconstruction loss
            recon = recon_loss_fn(reconstructed, states, mask)
            total_loss = recon_weight * recon["loss"]

            epoch_metrics["recon_mse"].append(recon["mse"].item())
            epoch_metrics["recon_cos"].append(recon["cosine_loss"].item() if isinstance(recon["cosine_loss"], torch.Tensor) else recon["cosine_loss"])

            # VQ loss if applicable
            if "vq_loss" in output:
                total_loss = total_loss + output["vq_loss"]

            # Masked prediction loss
            if masked_loss_fn and any(seqs):
                masked = masked_loss_fn(reconstructed, seqs, mask)
                total_loss = total_loss + masked_weight * masked["loss"]
                epoch_metrics.setdefault("masked_loss", []).append(masked["loss"].item())
                epoch_metrics.setdefault("masked_acc", []).append(masked["accuracy"].item())

            # Contrastive loss
            if contrastive_loss_fn:
                aug_states, aug_mask = gaussian_noise(states, mask, std=0.1)
                aug_output = model(aug_states, aug_mask)
                z1 = model.get_pooled(latent, mask=mask)
                z2 = model.get_pooled(aug_output["latent"], mask=aug_mask)
                cl = contrastive_loss_fn(z1, z2)
                total_loss = total_loss + contrastive_weight * cl
                epoch_metrics.setdefault("contrastive_loss", []).append(cl.item())

            # Pooled reconstruction loss
            if pool_recon_loss_fn and hasattr(model, "pool_and_project"):
                pooled_pred = model.pool_and_project(latent)  # (B, D)
                # Compute mean-pooled target from input
                mask_f = mask.unsqueeze(-1)  # (B, L, 1)
                target_mean = (states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, D)
                pr = pool_recon_loss_fn(pooled_pred, target_mean)
                total_loss = total_loss + pool_recon_weight * pr["loss"]
                epoch_metrics.setdefault("pool_recon_mse", []).append(pr["mse"].item())

            # VICReg loss
            if vicreg_loss_fn:
                z_pooled = model.get_pooled(latent)  # (B, D')
                vr = vicreg_loss_fn(z_pooled)
                total_loss = total_loss + vicreg_weight * vr["loss"]
                epoch_metrics.setdefault("vicreg_var", []).append(vr["var_loss"].item())
                epoch_metrics.setdefault("vicreg_cov", []).append(vr["cov_loss"].item())

            # Token orthogonality loss
            if token_ortho_loss_fn:
                to = token_ortho_loss_fn(latent)
                total_loss = total_loss + token_ortho_weight * to["loss"]
                epoch_metrics.setdefault("token_ortho", []).append(to["mean_cos"].item())

            total_loss.backward()
            # Sanitize NaN/inf gradients before clipping to avoid
            # 0 * inf = NaN from clip_grad_norm_ (known issue with
            # contrastive losses that can produce inf logits).
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        history["epoch"].append(epoch)
        history["total_loss"].append(avg_loss)
        for k in epoch_metrics:
            if k in history and epoch_metrics[k]:
                history[k].append(np.mean(epoch_metrics[k]))

        # Validation loss (if validation set provided)
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    states = batch["states"].to(device)
                    mask = batch["mask"].to(device)
                    output = model(states, mask)
                    recon = recon_loss_fn(output["reconstructed"], states, mask)
                    val_losses.append((recon_weight * recon["loss"]).item())
            val_loss = np.mean(val_losses)
            history.setdefault("val_loss", []).append(val_loss)
            model.train()
            if masked_loss_fn:
                masked_loss_fn.train()

        # Checkpoint on validation loss if available, else training loss
        checkpoint_loss = val_loss if val_loss is not None else avg_loss
        if checkpoint_loss < best_loss:
            best_loss = checkpoint_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_dir:
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            if checkpoint_dir and (checkpoint_dir / "best_model.pt").exists():
                model.load_state_dict(
                    torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True)
                )
                print(f"  Reloaded best checkpoint from epoch {best_epoch}")
            break

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            mse_str = f"MSE={np.mean(epoch_metrics['recon_mse']):.4f}" if epoch_metrics["recon_mse"] else ""
            val_str = f" | ValLoss={val_loss:.4f}" if val_loss is not None else ""
            print(f"  Epoch {epoch:3d}/{epochs} | Loss={avg_loss:.4f} | {mse_str}{val_str} | {elapsed:.0f}s")

    if checkpoint_dir:
        torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return {"losses": history, "best_epoch": best_epoch, "best_loss": best_loss}
