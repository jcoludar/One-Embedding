"""Stage 3 continuous regression helpers for Seq2OE.

Unlike `seq2oe.py` (which uses binary BCE targets), Stage 3 predicts the
continuous 896d projected ProtT5 vector directly and trains with cosine +
MSE loss. This module provides the target-preparation, loss, and evaluation
utilities specific to that setup.

The model class itself (`Seq2OE_CNN`) is reused unchanged from `seq2oe.py` —
its 896-dim linear head outputs floats that we now interpret as continuous
regression values instead of pre-sigmoid logits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def prepare_continuous_targets(
    train_embeddings: dict[str, np.ndarray],
    all_embeddings: dict[str, np.ndarray],
    d_out: int = 896,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Fit `OneEmbeddingCodec` on train embeddings only, then apply the
    preprocessing pipeline (centering + random projection) to every protein
    in `all_embeddings`. Returns the continuous 896d projected vectors
    (pre-binarization) so Stage 3 can regress on them directly.

    Args:
        train_embeddings: {pid: (L, D) float32} used to fit codec centering
            stats. Typically the CATH20 H-split train fold.
        all_embeddings: {pid: (L, D) float32} the full set to encode. May be
            a strict superset of train_embeddings.
        d_out: Random projection target dimension. Matches Stage 2.
        seed: Deterministic RP matrix seed.

    Returns:
        {pid: (L, d_out) float32} continuous projected targets. Every key
        from `all_embeddings` appears in the result.
    """
    from src.one_embedding.codec_v2 import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(train_embeddings)

    targets: dict[str, np.ndarray] = {}
    for pid, raw in all_embeddings.items():
        projected = codec._preprocess(raw)  # (L, d_out), float32
        targets[pid] = projected.astype(np.float32, copy=False)
    return targets


def cosine_distance_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Masked per-residue cosine distance, averaged over valid positions.

    For each (batch, residue) position, compute
        1 - cos(pred[b, r, :], target[b, r, :])
    then average over positions where mask == 1.

    Args:
        pred: (B, L, D) continuous predictions.
        target: (B, L, D) continuous targets.
        mask: (B, L) float, 1.0 for valid positions, 0.0 for padding.
        eps: Numerical stability floor on the norms.

    Returns:
        Scalar tensor in [0, 2]. 0 = perfect alignment, 2 = antiparallel.
    """
    # (B, L) cosine similarity per residue
    pred_norm = pred.norm(dim=-1).clamp_min(eps)
    target_norm = target.norm(dim=-1).clamp_min(eps)
    cos_sim = (pred * target).sum(dim=-1) / (pred_norm * target_norm)
    cos_dist = 1.0 - cos_sim  # (B, L)

    # Masked average
    n_valid = mask.sum().clamp_min(1.0)
    return (cos_dist * mask).sum() / n_valid


def mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Masked per-element MSE, averaged over valid positions AND all D dims.

    Args:
        pred: (B, L, D) continuous predictions.
        target: (B, L, D) continuous targets.
        mask: (B, L) float, 1.0 for valid positions, 0.0 for padding.

    Returns:
        Scalar tensor. Mean squared error over valid (b, r, d) cells.
    """
    # (B, L, D) squared error
    sq = (pred - target) ** 2
    # Broadcast mask to (B, L, 1)
    mask_3d = mask.unsqueeze(-1)
    # Total valid cells = valid_residues * D
    d = pred.shape[-1]
    n_valid = mask.sum().clamp_min(1.0) * d
    return (sq * mask_3d).sum() / n_valid


def evaluate_continuous(*args, **kwargs):  # implemented in Task 3
    raise NotImplementedError("Implemented in Task 3")
