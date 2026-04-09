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


def evaluate_continuous(
    model,
    sequences: dict[str, str],
    targets: dict[str, np.ndarray],
    ids: set,
    device: torch.device,
    batch_size: int = 8,
    max_len: int = 512,
) -> dict:
    """Evaluate a Seq2OE model on a held-out set with continuous metrics.

    Computes the four Stage 3 primary metrics at the per-residue level:
        - cosine_sim: mean cosine similarity per residue, averaged over all
          valid positions.
        - cosine_distance: 1 - cosine_sim (redundant but reported for
          symmetry with the loss name).
        - mse: mean squared error per (residue, dim) cell.
        - bit_accuracy: re-binarize both pred and target via
          `sign(x - per-protein-mean)` and compare bit-by-bit.

    Also returns `dim_accuracies` — a length-D array of per-dimension bit
    accuracies, matching Stage 2's reporting format for the Intersect@60
    aggregation.

    Args:
        model: any `nn.Module` with signature `forward(input_ids, mask)
            -> (B, L, D)`.
        sequences: {pid: str} amino-acid sequences.
        targets: {pid: (L, D) float32} continuous targets.
        ids: set of protein IDs to evaluate on (intersection with sequences
            and targets keys is used).
        device: torch device.
        batch_size: eval batch size.
        max_len: sequence length cap (padding/truncation matches training).

    Returns:
        dict with keys: cosine_sim, cosine_distance, mse, bit_accuracy,
        per_protein_bit_acc_mean, per_protein_bit_acc_std,
        per_protein_bit_acc_min, per_protein_bit_acc_max,
        dim_accuracies (list of length D), n_test.
    """
    from torch.utils.data import DataLoader
    from src.one_embedding.seq2oe import Seq2OEDataset

    eval_seqs = {k: sequences[k] for k in ids if k in sequences and k in targets}
    eval_tgts = {k: targets[k] for k in ids if k in sequences and k in targets}
    ds = Seq2OEDataset(eval_seqs, eval_tgts, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Accumulators
    cos_sum = 0.0
    cos_count = 0
    mse_sum = 0.0
    mse_count = 0
    bit_correct_total = 0
    bit_total = 0
    dim_bit_correct = None  # lazy-init once we see D
    dim_bit_total = 0
    per_protein_accs: list[float] = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)  # (B, L, D), float32
            mask = batch["mask"].to(device)       # (B, L)
            lengths = batch["length"]

            pred = model(input_ids, mask)         # (B, L, D)

            # Cosine sim per residue
            pred_norm = pred.norm(dim=-1).clamp_min(1e-8)
            target_norm = target.norm(dim=-1).clamp_min(1e-8)
            cos = (pred * target).sum(dim=-1) / (pred_norm * target_norm)
            cos_sum += (cos * mask).sum().item()
            cos_count += mask.sum().item()

            # MSE
            d = pred.shape[-1]
            sq = (pred - target) ** 2
            mse_sum += (sq * mask.unsqueeze(-1)).sum().item()
            mse_count += mask.sum().item() * d

            # Bit accuracy after re-binarization (per-protein mean subtraction)
            if dim_bit_correct is None:
                dim_bit_correct = np.zeros(d, dtype=np.int64)

            for b in range(pred.shape[0]):
                L = int(lengths[b].item())
                if L == 0:
                    continue
                p = pred[b, :L].cpu().numpy()        # (L, D)
                t = target[b, :L].cpu().numpy()      # (L, D)
                p_bits = (p - p.mean(axis=0, keepdims=True)) > 0
                t_bits = (t - t.mean(axis=0, keepdims=True)) > 0
                eq = (p_bits == t_bits)              # (L, D) bool
                bit_correct_total += int(eq.sum())
                bit_total += eq.size
                dim_bit_correct += eq.sum(axis=0)
                dim_bit_total += L
                per_protein_accs.append(float(eq.mean()))

    cosine_sim = cos_sum / max(cos_count, 1.0)
    mse = mse_sum / max(mse_count, 1.0)
    bit_accuracy = bit_correct_total / max(bit_total, 1)
    dim_acc = (dim_bit_correct / max(dim_bit_total, 1)).tolist()
    per_prot = np.array(per_protein_accs) if per_protein_accs else np.zeros(1)

    return {
        "cosine_sim": float(cosine_sim),
        "cosine_distance": float(1.0 - cosine_sim),
        "mse": float(mse),
        "bit_accuracy": float(bit_accuracy),
        "per_protein_bit_acc_mean": float(per_prot.mean()),
        "per_protein_bit_acc_std": float(per_prot.std()),
        "per_protein_bit_acc_min": float(per_prot.min()),
        "per_protein_bit_acc_max": float(per_prot.max()),
        "dim_accuracies": dim_acc,
        "n_test": len(per_protein_accs),
    }
