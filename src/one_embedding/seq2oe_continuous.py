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


# Stubs to be implemented in Tasks 2 and 3
def cosine_distance_loss(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 2")


def mse_loss(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 2")


def evaluate_continuous(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 3")
