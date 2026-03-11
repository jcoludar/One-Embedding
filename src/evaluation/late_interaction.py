"""ColBERT-style late interaction retrieval using all K latent tokens.

Instead of mean-pooling K latent tokens into a single vector, this module
computes retrieval scores using the full multi-token representation:

    score(q, d) = sum_i max_j cos(q_token_i, d_token_j)

This preserves the information distributed across K tokens and is the
standard approach in multi-vector retrieval (Khattab & Zaharia, SIGIR 2020).
"""

import numpy as np
import torch

from src.compressors.base import SequenceCompressor
from src.utils.device import get_device


def _get_latent_tokens(
    model: SequenceCompressor,
    embeddings: dict[str, np.ndarray],
    device=None,
    max_len: int = 512,
) -> dict[str, np.ndarray]:
    """Get all K latent tokens per protein (not pooled).

    Returns:
        {protein_id: (K, D') ndarray}
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    tokens = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)  # (1, K, D')
            tokens[pid] = latent[0].cpu().numpy()   # (K, D')

    return tokens


def _late_interaction_score(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    """Compute ColBERT-style late interaction score.

    score = sum_i max_j cos(q_i, d_j)

    Args:
        q_tokens: (K_q, D') query latent tokens
        d_tokens: (K_d, D') document latent tokens

    Returns:
        float score
    """
    # Normalize
    q_norms = np.linalg.norm(q_tokens, axis=1, keepdims=True).clip(1e-8)
    d_norms = np.linalg.norm(d_tokens, axis=1, keepdims=True).clip(1e-8)
    q_normed = q_tokens / q_norms
    d_normed = d_tokens / d_norms

    # (K_q, K_d) cosine similarities
    sim_matrix = q_normed @ d_normed.T

    # Sum of max over d for each q token
    return float(sim_matrix.max(axis=1).sum())


def evaluate_late_interaction(
    model: SequenceCompressor,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    k_values: list[int] | None = None,
    device=None,
    query_ids: list[str] | None = None,
    database_ids: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval using ColBERT-style late interaction over K latent tokens.

    Instead of pooling K tokens to one vector, computes:
        score(q, d) = sum_i max_j cos(q_token_i, d_token_j)

    Args:
        model: Trained compressor with K latent tokens.
        embeddings: Per-residue embeddings {id: (L, D)}.
        metadata: List of dicts with at least 'id' and label columns.
        label_key: Which label to evaluate on.
        k_values: List of k for precision@k.
        query_ids: If provided, only compute precision for these proteins.
        database_ids: If provided, restrict retrieval database to these IDs.

    Returns:
        Dict with precision@k for each k, plus mean_precision.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # Get all K latent tokens per protein
    tokens = _get_latent_tokens(model, embeddings, device)

    # Build label mapping
    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}

    # Determine database and query sets
    db_ids = [pid for pid in tokens if pid in id_to_label]
    if database_ids is not None:
        db_set = set(database_ids)
        db_ids = [pid for pid in db_ids if pid in db_set]

    if query_ids is not None:
        q_ids = [pid for pid in query_ids if pid in id_to_label and pid in tokens]
    else:
        q_ids = db_ids

    if len(db_ids) < 2 or len(q_ids) < 1:
        return {f"precision@{k}": 0.0 for k in k_values}

    db_labels = [id_to_label[pid] for pid in db_ids]
    db_id_to_idx = {pid: i for i, pid in enumerate(db_ids)}

    # Precompute normalized db tokens
    db_tokens = [tokens[pid] for pid in db_ids]
    db_tokens_normed = []
    for t in db_tokens:
        n = np.linalg.norm(t, axis=1, keepdims=True).clip(1e-8)
        db_tokens_normed.append(t / n)

    results = {}
    for k in k_values:
        precisions = []
        for qi, qid in enumerate(q_ids):
            q_tok = tokens[qid]
            q_norm = np.linalg.norm(q_tok, axis=1, keepdims=True).clip(1e-8)
            q_normed = q_tok / q_norm

            # Compute late interaction scores against all db proteins
            scores = np.empty(len(db_ids))
            for di in range(len(db_ids)):
                if db_ids[di] == qid:
                    scores[di] = -np.inf  # exclude self
                else:
                    sim_matrix = q_normed @ db_tokens_normed[di].T
                    scores[di] = sim_matrix.max(axis=1).sum()

            top_k_idx = np.argsort(scores)[-k:]
            top_k_labels = [db_labels[j] for j in top_k_idx]
            q_label = id_to_label[qid]
            correct = sum(1 for lbl in top_k_labels if lbl == q_label)
            precisions.append(correct / k)

        results[f"precision@{k}"] = float(np.mean(precisions))

    results["mean_precision"] = float(np.mean([
        results[f"precision@{k}"] for k in k_values
    ]))
    results["n_queries"] = len(q_ids)
    results["n_database"] = len(db_ids)
    results["scoring"] = "late_interaction"
    return results
