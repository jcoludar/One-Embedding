"""kNN retrieval benchmarks by family/fold using cosine similarity."""

import numpy as np

from src.compressors.base import SequenceCompressor
from src.utils.device import get_device


def _get_latent_vectors(
    model: SequenceCompressor | None,
    embeddings: dict[str, np.ndarray],
    device=None,
    max_len: int = 512,
    pooling_strategy: str = "mean",
) -> dict[str, np.ndarray]:
    """Get pooled latent vectors from a compressor model (or mean-pool raw embeddings if model is None)."""
    import torch

    if device is None:
        device = get_device()

    vectors = {}

    if model is None:
        # Mean pool raw embeddings
        for pid, emb in embeddings.items():
            vectors[pid] = emb.mean(axis=0)
    else:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for pid, emb in embeddings.items():
                L = min(emb.shape[0], max_len)
                emb_t = emb[:L]
                states = torch.from_numpy(emb_t).unsqueeze(0).to(device)
                mask = torch.ones(1, L, device=device)
                latent = model.compress(states, mask)
                # For channel compressors (num_tokens=-1), pass mask for aware pooling
                if model.num_tokens == -1:
                    pooled = model.get_pooled(latent, strategy=pooling_strategy, mask=mask)
                else:
                    pooled = model.get_pooled(latent, strategy=pooling_strategy)
                vectors[pid] = pooled[0].cpu().numpy()

    return vectors


def evaluate_retrieval(
    model: SequenceCompressor | None,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    k_values: list[int] | None = None,
    device=None,
    query_ids: list[str] | None = None,
    database_ids: list[str] | None = None,
    pooling_strategy: str = "mean",
    return_per_query: bool = False,
) -> dict[str, float]:
    """Evaluate kNN retrieval precision.

    Args:
        query_ids: If provided, only compute precision for these proteins.
        database_ids: If provided, restrict the retrieval database to these IDs.
            When None, uses all proteins in embeddings as the database.
        pooling_strategy: Pooling strategy for get_pooled() ("mean", "first",
            "mean_std", "concat").

    Returns dict with: precision@k for each k, mean_precision.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    vectors = _get_latent_vectors(model, embeddings, device, pooling_strategy=pooling_strategy)

    # Build label mapping
    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}

    # Determine database and query sets
    db_ids = [pid for pid in vectors if pid in id_to_label]
    if database_ids is not None:
        db_set = set(database_ids)
        db_ids = [pid for pid in db_ids if pid in db_set]

    if query_ids is not None:
        q_ids = [pid for pid in query_ids if pid in id_to_label and pid in vectors]
    else:
        q_ids = db_ids

    if len(db_ids) < 2 or len(q_ids) < 1:
        return {f"precision@{k}": 0.0 for k in k_values}

    # Build database matrix
    db_matrix = np.array([vectors[pid] for pid in db_ids])
    db_labels = [id_to_label[pid] for pid in db_ids]

    # Normalize database
    db_norms = np.linalg.norm(db_matrix, axis=1, keepdims=True)
    db_norms[db_norms == 0] = 1
    db_matrix = db_matrix / db_norms

    # Build query matrix
    q_matrix = np.array([vectors[pid] for pid in q_ids])
    q_labels = [id_to_label[pid] for pid in q_ids]

    # Normalize queries
    q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1
    q_matrix = q_matrix / q_norms

    # Similarity: (n_queries, n_database)
    sims = q_matrix @ db_matrix.T

    # Build a set for fast lookup of query positions in database
    db_id_to_idx = {pid: i for i, pid in enumerate(db_ids)}

    results = {}
    per_query_scores = {} if return_per_query else None

    for k in k_values:
        precisions = []
        for qi, qid in enumerate(q_ids):
            row = sims[qi].copy()
            # Exclude self if query is in the database
            if qid in db_id_to_idx:
                row[db_id_to_idx[qid]] = -1
            top_k_idx = np.argsort(row)[-k:]
            top_k_labels = [db_labels[j] for j in top_k_idx]
            correct = sum(1 for lbl in top_k_labels if lbl == q_labels[qi])
            precisions.append(correct / k)
        results[f"precision@{k}"] = float(np.mean(precisions))
        if return_per_query and k == 1:
            per_query_scores["precision@1"] = {qid: p for qid, p in zip(q_ids, precisions)}

    results["mean_precision"] = float(np.mean([v for k, v in results.items() if k.startswith("precision@")]))

    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for qi, qid in enumerate(q_ids):
        row = sims[qi].copy()
        if qid in db_id_to_idx:
            row[db_id_to_idx[qid]] = -1
        sorted_idx = np.argsort(row)[::-1]
        for rank, idx in enumerate(sorted_idx, 1):
            if db_labels[idx] == q_labels[qi]:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    results["mrr"] = float(np.mean(reciprocal_ranks))
    if return_per_query:
        per_query_scores["mrr"] = {qid: rr for qid, rr in zip(q_ids, reciprocal_ranks)}

    # MAP (Mean Average Precision)
    avg_precisions = []
    for qi, qid in enumerate(q_ids):
        row = sims[qi].copy()
        if qid in db_id_to_idx:
            row[db_id_to_idx[qid]] = -1
        sorted_idx = np.argsort(row)[::-1]
        n_relevant = 0
        sum_precision = 0.0
        for rank, idx in enumerate(sorted_idx, 1):
            if db_labels[idx] == q_labels[qi]:
                n_relevant += 1
                sum_precision += n_relevant / rank
        if n_relevant > 0:
            avg_precisions.append(sum_precision / n_relevant)
        else:
            avg_precisions.append(0.0)
    results["map"] = float(np.mean(avg_precisions))

    results["n_queries"] = len(q_ids)
    results["n_database"] = len(db_ids)

    if return_per_query:
        results["per_query"] = per_query_scores

    return results


def evaluate_retrieval_from_vectors(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    k_values: list[int] | None = None,
    query_ids: list[str] | None = None,
    database_ids: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval from pre-computed vectors (any dimensionality).

    Same logic as evaluate_retrieval but accepts pre-computed vectors
    directly instead of running a model. Includes precision@k, MRR, and MAP.

    Args:
        vectors: Mapping from protein ID to embedding vector.
        metadata: List of dicts with at least 'id' and label_key fields.
        label_key: Metadata field to use as retrieval label.
        k_values: List of k values for precision@k.
        query_ids: If provided, only compute metrics for these proteins.
        database_ids: If provided, restrict the database to these IDs.

    Returns:
        Dict with precision@k, mrr, map, n_queries, n_database.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}

    db_ids = [pid for pid in vectors if pid in id_to_label]
    if database_ids is not None:
        db_set = set(database_ids)
        db_ids = [pid for pid in db_ids if pid in db_set]

    if query_ids is not None:
        q_ids = [pid for pid in query_ids if pid in id_to_label and pid in vectors]
    else:
        q_ids = db_ids

    if len(db_ids) < 2 or len(q_ids) < 1:
        return {f"precision@{k}": 0.0 for k in k_values}

    db_matrix = np.array([vectors[pid] for pid in db_ids])
    db_labels = [id_to_label[pid] for pid in db_ids]

    db_norms = np.linalg.norm(db_matrix, axis=1, keepdims=True).clip(1e-8)
    db_matrix = db_matrix / db_norms

    q_matrix = np.array([vectors[pid] for pid in q_ids])
    q_labels = [id_to_label[pid] for pid in q_ids]

    q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True).clip(1e-8)
    q_matrix = q_matrix / q_norms

    sims = q_matrix @ db_matrix.T
    db_id_to_idx = {pid: i for i, pid in enumerate(db_ids)}

    results = {}
    mrr_sum = 0.0
    avg_precisions = []

    for qi, qid in enumerate(q_ids):
        q_label = q_labels[qi]
        row = sims[qi].copy()

        # Exclude self
        if qid in db_id_to_idx:
            row[db_id_to_idx[qid]] = -np.inf

        ranked = np.argsort(row)[::-1]

        for k in k_values:
            top_k_labels = [db_labels[j] for j in ranked[:k]]
            correct = sum(1 for lbl in top_k_labels if lbl == q_label)
            key = f"precision@{k}"
            results.setdefault(key, 0.0)
            results[key] += correct / k

        # MRR
        for rank, idx in enumerate(ranked, 1):
            if db_labels[idx] == q_label:
                mrr_sum += 1.0 / rank
                break

        # MAP: average precision for this query
        n_relevant = 0
        sum_precision = 0.0
        for rank, idx in enumerate(ranked, 1):
            if db_labels[idx] == q_label:
                n_relevant += 1
                sum_precision += n_relevant / rank
        if n_relevant > 0:
            avg_precisions.append(sum_precision / n_relevant)
        else:
            avg_precisions.append(0.0)

    n_queries = len(q_ids)
    for k in k_values:
        results[f"precision@{k}"] /= n_queries
    results["mrr"] = mrr_sum / n_queries
    results["map"] = float(np.mean(avg_precisions))
    results["n_queries"] = n_queries
    results["n_database"] = len(db_ids)

    return results
