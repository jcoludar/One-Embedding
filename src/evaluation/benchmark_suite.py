"""Orchestrate all benchmarks into a single JSON report."""

import json
from pathlib import Path

import numpy as np

from src.compressors.base import SequenceCompressor
from src.evaluation.reconstruction import evaluate_reconstruction
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe


def compute_compression_ratio(
    model: SequenceCompressor,
    embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute compression ratio.

    For fixed-token models: (K * D') / mean(L * D)
    For channel compressors (num_tokens=-1): (mean_L * D') / (mean_L * D) = D'/D
    """
    K = model.num_tokens
    D_prime = model.latent_dim
    D = next(iter(embeddings.values())).shape[-1]
    lengths = [emb.shape[0] for emb in embeddings.values()]
    mean_L = np.mean(lengths)

    if K == -1:
        # Channel compressor: same L, reduced D
        compressed_size = float(mean_L * D_prime)
        original_size = float(mean_L * D)
    else:
        compressed_size = K * D_prime
        original_size = mean_L * D

    ratio = compressed_size / original_size

    return {
        "K": K,
        "D_prime": D_prime,
        "D_original": D,
        "mean_length": float(mean_L),
        "compressed_elements": float(compressed_size),
        "original_elements": float(original_size),
        "compression_ratio": float(ratio),
    }


def run_benchmark_suite(
    model: SequenceCompressor | None,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    name: str = "unnamed",
    device=None,
    label_keys: list[str] | None = None,
    train_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    eval_ids: list[str] | None = None,
    pooling_strategy: str = "mean",
    cls_train_ids: list[str] | None = None,
    cls_test_ids: list[str] | None = None,
) -> dict:
    """Run full benchmark suite and return results dict.

    Args:
        model: Compressor model (None for raw mean-pool baseline).
        embeddings: Per-residue embeddings {id: (L, D)}.
        metadata: List of dicts with at least 'id' and label columns.
        name: Name of this configuration.
        label_keys: Which label columns to evaluate on.
        train_ids: Train protein IDs for retrieval database and model training split.
        test_ids: Test protein IDs for retrieval database.
        eval_ids: Eval protein IDs for retrieval queries (subset of test_ids
            with enough family members). If None, uses test_ids.
        pooling_strategy: Pooling strategy for get_pooled().
        cls_train_ids: Train IDs for classification (family-stratified split).
            If None, falls back to train_ids.
        cls_test_ids: Test IDs for classification (family-stratified split).
            If None, falls back to test_ids.
    """
    if label_keys is None:
        label_keys = ["family", "superfamily", "fold"]

    split_mode = "held_out" if train_ids is not None else "legacy_cv"
    results = {"name": name, "split_mode": split_mode}

    # Use eval_ids for retrieval queries; default to test_ids
    retrieval_query_ids = eval_ids if eval_ids is not None else test_ids
    # For retrieval database in held-out mode, use test_ids only
    retrieval_db_ids = test_ids

    # Reconstruction (skip for pooling-only methods that can't actually reconstruct)
    if model is not None:
        D = next(iter(embeddings.values())).shape[-1]
        can_reconstruct = model.latent_dim != D or model.num_tokens > 1
        try:
            import torch
            _test = torch.randn(1, 10, D, device=device)
            _mask = torch.ones(1, 10, device=device)
            _out = model.to(device)(_test, _mask)
            can_reconstruct = _out["reconstructed"].shape[-1] == D
        except Exception:
            can_reconstruct = False

        if can_reconstruct:
            # Evaluate reconstruction on test set only if split provided
            recon_emb = embeddings
            if test_ids is not None:
                test_set = set(test_ids)
                recon_emb = {k: v for k, v in embeddings.items() if k in test_set}
            print(f"  [{name}] Evaluating reconstruction...")
            results["reconstruction"] = evaluate_reconstruction(model, recon_emb, device)
        else:
            results["reconstruction"] = {"mse": float("nan"), "cosine_sim": float("nan"), "note": "pooling-only, no decoder"}

        print(f"  [{name}] Computing compression ratio...")
        results["compression"] = compute_compression_ratio(model, embeddings)
    else:
        D = next(iter(embeddings.values())).shape[-1]
        mean_L = np.mean([e.shape[0] for e in embeddings.values()])
        results["compression"] = {
            "K": 1, "D_prime": D, "D_original": D,
            "mean_length": float(mean_L),
            "compression_ratio": float(1.0 / mean_L),
        }

    # Retrieval
    for label_key in label_keys:
        if not any(label_key in m for m in metadata):
            continue
        print(f"  [{name}] Evaluating retrieval by {label_key}...")
        results[f"retrieval_{label_key}"] = evaluate_retrieval(
            model, embeddings, metadata, label_key=label_key, device=device,
            query_ids=retrieval_query_ids, database_ids=retrieval_db_ids,
            pooling_strategy=pooling_strategy,
        )

    # Classification (use family-stratified split if provided, else fall back)
    cls_tr = cls_train_ids if cls_train_ids is not None else train_ids
    cls_te = cls_test_ids if cls_test_ids is not None else test_ids
    for label_key in label_keys:
        if not any(label_key in m for m in metadata):
            continue
        print(f"  [{name}] Evaluating linear probe on {label_key}...")
        results[f"classification_{label_key}"] = evaluate_linear_probe(
            model, embeddings, metadata, label_key=label_key, device=device,
            train_ids=cls_tr, test_ids=cls_te,
            pooling_strategy=pooling_strategy,
        )

    return results


def save_benchmark_results(results: dict | list[dict], output_path: Path | str) -> None:
    """Save benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to {output_path}")
