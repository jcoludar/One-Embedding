#!/usr/bin/env python3
"""Experiment 23: Euclidean Distance + Biologically-Grounded Evaluation.

All 22 prior experiments used cosine similarity for retrieval. This experiment
tests whether Euclidean distance gives better family separation, re-evaluates
codec candidates that "failed" under cosine, and adds SCOP hierarchy-aware
evaluation to check if embeddings capture biological truth.

Steps:
  E1: Euclidean vs Cosine baseline (raw mean pool)
  E2: Re-evaluate codec candidates (Euclidean)
  E3: Re-evaluate "failed" concatenations (Euclidean)
  E4: Novel pooling approaches (both metrics)
  E5: Best approaches on feature hash d=512 (actual codec test)
  E6: Hierarchy evaluation for top methods
  E7: Summary table

Usage:
  uv run python experiments/23_euclidean_eval.py --step E1
  uv run python experiments/23_euclidean_eval.py              # run all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import dct as scipy_dct

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.hierarchy import (
    evaluate_hierarchy_distances,
    plot_distance_distributions,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.universal_transforms import feature_hash, random_orthogonal_project
from src.one_embedding.path_transforms import lag_cross_covariance_eigenvalues
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "euclidean_eval_results.json"
PLOTS_DIR = DATA_DIR / "plots" / "exp23"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512


# ── Helpers ──────────────────────────────────────────────────────


def monitor():
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except OSError:
        pass


def load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {len(results)} results to {RESULTS_PATH}")


def is_done(results: list[dict], name: str) -> bool:
    return any(r.get("name") == name for r in results)


def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def load_raw_embeddings(plm: str, dataset: str = "medium5k") -> dict[str, np.ndarray]:
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm}_{dataset}.h5"
    if not h5_path.exists():
        print(f"  ERROR: {h5_path} not found")
        return {}
    return load_residue_embeddings(h5_path)


def load_metadata() -> list[dict]:
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    return metadata


def compute_retrieval(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    test_ids: list[str],
    metric: str = "cosine",
) -> dict[str, float]:
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
        metric=metric,
    )


def compute_both_metrics(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    test_ids: list[str],
) -> dict[str, float]:
    """Run retrieval with both cosine and Euclidean, return combined dict."""
    cos = compute_retrieval(vectors, metadata, test_ids, metric="cosine")
    euc = compute_retrieval(vectors, metadata, test_ids, metric="euclidean")
    return {
        "cos_ret1": cos["precision@1"],
        "cos_mrr": cos["mrr"],
        "cos_map": cos["map"],
        "euc_ret1": euc["precision@1"],
        "euc_mrr": euc["mrr"],
        "euc_map": euc["map"],
        "n_queries": cos["n_queries"],
        "n_database": cos["n_database"],
    }


def pool_mean(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    return {pid: embeddings[pid][:MAX_LEN].mean(axis=0) for pid in ids if pid in embeddings}


def pool_max(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    return {pid: embeddings[pid][:MAX_LEN].max(axis=0) for pid in ids if pid in embeddings}


def pool_mean_max(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            vecs[pid] = np.concatenate([emb.mean(axis=0), emb.max(axis=0)])
    return vecs


def pool_mean_std(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            vecs[pid] = np.concatenate([emb.mean(axis=0), emb.std(axis=0)])
    return vecs


def pool_mean_half_diff(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            mu = emb.mean(axis=0)
            mid = len(emb) // 2
            half_diff = emb[:mid].mean(axis=0) - emb[mid:].mean(axis=0) if mid > 0 else np.zeros_like(mu)
            vecs[pid] = np.concatenate([mu, half_diff])
    return vecs


def pool_self_attention(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    """Self-attention weighted mean: query=mean, keys=residues, no learned params."""
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN].astype(np.float32)
            mu = emb.mean(axis=0)
            # Attention weights: softmax(emb @ mu / sqrt(d))
            d = emb.shape[1]
            scores = emb @ mu / np.sqrt(d)
            scores = scores - scores.max()  # numerical stability
            weights = np.exp(scores)
            weights = weights / weights.sum()
            vecs[pid] = (weights[:, np.newaxis] * emb).sum(axis=0)
    return vecs


def pool_block_norm_mean_std(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    """Block-normalized [norm(mean) | norm(std)] — each block L2-normalized before concat."""
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            mu = emb.mean(axis=0)
            sd = emb.std(axis=0)
            mu_norm = mu / max(np.linalg.norm(mu), 1e-8)
            sd_norm = sd / max(np.linalg.norm(sd), 1e-8)
            vecs[pid] = np.concatenate([mu_norm, sd_norm])
    return vecs


def pool_mean_xcov(embeddings: dict[str, np.ndarray], ids: list[str], k: int = 64) -> dict[str, np.ndarray]:
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            mu = emb.mean(axis=0)
            xcov = lag_cross_covariance_eigenvalues(emb, k=k)
            vecs[pid] = np.concatenate([mu, xcov])
    return vecs


def pool_path_sig(embeddings: dict[str, np.ndarray], ids: list[str], p: int = 32) -> dict[str, np.ndarray]:
    """Path signature depth 2 on random projection to p dims, prepend mean."""
    from src.one_embedding.path_transforms import path_signature_depth2
    vecs = {}
    # Pre-compute projection matrix
    D = next(iter(embeddings.values())).shape[1]
    rng = np.random.RandomState(42)
    R = rng.randn(D, p).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    proj = Q * np.sqrt(D / p)

    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            mu = emb.mean(axis=0)
            projected = emb @ proj
            sig = path_signature_depth2(projected)
            vecs[pid] = np.concatenate([mu, sig])
    return vecs


def pool_dct_k2_fhash(embeddings: dict[str, np.ndarray], ids: list[str], d_out: int = 512) -> dict[str, np.ndarray]:
    """DCT K=2 of feature-hashed residues."""
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            hashed = feature_hash(emb, d_out=d_out)
            L = hashed.shape[0]
            K = min(2, L)
            coeffs = scipy_dct(hashed, type=2, axis=0, norm="ortho")[:K]
            vecs[pid] = coeffs.ravel().astype(np.float32)
    return vecs


# ── Steps ────────────────────────────────────────────────────────


def step_E1(results: list[dict]):
    """E1: Euclidean vs Cosine baseline — raw mean pool."""
    print("\n═══ E1: Euclidean vs Cosine Baseline ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    # Raw mean pool
    name = "E1_raw_mean_pool"
    if not is_done(results, name):
        print("  Computing raw mean pool with both metrics...")
        vectors = pool_mean(embeddings, test_ids)
        both = compute_both_metrics(vectors, metadata, test_ids)
        dim = next(iter(vectors.values())).shape[0]

        result = {"name": name, "plm": "prot_t5_xl", "transform": "mean_pool",
                  "dim": dim, **both}
        results.append(result)
        save_results(results)
        print(f"  Mean pool: Cosine Ret@1={both['cos_ret1']:.3f}, Euclidean Ret@1={both['euc_ret1']:.3f}")

    # Hierarchy distances — both metrics
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    vectors = pool_mean(embeddings, test_ids)

    for metric in ["cosine", "euclidean"]:
        hier_name = f"E1_hierarchy_{metric}"
        if not is_done(results, hier_name):
            print(f"  Hierarchy distances ({metric})...")
            hier = evaluate_hierarchy_distances(vectors, metadata, metric=metric)
            result = {"name": hier_name, "plm": "prot_t5_xl", "transform": "mean_pool",
                      "metric": metric, **hier}
            results.append(result)
            save_results(results)

            levels = ["within_family", "same_superfamily", "same_fold", "unrelated"]
            print(f"    {metric} distances:")
            for lv in levels:
                m = hier.get(f"{lv}_mean")
                n = hier.get(f"{lv}_n_pairs", 0)
                if m is not None:
                    print(f"      {lv}: {m:.4f} (n={n:,})")
            print(f"    Separation ratio: {hier.get('separation_ratio', 'N/A')}")
            print(f"    Ordering correct: {hier.get('ordering_correct', 'N/A')}")

    # Distance distribution plots
    for metric in ["cosine", "euclidean"]:
        plot_path = str(PLOTS_DIR / f"dist_distrib_mean_pool_{metric}.png")
        if not Path(plot_path).exists():
            print(f"  Plotting {metric} distance distributions...")
            plot_distance_distributions(
                vectors, metadata, metric=metric,
                output_path=plot_path,
                title=f"Raw Mean Pool — {metric.title()} Distance",
            )
            print(f"    Saved to {plot_path}")

    monitor()


def step_E2(results: list[dict]):
    """E2: Re-evaluate codec candidates with Euclidean."""
    print("\n═══ E2: Codec Candidates — Euclidean Re-evaluation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    candidates = [
        ("E2_fhash_d256", "feature_hash_d256_mean", lambda e, ids: {
            pid: feature_hash(e[pid][:MAX_LEN], d_out=256).mean(axis=0)
            for pid in ids if pid in e
        }),
        ("E2_fhash_d512", "feature_hash_d512_mean", lambda e, ids: {
            pid: feature_hash(e[pid][:MAX_LEN], d_out=512).mean(axis=0)
            for pid in ids if pid in e
        }),
        ("E2_hybrid_K4_d256", "hybrid_K4_d256", lambda e, ids: _hybrid_pool(e, ids, K=4, d_out=256)),
        ("E2_hybrid_K4_d512", "hybrid_K4_d512", lambda e, ids: _hybrid_pool(e, ids, K=4, d_out=512)),
        ("E2_rproj_d256", "random_proj_d256_mean", lambda e, ids: _rproj_pool(e, ids, d_out=256)),
        ("E2_rproj_d512", "random_proj_d512_mean", lambda e, ids: _rproj_pool(e, ids, d_out=512)),
    ]

    for name, transform, pool_fn in candidates:
        if is_done(results, name):
            print(f"  {transform} already done, skipping.")
            continue

        print(f"  {transform}...")
        t0 = time.time()
        vectors = pool_fn(embeddings, test_ids)
        elapsed = time.time() - t0

        both = compute_both_metrics(vectors, metadata, test_ids)
        dim = next(iter(vectors.values())).shape[0]

        result = {"name": name, "plm": "prot_t5_xl", "transform": transform,
                  "dim": dim, "encode_time_s": round(elapsed, 2), **both}
        results.append(result)
        save_results(results)
        print(f"    Cos Ret@1={both['cos_ret1']:.3f}, Euc Ret@1={both['euc_ret1']:.3f} "
              f"(dim={dim}, {elapsed:.1f}s)")

    monitor()


def _hybrid_pool(embeddings, ids, K, d_out):
    vecs = {}
    for pid in ids:
        if pid in embeddings:
            emb = embeddings[pid][:MAX_LEN]
            hashed = feature_hash(emb, d_out=d_out)
            L = hashed.shape[0]
            k = min(K, L)
            coeffs = scipy_dct(hashed, type=2, axis=0, norm="ortho")[:k]
            vecs[pid] = coeffs.ravel().astype(np.float32)
    return vecs


def _rproj_pool(embeddings, ids, d_out):
    D = next(iter(embeddings.values())).shape[1]
    rng = np.random.RandomState(42)
    R = rng.randn(D, d_out).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    proj = Q * np.sqrt(D / d_out)
    return {
        pid: (embeddings[pid][:MAX_LEN] @ proj).mean(axis=0).astype(np.float32)
        for pid in ids if pid in embeddings
    }


def step_E3(results: list[dict]):
    """E3: Re-evaluate 'failed' concatenations with Euclidean."""
    print("\n═══ E3: Failed Concatenations — Euclidean Re-evaluation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    candidates = [
        ("E3_mean_std", "[mean|std]", pool_mean_std),
        ("E3_mean_half_diff", "[mean|half_diff]", pool_mean_half_diff),
        ("E3_mean_xcov_k64", "[mean|xcov_k64]", lambda e, ids: pool_mean_xcov(e, ids, k=64)),
        ("E3_path_sig_p32", "[mean|sig2_p32]", lambda e, ids: pool_path_sig(e, ids, p=32)),
    ]

    for name, transform, pool_fn in candidates:
        if is_done(results, name):
            print(f"  {transform} already done, skipping.")
            continue

        print(f"  {transform}...")
        t0 = time.time()
        vectors = pool_fn(embeddings, test_ids)
        elapsed = time.time() - t0

        both = compute_both_metrics(vectors, metadata, test_ids)
        dim = next(iter(vectors.values())).shape[0]

        result = {"name": name, "plm": "prot_t5_xl", "transform": transform,
                  "dim": dim, "encode_time_s": round(elapsed, 2), **both}
        results.append(result)
        save_results(results)
        print(f"    Cos Ret@1={both['cos_ret1']:.3f}, Euc Ret@1={both['euc_ret1']:.3f} "
              f"(dim={dim}, {elapsed:.1f}s)")

    monitor()


def step_E4(results: list[dict]):
    """E4: Novel pooling approaches with both metrics."""
    print("\n═══ E4: Novel Pooling Approaches ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    candidates = [
        ("E4_max_pool", "max_pool", pool_max),
        ("E4_mean_max", "[mean|max]", pool_mean_max),
        ("E4_self_attn", "self_attn_mean", pool_self_attention),
        ("E4_block_norm_mean_std", "block_norm[mean|std]", pool_block_norm_mean_std),
        ("E4_dct_k2_fhash512", "dct_K2_fhash512", lambda e, ids: pool_dct_k2_fhash(e, ids, d_out=512)),
    ]

    for name, transform, pool_fn in candidates:
        if is_done(results, name):
            print(f"  {transform} already done, skipping.")
            continue

        print(f"  {transform}...")
        t0 = time.time()
        vectors = pool_fn(embeddings, test_ids)
        elapsed = time.time() - t0

        both = compute_both_metrics(vectors, metadata, test_ids)
        dim = next(iter(vectors.values())).shape[0]

        result = {"name": name, "plm": "prot_t5_xl", "transform": transform,
                  "dim": dim, "encode_time_s": round(elapsed, 2), **both}
        results.append(result)
        save_results(results)
        print(f"    Cos Ret@1={both['cos_ret1']:.3f}, Euc Ret@1={both['euc_ret1']:.3f} "
              f"(dim={dim}, {elapsed:.1f}s)")

    monitor()


def step_E5(results: list[dict]):
    """E5: Best approaches on feature hash d=512 (the actual codec test)."""
    print("\n═══ E5: Codec Test — Best Pooling on Feature Hash d=512 ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    # Pre-hash all embeddings to d=512
    print("  Feature hashing all embeddings to d=512...")
    hashed_emb = {}
    for pid in test_ids:
        if pid in embeddings:
            hashed_emb[pid] = feature_hash(embeddings[pid][:MAX_LEN], d_out=512)

    candidates = [
        ("E5_fh512_mean", "fh512_mean", lambda h, ids: {pid: h[pid].mean(axis=0) for pid in ids if pid in h}),
        ("E5_fh512_max", "fh512_max", lambda h, ids: {pid: h[pid].max(axis=0) for pid in ids if pid in h}),
        ("E5_fh512_mean_max", "fh512_[mean|max]", lambda h, ids: {
            pid: np.concatenate([h[pid].mean(axis=0), h[pid].max(axis=0)])
            for pid in ids if pid in h
        }),
        ("E5_fh512_mean_std", "fh512_[mean|std]", lambda h, ids: {
            pid: np.concatenate([h[pid].mean(axis=0), h[pid].std(axis=0)])
            for pid in ids if pid in h
        }),
        ("E5_fh512_self_attn", "fh512_self_attn", lambda h, ids: _self_attn_dict(h, ids)),
        ("E5_fh512_block_norm", "fh512_block_norm[mean|std]", lambda h, ids: _block_norm_dict(h, ids)),
    ]

    for name, transform, pool_fn in candidates:
        if is_done(results, name):
            print(f"  {transform} already done, skipping.")
            continue

        print(f"  {transform}...")
        t0 = time.time()
        vectors = pool_fn(hashed_emb, test_ids)
        elapsed = time.time() - t0

        both = compute_both_metrics(vectors, metadata, test_ids)
        dim = next(iter(vectors.values())).shape[0]

        result = {"name": name, "plm": "prot_t5_xl", "transform": transform,
                  "dim": dim, "codec": "feature_hash_d512",
                  "encode_time_s": round(elapsed, 2), **both}
        results.append(result)
        save_results(results)
        print(f"    Cos Ret@1={both['cos_ret1']:.3f}, Euc Ret@1={both['euc_ret1']:.3f} "
              f"(dim={dim}, {elapsed:.1f}s)")

    monitor()


def _self_attn_dict(hashed_emb, ids):
    vecs = {}
    for pid in ids:
        if pid in hashed_emb:
            emb = hashed_emb[pid].astype(np.float32)
            mu = emb.mean(axis=0)
            d = emb.shape[1]
            scores = emb @ mu / np.sqrt(d)
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights = weights / weights.sum()
            vecs[pid] = (weights[:, np.newaxis] * emb).sum(axis=0)
    return vecs


def _block_norm_dict(hashed_emb, ids):
    vecs = {}
    for pid in ids:
        if pid in hashed_emb:
            emb = hashed_emb[pid]
            mu = emb.mean(axis=0)
            sd = emb.std(axis=0)
            mu_n = mu / max(np.linalg.norm(mu), 1e-8)
            sd_n = sd / max(np.linalg.norm(sd), 1e-8)
            vecs[pid] = np.concatenate([mu_n, sd_n])
    return vecs


def step_E6(results: list[dict]):
    """E6: Hierarchy evaluation for top methods."""
    print("\n═══ E6: Hierarchy Evaluation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick top methods from E1-E5 results
    # Always evaluate: mean, [mean|max], [mean|std], self_attn, fhash_d512_mean
    methods = [
        ("E6_hier_mean", "mean_pool", pool_mean),
        ("E6_hier_mean_max", "[mean|max]", pool_mean_max),
        ("E6_hier_mean_std", "[mean|std]", pool_mean_std),
        ("E6_hier_self_attn", "self_attn", pool_self_attention),
        ("E6_hier_max", "max_pool", pool_max),
        ("E6_hier_fhash512_mean", "fhash512_mean", lambda e, ids: {
            pid: feature_hash(e[pid][:MAX_LEN], d_out=512).mean(axis=0)
            for pid in ids if pid in e
        }),
    ]

    for name, transform, pool_fn in methods:
        if is_done(results, name):
            print(f"  {transform} hierarchy already done, skipping.")
            continue

        print(f"  Hierarchy: {transform}...")
        vectors = pool_fn(embeddings, test_ids)

        for metric in ["cosine", "euclidean"]:
            hier = evaluate_hierarchy_distances(vectors, metadata, metric=metric)
            full_name = f"{name}_{metric}"
            result = {"name": full_name, "plm": "prot_t5_xl", "transform": transform,
                      "metric": metric, **hier}
            results.append(result)

            # Plot
            plot_path = str(PLOTS_DIR / f"dist_distrib_{transform.replace('|', '_').replace('[', '').replace(']', '')}_{metric}.png")
            if not Path(plot_path).exists():
                plot_distance_distributions(
                    vectors, metadata, metric=metric,
                    output_path=plot_path,
                    title=f"{transform} — {metric.title()} Distance",
                )

        # Mark parent as done
        results.append({"name": name, "transform": transform, "done": True})
        save_results(results)

        levels = ["within_family", "same_superfamily", "same_fold", "unrelated"]
        print(f"    Euclidean distances:")
        for lv in levels:
            m = hier.get(f"{lv}_mean")
            if m is not None:
                print(f"      {lv}: {m:.4f}")
        print(f"    Separation ratio: {hier.get('separation_ratio', 'N/A')}")

    monitor()


def step_E7(results: list[dict]):
    """E7: Summary table — all results with both metrics."""
    print("\n═══ E7: Summary Table ═══")

    # Collect results that have both metrics
    both_metric_results = [r for r in results if "cos_ret1" in r and "euc_ret1" in r]

    if not both_metric_results:
        print("  No dual-metric results found. Run E1-E5 first.")
        return

    # Sort by Euclidean Ret@1
    both_metric_results.sort(key=lambda r: r.get("euc_ret1", 0), reverse=True)

    print("\n  ═══ Retrieval: Cosine vs Euclidean (ProtT5, SCOPe 5K) ═══")
    print("  ┌────────────────────────────┬──────┬──────────────┬──────────────┬────────────┐")
    print("  │ Method                     │ Dim  │  Cos Ret@1   │  Euc Ret@1   │ Delta      │")
    print("  ├────────────────────────────┼──────┼──────────────┼──────────────┼────────────┤")

    for r in both_metric_results:
        tf = r.get("transform", "?")[:26]
        dim = r.get("dim", "?")
        cos = r.get("cos_ret1", 0)
        euc = r.get("euc_ret1", 0)
        delta = euc - cos
        sign = "+" if delta >= 0 else ""
        print(f"  │ {tf:<26s} │ {str(dim):>4s} │ {cos:>12.3f} │ {euc:>12.3f} │ {sign}{delta:>9.3f} │")

    print("  └────────────────────────────┴──────┴──────────────┴──────────────┴────────────┘")

    # Find best
    best_cos = max(both_metric_results, key=lambda r: r["cos_ret1"])
    best_euc = max(both_metric_results, key=lambda r: r["euc_ret1"])
    print(f"\n  Best cosine:    {best_cos['transform']} Ret@1={best_cos['cos_ret1']:.3f}")
    print(f"  Best Euclidean: {best_euc['transform']} Ret@1={best_euc['euc_ret1']:.3f}")

    # Hierarchy summary
    hier_results = [r for r in results if "separation_ratio" in r and r.get("separation_ratio") is not None]
    if hier_results:
        print("\n  ═══ Hierarchy Separation Ratios ═══")
        print("  ┌────────────────────────────┬──────────┬──────────────┬──────────────┐")
        print("  │ Method                     │ Metric   │ Sep. Ratio   │ Order OK?    │")
        print("  ├────────────────────────────┼──────────┼──────────────┼──────────────┤")

        hier_results.sort(key=lambda r: r.get("separation_ratio", 0), reverse=True)
        for r in hier_results:
            tf = r.get("transform", "?")[:26]
            metric = r.get("metric", "?")
            sep = r.get("separation_ratio", 0)
            order = "YES" if r.get("ordering_correct") else "no"
            print(f"  │ {tf:<26s} │ {metric:<8s} │ {sep:>12.3f} │ {order:>12s} │")

        print("  └────────────────────────────┴──────────┴──────────────┴──────────────┘")

    # Baselines reference
    print(f"\n  Reference baselines:")
    print(f"    Raw mean pool cosine:     Ret@1=0.734 (ground zero)")
    print(f"    Trained ChannelComp:      Ret@1=0.808 (trained model)")
    print(f"    Feature hash d=512 cos:   Ret@1=0.738 (Exp 21)")

    print("\n  E7 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 23: Euclidean + Hierarchy Evaluation")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step (E1-E7)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "E1": step_E1, "E2": step_E2, "E3": step_E3,
        "E4": step_E4, "E5": step_E5, "E6": step_E6, "E7": step_E7,
    }

    if args.step:
        step_name = args.step.upper()
        if step_name in steps:
            steps[step_name](results)
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
    else:
        for step_name, step_fn in steps.items():
            step_fn(results)

    print("\n Done.")


if __name__ == "__main__":
    main()
