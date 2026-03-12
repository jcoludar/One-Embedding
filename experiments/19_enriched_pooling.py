#!/usr/bin/env python3
"""Experiment 19: Enriched Pooling — Beat Mean Pool.

Tests whether enriching protein representations with richer statistics
(variance, autocovariance, spectral features) and PCA-reducing back
to 256d can beat naive mean pooling at 256d.

Usage:
  uv run python experiments/19_enriched_pooling.py --step E1  # ProtT5 enriched
  uv run python experiments/19_enriched_pooling.py --step E2  # ESM2 cross-validation
  uv run python experiments/19_enriched_pooling.py --step E3  # Variance analysis
  uv run python experiments/19_enriched_pooling.py --step E4  # Feature importance
  uv run python experiments/19_enriched_pooling.py --step E5  # Statistical test
  uv run python experiments/19_enriched_pooling.py             # run all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.mixture import GaussianMixture

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.splitting import load_split
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.enriched_transforms import (
    EnrichedTransformPipeline,
    autocovariance_pool,
    dct_pool,
    fisher_vector,
    gram_features,
    haar_pool,
    moment_pool,
)
from src.one_embedding.pipeline import compress_embeddings
from src.one_embedding.registry import PLMRegistry
from src.utils.device import get_device
from src.evaluation.retrieval import evaluate_retrieval_from_vectors as retrieval_from_vectors
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "enriched_pooling_results.json"
SPLIT_DIR = DATA_DIR / "splits"
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


# ── Data Loading ─────────────────────────────────────────────────


def load_prot_t5_data():
    emb_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    embeddings = load_residue_embeddings(emb_path)
    metadata = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    valid_ids = {m["id"] for m in metadata}
    embeddings = {k: v for k, v in embeddings.items() if k in valid_ids}
    train_ids, test_ids, eval_ids = load_split(SPLIT_DIR / "esm2_650m_5k_split.json")
    split = {"train_ids": train_ids, "test_ids": test_ids, "eval_ids": eval_ids}
    return embeddings, metadata, split


def load_esm2_data():
    emb_path = DATA_DIR / "residue_embeddings" / "esm2_650m_medium5k.h5"
    embeddings = load_residue_embeddings(emb_path)
    metadata = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    valid_ids = {m["id"] for m in metadata}
    embeddings = {k: v for k, v in embeddings.items() if k in valid_ids}
    train_ids, test_ids, eval_ids = load_split(SPLIT_DIR / "esm2_650m_5k_split.json")
    split = {"train_ids": train_ids, "test_ids": test_ids, "eval_ids": eval_ids}
    return embeddings, metadata, split


def get_compressed(plm_name: str, embeddings: dict, device=None) -> dict[str, np.ndarray]:
    registry = PLMRegistry(checkpoint_base=CHECKPOINTS_DIR)
    model = registry.get_compressor(plm_name, device=device)
    return compress_embeddings(model, embeddings, device=device, max_len=MAX_LEN)


# ── Steps ────────────────────────────────────────────────────────


def step_E1(results, compressed, metadata, split):
    """E1: ProtT5 enriched retrieval — all transforms × {256, 512} dims."""
    print("\n═══ E1: ProtT5 Enriched Retrieval ═══")

    train_set = set(split["train_ids"])
    test_ids = [pid for pid in split["test_ids"] if pid in compressed]
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}

    # Define transforms
    transforms = {
        "moment_pool": (moment_pool, {}),
        "autocov_pool": (autocovariance_pool, {}),
        "gram_features": (gram_features, {}),
        "dct_K8_pca": (dct_pool, {"K": 8}),
        "dct_K16_pca": (dct_pool, {"K": 16}),
        "haar_L3_pca": (haar_pool, {"levels": 3}),
    }

    target_dims = [256, 512]

    for dim in target_dims:
        for tname, (tfn, tkwargs) in transforms.items():
            result_name = f"E1_{tname}_d{dim}"
            if is_done(results, result_name):
                print(f"  {result_name}: already done, skipping")
                continue

            print(f"  Fitting {tname} → PCA-{dim}...")
            t0 = time.time()

            pipe = EnrichedTransformPipeline(tfn, tkwargs)
            pipe.fit(train_matrices, target_dim=dim)

            # Transform test proteins (match Exp 18 protocol: database=test_ids)
            test_matrices = {k: compressed[k] for k in test_ids}
            vectors = pipe.transform_batch(test_matrices)

            fit_time = time.time() - t0

            # Evaluate retrieval — database=test_ids to match Exp 18
            ret = retrieval_from_vectors(
                vectors, metadata,
                query_ids=test_ids,
                database_ids=test_ids,
            )

            result = {
                "name": result_name,
                "plm": "prot_t5_xl",
                "transform": tname,
                "target_dim": dim,
                "raw_dim": pipe.raw_dim,
                "actual_dim": pipe.pca.n_components_,
                "variance_explained_total": float(pipe.variance_explained[-1]),
                "fit_time_s": round(fit_time, 2),
                **{k: v for k, v in ret.items()},
            }
            results.append(result)
            save_results(results)
            print(
                f"    {result_name}: Ret@1={ret['precision@1']:.3f}, "
                f"MRR={ret['mrr']:.3f}, "
                f"raw={pipe.raw_dim}→{pipe.pca.n_components_}d, "
                f"var={pipe.variance_explained[-1]:.3f}"
            )

    # Also add mean baseline at 256d for comparison (match Exp 18 protocol)
    result_name = "E1_mean_baseline"
    if not is_done(results, result_name):
        print("  Computing mean baseline...")
        mean_vectors = {pid: compressed[pid].mean(axis=0) for pid in test_ids}
        ret = retrieval_from_vectors(
            mean_vectors, metadata,
            query_ids=test_ids,
            database_ids=test_ids,
        )
        result = {
            "name": result_name,
            "plm": "prot_t5_xl",
            "transform": "mean",
            "target_dim": 256,
            "raw_dim": 256,
            **{k: v for k, v in ret.items()},
        }
        results.append(result)
        save_results(results)
        print(f"    mean baseline: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}")

    # Print ranked summary
    print("\n  ┌──────────────────────────┬────────┬────────┬────────┐")
    print("  │ Transform                │ Ret@1  │ MRR    │ Dim    │")
    print("  ├──────────────────────────┼────────┼────────┼────────┤")
    e1_results = sorted(
        [r for r in results if r["name"].startswith("E1_")],
        key=lambda r: r.get("precision@1", 0),
        reverse=True,
    )
    for r in e1_results:
        print(
            f"  │ {r['transform']:<24s} │ {r.get('precision@1', 0):.3f}  │ "
            f"{r.get('mrr', 0):.3f}  │ {r.get('target_dim', '?'):>5}  │"
        )
    print("  └──────────────────────────┴────────┴────────┴────────┘")

    return results


def step_E2(results, metadata, split):
    """E2: ESM2 cross-validation with top transforms."""
    print("\n═══ E2: ESM2 Cross-Validation ═══")

    embeddings, _, _ = load_esm2_data()
    device = get_device()
    compressed = get_compressed("esm2_650m", embeddings, device=device)
    del embeddings

    train_set = set(split["train_ids"])
    test_ids = [pid for pid in split["test_ids"] if pid in compressed]
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}

    transforms = {
        "moment_pool": (moment_pool, {}),
        "autocov_pool": (autocovariance_pool, {}),
        "dct_K8_pca": (dct_pool, {"K": 8}),
    }

    for dim in [256, 512]:
        for tname, (tfn, tkwargs) in transforms.items():
            result_name = f"E2_esm2_{tname}_d{dim}"
            if is_done(results, result_name):
                print(f"  {result_name}: already done, skipping")
                continue

            print(f"  Fitting {tname} → PCA-{dim} on ESM2...")
            pipe = EnrichedTransformPipeline(tfn, tkwargs)
            pipe.fit(train_matrices, target_dim=dim)
            test_matrices = {k: compressed[k] for k in test_ids}
            vectors = pipe.transform_batch(test_matrices)

            ret = retrieval_from_vectors(
                vectors, metadata,
                query_ids=test_ids,
                database_ids=test_ids,
            )

            result = {
                "name": result_name,
                "plm": "esm2_650m",
                "transform": tname,
                "target_dim": dim,
                "raw_dim": pipe.raw_dim,
                **{k: v for k, v in ret.items()},
            }
            results.append(result)
            save_results(results)
            print(f"    {result_name}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}")

    # ESM2 mean baseline
    result_name = "E2_esm2_mean_baseline"
    if not is_done(results, result_name):
        mean_vectors = {pid: compressed[pid].mean(axis=0) for pid in test_ids}
        ret = retrieval_from_vectors(
            mean_vectors, metadata,
            query_ids=test_ids,
            database_ids=test_ids,
        )
        result = {
            "name": result_name,
            "plm": "esm2_650m",
            "transform": "mean",
            "target_dim": 256,
            **{k: v for k, v in ret.items()},
        }
        results.append(result)
        save_results(results)
        print(f"    ESM2 mean baseline: Ret@1={ret['precision@1']:.3f}")

    return results


def step_E3(results, compressed, split):
    """E3: Variance explained analysis for key transforms."""
    print("\n═══ E3: PCA Variance Analysis ═══")

    result_name = "E3_variance_analysis"
    if is_done(results, result_name):
        print("  Already done, skipping")
        return results

    train_set = set(split["train_ids"])
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}

    transforms = {
        "moment_pool": (moment_pool, {}),
        "autocov_pool": (autocovariance_pool, {}),
        "gram_features": (gram_features, {}),
        "dct_K8": (dct_pool, {"K": 8}),
        "haar_L3": (haar_pool, {"levels": 3}),
    }

    analysis = {}
    for tname, (tfn, tkwargs) in transforms.items():
        pipe = EnrichedTransformPipeline(tfn, tkwargs)
        # Fit with high dim to see full variance curve
        max_dim = min(512, len(train_matrices))
        pipe.fit(train_matrices, target_dim=max_dim)

        cumvar = pipe.variance_explained
        analysis[tname] = {
            "raw_dim": pipe.raw_dim,
            "var_at_64": float(cumvar[min(63, len(cumvar) - 1)]),
            "var_at_128": float(cumvar[min(127, len(cumvar) - 1)]),
            "var_at_256": float(cumvar[min(255, len(cumvar) - 1)]),
            "var_at_512": float(cumvar[min(511, len(cumvar) - 1)]) if len(cumvar) > 511 else float(cumvar[-1]),
        }
        print(
            f"  {tname} (raw={pipe.raw_dim}d): "
            f"@64={analysis[tname]['var_at_64']:.3f}, "
            f"@128={analysis[tname]['var_at_128']:.3f}, "
            f"@256={analysis[tname]['var_at_256']:.3f}, "
            f"@512={analysis[tname]['var_at_512']:.3f}"
        )

    result = {"name": result_name, "analysis": analysis}
    results.append(result)
    save_results(results)

    return results


def step_E4(results, compressed, split):
    """E4: Feature group importance — which components contribute to top PCA dims."""
    print("\n═══ E4: Feature Importance Analysis ═══")

    result_name = "E4_feature_importance"
    if is_done(results, result_name):
        print("  Already done, skipping")
        return results

    train_set = set(split["train_ids"])
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}
    d = next(iter(train_matrices.values())).shape[1]  # 256

    pipe = EnrichedTransformPipeline(moment_pool)
    pipe.fit(train_matrices, target_dim=256)

    # Analyze PCA loadings: which feature groups load onto top components
    # moment_pool layout: [mean(d) | std(d) | skew(d) | autocov(d) | half_diff(d)]
    components = pipe.pca.components_  # (256, 1280)
    feature_groups = {
        "mean": (0, d),
        "std": (d, 2 * d),
        "skewness": (2 * d, 3 * d),
        "lag1_autocov": (3 * d, 4 * d),
        "half_diff": (4 * d, 5 * d),
    }

    importance = {}
    for group, (start, end) in feature_groups.items():
        # Total squared loading on top 10 components
        loading = float(np.sum(components[:10, start:end] ** 2))
        # Total squared loading on all components
        loading_all = float(np.sum(components[:, start:end] ** 2))
        importance[group] = {
            "top10_loading": round(loading, 4),
            "total_loading": round(loading_all, 4),
        }

    print("  Feature group loadings on PCA (moment_pool):")
    for group, vals in sorted(importance.items(), key=lambda x: -x[1]["top10_loading"]):
        print(
            f"    {group:<15s}: top10={vals['top10_loading']:.4f}, "
            f"total={vals['total_loading']:.4f}"
        )

    result = {"name": result_name, "moment_pool_importance": importance}
    results.append(result)
    save_results(results)

    return results


def step_E5(results, compressed, metadata, split):
    """E5: Statistical significance test — best enriched vs mean."""
    print("\n═══ E5: Statistical Significance ═══")

    result_name = "E5_significance"
    if is_done(results, result_name):
        print("  Already done, skipping")
        return results

    # Find best enriched method from E1
    e1_results = [r for r in results if r["name"].startswith("E1_") and r["name"] != "E1_mean_baseline"]
    if not e1_results:
        print("  No E1 results yet, skipping")
        return results

    best = max(e1_results, key=lambda r: r.get("precision@1", 0))
    best_name = best["transform"]
    best_dim = best["target_dim"]
    print(f"  Best enriched: {best_name} at d={best_dim} (Ret@1={best['precision@1']:.3f})")

    train_set = set(split["train_ids"])
    test_ids = [pid for pid in split["test_ids"] if pid in compressed]
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}

    # Re-compute per-query results for the best enriched method
    transform_map = {
        "moment_pool": (moment_pool, {}),
        "autocov_pool": (autocovariance_pool, {}),
        "gram_features": (gram_features, {}),
        "dct_K8_pca": (dct_pool, {"K": 8}),
        "dct_K16_pca": (dct_pool, {"K": 16}),
        "haar_L3_pca": (haar_pool, {"levels": 3}),
    }
    tfn, tkwargs = transform_map[best_name]
    pipe = EnrichedTransformPipeline(tfn, tkwargs)
    pipe.fit(train_matrices, target_dim=best_dim)
    enriched_vectors = pipe.transform_batch({k: compressed[k] for k in test_ids})

    # Mean baseline vectors (match Exp 18 protocol: test-only database)
    mean_vectors = {pid: compressed[pid].mean(axis=0) for pid in test_ids}

    # Per-query Ret@1 for both methods
    id_to_label = {m["id"]: m["family"] for m in metadata if "family" in m}

    def per_query_precision(vectors, q_ids, db_ids):
        db_matrix = np.array([vectors[pid] for pid in db_ids])
        db_norms = np.linalg.norm(db_matrix, axis=1, keepdims=True).clip(1e-8)
        db_matrix = db_matrix / db_norms
        db_labels = [id_to_label[pid] for pid in db_ids]
        db_id_to_idx = {pid: i for i, pid in enumerate(db_ids)}

        scores = []
        for qid in q_ids:
            q = vectors[qid]
            q = q / np.linalg.norm(q).clip(1e-8)
            sims = q @ db_matrix.T
            if qid in db_id_to_idx:
                sims[db_id_to_idx[qid]] = -np.inf
            top1_idx = np.argmax(sims)
            scores.append(1.0 if db_labels[top1_idx] == id_to_label[qid] else 0.0)
        return np.array(scores)

    valid_test = [pid for pid in test_ids if pid in id_to_label]

    enriched_scores = per_query_precision(enriched_vectors, valid_test, valid_test)
    mean_scores = per_query_precision(mean_vectors, valid_test, valid_test)

    # Paired permutation test
    observed_diff = enriched_scores.mean() - mean_scores.mean()
    n_perm = 10000
    rng = np.random.RandomState(42)
    diffs = enriched_scores - mean_scores
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = (diffs * signs).mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    result = {
        "name": result_name,
        "best_enriched": best_name,
        "best_dim": best_dim,
        "enriched_ret1": float(enriched_scores.mean()),
        "mean_ret1": float(mean_scores.mean()),
        "diff": float(observed_diff),
        "p_value": float(p_value),
        "n_queries": len(valid_test),
    }
    results.append(result)
    save_results(results)

    sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(
        f"  Enriched Ret@1={enriched_scores.mean():.3f} vs Mean Ret@1={mean_scores.mean():.3f}"
    )
    print(f"  Diff={observed_diff:+.4f}, p={p_value:.4f} ({sig})")

    return results


def step_E6_fisher(results, compressed, metadata, split):
    """E6: Fisher Vector encoding (requires GMM fitting)."""
    print("\n═══ E6: Fisher Vector Encoding ═══")

    train_set = set(split["train_ids"])
    test_ids = [pid for pid in split["test_ids"] if pid in compressed]
    train_matrices = {k: v for k, v in compressed.items() if k in train_set}

    # Fit GMM on subsample of train residues
    result_name = "E6_fisher_gmm_fit"
    n_components = 8

    if not is_done(results, result_name):
        print(f"  Fitting GMM with k={n_components} on train residues...")
        t0 = time.time()

        # Subsample residues: take up to 200 residues from each train protein
        all_residues = []
        rng = np.random.RandomState(42)
        for pid, mat in train_matrices.items():
            if mat.shape[0] > 200:
                idx = rng.choice(mat.shape[0], 200, replace=False)
                all_residues.append(mat[idx])
            else:
                all_residues.append(mat)

        residue_matrix = np.vstack(all_residues).astype(np.float64)
        print(f"  GMM training on {residue_matrix.shape[0]} residues × {residue_matrix.shape[1]}d")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=42,
            max_iter=100,
            n_init=3,
        )
        gmm.fit(residue_matrix)
        gmm_time = time.time() - t0
        print(f"  GMM fitted in {gmm_time:.1f}s")

        gmm_result = {
            "name": result_name,
            "n_components": n_components,
            "n_residues": residue_matrix.shape[0],
            "fit_time_s": round(gmm_time, 1),
        }
        results.append(gmm_result)
        save_results(results)
    else:
        print("  GMM already fitted, re-fitting for vectors...")
        # Re-fit since we don't cache the GMM object
        all_residues = []
        rng = np.random.RandomState(42)
        for pid, mat in train_matrices.items():
            if mat.shape[0] > 200:
                idx = rng.choice(mat.shape[0], 200, replace=False)
                all_residues.append(mat[idx])
            else:
                all_residues.append(mat)
        residue_matrix = np.vstack(all_residues).astype(np.float64)
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="diag",
            random_state=42, max_iter=100, n_init=3,
        )
        gmm.fit(residue_matrix)

    gmm_means = gmm.means_.astype(np.float32)
    gmm_covars = gmm.covariances_.astype(np.float32)
    gmm_weights = gmm.weights_.astype(np.float32)

    # Compute Fisher vectors for all proteins
    for dim in [256, 512]:
        result_name = f"E6_fisher_d{dim}"
        if is_done(results, result_name):
            print(f"  {result_name}: already done, skipping")
            continue

        print(f"  Computing Fisher vectors → PCA-{dim}...")
        t0 = time.time()

        # Compute raw Fisher vectors for train set
        train_raw = {}
        for pid, mat in train_matrices.items():
            train_raw[pid] = fisher_vector(mat, gmm_means, gmm_covars, gmm_weights)

        raw_dim = next(iter(train_raw.values())).shape[0]

        # Fit PCA on train Fisher vectors
        from sklearn.decomposition import PCA
        train_array = np.array(list(train_raw.values()))
        actual_dim = min(dim, raw_dim, train_array.shape[0])
        pca = PCA(n_components=actual_dim, random_state=42)
        pca.fit(train_array)

        # Transform test proteins (match Exp 18 protocol: database=test_ids)
        fv_vectors = {}
        for pid in test_ids:
            fv = fisher_vector(compressed[pid], gmm_means, gmm_covars, gmm_weights)
            fv_vectors[pid] = pca.transform(fv.reshape(1, -1))[0].astype(np.float32)

        fit_time = time.time() - t0
        var_explained = float(np.cumsum(pca.explained_variance_ratio_)[-1])

        ret = retrieval_from_vectors(
            fv_vectors, metadata,
            query_ids=test_ids,
            database_ids=test_ids,
        )

        result = {
            "name": result_name,
            "plm": "prot_t5_xl",
            "transform": f"fisher_k{n_components}",
            "target_dim": dim,
            "raw_dim": raw_dim,
            "variance_explained_total": var_explained,
            "fit_time_s": round(fit_time, 1),
            **{k: v for k, v in ret.items()},
        }
        results.append(result)
        save_results(results)
        print(
            f"    {result_name}: Ret@1={ret['precision@1']:.3f}, "
            f"MRR={ret['mrr']:.3f}, raw={raw_dim}→{actual_dim}d"
        )

    return results


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default=None)
    args = parser.parse_args()

    results = load_results()

    # Load ProtT5 data (needed for most steps)
    print("Loading ProtT5 data...")
    embeddings, metadata, split = load_prot_t5_data()
    device = get_device()
    compressed = get_compressed("prot_t5_xl", embeddings, device=device)
    del embeddings
    monitor()

    if args.step is None or args.step == "E1":
        results = step_E1(results, compressed, metadata, split)
        monitor()

    if args.step is None or args.step == "E2":
        results = step_E2(results, metadata, split)
        monitor()

    if args.step is None or args.step == "E3":
        results = step_E3(results, compressed, split)

    if args.step is None or args.step == "E4":
        results = step_E4(results, compressed, split)

    if args.step is None or args.step == "E5":
        results = step_E5(results, compressed, metadata, split)

    if args.step is None or args.step == "E6":
        results = step_E6_fisher(results, compressed, metadata, split)
        monitor()

    print("\n✓ Experiment 19 complete.")


if __name__ == "__main__":
    main()
