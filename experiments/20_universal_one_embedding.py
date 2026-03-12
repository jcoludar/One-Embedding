#!/usr/bin/env python3
"""Experiment 20: Universal One Embedding — HRR + Gap Closure.

Tests Holographic Reduced Representations and other training-free transforms
on RAW PLM per-residue embeddings (no compression step). Benchmarks a
universal "one embedding" that works for both per-protein (retrieval) and
per-residue (SS3 probe) tasks without learned weights.

Usage:
  uv run python experiments/20_universal_one_embedding.py --step U1   # Raw mean pool baselines
  uv run python experiments/20_universal_one_embedding.py --step U2   # HRR K=1 retrieval
  uv run python experiments/20_universal_one_embedding.py --step U3   # HRR per-residue recovery
  uv run python experiments/20_universal_one_embedding.py --step U4   # K-slot HRR retrieval
  uv run python experiments/20_universal_one_embedding.py --step U5   # DCT on raw embeddings
  uv run python experiments/20_universal_one_embedding.py --step U6   # Enriched pooling on raw
  uv run python experiments/20_universal_one_embedding.py --step U7   # Universal transforms
  uv run python experiments/20_universal_one_embedding.py --step U8   # Structural correlation
  uv run python experiments/20_universal_one_embedding.py --step U9   # Summary tables
  uv run python experiments/20_universal_one_embedding.py              # run all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.per_residue_tasks import evaluate_ss3_probe, load_cb513_csv
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.structural_validation import (
    compute_pairwise_tm_scores,
    fetch_pdb_structures,
    validate_embedding_vs_structure,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.enriched_transforms import (
    EnrichedTransformPipeline,
    autocovariance_pool,
    moment_pool,
)
from src.one_embedding.hrr import (
    hrr_decode,
    hrr_encode,
    hrr_kslot_decode,
    hrr_kslot_encode,
    hrr_per_protein,
    hrr_per_residue,
)
from src.one_embedding.transforms import dct_summary, inverse_dct
from src.one_embedding.universal_transforms import (
    kernel_mean_embedding,
    norm_weighted_mean,
    power_mean_pool,
    svd_spectrum,
)
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "universal_oe_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
STRUCTURES_DIR = DATA_DIR / "structures"
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
    """Load raw per-residue embeddings from H5."""
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
) -> dict[str, float]:
    """Run retrieval evaluation on test set."""
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )


def mean_cosine_sim(original: dict, decoded: dict) -> float:
    """Mean per-residue cosine similarity between original and decoded."""
    cos_sims = []
    for pid in decoded:
        if pid not in original:
            continue
        orig = original[pid]
        dec = decoded[pid]
        L = min(orig.shape[0], dec.shape[0])
        for i in range(L):
            n1 = np.linalg.norm(orig[i])
            n2 = np.linalg.norm(dec[i])
            if n1 > 1e-8 and n2 > 1e-8:
                cos_sims.append(np.dot(orig[i], dec[i]) / (n1 * n2))
    return float(np.mean(cos_sims)) if cos_sims else 0.0


# ── Steps ────────────────────────────────────────────────────────


def step_U1(results: list[dict]):
    """U1: Raw mean pool baselines (ProtT5 + ESM2)."""
    print("\n═══ U1: Raw Mean Pool Baselines ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    for plm, dim_name in [("prot_t5_xl", "1024"), ("esm2_650m", "1280")]:
        name = f"U1_mean_pool_{plm}"
        if is_done(results, name):
            print(f"  {plm} already done, skipping.")
            continue

        print(f"\n  Loading raw {plm} embeddings...")
        embeddings = load_raw_embeddings(plm)
        if not embeddings:
            continue

        # Mean pool
        vectors = {}
        for pid in test_ids:
            if pid in embeddings:
                emb = embeddings[pid][:MAX_LEN]
                vectors[pid] = emb.mean(axis=0)

        D = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids)

        result = {
            "name": name, "plm": plm, "transform": "mean_pool",
            "dim": D, **ret,
        }
        results.append(result)
        save_results(results)
        print(f"  {plm}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D}")

    monitor()


def step_U2(results: list[dict]):
    """U2: HRR K=1 on raw embeddings."""
    print("\n═══ U2: HRR K=1 Retrieval ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        name = f"U2_hrr_k1_{plm}"
        if is_done(results, name):
            print(f"  {plm} already done, skipping.")
            continue

        print(f"\n  Loading raw {plm} embeddings...")
        embeddings = load_raw_embeddings(plm)
        if not embeddings:
            continue

        print(f"  Computing HRR K=1 for {len(embeddings)} proteins...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids:
            if pid in embeddings:
                emb = embeddings[pid][:MAX_LEN]
                vectors[pid] = hrr_encode(emb)
        elapsed = time.time() - t0

        D = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids)

        result = {
            "name": name, "plm": plm, "transform": "hrr_k1",
            "dim": D, "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"  {plm}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D}, {elapsed:.1f}s")

    monitor()


def step_U3(results: list[dict]):
    """U3: HRR per-residue recovery — cosine sim + SS3 Q3 on CB513."""
    print("\n═══ U3: HRR Per-Residue Recovery ═══")

    name = "U3_hrr_per_residue"
    if is_done(results, name):
        print("  Already done, skipping.")
        return

    # Load CB513 labels
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    if not ss3_labels:
        print("  ERROR: CB513 data not found")
        return

    # Load raw ProtT5 CB513 embeddings
    embeddings = load_raw_embeddings("prot_t5_xl", "cb513")
    if not embeddings:
        return

    # Match protein IDs between CB513 labels and embeddings
    common_ids = sorted(set(embeddings.keys()) & set(ss3_labels.keys()))
    if not common_ids:
        print("  ERROR: No matching protein IDs between embeddings and CB513 labels")
        return

    print(f"  Found {len(common_ids)} CB513 proteins with ProtT5 embeddings")

    # Split into train/test (80/20)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(common_ids))
    n_train = int(0.8 * len(common_ids))
    train_ids = [common_ids[i] for i in perm[:n_train]]
    test_ids_cb = [common_ids[i] for i in perm[n_train:]]

    # 1. Raw baseline SS3
    print("  Computing raw SS3 baseline...")
    raw_ss3 = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids_cb)
    print(f"    Raw SS3 Q3 = {raw_ss3['q3']:.3f}")

    # 2. HRR encode → decode
    print("  HRR encode → decode...")
    decoded_k1 = {}
    for pid in common_ids:
        emb = embeddings[pid][:MAX_LEN]
        trace = hrr_encode(emb)
        decoded_k1[pid] = hrr_decode(trace, L=emb.shape[0])

    cos_sim_k1 = mean_cosine_sim(embeddings, decoded_k1)
    print(f"    HRR K=1 mean cosine sim = {cos_sim_k1:.3f}")

    # 3. SS3 on HRR-decoded
    hrr_ss3 = evaluate_ss3_probe(decoded_k1, ss3_labels, train_ids, test_ids_cb)
    print(f"    HRR K=1 decoded SS3 Q3 = {hrr_ss3['q3']:.3f}")

    # 4. K=8 slots
    print("  K=8 slot HRR encode → decode...")
    decoded_k8 = {}
    for pid in common_ids:
        emb = embeddings[pid][:MAX_LEN]
        slots = hrr_kslot_encode(emb, K=8)
        decoded_k8[pid] = hrr_kslot_decode(slots, L=emb.shape[0])

    cos_sim_k8 = mean_cosine_sim(embeddings, decoded_k8)
    print(f"    HRR K=8 mean cosine sim = {cos_sim_k8:.3f}")

    hrr_k8_ss3 = evaluate_ss3_probe(decoded_k8, ss3_labels, train_ids, test_ids_cb)
    print(f"    HRR K=8 decoded SS3 Q3 = {hrr_k8_ss3['q3']:.3f}")

    result = {
        "name": name, "plm": "prot_t5_xl",
        "raw_ss3_q3": raw_ss3["q3"],
        "hrr_k1_cos_sim": cos_sim_k1,
        "hrr_k1_ss3_q3": hrr_ss3["q3"],
        "hrr_k8_cos_sim": cos_sim_k8,
        "hrr_k8_ss3_q3": hrr_k8_ss3["q3"],
        "n_proteins": len(common_ids),
    }
    results.append(result)
    save_results(results)
    monitor()


def step_U4(results: list[dict]):
    """U4: K-slot HRR K=4,8,16 retrieval + per-residue recovery."""
    print("\n═══ U4: K-Slot HRR ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings = None  # lazy load

        for K in [4, 8, 16]:
            name = f"U4_hrr_k{K}_{plm}"
            if is_done(results, name):
                print(f"  {plm} K={K} already done, skipping.")
                continue

            if embeddings is None:
                print(f"\n  Loading raw {plm} embeddings...")
                embeddings = load_raw_embeddings(plm)
                if not embeddings:
                    break

            print(f"  Computing HRR K={K} for {plm}...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids:
                if pid in embeddings:
                    emb = embeddings[pid][:MAX_LEN]
                    vectors[pid] = hrr_per_protein(emb, K=K)
            elapsed = time.time() - t0

            D = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids)

            result = {
                "name": name, "plm": plm, "transform": f"hrr_k{K}",
                "dim": D, "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    K={K}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D}, {elapsed:.1f}s")

    monitor()


def step_U5(results: list[dict]):
    """U5: DCT on raw embeddings K=1,2,4,8 (gap closure)."""
    print("\n═══ U5: DCT on Raw Embeddings ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings = None

        for K in [1, 2, 4, 8]:
            name = f"U5_dct_k{K}_{plm}"
            if is_done(results, name):
                print(f"  {plm} DCT K={K} already done, skipping.")
                continue

            if embeddings is None:
                print(f"\n  Loading raw {plm} embeddings...")
                embeddings = load_raw_embeddings(plm)
                if not embeddings:
                    break

            print(f"  Computing DCT K={K} for {plm}...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids:
                if pid in embeddings:
                    emb = embeddings[pid][:MAX_LEN]
                    vectors[pid] = dct_summary(emb, K=K)
            elapsed = time.time() - t0

            D = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids)

            result = {
                "name": name, "plm": plm, "transform": f"dct_k{K}",
                "dim": D, "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    K={K}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D}, {elapsed:.1f}s")

    monitor()


def step_U6(results: list[dict]):
    """U6: Enriched pooling on raw embeddings — moments, autocovariance."""
    print("\n═══ U6: Enriched Pooling on Raw ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings = None

        for transform_name, transform_fn in [
            ("moment_pool", moment_pool),
            ("autocovariance_pool", autocovariance_pool),
        ]:
            name = f"U6_{transform_name}_{plm}"
            if is_done(results, name):
                print(f"  {plm} {transform_name} already done, skipping.")
                continue

            if embeddings is None:
                print(f"\n  Loading raw {plm} embeddings...")
                embeddings = load_raw_embeddings(plm)
                if not embeddings:
                    break

            print(f"  Computing {transform_name} for {plm}...")
            t0 = time.time()

            # Raw vectors (high dim)
            raw_vectors = {}
            for pid in list(set(train_ids + test_ids)):
                if pid in embeddings:
                    emb = embeddings[pid][:MAX_LEN]
                    raw_vectors[pid] = transform_fn(emb)

            raw_dim = next(iter(raw_vectors.values())).shape[0]
            print(f"    Raw dimension: {raw_dim}")

            # If dim > 2048, apply PCA to reasonable target
            D_orig = next(iter(embeddings.values())).shape[1]
            target_dim = D_orig  # e.g. 1024 for ProtT5, 1280 for ESM2

            if raw_dim > target_dim:
                print(f"    Applying PCA: {raw_dim} → {target_dim}")
                pipeline = EnrichedTransformPipeline(transform_fn=transform_fn)
                train_matrices = {pid: embeddings[pid][:MAX_LEN] for pid in train_ids if pid in embeddings}
                pipeline.fit(train_matrices, target_dim=target_dim)

                vectors = {}
                for pid in test_ids:
                    if pid in embeddings:
                        vectors[pid] = pipeline.transform(embeddings[pid][:MAX_LEN])
                D_final = target_dim
            else:
                vectors = {pid: raw_vectors[pid] for pid in test_ids if pid in raw_vectors}
                D_final = raw_dim

            elapsed = time.time() - t0
            ret = compute_retrieval(vectors, metadata, test_ids)

            result = {
                "name": name, "plm": plm, "transform": transform_name,
                "dim": D_final, "raw_dim": raw_dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    {transform_name}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D_final}")

    monitor()


def step_U7(results: list[dict]):
    """U7: Universal transforms — power mean, norm-weighted, kernel ME, SVD."""
    print("\n═══ U7: Universal Transforms ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    transforms = [
        ("power_mean_p2", lambda m: power_mean_pool(m, p=2.0)),
        ("power_mean_p3", lambda m: power_mean_pool(m, p=3.0)),
        ("norm_weighted", norm_weighted_mean),
        ("kernel_me_2048", lambda m: kernel_mean_embedding(m, D_out=2048)),
        ("svd_spectrum_16", lambda m: svd_spectrum(m, k=16)),
        ("svd_spectrum_64", lambda m: svd_spectrum(m, k=64)),
    ]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings = None

        for tf_name, tf_fn in transforms:
            name = f"U7_{tf_name}_{plm}"
            if is_done(results, name):
                print(f"  {plm} {tf_name} already done, skipping.")
                continue

            if embeddings is None:
                print(f"\n  Loading raw {plm} embeddings...")
                embeddings = load_raw_embeddings(plm)
                if not embeddings:
                    break

            print(f"  Computing {tf_name} for {plm}...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids:
                if pid in embeddings:
                    emb = embeddings[pid][:MAX_LEN]
                    vectors[pid] = tf_fn(emb)
            elapsed = time.time() - t0

            D = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids)

            result = {
                "name": name, "plm": plm, "transform": tf_name,
                "dim": D, "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    {tf_name}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={D}, {elapsed:.1f}s")

    monitor()


def step_U8(results: list[dict]):
    """U8: Structural correlation — TM-score vs embedding similarity."""
    print("\n═══ U8: Structural Correlation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    # Use a subset for TM-score computation (200 proteins, same as Exp 18)
    rng = np.random.RandomState(42)
    struct_ids = list(rng.choice(test_ids, size=min(200, len(test_ids)), replace=False))

    # Check for cached TM-scores
    tm_cache = STRUCTURES_DIR / "tm_scores_200.npz"

    # We need PDB structures
    print("  Fetching PDB structures...")
    pdb_paths = fetch_pdb_structures(struct_ids, cache_dir=STRUCTURES_DIR / "pdb")
    valid_ids = [pid for pid in struct_ids if pid in pdb_paths]
    print(f"  {len(valid_ids)} / {len(struct_ids)} structures available")

    if len(valid_ids) < 20:
        print("  ERROR: Too few structures for meaningful correlation")
        return

    # Compute TM-scores
    print("  Computing TM-scores (cached if available)...")
    tm_scores = compute_pairwise_tm_scores(valid_ids, pdb_paths, cache_path=tm_cache)

    # Load ProtT5 raw embeddings
    embeddings = load_raw_embeddings("prot_t5_xl")
    if not embeddings:
        return

    # Compute cosine similarity matrices for different transforms
    transform_configs = [
        ("mean_pool", lambda m: m.mean(axis=0)),
        ("hrr_k1", hrr_encode),
        ("norm_weighted", norm_weighted_mean),
        ("power_mean_p3", lambda m: power_mean_pool(m, p=3.0)),
    ]

    for tf_name, tf_fn in transform_configs:
        name = f"U8_struct_{tf_name}"
        if is_done(results, name):
            print(f"  {tf_name} already done, skipping.")
            continue

        print(f"\n  Computing {tf_name} structural correlation...")
        vectors = {}
        for pid in valid_ids:
            if pid in embeddings:
                emb = embeddings[pid][:MAX_LEN]
                vectors[pid] = tf_fn(emb)

        # Build similarity matrix
        ids_with_vec = [pid for pid in valid_ids if pid in vectors]
        n = len(ids_with_vec)
        vec_matrix = np.array([vectors[pid] for pid in ids_with_vec])
        norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True).clip(1e-8)
        vec_matrix = vec_matrix / norms
        emb_sim = vec_matrix @ vec_matrix.T

        # Reindex TM-score matrix
        id_to_idx = {pid: i for i, pid in enumerate(valid_ids)}
        tm_sub = np.zeros((n, n), dtype=np.float32)
        for i, pid_i in enumerate(ids_with_vec):
            for j, pid_j in enumerate(ids_with_vec):
                tm_sub[i, j] = tm_scores[id_to_idx[pid_i], id_to_idx[pid_j]]

        corr = validate_embedding_vs_structure(emb_sim, tm_sub, ids_with_vec, metadata)

        result = {
            "name": name, "plm": "prot_t5_xl", "transform": tf_name,
            "spearman_rho": corr.get("spearman_rho", 0),
            "pearson_r": corr.get("pearson_r", 0),
            "n_proteins": corr.get("n_proteins", 0),
            "n_valid_pairs": corr.get("n_valid_pairs", 0),
        }
        results.append(result)
        save_results(results)
        print(f"    {tf_name}: Spearman={corr.get('spearman_rho', 0):.3f}, "
              f"Pearson={corr.get('pearson_r', 0):.3f}")

    monitor()


def step_U9(results: list[dict]):
    """U9: Summary comparison tables."""
    print("\n═══ U9: Summary Tables ═══")

    # Per-protein retrieval table
    print("\n  ┌─────────────────────┬──────┬───────────────────────┬───────────────────────┐")
    print("  │ Transform           │ Dim  │ ProtT5 Ret@1 / MRR   │ ESM2 Ret@1 / MRR     │")
    print("  ├─────────────────────┼──────┼───────────────────────┼───────────────────────┤")

    # Collect results by transform
    transform_results = {}
    for r in results:
        if "precision@1" not in r:
            continue
        tf = r.get("transform", "")
        plm = r.get("plm", "")
        if tf not in transform_results:
            transform_results[tf] = {}
        transform_results[tf][plm] = r

    # Sort by ProtT5 Ret@1
    sorted_tfs = sorted(
        transform_results.keys(),
        key=lambda tf: transform_results[tf].get("prot_t5_xl", {}).get("precision@1", 0),
        reverse=True,
    )

    for tf in sorted_tfs:
        pt5 = transform_results[tf].get("prot_t5_xl", {})
        esm2 = transform_results[tf].get("esm2_650m", {})
        dim = pt5.get("dim", esm2.get("dim", "?"))

        pt5_str = f"{pt5.get('precision@1', 0):.3f} / {pt5.get('mrr', 0):.3f}" if pt5 else "  -  /  -  "
        esm2_str = f"{esm2.get('precision@1', 0):.3f} / {esm2.get('mrr', 0):.3f}" if esm2 else "  -  /  -  "

        print(f"  │ {tf:<19s} │ {str(dim):>4s} │ {pt5_str:>21s} │ {esm2_str:>21s} │")

    print("  └─────────────────────┴──────┴───────────────────────┴───────────────────────┘")

    # Per-residue recovery table (from U3)
    u3 = next((r for r in results if r.get("name") == "U3_hrr_per_residue"), None)
    if u3:
        print("\n  Per-Residue Recovery (ProtT5, CB513):")
        print("  ┌──────────────────┬──────────┬────────┐")
        print("  │ Method           │ CosSim   │ SS3 Q3 │")
        print("  ├──────────────────┼──────────┼────────┤")
        print(f"  │ Raw (baseline)   │  1.000   │ {u3['raw_ss3_q3']:.3f}  │")
        print(f"  │ HRR K=1 decoded  │  {u3['hrr_k1_cos_sim']:.3f}   │ {u3['hrr_k1_ss3_q3']:.3f}  │")
        print(f"  │ HRR K=8 decoded  │  {u3['hrr_k8_cos_sim']:.3f}   │ {u3['hrr_k8_ss3_q3']:.3f}  │")
        print("  └──────────────────┴──────────┴────────┘")

    # Structural correlation table (from U8)
    u8_results = [r for r in results if r.get("name", "").startswith("U8_struct_")]
    if u8_results:
        print("\n  Structural Correlation (ProtT5, TM-score):")
        print("  ┌──────────────────┬───────────┬──────────┐")
        print("  │ Transform        │ Spearman  │ Pearson  │")
        print("  ├──────────────────┼───────────┼──────────┤")
        for r in sorted(u8_results, key=lambda x: x.get("spearman_rho", 0), reverse=True):
            tf = r.get("transform", "")
            print(f"  │ {tf:<16s} │  {r.get('spearman_rho', 0):>6.3f}   │  {r.get('pearson_r', 0):>5.3f}  │")
        print("  └──────────────────┴───────────┴──────────┘")

    print("\n  U9 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 20: Universal One Embedding")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step (U1-U9)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "U1": step_U1, "U2": step_U2, "U3": step_U3, "U4": step_U4,
        "U5": step_U5, "U6": step_U6, "U7": step_U7, "U8": step_U8,
        "U9": step_U9,
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
