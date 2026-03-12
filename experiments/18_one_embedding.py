#!/usr/bin/env python3
"""Experiment 18: One Embedding — Unified Protein Representation.

Compares mathematical transforms (DCT, spectral fingerprint, Haar wavelet)
against mean pooling for protein-level retrieval, per-residue probes, and
structural similarity correlation (TM-score vs PDB).

Usage:
  uv run python experiments/18_one_embedding.py --step O1  # Build One Embeddings
  uv run python experiments/18_one_embedding.py --step O2  # Retrieval comparison
  uv run python experiments/18_one_embedding.py --step O3  # Per-residue equivalence
  uv run python experiments/18_one_embedding.py --step O4  # Late interaction
  uv run python experiments/18_one_embedding.py --step O5  # Cross-PLM (ESM2)
  uv run python experiments/18_one_embedding.py --step O6  # TM-score validation
  uv run python experiments/18_one_embedding.py --step O7  # Storage efficiency
  uv run python experiments/18_one_embedding.py             # run all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.splitting import load_split
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.embedding import OneEmbedding
from src.one_embedding.io import load_one_embeddings, save_one_embeddings
from src.one_embedding.pipeline import apply_transform, compress_embeddings
from src.one_embedding.registry import PLMRegistry
from src.one_embedding.transforms import (
    dct_summary,
    haar_summary,
    inverse_dct,
    inverse_haar,
    spectral_fingerprint,
    spectral_moments,
)
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "one_embedding_results.json"
SPLIT_DIR = DATA_DIR / "splits"
ONE_EMB_DIR = DATA_DIR / "one_embeddings"
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


def retrieval_from_vectors(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    k_values: list[int] | None = None,
    query_ids: list[str] | None = None,
    database_ids: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval from pre-computed vectors (any dimensionality).

    Same logic as src.evaluation.retrieval.evaluate_retrieval but accepts
    pre-computed vectors directly instead of running a model.
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

    n_queries = len(q_ids)
    for k in k_values:
        results[f"precision@{k}"] /= n_queries
    results["mrr"] = mrr_sum / n_queries
    results["n_queries"] = n_queries
    results["n_database"] = len(db_ids)

    return results


# ── Data Loading ─────────────────────────────────────────────────


def load_prot_t5_data():
    """Load ProtT5 embeddings, metadata, and split."""
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
    """Load ESM2-650M embeddings, metadata, and split."""
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
    """Compress raw embeddings using trained ChannelCompressor."""
    registry = PLMRegistry(checkpoint_base=CHECKPOINTS_DIR)
    model = registry.get_compressor(plm_name, device=device)
    return compress_embeddings(model, embeddings, device=device, max_len=MAX_LEN)


# ── Steps ────────────────────────────────────────────────────────


def step_O1(results: list[dict]):
    """Build One Embeddings from all transforms for ProtT5."""
    print("\n═══ O1: Build One Embeddings (all transforms, ProtT5) ═══")

    embeddings, metadata, split = load_prot_t5_data()
    device = get_device()

    # Compress with trained ChannelCompressor
    print(f"  Compressing {len(embeddings)} proteins with ProtT5 compressor...")
    t0 = time.time()
    compressed = get_compressed("prot_t5_xl", embeddings, device=device)
    print(f"  Compression done in {time.time() - t0:.1f}s")
    monitor()

    # Define transform configs
    configs = [
        ("mean", {}),
        ("dct_K1", {"transform": "dct", "K": 1}),
        ("dct_K2", {"transform": "dct", "K": 2}),
        ("dct_K4", {"transform": "dct", "K": 4}),
        ("dct_K8", {"transform": "dct", "K": 8}),
        ("dct_K16", {"transform": "dct", "K": 16}),
        ("spectral_4bands", {"transform": "spectral", "n_bands": 4}),
        ("spectral_8bands", {"transform": "spectral", "n_bands": 8}),
        ("spectral_16bands", {"transform": "spectral", "n_bands": 16}),
        ("spectral_moments", {"transform": "spectral_moments"}),
        ("haar_L1", {"transform": "haar", "haar_levels": 1}),
        ("haar_L2", {"transform": "haar", "haar_levels": 2}),
        ("haar_L3", {"transform": "haar", "haar_levels": 3}),
        ("haar_L4", {"transform": "haar", "haar_levels": 4}),
    ]

    for cfg_name, cfg_kwargs in configs:
        name = f"O1_build_{cfg_name}"
        if is_done(results, name):
            print(f"  Skipping {cfg_name} (already done)")
            continue

        print(f"\n  Building: {cfg_name}...")
        transform = cfg_kwargs.get("transform", "mean")

        t0 = time.time()
        one_embeddings = {}
        summary_dims = set()

        for pid, matrix in compressed.items():
            summary = apply_transform(matrix, **cfg_kwargs) if cfg_kwargs else matrix.mean(axis=0)
            one_embeddings[pid] = OneEmbedding.from_compressed(
                protein_id=pid,
                plm="prot_t5_xl",
                matrix=matrix,
                transform=transform,
                summary=summary,
            )
            summary_dims.add(one_embeddings[pid].summary_dim)

        elapsed = time.time() - t0
        summary_dim = summary_dims.pop() if len(summary_dims) == 1 else list(summary_dims)

        # Save
        out_path = ONE_EMB_DIR / f"prot_t5_xl_{cfg_name}.h5"
        save_one_embeddings(one_embeddings, out_path)
        file_size_mb = out_path.stat().st_size / 1024 / 1024

        result = {
            "name": name,
            "plm": "prot_t5_xl",
            "transform": cfg_name,
            "summary_dim": summary_dim,
            "n_proteins": len(one_embeddings),
            "build_time_s": round(elapsed, 2),
            "file_size_mb": round(file_size_mb, 2),
        }
        results.append(result)
        save_results(results)
        print(f"    {cfg_name}: dim={summary_dim}, {file_size_mb:.1f} MB, {elapsed:.1f}s")

    print("\n  O1 complete.")


def step_O2(results: list[dict]):
    """Retrieval comparison across all transforms."""
    print("\n═══ O2: Retrieval Comparison ═══")

    _, metadata, split = load_prot_t5_data()
    test_ids = split["test_ids"]

    # Find all built One Embedding files
    if not ONE_EMB_DIR.exists():
        print("  ERROR: Run O1 first to build One Embeddings")
        return

    h5_files = sorted(ONE_EMB_DIR.glob("prot_t5_xl_*.h5"))
    if not h5_files:
        print("  ERROR: No One Embedding files found. Run O1 first.")
        return

    print(f"  Found {len(h5_files)} One Embedding variants")

    for h5_path in h5_files:
        cfg_name = h5_path.stem.replace("prot_t5_xl_", "")
        name = f"O2_retrieval_{cfg_name}"
        if is_done(results, name):
            print(f"  Skipping {cfg_name} (already done)")
            continue

        print(f"\n  Evaluating retrieval: {cfg_name}...")
        one_embeddings = load_one_embeddings(h5_path)

        # Get summary vectors for retrieval
        vectors = {pid: emb.summary for pid, emb in one_embeddings.items()}
        summary_dim = next(iter(vectors.values())).shape[0]

        ret = retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids,
        )

        result = {
            "name": name,
            "plm": "prot_t5_xl",
            "transform": cfg_name,
            "summary_dim": summary_dim,
            "precision@1": ret["precision@1"],
            "precision@3": ret.get("precision@3", 0),
            "precision@5": ret.get("precision@5", 0),
            "mrr": ret.get("mrr", 0),
            "n_queries": ret["n_queries"],
        }
        results.append(result)
        save_results(results)

        print(f"    {cfg_name}: Ret@1={ret['precision@1']:.3f}, "
              f"MRR={ret.get('mrr', 0):.3f}, dim={summary_dim}")

    # Print summary table
    print("\n  ┌─────────────────────────┬────────┬────────┬──────────┐")
    print("  │ Transform               │ Ret@1  │  MRR   │ Sum. Dim │")
    print("  ├─────────────────────────┼────────┼────────┼──────────┤")
    o2_results = [r for r in results if r["name"].startswith("O2_retrieval_")]
    o2_results.sort(key=lambda r: r["precision@1"], reverse=True)
    for r in o2_results:
        print(f"  │ {r['transform']:<23s} │ {r['precision@1']:.3f}  │ {r['mrr']:.3f}  │ {r['summary_dim']:>8} │")
    print("  └─────────────────────────┴────────┴────────┴──────────┘")


def step_O3(results: list[dict]):
    """Per-residue equivalence verification + inverse DCT quality."""
    print("\n═══ O3: Per-Residue Equivalence ═══")

    name = "O3_per_residue"
    if is_done(results, name):
        print("  Already done, skipping.")
        return

    # Load One Embedding (any variant — residues are identical across transforms)
    mean_path = ONE_EMB_DIR / "prot_t5_xl_mean.h5"
    if not mean_path.exists():
        print("  ERROR: Run O1 first")
        return

    one_embeddings = load_one_embeddings(mean_path)
    residue_dict = {pid: emb.residues for pid, emb in one_embeddings.items()}

    # Verify residue matrices match across transforms
    dct_path = ONE_EMB_DIR / "prot_t5_xl_dct_K8.h5"
    if dct_path.exists():
        dct_embeddings = load_one_embeddings(dct_path)
        mismatches = 0
        for pid in list(residue_dict.keys())[:100]:
            if pid in dct_embeddings:
                if not np.allclose(residue_dict[pid], dct_embeddings[pid].residues, atol=1e-6):
                    mismatches += 1
        print(f"  Residue matrix consistency across transforms: "
              f"{100-mismatches}% match ({mismatches} mismatches in 100 samples)")

    # Inverse DCT reconstruction quality
    print("\n  Inverse DCT reconstruction quality (from K coefficients):")
    result_data = {"name": name}

    for K in [1, 4, 8, 16]:
        mse_list, cos_list = [], []
        for pid in list(residue_dict.keys())[:200]:
            original = residue_dict[pid]
            L, d = original.shape
            coeffs = dct_summary(original, K=K)
            reconstructed = inverse_dct(coeffs, d=d, target_len=L)
            mse = float(np.mean((original - reconstructed) ** 2))
            # Per-residue cosine similarity
            dot_prods = np.sum(original * reconstructed, axis=1)
            orig_norms = np.linalg.norm(original, axis=1).clip(1e-8)
            recon_norms = np.linalg.norm(reconstructed, axis=1).clip(1e-8)
            cos = float(np.mean(dot_prods / (orig_norms * recon_norms)))
            mse_list.append(mse)
            cos_list.append(cos)

        avg_mse = float(np.mean(mse_list))
        avg_cos = float(np.mean(cos_list))
        print(f"    DCT K={K:>2d}: MSE={avg_mse:.4f}, CosSim={avg_cos:.4f}")
        result_data[f"dct_K{K}_mse"] = round(avg_mse, 6)
        result_data[f"dct_K{K}_cossim"] = round(avg_cos, 4)

    # Haar reconstruction quality
    print("\n  Haar wavelet reconstruction quality:")
    from src.one_embedding.transforms import haar_full_coefficients
    for levels in [3, 5, 7]:
        mse_list, cos_list = [], []
        for pid in list(residue_dict.keys())[:200]:
            original = residue_dict[pid]
            L, d = original.shape
            approx, details = haar_full_coefficients(original, levels=levels)
            reconstructed = inverse_haar(approx, details, target_len=L)
            mse = float(np.mean((original - reconstructed) ** 2))
            dot_prods = np.sum(original * reconstructed, axis=1)
            orig_norms = np.linalg.norm(original, axis=1).clip(1e-8)
            recon_norms = np.linalg.norm(reconstructed, axis=1).clip(1e-8)
            cos = float(np.mean(dot_prods / (orig_norms * recon_norms)))
            mse_list.append(mse)
            cos_list.append(cos)

        avg_mse = float(np.mean(mse_list))
        avg_cos = float(np.mean(cos_list))
        print(f"    Haar L={levels}: MSE={avg_mse:.8f}, CosSim={avg_cos:.6f}")
        result_data[f"haar_L{levels}_mse"] = round(avg_mse, 8)
        result_data[f"haar_L{levels}_cossim"] = round(avg_cos, 6)

    results.append(result_data)
    save_results(results)


def step_O4(results: list[dict]):
    """Late interaction retrieval using full residue matrices."""
    print("\n═══ O4: Late Interaction Retrieval ═══")

    name = "O4_late_interaction"
    if is_done(results, name):
        print("  Already done, skipping.")
        return

    mean_path = ONE_EMB_DIR / "prot_t5_xl_mean.h5"
    if not mean_path.exists():
        print("  ERROR: Run O1 first")
        return

    _, metadata, split = load_prot_t5_data()
    test_ids = split["test_ids"]

    one_embeddings = load_one_embeddings(mean_path)

    # Get residue matrices for test proteins
    test_embeddings = {pid: one_embeddings[pid].residues
                       for pid in test_ids if pid in one_embeddings}
    test_ids_valid = list(test_embeddings.keys())

    if len(test_ids_valid) < 10:
        print("  Too few test proteins, skipping")
        return

    # Use existing late interaction evaluator
    # We need a model-like wrapper — use ChannelCompressor with identity compression
    # Or compute directly
    from src.one_embedding.similarity import late_interaction_score

    print(f"  Computing late interaction scores for {len(test_ids_valid)} proteins...")
    print(f"  ({len(test_ids_valid) ** 2} pairs — this may take a while)")

    id_to_label = {m["id"]: m["family"] for m in metadata if "family" in m}

    # Precompute normalized residue matrices
    normed_residues = {}
    for pid in test_ids_valid:
        res = test_embeddings[pid]
        norms = np.linalg.norm(res, axis=1, keepdims=True).clip(1e-8)
        normed_residues[pid] = res / norms

    # Compute late interaction retrieval
    k_values = [1, 3, 5]
    precisions = {k: [] for k in k_values}
    mrr_sum = 0.0

    t0 = time.time()
    for qi, qid in enumerate(test_ids_valid):
        if qi % 50 == 0 and qi > 0:
            print(f"    Progress: {qi}/{len(test_ids_valid)}")

        q_res = normed_residues[qid]
        scores = []
        for did in test_ids_valid:
            if did == qid:
                scores.append(-np.inf)
            else:
                d_res = normed_residues[did]
                sim = q_res @ d_res.T
                scores.append(float(sim.max(axis=1).sum()))
        scores = np.array(scores)
        ranked = np.argsort(scores)[::-1]

        q_label = id_to_label.get(qid, "")
        for k in k_values:
            top_labels = [id_to_label.get(test_ids_valid[j], "") for j in ranked[:k]]
            correct = sum(1 for lbl in top_labels if lbl == q_label)
            precisions[k].append(correct / k)

        for rank, idx in enumerate(ranked, 1):
            if id_to_label.get(test_ids_valid[idx], "") == q_label:
                mrr_sum += 1.0 / rank
                break

    elapsed = time.time() - t0
    n = len(test_ids_valid)

    result = {
        "name": name,
        "method": "late_interaction",
        "precision@1": float(np.mean(precisions[1])),
        "precision@3": float(np.mean(precisions[3])),
        "precision@5": float(np.mean(precisions[5])),
        "mrr": mrr_sum / n,
        "n_queries": n,
        "elapsed_s": round(elapsed, 1),
    }
    results.append(result)
    save_results(results)

    print(f"  Late interaction: Ret@1={result['precision@1']:.3f}, "
          f"MRR={result['mrr']:.3f} ({elapsed:.0f}s)")


def step_O5(results: list[dict]):
    """Cross-PLM: repeat retrieval with ESM2-650M."""
    print("\n═══ O5: Cross-PLM (ESM2-650M) ═══")

    embeddings, metadata, split = load_esm2_data()
    test_ids = split["test_ids"]
    device = get_device()

    # Compress
    print(f"  Compressing {len(embeddings)} proteins with ESM2 compressor...")
    compressed = get_compressed("esm2_650m", embeddings, device=device)

    configs = [
        ("mean", {}),
        ("dct_K8", {"transform": "dct", "K": 8}),
        ("spectral_8bands", {"transform": "spectral", "n_bands": 8}),
        ("haar_L3", {"transform": "haar", "haar_levels": 3}),
    ]

    for cfg_name, cfg_kwargs in configs:
        name = f"O5_esm2_{cfg_name}"
        if is_done(results, name):
            print(f"  Skipping {cfg_name} (already done)")
            continue

        print(f"\n  Evaluating ESM2 {cfg_name}...")

        vectors = {}
        for pid, matrix in compressed.items():
            vectors[pid] = apply_transform(matrix, **cfg_kwargs) if cfg_kwargs else matrix.mean(axis=0)

        ret = retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids,
        )

        dim = next(iter(vectors.values())).shape[0]
        result = {
            "name": name,
            "plm": "esm2_650m",
            "transform": cfg_name,
            "summary_dim": dim,
            "precision@1": ret["precision@1"],
            "mrr": ret.get("mrr", 0),
            "n_queries": ret["n_queries"],
        }
        results.append(result)
        save_results(results)
        print(f"    {cfg_name}: Ret@1={ret['precision@1']:.3f}, MRR={ret.get('mrr', 0):.3f}")


def step_O6(results: list[dict]):
    """TM-score structural validation."""
    print("\n═══ O6: TM-Score Structural Validation ═══")

    from src.evaluation.structural_validation import (
        compute_pairwise_tm_scores,
        extract_pdb_info,
        fetch_pdb_structures,
        validate_embedding_vs_structure,
    )

    name_base = "O6_tm_score"

    _, metadata, split = load_prot_t5_data()
    test_ids = split["test_ids"]

    # Subsample for O(n²) computation — use 200 proteins
    np.random.seed(42)
    if len(test_ids) > 200:
        test_subset = list(np.random.choice(test_ids, size=200, replace=False))
    else:
        test_subset = test_ids
    print(f"  Using {len(test_subset)} proteins for TM-score validation")

    # Step 1: Fetch PDB structures
    print("  Fetching PDB structures from RCSB...")
    pdb_cache = STRUCTURES_DIR / "pdb"
    pdb_paths = fetch_pdb_structures(test_subset, cache_dir=pdb_cache)
    print(f"  Got {len(pdb_paths)}/{len(test_subset)} PDB structures")
    monitor()

    # Step 2: Compute pairwise TM-scores
    tm_cache = STRUCTURES_DIR / "tm_scores_200.npz"
    print("  Computing pairwise TM-scores (this may take several minutes)...")
    t0 = time.time()
    tm_scores = compute_pairwise_tm_scores(
        test_subset, pdb_paths, cache_path=tm_cache,
    )
    print(f"  TM-scores computed in {time.time() - t0:.0f}s")
    monitor()

    # Step 3: Compare embedding approaches vs TM-scores
    h5_files = sorted(ONE_EMB_DIR.glob("prot_t5_xl_*.h5"))
    approaches_to_test = [
        "mean", "dct_K4", "dct_K8", "dct_K16",
        "spectral_4bands", "spectral_8bands", "spectral_16bands",
        "spectral_moments", "haar_L2", "haar_L3",
    ]

    for h5_path in h5_files:
        cfg_name = h5_path.stem.replace("prot_t5_xl_", "")
        if cfg_name not in approaches_to_test:
            continue

        name = f"{name_base}_{cfg_name}"
        if is_done(results, name):
            print(f"  Skipping {cfg_name} (already done)")
            continue

        print(f"\n  Correlating {cfg_name} with TM-scores...")
        one_embeddings = load_one_embeddings(h5_path)

        # Get summary vectors for test subset
        vectors = {}
        for pid in test_subset:
            if pid in one_embeddings:
                vectors[pid] = one_embeddings[pid].summary

        if len(vectors) < 50:
            print(f"    Too few vectors ({len(vectors)}), skipping")
            continue

        # Compute embedding cosine similarity matrix
        ordered_ids = test_subset
        n = len(ordered_ids)
        emb_matrix = np.zeros((n, n), dtype=np.float32)

        vec_array = np.array([vectors.get(pid, np.zeros_like(next(iter(vectors.values()))))
                              for pid in ordered_ids])
        norms = np.linalg.norm(vec_array, axis=1, keepdims=True).clip(1e-8)
        vec_normed = vec_array / norms
        emb_matrix = vec_normed @ vec_normed.T

        val = validate_embedding_vs_structure(
            emb_matrix, tm_scores, ordered_ids, metadata=metadata,
        )

        result = {
            "name": name,
            "transform": cfg_name,
            **val,
        }
        results.append(result)
        save_results(results)

        print(f"    {cfg_name}: Spearman ρ={val.get('spearman_rho', 0):.3f} "
              f"(p={val.get('spearman_p', 1):.2e}), "
              f"Pearson r={val.get('pearson_r', 0):.3f}")

    # Print summary
    print("\n  ┌──────────────────────┬──────────┬──────────┐")
    print("  │ Transform            │ Spearman │ Pearson  │")
    print("  ├──────────────────────┼──────────┼──────────┤")
    o6_results = [r for r in results if r["name"].startswith(name_base)]
    o6_results.sort(key=lambda r: r.get("spearman_rho", 0), reverse=True)
    for r in o6_results:
        rho = r.get("spearman_rho", 0)
        pr = r.get("pearson_r", 0)
        print(f"  │ {r['transform']:<20s} │ {rho:>7.3f}  │ {pr:>7.3f}  │")
    print("  └──────────────────────┴──────────┴──────────┘")


def step_O7(results: list[dict]):
    """Storage efficiency analysis."""
    print("\n═══ O7: Storage Efficiency ═══")

    name = "O7_storage"
    if is_done(results, name):
        print("  Already done, skipping.")
        return

    # Original embeddings size
    orig_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    orig_size_mb = orig_path.stat().st_size / 1024 / 1024 if orig_path.exists() else 0

    # One Embedding sizes
    sizes = {}
    if ONE_EMB_DIR.exists():
        for h5 in ONE_EMB_DIR.glob("prot_t5_xl_*.h5"):
            cfg_name = h5.stem.replace("prot_t5_xl_", "")
            sizes[cfg_name] = h5.stat().st_size / 1024 / 1024

    print(f"  Original ProtT5 (L, 1024): {orig_size_mb:.1f} MB")
    for cfg, sz in sorted(sizes.items()):
        ratio = orig_size_mb / sz if sz > 0 else 0
        print(f"  One Embedding ({cfg}): {sz:.1f} MB ({ratio:.1f}x compression)")

    # Timing: encode 100 proteins
    mean_path = ONE_EMB_DIR / "prot_t5_xl_mean.h5"
    if mean_path.exists():
        one_embeddings = load_one_embeddings(mean_path)
        sample = dict(list(one_embeddings.items())[:100])
        matrices = {pid: emb.residues for pid, emb in sample.items()}

        timings = {}
        for tfm, kwargs in [
            ("mean", {}),
            ("dct_K8", {"transform": "dct", "K": 8}),
            ("spectral_8bands", {"transform": "spectral", "n_bands": 8}),
            ("haar_L3", {"transform": "haar", "haar_levels": 3}),
        ]:
            t0 = time.time()
            for _ in range(10):
                for pid, m in matrices.items():
                    apply_transform(m, **kwargs) if kwargs else m.mean(axis=0)
            timings[tfm] = (time.time() - t0) / 10 / len(matrices) * 1000  # ms per protein
            print(f"  Transform time ({tfm}): {timings[tfm]:.2f} ms/protein")

    result = {
        "name": name,
        "original_size_mb": round(orig_size_mb, 2),
        "one_embedding_sizes_mb": {k: round(v, 2) for k, v in sizes.items()},
    }
    results.append(result)
    save_results(results)


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 18: One Embedding")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step (O1-O7) or all if omitted")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "O1": step_O1,
        "O2": step_O2,
        "O3": step_O3,
        "O4": step_O4,
        "O5": step_O5,
        "O6": step_O6,
        "O7": step_O7,
    }

    if args.step:
        step = args.step.upper()
        if step not in steps:
            print(f"Unknown step: {step}. Available: {list(steps.keys())}")
            return
        steps[step](results)
    else:
        for step_name, step_fn in steps.items():
            step_fn(results)
            monitor()

    print("\n✓ Experiment 18 complete.")


if __name__ == "__main__":
    main()
