#!/usr/bin/env python3
"""Experiment 22: Path Geometry Codec — Topology meets Protein Embeddings.

Treats protein per-residue embeddings as discrete paths through R^D and applies
concepts from rough path theory, differential geometry, and polymer physics to
extract training-free features for both per-protein retrieval and per-residue SS3.

Usage:
  uv run python experiments/22_path_geometry.py --step P1   # Displacement DCT
  uv run python experiments/22_path_geometry.py --step P2   # Path signature depth 2
  uv run python experiments/22_path_geometry.py --step P3   # Path signature depth 3
  uv run python experiments/22_path_geometry.py --step P4   # Cross-covariance eigenspectrum
  uv run python experiments/22_path_geometry.py --step P5   # Curvature-enriched per-residue
  uv run python experiments/22_path_geometry.py --step P6   # Feature hash + path stats
  uv run python experiments/22_path_geometry.py --step P7   # Gyration tensor
  uv run python experiments/22_path_geometry.py --step P8   # Best-of-breed combinations
  uv run python experiments/22_path_geometry.py --step P9   # Summary table
  uv run python experiments/22_path_geometry.py              # run all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import dct, idct

from src.evaluation.per_residue_tasks import evaluate_ss3_probe, load_cb513_csv
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.path_transforms import (
    curvature_enriched,
    displacement_dct,
    displacement_encode,
    gyration_eigenspectrum,
    inverse_displacement_dct,
    lag_cross_covariance_eigenvalues,
    path_signature_depth2,
    path_signature_depth3,
    path_statistics,
    shape_descriptors,
)
from src.one_embedding.universal_transforms import feature_hash
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "path_geometry_results.json"
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


def compute_retrieval(vectors, metadata, test_ids):
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )


def load_cb513_data():
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    if not ss3_labels:
        print("  ERROR: CB513 data not found")
        return None, None, None, None, None
    embeddings = load_raw_embeddings("prot_t5_xl", "cb513")
    if not embeddings:
        return None, None, None, None, None
    common_ids = sorted(set(embeddings.keys()) & set(ss3_labels.keys()))
    if not common_ids:
        return None, None, None, None, None
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(common_ids))
    n_train = int(0.8 * len(common_ids))
    train_ids = [common_ids[i] for i in perm[:n_train]]
    test_ids = [common_ids[i] for i in perm[n_train:]]
    return embeddings, ss3_labels, common_ids, train_ids, test_ids


def mean_cosine_sim(original: dict, decoded: dict) -> float:
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


def step_P1(results: list[dict]):
    """P1: Displacement DCT — are path derivatives more compressible?

    Hypothesis: DCT of displacements compresses better than DCT of raw
    embeddings, because displacements are smoother. Like MPEG vs raw video.
    """
    print("\n═══ P1: Displacement DCT ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    # Per-protein retrieval
    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    D = next(iter(embeddings_5k.values())).shape[1]

    for K in [1, 2, 4, 8]:
        name = f"P1_disp_dct_K{K}_retrieval"
        if is_done(results, name):
            print(f"  Disp DCT K={K} retrieval already done, skipping.")
            continue

        print(f"\n  Displacement DCT K={K}: retrieval...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                vectors[pid] = displacement_dct(emb, K=K)
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)

        result = {
            "name": name, "plm": "prot_t5_xl", "transform": f"disp_dct_K{K}",
            "K": K, "dim": dim, "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    K={K}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    # Per-residue: displacement DCT inverse on CB513
    print("\n  Displacement DCT per-residue (CB513)...")
    cb_data = load_cb513_data()
    if cb_data[0] is not None:
        embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data

        for K in [4, 8, 16]:
            ss3_name = f"P1_disp_dct_K{K}_ss3"
            if is_done(results, ss3_name):
                print(f"  Disp DCT K={K} SS3 already done, skipping.")
                continue

            D_cb = next(iter(embeddings_cb.values())).shape[1]
            decoded = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                L = emb.shape[0]
                coeffs = displacement_dct(emb, K=K)
                decoded[pid] = inverse_displacement_dct(coeffs, D=D_cb, target_len=L, x0=emb[0])

            cos_sim = mean_cosine_sim(embeddings_cb, decoded)
            ss3 = evaluate_ss3_probe(decoded, ss3_labels, train_ids_cb, test_ids_cb)

            result = {
                "name": ss3_name, "plm": "prot_t5_xl",
                "transform": f"disp_dct_inverse_K{K}",
                "K": K, "cos_sim": cos_sim, "ss3_q3": ss3["q3"],
            }
            results.append(result)
            save_results(results)
            print(f"    K={K}: CosSim={cos_sim:.3f}, SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_P2(results: list[dict]):
    """P2: Path Signature depth 2 — universal nonlinear path feature.

    Random project (L, D) → (L, p), then depth-2 signature.
    Also test: signature concatenated with mean pool of projected residues.
    """
    print("\n═══ P2: Path Signature Depth 2 ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    D = next(iter(embeddings_5k.values())).shape[1]

    for p in [16, 24, 32]:
        # Generate fixed random projection
        rng = np.random.RandomState(42)
        R = rng.randn(D, p).astype(np.float32)
        Q, _ = np.linalg.qr(R, mode="reduced")
        proj = Q * np.sqrt(D / p)

        # Signature-only retrieval
        sig_name = f"P2_sig2_p{p}_retrieval"
        if not is_done(results, sig_name):
            print(f"\n  Path signature depth 2, p={p}: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    projected = (emb @ proj).astype(np.float32)
                    vectors[pid] = path_signature_depth2(projected)
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)

            result = {
                "name": sig_name, "plm": "prot_t5_xl",
                "transform": f"sig2_p{p}", "p": p, "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    sig2 p={p}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

        # Signature + mean pool concatenation
        concat_name = f"P2_sig2_p{p}_plus_mean_retrieval"
        if not is_done(results, concat_name):
            print(f"  Signature p={p} + mean pool: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    projected = (emb @ proj).astype(np.float32)
                    sig = path_signature_depth2(projected)
                    mean_vec = emb.mean(axis=0)
                    vectors[pid] = np.concatenate([mean_vec, sig])
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)

            result = {
                "name": concat_name, "plm": "prot_t5_xl",
                "transform": f"sig2_p{p}_plus_mean", "p": p, "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    sig2_p{p}+mean: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    # Per-residue: projected residues → SS3
    print("\n  Projected residues per-residue (CB513)...")
    cb_data = load_cb513_data()
    if cb_data[0] is not None:
        embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data
        D_cb = next(iter(embeddings_cb.values())).shape[1]

        for p in [16, 32, 64]:
            ss3_name = f"P2_proj_p{p}_ss3"
            if is_done(results, ss3_name):
                continue

            rng = np.random.RandomState(42)
            R = rng.randn(D_cb, p).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            proj = Q * np.sqrt(D_cb / p)

            proj_emb = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                proj_emb[pid] = (emb @ proj).astype(np.float32)

            ss3 = evaluate_ss3_probe(proj_emb, ss3_labels, train_ids_cb, test_ids_cb)
            result = {
                "name": ss3_name, "plm": "prot_t5_xl",
                "transform": f"random_proj_p{p}", "p": p, "ss3_q3": ss3["q3"],
            }
            results.append(result)
            save_results(results)
            print(f"    Proj p={p}: SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_P3(results: list[dict]):
    """P3: Path Signature depth 3 — does third-order structure help?"""
    print("\n═══ P3: Path Signature Depth 3 ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    D = next(iter(embeddings_5k.values())).shape[1]
    p = 16  # Only p=16 practical for depth 3 (4369 dims)

    rng = np.random.RandomState(42)
    R = rng.randn(D, p).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    proj = Q * np.sqrt(D / p)

    name = "P3_sig3_p16_retrieval"
    if not is_done(results, name):
        print(f"\n  Path signature depth 3, p={p}: retrieval...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                projected = (emb @ proj).astype(np.float32)
                vectors[pid] = path_signature_depth3(projected)
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)

        result = {
            "name": name, "plm": "prot_t5_xl",
            "transform": "sig3_p16", "p": p, "dim": dim,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    sig3 p=16: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    # Sig3 + mean pool
    concat_name = "P3_sig3_p16_plus_mean_retrieval"
    if not is_done(results, concat_name):
        print(f"  Signature depth 3 + mean pool: retrieval...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                projected = (emb @ proj).astype(np.float32)
                sig = path_signature_depth3(projected)
                mean_vec = emb.mean(axis=0)
                vectors[pid] = np.concatenate([mean_vec, sig])
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)

        result = {
            "name": concat_name, "plm": "prot_t5_xl",
            "transform": "sig3_p16_plus_mean", "p": p, "dim": dim,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    sig3_p16+mean: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    monitor()


def step_P4(results: list[dict]):
    """P4: Lag-1 cross-covariance eigenspectrum — captures ordering info."""
    print("\n═══ P4: Cross-Covariance Eigenspectrum ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    for k in [32, 64, 128]:
        # Eigenvalues only
        eig_name = f"P4_xcov_k{k}_retrieval"
        if not is_done(results, eig_name):
            print(f"\n  Cross-cov top-{k} eigenvalues: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    vectors[pid] = lag_cross_covariance_eigenvalues(emb, k=k)
            elapsed = time.time() - t0

            ret = compute_retrieval(vectors, metadata, test_ids_5k)
            result = {
                "name": eig_name, "plm": "prot_t5_xl",
                "transform": f"xcov_k{k}", "k": k, "dim": k,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    xcov k={k}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}")

        # Eigenvalues + mean pool concatenation
        concat_name = f"P4_xcov_k{k}_plus_mean_retrieval"
        if not is_done(results, concat_name):
            print(f"  Cross-cov k={k} + mean pool: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    eig = lag_cross_covariance_eigenvalues(emb, k=k)
                    mean_vec = emb.mean(axis=0)
                    vectors[pid] = np.concatenate([mean_vec, eig])
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)
            result = {
                "name": concat_name, "plm": "prot_t5_xl",
                "transform": f"xcov_k{k}_plus_mean", "k": k, "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    xcov_k{k}+mean: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    monitor()


def step_P5(results: list[dict]):
    """P5: Curvature-enriched per-residue — geometric context for SS3."""
    print("\n═══ P5: Curvature-Enriched Per-Residue ═══")

    cb_data = load_cb513_data()
    if cb_data[0] is None:
        return
    embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data

    # Curvature-enriched SS3
    name = "P5_curvature_enriched_ss3"
    if not is_done(results, name):
        print("  Curvature-enriched per-residue: SS3...")
        enriched_emb = {}
        for pid in common_ids:
            emb = embeddings_cb[pid][:MAX_LEN]
            enriched_emb[pid] = curvature_enriched(emb)

        ss3 = evaluate_ss3_probe(enriched_emb, ss3_labels, train_ids_cb, test_ids_cb)
        D_new = next(iter(enriched_emb.values())).shape[1]

        result = {
            "name": name, "plm": "prot_t5_xl",
            "transform": "curvature_enriched", "dim": D_new,
            "ss3_q3": ss3["q3"],
        }
        results.append(result)
        save_results(results)
        print(f"    Curvature-enriched: SS3 Q3={ss3['q3']:.3f} (dim={D_new})")

    # Curvature-enriched per-protein retrieval
    ret_name = "P5_curvature_enriched_retrieval"
    if not is_done(results, ret_name):
        print("  Curvature-enriched: retrieval...")
        metadata = load_metadata()
        split = load_split()
        test_ids_5k = split["test_ids"]

        embeddings_5k = load_raw_embeddings("prot_t5_xl")
        if embeddings_5k:
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    enriched = curvature_enriched(emb)
                    vectors[pid] = enriched.mean(axis=0)
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)
            result = {
                "name": ret_name, "plm": "prot_t5_xl",
                "transform": "curvature_enriched_mean", "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    Curvature+mean pool: Ret@1={ret['precision@1']:.3f}, dim={dim}")

    monitor()


def step_P6(results: list[dict]):
    """P6: Feature hash + path statistics — the pragmatic composite codec."""
    print("\n═══ P6: Feature Hash + Path Statistics ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    for d_hash in [256, 512]:
        # Path stats on feature-hashed embeddings
        name = f"P6_fhash_d{d_hash}_pathstats_retrieval"
        if not is_done(results, name):
            print(f"\n  Feature hash d={d_hash} + path stats: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    hashed = feature_hash(emb, d_out=d_hash)
                    mean_h = hashed.mean(axis=0)
                    std_h = hashed.std(axis=0)
                    pstats = path_statistics(hashed)
                    vectors[pid] = np.concatenate([mean_h, std_h, pstats])
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)

            result = {
                "name": name, "plm": "prot_t5_xl",
                "transform": f"fhash_d{d_hash}_pathstats",
                "d_hash": d_hash, "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    d={d_hash}+pathstats: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

        # Mean + std + path stats on RAW embeddings (no hash)
        raw_name = f"P6_raw_pathstats_retrieval"
        if not is_done(results, raw_name):
            print(f"\n  Raw mean + std + path stats: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    mean_r = emb.mean(axis=0)
                    std_r = emb.std(axis=0)
                    pstats = path_statistics(emb)
                    vectors[pid] = np.concatenate([mean_r, std_r, pstats])
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)

            result = {
                "name": raw_name, "plm": "prot_t5_xl",
                "transform": "raw_mean_std_pathstats", "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    raw+pathstats: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, dim={dim}")

    monitor()


def step_P7(results: list[dict]):
    """P7: Gyration tensor eigenspectrum + shape descriptors."""
    print("\n═══ P7: Gyration Tensor ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    for k in [32, 64]:
        # Gyration eigenvalues only
        eig_name = f"P7_gyration_k{k}_retrieval"
        if not is_done(results, eig_name):
            print(f"\n  Gyration top-{k} eigenvalues: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    vectors[pid] = gyration_eigenspectrum(emb, k=k)
            elapsed = time.time() - t0

            ret = compute_retrieval(vectors, metadata, test_ids_5k)
            result = {
                "name": eig_name, "plm": "prot_t5_xl",
                "transform": f"gyration_k{k}", "k": k, "dim": k,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    gyration k={k}: Ret@1={ret['precision@1']:.3f}")

        # Gyration + shape descriptors + mean pool
        concat_name = f"P7_gyration_k{k}_plus_mean_retrieval"
        if not is_done(results, concat_name):
            print(f"  Gyration k={k} + shape + mean pool: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    eig = gyration_eigenspectrum(emb, k=k)
                    sd = shape_descriptors(emb)
                    mean_vec = emb.mean(axis=0)
                    vectors[pid] = np.concatenate([mean_vec, eig, sd])
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)
            result = {
                "name": concat_name, "plm": "prot_t5_xl",
                "transform": f"gyration_k{k}_shape_plus_mean",
                "k": k, "dim": dim,
                "encode_time_s": round(elapsed, 2), **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    gyration_k{k}+shape+mean: Ret@1={ret['precision@1']:.3f}, dim={dim}")

    monitor()


def step_P8(results: list[dict]):
    """P8: Best-of-breed combinations from P1-P7."""
    print("\n═══ P8: Best-of-Breed Combinations ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    D = next(iter(embeddings_5k.values())).shape[1]

    # Combo 1: Mean pool + sig2 p=32 + xcov k=64
    name = "P8_mean_sig2p32_xcov64"
    if not is_done(results, name):
        print("\n  Combo: mean + sig2_p32 + xcov_k64...")
        rng = np.random.RandomState(42)
        R = rng.randn(D, 32).astype(np.float32)
        Q, _ = np.linalg.qr(R, mode="reduced")
        proj = Q * np.sqrt(D / 32)

        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                mean_vec = emb.mean(axis=0)
                projected = (emb @ proj).astype(np.float32)
                sig = path_signature_depth2(projected)
                xcov = lag_cross_covariance_eigenvalues(emb, k=64)
                vectors[pid] = np.concatenate([mean_vec, sig, xcov])
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)
        result = {
            "name": name, "plm": "prot_t5_xl",
            "transform": "mean+sig2_p32+xcov_k64", "dim": dim,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    mean+sig2+xcov: Ret@1={ret['precision@1']:.3f}, dim={dim}")

    # Combo 2: Mean + std + path_stats (raw) + gyration k=32
    name = "P8_mean_std_pathstats_gyration"
    if not is_done(results, name):
        print("  Combo: mean + std + path_stats + gyration_k32...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                mean_vec = emb.mean(axis=0)
                std_vec = emb.std(axis=0)
                pstats = path_statistics(emb)
                eig = gyration_eigenspectrum(emb, k=32)
                sd = shape_descriptors(emb)
                vectors[pid] = np.concatenate([mean_vec, std_vec, pstats, eig, sd])
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)
        result = {
            "name": name, "plm": "prot_t5_xl",
            "transform": "mean+std+pathstats+gyration+shape", "dim": dim,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    kitchen_sink: Ret@1={ret['precision@1']:.3f}, dim={dim}")

    # Combo 3: Displacement DCT K=4 + signature (hybrid path codec)
    name = "P8_disp_dct_K4_plus_sig2_p32"
    if not is_done(results, name):
        print("  Combo: disp_dct_K4 + sig2_p32...")
        rng = np.random.RandomState(42)
        R = rng.randn(D, 32).astype(np.float32)
        Q, _ = np.linalg.qr(R, mode="reduced")
        proj = Q * np.sqrt(D / 32)

        t0 = time.time()
        vectors = {}
        for pid in test_ids_5k:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                dct_vec = displacement_dct(emb, K=4)
                projected = (emb @ proj).astype(np.float32)
                sig = path_signature_depth2(projected)
                vectors[pid] = np.concatenate([dct_vec, sig])
        elapsed = time.time() - t0

        dim = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids_5k)
        result = {
            "name": name, "plm": "prot_t5_xl",
            "transform": "disp_dct_K4+sig2_p32", "dim": dim,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    disp_dct+sig2: Ret@1={ret['precision@1']:.3f}, dim={dim}")

    monitor()


def step_P9(results: list[dict]):
    """P9: Summary comparison table."""
    print("\n═══ P9: Summary ═══")

    print("\n  ═══ Per-Protein Retrieval (ProtT5, SCOPe 5K) ═══")
    print("  ┌──────────────────────────────────┬──────┬────────┬────────┐")
    print("  │ Method                            │ Dim  │ Ret@1  │ MRR    │")
    print("  ├──────────────────────────────────┼──────┼────────┼────────┤")

    ret_results = [r for r in results if "precision@1" in r]
    ret_results.sort(key=lambda r: r.get("precision@1", 0), reverse=True)
    for r in ret_results:
        tf = r.get("transform", "?")
        dim = r.get("dim", "?")
        ret1 = r.get("precision@1", 0)
        mrr = r.get("mrr", 0)
        print(f"  │ {tf:<34s} │ {str(dim):>4s} │ {ret1:.3f}  │ {mrr:.3f}  │")
    print("  └──────────────────────────────────┴──────┴────────┴────────┘")

    print("\n  ═══ Per-Residue SS3 Q3 (ProtT5, CB513) ═══")
    print("  ┌──────────────────────────────────┬──────────┬────────┐")
    print("  │ Method                            │ CosSim   │ SS3 Q3 │")
    print("  ├──────────────────────────────────┼──────────┼────────┤")

    ss3_results = [r for r in results if "ss3_q3" in r]
    ss3_results.sort(key=lambda r: r.get("ss3_q3", 0), reverse=True)
    for r in ss3_results:
        tf = r.get("transform", "?")
        cos = r.get("cos_sim", "?")
        q3 = r.get("ss3_q3", 0)
        cos_str = f"{cos:.3f}" if isinstance(cos, float) else str(cos)
        print(f"  │ {tf:<34s} │ {cos_str:>8s} │ {q3:.3f}  │")
    print("  └──────────────────────────────────┴──────────┴────────┘")

    print("\n  Baselines:")
    print("    Raw mean pool:     Ret@1=0.734, SS3 Q3=0.845")
    print("    Hybrid K4 d256:    Ret@1=0.779, SS3 Q3=0.499")
    print("    Feature hash d512: Ret@1=0.738, SS3 Q3=0.807")
    print("    Trained ChannelC:  Ret@1=0.808, SS3 Q3=0.834")
    print("    Moment pool+PCA:   Ret@1=0.800 (disqualified)")

    print("\n  P9 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 22: Path Geometry Codec")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step (P1-P9)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "P1": step_P1, "P2": step_P2, "P3": step_P3, "P4": step_P4,
        "P5": step_P5, "P6": step_P6, "P7": step_P7, "P8": step_P8,
        "P9": step_P9,
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
