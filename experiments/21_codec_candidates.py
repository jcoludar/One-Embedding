#!/usr/bin/env python3
"""Experiment 21: Codec Candidates — Training-Free Universal Representation.

Tests candidate codecs that produce a single object from ANY PLM per-residue
embeddings, from which BOTH per-protein vectors (retrieval) and per-residue
matrices (SS3 probes) can be derived. Zero training, zero fitting.

Candidates:
  V1: DCT inverse per-residue quality (the CRITICAL gap from Exp 20)
  V2: Haar wavelet on raw embeddings — per-residue baseline
  V3: Feature hashing (D → d_out, PLM-universal)
  V4: Random orthogonal projection (D → d_out, JL guarantee)
  V5: Hybrid codec (feature hash + DCT = fixed-size storage)
  V6: Summary comparison table

Usage:
  uv run python experiments/21_codec_candidates.py --step V1
  uv run python experiments/21_codec_candidates.py --step V2
  uv run python experiments/21_codec_candidates.py --step V3
  uv run python experiments/21_codec_candidates.py --step V4
  uv run python experiments/21_codec_candidates.py --step V5
  uv run python experiments/21_codec_candidates.py --step V6
  uv run python experiments/21_codec_candidates.py              # run all
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
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.transforms import dct_summary, inverse_dct, haar_full_coefficients, inverse_haar
from src.one_embedding.universal_transforms import (
    feature_hash,
    random_orthogonal_project,
)
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "codec_candidates_results.json"
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
) -> dict[str, float]:
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )


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


def load_cb513_data():
    """Load CB513 labels and raw ProtT5 embeddings, return common IDs and splits."""
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
        print("  ERROR: No matching IDs between embeddings and CB513 labels")
        return None, None, None, None, None

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(common_ids))
    n_train = int(0.8 * len(common_ids))
    train_ids = [common_ids[i] for i in perm[:n_train]]
    test_ids = [common_ids[i] for i in perm[n_train:]]

    return embeddings, ss3_labels, common_ids, train_ids, test_ids


# ── Steps ────────────────────────────────────────────────────────


def step_V1(results: list[dict]):
    """V1: DCT inverse per-residue quality — THE critical gap.

    Tests: for K=1,2,4,8,16, encode raw ProtT5 CB513 with DCT,
    then inverse DCT back to (L, D). Measure CosSim and SS3 Q3.
    """
    print("\n═══ V1: DCT Inverse Per-Residue Quality ═══")

    cb_data = load_cb513_data()
    if cb_data[0] is None:
        return
    embeddings, ss3_labels, common_ids, train_ids, test_ids = cb_data

    print(f"  {len(common_ids)} CB513 proteins with ProtT5 embeddings")

    # Raw baseline (if not already done)
    raw_name = "V1_raw_baseline"
    if not is_done(results, raw_name):
        print("  Computing raw SS3 baseline...")
        raw_ss3 = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids)
        result = {
            "name": raw_name, "plm": "prot_t5_xl", "transform": "raw",
            "ss3_q3": raw_ss3["q3"], "cos_sim": 1.0,
            "n_proteins": len(common_ids),
        }
        results.append(result)
        save_results(results)
        print(f"  Raw SS3 Q3 = {raw_ss3['q3']:.3f}")

    D = next(iter(embeddings.values())).shape[1]

    for K in [1, 2, 4, 8, 16]:
        name = f"V1_dct_inverse_K{K}"
        if is_done(results, name):
            print(f"  DCT K={K} already done, skipping.")
            continue

        print(f"\n  DCT K={K}: encode → inverse decode...")
        t0 = time.time()
        decoded = {}
        for pid in common_ids:
            emb = embeddings[pid][:MAX_LEN]
            L = emb.shape[0]
            coeffs = dct_summary(emb, K=K)
            decoded[pid] = inverse_dct(coeffs, d=D, target_len=L)
        encode_time = time.time() - t0

        cos_sim = mean_cosine_sim(embeddings, decoded)
        print(f"    CosSim = {cos_sim:.3f} ({encode_time:.1f}s)")

        print(f"    Training SS3 probe on DCT K={K} decoded embeddings...")
        ss3 = evaluate_ss3_probe(decoded, ss3_labels, train_ids, test_ids)
        print(f"    SS3 Q3 = {ss3['q3']:.3f}")

        result = {
            "name": name, "plm": "prot_t5_xl", "transform": f"dct_inverse_K{K}",
            "K": K, "dim": D, "storage_per_protein": f"K*D={K}*{D}={K*D}",
            "cos_sim": cos_sim, "ss3_q3": ss3["q3"],
            "encode_time_s": round(encode_time, 2),
            "n_proteins": len(common_ids),
        }
        results.append(result)
        save_results(results)

    monitor()


def step_V2(results: list[dict]):
    """V2: Haar wavelet on raw — lossless per-residue + per-protein retrieval.

    Haar is mathematically lossless for per-residue (verified in Exp 18).
    Here we test: given Haar coefficients stored as the "codec", can we
    get competitive per-protein retrieval from the approximation coefficients?
    Also test per-residue SS3 from inverse Haar reconstruction.
    """
    print("\n═══ V2: Haar Wavelet Codec ═══")

    # Per-protein retrieval
    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    embeddings_5k = load_raw_embeddings("prot_t5_xl")
    if not embeddings_5k:
        return

    for levels in [1, 2, 3, 4]:
        name = f"V2_haar_L{levels}_retrieval"
        if is_done(results, name):
            print(f"  Haar L={levels} retrieval already done, skipping.")
            continue

        print(f"\n  Haar L={levels}: per-protein retrieval...")
        t0 = time.time()
        vectors = {}
        for pid in test_ids:
            if pid in embeddings_5k:
                emb = embeddings_5k[pid][:MAX_LEN]
                approx, details = haar_full_coefficients(emb, levels=levels)
                # Per-protein: mean pool the approximation coefficients
                vectors[pid] = approx.mean(axis=0)
        elapsed = time.time() - t0

        D = next(iter(vectors.values())).shape[0]
        ret = compute_retrieval(vectors, metadata, test_ids)

        result = {
            "name": name, "plm": "prot_t5_xl", "transform": f"haar_L{levels}",
            "levels": levels, "dim": D,
            "encode_time_s": round(elapsed, 2), **ret,
        }
        results.append(result)
        save_results(results)
        print(f"    Haar L={levels}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}")

    # Per-residue: Haar inverse on CB513 (should be lossless)
    name = "V2_haar_per_residue"
    if not is_done(results, name):
        print("\n  Haar per-residue verification on CB513...")
        cb_data = load_cb513_data()
        if cb_data[0] is not None:
            embeddings_cb, ss3_labels, common_ids, train_ids, test_ids_cb = cb_data

            decoded_haar = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                approx, details = haar_full_coefficients(emb, levels=3)
                decoded_haar[pid] = inverse_haar(approx, details, target_len=emb.shape[0])

            cos_sim = mean_cosine_sim(embeddings_cb, decoded_haar)
            ss3 = evaluate_ss3_probe(decoded_haar, ss3_labels, train_ids, test_ids_cb)

            result = {
                "name": name, "plm": "prot_t5_xl", "transform": "haar_inverse_L3",
                "cos_sim": cos_sim, "ss3_q3": ss3["q3"],
                "n_proteins": len(common_ids),
            }
            results.append(result)
            save_results(results)
            print(f"    Haar inverse: CosSim={cos_sim:.3f}, SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_V3(results: list[dict]):
    """V3: Feature hashing — PLM-universal feature compression.

    Tests feature hashing at various d_out for:
    a) Per-protein: mean pool of hashed residues → retrieval
    b) Per-residue: hashed residues → SS3 probe

    Feature hashing is the ONLY approach that works for any D without
    storing a projection matrix. The codec = two hash functions (seed).
    """
    print("\n═══ V3: Feature Hashing Codec ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]
    train_ids_5k = split["train_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings_5k = None

        for d_out in [128, 256, 512, 1024]:
            # Per-protein retrieval
            ret_name = f"V3_fhash_d{d_out}_{plm}_retrieval"
            if not is_done(results, ret_name):
                if embeddings_5k is None:
                    print(f"\n  Loading raw {plm} embeddings...")
                    embeddings_5k = load_raw_embeddings(plm)
                    if not embeddings_5k:
                        break

                print(f"  Feature hash d={d_out} for {plm}: retrieval...")
                t0 = time.time()
                vectors = {}
                for pid in test_ids_5k:
                    if pid in embeddings_5k:
                        emb = embeddings_5k[pid][:MAX_LEN]
                        hashed = feature_hash(emb, d_out=d_out)
                        vectors[pid] = hashed.mean(axis=0)
                elapsed = time.time() - t0

                ret = compute_retrieval(vectors, metadata, test_ids_5k)
                result = {
                    "name": ret_name, "plm": plm,
                    "transform": f"feature_hash_d{d_out}", "d_out": d_out,
                    "dim": d_out, "encode_time_s": round(elapsed, 2),
                    "plm_universal": True, **ret,
                }
                results.append(result)
                save_results(results)
                print(f"    d={d_out}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, {elapsed:.1f}s")

    # Per-residue: feature-hashed CB513 → SS3
    print("\n  Feature hash per-residue (CB513)...")
    cb_data = load_cb513_data()
    if cb_data[0] is not None:
        embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data
        D = next(iter(embeddings_cb.values())).shape[1]

        for d_out in [128, 256, 512, 1024]:
            ss3_name = f"V3_fhash_d{d_out}_prot_t5_xl_ss3"
            if is_done(results, ss3_name):
                print(f"  Feature hash d={d_out} SS3 already done, skipping.")
                continue

            print(f"  Feature hash d={d_out}: SS3 probe...")
            hashed_emb = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                hashed_emb[pid] = feature_hash(emb, d_out=d_out)

            ss3 = evaluate_ss3_probe(hashed_emb, ss3_labels, train_ids_cb, test_ids_cb)
            result = {
                "name": ss3_name, "plm": "prot_t5_xl",
                "transform": f"feature_hash_d{d_out}", "d_out": d_out,
                "original_dim": D, "ss3_q3": ss3["q3"],
                "n_proteins": len(common_ids),
            }
            results.append(result)
            save_results(results)
            print(f"    d={d_out}: SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_V4(results: list[dict]):
    """V4: Random orthogonal projection — JL distance preservation.

    Tests random projection at various d_out for:
    a) Per-protein: mean pool of projected residues → retrieval
    b) Per-residue: projected residues → SS3 probe

    Unlike feature hashing, needs different projection matrix per PLM
    (depends on D). But denser → potentially better quality.
    """
    print("\n═══ V4: Random Orthogonal Projection ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings_5k = None

        for d_out in [128, 256, 512]:
            ret_name = f"V4_rproj_d{d_out}_{plm}_retrieval"
            if not is_done(results, ret_name):
                if embeddings_5k is None:
                    print(f"\n  Loading raw {plm} embeddings...")
                    embeddings_5k = load_raw_embeddings(plm)
                    if not embeddings_5k:
                        break

                D = next(iter(embeddings_5k.values())).shape[1]
                print(f"  Random projection {D}→{d_out} for {plm}: retrieval...")

                # Pre-compute projection matrix once
                rng = np.random.RandomState(42)
                R = rng.randn(D, d_out).astype(np.float32)
                Q, _ = np.linalg.qr(R, mode="reduced")
                proj_matrix = Q * np.sqrt(D / d_out)

                t0 = time.time()
                vectors = {}
                for pid in test_ids_5k:
                    if pid in embeddings_5k:
                        emb = embeddings_5k[pid][:MAX_LEN]
                        projected = emb @ proj_matrix
                        vectors[pid] = projected.mean(axis=0).astype(np.float32)
                elapsed = time.time() - t0

                ret = compute_retrieval(vectors, metadata, test_ids_5k)
                result = {
                    "name": ret_name, "plm": plm,
                    "transform": f"random_proj_d{d_out}", "d_out": d_out,
                    "original_dim": D, "dim": d_out,
                    "encode_time_s": round(elapsed, 2),
                    "plm_universal": False, **ret,
                }
                results.append(result)
                save_results(results)
                print(f"    d={d_out}: Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}, {elapsed:.1f}s")

    # Per-residue: projected CB513 → SS3
    print("\n  Random projection per-residue (CB513)...")
    cb_data = load_cb513_data()
    if cb_data[0] is not None:
        embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data
        D = next(iter(embeddings_cb.values())).shape[1]

        for d_out in [128, 256, 512]:
            ss3_name = f"V4_rproj_d{d_out}_prot_t5_xl_ss3"
            if is_done(results, ss3_name):
                print(f"  Random proj d={d_out} SS3 already done, skipping.")
                continue

            print(f"  Random projection {D}→{d_out}: SS3 probe...")
            rng = np.random.RandomState(42)
            R = rng.randn(D, d_out).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            proj_matrix = Q * np.sqrt(D / d_out)

            proj_emb = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                proj_emb[pid] = (emb @ proj_matrix).astype(np.float32)

            ss3 = evaluate_ss3_probe(proj_emb, ss3_labels, train_ids_cb, test_ids_cb)
            result = {
                "name": ss3_name, "plm": "prot_t5_xl",
                "transform": f"random_proj_d{d_out}", "d_out": d_out,
                "original_dim": D, "ss3_q3": ss3["q3"],
                "n_proteins": len(common_ids),
            }
            results.append(result)
            save_results(results)
            print(f"    d={d_out}: SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_V5(results: list[dict]):
    """V5: Hybrid codec — feature hash (D→d) + DCT (L→K) = fixed-size.

    The "ProteJPG" codec:
    1. Feature hash: (L, D) → (L, d)         [compress features, PLM-universal]
    2. DCT along L:  (L, d) → (K, d)          [compress sequence]
    3. Store: (K, d) = FIXED SIZE per protein  [like JPG block]

    Per-protein: flatten (K*d,) → cosine similarity
    Per-residue: inverse DCT (K,d) → (L,d) → SS3 probe on d-dim residues

    With K=4, d=256: store 1024 values per protein regardless of PLM or length.
    """
    print("\n═══ V5: Hybrid Codec (Feature Hash + DCT) ═══")

    from scipy.fft import dct as scipy_dct, idct as scipy_idct

    metadata = load_metadata()
    split = load_split()
    test_ids_5k = split["test_ids"]

    # Test various (K, d_out) combinations
    configs = [
        (2, 256),   # 512 values
        (4, 256),   # 1024 values — the "JPG quality 75"
        (4, 512),   # 2048 values
        (8, 256),   # 2048 values
        (8, 512),   # 4096 values
    ]

    for plm in ["prot_t5_xl", "esm2_650m"]:
        embeddings_5k = None

        for K, d_out in configs:
            ret_name = f"V5_hybrid_K{K}_d{d_out}_{plm}_retrieval"
            if is_done(results, ret_name):
                print(f"  {plm} K={K} d={d_out} already done, skipping.")
                continue

            if embeddings_5k is None:
                print(f"\n  Loading raw {plm} embeddings...")
                embeddings_5k = load_raw_embeddings(plm)
                if not embeddings_5k:
                    break

            print(f"  Hybrid K={K}, d={d_out} for {plm}: retrieval...")
            t0 = time.time()
            vectors = {}
            for pid in test_ids_5k:
                if pid in embeddings_5k:
                    emb = embeddings_5k[pid][:MAX_LEN]
                    # Step 1: Feature hash D → d_out
                    hashed = feature_hash(emb, d_out=d_out)
                    # Step 2: DCT along sequence, keep K coefficients
                    L = hashed.shape[0]
                    k = min(K, L)
                    coeffs = scipy_dct(hashed, type=2, axis=0, norm="ortho")[:k]
                    # Per-protein: flatten
                    vectors[pid] = coeffs.ravel().astype(np.float32)
            elapsed = time.time() - t0

            dim = next(iter(vectors.values())).shape[0]
            ret = compute_retrieval(vectors, metadata, test_ids_5k)

            result = {
                "name": ret_name, "plm": plm,
                "transform": f"hybrid_K{K}_d{d_out}",
                "K": K, "d_out": d_out, "dim": dim,
                "storage_values": K * d_out,
                "encode_time_s": round(elapsed, 2),
                "plm_universal": True, **ret,
            }
            results.append(result)
            save_results(results)
            print(f"    K={K}, d={d_out}: Ret@1={ret['precision@1']:.3f}, "
                  f"MRR={ret['mrr']:.3f}, dim={dim}, {elapsed:.1f}s")

    # Per-residue: hybrid decode on CB513
    print("\n  Hybrid codec per-residue (CB513)...")
    cb_data = load_cb513_data()
    if cb_data[0] is not None:
        embeddings_cb, ss3_labels, common_ids, train_ids_cb, test_ids_cb = cb_data

        for K, d_out in [(4, 256), (4, 512), (8, 256), (8, 512)]:
            ss3_name = f"V5_hybrid_K{K}_d{d_out}_prot_t5_xl_ss3"
            if is_done(results, ss3_name):
                print(f"  Hybrid K={K} d={d_out} SS3 already done, skipping.")
                continue

            print(f"  Hybrid K={K}, d={d_out}: encode → decode → SS3...")
            decoded = {}
            for pid in common_ids:
                emb = embeddings_cb[pid][:MAX_LEN]
                L = emb.shape[0]
                # Encode
                hashed = feature_hash(emb, d_out=d_out)
                k = min(K, L)
                coeffs = scipy_dct(hashed, type=2, axis=0, norm="ortho")[:k]
                # Decode: inverse DCT back to (L, d_out)
                full_coeffs = np.zeros((L, d_out), dtype=np.float32)
                full_coeffs[:k] = coeffs
                decoded[pid] = scipy_idct(full_coeffs, type=2, axis=0, norm="ortho").astype(np.float32)

            # CosSim against feature-hashed (not original!) — measures DCT loss only
            hashed_originals = {}
            for pid in common_ids:
                hashed_originals[pid] = feature_hash(embeddings_cb[pid][:MAX_LEN], d_out=d_out)
            cos_sim_dct = mean_cosine_sim(hashed_originals, decoded)

            ss3 = evaluate_ss3_probe(decoded, ss3_labels, train_ids_cb, test_ids_cb)

            result = {
                "name": ss3_name, "plm": "prot_t5_xl",
                "transform": f"hybrid_K{K}_d{d_out}",
                "K": K, "d_out": d_out,
                "cos_sim_dct_only": cos_sim_dct,
                "ss3_q3": ss3["q3"],
                "storage_values": K * d_out,
                "n_proteins": len(common_ids),
            }
            results.append(result)
            save_results(results)
            print(f"    K={K}, d={d_out}: DCT CosSim={cos_sim_dct:.3f}, SS3 Q3={ss3['q3']:.3f}")

    monitor()


def step_V6(results: list[dict]):
    """V6: Summary comparison — all codec candidates vs baselines."""
    print("\n═══ V6: Codec Candidate Summary ═══")

    # ── Per-protein retrieval ──
    print("\n  ═══ Per-Protein Retrieval (ProtT5, SCOPe 5K) ═══")
    print("  ┌────────────────────────────┬──────┬────────┬────────┬─────────┐")
    print("  │ Codec                      │ Dim  │ Ret@1  │ MRR    │ PLM-Uni │")
    print("  ├────────────────────────────┼──────┼────────┼────────┼─────────┤")

    # Collect retrieval results for ProtT5
    ret_results = []
    for r in results:
        if r.get("plm") == "prot_t5_xl" and "precision@1" in r:
            ret_results.append(r)

    # Sort by Ret@1
    ret_results.sort(key=lambda r: r.get("precision@1", 0), reverse=True)

    for r in ret_results:
        tf = r.get("transform", r.get("name", "?"))
        dim = r.get("dim", "?")
        ret1 = r.get("precision@1", 0)
        mrr = r.get("mrr", 0)
        univ = "YES" if r.get("plm_universal") else "no"
        print(f"  │ {tf:<26s} │ {str(dim):>4s} │ {ret1:.3f}  │ {mrr:.3f}  │ {univ:>7s} │")

    print("  └────────────────────────────┴──────┴────────┴────────┴─────────┘")

    # ── Per-residue SS3 ──
    print("\n  ═══ Per-Residue SS3 Q3 (ProtT5, CB513) ═══")
    print("  ┌────────────────────────────┬──────────┬────────┐")
    print("  │ Codec                      │ CosSim   │ SS3 Q3 │")
    print("  ├────────────────────────────┼──────────┼────────┤")

    ss3_results = []
    for r in results:
        if r.get("plm") == "prot_t5_xl" and "ss3_q3" in r:
            ss3_results.append(r)

    ss3_results.sort(key=lambda r: r.get("ss3_q3", 0), reverse=True)

    for r in ss3_results:
        tf = r.get("transform", r.get("name", "?"))
        cos = r.get("cos_sim", r.get("cos_sim_dct_only", "?"))
        q3 = r.get("ss3_q3", 0)
        cos_str = f"{cos:.3f}" if isinstance(cos, float) else str(cos)
        print(f"  │ {tf:<26s} │ {cos_str:>8s} │ {q3:.3f}  │")

    print("  └────────────────────────────┴──────────┴────────┘")

    # ── ESM2 retrieval (cross-PLM) ──
    esm_ret = [r for r in results if r.get("plm") == "esm2_650m" and "precision@1" in r]
    if esm_ret:
        print("\n  ═══ ESM2-650M Retrieval (Cross-PLM Validation) ═══")
        print("  ┌────────────────────────────┬──────┬────────┬────────┐")
        print("  │ Codec                      │ Dim  │ Ret@1  │ MRR    │")
        print("  ├────────────────────────────┼──────┼────────┼────────┤")
        esm_ret.sort(key=lambda r: r.get("precision@1", 0), reverse=True)
        for r in esm_ret:
            tf = r.get("transform", "?")
            dim = r.get("dim", "?")
            ret1 = r.get("precision@1", 0)
            mrr = r.get("mrr", 0)
            print(f"  │ {tf:<26s} │ {str(dim):>4s} │ {ret1:.3f}  │ {mrr:.3f}  │")
        print("  └────────────────────────────┴──────┴────────┴────────┘")

    # ── Key insights ──
    print("\n  ═══ Key Insights ═══")

    # Find best codec that serves both tasks
    best_ret = max((r for r in results if r.get("plm") == "prot_t5_xl" and "precision@1" in r),
                   key=lambda r: r["precision@1"], default=None)
    best_ss3 = max((r for r in results if r.get("plm") == "prot_t5_xl" and "ss3_q3" in r),
                   key=lambda r: r["ss3_q3"], default=None)

    if best_ret:
        print(f"  Best per-protein: {best_ret.get('transform', '?')} "
              f"Ret@1={best_ret['precision@1']:.3f}")
    if best_ss3:
        print(f"  Best per-residue: {best_ss3.get('transform', '?')} "
              f"SS3 Q3={best_ss3['ss3_q3']:.3f}")

    # Reference baselines
    print(f"\n  Baselines (from Exp 20):")
    print(f"    Raw mean pool ProtT5:    Ret@1=0.734")
    print(f"    Raw SS3 ProtT5:          Q3=0.845")
    print(f"    Trained ChannelComp:     Ret@1=0.808, Q3=0.834")

    print("\n  V6 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 21: Codec Candidates")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step (V1-V6)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "V1": step_V1, "V2": step_V2, "V3": step_V3,
        "V4": step_V4, "V5": step_V5, "V6": step_V6,
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
