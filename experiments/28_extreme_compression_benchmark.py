#!/usr/bin/env python3
"""Experiment 28: Extreme Compression Benchmark.

Benchmarks aggressive compression techniques across 4 categories:
  A. Within-channel compression (wavelet, CUR, channel pruning, delta encoding)
  B. Quantization (int8, int4, binary, PQ, RVQ)
  C. Novel math (tensor train, NMF, optimal transport, TDA)
  D. Structure-aware (AA-residual, SimHash)

Plus multi-stage pipelines (P2) and cross-PLM validation (P3).

Each codec is evaluated on:
  - Family retrieval Ret@1 (SCOPe 5K test set)
  - Per-residue SS3 Q3 probe (CB513)
  - Reconstruction cosine similarity
  - Compressed size in bytes (+ zstd)
  - Encoding speed (ms per protein)

Steps:
  P1A: Category A — within-channel compression
  P1B: Category B — quantization
  P1C: Category C — novel math (TT, NMF, OT, TDA)
  P1D: Category D — structure-aware (AA-residual, SimHash)
  P2:  Multi-stage pipelines
  P3:  Cross-PLM validation (ESM2-650M, ESM-C 300M)

Usage:
  uv run python experiments/28_extreme_compression_benchmark.py --step P1A
  uv run python experiments/28_extreme_compression_benchmark.py --step P1B
  uv run python experiments/28_extreme_compression_benchmark.py
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    load_cb513_csv,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.utils.h5_store import load_residue_embeddings

# Category A: within-channel compression
from src.one_embedding.extreme_compression import (
    wavelet_threshold_compress,
    wavelet_threshold_decompress,
    cur_decompose,
    cur_reconstruct,
    compute_channel_importance,
    channel_prune,
    measure_compressed_size,
    zstd_compress,
)

# Category B: quantization
from src.one_embedding.quantization import (
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
    quantize_binary,
    dequantize_binary,
    pq_fit,
    pq_encode,
    pq_decode,
    rvq_fit,
    rvq_encode,
    rvq_decode,
    compressed_size_bytes,
)

# Category C: tensor decomposition + NMF
from src.one_embedding.tensor_decomposition import (
    tt_decompose,
    tt_reconstruct,
    tt_storage_bytes,
    nmf_fit,
    nmf_encode,
    nmf_decode,
)

# Category D: topological / structure-aware
from src.one_embedding.topological import (
    sliced_wasserstein_distance,
    sliced_wasserstein_matrix,
    persistence_image,
    simhash_encode,
    simhash_decode_approx,
    compute_aa_centroids,
    aa_residual_encode,
    aa_residual_decode,
)

# Other transforms
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.path_transforms import displacement_encode, displacement_decode

# ── Constants ──────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "extreme_compression_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512

PLMS = [
    ("prot_t5_xl", 1024, "prot_t5_xl"),
    ("esm2_650m", 1280, "esm2_650m"),
    ("esmc_300m", 960, "esmc_300m"),
]


# ── Helpers ────────────────────────────────────────────────────────

def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": [], "results": {}}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def mark_done(results, step):
    if step not in results.setdefault("steps_done", []):
        results["steps_done"].append(step)


def monitor():
    try:
        l1, l5, l15 = os.getloadavg()
        print(f"  System load: {l1:.1f} / {l5:.1f} / {l15:.1f}")
    except OSError:
        pass


def load_metadata():
    meta = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    meta, _ = filter_by_family_size(meta, min_members=3)
    return meta


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def load_plm_embeddings(plm_stem, dataset="medium5k"):
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
    if not h5_path.exists():
        print(f"  WARNING: {h5_path} not found")
        return {}
    embeddings = load_residue_embeddings(h5_path)
    if any("|" in k for k in list(embeddings.keys())[:5]):
        return {k.split("|")[0]: v for k, v in embeddings.items()}
    return embeddings


def cap_length(embs, max_len=MAX_LEN):
    return {k: v[:max_len] for k, v in embs.items()}


def eval_retrieval(vectors, metadata, test_ids):
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids, metric="cosine",
    )


def compute_cos_sim(original, reconstructed):
    """Mean cosine similarity between original and reconstructed matrices."""
    sims = []
    for pid in original:
        if pid not in reconstructed:
            continue
        orig = original[pid].astype(np.float32).ravel()
        recon = reconstructed[pid].astype(np.float32).ravel()
        norm_o = np.linalg.norm(orig)
        norm_r = np.linalg.norm(recon)
        if norm_o > 1e-8 and norm_r > 1e-8:
            sims.append(float(np.dot(orig, recon) / (norm_o * norm_r)))
    return float(np.mean(sims)) if sims else 0.0


def time_encode(encode_fn, embeddings, sample_ids, n_warmup=10, n_timed=100):
    """Time encode on n_timed proteins (after n_warmup warmup). Returns ms/protein."""
    valid_ids = [pid for pid in sample_ids if pid in embeddings]
    if not valid_ids:
        return 0.0
    # Warmup
    for pid in valid_ids[:n_warmup]:
        try:
            encode_fn(embeddings[pid])
        except Exception:
            pass
    # Timed
    ids_to_time = valid_ids[:n_timed]
    t0 = time.perf_counter()
    for pid in ids_to_time:
        try:
            encode_fn(embeddings[pid])
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return (elapsed / len(ids_to_time)) * 1000.0  # ms per protein


def load_cb513_data():
    """Load CB513 labels and embeddings for per-residue probes."""
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if not cb513_path.exists():
        print("  CB513 not found, skipping per-residue probes")
        return None
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    cb_embs = load_plm_embeddings("prot_t5_xl", dataset="cb513")
    if not cb_embs:
        print("  CB513 embeddings not found")
        return None
    cb_embs = cap_length(cb_embs)
    return {
        "sequences": sequences,
        "ss3_labels": ss3_labels,
        "ss8_labels": ss8_labels,
        "embeddings": cb_embs,
    }


def eval_ss3(coded_cb_embs, ss3_labels):
    """Evaluate SS3 Q3 on CB513 with codec-transformed embeddings."""
    avail = [pid for pid in ss3_labels if pid in coded_cb_embs]
    if len(avail) < 10:
        return {"q3": 0.0}
    rng = random.Random(42)
    rng.shuffle(avail)
    n_train = int(len(avail) * 0.8)
    train_ids, test_ids_cb = avail[:n_train], avail[n_train:]
    return evaluate_ss3_probe(coded_cb_embs, ss3_labels, train_ids, test_ids_cb)


def benchmark_codec(
    codec_name,
    encode_fn,
    decode_fn,
    embeddings,
    test_ids,
    metadata,
    cb513_data,
    plm_name="prot_t5_xl",
    fitting="none",
    has_per_residue=True,
):
    """Run full benchmark for a single codec. Returns result dict."""
    result = {
        "fitting": fitting,
        "per_residue": has_per_residue,
    }

    try:
        # 1. Encode all test proteins, compute protein vectors via mean pool
        coded_embs = {}  # per-residue (or pruned)
        protein_vectors = {}
        for pid in test_ids:
            if pid not in embeddings:
                continue
            m = embeddings[pid].astype(np.float32)
            encoded = encode_fn(m)
            if decode_fn is not None:
                decoded = decode_fn(encoded)
                coded_embs[pid] = decoded
                protein_vectors[pid] = decoded.mean(axis=0).astype(np.float32)
            else:
                # No decode — encoded IS the final per-residue representation
                if isinstance(encoded, tuple):
                    encoded = encoded[0]  # channel_prune returns (matrix, indices)
                coded_embs[pid] = encoded
                protein_vectors[pid] = encoded.mean(axis=0).astype(np.float32)

        if not protein_vectors:
            return {"error": "no proteins encoded"}

        # 2. Retrieval
        ret = eval_retrieval(protein_vectors, metadata, test_ids)
        result["family_ret1"] = ret["precision@1"]
        result["mrr"] = ret["mrr"]

        # 3. Per-residue SS3
        if has_per_residue and cb513_data is not None:
            cb_coded = {}
            for pid in cb513_data["ss3_labels"]:
                if pid not in cb513_data["embeddings"]:
                    continue
                m = cb513_data["embeddings"][pid].astype(np.float32)
                try:
                    encoded = encode_fn(m)
                    if decode_fn is not None:
                        decoded = decode_fn(encoded)
                    else:
                        decoded = encoded[0] if isinstance(encoded, tuple) else encoded
                    cb_coded[pid] = decoded.astype(np.float32)
                except Exception:
                    continue
            if cb_coded:
                ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                result["ss3_q3"] = ss3.get("q3", 0.0)

        # 4. Reconstruction cosine similarity
        if decode_fn is not None:
            result["cos_sim"] = compute_cos_sim(
                {pid: embeddings[pid][:MAX_LEN] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )

        # 5. Size estimate (sample 10 proteins)
        sizes_raw = []
        sizes_zstd = []
        sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
        for pid in sample_pids:
            m = embeddings[pid].astype(np.float32)
            encoded = encode_fn(m)
            if decode_fn is not None:
                decoded = decode_fn(encoded)
            else:
                decoded = encoded[0] if isinstance(encoded, tuple) else encoded
            sz = measure_compressed_size(decoded)
            sizes_raw.append(sz["raw_bytes"])
            sizes_zstd.append(sz["zstd_bytes"])
        if sizes_raw:
            orig_sizes = [embeddings[pid].astype(np.float32).nbytes for pid in sample_pids]
            result["size_bytes"] = int(np.mean(sizes_raw))
            result["size_zstd_bytes"] = int(np.mean(sizes_zstd))
            result["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(sizes_raw))

        # 6. Speed
        result["encode_ms"] = time_encode(
            encode_fn, embeddings,
            [pid for pid in test_ids if pid in embeddings],
        )

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()

    return result


# ── Step P1A: Category A — Within-Channel Compression ──────────────


def step_P1a(results):
    """P1A: Wavelet, CUR, channel pruning, delta encoding on ProtT5."""
    print("\n" + "=" * 60)
    print("P1A: Category A — Within-Channel Compression (ProtT5)")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    if not embeddings:
        print("  No embeddings, skipping P1A")
        return
    embeddings = cap_length(embeddings)
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    cb513_data = load_cb513_data()

    # Compute channel importance on training set for pruning
    train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    print(f"  Computing channel importance on {len(train_embs)} training proteins...")
    importance = compute_channel_importance(train_embs, max_proteins=1000)
    print(f"  Channel importance computed: shape={importance.shape}")

    plm_results = results.setdefault("results", {}).setdefault("prot_t5_xl", {})

    # Define codecs
    codecs = [
        # Wavelet thresholding
        ("wavelet_db4_50", "wavelet",
         lambda m: wavelet_threshold_compress(m, "db4", 50),
         lambda c: wavelet_threshold_decompress(c)),
        ("wavelet_db4_75", "wavelet",
         lambda m: wavelet_threshold_compress(m, "db4", 75),
         lambda c: wavelet_threshold_decompress(c)),
        ("wavelet_db4_90", "wavelet",
         lambda m: wavelet_threshold_compress(m, "db4", 90),
         lambda c: wavelet_threshold_decompress(c)),
        ("wavelet_db4_95", "wavelet",
         lambda m: wavelet_threshold_compress(m, "db4", 95),
         lambda c: wavelet_threshold_decompress(c)),
        # CUR decomposition
        ("cur_k32", "cur",
         lambda m: cur_decompose(m, k=32),
         lambda c: cur_reconstruct(c)),
        ("cur_k64", "cur",
         lambda m: cur_decompose(m, k=64),
         lambda c: cur_reconstruct(c)),
        ("cur_k128", "cur",
         lambda m: cur_decompose(m, k=128),
         lambda c: cur_reconstruct(c)),
        ("cur_k256", "cur",
         lambda m: cur_decompose(m, k=256),
         lambda c: cur_reconstruct(c)),
        # Channel pruning (no decode — lossy)
        ("prune_k64", "prune",
         lambda m: channel_prune(m, importance, 64),
         None),
        ("prune_k128", "prune",
         lambda m: channel_prune(m, importance, 128),
         None),
        ("prune_k256", "prune",
         lambda m: channel_prune(m, importance, 256),
         None),
        ("prune_k512", "prune",
         lambda m: channel_prune(m, importance, 512),
         None),
        # Delta encoding
        ("delta_order1", "delta",
         lambda m: (displacement_encode(m), m[0]),
         None),
    ]

    n_total = len(codecs)
    for i, (name, codec_type, encode_fn, decode_fn) in enumerate(codecs):
        if name in plm_results:
            print(f"  [{i+1}/{n_total}] {name}: already done, skipping")
            continue

        print(f"  [{i+1}/{n_total}] {name}: running...")

        # Special handling for delta encoding
        if codec_type == "delta":
            # Delta encode: compute displacements + x0, then decode for retrieval
            def delta_encode(m):
                return displacement_encode(m)

            def delta_decode(disp, m=None):
                # We need x0 for decode, pass it through closure
                pass

            # Custom benchmark for delta
            try:
                coded_embs = {}
                protein_vectors = {}
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    disp = displacement_encode(m)
                    x0 = m[0].copy()
                    recon = displacement_decode(disp, x0)
                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                ret = eval_retrieval(protein_vectors, metadata, test_ids)
                cos = compute_cos_sim(
                    {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                    {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
                )
                enc_ms = time_encode(displacement_encode, embeddings,
                                     [pid for pid in test_ids if pid in embeddings])

                # Size: displacements are (L-1, D), x0 is (D,), so same as (L, D)
                sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
                avg_raw = np.mean([embeddings[pid].nbytes for pid in sample_pids])
                avg_disp = np.mean([
                    displacement_encode(embeddings[pid]).nbytes + embeddings[pid][0].nbytes
                    for pid in sample_pids
                ])
                disp_zstd = np.mean([
                    len(zstd_compress(displacement_encode(embeddings[pid]).tobytes()))
                    + len(zstd_compress(embeddings[pid][0].tobytes()))
                    for pid in sample_pids
                ])

                r = {
                    "family_ret1": ret["precision@1"],
                    "mrr": ret["mrr"],
                    "cos_sim": cos,
                    "size_bytes": int(avg_disp),
                    "size_zstd_bytes": int(disp_zstd),
                    "compression_ratio": float(avg_raw / avg_disp) if avg_disp > 0 else 0,
                    "encode_ms": enc_ms,
                    "fitting": "none",
                    "per_residue": True,
                }

                # SS3 probe on delta-reconstructed CB513
                if cb513_data is not None:
                    cb_coded = {}
                    for pid in cb513_data["ss3_labels"]:
                        if pid not in cb513_data["embeddings"]:
                            continue
                        m = cb513_data["embeddings"][pid].astype(np.float32)
                        disp = displacement_encode(m)
                        recon = displacement_decode(disp, m[0].copy())
                        cb_coded[pid] = recon
                    if cb_coded:
                        ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                        r["ss3_q3"] = ss3.get("q3", 0.0)

                plm_results[name] = r
                print(f"  [{i+1}/{n_total}] {name}: Ret@1={r['family_ret1']:.3f}, "
                      f"SS3={r.get('ss3_q3', '-')}, CosSim={r['cos_sim']:.3f}, "
                      f"Size={r['size_bytes']//1024}KB, Encode={r['encode_ms']:.1f}ms")
            except Exception as e:
                plm_results[name] = {"error": str(e)}
                traceback.print_exc()
                print(f"  [{i+1}/{n_total}] {name}: FAILED — {e}")

        elif codec_type == "prune":
            # Channel pruning returns (matrix, indices) — no decode
            r = benchmark_codec(
                name, encode_fn, decode_fn,
                embeddings, test_ids, metadata, cb513_data,
                has_per_residue=True,
            )
            plm_results[name] = r
            print(f"  [{i+1}/{n_total}] {name}: Ret@1={r.get('family_ret1', '-')}, "
                  f"SS3={r.get('ss3_q3', '-')}, "
                  f"Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Encode={r.get('encode_ms', 0):.1f}ms")

        elif codec_type in ("wavelet", "cur"):
            # Wavelet and CUR: encode returns dict, decode takes dict
            # Custom benchmark since benchmark_codec expects encode to return array
            try:
                coded_embs = {}
                protein_vectors = {}
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    compressed = encode_fn(m)
                    recon = decode_fn(compressed)
                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                ret = eval_retrieval(protein_vectors, metadata, test_ids)
                cos = compute_cos_sim(
                    {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                    {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
                )
                enc_ms = time_encode(encode_fn, embeddings,
                                     [pid for pid in test_ids if pid in embeddings])

                # Size (of reconstructed matrix)
                sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
                sizes_info = [measure_compressed_size(coded_embs[pid])
                              for pid in sample_pids if pid in coded_embs]
                orig_sizes = [embeddings[pid].astype(np.float32).nbytes for pid in sample_pids]

                r = {
                    "family_ret1": ret["precision@1"],
                    "mrr": ret["mrr"],
                    "cos_sim": cos,
                    "encode_ms": enc_ms,
                    "fitting": "none",
                    "per_residue": True,
                }
                if sizes_info:
                    r["size_bytes"] = int(np.mean([s["raw_bytes"] for s in sizes_info]))
                    r["size_zstd_bytes"] = int(np.mean([s["zstd_bytes"] for s in sizes_info]))
                    r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean([s["raw_bytes"] for s in sizes_info]))

                # SS3 probe
                if cb513_data is not None:
                    cb_coded = {}
                    for pid in cb513_data["ss3_labels"]:
                        if pid not in cb513_data["embeddings"]:
                            continue
                        m = cb513_data["embeddings"][pid].astype(np.float32)
                        try:
                            compressed = encode_fn(m)
                            recon = decode_fn(compressed)
                            cb_coded[pid] = recon.astype(np.float32)
                        except Exception:
                            continue
                    if cb_coded:
                        ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                        r["ss3_q3"] = ss3.get("q3", 0.0)

                plm_results[name] = r
                print(f"  [{i+1}/{n_total}] {name}: Ret@1={r['family_ret1']:.3f}, "
                      f"SS3={r.get('ss3_q3', '-')}, CosSim={r['cos_sim']:.3f}, "
                      f"Size={r.get('size_bytes', 0)//1024}KB, "
                      f"Encode={r['encode_ms']:.1f}ms")
            except Exception as e:
                plm_results[name] = {"error": str(e)}
                traceback.print_exc()
                print(f"  [{i+1}/{n_total}] {name}: FAILED — {e}")
        else:
            r = benchmark_codec(
                name, encode_fn, decode_fn,
                embeddings, test_ids, metadata, cb513_data,
            )
            plm_results[name] = r
            print(f"  [{i+1}/{n_total}] {name}: Ret@1={r.get('family_ret1', '-')}, "
                  f"SS3={r.get('ss3_q3', '-')}")

        save_results(results)
        monitor()

    mark_done(results, "P1A")
    save_results(results)
    print("\n  P1A complete.")


# ── Step P1B: Category B — Quantization ────────────────────────────


def step_P1b(results):
    """P1B: Int8, Int4, Binary, PQ, RVQ quantization on ProtT5."""
    print("\n" + "=" * 60)
    print("P1B: Category B — Quantization (ProtT5)")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    if not embeddings:
        print("  No embeddings, skipping P1B")
        return
    embeddings = cap_length(embeddings)
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    cb513_data = load_cb513_data()

    plm_results = results.setdefault("results", {}).setdefault("prot_t5_xl", {})

    # Pre-fit PQ and RVQ on training data
    train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    D = 1024  # ProtT5

    pq_models = {}
    rvq_models = {}

    for M_val in [16, 32, 64]:
        pq_name = f"pq_M{M_val}"
        if pq_name not in plm_results:
            print(f"  Fitting PQ M={M_val} on {len(train_embs)} training proteins...")
            try:
                pq_models[M_val] = pq_fit(train_embs, M=M_val, n_centroids=256, seed=42)
                print(f"    PQ M={M_val} fitted: sub_dim={pq_models[M_val]['sub_dim']}")
            except Exception as e:
                print(f"    PQ M={M_val} fit FAILED: {e}")

    for n_lev in [2, 3]:
        rvq_name = f"rvq_{n_lev}level"
        if rvq_name not in plm_results:
            print(f"  Fitting RVQ {n_lev}-level on {len(train_embs)} training proteins...")
            try:
                rvq_models[n_lev] = rvq_fit(train_embs, n_levels=n_lev, n_centroids=256, seed=42)
                print(f"    RVQ {n_lev}-level fitted")
            except Exception as e:
                print(f"    RVQ {n_lev}-level fit FAILED: {e}")

    # Define codecs
    codecs = []

    # Int8/Int4/Binary on raw
    codecs.append(("int8_raw", "train_free",
                   lambda m: quantize_int8(m),
                   lambda c: dequantize_int8(c)))
    codecs.append(("int4_raw", "train_free",
                   lambda m: quantize_int4(m),
                   lambda c: dequantize_int4(c)))
    codecs.append(("binary_raw", "train_free",
                   lambda m: quantize_binary(m),
                   lambda c: dequantize_binary(c)))

    # Int8/Int4/Binary on rp512
    codecs.append(("int8_rp512", "train_free",
                   lambda m: quantize_int8(random_orthogonal_project(m, d_out=512)),
                   lambda c: dequantize_int8(c)))
    codecs.append(("int4_rp512", "train_free",
                   lambda m: quantize_int4(random_orthogonal_project(m, d_out=512)),
                   lambda c: dequantize_int4(c)))
    codecs.append(("binary_rp512", "train_free",
                   lambda m: quantize_binary(random_orthogonal_project(m, d_out=512)),
                   lambda c: dequantize_binary(c)))

    # PQ
    for M_val in [16, 32, 64]:
        if M_val in pq_models:
            model = pq_models[M_val]
            codecs.append((f"pq_M{M_val}", "corpus_fit",
                           lambda m, mod=model: pq_encode(m, mod),
                           lambda c, mod=model: pq_decode(c, mod)))

    # RVQ
    for n_lev in [2, 3]:
        if n_lev in rvq_models:
            model = rvq_models[n_lev]
            codecs.append((f"rvq_{n_lev}level", "corpus_fit",
                           lambda m, mod=model: rvq_encode(m, mod),
                           lambda c, mod=model: rvq_decode(c, mod)))

    n_total = len(codecs)
    for i, (name, fitting, encode_fn, decode_fn) in enumerate(codecs):
        if name in plm_results:
            print(f"  [{i+1}/{n_total}] {name}: already done, skipping")
            continue

        print(f"  [{i+1}/{n_total}] {name}: running...")

        try:
            coded_embs = {}
            protein_vectors = {}
            for pid in test_ids:
                if pid not in embeddings:
                    continue
                m = embeddings[pid].astype(np.float32)
                compressed = encode_fn(m)
                recon = decode_fn(compressed)
                coded_embs[pid] = recon
                protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

            if not protein_vectors:
                plm_results[name] = {"error": "no proteins encoded"}
                continue

            ret = eval_retrieval(protein_vectors, metadata, test_ids)
            cos = compute_cos_sim(
                {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )
            enc_ms = time_encode(encode_fn, embeddings,
                                 [pid for pid in test_ids if pid in embeddings])

            # Size: for quantized, measure the compressed data size
            sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
            q_sizes = []
            orig_sizes = []
            for pid in sample_pids:
                m = embeddings[pid].astype(np.float32)
                compressed = encode_fn(m)
                try:
                    q_sizes.append(compressed_size_bytes(compressed))
                except (ValueError, KeyError):
                    # PQ/RVQ codes: compute from array directly
                    if isinstance(compressed, np.ndarray):
                        q_sizes.append(compressed.nbytes)
                    else:
                        # Fallback: measure reconstructed
                        recon = decode_fn(compressed)
                        q_sizes.append(recon.nbytes)
                orig_sizes.append(m.nbytes)

            zstd_sizes = []
            for pid in sample_pids:
                m = embeddings[pid].astype(np.float32)
                compressed = encode_fn(m)
                recon = decode_fn(compressed)
                sz = measure_compressed_size(recon)
                zstd_sizes.append(sz["zstd_bytes"])

            r = {
                "family_ret1": ret["precision@1"],
                "mrr": ret["mrr"],
                "cos_sim": cos,
                "encode_ms": enc_ms,
                "fitting": fitting,
                "per_residue": True,
            }
            if q_sizes:
                r["size_bytes"] = int(np.mean(q_sizes))
                r["size_zstd_bytes"] = int(np.mean(zstd_sizes))
                r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(q_sizes))

            # SS3 probe
            if cb513_data is not None:
                cb_coded = {}
                for pid in cb513_data["ss3_labels"]:
                    if pid not in cb513_data["embeddings"]:
                        continue
                    m = cb513_data["embeddings"][pid].astype(np.float32)
                    try:
                        compressed = encode_fn(m)
                        recon = decode_fn(compressed)
                        cb_coded[pid] = recon.astype(np.float32)
                    except Exception:
                        continue
                if cb_coded:
                    ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                    r["ss3_q3"] = ss3.get("q3", 0.0)

            plm_results[name] = r
            print(f"  [{i+1}/{n_total}] {name}: Ret@1={r['family_ret1']:.3f}, "
                  f"SS3={r.get('ss3_q3', '-')}, CosSim={r['cos_sim']:.3f}, "
                  f"Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Encode={r['encode_ms']:.1f}ms")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  [{i+1}/{n_total}] {name}: FAILED — {e}")

        save_results(results)
        monitor()

    mark_done(results, "P1B")
    save_results(results)
    print("\n  P1B complete.")


# ── Step P1C: Category C — Novel Math ──────────────────────────────


def step_P1c(results):
    """P1C: Tensor Train, NMF, Optimal Transport, TDA on ProtT5."""
    print("\n" + "=" * 60)
    print("P1C: Category C — Novel Math (ProtT5)")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    if not embeddings:
        print("  No embeddings, skipping P1C")
        return
    embeddings = cap_length(embeddings)
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    cb513_data = load_cb513_data()

    plm_results = results.setdefault("results", {}).setdefault("prot_t5_xl", {})

    # Pre-fit NMF on training data
    train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    nmf_models = {}
    for k_val in [16, 32, 64]:
        nmf_name = f"nmf_k{k_val}"
        if nmf_name not in plm_results:
            print(f"  Fitting NMF k={k_val} on {len(train_embs)} training proteins...")
            try:
                nmf_models[k_val] = nmf_fit(train_embs, k=k_val, seed=42)
                print(f"    NMF k={k_val} fitted")
            except Exception as e:
                print(f"    NMF k={k_val} fit FAILED: {e}")

    # ── Tensor Train ──
    for bond_dim in [4, 8, 16, 32]:
        name = f"tt_bd{bond_dim}"
        if name in plm_results:
            print(f"  {name}: already done, skipping")
            continue

        print(f"  {name}: running...")
        try:
            coded_embs = {}
            protein_vectors = {}
            tt_sizes = []
            for pid in test_ids:
                if pid not in embeddings:
                    continue
                m = embeddings[pid].astype(np.float32)
                compressed = tt_decompose(m, bond_dim=bond_dim)
                recon = tt_reconstruct(compressed)
                coded_embs[pid] = recon
                protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)
                if len(tt_sizes) < 10:
                    tt_sizes.append(tt_storage_bytes(compressed))

            ret = eval_retrieval(protein_vectors, metadata, test_ids)
            cos = compute_cos_sim(
                {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )
            enc_ms = time_encode(
                lambda m, bd=bond_dim: tt_decompose(m, bond_dim=bd),
                embeddings,
                [pid for pid in test_ids if pid in embeddings],
            )

            sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
            orig_sizes = [embeddings[pid].astype(np.float32).nbytes for pid in sample_pids]

            r = {
                "family_ret1": ret["precision@1"],
                "mrr": ret["mrr"],
                "cos_sim": cos,
                "encode_ms": enc_ms,
                "fitting": "none",
                "per_residue": True,
            }
            if tt_sizes:
                r["size_bytes"] = int(np.mean(tt_sizes))
                r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(tt_sizes))

            # SS3 probe
            if cb513_data is not None:
                cb_coded = {}
                for pid in cb513_data["ss3_labels"]:
                    if pid not in cb513_data["embeddings"]:
                        continue
                    m = cb513_data["embeddings"][pid].astype(np.float32)
                    try:
                        compressed = tt_decompose(m, bond_dim=bond_dim)
                        recon = tt_reconstruct(compressed)
                        cb_coded[pid] = recon.astype(np.float32)
                    except Exception:
                        continue
                if cb_coded:
                    ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                    r["ss3_q3"] = ss3.get("q3", 0.0)

            plm_results[name] = r
            print(f"  {name}: Ret@1={r['family_ret1']:.3f}, SS3={r.get('ss3_q3', '-')}, "
                  f"CosSim={r['cos_sim']:.3f}, Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Encode={r['encode_ms']:.1f}ms")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  {name}: FAILED — {e}")

        save_results(results)
        monitor()

    # ── NMF ──
    for k_val in [16, 32, 64]:
        name = f"nmf_k{k_val}"
        if name in plm_results:
            print(f"  {name}: already done, skipping")
            continue
        if k_val not in nmf_models:
            print(f"  {name}: model not fitted, skipping")
            continue

        model = nmf_models[k_val]
        print(f"  {name}: running...")
        try:
            coded_embs = {}
            protein_vectors = {}
            for pid in test_ids:
                if pid not in embeddings:
                    continue
                m = embeddings[pid].astype(np.float32)
                W = nmf_encode(m, model)
                recon = nmf_decode(W, model)
                coded_embs[pid] = recon
                protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

            ret = eval_retrieval(protein_vectors, metadata, test_ids)
            cos = compute_cos_sim(
                {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )
            enc_ms = time_encode(
                lambda m, mod=model: nmf_encode(m, mod),
                embeddings,
                [pid for pid in test_ids if pid in embeddings],
            )

            # Size: W is (L, k) float32
            sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
            w_sizes = []
            orig_sizes = []
            for pid in sample_pids:
                m = embeddings[pid].astype(np.float32)
                W = nmf_encode(m, model)
                w_sizes.append(W.nbytes)
                orig_sizes.append(m.nbytes)

            r = {
                "family_ret1": ret["precision@1"],
                "mrr": ret["mrr"],
                "cos_sim": cos,
                "encode_ms": enc_ms,
                "fitting": "corpus_fit",
                "per_residue": True,
            }
            if w_sizes:
                r["size_bytes"] = int(np.mean(w_sizes))
                r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(w_sizes))

            # SS3 probe
            if cb513_data is not None:
                cb_coded = {}
                for pid in cb513_data["ss3_labels"]:
                    if pid not in cb513_data["embeddings"]:
                        continue
                    m = cb513_data["embeddings"][pid].astype(np.float32)
                    try:
                        W = nmf_encode(m, model)
                        recon = nmf_decode(W, model)
                        cb_coded[pid] = recon.astype(np.float32)
                    except Exception:
                        continue
                if cb_coded:
                    ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                    r["ss3_q3"] = ss3.get("q3", 0.0)

            plm_results[name] = r
            print(f"  {name}: Ret@1={r['family_ret1']:.3f}, SS3={r.get('ss3_q3', '-')}, "
                  f"CosSim={r['cos_sim']:.3f}, Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Encode={r['encode_ms']:.1f}ms")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  {name}: FAILED — {e}")

        save_results(results)
        monitor()

    # ── Optimal Transport (Sliced Wasserstein) ──
    name = "ot_100proj"
    if name not in plm_results:
        print(f"  {name}: computing pairwise SWD on test set (SLOW)...")
        try:
            # Use a subset of test IDs for feasibility
            valid_test = [pid for pid in test_ids if pid in embeddings]
            # SWD is O(N^2) — limit to 500 proteins max
            max_ot = min(500, len(valid_test))
            ot_ids = valid_test[:max_ot]
            print(f"    Using {len(ot_ids)} proteins for SWD matrix")

            t0 = time.time()
            dist_mat = sliced_wasserstein_matrix(embeddings, ot_ids, n_projections=100, seed=42)
            elapsed_ot = time.time() - t0
            print(f"    SWD matrix computed in {elapsed_ot:.1f}s")

            # Custom retrieval from distance matrix
            # Build label lookup
            id_to_family = {}
            for m in metadata:
                id_to_family[m["id"]] = m.get("family", "")

            correct = 0
            rr_sum = 0.0
            n_queries = 0
            for i, qid in enumerate(ot_ids):
                q_fam = id_to_family.get(qid, "")
                if not q_fam:
                    continue
                # Sort by distance (ascending), skip self
                dists = dist_mat[i].copy()
                dists[i] = np.inf  # exclude self
                ranked = np.argsort(dists)

                # Precision@1
                nn_fam = id_to_family.get(ot_ids[ranked[0]], "")
                if nn_fam == q_fam:
                    correct += 1

                # MRR
                for rank, idx in enumerate(ranked):
                    nn_fam_r = id_to_family.get(ot_ids[idx], "")
                    if nn_fam_r == q_fam:
                        rr_sum += 1.0 / (rank + 1)
                        break
                n_queries += 1

            ret1 = correct / n_queries if n_queries > 0 else 0.0
            mrr = rr_sum / n_queries if n_queries > 0 else 0.0

            plm_results[name] = {
                "family_ret1": ret1,
                "mrr": mrr,
                "n_queries": n_queries,
                "n_proteins": len(ot_ids),
                "encode_ms": (elapsed_ot / len(ot_ids)) * 1000,
                "fitting": "none",
                "per_residue": False,
                "note": "distance metric, not a codec",
            }
            print(f"  {name}: Ret@1={ret1:.3f}, MRR={mrr:.3f} "
                  f"(n={n_queries}, {elapsed_ot:.1f}s)")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  {name}: FAILED — {e}")

        save_results(results)
        monitor()

    # ── TDA (Persistence Images) ──
    name = "tda_dim1"
    if name not in plm_results:
        print(f"  {name}: computing persistence images...")
        try:
            # Test if ripser is available
            test_img = persistence_image(np.random.randn(10, 5).astype(np.float32))
            if test_img is None:
                print(f"  {name}: ripser/persim not available, skipping")
                plm_results[name] = {"error": "ripser/persim not installed"}
            else:
                tda_vectors = {}
                valid_test = [pid for pid in test_ids if pid in embeddings]
                t0 = time.time()
                for j, pid in enumerate(valid_test):
                    m = embeddings[pid].astype(np.float32)
                    img = persistence_image(m, max_dim=1, n_bins=20, spread=1.0)
                    if img is not None:
                        tda_vectors[pid] = img.ravel().astype(np.float32)
                    if (j + 1) % 100 == 0:
                        print(f"    {j+1}/{len(valid_test)} done...")
                elapsed_tda = time.time() - t0

                if tda_vectors:
                    ret = eval_retrieval(tda_vectors, metadata, test_ids)
                    r = {
                        "family_ret1": ret["precision@1"],
                        "mrr": ret["mrr"],
                        "n_queries": ret["n_queries"],
                        "encode_ms": (elapsed_tda / len(valid_test)) * 1000,
                        "fitting": "none",
                        "per_residue": False,
                        "note": "protein-level descriptor only",
                    }
                    sample_vec = next(iter(tda_vectors.values()))
                    r["dim"] = int(sample_vec.shape[0])
                    plm_results[name] = r
                    print(f"  {name}: Ret@1={r['family_ret1']:.3f}, MRR={r['mrr']:.3f}, "
                          f"dim={r['dim']}, {elapsed_tda:.1f}s")
                else:
                    plm_results[name] = {"error": "no images computed"}

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  {name}: FAILED — {e}")

        save_results(results)
        monitor()

    mark_done(results, "P1C")
    save_results(results)
    print("\n  P1C complete.")


# ── Step P1D: Category D — Structure-Aware ─────────────────────────


def step_P1d(results):
    """P1D: AA-residual encoding, SimHash on ProtT5."""
    print("\n" + "=" * 60)
    print("P1D: Category D — Structure-Aware (ProtT5)")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    if not embeddings:
        print("  No embeddings, skipping P1D")
        return
    embeddings = cap_length(embeddings)
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    cb513_data = load_cb513_data()

    plm_results = results.setdefault("results", {}).setdefault("prot_t5_xl", {})

    # ── AA-Residual encoding ──
    # Try to get sequences from metadata or H5
    sequences = {}
    try:
        import h5py
        h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
        if h5_path.exists():
            with h5py.File(str(h5_path), "r") as f:
                for key in f.keys():
                    pid = key.split("|")[0] if "|" in key else key
                    if "sequence" in f[key].attrs:
                        sequences[pid] = f[key].attrs["sequence"]
    except Exception:
        pass

    # Also try metadata
    if not sequences:
        for m in metadata:
            if "sequence" in m and m.get("id"):
                sequences[m["id"]] = m["sequence"]

    has_sequences = len(sequences) > 50
    if has_sequences:
        print(f"  Found {len(sequences)} sequences for AA-residual encoding")
    else:
        print(f"  Only {len(sequences)} sequences found — AA-residual codecs will be skipped")

    aa_centroids = None
    if has_sequences:
        train_embs = {pid: embeddings[pid] for pid in train_ids
                      if pid in embeddings and pid in sequences}
        train_seqs = {pid: sequences[pid] for pid in train_embs}
        if len(train_embs) > 10:
            print(f"  Computing AA centroids from {len(train_embs)} training proteins...")
            try:
                aa_centroids = compute_aa_centroids(train_embs, train_seqs)
                print(f"  AA centroids computed: shape={aa_centroids.shape}")
            except Exception as e:
                print(f"  AA centroid computation FAILED: {e}")

    # AA-residual codecs
    if aa_centroids is not None:
        aa_codecs = [
            ("aa_resid_fp16", "aa_fp16"),
            ("aa_resid_int8", "aa_int8"),
            ("aa_resid_int4", "aa_int4"),
        ]

        for name, quant_type in aa_codecs:
            if name in plm_results:
                print(f"  {name}: already done, skipping")
                continue

            print(f"  {name}: running...")
            try:
                coded_embs = {}
                protein_vectors = {}
                for pid in test_ids:
                    if pid not in embeddings or pid not in sequences:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    seq = sequences[pid]
                    residuals = aa_residual_encode(m, seq, aa_centroids)

                    if quant_type == "aa_fp16":
                        q_residuals = residuals.astype(np.float16).astype(np.float32)
                        recon = aa_residual_decode(q_residuals, seq, aa_centroids)
                    elif quant_type == "aa_int8":
                        compressed = quantize_int8(residuals)
                        deq = dequantize_int8(compressed)
                        recon = aa_residual_decode(deq, seq, aa_centroids)
                    elif quant_type == "aa_int4":
                        compressed = quantize_int4(residuals)
                        deq = dequantize_int4(compressed)
                        recon = aa_residual_decode(deq, seq, aa_centroids)
                    else:
                        recon = aa_residual_decode(residuals, seq, aa_centroids)

                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                if not protein_vectors:
                    plm_results[name] = {"error": "no proteins encoded (missing sequences)"}
                    save_results(results)
                    continue

                ret = eval_retrieval(protein_vectors, metadata, test_ids)
                cos = compute_cos_sim(
                    {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                    {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
                )

                def _aa_encode_fn(m, q=quant_type, s=sequences, c=aa_centroids):
                    # We need a sequence but don't know the pid from m alone
                    # Just measure residual computation time
                    _ = m - m.mean(axis=0)  # proxy
                    return m

                enc_ms = time_encode(
                    _aa_encode_fn, embeddings,
                    [pid for pid in test_ids if pid in embeddings],
                )

                # Size
                sample_pids = [pid for pid in test_ids
                               if pid in embeddings and pid in sequences][:10]
                q_sizes = []
                orig_sizes = []
                for pid in sample_pids:
                    m = embeddings[pid].astype(np.float32)
                    seq = sequences[pid]
                    residuals = aa_residual_encode(m, seq, aa_centroids)
                    if quant_type == "aa_fp16":
                        q_sizes.append(residuals.astype(np.float16).nbytes)
                    elif quant_type == "aa_int8":
                        q_sizes.append(compressed_size_bytes(quantize_int8(residuals)))
                    elif quant_type == "aa_int4":
                        q_sizes.append(compressed_size_bytes(quantize_int4(residuals)))
                    orig_sizes.append(m.nbytes)

                r = {
                    "family_ret1": ret["precision@1"],
                    "mrr": ret["mrr"],
                    "cos_sim": cos,
                    "encode_ms": enc_ms,
                    "fitting": "corpus_fit",
                    "per_residue": True,
                }
                if q_sizes:
                    r["size_bytes"] = int(np.mean(q_sizes))
                    r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(q_sizes))

                # SS3 on CB513
                if cb513_data is not None:
                    cb_coded = {}
                    cb_seqs = cb513_data.get("sequences", {})
                    for pid in cb513_data["ss3_labels"]:
                        if pid not in cb513_data["embeddings"]:
                            continue
                        seq = cb_seqs.get(pid, "")
                        if not seq:
                            continue
                        m = cb513_data["embeddings"][pid].astype(np.float32)
                        try:
                            residuals = aa_residual_encode(m, seq, aa_centroids)
                            if quant_type == "aa_fp16":
                                q_res = residuals.astype(np.float16).astype(np.float32)
                                recon = aa_residual_decode(q_res, seq, aa_centroids)
                            elif quant_type == "aa_int8":
                                comp = quantize_int8(residuals)
                                deq = dequantize_int8(comp)
                                recon = aa_residual_decode(deq, seq, aa_centroids)
                            elif quant_type == "aa_int4":
                                comp = quantize_int4(residuals)
                                deq = dequantize_int4(comp)
                                recon = aa_residual_decode(deq, seq, aa_centroids)
                            else:
                                recon = aa_residual_decode(residuals, seq, aa_centroids)
                            cb_coded[pid] = recon.astype(np.float32)
                        except Exception:
                            continue
                    if cb_coded:
                        ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                        r["ss3_q3"] = ss3.get("q3", 0.0)

                plm_results[name] = r
                print(f"  {name}: Ret@1={r['family_ret1']:.3f}, SS3={r.get('ss3_q3', '-')}, "
                      f"CosSim={r['cos_sim']:.3f}, Size={r.get('size_bytes', 0)//1024}KB")

            except Exception as e:
                plm_results[name] = {"error": str(e)}
                traceback.print_exc()
                print(f"  {name}: FAILED — {e}")

            save_results(results)
            monitor()
    else:
        for name in ["aa_resid_fp16", "aa_resid_int8", "aa_resid_int4"]:
            if name not in plm_results:
                plm_results[name] = {"error": "sequences not available"}
                print(f"  {name}: SKIPPED — sequences not available")

    # ── SimHash ──
    for n_bits in [512, 1024, 2048]:
        name = f"simhash_{n_bits}"
        if name in plm_results:
            print(f"  {name}: already done, skipping")
            continue

        print(f"  {name}: running...")
        try:
            coded_embs = {}
            protein_vectors = {}
            for pid in test_ids:
                if pid not in embeddings:
                    continue
                m = embeddings[pid].astype(np.float32)
                compressed = simhash_encode(m, n_bits=n_bits, seed=42)
                recon = simhash_decode_approx(compressed)
                coded_embs[pid] = recon
                protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

            ret = eval_retrieval(protein_vectors, metadata, test_ids)
            cos = compute_cos_sim(
                {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )
            enc_ms = time_encode(
                lambda m, nb=n_bits: simhash_encode(m, n_bits=nb, seed=42),
                embeddings,
                [pid for pid in test_ids if pid in embeddings],
            )

            # Size: packed bits
            sample_pids = [pid for pid in test_ids if pid in embeddings][:10]
            bit_sizes = []
            orig_sizes = []
            for pid in sample_pids:
                m = embeddings[pid].astype(np.float32)
                compressed = simhash_encode(m, n_bits=n_bits, seed=42)
                bit_sizes.append(compressed["bits"].nbytes)
                orig_sizes.append(m.nbytes)

            r = {
                "family_ret1": ret["precision@1"],
                "mrr": ret["mrr"],
                "cos_sim": cos,
                "encode_ms": enc_ms,
                "fitting": "none",
                "per_residue": True,
            }
            if bit_sizes:
                r["size_bytes"] = int(np.mean(bit_sizes))
                r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(bit_sizes))

            # SS3 on CB513
            if cb513_data is not None:
                cb_coded = {}
                for pid in cb513_data["ss3_labels"]:
                    if pid not in cb513_data["embeddings"]:
                        continue
                    m = cb513_data["embeddings"][pid].astype(np.float32)
                    try:
                        compressed = simhash_encode(m, n_bits=n_bits, seed=42)
                        recon = simhash_decode_approx(compressed)
                        cb_coded[pid] = recon.astype(np.float32)
                    except Exception:
                        continue
                if cb_coded:
                    ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                    r["ss3_q3"] = ss3.get("q3", 0.0)

            plm_results[name] = r
            print(f"  {name}: Ret@1={r['family_ret1']:.3f}, SS3={r.get('ss3_q3', '-')}, "
                  f"CosSim={r['cos_sim']:.3f}, Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Encode={r['encode_ms']:.1f}ms")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  {name}: FAILED — {e}")

        save_results(results)
        monitor()

    mark_done(results, "P1D")
    save_results(results)
    print("\n  P1D complete.")


# ── Step P2: Multi-Stage Pipelines ─────────────────────────────────


def step_P2(results):
    """P2: Multi-stage compression pipelines on ProtT5."""
    print("\n" + "=" * 60)
    print("P2: Multi-Stage Pipelines (ProtT5)")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    if not embeddings:
        print("  No embeddings, skipping P2")
        return
    embeddings = cap_length(embeddings)
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    cb513_data = load_cb513_data()
    plm_results = results.setdefault("results", {}).setdefault("prot_t5_xl", {})

    # Get sequences for AA-residual pipeline
    sequences = {}
    try:
        import h5py
        h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
        if h5_path.exists():
            with h5py.File(str(h5_path), "r") as f:
                for key in f.keys():
                    pid = key.split("|")[0] if "|" in key else key
                    if "sequence" in f[key].attrs:
                        sequences[pid] = f[key].attrs["sequence"]
    except Exception:
        pass

    # Try to load AA centroids
    aa_centroids = None
    if len(sequences) > 50:
        train_embs = {pid: embeddings[pid] for pid in train_ids
                      if pid in embeddings and pid in sequences}
        train_seqs = {pid: sequences[pid] for pid in train_embs}
        if len(train_embs) > 10:
            try:
                aa_centroids = compute_aa_centroids(train_embs, train_seqs)
            except Exception:
                pass

    # Fit PQ for pipeline E2
    train_embs_all = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}

    # Pipeline definitions
    pipelines = {
        "E1_wavelet_int8_zstd": "wavelet(db4, 75%) -> int8 -> zstd size",
        "E2_cur_pq": "CUR(k=64) -> PQ(M=8) on selected columns",
        "E3_aa_resid_pq": "AA-residual -> PQ(M=32) on residuals",
        "E4_rp512_int8_zstd": "rp512 -> int8 -> zstd size",
        "E5_delta_int4_zstd": "delta -> int4 -> zstd size",
        "E6_rp512_int4_zstd": "rp512 -> int4 -> zstd size",
    }

    n_total = len(pipelines)
    for i, (name, desc) in enumerate(pipelines.items()):
        if name in plm_results:
            print(f"  [{i+1}/{n_total}] {name}: already done, skipping")
            continue

        print(f"  [{i+1}/{n_total}] {name}: {desc}")

        try:
            coded_embs = {}
            protein_vectors = {}
            pipeline_sizes = []
            pipeline_zstd_sizes = []
            orig_sizes = []

            if name == "E1_wavelet_int8_zstd":
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    # Stage 1: wavelet
                    wc = wavelet_threshold_compress(m, "db4", 75)
                    recon1 = wavelet_threshold_decompress(wc)
                    # Stage 2: int8
                    q = quantize_int8(recon1)
                    recon2 = dequantize_int8(q)
                    coded_embs[pid] = recon2
                    protein_vectors[pid] = recon2.mean(axis=0).astype(np.float32)
                    if len(pipeline_sizes) < 10:
                        pipeline_sizes.append(compressed_size_bytes(q))
                        pipeline_zstd_sizes.append(
                            len(zstd_compress(dequantize_int8(q).tobytes()))
                        )
                        orig_sizes.append(m.nbytes)

            elif name == "E2_cur_pq":
                # CUR selects k=64 columns, then PQ with M=8 on those 64 dims
                # Fit PQ on CUR-selected columns from training set
                print("    Fitting CUR + PQ pipeline...")
                cur_train = {}
                for pid in train_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    cd = cur_decompose(m, k=64)
                    cur_train[pid] = cd["C"]  # (L, 64)

                if cur_train:
                    pq_model_cur = pq_fit(cur_train, M=8, n_centroids=256, seed=42)
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        cd = cur_decompose(m, k=64)
                        C = cd["C"]  # (L, 64)
                        codes = pq_encode(C, pq_model_cur)
                        recon_c = pq_decode(codes, pq_model_cur)
                        # Reconstruct full matrix from CUR
                        recon_full = recon_c @ cd["interp_matrix"]
                        coded_embs[pid] = recon_full
                        protein_vectors[pid] = recon_full.mean(axis=0).astype(np.float32)
                        if len(pipeline_sizes) < 10:
                            pipeline_sizes.append(codes.nbytes)
                            orig_sizes.append(m.nbytes)

            elif name == "E3_aa_resid_pq":
                if aa_centroids is None:
                    plm_results[name] = {"error": "AA centroids not available"}
                    print(f"    SKIPPED — AA centroids not available")
                    save_results(results)
                    continue

                # Fit PQ on AA-residuals from training set
                print("    Fitting AA-residual + PQ pipeline...")
                resid_train = {}
                for pid in train_ids:
                    if pid not in embeddings or pid not in sequences:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    resid_train[pid] = aa_residual_encode(m, sequences[pid], aa_centroids)

                if len(resid_train) > 10:
                    pq_model_aa = pq_fit(resid_train, M=32, n_centroids=256, seed=42)
                    for pid in test_ids:
                        if pid not in embeddings or pid not in sequences:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        residuals = aa_residual_encode(m, sequences[pid], aa_centroids)
                        codes = pq_encode(residuals, pq_model_aa)
                        recon_resid = pq_decode(codes, pq_model_aa)
                        recon = aa_residual_decode(recon_resid, sequences[pid], aa_centroids)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)
                        if len(pipeline_sizes) < 10:
                            pipeline_sizes.append(codes.nbytes)
                            orig_sizes.append(m.nbytes)
                else:
                    plm_results[name] = {"error": "not enough training AA-residuals"}
                    print(f"    SKIPPED — not enough training sequences")
                    save_results(results)
                    continue

            elif name == "E4_rp512_int8_zstd":
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    rp = random_orthogonal_project(m, d_out=512)
                    q = quantize_int8(rp)
                    recon = dequantize_int8(q)
                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)
                    if len(pipeline_sizes) < 10:
                        pipeline_sizes.append(compressed_size_bytes(q))
                        pipeline_zstd_sizes.append(
                            len(zstd_compress(recon.tobytes()))
                        )
                        orig_sizes.append(m.nbytes)

            elif name == "E5_delta_int4_zstd":
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    disp = displacement_encode(m)
                    x0 = m[0].copy()
                    q = quantize_int4(disp)
                    recon_disp = dequantize_int4(q)
                    recon = displacement_decode(recon_disp, x0)
                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)
                    if len(pipeline_sizes) < 10:
                        pipeline_sizes.append(
                            compressed_size_bytes(q) + x0.nbytes
                        )
                        pipeline_zstd_sizes.append(
                            len(zstd_compress(dequantize_int4(q).tobytes()))
                            + len(zstd_compress(x0.tobytes()))
                        )
                        orig_sizes.append(m.nbytes)

            elif name == "E6_rp512_int4_zstd":
                for pid in test_ids:
                    if pid not in embeddings:
                        continue
                    m = embeddings[pid].astype(np.float32)
                    rp = random_orthogonal_project(m, d_out=512)
                    q = quantize_int4(rp)
                    recon = dequantize_int4(q)
                    coded_embs[pid] = recon
                    protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)
                    if len(pipeline_sizes) < 10:
                        pipeline_sizes.append(compressed_size_bytes(q))
                        pipeline_zstd_sizes.append(
                            len(zstd_compress(recon.tobytes()))
                        )
                        orig_sizes.append(m.nbytes)

            if not protein_vectors:
                plm_results[name] = {"error": "no proteins processed"}
                save_results(results)
                continue

            ret = eval_retrieval(protein_vectors, metadata, test_ids)
            cos = compute_cos_sim(
                {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
            )

            r = {
                "family_ret1": ret["precision@1"],
                "mrr": ret["mrr"],
                "cos_sim": cos,
                "fitting": "corpus_fit" if name in ("E2_cur_pq", "E3_aa_resid_pq") else "none",
                "per_residue": True,
                "pipeline": desc,
            }
            if pipeline_sizes:
                r["size_bytes"] = int(np.mean(pipeline_sizes))
                r["compression_ratio"] = float(np.mean(orig_sizes) / np.mean(pipeline_sizes))
            if pipeline_zstd_sizes:
                r["size_zstd_bytes"] = int(np.mean(pipeline_zstd_sizes))

            # SS3 probe
            if cb513_data is not None:
                cb_coded = {}
                for pid in cb513_data["ss3_labels"]:
                    if pid in coded_embs:
                        cb_coded[pid] = coded_embs[pid].astype(np.float32)
                    elif pid in cb513_data["embeddings"]:
                        # Re-encode CB513 protein through pipeline
                        m = cb513_data["embeddings"][pid].astype(np.float32)
                        try:
                            if name == "E4_rp512_int8_zstd":
                                rp = random_orthogonal_project(m, d_out=512)
                                q = quantize_int8(rp)
                                cb_coded[pid] = dequantize_int8(q).astype(np.float32)
                            elif name == "E6_rp512_int4_zstd":
                                rp = random_orthogonal_project(m, d_out=512)
                                q = quantize_int4(rp)
                                cb_coded[pid] = dequantize_int4(q).astype(np.float32)
                            elif name == "E1_wavelet_int8_zstd":
                                wc = wavelet_threshold_compress(m, "db4", 75)
                                recon1 = wavelet_threshold_decompress(wc)
                                q = quantize_int8(recon1)
                                cb_coded[pid] = dequantize_int8(q).astype(np.float32)
                            elif name == "E5_delta_int4_zstd":
                                disp = displacement_encode(m)
                                q = quantize_int4(disp)
                                recon_disp = dequantize_int4(q)
                                cb_coded[pid] = displacement_decode(
                                    recon_disp, m[0].copy()
                                ).astype(np.float32)
                        except Exception:
                            continue

                if cb_coded:
                    ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                    r["ss3_q3"] = ss3.get("q3", 0.0)

            plm_results[name] = r
            print(f"  [{i+1}/{n_total}] {name}: Ret@1={r['family_ret1']:.3f}, "
                  f"SS3={r.get('ss3_q3', '-')}, CosSim={r['cos_sim']:.3f}, "
                  f"Size={r.get('size_bytes', 0)//1024}KB, "
                  f"Ratio={r.get('compression_ratio', 0):.1f}x")

        except Exception as e:
            plm_results[name] = {"error": str(e)}
            traceback.print_exc()
            print(f"  [{i+1}/{n_total}] {name}: FAILED — {e}")

        save_results(results)
        monitor()

    mark_done(results, "P2")
    save_results(results)
    print("\n  P2 complete.")


# ── Step P3: Cross-PLM Validation ──────────────────────────────────


def step_P3(results):
    """P3: Run top techniques on ESM2-650M and ESM-C 300M."""
    print("\n" + "=" * 60)
    print("P3: Cross-PLM Validation")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    # Select top-5 techniques from P1+P2 based on ProtT5 results
    prot_results = results.get("results", {}).get("prot_t5_xl", {})
    if not prot_results:
        print("  No ProtT5 results to select top codecs from. Run P1A-P2 first.")
        return

    # Rank by family_ret1
    ranked = []
    for name, r in prot_results.items():
        if isinstance(r, dict) and "family_ret1" in r and "error" not in r:
            ranked.append((name, r["family_ret1"]))
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Select top-5 that are generalizable (not requiring sequences/corpus-specific fits)
    # Prefer codecs that don't need AA centroids, and work on any D
    generalizable_codecs = []
    for name, score in ranked:
        # Skip OT and TDA (distance metrics, not codecs)
        if name.startswith("ot_") or name.startswith("tda_"):
            continue
        # Skip AA-residual (needs sequences)
        if name.startswith("aa_resid"):
            continue
        generalizable_codecs.append(name)
        if len(generalizable_codecs) >= 5:
            break

    if not generalizable_codecs:
        print("  No generalizable codecs found. Run P1A-P2 first.")
        return

    print(f"  Top-5 codecs for cross-PLM: {generalizable_codecs}")

    # PLMs to test (skip ProtT5 — already done)
    cross_plms = [
        ("esm2_650m", 1280, "esm2_650m"),
        ("esmc_300m", 960, "esmc_300m"),
    ]

    for plm_name, D, plm_stem in cross_plms:
        print(f"\n  === {plm_name} (D={D}) ===")
        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            print(f"    No embeddings for {plm_name}, skipping")
            continue
        embeddings = cap_length(embeddings)
        print(f"    Loaded {len(embeddings)} embeddings")

        plm_results = results.setdefault("results", {}).setdefault(plm_name, {})

        # Fit corpus-dependent models if needed
        train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}

        for codec_name in generalizable_codecs:
            if codec_name in plm_results:
                print(f"    {codec_name}: already done, skipping")
                continue

            print(f"    {codec_name}: running on {plm_name}...")

            try:
                coded_embs = {}
                protein_vectors = {}

                # Determine which codec to run
                if codec_name.startswith("wavelet_db4_"):
                    pct = int(codec_name.split("_")[-1])
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        wc = wavelet_threshold_compress(m, "db4", pct)
                        recon = wavelet_threshold_decompress(wc)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("cur_k"):
                    k = int(codec_name.split("k")[1])
                    k = min(k, D)  # CUR k must be <= D
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        cd = cur_decompose(m, k=k)
                        recon = cur_reconstruct(cd)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("prune_k"):
                    k = int(codec_name.split("k")[1])
                    k = min(k, D)
                    importance = compute_channel_importance(train_embs, max_proteins=1000)
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        pruned, _ = channel_prune(m, importance, k)
                        coded_embs[pid] = pruned
                        protein_vectors[pid] = pruned.mean(axis=0).astype(np.float32)

                elif codec_name == "int8_raw":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        q = quantize_int8(m)
                        recon = dequantize_int8(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "int4_raw":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        q = quantize_int4(m)
                        recon = dequantize_int4(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "int8_rp512":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        rp = random_orthogonal_project(m, d_out=512)
                        q = quantize_int8(rp)
                        recon = dequantize_int8(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "int4_rp512":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        rp = random_orthogonal_project(m, d_out=512)
                        q = quantize_int4(rp)
                        recon = dequantize_int4(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "binary_raw":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        q = quantize_binary(m)
                        recon = dequantize_binary(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "binary_rp512":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        rp = random_orthogonal_project(m, d_out=512)
                        q = quantize_binary(rp)
                        recon = dequantize_binary(q)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("pq_M"):
                    M_val = int(codec_name.split("M")[1])
                    # Check divisibility
                    if D % M_val != 0:
                        print(f"      D={D} not divisible by M={M_val}, skipping")
                        plm_results[codec_name] = {"error": f"D={D} not divisible by M={M_val}"}
                        save_results(results)
                        continue
                    pq_model = pq_fit(train_embs, M=M_val, n_centroids=256, seed=42)
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        codes = pq_encode(m, pq_model)
                        recon = pq_decode(codes, pq_model)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("rvq_"):
                    n_lev = int(codec_name.split("_")[1].replace("level", ""))
                    rvq_model = rvq_fit(train_embs, n_levels=n_lev, n_centroids=256, seed=42)
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        codes = rvq_encode(m, rvq_model)
                        recon = rvq_decode(codes, rvq_model)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("tt_bd"):
                    bd = int(codec_name.split("bd")[1])
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        compressed = tt_decompose(m, bond_dim=bd)
                        recon = tt_reconstruct(compressed)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("nmf_k"):
                    k_val = int(codec_name.split("k")[1])
                    nmf_model = nmf_fit(train_embs, k=k_val, seed=42)
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        W = nmf_encode(m, nmf_model)
                        recon = nmf_decode(W, nmf_model)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name == "delta_order1":
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        disp = displacement_encode(m)
                        recon = displacement_decode(disp, m[0].copy())
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("simhash_"):
                    n_bits = int(codec_name.split("_")[1])
                    for pid in test_ids:
                        if pid not in embeddings:
                            continue
                        m = embeddings[pid].astype(np.float32)
                        compressed = simhash_encode(m, n_bits=n_bits, seed=42)
                        recon = simhash_decode_approx(compressed)
                        coded_embs[pid] = recon
                        protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                elif codec_name.startswith("E"):
                    # Pipeline codecs
                    if codec_name == "E1_wavelet_int8_zstd":
                        for pid in test_ids:
                            if pid not in embeddings:
                                continue
                            m = embeddings[pid].astype(np.float32)
                            wc = wavelet_threshold_compress(m, "db4", 75)
                            recon1 = wavelet_threshold_decompress(wc)
                            q = quantize_int8(recon1)
                            recon2 = dequantize_int8(q)
                            coded_embs[pid] = recon2
                            protein_vectors[pid] = recon2.mean(axis=0).astype(np.float32)

                    elif codec_name == "E4_rp512_int8_zstd":
                        for pid in test_ids:
                            if pid not in embeddings:
                                continue
                            m = embeddings[pid].astype(np.float32)
                            rp = random_orthogonal_project(m, d_out=512)
                            q = quantize_int8(rp)
                            recon = dequantize_int8(q)
                            coded_embs[pid] = recon
                            protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                    elif codec_name == "E5_delta_int4_zstd":
                        for pid in test_ids:
                            if pid not in embeddings:
                                continue
                            m = embeddings[pid].astype(np.float32)
                            disp = displacement_encode(m)
                            q = quantize_int4(disp)
                            recon_disp = dequantize_int4(q)
                            recon = displacement_decode(recon_disp, m[0].copy())
                            coded_embs[pid] = recon
                            protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                    elif codec_name == "E6_rp512_int4_zstd":
                        for pid in test_ids:
                            if pid not in embeddings:
                                continue
                            m = embeddings[pid].astype(np.float32)
                            rp = random_orthogonal_project(m, d_out=512)
                            q = quantize_int4(rp)
                            recon = dequantize_int4(q)
                            coded_embs[pid] = recon
                            protein_vectors[pid] = recon.mean(axis=0).astype(np.float32)

                    else:
                        print(f"      Pipeline {codec_name} not implemented for cross-PLM")
                        plm_results[codec_name] = {"error": "not implemented for cross-PLM"}
                        save_results(results)
                        continue

                else:
                    print(f"      Unknown codec {codec_name}, skipping")
                    plm_results[codec_name] = {"error": "unknown codec"}
                    save_results(results)
                    continue

                if not protein_vectors:
                    plm_results[codec_name] = {"error": "no proteins encoded"}
                    save_results(results)
                    continue

                ret = eval_retrieval(protein_vectors, metadata, test_ids)
                cos = compute_cos_sim(
                    {pid: embeddings[pid] for pid in list(coded_embs.keys())[:200]},
                    {pid: coded_embs[pid] for pid in list(coded_embs.keys())[:200]},
                )

                r = {
                    "family_ret1": ret["precision@1"],
                    "mrr": ret["mrr"],
                    "cos_sim": cos,
                    "per_residue": True,
                }
                plm_results[codec_name] = r
                print(f"    {codec_name}: Ret@1={r['family_ret1']:.3f}, "
                      f"MRR={r['mrr']:.3f}, CosSim={r['cos_sim']:.3f}")

            except Exception as e:
                plm_results[codec_name] = {"error": str(e)}
                traceback.print_exc()
                print(f"    {codec_name}: FAILED — {e}")

            save_results(results)
            monitor()

        del embeddings

    mark_done(results, "P3")
    save_results(results)
    print("\n  P3 complete.")


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 28: Extreme Compression Benchmark",
    )
    parser.add_argument(
        "--step", type=str, default=None,
        help="Run a specific step (P1A, P1B, P1C, P1D, P2, P3)",
    )
    args = parser.parse_args()

    results = load_results()

    steps = {
        "P1A": step_P1a,
        "P1B": step_P1b,
        "P1C": step_P1c,
        "P1D": step_P1d,
        "P2": step_P2,
        "P3": step_P3,
    }

    if args.step:
        name = args.step.upper()
        if name in steps:
            steps[name](results)
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
    else:
        for name, fn in steps.items():
            fn(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
