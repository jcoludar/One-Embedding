#!/usr/bin/env python3
"""Experiment 29: Exhaustive Low-Hanging Fruit Sweep.

Tests ~30 untried techniques identified through comprehensive project audit.

Parts:
  F:  Data characterization (intrinsic dim, channel analysis) — run FIRST
  A:  Pre-processing (centering, z-score, All-but-the-Top, PCA rotation)
  B:  Transposed matrix view (channel resample, per-protein SVD, channel stats)
  C:  Improved pooling (percentile, trimmed mean)
  D:  RP variants (multi-seed, sparse RP)
  E:  Quantization combinations (int4+codec, JPEG pipeline, predictive coding)
  G:  Evaluation enhancements (RNS, remote homology, MLP probes, Matryoshka)
  H:  Reference corpus approaches (k-means residual, PCA as D-compression)
  I:  Multi-resolution framing (3-level retrieval cascade)

Usage:
  uv run python experiments/29_exhaustive_fruit_sweep.py --step F
  uv run python experiments/29_exhaustive_fruit_sweep.py --step A
  uv run python experiments/29_exhaustive_fruit_sweep.py           # all steps
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
from scipy.spatial.distance import cdist

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    load_cb513_csv,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.utils.h5_store import load_residue_embeddings
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.transforms import dct_summary

# ── Constants ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "exhaustive_sweep_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512


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
        raw = json.load(f)
    # Normalize key names (file uses train_ids/test_ids)
    return {
        "train": raw.get("train_ids", raw.get("train", [])),
        "test": raw.get("test_ids", raw.get("test", [])),
    }


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


def eval_retrieval(vectors, metadata, test_ids, metric="cosine"):
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids, metric=metric,
    )


def load_cb513_data():
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if not cb513_path.exists():
        print("  CB513 not found, skipping per-residue probes")
        return None
    sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    cb_embs = load_plm_embeddings("prot_t5_xl", dataset="cb513")
    if not cb_embs:
        print("  CB513 embeddings not found")
        return None
    return {"ss3_labels": ss3_labels, "embeddings": cap_length(cb_embs)}


def eval_ss3(coded_cb_embs, ss3_labels):
    avail = [pid for pid in ss3_labels if pid in coded_cb_embs]
    if len(avail) < 10:
        return {"q3": 0.0}
    rng = random.Random(42)
    rng.shuffle(avail)
    n_train = int(len(avail) * 0.8)
    train_ids, test_ids_cb = avail[:n_train], avail[n_train:]
    return evaluate_ss3_probe(coded_cb_embs, ss3_labels, train_ids, test_ids_cb)


def time_encode(encode_fn, embeddings, sample_ids, n_timed=100):
    valid_ids = [pid for pid in sample_ids if pid in embeddings][:n_timed]
    if not valid_ids:
        return 0.0
    for pid in valid_ids[:10]:
        try:
            encode_fn(embeddings[pid])
        except Exception:
            pass
    t0 = time.perf_counter()
    for pid in valid_ids:
        encode_fn(embeddings[pid])
    elapsed = time.perf_counter() - t0
    return (elapsed / len(valid_ids)) * 1000.0


def benchmark_protein_vec(
    name, vec_fn, embeddings, test_ids, metadata, cb513_data=None,
    per_residue_fn=None, metric="cosine",
):
    """Benchmark a protein-level vector method."""
    result = {"codec": name}
    try:
        # Protein vectors
        vectors = {}
        for pid in test_ids:
            if pid not in embeddings:
                continue
            m = embeddings[pid].astype(np.float32)[:MAX_LEN]
            try:
                vectors[pid] = vec_fn(m)
            except Exception:
                continue

        if not vectors:
            return {"codec": name, "error": "no vectors"}

        # Retrieval
        ret = eval_retrieval(vectors, metadata, test_ids, metric=metric)
        result["family_ret1"] = ret["precision@1"]
        result["mrr"] = ret["mrr"]
        result["vec_dim"] = len(next(iter(vectors.values())))

        # SS3 probe (if per-residue function provided)
        if per_residue_fn is not None and cb513_data is not None:
            cb_coded = {}
            for pid in cb513_data["ss3_labels"]:
                if pid not in cb513_data["embeddings"]:
                    continue
                m = cb513_data["embeddings"][pid].astype(np.float32)
                try:
                    cb_coded[pid] = per_residue_fn(m).astype(np.float32)
                except Exception:
                    continue
            if cb_coded:
                ss3 = eval_ss3(cb_coded, cb513_data["ss3_labels"])
                result["ss3_q3"] = ss3.get("q3", 0.0)

        # Size estimate
        sample = list(vectors.values())[:10]
        if sample:
            result["vec_bytes"] = int(np.mean([v.nbytes for v in sample]))

        # Speed
        result["encode_ms"] = time_encode(
            vec_fn, embeddings,
            [pid for pid in test_ids if pid in embeddings],
        )

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()

    return result


# ══════════════════════════════════════════════════════════════════
# STEP F: Data Characterization
# ══════════════════════════════════════════════════════════════════

def step_F(results):
    """Part F: Intrinsic dimensionality + channel distributions."""
    print("\n" + "=" * 60)
    print("STEP F: Data Characterization")
    print("=" * 60)

    from src.one_embedding.data_analysis import (
        intrinsic_dimensionality,
        channel_distributions,
    )

    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    if not embeddings:
        print("  ERROR: no embeddings")
        return

    print(f"  Loaded {len(embeddings)} proteins")
    monitor()

    # F1: Intrinsic dimensionality — ProtT5
    print("\n--- F1: Intrinsic Dimensionality (ProtT5) ---")
    idim = intrinsic_dimensionality(embeddings, n_sample=50_000)
    print(f"  Participation ratio: {idim['participation_ratio']:.1f}")
    print(f"  Dims for 90% var: {idim['dims_90pct']} / {idim['total_dims']}")
    print(f"  Dims for 95% var: {idim['dims_95pct']} / {idim['total_dims']}")
    print(f"  Dims for 99% var: {idim['dims_99pct']} / {idim['total_dims']}")

    results.setdefault("data_analysis", {})["intrinsic_dim_prott5"] = {
        "participation_ratio": idim["participation_ratio"],
        "dims_90pct": idim["dims_90pct"],
        "dims_95pct": idim["dims_95pct"],
        "dims_99pct": idim["dims_99pct"],
        "total_dims": idim["total_dims"],
        "sv_top20": idim["singular_values"][:20],
    }

    # F2: Channel distributions
    print("\n--- F2: Channel Distributions (ProtT5) ---")
    chdist = channel_distributions(embeddings, n_sample=50_000)
    print(f"  Mean discriminative ratio: {chdist['mean_disc_ratio']:.3f}")
    print(f"  Top-10 discriminative channels: {chdist['top_10_discriminative_channels']}")
    print(f"  Mean abs inter-channel correlation: {chdist['mean_abs_correlation']:.3f}")
    print(f"  Highly correlated pairs (|r|>0.8): {chdist['n_highly_correlated_pairs']}")

    results["data_analysis"]["channel_dist_prott5"] = {
        "mean_disc_ratio": chdist["mean_disc_ratio"],
        "top_10_discriminative_channels": chdist["top_10_discriminative_channels"],
        "bottom_10_channels": chdist["bottom_10_channels"],
        "mean_abs_correlation": chdist["mean_abs_correlation"],
        "n_highly_correlated_pairs": chdist["n_highly_correlated_pairs"],
    }

    # F1b: ESM2
    print("\n--- F1b: Intrinsic Dimensionality (ESM2) ---")
    esm_embs = cap_length(load_plm_embeddings("esm2_650m"))
    if esm_embs:
        idim_esm = intrinsic_dimensionality(esm_embs, n_sample=50_000)
        print(f"  ESM2 participation ratio: {idim_esm['participation_ratio']:.1f}")
        print(f"  ESM2 dims for 95% var: {idim_esm['dims_95pct']} / {idim_esm['total_dims']}")
        results["data_analysis"]["intrinsic_dim_esm2"] = {
            "participation_ratio": idim_esm["participation_ratio"],
            "dims_90pct": idim_esm["dims_90pct"],
            "dims_95pct": idim_esm["dims_95pct"],
            "dims_99pct": idim_esm["dims_99pct"],
            "total_dims": idim_esm["total_dims"],
        }

    mark_done(results, "F")
    save_results(results)
    print("\n  Step F DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP A: Pre-Processing Transforms
# ══════════════════════════════════════════════════════════════════

def step_A(results):
    """Part A: Centering, z-score, All-but-the-Top, PCA rotation before RP."""
    print("\n" + "=" * 60)
    print("STEP A: Pre-Processing Transforms")
    print("=" * 60)

    from src.one_embedding.preprocessing import (
        compute_corpus_stats,
        center_embeddings,
        zscore_embeddings,
        all_but_the_top,
        pca_rotate,
    )

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    train_ids = split["train"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # Compute corpus stats from training set
    train_embs = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    print(f"  Computing corpus stats from {len(train_embs)} training proteins...")
    stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5)
    print(f"  Top-5 PC variance explained: {[f'{v:.4f}' for v in stats['explained_variance']]}")

    # Ground zero: raw rp512 + dct_K4 (reference)
    def _raw_rp_dct(m):
        return dct_summary(random_orthogonal_project(m, d_out=512), K=4)

    def _raw_rp(m):
        return random_orthogonal_project(m, d_out=512)

    print("\n  Ground zero: raw rp512+dct_K4...")
    r0 = benchmark_protein_vec("raw_rp512_dct_K4", _raw_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _raw_rp)
    results.setdefault("part_A", {})["raw_rp512_dct_K4"] = r0
    print(f"    Ret@1={r0.get('family_ret1', 'ERR')}, SS3={r0.get('ss3_q3', 'N/A')}")

    # A1: Centering
    def _centered_rp_dct(m):
        mc = center_embeddings(m, stats["mean_vec"])
        return dct_summary(random_orthogonal_project(mc, d_out=512), K=4)

    def _centered_rp(m):
        return random_orthogonal_project(center_embeddings(m, stats["mean_vec"]), d_out=512)

    print("\n  A1: centered + rp512 + dct_K4...")
    r1 = benchmark_protein_vec("centered_rp512_dct_K4", _centered_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _centered_rp)
    results["part_A"]["centered_rp512_dct_K4"] = r1
    print(f"    Ret@1={r1.get('family_ret1', 'ERR')}, SS3={r1.get('ss3_q3', 'N/A')}")

    # A2: Z-score
    def _zscore_rp_dct(m):
        mz = zscore_embeddings(m, stats["mean_vec"], stats["std_vec"])
        return dct_summary(random_orthogonal_project(mz, d_out=512), K=4)

    def _zscore_rp(m):
        return random_orthogonal_project(
            zscore_embeddings(m, stats["mean_vec"], stats["std_vec"]), d_out=512)

    print("\n  A2: z-score + rp512 + dct_K4...")
    r2 = benchmark_protein_vec("zscore_rp512_dct_K4", _zscore_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _zscore_rp)
    results["part_A"]["zscore_rp512_dct_K4"] = r2
    print(f"    Ret@1={r2.get('family_ret1', 'ERR')}, SS3={r2.get('ss3_q3', 'N/A')}")

    # A3: All-but-the-Top k=1,3
    for k in [1, 3]:
        def _abtt_rp_dct(m, _k=k):
            ma = all_but_the_top(m, stats["top_pcs"][:_k])
            return dct_summary(random_orthogonal_project(ma, d_out=512), K=4)

        def _abtt_rp(m, _k=k):
            return random_orthogonal_project(
                all_but_the_top(m, stats["top_pcs"][:_k]), d_out=512)

        name = f"abtt_k{k}_rp512_dct_K4"
        print(f"\n  A3: All-but-the-Top k={k} + rp512 + dct_K4...")
        r3 = benchmark_protein_vec(name, _abtt_rp_dct, embeddings,
                                    test_ids, metadata, cb513_data, _abtt_rp)
        results["part_A"][name] = r3
        print(f"    Ret@1={r3.get('family_ret1', 'ERR')}, SS3={r3.get('ss3_q3', 'N/A')}")

    # A4: PCA rotation
    def _pca_rot_rp_dct(m):
        mr = pca_rotate(m, stats["rotation_matrix"])
        return dct_summary(random_orthogonal_project(mr, d_out=512), K=4)

    def _pca_rot_rp(m):
        return random_orthogonal_project(pca_rotate(m, stats["rotation_matrix"]), d_out=512)

    print("\n  A4: PCA rotation + rp512 + dct_K4...")
    r4 = benchmark_protein_vec("pca_rot_rp512_dct_K4", _pca_rot_rp_dct, embeddings,
                                test_ids, metadata, cb513_data, _pca_rot_rp)
    results["part_A"]["pca_rot_rp512_dct_K4"] = r4
    print(f"    Ret@1={r4.get('family_ret1', 'ERR')}, SS3={r4.get('ss3_q3', 'N/A')}")

    # A5: Center + ABTT k=1 (combined best pre-processing)
    def _center_abtt_rp_dct(m):
        mc = center_embeddings(m, stats["mean_vec"])
        ma = all_but_the_top(mc, stats["top_pcs"][:1])
        return dct_summary(random_orthogonal_project(ma, d_out=512), K=4)

    def _center_abtt_rp(m):
        mc = center_embeddings(m, stats["mean_vec"])
        ma = all_but_the_top(mc, stats["top_pcs"][:1])
        return random_orthogonal_project(ma, d_out=512)

    print("\n  A5: center + ABTT k=1 + rp512 + dct_K4...")
    r5 = benchmark_protein_vec("center_abtt1_rp512_dct_K4", _center_abtt_rp_dct,
                                embeddings, test_ids, metadata, cb513_data, _center_abtt_rp)
    results["part_A"]["center_abtt1_rp512_dct_K4"] = r5
    print(f"    Ret@1={r5.get('family_ret1', 'ERR')}, SS3={r5.get('ss3_q3', 'N/A')}")

    # A6-A8: Pre-processing on raw mean pool (no RP)
    print("\n  A6-A8: Pre-processing on raw mean pool (no RP)...")
    for preproc_name, preproc_fn in [
        ("centered_mean", lambda m: center_embeddings(m, stats["mean_vec"]).mean(axis=0)),
        ("zscore_mean", lambda m: zscore_embeddings(m, stats["mean_vec"], stats["std_vec"]).mean(axis=0)),
        ("abtt1_mean", lambda m: all_but_the_top(m, stats["top_pcs"][:1]).mean(axis=0)),
    ]:
        r = benchmark_protein_vec(preproc_name, preproc_fn, embeddings, test_ids, metadata)
        results["part_A"][preproc_name] = r
        print(f"    {preproc_name}: Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "A")
    save_results(results)
    print("\n  Step A DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP B: Transposed Matrix View
# ══════════════════════════════════════════════════════════════════

def step_B(results):
    """Part B: Channel resampling, per-protein SVD, channel statistics."""
    print("\n" + "=" * 60)
    print("STEP B: Transposed Matrix View")
    print("=" * 60)

    from src.one_embedding.transposed_transforms import (
        channel_resample,
        per_protein_svd,
        channel_statistics,
    )

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # B1: Channel resampling
    for l_out in [32, 64, 128]:
        def _resample_vec(m, _l=l_out):
            return channel_resample(m, l_out=_l).flatten()

        name = f"channel_resample_l{l_out}"
        print(f"\n  B1: {name}...")
        r = benchmark_protein_vec(name, _resample_vec, embeddings, test_ids, metadata)
        results.setdefault("part_B", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B1b: RP + resample
    for l_out in [32, 64]:
        def _rp_resample(m, _l=l_out):
            compressed = random_orthogonal_project(m, d_out=512)
            return channel_resample(compressed, l_out=_l).flatten()

        name = f"rp512_resample_l{l_out}"
        print(f"\n  B1b: {name}...")
        r = benchmark_protein_vec(name, _rp_resample, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B2: Per-protein SVD
    for k in [1, 2, 4, 8]:
        def _svd_vec(m, _k=k):
            return per_protein_svd(m, k=_k)

        name = f"protein_svd_k{k}"
        print(f"\n  B2: {name}...")
        r = benchmark_protein_vec(name, _svd_vec, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B3: Channel statistics
    for stat_combo, stat_name in [
        (["mean", "std"], "mean_std"),
        (["mean", "std", "skew"], "mean_std_skew"),
        (["mean", "std", "min", "max"], "mean_std_min_max"),
    ]:
        def _chstat_vec(m, _stats=stat_combo):
            return channel_statistics(m, stats=_stats)

        name = f"channel_stats_{stat_name}"
        print(f"\n  B3: {name}...")
        r = benchmark_protein_vec(name, _chstat_vec, embeddings, test_ids, metadata)
        results["part_B"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # B4: [mean|std] simplest transposed insight
    def _mean_std_vec(m):
        return np.concatenate([m.mean(axis=0), m.std(axis=0)]).astype(np.float32)

    print(f"\n  B4: [mean|std] concatenation...")
    r = benchmark_protein_vec("mean_std_concat", _mean_std_vec, embeddings,
                               test_ids, metadata)
    results["part_B"]["mean_std_concat"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "B")
    save_results(results)
    print("\n  Step B DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP C: Improved Pooling
# ══════════════════════════════════════════════════════════════════

def step_C(results):
    """Part C: Percentile pooling, trimmed mean."""
    print("\n" + "=" * 60)
    print("STEP C: Improved Pooling Strategies")
    print("=" * 60)

    from src.one_embedding.universal_transforms import percentile_pool, trimmed_mean_pool

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # C1: Percentile pooling
    for pcts, pname in [
        ([25, 50, 75], "p25_50_75"),
        ([10, 50, 90], "p10_50_90"),
        ([10, 25, 50, 75, 90], "p10_25_50_75_90"),
    ]:
        def _pct_vec(m, _p=pcts):
            return percentile_pool(m, percentiles=_p)

        name = f"percentile_{pname}"
        print(f"\n  C1: {name}...")
        r = benchmark_protein_vec(name, _pct_vec, embeddings, test_ids, metadata)
        results.setdefault("part_C", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # C1b: [mean | IQR]
    def _mean_iqr(m):
        p25 = np.percentile(m, 25, axis=0)
        p75 = np.percentile(m, 75, axis=0)
        return np.concatenate([m.mean(axis=0), p75 - p25]).astype(np.float32)

    print(f"\n  C1b: [mean | IQR]...")
    r = benchmark_protein_vec("mean_iqr", _mean_iqr, embeddings, test_ids, metadata)
    results["part_C"]["mean_iqr"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    # C2: Trimmed mean
    for prop in [0.05, 0.1, 0.2]:
        def _trim_vec(m, _p=prop):
            return trimmed_mean_pool(m, proportion=_p)

        name = f"trimmed_mean_{int(prop*100)}pct"
        print(f"\n  C2: {name}...")
        r = benchmark_protein_vec(name, _trim_vec, embeddings, test_ids, metadata)
        results["part_C"][name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    # C3: [mean | max | std]
    def _mean_max_std(m):
        return np.concatenate([m.mean(axis=0), m.max(axis=0), m.std(axis=0)]).astype(np.float32)

    print(f"\n  C3: [mean|max|std]...")
    r = benchmark_protein_vec("mean_max_std", _mean_max_std, embeddings, test_ids, metadata)
    results["part_C"]["mean_max_std"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}, dim={r.get('vec_dim', '?')}")

    # C4: [mean | min | max] (captures range)
    def _mean_min_max(m):
        return np.concatenate([m.mean(axis=0), m.min(axis=0), m.max(axis=0)]).astype(np.float32)

    print(f"\n  C4: [mean|min|max]...")
    r = benchmark_protein_vec("mean_min_max", _mean_min_max, embeddings, test_ids, metadata)
    results["part_C"]["mean_min_max"] = r
    print(f"    Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "C")
    save_results(results)
    print("\n  Step C DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP D: RP Variants and Multi-Seed
# ══════════════════════════════════════════════════════════════════

def step_D(results):
    """Part D: Multi-seed RP variance, sparse RP."""
    print("\n" + "=" * 60)
    print("STEP D: RP Variants and Multi-Seed Characterization")
    print("=" * 60)

    from src.one_embedding.universal_transforms import sparse_random_project

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # D1: Multi-seed RP variance
    seeds = [42, 123, 456, 789, 0, 7, 99, 2024, 31415, 271828]
    seed_results = []
    print("\n  D1: Multi-seed RP variance (10 seeds)...")
    for seed in seeds:
        def _rp_dct_seed(m, _s=seed):
            return dct_summary(random_orthogonal_project(m, d_out=512, seed=_s), K=4)

        def _rp_seed(m, _s=seed):
            return random_orthogonal_project(m, d_out=512, seed=_s)

        r = benchmark_protein_vec(f"rp512_dct_K4_s{seed}", _rp_dct_seed,
                                   embeddings, test_ids, metadata, cb513_data, _rp_seed)
        seed_results.append(r)
        print(f"    seed={seed}: Ret@1={r.get('family_ret1', 'ERR'):.4f}, "
              f"SS3={r.get('ss3_q3', 'N/A')}")

    ret1_values = [r["family_ret1"] for r in seed_results if "family_ret1" in r]
    ss3_values = [r.get("ss3_q3", 0) for r in seed_results if r.get("ss3_q3")]
    results.setdefault("part_D", {})["multi_seed"] = {
        "seeds": seeds,
        "ret1_mean": float(np.mean(ret1_values)),
        "ret1_std": float(np.std(ret1_values)),
        "ret1_min": float(np.min(ret1_values)),
        "ret1_max": float(np.max(ret1_values)),
        "ss3_mean": float(np.mean(ss3_values)) if ss3_values else None,
        "ss3_std": float(np.std(ss3_values)) if ss3_values else None,
        "per_seed": {f"s{s}": r for s, r in zip(seeds, seed_results)},
    }
    print(f"\n  >>> Multi-seed Ret@1: {np.mean(ret1_values):.4f} +/- {np.std(ret1_values):.4f}")
    if ss3_values:
        print(f"  >>> Multi-seed SS3:   {np.mean(ss3_values):.4f} +/- {np.std(ss3_values):.4f}")

    # D2: Sparse RP (Achlioptas)
    print("\n  D2: Sparse random projection...")
    def _sparse_rp_dct(m):
        return dct_summary(sparse_random_project(m, d_out=512), K=4)

    def _sparse_rp(m):
        return sparse_random_project(m, d_out=512)

    r_sparse = benchmark_protein_vec("sparse_rp512_dct_K4", _sparse_rp_dct,
                                      embeddings, test_ids, metadata, cb513_data, _sparse_rp)
    results["part_D"]["sparse_rp512_dct_K4"] = r_sparse
    print(f"    Ret@1={r_sparse.get('family_ret1', 'ERR')}, SS3={r_sparse.get('ss3_q3', 'N/A')}")

    mark_done(results, "D")
    save_results(results)
    print("\n  Step D DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP E: Quantization and Coding Combinations
# ══════════════════════════════════════════════════════════════════

def step_E(results):
    """Part E: int4 on codec output, JPEG-style DCT+quantize, predictive coding."""
    print("\n" + "=" * 60)
    print("STEP E: Quantization & Coding Combinations")
    print("=" * 60)

    from src.one_embedding.quantization import (
        quantize_int4, dequantize_int4,
        quantize_int8, dequantize_int8,
    )
    from scipy.fft import dct as scipy_dct, idct as scipy_idct

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # E1: int4 of rp512 output
    print("\n  E1: int4 of rp512 output...")
    def _rp_int4_dct(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q = quantize_int4(compressed)
        deq = dequantize_int4(q)
        return dct_summary(deq, K=4)

    def _rp_int4(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q = quantize_int4(compressed)
        return dequantize_int4(q)

    r_e1 = benchmark_protein_vec("rp512_int4_dct_K4", _rp_int4_dct, embeddings,
                                  test_ids, metadata, cb513_data, _rp_int4)
    results.setdefault("part_E", {})["rp512_int4_dct_K4"] = r_e1
    print(f"    Ret@1={r_e1.get('family_ret1', 'ERR')}, SS3={r_e1.get('ss3_q3', 'N/A')}")

    # E1b: int8 of rp512
    print("\n  E1b: int8 of rp512 output...")
    def _rp_int8_dct(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q = quantize_int8(compressed)
        deq = dequantize_int8(q)
        return dct_summary(deq, K=4)

    def _rp_int8(m):
        compressed = random_orthogonal_project(m, d_out=512)
        q = quantize_int8(compressed)
        return dequantize_int8(q)

    r_e1b = benchmark_protein_vec("rp512_int8_dct_K4", _rp_int8_dct, embeddings,
                                   test_ids, metadata, cb513_data, _rp_int8)
    results["part_E"]["rp512_int8_dct_K4"] = r_e1b
    print(f"    Ret@1={r_e1b.get('family_ret1', 'ERR')}, SS3={r_e1b.get('ss3_q3', 'N/A')}")

    # E2: JPEG-style pipeline
    print("\n  E2: JPEG-style DCT + coefficient truncation...")
    for keep_frac in [0.25, 0.50, 0.75]:
        def _jpeg_encode(m, _frac=keep_frac):
            L, D = m.shape
            Xt = m.T.astype(np.float64)
            dct_coeffs = scipy_dct(Xt, type=2, norm='ortho', axis=1)
            n_keep = max(1, int(L * _frac))
            dct_coeffs[:, n_keep:] = 0
            reconstructed = scipy_idct(dct_coeffs, type=2, norm='ortho', axis=1)
            return reconstructed.T.astype(np.float32)

        def _jpeg_vec(m, _frac=keep_frac):
            return _jpeg_encode(m, _frac).mean(axis=0)

        def _jpeg_pr(m, _frac=keep_frac):
            return _jpeg_encode(m, _frac)

        name = f"jpeg_dct_keep{int(keep_frac*100)}pct"
        r = benchmark_protein_vec(name, _jpeg_vec, embeddings, test_ids, metadata,
                                   cb513_data, _jpeg_pr)
        results["part_E"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

    # E2b: JPEG + int4 on truncated coefficients
    print("\n  E2b: JPEG DCT 50% + int4 quantize coefficients...")
    def _jpeg_int4_encode(m):
        L, D = m.shape
        Xt = m.T.astype(np.float64)
        dct_coeffs = scipy_dct(Xt, type=2, norm='ortho', axis=1)
        n_keep = max(1, int(L * 0.5))
        truncated = dct_coeffs[:, :n_keep].T.astype(np.float32)
        q = quantize_int4(truncated)
        deq = dequantize_int4(q)
        full_coeffs = np.zeros((D, L), dtype=np.float64)
        full_coeffs[:, :n_keep] = deq.T
        return scipy_idct(full_coeffs, type=2, norm='ortho', axis=1).T.astype(np.float32)

    def _jpeg_int4_vec(m):
        return _jpeg_int4_encode(m).mean(axis=0)

    r_j4 = benchmark_protein_vec("jpeg_dct50_int4", _jpeg_int4_vec, embeddings,
                                  test_ids, metadata, cb513_data, _jpeg_int4_encode)
    results["part_E"]["jpeg_dct50_int4"] = r_j4
    print(f"    Ret@1={r_j4.get('family_ret1', 'ERR')}, SS3={r_j4.get('ss3_q3', 'N/A')}")

    # E3: DPCM (predictive coding)
    print("\n  E3: Predictive coding (DPCM order-1)...")
    # Note: lossless DPCM reconstructs exactly — test with int4 quantized deltas
    def _dpcm_int4_encode(m):
        L, D = m.shape
        first_row = m[0:1]  # (1, D) — stored exactly
        deltas = np.diff(m, axis=0)  # (L-1, D)
        q = quantize_int4(deltas)
        deq = dequantize_int4(q)
        recon = np.zeros_like(m)
        recon[0] = first_row[0]
        recon[1:] = first_row[0] + np.cumsum(deq, axis=0)
        return recon

    def _dpcm_int4_vec(m):
        return _dpcm_int4_encode(m).mean(axis=0)

    r_dpcm = benchmark_protein_vec("dpcm_int4", _dpcm_int4_vec, embeddings,
                                    test_ids, metadata, cb513_data, _dpcm_int4_encode)
    results["part_E"]["dpcm_int4"] = r_dpcm
    print(f"    Ret@1={r_dpcm.get('family_ret1', 'ERR')}, SS3={r_dpcm.get('ss3_q3', 'N/A')}")

    # E3b: DPCM + int8 (more precision for deltas)
    def _dpcm_int8_encode(m):
        L, D = m.shape
        first_row = m[0:1]
        deltas = np.diff(m, axis=0)
        q = quantize_int8(deltas)
        deq = dequantize_int8(q)
        recon = np.zeros_like(m)
        recon[0] = first_row[0]
        recon[1:] = first_row[0] + np.cumsum(deq, axis=0)
        return recon

    def _dpcm_int8_vec(m):
        return _dpcm_int8_encode(m).mean(axis=0)

    r_dpcm8 = benchmark_protein_vec("dpcm_int8", _dpcm_int8_vec, embeddings,
                                     test_ids, metadata, cb513_data, _dpcm_int8_encode)
    results["part_E"]["dpcm_int8"] = r_dpcm8
    print(f"    Ret@1={r_dpcm8.get('family_ret1', 'ERR')}, SS3={r_dpcm8.get('ss3_q3', 'N/A')}")

    # E4: Delta compressibility analysis
    print("\n  E4: Delta compressibility analysis...")
    raw_vars, delta_vars = [], []
    for pid in list(test_ids)[:200]:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        raw_vars.append(float(m.var()))
        delta_vars.append(float(np.diff(m, axis=0).var()))
    results["part_E"]["delta_analysis"] = {
        "mean_raw_var": float(np.mean(raw_vars)),
        "mean_delta_var": float(np.mean(delta_vars)),
        "variance_reduction": float(1 - np.mean(delta_vars) / np.mean(raw_vars)),
    }
    print(f"    Raw var: {np.mean(raw_vars):.4f}, Delta var: {np.mean(delta_vars):.4f}")
    print(f"    Variance reduction: {1 - np.mean(delta_vars)/np.mean(raw_vars):.1%}")

    mark_done(results, "E")
    save_results(results)
    print("\n  Step E DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP G: Evaluation Enhancements
# ══════════════════════════════════════════════════════════════════

def step_G(results):
    """Part G: RNS, remote homology, MLP probes, Matryoshka."""
    print("\n" + "=" * 60)
    print("STEP G: Evaluation Enhancements")
    print("=" * 60)

    from src.evaluation.retrieval import compute_rns

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # G1: RNS
    print("\n  G1: Random Neighbor Score...")
    raw_vecs = {pid: embeddings[pid].mean(axis=0) for pid in test_ids if pid in embeddings}
    rns_raw = compute_rns(raw_vecs, metadata, k=10, label_key="family")
    print(f"    Raw RNS: {rns_raw['mean_rns']:.4f} (high RNS proteins: {rns_raw['n_high_rns']})")

    codec_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        codec_vecs[pid] = dct_summary(random_orthogonal_project(m, d_out=512), K=4)
    rns_codec = compute_rns(codec_vecs, metadata, k=10, label_key="family")
    print(f"    Codec RNS: {rns_codec['mean_rns']:.4f} (high RNS: {rns_codec['n_high_rns']})")

    results.setdefault("part_G", {})["rns"] = {
        "raw_mean_pool": {k: v for k, v in rns_raw.items() if k != "per_protein_rns"},
        "codec_rp512_dct_K4": {k: v for k, v in rns_codec.items() if k != "per_protein_rns"},
    }

    # G2: Remote homology
    print("\n  G2: Remote homology (superfamily + fold)...")
    for level in ["superfamily", "fold"]:
        ret_raw = evaluate_retrieval_from_vectors(
            raw_vecs, metadata, label_key=level,
            query_ids=test_ids, database_ids=test_ids, metric="cosine",
        )
        ret_codec = evaluate_retrieval_from_vectors(
            codec_vecs, metadata, label_key=level,
            query_ids=test_ids, database_ids=test_ids, metric="cosine",
        )
        results["part_G"][f"remote_homology_{level}"] = {
            "raw_ret1": ret_raw["precision@1"],
            "codec_ret1": ret_codec["precision@1"],
            "raw_mrr": ret_raw["mrr"],
            "codec_mrr": ret_codec["mrr"],
        }
        print(f"    {level}: raw={ret_raw['precision@1']:.4f}, codec={ret_codec['precision@1']:.4f}")

    # G3: MLP probes
    print("\n  G3: MLP probes on SS3...")
    if cb513_data is not None:
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression

        avail = [pid for pid in cb513_data["ss3_labels"] if pid in cb513_data["embeddings"]]
        rng = random.Random(42)
        rng.shuffle(avail)
        n_train = int(len(avail) * 0.8)
        train_cb, test_cb = avail[:n_train], avail[n_train:]

        ss3_map = {"H": 0, "E": 1, "C": 2}
        X_tr_raw, X_te_raw, X_tr_cod, X_te_cod = [], [], [], []
        y_tr, y_te = [], []

        for pid in train_cb:
            emb = cb513_data["embeddings"][pid].astype(np.float32)
            labels = cb513_data["ss3_labels"][pid]
            ml = min(len(emb), len(labels))
            X_tr_raw.append(emb[:ml])
            X_tr_cod.append(random_orthogonal_project(emb, d_out=512)[:ml])
            y_tr.extend([ss3_map.get(l, 2) for l in labels[:ml]])

        for pid in test_cb:
            emb = cb513_data["embeddings"][pid].astype(np.float32)
            labels = cb513_data["ss3_labels"][pid]
            ml = min(len(emb), len(labels))
            X_te_raw.append(emb[:ml])
            X_te_cod.append(random_orthogonal_project(emb, d_out=512)[:ml])
            y_te.extend([ss3_map.get(l, 2) for l in labels[:ml]])

        X_tr_raw = np.vstack(X_tr_raw)
        X_te_raw = np.vstack(X_te_raw)
        X_tr_cod = np.vstack(X_tr_cod)
        X_te_cod = np.vstack(X_te_cod)
        y_tr, y_te = np.array(y_tr), np.array(y_te)

        # LR baseline
        lr_raw_q3 = float((LogisticRegression(max_iter=500, random_state=42)
                           .fit(X_tr_raw, y_tr).predict(X_te_raw) == y_te).mean())
        lr_cod_q3 = float((LogisticRegression(max_iter=500, random_state=42)
                           .fit(X_tr_cod, y_tr).predict(X_te_cod) == y_te).mean())

        # MLP
        mlp_raw_q3 = float((MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                            random_state=42, early_stopping=True)
                           .fit(X_tr_raw, y_tr).predict(X_te_raw) == y_te).mean())
        mlp_cod_q3 = float((MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                            random_state=42, early_stopping=True)
                           .fit(X_tr_cod, y_tr).predict(X_te_cod) == y_te).mean())

        results["part_G"]["mlp_probes"] = {
            "lr_raw_q3": lr_raw_q3, "lr_codec_q3": lr_cod_q3,
            "mlp_raw_q3": mlp_raw_q3, "mlp_codec_q3": mlp_cod_q3,
            "lr_gap": lr_raw_q3 - lr_cod_q3, "mlp_gap": mlp_raw_q3 - mlp_cod_q3,
        }
        print(f"    LR:  raw={lr_raw_q3:.4f}, codec={lr_cod_q3:.4f}, gap={lr_raw_q3-lr_cod_q3:.4f}")
        print(f"    MLP: raw={mlp_raw_q3:.4f}, codec={mlp_cod_q3:.4f}, gap={mlp_raw_q3-mlp_cod_q3:.4f}")

    # G4: Matryoshka dimension ordering
    print("\n  G4: Matryoshka dimension ordering...")
    train_embs = {pid: embeddings[pid] for pid in split["train"] if pid in embeddings}
    all_projected = []
    for pid in list(train_embs.keys())[:500]:
        projected = random_orthogonal_project(train_embs[pid][:MAX_LEN], d_out=512)
        all_projected.append(projected.mean(axis=0))
    all_projected = np.array(all_projected)
    dim_variance = all_projected.var(axis=0)
    sorted_dims = np.argsort(dim_variance)[::-1]

    for n_dims in [128, 256, 384, 512]:
        keep = sorted_dims[:n_dims]
        def _matr_mean(m, _keep=keep):
            compressed = random_orthogonal_project(m, d_out=512)
            return compressed.mean(axis=0)[_keep]

        name = f"matryoshka_mean_d{n_dims}"
        r = benchmark_protein_vec(name, _matr_mean, embeddings, test_ids, metadata)
        results["part_G"].setdefault("matryoshka", {})[name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}")

    # Random dim selection baseline
    rng_np = np.random.RandomState(42)
    random_dims = rng_np.permutation(512)
    for n_dims in [128, 256]:
        keep = random_dims[:n_dims]
        def _rand_mean(m, _keep=keep):
            compressed = random_orthogonal_project(m, d_out=512)
            return compressed.mean(axis=0)[_keep]

        name = f"random_subset_d{n_dims}"
        r = benchmark_protein_vec(name, _rand_mean, embeddings, test_ids, metadata)
        results["part_G"]["matryoshka"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}")

    mark_done(results, "G")
    save_results(results)
    print("\n  Step G DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP H: Reference Corpus Approaches
# ══════════════════════════════════════════════════════════════════

def step_H(results):
    """Part H: k-means centroid residual, PCA as D-compression."""
    print("\n" + "=" * 60)
    print("STEP H: Reference Corpus Approaches")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    train_ids = split["train"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))
    cb513_data = load_cb513_data()

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    # Collect training residues
    print("  Collecting training residues...")
    train_residues = np.vstack([
        embeddings[pid].astype(np.float32)[:MAX_LEN]
        for pid in train_ids if pid in embeddings
    ])
    print(f"  Training residues: {train_residues.shape}")

    # H1: k-means residual coding
    from sklearn.cluster import MiniBatchKMeans

    for n_clusters in [64, 256]:
        print(f"\n  H1: k-means residual k={n_clusters}...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 batch_size=10000, n_init=3)
        kmeans.fit(train_residues)
        centroids = kmeans.cluster_centers_

        def _km_vec(m, _km=kmeans, _c=centroids):
            labels = _km.predict(m)
            residuals = m - _c[labels]
            return np.concatenate([_c[labels].mean(axis=0),
                                   residuals.mean(axis=0)]).astype(np.float32)

        def _km_pr(m, _km=kmeans, _c=centroids):
            labels = _km.predict(m)
            return m - _c[labels]

        name = f"kmeans_residual_k{n_clusters}"
        r = benchmark_protein_vec(name, _km_vec, embeddings, test_ids, metadata,
                                   cb513_data, _km_pr)
        results.setdefault("part_H", {})[name] = r
        print(f"    Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

    # H2: PCA as D-compression
    from sklearn.decomposition import PCA

    if len(train_residues) > 50_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_residues), 50_000, replace=False)
        pca_train = train_residues[idx]
    else:
        pca_train = train_residues

    for n_comp in [256, 512]:
        print(f"\n  H2: PCA d={n_comp}...")
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(pca_train)
        var_exp = pca.explained_variance_ratio_.sum()
        print(f"    Variance explained: {var_exp:.4f}")

        def _pca_dct(m, _pca=pca):
            return dct_summary(_pca.transform(m).astype(np.float32), K=4)

        def _pca_pr(m, _pca=pca):
            return _pca.transform(m).astype(np.float32)

        name = f"pca{n_comp}_dct_K4"
        r = benchmark_protein_vec(name, _pca_dct, embeddings, test_ids, metadata,
                                   cb513_data, _pca_pr)
        results["part_H"][name] = r
        print(f"    {name}: Ret@1={r.get('family_ret1', 'ERR')}, SS3={r.get('ss3_q3', 'N/A')}")

        # PCA mean pool (no DCT)
        def _pca_mean(m, _pca=pca):
            return _pca.transform(m).astype(np.float32).mean(axis=0)

        name2 = f"pca{n_comp}_mean"
        r2 = benchmark_protein_vec(name2, _pca_mean, embeddings, test_ids, metadata)
        results["part_H"][name2] = r2
        print(f"    {name2}: Ret@1={r2.get('family_ret1', 'ERR')}")

    del train_residues, pca_train

    mark_done(results, "H")
    save_results(results)
    print("\n  Step H DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# STEP I: Multi-Resolution Framing
# ══════════════════════════════════════════════════════════════════

def step_I(results):
    """Part I: Three-level retrieval cascade."""
    print("\n" + "=" * 60)
    print("STEP I: Multi-Resolution Retrieval")
    print("=" * 60)

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test"]
    embeddings = cap_length(load_plm_embeddings("prot_t5_xl"))

    if not embeddings:
        print("  ERROR: no embeddings")
        return

    from src.one_embedding.topological import simhash_encode

    # Level 1: SimHash
    print("\n  Level 1: SimHash 1024-bit (Hamming)...")
    sh_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        sh_result = simhash_encode(embeddings[pid][:MAX_LEN].astype(np.float32),
                                    n_bits=1024)
        # Unpack bits from dict, convert to float for distance computation
        bits = np.unpackbits(sh_result["bits"].flatten())[:1024]
        sh_vecs[pid] = bits.astype(np.float32)

    ret_sh = eval_retrieval(sh_vecs, metadata, test_ids, metric="cosine")
    print(f"    SimHash Ret@1={ret_sh['precision@1']:.4f}")

    # Level 2: Codec protein_vec
    print("\n  Level 2: Codec protein_vec 2048d (cosine)...")
    codec_vecs = {}
    for pid in test_ids:
        if pid not in embeddings:
            continue
        m = embeddings[pid].astype(np.float32)[:MAX_LEN]
        codec_vecs[pid] = dct_summary(random_orthogonal_project(m, d_out=512), K=4)

    ret_codec = eval_retrieval(codec_vecs, metadata, test_ids)
    print(f"    Codec Ret@1={ret_codec['precision@1']:.4f}")

    # Cascade: SimHash top-100 → codec re-rank
    print("\n  Cascade: SimHash top-100 → codec rerank...")
    pids_list = [pid for pid in test_ids if pid in sh_vecs and pid in codec_vecs]
    sh_matrix = np.array([sh_vecs[pid] for pid in pids_list])
    codec_matrix = np.array([codec_vecs[pid] for pid in pids_list])
    id_to_family = {m.get("protein_id", m.get("id", "")): m.get("family", "") for m in metadata}

    sh_dists = cdist(sh_matrix, sh_matrix, metric="cosine")

    correct, total = 0, 0
    for i, pid in enumerate(pids_list):
        if pid not in id_to_family or not id_to_family[pid]:
            continue
        top100 = np.argsort(sh_dists[i])[1:101]
        query_codec = codec_vecs[pid]
        cands = np.array([codec_matrix[j] for j in top100])
        norms = np.linalg.norm(cands, axis=1) * np.linalg.norm(query_codec)
        cos_sims = (cands @ query_codec) / np.maximum(norms, 1e-10)
        best_j = top100[np.argmax(cos_sims)]
        if id_to_family.get(pids_list[best_j]) == id_to_family[pid]:
            correct += 1
        total += 1

    cascade_ret1 = correct / total if total > 0 else 0
    print(f"    Cascade Ret@1={cascade_ret1:.4f}")

    results.setdefault("part_I", {})["multi_resolution"] = {
        "level1_simhash_ret1": ret_sh["precision@1"],
        "level2_codec_ret1": ret_codec["precision@1"],
        "cascade_sh100_codec_ret1": cascade_ret1,
    }

    mark_done(results, "I")
    save_results(results)
    print("\n  Step I DONE")
    monitor()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

ALL_STEPS = {
    "F": step_F, "A": step_A, "B": step_B, "C": step_C,
    "D": step_D, "E": step_E, "G": step_G, "H": step_H, "I": step_I,
}
STEP_ORDER = ["F", "A", "B", "C", "D", "E", "G", "H", "I"]


def main():
    parser = argparse.ArgumentParser(description="Experiment 29: Exhaustive Fruit Sweep")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step(s), comma-separated (F/A/B/C/D/E/G/H/I).")
    args = parser.parse_args()

    results = load_results()

    steps = [s.strip() for s in args.step.split(",")] if args.step else STEP_ORDER

    for step in steps:
        if step in results.get("steps_done", []):
            print(f"\n  Step {step} already done, skipping.")
            continue
        if step not in ALL_STEPS:
            print(f"\n  Unknown step: {step}")
            continue
        ALL_STEPS[step](results)

    print("\n" + "=" * 60)
    print("EXPERIMENT 29 SUMMARY")
    print("=" * 60)
    print(f"Steps completed: {results.get('steps_done', [])}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
