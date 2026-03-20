#!/usr/bin/env python3
"""Experiment 38: Projection quality diagnostic.

Traces per-residue quality loss through RP vs PCA at multiple dimensions.
Identifies which raw embedding channels carry TM/disorder signal.

Steps:
    A: RP dimension sweep (256, 384, 512, 640, 768, 896, 1024) — SS3, SS8, disorder, TM, retrieval
    B: PCA dimension sweep (same dimensions) — same metrics
    C: Channel importance for TM and disorder
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.transforms import dct_summary
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, evaluate_ss8_probe,
    evaluate_disorder_probe, evaluate_tm_probe,
    load_cb513_csv, load_chezod_seth, load_tmbed_annotated,
)
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size
from src.utils.h5_store import load_residue_embeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "projection_diagnostic_results.json"

EMB_5K = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
EMB_CB513 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
EMB_CHEZOD = DATA_DIR / "residue_embeddings" / "prot_t5_xl_chezod.h5"
EMB_VALID = DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"
META_5K = DATA_DIR / "proteins" / "metadata_5k.csv"
SPLIT_5K = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
CB513_CSV = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
# TMbed annotated FASTA is in the TMbed subdirectory
TMBED_FASTA = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"

DIMS = [256, 384, 512, 640, 768, 896, 1024]


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": []}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Helper: load tmbed embeddings (stored with 'tmbed_' prefix in validation h5)
# ---------------------------------------------------------------------------

def load_tmbed_embeddings() -> dict:
    """Load TMbed embeddings from validation H5, stripping the 'tmbed_' prefix.

    Uses h5py directly to avoid loading all chezod_ keys into memory.
    """
    import h5py
    result = {}
    if EMB_VALID.exists():
        with h5py.File(str(EMB_VALID), "r") as f:
            for key in f.keys():
                if key.startswith("tmbed_"):
                    original_id = key[len("tmbed_"):]
                    result[original_id] = np.array(f[key], dtype=np.float32)
    if result:
        print(f"  Loaded {len(result)} TMbed embeddings from {EMB_VALID.name}")
    return result


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_pca(embeddings: dict, rotation_matrix: np.ndarray,
                mean_vec: np.ndarray, d_out: int) -> dict:
    """Apply PCA projection: center, rotate, truncate to top d_out components.

    rotation_matrix: (D, D) where rows are PC directions (from compute_corpus_stats).
    For d_out >= D, applies full rotation without truncation.
    """
    D = rotation_matrix.shape[1]
    if d_out >= D:
        # Full PCA rotation, no truncation: R rows are PC directions; X @ R.T
        R = rotation_matrix.T  # (D, D)
    else:
        R = rotation_matrix[:d_out].T  # (D, d_out) — top d_out PCs as columns
    result = {}
    for pid, emb in embeddings.items():
        centered = emb.astype(np.float32) - mean_vec
        result[pid] = (centered @ R).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_all_per_residue(embeddings_cb513, embeddings_chezod, embeddings_tm,
                              ss3_labels, ss8_labels,
                              disorder_scores, disorder_train, disorder_test,
                              tm_labels, tm_train, tm_test,
                              cb_train, cb_test):
    """Run all 4 per-residue probes. Returns dict of metrics."""
    results = {}

    ss3 = evaluate_ss3_probe(embeddings_cb513, ss3_labels, cb_train, cb_test)
    results["ss3_q3"] = ss3["q3"]

    ss8 = evaluate_ss8_probe(embeddings_cb513, ss8_labels, cb_train, cb_test)
    results["ss8_q8"] = ss8["q8"]

    disorder = evaluate_disorder_probe(
        embeddings_chezod, disorder_scores, disorder_train, disorder_test
    )
    results["disorder_rho"] = disorder["spearman_rho"]

    tm = evaluate_tm_probe(embeddings_tm, tm_labels, tm_train, tm_test)
    results["tm_f1"] = tm.get("macro_f1", tm.get("accuracy", 0.0))

    return results


def evaluate_retrieval(embeddings_5k, metadata, test_ids):
    """Compute protein vectors via DCT K=4 and evaluate retrieval."""
    vecs = {}
    for pid, emb in embeddings_5k.items():
        vecs[pid] = dct_summary(emb, K=4)
    ret = evaluate_retrieval_from_vectors(
        vecs, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )
    return {"family_ret1": ret.get("precision@1", 0.0), "mrr": ret.get("mrr", 0.0)}


# ---------------------------------------------------------------------------
# Step A: RP dimension sweep
# ---------------------------------------------------------------------------

def step_A(results):
    """RP dimension sweep across all 7 dims — all per-residue tasks + retrieval."""
    if "A" in results.get("steps_done", []):
        print("Step A already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step A: RP Dimension Sweep")
    print("=" * 60)

    # Load data
    print("  Loading embeddings...")
    emb_5k = load_residue_embeddings(EMB_5K)
    emb_cb513 = load_residue_embeddings(EMB_CB513)
    emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    emb_tm = load_tmbed_embeddings()

    print(f"  CB513: {len(emb_cb513)}, CheZOD: {len(emb_chezod)}, TM: {len(emb_tm)}, 5K: {len(emb_5k)}")

    # Load labels
    _, ss3_labels, ss8_labels, _ = load_cb513_csv(CB513_CSV)
    _, disorder_scores, disorder_train_ids, disorder_test_ids = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    # CB513 split (80/20 of available proteins with embeddings)
    cb513_avail = sorted(set(emb_cb513.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]
    print(f"  CB513 available: {len(cb513_avail)} (train={len(cb_train)}, test={len(cb_test)})")

    # TM split — proteins with both embeddings and labels
    tm_avail = sorted(set(emb_tm.keys()) & set(tm_labels.keys()))
    rng2 = random.Random(42)
    rng2.shuffle(tm_avail)
    n_tr_tm = int(len(tm_avail) * 0.8)
    tm_train, tm_test = tm_avail[:n_tr_tm], tm_avail[n_tr_tm:]
    print(f"  TM available: {len(tm_avail)} (train={len(tm_train)}, test={len(tm_test)})")

    # Disorder uses the SETH pre-defined train/test split
    dis_train = [p for p in disorder_train_ids if p in emb_chezod]
    dis_test = [p for p in disorder_test_ids if p in emb_chezod]
    print(f"  Disorder: {len(dis_train)} train, {len(dis_test)} test")

    # Retrieval metadata + split
    metadata = load_metadata_csv(META_5K)
    metadata, kept_ids = filter_by_family_size(metadata, min_members=3)
    with open(SPLIT_5K) as f:
        split = json.load(f)
    ret_test_ids = [i for i in split["test_ids"] if i in kept_ids]
    print(f"  Retrieval: {len(ret_test_ids)} test IDs")

    # Compute corpus stats (for ABTT3 top-3 PCs)
    print("  Computing corpus stats (ABTT3 top-3 PCs)...")
    stats = compute_corpus_stats(emb_5k, n_sample=50_000, n_pcs=3, seed=42)
    top3 = stats["top_pcs"]  # (3, D)
    mean_vec = stats["mean_vec"]

    # --- Raw 1024d baseline ---
    print("\n  Evaluating raw 1024d baseline...")
    t0 = time.time()
    raw_pr = evaluate_all_per_residue(
        emb_cb513, emb_chezod, emb_tm,
        ss3_labels, ss8_labels,
        disorder_scores, dis_train, dis_test,
        tm_labels, tm_train, tm_test,
        cb_train, cb_test,
    )
    elapsed_raw = time.time() - t0
    print(f"    Raw: SS3={raw_pr['ss3_q3']:.3f} SS8={raw_pr['ss8_q8']:.3f} "
          f"Dis={raw_pr['disorder_rho']:.3f} TM={raw_pr['tm_f1']:.3f} ({elapsed_raw:.1f}s)")

    step_results = {"raw_1024d": raw_pr, "rp": {}}

    # --- RP sweep ---
    for d_out in DIMS:
        print(f"\n  RP d={d_out}...")
        t0 = time.time()

        # Apply ABTT3 first, then RP (or skip RP if already at full dimension)
        def apply_abtt_rp(embs):
            projected = {}
            for pid, emb in embs.items():
                e = all_but_the_top(
                    (emb.astype(np.float32) - mean_vec), top3
                )
                if d_out < e.shape[1]:
                    projected[pid] = random_orthogonal_project(e, d_out=d_out, seed=42)
                else:
                    projected[pid] = e  # No projection at full dim
            return projected

        cb_proj = apply_abtt_rp(emb_cb513)
        cz_proj = apply_abtt_rp(emb_chezod)
        tm_proj = apply_abtt_rp(emb_tm)
        fk_proj = apply_abtt_rp(emb_5k)

        # Per-residue evaluation
        pr = evaluate_all_per_residue(
            cb_proj, cz_proj, tm_proj,
            ss3_labels, ss8_labels,
            disorder_scores, dis_train, dis_test,
            tm_labels, tm_train, tm_test,
            cb_train, cb_test,
        )

        # Retrieval
        ret = evaluate_retrieval(fk_proj, metadata, ret_test_ids)

        elapsed = time.time() - t0
        step_results["rp"][str(d_out)] = {**pr, **ret, "elapsed_s": round(elapsed, 1)}
        print(f"    RP-{d_out}: SS3={pr['ss3_q3']:.3f} SS8={pr['ss8_q8']:.3f} "
              f"Dis={pr['disorder_rho']:.3f} TM={pr['tm_f1']:.3f} "
              f"Ret@1={ret['family_ret1']:.3f} ({elapsed:.1f}s)")

    results["A"] = step_results
    results["steps_done"].append("A")
    save_results(results)
    print("\n  Step A complete!")
    return results


# ---------------------------------------------------------------------------
# Step B: PCA dimension sweep
# ---------------------------------------------------------------------------

def step_B(results):
    """PCA dimension sweep across all 7 dims — all per-residue tasks + retrieval."""
    if "B" in results.get("steps_done", []):
        print("Step B already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step B: PCA Dimension Sweep")
    print("=" * 60)

    # Load same data as Step A
    print("  Loading embeddings...")
    emb_5k = load_residue_embeddings(EMB_5K)
    emb_cb513 = load_residue_embeddings(EMB_CB513)
    emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    emb_tm = load_tmbed_embeddings()

    _, ss3_labels, ss8_labels, _ = load_cb513_csv(CB513_CSV)
    _, disorder_scores, disorder_train_ids, disorder_test_ids = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    cb513_avail = sorted(set(emb_cb513.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    tm_avail = sorted(set(emb_tm.keys()) & set(tm_labels.keys()))
    rng2 = random.Random(42)
    rng2.shuffle(tm_avail)
    n_tr_tm = int(len(tm_avail) * 0.8)
    tm_train, tm_test = tm_avail[:n_tr_tm], tm_avail[n_tr_tm:]

    dis_train = [p for p in disorder_train_ids if p in emb_chezod]
    dis_test = [p for p in disorder_test_ids if p in emb_chezod]

    metadata = load_metadata_csv(META_5K)
    metadata, kept_ids = filter_by_family_size(metadata, min_members=3)
    with open(SPLIT_5K) as f:
        split = json.load(f)
    ret_test_ids = [i for i in split["test_ids"] if i in kept_ids]

    # Compute full PCA on 5K corpus (rotation matrix + mean)
    print("  Computing full PCA rotation matrix...")
    stats = compute_corpus_stats(emb_5k, n_sample=50_000, n_pcs=3, seed=42)
    rotation = stats["rotation_matrix"]  # (D, D) where rows are PC directions
    mean_vec = stats["mean_vec"]         # (D,)
    print(f"  Rotation matrix shape: {rotation.shape}")

    step_results = {"pca": {}}

    for d_out in DIMS:
        print(f"\n  PCA d={d_out}...")
        t0 = time.time()

        # PCA project: center + rotate + truncate
        cb_proj = project_pca(emb_cb513, rotation, mean_vec, d_out)
        cz_proj = project_pca(emb_chezod, rotation, mean_vec, d_out)
        tm_proj = project_pca(emb_tm, rotation, mean_vec, d_out)
        fk_proj = project_pca(emb_5k, rotation, mean_vec, d_out)

        # Per-residue evaluation
        pr = evaluate_all_per_residue(
            cb_proj, cz_proj, tm_proj,
            ss3_labels, ss8_labels,
            disorder_scores, dis_train, dis_test,
            tm_labels, tm_train, tm_test,
            cb_train, cb_test,
        )

        # Retrieval (DCT K=4 protein vector)
        ret = evaluate_retrieval(fk_proj, metadata, ret_test_ids)

        elapsed = time.time() - t0
        step_results["pca"][str(d_out)] = {**pr, **ret, "elapsed_s": round(elapsed, 1)}
        print(f"    PCA-{d_out}: SS3={pr['ss3_q3']:.3f} SS8={pr['ss8_q8']:.3f} "
              f"Dis={pr['disorder_rho']:.3f} TM={pr['tm_f1']:.3f} "
              f"Ret@1={ret['family_ret1']:.3f} ({elapsed:.1f}s)")

    results["B"] = step_results
    results["steps_done"].append("B")
    save_results(results)
    print("\n  Step B complete!")
    return results


# ---------------------------------------------------------------------------
# Step C: Channel importance analysis
# ---------------------------------------------------------------------------

def step_C(results):
    """Channel importance: which raw 1024d channels matter for TM and disorder?"""
    if "C" in results.get("steps_done", []):
        print("Step C already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step C: Channel Importance Analysis")
    print("=" * 60)

    # Load embeddings + labels
    emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    emb_tm = load_tmbed_embeddings()

    _, disorder_scores, disorder_train_ids, disorder_test_ids = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    # Compute per-channel Spearman correlation with disorder scores
    print("  Computing per-channel disorder correlation...")
    dis_proteins = sorted(set(emb_chezod.keys()) & set(disorder_scores.keys()))
    print(f"  Disorder proteins with embeddings: {len(dis_proteins)}")
    all_emb_rows = []
    all_dis_scores = []
    for pid in dis_proteins:
        emb = emb_chezod[pid].astype(np.float32)
        scores = disorder_scores[pid]
        L = min(len(scores), emb.shape[0])
        valid_mask = ~np.isnan(scores[:L])
        if valid_mask.sum() > 0:
            all_emb_rows.append(emb[:L][valid_mask])
            all_dis_scores.append(scores[:L][valid_mask])

    X_dis = np.vstack(all_emb_rows)   # (N_residues, 1024)
    y_dis = np.concatenate(all_dis_scores)  # (N_residues,)
    D = X_dis.shape[1]
    print(f"  Total residues for disorder: {X_dis.shape[0]}")

    # Subsample for speed if very large
    max_rows = 100_000
    if X_dis.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_dis.shape[0], size=max_rows, replace=False)
        X_dis_sub = X_dis[idx]
        y_dis_sub = y_dis[idx]
        print(f"  Subsampled to {max_rows} residues for correlation computation")
    else:
        X_dis_sub = X_dis
        y_dis_sub = y_dis

    from scipy.stats import spearmanr
    dis_channel_corr = np.zeros(D, dtype=np.float32)
    for ch in range(D):
        rho, _ = spearmanr(X_dis_sub[:, ch], y_dis_sub)
        dis_channel_corr[ch] = abs(rho) if not np.isnan(rho) else 0.0
        if (ch + 1) % 128 == 0:
            print(f"    Disorder channel {ch+1}/{D} done")

    # Compute per-channel F-statistic for TM topology (multi-class)
    print("  Computing per-channel TM importance (F-statistic)...")
    TM_MAP = {"H": 0, "B": 1, "S": 2, "O": 3}
    tm_avail = sorted(set(emb_tm.keys()) & set(tm_labels.keys()))
    print(f"  TM proteins with embeddings: {len(tm_avail)}")
    all_tm_emb = []
    all_tm_lbl = []
    for pid in tm_avail:
        emb = emb_tm[pid].astype(np.float32)
        labels_str = tm_labels[pid]
        L = min(len(labels_str), emb.shape[0])
        mapped = [TM_MAP.get(c, 3) for c in labels_str[:L]]
        all_tm_emb.append(emb[:L])
        all_tm_lbl.append(np.array(mapped, dtype=np.int32))

    if all_tm_emb:
        X_tm = np.vstack(all_tm_emb)
        y_tm = np.concatenate(all_tm_lbl)
        print(f"  Total residues for TM: {X_tm.shape[0]}")

        # Subsample if very large
        if X_tm.shape[0] > max_rows:
            rng2 = np.random.RandomState(42)
            idx2 = rng2.choice(X_tm.shape[0], size=max_rows, replace=False)
            X_tm_sub = X_tm[idx2]
            y_tm_sub = y_tm[idx2]
            print(f"  Subsampled to {max_rows} residues for F-stat computation")
        else:
            X_tm_sub = X_tm
            y_tm_sub = y_tm

        from sklearn.feature_selection import f_classif
        f_scores, _ = f_classif(X_tm_sub, y_tm_sub)
        tm_channel_f = np.nan_to_num(f_scores, nan=0.0).astype(np.float32)
        n_residues_tm = int(X_tm.shape[0])
    else:
        tm_channel_f = np.zeros(D, dtype=np.float32)
        n_residues_tm = 0
        print("  WARNING: No TM proteins found!")

    # Compute PCA stats to find where task-important channels concentrate
    print("  Computing PCA concentration of task-important channels...")
    emb_5k = load_residue_embeddings(EMB_5K)
    stats = compute_corpus_stats(emb_5k, n_sample=50_000, n_pcs=3, seed=42)
    rotation = stats["rotation_matrix"]  # (D, D) — rows are PC directions

    # For each PC direction, weighted importance = |PC_weights| dot channel_importance
    dis_pc_importance = np.abs(rotation) @ dis_channel_corr  # (D,)
    tm_pc_importance = np.abs(rotation) @ tm_channel_f       # (D,)

    # Cumulative importance captured by top-k PCs (sorted by PC index, already in var order)
    dis_cumulative = np.cumsum(dis_pc_importance) / dis_pc_importance.sum()
    tm_cumulative = np.cumsum(tm_pc_importance) / tm_pc_importance.sum()

    checkpoints = [64, 128, 256, 384, 512, 640, 768, 896, 1024]
    D_actual = len(dis_cumulative)

    step_results = {
        "disorder_channel_corr_stats": {
            "max": float(dis_channel_corr.max()),
            "mean": float(dis_channel_corr.mean()),
            "std": float(dis_channel_corr.std()),
            "top20_channels": np.argsort(dis_channel_corr)[-20:][::-1].tolist(),
            "top20_values": dis_channel_corr[np.argsort(dis_channel_corr)[-20:][::-1]].tolist(),
        },
        "tm_channel_f_stats": {
            "max": float(tm_channel_f.max()),
            "mean": float(tm_channel_f.mean()),
            "std": float(tm_channel_f.std()),
            "top20_channels": np.argsort(tm_channel_f)[-20:][::-1].tolist(),
            "top20_values": tm_channel_f[np.argsort(tm_channel_f)[-20:][::-1]].tolist(),
        },
        "pca_cumulative_disorder": {
            f"top_{k}": float(dis_cumulative[k - 1])
            for k in checkpoints if k <= D_actual
        },
        "pca_cumulative_tm": {
            f"top_{k}": float(tm_cumulative[k - 1])
            for k in checkpoints if k <= D_actual
        },
        "n_residues_disorder": int(X_dis.shape[0]),
        "n_residues_tm": n_residues_tm,
    }

    results["C"] = step_results
    results["steps_done"].append("C")
    save_results(results)
    print("\n  Step C complete!")
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print a comparison table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Projection Quality Diagnostic")
    print("=" * 80)

    if "A" in results:
        raw = results["A"]["raw_1024d"]
        print(f"\n{'Method':<20} {'SS3 Q3':>8} {'SS8 Q8':>8} {'Dis rho':>8} {'TM F1':>8} {'Ret@1':>8}")
        print("-" * 72)
        print(f"{'Raw 1024d':<20} {raw['ss3_q3']:>8.3f} {raw['ss8_q8']:>8.3f} "
              f"{raw['disorder_rho']:>8.3f} {raw['tm_f1']:>8.3f} {'—':>8}")

        for d in DIMS:
            ds = str(d)
            if ds in results["A"]["rp"]:
                r = results["A"]["rp"][ds]
                print(f"{'RP ' + ds:<20} {r['ss3_q3']:>8.3f} {r['ss8_q8']:>8.3f} "
                      f"{r['disorder_rho']:>8.3f} {r['tm_f1']:>8.3f} {r['family_ret1']:>8.3f}")

    if "B" in results:
        for d in DIMS:
            ds = str(d)
            if ds in results["B"]["pca"]:
                r = results["B"]["pca"][ds]
                print(f"{'PCA ' + ds:<20} {r['ss3_q3']:>8.3f} {r['ss8_q8']:>8.3f} "
                      f"{r['disorder_rho']:>8.3f} {r['tm_f1']:>8.3f} {r['family_ret1']:>8.3f}")

    if "C" in results:
        print(f"\n  PCA cumulative disorder importance:")
        for k, v in results["C"]["pca_cumulative_disorder"].items():
            print(f"    {k}: {v:.3f}")
        print(f"\n  PCA cumulative TM importance:")
        for k, v in results["C"]["pca_cumulative_tm"].items():
            print(f"    {k}: {v:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Experiment 38: Projection Quality Diagnostic")
    print("=" * 60)

    results = load_results()
    print(f"Steps already done: {results.get('steps_done', [])}")

    results = step_A(results)
    results = step_B(results)
    results = step_C(results)

    print_summary(results)
    save_results(results)
    print(f"\nResults saved to {RESULTS_PATH}")
