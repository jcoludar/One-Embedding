#!/usr/bin/env python3
"""Experiment 41: 768d Codec Retention Benchmarks.

Measures how much per-residue task quality the 768d One Embedding codec preserves
compared to raw ProtT5-XL 1024d embeddings.

Tasks:
    1. SS3  (Q3 accuracy)      -- CB513, 80/20 split
    2. SS8  (Q8 accuracy)      -- CB513, same split
    3. Disorder (Spearman rho) -- CheZOD SETH 1174/117 split
    4. TM topology (Macro F1)  -- TMbed cv_00
    5. Family retrieval (Ret@1) -- SCOPe 5K

Compares raw 1024d ProtT5 embeddings against 768d compressed .one.h5 files.
For retrieval, uses the pre-stored protein_vec from compressed files directly.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    evaluate_disorder_probe,
    evaluate_tm_probe,
    load_cb513_csv,
    load_chezod_seth,
    load_tmbed_annotated,
)
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size
from src.one_embedding.io import read_one_h5_batch
from src.one_embedding.transforms import dct_summary
from src.utils.h5_store import load_residue_embeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Raw ProtT5 1024d embeddings
RAW_CB513 = DATA / "residue_embeddings" / "prot_t5_xl_cb513.h5"
RAW_CHEZOD = DATA / "residue_embeddings" / "prot_t5_xl_chezod.h5"
RAW_SCOPE = DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
RAW_VALIDATION = DATA / "residue_embeddings" / "prot_t5_xl_validation.h5"

# Compressed 768d embeddings
COMP_CB513 = DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "cb513.one.h5"
COMP_CHEZOD = DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "chezod.one.h5"
COMP_SCOPE = DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "scope_5k.one.h5"
COMP_VALIDATION = DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "validation.one.h5"

# Splits
CB513_SPLIT = DATA / "benchmark_suite" / "splits" / "cb513_80_20.json"
CHEZOD_SPLIT = DATA / "benchmark_suite" / "splits" / "chezod_seth.json"
SCOPE_SPLIT = DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json"

# Labels
CB513_CSV = DATA / "per_residue_benchmarks" / "CB513.csv"
SETH_DIR = DATA / "per_residue_benchmarks"
TMBED_FASTA = DATA / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"

# Metadata
SCOPE_META = DATA / "proteins" / "metadata_5k.csv"

# Output
RESULTS_PATH = DATA / "benchmarks" / "retention_768d_results.json"


def load_compressed_per_residue(path: Path) -> dict[str, np.ndarray]:
    """Load compressed .one.h5 and return {pid: per_residue} dict."""
    batch = read_one_h5_batch(path)
    return {pid: data["per_residue"] for pid, data in batch.items()}


def load_compressed_protein_vecs(path: Path) -> dict[str, np.ndarray]:
    """Load compressed .one.h5 and return {pid: protein_vec} dict."""
    batch = read_one_h5_batch(path)
    return {pid: np.asarray(data["protein_vec"], dtype=np.float32) for pid, data in batch.items()}


def compute_raw_protein_vecs(
    embeddings: dict[str, np.ndarray], dct_k: int = 4,
) -> dict[str, np.ndarray]:
    """Compute protein-level vectors from raw per-residue embeddings using DCT K=4."""
    vecs = {}
    for pid, emb in embeddings.items():
        vecs[pid] = dct_summary(emb, K=dct_k)
    return vecs


# ======================================================================
# Benchmark runners
# ======================================================================


def benchmark_ss3(results: dict) -> None:
    """SS3 Q3 accuracy on CB513."""
    print("\n" + "=" * 60)
    print("TASK: SS3 (Q3 accuracy) -- CB513")
    print("=" * 60)

    # Load labels (dict-based lookup by protein ID)
    _, ss3_labels, _, _ = load_cb513_csv(CB513_CSV)
    print(f"  Labels: {len(ss3_labels)} proteins")

    # Load split
    split = json.loads(CB513_SPLIT.read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    print(f"  Split: {len(train_ids)} train / {len(test_ids)} test")

    # --- Raw 1024d ---
    print("\n  [Raw 1024d] Loading embeddings...")
    raw_emb = load_residue_embeddings(RAW_CB513)
    avail = len(set(train_ids + test_ids) & set(raw_emb.keys()) & set(ss3_labels.keys()))
    print(f"  [Raw 1024d] {avail} proteins with embeddings + labels")

    t0 = time.time()
    raw_result = evaluate_ss3_probe(raw_emb, ss3_labels, train_ids, test_ids)
    print(f"  [Raw 1024d] SS3 Q3 = {raw_result['q3']:.4f}  ({time.time() - t0:.1f}s)")
    del raw_emb

    # --- Compressed 768d ---
    print("\n  [768d] Loading compressed embeddings...")
    comp_emb = load_compressed_per_residue(COMP_CB513)
    avail = len(set(train_ids + test_ids) & set(comp_emb.keys()) & set(ss3_labels.keys()))
    print(f"  [768d] {avail} proteins with embeddings + labels")

    t0 = time.time()
    comp_result = evaluate_ss3_probe(comp_emb, ss3_labels, train_ids, test_ids)
    print(f"  [768d] SS3 Q3 = {comp_result['q3']:.4f}  ({time.time() - t0:.1f}s)")
    del comp_emb

    retention = comp_result["q3"] / raw_result["q3"] * 100 if raw_result["q3"] > 0 else 0
    print(f"\n  Retention: {retention:.1f}%")

    results["ss3"] = {
        "raw_1024d": raw_result,
        "compressed_768d": comp_result,
        "retention_pct": retention,
    }


def benchmark_ss8(results: dict) -> None:
    """SS8 Q8 accuracy on CB513."""
    print("\n" + "=" * 60)
    print("TASK: SS8 (Q8 accuracy) -- CB513")
    print("=" * 60)

    # Load labels
    _, _, ss8_labels, _ = load_cb513_csv(CB513_CSV)
    print(f"  Labels: {len(ss8_labels)} proteins")

    # Load split
    split = json.loads(CB513_SPLIT.read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    print(f"  Split: {len(train_ids)} train / {len(test_ids)} test")

    # --- Raw 1024d ---
    print("\n  [Raw 1024d] Loading embeddings...")
    raw_emb = load_residue_embeddings(RAW_CB513)

    t0 = time.time()
    raw_result = evaluate_ss8_probe(raw_emb, ss8_labels, train_ids, test_ids)
    print(f"  [Raw 1024d] SS8 Q8 = {raw_result['q8']:.4f}  ({time.time() - t0:.1f}s)")
    del raw_emb

    # --- Compressed 768d ---
    print("\n  [768d] Loading compressed embeddings...")
    comp_emb = load_compressed_per_residue(COMP_CB513)

    t0 = time.time()
    comp_result = evaluate_ss8_probe(comp_emb, ss8_labels, train_ids, test_ids)
    print(f"  [768d] SS8 Q8 = {comp_result['q8']:.4f}  ({time.time() - t0:.1f}s)")
    del comp_emb

    retention = comp_result["q8"] / raw_result["q8"] * 100 if raw_result["q8"] > 0 else 0
    print(f"\n  Retention: {retention:.1f}%")

    results["ss8"] = {
        "raw_1024d": raw_result,
        "compressed_768d": comp_result,
        "retention_pct": retention,
    }


def benchmark_disorder(results: dict) -> None:
    """Disorder Spearman rho on CheZOD SETH."""
    print("\n" + "=" * 60)
    print("TASK: Disorder (Spearman rho) -- CheZOD SETH")
    print("=" * 60)

    # Load labels from SETH directory
    _, disorder_scores, _, _ = load_chezod_seth(SETH_DIR)
    print(f"  Labels: {len(disorder_scores)} proteins")

    # Load split (matches SETH train/test exactly)
    split = json.loads(CHEZOD_SPLIT.read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    print(f"  Split: {len(train_ids)} train / {len(test_ids)} test")

    # --- Raw 1024d ---
    print("\n  [Raw 1024d] Loading embeddings...")
    raw_emb = load_residue_embeddings(RAW_CHEZOD)
    avail_train = len(set(train_ids) & set(raw_emb.keys()) & set(disorder_scores.keys()))
    avail_test = len(set(test_ids) & set(raw_emb.keys()) & set(disorder_scores.keys()))
    print(f"  [Raw 1024d] {avail_train} train / {avail_test} test with embeddings + labels")

    t0 = time.time()
    raw_result = evaluate_disorder_probe(raw_emb, disorder_scores, train_ids, test_ids)
    print(f"  [Raw 1024d] Disorder rho = {raw_result['spearman_rho']:.4f}  ({time.time() - t0:.1f}s)")
    del raw_emb

    # --- Compressed 768d ---
    print("\n  [768d] Loading compressed embeddings...")
    comp_emb = load_compressed_per_residue(COMP_CHEZOD)
    avail_train = len(set(train_ids) & set(comp_emb.keys()) & set(disorder_scores.keys()))
    avail_test = len(set(test_ids) & set(comp_emb.keys()) & set(disorder_scores.keys()))
    print(f"  [768d] {avail_train} train / {avail_test} test with embeddings + labels")

    t0 = time.time()
    comp_result = evaluate_disorder_probe(comp_emb, disorder_scores, train_ids, test_ids)
    print(f"  [768d] Disorder rho = {comp_result['spearman_rho']:.4f}  ({time.time() - t0:.1f}s)")
    del comp_emb

    raw_rho = raw_result["spearman_rho"]
    comp_rho = comp_result["spearman_rho"]
    retention = comp_rho / raw_rho * 100 if raw_rho > 0 else 0
    print(f"\n  Retention: {retention:.1f}%")

    results["disorder"] = {
        "raw_1024d": raw_result,
        "compressed_768d": comp_result,
        "retention_pct": retention,
    }


def benchmark_tm(results: dict) -> None:
    """TM topology Macro F1 on TMbed cv_00."""
    print("\n" + "=" * 60)
    print("TASK: TM topology (Macro F1) -- TMbed")
    print("=" * 60)

    # Load labels
    tm_seqs, tm_labels = load_tmbed_annotated(TMBED_FASTA)
    print(f"  Labels: {len(tm_labels)} proteins")

    # TMbed uses 80/20 random split (no predefined split file)
    import random
    rng = random.Random(42)
    all_label_ids = list(tm_labels.keys())

    # --- Raw 1024d ---
    # Validation H5 has tmbed_ prefix: "tmbed_XXX" in H5, "XXX" in labels
    print("\n  [Raw 1024d] Loading validation embeddings...")
    raw_emb_full = load_residue_embeddings(RAW_VALIDATION)

    # Build mapping: strip tmbed_ prefix to match label keys
    raw_emb_tmbed = {}
    for h5_key, emb in raw_emb_full.items():
        if h5_key.startswith("tmbed_"):
            label_key = h5_key[6:]  # strip "tmbed_" prefix
            raw_emb_tmbed[label_key] = emb

    avail_ids = [pid for pid in all_label_ids if pid in raw_emb_tmbed]
    rng_copy = random.Random(42)
    rng_copy.shuffle(avail_ids)
    n_train = int(len(avail_ids) * 0.8)
    train_ids = avail_ids[:n_train]
    test_ids = avail_ids[n_train:]
    print(f"  [Raw 1024d] {len(avail_ids)} proteins with embeddings + labels")
    print(f"  Split: {len(train_ids)} train / {len(test_ids)} test")

    t0 = time.time()
    raw_result = evaluate_tm_probe(raw_emb_tmbed, tm_labels, train_ids, test_ids)
    print(f"  [Raw 1024d] TM Macro F1 = {raw_result['macro_f1']:.4f}  ({time.time() - t0:.1f}s)")
    del raw_emb_full, raw_emb_tmbed

    # --- Compressed 768d ---
    print("\n  [768d] Loading compressed embeddings...")
    comp_batch = read_one_h5_batch(COMP_VALIDATION)

    # Same prefix stripping
    comp_emb_tmbed = {}
    for h5_key, data in comp_batch.items():
        if h5_key.startswith("tmbed_"):
            label_key = h5_key[6:]
            comp_emb_tmbed[label_key] = data["per_residue"]

    avail_comp = len(set(train_ids + test_ids) & set(comp_emb_tmbed.keys()))
    print(f"  [768d] {avail_comp} proteins with embeddings + labels")

    t0 = time.time()
    comp_result = evaluate_tm_probe(comp_emb_tmbed, tm_labels, train_ids, test_ids)
    print(f"  [768d] TM Macro F1 = {comp_result['macro_f1']:.4f}  ({time.time() - t0:.1f}s)")
    del comp_batch, comp_emb_tmbed

    raw_f1 = raw_result["macro_f1"]
    comp_f1 = comp_result["macro_f1"]
    retention = comp_f1 / raw_f1 * 100 if raw_f1 > 0 else 0
    print(f"\n  Retention: {retention:.1f}%")

    results["tm_topology"] = {
        "raw_1024d": raw_result,
        "compressed_768d": comp_result,
        "retention_pct": retention,
    }


def benchmark_retrieval(results: dict) -> None:
    """Family retrieval Ret@1 on SCOPe 5K."""
    print("\n" + "=" * 60)
    print("TASK: Family Retrieval (Ret@1) -- SCOPe 5K")
    print("=" * 60)

    # Load metadata for family labels
    metadata = load_metadata_csv(SCOPE_META)
    print(f"  Metadata: {len(metadata)} entries")

    # Filter to families with >= 3 members for meaningful retrieval
    metadata_filtered, kept_ids = filter_by_family_size(metadata, min_members=3)
    print(f"  After filtering (>=3 members): {len(metadata_filtered)} proteins")

    # Load split
    split = json.loads(SCOPE_SPLIT.read_text())
    eval_ids = split["eval_ids"]
    print(f"  Eval IDs: {len(eval_ids)}")

    # --- Raw 1024d ---
    # For raw embeddings, compute protein_vec using mean pool (standard for retrieval)
    print("\n  [Raw 1024d] Loading embeddings...")
    raw_emb = load_residue_embeddings(RAW_SCOPE)

    # Mean pool for retrieval (raw baseline uses mean-pooled vectors)
    print("  [Raw 1024d] Computing mean-pooled vectors...")
    raw_vecs = {pid: emb.mean(axis=0) for pid, emb in raw_emb.items()}
    del raw_emb

    t0 = time.time()
    raw_result = evaluate_retrieval_from_vectors(
        raw_vecs, metadata_filtered, label_key="family",
        k_values=[1, 3, 5], metric="cosine",
    )
    print(f"  [Raw 1024d] Ret@1 = {raw_result['precision@1']:.4f}  ({time.time() - t0:.1f}s)")
    del raw_vecs

    # --- Compressed 768d ---
    # Use pre-stored protein_vec from compressed file (DCT K=4 of 768d projected)
    print("\n  [768d] Loading compressed protein vectors...")
    comp_vecs = load_compressed_protein_vecs(COMP_SCOPE)
    avail = len(set(comp_vecs.keys()) & kept_ids)
    print(f"  [768d] {avail} proteins with vectors + family labels")

    t0 = time.time()
    comp_result = evaluate_retrieval_from_vectors(
        comp_vecs, metadata_filtered, label_key="family",
        k_values=[1, 3, 5], metric="cosine",
    )
    print(f"  [768d] Ret@1 = {comp_result['precision@1']:.4f}  ({time.time() - t0:.1f}s)")
    del comp_vecs

    raw_ret1 = raw_result["precision@1"]
    comp_ret1 = comp_result["precision@1"]
    retention = comp_ret1 / raw_ret1 * 100 if raw_ret1 > 0 else 0
    print(f"\n  Retention: {retention:.1f}%")

    results["retrieval"] = {
        "raw_1024d": raw_result,
        "compressed_768d": comp_result,
        "retention_pct": retention,
    }


# ======================================================================
# Main
# ======================================================================


def print_summary(results: dict) -> None:
    """Print a nicely formatted summary table."""
    print("\n")
    print("=" * 64)
    print("768d Codec Retention Benchmarks")
    print("=" * 64)
    print(f"{'Task':<18} {'Raw 1024d':>12} {'Compressed 768d':>16} {'Retention':>10}")
    print("-" * 64)

    rows = []

    if "ss3" in results:
        raw = results["ss3"]["raw_1024d"]["q3"]
        comp = results["ss3"]["compressed_768d"]["q3"]
        ret = results["ss3"]["retention_pct"]
        print(f"{'SS3 Q3':<18} {raw:>12.3f} {comp:>16.3f} {ret:>9.1f}%")
        rows.append(("SS3 Q3", raw, comp, ret))

    if "ss8" in results:
        raw = results["ss8"]["raw_1024d"]["q8"]
        comp = results["ss8"]["compressed_768d"]["q8"]
        ret = results["ss8"]["retention_pct"]
        print(f"{'SS8 Q8':<18} {raw:>12.3f} {comp:>16.3f} {ret:>9.1f}%")
        rows.append(("SS8 Q8", raw, comp, ret))

    if "disorder" in results:
        raw = results["disorder"]["raw_1024d"]["spearman_rho"]
        comp = results["disorder"]["compressed_768d"]["spearman_rho"]
        ret = results["disorder"]["retention_pct"]
        print(f"{'Disorder rho':<18} {raw:>12.3f} {comp:>16.3f} {ret:>9.1f}%")
        rows.append(("Disorder rho", raw, comp, ret))

    if "tm_topology" in results:
        raw = results["tm_topology"]["raw_1024d"]["macro_f1"]
        comp = results["tm_topology"]["compressed_768d"]["macro_f1"]
        ret = results["tm_topology"]["retention_pct"]
        print(f"{'TM F1':<18} {raw:>12.3f} {comp:>16.3f} {ret:>9.1f}%")
        rows.append(("TM F1", raw, comp, ret))

    if "retrieval" in results:
        raw = results["retrieval"]["raw_1024d"]["precision@1"]
        comp = results["retrieval"]["compressed_768d"]["precision@1"]
        ret = results["retrieval"]["retention_pct"]
        print(f"{'Family Ret@1':<18} {raw:>12.3f} {comp:>16.3f} {ret:>9.1f}%")
        rows.append(("Family Ret@1", raw, comp, ret))

    print("-" * 64)

    if rows:
        avg_retention = np.mean([r[3] for r in rows])
        print(f"{'Mean retention':<18} {'':>12} {'':>16} {avg_retention:>9.1f}%")

    print("=" * 64)


def main():
    print("Experiment 41: 768d Codec Retention Benchmarks")
    print(f"  Raw embeddings: ProtT5-XL, 1024d")
    print(f"  Compressed: One Embedding 1.0 codec, 768d")
    print()

    # Verify all input files exist
    required_files = [
        RAW_CB513, RAW_CHEZOD, RAW_SCOPE, RAW_VALIDATION,
        COMP_CB513, COMP_CHEZOD, COMP_SCOPE, COMP_VALIDATION,
        CB513_SPLIT, CHEZOD_SPLIT, SCOPE_SPLIT,
        CB513_CSV, TMBED_FASTA, SCOPE_META,
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    print("  All input files verified.")

    results = {
        "experiment": "41_768d_retention_benchmarks",
        "raw_model": "prot_t5_xl",
        "raw_dim": 1024,
        "compressed_dim": 768,
        "codec": "One Embedding 1.0 (ABTT3 + RP768 + DCT K=4)",
    }

    t_start = time.time()

    # Run all benchmarks
    benchmark_ss3(results)
    benchmark_ss8(results)
    benchmark_disorder(results)
    benchmark_tm(results)
    benchmark_retrieval(results)

    total_time = time.time() - t_start
    results["total_time_seconds"] = total_time

    # Print summary table
    print_summary(results)
    print(f"\nTotal time: {total_time:.1f}s")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
