#!/usr/bin/env python3
"""Phase A — Bit-width sweep on ABTT3+RP512 preprocessed embeddings.

Tests: binary (1-bit), int2 (2-bit), int4 (4-bit), int8 (8-bit), fp16
on ABTT3-preprocessed + RP512-projected embeddings, with DCT K=4 retrieval.

Key hypothesis: all prior extreme quantization benchmarks (Exp 28) used raw
1024d embeddings. The ABTT3+RP512 space is decorrelated and more isotropic,
so lower bit-widths should be much better than on raw data.

Steps:
  A1: Full pipeline benchmark at each bit width
  A2: Non-uniform bit allocation (variance-based triage)
  A3: RaBitQ-style double rotation before binarization
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
from src.one_embedding.quantization import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
    quantize_int2, dequantize_int2,
    quantize_binary, dequantize_binary,
)
from src.one_embedding.transforms import dct_summary
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, evaluate_ss8_probe, load_cb513_csv,
)
from src.utils.h5_store import load_residue_embeddings

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "bitwidth_sweep_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"

# ── Helpers ───────────────────────────────────────────────────────────────

def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": []}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_split():
    with open(SPLIT_PATH) as f:
        s = json.load(f)
    return s["train_ids"], s["test_ids"]


def load_metadata():
    from src.extraction.data_loader import load_metadata_csv, filter_by_family_size
    meta = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    meta, _ = filter_by_family_size(meta, min_members=3)
    return meta


def apply_abtt3_rp512(embeddings, stats, seed=42):
    """Apply ABTT3 + RP512 pipeline to a dict of embeddings."""
    top3 = stats["top_pcs"][:3]
    coded = {}
    for pid, m in embeddings.items():
        ma = all_but_the_top(m.astype(np.float32), top3)
        coded[pid] = random_orthogonal_project(ma, d_out=512, seed=seed)
    return coded


QUANT_METHODS = {
    "binary": (quantize_binary, dequantize_binary),
    "int2":   (quantize_int2,   dequantize_int2),
    "int4":   (quantize_int4,   dequantize_int4),
    "int8":   (quantize_int8,   dequantize_int8),
}


def quantize_dequantize(matrix, method):
    """Apply quantize then dequantize for a named method."""
    if method == "fp16":
        return matrix.astype(np.float16).astype(np.float32)
    if method == "fp32":
        return matrix.copy()
    enc_fn, dec_fn = QUANT_METHODS[method]
    return dec_fn(enc_fn(matrix))


def storage_bytes(L, D, method):
    """Compute per-protein storage for per-residue + protein_vec fp16."""
    protein_vec_bytes = 2048 * 2  # DCT K=4 * 512d * fp16
    if method == "fp32":
        return L * D * 4 + protein_vec_bytes
    elif method == "fp16":
        return L * D * 2 + protein_vec_bytes
    elif method == "int8":
        return L * D * 1 + protein_vec_bytes
    elif method == "int4":
        return L * D // 2 + protein_vec_bytes
    elif method == "int2":
        return L * D // 4 + protein_vec_bytes
    elif method == "binary":
        return L * D // 8 + protein_vec_bytes
    return 0


# ── Step A1: Full pipeline benchmark ─────────────────────────────────────

def step_A1(results):
    print("\n" + "=" * 60)
    print("STEP A1: ABTT3+RP512 bit-width sweep")
    print("=" * 60)

    # Load data
    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    print(f"Loaded {len(embeddings)} proteins")

    # Compute corpus stats from training set
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    print(f"Computing ABTT3 stats from {len(train_embs)} train proteins...")
    stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)
    del train_embs

    # Apply ABTT3+RP512 to all proteins
    print("Applying ABTT3+RP512 to all proteins...")
    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    # Methods to test
    methods = ["fp16", "int8", "int4", "int2", "binary"]

    a1_results = {}
    for method in methods:
        print(f"\n  --- {method} ---")
        t0 = time.time()

        # Quantize/dequantize per-residue
        deq = {}
        for pid, m in coded.items():
            deq[pid] = quantize_dequantize(m, method)

        # Retrieval: DCT K=4 protein vectors
        vectors = {}
        for pid, m in deq.items():
            vectors[pid] = dct_summary(m, K=4)

        ret = evaluate_retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids,
        )

        # Per-residue: SS3 on CB513
        cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
        cb513_embs = load_residue_embeddings(
            DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
        )
        cb513_coded = apply_abtt3_rp512(cb513_embs, stats)
        cb513_deq = {pid: quantize_dequantize(m, method)
                     for pid, m in cb513_coded.items()}

        sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
        avail = sorted(set(cb513_deq.keys()) & set(ss3_labels.keys()))
        rng = random.Random(42)
        rng.shuffle(avail)
        n_tr = int(len(avail) * 0.8)
        cb_train, cb_test = avail[:n_tr], avail[n_tr:]

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        # Storage estimate
        mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
        size = storage_bytes(mean_L, 512, method)

        elapsed = time.time() - t0

        row = {
            "method": method,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
            "mean_L": mean_L,
            "elapsed_s": round(elapsed, 1),
        }
        a1_results[method] = row
        print(f"    Ret@1={ret['precision@1']:.3f}  MRR={ret['mrr']:.3f}  "
              f"SS3={ss3['q3']:.3f}  size={size/1024:.1f}KB  "
              f"({size/2048:.1f}x mean_pool)")

        del deq, vectors, cb513_deq

    results["A1"] = a1_results
    results["steps_done"].append("A1")
    save_results(results)
    print("\n  A1 complete!")
    return results


# ── Step A2: Non-uniform bit allocation ───────────────────────────────────

def step_A2(results):
    print("\n" + "=" * 60)
    print("STEP A2: Non-uniform bit allocation (variance triage)")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)
    del train_embs

    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    # Compute per-dimension variance on training set
    train_coded = {k: v for k, v in coded.items() if k in set(train_ids)}
    all_residues = np.concatenate(list(train_coded.values()), axis=0)
    dim_var = np.var(all_residues, axis=0)  # (512,)
    del all_residues, train_coded

    # Sort dims by variance
    sorted_dims = np.argsort(dim_var)[::-1]  # descending variance
    n_dims = len(sorted_dims)
    tier1 = set(sorted_dims[:n_dims // 4])      # top 25% → 4 bits
    tier2 = set(sorted_dims[n_dims // 4: 3 * n_dims // 4])  # middle 50% → 2 bits
    tier3 = set(sorted_dims[3 * n_dims // 4:])   # bottom 25% → 1 bit

    print(f"  Variance tiers: {len(tier1)} dims @4bit, {len(tier2)} @2bit, {len(tier3)} @1bit")
    avg_bits = (len(tier1) * 4 + len(tier2) * 2 + len(tier3) * 1) / n_dims
    print(f"  Average bits/dim: {avg_bits:.2f}")

    def nonuniform_quantize(matrix):
        """Apply different quantization per dimension group."""
        L, D = matrix.shape
        result = np.zeros_like(matrix)
        for d in range(D):
            col = matrix[:, d:d+1]
            if d in tier1:
                c = quantize_int4(col)
                result[:, d:d+1] = dequantize_int4(c)
            elif d in tier2:
                c = quantize_int2(col)
                result[:, d:d+1] = dequantize_int2(c)
            else:
                c = quantize_binary(col)
                result[:, d:d+1] = dequantize_binary(c)
        return result

    # Apply and evaluate
    deq = {pid: nonuniform_quantize(m) for pid, m in coded.items()}
    vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
    ret = evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )

    # SS3
    cb513_embs = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_embs, stats)
    cb513_deq = {pid: nonuniform_quantize(m) for pid, m in cb513_coded.items()}

    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    sequences, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    avail = sorted(set(cb513_deq.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(avail)
    n_tr = int(len(avail) * 0.8)
    ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, avail[:n_tr], avail[n_tr:])

    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
    # Storage: 128 dims * 4bit + 256 dims * 2bit + 128 dims * 1bit per residue
    pr_bits = 128 * 4 + 256 * 2 + 128 * 1  # = 1152 bits = 144 bytes per residue
    size = mean_L * 144 + 2048 * 2

    row = {
        "method": f"nonuniform_avg{avg_bits:.1f}bit",
        "family_ret1": ret["precision@1"],
        "family_mrr": ret["mrr"],
        "ss3_q3": ss3["q3"],
        "per_protein_bytes": size,
        "per_protein_kb": round(size / 1024, 1),
        "vs_mean_pool": round(size / 2048, 1),
        "avg_bits_per_dim": avg_bits,
    }
    results["A2"] = {"nonuniform": row}
    results["steps_done"].append("A2")
    save_results(results)
    print(f"\n  Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
          f"size={size/1024:.1f}KB ({size/2048:.1f}x mean_pool)")
    print("  A2 complete!")
    return results


# ── Step A3: RaBitQ-style Hadamard rotation ───────────────────────────────

def step_A3(results):
    print("\n" + "=" * 60)
    print("STEP A3: RaBitQ double rotation (Hadamard + binary)")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)
    del train_embs

    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    # Second random orthogonal rotation for isotropy maximization
    rng = np.random.RandomState(123)  # different seed from RP
    R2 = rng.randn(512, 512).astype(np.float32)
    Q2, _ = np.linalg.qr(R2, mode="reduced")

    a3_results = {}
    for method in ["binary", "int2"]:
        print(f"\n  --- double_rotation + {method} ---")

        deq = {}
        for pid, m in coded.items():
            rotated = m @ Q2  # second rotation
            deq_m = quantize_dequantize(rotated, method)
            deq[pid] = deq_m @ Q2.T  # rotate back for per-residue

        # Retrieval from double-rotated (quantized) space
        vectors = {pid: dct_summary(deq[pid], K=4) for pid, m in coded.items()}
        ret = evaluate_retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids,
        )

        # SS3
        cb513_embs = load_residue_embeddings(
            DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
        )
        cb513_coded = apply_abtt3_rp512(cb513_embs, stats)
        cb513_deq = {}
        for pid, m in cb513_coded.items():
            rotated = m @ Q2
            deq_m = quantize_dequantize(rotated, method)
            cb513_deq[pid] = deq_m @ Q2.T

        cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
        sequences, ss3_labels, _, _ = load_cb513_csv(cb513_path)
        avail = sorted(set(cb513_deq.keys()) & set(ss3_labels.keys()))
        rng_split = random.Random(42)
        rng_split.shuffle(avail)
        n_tr = int(len(avail) * 0.8)
        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, avail[:n_tr], avail[n_tr:])

        mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
        size = storage_bytes(mean_L, 512, method)

        key = f"rabitq_{method}"
        a3_results[key] = {
            "method": key,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
              f"size={size/1024:.1f}KB")

        del deq, cb513_deq

    results["A3"] = a3_results
    results["steps_done"].append("A3")
    save_results(results)
    print("\n  A3 complete!")
    return results


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 60)
    print("PHASE A SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30s} {'Ret@1':>7s} {'SS3 Q3':>7s} {'KB':>7s} {'×mean':>6s}")
    print("-" * 60)

    all_rows = []
    if "A1" in results:
        all_rows.extend(results["A1"].values())
    if "A2" in results:
        all_rows.extend(results["A2"].values())
    if "A3" in results:
        all_rows.extend(results["A3"].values())

    for r in sorted(all_rows, key=lambda x: x.get("family_ret1", 0), reverse=True):
        name = r.get("method", "?")
        ret1 = r.get("family_ret1", 0)
        ss3 = r.get("ss3_q3", 0)
        kb = r.get("per_protein_kb", 0)
        vs = r.get("vs_mean_pool", 0)
        print(f"  {name:<28s} {ret1:>7.3f} {ss3:>7.3f} {kb:>7.1f} {vs:>5.1f}x")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()

    step = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--step"):
                step = sys.argv[sys.argv.index(arg) + 1]

    if step is None or step == "A1":
        if "A1" not in results.get("steps_done", []):
            results = step_A1(results)
        else:
            print("A1 already done, skipping")

    if step is None or step == "A2":
        if "A2" not in results.get("steps_done", []):
            results = step_A2(results)
        else:
            print("A2 already done, skipping")

    if step is None or step == "A3":
        if "A3" not in results.get("steps_done", []):
            results = step_A3(results)
        else:
            print("A3 already done, skipping")

    print_summary(results)
    print(f"\nResults saved to {RESULTS_PATH}")
