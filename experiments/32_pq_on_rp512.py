#!/usr/bin/env python3
"""Phase B — Product Quantization on ABTT3+RP512 preprocessed embeddings.

Previous PQ benchmarks (Exp 28) applied PQ to raw 1024d and got Ret@1=0.701
(M=64). This experiment applies PQ to the ABTT3+RP512 space, which is
decorrelated and more isotropic — ideal for sub-vector quantization.

Steps:
  B1: PQ sweep — M ∈ {8, 16, 32, 64}, K=256 on ABTT3+RP512
  B2: OPQ — learn per-subspace PCA rotation before PQ
  B3: PQ + scalar residual refinement (PQ M=16 + int4/int2 residuals)
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
    pq_fit, pq_encode, pq_decode,
    rvq_fit, rvq_encode, rvq_decode,
    quantize_int4, dequantize_int4,
    quantize_int2, dequantize_int2,
)
from src.one_embedding.transforms import dct_summary
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, load_cb513_csv,
)
from src.utils.h5_store import load_residue_embeddings
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "pq_rp512_results.json"
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
    meta = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    meta, _ = filter_by_family_size(meta, min_members=3)
    return meta


def apply_abtt3_rp512(embeddings, stats, seed=42):
    """Apply ABTT3 + RP512 pipeline."""
    top3 = stats["top_pcs"][:3]
    coded = {}
    for pid, m in embeddings.items():
        ma = all_but_the_top(m.astype(np.float32), top3)
        coded[pid] = random_orthogonal_project(ma, d_out=512, seed=seed)
    return coded


def get_corpus_stats(embeddings, train_ids):
    """Compute ABTT3 corpus stats from training set."""
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    print(f"  Computing ABTT3 stats from {len(train_embs)} train proteins...")
    return compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)


def eval_retrieval(vectors, metadata, test_ids):
    """Evaluate retrieval from protein vectors."""
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )


def eval_ss3(per_residue_embs, stats):
    """Evaluate SS3 on CB513 with given per-residue embeddings."""
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_raw, stats)

    # Apply the same transform that was applied to per_residue_embs
    # (caller passes a transform function via cb513_coded)
    sequences, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    avail = sorted(set(cb513_coded.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(avail)
    n_tr = int(len(avail) * 0.8)
    return cb513_coded, ss3_labels, avail[:n_tr], avail[n_tr:]


def storage_pq(mean_L, M, n_centroids=256):
    """Per-protein bytes for PQ + protein_vec fp16."""
    bits_per_code = int(np.ceil(np.log2(n_centroids)))
    bytes_per_code = 1 if bits_per_code <= 8 else 2
    return mean_L * M * bytes_per_code + 2048 * 2


# ── Step B1: PQ sweep ─────────────────────────────────────────────────────

def step_B1(results):
    print("\n" + "=" * 60)
    print("STEP B1: PQ sweep on ABTT3+RP512 (M ∈ {8, 16, 32, 64, 128})")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    stats = get_corpus_stats(embeddings, train_ids)

    # Apply ABTT3+RP512
    print("  Applying ABTT3+RP512...")
    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    # Prepare CB513 for SS3
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_raw, stats)
    del cb513_raw

    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    sequences, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    cb513_avail = sorted(set(cb513_coded.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    # Fit PQ on training set only
    train_coded = {k: v for k, v in coded.items() if k in set(train_ids)}

    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
    b1_results = {}

    for M in [8, 16, 32, 64, 128]:
        if 512 % M != 0:
            print(f"\n  Skipping M={M}: 512 not divisible by {M}")
            continue

        print(f"\n  --- PQ M={M}, K=256 ---")
        t0 = time.time()

        # Fit
        print(f"    Fitting PQ on {len(train_coded)} train proteins...")
        pq_model = pq_fit(train_coded, M=M, n_centroids=256,
                          max_residues=500_000, seed=42)
        sub_dim = pq_model["sub_dim"]
        print(f"    sub_dim={sub_dim}, codebook shape: ({M}, 256, {sub_dim})")

        # Encode + decode all proteins
        deq = {}
        for pid, m in coded.items():
            codes = pq_encode(m, pq_model)
            deq[pid] = pq_decode(codes, pq_model)

        # Retrieval
        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        # SS3 on CB513
        cb513_deq = {}
        for pid, m in cb513_coded.items():
            codes = pq_encode(m, pq_model)
            cb513_deq[pid] = pq_decode(codes, pq_model)

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        # Reconstruction quality
        cos_sims = []
        for pid in list(coded.keys())[:200]:
            orig = coded[pid]
            rec = deq[pid]
            cos = np.sum(orig * rec, axis=1) / (
                np.linalg.norm(orig, axis=1) * np.linalg.norm(rec, axis=1) + 1e-8
            )
            cos_sims.append(cos.mean())
        mean_cos = np.mean(cos_sims)

        size = storage_pq(mean_L, M)
        codebook_kb = pq_model["codebook"].nbytes / 1024
        elapsed = time.time() - t0

        key = f"pq_M{M}"
        b1_results[key] = {
            "method": key,
            "M": M,
            "sub_dim": sub_dim,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "cos_sim": round(float(mean_cos), 4),
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
            "codebook_kb": round(codebook_kb, 1),
            "mean_L": mean_L,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  MRR={ret['mrr']:.3f}  "
              f"SS3={ss3['q3']:.3f}  CosSim={mean_cos:.3f}  "
              f"size={size/1024:.1f}KB ({size/2048:.1f}x mean_pool)  "
              f"codebook={codebook_kb:.0f}KB")

        del deq, vectors, cb513_deq

    results["B1"] = b1_results
    results["steps_done"].append("B1")
    save_results(results)
    print("\n  B1 complete!")
    return results


# ── Step B2: OPQ (rotation before PQ) ────────────────────────────────────

def step_B2(results):
    print("\n" + "=" * 60)
    print("STEP B2: OPQ — per-subspace PCA rotation before PQ")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    stats = get_corpus_stats(embeddings, train_ids)

    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    train_coded = {k: v for k, v in coded.items() if k in set(train_ids)}

    # Learn global PCA rotation on training residues
    print("  Learning PCA rotation on training residues...")
    all_train_residues = np.concatenate(list(train_coded.values()), axis=0)
    if len(all_train_residues) > 500_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_train_residues), 500_000, replace=False)
        all_train_residues = all_train_residues[idx]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=512, random_state=42)
    pca.fit(all_train_residues.astype(np.float32))
    rotation = pca.components_.astype(np.float32)  # (512, 512)
    del all_train_residues

    # Apply rotation to all proteins
    print("  Applying PCA rotation...")
    coded_rot = {pid: m @ rotation.T for pid, m in coded.items()}
    train_coded_rot = {k: v for k, v in coded_rot.items() if k in set(train_ids)}

    # CB513
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_raw, stats)
    cb513_rot = {pid: m @ rotation.T for pid, m in cb513_coded.items()}
    del cb513_raw, cb513_coded

    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    _, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    cb513_avail = sorted(set(cb513_rot.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
    b2_results = {}

    for M in [16, 32]:
        print(f"\n  --- OPQ M={M}, K=256 ---")
        t0 = time.time()

        pq_model = pq_fit(train_coded_rot, M=M, n_centroids=256,
                          max_residues=500_000, seed=42)

        # Encode/decode in rotated space, then rotate back for per-residue
        deq = {}
        for pid, m in coded_rot.items():
            codes = pq_encode(m, pq_model)
            rec_rot = pq_decode(codes, pq_model)
            deq[pid] = rec_rot @ rotation  # rotate back

        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        # SS3 (in original space)
        cb513_deq = {}
        for pid, m in cb513_rot.items():
            codes = pq_encode(m, pq_model)
            rec_rot = pq_decode(codes, pq_model)
            cb513_deq[pid] = rec_rot @ rotation

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        size = storage_pq(mean_L, M)
        elapsed = time.time() - t0

        key = f"opq_M{M}"
        b2_results[key] = {
            "method": key,
            "M": M,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
              f"size={size/1024:.1f}KB ({size/2048:.1f}x)")

        del deq, cb513_deq

    results["B2"] = b2_results
    results["steps_done"].append("B2")
    save_results(results)
    print("\n  B2 complete!")
    return results


# ── Step B3: PQ + scalar residual refinement ──────────────────────────────

def step_B3(results):
    print("\n" + "=" * 60)
    print("STEP B3: PQ M=16 + scalar residual (int4, int2)")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    stats = get_corpus_stats(embeddings, train_ids)

    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    train_coded = {k: v for k, v in coded.items() if k in set(train_ids)}

    # Fit PQ M=16
    print("  Fitting PQ M=16...")
    pq_model = pq_fit(train_coded, M=16, n_centroids=256,
                      max_residues=500_000, seed=42)

    # CB513
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_raw, stats)
    del cb513_raw

    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    _, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    cb513_avail = sorted(set(cb513_coded.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))
    b3_results = {}

    # PQ base (no refinement — same as B1 M=16 but recomputed for consistency)
    for refine_name, refine_fn in [
        ("none", lambda res: np.zeros_like(res)),
        ("int4", lambda res: dequantize_int4(quantize_int4(res))),
        ("int2", lambda res: dequantize_int2(quantize_int2(res))),
    ]:
        print(f"\n  --- PQ M=16 + residual {refine_name} ---")

        deq = {}
        for pid, m in coded.items():
            codes = pq_encode(m, pq_model)
            pq_rec = pq_decode(codes, pq_model)
            residual = m - pq_rec
            refined_residual = refine_fn(residual)
            deq[pid] = pq_rec + refined_residual

        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        # SS3
        cb513_deq = {}
        for pid, m in cb513_coded.items():
            codes = pq_encode(m, pq_model)
            pq_rec = pq_decode(codes, pq_model)
            residual = m - pq_rec
            refined_residual = refine_fn(residual)
            cb513_deq[pid] = pq_rec + refined_residual

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        # Storage
        pq_bytes = mean_L * 16  # M=16, 1 byte per sub-vec
        pv_bytes = 2048 * 2
        if refine_name == "int4":
            res_bytes = mean_L * 512 // 2
        elif refine_name == "int2":
            res_bytes = mean_L * 512 // 4
        else:
            res_bytes = 0
        total = pq_bytes + res_bytes + pv_bytes

        key = f"pq16_res_{refine_name}"
        b3_results[key] = {
            "method": key,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "per_protein_bytes": total,
            "per_protein_kb": round(total / 1024, 1),
            "vs_mean_pool": round(total / 2048, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
              f"size={total/1024:.1f}KB ({total/2048:.1f}x)")

        del deq, cb513_deq

    # Also test RVQ on ABTT3+RP512
    print(f"\n  --- RVQ (2-4 levels, K=256) on ABTT3+RP512 ---")
    for n_levels in [2, 3, 4]:
        print(f"\n    RVQ {n_levels}-level K=256:")
        rvq_model = rvq_fit(train_coded, n_levels=n_levels, n_centroids=256,
                            max_residues=500_000, seed=42)

        deq = {}
        for pid, m in coded.items():
            codes = rvq_encode(m, rvq_model)
            deq[pid] = rvq_decode(codes, rvq_model)

        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        cb513_deq = {}
        for pid, m in cb513_coded.items():
            codes = rvq_encode(m, rvq_model)
            cb513_deq[pid] = rvq_decode(codes, rvq_model)

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        size = mean_L * n_levels + 2048 * 2  # n_levels bytes per residue + pv
        key = f"rvq_{n_levels}level"
        b3_results[key] = {
            "method": key,
            "n_levels": n_levels,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
        }
        print(f"      Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
              f"size={size/1024:.1f}KB ({size/2048:.1f}x)")

        del deq, cb513_deq

    results["B3"] = b3_results
    results["steps_done"].append("B3")
    save_results(results)
    print("\n  B3 complete!")
    return results


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 60)
    print("PHASE B SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30s} {'Ret@1':>7s} {'SS3 Q3':>7s} {'KB':>7s} {'×mean':>6s}")
    print("-" * 60)

    all_rows = []
    for step_key in ["B1", "B2", "B3"]:
        if step_key in results:
            all_rows.extend(results[step_key].values())

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

    if step is None or step == "B1":
        if "B1" not in results.get("steps_done", []):
            results = step_B1(results)
        else:
            print("B1 already done, skipping")

    if step is None or step == "B2":
        if "B2" not in results.get("steps_done", []):
            results = step_B2(results)
        else:
            print("B2 already done, skipping")

    if step is None or step == "B3":
        if "B3" not in results.get("steps_done", []):
            results = step_B3(results)
        else:
            print("B3 already done, skipping")

    print_summary(results)
    print(f"\nResults saved to {RESULTS_PATH}")
