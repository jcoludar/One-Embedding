#!/usr/bin/env python3
"""Phase C — Residue-Level Vector Quantization on ABTT3+RP512.

Most aggressive compression: one codebook index per residue position.
RVQ failed in Phase B (0.595 Ret@1 on full 512d), so this phase tests
whether smarter codebook strategies can make VQ work at sub-10 KB.

Steps:
  C1: K-means VQ sweep — K ∈ {256, 1024, 4096, 16384} on ABTT3+RP512
  C2: RVQ with larger codebooks — K=1024 per level (uint16)
  C3: Codebook utilization audit + dead-code reset
  C4: Hybrid VQ+PQ — VQ for coarse, PQ on residuals (Phase B+C fusion)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.quantization import (
    pq_fit, pq_encode, pq_decode,
)
from src.one_embedding.transforms import dct_summary
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import evaluate_ss3_probe, load_cb513_csv
from src.utils.h5_store import load_residue_embeddings
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "vq_codec_results.json"
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
    top3 = stats["top_pcs"][:3]
    coded = {}
    for pid, m in embeddings.items():
        ma = all_but_the_top(m.astype(np.float32), top3)
        coded[pid] = random_orthogonal_project(ma, d_out=512, seed=seed)
    return coded


def get_corpus_stats(embeddings, train_ids):
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    print(f"  Computing ABTT3 stats from {len(train_embs)} train proteins...")
    return compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)


def eval_retrieval(vectors, metadata, test_ids):
    return evaluate_retrieval_from_vectors(
        vectors, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )


def prepare_cb513(stats):
    """Load and preprocess CB513 for SS3 evaluation."""
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_coded = apply_abtt3_rp512(cb513_raw, stats)
    del cb513_raw

    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    _, ss3_labels, _, _ = load_cb513_csv(cb513_path)
    avail = sorted(set(cb513_coded.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(avail)
    n_tr = int(len(avail) * 0.8)
    return cb513_coded, ss3_labels, avail[:n_tr], avail[n_tr:]


def collect_train_residues(train_coded, max_residues=500_000):
    """Gather all residues from training proteins, subsample if needed."""
    all_res = np.concatenate(list(train_coded.values()), axis=0)
    if len(all_res) > max_residues:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_res), max_residues, replace=False)
        all_res = all_res[idx]
    return all_res.astype(np.float32)


def vq_encode(matrix, centroids):
    """Assign each residue to nearest centroid. Returns (L,) indices."""
    # Efficient distance: ||x - c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
    x_sq = np.sum(matrix ** 2, axis=1, keepdims=True)  # (L, 1)
    c_sq = np.sum(centroids ** 2, axis=1)  # (K,)
    dists = x_sq - 2.0 * (matrix @ centroids.T) + c_sq  # (L, K)
    return np.argmin(dists, axis=1)


def vq_decode(indices, centroids):
    """Look up centroid vectors for each index."""
    return centroids[indices]


def storage_vq(mean_L, K, extra_bytes=0):
    """Per-protein bytes: indices + protein_vec fp16."""
    bits = int(np.ceil(np.log2(K)))
    if bits <= 8:
        idx_bytes = mean_L * 1  # uint8
    else:
        idx_bytes = mean_L * 2  # uint16
    return idx_bytes + 2048 * 2 + extra_bytes


# ── Step C1: K-means VQ sweep ────────────────────────────────────────────

def step_C1(results):
    print("\n" + "=" * 60)
    print("STEP C1: K-means VQ on ABTT3+RP512")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    stats = get_corpus_stats(embeddings, train_ids)

    print("  Applying ABTT3+RP512...")
    coded = apply_abtt3_rp512(embeddings, stats)
    del embeddings

    train_coded = {k: v for k, v in coded.items() if k in set(train_ids)}
    train_residues = collect_train_residues(train_coded)
    print(f"  Training residues: {len(train_residues)} × {train_residues.shape[1]}d")

    cb513_coded, ss3_labels, cb_train, cb_test = prepare_cb513(stats)
    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))

    c1_results = {}

    for K in [256, 1024, 4096, 16384]:
        print(f"\n  --- VQ K={K} ---")
        t0 = time.time()

        km = MiniBatchKMeans(
            n_clusters=K,
            random_state=42,
            n_init=3,
            batch_size=min(10_000, len(train_residues)),
            max_iter=300,
        )
        km.fit(train_residues)
        centroids = km.cluster_centers_.astype(np.float32)

        # Check codebook utilization
        train_labels = km.labels_
        used = len(set(train_labels))
        utilization = used / K * 100
        print(f"    Codebook utilization: {used}/{K} ({utilization:.1f}%)")

        # Encode/decode all proteins
        deq = {}
        for pid, m in coded.items():
            indices = vq_encode(m, centroids)
            deq[pid] = vq_decode(indices, centroids)

        # Retrieval: DCT K=4 on reconstructed per-residue
        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        # SS3 on CB513
        cb513_deq = {}
        for pid, m in cb513_coded.items():
            indices = vq_encode(m, centroids)
            cb513_deq[pid] = vq_decode(indices, centroids)

        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        # Reconstruction quality
        cos_sims = []
        for pid in list(coded.keys())[:200]:
            orig = coded[pid]
            rec = deq[pid]
            norms = np.linalg.norm(orig, axis=1) * np.linalg.norm(rec, axis=1)
            norms = np.clip(norms, 1e-8, None)
            cos = np.sum(orig * rec, axis=1) / norms
            cos_sims.append(cos.mean())
        mean_cos = np.mean(cos_sims)

        size = storage_vq(mean_L, K)
        codebook_kb = centroids.nbytes / 1024
        elapsed = time.time() - t0

        key = f"vq_K{K}"
        c1_results[key] = {
            "method": key,
            "K": K,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "cos_sim": round(float(mean_cos), 4),
            "utilization_pct": round(utilization, 1),
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
            "codebook_kb": round(codebook_kb, 1),
            "mean_L": mean_L,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  MRR={ret['mrr']:.3f}  "
              f"SS3={ss3['q3']:.3f}  CosSim={mean_cos:.3f}  "
              f"size={size/1024:.1f}KB ({size/2048:.1f}x)  "
              f"codebook={codebook_kb:.0f}KB  [{elapsed:.0f}s]")

        del deq, vectors, cb513_deq

    results["C1"] = c1_results
    results["steps_done"].append("C1")
    save_results(results)
    print("\n  C1 complete!")
    return results


# ── Step C2: RVQ with larger codebooks ────────────────────────────────────

def step_C2(results):
    print("\n" + "=" * 60)
    print("STEP C2: RVQ with K=1024 per level on ABTT3+RP512")
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
    train_residues = collect_train_residues(train_coded)

    cb513_coded, ss3_labels, cb_train, cb_test = prepare_cb513(stats)
    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))

    c2_results = {}

    for n_levels, K in [(2, 1024), (3, 1024), (4, 1024), (2, 4096), (3, 4096)]:
        print(f"\n  --- RVQ {n_levels}-level K={K} ---")
        t0 = time.time()

        # Fit RVQ manually (existing rvq_fit caps at K=256 uint8)
        codebooks = []
        residuals = train_residues.copy()
        for level in range(n_levels):
            km = MiniBatchKMeans(
                n_clusters=K, random_state=42 + level,
                n_init=3, batch_size=min(10_000, len(residuals)),
                max_iter=200,
            )
            km.fit(residuals)
            centers = km.cluster_centers_.astype(np.float32)
            codebooks.append(centers)
            assignments = km.predict(residuals)
            residuals = residuals - centers[assignments]
            print(f"    Level {level}: residual norm = {np.linalg.norm(residuals, axis=1).mean():.3f}")

        # Encode/decode
        def rvq_enc_dec(matrix):
            rec = np.zeros_like(matrix)
            res = matrix.copy()
            for centers in codebooks:
                idx = vq_encode(res, centers)
                rec += centers[idx]
                res = res - centers[idx]
            return rec

        deq = {pid: rvq_enc_dec(m) for pid, m in coded.items()}
        vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
        ret = eval_retrieval(vectors, metadata, test_ids)

        cb513_deq = {pid: rvq_enc_dec(m) for pid, m in cb513_coded.items()}
        ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

        # Storage: uint16 indices per level
        bits_per_code = int(np.ceil(np.log2(K)))
        bytes_per_code = 2 if bits_per_code > 8 else 1
        size = mean_L * n_levels * bytes_per_code + 2048 * 2
        codebook_kb = sum(c.nbytes for c in codebooks) / 1024

        cos_sims = []
        for pid in list(coded.keys())[:200]:
            orig = coded[pid]
            rec = deq[pid]
            norms = np.linalg.norm(orig, axis=1) * np.linalg.norm(rec, axis=1)
            norms = np.clip(norms, 1e-8, None)
            cos = np.sum(orig * rec, axis=1) / norms
            cos_sims.append(cos.mean())

        elapsed = time.time() - t0
        key = f"rvq_{n_levels}L_K{K}"
        c2_results[key] = {
            "method": key,
            "n_levels": n_levels,
            "K": K,
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "cos_sim": round(float(np.mean(cos_sims)), 4),
            "per_protein_bytes": size,
            "per_protein_kb": round(size / 1024, 1),
            "vs_mean_pool": round(size / 2048, 1),
            "codebook_kb": round(codebook_kb, 1),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
              f"CosSim={np.mean(cos_sims):.3f}  "
              f"size={size/1024:.1f}KB ({size/2048:.1f}x)  [{elapsed:.0f}s]")

        del deq, cb513_deq

    results["C2"] = c2_results
    results["steps_done"].append("C2")
    save_results(results)
    print("\n  C2 complete!")
    return results


# ── Step C4: Hybrid VQ+PQ ─────────────────────────────────────────────────

def step_C4(results):
    print("\n" + "=" * 60)
    print("STEP C4: Hybrid VQ + PQ residual refinement")
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
    train_residues = collect_train_residues(train_coded)

    cb513_coded, ss3_labels, cb_train, cb_test = prepare_cb513(stats)
    mean_L = int(np.mean([m.shape[0] for m in coded.values()]))

    c4_results = {}

    # VQ base codebook sizes to try
    for vq_K in [256, 1024, 4096]:
        print(f"\n  Fitting VQ K={vq_K}...")
        km = MiniBatchKMeans(
            n_clusters=vq_K, random_state=42, n_init=3,
            batch_size=min(10_000, len(train_residues)), max_iter=300,
        )
        km.fit(train_residues)
        vq_centroids = km.cluster_centers_.astype(np.float32)

        # Compute VQ residuals on training set
        train_vq_residuals = {}
        for pid, m in train_coded.items():
            idx = vq_encode(m, vq_centroids)
            train_vq_residuals[pid] = m - vq_centroids[idx]

        # PQ on residuals with M ∈ {8, 16, 32}
        for pq_M in [8, 16, 32]:
            if 512 % pq_M != 0:
                continue
            print(f"\n    --- VQ K={vq_K} + PQ M={pq_M} residual ---")
            t0 = time.time()

            pq_model = pq_fit(train_vq_residuals, M=pq_M, n_centroids=256,
                              max_residues=500_000, seed=42)

            # Full encode/decode
            deq = {}
            for pid, m in coded.items():
                idx = vq_encode(m, vq_centroids)
                vq_rec = vq_centroids[idx]
                residual = m - vq_rec
                pq_codes = pq_encode(residual, pq_model)
                pq_rec = pq_decode(pq_codes, pq_model)
                deq[pid] = vq_rec + pq_rec

            vectors = {pid: dct_summary(m, K=4) for pid, m in deq.items()}
            ret = eval_retrieval(vectors, metadata, test_ids)

            cb513_deq = {}
            for pid, m in cb513_coded.items():
                idx = vq_encode(m, vq_centroids)
                vq_rec = vq_centroids[idx]
                residual = m - vq_rec
                pq_codes = pq_encode(residual, pq_model)
                pq_rec = pq_decode(pq_codes, pq_model)
                cb513_deq[pid] = vq_rec + pq_rec

            ss3 = evaluate_ss3_probe(cb513_deq, ss3_labels, cb_train, cb_test)

            # Storage: VQ index (uint8 or uint16) + PQ codes (M bytes) per residue
            vq_idx_bytes = 1 if vq_K <= 256 else 2
            size = mean_L * (vq_idx_bytes + pq_M) + 2048 * 2
            elapsed = time.time() - t0

            key = f"vq{vq_K}_pq{pq_M}"
            c4_results[key] = {
                "method": key,
                "vq_K": vq_K,
                "pq_M": pq_M,
                "family_ret1": ret["precision@1"],
                "family_mrr": ret["mrr"],
                "ss3_q3": ss3["q3"],
                "per_protein_bytes": size,
                "per_protein_kb": round(size / 1024, 1),
                "vs_mean_pool": round(size / 2048, 1),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"      Ret@1={ret['precision@1']:.3f}  SS3={ss3['q3']:.3f}  "
                  f"size={size/1024:.1f}KB ({size/2048:.1f}x)  [{elapsed:.0f}s]")

            del deq, cb513_deq

    results["C4"] = c4_results
    results["steps_done"].append("C4")
    save_results(results)
    print("\n  C4 complete!")
    return results


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 60)
    print("PHASE C SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30s} {'Ret@1':>7s} {'SS3 Q3':>7s} {'KB':>7s} {'×mean':>6s}")
    print("-" * 60)

    all_rows = []
    for step_key in ["C1", "C2", "C4"]:
        if step_key in results:
            all_rows.extend(results[step_key].values())

    for r in sorted(all_rows, key=lambda x: x.get("family_ret1", 0), reverse=True):
        name = r.get("method", "?")
        ret1 = r.get("family_ret1", 0)
        ss3 = r.get("ss3_q3", 0)
        kb = r.get("per_protein_kb", 0)
        vs = r.get("vs_mean_pool", 0)
        extra = ""
        if "cos_sim" in r:
            extra = f"  cos={r['cos_sim']:.3f}"
        if "utilization_pct" in r:
            extra += f"  util={r['utilization_pct']:.0f}%"
        print(f"  {name:<28s} {ret1:>7.3f} {ss3:>7.3f} {kb:>7.1f} {vs:>5.1f}x{extra}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()

    step = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--step"):
                step = sys.argv[sys.argv.index(arg) + 1]

    if step is None or step == "C1":
        if "C1" not in results.get("steps_done", []):
            results = step_C1(results)
        else:
            print("C1 already done, skipping")

    if step is None or step == "C2":
        if "C2" not in results.get("steps_done", []):
            results = step_C2(results)
        else:
            print("C2 already done, skipping")

    if step is None or step == "C4":
        if "C4" not in results.get("steps_done", []):
            results = step_C4(results)
        else:
            print("C4 already done, skipping")

    print_summary(results)
    print(f"\nResults saved to {RESULTS_PATH}")
