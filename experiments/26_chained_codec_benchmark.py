#!/usr/bin/env python3
"""Experiment 26: Chained & Improved Codec Benchmark.

Tests 11 new codecs that Experiment 25 missed:
  A. Chained codecs: D-compress (rp512/fh512) → smart pool (mean_max/dct_K4)
  B. Importance-weighted pooling (entropy, cosine deviation, attention-to-mean)
  C. Fixes for broken codecs (kernel_mean auto-gamma, svd_spectrum k=64)
  D. Token merging (adjacent residues cos>0.95)

The key idea: chained codecs give BOTH per-residue (L,512) AND smart per-protein
retrieval, closing the gap between D-compression and smart pooling.

Steps:
  D1: Per-protein retrieval (11 codecs × 3 PLMs × cosine; #5 euclidean)
  D2: Hierarchy separation (11 codecs × 3 PLMs)
  D3: Biology correlations (11 codecs × 3 PLMs × GO/EC/Pfam/taxonomy)
  D4: Per-residue probes (chained codecs × 3 PLMs × SS3/SS8/disorder/TM/SignalP)
  D5: Aggregator — cross-experiment comparison table merging Exp 25 + 26

Usage:
  uv run python experiments/26_chained_codec_benchmark.py --step D1
  uv run python experiments/26_chained_codec_benchmark.py --step D5
  uv run python experiments/26_chained_codec_benchmark.py           # all steps
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.distance import pdist

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.hierarchy import evaluate_hierarchy_distances
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    evaluate_disorder_probe,
    evaluate_tm_probe,
    evaluate_signalp_probe,
    load_cb513_csv,
    load_chezod_seth,
    load_tmbed_annotated,
    load_signalp6_annotated,
)
from src.evaluation.biological_annotations import (
    map_scope_to_uniprot,
    fetch_uniprot_annotations,
    fetch_pdb_organisms,
    load_ncbi_taxonomy,
    parse_scope_to_pdb,
    evaluate_go_correlation,
    evaluate_ec_retrieval,
    evaluate_pfam_retrieval,
    evaluate_taxonomy_correlation,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.utils.h5_store import load_residue_embeddings

# Codec transforms
from src.one_embedding.universal_transforms import (
    kernel_mean_embedding,
    svd_spectrum,
    feature_hash,
    random_orthogonal_project,
)
from src.one_embedding.transforms import dct_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "chained_codec_results.json"
EXP25_RESULTS_PATH = DATA_DIR / "benchmarks" / "universal_codec_results.json"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512

PLMS = [
    ("prot_t5_xl", 1024, "prot_t5_xl"),
    ("esm2_650m", 1280, "esm2_650m"),
    ("esmc_300m", 960, "esmc_300m"),
]


# ── Codec functions ─────────────────────────────────────────────────

# --- A. Chained codecs: D-compress → smart pool ---

def _rp512_mean_max(m):
    """RP(L,D)→(L,512), then [mean|max]→(1024,)."""
    compressed = random_orthogonal_project(m[:MAX_LEN], d_out=512)
    return np.concatenate([compressed.mean(axis=0), compressed.max(axis=0)]).astype(np.float32)

def _rp512_dct_K4(m):
    """RP(L,D)→(L,512), then DCT K=4→(2048,)."""
    compressed = random_orthogonal_project(m[:MAX_LEN], d_out=512)
    return dct_summary(compressed, K=4)

def _fh512_mean_max(m):
    """FH(L,D)→(L,512), then [mean|max]→(1024,)."""
    compressed = feature_hash(m[:MAX_LEN], d_out=512)
    return np.concatenate([compressed.mean(axis=0), compressed.max(axis=0)]).astype(np.float32)

def _fh512_dct_K4(m):
    """FH(L,D)→(L,512), then DCT K=4→(2048,)."""
    compressed = feature_hash(m[:MAX_LEN], d_out=512)
    return dct_summary(compressed, K=4)

def _rp512_mean_max_euc(m):
    """Same vector as rp512_mean_max — evaluated with euclidean metric."""
    return _rp512_mean_max(m)

# Per-residue for chained codecs (same D-compression as Exp 25)
def _rp512_residue(m):
    return random_orthogonal_project(m[:MAX_LEN], d_out=512)

def _fh512_residue(m):
    return feature_hash(m[:MAX_LEN], d_out=512)


# --- B. Importance-weighted pooling ---

def _entropy_weighted(m):
    """Weight residues by softmax entropy — high-entropy = more 'informative'."""
    m = m[:MAX_LEN].astype(np.float32)
    # Filter zero-norm rows (padding artifacts)
    norms = np.linalg.norm(m, axis=1)
    valid = norms > 1e-8
    if valid.sum() == 0:
        return np.zeros(m.shape[1], dtype=np.float32)
    m_valid = m[valid]

    # Softmax per residue → entropy
    # Shift for numerical stability
    shifted = m_valid - m_valid.max(axis=1, keepdims=True)
    exp_m = np.exp(shifted)
    softmax = exp_m / exp_m.sum(axis=1, keepdims=True).clip(1e-12)
    # Entropy: -sum(p * log(p))
    log_softmax = np.log(softmax.clip(1e-12))
    entropy = -(softmax * log_softmax).sum(axis=1)  # (L_valid,)

    weights = entropy / entropy.sum().clip(1e-12)
    return (weights[:, np.newaxis] * m_valid).sum(axis=0).astype(np.float32)

def _cosine_deviation(m):
    """Weight residues by 1 - cos(residue, mean) — outliers contribute more."""
    m = m[:MAX_LEN].astype(np.float32)
    mean_vec = m.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec).clip(1e-8)
    row_norms = np.linalg.norm(m, axis=1).clip(1e-8)
    cos_sims = (m @ mean_vec) / (row_norms * mean_norm)
    deviations = 1.0 - cos_sims  # higher = more deviant
    # Clamp negative deviations (shouldn't happen with real data but safety)
    deviations = deviations.clip(0)
    weights = deviations / deviations.sum().clip(1e-12)
    return (weights[:, np.newaxis] * m).sum(axis=0).astype(np.float32)

def _attention_to_mean(m):
    """Attention pooling: softmax(residues @ mean / sqrt(d))."""
    m = m[:MAX_LEN].astype(np.float32)
    mean_vec = m.mean(axis=0)  # (D,)
    d = m.shape[1]
    scores = (m @ mean_vec) / np.sqrt(d)  # (L,)
    # Softmax
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / weights.sum().clip(1e-12)
    return (weights[:, np.newaxis] * m).sum(axis=0).astype(np.float32)


# --- C. Fixes for broken codecs ---

def _kernel_mean_auto(m):
    """Kernel mean embedding with median heuristic for gamma."""
    m = m[:MAX_LEN].astype(np.float32)
    L = m.shape[0]
    # Subsample for speed
    n_sub = min(50, L)
    rng = np.random.RandomState(42)
    idx = rng.choice(L, n_sub, replace=False) if L > n_sub else np.arange(L)
    sub = m[idx]
    # Median heuristic: gamma = 1 / median(pairwise squared euclidean distances)
    if len(sub) > 1:
        dists_sq = pdist(sub, metric="sqeuclidean")
        median_dist_sq = np.median(dists_sq)
        gamma = 1.0 / max(median_dist_sq, 1e-8)
    else:
        gamma = 1.0
    return kernel_mean_embedding(m, D_out=2048, gamma=gamma)

def _svd_spectrum_k64(m):
    """SVD spectrum with k=64 instead of k=16."""
    return svd_spectrum(m[:MAX_LEN], k=64)


# --- D. Token merging ---

def _token_merge_mean(m):
    """Merge adjacent residues with cos>0.95, then mean pool."""
    m = m[:MAX_LEN].astype(np.float32)
    L = m.shape[0]
    if L <= 1:
        return m.mean(axis=0)

    # Compute cosine similarity between adjacent pairs
    norms = np.linalg.norm(m, axis=1).clip(1e-8)
    m_normed = m / norms[:, np.newaxis]
    adj_cos = (m_normed[:-1] * m_normed[1:]).sum(axis=1)  # (L-1,)

    # Greedy merge: group consecutive residues with cos > threshold
    threshold = 0.95
    groups = []
    current_group = [0]
    for i in range(L - 1):
        if adj_cos[i] > threshold:
            current_group.append(i + 1)
        else:
            groups.append(current_group)
            current_group = [i + 1]
    groups.append(current_group)

    # Mean pool within each group, then mean pool across groups
    merged = np.array([m[g].mean(axis=0) for g in groups], dtype=np.float32)
    return merged.mean(axis=0).astype(np.float32)


# ── Codec registry ──────────────────────────────────────────────────

CODECS_PER_PROTEIN = [
    # A. Chained
    ("rp512_mean_max", _rp512_mean_max),
    ("rp512_dct_K4", _rp512_dct_K4),
    ("fh512_mean_max", _fh512_mean_max),
    ("fh512_dct_K4", _fh512_dct_K4),
    # rp512_mean_max_euc uses same fn, different metric — handled separately
    # B. Importance-weighted
    ("entropy_weighted", _entropy_weighted),
    ("cosine_deviation", _cosine_deviation),
    ("attention_to_mean", _attention_to_mean),
    # C. Fixes
    ("kernel_mean_auto", _kernel_mean_auto),
    ("svd_spectrum_k64", _svd_spectrum_k64),
    # D. Token merge
    ("token_merge_mean", _token_merge_mean),
]

CODECS_PER_RESIDUE = [
    ("rp512_mean_max", _rp512_residue),
    ("rp512_dct_K4", _rp512_residue),
    ("fh512_mean_max", _fh512_residue),
    ("fh512_dct_K4", _fh512_residue),
    ("rp512_mean_max_euc", _rp512_residue),
]


# ── Helpers (from Exp 25) ───────────────────────────────────────────

def monitor():
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except OSError:
        pass

def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": [], "results": {}}

def save_results(results: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

def mark_done(results: dict, step: str):
    results.setdefault("steps_done", [])
    if step not in results["steps_done"]:
        results["steps_done"].append(step)

def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)

def load_metadata() -> list[dict]:
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    return metadata

def load_plm_embeddings(plm_stem: str, dataset: str = "medium5k") -> dict[str, np.ndarray]:
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
    if not h5_path.exists():
        print(f"  WARNING: {h5_path} not found, skipping")
        return {}
    embeddings = load_residue_embeddings(h5_path)
    needs_remap = any("|" in k for k in list(embeddings.keys())[:5])
    if needs_remap:
        remapped = {}
        for k, v in embeddings.items():
            remapped[k.split("|")[0]] = v
        return remapped
    return embeddings

def load_validation_embeddings(plm_stem: str, prefix: str) -> dict[str, np.ndarray]:
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_validation.h5"
    if not h5_path.exists():
        return {}
    import h5py
    result = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            if key.startswith(prefix):
                result[key[len(prefix):]] = np.array(f[key], dtype=np.float32)
    if result:
        print(f"    Loaded {len(result)} from validation H5 (prefix={prefix})")
    return result

def apply_codec_per_protein(
    embeddings: dict[str, np.ndarray],
    ids: list[str],
    codec_fn,
) -> dict[str, np.ndarray]:
    vectors = {}
    for pid in ids:
        if pid not in embeddings:
            continue
        vectors[pid] = codec_fn(embeddings[pid])
    return vectors

def apply_codec_per_residue(
    embeddings: dict[str, np.ndarray],
    ids: list[str],
    codec_fn,
) -> dict[str, np.ndarray]:
    result = {}
    for pid in ids:
        if pid not in embeddings:
            continue
        result[pid] = codec_fn(embeddings[pid])
    return result

def _load_biology_context(metadata, split):
    """Load all biology annotation data (shared across codecs)."""
    scope_ids = [m["id"] for m in metadata]

    scope_to_uniprot = map_scope_to_uniprot(
        scope_ids,
        sifts_path=str(DATA_DIR / "annotations" / "sifts_mapping.json"),
    )

    annotations_cache = DATA_DIR / "annotations" / "uniprot_annotations.json"
    if not annotations_cache.exists():
        print("  WARNING: annotations not found. Run Exp 24 B10 first.")
        return None
    with open(annotations_cache) as f:
        all_annotations = json.load(f)

    go_terms, ec_numbers, pfam_domains = {}, {}, {}
    for sid, uid in scope_to_uniprot.items():
        ann = all_annotations.get(uid, {})
        if ann.get("go"):
            go_terms[sid] = ann["go"]
        if ann.get("ec"):
            ec_numbers[sid] = ann["ec"]
        if ann.get("pfam"):
            pfam_domains[sid] = ann["pfam"]

    # Taxonomy
    pdb_organisms_cache = DATA_DIR / "annotations" / "pdb_organisms.json"
    pdb_organisms = {}
    if pdb_organisms_cache.exists():
        with open(pdb_organisms_cache) as f:
            pdb_organisms = {k: int(v) for k, v in json.load(f).items()}

    ncbi_taxonomy = load_ncbi_taxonomy()
    protein_taxonomy = {}
    for sid in scope_ids:
        pdb_id, _ = parse_scope_to_pdb(sid)
        taxid = pdb_organisms.get(pdb_id.upper()) or pdb_organisms.get(pdb_id.lower())
        if taxid and taxid in ncbi_taxonomy:
            protein_taxonomy[sid] = ncbi_taxonomy[taxid]

    print(f"  Biology context: GO={len(go_terms)}, EC={len(ec_numbers)}, "
          f"Pfam={len(pfam_domains)}, Taxonomy={len(protein_taxonomy)}")

    return {
        "go_terms": go_terms,
        "ec_numbers": ec_numbers,
        "pfam_domains": pfam_domains,
        "protein_taxonomy": protein_taxonomy,
    }


# ── Step D1: Retrieval ─────────────────────────────────────────────

def step_D1(results: dict):
    """D1: Per-protein retrieval for all codecs × all PLMs."""
    print("\n═══ D1: Per-Protein Retrieval ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    step_results = results.setdefault("results", {}).setdefault("retrieval", {})

    for plm_name, dim, plm_stem in PLMS:
        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"\n  {plm_name} ({dim}d):")

        # Standard codecs with cosine
        for codec_name, codec_fn in CODECS_PER_PROTEIN:
            rkey = f"{plm_name}_{codec_name}"
            if rkey in step_results:
                print(f"    {codec_name}: already done, skipping")
                continue

            t0 = time.time()
            vectors = apply_codec_per_protein(embeddings, test_ids, codec_fn)
            if not vectors:
                continue

            ret = evaluate_retrieval_from_vectors(
                vectors, metadata, label_key="family",
                query_ids=test_ids, database_ids=test_ids, metric="cosine",
            )
            sf = evaluate_retrieval_from_vectors(
                vectors, metadata, label_key="superfamily",
                query_ids=test_ids, database_ids=test_ids, metric="cosine",
            )
            fold = evaluate_retrieval_from_vectors(
                vectors, metadata, label_key="fold",
                query_ids=test_ids, database_ids=test_ids, metric="cosine",
            )
            sample_vec = next(iter(vectors.values()))
            step_results[rkey] = {
                "family_ret1": ret["precision@1"],
                "family_mrr": ret["mrr"],
                "sf_ret1": sf["precision@1"],
                "fold_ret1": fold["precision@1"],
                "dim": int(sample_vec.shape[0]),
                "n_queries": ret["n_queries"],
            }
            elapsed = time.time() - t0
            print(f"    {codec_name} (d={sample_vec.shape[0]}): "
                  f"Ret@1={ret['precision@1']:.3f}, SF={sf['precision@1']:.3f}, "
                  f"Fold={fold['precision@1']:.3f} [{elapsed:.1f}s]")
            save_results(results)

        # #5: rp512_mean_max with euclidean metric
        rkey_euc = f"{plm_name}_rp512_mean_max_euc"
        if rkey_euc not in step_results:
            vectors = apply_codec_per_protein(embeddings, test_ids, _rp512_mean_max_euc)
            if vectors:
                ret = evaluate_retrieval_from_vectors(
                    vectors, metadata, label_key="family",
                    query_ids=test_ids, database_ids=test_ids, metric="euclidean",
                )
                sf = evaluate_retrieval_from_vectors(
                    vectors, metadata, label_key="superfamily",
                    query_ids=test_ids, database_ids=test_ids, metric="euclidean",
                )
                fold = evaluate_retrieval_from_vectors(
                    vectors, metadata, label_key="fold",
                    query_ids=test_ids, database_ids=test_ids, metric="euclidean",
                )
                sample_vec = next(iter(vectors.values()))
                step_results[rkey_euc] = {
                    "family_ret1": ret["precision@1"],
                    "family_mrr": ret["mrr"],
                    "sf_ret1": sf["precision@1"],
                    "fold_ret1": fold["precision@1"],
                    "dim": int(sample_vec.shape[0]),
                    "metric": "euclidean",
                    "n_queries": ret["n_queries"],
                }
                print(f"    rp512_mean_max_euc (d={sample_vec.shape[0]}): "
                      f"Ret@1={ret['precision@1']:.3f}, SF={sf['precision@1']:.3f}, "
                      f"Fold={fold['precision@1']:.3f}")
                save_results(results)

        del embeddings

    mark_done(results, "D1")
    monitor()


# ── Step D2: Hierarchy ─────────────────────────────────────────────

def step_D2(results: dict):
    """D2: SCOP hierarchy separation for all codecs × all PLMs."""
    print("\n═══ D2: Hierarchy Separation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    step_results = results.setdefault("results", {}).setdefault("hierarchy", {})

    all_codecs = CODECS_PER_PROTEIN + [("rp512_mean_max_euc", _rp512_mean_max_euc)]

    for plm_name, dim, plm_stem in PLMS:
        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"\n  {plm_name} ({dim}d):")

        for codec_name, codec_fn in all_codecs:
            rkey = f"{plm_name}_{codec_name}"
            if rkey in step_results:
                print(f"    {codec_name}: already done, skipping")
                continue

            vectors = apply_codec_per_protein(embeddings, test_ids, codec_fn)
            if not vectors:
                continue

            metric = "euclidean" if codec_name == "rp512_mean_max_euc" else "cosine"
            hier = evaluate_hierarchy_distances(vectors, metadata, metric=metric)
            step_results[rkey] = {
                "sep_ratio": hier.get("separation_ratio"),
                "ordering_ok": hier.get("ordering_correct", False),
            }
            if metric != "cosine":
                step_results[rkey]["metric"] = metric
            print(f"    {codec_name}: sep_ratio={hier.get('separation_ratio', 0):.3f}")
            save_results(results)

        del embeddings

    mark_done(results, "D2")
    monitor()


# ── Step D3: Biology ───────────────────────────────────────────────

def step_D3(results: dict):
    """D3: Biology correlations for all codecs × all PLMs."""
    print("\n═══ D3: Biology Correlations ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    bio_ctx = _load_biology_context(metadata, split)
    if bio_ctx is None:
        return

    step_results = results.setdefault("results", {}).setdefault("biology", {})

    all_codecs = CODECS_PER_PROTEIN + [("rp512_mean_max_euc", _rp512_mean_max_euc)]

    for plm_name, dim, plm_stem in PLMS:
        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"\n  {plm_name} ({dim}d):")

        for codec_name, codec_fn in all_codecs:
            rkey = f"{plm_name}_{codec_name}"
            if rkey in step_results:
                print(f"    {codec_name}: already done, skipping")
                continue

            t0 = time.time()
            vectors = apply_codec_per_protein(embeddings, test_ids, codec_fn)
            if not vectors:
                continue

            metric = "euclidean" if codec_name == "rp512_mean_max_euc" else "cosine"
            bio = {}

            if bio_ctx["go_terms"]:
                go_res = evaluate_go_correlation(vectors, bio_ctx["go_terms"], metric=metric)
                bio["go_rho"] = go_res.get("spearman_rho", 0)

            if bio_ctx["ec_numbers"]:
                ec_res = evaluate_ec_retrieval(vectors, bio_ctx["ec_numbers"], metric=metric)
                bio["ec_ret1"] = ec_res.get("ec_full_ret1", 0)
                bio["ec_level1_ret1"] = ec_res.get("ec_level1_ret1", 0)

            if bio_ctx["pfam_domains"]:
                pfam_res = evaluate_pfam_retrieval(vectors, bio_ctx["pfam_domains"], metric=metric)
                bio["pfam_ret1"] = pfam_res.get("pfam_ret1", 0)

            if bio_ctx["protein_taxonomy"]:
                tax_res = evaluate_taxonomy_correlation(
                    vectors, bio_ctx["protein_taxonomy"], metric=metric,
                )
                bio["tax_rho"] = tax_res.get("spearman_rho", 0)

            step_results[rkey] = bio
            elapsed = time.time() - t0
            print(f"    {codec_name}: GO={bio.get('go_rho', 0):.3f}, "
                  f"EC={bio.get('ec_ret1', 0):.3f}, Pfam={bio.get('pfam_ret1', 0):.3f}, "
                  f"Tax={bio.get('tax_rho', 0):.3f} [{elapsed:.1f}s]")
            save_results(results)

        del embeddings

    mark_done(results, "D3")
    monitor()


# ── Step D4: Per-Residue Probes ────────────────────────────────────

def step_D4(results: dict):
    """D4: Per-residue probes for chained codecs (A1-A5)."""
    print("\n═══ D4: Per-Residue Probes ═══")

    step_results = results.setdefault("results", {}).setdefault("per_residue", {})

    # ── SS3 / SS8 (CB513) ──
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if cb513_path.exists():
        sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
        print(f"\n  CB513: {len(ss3_labels)} proteins")

        for plm_name, dim, plm_stem in PLMS:
            embeddings = load_plm_embeddings(plm_stem, "cb513")
            if not embeddings:
                continue

            for codec_name, codec_fn in CODECS_PER_RESIDUE:
                ss3_key = f"{plm_name}_{codec_name}_ss3"
                if ss3_key in step_results:
                    print(f"    {plm_name}/{codec_name} SS3/SS8: already done, skipping")
                    continue

                avail_ids = [pid for pid in ss3_labels if pid in embeddings]
                coded_embs = apply_codec_per_residue(embeddings, avail_ids, codec_fn)

                rng = random.Random(42)
                rng.shuffle(avail_ids)
                n_train = int(len(avail_ids) * 0.8)
                train_ids = avail_ids[:n_train]
                test_ids_cb = avail_ids[n_train:]

                print(f"    {plm_name}/{codec_name} SS3 ({len(train_ids)} train, {len(test_ids_cb)} test)...")
                ss3 = evaluate_ss3_probe(coded_embs, ss3_labels, train_ids, test_ids_cb)
                step_results[ss3_key] = {"q3": ss3.get("q3", 0)}
                print(f"      Q3={ss3.get('q3', 0):.3f}")

                ss8 = evaluate_ss8_probe(coded_embs, ss8_labels, train_ids, test_ids_cb)
                step_results[f"{plm_name}_{codec_name}_ss8"] = {"q8": ss8.get("q8", 0)}
                print(f"      Q8={ss8.get('q8', 0):.3f}")

                save_results(results)

            del embeddings
    else:
        print("  CB513 not found, skipping SS3/SS8")

    # ── Disorder (SETH/CheZOD) ──
    seth_dir = DATA_DIR / "per_residue_benchmarks"
    seth_train_fasta = seth_dir / "SETH" / "CheZOD1174_training_set_sequences.fasta"
    if seth_train_fasta.exists():
        sequences, disorder_scores, train_ids_seth, test_ids_seth = load_chezod_seth(seth_dir)
        print(f"\n  SETH disorder: {len(disorder_scores)} proteins")

        for plm_name, dim, plm_stem in PLMS:
            embeddings = load_plm_embeddings(plm_stem, "seth")
            if not embeddings:
                embeddings = load_validation_embeddings(plm_stem, "chezod_")
            if not embeddings:
                continue

            for codec_name, codec_fn in CODECS_PER_RESIDUE:
                dis_key = f"{plm_name}_{codec_name}_disorder"
                if dis_key in step_results:
                    print(f"    {plm_name}/{codec_name} disorder: already done, skipping")
                    continue

                coded_embs = apply_codec_per_residue(embeddings, list(disorder_scores.keys()), codec_fn)
                print(f"    {plm_name}/{codec_name} disorder probe...")
                dis = evaluate_disorder_probe(coded_embs, disorder_scores, train_ids_seth, test_ids_seth)
                step_results[dis_key] = {"rho": dis.get("spearman_rho", 0)}
                print(f"      rho={dis.get('spearman_rho', 0):.3f}")
                save_results(results)

            del embeddings
    else:
        print("  SETH data not found, skipping disorder")

    # ── TM Topology (TMbed) ──
    tmbed_path = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
    if tmbed_path.exists():
        tm_sequences, tm_labels = load_tmbed_annotated(tmbed_path)
        print(f"\n  TMbed: {len(tm_labels)} proteins")

        for plm_name, dim, plm_stem in PLMS:
            embeddings = load_plm_embeddings(plm_stem, "tmbed")
            if not embeddings:
                embeddings = load_validation_embeddings(plm_stem, "tmbed_")
            if not embeddings:
                continue

            for codec_name, codec_fn in CODECS_PER_RESIDUE:
                tm_key = f"{plm_name}_{codec_name}_tm"
                if tm_key in step_results:
                    print(f"    {plm_name}/{codec_name} TM: already done, skipping")
                    continue

                avail_tm = [pid for pid in tm_labels if pid in embeddings]
                if not avail_tm:
                    print(f"    {plm_name}/{codec_name} TM: no ID overlap, skipping")
                    continue

                coded_embs = apply_codec_per_residue(embeddings, avail_tm, codec_fn)
                rng = random.Random(42)
                rng.shuffle(avail_tm)
                n_train = int(len(avail_tm) * 0.8)
                train_tm = avail_tm[:n_train]
                test_tm = avail_tm[n_train:]

                print(f"    {plm_name}/{codec_name} TM probe ({len(train_tm)} train, {len(test_tm)} test)...")
                tm = evaluate_tm_probe(coded_embs, tm_labels, train_tm, test_tm)
                step_results[tm_key] = {"macro_f1": tm.get("macro_f1", 0)}
                print(f"      macro_f1={tm.get('macro_f1', 0):.3f}")
                save_results(results)

            del embeddings
    else:
        print("  TMbed not found, skipping TM topology")

    # ── Signal Peptide (SignalP6) ──
    signalp_path = DATA_DIR / "per_residue_benchmarks" / "SignalP6" / "train_set.fasta"
    if signalp_path.exists():
        sp_sequences, sp_labels = load_signalp6_annotated(signalp_path)
        avail_sp = list(sp_sequences.keys())
        rng = random.Random(42)
        rng.shuffle(avail_sp)
        n_train = int(len(avail_sp) * 0.8)
        train_sp = avail_sp[:n_train]
        test_sp = avail_sp[n_train:]
        print(f"\n  SignalP: {len(sp_labels)} proteins ({len(train_sp)} train, {len(test_sp)} test)")

        for plm_name, dim, plm_stem in PLMS:
            embeddings = load_plm_embeddings(plm_stem, "signalp")
            if not embeddings:
                continue

            for codec_name, codec_fn in CODECS_PER_RESIDUE:
                sp_key = f"{plm_name}_{codec_name}_signalp"
                if sp_key in step_results:
                    print(f"    {plm_name}/{codec_name} SignalP: already done, skipping")
                    continue

                coded_embs = apply_codec_per_residue(embeddings, avail_sp, codec_fn)
                print(f"    {plm_name}/{codec_name} SignalP probe...")
                sp = evaluate_signalp_probe(coded_embs, sp_labels, train_sp, test_sp)
                step_results[sp_key] = {
                    "macro_f1": sp.get("macro_f1", 0),
                    "signal_f1": sp.get("signal_f1", 0),
                }
                print(f"      macro_f1={sp.get('macro_f1', 0):.3f}, signal_f1={sp.get('signal_f1', 0):.3f}")
                save_results(results)

            del embeddings
    else:
        print("  SignalP6 not found, skipping signal peptide")

    mark_done(results, "D4")
    monitor()


# ── Step D5: Aggregator ────────────────────────────────────────────

def step_D5(results: dict):
    """D5: Cross-experiment comparison merging Exp 25 + Exp 26."""
    print("\n═══ D5: Cross-Experiment Comparison ═══\n")

    # Load Exp 25 results
    exp25 = {}
    if EXP25_RESULTS_PATH.exists():
        with open(EXP25_RESULTS_PATH) as f:
            exp25 = json.load(f)
        print(f"  Loaded Exp 25 results from {EXP25_RESULTS_PATH}")
    else:
        print("  WARNING: Exp 25 results not found")

    all_results = results.get("results", {})
    retrieval = all_results.get("retrieval", {})
    hierarchy = all_results.get("hierarchy", {})
    biology = all_results.get("biology", {})
    per_residue = all_results.get("per_residue", {})

    exp25_results = exp25.get("results", {})
    exp25_retrieval = exp25_results.get("retrieval", {})
    exp25_per_residue = exp25_results.get("per_residue", {})

    codec_names = [c[0] for c in CODECS_PER_PROTEIN] + ["rp512_mean_max_euc"]
    per_res_codec_names = [c[0] for c in CODECS_PER_RESIDUE]

    # Build summary table for Exp 26 codecs
    rows = []
    for plm_name, dim, _ in PLMS:
        for codec_name in codec_names:
            rkey = f"{plm_name}_{codec_name}"
            ret = retrieval.get(rkey, {})
            hier = hierarchy.get(rkey, {})
            bio = biology.get(rkey, {})

            row = {
                "plm": plm_name,
                "codec": codec_name,
                "experiment": 26,
                "dim": ret.get("dim", "?"),
                "family_ret1": ret.get("family_ret1"),
                "family_mrr": ret.get("family_mrr"),
                "sf_ret1": ret.get("sf_ret1"),
                "fold_ret1": ret.get("fold_ret1"),
                "sep_ratio": hier.get("sep_ratio"),
                "go_rho": bio.get("go_rho"),
                "ec_ret1": bio.get("ec_ret1"),
                "pfam_ret1": bio.get("pfam_ret1"),
                "tax_rho": bio.get("tax_rho"),
            }

            # Per-residue
            if codec_name in per_res_codec_names:
                row["ss3_q3"] = per_residue.get(f"{plm_name}_{codec_name}_ss3", {}).get("q3")
                row["ss8_q8"] = per_residue.get(f"{plm_name}_{codec_name}_ss8", {}).get("q8")
                row["disorder_rho"] = per_residue.get(f"{plm_name}_{codec_name}_disorder", {}).get("rho")
                row["tm_f1"] = per_residue.get(f"{plm_name}_{codec_name}_tm", {}).get("macro_f1")
                row["signalp_f1"] = per_residue.get(f"{plm_name}_{codec_name}_signalp", {}).get("macro_f1")

            rows.append(row)

    # Print Exp 26 results per PLM
    for plm_name, _, _ in PLMS:
        plm_rows = [r for r in rows if r["plm"] == plm_name]
        print(f"\n### {plm_name} — Exp 26 codecs")
        print("| Codec | Dim | Fam Ret@1 | SF Ret@1 | Fold | Sep | GO | EC | Pfam | Tax | SS3 | SS8 | Dis | TM | SP |")
        print("|-------|-----|-----------|----------|------|-----|----|----|------|-----|-----|-----|-----|----|-----|")
        for r in plm_rows:
            def fmt(v):
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.3f}"
                return str(v)
            cols = [
                r["codec"][:20],
                fmt(r["dim"]),
                fmt(r.get("family_ret1")),
                fmt(r.get("sf_ret1")),
                fmt(r.get("fold_ret1")),
                fmt(r.get("sep_ratio")),
                fmt(r.get("go_rho")),
                fmt(r.get("ec_ret1")),
                fmt(r.get("pfam_ret1")),
                fmt(r.get("tax_rho")),
                fmt(r.get("ss3_q3")),
                fmt(r.get("ss8_q8")),
                fmt(r.get("disorder_rho")),
                fmt(r.get("tm_f1")),
                fmt(r.get("signalp_f1")),
            ]
            print(f"| {' | '.join(cols)} |")

    # Cross-experiment comparison: ProtT5 head-to-head
    print("\n\n### ProtT5-XL Head-to-Head (Exp 25 vs Exp 26)")
    print("| Codec | Exp | Dim | Fam Ret@1 | MRR | Per-Res? | SS3 |")
    print("|-------|-----|-----|-----------|-----|----------|-----|")

    # Key Exp 25 codecs for comparison
    exp25_codecs = ["mean_pool", "mean_max", "mean_max_euclidean", "rp512", "fh512", "dct_K4", "kernel_mean"]
    for codec_name in exp25_codecs:
        rkey = f"prot_t5_xl_{codec_name}"
        ret = exp25_retrieval.get(rkey, {})
        # Check per-residue from Exp 25
        ss3_val = exp25_per_residue.get(f"prot_t5_xl_{codec_name}_ss3", {}).get("q3")
        if ss3_val is None and codec_name == "dct_K4":
            ss3_val = exp25_per_residue.get("prot_t5_xl_dct_K4_inv_ss3", {}).get("q3")
        has_per_res = codec_name in ("rp512", "fh512", "dct_K4", "hrr_K1", "hrr_K8")
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        print(f"| {codec_name:<20} | 25  | {fmt(ret.get('dim', '?'))} | "
              f"{fmt(ret.get('family_ret1'))} | {fmt(ret.get('family_mrr'))} | "
              f"{'Yes' if has_per_res else 'No':>8} | {fmt(ss3_val)} |")

    # Key Exp 26 codecs
    exp26_highlight = ["rp512_mean_max", "rp512_dct_K4", "fh512_mean_max", "fh512_dct_K4",
                       "rp512_mean_max_euc", "entropy_weighted", "attention_to_mean",
                       "kernel_mean_auto", "token_merge_mean"]
    for codec_name in exp26_highlight:
        rkey = f"prot_t5_xl_{codec_name}"
        ret = retrieval.get(rkey, {})
        ss3_val = per_residue.get(f"prot_t5_xl_{codec_name}_ss3", {}).get("q3")
        has_per_res = codec_name in per_res_codec_names
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        print(f"| {codec_name:<20} | 26  | {fmt(ret.get('dim', '?'))} | "
              f"{fmt(ret.get('family_ret1'))} | {fmt(ret.get('family_mrr'))} | "
              f"{'Yes' if has_per_res else 'No':>8} | {fmt(ss3_val)} |")

    # Save
    results["results"]["summary_rows"] = rows
    save_results(results)
    print(f"\n  Results saved to {RESULTS_PATH}")

    mark_done(results, "D5")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 26: Chained & Improved Codec Benchmark",
    )
    parser.add_argument("--step", type=str, default=None, help="Run a specific step (D1-D5)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "D1": step_D1, "D2": step_D2, "D3": step_D3,
        "D4": step_D4, "D5": step_D5,
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

    print("\nDone.")


if __name__ == "__main__":
    main()
