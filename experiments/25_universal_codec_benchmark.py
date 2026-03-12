#!/usr/bin/env python3
"""Experiment 25: Universal Codec Benchmark.

Benchmarks 14 training-free codec methods across 3 PLMs (ProtT5-XL, ESM2-650M,
ESM-C 300M) on the full task suite: retrieval, hierarchy, biology, and per-residue
probes. Produces a unified comparison table.

Steps:
  C1: Per-protein retrieval (14 codecs × 3 PLMs)
  C2: Hierarchy separation (14 codecs × 3 PLMs)
  C3: Biology correlations (14 codecs × 3 PLMs × GO/EC/Pfam/taxonomy)
  C4: Per-residue probes (5 codecs × 3 PLMs × SS3/SS8/disorder/TM/SignalP)
  C5: Aggregator — unified table + JSON

Usage:
  uv run python experiments/25_universal_codec_benchmark.py --step C1
  uv run python experiments/25_universal_codec_benchmark.py --step C5
  uv run python experiments/25_universal_codec_benchmark.py           # all steps
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
    power_mean_pool,
    norm_weighted_mean,
    kernel_mean_embedding,
    svd_spectrum,
    feature_hash,
    random_orthogonal_project,
)
from src.one_embedding.transforms import (
    dct_summary,
    inverse_dct,
    haar_summary,
)
from src.one_embedding.hrr import (
    hrr_encode,
    hrr_decode,
    hrr_kslot_encode,
    hrr_kslot_decode,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "universal_codec_results.json"
PLOTS_DIR = DATA_DIR / "plots" / "exp25"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512

PLMS = [
    ("prot_t5_xl", 1024, "prot_t5_xl"),
    ("esm2_650m", 1280, "esm2_650m"),
    ("esmc_300m", 960, "esmc_300m"),
]

# ── Codec definitions ──────────────────────────────────────────────
# Each codec: (name, per_protein_fn, per_residue_fn_or_None)
# per_protein_fn: (L, D) -> (V,)
# per_residue_fn: (L, D) -> (L, d) or None

def _mean_pool(m):
    return m[:MAX_LEN].mean(axis=0).astype(np.float32)

def _max_pool(m):
    return m[:MAX_LEN].max(axis=0).astype(np.float32)

def _mean_max(m):
    m = m[:MAX_LEN]
    return np.concatenate([m.mean(axis=0), m.max(axis=0)]).astype(np.float32)

def _power_mean_p3(m):
    return power_mean_pool(m[:MAX_LEN], p=3.0)

def _norm_weighted(m):
    return norm_weighted_mean(m[:MAX_LEN])

def _kernel_mean(m):
    return kernel_mean_embedding(m[:MAX_LEN], D_out=2048)

def _svd_spec(m):
    return svd_spectrum(m[:MAX_LEN], k=16)

def _dct_K4_protein(m):
    """Per-protein: use full DCT K=4 summary as protein vector."""
    return dct_summary(m[:MAX_LEN], K=4)

def _dct_K4_residue(m):
    """Per-residue: DCT K=4 → inverse → approximate (L, D)."""
    m = m[:MAX_LEN]
    L, D = m.shape
    coeffs = dct_summary(m, K=4)
    return inverse_dct(coeffs, D, L)

def _haar_L3(m):
    return haar_summary(m[:MAX_LEN], levels=3)

def _fh512_protein(m):
    return feature_hash(m[:MAX_LEN], d_out=512).mean(axis=0)

def _fh512_residue(m):
    return feature_hash(m[:MAX_LEN], d_out=512)

def _rp512_protein(m):
    return random_orthogonal_project(m[:MAX_LEN], d_out=512).mean(axis=0)

def _rp512_residue(m):
    return random_orthogonal_project(m[:MAX_LEN], d_out=512)

def _hrr_K1_protein(m):
    return hrr_encode(m[:MAX_LEN])

def _hrr_K1_residue(m):
    m = m[:MAX_LEN]
    trace = hrr_encode(m)
    return hrr_decode(trace, m.shape[0])

def _hrr_K8_protein(m):
    return hrr_kslot_encode(m[:MAX_LEN], K=8).ravel()

def _hrr_K8_residue(m):
    m = m[:MAX_LEN]
    slots = hrr_kslot_encode(m, K=8)
    return hrr_kslot_decode(slots, m.shape[0])

# mean_max_euclidean uses same vector as mean_max but different metric — handled in eval


CODECS_PER_PROTEIN = [
    ("mean_pool", _mean_pool),
    ("max_pool", _max_pool),
    ("mean_max", _mean_max),
    ("power_mean_p3", _power_mean_p3),
    ("norm_weighted", _norm_weighted),
    ("kernel_mean", _kernel_mean),
    ("svd_spectrum", _svd_spec),
    ("dct_K4", _dct_K4_protein),
    ("haar_L3", _haar_L3),
    ("fh512", _fh512_protein),
    ("rp512", _rp512_protein),
    ("hrr_K1", _hrr_K1_protein),
    ("hrr_K8", _hrr_K8_protein),
    # mean_max_euclidean is #14 — same vector as mean_max, evaluated with euclidean
]

CODECS_PER_RESIDUE = [
    ("fh512", _fh512_residue),
    ("rp512", _rp512_residue),
    ("hrr_K1", _hrr_K1_residue),
    ("hrr_K8", _hrr_K8_residue),
    ("dct_K4_inv", _dct_K4_residue),
]


# ── Helpers ──────────────────────────────────────────────────────

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
    """Apply a per-protein codec to embeddings, returning {id: vector}."""
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
    """Apply a per-residue codec to embeddings, returning {id: (L, d)}."""
    result = {}
    for pid in ids:
        if pid not in embeddings:
            continue
        result[pid] = codec_fn(embeddings[pid])
    return result


# ── Step C1: Retrieval ─────────────────────────────────────────────

def step_C1(results: dict):
    """C1: Per-protein retrieval for all codecs × all PLMs."""
    print("\n═══ C1: Per-Protein Retrieval ═══")

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

        # #14: mean_max with euclidean metric
        rkey_euc = f"{plm_name}_mean_max_euclidean"
        if rkey_euc not in step_results:
            vectors = apply_codec_per_protein(embeddings, test_ids, _mean_max)
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
                print(f"    mean_max_euclidean (d={sample_vec.shape[0]}): "
                      f"Ret@1={ret['precision@1']:.3f}, SF={sf['precision@1']:.3f}, "
                      f"Fold={fold['precision@1']:.3f}")
                save_results(results)

        del embeddings

    mark_done(results, "C1")
    monitor()


# ── Step C2: Hierarchy ─────────────────────────────────────────────

def step_C2(results: dict):
    """C2: SCOP hierarchy separation for all codecs × all PLMs."""
    print("\n═══ C2: Hierarchy Separation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    step_results = results.setdefault("results", {}).setdefault("hierarchy", {})

    for plm_name, dim, plm_stem in PLMS:
        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"\n  {plm_name} ({dim}d):")

        for codec_name, codec_fn in CODECS_PER_PROTEIN:
            rkey = f"{plm_name}_{codec_name}"
            if rkey in step_results:
                print(f"    {codec_name}: already done, skipping")
                continue

            vectors = apply_codec_per_protein(embeddings, test_ids, codec_fn)
            if not vectors:
                continue

            hier = evaluate_hierarchy_distances(vectors, metadata, metric="cosine")
            step_results[rkey] = {
                "sep_ratio": hier.get("separation_ratio"),
                "ordering_ok": hier.get("ordering_correct", False),
            }
            print(f"    {codec_name}: sep_ratio={hier.get('separation_ratio', 0):.3f}")
            save_results(results)

        # #14: mean_max euclidean
        rkey_euc = f"{plm_name}_mean_max_euclidean"
        if rkey_euc not in step_results:
            vectors = apply_codec_per_protein(embeddings, test_ids, _mean_max)
            if vectors:
                hier = evaluate_hierarchy_distances(vectors, metadata, metric="euclidean")
                step_results[rkey_euc] = {
                    "sep_ratio": hier.get("separation_ratio"),
                    "ordering_ok": hier.get("ordering_correct", False),
                    "metric": "euclidean",
                }
                print(f"    mean_max_euclidean: sep_ratio={hier.get('separation_ratio', 0):.3f}")
                save_results(results)

        del embeddings

    mark_done(results, "C2")
    monitor()


# ── Step C3: Biology ───────────────────────────────────────────────

def _load_biology_context(metadata, split):
    """Load all biology annotation data (shared across codecs)."""
    test_ids = split["test_ids"]
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
    pdb_ids = list(set(parse_scope_to_pdb(sid)[0] for sid in scope_ids))
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


def step_C3(results: dict):
    """C3: Biology correlations for all codecs × all PLMs."""
    print("\n═══ C3: Biology Correlations ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    bio_ctx = _load_biology_context(metadata, split)
    if bio_ctx is None:
        return

    step_results = results.setdefault("results", {}).setdefault("biology", {})

    all_codecs = CODECS_PER_PROTEIN + [("mean_max_euclidean", _mean_max)]

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

            metric = "euclidean" if codec_name == "mean_max_euclidean" else "cosine"
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

    mark_done(results, "C3")
    monitor()


# ── Step C4: Per-Residue Probes ────────────────────────────────────

def step_C4(results: dict):
    """C4: Per-residue probes for codecs that preserve residue-level info."""
    print("\n═══ C4: Per-Residue Probes ═══")

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

                # Apply codec to get per-residue representations
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

    mark_done(results, "C4")
    monitor()


# ── Step C5: Aggregator ────────────────────────────────────────────

def step_C5(results: dict):
    """C5: Unified comparison table."""
    print("\n═══ C5: Universal Codec Comparison ═══\n")

    all_results = results.get("results", {})
    retrieval = all_results.get("retrieval", {})
    hierarchy = all_results.get("hierarchy", {})
    biology = all_results.get("biology", {})
    per_residue = all_results.get("per_residue", {})

    codec_names = [c[0] for c in CODECS_PER_PROTEIN] + ["mean_max_euclidean"]
    per_res_names = [c[0] for c in CODECS_PER_RESIDUE]

    # Build summary table
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
                "dim": ret.get("dim", "?"),
                "family_ret1": ret.get("family_ret1", None),
                "sf_ret1": ret.get("sf_ret1", None),
                "fold_ret1": ret.get("fold_ret1", None),
                "sep_ratio": hier.get("sep_ratio", None),
                "go_rho": bio.get("go_rho", None),
                "ec_ret1": bio.get("ec_ret1", None),
                "pfam_ret1": bio.get("pfam_ret1", None),
                "tax_rho": bio.get("tax_rho", None),
            }

            # Per-residue (if applicable)
            if codec_name in per_res_names or codec_name == "dct_K4":
                pr_name = "dct_K4_inv" if codec_name == "dct_K4" else codec_name
                row["ss3_q3"] = per_residue.get(f"{plm_name}_{pr_name}_ss3", {}).get("q3", None)
                row["ss8_q8"] = per_residue.get(f"{plm_name}_{pr_name}_ss8", {}).get("q8", None)
                row["disorder_rho"] = per_residue.get(f"{plm_name}_{pr_name}_disorder", {}).get("rho", None)
                row["tm_f1"] = per_residue.get(f"{plm_name}_{pr_name}_tm", {}).get("macro_f1", None)
                row["signalp_f1"] = per_residue.get(f"{plm_name}_{pr_name}_signalp", {}).get("macro_f1", None)

            rows.append(row)

    # Print per-PLM tables
    for plm_name, _, _ in PLMS:
        plm_rows = [r for r in rows if r["plm"] == plm_name]
        print(f"\n### {plm_name}")
        print(f"| Codec | Dim | Fam Ret@1 | SF Ret@1 | Fold | Sep | GO | EC | Pfam | Tax | SS3 | SS8 | Dis | TM | SP |")
        print(f"|-------|-----|-----------|----------|------|-----|----|----|------|-----|-----|-----|-----|----|-----|")
        for r in plm_rows:
            def fmt(v):
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.3f}"
                return str(v)
            cols = [
                r["codec"][:18],
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

    # Save full results
    results["results"]["summary_rows"] = rows
    save_results(results)
    print(f"\n  Results saved to {RESULTS_PATH}")

    mark_done(results, "C5")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 25: Universal Codec Benchmark",
    )
    parser.add_argument("--step", type=str, default=None, help="Run a specific step (C1-C5)")
    args = parser.parse_args()

    results = load_results()

    steps = {
        "C1": step_C1, "C2": step_C2, "C3": step_C3,
        "C4": step_C4, "C5": step_C5,
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
