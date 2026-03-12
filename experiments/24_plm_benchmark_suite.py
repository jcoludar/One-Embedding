#!/usr/bin/env python3
"""Experiment 24: Comprehensive PLM Benchmark Suite.

Compares ProtT5-XL (1024d), ESM2-650M (1280d), and ESM-C 300M (960d) across
retrieval, per-residue probes, biological annotation correlations, per-protein
regression, and efficiency metrics. Produces a unified aggregator score.

Addon to Exp 23 — reuses Euclidean metric, hierarchy evaluation infrastructure.

Steps:
  B1:  Extract ESM-C 300M for SCOPe medium5k          (~60 min CPU)
  B2:  Extract ESM-C 300M for CB513                    (~10 min)
  B3:  Extract ESM-C 300M for SETH + TMbed subsets     (~30 min)
  B4:  Download ProteinGLUE PPI + EPI data             (~1 min)
  B5:  Download eSol + Meltome datasets                (~2 min)
  B6:  Extract all 3 PLMs for new datasets             (~3-4 hrs)
  B7:  Per-protein retrieval (3 PLMs × cosine+euclidean)
  B8:  SCOP hierarchy evaluation (3 PLMs)
  B9:  Per-residue probes (3 PLMs)
  B10: Download SIFTS + fetch UniProt annotations
  B11: Biological correlations (3 PLMs)
  B12: Per-protein regression: solubility + thermostability
  B13: Aggregator: summary table + radar plot + JSON

Usage:
  uv run python experiments/24_plm_benchmark_suite.py --step B7   # retrieval
  uv run python experiments/24_plm_benchmark_suite.py --step B13  # aggregator
  uv run python experiments/24_plm_benchmark_suite.py             # all steps
"""

import argparse
import json
import os
import subprocess
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
    evaluate_ppi_probe,
    evaluate_epitope_probe,
    load_cb513_csv,
    load_chezod_seth,
    load_tmbed_annotated,
    load_signalp6_annotated,
    load_proteinglue_binary,
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
from src.evaluation.aggregator import BenchmarkAggregator
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "plm_benchmark_suite_results.json"
PLOTS_DIR = DATA_DIR / "plots" / "exp24"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
MAX_LEN = 512

# PLM configs: (name, embedding_dim, h5_stem)
PLMS = [
    ("prot_t5_xl", 1024, "prot_t5_xl"),
    ("esm2_650m", 1280, "esm2_650m"),
    ("esmc_300m", 960, "esmc_300m"),
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
    print(f"  Saved results to {RESULTS_PATH}")


def is_done(results: dict, step: str) -> bool:
    return step in results.get("steps_done", [])


def mark_done(results: dict, step: str):
    results.setdefault("steps_done", [])
    if step not in results["steps_done"]:
        results["steps_done"].append(step)
    save_results(results)


def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def load_metadata() -> list[dict]:
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    return metadata


def load_plm_embeddings(plm_stem: str, dataset: str = "medium5k") -> dict[str, np.ndarray]:
    """Load H5 embeddings for a PLM + dataset combination."""
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
    if not h5_path.exists():
        print(f"  WARNING: {h5_path} not found, skipping")
        return {}
    return load_residue_embeddings(h5_path)


def pool_mean(embeddings: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    return {pid: embeddings[pid][:MAX_LEN].mean(axis=0) for pid in ids if pid in embeddings}


# ── Steps ────────────────────────────────────────────────────────


def step_B1(results: dict):
    """B1: Extract ESM-C 300M for SCOPe medium5k."""
    print("\n═══ B1: Extract ESM-C 300M — SCOPe medium5k ═══")

    h5_path = DATA_DIR / "residue_embeddings" / "esmc_300m_medium5k.h5"
    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as f:
            n = len(f.keys())
        if n >= 2400:  # medium5k has ~2493 proteins
            print(f"  Already extracted ({n} proteins), skipping.")
            mark_done(results, "B1")
            return

    script = Path(__file__).resolve().parent.parent / "scripts" / "extract_esmc.py"
    print(f"  Running: uv run --with esm python {script} --dataset medium5k")
    print("  This will take ~60 min on CPU...")
    monitor()

    ret = subprocess.run(
        ["uv", "run", "--with", "esm", "python", str(script), "--dataset", "medium5k"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if ret.returncode != 0:
        print(f"  ERROR: ESM-C extraction failed (exit code {ret.returncode})")
        return

    mark_done(results, "B1")
    monitor()


def step_B2(results: dict):
    """B2: Extract ESM-C 300M for CB513."""
    print("\n═══ B2: Extract ESM-C 300M — CB513 ═══")

    h5_path = DATA_DIR / "residue_embeddings" / "esmc_300m_cb513.h5"
    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as f:
            n = len(f.keys())
        if n >= 500:
            print(f"  Already extracted ({n} proteins), skipping.")
            mark_done(results, "B2")
            return

    script = Path(__file__).resolve().parent.parent / "scripts" / "extract_esmc.py"
    print(f"  Running ESM-C extraction for CB513...")
    monitor()

    ret = subprocess.run(
        ["uv", "run", "--with", "esm", "python", str(script), "--dataset", "cb513"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if ret.returncode != 0:
        print(f"  ERROR: ESM-C CB513 extraction failed")
        return

    mark_done(results, "B2")


def step_B3(results: dict):
    """B3: Extract ESM-C 300M for SETH + TMbed + SignalP subsets."""
    print("\n═══ B3: Extract ESM-C 300M — SETH + TMbed + SignalP ═══")

    script = Path(__file__).resolve().parent.parent / "scripts" / "extract_esmc.py"
    cwd = str(Path(__file__).resolve().parent.parent)

    for dataset in ["seth", "tmbed", "signalp"]:
        h5_path = DATA_DIR / "residue_embeddings" / f"esmc_300m_{dataset}.h5"
        if h5_path.exists():
            print(f"  {dataset} already extracted, skipping.")
            continue

        print(f"  Extracting {dataset}...")
        monitor()
        ret = subprocess.run(
            ["uv", "run", "--with", "esm", "python", str(script), "--dataset", dataset],
            cwd=cwd,
        )
        if ret.returncode != 0:
            print(f"  ERROR: ESM-C {dataset} extraction failed")

    mark_done(results, "B3")


def step_B4(results: dict):
    """B4: Download ProteinGLUE PPI + EPI data."""
    print("\n═══ B4: Download ProteinGLUE Data ═══")

    proteinglue_dir = DATA_DIR / "per_residue_benchmarks" / "proteinglue"
    if proteinglue_dir.exists() and (proteinglue_dir / "ppi").exists():
        print("  ProteinGLUE data already exists, skipping.")
        mark_done(results, "B4")
        return

    print("  ProteinGLUE requires manual download from https://www.ibi.vu.nl/downloads/ProteinGLUE/")
    print("  TFRecord format needs conversion to FASTA + labels.")
    print("  Expected structure:")
    print("    data/per_residue_benchmarks/proteinglue/ppi/train_sequences.fasta")
    print("    data/per_residue_benchmarks/proteinglue/ppi/train_labels.txt")
    print("    data/per_residue_benchmarks/proteinglue/epitope/train_sequences.fasta")
    print("    data/per_residue_benchmarks/proteinglue/epitope/train_labels.txt")
    print("  Skipping for now — PPI/epitope probes will be skipped if data not present.")

    mark_done(results, "B4")


def step_B5(results: dict):
    """B5: Download eSol + Meltome datasets."""
    print("\n═══ B5: Download eSol + Meltome Datasets ═══")

    esol_dir = DATA_DIR / "per_protein_benchmarks" / "esol"
    meltome_dir = DATA_DIR / "per_protein_benchmarks" / "meltome"

    # eSol: E. coli solubility dataset
    if not esol_dir.exists():
        esol_dir.mkdir(parents=True, exist_ok=True)
        print("  eSol dataset needs manual download:")
        print("    https://www.tanpaku.org/tp-esol/")
        print("    Expected: data/per_protein_benchmarks/esol/esol_data.csv")
        print("    Columns: uniprot_id, sequence, solubility")
    else:
        print(f"  eSol directory exists: {esol_dir}")

    # Meltome: thermostability dataset
    if not meltome_dir.exists():
        meltome_dir.mkdir(parents=True, exist_ok=True)
        print("  Meltome dataset needs manual download:")
        print("    https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome")
        print("    Expected: data/per_protein_benchmarks/meltome/meltome_data.csv")
        print("    Columns: uniprot_id, sequence, melting_temp")
    else:
        print(f"  Meltome directory exists: {meltome_dir}")

    mark_done(results, "B5")


def step_B6(results: dict):
    """B6: Extract all 3 PLMs for new datasets."""
    print("\n═══ B6: Extract All PLMs for New Datasets ═══")

    print("  Note: ProtT5 and ESM2 extraction for SignalP/SETH/TMbed")
    print("  uses existing extraction infrastructure from src/extraction/")
    print("  ESM-C extraction handled by B1-B3.")
    print("  eSol/Meltome/ProteinGLUE extraction depends on B4-B5 data availability.")
    print("  Skipping — extraction for new datasets will be handled on demand.")

    # For SignalP, we need ProtT5 and ESM2 embeddings
    # These can be extracted using the existing extraction scripts
    for plm_name, _, plm_stem in PLMS[:2]:  # ProtT5 and ESM2 only
        for dataset in ["signalp"]:
            h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
            if h5_path.exists():
                print(f"  {plm_stem}_{dataset}.h5 exists")
            else:
                print(f"  NOTE: {plm_stem}_{dataset}.h5 not found — SignalP probes will be ESM-C only")

    mark_done(results, "B6")


def step_B7(results: dict):
    """B7: Per-protein retrieval (3 PLMs × cosine+euclidean)."""
    print("\n═══ B7: Per-Protein Retrieval ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    step_results = results.setdefault("results", {}).setdefault("retrieval", {})

    for plm_name, dim, plm_stem in PLMS:
        key = f"{plm_name}_family"
        if key in step_results and "cos_ret1" in step_results[key]:
            print(f"  {plm_name} retrieval already done, skipping.")
            continue

        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"  {plm_name} (dim={dim}): computing retrieval...")
        t0 = time.time()
        vectors = pool_mean(embeddings, test_ids)

        # Family retrieval — both metrics
        for metric in ["cosine", "euclidean"]:
            for label_key in ["family", "superfamily", "fold"]:
                ret = evaluate_retrieval_from_vectors(
                    vectors, metadata, label_key=label_key,
                    query_ids=test_ids, database_ids=test_ids,
                    metric=metric,
                )
                rkey = f"{plm_name}_{label_key}_{metric}"
                step_results[rkey] = {
                    "ret1": ret["precision@1"],
                    "mrr": ret["mrr"],
                    "map": ret["map"],
                    "n_queries": ret["n_queries"],
                }
                print(f"    {label_key} ({metric}): Ret@1={ret['precision@1']:.3f}, MRR={ret['mrr']:.3f}")

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        save_results(results)
        del embeddings

    mark_done(results, "B7")
    monitor()


def step_B8(results: dict):
    """B8: SCOP hierarchy evaluation (3 PLMs)."""
    print("\n═══ B8: SCOP Hierarchy Evaluation ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    step_results = results.setdefault("results", {}).setdefault("hierarchy", {})
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for plm_name, dim, plm_stem in PLMS:
        key = f"{plm_name}_cosine"
        if key in step_results:
            print(f"  {plm_name} hierarchy already done, skipping.")
            continue

        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        print(f"  {plm_name}: hierarchy distances...")
        vectors = pool_mean(embeddings, test_ids)

        for metric in ["cosine", "euclidean"]:
            hier = evaluate_hierarchy_distances(vectors, metadata, metric=metric)
            step_results[f"{plm_name}_{metric}"] = hier
            print(f"    {metric}: sep_ratio={hier.get('separation_ratio', 'N/A'):.3f}, "
                  f"ordering={'OK' if hier.get('ordering_correct') else 'FAIL'}")

        save_results(results)
        del embeddings

    mark_done(results, "B8")
    monitor()


def step_B9(results: dict):
    """B9: Per-residue probes (3 PLMs)."""
    print("\n═══ B9: Per-Residue Probes ═══")

    import random

    step_results = results.setdefault("results", {}).setdefault("per_residue", {})

    # ── SS3 / SS8 (CB513) ──
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    if cb513_path.exists():
        sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(cb513_path)

        for plm_name, dim, plm_stem in PLMS:
            ss3_key = f"{plm_name}_ss3"
            if ss3_key in step_results:
                print(f"  {plm_name} SS3/SS8 already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "cb513")
            if not embeddings:
                continue

            avail_ids = [pid for pid in ss3_labels if pid in embeddings]
            rng = random.Random(42)
            rng.shuffle(avail_ids)
            n_train = int(len(avail_ids) * 0.8)
            train_ids = avail_ids[:n_train]
            test_ids_cb = avail_ids[n_train:]

            print(f"  {plm_name} SS3 probe ({len(train_ids)} train, {len(test_ids_cb)} test)...")
            ss3 = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids_cb)
            step_results[ss3_key] = ss3
            print(f"    SS3 Q3={ss3.get('q3', 0):.3f}")

            print(f"  {plm_name} SS8 probe...")
            ss8 = evaluate_ss8_probe(embeddings, ss8_labels, train_ids, test_ids_cb)
            step_results[f"{plm_name}_ss8"] = ss8
            print(f"    SS8 Q8={ss8.get('q8', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  CB513.csv not found, skipping SS3/SS8")

    # ── Disorder (SETH/CheZOD) ──
    seth_dir = DATA_DIR / "per_residue_benchmarks"
    seth_train_fasta = seth_dir / "SETH" / "CheZOD1174_training_set_sequences.fasta"
    if seth_train_fasta.exists():
        sequences, disorder_scores, train_ids_seth, test_ids_seth = load_chezod_seth(seth_dir)

        for plm_name, dim, plm_stem in PLMS:
            dis_key = f"{plm_name}_disorder"
            if dis_key in step_results:
                print(f"  {plm_name} disorder already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "seth")
            if not embeddings:
                # Try medium5k as fallback (SETH proteins may have been extracted there)
                embeddings = load_plm_embeddings(plm_stem, "medium5k")
                if not embeddings:
                    continue

            print(f"  {plm_name} disorder probe...")
            dis = evaluate_disorder_probe(embeddings, disorder_scores, train_ids_seth, test_ids_seth)
            step_results[dis_key] = dis
            print(f"    Disorder rho={dis.get('spearman_rho', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  SETH data not found, skipping disorder")

    # ── TM Topology (TMbed) ──
    tmbed_path = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
    if tmbed_path.exists():
        tm_sequences, tm_labels = load_tmbed_annotated(tmbed_path)
        avail_tm = list(tm_sequences.keys())
        rng = random.Random(42)
        rng.shuffle(avail_tm)
        n_train = int(len(avail_tm) * 0.8)
        train_ids_tm = avail_tm[:n_train]
        test_ids_tm = avail_tm[n_train:]

        for plm_name, dim, plm_stem in PLMS:
            tm_key = f"{plm_name}_tm"
            if tm_key in step_results:
                print(f"  {plm_name} TM already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "tmbed")
            if not embeddings:
                continue

            print(f"  {plm_name} TM probe ({len(train_ids_tm)} train, {len(test_ids_tm)} test)...")
            tm = evaluate_tm_probe(embeddings, tm_labels, train_ids_tm, test_ids_tm)
            step_results[tm_key] = tm
            print(f"    TM macro_f1={tm.get('macro_f1', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  TMbed data not found, skipping TM topology")

    # ── Signal Peptide (SignalP6) ──
    signalp_path = DATA_DIR / "per_residue_benchmarks" / "SignalP6" / "train_set.fasta"
    if signalp_path.exists():
        sp_sequences, sp_labels = load_signalp6_annotated(signalp_path)
        avail_sp = list(sp_sequences.keys())
        rng = random.Random(42)
        rng.shuffle(avail_sp)
        n_train = int(len(avail_sp) * 0.8)
        train_ids_sp = avail_sp[:n_train]
        test_ids_sp = avail_sp[n_train:]

        for plm_name, dim, plm_stem in PLMS:
            sp_key = f"{plm_name}_signalp"
            if sp_key in step_results:
                print(f"  {plm_name} SignalP already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "signalp")
            if not embeddings:
                continue

            print(f"  {plm_name} SignalP probe ({len(train_ids_sp)} train, {len(test_ids_sp)} test)...")
            sp = evaluate_signalp_probe(embeddings, sp_labels, train_ids_sp, test_ids_sp)
            step_results[sp_key] = sp
            print(f"    SignalP macro_f1={sp.get('macro_f1', 0):.3f}, signal_f1={sp.get('signal_f1', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  SignalP6 data not found, skipping signal peptide")

    # ── PPI Interface (ProteinGLUE) ──
    ppi_seqs, ppi_labels, ppi_train, ppi_test = load_proteinglue_binary(
        DATA_DIR / "per_residue_benchmarks", task="ppi"
    )
    if ppi_labels:
        for plm_name, dim, plm_stem in PLMS:
            ppi_key = f"{plm_name}_ppi"
            if ppi_key in step_results:
                print(f"  {plm_name} PPI already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "proteinglue")
            if not embeddings:
                continue

            print(f"  {plm_name} PPI probe...")
            ppi = evaluate_ppi_probe(embeddings, ppi_labels, ppi_train, ppi_test)
            step_results[ppi_key] = ppi
            print(f"    PPI macro_f1={ppi.get('macro_f1', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  ProteinGLUE PPI data not found, skipping PPI")

    # ── Epitope (ProteinGLUE) ──
    epi_seqs, epi_labels, epi_train, epi_test = load_proteinglue_binary(
        DATA_DIR / "per_residue_benchmarks", task="epitope"
    )
    if epi_labels:
        for plm_name, dim, plm_stem in PLMS:
            epi_key = f"{plm_name}_epitope"
            if epi_key in step_results:
                print(f"  {plm_name} epitope already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, "proteinglue")
            if not embeddings:
                continue

            print(f"  {plm_name} epitope probe...")
            epi = evaluate_epitope_probe(embeddings, epi_labels, epi_train, epi_test)
            step_results[epi_key] = epi
            print(f"    Epitope macro_f1={epi.get('macro_f1', 0):.3f}")

            save_results(results)
            del embeddings
    else:
        print("  ProteinGLUE epitope data not found, skipping epitope")

    mark_done(results, "B9")
    monitor()


def step_B10(results: dict):
    """B10: Download SIFTS + fetch UniProt GO/EC/Pfam annotations."""
    print("\n═══ B10: Download Annotations ═══")

    metadata = load_metadata()
    scope_ids = [m["id"] for m in metadata]

    # SCOPe → UniProt mapping
    print("  Mapping SCOPe → UniProt via SIFTS...")
    scope_to_uniprot = map_scope_to_uniprot(
        scope_ids,
        sifts_path=str(DATA_DIR / "annotations" / "sifts_mapping.json"),
    )
    results.setdefault("results", {})["scope_to_uniprot_coverage"] = len(scope_to_uniprot) / len(scope_ids)

    # Fetch UniProt annotations
    uniprot_ids = list(set(scope_to_uniprot.values()))
    print(f"  Fetching annotations for {len(uniprot_ids)} unique UniProt IDs...")
    annotations = fetch_uniprot_annotations(
        uniprot_ids,
        cache_path=str(DATA_DIR / "annotations" / "uniprot_annotations.json"),
    )

    # Count coverage
    n_go = sum(1 for a in annotations.values() if a.get("go"))
    n_ec = sum(1 for a in annotations.values() if a.get("ec"))
    n_pfam = sum(1 for a in annotations.values() if a.get("pfam"))
    print(f"  GO: {n_go}/{len(annotations)}, EC: {n_ec}/{len(annotations)}, Pfam: {n_pfam}/{len(annotations)}")

    # Fetch PDB organisms for taxonomy
    pdb_ids = list(set(parse_scope_to_pdb(sid)[0] for sid in scope_ids))
    print(f"  Fetching organisms for {len(pdb_ids)} PDB IDs...")
    pdb_organisms = fetch_pdb_organisms(
        pdb_ids,
        cache_path=str(DATA_DIR / "annotations" / "pdb_organisms.json"),
    )
    print(f"  PDB organisms: {len(pdb_organisms)}/{len(pdb_ids)} mapped")

    # Load NCBI taxonomy
    taxonomy = load_ncbi_taxonomy()
    if taxonomy:
        n_mapped = sum(1 for tid in pdb_organisms.values() if tid in taxonomy)
        print(f"  Taxonomy lineages: {n_mapped}/{len(pdb_organisms)} TaxIDs found in NCBI")

    mark_done(results, "B10")
    save_results(results)


def step_B11(results: dict):
    """B11: Biological correlations (3 PLMs)."""
    print("\n═══ B11: Biological Correlations ═══")

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    # Load annotations
    scope_ids = [m["id"] for m in metadata]
    scope_to_uniprot = map_scope_to_uniprot(
        scope_ids,
        sifts_path=str(DATA_DIR / "annotations" / "sifts_mapping.json"),
    )

    annotations_cache = DATA_DIR / "annotations" / "uniprot_annotations.json"
    if not annotations_cache.exists():
        print("  Run B10 first to download annotations!")
        return

    with open(annotations_cache) as f:
        all_annotations = json.load(f)

    # Build per-protein annotation dicts (keyed by SCOPe ID)
    go_terms: dict[str, list[str]] = {}
    ec_numbers: dict[str, list[str]] = {}
    pfam_domains: dict[str, list[str]] = {}

    for sid, uniprot_id in scope_to_uniprot.items():
        ann = all_annotations.get(uniprot_id, {})
        if ann.get("go"):
            go_terms[sid] = ann["go"]
        if ann.get("ec"):
            ec_numbers[sid] = ann["ec"]
        if ann.get("pfam"):
            pfam_domains[sid] = ann["pfam"]

    print(f"  Annotations mapped: GO={len(go_terms)}, EC={len(ec_numbers)}, Pfam={len(pfam_domains)}")

    # Taxonomy
    pdb_ids = list(set(parse_scope_to_pdb(sid)[0] for sid in scope_ids))
    pdb_organisms_cache = DATA_DIR / "annotations" / "pdb_organisms.json"
    pdb_organisms: dict[str, int] = {}
    if pdb_organisms_cache.exists():
        with open(pdb_organisms_cache) as f:
            pdb_organisms = {k: int(v) for k, v in json.load(f).items()}

    ncbi_taxonomy = load_ncbi_taxonomy()

    # Build per-protein taxonomy dict (SCOPe ID → lineage)
    protein_taxonomy: dict[str, list[str]] = {}
    for sid in scope_ids:
        pdb_id, _ = parse_scope_to_pdb(sid)
        taxid = pdb_organisms.get(pdb_id.upper()) or pdb_organisms.get(pdb_id.lower())
        if taxid and taxid in ncbi_taxonomy:
            protein_taxonomy[sid] = ncbi_taxonomy[taxid]

    print(f"  Taxonomy mapped: {len(protein_taxonomy)} proteins")

    step_results = results.setdefault("results", {}).setdefault("biology", {})

    for plm_name, dim, plm_stem in PLMS:
        bio_key = f"{plm_name}_go"
        if bio_key in step_results:
            print(f"  {plm_name} biology already done, skipping.")
            continue

        embeddings = load_plm_embeddings(plm_stem, "medium5k")
        if not embeddings:
            continue

        vectors = pool_mean(embeddings, test_ids)
        print(f"  {plm_name}: biological correlations ({len(vectors)} proteins)...")

        # GO correlation
        if go_terms:
            go_res = evaluate_go_correlation(vectors, go_terms, metric="cosine")
            step_results[f"{plm_name}_go"] = go_res
            print(f"    GO Spearman rho={go_res.get('spearman_rho', 0):.3f} "
                  f"(n_pairs={go_res.get('n_pairs', 0):,})")

        # EC retrieval
        if ec_numbers:
            ec_res = evaluate_ec_retrieval(vectors, ec_numbers, metric="cosine")
            step_results[f"{plm_name}_ec"] = ec_res
            print(f"    EC full Ret@1={ec_res.get('ec_full_ret1', 0):.3f}, "
                  f"level1={ec_res.get('ec_level1_ret1', 0):.3f}")

        # Pfam retrieval
        if pfam_domains:
            pfam_res = evaluate_pfam_retrieval(vectors, pfam_domains, metric="cosine")
            step_results[f"{plm_name}_pfam"] = pfam_res
            print(f"    Pfam Ret@1={pfam_res.get('pfam_ret1', 0):.3f}")

        # Taxonomy correlation
        if protein_taxonomy:
            tax_res = evaluate_taxonomy_correlation(vectors, protein_taxonomy, metric="cosine")
            step_results[f"{plm_name}_taxonomy"] = tax_res
            print(f"    Taxonomy Spearman rho={tax_res.get('spearman_rho', 0):.3f}")

        save_results(results)
        del embeddings

    mark_done(results, "B11")
    monitor()


def step_B12(results: dict):
    """B12: Per-protein regression (solubility + thermostability)."""
    print("\n═══ B12: Per-Protein Regression ═══")

    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr as scipy_spearmanr

    step_results = results.setdefault("results", {}).setdefault("regression", {})

    for task_name, data_dir_name, label_col in [
        ("solubility", "esol", "solubility"),
        ("thermostability", "meltome", "melting_temp"),
    ]:
        data_csv = DATA_DIR / "per_protein_benchmarks" / data_dir_name / f"{data_dir_name}_data.csv"
        if not data_csv.exists():
            print(f"  {task_name}: data not found at {data_csv}, skipping.")
            continue

        # Load labels
        import csv
        labels: dict[str, float] = {}
        with open(data_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("uniprot_id", row.get("id", ""))
                try:
                    labels[pid] = float(row[label_col])
                except (ValueError, KeyError):
                    continue

        if len(labels) < 20:
            print(f"  {task_name}: too few labels ({len(labels)}), skipping.")
            continue

        print(f"  {task_name}: {len(labels)} proteins with labels")

        for plm_name, dim, plm_stem in PLMS:
            reg_key = f"{plm_name}_{task_name}"
            if reg_key in step_results:
                print(f"    {plm_name} {task_name} already done, skipping.")
                continue

            embeddings = load_plm_embeddings(plm_stem, data_dir_name)
            if not embeddings:
                continue

            # Pool and align
            import random
            avail = [pid for pid in labels if pid in embeddings]
            if len(avail) < 20:
                print(f"    {plm_name}: only {len(avail)} proteins matched, skipping.")
                del embeddings
                continue

            rng = random.Random(42)
            rng.shuffle(avail)
            n_train = int(len(avail) * 0.8)
            train = avail[:n_train]
            test = avail[n_train:]

            X_train = np.array([embeddings[pid][:MAX_LEN].mean(axis=0) for pid in train], dtype=np.float32)
            y_train = np.array([labels[pid] for pid in train], dtype=np.float32)
            X_test = np.array([embeddings[pid][:MAX_LEN].mean(axis=0) for pid in test], dtype=np.float32)
            y_test = np.array([labels[pid] for pid in test], dtype=np.float32)

            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            rho, p_val = scipy_spearmanr(y_test, y_pred)

            step_results[reg_key] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_train": len(train),
                "n_test": len(test),
            }
            print(f"    {plm_name}: rho={rho:.3f} ({len(train)} train, {len(test)} test)")

            save_results(results)
            del embeddings

    mark_done(results, "B12")
    monitor()


def step_B13(results: dict):
    """B13: Aggregator — summary table + radar plot + JSON."""
    print("\n═══ B13: Unified Aggregator ═══")

    agg = BenchmarkAggregator()
    all_results = results.get("results", {})

    # ── Retrieval ──
    retrieval = all_results.get("retrieval", {})
    for plm_name, dim, plm_stem in PLMS:
        fam = retrieval.get(f"{plm_name}_family_cosine", {})
        if fam:
            agg.add_result(plm_name, "scop_family_ret1", fam["ret1"])
        sf = retrieval.get(f"{plm_name}_superfamily_cosine", {})
        if sf:
            agg.add_result(plm_name, "scop_sf_ret1", sf["ret1"])
        fold = retrieval.get(f"{plm_name}_fold_cosine", {})
        if fold:
            agg.add_result(plm_name, "scop_fold_ret1", fold["ret1"])

    # ── Hierarchy ──
    hierarchy = all_results.get("hierarchy", {})
    for plm_name, dim, plm_stem in PLMS:
        hier = hierarchy.get(f"{plm_name}_cosine", {})
        if hier and hier.get("separation_ratio") is not None:
            agg.add_result(plm_name, "scop_separation", hier["separation_ratio"])

    # ── Per-residue ──
    per_residue = all_results.get("per_residue", {})
    for plm_name, dim, plm_stem in PLMS:
        ss3 = per_residue.get(f"{plm_name}_ss3", {})
        if ss3:
            agg.add_result(plm_name, "ss3_q3", ss3.get("q3", 0))
        ss8 = per_residue.get(f"{plm_name}_ss8", {})
        if ss8:
            agg.add_result(plm_name, "ss8_q8", ss8.get("q8", 0))
        dis = per_residue.get(f"{plm_name}_disorder", {})
        if dis:
            agg.add_result(plm_name, "disorder_rho", dis.get("spearman_rho", 0))
        tm = per_residue.get(f"{plm_name}_tm", {})
        if tm:
            agg.add_result(plm_name, "tm_f1", tm.get("macro_f1", 0))
        sp = per_residue.get(f"{plm_name}_signalp", {})
        if sp:
            agg.add_result(plm_name, "signalp_f1", sp.get("macro_f1", 0))
        ppi = per_residue.get(f"{plm_name}_ppi", {})
        if ppi:
            agg.add_result(plm_name, "ppi_f1", ppi.get("macro_f1", 0))
        epi = per_residue.get(f"{plm_name}_epitope", {})
        if epi:
            agg.add_result(plm_name, "epitope_f1", epi.get("macro_f1", 0))

    # ── Biology ──
    biology = all_results.get("biology", {})
    for plm_name, dim, plm_stem in PLMS:
        go = biology.get(f"{plm_name}_go", {})
        if go:
            agg.add_result(plm_name, "go_rho", go.get("spearman_rho", 0))
        ec = biology.get(f"{plm_name}_ec", {})
        if ec:
            agg.add_result(plm_name, "ec_ret1", ec.get("ec_full_ret1", 0))
        pfam = biology.get(f"{plm_name}_pfam", {})
        if pfam:
            agg.add_result(plm_name, "pfam_ret1", pfam.get("pfam_ret1", 0))
        tax = biology.get(f"{plm_name}_taxonomy", {})
        if tax:
            agg.add_result(plm_name, "taxonomy_rho", tax.get("spearman_rho", 0))

    # ── Regression ──
    regression = all_results.get("regression", {})
    for plm_name, dim, plm_stem in PLMS:
        sol = regression.get(f"{plm_name}_solubility", {})
        if sol:
            agg.add_result(plm_name, "solubility_rho", sol.get("spearman_rho", 0))
        thermo = regression.get(f"{plm_name}_thermostability", {})
        if thermo:
            agg.add_result(plm_name, "thermostability_rho", thermo.get("spearman_rho", 0))

    # ── Efficiency ──
    for plm_name, dim, plm_stem in PLMS:
        agg.add_result(plm_name, "embedding_dim", float(dim))

    # ── Output ──
    print("\n" + agg.summary_table())

    composites = agg.composite_score()
    print("\n  Composite scores:")
    for plm, score in sorted(composites.items(), key=lambda x: x[1], reverse=True):
        print(f"    {plm}: {score:.3f}")

    # Radar plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    radar_path = str(PLOTS_DIR / "plm_benchmark_radar.png")
    agg.plot_radar(radar_path, title="PLM Benchmark Suite — Exp 24")

    # Save aggregator JSON
    agg_path = str(DATA_DIR / "benchmarks" / "plm_benchmark_aggregator.json")
    agg.save(agg_path)

    mark_done(results, "B13")
    print("\n  B13 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 24: Comprehensive PLM Benchmark Suite",
    )
    parser.add_argument(
        "--step", type=str, default=None,
        help="Run a specific step (B1-B13)",
    )
    args = parser.parse_args()

    results = load_results()

    steps = {
        "B1": step_B1, "B2": step_B2, "B3": step_B3,
        "B4": step_B4, "B5": step_B5, "B6": step_B6,
        "B7": step_B7, "B8": step_B8, "B9": step_B9,
        "B10": step_B10, "B11": step_B11, "B12": step_B12,
        "B13": step_B13,
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
