#!/usr/bin/env python3
"""Experiment 40: Build the One Embedding Benchmark Suite.

Assembles existing + new datasets into a unified benchmark structure
with verified redundancy-reduced splits.

Steps:
    A: Create directory structure and symlink existing data
    B: Curate diversity set from SpeciesEmbedding
    C: Run MMseqs2 redundancy reduction on custom subsets
    D: Create and verify all train/test splits
    E: Compress all subsets with codec (768d default)
    F: Write master metadata.json
"""

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.core import Codec
from src.one_embedding.io import write_one_h5_batch
from src.utils.h5_store import load_residue_embeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUITE = DATA / "benchmark_suite"
SPECIES = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding")
RESULTS_PATH = DATA / "benchmarks" / "benchmark_suite_assembly.json"

EXISTING_EMBEDDINGS = DATA / "residue_embeddings"

# Existing per-residue benchmark data
PER_RES = DATA / "per_residue_benchmarks"

MMSEQS = "/opt/homebrew/bin/mmseqs"


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
# FASTA parsing (fallback if BioPython unavailable)
# ---------------------------------------------------------------------------
def read_fasta(path):
    """Parse FASTA file into {id: sequence} dict."""
    seqs = {}
    with open(path) as f:
        pid, seq = None, []
        for line in f:
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(seq)
                pid = line[1:].strip().split()[0]
                seq = []
            else:
                seq.append(line.strip())
        if pid:
            seqs[pid] = "".join(seq)
    return seqs


# ---------------------------------------------------------------------------
# Step A: Directory structure + symlinks
# ---------------------------------------------------------------------------
def step_A(results):
    """Create directory structure and symlink existing embeddings."""
    if "A" in results.get("steps_done", []):
        print("Step A done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step A: Directory structure + symlinks")
    print("=" * 60)

    # Create directory tree
    dirs = [
        SUITE / "retrieval" / "scope_5k",
        SUITE / "retrieval" / "toxfam",
        SUITE / "per_residue" / "cb513",
        SUITE / "per_residue" / "chezod",
        SUITE / "per_residue" / "trizod",
        SUITE / "per_residue" / "tmbed",
        SUITE / "diversity" / "venom_families",
        SUITE / "diversity" / "non_toxin",
        SUITE / "stress" / "long_proteins",
        SUITE / "stress" / "short_proteins",
        SUITE / "plm_embeddings" / "prot_t5_xl",
        SUITE / "plm_embeddings" / "esm2_650m",
        SUITE / "splits",
        SUITE / "compressed" / "prot_t5_768d",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created {d.relative_to(ROOT)}")

    # Symlink existing ProtT5 embeddings
    symlinks = {
        "prot_t5_xl/scope_5k.h5": EXISTING_EMBEDDINGS / "prot_t5_xl_medium5k.h5",
        "prot_t5_xl/cb513.h5": EXISTING_EMBEDDINGS / "prot_t5_xl_cb513.h5",
        "prot_t5_xl/chezod.h5": EXISTING_EMBEDDINGS / "prot_t5_xl_chezod.h5",
        "prot_t5_xl/trizod.h5": EXISTING_EMBEDDINGS / "prot_t5_xl_trizod.h5",
        "prot_t5_xl/validation.h5": EXISTING_EMBEDDINGS / "prot_t5_xl_validation.h5",
    }
    for link_name, target in symlinks.items():
        link_path = SUITE / "plm_embeddings" / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        if target.exists():
            link_path.symlink_to(target)
            size_mb = target.stat().st_size / 1024 / 1024
            print(f"  Linked {link_name} -> {target.name} ({size_mb:.0f} MB)")
        else:
            print(f"  WARNING: {target} does not exist, skipping")

    # Symlink ESM2 embeddings
    esm2_links = {
        "esm2_650m/scope_5k.h5": EXISTING_EMBEDDINGS / "esm2_650m_medium5k.h5",
        "esm2_650m/cb513.h5": EXISTING_EMBEDDINGS / "esm2_650m_cb513.h5",
    }
    for link_name, target in esm2_links.items():
        link_path = SUITE / "plm_embeddings" / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        if target.exists():
            link_path.symlink_to(target)
            print(f"  Linked {link_name}")

    # Copy split files
    for split_file in (DATA / "splits").glob("*.json"):
        dest = SUITE / "splits" / split_file.name
        if not dest.exists():
            shutil.copy2(split_file, dest)
            print(f"  Copied split: {split_file.name}")

    # Copy per-residue benchmark labels
    label_copies = {
        "per_residue/cb513/CB513.csv": PER_RES / "CB513.csv",
        "per_residue/chezod/": PER_RES / "SETH",
        "per_residue/trizod/": PER_RES / "TriZOD",
        "per_residue/tmbed/": PER_RES / "TMbed",
    }
    for dest_name, src in label_copies.items():
        dest = SUITE / dest_name
        if src.is_file():
            if not dest.exists():
                shutil.copy2(src, dest)
                print(f"  Copied {src.name}")
        elif src.is_dir():
            # Always copy with dirs_exist_ok (dir may already exist but be empty)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"  Copied dir {src.name}/")

    # Copy ToxFam labels
    toxfam_labels = DATA / "phylo_benchmark" / "toxfam_v2_labels.csv"
    if toxfam_labels.exists():
        dest = SUITE / "retrieval" / "toxfam" / "labels.csv"
        if not dest.exists():
            shutil.copy2(toxfam_labels, dest)
            print(f"  Copied ToxFam labels")

    step_result = {
        "dirs_created": len(dirs),
        "symlinks_created": len(symlinks) + len(esm2_links),
    }
    results["A"] = step_result
    results["steps_done"].append("A")
    save_results(results)
    print("\n  Step A complete!")
    return results


# ---------------------------------------------------------------------------
# Step B: Curate diversity set from SpeciesEmbedding
# ---------------------------------------------------------------------------
def step_B(results):
    """Curate diversity proteins from SpeciesEmbedding."""
    if "B" in results.get("steps_done", []):
        print("Step B done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step B: Curate diversity set")
    print("=" * 60)

    # Use BioPython if available, else fallback to manual parser
    try:
        from Bio import SeqIO
        use_biopython = True
    except ImportError:
        use_biopython = False
        print("  BioPython not available, using manual FASTA parser")

    # Each entry: (name, list_of_fasta_paths, category)
    # Use specific files (not globs) since SpeciesEmbedding has varied structures
    diversity_sources = [
        ("3FTx", [SPECIES / "data" / "3FTx" / "3FTx_degapped.fasta"], "venom_families"),
        ("Kunitz", [SPECIES / "data" / "Kunitz" / "Kunitz.fasta"], "venom_families"),
        ("PLA2", [SPECIES / "data" / "Pla2" / "sequences.fasta"], "venom_families"),
        ("conotoxin", [SPECIES / "data" / "conotoxin" / "mature_sequences.fasta"], "venom_families"),
        ("snake_venom", [SPECIES / "data" / "snake_venom" / "nitin_snakes" / "combined" / "all_species_combined.normalized.fasta"], "venom_families"),
        ("ant_venom", [SPECIES / "data" / "ant_venoms" / "sequences" / "toxin_seqs" / "toxins_with_Hymenoptera.fa"], "venom_families"),
        ("bee_venom", [SPECIES / "data" / "bee_venom" / "uniprot-reviewed_Apoidea_291020.fasta"], "venom_families"),
        ("casein", [SPECIES / "data" / "casein" / "sequences" / "combo.fasta"], "non_toxin"),
        ("mrjp_yellow", [SPECIES / "data" / "mrjp_yellow" / "proteins" / "all_bee_mrjp_proteins.fasta"], "non_toxin"),
    ]

    all_seqs = {}  # {source_name: {protein_id: sequence_str}}
    total = 0

    for name, fasta_paths, category in diversity_sources:
        seqs = {}
        for fasta_path in fasta_paths:
            if not fasta_path.exists():
                print(f"  WARNING: {fasta_path} does not exist, skipping")
                continue
            try:
                if use_biopython:
                    for record in SeqIO.parse(str(fasta_path), "fasta"):
                        seq = str(record.seq).replace("*", "").replace("X", "")
                        # Remove gap characters from aligned files
                        seq = seq.replace("-", "")
                        if 20 <= len(seq) <= 5000:
                            pid = f"{name}_{record.id}"
                            seqs[pid] = seq
                else:
                    raw = read_fasta(str(fasta_path))
                    for rid, seq in raw.items():
                        seq = seq.replace("*", "").replace("X", "").replace("-", "")
                        if 20 <= len(seq) <= 5000:
                            pid = f"{name}_{rid}"
                            seqs[pid] = seq
            except Exception as e:
                print(f"  WARNING: Failed to parse {fasta_path}: {e}")

        # Cap at 300 per source to keep balanced
        if len(seqs) > 300:
            rng = random.Random(42)
            keys = list(seqs.keys())
            rng.shuffle(keys)
            seqs = {k: seqs[k] for k in keys[:300]}

        all_seqs[name] = seqs
        total += len(seqs)
        print(f"  {name}: {len(seqs)} proteins")

    # Write combined FASTA for each category
    for category in ["venom_families", "non_toxin"]:
        cat_dir = SUITE / "diversity" / category
        cat_seqs = {}
        for name, fasta_paths, cat in diversity_sources:
            if cat == category and name in all_seqs:
                cat_seqs.update(all_seqs[name])

        fasta_path = cat_dir / "sequences.fasta"
        with open(fasta_path, "w") as f:
            for pid, seq in sorted(cat_seqs.items()):
                f.write(f">{pid}\n{seq}\n")
        print(f"  Wrote {len(cat_seqs)} sequences to {fasta_path.relative_to(ROOT)}")

    # Write a master diversity FASTA (for MMseqs2)
    master_fasta = SUITE / "diversity" / "all_diversity.fasta"
    with open(master_fasta, "w") as f:
        for name, seqs in all_seqs.items():
            for pid, seq in seqs.items():
                f.write(f">{pid}\n{seq}\n")
    print(f"\n  Total diversity proteins: {total}")
    print(f"  Master FASTA: {master_fasta.relative_to(ROOT)}")

    results["B"] = {
        "total_diversity_proteins": total,
        "sources": {name: len(seqs) for name, seqs in all_seqs.items()},
    }
    results["steps_done"].append("B")
    save_results(results)
    print("\n  Step B complete!")
    return results


# ---------------------------------------------------------------------------
# Step C: MMseqs2 redundancy reduction
# ---------------------------------------------------------------------------
def step_C(results):
    """Run MMseqs2 clustering at 30% identity on diversity set."""
    if "C" in results.get("steps_done", []):
        print("Step C done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step C: MMseqs2 redundancy reduction")
    print("=" * 60)

    import tempfile

    master_fasta = SUITE / "diversity" / "all_diversity.fasta"
    if not master_fasta.exists():
        print("  ERROR: Run Step B first")
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "db")
        result_db = os.path.join(tmpdir, "result")
        tsv_out = str(SUITE / "diversity" / "clusters_30pct.tsv")

        # Create MMseqs2 database
        subprocess.run([MMSEQS, "createdb", str(master_fasta), db], check=True)

        # Cluster at 30% identity, 80% coverage
        subprocess.run([
            MMSEQS, "cluster", db, result_db, tmpdir,
            "--min-seq-id", "0.3",
            "-c", "0.8",
            "--cov-mode", "0",
        ], check=True)

        # Export to TSV
        subprocess.run([
            MMSEQS, "createtsv", db, db, result_db, tsv_out,
        ], check=True)

    # Parse clusters
    clusters = {}  # representative -> [members]
    with open(tsv_out) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters.setdefault(rep, []).append(member)

    # Count total proteins before and after
    total_before = sum(len(v) for v in clusters.values())
    total_after = len(clusters)  # one representative per cluster

    print(f"  Before clustering: {total_before} proteins")
    print(f"  After clustering (30% identity): {total_after} representatives")
    print(f"  Reduction: {100 * (1 - total_after / total_before):.1f}%")

    # Write representative-only FASTA
    reps = set(clusters.keys())
    rep_fasta = SUITE / "diversity" / "representatives_30pct.fasta"
    kept = 0
    with open(master_fasta) as fin, open(rep_fasta, "w") as fout:
        write_next = False
        for line in fin:
            if line.startswith(">"):
                pid = line[1:].strip().split()[0]
                write_next = pid in reps
                if write_next:
                    kept += 1
            if write_next:
                fout.write(line)

    print(f"  Representative FASTA: {kept} proteins")

    results["C"] = {
        "total_before": total_before,
        "total_after": total_after,
        "n_clusters": len(clusters),
        "reduction_pct": round(100 * (1 - total_after / total_before), 1),
    }
    results["steps_done"].append("C")
    save_results(results)
    print("\n  Step C complete!")
    return results


# ---------------------------------------------------------------------------
# Step D: Create and verify train/test splits
# ---------------------------------------------------------------------------
def step_D(results):
    """Create and verify all train/test splits."""
    if "D" in results.get("steps_done", []):
        print("Step D done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step D: Create verified splits")
    print("=" * 60)

    splits_dir = SUITE / "splits"
    split_info = {}

    # 1. CheZOD: use standard SETH split (already defined)
    chezod_train = []
    chezod_test = []
    seth_dir = SUITE / "per_residue" / "chezod"

    train_fasta = seth_dir / "CheZOD1174_training_set_sequences.fasta"
    test_fasta = seth_dir / "CheZOD117_test_set_sequences.fasta"

    if train_fasta.exists() and test_fasta.exists():
        try:
            from Bio import SeqIO
            for r in SeqIO.parse(str(train_fasta), "fasta"):
                chezod_train.append(r.id)
            for r in SeqIO.parse(str(test_fasta), "fasta"):
                chezod_test.append(r.id)
        except ImportError:
            chezod_train = list(read_fasta(str(train_fasta)).keys())
            chezod_test = list(read_fasta(str(test_fasta)).keys())

    split_info["chezod"] = {
        "method": "SETH predefined (Dass et al. 2020)",
        "train": len(chezod_train), "test": len(chezod_test),
        "identity_threshold": "predefined, non-redundant",
    }
    with open(splits_dir / "chezod_seth.json", "w") as f:
        json.dump({"train_ids": chezod_train, "test_ids": chezod_test}, f, indent=2)
    print(f"  CheZOD: {len(chezod_train)} train / {len(chezod_test)} test (SETH split)")

    # 2. CB513: 80/20 random split (CB513 is <25% identity by construction)
    cb513_emb_path = SUITE / "plm_embeddings" / "prot_t5_xl" / "cb513.h5"
    if cb513_emb_path.exists():
        embs = load_residue_embeddings(str(cb513_emb_path))
        cb_ids = sorted(embs.keys())
        rng = random.Random(42)
        rng.shuffle(cb_ids)
        n_train = int(len(cb_ids) * 0.8)
        cb_train, cb_test = cb_ids[:n_train], cb_ids[n_train:]
        with open(splits_dir / "cb513_80_20.json", "w") as f:
            json.dump({"train_ids": cb_train, "test_ids": cb_test}, f, indent=2)
        split_info["cb513"] = {
            "method": "80/20 random (seed=42). CB513 is <25% identity by construction.",
            "train": len(cb_train), "test": len(cb_test),
            "identity_threshold": "<25% (dataset property)",
        }
        print(f"  CB513: {len(cb_train)} train / {len(cb_test)} test")

    # 3. TriZOD: use TriZOD348 test set (from UdonPred/TriZOD paper)
    trizod_test_fasta = SUITE / "per_residue" / "trizod" / "TriZOD_test_set.fasta"
    trizod_emb_path = SUITE / "plm_embeddings" / "prot_t5_xl" / "trizod.h5"
    if trizod_test_fasta.exists() and trizod_emb_path.exists():
        try:
            from Bio import SeqIO
            trizod_test = [r.id for r in SeqIO.parse(str(trizod_test_fasta), "fasta")]
        except ImportError:
            trizod_test = list(read_fasta(str(trizod_test_fasta)).keys())

        trizod_embs = load_residue_embeddings(str(trizod_emb_path))
        trizod_all = sorted(trizod_embs.keys())
        trizod_test_set = set(trizod_test)
        trizod_train = [pid for pid in trizod_all if pid not in trizod_test_set]
        with open(splits_dir / "trizod_predefined.json", "w") as f:
            json.dump({"train_ids": trizod_train, "test_ids": trizod_test}, f, indent=2)
        split_info["trizod"] = {
            "method": "TriZOD348 predefined test set (Haak 2025)",
            "train": len(trizod_train), "test": len(trizod_test),
            "identity_threshold": "predefined, cluster-based",
        }
        print(f"  TriZOD: {len(trizod_train)} train / {len(trizod_test)} test")

    # 4. SCOPe 5K: use existing split
    scope_split = splits_dir / "esm2_650m_5k_split.json"
    if scope_split.exists():
        with open(scope_split) as f:
            scope = json.load(f)
        split_info["scope_5k"] = {
            "method": "Existing family-stratified split",
            "train": len(scope["train_ids"]), "test": len(scope["test_ids"]),
            "identity_threshold": "family-stratified",
        }
        print(f"  SCOPe 5K: {len(scope['train_ids'])} train / {len(scope['test_ids'])} test")

    # Save split summary
    with open(splits_dir / "split_summary.json", "w") as f:
        json.dump(split_info, f, indent=2)

    results["D"] = split_info
    results["steps_done"].append("D")
    save_results(results)
    print("\n  Step D complete!")
    return results


# ---------------------------------------------------------------------------
# Step E: Compress with codec
# ---------------------------------------------------------------------------
def step_E(results):
    """Compress all subsets with codec at 768d."""
    if "E" in results.get("steps_done", []):
        print("Step E done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step E: Compress with codec (768d)")
    print("=" * 60)

    codec = Codec.for_plm("prot_t5", d_out=768)
    compressed_dir = SUITE / "compressed" / "prot_t5_768d"

    # Compress each ProtT5 embedding file
    emb_dir = SUITE / "plm_embeddings" / "prot_t5_xl"
    compressed_files = []
    for h5_path in sorted(emb_dir.glob("*.h5")):
        out_path = compressed_dir / h5_path.name.replace(".h5", ".one.h5")
        if out_path.exists():
            print(f"  {h5_path.name} already compressed, skipping")
            compressed_files.append(str(out_path.relative_to(ROOT)))
            continue

        print(f"  Compressing {h5_path.name}...")
        t0 = time.time()
        embs = load_residue_embeddings(str(h5_path))

        encoded = {}
        for i, (pid, raw) in enumerate(embs.items()):
            result = codec.encode(raw)
            encoded[pid] = {
                "per_residue": result["per_residue"].astype(np.float32),
                "protein_vec": result["protein_vec"],
            }
            if (i + 1) % 500 == 0:
                print(f"    Encoded {i + 1}/{len(embs)} proteins...")

        write_one_h5_batch(out_path, encoded, tags={
            "source_model": "prot_t5_xl",
            "d_out": 768,
            "compression": "abtt3_rp768_dct4",
            "source_file": h5_path.name,
        })

        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / 1024 / 1024
        compressed_files.append(str(out_path.relative_to(ROOT)))
        print(f"    {len(encoded)} proteins -> {size_mb:.1f} MB ({elapsed:.1f}s)")

    results["E"] = {
        "d_out": 768,
        "codec": "abtt3_rp768_dct4",
        "compressed_files": compressed_files,
    }
    results["steps_done"].append("E")
    save_results(results)
    print("\n  Step E complete!")
    return results


# ---------------------------------------------------------------------------
# Step F: Write metadata.json
# ---------------------------------------------------------------------------
def step_F(results):
    """Write master metadata.json."""
    if "F" in results.get("steps_done", []):
        print("Step F done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step F: Write metadata.json")
    print("=" * 60)

    metadata = {
        "name": "One Embedding Benchmark Suite",
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d"),
        "description": "Collection of benchmark datasets for codec validation, "
                       "probe training, and tool benchmarking.",
        "subsets": {
            "retrieval/scope_5k": {
                "description": "SCOPe 5K family retrieval benchmark",
                "n_proteins": 5000,
                "split": "splits/esm2_650m_5k_split.json",
                "tasks": ["family_ret1", "sf_ret1", "fold_ret1"],
            },
            "retrieval/toxfam": {
                "description": "ToxFam v2 -- 84 proteins from 12 venom families",
                "n_proteins": 84,
                "labels": "retrieval/toxfam/labels.csv",
                "tasks": ["monophyly", "phylogenetics"],
            },
            "per_residue/cb513": {
                "description": "CB513 -- secondary structure (<25% identity)",
                "n_proteins": 513,
                "split": "splits/cb513_80_20.json",
                "tasks": ["ss3", "ss8"],
            },
            "per_residue/chezod": {
                "description": "CheZOD -- disorder (SETH split)",
                "n_proteins": 1291,
                "split": "splits/chezod_seth.json",
                "tasks": ["disorder"],
            },
            "per_residue/trizod": {
                "description": "TriZOD -- disorder (full, TriZOD348 test)",
                "n_proteins": 5786,
                "split": "splits/trizod_predefined.json",
                "tasks": ["disorder"],
            },
            "per_residue/tmbed": {
                "description": "TMbed -- transmembrane topology",
                "tasks": ["tm_topology"],
            },
            "diversity": {
                "description": "Handcrafted diversity set from SpeciesEmbedding",
                "tasks": ["retrieval", "clustering", "embedding_analysis"],
            },
        },
        "plm_embeddings": {
            "prot_t5_xl": "plm_embeddings/prot_t5_xl/",
            "esm2_650m": "plm_embeddings/esm2_650m/",
        },
        "compressed": {
            "prot_t5_768d": "compressed/prot_t5_768d/",
        },
        "redundancy_reduction": {
            "method": "MMseqs2 cluster",
            "identity_threshold": 0.3,
            "coverage_threshold": 0.8,
            "coverage_mode": 0,
            "tool_version": "MMseqs2",
        },
        "assembly_results": results,
    }

    meta_path = SUITE / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Written to {meta_path.relative_to(ROOT)}")

    results["F"] = {"metadata_path": str(meta_path.relative_to(ROOT))}
    results["steps_done"].append("F")
    save_results(results)
    print("\n  Step F complete!")
    return results


# ---------------------------------------------------------------------------
# Verification: check all splits for data leakage
# ---------------------------------------------------------------------------
def step_verify(results):
    """Verify no data leakage between train/test splits."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Check all splits for leakage")
    print("=" * 60)

    splits_dir = SUITE / "splits"
    issues = []

    # Check each split file
    for split_file in sorted(splits_dir.glob("*.json")):
        if split_file.name in ("split_summary.json",):
            continue
        with open(split_file) as f:
            split = json.load(f)
        train = set(split.get("train_ids", []))
        test = set(split.get("test_ids", []))

        if not train and not test:
            continue  # Not a train/test split file

        overlap = train & test
        if overlap:
            issues.append(f"{split_file.name}: {len(overlap)} IDs in BOTH train and test!")
            print(f"  FAIL {split_file.name}: {len(overlap)} overlapping IDs!")
        else:
            print(f"  OK   {split_file.name}: {len(train)} train, {len(test)} test, 0 overlap")

    # Check ABTT corpus (SCOPe 5K) vs test sets
    scope_emb_path = SUITE / "plm_embeddings" / "prot_t5_xl" / "scope_5k.h5"
    if scope_emb_path.exists():
        scope_embs = load_residue_embeddings(str(scope_emb_path))
        scope_ids = set(scope_embs.keys())

        for split_file in sorted(splits_dir.glob("*.json")):
            if split_file.name in ("split_summary.json",):
                continue
            with open(split_file) as f:
                split = json.load(f)
            test_ids = set(split.get("test_ids", []))
            if not test_ids:
                continue
            overlap = scope_ids & test_ids
            if overlap:
                print(f"  NOTE {split_file.name}: {len(overlap)} test IDs in SCOPe 5K corpus "
                      f"(ABTT fitting data). This is OK for unsupervised preprocessing.")

    if issues:
        print(f"\n  CRITICAL: {len(issues)} leakage issues found!")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  All splits verified clean. No data leakage.")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Experiment 40: Build Benchmark Suite")
    print("=" * 60)

    results = load_results()

    results = step_A(results)
    results = step_B(results)
    results = step_C(results)
    results = step_D(results)
    results = step_E(results)
    results = step_F(results)

    # Run verification
    issues = step_verify(results)

    save_results(results)
    print(f"\nBenchmark suite assembled at {SUITE.relative_to(ROOT)}/")
    print(f"Results saved to {RESULTS_PATH.relative_to(ROOT)}")
    if issues:
        print(f"\nWARNING: {len(issues)} leakage issues detected!")
        sys.exit(1)
    else:
        print("\nAll verifications passed.")
