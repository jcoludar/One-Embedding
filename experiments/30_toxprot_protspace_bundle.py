#!/usr/bin/env python3
"""Experiment 30: ToxProt ProtSpace Bundle with ABTT3+rp512+int4+dct codec.

Full pipeline: extract per-residue ProtT5 → apply codec → build ProtSpace bundle.

Usage:
  uv run python experiments/30_toxprot_protspace_bundle.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py

import os
SPECIES_EMB_ROOT = Path(os.environ.get("SPECIES_EMB_ROOT", "../SpeciesEmbedding")).expanduser()
sys.path.insert(0, str(SPECIES_EMB_ROOT / "tools"))
from pipelines.protspace_pipeline import build_bundle, save_embeddings_h5

from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.transforms import dct_summary
from src.one_embedding.quantization import quantize_int4, dequantize_int4
from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOXFAM_DIR = DATA_DIR / "external_validation" / "toxfam"
OUTPUT_DIR = DATA_DIR / "protspace_bundles"
RESIDUE_H5 = TOXFAM_DIR / "residue_embeddings_prot_t5.h5"
CODEC_VECS_H5 = OUTPUT_DIR / "toxprot_codec_vecs.h5"
BUNDLE_PATH = OUTPUT_DIR / "ToxProt_ABTT3_rp512_int4.parquetbundle"


def step_S1_extract():
    """Extract per-residue ProtT5-XL embeddings."""
    if RESIDUE_H5.exists():
        with h5py.File(RESIDUE_H5, "r") as f:
            log.info(f"S1: Already extracted ({len(f.keys())} proteins), skipping.")
        return

    log.info("S1: Extracting per-residue ProtT5-XL...")
    subset = pd.read_csv(TOXFAM_DIR / "subset.csv")
    fasta_dict = dict(zip(subset["identifier"], subset["Sequence"]))
    log.info(f"  {len(fasta_dict)} proteins")

    embeddings = extract_prot_t5_embeddings(fasta_dict, batch_size=4)

    RESIDUE_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(RESIDUE_H5, "w") as f:
        for pid, emb in embeddings.items():
            f.create_dataset(pid, data=emb.astype(np.float16),
                             compression="gzip", compression_opts=4)

    log.info(f"  Saved {len(embeddings)} to {RESIDUE_H5}")


def step_S2_S3_codec():
    """Compute ABTT stats + apply full codec."""
    if CODEC_VECS_H5.exists():
        with h5py.File(CODEC_VECS_H5, "r") as f:
            log.info(f"S2-S3: Codec vectors exist ({len(f.keys())} proteins), skipping.")
        return

    log.info("S2: Computing corpus stats from training set...")
    subset = pd.read_csv(TOXFAM_DIR / "subset.csv")
    train_ids = set(subset[subset["Split"] == "train"]["identifier"])

    with h5py.File(RESIDUE_H5, "r") as f:
        all_ids = list(f.keys())
        train_embs = {pid: f[pid][:].astype(np.float32) for pid in all_ids if pid in train_ids}
        log.info(f"  {len(train_embs)} training proteins for ABTT stats")

        stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5)
        log.info(f"  Top-3 PC variance: {[f'{v:.4f}' for v in stats['explained_variance'][:3]]}")

        log.info("S3: Applying ABTT3+rp512+int4+dct codec...")
        protein_vecs = {}
        for idx, pid in enumerate(all_ids):
            m = f[pid][:].astype(np.float32)
            ma = all_but_the_top(m, stats["top_pcs"][:3])
            compressed = random_orthogonal_project(ma, d_out=512)
            deq = dequantize_int4(quantize_int4(compressed))
            protein_vecs[pid] = dct_summary(deq, K=4)

            if (idx + 1) % 1000 == 0:
                log.info(f"  {idx + 1}/{len(all_ids)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings_h5(protein_vecs, CODEC_VECS_H5)
    log.info(f"  {len(protein_vecs)} codec vectors → {CODEC_VECS_H5}")


def step_S4_bundle():
    """Build ProtSpace .parquetbundle."""
    log.info("S4: Building ProtSpace bundle...")

    subset = pd.read_csv(TOXFAM_DIR / "subset.csv")
    annotations = subset[["identifier"]].copy()
    annotations["Toxicity"] = subset["binary_label"]
    annotations["Family"] = subset["Protein families"].astype(str).str[:253]
    annotations["Split"] = subset["Split"]
    if "Organism (ID)" in subset.columns:
        annotations["Organism_ID"] = subset["Organism (ID)"].astype(str)

    build_bundle(
        annotations_df=annotations,
        h5_path=CODEC_VECS_H5,
        output_path=BUNDLE_PATH,
        column_order=["Toxicity", "Family", "Split", "Organism_ID"],
    )

    log.info(f"\nBundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024:.0f} KB)")


def main():
    log.info("=" * 60)
    log.info("Experiment 30: ToxProt ProtSpace Bundle")
    log.info("Codec: ABTT k=3 + rp512 + int4 + dct_K4")
    log.info("=" * 60)

    step_S1_extract()
    step_S2_S3_codec()
    step_S4_bundle()

    log.info("\nDONE.")


if __name__ == "__main__":
    main()
