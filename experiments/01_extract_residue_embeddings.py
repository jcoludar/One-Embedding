#!/usr/bin/env python3
"""Phase 1.1-1.2: Curate tiny protein set and extract per-residue embeddings.

Downloads SCOPe ASTRAL sequences, curates a diverse 100-protein set,
and extracts ESM2-8M per-residue embeddings.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ssl
import urllib.request
import gzip
import shutil

from src.extraction.data_loader import (
    curate_scope_set,
    write_fasta,
    save_metadata_csv,
    read_fasta,
)
from src.extraction.esm_extractor import extract_residue_embeddings
from src.utils.h5_store import save_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROTEINS_DIR = DATA_DIR / "proteins"
EMBEDDINGS_DIR = DATA_DIR / "residue_embeddings"

# SCOPe ASTRAL 40% identity representatives
SCOPE_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"


def download_scope():
    """Download SCOPe ASTRAL sequences if not present."""
    fasta_path = PROTEINS_DIR / "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    if fasta_path.exists():
        print(f"SCOPe FASTA already exists: {fasta_path}")
        return fasta_path

    PROTEINS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SCOPe ASTRAL sequences...")

    # Try direct URL first, then gzipped
    for url in [SCOPE_URL, SCOPE_URL + ".gz"]:
        try:
            print(f"  Trying: {url}")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=60, context=ctx)

            if url.endswith(".gz"):
                gz_path = fasta_path.with_suffix(".fa.gz")
                with open(gz_path, "wb") as f:
                    shutil.copyfileobj(response, f)
                with gzip.open(gz_path, "rt") as gz, open(fasta_path, "w") as out:
                    out.write(gz.read())
                gz_path.unlink()
            else:
                with open(fasta_path, "wb") as f:
                    shutil.copyfileobj(response, f)

            print(f"  Downloaded to: {fasta_path}")
            return fasta_path
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("ERROR: Could not download SCOPe sequences.")
    print("Please manually download from https://scop.berkeley.edu/downloads/scopeseq-2.08/")
    print(f"and place at: {fasta_path}")
    sys.exit(1)


def main():
    # Step 1: Download SCOPe
    scope_fasta = download_scope()

    # Step 2: Curate diverse 100-protein set
    print("\nCurating diverse 100-protein set...")
    fasta_dict, metadata = curate_scope_set(
        scope_fasta,
        n_proteins=100,
        min_length=50,
        max_length=500,
        n_per_family=5,
        seed=42,
    )
    print(f"  Selected {len(fasta_dict)} proteins from {len(set(m['family'] for m in metadata))} families")

    # Save curated set
    fasta_path = PROTEINS_DIR / "tiny_diverse_100.fasta"
    csv_path = PROTEINS_DIR / "metadata.csv"
    write_fasta(fasta_dict, fasta_path)
    save_metadata_csv(metadata, csv_path)
    print(f"  Saved FASTA: {fasta_path}")
    print(f"  Saved metadata: {csv_path}")

    # Print distribution
    from collections import Counter
    class_dist = Counter(m["class_label"] for m in metadata)
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    length_stats = [m["length"] for m in metadata]
    print(f"\n  Length range: {min(length_stats)}-{max(length_stats)}")
    print(f"  Mean length: {sum(length_stats)/len(length_stats):.0f}")

    # Step 3: Extract ESM2-8M per-residue embeddings
    print("\nExtracting ESM2-8M per-residue embeddings...")
    embeddings = extract_residue_embeddings(
        fasta_dict,
        model_name="esm2_t6_8M_UR50D",
        batch_size=8,
    )

    # Verify shapes
    for pid, emb in list(embeddings.items())[:3]:
        print(f"  {pid}: shape={emb.shape}, dtype={emb.dtype}")

    # Save
    h5_path = EMBEDDINGS_DIR / "esm2_8m_tiny100.h5"
    save_residue_embeddings(embeddings, h5_path)

    print(f"\nDone! {len(embeddings)} proteins embedded.")
    print(f"  FASTA: {fasta_path}")
    print(f"  Metadata: {csv_path}")
    print(f"  Embeddings: {h5_path}")


if __name__ == "__main__":
    main()
