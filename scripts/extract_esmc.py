#!/usr/bin/env python3
"""Standalone ESM-C 300M embedding extraction.

Uses the `esm` package (EvolutionaryScale), NOT `fair-esm`.
Run with: uv run --with esm python scripts/extract_esmc.py --dataset medium5k

No imports from src/ to avoid fair-esm package conflict.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root (two levels up from scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# FASTA / dataset readers (self-contained, no src imports)
# ---------------------------------------------------------------------------
def read_fasta(path):
    """Parse a FASTA file into {id: sequence} dict."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def read_annotated_fasta_triplets(path):
    """Parse annotated FASTA with header/sequence/topology triplets.

    Used for TMbed and SignalP6 formats where every three lines form:
      >header
      SEQUENCE
      TOPOLOGY_ANNOTATION
    Only the header and sequence are returned.
    """
    sequences = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            prot_id = lines[i][1:].split()[0]
            seq = lines[i + 1] if i + 1 < len(lines) else ""
            sequences[prot_id] = seq
            i += 3  # skip header, sequence, annotation
        else:
            i += 1
    return sequences


def load_dataset(name: str) -> dict[str, str]:
    """Load sequences for a named dataset. Returns {id: sequence}."""
    if name == "medium5k":
        path = PROJECT_ROOT / "data" / "proteins" / "medium_diverse_5k.fasta"
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        return read_fasta(path)

    elif name == "cb513":
        import csv

        path = PROJECT_ROOT / "data" / "per_residue_benchmarks" / "CB513.csv"
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        sequences = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                sequences[f"cb513_{idx}"] = row["input"]
        return sequences

    elif name == "seth":
        seqs = {}
        for fname in [
            "SETH/CheZOD1174_training_set_sequences.fasta",
            "SETH/CheZOD117_test_set_sequences.fasta",
        ]:
            path = PROJECT_ROOT / "data" / "per_residue_benchmarks" / fname
            if path.exists():
                seqs.update(read_fasta(path))
            else:
                print(f"WARNING: {path} not found, skipping")
        if not seqs:
            print("ERROR: No SETH FASTA files found")
            sys.exit(1)
        return seqs

    elif name == "tmbed":
        path = (
            PROJECT_ROOT
            / "data"
            / "per_residue_benchmarks"
            / "TMbed"
            / "cv_00_annotated.fasta"
        )
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        return read_annotated_fasta_triplets(path)

    elif name == "signalp":
        path = (
            PROJECT_ROOT
            / "data"
            / "per_residue_benchmarks"
            / "SignalP6"
            / "train_set.fasta"
        )
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        return read_annotated_fasta_triplets(path)

    elif name in ("esol", "meltome", "proteinglue"):
        path = (
            PROJECT_ROOT
            / "data"
            / "per_protein_benchmarks"
            / name
            / "sequences.fasta"
        )
        if not path.exists():
            print(f"WARNING: {path} not found — dataset '{name}' not available yet")
            return {}
        return read_fasta(path)

    else:
        print(f"ERROR: Unknown dataset '{name}'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------
def save_embedding(h5_path: Path, protein_id: str, embedding: np.ndarray):
    """Append one embedding to the H5 file (skip if already present)."""
    with h5py.File(h5_path, "a") as f:
        if protein_id not in f:
            f.create_dataset(
                protein_id,
                data=embedding.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )


def get_existing_ids(h5_path: Path) -> set[str]:
    """Return set of protein IDs already in the H5 file."""
    if not h5_path.exists():
        return set()
    with h5py.File(h5_path, "r") as f:
        return set(f.keys())


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 2000


def extract(dataset_name: str):
    """Extract ESM-C 300M embeddings for the given dataset."""
    # Load sequences
    print(f"Loading dataset: {dataset_name}")
    sequences = load_dataset(dataset_name)
    if not sequences:
        print("No sequences loaded — exiting.")
        return

    print(f"  Loaded {len(sequences)} sequences")

    # Output path
    out_dir = PROJECT_ROOT / "data" / "residue_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"esmc_300m_{dataset_name}.h5"

    # Check for resume
    existing = get_existing_ids(h5_path)
    if existing:
        print(f"  Resuming: {len(existing)} proteins already extracted in {h5_path}")

    todo = {pid: seq for pid, seq in sequences.items() if pid not in existing}
    if not todo:
        print("  All proteins already extracted — nothing to do.")
        return

    print(f"  Proteins to extract: {len(todo)}")

    # Load model (CPU only — MPS not supported for ESM-C)
    print("Loading ESM-C 300M model (CPU only)...")
    t0 = time.time()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    model = ESMC.from_pretrained("esmc_300m")
    model = model.to("cpu")
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Extract embeddings one at a time
    print(f"Extracting embeddings to {h5_path}")
    load1, load5, load15 = os.getloadavg()
    print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")

    t_start = time.time()
    n_done = 0
    n_skipped_long = 0
    protein_items = list(todo.items())

    for i, (pid, seq) in enumerate(protein_items):
        # Cap sequence length
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
            n_skipped_long += 1

        # Skip empty/invalid sequences
        if not seq:
            print(f"  WARNING: Empty sequence for {pid}, skipping")
            continue

        try:
            with torch.no_grad():
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(return_embeddings=True)
                )
                embeddings = logits_output.embeddings

                # Handle different possible shapes
                if isinstance(embeddings, torch.Tensor):
                    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
                else:
                    emb_np = np.array(embeddings, dtype=np.float32)

                # If shape is (1, L+2, D), squeeze batch dim
                if emb_np.ndim == 3 and emb_np.shape[0] == 1:
                    emb_np = emb_np[0]

                # Strip BOS and EOS tokens: (L+2, D) -> (L, D)
                if emb_np.shape[0] == len(seq) + 2:
                    emb_np = emb_np[1:-1]

                # Verify shape: should be (L, 960)
                if emb_np.ndim != 2:
                    print(
                        f"  WARNING: Unexpected embedding shape {emb_np.shape} for {pid}"
                    )
                    # Try to print available info for debugging
                    print(f"    logits_output type: {type(logits_output)}")
                    print(
                        f"    logits_output attrs: {[a for a in dir(logits_output) if not a.startswith('_')]}"
                    )
                    continue

            save_embedding(h5_path, pid, emb_np)
            n_done += 1

        except Exception as e:
            print(f"  ERROR extracting {pid} (len={len(seq)}): {e}")
            continue

        # Progress reporting every 50 proteins
        if (i + 1) % 50 == 0 or (i + 1) == len(protein_items):
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(protein_items) - i - 1) / rate if rate > 0 else 0
            load1, load5, load15 = os.getloadavg()
            print(
                f"  [{i + 1}/{len(protein_items)}] "
                f"{n_done} done, {elapsed:.0f}s elapsed, "
                f"{rate:.1f} prot/s, ETA {eta:.0f}s | "
                f"load: {load1:.1f}/{load5:.1f}/{load15:.1f}"
            )

    # Summary
    elapsed_total = time.time() - t_start
    print(f"\nDone! Extracted {n_done} proteins in {elapsed_total:.1f}s")
    if n_skipped_long:
        print(f"  {n_skipped_long} proteins truncated to {MAX_SEQ_LEN} residues")

    # Verify output
    with h5py.File(h5_path, "r") as f:
        total_in_file = len(f.keys())
        # Check first entry for shape
        first_key = list(f.keys())[0]
        shape = f[first_key].shape
        print(f"  Output: {h5_path}")
        print(f"  Total proteins in file: {total_in_file}")
        print(f"  Example shape ({first_key}): {shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DATASETS = [
    "medium5k",
    "cb513",
    "seth",
    "tmbed",
    "signalp",
    "esol",
    "meltome",
    "proteinglue",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM-C 300M per-residue embeddings (CPU only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run --with esm python scripts/extract_esmc.py --dataset medium5k\n"
            "  uv run --with esm python scripts/extract_esmc.py --dataset cb513\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset to extract embeddings for.",
    )
    args = parser.parse_args()

    # Force line-buffered output for subprocess visibility
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("ESM-C 300M Embedding Extraction (Standalone)")
    print("=" * 60)
    print(f"  Device: CPU (MPS not supported for ESM-C)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max seq length: {MAX_SEQ_LEN}")
    print(f"  Project root: {PROJECT_ROOT}")
    print()

    extract(args.dataset)


if __name__ == "__main__":
    main()
