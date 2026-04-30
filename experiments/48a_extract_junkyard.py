"""Extract ProtT5 embeddings for residue-shuffled junkyard copies of CB513.

For each CB513 sequence, generate n_shuffles random permutations of its
residues, extract ProtT5-XL per-residue embeddings, and save to H5 keyed as
``cb513_<i>_shuf<j>``. This is the reference data for Exp 48 (RNS under
compression).

Usage:
    uv run python experiments/48a_extract_junkyard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
from src.one_embedding.rns import generate_junkyard_sequences


CB513_CSV = ROOT / "data" / "per_residue_benchmarks" / "CB513.csv"
OUT_H5 = ROOT / "data" / "residue_embeddings" / "prot_t5_xl_cb513_junkyard.h5"
N_SHUFFLES = 5
SEED = 42


def load_cb513_sequences() -> dict[str, str]:
    """Load CB513 sequences keyed to match prot_t5_xl_cb513.h5 (cb513_0..510)."""
    df = pd.read_csv(CB513_CSV)
    return {f"cb513_{i}": seq for i, seq in enumerate(df["input"].tolist())}


def main() -> None:
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)

    if OUT_H5.exists():
        with h5py.File(OUT_H5, "r") as f:
            n_existing = len(f.keys())
        print(f"Output already exists with {n_existing} entries: {OUT_H5}")
        print("Delete it first to re-extract.")
        return

    real = load_cb513_sequences()
    print(f"Loaded {len(real)} CB513 sequences")

    junk = generate_junkyard_sequences(real, n_shuffles=N_SHUFFLES, seed=SEED)
    print(f"Generated {len(junk)} junkyard sequences "
          f"({len(real)} proteins x {N_SHUFFLES} shuffles)")

    embeddings = extract_prot_t5_embeddings(junk, batch_size=4)

    print(f"Writing {len(embeddings)} embeddings to {OUT_H5}")
    with h5py.File(OUT_H5, "w") as f:
        for pid, arr in embeddings.items():
            f.create_dataset(pid, data=arr, compression="gzip")
        f.attrs["source"] = "cb513"
        f.attrs["n_real"] = len(real)
        f.attrs["n_shuffles"] = N_SHUFFLES
        f.attrs["seed"] = SEED

    print("Done.")


if __name__ == "__main__":
    main()
