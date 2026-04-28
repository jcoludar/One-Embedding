"""One-off: extract ESM2-650M per-residue embeddings for ToxProt.

Mirrors the ProtT5 extraction at
data/external_validation/toxfam/residue_embeddings_prot_t5.h5

Output → data/residue_embeddings/esm2_650m_toxprot.h5  (fp16, gzipped)

Run:
    uv run python experiments/extract_esm2_toxprot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.extraction.esm_extractor import extract_residue_embeddings  # noqa: E402

SUBSET_CSV = REPO / "data" / "external_validation" / "toxfam" / "subset.csv"
OUT_H5 = REPO / "data" / "residue_embeddings" / "esm2_650m_toxprot.h5"

MODEL = "esm2_t33_650M_UR50D"
MAX_LEN = 1024  # cap to avoid OOM on M3 Max; matches typical extraction protocol
BATCH_SIZE = 2  # conservative for L up to 1024


def main() -> None:
    df = pd.read_csv(SUBSET_CSV)
    print(f"Loaded {len(df)} ToxProt entries from {SUBSET_CSV.name}")

    # Filter sequences and cap length
    fasta = {}
    n_capped = 0
    for _, row in df.iterrows():
        sid = str(row["identifier"])
        seq = str(row["Sequence"]).strip()
        if not seq or any(c not in "ACDEFGHIKLMNPQRSTVWYBXZJUO" for c in seq.upper()):
            continue
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
            n_capped += 1
        fasta[sid] = seq

    print(f"  {len(fasta)} valid sequences (capped {n_capped} at L={MAX_LEN})")

    OUT_H5.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip IDs already in the H5
    done: set[str] = set()
    if OUT_H5.exists():
        with h5py.File(OUT_H5, "r") as f:
            done = set(f.keys())
        print(f"  Resume: {len(done)} already extracted; skipping")
        fasta = {sid: seq for sid, seq in fasta.items() if sid not in done}
        if not fasta:
            print("  Nothing to do.")
            return

    print(f"  Extracting {len(fasta)} sequences with {MODEL} (batch_size={BATCH_SIZE})")

    embeddings = extract_residue_embeddings(
        fasta_dict=fasta,
        model_name=MODEL,
        batch_size=BATCH_SIZE,
    )

    mode = "a" if OUT_H5.exists() else "w"
    with h5py.File(OUT_H5, mode) as f:
        for sid, emb in embeddings.items():
            if sid in f:
                continue
            f.create_dataset(sid, data=emb.astype(np.float16),
                             compression="gzip", compression_opts=4)

    with h5py.File(OUT_H5, "r") as f:
        n_total = len(f.keys())
    size_mb = OUT_H5.stat().st_size / (1024 * 1024)
    print(f"\nDone. {OUT_H5.name}: {n_total} proteins, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
