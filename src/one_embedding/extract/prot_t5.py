"""ProtT5-XL per-residue embedding extraction."""
import h5py
from ._base import read_fasta


def extract(input_path, output_path, batch_size=4, device=None, max_length=2000):
    """Extract ProtT5 per-residue embeddings from FASTA.

    Args:
        input_path: FASTA file
        output_path: H5 output file
        batch_size: proteins per batch
        device: torch device (auto-detected if None)
        max_length: skip proteins longer than this
    """
    from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings

    seqs = read_fasta(input_path)
    seqs = {k: v for k, v in seqs.items() if len(v) <= max_length}

    embeddings = extract_prot_t5_embeddings(seqs, batch_size=batch_size, device=device)

    with h5py.File(output_path, "w") as f:
        for pid, emb in embeddings.items():
            f.create_dataset(pid, data=emb, compression="gzip", compression_opts=4)
