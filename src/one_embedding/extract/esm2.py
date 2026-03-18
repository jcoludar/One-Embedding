"""ESM2 per-residue embedding extraction."""
import h5py
from ._base import read_fasta


def extract(
    input_path,
    output_path,
    model_name="esm2_t33_650M_UR50D",
    batch_size=8,
    device=None,
    max_length=2000,
):
    """Extract ESM2 per-residue embeddings from FASTA.

    Args:
        input_path: FASTA file
        output_path: H5 output file
        model_name: ESM2 model variant (e.g. 'esm2_t33_650M_UR50D', 'esm2_t6_8M_UR50D')
        batch_size: proteins per batch
        device: torch device (auto-detected if None)
        max_length: skip proteins longer than this
    """
    from src.extraction.esm_extractor import extract_residue_embeddings

    seqs = read_fasta(input_path)
    seqs = {k: v for k, v in seqs.items() if len(v) <= max_length}

    embeddings = extract_residue_embeddings(
        seqs, model_name=model_name, batch_size=batch_size, device=device
    )

    with h5py.File(output_path, "w") as f:
        for pid, emb in embeddings.items():
            f.create_dataset(pid, data=emb, compression="gzip", compression_opts=4)
