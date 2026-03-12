"""H5 serialization for OneEmbedding collections."""

from pathlib import Path

import h5py
import numpy as np

from src.one_embedding.embedding import OneEmbedding


def save_one_embeddings(
    embeddings: dict[str, OneEmbedding],
    h5_path: Path | str,
) -> None:
    """Save OneEmbedding collection to H5.

    Each protein is stored as a group with two datasets:
        summary: (summary_dim,) — fixed-size protein-level vector
        residues: (L, d) — per-residue compressed matrix

    File-level attrs store format metadata.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Get common metadata from first embedding
    first = next(iter(embeddings.values()))

    with h5py.File(h5_path, "w") as f:
        f.attrs["format"] = "one_embedding"
        f.attrs["plm"] = first.plm
        f.attrs["latent_dim"] = first.latent_dim
        f.attrs["transform"] = first.transform
        f.attrs["n_proteins"] = len(embeddings)

        for pid, emb in embeddings.items():
            grp = f.create_group(pid)
            grp.create_dataset(
                "summary",
                data=emb.summary,
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset(
                "residues",
                data=emb.residues,
                compression="gzip",
                compression_opts=4,
            )
            grp.attrs["seq_len"] = emb.seq_len


def load_one_embeddings(h5_path: Path | str) -> dict[str, OneEmbedding]:
    """Load OneEmbedding collection from H5."""
    embeddings = {}

    with h5py.File(h5_path, "r") as f:
        plm = str(f.attrs["plm"])
        transform = str(f.attrs["transform"])

        for pid in f.keys():
            grp = f[pid]
            summary = np.array(grp["summary"], dtype=np.float32)
            residues = np.array(grp["residues"], dtype=np.float32)

            embeddings[pid] = OneEmbedding(
                protein_id=pid,
                plm=plm,
                latent_dim=residues.shape[1],
                seq_len=residues.shape[0],
                transform=transform,
                _summary=summary,
                _residues=residues,
            )

    return embeddings
