"""HDF5 read/write for variable-length per-residue tensors."""

from pathlib import Path

import h5py
import numpy as np


def save_residue_embeddings(
    embeddings: dict[str, np.ndarray],
    h5_path: Path | str,
) -> None:
    """Save per-residue embeddings to H5. Each key = protein_id, value = (L, D) float32."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        for sid, emb in embeddings.items():
            f.create_dataset(sid, data=emb.astype(np.float32), compression="gzip", compression_opts=4)
    print(f"Saved {len(embeddings)} residue embeddings to {h5_path}")


def load_residue_embeddings(h5_path: Path | str) -> dict[str, np.ndarray]:
    """Load per-residue embeddings from H5."""
    embeddings = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)
    print(f"Loaded {len(embeddings)} residue embeddings from {h5_path}")
    return embeddings


def save_compressed_embeddings(
    embeddings: dict[str, np.ndarray],
    h5_path: Path | str,
) -> None:
    """Save compressed embeddings (K, D') to H5."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        for sid, emb in embeddings.items():
            f.create_dataset(sid, data=emb.astype(np.float32), compression="gzip", compression_opts=4)
    print(f"Saved {len(embeddings)} compressed embeddings to {h5_path}")


def load_compressed_embeddings(h5_path: Path | str) -> dict[str, np.ndarray]:
    """Load compressed embeddings from H5."""
    return load_residue_embeddings(h5_path)
