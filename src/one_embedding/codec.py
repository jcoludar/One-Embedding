"""One Embedding Codec: self-contained save/load for compressed protein embeddings.

Encodes raw PLM per-residue embeddings into a single file containing both
per-residue and protein-level representations, with metadata for reproducibility.

Usage:
    # Encode
    codec = OneEmbeddingCodec(d_out=512, dct_k=4)
    codec.encode_h5(raw_embeddings_h5, "output_dir/")

    # Decode (receiver side — no knowledge of codec internals needed)
    emb = OneEmbeddingCodec.load("output_dir/protein_id.h5")
    emb["per_residue"]   # (L, 512) for per-residue probes
    emb["protein_vec"]   # (2048,) for retrieval/UMAP/clustering
    emb["metadata"]      # dict with codec params
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from src.one_embedding.transforms import dct_summary
from src.one_embedding.universal_transforms import random_orthogonal_project


class OneEmbeddingCodec:
    """Training-free codec: random projection + DCT pooling.

    Args:
        d_out: Output dimensionality for per-residue (default 512).
        dct_k: Number of DCT coefficients for protein vector (default 4).
        seed: Fixed seed for projection matrix (default 42).
        dtype: Storage dtype — "float16" (default, 25% of raw) or "float32".
               Computation is always float32; dtype only affects save/load.
    """

    def __init__(
        self,
        d_out: int = 512,
        dct_k: int = 4,
        seed: int = 42,
        dtype: str = "float16",
    ):
        self.d_out = d_out
        self.dct_k = dct_k
        self.seed = seed
        self.dtype = np.dtype(dtype)
        self._proj_cache: dict[int, np.ndarray] = {}

    def _get_projection_matrix(self, d_in: int) -> np.ndarray:
        """Get or create the projection matrix for a given input dim."""
        if d_in not in self._proj_cache:
            rng = np.random.RandomState(self.seed)
            R = rng.randn(d_in, self.d_out).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            self._proj_cache[d_in] = Q * np.sqrt(d_in / self.d_out)
        return self._proj_cache[d_in]

    def encode(self, raw: np.ndarray) -> dict:
        """Encode a single protein's raw per-residue embeddings.

        Args:
            raw: (L, D) raw PLM per-residue embeddings, float32.

        Returns:
            dict with 'per_residue' (L, d_out), 'protein_vec' (dct_k * d_out,),
            and 'metadata'.
        """
        L, D = raw.shape
        R = self._get_projection_matrix(D)
        per_residue = (raw @ R).astype(np.float32)
        protein_vec = dct_summary(per_residue, K=self.dct_k)

        return {
            "per_residue": per_residue.astype(self.dtype),
            "protein_vec": protein_vec.astype(self.dtype),
            "metadata": {
                "codec": "rp_dct",
                "version": 1,
                "d_in": D,
                "d_out": self.d_out,
                "dct_k": self.dct_k,
                "seed": self.seed,
                "seq_len": L,
                "protein_vec_dim": self.dct_k * self.d_out,
                "dtype": str(self.dtype),
            },
        }

    def save(self, encoded: dict, path: str, protein_id: str = "protein") -> Path:
        """Save an encoded protein to a self-contained H5 file.

        Args:
            encoded: Output of encode().
            path: Output file path (.h5).
            protein_id: Identifier stored in metadata.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.create_dataset("per_residue", data=encoded["per_residue"],
                             compression="gzip", compression_opts=4)
            f.create_dataset("protein_vec", data=encoded["protein_vec"])

            meta = encoded["metadata"].copy()
            meta["protein_id"] = protein_id
            f.attrs["metadata"] = json.dumps(meta)

        return path

    @staticmethod
    def load(path: str) -> dict:
        """Load a One Embedding file. No codec knowledge needed.

        Args:
            path: Path to .h5 file.

        Returns:
            dict with 'per_residue' (L, d_out), 'protein_vec' (K*d_out,),
            and 'metadata' dict.
        """
        with h5py.File(path, "r") as f:
            return {
                "per_residue": f["per_residue"][:],
                "protein_vec": f["protein_vec"][:],
                "metadata": json.loads(f.attrs["metadata"]),
            }

    def encode_h5(
        self,
        input_h5: str,
        output_dir: str,
        max_proteins: Optional[int] = None,
    ) -> list[Path]:
        """Batch-encode all proteins from a raw embedding H5 file.

        Args:
            input_h5: Path to H5 file with datasets keyed by protein ID,
                      each of shape (L, D).
            output_dir: Directory for output .h5 files (one per protein).
            max_proteins: Optional limit for testing.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        with h5py.File(input_h5, "r") as f:
            keys = list(f.keys())
            if max_proteins:
                keys = keys[:max_proteins]

            for pid in keys:
                raw = f[pid][:].astype(np.float32)
                encoded = self.encode(raw)
                out_path = self.save(encoded, output_dir / f"{pid}.h5",
                                     protein_id=pid)
                saved.append(out_path)

        return saved

    def encode_h5_to_h5(
        self,
        input_h5: str,
        output_h5: str,
        max_proteins: Optional[int] = None,
    ) -> Path:
        """Batch-encode into a single H5 file (one group per protein).

        More practical for large datasets than one-file-per-protein.

        Args:
            input_h5: Path to raw embedding H5.
            output_h5: Path for output H5.
            max_proteins: Optional limit.

        Returns:
            Path to output H5.
        """
        output_h5 = Path(output_h5)
        output_h5.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(input_h5, "r") as fin, \
             h5py.File(output_h5, "w") as fout:

            keys = list(fin.keys())
            if max_proteins:
                keys = keys[:max_proteins]

            # Store codec metadata at root
            fout.attrs["metadata"] = json.dumps({
                "codec": "rp_dct",
                "version": 1,
                "d_out": self.d_out,
                "dct_k": self.dct_k,
                "seed": self.seed,
                "dtype": str(self.dtype),
                "n_proteins": len(keys),
            })

            for pid in keys:
                raw = fin[pid][:].astype(np.float32)
                encoded = self.encode(raw)

                grp = fout.create_group(pid)
                grp.create_dataset("per_residue", data=encoded["per_residue"],
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("protein_vec", data=encoded["protein_vec"])
                grp.attrs["seq_len"] = raw.shape[0]
                grp.attrs["d_in"] = raw.shape[1]

        return output_h5

    @staticmethod
    def load_batch(path: str, protein_ids: Optional[list[str]] = None) -> dict:
        """Load protein vectors and/or per-residue from a batch H5.

        Args:
            path: Path to batch H5 file.
            protein_ids: Optional subset. None = all.

        Returns:
            dict with 'protein_vecs' {id: (K*d,)},
            'per_residue' {id: (L, d)}, 'metadata' dict.
        """
        protein_vecs = {}
        per_residue = {}

        with h5py.File(path, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
            ids = protein_ids or [k for k in f.keys()]

            for pid in ids:
                grp = f[pid]
                protein_vecs[pid] = grp["protein_vec"][:]
                per_residue[pid] = grp["per_residue"][:]

        return {
            "protein_vecs": protein_vecs,
            "per_residue": per_residue,
            "metadata": metadata,
        }
