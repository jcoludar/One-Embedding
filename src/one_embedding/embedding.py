"""OneEmbedding dataclass — the unified protein representation."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class OneEmbedding:
    """A single embedding per protein that serves both protein-level and residue-level tasks.

    Layout (summary-prefixed):
        v = [summary | residue_1 | residue_2 | ... | residue_L]

    The summary is a fixed-size protein-level embedding derived from one of
    several mathematical transforms (DCT, Haar wavelet, spectral fingerprint).
    The residues are the compressed per-residue embeddings from ChannelCompressor.

    Properties:
        summary: fixed-size protein-level vector
        residues: (L, d) per-residue matrix
        flat: full linearized vector [summary | residues.ravel()]
    """

    protein_id: str
    plm: str
    latent_dim: int
    seq_len: int
    transform: str
    _summary: np.ndarray
    _residues: np.ndarray

    @property
    def summary(self) -> np.ndarray:
        """Fixed-size protein-level embedding vector."""
        return self._summary

    @property
    def summary_dim(self) -> int:
        """Dimensionality of the protein-level summary."""
        return self._summary.shape[0]

    @property
    def residues(self) -> np.ndarray:
        """Per-residue embedding matrix (L, d)."""
        return self._residues

    @property
    def flat(self) -> np.ndarray:
        """Full linearized vector: [summary | residues.ravel()]."""
        return np.concatenate([self._summary, self._residues.ravel()])

    @classmethod
    def from_compressed(
        cls,
        protein_id: str,
        plm: str,
        matrix: np.ndarray,
        transform: str = "mean",
        summary: np.ndarray | None = None,
    ) -> "OneEmbedding":
        """Build from compressed per-residue matrix (L, d).

        Args:
            protein_id: Protein identifier.
            plm: PLM name (e.g. "prot_t5_xl").
            matrix: Compressed per-residue embeddings, shape (L, d).
            transform: Name of the transform used for the summary.
            summary: Pre-computed summary vector. If None, uses mean pool.
        """
        matrix = matrix.astype(np.float32)
        if summary is None:
            summary = matrix.mean(axis=0).astype(np.float32)
        else:
            summary = summary.astype(np.float32)

        return cls(
            protein_id=protein_id,
            plm=plm,
            latent_dim=matrix.shape[1],
            seq_len=matrix.shape[0],
            transform=transform,
            _summary=summary,
            _residues=matrix,
        )
