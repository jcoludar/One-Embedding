"""Core Codec: ABTT3 + RP512 + DCT K=4 — the recommended protein embedding codec.

Pipeline per protein:
    raw (L, D)
      → ABTT(k=3): remove top-3 PCs for isotropy       → (L, D)
      → RP(d_out=512): random orthogonal projection     → (L, 512)
      → per_residue output (L, 512) fp16
      → DCT(K=4): spectral pooling along sequence axis  → (4, 512)
      → protein_vec output (2048,) fp16

The ABTT params (mean, top_pcs) are fitted once on a corpus and persisted.
The projection matrix is seeded — no fitting required.

Reference results (V1, SCOPe 5K):
    Ret@1=0.786, SS3 Q3=0.809, size≈48 KB/protein (int16 per_residue).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.fft import dct

from .preprocessing import fit_abtt, apply_abtt
from .projection import project


class Codec:
    """Training-free codec for PLM per-residue embeddings.

    Encodes raw (L, D) embeddings into a compact representation that
    supports two downstream modes:
        - per_residue (L, d_out) — for per-residue probes (SS3, disorder, ...)
        - protein_vec (dct_k * d_out,) — for retrieval, clustering, UMAP

    Args:
        d_out: Per-residue output dimensionality (default 512).
        dct_k: Number of DCT coefficients for the protein vector (default 4).
        seed: Projection matrix seed (default 42). Acts as the "codec key".
    """

    def __init__(
        self,
        d_out: int = 512,
        dct_k: int = 4,
        seed: int = 42,
    ) -> None:
        self.d_out = d_out
        self.dct_k = dct_k
        self.seed = seed
        self._abtt_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: dict, k: int = 3) -> "Codec":
        """Fit ABTT preprocessing on a corpus of raw embeddings.

        Args:
            corpus: {protein_id: (L, D) ndarray} mapping.
            k: Number of dominant PCs to remove (default 3).

        Returns:
            self (for chaining).
        """
        # Stack all residues from all proteins
        residues = np.vstack(
            [np.asarray(v, dtype=np.float32) for v in corpus.values()]
        )
        self._abtt_params = fit_abtt(residues, k=k, seed=self.seed)
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, raw: np.ndarray) -> dict:
        """Encode a single protein's raw per-residue embeddings.

        If fit() has been called, ABTT preprocessing is applied.
        Otherwise the raw embeddings are projected directly.

        Args:
            raw: (L, D) raw PLM per-residue embeddings, float32.

        Returns:
            dict with:
                per_residue: (L, d_out) float16 — compressed per-residue.
                protein_vec: (dct_k * d_out,) float16 — protein-level vector.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D array (L, D), got shape {raw.shape}")
        L, D = raw.shape

        # Step 1: ABTT preprocessing (if fitted)
        if self._abtt_params is not None:
            preprocessed = apply_abtt(raw, self._abtt_params)
        else:
            preprocessed = raw

        # Step 2: Random orthogonal projection D → d_out
        projected = project(preprocessed, d_out=self.d_out, seed=self.seed)
        # projected: (L, d_out) float32

        # Step 3: DCT along sequence axis for protein vector
        # dct(X, axis=0) treats each column as a 1D signal over L residues.
        k = min(self.dct_k, L)
        coeffs = dct(projected, axis=0, type=2, norm="ortho")  # (L, d_out)
        protein_vec = coeffs[:k].ravel()  # (k * d_out,)

        # Zero-pad protein_vec if L < dct_k (edge case: very short proteins)
        if k < self.dct_k:
            pad = np.zeros(
                (self.dct_k - k) * self.d_out, dtype=np.float32
            )
            protein_vec = np.concatenate([protein_vec, pad])

        return {
            "per_residue": projected.astype(np.float16),
            "protein_vec": protein_vec.astype(np.float16),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_params(self, path: str) -> Path:
        """Save ABTT parameters to a JSON file.

        Args:
            path: Output path (e.g. "abtt_params.json").

        Returns:
            Path to saved file.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._abtt_params is None:
            raise RuntimeError("No params to save — call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = {
            "codec": "core_v1",
            "d_out": self.d_out,
            "dct_k": self.dct_k,
            "seed": self.seed,
            "mean": self._abtt_params["mean"].tolist(),
            "top_pcs": self._abtt_params["top_pcs"].tolist(),
        }
        path.write_text(json.dumps(serialisable))
        return path

    def load_params(self, path: str) -> "Codec":
        """Load ABTT parameters from a JSON file.

        Args:
            path: Path to params file saved by save_params().

        Returns:
            self (for chaining).
        """
        data = json.loads(Path(path).read_text())
        self._abtt_params = {
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "top_pcs": np.asarray(data["top_pcs"], dtype=np.float32),
        }
        # Restore codec hyperparams if present
        if "d_out" in data:
            self.d_out = data["d_out"]
        if "dct_k" in data:
            self.dct_k = data["dct_k"]
        if "seed" in data:
            self.seed = data["seed"]
        return self
