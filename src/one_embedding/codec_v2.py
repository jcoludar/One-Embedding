"""One Embedding Codec V2: progressive compression with selectable quality tiers.

Extends V1 with Product Quantization for dramatically smaller per-protein
storage, while preserving both retrieval and per-residue task quality.

Compression modes (mean L=175, D_in=1024):
    'full'     — int4 per-residue, 48 KB (same as V1)
    'balanced' — PQ M=128 per-residue, 26 KB  (Ret@1=0.784, SS3=0.807)
    'compact'  — PQ M=64 per-residue, 15 KB   (Ret@1=0.779, SS3=0.778)
    'micro'    — PQ M=32 per-residue, 10 KB    (Ret@1=0.766, SS3=0.739)
    'binary'   — 1-bit per-residue, 15 KB      (Ret@1=0.787, SS3=0.776)

Usage:
    # One-time: fit codebook on training corpus
    codec = OneEmbeddingCodec(quantization='pq', pq_m=64)
    codec.fit(training_embeddings)
    codec.save_codebook('codebook.h5')

    # Encode (uses fitted codebook)
    codec = OneEmbeddingCodec(quantization='pq', pq_m=64, codebook_path='codebook.h5')
    codec.encode_h5_to_h5('raw.h5', 'compressed.h5')

    # Decode (receiver side: h5py + numpy + codebook)
    data = OneEmbeddingCodec.load('compressed.h5', codebook_path='codebook.h5')
    data['per_residue']   # (L, 512)
    data['protein_vec']   # (2048,)

    # Legacy V2 compat
    codec = OneEmbeddingCodecV2(mode='compact')
"""

import json
from pathlib import Path

import h5py
import numpy as np

from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.transforms import dct_summary
from src.one_embedding.quantization import (
    quantize_int4, dequantize_int4,
    quantize_binary, dequantize_binary,
    pq_fit, pq_encode, pq_decode,
)


def auto_pq_m(d_out: int) -> int:
    """Compute default PQ M targeting ~6d sub-vectors (~30x compression).

    Finds the largest factor of d_out that is <= d_out // 6.
    For d_out=768: returns 128 (6d sub-vectors, 30x compression).
    For d_out=512: returns 64 (8d sub-vectors, 32x compression).
    """
    target = d_out // 6
    for m in range(target, 0, -1):
        if d_out % m == 0:
            return m
    return 1


_LEGACY_MODES = {
    "full":     {"type": "int4",   "desc": "int4 per-residue (V1 compatible)"},
    "balanced": {"type": "pq",     "M": 128, "K": 256, "desc": "PQ M=128"},
    "compact":  {"type": "pq",     "M": 64,  "K": 256, "desc": "PQ M=64"},
    "micro":    {"type": "pq",     "M": 32,  "K": 256, "desc": "PQ M=32"},
    "binary":   {"type": "binary", "desc": "1-bit sign quantization"},
}

# Alias so existing code importing MODES still works
MODES = _LEGACY_MODES

_VALID_QUANTIZATIONS = {None, "int4", "pq", "binary"}


class OneEmbeddingCodec:
    """Progressive protein embedding codec with PQ support.

    Args:
        d_out: RP projection dimensionality (default 768).
        quantization: Compression type — None (fp16), 'int4', 'pq', 'binary'.
        pq_m: Number of PQ sub-spaces (auto-computed from d_out when None).
        dct_k: DCT coefficients for protein vector (default 4).
        seed: Fixed seed for RP matrix (default 42).
        codebook_path: Path to pre-fitted codebook H5 (required for PQ modes).
        mode: Legacy V2 parameter — maps to quantization (backward compat).
    """

    def __init__(
        self,
        d_out: int = 768,
        quantization: str | None = "pq",
        pq_m: int | None = None,
        dct_k: int = 4,
        seed: int = 42,
        codebook_path: str | None = None,
        # Legacy V2 compat — if mode is passed, map to new API
        mode: str | None = None,
    ):
        # Legacy V2 compat: map mode to quantization
        if mode is not None:
            if mode not in _LEGACY_MODES:
                raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_LEGACY_MODES)}")
            cfg = _LEGACY_MODES[mode]
            quantization = cfg["type"]
            if quantization == "pq":
                pq_m = cfg["M"]
            if d_out == 768:  # only override if user didn't set it
                d_out = 512  # legacy V2 was 512d

        if quantization not in _VALID_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_VALID_QUANTIZATIONS}, got '{quantization}'"
            )

        self.d_out = d_out
        self.quantization = quantization
        self.dct_k = dct_k
        self.seed = seed
        self._proj_cache: dict[int, np.ndarray] = {}
        self._corpus_stats = None
        self._pq_model = None

        # Resolve pq_m
        if quantization == "pq":
            if pq_m is None:
                pq_m = auto_pq_m(d_out)
            if d_out % pq_m != 0:
                factors = [f for f in range(1, d_out + 1) if d_out % f == 0 and f <= d_out // 2]
                raise ValueError(
                    f"pq_m={pq_m} must divide d_out={d_out} evenly. "
                    f"Some valid values: {factors[-10:]}"
                )
            self.pq_m = pq_m
            self.pq_k = 256
        else:
            self.pq_m = None
            self.pq_k = None

        if codebook_path is not None:
            self._load_codebook(codebook_path)

    # ── Legacy compat properties ───────────────────────────────────────────

    @property
    def mode(self):
        """Legacy compat."""
        if self.quantization == "int4":
            return "full"
        elif self.quantization == "pq":
            return "balanced"
        elif self.quantization == "binary":
            return "binary"
        return "preserve"

    @property
    def mode_cfg(self):
        """Legacy compat — maps quantization to old mode config."""
        if self.quantization == "int4":
            return {"type": "int4"}
        elif self.quantization == "pq":
            return {"type": "pq", "M": self.pq_m, "K": self.pq_k}
        elif self.quantization == "binary":
            return {"type": "binary"}
        return {"type": "fp16"}

    # ── Projection ────────────────────────────────────────────────────────

    def _get_projection_matrix(self, d_in: int) -> np.ndarray:
        if d_in not in self._proj_cache:
            rng = np.random.RandomState(self.seed)
            R = rng.randn(d_in, self.d_out).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            self._proj_cache[d_in] = Q * np.sqrt(d_in / self.d_out)
        return self._proj_cache[d_in]

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Apply ABTT3 + RP projection (skip RP if d_out >= d_in)."""
        raw = raw.astype(np.float32)
        if self._corpus_stats is not None:
            top3 = self._corpus_stats["top_pcs"][:3]
            raw = all_but_the_top(raw, top3)
        if self.d_out >= raw.shape[1]:
            return raw  # skip RP — lossless mode
        R = self._get_projection_matrix(raw.shape[1])
        return (raw @ R).astype(np.float32)

    # ── Fit codebook ──────────────────────────────────────────────────────

    def fit(self, embeddings: dict[str, np.ndarray], max_residues: int = 500_000):
        """Fit corpus stats and PQ codebook from training embeddings.

        Args:
            embeddings: Dict of protein_id → (L, D) float32 arrays.
            max_residues: Max residues for PQ fitting.
        """
        # Corpus stats for ABTT3
        self._corpus_stats = compute_corpus_stats(
            embeddings, n_sample=50_000, n_pcs=5, seed=self.seed
        )

        # Fit PQ if needed
        if self.quantization == "pq":
            preprocessed = {}
            for pid, m in embeddings.items():
                preprocessed[pid] = self._preprocess(m)
            self._pq_model = pq_fit(
                preprocessed, M=self.pq_m, n_centroids=self.pq_k,
                max_residues=max_residues, seed=self.seed,
            )

    def save_codebook(self, path: str) -> Path:
        """Save fitted codebook (corpus stats + PQ model) to H5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            # Corpus stats
            f.create_dataset("top_pcs", data=self._corpus_stats["top_pcs"])
            f.create_dataset("mean_vec", data=self._corpus_stats["mean_vec"])

            # PQ model
            if self._pq_model is not None:
                f.create_dataset("pq_codebook", data=self._pq_model["codebook"])
                f.attrs["pq_M"] = self._pq_model["M"]
                f.attrs["pq_K"] = self._pq_model["n_centroids"]
                f.attrs["pq_sub_dim"] = self._pq_model["sub_dim"]
                f.attrs["pq_D"] = self._pq_model["D"]

            f.attrs["mode"] = self.mode
            f.attrs["quantization"] = self.quantization or "none"
            f.attrs["d_out"] = self.d_out
            f.attrs["dct_k"] = self.dct_k
            f.attrs["seed"] = self.seed

        return path

    def _load_codebook(self, path: str):
        """Load codebook from H5."""
        with h5py.File(path, "r") as f:
            self._corpus_stats = {
                "top_pcs": f["top_pcs"][:],
                "mean_vec": f["mean_vec"][:],
            }
            if "pq_codebook" in f:
                self._pq_model = {
                    "codebook": f["pq_codebook"][:],
                    "M": int(f.attrs["pq_M"]),
                    "n_centroids": int(f.attrs["pq_K"]),
                    "sub_dim": int(f.attrs["pq_sub_dim"]),
                    "D": int(f.attrs["pq_D"]),
                }

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, raw: np.ndarray) -> dict:
        """Encode a single protein's raw PLM embeddings.

        Args:
            raw: (L, D) raw PLM per-residue embeddings.

        Returns:
            dict with compressed data + metadata.
        """
        L, D = raw.shape
        projected = self._preprocess(raw)
        protein_vec = dct_summary(projected, K=self.dct_k).astype(np.float16)

        result = {
            "protein_vec": protein_vec,
            "metadata": {
                "codec": "one_embedding",
                "version": 3,
                "d_in": D,
                "d_out": self.d_out,
                "quantization": self.quantization,
                "pq_m": self.pq_m,
                "dct_k": self.dct_k,
                "seed": self.seed,
                "seq_len": L,
            },
        }

        if self.quantization is None:
            result["per_residue_fp16"] = projected.astype(np.float16)
        elif self.quantization == "int4":
            compressed = quantize_int4(projected)
            result["per_residue_data"] = compressed["data"]
            result["per_residue_scales"] = compressed["scales"]
            result["per_residue_zp"] = compressed["zero_points"]
        elif self.quantization == "binary":
            compressed = quantize_binary(projected)
            result["per_residue_bits"] = compressed["bits"]
            result["per_residue_means"] = compressed["means"]
            result["per_residue_scales"] = compressed["scales"]
        elif self.quantization == "pq":
            codes = pq_encode(projected, self._pq_model)
            result["pq_codes"] = codes  # (L, M) uint8

        return result

    def decode_per_residue(self, encoded: dict) -> np.ndarray:
        """Decode per-residue embeddings from encoded dict."""
        meta = encoded["metadata"]
        L = meta["seq_len"]
        d_out = meta.get("d_out", self.d_out)

        # V3 format: "quantization" key. V2 legacy: "mode" key
        quantization = meta.get("quantization")
        if quantization is None and "mode" in meta:
            legacy_mode = meta["mode"]
            cfg = _LEGACY_MODES.get(legacy_mode, {})
            quantization = cfg.get("type")

        if quantization is None or quantization == "fp16":
            return encoded["per_residue_fp16"].astype(np.float32)
        elif quantization == "int4":
            compressed = {
                "data": encoded["per_residue_data"],
                "scales": encoded["per_residue_scales"],
                "zero_points": encoded["per_residue_zp"],
                "original_shape": (L, d_out),
                "dtype": "int4",
            }
            return dequantize_int4(compressed)
        elif quantization == "binary":
            compressed = {
                "bits": encoded["per_residue_bits"],
                "means": encoded["per_residue_means"],
                "scales": encoded["per_residue_scales"],
                "original_shape": (L, d_out),
                "dtype": "binary",
            }
            return dequantize_binary(compressed)
        elif quantization == "pq":
            return pq_decode(encoded["pq_codes"], self._pq_model)
        raise ValueError(f"Unknown quantization: {quantization}")

    # ── H5 I/O ────────────────────────────────────────────────────────────

    def save(self, encoded: dict, path: str, protein_id: str = "protein") -> Path:
        """Save encoded protein to H5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.create_dataset("protein_vec", data=encoded["protein_vec"])

            if self.quantization is None:
                f.create_dataset("per_residue", data=encoded["per_residue_fp16"],
                                 compression="gzip", compression_opts=4)
            elif self.quantization == "int4":
                f.create_dataset("per_residue_data",
                                 data=encoded["per_residue_data"],
                                 compression="gzip", compression_opts=4)
                f.create_dataset("per_residue_scales",
                                 data=encoded["per_residue_scales"])
                f.create_dataset("per_residue_zp",
                                 data=encoded["per_residue_zp"])
            elif self.quantization == "binary":
                f.create_dataset("per_residue_bits",
                                 data=encoded["per_residue_bits"],
                                 compression="gzip", compression_opts=4)
                f.create_dataset("per_residue_means",
                                 data=encoded["per_residue_means"])
                f.create_dataset("per_residue_scales",
                                 data=encoded["per_residue_scales"])
            elif self.quantization == "pq":
                f.create_dataset("pq_codes", data=encoded["pq_codes"],
                                 compression="gzip", compression_opts=4)

            meta = encoded["metadata"].copy()
            meta["protein_id"] = protein_id
            f.attrs["metadata"] = json.dumps(meta)

        return path

    @staticmethod
    def load(path: str, codebook_path: str | None = None) -> dict:
        """Load and decode a compressed protein.

        Args:
            path: Path to compressed H5.
            codebook_path: Path to shared codebook (required for PQ modes).

        Returns:
            dict with 'per_residue', 'protein_vec', 'metadata'.
        """
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata"])
            protein_vec = f["protein_vec"][:]

            L = meta["seq_len"]
            d_out = meta.get("d_out", 512)

            # Determine quantization type
            quantization = meta.get("quantization")
            if quantization is None and "mode" in meta:
                legacy_mode = meta["mode"]
                cfg = _LEGACY_MODES.get(legacy_mode, {})
                quantization = cfg.get("type")

            if quantization is None or quantization == "fp16":
                # fp16 format — dataset named "per_residue"
                per_residue = f["per_residue"][:].astype(np.float32)
            elif quantization == "int4":
                compressed = {
                    "data": f["per_residue_data"][:],
                    "scales": f["per_residue_scales"][:],
                    "zero_points": f["per_residue_zp"][:],
                    "original_shape": (L, d_out),
                    "dtype": "int4",
                }
                per_residue = dequantize_int4(compressed)
            elif quantization == "binary":
                compressed = {
                    "bits": f["per_residue_bits"][:],
                    "means": f["per_residue_means"][:],
                    "scales": f["per_residue_scales"][:],
                    "original_shape": (L, d_out),
                    "dtype": "binary",
                }
                per_residue = dequantize_binary(compressed)
            elif quantization == "pq":
                codes = f["pq_codes"][:]
                if codebook_path is None:
                    raise ValueError("codebook_path required for PQ modes")
                with h5py.File(codebook_path, "r") as cb:
                    pq_model = {
                        "codebook": cb["pq_codebook"][:],
                        "M": int(cb.attrs["pq_M"]),
                        "n_centroids": int(cb.attrs["pq_K"]),
                        "sub_dim": int(cb.attrs["pq_sub_dim"]),
                        "D": int(cb.attrs["pq_D"]),
                    }
                per_residue = pq_decode(codes, pq_model)
            else:
                raise ValueError(f"Unknown quantization: {quantization}")

        return {
            "per_residue": per_residue,
            "protein_vec": protein_vec,
            "metadata": meta,
        }

    def encode_h5_to_h5(
        self, input_h5: str, output_h5: str,
        max_proteins: int | None = None,
    ) -> Path:
        """Batch-encode all proteins from raw H5 to compressed H5."""
        output_h5 = Path(output_h5)
        output_h5.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(input_h5, "r") as fin, \
             h5py.File(output_h5, "w") as fout:

            keys = list(fin.keys())
            if max_proteins:
                keys = keys[:max_proteins]

            fout.attrs["metadata"] = json.dumps({
                "codec": "one_embedding",
                "version": 3,
                "d_out": self.d_out,
                "quantization": self.quantization,
                "pq_m": self.pq_m,
                "dct_k": self.dct_k,
                "seed": self.seed,
                "n_proteins": len(keys),
            })

            for pid in keys:
                raw = fin[pid][:].astype(np.float32)
                encoded = self.encode(raw)

                grp = fout.create_group(pid)
                grp.create_dataset("protein_vec", data=encoded["protein_vec"])
                grp.attrs["seq_len"] = raw.shape[0]
                grp.attrs["d_in"] = raw.shape[1]

                if self.quantization is None:
                    grp.create_dataset("per_residue",
                                       data=encoded["per_residue_fp16"],
                                       compression="gzip", compression_opts=4)
                elif self.quantization == "int4":
                    grp.create_dataset("per_residue_data",
                                       data=encoded["per_residue_data"],
                                       compression="gzip", compression_opts=4)
                    grp.create_dataset("per_residue_scales",
                                       data=encoded["per_residue_scales"])
                    grp.create_dataset("per_residue_zp",
                                       data=encoded["per_residue_zp"])
                elif self.quantization == "binary":
                    grp.create_dataset("per_residue_bits",
                                       data=encoded["per_residue_bits"],
                                       compression="gzip", compression_opts=4)
                    grp.create_dataset("per_residue_means",
                                       data=encoded["per_residue_means"])
                    grp.create_dataset("per_residue_scales",
                                       data=encoded["per_residue_scales"])
                elif self.quantization == "pq":
                    grp.create_dataset("pq_codes", data=encoded["pq_codes"],
                                       compression="gzip", compression_opts=4)

        return output_h5


# Backward-compat alias
OneEmbeddingCodecV2 = OneEmbeddingCodec
