"""One Embedding Codec — universal protein embedding compression.

Three knobs: d_out (768), quantization ('pq'), pq_m (auto).
Default: ABTT3 + RP 768d + PQ M=128 → ~30x compression.

Usage:
    # Default — ~30x compression, just works
    codec = OneEmbeddingCodec()
    codec.fit(training_embeddings)
    codec.save_codebook('codebook.h5')

    # Encode
    codec = OneEmbeddingCodec(codebook_path='codebook.h5')
    codec.encode_h5_to_h5('raw.h5', 'compressed.h5')

    # Decode (receiver side: h5py + numpy + codebook)
    data = OneEmbeddingCodec.load('compressed.h5', codebook_path='codebook.h5')
    data['per_residue']   # (L, 768)
    data['protein_vec']   # (3072,)

    # Max fidelity — no RP, fp16 only
    codec = OneEmbeddingCodec(d_out=1024, quantization=None)

    # More compression — int4 (10x)
    codec = OneEmbeddingCodec(quantization='int4')
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


_VALID_QUANTIZATIONS = {None, "int4", "pq", "binary"}


class OneEmbeddingCodec:
    """Universal protein embedding codec.

    Compresses raw PLM per-residue embeddings (any PLM, any protein) into
    compact representations for storage and downstream tasks.

    Default: ABTT3 + RP to 768d + PQ M=128 → ~30x compression.

    Three knobs control the compression/fidelity trade-off:
        d_out: Dimensions after random projection (default 768).
            Higher = more fidelity, less compression.
            Set to input dim (e.g. 1024) to skip RP entirely.
        quantization: Per-residue storage method (default 'pq').
            None = fp16 (2.7x), 'int4' (10x), 'pq' (~30x), 'binary' (41x).
        pq_m: PQ subquantizers (default auto = d_out // 6).
            Only used when quantization='pq'. Must divide d_out evenly.
    """

    def __init__(
        self,
        d_out: int = 768,
        quantization: str | None = "pq",
        pq_m: int | None = None,
        dct_k: int = 4,
        seed: int = 42,
        codebook_path: str | None = None,
    ):
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

        quantization = meta.get("quantization")

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

            quantization = meta.get("quantization")

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


