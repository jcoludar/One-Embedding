"""One Embedding: universal codec for PLM per-residue embeddings.

Compresses raw PLM output (L, D) into a self-contained float16 representation
that serves both protein-level retrieval and per-residue structure prediction.
The 1.0 codec defaults to 768d (2.5x compression, 100.1% mean retention).

Quick start::

    from src.one_embedding.codec import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=768, dct_k=4)  # 768d float16 default
    encoded = codec.encode(raw)       # raw: (L, 1024) -> {per_residue, protein_vec}
    codec.save(encoded, "out.one.h5") # self-contained .one.h5 file (float16)

    loaded = OneEmbeddingCodec.load("out.one.h5")
    loaded["protein_vec"]             # (3072,) for UMAP / retrieval
    loaded["per_residue"]             # (L, 768) for SS3 / disorder probes

Key classes:
    OneEmbeddingCodec  -- encode/save/load compressed embeddings (codec.py)
    OneEmbedding       -- dataclass for embedding + metadata (embedding.py)

Transform modules:
    transforms.py           -- DCT, Haar wavelet, spectral fingerprint
    universal_transforms.py -- random projection, feature hashing, power mean
    enriched_transforms.py  -- moment pool, autocovariance, Fisher vector, PCA
    path_transforms.py      -- displacement DCT, signatures, curvature, gyration
"""

from src.one_embedding.codec import OneEmbeddingCodec
from src.one_embedding.embedding import OneEmbedding
from src.one_embedding.hrr import (
    hrr_bind,
    hrr_decode,
    hrr_encode,
    hrr_kslot_decode,
    hrr_kslot_encode,
    hrr_per_protein,
    hrr_per_residue,
    hrr_unbind,
)
from src.one_embedding.enriched_transforms import (
    EnrichedTransformPipeline,
    autocovariance_pool,
    dct_pool,
    fisher_vector,
    gram_features,
    haar_pool,
    moment_pool,
)
from src.one_embedding.transforms import (
    dct_summary,
    haar_full_coefficients,
    haar_summary,
    inverse_dct,
    inverse_haar,
    spectral_fingerprint,
    spectral_moments,
)
from src.one_embedding.path_transforms import (
    curvature_enriched,
    discrete_curvature,
    displacement_decode,
    displacement_dct,
    displacement_encode,
    displacement_magnitude,
    gyration_eigenspectrum,
    inverse_displacement_dct,
    lag_cross_covariance_eigenvalues,
    path_signature_depth2,
    path_signature_depth3,
    path_statistics,
    shape_descriptors,
)
from src.one_embedding.universal_transforms import (
    feature_hash,
    kernel_mean_embedding,
    norm_weighted_mean,
    percentile_pool,
    power_mean_pool,
    random_orthogonal_project,
    sparse_random_project,
    svd_spectrum,
    trimmed_mean_pool,
)
from src.one_embedding.data_analysis import (
    channel_distributions,
    intrinsic_dimensionality,
)

__all__ = [
    "EnrichedTransformPipeline",
    "OneEmbedding",
    "OneEmbeddingCodec",
    "autocovariance_pool",
    "channel_distributions",
    "curvature_enriched",
    "dct_pool",
    "dct_summary",
    "discrete_curvature",
    "displacement_decode",
    "displacement_dct",
    "displacement_encode",
    "displacement_magnitude",
    "feature_hash",
    "fisher_vector",
    "gram_features",
    "gyration_eigenspectrum",
    "haar_full_coefficients",
    "haar_pool",
    "haar_summary",
    "hrr_bind",
    "hrr_decode",
    "hrr_encode",
    "hrr_kslot_decode",
    "hrr_kslot_encode",
    "hrr_per_protein",
    "hrr_per_residue",
    "hrr_unbind",
    "intrinsic_dimensionality",
    "inverse_dct",
    "inverse_displacement_dct",
    "inverse_haar",
    "kernel_mean_embedding",
    "lag_cross_covariance_eigenvalues",
    "moment_pool",
    "norm_weighted_mean",
    "path_signature_depth2",
    "path_signature_depth3",
    "path_statistics",
    "percentile_pool",
    "power_mean_pool",
    "random_orthogonal_project",
    "shape_descriptors",
    "sparse_random_project",
    "spectral_fingerprint",
    "spectral_moments",
    "svd_spectrum",
    "trimmed_mean_pool",
    "Codec",
    "encode",
    "decode",
    "embed",
    "__version__",
]

# ── Production API ──────────────────────────────────────────────────
__version__ = "1.0.0"

from src.one_embedding.core import Codec


def encode(input_path, output_path, d_out=768, dct_k=4, seed=42):
    """Compress per-residue embeddings H5 to .one.h5 format."""
    import h5py
    import numpy as np
    from src.one_embedding.io import ONE_H5_FORMAT, ONE_H5_VERSION

    codec = Codec(d_out=d_out, dct_k=dct_k, seed=seed)

    # Fit ABTT from corpus (sample residues without loading all into memory)
    # Create the RNG once and reuse it across all proteins
    rng = np.random.RandomState(seed)
    residues = []
    with h5py.File(input_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            emb = f[key][:]
            # Sample up to 200 residues per protein for fitting
            if emb.shape[0] > 200:
                idx = rng.choice(emb.shape[0], 200, replace=False)
                residues.append(emb[idx])
            else:
                residues.append(emb)
            if sum(r.shape[0] for r in residues) > 50000:
                break
    stacked = np.concatenate(residues, axis=0).astype(np.float32)
    from src.one_embedding.core.preprocessing import fit_abtt
    codec._abtt_params = fit_abtt(stacked, k=3, seed=seed)

    # Tags for the .one.h5 file
    tags = {
        "source_model": "unknown",
        "d_out": d_out,
        "compression": f"abtt3_rp{d_out}_dct{dct_k}",
        "seed": seed,
    }

    # Encode proteins incrementally — write each protein directly to H5
    # to avoid accumulating all encoded data in memory.
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as out_f:
        # Root attributes for .one.h5 format
        out_f.attrs["format"] = ONE_H5_FORMAT
        out_f.attrs["version"] = ONE_H5_VERSION
        out_f.attrs["n_proteins"] = len(keys)
        for k, v in tags.items():
            out_f.attrs[k] = v

        with h5py.File(input_path, "r") as in_f:
            for key in in_f.keys():
                encoded = codec.encode(in_f[key][:])
                per_residue = np.asarray(encoded["per_residue"])
                protein_vec = np.asarray(encoded["protein_vec"], dtype=np.float16)
                grp = out_f.create_group(key)
                grp.create_dataset(
                    "per_residue", data=per_residue,
                    compression="gzip", compression_opts=4,
                )
                grp.create_dataset("protein_vec", data=protein_vec)
                grp.attrs["seq_len"] = per_residue.shape[0]


def decode(path, protein_id=None):
    """Read .one.h5 or .oemb file into arrays.

    Auto-detects format: tries .one.h5 (format="one_embedding") first,
    then falls back to legacy .oemb reading.
    """
    import numpy as np
    import h5py
    from src.one_embedding.io import (
        read_one_h5, read_one_h5_batch,
        read_oemb, read_oemb_batch, ONE_H5_FORMAT,
    )

    with h5py.File(str(path), "r") as f:
        fmt = str(f.attrs.get("format", ""))
        is_one_h5 = (fmt == ONE_H5_FORMAT)
        n_proteins = int(f.attrs.get("n_proteins", 0))
        has_root_per_residue = "per_residue" in f

    if is_one_h5:
        # New .one.h5 format
        if protein_id:
            batch = read_one_h5_batch(str(path), protein_ids=[protein_id])
            if protein_id not in batch:
                raise KeyError(f"Protein '{protein_id}' not found")
            return batch[protein_id]
        elif n_proteins == 1:
            return read_one_h5(str(path))
        else:
            return read_one_h5_batch(str(path))

    # Legacy .oemb format
    if has_root_per_residue:
        # Single protein file
        with h5py.File(str(path), "r") as f:
            result = {
                "per_residue": np.array(f["per_residue"]),
                "protein_vec": np.array(f["protein_vec"]),
            }
            for attr in ("sequence", "source_model", "codec", "protein_id"):
                if attr in f.attrs:
                    result[attr] = str(f.attrs[attr])
            return result
    elif protein_id:
        # Specific protein from batch file
        with h5py.File(str(path), "r") as f:
            if protein_id not in f:
                raise KeyError(f"Protein '{protein_id}' not found")
            g = f[protein_id]
            entry = {
                "per_residue": np.array(g["per_residue"]),
                "protein_vec": np.array(g["protein_vec"]),
            }
            if "sequence" in g.attrs:
                entry["sequence"] = str(g.attrs["sequence"])
            return entry
    else:
        # Full batch — delegate to read_oemb_batch
        return read_oemb_batch(str(path))


def embed(input_path, output_path, model="prot_t5"):
    """Extract per-residue embeddings from FASTA sequences."""
    from src.one_embedding.extract import extract_embeddings
    extract_embeddings(input_path, output_path, model=model)
