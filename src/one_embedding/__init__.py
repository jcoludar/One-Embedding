"""One Embedding: universal codec for PLM per-residue embeddings.

Compresses raw PLM output (L, D) into a self-contained float16 representation
(~26% of raw size) that serves both protein-level retrieval and per-residue
structure prediction with zero quality loss vs float32.

Quick start::

    from src.one_embedding.codec import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=512, dct_k=4)  # float16 default
    encoded = codec.encode(raw)       # raw: (L, 1024) → {per_residue, protein_vec}
    codec.save(encoded, "out.h5")     # self-contained H5 file (float16)

    loaded = OneEmbeddingCodec.load("out.h5")
    loaded["protein_vec"]             # (2048,) for UMAP / retrieval
    loaded["per_residue"]             # (L, 512) for SS3 / disorder probes

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
__version__ = "0.1.0"

from src.one_embedding.core import Codec


def encode(input_path, output_path, d_out=512, dct_k=4, seed=42):
    """Compress per-residue embeddings H5 to .oemb format."""
    import h5py
    from src.one_embedding.io import write_oemb_batch

    codec = Codec(d_out=d_out, dct_k=dct_k, seed=seed)

    # Fit ABTT from corpus (sample residues without loading all into memory)
    import numpy as np
    residues = []
    with h5py.File(input_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            emb = f[key][:]
            # Sample up to 200 residues per protein for fitting
            if emb.shape[0] > 200:
                idx = np.random.RandomState(seed).choice(emb.shape[0], 200, replace=False)
                residues.append(emb[idx])
            else:
                residues.append(emb)
            if sum(r.shape[0] for r in residues) > 50000:
                break
    stacked = np.concatenate(residues, axis=0).astype(np.float32)
    from src.one_embedding.core.preprocessing import fit_abtt
    codec._abtt_params = fit_abtt(stacked, k=3, seed=seed)

    # Encode proteins one at a time (streaming)
    proteins = {}
    with h5py.File(input_path, "r") as f:
        for key in f.keys():
            proteins[key] = codec.encode(f[key][:])

    write_oemb_batch(output_path, proteins)


def decode(path, protein_id=None):
    """Read .oemb file into arrays."""
    import h5py
    from src.one_embedding.io import read_oemb, read_oemb_batch
    with h5py.File(path, "r") as f:
        if "per_residue" in f:
            return read_oemb(path)
        elif protein_id:
            return read_oemb_batch(path, [protein_id]).get(protein_id)
        else:
            return read_oemb_batch(path)


def embed(input_path, output_path, model="prot_t5"):
    """Extract per-residue embeddings from FASTA sequences."""
    from src.one_embedding.extract import extract_embeddings
    extract_embeddings(input_path, output_path, model=model)
