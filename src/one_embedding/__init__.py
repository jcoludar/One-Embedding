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
    power_mean_pool,
    random_orthogonal_project,
    svd_spectrum,
)

__all__ = [
    "EnrichedTransformPipeline",
    "OneEmbedding",
    "OneEmbeddingCodec",
    "autocovariance_pool",
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
    "power_mean_pool",
    "random_orthogonal_project",
    "shape_descriptors",
    "spectral_fingerprint",
    "spectral_moments",
    "svd_spectrum",
]
