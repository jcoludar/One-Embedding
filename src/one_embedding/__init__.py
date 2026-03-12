"""One Embedding: unified per-protein + per-residue representation.

Mathematically principled transforms (DCT, Haar wavelet, spectral fingerprint)
applied to compressed PLM per-residue embeddings to create a single embedding
that serves both protein-level and residue-level downstream tasks.
"""

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
