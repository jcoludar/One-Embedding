"""Backward compatibility — imports unified codec from codec_v2.

All new code should use:
    from src.one_embedding.codec_v2 import OneEmbeddingCodec
"""

from src.one_embedding.codec_v2 import OneEmbeddingCodec

__all__ = ["OneEmbeddingCodec"]
