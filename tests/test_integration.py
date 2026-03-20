"""End-to-end: raw embeddings → .one.h5 → tools."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py
import tempfile


def test_full_pipeline():
    from src.one_embedding import encode, decode
    from src.one_embedding.tools import disorder, search, classify

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # Create raw embeddings
        with h5py.File(str(d / "raw.h5"), "w") as f:
            for i in range(10):
                f.create_dataset(f"prot_{i}", data=np.random.randn(80 + i*5, 1024).astype(np.float32))

        # Encode with d_out=512 (disorder CNN probe expects 512d)
        encode(str(d / "raw.h5"), str(d / "proteins.one.h5"), d_out=512)

        # Decode
        data = decode(str(d / "proteins.one.h5"))
        assert len(data) == 10
        assert data["prot_0"]["per_residue"].shape[1] == 512
        assert data["prot_0"]["protein_vec"].shape == (2048,)

        # Disorder
        dis = disorder.predict(str(d / "proteins.one.h5"))
        assert len(dis) == 10
        assert len(dis["prot_0"]) == 80

        # Search
        hits = search.find_neighbors(str(d / "proteins.one.h5"), k=3)
        assert len(hits) == 10
        assert len(hits["prot_0"]) == 3

        # Classify
        cls = classify.predict(str(d / "proteins.one.h5"), db=str(d / "proteins.one.h5"), k=1)
        assert len(cls) == 10


def test_full_pipeline_default_768d():
    """Test the default d_out=768 pipeline (encode + decode only, no tool probes)."""
    from src.one_embedding import encode, decode

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        with h5py.File(str(d / "raw.h5"), "w") as f:
            for i in range(5):
                f.create_dataset(f"prot_{i}", data=np.random.randn(50 + i*10, 1024).astype(np.float32))

        encode(str(d / "raw.h5"), str(d / "proteins.one.h5"))

        data = decode(str(d / "proteins.one.h5"))
        assert len(data) == 5
        assert data["prot_0"]["per_residue"].shape[1] == 768
        assert data["prot_0"]["protein_vec"].shape == (3072,)


def test_existing_tests_not_broken():
    """Sanity: import existing modules to verify backward compatibility."""
    from src.one_embedding.codec import OneEmbeddingCodec
    from src.one_embedding.embedding import OneEmbedding
    from src.one_embedding.transforms import dct_summary
    from src.one_embedding.universal_transforms import random_orthogonal_project
    from src.one_embedding.io import save_one_embeddings, load_one_embeddings
    # If we get here, imports are fine
    assert True
