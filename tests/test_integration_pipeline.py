"""Integration test: Codec -> .one.h5 -> tools pipeline."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.core import Codec
from src.one_embedding.io import write_one_h5_batch, read_one_h5_batch, inspect_one_h5
from src.one_embedding.tools._base import load_per_residue, load_protein_vecs


def _make_corpus(n=20, D=1024, seed=42):
    rng = np.random.RandomState(seed)
    return {f"prot_{i}": rng.randn(30 + i * 5, D).astype(np.float32) for i in range(n)}


class TestFullPipeline:
    @pytest.mark.parametrize("d_out", [512, 768])
    def test_encode_write_load_roundtrip(self, tmp_path, d_out):
        corpus = _make_corpus(n=10, D=1024)
        codec = Codec(d_out=d_out)
        codec.fit(corpus)

        encoded = {}
        for pid, raw in corpus.items():
            result = codec.encode(raw)
            encoded[pid] = {
                "per_residue": result["per_residue"].astype(np.float32),
                "protein_vec": result["protein_vec"],
            }

        path = tmp_path / "encoded.one.h5"
        write_one_h5_batch(path, encoded, tags={
            "source_model": "prot_t5_xl",
            "d_out": d_out,
            "compression": f"abtt3_rp{d_out}",
        })

        info = inspect_one_h5(path)
        assert info["n_proteins"] == 10
        assert info["d_out"] == d_out

        per_res = load_per_residue(str(path))
        vecs = load_protein_vecs(str(path))
        assert len(per_res) == 10
        assert len(vecs) == 10
        assert next(iter(per_res.values())).shape[1] == d_out
        assert next(iter(vecs.values())).shape[0] == 4 * d_out

    def test_tags_survive_roundtrip(self, tmp_path):
        corpus = _make_corpus(n=3, D=1024)
        codec = Codec(d_out=768)
        codec.fit(corpus)

        encoded = {}
        for pid, raw in corpus.items():
            result = codec.encode(raw)
            encoded[pid] = {
                "per_residue": result["per_residue"].astype(np.float32),
                "protein_vec": result["protein_vec"],
                "tags": {"organism": "human", "length": raw.shape[0]},
            }

        path = tmp_path / "tagged.one.h5"
        write_one_h5_batch(path, encoded, tags={"lab": "rostlab"})

        info = inspect_one_h5(path)
        assert info["tags"]["lab"] == "rostlab"
