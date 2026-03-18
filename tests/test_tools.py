import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile, pytest


def _make_oemb(d, n=5, L=80, D=512):
    from src.one_embedding.io import write_oemb_batch
    proteins = {}
    for i in range(n):
        proteins[f"prot_{i}"] = {
            "per_residue": np.random.randn(L, D).astype(np.float16),
            "protein_vec": np.random.randn(2048).astype(np.float16),
            "sequence": "A" * L,
        }
    path = Path(d) / "test.oemb"
    write_oemb_batch(str(path), proteins)
    return str(path)


class TestDisorder:
    def test_returns_scores(self):
        from src.one_embedding.tools.disorder import predict
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = predict(path)
            assert "prot_0" in r
            assert len(r["prot_0"]) == 80
            assert np.isfinite(r["prot_0"]).all()  # CNN outputs Z-scores (unbounded)


class TestClassify:
    def test_returns_neighbors(self):
        from src.one_embedding.tools.classify import predict
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = predict(path, db=path, k=2)
            assert "prot_0" in r
            assert r["prot_0"]["top_match"] is not None


class TestSearch:
    def test_returns_hits(self):
        from src.one_embedding.tools.search import find_neighbors
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = find_neighbors(path, k=3)
            assert "prot_0" in r
            assert len(r["prot_0"]) == 3
            assert "similarity" in r["prot_0"][0]


class TestAlign:
    def test_align_pair(self):
        from src.one_embedding.tools.align import align_pair
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = align_pair(path, "prot_0", "prot_1")
            assert "score" in r
            assert "n_aligned" in r


class TestSS3:
    def test_returns_predictions(self):
        from src.one_embedding.tools.ss3 import predict
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = predict(path)
            assert "prot_0" in r
            assert set(np.unique(r["prot_0"])).issubset({0, 1, 2})


class TestConserve:
    def test_returns_scores(self):
        from src.one_embedding.tools.conserve import score
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = score(path)
            assert "prot_0" in r
            assert len(r["prot_0"]) == 80


class TestMutate:
    def test_returns_sensitivity(self):
        from src.one_embedding.tools.mutate import scan
        with tempfile.TemporaryDirectory() as d:
            path = _make_oemb(d)
            r = scan(path)
            assert "prot_0" in r
            assert len(r["prot_0"]) == 80
